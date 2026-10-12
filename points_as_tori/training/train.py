"""
Torch port of training/train.py (JAX / flax.nnx / optax).

	python -m points_as_tori.training.train          # the original five-phase schedule (train())

Function names, arguments, defaults, printed logs and the training schedule follow the original.

Correspondences:
	nnx.value_and_grad + zero_out_nans + nnx.Optimizer(optax.adamw(schedule))
	    -> loss.backward(); NaN gradient entries set to 0; torch.optim.AdamW with lr set to schedule(count) before
	       each update (optax evaluates the schedule at the count of previous updates). Adam/AdamW formulas are the
	       same: m ← β1 m + (1-β1) g, v ← β2 v + (1-β2) g², p ← p - lr (m̂/(√v̂ + ε) + wd p), m̂ = m/(1-β1ᵗ), v̂ = v/(1-β2ᵗ)
	optax.warmup_cosine_decay_schedule -> warmup_cosine_decay_schedule below (same piecewise formula)
	jax.lax.cond(n > 0, sum / n, 0)    -> torch.where(n > 0, sum / max(n, 1), 0) (no host branch)
	jax.lax.dynamic_slice_in_dim       -> _dynamic_slice_start (start index clamped so the slice stays in bounds, so
	                                      the last batch of a shape overlaps the previous one, as in JAX)
	JAX keys fold_in(epoch, shape)     -> torch.Generator seeded from (seed, epoch, shape index)
	flax .pkl checkpoints              -> torch .pt checkpoints (FundamentalFormPredictor.save_model); warm starts
	                                      accept either

Device: everything runs on config.get_device(). Each shape is uploaded once per epoch (pinned, non-blocking on CUDA);
losses are accumulated on the device and read back only where the original prints (log/report points).
Training data: the same .npz files as the original (mesh_data.py / the pre-generated datasets).
"""

from __future__ import annotations

import glob
import math
import os
import time
from typing import Callable, Optional

import numpy as np
import torch

from .. import config
from ..network import (
	MODELS_DIR,
	FundamentalFormPredictor,
	convert_flax_checkpoint,
	torus_distance_gradient_masked,
	torus_precompute_masked,
)
from .mesh_sampling import generate_shape_features


def load_mesh_training_data(filepath):
	"""
	Load training data with mesh information from NPZ file (host arrays; uploaded per shape during training).

	Args:
		filepath: Path to NPZ file

	Returns:
		Tuple of NumPy arrays:
			all_positions: (n_shapes, max_points, 3)
			all_normals: (n_shapes, max_points, 3)
			all_neighbors: (n_shapes, max_points, k_neighbors)
			all_masks: (n_shapes, max_points)
			all_mesh_vertices: (n_shapes, max_vertices, 3)
			all_mesh_faces: (n_shapes, max_faces, 3)
			all_face_masks: (n_shapes, max_faces)
	"""
	data = np.load(filepath)

	all_positions = data['positions']
	all_normals = data['normals']
	all_neighbors = data['neighbors']
	all_masks = data['masks']
	all_mesh_vertices = data['mesh_vertices']
	all_mesh_faces = data['mesh_faces']
	all_face_masks = data['face_masks']

	print(f'Loaded mesh training data from {filepath}')
	print(f'  Shapes: {len(all_positions)} shapes')
	print(f'  Max points per shape: {all_positions.shape[1]}')
	print(f'  k_neighbors: {all_neighbors.shape[2]}')
	print(f'  Max vertices: {all_mesh_vertices.shape[1]}')
	print(f'  Max faces: {all_mesh_faces.shape[1]}')

	return (
		all_positions,
		all_normals,
		all_neighbors,
		all_masks,
		all_mesh_vertices,
		all_mesh_faces,
		all_face_masks,
	)


def upload_shape(data, shape_idx: int, dtype: Optional[torch.dtype] = None) -> dict:
	"""
	Import boundary: one shape of a dataset (host arrays) -> device tensors in the wide layout.

	Returns dict with positions, normals (3, max_points); neighbors (max_points, k) int64; mask (max_points,) bool;
	mesh_vertices (3, max_vertices); mesh_faces (3, max_faces) int64; face_mask (max_faces,) bool.
	"""
	device = config.get_device()
	dtype = config.get_dtype() if dtype is None else dtype
	pin = device.type == 'cuda'

	def upload(x, dt, wide):
		t = torch.from_numpy(np.ascontiguousarray(x))
		if pin:
			t = t.pin_memory()
		t = t.to(device=device, dtype=dt, non_blocking=pin)
		return t.T.contiguous() if wide else t

	all_positions, all_normals, all_neighbors, all_masks, all_mesh_vertices, all_mesh_faces, all_face_masks = data
	return dict(
		positions=upload(all_positions[shape_idx], dtype, True),
		normals=upload(all_normals[shape_idx], dtype, True),
		neighbors=upload(all_neighbors[shape_idx], torch.long, False),
		mask=upload(all_masks[shape_idx], torch.bool, False),
		mesh_vertices=upload(all_mesh_vertices[shape_idx], dtype, True),
		mesh_faces=upload(all_mesh_faces[shape_idx], torch.long, True),
		face_mask=upload(all_face_masks[shape_idx], torch.bool, False),
	)


def shape_generator(seed: int, *indices: int) -> torch.Generator:
	"""Deterministic generator per (seed, epoch, shape, ...) — the analog of jax.random.fold_in chains."""
	state = np.random.SeedSequence([seed, *indices]).generate_state(1, dtype=np.uint64)[0]
	generator = torch.Generator(device=config.get_device())
	generator.manual_seed(int(state) & 0x7FFF_FFFF_FFFF_FFFF)
	return generator


def _dynamic_slice_start(start_idx: int, size: int, slice_size: int) -> int:
	"""jax.lax.dynamic_slice_in_dim clamps the start index into [0, size - slice_size]."""
	if slice_size > size:
		raise ValueError(f'slice size {slice_size} larger than dimension {size} (as in jax.lax.dynamic_slice_in_dim)')
	return min(max(start_idx, 0), size - slice_size)


# ========================================================================
# TRAINING
# ========================================================================


def _masked_mean(total: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
	"""jax.lax.cond(count > 0, total / count, 0.0) without a host branch (safe denominator in the unused branch)."""
	return torch.where(count > 0, total / torch.clamp_min(count, 1).to(total.dtype), torch.zeros_like(total))


def _sdf_losses(mask, batch_mask, positions, normals, coefficients, queries, true_distances):
	"""Shared body of SDF_loss / validation_loss."""
	centers, axes, major_radii, minor_radii = torus_precompute_masked(mask, positions, normals, coefficients)

	predicted_distances, predicted_gradients = torus_distance_gradient_masked(
		mask, queries, positions, centers, axes, major_radii, minor_radii
	)

	distance_error = torch.abs(predicted_distances - true_distances)  # (batch_size, n_queries, )
	distance_valid_mask = ~torch.isnan(distance_error) & batch_mask[:, None]
	# As in the original, the error is multiplied by the mask (not selected), so a NaN prediction makes the
	# distance loss NaN even though its query is marked invalid (NaN * 0 = NaN).
	distance_error_masked = torch.sum(distance_error * distance_valid_mask, dim=-1)  # (batch_size,)
	num_valid_distance_queries = torch.sum(distance_valid_mask)

	eikonality_error = torch.abs(
		1.0 - torch.sqrt(torch.sum(predicted_gradients * predicted_gradients, dim=0))
	)  # (batch_size, n_queries,)
	eikonality_valid_mask = ~torch.isnan(eikonality_error) & batch_mask[:, None]  # (batch_size, n_queries)
	eikonality_error_masked = torch.sum(
		torch.where(eikonality_valid_mask, eikonality_error, 0.0), dim=-1
	)  # (batch_size, )
	num_valid_eikonality_queries = torch.sum(eikonality_valid_mask)

	# Take mean only over valid points in the neighborhood.
	distance_loss = _masked_mean(torch.sum(distance_error_masked), num_valid_distance_queries)
	eikonality_loss = _masked_mean(torch.sum(eikonality_error_masked), num_valid_eikonality_queries)
	return distance_loss, eikonality_loss


def SDF_loss(
	mask,
	batch_mask,
	positions,
	normals,
	coefficients,
	queries,
	true_distances,
):
	"""
	Compute a loss based on the end-to-end SDF process.
	Returns loss that is averaged over all neighborhoods in the batch.

	Loss = mean |φ(q) - d_true(q)| + mean | 1 - |∇φ(q)| |  over valid queries of valid neighborhoods
	(L1 distance error + eikonal error, φ the torus blend of torus_distance_gradient_masked).

	Args:
		mask: (max_points, ) boolean-valued array indicating which points are actually valid and belong to the input point cloud
		batch_mask: (batch_size, ) boolean-valued array indicating which query points in the batch are valid
		positions, normals: (3, max_points) the entire input point cloud of a single shape
		coefficients: (6, max_points) all point coefficients
		queries: (3, batch_size, n_queries)
		true_distances: (batch_size, n_queries,)
	"""
	distance_loss, eikonality_loss = _sdf_losses(
		mask, batch_mask, positions, normals, coefficients, queries, true_distances
	)

	loss = distance_loss + eikonality_loss

	# Return individual components for logging
	loss_dict = {
		'distance_loss': distance_loss,
		'eikonality_loss': eikonality_loss,
	}

	return loss, loss_dict


def validation_loss(
	mask,
	batch_mask,
	positions,
	normals,
	coefficients,
	queries,
	true_distances,
):
	return _sdf_losses(mask, batch_mask, positions, normals, coefficients, queries, true_distances)


def warmup_cosine_decay_schedule(
	init_value: float,
	peak_value: float,
	warmup_steps: float,
	decay_steps: float,
	end_value: float = 0.0,
	exponent: float = 1.0,
) -> Callable[[int], float]:
	"""
	optax.warmup_cosine_decay_schedule (host-side, float64):
		count < W:  init + (peak - init) count / W                         (linear warmup; W <= 0: no warmup)
		count >= W: peak ((1 - α) c(t)^exponent + α),  t = min(count - W, T),  c(t) = ½ (1 + cos(π t / T)),
		            T = decay_steps - W,  α = end / peak
	"""
	alpha = 0.0 if peak_value == 0.0 else end_value / peak_value
	cosine_steps = decay_steps - warmup_steps
	if not cosine_steps > 0:
		raise ValueError(f'The cosine_decay_schedule requires positive decay_steps, got decay_steps={cosine_steps}.')

	def linear(count):
		if warmup_steps <= 0:
			return init_value
		count = min(max(count, 0), warmup_steps)
		frac = 1 - count / warmup_steps
		return (init_value - peak_value) * frac + peak_value

	def cosine(count):
		count = min(count, cosine_steps)
		cosine_decay = 0.5 * (1 + math.cos(math.pi * count / cosine_steps))
		return peak_value * ((1 - alpha) * cosine_decay**exponent + alpha)

	def schedule(count):
		return linear(count) if count < warmup_steps else cosine(count - warmup_steps)

	return schedule


def FundamentalFormPredictorTrainer(
	embed_dim=128,
	n_freq=4,
	n_layers=4,
	n_heads=4,
	mlp_dim=256,
	dropout_rate=0.0,
	learning_rate=1e-3,
	weight_decay=0.0,
	warmup_steps=1000,
	total_steps=100000,
	seed=42,
):
	"""
	Set up network (model) with the given parameters, on the configured device.
	(The original takes `rngs=nnx.Rngs(0)`; here `seed` seeds the parameter initialization.)

	Returns: (model, optimizer, train_step) where train_step() is a function that trains the model for a single step.
	"""
	# Initialize on the CPU generator (device-independent values), then move to the device once.
	with torch.random.fork_rng(devices=[]):
		torch.manual_seed(seed)
		model = FundamentalFormPredictor(
			embed_dim=embed_dim,
			n_freq=n_freq,
			n_layers=n_layers,
			n_heads=n_heads,
			mlp_dim=mlp_dim,
			dropout_rate=dropout_rate,
		)
	model = model.to(config.get_device())

	# Learning rate schedule
	schedule = warmup_cosine_decay_schedule(
		init_value=0.0,  # Start from 0
		peak_value=learning_rate,  # Warmup to this
		warmup_steps=warmup_steps,  # Linear warmup
		decay_steps=total_steps,  # Cosine decay over this many steps
		end_value=learning_rate * 0.01,
	)

	# Optimizer (optax.adamw defaults: b1=0.9, b2=0.999, eps=1e-8)
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=schedule(0), betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay
	)
	optimizer.schedule = schedule
	optimizer.update_count = 0  # optax's ScaleByScheduleState.count

	def train_step(
		model: FundamentalFormPredictor,
		optimizer: torch.optim.Optimizer,
		mask,
		batch_mask,
		positions,
		normals,
		neighbors,
		queries,
		true_distances,
	):
		"""
		Single training step on a single batch (neighborhood) of one shape.

		Args:
			mask: (max_points,)
			batch_mask: (batch_size, )
			positions, normals: (3, |P|)
			neighbors: (|P|, k)
			queries: (3, batch_size, n_queries)
			true_distances: (batch_size, n_queries)

		Returns:
			Loss averaged over batch (device tensors, detached).
		"""

		def loss_fn(model):
			dt = model._network_dtype()
			coefficients = model(positions[:, neighbors].to(dt), normals[:, neighbors].to(dt))

			loss, loss_dict = SDF_loss(mask, batch_mask, positions, normals, coefficients, queries, true_distances)

			return loss, loss_dict

		optimizer.zero_grad(set_to_none=False)
		batch_loss, loss_dict = loss_fn(model)
		batch_loss.backward()

		# zero_out_nans: apply to all gradients (inf is kept, as in the original)
		for p in model.parameters():
			if p.grad is not None:
				p.grad.masked_fill_(torch.isnan(p.grad), 0.0)

		# Update with this batch's gradients; the learning rate of update number `count` is schedule(count)
		lr = optimizer.schedule(optimizer.update_count)
		for group in optimizer.param_groups:
			group['lr'] = lr
		optimizer.step()
		optimizer.update_count += 1

		return batch_loss.detach(), {k: v.detach() for k, v in loss_dict.items()}

	return model, optimizer, train_step


def _read_checkpoint(path: str) -> dict:
	if path.endswith('.pkl'):
		return convert_flax_checkpoint(path)
	return torch.load(path, map_location=config.get_device(), weights_only=True)


def TrainFundamentalFormPredictor(
	training_data_filenames,
	num_epochs,
	learning_rate,
	batch_size,
	warmup_steps,
	n_epochs_to_save,
	n_shapes_to_save,
	load_model_path=None,
	final_save_filepath=None,
	*,
	models_dir: str = MODELS_DIR,
	val_data_filename: str = 'datasets/ABC17_validation_3.npz',
	plot_filepath: Optional[str] = 'PlotTrainingHistory.png',
	embed_dim: int = 128,
	n_freq: int = 0,
	n_layers: int = 8,
	n_heads: int = 8,
	mlp_dim: Optional[int] = None,
	n_queries_per_neighborhood: int = 120,
	narrow_band_offset: float = 0.2,
	bbox_extension: float = 1.0,
	seed: int = 42,
):
	"""
	Args:
		load_model_path: Optional path to a saved model file (.pt, or the original .pkl) to use as warm start
	Keyword-only arguments (added in the port; defaults are the values hard-coded in the original):
		models_dir, val_data_filename, plot_filepath (None: no plot), architecture and sampling parameters, seed.

	Returns: the trained model (added in the port; the original returns None).
	"""
	LABEL_WIDTH = 18  # Width for "distance loss = ", etc.
	VALUE_WIDTH = 12  # Width for the numeric values

	device = config.get_device()
	print(device)

	# NOTE: Could probably train on much larger meshes and point clouds using chunked versions of torus_precompute_masked() and precompute_coefficients(), to avoid GPU memory constraints (at the cost of increasing compute time).
	# The current (naive) versions of these functions are called on all points at once (usually in a vectorized manner).

	# Architecture parameters
	mlp_dim = embed_dim * 4 if mlp_dim is None else mlp_dim

	# Learning parameters
	dropout_rate = 0.0
	weight_decay = 0.0
	total_steps = 1e8

	model, optimizer, train_step = FundamentalFormPredictorTrainer(
		embed_dim=embed_dim,
		n_freq=n_freq,
		n_layers=n_layers,
		n_heads=n_heads,
		mlp_dim=mlp_dim,
		dropout_rate=dropout_rate,
		learning_rate=learning_rate,
		weight_decay=weight_decay,
		warmup_steps=warmup_steps,
		total_steps=total_steps,
		seed=seed,
	)

	# Load pretrained model if specified
	if load_model_path is not None:
		print(f'Loading pretrained model from {load_model_path}')
		saved_data = _read_checkpoint(load_model_path)

		# Verify architecture matches
		assert saved_data['embed_dim'] == embed_dim, 'embed_dim mismatch'
		assert saved_data['n_freq'] == n_freq, 'n_freq mismatch'
		assert saved_data['n_heads'] == n_heads, 'n_heads mismatch'
		assert saved_data['n_layers'] == n_layers, 'n_layers mismatch'
		assert saved_data['mlp_dim'] == mlp_dim, 'mlp_dim mismatch'

		# Load parameters into model
		model.load_state_dict(saved_data['state_dict'])
		print(f'Successfully loaded pretrained weights from {load_model_path}')

	# Load mesh training data from multiple files.
	all_training_data = []
	t0 = time.time()
	k_neighbors = None
	for training_data_filename in training_data_filenames:
		if not os.path.exists(training_data_filename):
			raise ValueError(f"The path '{training_data_filename}' does not exist")

		print(f"Loading training data from '{training_data_filename}'...")

		data = load_mesh_training_data(training_data_filename)

		(all_positions, all_normals, all_neighbors, all_masks, all_mesh_vertices, all_mesh_faces, all_face_masks) = data

		k_neighbors = all_neighbors.shape[2]
		n_shapes = len(all_positions)
		max_points = all_positions.shape[1]
		print(f'  Loaded {n_shapes} shapes with max {max_points} points each')
		print(f'  k_neighbors: {k_neighbors}')

		all_training_data.append(data)

	t1 = time.time()
	print(f'Load data: {t1 - t0:.6f} s')

	# Create combined list of (dataset_idx, shape_idx) for all shapes (across possibly multiple datasets)
	all_shape_indices = []
	for dataset_idx, data in enumerate(all_training_data):
		n_shapes = len(data[0])  # data[0] is all_positions
		for shape_idx in range(n_shapes):
			all_shape_indices.append((dataset_idx, shape_idx))

	total_shapes = len(all_shape_indices)
	print(f'Total shapes across all datasets: {total_shapes}')

	# Load validation data
	val_data = None
	if val_data_filename is not None and os.path.exists(val_data_filename):
		try:
			val_data = load_mesh_training_data(val_data_filename)
			print(f'Loaded validation data: {len(val_data[0])} shapes')
		except Exception as e:
			print(f'Warning: Could not load validation data: {e}')
			print('Continuing without validation...')
	else:
		print(f'Validation file not found: {val_data_filename}')

	def save_model(curr_model, epoch, save_filepath: str = None):
		# Save weights as a torch checkpoint (the original pickles the flax state)
		curr_model.eval()  # as nnx.eval_mode in the original (no effect with dropout_rate = 0)
		model_name = f'FundamentalFormPredictor_{epoch}' if save_filepath is None else save_filepath
		os.makedirs(models_dir, exist_ok=True)
		filepath = os.path.join(models_dir, model_name + '.pt')
		curr_model.save_model(filepath, k_neighbors)

		print(f'Model saved to {filepath}')

	# Training loop
	model.train()

	t0 = time.time()
	all_train_losses = []
	all_distance_losses = []
	all_eikonality_losses = []

	validation_losses = []
	validation_epochs = []

	print('\nBeginning training...')
	print(f'  Batch size: {batch_size}')

	rng = np.random.RandomState(seed)  # np.random.seed(seed) + np.random.permutation in the original

	for epoch in range(num_epochs):
		# Losses accumulate on the device; read back only when printed
		epoch_loss_sum = torch.zeros((), device=device, dtype=config.get_dtype())
		epoch_distance_loss_sum = torch.zeros((), device=device, dtype=config.get_dtype())
		epoch_eikonality_loss_sum = torch.zeros((), device=device, dtype=config.get_dtype())
		total_num_batches = 0

		# Shuffle all shapes across all datasets together
		shuffled_indices = rng.permutation(total_shapes)

		for i, global_idx in enumerate(shuffled_indices):
			dataset_idx, shape_idx = all_shape_indices[global_idx]

			max_points = all_training_data[dataset_idx][0].shape[1]
			n_batches = (max_points + batch_size - 1) // batch_size

			# deterministic generator per (epoch, shape)
			shape_rng = shape_generator(seed, epoch, int(global_idx))

			# Upload single shape to the device
			shape = upload_shape(all_training_data[dataset_idx], shape_idx)
			positions = shape['positions']
			normals = shape['normals']
			neighbors = shape['neighbors']
			mask = shape['mask']

			# Generate query points and ground-truth distances
			with torch.no_grad():
				shape_queries, shape_distances = generate_shape_features(
					positions,
					neighbors,
					shape['mesh_vertices'],
					shape['mesh_faces'],
					shape['face_mask'],
					shape_rng,
					n_queries=n_queries_per_neighborhood,
					narrow_band_offset=narrow_band_offset,
					bbox_extension=bbox_extension,
				)

			# Loop over batches for this shape
			for batch_idx in range(n_batches):
				start_idx = _dynamic_slice_start(batch_idx * batch_size, max_points, batch_size)
				end_idx = start_idx + batch_size

				batch_mask = mask[start_idx:end_idx]
				batch_queries = shape_queries[:, start_idx:end_idx]
				batch_distances = shape_distances[start_idx:end_idx]

				# Train step
				loss, loss_dict = train_step(
					model,
					optimizer,
					mask,
					batch_mask,
					positions,
					normals,
					neighbors,
					batch_queries,
					batch_distances,
				)

				epoch_loss_sum += loss
				epoch_distance_loss_sum += loss_dict['distance_loss']
				epoch_eikonality_loss_sum += loss_dict['eikonality_loss']

			total_num_batches += n_batches

			# Release this shape's device memory
			del shape, positions, normals, neighbors, mask, shape_queries, shape_distances

			if i % n_shapes_to_save == 0 and i > 0:
				epoch_train_loss = epoch_loss_sum.item() / total_num_batches
				epoch_distance_loss = epoch_distance_loss_sum.item() / total_num_batches
				epoch_eikonality_loss = epoch_eikonality_loss_sum.item() / total_num_batches

				print(
					f'  Epoch {epoch}, shape {i}/{total_shapes}: '
					f'loss = {epoch_train_loss:.6f}  '
					f'distance = {epoch_distance_loss:.6f}  '
					f'eikonality = {epoch_eikonality_loss:.6f}'
				)

		# Compute epoch averages (device -> host: report point)
		epoch_train_loss = epoch_loss_sum.item() / total_num_batches
		epoch_distance_loss = epoch_distance_loss_sum.item() / total_num_batches
		epoch_eikonality_loss = epoch_eikonality_loss_sum.item() / total_num_batches

		all_train_losses.append(epoch_train_loss)
		all_distance_losses.append(epoch_distance_loss)
		all_eikonality_losses.append(epoch_eikonality_loss)

		if epoch % n_epochs_to_save == 0:
			print(
				f'Epoch {epoch}:\n'
				f'	{"total loss:":<{LABEL_WIDTH}}{epoch_train_loss:<{VALUE_WIDTH}.6f}'
				f'{"distance:":<{LABEL_WIDTH}}{epoch_distance_loss:<{VALUE_WIDTH}.6f}'
				f'{"eikonality:":<{LABEL_WIDTH}}{epoch_eikonality_loss:<{VALUE_WIDTH}.6f}'
			)

			# Save model checkpoint
			save_model(model, epoch)

			# Validation loss
			if val_data is not None:
				model.eval()
				val_loss, val_eikonality_loss = _validation(
					model, val_data, epoch, batch_size, n_queries_per_neighborhood, seed
				)
				model.train()

				validation_losses.append(val_loss)
				validation_epochs.append(epoch)

				print(
					f'	{"validation:":<{LABEL_WIDTH}}{val_loss:<{VALUE_WIDTH}.6f}'
					f'{"eikonality:":<{LABEL_WIDTH}}{val_eikonality_loss:<{VALUE_WIDTH}.6f}'
				)

	t1 = time.time()
	print(f'Total training time: {t1 - t0:.2f} s')

	train_losses = np.array(all_train_losses)
	distance_losses = np.array(all_distance_losses)
	eikonality_losses = np.array(all_eikonality_losses)

	# Save final model
	save_model(model, num_epochs - 1, final_save_filepath)

	# Plot training history
	if plot_filepath is not None:
		_plot_training_history(
			plot_filepath, train_losses, distance_losses, eikonality_losses, validation_epochs, validation_losses
		)

	return model


@torch.no_grad()
def _validation(model, val_data, epoch, batch_size, n_queries_per_neighborhood, seed):
	"""
	Validation block of the original training loop. Per shape: mean of the per-batch losses over batches that contain
	valid points; then the mean over shapes with at least one such batch. The `if jnp.sum(batch_mask) > 0` host
	branches of the original are replaced by weights (valid batch -> 1, else 0), so only the final values are read back.
	"""
	device, dtype = config.get_device(), config.get_dtype()
	n_val_shapes = len(val_data[0])
	shape_loss_sum = torch.zeros((), device=device, dtype=dtype)
	shape_eikonality_sum = torch.zeros((), device=device, dtype=dtype)
	n_valid_shapes = torch.zeros((), device=device, dtype=dtype)

	for val_shape_idx in range(n_val_shapes):
		# Generate validation generator (deterministic per epoch)
		val_rng = shape_generator(seed, epoch, 10000 + val_shape_idx)

		shape = upload_shape(val_data, val_shape_idx)
		val_pos = shape['positions']
		val_norm = shape['normals']
		val_neigh = shape['neighbors']
		val_mask = shape['mask']

		# Generate validation queries on-the-fly
		val_queries, val_distances = generate_shape_features(
			val_pos,
			val_neigh,
			shape['mesh_vertices'],
			shape['mesh_faces'],
			shape['face_mask'],
			val_rng,
			n_queries=n_queries_per_neighborhood,
		)

		# Compute weights for this validation shape
		dt = model._network_dtype()
		val_weights = model(val_pos[:, val_neigh].to(dt), val_norm[:, val_neigh].to(dt))

		# Compute loss per batch
		n_points = val_pos.shape[1]
		n_val_batches = (n_points + batch_size - 1) // batch_size
		batch_loss_sum = torch.zeros((), device=device, dtype=dtype)
		batch_eikonality_sum = torch.zeros((), device=device, dtype=dtype)
		n_valid_batches = torch.zeros((), device=device, dtype=dtype)

		for batch_idx in range(n_val_batches):
			start_idx = _dynamic_slice_start(batch_idx * batch_size, n_points, batch_size)
			end_idx = start_idx + batch_size
			batch_mask = val_mask[start_idx:end_idx]

			val_loss, val_eikonality_loss = validation_loss(
				val_mask,
				batch_mask,
				val_pos,
				val_norm,
				val_weights,
				val_queries[:, start_idx:end_idx],
				val_distances[start_idx:end_idx],
			)
			# Only count batches with valid points
			has_valid = batch_mask.any()
			batch_loss_sum += torch.where(has_valid, val_loss, 0.0)
			batch_eikonality_sum += torch.where(has_valid, val_eikonality_loss, 0.0)
			n_valid_batches += has_valid

		# Average over batches for this shape
		shape_valid = n_valid_batches > 0
		shape_loss_sum += torch.where(shape_valid, batch_loss_sum / torch.clamp_min(n_valid_batches, 1), 0.0)
		shape_eikonality_sum += torch.where(
			shape_valid, batch_eikonality_sum / torch.clamp_min(n_valid_batches, 1), 0.0
		)
		n_valid_shapes += shape_valid

	# Average over all validation shapes (device -> host: report point)
	n = n_valid_shapes.item()
	val_loss = shape_loss_sum.item() / n if n > 0 else 0.0
	val_eikonality_loss = shape_eikonality_sum.item() / n if n > 0 else 0.0
	return val_loss, val_eikonality_loss


def _plot_training_history(plot_filepath, train_losses, distance_losses, eikonality_losses, val_epochs, val_losses):
	try:
		import matplotlib

		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
	except ImportError:
		print('matplotlib not installed: skipping the training history plot')
		return

	fig, axes = plt.subplots(2, 2, figsize=(15, 10))

	# Total training loss
	axes[0, 0].plot(train_losses, label='Total Training Loss', color='blue')
	axes[0, 0].set_xlabel('Epoch')
	axes[0, 0].set_ylabel('Loss')
	axes[0, 0].set_title('Total Training Loss')
	axes[0, 0].legend()
	axes[0, 0].grid(True, alpha=0.3)

	# Distance loss
	axes[0, 1].plot(distance_losses, label='Distance Error', color='green')
	axes[0, 1].set_xlabel('Epoch')
	axes[0, 1].set_ylabel('Loss')
	axes[0, 1].set_title('Distance Error')
	axes[0, 1].legend()
	axes[0, 1].grid(True, alpha=0.3)

	# Eikonality loss
	axes[1, 0].plot(eikonality_losses, label='Eikonality Error', color='orange')
	axes[1, 0].set_xlabel('Epoch')
	axes[1, 0].set_ylabel('Loss')
	axes[1, 0].set_title('Eikonality Error')
	axes[1, 0].legend()
	axes[1, 0].grid(True, alpha=0.3)

	# Validation loss
	if len(val_losses) > 0:
		axes[1, 1].plot(val_epochs, val_losses, label='Validation Loss', color='red', marker='o')
		axes[1, 1].set_xlabel('Epoch')
		axes[1, 1].set_ylabel('Loss')
		axes[1, 1].set_title('Validation Loss')
		axes[1, 1].legend()
		axes[1, 1].grid(True, alpha=0.3)

	plt.tight_layout()
	plt.savefig(plot_filepath, dpi=150)
	plt.close(fig)
	print(f'Training history plot saved to {plot_filepath}')


def train(dataset_dir: str = 'datasets/nice', models_dir: str = MODELS_DIR):
	"""The original five-phase schedule; paths relative to the working directory, as in the original."""
	ckpt = lambda name: os.path.join(models_dir, name + '.pt')  # noqa: E731

	# Pretraining
	training_data_filenames = ['datasets/pretraining_blob.npz']
	TrainFundamentalFormPredictor(
		training_data_filenames,
		num_epochs=64,
		learning_rate=1e-3,
		warmup_steps=0,
		batch_size=32,
		n_epochs_to_save=1,
		n_shapes_to_save=1,
		load_model_path=None,
		final_save_filepath='FundamentalFormPredictor_pretrained',
		models_dir=models_dir,
	)

	# Phase 0: nicely sampled blobs
	training_data_filenames = sorted(glob.glob(os.path.join(dataset_dir, 'blobs_train_mode0*.npz')))
	TrainFundamentalFormPredictor(
		training_data_filenames,
		num_epochs=24,
		learning_rate=1e-5,
		warmup_steps=1e2,
		batch_size=1024,
		n_epochs_to_save=1,
		n_shapes_to_save=1,
		load_model_path=ckpt('FundamentalFormPredictor_pretrained'),
		final_save_filepath='FundamentalFormPredictor_phase0',
		models_dir=models_dir,
	)

	# Phase 1: nicely sampled blobs and ABC meshes
	blob_filenames = glob.glob(os.path.join(dataset_dir, 'blobs_train_mode0*.npz'))
	ABC_filenames = glob.glob(os.path.join(dataset_dir, 'ABC17_train_mode0_*.npz'))
	lists = [blob_filenames, ABC_filenames]
	training_data_filenames = [val for tup in zip(*lists) for val in tup]  # interleave lists
	TrainFundamentalFormPredictor(
		training_data_filenames,
		num_epochs=8,
		learning_rate=1e-5,
		warmup_steps=1e3,
		batch_size=1024,
		n_epochs_to_save=1,
		n_shapes_to_save=100,
		load_model_path=ckpt('FundamentalFormPredictor_phase0'),
		final_save_filepath='FundamentalFormPredictor_phase1',
		models_dir=models_dir,
	)

	# Phase 2: nicely sampled blobs and more ABC meshes
	blob_filenames = glob.glob(os.path.join(dataset_dir, 'blobs_train_mode0*.npz'))
	ABC17_filenames = glob.glob(os.path.join(dataset_dir, 'ABC17_train_mode0_*.npz'))
	ABC2_filenames = glob.glob(os.path.join(dataset_dir, 'ABC2_train_mode0_*.npz'))
	training_data_filenames = blob_filenames + ABC17_filenames + ABC2_filenames
	TrainFundamentalFormPredictor(
		training_data_filenames,
		num_epochs=8,
		learning_rate=1e-6,
		warmup_steps=1e3,
		batch_size=1024,
		n_epochs_to_save=1,
		n_shapes_to_save=100,
		load_model_path=ckpt('FundamentalFormPredictor_phase1'),
		final_save_filepath='FundamentalFormPredictor_phase2',
		models_dir=models_dir,
	)

	# Phase 3: nicely sampled blobs and even more ABC meshes
	blob_filenames = glob.glob(os.path.join(dataset_dir, 'blobs_train_mode0*.npz'))
	ABC_filenames = glob.glob(os.path.join(dataset_dir, 'ABC*_train_mode0_*.npz'))
	training_data_filenames = blob_filenames + ABC_filenames
	TrainFundamentalFormPredictor(
		training_data_filenames,
		num_epochs=5,
		learning_rate=1e-6,
		warmup_steps=1e3,
		batch_size=1024,
		n_epochs_to_save=1,
		n_shapes_to_save=100,
		load_model_path=ckpt('FundamentalFormPredictor_phase2'),
		final_save_filepath='FundamentalFormPredictor_phase3',
		models_dir=models_dir,
	)

	# Phase 4: everything from before, plus unevenly sampled data possibly with small gaps from a simulated scanning process
	blob_filenames = glob.glob(os.path.join(dataset_dir, 'blobs_train_mode0*.npz'))
	ABC_filenames = glob.glob(os.path.join(dataset_dir, 'ABC*_train_mode0_*.npz'))
	blob_filenames_simulated = glob.glob(os.path.join('datasets/simulated', 'blobs_train_mode*.npz'))
	ABC_filenames_simulated = glob.glob(os.path.join('datasets/simulated', 'ABC*_train_mode*.npz'))
	training_data_filenames = blob_filenames + ABC_filenames + blob_filenames_simulated + ABC_filenames_simulated
	TrainFundamentalFormPredictor(
		training_data_filenames,
		num_epochs=2,
		learning_rate=1e-7,
		warmup_steps=1e3,
		batch_size=1024,
		n_epochs_to_save=1,
		n_shapes_to_save=100,
		load_model_path=ckpt('FundamentalFormPredictor_phase3'),
		final_save_filepath='FundamentalFormPredictor_phase4',
		models_dir=models_dir,
	)


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Train the FundamentalFormPredictor (torch port of training/train.py)')
	parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
	parser.add_argument('--dataset-dir', default='datasets/nice')
	parser.add_argument('--models-dir', default=MODELS_DIR)
	args = parser.parse_args()
	config.set_device(args.device, torch.float32)
	train(args.dataset_dir, args.models_dir)
