"""
Training port, without reference implementations: schedule, slicing, an end-to-end training run on synthetic data,
the behavior on padded shapes, and the absence of host syncs (meta device).
"""

import math
import os

import numpy as np
import pytest
import torch

import points_as_tori as pat
from points_as_tori import config
from points_as_tori.network import FundamentalFormPredictor
from points_as_tori.training import mesh_sampling as pms
from points_as_tori.training import train as ptrain
from helpers import save_synthetic_dataset, synthetic_dataset

TINY = dict(embed_dim=16, n_freq=0, n_layers=1, n_heads=2, mlp_dim=32)


def test_schedule_closed_form():
	lr, W, T = 1e-3, 100, 1000
	s = ptrain.warmup_cosine_decay_schedule(0.0, lr, W, T, lr * 0.01)
	assert s(0) == 0.0 and s(50) == pytest.approx(0.5 * lr) and s(W) == pytest.approx(lr)
	t = 300
	assert s(W + t) == pytest.approx(lr * (0.99 * 0.5 * (1 + math.cos(math.pi * t / (T - W))) + 0.01))
	assert s(10 * T) == pytest.approx(lr * 0.01)
	assert ptrain.warmup_cosine_decay_schedule(0.0, lr, 0, 1e8, lr * 0.01)(0) == pytest.approx(lr)


def test_dynamic_slice_start():
	assert ptrain._dynamic_slice_start(0, 2048, 1024) == 0
	assert ptrain._dynamic_slice_start(1024, 2048, 1024) == 1024
	assert ptrain._dynamic_slice_start(1024, 2000, 1024) == 976  # last batch overlaps, as jax.lax.dynamic_slice
	with pytest.raises(ValueError):
		ptrain._dynamic_slice_start(0, 10, 32)


def test_upload_shape_layout(cpu32):
	data = synthetic_dataset(n_shapes=1, n_points=50, k=8)
	arrays = tuple(
		data[k] for k in ('positions', 'normals', 'neighbors', 'masks', 'mesh_vertices', 'mesh_faces', 'face_masks')
	)
	s = ptrain.upload_shape(arrays, 0)
	assert s['positions'].shape == (3, 58) and s['positions'].is_contiguous()
	assert s['mesh_faces'].shape[0] == 3 and s['mesh_faces'].dtype == torch.long
	assert s['neighbors'].shape == (58, 8) and s['mask'].dtype == torch.bool
	np.testing.assert_array_equal(s['positions'].numpy().T, data['positions'][0].astype(np.float32))


def test_generate_shape_features_ground_truth(cpu64):
	"""Queries have the requested structure and the ground truth is the signed distance to the mesh."""
	data = synthetic_dataset(n_shapes=1, n_points=80, k=16, pad_points=0)
	arrays = tuple(
		data[k] for k in ('positions', 'normals', 'neighbors', 'masks', 'mesh_vertices', 'mesh_faces', 'face_masks')
	)
	s = ptrain.upload_shape(arrays, 0)
	gen = ptrain.shape_generator(0, 0, 0)
	q, d = pms.generate_shape_features(
		s['positions'], s['neighbors'], s['mesh_vertices'], s['mesh_faces'], s['face_mask'], gen
	)
	assert q.shape == (3, 80, 120) and d.shape == (80, 120)
	# first third: on the surface -> distance ~ 0; middle third: within the narrow band (|offset| <= 0.2)
	assert d[:, :40].abs().max() < 1e-12
	assert d[:, 40:80].abs().max() <= 0.2 + 1e-12
	# sign: points inside the torus tube are negative
	assert (d < 0).any() and (d > 0).any()
	# same generator seed -> same samples
	q2, d2 = pms.generate_shape_features(
		s['positions'],
		s['neighbors'],
		s['mesh_vertices'],
		s['mesh_faces'],
		s['face_mask'],
		ptrain.shape_generator(0, 0, 0),
	)
	assert torch.equal(q, q2) and torch.equal(d, d2)


def _tiny_step_inputs(pad_points):
	data = synthetic_dataset(n_shapes=1, n_points=60, k=16, pad_points=pad_points, seed=5)
	arrays = tuple(
		data[k] for k in ('positions', 'normals', 'neighbors', 'masks', 'mesh_vertices', 'mesh_faces', 'face_masks')
	)
	s = ptrain.upload_shape(arrays, 0)
	q, d = pms.generate_shape_features(
		s['positions'],
		s['neighbors'],
		s['mesh_vertices'],
		s['mesh_faces'],
		s['face_mask'],
		ptrain.shape_generator(1, 2, 3),
		n_queries=30,
	)
	return s, q, d


def test_padded_shapes_give_nan_gradients(cpu32):
	"""
	Behavior inherited from the original (measured identically in JAX, see test_training_parity): padded points have
	neighbor rows of -1, i.e. a neighborhood of identical (zero) points, whose median neighbor distance is 0, so the
	input features are 0/0 = NaN. The loss masks these points, but backpropagating a zero cotangent through NaN
	activations gives NaN (0 · NaN), and every weight gradient sums over all points: all gradient entries are NaN and
	zero_out_nans turns the step into a zero-gradient step.
	"""
	for pad, expect_nan in ((0, False), (4, True)):
		model, optimizer, train_step = ptrain.FundamentalFormPredictorTrainer(
			**TINY, learning_rate=1e-3, warmup_steps=0
		)
		s, q, d = _tiny_step_inputs(pad)
		n = s['positions'].shape[1]
		dt = model._network_dtype()
		coefficients = model(s['positions'][:, s['neighbors']].to(dt), s['normals'][:, s['neighbors']].to(dt))
		if pad:
			assert torch.isnan(coefficients[:, n - pad :]).all() and torch.isfinite(coefficients[:, : n - pad]).all()
		else:
			assert torch.isfinite(coefficients).all()
		loss, _ = ptrain.SDF_loss(
			s['mask'], s['mask'][:32], s['positions'], s['normals'], coefficients, q[:, :32], d[:32]
		)
		loss.backward()
		grads = torch.cat([p.grad.flatten() for p in model.parameters()])
		assert bool(torch.isnan(grads).all()) == expect_nan
		assert torch.isfinite(loss)


def test_train_step_on_meta_device():
	"""Sampling, ground truth, loss, backward and the AdamW update run on `meta`: no data-dependent host operations."""
	config.set_device('meta', torch.float32, force=True)
	try:
		data = synthetic_dataset(n_shapes=1, n_points=60, k=16, pad_points=4)
		arrays = tuple(
			data[k] for k in ('positions', 'normals', 'neighbors', 'masks', 'mesh_vertices', 'mesh_faces', 'face_masks')
		)
		s = ptrain.upload_shape(arrays, 0)
		q, d = pms.generate_shape_features(
			s['positions'], s['neighbors'], s['mesh_vertices'], s['mesh_faces'], s['face_mask'], None, n_queries=30
		)
		assert q.shape == (3, 64, 30) and d.shape == (64, 30) and q.device.type == 'meta'
		model, optimizer, train_step = ptrain.FundamentalFormPredictorTrainer(**TINY, warmup_steps=0)
		assert next(model.parameters()).device.type == 'meta'
		loss, loss_dict = train_step(
			model, optimizer, s['mask'], s['mask'][:32], s['positions'], s['normals'], s['neighbors'], q[:, :32], d[:32]
		)
		assert loss.device.type == 'meta' and loss.shape == ()
		assert optimizer.update_count == 1
	finally:
		config.set_device('cpu', torch.float32, force=True)


def test_training_loop_end_to_end(cpu32, tmp_path):
	"""Two epochs on synthetic data (unpadded), with validation; checkpoints load into PointsAsTori."""
	train_file = save_synthetic_dataset(
		str(tmp_path / 'train.npz'), n_shapes=3, n_points=96, k=16, pad_points=0, seed=1
	)
	val_file = save_synthetic_dataset(str(tmp_path / 'val.npz'), n_shapes=1, n_points=96, k=16, pad_points=0, seed=2)
	models_dir = str(tmp_path / 'models')

	model = ptrain.TrainFundamentalFormPredictor(
		[train_file],
		num_epochs=2,
		learning_rate=1e-3,
		batch_size=32,
		warmup_steps=0,
		n_epochs_to_save=1,
		n_shapes_to_save=1,
		final_save_filepath='final',
		models_dir=models_dir,
		val_data_filename=val_file,
		plot_filepath=str(tmp_path / 'history.png'),
		n_queries_per_neighborhood=30,
		**TINY,
	)
	for name in ('FundamentalFormPredictor_0.pt', 'FundamentalFormPredictor_1.pt', 'final.pt'):
		assert os.path.isfile(os.path.join(models_dir, name))

	loaded, k = FundamentalFormPredictor.load_saved_model(os.path.join(models_dir, 'final.pt'))
	assert k == 16
	for a, b in zip(loaded.state_dict().values(), model.state_dict().values()):
		assert torch.equal(a, b)

	# warm start from the checkpoint (as the phases of train() do)
	ptrain.TrainFundamentalFormPredictor(
		[train_file],
		1,
		1e-4,
		32,
		0,
		1,
		10,
		load_model_path=os.path.join(models_dir, 'final.pt'),
		final_save_filepath='final2',
		models_dir=models_dir,
		val_data_filename=None,
		plot_filepath=None,
		n_queries_per_neighborhood=30,
		**TINY,
	)

	# the trained model drives inference
	data = synthetic_dataset(n_shapes=1, n_points=96, k=16, pad_points=0, seed=1)
	sdf = pat.PointsAsTori(data['positions'][0], data['normals'][0], model_path=os.path.join(models_dir, 'final.pt'))
	d = sdf.signed_distance(config.as_wide(np.zeros((4, 3)), 3))
	assert d.shape == (4,) and torch.isfinite(d).all()


def test_training_reduces_loss(cpu32):
	"""
	Repeated steps on one batch decrease the loss (sanity of gradients + optimizer wiring).
	Note: the same seed gives different initial weights in different torch versions (trunc_normal_ draws differ,
	measured 2.1.2 vs 2.14.1), so the trajectory is version dependent; a moderate learning rate keeps it monotone.
	"""
	model, optimizer, train_step = ptrain.FundamentalFormPredictorTrainer(
		**TINY, learning_rate=3e-4, warmup_steps=0, seed=0
	)
	s, q, d = _tiny_step_inputs(0)
	args = (s['mask'], s['mask'][:32], s['positions'], s['normals'], s['neighbors'], q[:, :32], d[:32])
	losses = [float(train_step(model, optimizer, *args)[0]) for _ in range(30)]
	print(f'loss: first {losses[0]:.4f}, last {losses[-1]:.4f}')
	assert losses[-1] < 0.95 * losses[0]


def _shape(pad_points, dtype_seed=3, n_points=60, k=16):
	data = synthetic_dataset(n_shapes=1, n_points=n_points, k=k, pad_points=pad_points, seed=dtype_seed)
	return ptrain.upload_shape(tuple(data[k] for k in KEYS), 0)


KEYS = ('positions', 'normals', 'neighbors', 'masks', 'mesh_vertices', 'mesh_faces', 'face_masks')


def test_fill_padding_step_equals_unpadded(cpu64):
	"""
	With fill_padding, one AdamW step on a padded shape gives the same loss and parameters as on the same shape
	without padding (padded points stay masked); without it, the gradient is all NaN (zeroed).
	"""
	s_unpadded, s_padded = _shape(0), _shape(8)
	rng = np.random.default_rng(1)
	q = torch.from_numpy(rng.uniform(-0.8, 0.8, size=(3, 32, 30)))
	d = torch.from_numpy(rng.normal(scale=0.2, size=(32, 30)))

	def one_step(s):
		model, optimizer, train_step = ptrain.FundamentalFormPredictorTrainer(
			**TINY, learning_rate=1e-3, warmup_steps=0
		)
		model.double()
		optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, eps=1e-8, weight_decay=0.0)
		optimizer.schedule, optimizer.update_count = (lambda count: 1e-3), 0
		loss, _ = train_step(
			model, optimizer, s['mask'], s['mask'][:32], s['positions'], s['normals'], s['neighbors'], q, d
		)
		grads = torch.cat([p.grad.flatten() for p in model.parameters()])
		return float(loss), torch.cat([p.detach().flatten() for p in model.parameters()]), grads

	l0, p0, g0 = one_step(s_unpadded)
	l1, p1, g1 = one_step(ptrain.fill_padding(s_padded))
	l2, p2, g2 = one_step(s_padded)
	assert l0 == pytest.approx(l1, rel=1e-14) and l0 == pytest.approx(l2, rel=1e-14)  # the loss never saw padding
	assert float((g1 - g0).abs().max() / g0.abs().max()) < 1e-14
	assert float((p1 - p0).abs().max()) < 1e-14
	assert bool((g2 == 0).all())  # original behavior: all-NaN gradient, zeroed by zero_out_nans
	# the first AdamW step with zero gradient does not move the parameters
	model0, _, _ = ptrain.FundamentalFormPredictorTrainer(**TINY)
	assert torch.equal(p2, torch.cat([p.detach().double().flatten() for p in model0.parameters()]))


def test_fill_padding_leaves_valid_points_and_validation_unchanged(cpu32):
	s = _shape(8)
	f = ptrain.fill_padding(s)
	m = s['mask']
	assert torch.equal(f['positions'][:, m], s['positions'][:, m]) and torch.equal(f['neighbors'][m], s['neighbors'][m])
	assert torch.equal(f['mask'], m)
	assert torch.equal(f['positions'][:, ~m], s['positions'][:, :1].expand(3, int((~m).sum())))
	assert (f['neighbors'] >= 0).all()

	data = synthetic_dataset(n_shapes=2, n_points=60, k=16, pad_points=8, seed=4)
	val = tuple(data[k] for k in KEYS)
	model, _, _ = ptrain.FundamentalFormPredictorTrainer(**TINY)
	model.eval()
	a = ptrain._validation(model, val, 0, 32, 30, 42, fix_padding=True)
	b = ptrain._validation(model, val, 0, 32, 30, 42, fix_padding=False)
	assert a == pytest.approx(b, rel=1e-6) and np.isfinite(a).all()


def test_fix_padding_option(cpu32, tmp_path):
	"""The option reaches the loop: on padded data the two settings train different weights."""
	train_file = save_synthetic_dataset(
		str(tmp_path / 'padded.npz'), n_shapes=2, n_points=60, k=16, pad_points=8, seed=6
	)
	common = dict(
		models_dir=str(tmp_path / 'models'),
		val_data_filename=None,
		plot_filepath=None,
		n_queries_per_neighborhood=30,
		**TINY,
	)
	fixed = ptrain.TrainFundamentalFormPredictor([train_file], 1, 1e-3, 32, 0, 1, 10, fix_padding=True, **common)
	original = ptrain.TrainFundamentalFormPredictor([train_file], 1, 1e-3, 32, 0, 1, 10, fix_padding=False, **common)
	init, _, _ = ptrain.FundamentalFormPredictorTrainer(**TINY)
	flat = lambda m: torch.cat([p.detach().flatten() for p in m.parameters()])  # noqa: E731
	moved_fixed = float((flat(fixed) - flat(init)).abs().max())
	moved_original = float((flat(original) - flat(init)).abs().max())
	print(f'max parameter change after one epoch: fix_padding=True {moved_fixed:.2e}, False {moved_original:.2e}')
	assert moved_fixed > 0
	assert moved_original == 0.0  # every shape is padded -> every gradient is zeroed; Adam moments stay 0
