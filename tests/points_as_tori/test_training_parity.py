"""
Parity of the training port against the original JAX training code (training/train.py, training/mesh_sampling_jax.py).

* Random sampling: JAX keys cannot be reproduced by a torch generator, so the exact JAX uniform draws are replayed
  through points_as_tori.training.mesh_sampling._uniform (same draws, same order) and results compared value by value.
* float64 network comparisons give the JAX model an exact float64 attention: flax.nnx.MultiHeadAttention (dropout 0)
  calls jax.nn.dot_product_attention, whose XLA path casts the logits to float32 before the softmax
  (jax/_src/nn/functions.py), which limits float64 agreement to ~1e-7. The torch attention matches a NumPy float64
  reference to ~6e-16. The float32 tests use the original unmodified.
"""

import pickle

import numpy as np
import pytest
import torch

from points_as_tori import config
from points_as_tori.network import convert_flax_params
from points_as_tori.training import mesh_sampling as pms
from points_as_tori.training import train as ptrain
from helpers import FLAX_MODEL, TORCH_MODEL, synthetic_dataset

ARCH = dict(embed_dim=128, n_freq=0, n_layers=8, n_heads=8, mlp_dim=512)  # the pre-trained model


def wide(x, dtype=None):
	t = torch.from_numpy(np.array(np.moveaxis(np.asarray(x), -1, 0)))
	return t if dtype is None else t.to(dtype)


def rel_err(a, b):
	a = np.asarray(a, dtype=np.float64)
	b = np.asarray(b, dtype=np.float64)
	assert a.shape == b.shape, (a.shape, b.shape)
	return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def jax_draws(key, n_points, n, dtype):
	"""
	The uniform draws of mesh_sampling_jax.generate_shape_features(key), in the order the torch port requests them:
	zero set (choice, u, v), narrow band (choice, u, v, offset), extended bbox ((n, 3)), each stacked over points.
	"""
	import jax

	calls = [[] for _ in range(8)]
	for k_point in jax.random.split(key, n_points):
		_, key_zero, key_narrow, key_bbox = jax.random.split(k_point, 4)
		k, sub = jax.random.split(key_zero)
		calls[0].append(jax.random.uniform(sub, (n,), dtype))
		k, s1, s2 = jax.random.split(k, 3)
		calls[1].append(jax.random.uniform(s1, (n,), dtype))
		calls[2].append(jax.random.uniform(s2, (n,), dtype))
		k, sub = jax.random.split(key_narrow)
		k2, sub2 = jax.random.split(sub)
		calls[3].append(jax.random.uniform(sub2, (n,), dtype))
		k2, s1, s2 = jax.random.split(k2, 3)
		calls[4].append(jax.random.uniform(s1, (n,), dtype))
		calls[5].append(jax.random.uniform(s2, (n,), dtype))
		k, sub = jax.random.split(k)
		calls[6].append(jax.random.uniform(sub, (n,), dtype))
		calls[7].append(jax.random.uniform(key_bbox, (n, 3), dtype))
	return [torch.from_numpy(np.array(np.stack(c))) for c in calls]


def replay(monkeypatch, draws):
	it = iter(draws)

	def _uniform(shape, generator, dtype, device):
		x = next(it)
		assert tuple(x.shape) == tuple(shape), (x.shape, shape)
		return x.to(device=device, dtype=dtype)

	monkeypatch.setattr(pms, '_uniform', _uniform)


def test_schedule_matches_optax(original_training):
	"""
	optax evaluates the schedule in float32; the port in float64. Two forms cancel digits in float32: the warmup
	(init - peak) * (1 - t/W) + peak near t = 0 (at t = 1, W = 1000: ~2e-5 relative; measured maximum over the
	original's five phase schedules: 1.97e-5), and 1 + cos(π t/T) near t = T (short schedules only).
	"""
	import optax

	for lr, warmup, total in [(1e-3, 0, 1e8), (1e-5, 1e2, 1e8), (1e-3, 10, 200), (1e-6, 1e3, 1e8), (1e-7, 1e3, 1e8)]:
		ref = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, total, lr * 0.01)
		port = ptrain.warmup_cosine_decay_schedule(0.0, lr, warmup, total, lr * 0.01)
		for step in [0, 1, 5, 9, 10, 11, 99, 100, 101, 150, 199, 200, 500, 999, 1000, 1001, 5000]:
			assert port(step) == pytest.approx(float(ref(step)), rel=5e-5, abs=1e-15), (lr, warmup, total, step)


def test_mesh_geometry_float64(original_training, jax_x64, cpu64):
	import jax
	import jax.numpy as jnp

	oms = original_training.mesh_sampling
	rng = np.random.default_rng(0)
	tri = rng.normal(size=(500, 3, 3))
	tri[:20, 2] = tri[:20, 0]  # degenerate triangles
	q = rng.normal(size=(500, 3))
	v0, v1, v2 = (tri[:, i] for i in range(3))
	ref = jax.vmap(oms.closest_point_on_triangle)(*map(jnp.asarray, (q, v0, v1, v2)))
	assert rel_err(pms.closest_point_on_triangle(*map(wide, (q, v0, v1, v2))).numpy().T, ref) < 1e-13
	ref = jax.vmap(oms.gwn_single_face)(*map(jnp.asarray, (q, v0, v1, v2)))
	assert rel_err(pms.gwn_single_face(*map(wide, (q, v0, v1, v2))).numpy(), ref) < 1e-13

	d = synthetic_dataset(n_shapes=1, n_points=150, k=16)
	V, F, FM = d['mesh_vertices'][0], d['mesh_faces'][0], d['face_masks'][0]
	tV, tF, tFM = wide(V), wide(F).long(), torch.from_numpy(FM)
	for p, r in zip(
		pms.compute_face_properties(tV, tF, tFM), oms.compute_face_properties(*map(jnp.asarray, (V, F, FM)))
	):
		assert rel_err(p.numpy().T if p.ndim == 2 else p.numpy(), r) < 1e-13

	queries = rng.uniform(-1, 1, size=(700, 3))
	ref = oms.signed_distance_to_mesh(*map(jnp.asarray, (queries, V, F, FM)))
	assert (np.asarray(ref) < 0).sum() > 10  # some queries inside
	assert rel_err(pms.signed_distance_to_mesh(wide(queries), tV, tF, tFM).numpy(), ref) < 1e-13
	ref = oms.unsigned_distance_to_mesh(*map(jnp.asarray, (queries, V, F, FM)))
	assert rel_err(pms.unsigned_distance_to_mesh(wide(queries), tV, tF, tFM).numpy(), ref) < 1e-13
	ref = oms.gwn_mesh(*map(jnp.asarray, (queries, V, F, FM)))
	assert rel_err(pms.gwn_mesh(wide(queries), tV, tF, tFM).numpy(), ref) < 1e-13


@pytest.mark.parametrize('x64', [True, False])
def test_generate_shape_features_replayed(original_training, monkeypatch, x64):
	"""Same random draws -> same queries and ground-truth distances (float64: to rounding; float32: to float32 rounding)."""
	import jax
	import jax.numpy as jnp

	jax.config.update('jax_enable_x64', x64)
	try:
		jdt, tdt = (jnp.float64, torch.float64) if x64 else (jnp.float32, torch.float32)
		config.set_device('cpu', tdt, force=True)
		d = synthetic_dataset(n_shapes=1, n_points=150, k=16, pad_points=8)
		pos, nb, V, F, FM = (
			d['positions'][0],
			d['neighbors'][0],
			d['mesh_vertices'][0],
			d['mesh_faces'][0],
			d['face_masks'][0],
		)
		key = jax.random.PRNGKey(7)
		n_queries = 120
		jq, jd = original_training.mesh_sampling.generate_shape_features(
			jnp.asarray(pos, jdt),
			jnp.asarray(nb),
			jnp.asarray(V, jdt),
			jnp.asarray(F),
			jnp.asarray(FM),
			key,
			n_queries=n_queries,
		)
		replay(monkeypatch, jax_draws(key, pos.shape[0], n_queries // 3, jdt))
		tq, td = pms.generate_shape_features(
			wide(pos, tdt),
			torch.from_numpy(nb).long(),
			wide(V, tdt),
			wide(F).long(),
			torch.from_numpy(FM),
			None,
			n_queries=n_queries,
		)
		eq, ed = rel_err(np.moveaxis(tq.numpy(), 0, -1), jq), rel_err(td.numpy(), jd)
		print(f'x64={x64}: queries {eq:.1e}, distances {ed:.1e}')
		tol = 1e-13 if x64 else 1e-5
		assert eq < tol and ed < tol
	finally:
		jax.config.update('jax_enable_x64', False)


def _exact_attention(query, key, value, *args, **kwargs):
	import jax
	import jax.numpy as jnp

	w = jnp.einsum('...qhd,...khd->...hqk', query / jnp.sqrt(query.shape[-1]).astype(query.dtype), key)
	return jnp.einsum('...hqk,...khd->...qhd', jax.nn.softmax(w, axis=-1), value)


def _jax_model_float64(original_training, model=None):
	import jax
	import jax.numpy as jnp
	from flax import nnx

	if model is None:
		model, _ = original_training.train.FundamentalFormPredictor.load_saved_model(FLAX_MODEL)
	else:
		with open(FLAX_MODEL, 'rb') as f:
			nnx.update(model, pickle.load(f)['params'])
	nnx.update(model, jax.tree.map(lambda x: x.astype(jnp.float64), nnx.state(model, nnx.Param)))
	for block in model.blocks:
		block.attention.attention_fn = _exact_attention
	return model


def _flat(state):
	from flax import nnx

	return convert_flax_params(
		{
			'/'.join(map(str, k)): np.asarray(v.get_value() if hasattr(v, 'get_value') else v.value)
			for k, v in nnx.to_flat_state(state)
		}
	)


def _batch_inputs(pad_points=0, seed=3, batch_size=64, n_queries=30):
	d = synthetic_dataset(n_shapes=1, n_points=160, k=64, pad_points=pad_points, seed=seed)
	rng = np.random.default_rng(1)
	batches = [
		(rng.uniform(-0.8, 0.8, size=(batch_size, n_queries, 3)), rng.normal(scale=0.2, size=(batch_size, n_queries)))
		for _ in range(4)
	]
	return d['positions'][0], d['normals'][0], d['neighbors'][0], d['masks'][0], batches


def test_sdf_loss_and_gradients_float64(original_training, jax_x64, cpu64):
	"""Loss and all 1.59M parameter gradients of the pre-trained model, through the network and the torus fit."""
	import jax.numpy as jnp
	from flax import nnx

	pos, nrm, nb, mask, batches = _batch_inputs()
	q, td = batches[0]
	bm = mask[:64]

	jm = _jax_model_float64(original_training)
	nnx.train_mode(jm)

	def loss_fn(model):
		c = model(jnp.asarray(pos)[nb], jnp.asarray(nrm)[nb])
		return original_training.train.SDF_loss(
			jnp.asarray(mask), jnp.asarray(bm), jnp.asarray(pos), jnp.asarray(nrm), c, jnp.asarray(q), jnp.asarray(td)
		)

	(jl, jld), jgrads = nnx.value_and_grad(loss_fn, has_aux=True)(jm)
	jg = _flat(jgrads)

	from points_as_tori.network import FundamentalFormPredictor

	tm, _ = FundamentalFormPredictor.load_saved_model(TORCH_MODEL)
	tm = tm.double().train()
	tnb = torch.from_numpy(nb).long()
	c = tm(wide(pos)[:, tnb], wide(nrm)[:, tnb])
	tl, tld = ptrain.SDF_loss(
		torch.from_numpy(mask), torch.from_numpy(bm), wide(pos), wide(nrm), c, wide(q), torch.from_numpy(td)
	)
	tl.backward()

	assert abs(float(tl) - float(jl)) / abs(float(jl)) < 1e-12
	for k in ('distance_loss', 'eikonality_loss'):
		assert abs(float(tld[k]) - float(jld[k])) <= 1e-12 * abs(float(jld[k]))
	grad_scale = max(float(v.abs().max()) for v in jg.values())
	grad_err = max(float((p.grad - jg[k]).abs().max()) for k, p in tm.named_parameters())
	print(f'loss {float(tl):.15g}; gradients: max|Δ| / max|g| = {grad_err / grad_scale:.1e}')
	assert grad_err / grad_scale < 1e-10


def test_train_steps_float64(original_training, jax_x64, cpu64):
	"""
	Original trainer (nnx.Optimizer + optax.adamw + schedule + NaN zeroing) vs port, four steps from the same weights.
	The first update agrees to rounding; afterwards Adam amplifies rounding differences of gradients near the ε scale
	(∂/∂g [g / (|g| + ε)] = ε / (|g| + ε)², up to 1/(4ε) · lr), so later steps agree to ~1e-8.
	"""
	import jax.numpy as jnp
	import optax
	from flax import nnx

	lr, warmup, total = 1e-3, 1, 100
	jm, _, jstep = original_training.train.FundamentalFormPredictorTrainer(
		**ARCH, learning_rate=lr, warmup_steps=warmup, total_steps=total
	)
	jm = _jax_model_float64(original_training, jm)
	schedule = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, total, lr * 0.01)
	jopt = nnx.Optimizer(jm, optax.chain(optax.adamw(learning_rate=schedule, weight_decay=0.0)), wrt=nnx.Param)
	nnx.train_mode(jm)

	tm, topt, tstep = ptrain.FundamentalFormPredictorTrainer(
		**ARCH, learning_rate=lr, warmup_steps=warmup, total_steps=total
	)
	tm.load_state_dict(torch.load(TORCH_MODEL, weights_only=True)['state_dict'])
	tm.double()
	topt = torch.optim.AdamW(tm.parameters(), lr=0.0, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
	topt.schedule = ptrain.warmup_cosine_decay_schedule(0.0, lr, warmup, total, lr * 0.01)
	topt.update_count = 0
	tm.train()

	pos, nrm, nb, mask, batches = _batch_inputs()
	bm = mask[:64]
	for step, (q, td) in enumerate(batches):
		jl, _ = jstep(jm, jopt, *map(jnp.asarray, (mask, bm, pos, nrm, nb, q, td)))
		tl, _ = tstep(
			tm,
			topt,
			torch.from_numpy(mask),
			torch.from_numpy(bm),
			wide(pos),
			wide(nrm),
			torch.from_numpy(nb).long(),
			wide(q),
			torch.from_numpy(td),
		)
		jp = _flat(nnx.state(jm, nnx.Param))
		param_err = max(float((p.detach() - jp[k]).abs().max()) for k, p in tm.named_parameters())
		loss_err = abs(float(tl) - float(jl)) / abs(float(jl))
		print(f'step {step}: loss rel {loss_err:.1e}, params max|Δ| {param_err:.1e}')
		assert loss_err < (1e-12 if step < 2 else 1e-6)
		assert param_err < (1e-9 if step < 2 else 1e-6)


def test_train_steps_float32(original_training, cpu32):
	"""The original as is (float32, its own attention) vs the port: losses over four AdamW steps."""
	import jax.numpy as jnp
	from flax import nnx

	jm, jopt, jstep = original_training.train.FundamentalFormPredictorTrainer(
		**ARCH, learning_rate=1e-4, warmup_steps=0, total_steps=1e8
	)
	with open(FLAX_MODEL, 'rb') as f:
		nnx.update(jm, pickle.load(f)['params'])
	nnx.train_mode(jm)
	tm, topt, tstep = ptrain.FundamentalFormPredictorTrainer(
		**ARCH, learning_rate=1e-4, warmup_steps=0, total_steps=1e8
	)
	tm.load_state_dict(torch.load(TORCH_MODEL, weights_only=True)['state_dict'])
	tm.train()

	pos, nrm, nb, mask, batches = _batch_inputs()
	bm = mask[:64]
	f32 = lambda x: jnp.asarray(x, jnp.float32) if np.asarray(x).dtype.kind == 'f' else jnp.asarray(x)  # noqa: E731
	for step, (q, td) in enumerate(batches):
		jl, _ = jstep(jm, jopt, *map(f32, (mask, bm, pos, nrm, nb, q, td)))
		tl, _ = tstep(
			tm,
			topt,
			torch.from_numpy(mask),
			torch.from_numpy(bm),
			wide(pos, torch.float32),
			wide(nrm, torch.float32),
			torch.from_numpy(nb).long(),
			wide(q, torch.float32),
			torch.from_numpy(td).float(),
		)
		loss_err = abs(float(tl) - float(jl)) / abs(float(jl))
		print(f'float32 step {step}: loss jax {float(jl):.7f} torch {float(tl):.7f} rel {loss_err:.1e}')
		assert loss_err < 1e-4


def test_padded_gradients_are_nan_in_the_original(original_training, cpu32):
	"""The all-NaN gradient on padded shapes (test_training.test_padded_shapes_give_nan_gradients) is the original's."""
	import jax.numpy as jnp
	from flax import nnx

	jm, _ = original_training.train.FundamentalFormPredictor.load_saved_model(FLAX_MODEL)
	pos, nrm, nb, mask, batches = _batch_inputs(pad_points=8)
	q, td = batches[0]

	def loss_fn(model):
		c = model(jnp.asarray(pos, jnp.float32)[nb], jnp.asarray(nrm, jnp.float32)[nb])
		return original_training.train.SDF_loss(
			jnp.asarray(mask),
			jnp.asarray(mask[:64]),
			jnp.asarray(pos, jnp.float32),
			jnp.asarray(nrm, jnp.float32),
			c,
			jnp.asarray(q, jnp.float32),
			jnp.asarray(td, jnp.float32),
		)

	(loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(jm)
	g = np.concatenate([np.asarray(v).ravel() for v in _flat(grads).values()])
	assert np.isfinite(float(loss))
	assert np.isnan(g).all()
