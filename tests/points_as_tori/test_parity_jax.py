"""
Parity of the torch port against the original JAX code (ext/points-as-tori/src/pointsastori) and the full original
pipeline (JAX network + C++ fitting/evaluation).

float64 checks validate the formulas (|Δ| / max|ref| <= 1e-11); float32 checks bound rounding of the network.
"""

import numpy as np
import pytest
import torch

import points_as_tori as pat
import points_as_tori.network as pnet
from points_as_tori import config
from helpers import FLAX_MODEL, TORCH_MODEL, fortran, read_ply


def rel_err(a, b):
	a = np.asarray(a, dtype=np.float64)
	b = np.asarray(b, dtype=np.float64)
	assert a.shape == b.shape, (a.shape, b.shape)
	return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def wide(x):
	"""(..., c) -> (c, ...) contiguous torch tensor"""
	return torch.from_numpy(np.array(np.moveaxis(np.asarray(x), -1, 0)))


def _random_fit_inputs(m=512, seed=0):
	rng = np.random.default_rng(seed)
	return rng.normal(size=(m, 3)), rng.normal(size=(m, 3)), rng.normal(scale=0.3, size=(m, 6)), rng


def test_formulas_float64(original, jax_x64):
	import jax.numpy as jnp

	onet = original.network
	p, n, c, rng = _random_fit_inputs()
	mask = np.ones(len(p), bool)

	ref = onet.torus_precompute_masked(jnp.asarray(mask), jnp.asarray(p), jnp.asarray(n), jnp.asarray(c))
	port = pnet.torus_precompute_masked(torch.from_numpy(mask), wide(p), wide(n), wide(c))
	assert rel_err(port[0].numpy().T, ref[0]) < 1e-11
	assert rel_err(port[1].numpy().T, ref[1]) < 1e-11
	assert rel_err(port[2].numpy(), ref[2]) < 1e-11
	assert rel_err(port[3].numpy(), ref[3]) < 1e-11

	centers, axes, R, r = [np.asarray(x) for x in ref]
	q = rng.uniform(-2, 2, size=(4, 200, 3))
	mask = rng.random(len(p)) > 0.1
	jd, jg = onet.torus_distance_gradient_masked(
		jnp.asarray(mask), jnp.asarray(q), jnp.asarray(p), *map(jnp.asarray, (centers, axes, R, r))
	)
	td, tg = pnet.torus_distance_gradient_masked(
		torch.from_numpy(mask),
		wide(q),
		wide(p),
		wide(centers),
		wide(axes),
		torch.from_numpy(R.copy()),
		torch.from_numpy(r.copy()),
	)
	assert rel_err(td.numpy(), jd) < 1e-11
	assert rel_err(np.moveaxis(tg.numpy(), 0, -1), jg) < 1e-11

	normals = rng.normal(size=(100, 3))
	for rotation in (0.0, 0.7, 2.5):
		js, jt = onet.orthonormal_basis_with_rotation(jnp.asarray(normals), rotation)
		ts, tt = pnet.orthonormal_basis_with_rotation(wide(normals), rotation)
		assert rel_err(ts.numpy().T, js) < 1e-12 and rel_err(tt.numpy().T, jt) < 1e-12

	u = rng.normal(size=(100, 3))
	assert (
		rel_err(pnet.cartesian_to_rotation_jax(wide(u)).numpy().T, onet.cartesian_to_rotation_jax(jnp.asarray(u)))
		< 1e-12
	)
	a = rng.uniform(-3, 3, size=(100, 2))
	assert (
		rel_err(pnet.rotation_to_cartesian_jax(wide(a)).numpy().T, onet.rotation_to_cartesian_jax(jnp.asarray(a)))
		< 1e-12
	)


def test_network_float32(cpu32, original, reference):
	"""Pre-trained weights converted from flax: coefficients agree to float32 rounding."""
	import jax.numpy as jnp

	points, normals = read_ply('tooth.ply')
	idx = reference.get_neighbors(fortran(points), 64, np.zeros(0, np.int32)).T  # identical neighborhoods
	p32, n32 = points.astype(np.float32), normals.astype(np.float32)

	jax_model, k_jax = original.network.FundamentalFormPredictor.load_saved_model(FLAX_MODEL)
	torch_model, k_torch = pnet.FundamentalFormPredictor.load_saved_model(TORCH_MODEL)
	assert k_jax == k_torch == idx.shape[1]

	ref = np.asarray(jax_model(jnp.asarray(p32[idx]), jnp.asarray(n32[idx])))  # (N, 6)
	with torch.inference_mode():
		port = torch_model(wide(p32)[:, torch.from_numpy(idx)], wide(n32)[:, torch.from_numpy(idx)]).numpy().T
	errs = [rel_err(port[:, i], ref[:, i]) for i in range(6)]
	print('network rel errors (a00, a01, a10, a11, a02, a20):', [f'{e:.1e}' for e in errs])
	assert max(errs) < 1e-5

	# the .pkl path converts on the fly to the same weights
	converted, _ = pnet.FundamentalFormPredictor.load_saved_model(FLAX_MODEL)
	for (k1, v1), (k2, v2) in zip(converted.state_dict().items(), torch_model.state_dict().items()):
		assert k1 == k2 and torch.equal(v1, v2)


@pytest.mark.parametrize('name', ['tooth.ply', 'snail.ply'])
def test_end_to_end(cpu64, original, name):
	"""
	Original PointsAsTori (JAX float32 network -> C++ float64 fit + evaluation) vs the port (torch float32 network ->
	float64 fit + evaluation). Differences come from float32 network rounding, amplified by 1/κ in the torus fit for
	nearly flat patches; with identical tori the evaluators agree to ~1e-15.
	"""
	points, normals = read_ply(name)
	orig = original.PointsAsTori(points, normals, model_path=FLAX_MODEL)
	port = pat.PointsAsTori(points, normals, model_path=TORCH_MODEL)

	rng = np.random.default_rng(0)
	lo, hi = points.min(0), points.max(0)
	ext = hi - lo
	queries = rng.uniform(lo - 0.2 * ext, hi + 0.2 * ext, size=(4000, 3))
	q = config.as_wide(queries, 3)

	for accelerate in (True, False):
		od = orig.signed_distance(queries, accelerate=accelerate)
		og = orig.sdf_gradient(queries, accelerate=accelerate)
		pd, pg = port.signed_distance_and_gradient(q, accelerate=accelerate)
		pd, pg = config.to_numpy(pd), config.to_numpy(pg)
		dist_err = np.max(np.abs(pd - od))
		grad_err = np.max(np.abs(pg - og))
		sign_agreement = np.mean(np.sign(pd) == np.sign(od))
		print(
			f'{name} accelerate={accelerate}: max|Δd|={dist_err:.2e} max|Δ∇d|={grad_err:.2e} sign agreement={sign_agreement}'
		)
		assert dist_err < 1e-6
		assert grad_err < 1e-3
		assert sign_agreement > 0.999

	# identical tori (original's) -> evaluator parity
	centers, axes, major_radii, minor_radii = orig._tdf.get_tori()
	port_same = pat.PointsAsTori(
		points, normals, tori={'centers': centers, 'axes': axes, 'major_radii': major_radii, 'minor_radii': minor_radii}
	)
	assert np.max(np.abs(config.to_numpy(port_same.signed_distance(q)) - orig.signed_distance(queries))) < 1e-12


def test_tori_file_interop(cpu64, original, tmp_path):
	"""save_tori files are interchangeable between the original and the port."""
	points, normals = read_ply('tooth.ply')
	orig = original.PointsAsTori(points, normals, model_path=FLAX_MODEL)
	orig_file = str(tmp_path / 'orig.pkl')
	orig.save_tori(orig_file)

	port = pat.PointsAsTori(points, normals, tori_filepath=orig_file)
	for p, o in zip(port.get_tori(), orig._tdf.get_tori()):
		assert np.array_equal(config.to_numpy(p), o)

	port_file = str(tmp_path / 'port.pkl')
	port.save_tori(port_file)
	orig2 = original.PointsAsTori(points, normals, tori_filepath=port_file)
	for a, b in zip(orig2._tdf.get_tori(), orig._tdf.get_tori()):
		assert np.array_equal(a, b)
