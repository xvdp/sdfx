"""
Device discipline, layout and shape contracts of the port (no reference implementation needed).

No host sync: the whole pipeline is run on torch's `meta` device, where tensors carry no data. Any operation that
needs tensor values on the host (.item(), .tolist(), .cpu(), nonzero / boolean-mask indexing, `if tensor:`) raises
there, so passing these tests shows the computation is free of data-dependent host round trips. On CUDA the same
property can be checked with torch.cuda.set_sync_debug_mode('error').
"""

import numpy as np
import pytest
import torch

import points_as_tori as pat
from points_as_tori import config
from helpers import TORCH_MODEL, random_tori, read_ply


@pytest.fixture
def meta():
	config.set_device('meta', torch.float32, force=True)
	yield
	config.set_device('cpu', torch.float32, force=True)


def test_pipeline_on_meta_device(meta):
	points, normals = read_ply('tooth.ply')
	sdf = pat.PointsAsTori(points, normals, model_path=TORCH_MODEL)
	q = torch.empty(3, 5000, device='meta')
	d = sdf.signed_distance(q)
	g = sdf.sdf_gradient(q, accelerate=False)
	d2, g2 = sdf.signed_distance_and_gradient(q)
	assert d.device.type == 'meta' and d.shape == (5000,) and g.shape == (3, 5000)
	assert d2.shape == (5000,) and g2.shape == (3, 5000)

	tdf = sdf._tdf
	tdf.set_outliers(np.arange(0, len(points), 5))
	tdf.set_k_evaluation(16)
	tdf.set_radius_evaluation(0.05)
	dist, grad, lap = tdf.evaluate_distance_gradient_laplacian(q)
	assert dist.shape == (5000,) and grad.shape == (3, 5000) and lap.shape == (5000,)

	assert pat.get_neighbors(config.as_wide(points, 3), 16, outliers=np.arange(10)).shape == (len(points), 16)
	assert pat.compute_optimal_radius(config.as_wide(points, 3)).shape == ()


def test_default_device(monkeypatch):
	"""Unset: get_device() picks cuda if torch sees one, else cpu (float32); an explicit set_device replaces it."""
	monkeypatch.setattr(config, '_DEVICE', None)
	monkeypatch.setattr(config, '_DTYPE', torch.float32)
	monkeypatch.setattr(config, '_DEFAULTED', False)
	assert config.default_device() == ('cuda' if torch.cuda.is_available() else 'cpu')
	with monkeypatch.context() as m:  # the rule itself, also without a GPU here
		m.setattr(torch.cuda, 'is_available', lambda: True)
		assert config.default_device() == 'cuda'
		m.setattr(torch.cuda, 'is_available', lambda: False)
		assert config.default_device() == 'cpu'
	assert config.get_device().type == config.default_device() and config.get_dtype() == torch.float32
	config.set_device('meta', torch.float64)  # replaces the default without force
	assert config.get_device().type == 'meta' and config.get_dtype() == torch.float64
	with pytest.raises(RuntimeError, match='already set'):
		config.set_device('cpu')  # an explicit choice stays


def test_device_set_once(monkeypatch):
	monkeypatch.setattr(config, '_DEVICE', None)
	monkeypatch.setattr(config, '_DEFAULTED', False)
	config.set_device('cpu')
	config.set_device('cpu')  # same device: fine
	with pytest.raises(RuntimeError, match='already set'):
		config.set_device('meta')


def test_no_implicit_transfer(cpu64):
	points, normals, coeffs = random_tori(50)
	tdf = pat.TorusDistanceField(config.as_wide(points, 3), config.as_wide(normals, 3))
	tdf.fit_tori_from_forms(config.as_wide(coeffs, 6))
	with pytest.raises(ValueError, match='expected'):
		tdf.evaluate_distance(torch.zeros(3, 4, device='meta', dtype=torch.float64))
	with pytest.raises(TypeError):
		tdf.evaluate_distance(np.zeros((3, 4)))


def test_layout_and_shapes(cpu64):
	points, normals, coeffs = random_tori(200)
	P = config.as_wide(points, 3)
	assert P.shape == (3, 200) and P.is_contiguous()
	assert np.array_equal(config.to_numpy(P), points)

	tdf = pat.TorusDistanceField(P, config.as_wide(normals, 3))
	centers, axes, R, r = tdf.fit_tori_from_forms(config.as_wide(coeffs, 6))
	assert centers.shape == (3, 200) and axes.shape == (2, 200) and R.shape == (200,) and r.shape == (200,)
	assert centers.is_contiguous() and axes.is_contiguous()

	tdf.set_k_evaluation(32)
	q = config.as_wide(np.random.default_rng(0).normal(size=(37, 3)), 3)
	d, g, lap = tdf.evaluate_distance_gradient_laplacian(q)
	assert d.shape == (37,) and g.shape == (3, 37) and lap.shape == (37,)
	assert g.is_contiguous()

	# a single (3,) query
	d0, g0, l0 = tdf.evaluate_distance_gradient_laplacian(q[:, 0])
	assert d0.shape == () and g0.shape == (3,) and l0.shape == ()
	assert torch.allclose(d0, d[0]) and torch.allclose(g0, g[:, 0]) and torch.allclose(l0, lap[0])


@pytest.mark.parametrize('k', [32, -1])
def test_chunking_invariance(cpu64, monkeypatch, k):
	points, normals, coeffs = random_tori(300)
	tdf = pat.TorusDistanceField(config.as_wide(points, 3), config.as_wide(normals, 3))
	tdf.fit_tori_from_forms(config.as_wide(coeffs, 6))
	tdf.set_k_evaluation(k)
	q = config.as_wide(np.random.default_rng(0).normal(size=(1000, 3)), 3)
	full = tdf.evaluate_distance_gradient_laplacian(q)
	monkeypatch.setattr(config, 'CHUNK_ELEMENTS', 700)  # ~2 queries per chunk
	chunked = tdf.evaluate_distance_gradient_laplacian(q)
	for a, b in zip(full, chunked):
		torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-13)


def test_gradient_is_the_frozen_lambda_gradient(cpu64):
	"""
	The closed-form gradient (C++ and port) differentiates φ = Σ w_i g_i / Σ w_i, w_i = exp(-λ(|q - p_i| - shift)),
	with λ held fixed. The shift cancels in the ratio, but λ = 64/shift and shift = ½ max_j |q - p_j| depend on q:

		∇φ_true = ∇φ_frozen + (∂φ/∂λ) ∇λ,   ∂φ/∂λ = -Cov_w(d, g) = -(Σ w d g / W - φ Σ w d / W),
		∇λ = -(λ / shift) ∇shift = -(λ / shift) · ½ r̂_far   (r̂_far: unit vector from the farthest neighbor to q)

	Checked against central differences: frozen-λ alone differs at the 1e-2 level, adding the λ term closes the gap
	to finite-difference accuracy. (Evaluated with all points as neighbors, so the neighbor set does not change.)
	"""
	from points_as_tori.shape_2d import norm
	from points_as_tori.torus_distance import torus_signed_distance_gradient

	points, normals, coeffs = random_tori(300, seed=4)
	tdf = pat.TorusDistanceField(config.as_wide(points, 3), config.as_wide(normals, 3))
	tdf.fit_tori_from_forms(config.as_wide(coeffs, 6))
	tdf.set_k_evaluation(-1)
	q = config.as_wide(np.random.default_rng(1).uniform(-1.5, 1.5, size=(64, 3)), 3)
	frozen = tdf.evaluate_gradient(q)

	eps = 1e-6
	fd = torch.stack(
		[
			(tdf.evaluate_distance(q + eps * e[:, None]) - tdf.evaluate_distance(q - eps * e[:, None])) / (2 * eps)
			for e in torch.eye(3, dtype=q.dtype)
		],
		dim=0,
	)

	rVec = q[:, :, None] - tdf.points[:, None, :]
	d = norm(rVec)
	shift = 0.5 * d.amax(dim=-1)
	lam = 64.0 / shift
	w = torch.exp(-lam[:, None] * (d - shift[:, None]))
	W = w.sum(dim=-1)
	g, _ = torus_signed_distance_gradient(
		q[:, :, None],
		tdf.centers[:, None, :],
		tdf._axes3d[:, None, :],
		tdf.major_radii[None],
		tdf.minor_radii[None],
		False,
	)
	phi = (w * g).sum(dim=-1) / W
	dphi_dlam = -((w * d * g).sum(dim=-1) / W - phi * (w * d).sum(dim=-1) / W)
	cols = torch.arange(q.shape[1])
	far = d.argmax(dim=-1)
	grad_lam = -(lam / shift) * 0.5 * rVec[:, cols, far] / d[cols, far]
	full = frozen + dphi_dlam * grad_lam

	def rel(a, b):
		return float((a - b).abs().max() / b.abs().max())

	print(f'frozen-λ gradient vs central differences: {rel(frozen, fd):.2e}; with the λ term: {rel(full, fd):.2e}')
	assert rel(frozen, fd) > 1e-4  # the frozen-λ approximation is measurable ...
	assert rel(full, fd) < 1e-7  # ... and fully accounted for by the λ term


def test_default_dtype_is_float32(monkeypatch):
	"""float32 unless set_device(..., torch.float64) is requested explicitly."""
	monkeypatch.setattr(config, '_DEVICE', None)
	monkeypatch.setattr(config, '_DTYPE', torch.float32)
	config.set_device('cpu')
	assert config.get_dtype() == torch.float32
	points, normals, coeffs = random_tori(100)  # float64 NumPy inputs
	P = config.as_wide(points, 3)
	assert P.dtype == torch.float32
	tdf = pat.TorusDistanceField(P, config.as_wide(normals, 3))
	tori = tdf.fit_tori_from_forms(config.as_wide(coeffs, 6))
	tdf.set_k_evaluation(16)
	outputs = (*tori, *tdf.evaluate_distance_gradient_laplacian(config.as_wide(np.zeros((5, 3)), 3)))
	assert all(t.dtype == torch.float32 for t in outputs)

	config.set_device('cpu', torch.float64, force=True)
	assert config.as_wide(points, 3).dtype == torch.float64
	config.set_device('cpu', torch.float32, force=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_cuda_device_has_index():
	"""set_device('cuda') must resolve to an indexed device ('cuda:0'), the device tensors report; otherwise every
	device check (tensor.device != get_device()) fails. (CPU and meta devices have no index, so CPU tests cannot see it.)"""
	config.set_device('cuda', torch.float32, force=True)
	try:
		assert config.get_device().index is not None
		assert torch.zeros(1, device=config.get_device()).device == config.get_device()
		points, normals, coeffs = random_tori(200)
		tdf = pat.TorusDistanceField(config.as_wide(points, 3), config.as_wide(normals, 3))
		tdf.fit_tori_from_forms(config.as_wide(coeffs, 6))
		assert tdf.evaluate_distance(config.as_wide(np.zeros((4, 3)), 3)).device == config.get_device()
	finally:
		config.set_device('cpu', torch.float32, force=True)
