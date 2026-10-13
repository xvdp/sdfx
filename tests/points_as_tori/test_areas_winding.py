"""
Point areas (igl::copyleft::cgal::point_areas), the area-based fields of TorusDistanceField (log-sum-exp variants,
winding numbers, fast winding number), neighbor subsampling, the exact-Laplacian option and isosurface sampling,
against the original C++ (pat_reference; point areas need it built with CGAL).
"""

import numpy as np
import pytest
import torch

import points_as_tori as pat
from points_as_tori.shape_2d import norm, subsample_point_cloud
from points_as_tori.torus_distance import torus_signed_distance_gradient
from helpers import random_tori, read_ply


def rel(a, b):
	a, b = np.asarray(a), np.asarray(b)
	return float(np.abs(a - b).max() / np.abs(b).max())


@pytest.fixture(scope='module')
def cgal(reference):
	if not getattr(reference, 'has_cgal', False):
		pytest.skip('pat_reference built without CGAL: point_areas is a stub')
	return reference


def clouds():
	P, N, _ = random_tori(1500, seed=3)
	tp, tn = read_ply('tooth.ply')
	return {'sphere': (P, N), 'tooth': (tp, tn)}


def queries(P, n=1200, seed=0):
	rng = np.random.default_rng(seed)
	near = P[rng.integers(0, len(P), n)] + 0.03 * rng.normal(size=(n, 3))
	return np.concatenate([near, rng.uniform(-1.2, 1.2, size=(n // 3, 3))])


@pytest.mark.parametrize('name', ['sphere', 'tooth'])
def test_point_areas(cgal, cpu64, name):
	P, N = clouds()[name]
	ref = cgal.TorusDistanceField(np.asfortranarray(P.T), np.asfortranarray(N.T)).get_point_areas()
	tdf = pat.TorusDistanceField(pat.as_wide(P, 3), pat.as_wide(N, 3))
	A = tdf.get_point_areas().numpy()
	print(f'\n{name}: point areas max |Δ| {np.abs(A - ref).max():.1e} (max area {ref.max():.1e})')
	np.testing.assert_allclose(A, ref, rtol=0, atol=1e-12 * ref.max())


def test_point_areas_grid_degenerate(cgal, cpu64):
	"""
	A regular grid: neighborhoods with cocircular points (decided by the exact fallback). Interior points match; at
	the grid border the 20th-nearest distance is tied and igl::knn (octree priority queue) breaks the tie
	differently, so the neighbor set, not the area computation, differs there.
	"""
	g = np.linspace(-1, 1, 30)
	X, Y = np.meshgrid(g, g)
	P = np.stack([X.ravel(), Y.ravel(), 0.1 * X.ravel() ** 2], 1)
	N = np.tile([0.0, 0.0, 1.0], (len(P), 1))
	ref = cgal.TorusDistanceField(np.asfortranarray(P.T), np.asfortranarray(N.T)).get_point_areas()
	A = pat.TorusDistanceField(pat.as_wide(P, 3), pat.as_wide(N, 3)).get_point_areas().numpy()
	err = np.abs(A - ref).reshape(X.shape)
	interior = err[3:-3, 3:-3]
	print(
		f'\ngrid: interior max |Δ| {interior.max():.1e}; {int((err > 1e-12).sum())} of {err.size} points differ, all within 2 rows of the border: {bool((err[2:-2, 2:-2] <= 1e-12).all())}'
	)
	assert interior.max() < 1e-12


@pytest.mark.parametrize('k', [32, -1])
def test_log_sum_exp_variants(cgal, cpu64, k):
	P, N = clouds()['tooth']
	Q = queries(P)
	ref = cgal.TorusDistanceField(np.asfortranarray(P.T), np.asfortranarray(N.T))
	ref.get_point_areas()
	tdf = pat.TorusDistanceField(pat.as_wide(P, 3), pat.as_wide(N, 3))
	Qw, Qf = pat.as_wide(Q, 3), np.asfortranarray(Q.T)
	for lam in (1.0, 0.5):
		ref.set_k_evaluation(k)
		tdf.set_k_evaluation(k)
		ref.set_lambda_scale(lam)
		tdf.set_lambda_scale(lam)
		for fn in (
			'signed_log_sum_exp',
			'signed_log_sum_exp_gradient',
			'self_normalized_signed_log_sum_exp',
			'self_normalized_signed_log_sum_exp_gradient',
			'log_sum_exp',
		):
			e = rel(getattr(tdf, fn)(Qw).numpy(), getattr(ref, fn)(Qf, True))
			print(f'k={k} λ_scale={lam} {fn}: {e:.1e}')
			assert e < 1e-11, fn


def test_winding_numbers(cgal, cpu64):
	P, N = clouds()['tooth']
	Q = queries(P, seed=1)
	ref = cgal.TorusDistanceField(np.asfortranarray(P.T), np.asfortranarray(N.T))
	ref.get_point_areas()
	tdf = pat.TorusDistanceField(pat.as_wide(P, 3), pat.as_wide(N, 3))
	Qw, Qf = pat.as_wide(Q, 3), np.asfortranarray(Q.T)
	assert rel(tdf.winding_number(Qw).numpy(), ref.winding_number(Qf, True)) < 1e-12
	assert rel(tdf.regularized_winding_number(Qw, 0.02).numpy(), ref.regularized_winding_number(Qf, 0.02, True)) < 1e-12
	ref.fast_winding_number_precompute()
	tdf.fast_winding_number_precompute()
	fast = tdf.fast_winding_number(Qw).numpy()
	fast_ref = ref.fast_winding_number(np.ascontiguousarray(Q))
	exact = -tdf.winding_number(Qw).numpy()  # libigl sign convention
	print(
		f'\nfast winding number vs C++ (libigl): {np.abs(fast - fast_ref).max():.1e}; Barnes-Hut vs exact sum: {np.abs(fast - exact).max():.1e}'
	)
	assert np.abs(fast - fast_ref).max() < 1e-10


def test_area_weighted_blend(cpu64):
	"""set_use_areas(True): weights w_i a_i in the blend (all modes), checked against the formula."""
	P, N, C = random_tori(600, seed=4)
	tdf = pat.TorusDistanceField(pat.as_wide(P, 3), pat.as_wide(N, 3))
	tdf.fit_tori_from_forms(pat.as_wide(C, 6))
	tdf.set_k_evaluation(-1)
	tdf.set_use_areas(True)
	q = pat.as_wide(np.random.default_rng(5).uniform(-1.3, 1.3, size=(100, 3)), 3)
	d = tdf.evaluate_distance(q)
	a = tdf.get_point_areas()
	rv = q[:, :, None] - tdf.points[:, None, :]
	r = norm(rv)
	shift = 0.5 * r.amax(-1)
	w = torch.exp(-(64 / shift)[:, None] * (r - shift[:, None])) * a[None, :]
	g, _ = torus_signed_distance_gradient(
		q[:, :, None],
		tdf.centers[:, None, :],
		tdf._axes3d[:, None, :],
		tdf.major_radii[None],
		tdf.minor_radii[None],
		False,
	)
	torch.testing.assert_close(d, (w * g).sum(-1) / w.sum(-1), rtol=1e-12, atol=1e-14)
	tdf.set_k_evaluation(16)
	d16, g16, l16 = tdf.evaluate_distance_gradient_laplacian(q)
	assert torch.isfinite(d16).all() and torch.isfinite(g16).all() and torch.isfinite(l16).all()


def test_exact_laplacian(cpu64):
	"""With the exact Δg of each torus, evaluate_laplacian equals the Laplacian of the blend (λ, shift frozen) by
	central differences; the original's (d-1)/|q - c| term (exact only for spheres) does not."""
	P, N, C = random_tori(400, seed=2)
	tdf = pat.TorusDistanceField(pat.as_wide(P, 3), pat.as_wide(N, 3))
	tdf.fit_tori_from_forms(pat.as_wide(C, 6))
	tdf.set_k_evaluation(-1)
	q = pat.as_wide(np.random.default_rng(1).uniform(-1.4, 1.4, size=(200, 3)), 3)
	d = norm(q[:, :, None] - tdf.points[:, None, :])
	shift = 0.5 * d.amax(-1)
	lam = 64 / shift

	def phi(x):
		dd = norm(x[:, :, None] - tdf.points[:, None, :])
		w = torch.exp(-lam[:, None] * (dd - shift[:, None]))
		g, _ = torus_signed_distance_gradient(
			x[:, :, None],
			tdf.centers[:, None, :],
			tdf._axes3d[:, None, :],
			tdf.major_radii[None],
			tdf.minor_radii[None],
			False,
		)
		return (w * g).sum(-1) / w.sum(-1)

	h = 1e-4
	fd = sum(phi(q + h * e[:, None]) + phi(q - h * e[:, None]) - 2 * phi(q) for e in torch.eye(3, dtype=q.dtype)) / h**2
	errs = {}
	for exact in (False, True):
		tdf.set_exact_laplacian(exact)
		errs[exact] = float(((tdf.evaluate_laplacian(q) - fd).abs() / fd.abs().max()).max())
	print(f'\nLaplacian vs finite differences: original {errs[False]:.1e}, exact Δg {errs[True]:.1e}')
	assert errs[True] < 1e-4 < errs[False]


def test_neighbors_and_subsample(reference, cpu64):
	"""Neighbors equal the C++; the subsample equals farthest-point sampling of each neighborhood (the C++ result
	itself depends on its OpenMP race, so the reference is subsample_point_cloud per neighborhood)."""
	P, N, C = random_tori(500, seed=6)
	tdf = pat.TorusDistanceField(pat.as_wide(P, 3), pat.as_wide(N, 3))
	ref = reference.TorusDistanceField(np.asfortranarray(P.T), np.asfortranarray(N.T))
	nb, sub = tdf.get_all_neighbors_and_subsample(24, 8)
	rnb, _ = ref.get_all_neighbors_and_subsample(24, 8)
	assert np.array_equal(nb.numpy(), rnb.T)
	for i in range(0, 500, 25):
		sel = subsample_point_cloud(tdf.points[:, nb[i]], 8)
		assert torch.equal(sub[i], nb[i, sel])


def test_sample_isosurface(cpu64):
	P, N, C = random_tori(800, seed=5)
	tdf = pat.TorusDistanceField(pat.as_wide(P, 3), pat.as_wide(N, 3))
	tdf.fit_tori_from_forms(pat.as_wide(C, 6))
	tdf.set_k_evaluation(32)
	loc, nrm = tdf.sample_isosurface(300, 0.0, seed=1)
	assert loc.shape == (3, 300) and nrm.shape == (3, 300)
	assert float(tdf.evaluate_distance(loc).abs().max()) < 1e-4
	g = tdf.evaluate_gradient(loc)
	torch.testing.assert_close(nrm, g / norm(g))
	loc2, _ = tdf.sample_isosurface(300, 0.0, seed=1)
	assert torch.equal(loc, loc2)  # reproducible (the C++ uses std::random_device)


def test_ray_intersections(cpu64):
	"""
	Tori fitted to the exact form of the unit sphere (φ = |x| - 1): a ray through the center hits at x = -1 and +1.
	(On fields with |∇φ| > 1 the |φ| steps of the original can step over a crossing; the port does the same.)
	"""
	rng = np.random.default_rng(0)
	P = rng.normal(size=(600, 3))
	P /= np.linalg.norm(P, axis=1, keepdims=True)
	coeffs = np.zeros((600, 6))
	coeffs[:, 4:] = -0.5
	tdf = pat.TorusDistanceField(pat.as_wide(P, 3), pat.as_wide(P, 3))
	tdf.fit_tori_from_forms(pat.as_wide(coeffs, 6))
	tdf.set_k_evaluation(16)
	ro = pat.as_wide(np.array([[-2.0, 0.0, 0.0], [0.0, -2.0, 0.0]]), 3)
	rd = pat.as_wide(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 3)
	hits = tdf.ray_intersections(ro, rd)
	assert hits.shape == (3, 4)
	assert float(tdf.evaluate_distance(hits).abs().max()) < 1e-4
	np.testing.assert_allclose(hits[0, :2].numpy(), [-1.0, 1.0], atol=1e-4)  # ray 0, in t order
	np.testing.assert_allclose(hits[1, 2:].numpy(), [-1.0, 1.0], atol=1e-4)  # ray 1
	single = tdf.ray_intersection_single(ro[:, 0], rd[:, 0], float(tdf.diameter))
	assert torch.equal(single, hits[:, :2])
