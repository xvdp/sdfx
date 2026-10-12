"""
Parity of the torch port against the original C++ evaluator (pat_reference build of src/cpp/torus_distance.cpp),
in float64. Metric: max |port - ref| / max |ref| over finite reference entries, and identical NaN/inf patterns.
"""

import numpy as np
import pytest
import torch

import points_as_tori as pat
from points_as_tori import config
from helpers import fortran, random_tori, read_ply

TOL = 1e-11  # measured: <= 6e-14 on x86-64 (see test output with -s)


def rel_err(a, b):
	a = np.asarray(a, dtype=np.float64)
	b = np.asarray(b, dtype=np.float64)
	assert a.shape == b.shape, (a.shape, b.shape)
	assert np.array_equal(np.isfinite(a), np.isfinite(b)), 'non-finite patterns differ'
	m = np.isfinite(b)
	if not m.any():
		return 0.0
	return float(np.max(np.abs(a[m] - b[m])) / max(np.max(np.abs(b[m])), 1e-300))


def test_fit_tori_from_forms(cpu64, reference):
	points, normals, coeffs = random_tori(3000)
	ref = reference.fit_tori_from_forms(fortran(points), fortran(normals), fortran(coeffs))
	port = pat.fit_tori_from_forms(config.as_wide(points, 3), config.as_wide(normals, 3), config.as_wide(coeffs, 6))
	for name, p, r in zip(('centers', 'axes', 'major_radii', 'minor_radii'), port, ref):
		err = rel_err(p.numpy(), r)
		print(f'fit {name}: {err:.2e}')
		assert err < TOL, name


def test_principal_curvatures(cpu64, reference):
	points, normals, coeffs = random_tori(1000, seed=3)
	ref_tdf = reference.TorusDistanceField(fortran(points), fortran(normals))
	tdf = pat.TorusDistanceField(config.as_wide(points, 3), config.as_wide(normals, 3))
	assert (
		rel_err(
			tdf.principal_curvatures(config.as_wide(coeffs, 6)).numpy(), ref_tdf.principal_curvatures(fortran(coeffs))
		)
		< TOL
	)
	for p, r in zip(tdf.principal_directions(config.as_wide(coeffs, 6)), ref_tdf.principal_directions(fortran(coeffs))):
		assert rel_err(p.numpy(), r) < TOL


@pytest.mark.parametrize('with_outliers', [False, True])
def test_get_neighbors(cpu64, reference, with_outliers):
	rng = np.random.default_rng(1)
	points = rng.normal(size=(3000, 3))
	outliers = rng.choice(3000, 50, replace=False).astype(np.int32) if with_outliers else np.zeros(0, np.int32)
	ref = reference.get_neighbors(fortran(points), 16, outliers).T
	port = pat.get_neighbors(config.as_wide(points, 3), 16, outliers if with_outliers else None).numpy()
	assert np.array_equal(port, ref)


def test_average_neighbor_distance(cpu64, reference):
	points, _ = read_ply('tooth.ply')
	# The C++ sums inside an OpenMP loop without a reduction (racy), so compare with a serial NumPy evaluation.
	idx = reference.get_neighbors(fortran(points), 64, np.zeros(0, np.int32)).T
	expected = np.linalg.norm(points[:, None, :] - points[idx], axis=-1).mean()
	got = pat.shape_3d.compute_optimal_radius(config.as_wide(points, 3), 64) / 128.0 * 1e3
	assert abs(float(got) - expected) / expected < 1e-12


MODES = {
	'knn32': dict(k=32, radius=-1.0, outliers=False, lambda_scale=1.0),
	'all_points': dict(k=-1, radius=-1.0, outliers=False, lambda_scale=1.0),
	'knn32_outliers': dict(k=32, radius=-1.0, outliers=True, lambda_scale=1.0),
	'knn8_lambda_half': dict(k=8, radius=-1.0, outliers=False, lambda_scale=0.5),
	'radius_with_knn_fallback': dict(k=32, radius=0.08, outliers=False, lambda_scale=1.0),
	'radius_outliers_k_all': dict(k=-1, radius=0.08, outliers=True, lambda_scale=1.0),
}


@pytest.mark.parametrize('mode', list(MODES))
def test_evaluate_distance_gradient_laplacian(cpu64, reference, mode):
	cfg = MODES[mode]
	points, normals = read_ply('tooth.ply')
	coeffs = np.random.default_rng(2).normal(scale=0.05, size=(len(points), 6))
	coeffs[:, 4:] -= 0.5

	ref_tdf = reference.TorusDistanceField(fortran(points), fortran(normals))
	ref_tdf.fit_tori_from_forms(fortran(coeffs))
	centers, axes, major_radii, minor_radii = ref_tdf.get_tori()

	# identical tori: isolates the evaluator
	tdf = pat.TorusDistanceField(config.as_wide(points, 3), config.as_wide(normals, 3))
	tdf.set_tori(
		torch.from_numpy(np.ascontiguousarray(centers)),
		torch.from_numpy(np.ascontiguousarray(axes)),
		torch.from_numpy(major_radii),
		torch.from_numpy(minor_radii),
	)

	outliers = np.arange(0, len(points), 7, dtype=np.int32) if cfg['outliers'] else np.zeros(0, np.int32)
	for obj in (ref_tdf, tdf):
		obj.set_k_evaluation(cfg['k'])
		obj.set_radius_evaluation(cfg['radius'])
		obj.set_lambda_scale(cfg['lambda_scale'])
		obj.set_outliers(outliers)

	rng = np.random.default_rng(0)
	lo, hi = points.min(0), points.max(0)
	ext = hi - lo
	queries = rng.uniform(lo - 0.25 * ext, hi + 0.25 * ext, size=(3000, 3))
	if cfg['radius'] > 0:
		# make sure both radius branches are exercised: some queries with and some without points within the radius
		n_inside = (np.linalg.norm(queries[:, None] - points[None], axis=-1) < cfg['radius']).sum(1)
		assert (n_inside == 0).any() and (n_inside > 0).any()

	q = config.as_wide(queries, 3)
	d, g, lap = ref_tdf.evaluate_distance_gradient_laplacian(fortran(queries), 0.0, True)
	pd, pg, plap = tdf.evaluate_distance_gradient_laplacian(q)
	errs = dict(
		distance=rel_err(pd.numpy(), d),
		gradient=rel_err(pg.numpy(), g),
		laplacian=rel_err(plap.numpy(), lap),
		distance_only=rel_err(
			tdf.evaluate_distance(q, 0.1).numpy(), ref_tdf.evaluate_distance(fortran(queries), 0.1, True)
		),
	)
	print(mode, {k: f'{v:.2e}' for k, v in errs.items()})
	for name, err in errs.items():
		assert err < TOL, name


def test_single_torus_queries(cpu64, reference):
	points, normals, coeffs = random_tori(10)
	ref_tdf = reference.TorusDistanceField(fortran(points), fortran(normals))
	tdf = pat.TorusDistanceField(config.as_wide(points, 3), config.as_wide(normals, 3))
	queries = np.random.default_rng(5).normal(size=(200, 3))
	center, axis = np.array([0.1, -0.2, 0.3]), np.array([0.4, 1.1])
	rd, rg = ref_tdf.torus_signed_distances_gradients(fortran(queries), center, axis, 0.7, -0.2)
	pd, pg = tdf.torus_signed_distances_gradients(config.as_wide(queries, 3), center, axis, 0.7, -0.2)
	assert rel_err(pd.numpy(), rd) < TOL
	assert rel_err(pg.numpy(), rg) < TOL
	assert rel_err(tdf.torus_signed_distances(config.as_wide(queries, 3), center, axis, 0.7, -0.2).numpy(), rd) < TOL
