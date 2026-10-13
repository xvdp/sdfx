"""
BVH (points_as_tori/bvh.py, port of src/cpp/bvh.cpp with an LBVH build and batched traversal) vs the original C++ BVH
and brute force; the k-NN backends of the evaluator.
"""

import numpy as np
import pytest
import torch

import points_as_tori as pat
from points_as_tori import config
from points_as_tori.bvh import BVH, BoundingVolumeHierarchy, intersect_sphere
from points_as_tori.torus_distance import get_neighbors
from helpers import random_tori, torus_mesh


def cloud(n=4000, seed=0):
	rng = np.random.default_rng(seed)
	P = rng.normal(size=(n, 3))
	P[: n // 3] *= 0.05  # dense cluster inside a sparse cloud
	return P


@pytest.fixture
def knn_backend():
	saved = config.KNN_BACKEND
	yield
	config.KNN_BACKEND = saved


def test_point_queries_match_cpp(reference, cpu64):
	P = cloud()
	bvh = BVH(pat.as_wide(P, 3))
	c = reference.BVH(np.asfortranarray(P.T))
	rng = np.random.default_rng(1)
	Q = np.concatenate([rng.normal(size=(200, 3)), 3 * rng.normal(size=(50, 3)), P[:50]])
	idx, dist = bvh.find_k_nearest(pat.as_wide(Q, 3), 16)
	ci, cd = bvh.find_closest_point(pat.as_wide(Q, 3))
	ri, rdist, rc = bvh.find_points_in_radius(pat.as_wide(Q, 3), 0.4)
	for i, q in enumerate(Q):
		ref_i, ref_d = c.find_k_nearest(q, 16)
		np.testing.assert_allclose(dist[i].numpy(), np.sort(ref_d), rtol=0, atol=1e-14)
		assert set(idx[i].numpy()) == set(ref_i)  # random data: no exact ties
		cidx, cdist = c.find_closest_point(q)
		assert int(ci[i]) == cidx and abs(float(cd[i]) - cdist) < 1e-14
		ref_ri, _ = c.find_points_in_radius(q, 0.4)
		assert set(ri[i, : rc[i]].numpy()) == set(ref_ri)
	# k-NN with a single (3,) query and k > N
	i1, d1 = bvh.find_k_nearest(pat.as_wide(Q[0], 3)[:, 0], 5)
	assert i1.shape == (5,) and torch.equal(i1, idx[0, :5])
	small = BVH(pat.as_wide(P[:3], 3))
	assert small.find_k_nearest(pat.as_wide(Q[:4], 3), 10)[0].shape == (4, 3)


def test_triangle_queries_match_cpp(reference, cpu64):
	V, F = torus_mesh(n_major=48, n_minor=24)
	bvh = BVH(pat.as_wide(V, 3), pat.as_wide(F, 3, dtype=torch.long))
	c = reference.BVH(np.asfortranarray(V.T), np.asfortranarray(F.T.astype(np.int32)))
	rng = np.random.default_rng(2)
	Q = rng.normal(size=(300, 3))
	pts, dist, tri = bvh.closest_point_on_mesh(pat.as_wide(Q, 3))
	d_err = 0.0
	for i, q in enumerate(Q):
		cp, cdist, ctri = c.closest_point_on_mesh(q)
		d_err = max(d_err, abs(cdist - float(dist[i])))
		if ctri == int(tri[i]):
			np.testing.assert_allclose(pts[:, i].numpy(), cp, atol=1e-14)
	assert d_err < 1e-14  # equal distances; the triangle differs only at shared edges / vertices (ties)
	ro = 2 * rng.normal(size=(400, 3))
	rd = -ro + 0.5 * rng.normal(size=(400, 3))  # mostly toward the torus
	h, t, idx, bary = bvh.intersect_ray(pat.as_wide(ro, 3), pat.as_wide(rd, 3))
	n_hits = 0
	for i in range(len(ro)):
		ch, ct, cidx, cb = c.intersect_ray(ro[i], rd[i], 0.0, np.inf, 0.01)
		assert ch == bool(h[i])
		if ch:
			n_hits += 1
			assert abs(ct - float(t[i])) < 1e-12 and cidx == int(idx[i])
			np.testing.assert_allclose(bary[:, i].numpy(), cb, atol=1e-12)
	assert n_hits > 100
	node_data, prims = bvh.get_gpu_data()
	assert node_data.shape == (bvh.get_num_nodes(), 10) and prims.shape == (len(F),)


def test_sphere_rays_are_exact(reference, cpu64):
	"""Point BVH rays: the port returns the closest sphere hit (= brute force); the original can miss it because its
	node boxes do not include the sphere radius (measured and printed)."""
	P = cloud(3000, seed=3)
	bvh = BVH(pat.as_wide(P, 3))
	c = reference.BVH(np.asfortranarray(P.T))
	rng = np.random.default_rng(4)
	ro = 3 * rng.normal(size=(300, 3))
	rd = -ro + 0.3 * rng.normal(size=(300, 3))
	RO, RD = pat.as_wide(ro, 3), pat.as_wide(rd, 3)
	h, t, idx, _ = bvh.intersect_ray(RO, RD, 0.0, np.inf, 0.05)
	hb, tb = intersect_sphere(RO[:, :, None], RD[:, :, None], pat.as_wide(P, 3)[:, None, :], 0.05)
	tmin = torch.where(hb, tb, torch.inf).amin(dim=1)
	assert torch.equal(h, torch.isfinite(tmin))
	torch.testing.assert_close(t[h], tmin[h], rtol=0, atol=1e-12)
	cpp = [c.intersect_ray(ro[i], rd[i], 0.0, np.inf, 0.05) for i in range(len(ro))]
	tm = tmin.numpy()
	agree = np.mean(
		[(r[0] == bool(np.isfinite(tm[i]))) and (not r[0] or abs(r[1] - tm[i]) < 1e-12) for i, r in enumerate(cpp)]
	)
	print(f'\noriginal point-BVH rays equal to the closest sphere hit: {100 * agree:.1f}% of rays; port: 100%')


def test_compaction_and_chunks(cpu64, monkeypatch):
	"""Query compaction and chunking do not change results."""
	P = cloud(3000, seed=5)
	bvh = BVH(pat.as_wide(P, 3))
	Q = pat.as_wide(np.random.default_rng(6).normal(size=(1000, 3)) * 2, 3)
	i0, d0 = bvh.find_k_nearest(Q, 12)
	monkeypatch.setattr(BVH, 'QUERY_CHUNK', 97)
	monkeypatch.setattr(BVH, 'CHECK_EVERY', 1)
	i1, d1 = bvh.find_k_nearest(Q, 12)
	assert torch.equal(d0, d1) and torch.equal(i0, i1)
	bf = torch.cdist(Q.T, pat.as_wide(P, 3).T).sort(dim=1).values[:, :12]
	torch.testing.assert_close(d0, bf, rtol=0, atol=1e-12)


@pytest.mark.parametrize('backend', ['brute', 'bvh'])
def test_get_neighbors_backends_match_cpp(reference, cpu64, knn_backend, backend):
	config.KNN_BACKEND = backend
	P = cloud(2500, seed=7)
	outliers = np.random.default_rng(8).choice(len(P), 40, replace=False)
	ref = reference.get_neighbors(np.asfortranarray(P.T), 16, outliers.astype(np.int32)).T
	assert np.array_equal(get_neighbors(pat.as_wide(P, 3), 16, outliers).numpy(), ref)
	ref = reference.get_neighbors(np.asfortranarray(P.T), 16, np.zeros(0, np.int32)).T
	assert np.array_equal(get_neighbors(pat.as_wide(P, 3), 16).numpy(), ref)


def test_evaluator_backends_identical(cpu64, knn_backend):
	"""k-NN evaluation through the BVH gives the same neighbors in the same order: identical results."""
	P, N, C = random_tori(3000, seed=9)
	tdf = pat.TorusDistanceField(pat.as_wide(P, 3), pat.as_wide(N, 3))
	tdf.fit_tori_from_forms(pat.as_wide(C, 6))
	tdf.set_k_evaluation(32)
	q = pat.as_wide(np.random.default_rng(10).normal(size=(4000, 3)), 3)
	out = {}
	for backend in ('brute', 'bvh'):
		config.KNN_BACKEND = backend
		out[backend] = tdf.evaluate_distance_gradient_laplacian(q)
	for a, b in zip(out['brute'], out['bvh']):
		assert torch.equal(a, b)
	config.KNN_BACKEND = 'bvh'
	nb_brute = None
	for backend in ('brute', 'bvh'):
		config.KNN_BACKEND = backend
		nb = tdf.get_all_neighbors(20)
		nb_brute = nb if nb_brute is None else nb_brute
	assert torch.equal(nb, nb_brute)


def test_wrapper(cpu64):
	P = cloud(500, seed=11)
	w = BoundingVolumeHierarchy(P)
	assert w.get_num_primitives() == 500 and w.get_num_nodes() >= 1
	idx, dist = w.find_k_nearest(pat.as_wide(P[0], 3)[:, 0], 3)
	assert int(idx[0]) == 0 and float(dist[0]) == 0.0
	V, F = torus_mesh()
	m = BoundingVolumeHierarchy(V, F)
	p, d, t = m.closest_point_on_mesh(pat.as_wide(np.zeros(3), 3)[:, 0])
	assert d > 0 and 0 <= int(t) < len(F)
	with pytest.raises(ValueError):
		w.closest_point_on_mesh(pat.as_wide(np.zeros(3), 3)[:, 0])
