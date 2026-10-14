"""
Fused Slang k-NN kernel (points_as_tori/knn_slang.py, points_as_tori/slang/*.slang) vs exact float64 k-NN and the
torch BVH traversal; the 'slang' k-NN backend of the evaluator.

Runs on every slangpy device available: CPU tensors through a Vulkan device (any GPU, or Mesa's lavapipe without
one), and each visible CUDA device through slangpy's CUDA backend and through Vulkan with CUDA interop. Skipped
without slangpy or when the device cannot be created.

Exactness criterion: the kernel selects neighbors with float32 distances, so where two candidates' distances differ
by less than float32 rounding it may return either one. The check is on distances: the exact (float64) distances of
the returned neighbors, sorted, equal the exact k nearest distances within float32 rounding of the coordinates.
"""

import numpy as np
import pytest
import torch

import points_as_tori as pat
from points_as_tori import config
from helpers import random_tori

PARAMS = ['cpu']
for _i in range(torch.cuda.device_count()):
	PARAMS += [f'cuda:{_i}', f'cuda:{_i}/vulkan']


@pytest.fixture(params=PARAMS)
def slang_device(request):
	"""Torch device for the test data; config.SLANG_DEVICE_TYPE set for the '/vulkan' variants; float64 geometry."""
	pytest.importorskip('slangpy')
	from points_as_tori import knn_slang

	name, _, device_type = request.param.partition('/')
	device = torch.device(name)
	saved = config.SLANG_DEVICE_TYPE, config.KNN_BACKEND
	config.SLANG_DEVICE_TYPE = device_type or None
	try:
		knn_slang.get_device(device)
	except Exception as e:  # no Vulkan driver / adapter, CUDA interop unsupported, ...
		config.SLANG_DEVICE_TYPE = saved[0]
		pytest.skip(f'no slangpy device for {request.param}: {e} (details: python -m points_as_tori.knn_slang)')
	config.set_device(device, torch.float64, force=True)
	yield device
	config.SLANG_DEVICE_TYPE, config.KNN_BACKEND = saved
	config.set_device('cpu', torch.float64, force=True)


def cloud(n, seed):
	rng = np.random.default_rng(seed)
	P = rng.normal(size=(n, 3))
	P[: n // 3] *= 0.05  # dense cluster inside a sparse cloud
	return P


def exact_knn_distances(Q, P, k):
	"""(Q, k) float64 distances of the k nearest points, ascending (CPU, float64)."""
	d = torch.cdist(Q.T.double().cpu(), P.T.double().cpu())
	return d.topk(k, dim=1, largest=False).values


def check_exact(idx, Q, P, k):
	"""Rows hold k distinct valid indices whose exact distances, sorted, equal the exact k-NN distances."""
	idx = idx.cpu()
	assert idx.shape == (Q.shape[1], k) and idx.dtype == torch.long
	assert int(idx.min()) >= 0 and int(idx.max()) < P.shape[1]
	assert bool((idx.sort(dim=1).values.diff(dim=1) > 0).all()), 'repeated neighbor'
	Pc, Qc = P.double().cpu(), Q.double().cpu()
	d = torch.sqrt(((Qc[:, :, None] - Pc[:, idx]) ** 2).sum(dim=0)).sort(dim=1).values
	ref = exact_knn_distances(Q, P, k)
	tol = 4e-7 * float(P.abs().max() + Q.abs().max())  # float32 rounding of coordinates and distances
	err = float((d - ref).abs().max())
	assert err <= tol, f'max |distance - exact| = {err:.3g} > {tol:.3g}'
	return err


@pytest.mark.parametrize('variant', ['heap', 'scan'])
@pytest.mark.parametrize('k', [1, 7, 32, 100])
def test_knn_exact_float64(slang_device, k, variant):
	from points_as_tori.knn_slang import SlangKNN

	P = pat.as_wide(cloud(4000, seed=1), 3)
	Q = pat.as_wide(np.random.default_rng(2).normal(size=(3000, 3)), 3)
	s = SlangKNN(P, variant=variant)
	idx, dist = s.find_k_nearest(Q, k)
	check_exact(idx, Q, P, k)
	# distances in the queries' dtype, recomputed from the indices, nearest first up to float32 order
	assert dist.dtype == torch.float64 and dist.device == P.device
	assert torch.allclose(dist, torch.sqrt(((Q[:, :, None] - P[:, idx]) ** 2).sum(dim=0)), rtol=0, atol=1e-15)
	if k > 1:
		assert float((dist[:, :-1] - dist[:, 1:]).max()) <= 1e-6


def test_knn_float32_matches_torch_bvh(slang_device):
	"""float32 inputs: float32 kernel distances, sorted; same as the torch BVH up to float32 rounding."""
	from points_as_tori.knn_slang import SlangKNN

	P = pat.as_wide(cloud(5000, seed=3), 3, dtype=torch.float32)
	Q = pat.as_wide(np.random.default_rng(4).normal(size=(2000, 3)), 3, dtype=torch.float32)
	s = SlangKNN(P)
	idx, dist = s.find_k_nearest(Q, 16)
	assert dist.dtype == torch.float32
	assert bool((dist[:, 1:] >= dist[:, :-1]).all())
	ref_idx, ref_dist = s.bvh.find_k_nearest(Q, 16)
	assert float((dist - ref_dist).abs().max()) <= 1e-6 * float(P.abs().max() + Q.abs().max())
	same = (idx.sort(dim=1).values == ref_idx.sort(dim=1).values).all(dim=1).float().mean()
	assert float(same) > 0.99  # differences only at float32 near-ties
	check_exact(idx, Q, P, 16)


def test_chunks_limits_and_single_query(slang_device, monkeypatch):
	from points_as_tori import knn_slang
	from points_as_tori.knn_slang import SlangKNN

	P = pat.as_wide(cloud(1500, seed=5), 3)
	Q = pat.as_wide(np.random.default_rng(6).normal(size=(700, 3)), 3)
	s = SlangKNN(P)
	idx, dist = s.find_k_nearest(Q, 9)
	monkeypatch.setattr(knn_slang, 'QUERY_CHUNK_ELEMENTS', 9 * 97)  # 97 queries per dispatch
	idx_c, dist_c = s.find_k_nearest(Q, 9)
	assert torch.equal(idx, idx_c) and torch.equal(dist, dist_c)
	# single query
	i1, d1 = s.find_k_nearest(Q[:, 5], 9)
	assert torch.equal(i1, idx[5]) and torch.equal(d1, dist[5])
	# k larger than the point count: all points
	small = SlangKNN(P[:, :20])
	i_all, _ = small.find_k_nearest(Q, 50)
	assert i_all.shape == (700, 20)
	check_exact(i_all, Q, P[:, :20], 20)
	# above K_LIMIT: the torch BVH traversal
	monkeypatch.setattr(knn_slang, 'K_LIMIT', 4)
	i_big, _ = s.find_k_nearest(Q, 6)
	check_exact(i_big, Q, P, 6)


def test_evaluator_slang_backend(slang_device):
	"""TorusDistanceField k-NN evaluation and neighbor lists with KNN_BACKEND = 'slang' vs 'bvh'."""
	P, N, C = random_tori(3000, seed=9)
	tdf = pat.TorusDistanceField(pat.as_wide(P, 3), pat.as_wide(N, 3))
	tdf.fit_tori_from_forms(pat.as_wide(C, 6))
	tdf.set_k_evaluation(32)
	q = pat.as_wide(np.random.default_rng(10).normal(size=(4000, 3)), 3)
	out, nb = {}, {}
	for backend in ('bvh', 'slang'):
		config.KNN_BACKEND = backend
		out[backend] = tdf.evaluate_distance_gradient_laplacian(q)
		nb[backend] = tdf.get_all_neighbors(20)
	assert type(tdf._get_bvh()).__name__ == 'SlangKNN'
	for a, b in zip(out['bvh'], out['slang']):
		assert torch.allclose(a, b, rtol=1e-9, atol=1e-12)
	assert float((nb['bvh'] == nb['slang']).all(dim=1).float().mean()) > 0.99
	check_exact(nb['slang'], tdf.points, tdf.points, 20)
	# module-level neighbor functions
	from points_as_tori.torus_distance import get_neighbors

	check_exact(get_neighbors(tdf.points, 12), tdf.points, tdf.points, 12)
