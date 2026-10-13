"""
LBVH (sdfview/accel/lbvh.py): exact building blocks against plain-Python references, tree invariants, k-NN through
the tree against brute force, and agreement with the original C++ BVH layout.
"""

import os
import sys

import numpy as np
import pytest
import torch

from sdfview.accel import lbvh as L
from sdfview.accel import build_lbvh, from_cpp_layout, knn_reference


def test_bit_length_exact():
	rng = np.random.default_rng(0)
	vals = np.concatenate(
		[rng.integers(0, 2**63 - 1, 2000, dtype=np.int64), 2 ** np.arange(63, dtype=np.int64), [0, 1, 2**63 - 1]]
	)
	vals = np.concatenate([vals, np.maximum(vals - 1, 0)])
	got = L.bit_length(torch.from_numpy(vals)).numpy()
	assert np.array_equal(got, [int(v).bit_length() for v in vals])


def test_morton_codes():
	rng = np.random.default_rng(1)
	p = rng.uniform(-2, 3, size=(500, 3))
	codes = L.morton_codes(torch.from_numpy(p.T.copy())).numpy()
	lo, hi = p.min(0), p.max(0)
	q = np.minimum(np.floor((p - lo) / (hi - lo) * 2**21), 2**21 - 1).astype(np.int64)
	ref = []
	for x, y, z in q:
		c = 0
		for b in range(21):
			c |= (
				((int(x) >> b) & 1) << (3 * b + 2) | ((int(y) >> b) & 1) << (3 * b + 1) | ((int(z) >> b) & 1) << (3 * b)
			)
		ref.append(c)
	assert np.array_equal(codes, ref)


def _radix_tree_by_delta(keys):
	"""Karras 2012 written with δ values directly (the slower form the optimized predicate replaces)."""
	n = keys.shape[0]
	steps = max(1, int(np.ceil(np.log2(n))))
	i = torch.arange(n - 1)
	dl = L._delta
	d = torch.where(dl(keys, i, i + 1) > dl(keys, i, i - 1), 1, -1)
	dmin = dl(keys, i, i - d)
	lo_len = torch.zeros_like(i)
	for b in reversed(range(steps)):
		t = 1 << b
		lo_len = torch.where(dl(keys, i, i + (lo_len + t) * d) > dmin, lo_len + t, lo_len)
	j = i + lo_len * d
	dn = dl(keys, i, j)
	s = torch.zeros_like(i)
	for b in reversed(range(steps)):
		t = 1 << b
		s = torch.where(dl(keys, i, i + (s + t) * d) > dn, s + t, s)
	g = i + s * d + torch.clamp(d, max=0)
	lo, hi = torch.minimum(i, j), torch.maximum(i, j)
	return lo, hi, torch.where(lo == g, n - 1 + g, g), torch.where(hi == g + 1, n + g, g + 1)


@pytest.mark.parametrize('kind', ['random', 'few_values', 'extremes'])
def test_radix_tree_predicate(kind):
	rng = np.random.default_rng({'random': 0, 'few_values': 1, 'extremes': 2}[kind])
	for _ in range(10):
		n = int(rng.integers(2, 400))
		if kind == 'random':
			keys = rng.integers(0, 2**63 - 1, n, dtype=np.int64)
		elif kind == 'few_values':  # many duplicates: the index tie-break decides
			keys = rng.integers(0, 6, n, dtype=np.int64)
		else:
			keys = rng.choice(np.array([0, 1, 2**62, 2**63 - 1], dtype=np.int64), n)
		keys = torch.sort(torch.from_numpy(keys)).values
		for a, b in zip(L._radix_tree(keys), _radix_tree_by_delta(keys)):
			assert torch.equal(a, b)


def check_tree(bvh, points, leaf_size):
	"""Invariants: pre-order, each point in exactly one leaf, exact leaf boxes, nested boxes, depth."""
	n = len(points)
	left, right = bvh.left.numpy(), bvh.right.numpy()
	start, count = bvh.prim_start.numpy(), bvh.prim_count.numpy()
	prims = bvh.prim_indices.numpy()
	bmin, bmax = bvh.bounds_min.T.numpy(), bvh.bounds_max.T.numpy()
	seen = np.zeros(n, int)
	depth = np.zeros(bvh.n_nodes, int)
	for i in range(bvh.n_nodes):
		if count[i] > 0:
			assert left[i] == right[i] == -1 and 1 <= count[i] <= max(leaf_size, 1)
			ids = prims[start[i] : start[i] + count[i]]
			seen[ids] += 1
			assert np.array_equal(bmin[i], points[ids].min(0)) and np.array_equal(bmax[i], points[ids].max(0))
		else:
			assert left[i] > i and right[i] > left[i]  # depth-first pre-order
			for c in (left[i], right[i]):
				depth[c] = depth[i] + 1
				assert np.all(bmin[c] >= bmin[i]) and np.all(bmax[c] <= bmax[i])
			# a parent's box is the union of its children's
			assert np.array_equal(bmin[i], np.minimum(bmin[left[i]], bmin[right[i]]))
			assert np.array_equal(bmax[i], np.maximum(bmax[left[i]], bmax[right[i]]))
	assert np.all(seen == 1)
	assert np.array_equal(depth, bvh.depth.numpy())
	assert bvh.n_nodes <= 2 * n - 1


@pytest.mark.parametrize('n', [1, 2, 3, 4, 5, 17, 1000])
@pytest.mark.parametrize('leaf_size', [1, 4, 8])
def test_tree_invariants(n, leaf_size):
	points = np.random.default_rng(n).normal(size=(n, 3)).astype(np.float32)
	bvh = build_lbvh(torch.from_numpy(points.T.copy()), leaf_size)
	check_tree(bvh, points, leaf_size)


def test_duplicates_and_flat_clouds():
	"""Coincident points (equal Morton codes) and degenerate extents (a plane, a line)."""
	rng = np.random.default_rng(3)
	base = rng.normal(size=(50, 3))
	for points in (
		np.repeat(base, 20, axis=0),
		np.c_[rng.normal(size=(500, 2)), np.zeros(500)],
		np.c_[rng.normal(size=500), np.zeros((500, 2))],
		np.zeros((64, 3)),
	):
		points = points.astype(np.float32)
		bvh = build_lbvh(torch.from_numpy(points.T.copy()), 4)
		check_tree(bvh, points, 4)


@pytest.mark.parametrize('k', [1, 8, 32])
def test_knn_matches_brute_force(k):
	rng = np.random.default_rng(k)
	points = rng.normal(size=(3000, 3))
	points[:1000] *= 0.05  # a dense cluster inside a sparse cloud
	bvh = build_lbvh(torch.from_numpy(points.T.copy()), 4)
	host = {}
	for q in np.concatenate([rng.normal(size=(100, 3)), 4 * rng.normal(size=(20, 3)), points[:20]]):
		idx, dist = knn_reference(bvh, points, q, k, host=host)
		bf = np.linalg.norm(points - q, axis=1)
		np.testing.assert_allclose(dist, np.sort(bf)[:k], rtol=0, atol=1e-12)
		np.testing.assert_allclose(bf[idx], dist, rtol=0, atol=1e-12)


def _reference_module():
	repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
	sys.path.insert(0, os.environ.get('PAT_REFERENCE_DIR', os.path.join(repo, 'build', 'pat_reference')))
	try:
		import pat_reference

		return pat_reference
	except ImportError:
		return None


@pytest.mark.skipif(_reference_module() is None, reason='needs the C++ reference module (PAT_REFERENCE_DIR)')
def test_original_bvh_layout():
	"""The original C++ BVH (SAH build) read through from_cpp_layout satisfies the same invariants and gives the same
	k-NN through the same traversal; tree sizes and traversal costs are printed for comparison."""
	ref = _reference_module()
	rng = np.random.default_rng(5)
	points = rng.normal(size=(5000, 3))
	points /= np.linalg.norm(points, axis=1, keepdims=True)
	nd, idx = ref.BVH(np.asfortranarray(points.T)).get_gpu_data()
	sah = from_cpp_layout(nd, idx)
	lb = build_lbvh(torch.from_numpy(points.T.copy()), 4)
	check_tree(sah, points, 4)
	costs = {}
	for name, b in (('sah', sah), ('lbvh', lb)):
		stats, host = {}, {}
		for q in points[rng.integers(0, 5000, 100)] * (1 + 0.02 * rng.normal(size=(100, 1))):
			_, dist = knn_reference(b, points, q, 32, stats, host)
			np.testing.assert_allclose(dist, np.sort(np.linalg.norm(points - q, axis=1))[:32], atol=1e-12)
		costs[name] = {k: v / 100 for k, v in stats.items()}
	print(
		f'\nnodes: SAH {sah.n_nodes}, LBVH {lb.n_nodes}; depth: SAH {int(sah.depth.max())}, LBVH {int(lb.depth.max())}'
	)
	print(f'k-NN (k=32) per query: SAH {costs["sah"]}, LBVH {costs["lbvh"]}')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_cuda_build_equals_cpu():
	points = np.random.default_rng(7).normal(size=(20000, 3)).astype(np.float32)
	a = build_lbvh(torch.from_numpy(points.T.copy()))
	b = build_lbvh(torch.from_numpy(points.T.copy()).cuda())
	for name in ('left', 'right', 'prim_start', 'prim_count', 'prim_indices', 'depth', 'bounds_min', 'bounds_max'):
		assert torch.equal(getattr(a, name), getattr(b, name).cpu()), name
