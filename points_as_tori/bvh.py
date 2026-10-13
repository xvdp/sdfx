"""
Torch port of src/cpp/bvh.h / bvh.cpp: a bounding volume hierarchy over points or triangles, with the original's
queries (closest point, k nearest, points in radius, ray intersection, closest point on a mesh, primitives in a
box) and the original's flattened node layout (get_gpu_data): root = node 0, depth-first pre-order, per node
(aabb_min, aabb_max, left_child, right_child, prim_start, prim_count), prim_count = 0 for internal nodes, leaves
hold ≤ leaf_size primitives addressed through a primitive index array.

Deviation: construction. The original builds top-down with a 12-bin SAH split per node (sequential, recursive).
Here the tree is a linear BVH (Karras 2012) built with data-parallel torch ops on any device; leaves hold ≤ 4
primitives as in the original. Same node semantics; a different tree, so traversal orders (and the order of
equal-distance results) differ, not the sets returned by exact queries.

Queries are batched: many queries traverse at once, each with its own stack (a (Q, S) tensor), one node per query per
iteration, with the original's pruning rules (see the BATCHED TRAVERSAL section).

Construction (all steps are data-parallel tensor ops; host syncs only to read two output sizes):

 1. Morton codes. Points are normalized to the unit cube of their bounding box and quantized to 21 bits per axis,
    x̃ = min(⌊x · 2²¹⌋, 2²¹ - 1); the 63-bit code interleaves the bits (x̃ in bit 3b+2, ỹ in 3b+1, z̃ in 3b).
    Sorting by code orders the points along a Z-order curve: points sharing a long code prefix lie in a small octree
    cell.

 2. Binary radix tree (Karras 2012, "Maximizing parallelism in the construction of BVHs, octrees and k-d trees").
    Keys k_i (sorted codes, made unique by appending the sorted position i) define, for i, j ∈ [0, N),
        δ(i, j) = length of the common prefix of (k_i, i) and (k_j, j);  δ(i, j) = -1 if j ∉ [0, N).
    With 63-bit codes in int64 words: δ = clz64(k_i ⊕ k_j) if k_i ≠ k_j, else 64 + clz64(i ⊕ j) (always larger).
    Internal node i ∈ [0, N-1) covers a contiguous range of sorted keys with i at one end:
        direction  d = sign(δ(i, i+1) - δ(i, i-1))
        other end  j = i + l·d,  l = max{ l : δ(i, i + l·d) > δ(i, i - d) }
        split      γ = i + s·d + min(d, 0),  s = max{ s : δ(i, i + s·d) > δ(i, j) }
        children   left = leaf γ if min(i, j) = γ else internal γ;  right = leaf γ+1 if max(i, j) = γ+1 else internal γ+1
    Both maxima are found by binary search over powers of two (the predicates are monotone in l and s), with a fixed
    number of steps ⌈log2 N⌉ for every node, so the loop runs on the host with no data-dependent branching.

 3. Leaf collapse. A node whose range holds ≤ leaf_size points becomes a leaf; its descendants are dropped.
    Kept nodes = root ∪ {children of nodes with range size > leaf_size}.

 4. Depth-first pre-order: sorting kept nodes by (range start ascending, range size descending) visits every parent
    before its subtree and the left subtree before the right one (ranges are nested or disjoint), i.e. the order
    of the original's recursive build. Children indices are remapped through the inverse permutation.

 5. Bounding boxes: box(node) = min / max over its sorted range [lo, hi], answered with a sparse table
    T_k[x] = min(P[x .. x + 2^k - 1]) built one level at a time: box = min(T_m[lo], T_m[hi - 2^m + 1]),
    m = ⌊log2(hi - lo + 1)⌋. Each level is O(N) memory and is dropped after use: O(N log N) time, O(N) memory;
    nodes are grouped by m so that each node is gathered once.

 6. Depth (for the shader's traversal stack): pointer jumping on the parent array, ⌈log2(max depth)⌉ rounds.

Exact by construction: the BVH is only an index; k-NN through it returns the same neighbors as brute force (up to
ties). Quality (SAH cost) is lower than the original's SAH top-down build; Karras 2013 (treelet restructuring),
PLOC (Meister & Bittner 2018) and H-PLOC (Benthin et al. 2024) improve it at higher build cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from . import config

MORTON_BITS = 21  # per axis; 3 * 21 = 63 bits fit a non-negative int64


@dataclass
class LBVH:
	"""Flattened BVH (device tensors). M nodes, N primitives."""

	bounds_min: torch.Tensor  # (3, M) float
	bounds_max: torch.Tensor  # (3, M) float
	left: torch.Tensor  # (M,) int64, -1 for leaves
	right: torch.Tensor  # (M,) int64, -1 for leaves
	prim_start: torch.Tensor  # (M,) int64 (0 for internal nodes)
	prim_count: torch.Tensor  # (M,) int64 (0 for internal nodes)
	prim_indices: torch.Tensor  # (N,) int64, slot -> primitive (point) index
	depth: torch.Tensor  # (M,) int64, edges from the root

	@property
	def n_nodes(self) -> int:
		return self.left.shape[0]

	def gpu_arrays(self) -> dict:
		"""Export boundary (device -> host): arrays in the scene-format layout (see sdfview/scene.py)."""
		bounds = torch.zeros(self.n_nodes, 8, dtype=torch.float32, device=self.left.device)
		bounds[:, 0:3] = self.bounds_min.T
		bounds[:, 4:7] = self.bounds_max.T
		nodes = torch.stack([self.left, self.right, self.prim_start, self.prim_count], dim=1).to(torch.int32)
		return {
			'bvh_bounds': bounds.cpu().numpy(),
			'bvh_nodes': nodes.cpu().numpy(),
			'bvh_prims': self.prim_indices.to(torch.int32)[:, None].cpu().numpy(),
		}

	def to_cpp_layout(self) -> Tuple[np.ndarray, np.ndarray]:
		"""(n_nodes, 10) float64 rows like the original BVH::get_gpu_data(), and the (N,) primitive indices."""
		cols = [self.bounds_min.T, self.bounds_max.T]
		cols += [
			x[:, None].to(self.bounds_min.dtype) for x in (self.left, self.right, self.prim_start, self.prim_count)
		]
		return torch.cat(cols, dim=1).double().cpu().numpy(), self.prim_indices.cpu().numpy()


def _expand_bits(v: torch.Tensor) -> torch.Tensor:
	"""Spread the low 21 bits of v so that bit b moves to bit 3b (magic-number bit interleaving)."""
	v = v & 0x1FFFFF
	v = (v | (v << 32)) & 0x1F00000000FFFF
	v = (v | (v << 16)) & 0x1F0000FF0000FF
	v = (v | (v << 8)) & 0x100F00F00F00F00F
	v = (v | (v << 4)) & 0x10C30C30C30C30C3
	v = (v | (v << 2)) & 0x1249249249249249
	return v


def morton_codes(points: torch.Tensor, lo: Optional[torch.Tensor] = None, hi: Optional[torch.Tensor] = None):
	"""(3, N) points -> (N,) int64 63-bit Morton codes of the points quantized in the box [lo, hi] (default: bbox)."""
	lo = points.amin(dim=1) if lo is None else lo
	hi = points.amax(dim=1) if hi is None else hi
	extent = torch.clamp(hi - lo, min=torch.finfo(points.dtype).tiny)
	scale = float(1 << MORTON_BITS)
	q = torch.floor((points - lo[:, None]) / extent[:, None] * scale)
	q = q.clamp(0, scale - 1).to(torch.int64)
	return (_expand_bits(q[0]) << 2) | (_expand_bits(q[1]) << 1) | _expand_bits(q[2])


def bit_length(x: torch.Tensor) -> torch.Tensor:
	"""Exact number of significant bits of non-negative int64 values (0 -> 0); six shift-and-compare steps."""
	n = torch.zeros_like(x)
	for s in (32, 16, 8, 4, 2, 1):
		hit = (x >> s) != 0
		n = n + hit.to(x.dtype) * s
		x = torch.where(hit, x >> s, x)
	return n + (x != 0).to(x.dtype)


def _delta(keys: torch.Tensor, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
	"""Common prefix length δ(i, j) of the unique keys (k_i, i), (k_j, j); -1 where j is out of range."""
	n = keys.shape[0]
	valid = (j >= 0) & (j < n)
	jc = j.clamp(0, n - 1)
	x = keys[i] ^ keys[jc]
	prefix = torch.where(x == 0, 128 - bit_length(i ^ jc), 64 - bit_length(x))
	return torch.where(valid, prefix, torch.full_like(prefix, -1))


def _prefix_shifts(keys: torch.Tensor, i: torch.Tensor, r: torch.Tensor):
	"""
	Shifts (sx, sy) that turn the comparison δ(i, j) > δ(i, r) into
		j ∈ [0, N)  and  (k_i ⊕ k_j) >> sx = 0  and  (i ⊕ j) >> sy = 0,
	so the binary searches need no clz. With x = k_i ⊕ k_r, y = i ⊕ r, b(·) = bit length:
		r out of range:  δ(i, r) = -1          -> sx = sy = 63 (only the range test remains)
		x ≠ 0:          δ(i, r) = 64 - b(x)    -> need b(k_i ⊕ k_j) < b(x):  sx = b(x) - 1, sy = 63
		x = 0:          δ(i, r) = 128 - b(y)   -> need k_j = k_i and b(i ⊕ j) < b(y):  sx = 0, sy = b(y) - 1
	"""
	n = keys.shape[0]
	valid = (r >= 0) & (r < n)
	rc = r.clamp(0, n - 1)
	x = keys[i] ^ keys[rc]
	same = x == 0
	sx = torch.where(valid, torch.where(same, 0, bit_length(x) - 1), 63)
	sy = torch.where(valid & same, bit_length(i ^ rc) - 1, 63)
	return sx, sy


def _longer_prefix(keys: torch.Tensor, ki: torch.Tensor, i: torch.Tensor, j: torch.Tensor, sx, sy) -> torch.Tensor:
	"""δ(i, j) > δ(i, r) for the reference r encoded by (sx, sy); ki = keys[i]."""
	n = keys.shape[0]
	jc = j.clamp(0, n - 1)
	return (j >= 0) & (j < n) & (((ki ^ keys[jc]) >> sx) == 0) & (((i ^ jc) >> sy) == 0)


def _radix_tree(keys: torch.Tensor):
	"""Karras 2012 internal nodes for N >= 2 sorted keys: (lo, hi, left, right) with leaves numbered N-1 + j."""
	n = keys.shape[0]
	n_int = n - 1
	steps = max(1, math.ceil(math.log2(n)))
	i = torch.arange(n_int, device=keys.device)
	ki = keys[:n_int]
	# d = sign(δ(i, i+1) - δ(i, i-1)) (never 0: keys are unique)
	d = torch.where(_longer_prefix(keys, ki, i, i + 1, *_prefix_shifts(keys, i, i - 1)), 1, -1)

	# other end: largest span l with δ(i, i + l d) > δ_min = δ(i, i - d)
	sx, sy = _prefix_shifts(keys, i, i - d)
	span = torch.zeros_like(i)
	for b in reversed(range(steps)):
		t = 1 << b
		span = torch.where(_longer_prefix(keys, ki, i, i + (span + t) * d, sx, sy), span + t, span)
	j = i + span * d

	# split: largest s with δ(i, i + s d) > δ_node = δ(i, j)
	sx, sy = _prefix_shifts(keys, i, j)
	s = torch.zeros_like(i)
	for b in reversed(range(steps)):
		t = 1 << b
		s = torch.where(_longer_prefix(keys, ki, i, i + (s + t) * d, sx, sy), s + t, s)
	gamma = i + s * d + torch.clamp(d, max=0)

	lo, hi = torch.minimum(i, j), torch.maximum(i, j)
	left = torch.where(lo == gamma, n_int + gamma, gamma)
	right = torch.where(hi == gamma + 1, n_int + gamma + 1, gamma + 1)
	return lo, hi, left, right


def _range_bounds(sorted_min: torch.Tensor, sorted_max: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor):
	"""
	min of sorted_min[:, lo..hi] and max of sorted_max[:, lo..hi] per node (primitive boxes in sorted order; for
	points both are the points), by a sparse table built and consumed level by level. Nodes are grouped by
	m = ⌊log2(hi - lo + 1)⌋ (one argsort; the group sizes are read on the host once), so each node is gathered only at
	its own level: O(N log N) table work, O(M) gathers, O(N) memory.
	"""
	n = sorted_min.shape[1]
	level = bit_length(hi - lo + 1) - 1
	n_levels = n.bit_length()
	by_level = torch.argsort(level, stable=True)
	counts = torch.bincount(level, minlength=n_levels).tolist()  # host sync: group sizes
	groups = torch.split(by_level, counts)
	bmin = torch.empty((3, lo.shape[0]), dtype=sorted_min.dtype, device=sorted_min.device)
	bmax = torch.empty_like(bmin)
	tmin, tmax = sorted_min, sorted_max  # level 0: T_0[x] = box of primitive x
	for k in range(n_levels):
		w = 1 << k
		nodes = groups[k]
		if counts[k] > 0:
			a, b = lo[nodes], hi[nodes] - w + 1
			bmin[:, nodes] = torch.minimum(tmin[:, a], tmin[:, b])
			bmax[:, nodes] = torch.maximum(tmax[:, a], tmax[:, b])
		if 2 * w <= n and sum(counts[k + 1 :]) > 0:  # T_{k+1}[x] = op(T_k[x], T_k[x + w]), x ∈ [0, n - 2w]
			tmin = torch.minimum(tmin[:, : n - 2 * w + 1], tmin[:, w : n - w + 1])
			tmax = torch.maximum(tmax[:, : n - 2 * w + 1], tmax[:, w : n - w + 1])
	return bmin, bmax


def _depth(parent: torch.Tensor, max_rounds: int = 8) -> torch.Tensor:
	"""Edges to the root by pointer jumping: after r rounds, ancestors up to 2^r levels up are accounted for."""
	dist = (parent >= 0).to(torch.int64)
	jump = parent.clone()
	for _ in range(max_rounds):  # 2^8 = 256 > max depth (≤ 63 code bits + 31 index bits + 1)
		valid = jump >= 0
		jc = jump.clamp(min=0)
		dist = dist + torch.where(valid, dist[jc], 0)
		jump = torch.where(valid, jump[jc], -1)
	return dist


def build_lbvh(
	points: torch.Tensor,
	leaf_size: int = 4,
	prim_min: Optional[torch.Tensor] = None,
	prim_max: Optional[torch.Tensor] = None,
) -> LBVH:
	"""
	Args:
		points: (3, N) tensor (any device / float dtype): the points, or the centroids of other primitives
		leaf_size: max primitives per leaf (the original uses 4)
		prim_min, prim_max: (3, N) boxes of the primitives when they are not points (e.g. triangles)

	Returns:
		LBVH with M ≤ 2N - 1 nodes. Two host syncs: the number of kept nodes and the per-level node counts
		(output / group sizes); everything else is queued on the device.
	"""
	if points.ndim != 2 or points.shape[0] != 3 or points.shape[1] < 1:
		raise ValueError(f'points: expected (3, N) with N >= 1, got {tuple(points.shape)}')
	if leaf_size < 1:
		raise ValueError('leaf_size must be >= 1')
	device, n = points.device, points.shape[1]
	long = dict(dtype=torch.int64, device=device)
	prim_min = points if prim_min is None else prim_min
	prim_max = points if prim_max is None else prim_max
	codes = morton_codes(points)
	keys, order = torch.sort(codes, stable=True)

	if n == 1:
		z = torch.zeros(1, **long)
		return LBVH(prim_min.clone(), prim_max.clone(), z - 1, z - 1, z, z + 1, order, z)

	n_int = n - 1
	lo_i, hi_i, left_i, right_i = _radix_tree(keys)
	leaves = torch.arange(n, **long)
	lo = torch.cat([lo_i, leaves])  # node ids: internal 0 .. N-2, leaf j -> N-1 + j
	hi = torch.cat([hi_i, leaves])
	left = torch.cat([left_i, torch.full((n,), -1, **long)])
	right = torch.cat([right_i, torch.full((n,), -1, **long)])
	size = hi - lo + 1

	parent = torch.full((2 * n - 1,), -1, **long)
	parent[left_i] = torch.arange(n_int, **long)
	parent[right_i] = torch.arange(n_int, **long)

	# leaf collapse: keep the root and the children of nodes larger than leaf_size
	keep = (parent < 0) | (size[parent.clamp(min=0)] > leaf_size)
	is_leaf = size <= leaf_size

	# depth-first pre-order of the kept nodes: (lo ascending, size descending); dropped nodes sort last
	key = lo * (n + 1) + (n - size)
	key = torch.where(keep, key, torch.full_like(key, (n + 1) * (n + 1)))
	perm = torch.argsort(key)
	m = int(keep.sum())  # host sync: output size
	perm = perm[:m]
	new_index = torch.full((2 * n - 1,), -1, **long)
	new_index[perm] = torch.arange(m, **long)

	lo_k, hi_k, leaf_k = lo[perm], hi[perm], is_leaf[perm]
	minus1 = torch.full((m,), -1, **long)
	left_k = torch.where(leaf_k, minus1, new_index[left[perm].clamp(min=0)])
	right_k = torch.where(leaf_k, minus1, new_index[right[perm].clamp(min=0)])
	prim_start = torch.where(leaf_k, lo_k, 0)
	prim_count = torch.where(leaf_k, hi_k - lo_k + 1, 0)

	bmin, bmax = _range_bounds(prim_min[:, order], prim_max[:, order], lo_k, hi_k)

	# parent of each kept node (leaves scatter into a discarded slot m: no boolean-mask indexing)
	parent_k = torch.full((m + 1,), -1, **long)
	src = torch.arange(m, **long)
	parent_k[torch.where(leaf_k, m, left_k)] = src
	parent_k[torch.where(leaf_k, m, right_k)] = src
	parent_k = parent_k[:m]
	depth = _depth(parent_k)

	return LBVH(bmin, bmax, left_k, right_k, prim_start, prim_count, order, depth)


# ========================================================================
# REFERENCE QUERIES (host, one query at a time; same algorithm as bvhKNearestNeighbors in shaders/glsl/bvh.glsl)
# ========================================================================


def _point_aabb_distance_sq(q, bmin, bmax):
	c = np.minimum(np.maximum(q, bmin), bmax)
	return float(np.dot(q - c, q - c))


def from_cpp_layout(node_data: np.ndarray, prim_indices: np.ndarray, device=None) -> LBVH:
	"""Wrap the original BVH::get_gpu_data() output ((M, 10) rows, (N,) indices) as an LBVH (same node semantics)."""
	t = torch.as_tensor(np.asarray(node_data, dtype=np.float64), device=device)
	ints = [t[:, c].round().to(torch.int64) for c in range(6, 10)]
	m = t.shape[0]
	parent = torch.full((m + 1,), -1, dtype=torch.int64, device=t.device)
	src = torch.arange(m, device=t.device)
	for child in ints[:2]:
		parent[torch.where(child >= 0, child, m)] = src
	idx = torch.as_tensor(np.asarray(prim_indices, dtype=np.int64), device=t.device)
	return LBVH(t[:, 0:3].T.contiguous(), t[:, 3:6].T.contiguous(), *ints, idx, _depth(parent[:m]))


def knn_reference(
	bvh: LBVH, points: np.ndarray, query: np.ndarray, k: int, stats: Optional[dict] = None, host: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	k nearest neighbors of one query by stack traversal (nearer child processed first, prune by the current k-th
	distance), mirroring the shader. points: (N, 3) host array. Returns (indices, distances) sorted by distance.
	stats: if given, 'nodes' / 'points' are incremented by the nodes popped and points tested (traversal cost).
	host: cache for host copies of the BVH arrays (pass the same dict across queries).
	"""
	if host is None or not host:
		arrays = dict(
			bmin=bvh.bounds_min.T.cpu().numpy(),
			bmax=bvh.bounds_max.T.cpu().numpy(),
			left=bvh.left.cpu().numpy(),
			right=bvh.right.cpu().numpy(),
			start=bvh.prim_start.cpu().numpy(),
			count=bvh.prim_count.cpu().numpy(),
			prims=bvh.prim_indices.cpu().numpy(),
		)
		if host is not None:
			host.update(arrays)
		host = arrays
	bmin, bmax, left, right = host['bmin'], host['bmax'], host['left'], host['right']
	start, count, prims = host['start'], host['count'], host['prims']
	n_nodes = n_points = 0
	idx, dsq = [], []
	kth = np.inf
	stack = [0]
	while stack:
		node = stack.pop()
		n_nodes += 1
		if len(idx) >= k and _point_aabb_distance_sq(query, bmin[node], bmax[node]) > kth:
			continue
		if count[node] > 0:
			n_points += int(count[node])
			for slot in range(start[node], start[node] + count[node]):
				p = prims[slot]
				d2 = float(np.dot(query - points[p], query - points[p]))
				if len(idx) < k:
					idx.append(p)
					dsq.append(d2)
					kth = max(dsq) if len(idx) == k else np.inf
				elif d2 < kth:
					far = int(np.argmax(dsq))
					idx[far], dsq[far] = p, d2
					kth = max(dsq)
		else:
			dl = _point_aabb_distance_sq(query, bmin[left[node]], bmax[left[node]])
			dr = _point_aabb_distance_sq(query, bmin[right[node]], bmax[right[node]])
			near, far_child, dn, df = (
				(left[node], right[node], dl, dr) if dl < dr else (right[node], left[node], dr, dl)
			)
			if len(idx) < k or df <= kth:
				stack.append(far_child)
			if len(idx) < k or dn <= kth:
				stack.append(near)
	if stats is not None:
		stats['nodes'] = stats.get('nodes', 0) + n_nodes
		stats['points'] = stats.get('points', 0) + n_points
	o = np.argsort(dsq, kind='stable')
	return np.asarray(idx)[o], np.sqrt(np.asarray(dsq)[o])


# ========================================================================
# BATCHED TRAVERSAL
# ========================================================================
#
# Every query keeps its own stack of node indices, as the C++ `std::vector<int> stack` of one query: a row of a
# (Q, S) tensor with a stack pointer. One iteration pops one node for every query whose stack is not empty, applies
# the original's test for that node (box distance or slab test against the current bound), processes a leaf's
# ≤ leaf_size primitives at once, or pushes children. S = max depth + 2 suffices: a node at depth d leaves at most
# d + 1 entries on the stack. Iterations = the largest number of nodes any query visits; the loop ends when all
# stacks are empty (host check every few iterations). Queries are processed in chunks to bound memory.


def point_aabb_distance_sq(q: torch.Tensor, aabb_min: torch.Tensor, aabb_max: torch.Tensor) -> torch.Tensor:
	"""(3, ...) points and boxes -> squared distance to the box (0 inside)."""
	c = torch.minimum(torch.maximum(q, aabb_min), aabb_max)
	d = q - c
	return (d * d).sum(dim=0)


def intersect_aabb(ro, rd, aabb_min, aabb_max):
	"""Slab test (C++ intersect_aabb): (hit, t_near, t_far), hit = t_near <= t_far and t_far >= 0."""
	inv_dir = 1.0 / rd
	t0 = (aabb_min - ro) * inv_dir
	t1 = (aabb_max - ro) * inv_dir
	t_near = torch.minimum(t0, t1).amax(dim=0)
	t_far = torch.maximum(t0, t1).amin(dim=0)
	return (t_near <= t_far) & (t_far >= 0.0), t_near, t_far


def intersect_sphere(ro, rd, center, radius):
	"""
	C++ intersect_sphere: t along the normalized direction d̂ = rd/|rd| to the closest approach,
	t* = d̂·(c - o); miss if |o + t* d̂ - c| > radius; else t = t* - s if t* - s >= 0 else t* + s, s = √(r² - d²);
	hit if t > 0. Returns (hit, t).
	"""
	d_hat = rd / torch.sqrt((rd * rd).sum(dim=0))
	t = (d_hat * (center - ro)).sum(dim=0) / (d_hat * d_hat).sum(dim=0)
	off = ro + t * d_hat - center
	d = torch.sqrt((off * off).sum(dim=0))
	s = torch.sqrt(torch.clamp(radius * radius - d * d, min=0.0))
	t = torch.where(t - s >= 0.0, t - s, t + s)
	return (d <= radius) & (t > 0.0), t


def intersect_triangle(ro, rd, v0, v1, v2):
	"""
	Möller-Trumbore (C++ intersect_triangle, EPSILON = 1e-8): e1 = v1 - v0, e2 = v2 - v0, h = d × e2, a = e1·h
	(|a| < ε: parallel), f = 1/a, s = o - v0, u = f s·h ∈ [0, 1], q = s × e1, v = f d·q ≥ 0, u + v ≤ 1,
	t = f e2·q > ε. Returns (hit, t, u, v).
	"""
	eps = 1e-8
	e1, e2 = v1 - v0, v2 - v0
	h = torch.linalg.cross(rd, e2, dim=0)
	a = (e1 * h).sum(dim=0)
	ok = torch.abs(a) >= eps
	f = 1.0 / torch.where(ok, a, torch.ones_like(a))
	s = ro - v0
	u = f * (s * h).sum(dim=0)
	q = torch.linalg.cross(s, e1, dim=0)
	v = f * (rd * q).sum(dim=0)
	t = f * (e2 * q).sum(dim=0)
	hit = ok & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0) & (t > eps)
	return hit, t, u, v


def closest_point_on_triangle(query, v0, v1, v2):
	"""
	C++ BVH::closest_point_on_triangle: project onto the plane (n = (v1 - v0) × (v2 - v0) / |·|); if the projection is
	inside (all edge cross products agree with n) return it, else the closest of the three clamped edge points
	(v0v1, v1v2, v2v0, first strictly closer wins, starting from v0); degenerate triangles (|n| < 1e-10) return v0.
	All arguments (3, ...).
	"""
	e0 = v1 - v0
	n = torch.linalg.cross(e0, -(v0 - v2), dim=0)
	n_len = torch.sqrt((n * n).sum(dim=0))
	degenerate = n_len < 1e-10
	n = n / torch.where(degenerate, torch.ones_like(n_len), n_len)
	dist_to_plane = ((query - v0) * n).sum(dim=0)
	proj = query - dist_to_plane * n
	c0 = (torch.linalg.cross(v1 - v0, proj - v0, dim=0) * n).sum(dim=0)
	c1 = (torch.linalg.cross(v2 - v1, proj - v1, dim=0) * n).sum(dim=0)
	c2 = (torch.linalg.cross(v0 - v2, proj - v2, dim=0) * n).sum(dim=0)
	inside = (c0 >= 0.0) & (c1 >= 0.0) & (c2 >= 0.0)

	closest = v0
	min_dist_sq = ((query - v0) ** 2).sum(dim=0)
	for va, vb in ((v0, v1), (v1, v2), (v2, v0)):
		edge = vb - va
		t = ((query - va) * edge).sum(dim=0) / (edge * edge).sum(dim=0)
		t = torch.clamp(t, 0.0, 1.0)
		pt = va + t * edge
		dist_sq = ((query - pt) ** 2).sum(dim=0)
		better = dist_sq < min_dist_sq
		min_dist_sq = torch.where(better, dist_sq, min_dist_sq)
		closest = torch.where(better, pt, closest)
	closest = torch.where(inside, proj, closest)
	return torch.where(degenerate, v0, closest)


class _Stacks:
	"""Q traversal stacks of node indices, all starting with the root."""

	def __init__(self, n: int, size: int, device):
		self.data = torch.zeros((n, size), dtype=torch.int32, device=device)
		self.sp = torch.ones(n, dtype=torch.long, device=device)
		self.rows = torch.arange(n, device=device)
		self.size = size

	def pop(self):
		active = self.sp > 0
		self.sp = self.sp - active.long()
		node = self.data[self.rows, self.sp].long()  # inactive rows read slot 0: masked by `active`
		return node, active

	def push(self, node: torch.Tensor, cond: torch.Tensor) -> None:
		pos = self.sp.clamp(max=self.size - 1)
		self.data[self.rows, pos] = torch.where(cond, node, self.data[self.rows, pos].long()).to(torch.int32)
		self.sp = self.sp + cond.long()

	def empty(self) -> bool:
		return not bool((self.sp > 0).any())  # host sync

	def compact(self, keep: torch.Tensor) -> None:
		"""Keep only the rows `keep` (bool), e.g. queries still traversing."""
		self.data, self.sp = self.data[keep], self.sp[keep]
		self.rows = torch.arange(self.sp.shape[0], device=self.sp.device)


class BVH:
	"""
	BVH over points (3, N) or a triangle mesh (vertices (3, V), faces (3, F) long), on the device of the inputs.

	Batched queries take (3, Q) tensors (a single (3,) query is accepted and squeezed back).
	"""

	CHECK_EVERY = 8  # iterations between host checks for termination
	QUERY_CHUNK = 1 << 18  # queries traversed together

	def __init__(self, *args, leaf_size: int = 4):
		if len(args) == 1:
			points = args[0]
			self.primitive_type = 'point'
			self.primitives = points
			self.vertices = self.faces = None
			centroids, pmin, pmax = points, None, None
		elif len(args) == 2:
			vertices, faces = args
			self.primitive_type = 'triangle'
			self.vertices, self.faces = vertices, faces.long()
			tri = vertices[:, self.faces]  # (3, 3 corners, F)
			self.primitives = tri
			centroids = tri.sum(dim=1) / 3.0
			pmin, pmax = tri.amin(dim=1), tri.amax(dim=1)
		else:
			raise ValueError('BVH takes points (3, N) or (vertices (3, V), faces (3, F))')
		self.tree = build_lbvh(centroids, leaf_size, pmin, pmax)
		self.leaf_size = leaf_size
		self.n_primitives = centroids.shape[1]
		self.stack_size = int(self.tree.depth.max()) + 2  # host read once
		self.device = centroids.device

	# ---- original getters

	def get_num_nodes(self) -> int:
		return self.tree.n_nodes

	def get_num_primitives(self) -> int:
		return self.n_primitives

	def get_primitive_type(self) -> str:
		return self.primitive_type

	def get_gpu_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
		"""(n_nodes, 10) float64 rows [aabb_min, aabb_max, left, right, prim_start, prim_count], (N,) indices."""
		t = self.tree
		cols = [t.bounds_min.T, t.bounds_max.T]
		cols += [x[:, None].to(t.bounds_min.dtype) for x in (t.left, t.right, t.prim_start, t.prim_count)]
		return torch.cat(cols, dim=1).double(), t.prim_indices

	# ---- traversal helpers

	def _node(self, node):
		t = self.tree
		return (
			t.bounds_min[:, node],
			t.bounds_max[:, node],
			t.left[node],
			t.right[node],
			t.prim_start[node],
			t.prim_count[node],
		)

	def _leaf_slots(self, start, count, take):
		"""(Q, L) primitive indices of the popped leaves and their validity (`take`: queries at a leaf)."""
		j = torch.arange(self.leaf_size, device=self.device)
		valid = take[:, None] & (j[None, :] < count[:, None])
		slot = (start[:, None] + j[None, :]).clamp(max=self.n_primitives - 1)
		return self.tree.prim_indices[slot], valid

	def _chunks(self, n):
		for start in range(0, n, self.QUERY_CHUNK):
			yield start, min(n, start + self.QUERY_CHUNK)

	@staticmethod
	def _batch(q: torch.Tensor):
		single = q.ndim == 1
		return (q[:, None] if single else q), single

	# ---- queries

	def find_k_nearest(self, queries: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		k nearest points: (indices (Q, k), distances (Q, k)) sorted by distance (C++: max-heap, returned nearest
		first). Pruning as the C++: a node is skipped when its box is farther than the current k-th distance.
		Children are visited nearer first (the C++ pushes left, right; same result, fewer visits).
		"""
		if self.primitive_type != 'point':
			raise RuntimeError('find_k_nearest only works for point BVHs')
		q_all, single = self._batch(queries)
		k = min(int(k), self.n_primitives)
		n = q_all.shape[1]
		out_i = torch.empty((n, k), dtype=torch.long, device=self.device)
		out_d = torch.empty((n, k), dtype=q_all.dtype, device=self.device)
		pts = self.primitives
		for s, e in self._chunks(n):
			q = q_all[:, s:e]
			m = e - s
			rows = torch.arange(s, e, device=self.device)  # output row of each traversing query
			best_d = torch.full((m, k), float('inf'), dtype=q.dtype, device=self.device)
			best_i = torch.full((m, k), -1, dtype=torch.long, device=self.device)
			kth = torch.full((m,), float('inf'), dtype=q.dtype, device=self.device)
			stacks = _Stacks(m, self.stack_size, self.device)
			it = 0
			while True:
				node, active = stacks.pop()
				bmin, bmax, left, right, start, count = self._node(node)
				visit = active & ~(point_aabb_distance_sq(q, bmin, bmax) > kth)
				leaf = visit & (count > 0)
				internal = visit & (count == 0)
				# leaf: merge its points into the k best
				pid, valid = self._leaf_slots(start, count, leaf)
				diff = q[:, :, None] - pts[:, pid]
				d2 = torch.where(valid, (diff * diff).sum(dim=0), float('inf'))
				cand_d = torch.cat([best_d, d2], dim=1)
				cand_i = torch.cat([best_i, pid], dim=1)
				top = torch.topk(cand_d, k, dim=1, largest=False, sorted=False)
				best_d = top.values
				best_i = torch.gather(cand_i, 1, top.indices)
				kth = best_d.amax(dim=1)
				# internal: push the farther child first so that the nearer one is popped first
				lc, rc = left.clamp(min=0), right.clamp(min=0)
				dl = point_aabb_distance_sq(q, self.tree.bounds_min[:, lc], self.tree.bounds_max[:, lc])
				dr = point_aabb_distance_sq(q, self.tree.bounds_min[:, rc], self.tree.bounds_max[:, rc])
				right_first = dl >= dr  # ties: right popped first, as the original
				near, far = torch.where(right_first, rc, lc), torch.where(right_first, lc, rc)
				dn, df = torch.where(right_first, dr, dl), torch.where(right_first, dl, dr)
				stacks.push(far, internal & (df <= kth))
				stacks.push(near, internal & (dn <= kth))
				it += 1
				if it % self.CHECK_EVERY == 0:
					running = stacks.sp > 0
					n_running = int(running.sum())  # host sync
					if n_running < running.numel():
						# write back finished queries; keep traversing only the others (compaction)
						done = ~running
						order = torch.argsort(best_d[done], dim=1, stable=True)
						out_d[rows[done]] = torch.sqrt(torch.gather(best_d[done], 1, order))
						out_i[rows[done]] = torch.gather(best_i[done], 1, order)
						if n_running == 0:
							break
						q, rows = q[:, running], rows[running]
						best_d, best_i, kth = best_d[running], best_i[running], kth[running]
						stacks.compact(running)
		if single:
			return out_i[0], out_d[0]
		return out_i, out_d

	def find_closest_point(self, queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""(index, distance) of the closest point (C++ find_closest_point)."""
		idx, dist = self.find_k_nearest(queries, 1)
		return idx[..., 0], dist[..., 0]

	def find_points_in_radius(
		self, queries: torch.Tensor, radius: float, max_results: Optional[int] = None
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Points with |q - p| ≤ radius: (indices (Q, R), distances (Q, R), counts (Q,)), padded with -1 / inf, in
		traversal order. R = max_results (extra points are dropped, as the shader's max_results), or, if None, the
		largest count (one counting pass + one host read).
		"""
		if self.primitive_type != 'point':
			raise RuntimeError('find_points_in_radius only works for point BVHs')
		q_all, single = self._batch(queries)
		n = q_all.shape[1]
		if max_results is None:
			counts = self._radius_pass(q_all, radius, None)[2]
			max_results = max(int(counts.max()) if n else 0, 1)  # host sync
		idx, dist, counts = self._radius_pass(q_all, radius, max_results)
		if single:
			return idx[0], dist[0], counts[0]
		return idx, dist, counts

	def _radius_pass(self, q_all, radius, max_results):
		n = q_all.shape[1]
		r2 = float(radius) * float(radius)
		R = max_results or 1
		out_i = torch.full((n, R), -1, dtype=torch.long, device=self.device)
		out_d = torch.full((n, R), float('inf'), dtype=q_all.dtype, device=self.device)
		out_c = torch.zeros(n, dtype=torch.long, device=self.device)
		pts = self.primitives
		for s, e in self._chunks(n):
			q = q_all[:, s:e]
			m = e - s
			rows = torch.arange(m, device=self.device)
			cnt = torch.zeros(m, dtype=torch.long, device=self.device)
			res_i = torch.full((m, R), -1, dtype=torch.long, device=self.device)
			res_d = torch.full((m, R), float('inf'), dtype=q.dtype, device=self.device)
			stacks = _Stacks(m, self.stack_size, self.device)
			it = 0
			while True:
				node, active = stacks.pop()
				bmin, bmax, left, right, start, count = self._node(node)
				visit = active & ~(point_aabb_distance_sq(q, bmin, bmax) > r2)
				leaf = visit & (count > 0)
				pid, valid = self._leaf_slots(start, count, leaf)
				diff = q[:, :, None] - pts[:, pid]
				d2 = (diff * diff).sum(dim=0)
				inside = valid & (d2 <= r2)
				if max_results is not None:
					for j in range(self.leaf_size):
						put = inside[:, j] & (cnt < R)
						pos = cnt.clamp(max=R - 1)
						res_i[rows, pos] = torch.where(put, pid[:, j], res_i[rows, pos])
						res_d[rows, pos] = torch.where(put, torch.sqrt(d2[:, j]), res_d[rows, pos])
						cnt = cnt + put.long()
				else:
					cnt = cnt + inside.long().sum(dim=1)
				internal = visit & (count == 0)
				stacks.push(left.clamp(min=0), internal)  # C++ order: left pushed first, right popped first
				stacks.push(right.clamp(min=0), internal)
				it += 1
				if it % self.CHECK_EVERY == 0 and stacks.empty():
					break
			out_i[s:e], out_d[s:e], out_c[s:e] = res_i, res_d, cnt
		return out_i, out_d, out_c

	def intersect_ray(
		self,
		ray_origin: torch.Tensor,
		ray_dir: torch.Tensor,
		t_min: float = 0.0,
		t_max: float = float('inf'),
		sphere_radius: float = 0.01,
	):
		"""
		Closest hit along each ray: (hit (R,), t (R,), primitive (R,), barycentric (2, R)) as the C++ intersect_ray.
		Points are spheres of radius sphere_radius (t along the normalized direction); triangles use Möller-Trumbore
		(t in units of ray_dir). A node is skipped if the slab test misses or t_near > t_hit or t_far < t_min; a hit
		replaces the current one when t_min ≤ t < t_hit.

		Deviation for points (exact here, tree-dependent in the original): the C++ tests the boxes of the sphere
		centers, not of the spheres, and compares slab distances (units of ray_dir) with sphere distances (units of
		the normalized direction), so it can skip the node of the closest sphere. Here boxes are grown by
		sphere_radius and slab distances are scaled by |ray_dir|: the result is the closest sphere hit.
		"""
		ro_all, single = self._batch(ray_origin)
		rd_all, _ = self._batch(ray_dir)
		n = ro_all.shape[1]
		out = (
			torch.zeros(n, dtype=torch.bool, device=self.device),
			torch.full((n,), float(t_max), dtype=ro_all.dtype, device=self.device),
			torch.full((n,), -1, dtype=torch.long, device=self.device),
			torch.zeros((2, n), dtype=ro_all.dtype, device=self.device),
		)
		for s, e in self._chunks(n):
			ro, rd = ro_all[:, s:e], rd_all[:, s:e]
			m = e - s
			found = torch.zeros(m, dtype=torch.bool, device=self.device)
			hit_t = torch.full((m,), float(t_max), dtype=ro.dtype, device=self.device)
			hit_i = torch.full((m,), -1, dtype=torch.long, device=self.device)
			bary = torch.zeros((2, m), dtype=ro.dtype, device=self.device)
			if self.primitive_type == 'point':
				grow, scale = float(sphere_radius), torch.sqrt((rd * rd).sum(dim=0))
			else:
				grow, scale = 0.0, 1.0
			stacks = _Stacks(m, self.stack_size, self.device)
			it = 0
			while True:
				node, active = stacks.pop()
				bmin, bmax, left, right, start, count = self._node(node)
				box_hit, t_near, t_far = intersect_aabb(ro, rd, bmin - grow, bmax + grow)
				visit = active & box_hit & ~(t_near * scale > hit_t) & ~(t_far * scale < t_min)
				leaf = visit & (count > 0)
				pid, valid = self._leaf_slots(start, count, leaf)
				for j in range(self.leaf_size):  # leaf order, first strictly closer hit wins
					p = pid[:, j]
					if self.primitive_type == 'point':
						h, t = intersect_sphere(ro, rd, self.primitives[:, p], sphere_radius)
						u = v = torch.zeros_like(t)  # unused for spheres
					else:
						tri = self.primitives[:, :, p]
						h, t, u, v = intersect_triangle(ro, rd, tri[:, 0], tri[:, 1], tri[:, 2])
					take = valid[:, j] & h & (t >= t_min) & (t < hit_t)
					hit_t = torch.where(take, t, hit_t)
					hit_i = torch.where(take, p, hit_i)
					bary = torch.where(take, torch.stack([u, v]), bary)
					found = found | take
				internal = visit & (count == 0)
				stacks.push(left.clamp(min=0), internal)
				stacks.push(right.clamp(min=0), internal)
				it += 1
				if it % self.CHECK_EVERY == 0 and stacks.empty():
					break
			out[0][s:e], out[1][s:e], out[2][s:e], out[3][:, s:e] = found, hit_t, hit_i, bary
		if single:
			return out[0][0], out[1][0], out[2][0], out[3][:, 0]
		return out

	def closest_point_on_mesh(self, queries: torch.Tensor):
		"""(closest points (3, Q), distances (Q,), triangle indices (Q,)) as the C++ closest_point_on_mesh."""
		if self.primitive_type != 'triangle':
			raise RuntimeError('closest_point_on_mesh only works for triangle BVHs')
		q_all, single = self._batch(queries)
		n = q_all.shape[1]
		out_p = torch.empty_like(q_all)
		out_d = torch.empty(n, dtype=q_all.dtype, device=self.device)
		out_t = torch.empty(n, dtype=torch.long, device=self.device)
		for s, e in self._chunks(n):
			q = q_all[:, s:e]
			m = e - s
			min_d = torch.full((m,), float('inf'), dtype=q.dtype, device=self.device)
			best_p = torch.zeros_like(q)
			best_t = torch.full((m,), -1, dtype=torch.long, device=self.device)
			stacks = _Stacks(m, self.stack_size, self.device)
			it = 0
			while True:
				node, active = stacks.pop()
				bmin, bmax, left, right, start, count = self._node(node)
				visit = active & ~(point_aabb_distance_sq(q, bmin, bmax) > min_d * min_d)
				leaf = visit & (count > 0)
				pid, valid = self._leaf_slots(start, count, leaf)
				for j in range(self.leaf_size):
					tri = self.primitives[:, :, pid[:, j]]
					pt = closest_point_on_triangle(q, tri[:, 0], tri[:, 1], tri[:, 2])
					dist = torch.sqrt(((q - pt) ** 2).sum(dim=0))
					take = valid[:, j] & (dist < min_d)
					min_d = torch.where(take, dist, min_d)
					best_p = torch.where(take, pt, best_p)
					best_t = torch.where(take, pid[:, j], best_t)
				internal = visit & (count == 0)
				lc, rc = left.clamp(min=0), right.clamp(min=0)
				dl = point_aabb_distance_sq(q, self.tree.bounds_min[:, lc], self.tree.bounds_max[:, lc])
				dr = point_aabb_distance_sq(q, self.tree.bounds_min[:, rc], self.tree.bounds_max[:, rc])
				m2 = min_d * min_d
				left_first = dl < dr  # C++: if left closer push right then left
				near, far = torch.where(left_first, lc, rc), torch.where(left_first, rc, lc)
				dn, df = torch.where(left_first, dl, dr), torch.where(left_first, dr, dl)
				stacks.push(far, internal & (df <= m2))
				stacks.push(near, internal & (dn <= m2))
				it += 1
				if it % self.CHECK_EVERY == 0 and stacks.empty():
					break
			out_p[:, s:e], out_d[s:e], out_t[s:e] = best_p, min_d, best_t
		if single:
			return out_p[:, 0], out_d[0], out_t[0]
		return out_p, out_d, out_t

	def find_primitives_in_box(self, box_min: torch.Tensor, box_max: torch.Tensor) -> torch.Tensor:
		"""
		Primitives of the leaves whose boxes (and all ancestors' boxes) intersect the query box (C++ semantics: a
		candidate set; it depends on the leaves, so it differs from the original's tree). Breadth-first, one host
		read per level.
		"""
		t = self.tree
		frontier = torch.zeros(1, dtype=torch.long, device=self.device)
		result = []
		while frontier.numel():
			bmin, bmax = t.bounds_min[:, frontier], t.bounds_max[:, frontier]
			hit = ((bmin <= box_max[:, None]) & (bmax >= box_min[:, None])).all(dim=0)
			frontier = frontier[hit]
			leaf = t.prim_count[frontier] > 0
			for node in frontier[leaf].tolist():
				s = int(t.prim_start[node])
				result.append(t.prim_indices[s : s + int(t.prim_count[node])])
			inner = frontier[~leaf]
			frontier = torch.cat([t.left[inner], t.right[inner]])
		return torch.cat(result) if result else torch.zeros(0, dtype=torch.long, device=self.device)


class BoundingVolumeHierarchy:
	"""
	Python wrapper of the original (src/pointsastori/shape_3d.py): BoundingVolumeHierarchy(points) or
	BoundingVolumeHierarchy(vertices, faces) with (n, 3) arrays, imported to the configured device. Queries take
	(3,) or (3, Q) device tensors.
	"""

	def __init__(self, *args):
		if len(args) == 1:
			self.bound_object = BVH(config.as_wide(args[0], 3))
			self.primitive_type = 'point'
		elif len(args) == 2:
			vertices, faces = args
			self.bound_object = BVH(config.as_wide(vertices, 3), config.as_wide(faces, 3, dtype=torch.long))
			self.primitive_type = 'triangle'
		else:
			raise ValueError('BVH constructor takes 1 argument (points) or 2 arguments (vertices, faces)')

	def find_closest_point(self, query):
		return self.bound_object.find_closest_point(query)

	def find_k_nearest(self, query, k: int):
		return self.bound_object.find_k_nearest(query, k)

	def find_points_in_radius(self, query, radius: float):
		return self.bound_object.find_points_in_radius(query, radius)

	def intersect_ray(
		self, ray_origin, ray_dir, t_min: float = 0.0, t_max: float = np.inf, sphere_radius: float = 0.01
	):
		return self.bound_object.intersect_ray(ray_origin, ray_dir, t_min, t_max, sphere_radius)

	def closest_point_on_mesh(self, query):
		if self.primitive_type != 'triangle':
			raise ValueError('closest_point_on_mesh only works for triangle meshes')
		return self.bound_object.closest_point_on_mesh(query)

	def get_gpu_data(self):
		return self.bound_object.get_gpu_data()

	def get_num_nodes(self) -> int:
		return self.bound_object.get_num_nodes()

	def get_num_primitives(self) -> int:
		return self.bound_object.get_num_primitives()
