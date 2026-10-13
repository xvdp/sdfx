"""
Linear BVH over points (Karras 2012), built with torch on any device, emitted in the node layout of the original
Points-as-Tori BVH (ext/points-as-tori/src/cpp/bvh.h): root = node 0, depth-first pre-order, per node
(aabb_min, aabb_max, left_child, right_child, prim_start, prim_count), prim_count = 0 for internal nodes, leaves
hold ≤ leaf_size primitives addressed through a primitive index array.

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


def _range_bounds(sorted_points: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor):
	"""
	min / max of sorted_points[:, lo..hi] per node, by a sparse table built and consumed level by level.
	Nodes are grouped by m = ⌊log2(hi - lo + 1)⌋ (one argsort; the group sizes are read on the host once), so each
	node is gathered only at its own level: O(N log N) table work, O(M) gathers, O(N) memory.
	"""
	n = sorted_points.shape[1]
	level = bit_length(hi - lo + 1) - 1
	n_levels = n.bit_length()
	by_level = torch.argsort(level, stable=True)
	counts = torch.bincount(level, minlength=n_levels).tolist()  # host sync: group sizes
	groups = torch.split(by_level, counts)
	bmin = torch.empty((3, lo.shape[0]), dtype=sorted_points.dtype, device=sorted_points.device)
	bmax = torch.empty_like(bmin)
	tmin, tmax = sorted_points, sorted_points  # level 0: T_0[x] = P[x]
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


def build_lbvh(points: torch.Tensor, leaf_size: int = 4) -> LBVH:
	"""
	Args:
		points: (3, N) tensor (any device / float dtype)
		leaf_size: max primitives per leaf (the original uses 4)

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
	codes = morton_codes(points)
	keys, order = torch.sort(codes, stable=True)
	sorted_points = points[:, order]

	if n == 1:
		z = torch.zeros(1, **long)
		return LBVH(points.clone(), points.clone(), z - 1, z - 1, z, z + 1, order, z)

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

	bmin, bmax = _range_bounds(sorted_points, lo_k, hi_k)

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
