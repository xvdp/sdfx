"""
Torch ports of the libigl / CGAL routines the original C++ calls (ext/points-as-tori/deps/libigl/include/igl):

	igl::copyleft::cgal::point_areas   per-point Voronoi-like areas from a local 2D Delaunay triangulation
	igl::octree                        octree of a point cloud (cells split until they hold ≤ 1 point)
	igl::fast_winding_number           Barnes-Hut winding number of an oriented point cloud (Barill et al. 2018):
	                                   precompute (area-weighted Taylor expansions per cell) and query

The neighbor lists (igl::knn in the original) come from points_as_tori.get_neighbors (exact k-NN, nearest first,
the point itself first).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional

import torch

from . import config

_EPS = 2.0**-53  # unit roundoff of float64
_ICC_ERRBOUND = (10.0 + 96.0 * _EPS) * _EPS  # Shewchuk's incircle filter bound (iccerrboundA)
_CCW_ERRBOUND = (3.0 + 16.0 * _EPS) * _EPS  # Shewchuk's orientation filter bound (ccwerrboundA)


# ========================================================================
# POINT AREAS (igl::copyleft::cgal::point_areas)
# ========================================================================


def _exact(v: float) -> Fraction:
	return Fraction(v)


def _orient_exact(a, b, c) -> int:
	"""sign of (a - c) × (b - c) in exact rational arithmetic (a, b, c: (x, y) floats)."""
	ax, ay, bx, by, cx, cy = (_exact(v) for v in (*a, *b, *c))
	d = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)
	return (d > 0) - (d < 0)


def _incircle_exact(a, b, c, d) -> int:
	"""sign of the incircle determinant (> 0: d inside the circle through a, b, c in counterclockwise order)."""
	ax, ay, bx, by, cx, cy, dx, dy = (_exact(v) for v in (*a, *b, *c, *d))
	adx, ady, bdx, bdy, cdx, cdy = ax - dx, ay - dy, bx - dx, by - dy, cx - dx, cy - dy
	alift, blift, clift = adx * adx + ady * ady, bdx * bdx + bdy * bdy, cdx * cdx + cdy * cdy
	det = alift * (bdx * cdy - cdx * bdy) + blift * (cdx * ady - adx * cdy) + clift * (adx * bdy - bdx * ady)
	return (det > 0) - (det < 0)


def _local_delaunay_areas(x: torch.Tensor, y: torch.Tensor, keep: torch.Tensor, stats: Optional[dict] = None):
	"""
	Area of vertex 0 of each local point set: (C, k) plane coordinates of the kept neighbors (vertex 0 = the point).

	The Delaunay triangles incident to vertex 0 are the triangles (0, a, b), a < b, with nonzero orientation whose
	circumcircle contains no other kept point (empty-circle property; it characterizes the Delaunay triangulation
	in general position). Orientation and incircle signs are decided in float64 with Shewchuk's static error bounds
	and recomputed exactly (rationals) when the bound does not decide; points exactly on a circumcircle do not
	invalidate a triangle.

	For each such triangle XYZ (X = vertex 0), with x = |YZ|, y = |XZ|, z = |XY| and the angle cosines
	cos X = (y² + z² - x²)/(2yz) etc., libigl adds
		½ T            if X is obtuse
		¼ T            if Y or Z is obtuse
		(T_Y + T_Z)/2  otherwise, T_V = T · b_V / Σ b,  b = (x cos X, y cos Y, z cos Z)
	where T = ¼ √((x+(y+z))(z-(x-y))(z+(x-y))(x+(y-z))) (Heron, in libigl's evaluation order).
	"""
	C, k = x.shape
	device = x.device
	x, y = x.double(), y.double()
	ia, ib = torch.triu_indices(k, k, offset=1, device=device)
	sel = ia > 0  # pairs (a, b) with 1 <= a < b
	ia, ib = ia[sel], ib[sel]  # T candidate triangles (0, a, b)

	px0, py0 = x[:, :1], y[:, :1]
	ax, ay, bx, by = x[:, ia], y[:, ia], x[:, ib], y[:, ib]  # (C, T)
	cand = keep[:, ia] & keep[:, ib] & keep[:, :1]

	# orientation of (0, a, b): (p0 - b) × (a - b) has the sign of (a - p0) × (b - p0)
	o1, o2 = (ax - px0) * (by - py0), (ay - py0) * (bx - px0)
	orient = o1 - o2
	o_bound = _CCW_ERRBOUND * (o1.abs() + o2.abs())
	o_sure = orient.abs() > o_bound
	o_sign = torch.sign(orient).to(torch.int64)

	# incircle of (0, a, b) against every neighbor d (C, T, k), in the counterclockwise orientation
	dx, dy = x[:, None, :], y[:, None, :]
	adx, ady = px0[:, :, None] - dx, py0[:, :, None] - dy
	bdx, bdy = ax[:, :, None] - dx, ay[:, :, None] - dy
	cdx, cdy = bx[:, :, None] - dx, by[:, :, None] - dy
	alift, blift, clift = adx * adx + ady * ady, bdx * bdx + bdy * bdy, cdx * cdx + cdy * cdy
	t1, t2 = bdx * cdy, cdx * bdy
	t3, t4 = cdx * ady, adx * cdy
	t5, t6 = adx * bdy, bdx * ady
	det = alift * (t1 - t2) + blift * (t3 - t4) + clift * (t5 - t6)
	permanent = (t1.abs() + t2.abs()) * alift + (t3.abs() + t4.abs()) * blift + (t5.abs() + t6.abs()) * clift
	bound = _ICC_ERRBOUND * permanent
	det = det * o_sign[:, :, None].to(det.dtype)  # counterclockwise
	d_idx = torch.arange(k, device=device)
	others = (
		keep[:, None, :] & (d_idx != 0) & (d_idx[None, None, :] != ia[None, :, None]) & (d_idx != ib[None, :, None])
	)
	inside_sure = others & (det > bound)
	uncertain = others & ~(det.abs() > bound)
	orient_uncertain = cand & ~o_sure

	delaunay = cand & (o_sign != 0) & ~inside_sure.any(dim=2)
	n_unc = int(uncertain.any(dim=2).logical_and(delaunay).sum()) + int(orient_uncertain.sum())  # host sync
	if n_unc:
		# exact decisions for the cases the float filter leaves open (near-degenerate configurations)
		xs, ys = x.cpu().tolist(), y.cpu().tolist()
		ia_l, ib_l = ia.tolist(), ib.tolist()
		check = (uncertain.any(dim=2) & delaunay) | orient_uncertain
		fix = []
		for c, t in torch.nonzero(check).tolist():
			a, b = ia_l[t], ib_l[t]
			p0, pa, pb = (xs[c][0], ys[c][0]), (xs[c][a], ys[c][a]), (xs[c][b], ys[c][b])
			s = _orient_exact(pa, pb, p0)
			ok = s != 0
			if ok:
				tri = (p0, pa, pb) if s > 0 else (p0, pb, pa)
				for d in range(1, k):
					if d in (a, b) or not bool(keep[c, d]):
						continue
					if _incircle_exact(*tri, (xs[c][d], ys[c][d])) > 0:
						ok = False
						break
			fix.append((c, t, ok))
		if stats is not None:
			stats['exact_checks'] = stats.get('exact_checks', 0) + len(fix)
		cc = torch.tensor([f[0] for f in fix], dtype=torch.long, device=device)
		tt = torch.tensor([f[1] for f in fix], dtype=torch.long, device=device)
		delaunay[cc, tt] = torch.tensor([f[2] for f in fix], dtype=torch.bool, device=device)

	# libigl's area of vertex X = 0 in triangle XYZ, Y = a, Z = b
	xl = torch.sqrt((ax - bx) ** 2 + (ay - by) ** 2)  # |YZ|
	yl = torch.sqrt((px0 - bx) ** 2 + (py0 - by) ** 2)  # |XZ|
	zl = torch.sqrt((ax - px0) ** 2 + (ay - py0) ** 2)  # |XY|
	cosX = (zl * zl + yl * yl - xl * xl) / (2 * yl * zl)
	cosY = (zl * zl + xl * xl - yl * yl) / (2 * xl * zl)
	cosZ = (xl * xl + yl * yl - zl * zl) / (2 * yl * xl)
	b0, b1, b2 = xl * cosX, yl * cosY, zl * cosZ
	bsum = b0 + b1 + b2
	full = 0.25 * torch.sqrt((xl + (yl + zl)) * (zl - (xl - yl)) * (zl + (xl - yl)) * (xl + (yl - zl)))
	partial = (b1 / bsum * full + b2 / bsum * full) / 2
	contrib = torch.where(cosX < 0, 0.5 * full, torch.where((cosY < 0) | (cosZ < 0), 0.25 * full, partial))
	area = torch.where(delaunay, contrib, 0.0).sum(dim=1)
	return torch.where(torch.isfinite(area), area, 0.0)


def point_areas(P: torch.Tensor, I: torch.Tensor, N: torch.Tensor, stats: Optional[dict] = None) -> torch.Tensor:  # noqa: E741
	"""
	igl::copyleft::cgal::point_areas: P (3, n), I (n, k) neighbors (row i starts with i), N (3, n) -> areas (n,).

	Per point i (libigl, CGAL Delaunay): neighbors whose normal has a positive dot product with the normal of I[i, 0]
	are kept; with ≤ 2 kept the area is 0; otherwise the kept points are centered and projected onto their two
	principal directions (PCA plane: the SVD scores U·S of libigl = projections on the eigenvectors of the 3×3
	covariance with the two largest eigenvalues; the in-plane rotation/reflection does not change the result), and
	the area of the point is accumulated over the Delaunay triangles around it (_local_delaunay_areas).
	Non-finite areas become 0. Computed in float64 and returned in the dtype of P.
	"""
	n, k = I.shape
	device = P.device
	out = torch.empty(n, dtype=torch.float64, device=device)
	t = (k - 1) * (k - 2) // 2
	chunk = max(1, config.CHUNK_ELEMENTS // max(1, t * k))
	P64, N64 = P.double(), N.double()
	for s in range(0, n, chunk):
		e = min(n, s + chunk)
		idx = I[s:e]
		nb = P64[:, idx]  # (3, C, k)
		nn = N64[:, idx]
		keep = (nn[:, :, :1] * nn).sum(dim=0) > 0  # (C, k)
		cnt = keep.sum(dim=1)
		w = keep.to(nb.dtype)
		mean = (nb * w).sum(dim=2) / cnt.clamp(min=1).to(nb.dtype)
		centered = (nb - mean[:, :, None]) * w  # dropped rows are zero: they do not change the principal directions
		cov = torch.einsum('icj,kcj->cik', centered, centered)
		_, vecs = torch.linalg.eigh(cov)  # ascending eigenvalues
		x = torch.einsum('icj,ci->cj', centered, vecs[:, :, 2])
		y = torch.einsum('icj,ci->cj', centered, vecs[:, :, 1])
		area = _local_delaunay_areas(x, y, keep, stats)
		out[s:e] = torch.where(cnt > 2, area, 0.0)
	return out.to(P.dtype)


# ========================================================================
# OCTREE (igl::octree)
# ========================================================================


@dataclass
class Octree:
	"""Cells: children (M, 8) (-1 for leaves), centers (3, M), widths (M,), and per point its cell at every level."""

	children: torch.Tensor
	centers: torch.Tensor
	widths: torch.Tensor
	point_cells: torch.Tensor  # (L, n): cell of each point at depth 0 .. L-1 (-1 below its leaf)
	counts: torch.Tensor  # (M,) points in each cell's subtree


def octree(P: torch.Tensor, max_depth: int = 64) -> Octree:
	"""
	igl::octree, built level by level. Root: center = (min + max)/2 of P, width = largest extent. A cell holding more
	than one point gets 8 children (all created, empty ones included) with centers parent ± width/4 per axis (child
	index = (x ≥ c_x) + 2 (y ≥ c_y) + 4 (z ≥ c_z)) and half the width; points go to the child of their octant.
	Deviation: depth is capped at `max_depth` (libigl: 30000), which only matters for coincident points.
	Cell numbering is breadth-first (libigl allocates depth-first); the cells are the same.
	"""
	n = P.shape[1]
	device, dtype = P.device, P.dtype
	lo, hi = P.amin(dim=1), P.amax(dim=1)
	centers: List[torch.Tensor] = [((lo + hi) / 2.0)[:, None]]
	widths: List[torch.Tensor] = [(hi - lo).amax()[None]]
	children: List[torch.Tensor] = []
	cell = torch.zeros(n, dtype=torch.long, device=device)
	point_cells = [cell.clone()]
	offset = 1  # cells allocated so far
	level_start, level_size = 0, 1
	octant_dirs = torch.tensor(
		[[1 if i & 1 else -1, 1 if i & 2 else -1, 1 if i & 4 else -1] for i in range(8)], dtype=dtype, device=device
	).T  # (3, 8)
	for _ in range(max_depth):
		local = cell - level_start
		valid = cell >= 0
		counts = torch.bincount(local[valid], minlength=level_size)
		split = counts > 1
		n_split = int(split.sum())  # host sync per level
		level_children = torch.full((level_size, 8), -1, dtype=torch.long, device=device)
		if n_split == 0:
			children.append(level_children)
			break
		rank = torch.cumsum(split.long(), 0) - 1
		base = offset + 8 * rank  # first child of each split cell
		level_children[split] = base[split][:, None] + torch.arange(8, device=device)
		children.append(level_children)
		c = torch.cat(centers, dim=1)[:, level_start : level_start + level_size][:, split]  # (3, S)
		w = torch.cat(widths)[level_start : level_start + level_size][split]
		h = w / 2
		centers.append((c[:, :, None] + octant_dirs[:, None, :] * (h / 2)[None, :, None]).reshape(3, -1))
		widths.append(h[:, None].expand(-1, 8).reshape(-1))
		# move points of split cells to their child cell
		moving = valid & split[local.clamp(min=0)]
		pc = torch.cat(centers[:-1], dim=1)[:, cell.clamp(min=0)]
		octant = (P[0] >= pc[0]).long() + 2 * (P[1] >= pc[1]).long() + 4 * (P[2] >= pc[2]).long()
		new_cell = base[local.clamp(min=0)] + octant
		cell = torch.where(moving, new_cell, torch.full_like(cell, -1))
		point_cells.append(cell.clone())
		level_start, level_size = offset, 8 * n_split
		offset += 8 * n_split
	else:
		children.append(torch.full((level_size, 8), -1, dtype=torch.long, device=device))
	M = offset
	ch = torch.cat(children)
	if ch.shape[0] < M:
		ch = torch.cat([ch, torch.full((M - ch.shape[0], 8), -1, dtype=torch.long, device=device)])
	pcells = torch.stack(point_cells)
	counts = torch.zeros(M, dtype=torch.long, device=device)
	valid = pcells >= 0
	counts.index_add_(0, pcells[valid], torch.ones(int(valid.sum()), dtype=torch.long, device=device))
	return Octree(ch[:M], torch.cat(centers, dim=1), torch.cat(widths), pcells, counts)


# ========================================================================
# FAST WINDING NUMBER (igl::fast_winding_number, point clouds)
# ========================================================================


@dataclass
class WindingNumberExpansions:
	tree: Octree
	CM: torch.Tensor  # (3, M) area-weighted centers of mass
	R: torch.Tensor  # (M,) largest distance of a point of the cell to CM
	EC0: torch.Tensor  # (3, M)   Σ a n
	EC1: torch.Tensor  # (3, 3, M) Σ a (p - CM) nᵀ
	EC2: torch.Tensor  # (3, 3, 3, M) Σ ½ a (p - CM)_i (p - CM) nᵀ
	leaf_point: torch.Tensor  # (M,) the point of a leaf cell (-1: empty, or several points at the depth cap)


def fast_winding_number_precompute(P: torch.Tensor, N: torch.Tensor, A: torch.Tensor, tree: Octree):
	"""
	Expansion order 2 (libigl): per cell, over the points of its subtree,
		CM = Σ a p / Σ a,  EC0 = Σ a n,  EC1 = Σ a (p - CM) nᵀ,  EC2_i = Σ ½ (p - CM)_i a (p - CM) nᵀ,
		R = max |p - CM|.
	"""
	device, dtype = P.device, P.dtype
	M = tree.children.shape[0]
	pc = tree.point_cells  # (L, n)
	valid = pc >= 0
	L = pc.shape[0]
	cells = pc[valid]
	pts = torch.arange(P.shape[1], device=device).expand(L, -1)[valid]
	a = A[pts]
	area = torch.zeros(M, dtype=dtype, device=device).index_add_(0, cells, a)
	CM = torch.zeros((M, 3), dtype=dtype, device=device).index_add_(0, cells, a[:, None] * P[:, pts].T)
	CM = (CM / area[:, None]).T  # empty cells: NaN, as libigl
	EC0 = torch.zeros((M, 3), dtype=dtype, device=device).index_add_(0, cells, a[:, None] * N[:, pts].T).T
	d = P[:, pts] - CM[:, cells]  # (3, K)
	an = a * N[:, pts]  # (3, K)
	ec1 = d[:, None, :] * an[None, :, :]  # (3, 3, K): (p - CM)_u (a n)_v
	EC1 = torch.zeros((M, 9), dtype=dtype, device=device).index_add_(0, cells, ec1.reshape(9, -1).T).T.reshape(3, 3, M)
	ec2 = 0.5 * d[:, None, None, :] * ec1[None]  # (3, 3, 3, K)
	EC2 = torch.zeros((M, 27), dtype=dtype, device=device).index_add_(0, cells, ec2.reshape(27, -1).T)
	EC2 = EC2.T.reshape(3, 3, 3, M)
	R = torch.zeros(M, dtype=dtype, device=device).scatter_reduce_(0, cells, torch.sqrt((d * d).sum(0)), 'amax')
	# leaves: the point a leaf holds (cells hold ≤ 1 point except at the depth cap)
	is_leaf = tree.children[:, 0] < 0
	leaf_point = torch.full((M,), -1, dtype=torch.long, device=device)
	single = is_leaf & (tree.counts == 1)
	leaf_point.index_put_((cells,), pts, accumulate=False)
	leaf_point = torch.where(single, leaf_point, torch.full_like(leaf_point, -1))
	return WindingNumberExpansions(tree, CM, R, EC0, EC1, EC2, leaf_point)


def _direct_eval(loc: torch.Tensor, anorm: torch.Tensor) -> torch.Tensor:
	"""libigl direct_eval: loc·anorm / (4π |loc|³), and ½ when |loc| = 0."""
	r = torch.sqrt((loc * loc).sum(0))
	val = (loc * anorm).sum(0) / (4.0 * math.pi * (r * r * r))
	return torch.where(r == 0, torch.full_like(r, 0.5), val)


def _expansion_eval(loc: torch.Tensor, ex: WindingNumberExpansions, cells: torch.Tensor) -> torch.Tensor:
	"""
	libigl expansion_eval at loc = CM - q (3, K), order 2:
		w = loc·EC0 / (4π r³)
		  + Σ_uv [δ_uv / (4π r³) - 3 loc_u loc_v / (4π r⁵)] EC1_uv
		  + Σ_i Σ_uv [15 loc_i loc_u loc_v / (4π r⁷) - 3 (δ_uv loc_i + δ_ui loc_v + δ_vi loc_u) / (4π r⁵)] EC2_iuv
	"""
	wn = _direct_eval(loc, ex.EC0[:, cells])
	r = torch.sqrt((loc * loc).sum(0))
	pi4_r3 = 4.0 * math.pi * r * r * r
	pi4_r5 = pi4_r3 * r * r
	pi4_r7 = pi4_r5 * r * r
	eye = torch.eye(3, dtype=loc.dtype, device=loc.device)
	second = loc[:, None] * loc[None, :] * (-3.0 / pi4_r5) + eye[:, :, None] * (1.0 / pi4_r3)
	wn = wn + (second * ex.EC1[:, :, cells]).sum(dim=(0, 1))
	locTloc = loc[:, None] * loc[None, :] / pi4_r7  # (3, 3, K)
	third = 15.0 * loc[:, None, None] * locTloc[None]  # (i, u, v, K)
	diag = (
		eye[None, :, :, None] * loc[:, None, None, :]
		+ eye[:, :, None, None] * loc[None, None, :, :]
		+ eye[:, None, :, None] * loc[None, :, None, :]
	)  # δ_uv loc_i + δ_ui loc_v + δ_vi loc_u
	third = third + (-3.0 / pi4_r5) * diag
	return wn + (third * ex.EC2[:, :, :, cells]).sum(dim=(0, 1, 2))


def fast_winding_number(
	P: torch.Tensor,
	N: torch.Tensor,
	A: torch.Tensor,
	ex: WindingNumberExpansions,
	Q: torch.Tensor,
	beta: float = 2.0,
	chunk: int = 1 << 16,
) -> torch.Tensor:
	"""
	libigl query (point clouds): starting from the root, a cell's children with points are either far
	(|CM - q| > β R: a leaf child is summed directly, an inner child through its expansion) or near (visited next).
	Leaves reached as near cells are summed directly. Direct terms use the point areas `A` (the original passes the
	TorusDistanceField's areas here, while the expansions were built with the areas of the precompute).
	Breadth-first over (query, cell) pairs, one host read per level.
	"""
	tree = ex.tree
	ch = tree.children
	out = torch.zeros(Q.shape[1], dtype=P.dtype, device=P.device)
	aN = N * A
	for s in range(0, Q.shape[1], chunk):
		e = min(Q.shape[1], s + chunk)
		q = Q[:, s:e]
		wn = torch.zeros(e - s, dtype=P.dtype, device=P.device)
		pq = torch.arange(e - s, device=P.device)
		pn = torch.zeros(e - s, dtype=torch.long, device=P.device)
		while pq.numel():
			leaf = ch[pn, 0] < 0
			# near leaves: direct sum over their point
			lp = ex.leaf_point[pn[leaf]]
			has = lp >= 0
			if bool(has.any()):
				qq, pp = pq[leaf][has], lp[has]
				wn.index_add_(0, qq, _direct_eval(P[:, pp] - q[:, qq], aN[:, pp]))
			wn = _leaves_with_many_points(wn, q, pq[leaf], pn[leaf], ex, P, aN)
			# inner cells: classify the children
			qi, ni = pq[~leaf], pn[~leaf]
			kids = ch[ni]  # (K, 8)
			qk = qi[:, None].expand(-1, 8).reshape(-1)
			kk = kids.reshape(-1)
			nonempty = tree.counts[kk] > 0
			qk, kk = qk[nonempty], kk[nonempty]
			loc = ex.CM[:, kk] - q[:, qk]
			far = torch.sqrt((loc * loc).sum(0)) > beta * ex.R[kk]
			kid_leaf = ch[kk, 0] < 0
			# far leaf child: direct; far inner child: expansion
			fl = far & kid_leaf
			lp = ex.leaf_point[kk[fl]]
			has = lp >= 0
			if bool(has.any()):
				wn.index_add_(0, qk[fl][has], _direct_eval(P[:, lp[has]] - q[:, qk[fl][has]], aN[:, lp[has]]))
			wn = _leaves_with_many_points(wn, q, qk[fl], kk[fl], ex, P, aN)
			fi = far & ~kid_leaf
			if bool(fi.any()):
				wn.index_add_(0, qk[fi], _expansion_eval(loc[:, fi], ex, kk[fi]))
			pq, pn = qk[~far], kk[~far]
		out[s:e] = wn
	return out


def _leaves_with_many_points(wn, q, pq, pn, ex, P, aN):
	"""Direct sums for leaves holding several points (only at the depth cap: coincident points)."""
	many = ex.tree.counts[pn] > 1
	if not bool(many.any()):
		return wn
	pc = ex.tree.point_cells
	for qi, cell in zip(pq[many].tolist(), pn[many].tolist()):
		pts = torch.nonzero((pc == cell).any(dim=0)).reshape(-1)
		wn[qi] += _direct_eval(P[:, pts] - q[:, qi : qi + 1], aN[:, pts]).sum()
	return wn
