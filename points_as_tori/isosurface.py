"""
Isosurface extraction (port of `mesh_isocontour` in ext/points-as-tori/demo/demo_3d.py, which samples a field of a
TorusDistanceField on a grid and contours it with PyVista). The contouring is marching cubes in torch, on the device.

Marching cubes (Lorensen & Cline 1987) on samples v[i, j, k] at the grid vertices:
  * a corner is "positive" when v > isovalue; a cube's case is the 8-bit mask of its positive corners;
  * every grid edge whose ends differ in sign holds one vertex, at the linear zero of v - isovalue along the edge:
        p = p0 + t (p1 - p0),  t = (isovalue - v0) / (v1 - v0);
    vertices are shared between neighboring cubes (one per grid edge), so the mesh is indexed, not a triangle soup;
  * the triangles of each case come from a table.

The table is generated here instead of being copied: on each cube face the crossing points are joined by segments
(two crossings: one segment; four crossings, the ambiguous face: each positive corner is cut off by its own
segment); the segments of the six faces form closed loops (each crossing point lies on exactly two faces), and each
loop is triangulated as a fan. Segments are oriented so that the positive region of the face lies on their left
seen from outside the cube, which orients every triangle with its normal (right-hand rule) toward increasing
values: outward for a signed distance (negative inside). Because the face rule depends only on the four corner
signs of the face, two cubes sharing a face cut it the same way, and the mesh is closed wherever the grid encloses
the surface (the classic Lorensen table is not consistent on ambiguous faces and can leave holes). Interior
ambiguities (Chernyaev's cases) are resolved by the loop structure, without the trilinear-interior test of Lewiner et
al. 2003. Loops are triangulated without diagonals that lie in a cube face (a diagonal there could coincide with
one of the neighbor cube and make an edge shared by four triangles).
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Tuple

import numpy as np
import torch

# corner c = dx + 2 dy + 4 dz of the unit cube
CORNERS = np.array([[c & 1, (c >> 1) & 1, (c >> 2) & 1] for c in range(8)], dtype=np.int64)
# edges: pairs of corners differing in one coordinate; EDGE_AXIS = that coordinate; EDGE_ORIGIN = the lower corner
EDGES = np.array([(a, b) for a, b in itertools.combinations(range(8), 2) if np.abs(CORNERS[a] - CORNERS[b]).sum() == 1])
EDGE_AXIS = np.array([int(np.argmax(np.abs(CORNERS[b] - CORNERS[a]))) for a, b in EDGES])
EDGE_ORIGIN = CORNERS[EDGES[:, 0]]


def _faces():
	"""The 6 faces: (corners in counterclockwise order seen from outside, the edge between consecutive corners)."""
	edge_of = {frozenset((int(a), int(b))): e for e, (a, b) in enumerate(EDGES)}
	faces = []
	for axis in range(3):
		for side in (0, 1):
			corners = [c for c in range(8) if CORNERS[c][axis] == side]
			normal = np.zeros(3)
			normal[axis] = 1.0 if side else -1.0
			u = np.eye(3)[(axis + 1) % 3]
			v = np.cross(normal, u)  # (u, v, normal) right-handed: angle in (u, v) increases counterclockwise
			center = CORNERS[corners].mean(0)
			ang = [np.arctan2((CORNERS[c] - center) @ v, (CORNERS[c] - center) @ u) for c in corners]
			ccw = [corners[i] for i in np.argsort(ang)]
			faces.append((ccw, [edge_of[frozenset((ccw[i], ccw[(i + 1) % 4]))] for i in range(4)]))
	return faces


FACES = _faces()


def _case_triangles(case: int):
	"""Triangles (as triples of cube edges) of one corner-sign case."""
	pos = [(case >> c) & 1 for c in range(8)]
	nxt = {}
	for ccw, edges in FACES:
		crossing = [k for k in range(4) if pos[ccw[k]] != pos[ccw[(k + 1) % 4]]]
		if not crossing:
			continue
		if len(crossing) == 2:
			a, b = crossing
			pairs = [(a, b)]
		else:  # ambiguous face: cut off each positive corner c_k, i.e. join its edges k-1 and k
			pairs = [((k - 1) % 4, k) for k in range(4) if pos[ccw[k]]]
		for a, b in pairs:
			# P on edge a, Q on edge b: the left of P->Q holds the corners on the counterclockwise arc from edge b to
			# edge a, starting with c_{b+1}; orient so that this side is positive
			if not pos[ccw[(b + 1) % 4]]:
				a, b = b, a
			nxt[edges[a]] = edges[b]
	tris, seen = [], set()
	for start in sorted(nxt):
		if start in seen:
			continue
		loop, e = [], start
		while e not in seen:
			seen.add(e)
			loop.append(e)
			e = nxt[e]
		tris += _triangulate(loop)
	return tris


FACE_EDGES = [frozenset(edges) for _, edges in FACES]


def _on_common_face(a: int, b: int) -> bool:
	return any(a in f and b in f for f in FACE_EDGES)


def _triangulations(loop):
	"""All triangulations of a polygon (as index triples), first the fans in order of apex."""
	n = len(loop)
	if n < 3:
		return
	for apex in range(n):
		r = loop[apex:] + loop[:apex]
		yield [(r[0], r[i], r[i + 1]) for i in range(1, n - 1)]
	if n > 5:  # non-fan triangulations (recursive split on the edge (0, 1))
		for k in range(2, n):
			for left in _triangulations(loop[1 : k + 1]) if k > 2 else [[]]:
				for right in _triangulations([loop[0]] + loop[k:]) if n - k > 1 else [[]]:
					yield [(loop[0], loop[1], loop[k])] + left + right


def _triangulate(loop):
	"""
	Triangulate a loop so that no diagonal joins two points of a common cube face: such a diagonal lies in the face,
	where the neighbor cube may put the same diagonal (an edge of four triangles). Loop edges are face segments.
	"""
	n = len(loop)
	for tris in _triangulations(loop):
		diagonals = {frozenset(p) for t in tris for p in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
		diagonals -= {frozenset((loop[i], loop[(i + 1) % n])) for i in range(n)}
		if not any(_on_common_face(*d) for d in diagonals):
			return tris
	raise RuntimeError(f'no admissible triangulation for loop {loop}')


@lru_cache(maxsize=None)
def triangle_table() -> np.ndarray:
	"""(256, T, 3) cube-edge triples per case, padded with -1 (T = max triangles of a case)."""
	all_tris = [_case_triangles(c) for c in range(256)]
	t_max = max(len(t) for t in all_tris)
	table = np.full((256, t_max, 3), -1, dtype=np.int64)
	for c, tris in enumerate(all_tris):
		if tris:
			table[c, : len(tris)] = tris
	return table


def marching_cubes(
	values: torch.Tensor, bounds_min, bounds_max, isovalue: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Args:
		values: (X, Y, Z) samples at the grid vertices: values[i, j, k] at bounds_min + (i, j, k) · h,
			h = (bounds_max - bounds_min) / (shape - 1)
		bounds_min, bounds_max: (3,) sequences or tensors
	Returns:
		vertices (3, V) (dtype of `values`), faces (3, F) long; triangle normals point toward increasing values.
	One host read: the number of cubes the surface crosses (output size).
	"""
	if values.ndim != 3 or min(values.shape) < 2:
		raise ValueError(f'values must be (X, Y, Z) with every size >= 2, got {tuple(values.shape)}')
	device, dtype = values.device, values.dtype
	X, Y, Z = values.shape
	lo = torch.as_tensor(bounds_min, dtype=dtype, device=device).reshape(3)
	hi = torch.as_tensor(bounds_max, dtype=dtype, device=device).reshape(3)
	shape = torch.tensor([X, Y, Z], device=device)
	h = (hi - lo) / (shape - 1).to(dtype)

	# case index of every cube
	positive = values > isovalue
	case = torch.zeros((X - 1, Y - 1, Z - 1), dtype=torch.long, device=device)
	for c, (dx, dy, dz) in enumerate(CORNERS.tolist()):
		case |= positive[dx : X - 1 + dx, dy : Y - 1 + dy, dz : Z - 1 + dz].long() << c
	active = torch.nonzero((case != 0) & (case != 255))  # (A, 3) cube indices; host sync (output size)
	cases = case[active[:, 0], active[:, 1], active[:, 2]]

	table = torch.as_tensor(triangle_table(), device=device)  # (256, T, 3)
	tri_edges = table[cases]  # (A, T, 3) cube edges, -1 padding
	valid = tri_edges[:, :, 0] >= 0  # (A, T)

	# global id of a grid edge = (axis, lower corner): x-edges, then y-edges, then z-edges, row-major in (i, j, k)
	counts = [(X - 1) * Y * Z, X * (Y - 1) * Z, X * Y * (Z - 1)]
	dims = [(X - 1, Y, Z), (X, Y - 1, Z), (X, Y, Z - 1)]
	offsets = [0, counts[0], counts[0] + counts[1]]
	edge_axis = torch.as_tensor(EDGE_AXIS, device=device)
	edge_origin = torch.as_tensor(EDGE_ORIGIN, device=device)
	e = tri_edges.clamp(min=0)
	ijk = active[:, None, None, :] + edge_origin[e]  # (A, T, 3, 3) lower grid corner of each triangle edge
	axis = edge_axis[e]
	gid = torch.zeros_like(axis)
	for a in range(3):
		d1, d2, d3 = dims[a]
		ids = offsets[a] + (ijk[..., 0] * d2 + ijk[..., 1]) * d3 + ijk[..., 2]
		gid = torch.where(axis == a, ids, gid)

	gid_valid = gid[valid]  # (F, 3); host sync already paid for by `active`
	unique, inverse = torch.unique(gid_valid.reshape(-1), return_inverse=True)
	faces = inverse.reshape(-1, 3).T.contiguous()

	# vertex on each unique grid edge
	axis_u = torch.searchsorted(torch.tensor(offsets[1:], device=device), unique, right=True)
	local = unique - torch.as_tensor(offsets, device=device)[axis_u]
	d = torch.as_tensor(dims, device=device)[axis_u]  # (V, 3)
	k0 = local % d[:, 2]
	j0 = (local // d[:, 2]) % d[:, 1]
	i0 = local // (d[:, 2] * d[:, 1])
	p0 = torch.stack([i0, j0, k0], dim=0)  # (3, V)
	step = torch.nn.functional.one_hot(axis_u, 3).T  # (3, V)
	p1 = p0 + step
	v0 = values[p0[0], p0[1], p0[2]]
	v1 = values[p1[0], p1[1], p1[2]]
	t = (isovalue - v0) / (v1 - v0)
	vertices = lo[:, None] + (p0.to(dtype) + t * step.to(dtype)) * h[:, None]
	return vertices.contiguous(), faces


def mesh_isocontour(
	TDF,
	isoval: float = 0.0,
	grid_resolution: int = 64,
	eval_method: str = 'PAT',
	epsilon: float = 0.01,
	method: str = 'marching_cubes',
):
	"""
	Create mesh of contour of the torus SDF by contouring the SDF on a regular grid of the given resolution
	(demo_3d.py). The grid is the point cloud's bounding cube grown by |isovalue| + 10% of its largest extent.

	Args:
		TDF: TorusDistanceField with tori set
		isoval: isovalue to extract
		grid_resolution: samples per axis
		eval_method: 'PAT' (blended tori, TDF.evaluate_distance), 'GWN' (fast winding number, isovalue 0.5),
			'SHC' (signed_log_sum_exp), 'SSPD' (self_normalized_signed_log_sum_exp), 'LogSumExp' (log_sum_exp)
		epsilon: unused (kept for the original signature)
		method: 'marching_cubes' or 'flying_edges' (both run the marching cubes above: flying edges is a faster
			traversal of the same triangulation in VTK, not a different surface)
	Returns:
		vertices (3, V), faces (3, F) on the device
	"""
	if method.lower() not in ('marching_cubes', 'flying_edges'):
		raise ValueError(f"Unknown method '{method}'. Use 'flying_edges' or 'marching_cubes'")
	points = TDF.get_points()
	bbox_min, bbox_max = points.amin(dim=1), points.amax(dim=1)
	bbox_center = (bbox_min + bbox_max) / 2.0
	max_extent = torch.max(torch.abs(bbox_max - bbox_min))

	# Pad by isovalue to ensure isosurface doesn't touch boundary
	padding = abs(isoval) + max_extent * 0.1  # Add 10% extra margin
	grid_half_size = (max_extent / 2.0) + padding
	grid_min = bbox_center - grid_half_size
	grid_max = bbox_center + grid_half_size

	n = grid_resolution
	axes = [torch.linspace(0, 1, n, dtype=points.dtype, device=points.device) for _ in range(3)]
	xx, yy, zz = torch.meshgrid(*axes, indexing='ij')
	grid_points = grid_min[:, None] + torch.stack([xx, yy, zz]).reshape(3, -1) * (grid_max - grid_min)[:, None]

	isovalue = 0.0
	if eval_method == 'PAT':
		distances = TDF.evaluate_distance(grid_points, isovalue)
		isovalue = isoval
	elif eval_method == 'GWN':
		TDF.fast_winding_number_precompute()
		distances = TDF.fast_winding_number(grid_points)
		isovalue = 0.5
	elif eval_method == 'SHC':
		distances = TDF.signed_log_sum_exp(grid_points)
		isovalue = isoval
	elif eval_method == 'SSPD':
		distances = TDF.self_normalized_signed_log_sum_exp(grid_points)
		isovalue = isoval
	elif eval_method == 'LogSumExp':
		distances = TDF.log_sum_exp(grid_points)
		isovalue = isoval
	else:
		raise ValueError(f'unknown eval_method {eval_method!r}')
	return marching_cubes(distances.reshape(n, n, n), grid_min, grid_max, isovalue)
