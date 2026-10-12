"""
Torch port of training/mesh_sampling_jax.py: on-the-fly query sampling and ground-truth signed distances for training.

Function names are kept. Differences in form, not in math:
  * layout is structure-of-arrays (config.py): vertices (3, V), faces (3, F) int64, queries (3, ..., n);
  * the original vmaps a per-neighborhood function over the n_points neighborhoods; here every function takes all
    neighborhoods at once (bboxes (3, P), samples (3, P, n));
  * brute-force query x face evaluations are chunked over queries (config.CHUNK_ELEMENTS pairs per chunk).

Randomness: JAX keys are replaced by a torch.Generator. Every random number is drawn through `_uniform`, in the same
order and with the same per-neighborhood shapes as the JAX code consumes its keys, so tests can substitute the exact JAX
draws (tests/points_as_tori/test_training_parity.py) and compare results value by value.

Sampling formulas (as in the original):
  * area-weighted face choice (jax.random.choice with p, replace=True): c = cumsum(p), r = c[-1] (1 - u),
    face = searchsorted(c, r)  (inverse-CDF sampling);
  * uniform point in a triangle: α = √u (1 - v), β = v √u, γ = 1 - α - β  (density uniform w.r.t. area);
  * sign of the ground truth from the generalized winding number w(q) = Σ_f Ω_f(q) (solid angles, 4π inside):
    inside iff w > π;  Ω_f = 2 atan2(det[a b c], |a||b||c| + (a·b)|c| + (b·c)|a| + (c·a)|b|)  (Van Oosterom–Strackee).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from .. import config
from ..shape_2d import cross, dot, norm

EPS = 1e-10


def _uniform(shape, generator: Optional[torch.Generator], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
	"""Every random draw of this module goes through here: standard uniform on [0, 1)."""
	return torch.rand(shape, generator=generator, dtype=dtype, device=device)


def closest_point_on_triangle(
	query: torch.Tensor,  # (3, ...)
	v0: torch.Tensor,  # (3, ...)
	v1: torch.Tensor,  # (3, ...)
	v2: torch.Tensor,  # (3, ...)
) -> torch.Tensor:
	"""
	Find closest point on triangle to query point (broadcasting over the trailing axes).

	Args:
		query: Query point(s)
		v0, v1, v2: Triangle vertices

	Returns:
		Closest point on triangle (broadcast shape (3, ...))
	"""
	# Edges
	e0 = v1 - v0
	e1 = v2 - v1  # noqa: F841 (as in the original)
	e2 = v0 - v2

	# Face normal
	n = cross(e0, -e2)
	n_len = norm(n)

	# Handle degenerate triangle
	unit_x = torch.tensor([1.0, 0.0, 0.0], dtype=n.dtype, device=n.device).reshape((3,) + (1,) * (n.ndim - 1))
	n_normalized = torch.where(n_len < EPS, unit_x, n / n_len)

	# Project query onto plane
	v0_to_query = query - v0
	dist_to_plane = dot(v0_to_query, n_normalized)
	proj = query - dist_to_plane * n_normalized

	# Check if projection is inside triangle using barycentric sign test
	c0 = cross(v1 - v0, proj - v0)
	c1 = cross(v2 - v1, proj - v1)
	c2 = cross(v0 - v2, proj - v2)

	inside = (dot(c0, n_normalized) >= 0.0) & (dot(c1, n_normalized) >= 0.0) & (dot(c2, n_normalized) >= 0.0)

	# If inside, return projection
	# Otherwise, find closest point on edges

	def closest_on_edge(va: torch.Tensor, vb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Find closest point on edge and its squared distance."""
		edge = vb - va
		edge_len_sq = dot(edge, edge)
		t = dot(query - va, edge) / torch.clamp_min(edge_len_sq, EPS)
		t = torch.clamp(t, 0.0, 1.0)
		pt = va + t * edge
		dist_sq = torch.sum((query - pt) ** 2, dim=0)
		return pt, dist_sq

	# Check all three edges
	pt0, dist_sq0 = closest_on_edge(v0, v1)
	pt1, dist_sq1 = closest_on_edge(v1, v2)
	pt2, dist_sq2 = closest_on_edge(v2, v0)

	# Find minimum
	min_dist_sq = torch.minimum(torch.minimum(dist_sq0, dist_sq1), dist_sq2)

	# Select closest edge point
	edge_closest = torch.where(dist_sq0 == min_dist_sq, pt0, torch.where(dist_sq1 == min_dist_sq, pt1, pt2))

	# Return projection if inside, otherwise closest edge point
	return torch.where(inside, proj, edge_closest)


def unsigned_distance_to_triangle(
	query: torch.Tensor,  # (3, ...)
	v0: torch.Tensor,
	v1: torch.Tensor,
	v2: torch.Tensor,
) -> torch.Tensor:
	"""
	Compute unsigned distance from query point(s) to triangle(s).

	Returns:
		Unsigned distance(s) of shape (...)
	"""
	closest_pt = closest_point_on_triangle(query, v0, v1, v2)
	return norm(query - closest_pt)


def gwn_single_face(
	query: torch.Tensor,  # (3, ...)
	v0: torch.Tensor,
	v1: torch.Tensor,
	v2: torch.Tensor,
) -> torch.Tensor:
	"""
	Compute solid angle contribution of triangle(s) to generalized winding number.

	Port of the GWN computation in shape_3d.cpp (via the JAX version).

	Returns:
		Solid angle contribution(s) (...)
	"""
	# Vectors from query to vertices
	a_vec = v0 - query
	b_vec = v1 - query
	c_vec = v2 - query

	# Lengths
	a = norm(a_vec)
	b = norm(b_vec)
	c = norm(c_vec)

	# Solid angle formula from C++:
	# numer = -a_z * b_y * c_x + a_y * b_z * c_x + a_z * b_x * c_y
	# - a_x * b_z * c_y - a_y * b_x * c_z + a_x * b_y * c_z
	# This is the determinant det([a, b, c])
	numer = (
		-a_vec[2] * b_vec[1] * c_vec[0]
		+ a_vec[1] * b_vec[2] * c_vec[0]
		+ a_vec[2] * b_vec[0] * c_vec[1]
		- a_vec[0] * b_vec[2] * c_vec[1]
		- a_vec[1] * b_vec[0] * c_vec[2]
		+ a_vec[0] * b_vec[1] * c_vec[2]
	)

	denom = a * b * c + dot(a_vec, b_vec) * c + dot(b_vec, c_vec) * a + dot(c_vec, a_vec) * b

	return 2.0 * torch.atan2(numer, denom)


def _triangle_vertices(vertices: torch.Tensor, faces: torch.Tensor):
	# Handle invalid indices by clamping to 0 (will be masked out anyway)
	safe_faces = torch.clamp_min(faces, 0)
	return vertices[:, safe_faces[0]], vertices[:, safe_faces[1]], vertices[:, safe_faces[2]]  # (3, max_faces) each


def compute_face_properties(
	vertices: torch.Tensor,  # (3, max_vertices)
	faces: torch.Tensor,  # (3, max_faces) - integer indices
	face_mask: torch.Tensor,  # (max_faces,) - bool mask
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Compute face normals, areas, and centroids for all faces.

	Args:
		vertices: Vertex positions (padded)
		faces: Face indices (padded, invalid faces have -1)
		face_mask: Boolean mask indicating valid faces

	Returns:
		normals: (3, max_faces) - unit normals (zero for invalid faces)
		areas: (max_faces,) - face areas (zero for invalid faces)
		centroids: (3, max_faces) - face centers (zero for invalid faces)
	"""
	v0, v1, v2 = _triangle_vertices(vertices, faces)

	# Centroids
	centroids = (v0 + v1 + v2) / 3.0

	# Area normals (unnormalized)
	area_normals = cross(v1 - v0, v2 - v0)  # (3, max_faces)
	double_areas = norm(area_normals)  # (max_faces,)
	areas = 0.5 * double_areas

	# Normalized normals (handle zero-area faces)
	normals = area_normals / torch.clamp_min(double_areas, EPS)

	# Mask out invalid faces
	normals = torch.where(face_mask, normals, 0.0)
	areas = torch.where(face_mask, areas, 0.0)
	centroids = torch.where(face_mask, centroids, 0.0)

	return normals, areas, centroids


def sample_bbox_uniform(
	bbox_min: torch.Tensor,  # (3, P)
	bbox_max: torch.Tensor,  # (3, P)
	n: int,
	generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
	"""
	Sample points uniformly in bounding boxes (one box per neighborhood).

	Returns:
		samples: (3, P, n) points in each bbox
	"""
	P = bbox_min.shape[1]
	# drawn as (P, n, 3), i.e. per neighborhood the (n, 3) array of the original
	samples = _uniform((P, n, 3), generator, bbox_min.dtype, bbox_min.device).permute(2, 0, 1)
	return bbox_min[:, :, None] + samples * (bbox_max - bbox_min)[:, :, None]


def _area_weights_in_bbox(bbox_min, bbox_max, v0, v1, v2, face_areas, face_mask):
	"""(P, max_faces) normalized sampling weights: area of valid faces whose AABB intersects each bbox."""
	# Check which faces intersect bbox; approximate by just checking whether triangle bounding box intersects bbox
	tri_bbox_min = torch.minimum(torch.minimum(v0, v1), v2)  # (3, max_faces)
	tri_bbox_max = torch.maximum(torch.maximum(v0, v1), v2)

	# AABB-AABB intersection test, (P, max_faces)
	intersects = (
		(tri_bbox_min[0][None, :] <= bbox_max[0][:, None])
		& (tri_bbox_max[0][None, :] >= bbox_min[0][:, None])
		& (tri_bbox_min[1][None, :] <= bbox_max[1][:, None])
		& (tri_bbox_max[1][None, :] >= bbox_min[1][:, None])
		& (tri_bbox_min[2][None, :] <= bbox_max[2][:, None])
		& (tri_bbox_max[2][None, :] >= bbox_min[2][:, None])
	)

	# Combine with face mask
	valid_faces = face_mask[None, :] & intersects

	# Compute sampling weights (area for valid faces, 0 otherwise)
	weights = torch.where(valid_faces, face_areas[None, :], 0.0)

	# Normalize weights (handle case where no faces intersect)
	total_weight = torch.sum(weights, dim=-1, keepdim=True)
	return weights / torch.clamp_min(total_weight, EPS)


def _choice(weights: torch.Tensor, n: int, generator: Optional[torch.Generator]) -> torch.Tensor:
	"""
	jax.random.choice(key, n_faces, shape=(n,), p=weights) for each row of weights (P, n_faces) -> (P, n) indices.
	Inverse-CDF sampling, same arithmetic as JAX: c = cumsum(p), r = c[-1] (1 - u), searchsorted(c, r, 'left').
	"""
	p_cuml = torch.cumsum(weights, dim=-1)
	u = _uniform((weights.shape[0], n), generator, weights.dtype, weights.device)
	r = p_cuml[:, -1:] * (1 - u)
	ind = torch.searchsorted(p_cuml, r)
	return torch.clamp_max(ind, weights.shape[1] - 1)  # out-of-range gathers clamp in JAX


def _sample_triangles(face_indices, v0, v1, v2, generator):
	"""Uniform samples on the chosen faces: (P, n) face indices -> (3, P, n) points."""
	P, n = face_indices.shape
	u = _uniform((P, n), generator, v0.dtype, v0.device)
	v = _uniform((P, n), generator, v0.dtype, v0.device)

	sqrt_u = torch.sqrt(u)
	alpha = sqrt_u * (1.0 - v)
	beta = v * sqrt_u
	gamma = 1.0 - alpha - beta

	# Vertices for sampled faces
	sampled_v0 = v0[:, face_indices]  # (3, P, n)
	sampled_v1 = v1[:, face_indices]
	sampled_v2 = v2[:, face_indices]

	# Compute samples
	return alpha * sampled_v0 + beta * sampled_v1 + gamma * sampled_v2


def sample_zero_set_single_bbox(
	bbox_min: torch.Tensor,  # (3, P)
	bbox_max: torch.Tensor,  # (3, P)
	vertices: torch.Tensor,  # (3, max_vertices)
	faces: torch.Tensor,  # (3, max_faces)
	face_normals: torch.Tensor,  # (3, max_faces)
	face_areas: torch.Tensor,  # (max_faces,)
	face_mask: torch.Tensor,  # (max_faces,)
	n: int,
	generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
	"""
	Sample points uniformly on mesh surface within a bounding box (not exact), for each of the P bboxes.

	Strategy:
	1. Test all faces for bbox intersection (brute force)
	2. Sample faces proportionally to area
	3. Sample points uniformly on selected faces

	Returns:
		samples: (3, P, n) points on mesh surface within bbox
	"""
	v0, v1, v2 = _triangle_vertices(vertices, faces)
	weights = _area_weights_in_bbox(bbox_min, bbox_max, v0, v1, v2, face_areas, face_mask)

	# Sample face indices
	face_indices = _choice(weights, n, generator)

	# Sample triangle
	return _sample_triangles(face_indices, v0, v1, v2, generator)


def _mesh_distances_chunked(queries, vertices, faces, face_mask, signed: bool) -> torch.Tensor:
	"""
	Brute force over (query, face) pairs, chunked over queries.
	unsigned = min_f |q - closest_f(q)| over valid faces;  signed: sign = -1 where Σ_f Ω_f(q) > π (inside).
	"""
	v0, v1, v2 = _triangle_vertices(vertices, faces)
	v0, v1, v2 = v0[:, None, :], v1[:, None, :], v2[:, None, :]  # (3, 1, max_faces)
	n_queries = queries.shape[1]
	max_faces = faces.shape[1]
	chunk = max(1, config.CHUNK_ELEMENTS // max(1, max_faces))

	out = torch.empty(n_queries, dtype=queries.dtype, device=queries.device)
	for start in range(0, n_queries, chunk):
		q = queries[:, start : start + chunk, None]  # (3, Qc, 1)

		# Unsigned distance: mask invalid faces with inf, minimum over faces
		distances_all = unsigned_distance_to_triangle(q, v0, v1, v2)  # (Qc, max_faces)
		unsigned_dist = torch.where(face_mask[None, :], distances_all, torch.inf).amin(dim=1)

		if signed:
			# Sign from GWN
			solid_angles = torch.where(face_mask[None, :], gwn_single_face(q, v0, v1, v2), 0.0)
			gwn = torch.sum(solid_angles, dim=1)
			unsigned_dist = torch.where(gwn > torch.pi, -1.0, 1.0).to(unsigned_dist.dtype) * unsigned_dist

		out[start : start + chunk] = unsigned_dist
	return out


def unsigned_distance_to_mesh(
	queries: torch.Tensor,  # (3, n_queries)
	vertices: torch.Tensor,  # (3, max_vertices)
	faces: torch.Tensor,  # (3, max_faces)
	face_mask: torch.Tensor,  # (max_faces,)
) -> torch.Tensor:
	"""
	Compute unsigned distance from query points to mesh (brute force).

	Returns:
		distances: (n_queries,) unsigned distances to mesh
	"""
	return _mesh_distances_chunked(queries, vertices, faces, face_mask, signed=False)


def gwn_mesh(
	queries: torch.Tensor,  # (3, n_queries)
	vertices: torch.Tensor,  # (3, max_vertices)
	faces: torch.Tensor,  # (3, max_faces)
	face_mask: torch.Tensor,  # (max_faces,)
) -> torch.Tensor:
	"""
	Compute generalized winding number for query points.

	GWN is the sum of solid angles subtended by all faces.

	Returns:
		gwn: (n_queries,) generalized winding numbers (sum of solid angles; 4π inside a closed mesh)
	"""
	v0, v1, v2 = _triangle_vertices(vertices, faces)
	v0, v1, v2 = v0[:, None, :], v1[:, None, :], v2[:, None, :]
	n_queries = queries.shape[1]
	chunk = max(1, config.CHUNK_ELEMENTS // max(1, faces.shape[1]))
	out = torch.empty(n_queries, dtype=queries.dtype, device=queries.device)
	for start in range(0, n_queries, chunk):
		q = queries[:, start : start + chunk, None]
		# Mask invalid faces, sum over all faces for each query
		solid_angles = torch.where(face_mask[None, :], gwn_single_face(q, v0, v1, v2), 0.0)
		out[start : start + chunk] = torch.sum(solid_angles, dim=1)
	return out


def signed_distance_to_mesh(
	queries: torch.Tensor,  # (3, n_queries)
	vertices: torch.Tensor,  # (3, max_vertices)
	faces: torch.Tensor,  # (3, max_faces)
	face_mask: torch.Tensor,  # (max_faces,)
) -> torch.Tensor:
	"""
	Compute signed distance from query points to mesh.

	Uses unsigned distance, and GWN for sign determination.
	Negative inside, positive outside.

	Returns:
		signed_distances: (n_queries,)
	"""
	return _mesh_distances_chunked(queries, vertices, faces, face_mask, signed=True)


def sample_zero_set_with_normals(
	bbox_min: torch.Tensor,  # (3, P)
	bbox_max: torch.Tensor,  # (3, P)
	vertices: torch.Tensor,  # (3, max_vertices)
	faces: torch.Tensor,  # (3, max_faces)
	face_normals: torch.Tensor,  # (3, max_faces)
	face_areas: torch.Tensor,  # (max_faces,)
	face_mask: torch.Tensor,  # (max_faces,)
	n: int,
	generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Sample points on mesh surface with their normals.

	Returns:
		samples: (3, P, n) points
		normals: (3, P, n) surface normals at samples
	"""
	v0, v1, v2 = _triangle_vertices(vertices, faces)
	weights = _area_weights_in_bbox(bbox_min, bbox_max, v0, v1, v2, face_areas, face_mask)

	# Sample faces
	face_indices = _choice(weights, n, generator)

	# Sample triangle; get vertices and normals for sampled faces
	samples = _sample_triangles(face_indices, v0, v1, v2, generator)
	sampled_normals = face_normals[:, face_indices]

	return samples, sampled_normals


def sample_narrow_band_single_bbox(
	bbox_min: torch.Tensor,
	bbox_max: torch.Tensor,
	vertices: torch.Tensor,
	faces: torch.Tensor,
	face_normals: torch.Tensor,
	face_areas: torch.Tensor,
	face_mask: torch.Tensor,
	offset: float,
	n: int,
	generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
	"""
	Sample narrow band around a mesh (approximate), for each of the P bboxes -> (3, P, n)
	"""
	# Sample zero set with normals
	zero_samples, normals = sample_zero_set_with_normals(
		bbox_min, bbox_max, vertices, faces, face_normals, face_areas, face_mask, n, generator
	)

	# Offset along normals: uniform in [-offset, offset) (jax.random.uniform(minval, maxval) arithmetic)
	u = _uniform(zero_samples.shape[1:], generator, zero_samples.dtype, zero_samples.device)
	offsets = torch.clamp_min(u * (offset - (-offset)) + (-offset), -offset)

	samples = zero_samples + offsets * normals

	return samples


def generate_queries_for_neighborhood(
	bbox_min: torch.Tensor,  # (3, P)
	bbox_max: torch.Tensor,  # (3, P)
	vertices: torch.Tensor,  # (3, max_vertices)
	faces: torch.Tensor,  # (3, max_faces)
	face_normals: torch.Tensor,  # (3, max_faces)
	face_areas: torch.Tensor,  # (max_faces,)
	face_mask: torch.Tensor,  # (max_faces,)
	generator: Optional[torch.Generator] = None,
	n_queries: int = 120,
	n_queries_per_type: int = 40,  # n_queries // 3
	narrow_band_offset: float = 0.2,
	bbox_extension: float = 1.0,
) -> torch.Tensor:
	"""
	Generate query points for P neighborhoods at once.

	Samples (per neighborhood):
	- n_queries_per_type from zero set
	- n_queries_per_type from narrow band
	- n_queries_per_type from extended bbox

	Args:
		bbox_min, bbox_max: Neighborhood bounding boxes
		vertices, faces: Mesh geometry
		face_normals, face_areas: Pre-computed face properties
		face_mask: Valid face mask
		generator: torch random generator
		n_queries: Total queries (should be divisible by 3)
		n_queries_per_type: Queries per type (n_queries // 3)
		narrow_band_offset: Half-width of narrow band
		bbox_extension: Multiplicative extension factor

	Returns:
		queries: (3, P, 3 * n_queries_per_type) query points
	"""
	# Regularize bbox
	bbox_diag = bbox_max - bbox_min
	bbox_extents = torch.abs(bbox_diag)
	bbox_center = 0.5 * (bbox_max + bbox_min)
	bbox_extents = torch.clamp_min(bbox_extents, 0.1)  # Min extent
	bbox_min_reg = bbox_center - 0.5 * bbox_extents
	bbox_max_reg = bbox_center + 0.5 * bbox_extents

	# Zero set samples
	zero_samples = sample_zero_set_single_bbox(
		bbox_min_reg, bbox_max_reg, vertices, faces, face_normals, face_areas, face_mask, n_queries_per_type, generator
	)

	# Narrow band samples
	narrow_samples = sample_narrow_band_single_bbox(
		bbox_min_reg,
		bbox_max_reg,
		vertices,
		faces,
		face_normals,
		face_areas,
		face_mask,
		narrow_band_offset,
		n_queries_per_type,
		generator,
	)

	# Extended bbox samples
	# Compute extended bbox (cube based on max extent)
	sidelength = torch.clamp_min(torch.amax(torch.abs(bbox_diag), dim=0), 0.01)  # (P,)
	radius = 0.5 * math.sqrt(3.0) * sidelength
	bbox_min_ext = bbox_center - radius * bbox_extension
	bbox_max_ext = bbox_center + radius * bbox_extension

	bbox_samples = sample_bbox_uniform(bbox_min_ext, bbox_max_ext, n_queries_per_type, generator)

	# Concatenate all samples
	queries = torch.cat([zero_samples, narrow_samples, bbox_samples], dim=-1)

	return queries


def generate_shape_features(
	positions: torch.Tensor,  # (3, n_points)
	neighbors: torch.Tensor,  # (n_points, k) - neighbor indices
	vertices: torch.Tensor,  # (3, max_vertices)
	faces: torch.Tensor,  # (3, max_faces)
	face_mask: torch.Tensor,  # (max_faces,)
	generator: Optional[torch.Generator] = None,
	n_queries: int = 120,
	narrow_band_offset: float = 0.2,
	bbox_extension: float = 1.0,
	use_unsigned_distance: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Generate query points and ground-truth distances for all neighborhoods in a shape.

	Args:
		positions: Point cloud positions
		neighbors: Neighbor indices for each point
		vertices: Mesh vertices (padded to max_vertices)
		faces: Mesh faces (padded to max_faces, invalid = -1)
		face_mask: Boolean mask for valid faces
		generator: torch random generator (replaces the JAX key)
		n_queries: Total queries per neighborhood (must be divisible by 3)
		narrow_band_offset: Half-width of narrow band
		bbox_extension: Extension factor

	Returns:
		queries: (3, n_points, n_queries) query points per neighborhood
		distances: (n_points, n_queries) ground-truth signed distances
	"""
	n_points = positions.shape[1]
	n_queries_per_type = n_queries // 3

	# Pre-compute face properties (once for entire shape)
	face_normals, face_areas, _ = compute_face_properties(vertices, faces, face_mask)

	# Compute bboxes for all neighborhoods
	neighbor_positions = positions[:, neighbors]  # (3, n_points, k)
	bbox_mins = torch.amin(neighbor_positions, dim=-1)  # (3, n_points)
	bbox_maxs = torch.amax(neighbor_positions, dim=-1)

	# Vectorized query generation
	queries = generate_queries_for_neighborhood(
		bbox_mins,
		bbox_maxs,
		vertices,
		faces,
		face_normals,
		face_areas,
		face_mask,
		generator,
		n_queries,
		n_queries_per_type,
		narrow_band_offset,
		bbox_extension,
	)  # (3, n_points, n_queries)

	# Compute signed distances for all queries
	flat_queries = queries.reshape(3, -1)
	if use_unsigned_distance:
		distances = unsigned_distance_to_mesh(flat_queries, vertices, faces, face_mask)
	else:
		distances = signed_distance_to_mesh(flat_queries, vertices, faces, face_mask)

	return queries, distances.reshape(n_points, -1)  # (n_points, n_queries)
