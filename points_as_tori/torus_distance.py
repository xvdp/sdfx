"""
Torch port of src/cpp/torus_distance.h / torus_distance.cpp (the TorusDistanceField evaluator, torus fitting, k-NN).

The C++ loops over queries (OpenMP) and, per query, over its neighbor list. Here both loops are replaced by wide
(structure-of-arrays) tensor expressions over a chunk of Q queries and their K neighbors: per-pair quantities have
shape (Q, K) and per-pair vectors (3, Q, K); reductions are sums over the last axis. When every point is a neighbor
(k_evaluation <= 0 or >= |P|, and radius mode) the point arrays are broadcast as (3, 1, |P|) instead of gathered.

Nearest neighbors: nanoflann's kd-tree is replaced by one of three exact searches (config.KNN_BACKEND):
  * brute force: one GEMM for squared distances + torch.topk per chunk of queries; O(|Q| |P|) work, no host sync,
    chunks sized so that Q * |P| <= config.CHUNK_ELEMENTS;
  * a linear BVH built on the device (points_as_tori.bvh) traversed by all queries at once; O(|Q| · visited nodes),
    a host check every few traversal steps;
  * the same BVH traversed by a fused Slang kernel, one GPU thread per query (points_as_tori.knn_slang, 'slang').
'auto' takes the BVH from config.KNN_BVH_MIN_POINTS points on.

Mathematics (per query q, neighbors i ∈ N(q), point p_i, torus T_i):

	torus SDF   g_i(q) = s_i ( sqrt(d1² + d2²) - |r_i| ),   d1 = |(q - c_i) × a_i| - R_i,   d2 = (q - c_i)·a_i,
	            s_i = sign(r_i)   (the sign of the minor radius encodes the torus orientation)
	weights     w_i = exp(-λ (|q - p_i| - shift)),   shift = ½ max_{j ∈ N(q)} |q - p_j|,   λ = 64 / shift · λ_scale
	blend       φ(q) = Σ w_i g_i / Σ w_i
	gradient    ∇w_i = -λ w_i r̂_i  (shift, λ held fixed),  r̂_i = (q - p_i)/|q - p_i|
	            ∇φ = Σ w_i (∇g_i - λ r̂_i g_i) / W  +  φ λ Σ w_i r̂_i / W  =  A + φ B,   W = Σ w_i
	torus grad  ∇g_i = s_i (w_x G_1 + w_y G_2),  (w_x, w_y) = (d1, d2)/|(d1, d2)|,
	            G_1 = ∂v/∂q = (r - (r·a) a)/v (written out componentwise as in the C++), G_2 = a,  r = q - c, v = |r × a|
	The shift makes the exponent lie in [-64 λ_scale, 64 λ_scale] for k-NN / all-point evaluation, so exp() does not
	overflow in float32.

Layout and device: see config.py. All functions take/return (3, n) / (2, n) / (6, n) tensors on the configured device.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch

from . import config
from .shape_2d import sgn, sign, cross, dot, norm, orthonormal_basis

SPECIAL_VALUE = float('nan')
NAN_EPSILON = 1e-10  # reproduce Python results


def rotation_to_cartesian(angles: torch.Tensor) -> torch.Tensor:
	"""(2, ...) angles (theta, phi) -> (3, ...) unit vectors (cos θ sin φ, sin θ sin φ, cos φ)."""
	theta = angles[0]
	phi = angles[1]
	sin_phi = torch.sin(phi)
	return torch.stack([torch.cos(theta) * sin_phi, torch.sin(theta) * sin_phi, torch.cos(phi)], dim=0)


def cartesian_to_rotation(p: torch.Tensor) -> torch.Tensor:
	"""
	(3, ...) unit vectors -> (2, ...) angles (theta = atan2(y, x), phi = acos(z)).
	Deviation: z is clamped to [-1, 1] before acos (the C++ returns NaN when rounding pushes |z| above 1).
	"""
	theta = torch.atan2(p[1], p[0])
	phi = torch.acos(torch.clamp(p[2], -1.0, 1.0))
	return torch.stack([theta, phi], dim=0)


def compute_lambda(shift: torch.Tensor) -> torch.Tensor:
	return 64.0 / shift


def point_cloud_bounding_box(positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
	"""(3, |P|) -> (bbox_min, bbox_max), each (3,). (C++: shape_3d.cpp)"""
	return positions.amin(dim=1), positions.amax(dim=1)


# ========================================================================
# BRUTE-FORCE NEAREST NEIGHBORS (replaces nanoflann)
# ========================================================================


def _chunk_ranges(n_total: int, chunk: int):
	for start in range(0, n_total, chunk):
		yield start, min(start + chunk, n_total)


def _query_chunk_size(n_points: int) -> int:
	return max(1, config.CHUNK_ELEMENTS // max(1, n_points))


def _squared_distances(queries: torch.Tensor, points: torch.Tensor, points_sq: torch.Tensor) -> torch.Tensor:
	"""
	(3, Q), (3, N), (N,) -> (Q, N) squared distances |q|² + |p|² - 2 q·p (one GEMM). Used only to *select*
	neighbors; distances that enter the SDF are recomputed exactly from the gathered differences.
	"""
	d2 = torch.addmm(points_sq[None, :], queries.T, points, beta=1.0, alpha=-2.0)
	d2 += (queries * queries).sum(dim=0)[:, None]
	return d2.clamp_min_(0.0)


def _knn(
	queries: torch.Tensor,
	points: torch.Tensor,
	k: int,
	points_sq: torch.Tensor,
	exclude: Optional[torch.Tensor] = None,
) -> torch.Tensor:
	"""(3, Q) queries -> (Q, k) indices of the k nearest points, nearest first. `exclude`: (N,) bool, never returned."""
	d2 = _squared_distances(queries, points, points_sq)
	if exclude is not None:
		d2 = d2.masked_fill_(exclude[None, :], float('inf'))
	return torch.topk(d2, k, dim=1, largest=False, sorted=True).indices


def _use_bvh(n_points: int) -> bool:
	"""True if k-NN goes through a BVH (torch traversal or the Slang kernel), False for the brute force."""
	backend = config.KNN_BACKEND
	if backend not in ('auto', 'brute', 'bvh', 'slang'):
		raise ValueError(f"config.KNN_BACKEND must be 'auto', 'brute', 'bvh' or 'slang', got {backend!r}")
	return backend in ('bvh', 'slang') or (backend == 'auto' and n_points >= config.KNN_BVH_MIN_POINTS)


def _knn_searcher(points: torch.Tensor):
	"""Object with find_k_nearest(queries, k) over points: points_as_tori.bvh.BVH, or knn_slang.SlangKNN for 'slang'."""
	if config.KNN_BACKEND == 'slang':
		from .knn_slang import SlangKNN

		return SlangKNN(points)
	from .bvh import BVH

	return BVH(points)


def _bvh_knn(queries: torch.Tensor, points: torch.Tensor, k: int, bvh=None) -> torch.Tensor:
	"""(3, Q) queries -> (Q, k) indices of the k nearest points through a BVH (built if not given), nearest first."""
	bvh = _knn_searcher(points) if bvh is None else bvh
	return bvh.find_k_nearest(queries, k)[0]


# Return (|P|, k) matrix representing the neighbors of the point cloud
# Potentially ignore outliers
def get_neighbors(points: torch.Tensor, k_neighbors_: int, outliers=None) -> torch.Tensor:
	"""
	Args:
		points: (3, |P|)
		k_neighbors_: k (<= 0 or >= #inliers means all inliers)
		outliers: optional indices of outlier points; outliers are never returned as neighbors, and their own rows
			are filled with -1 (as in the C++).

	Returns:
		(|P|, k) int64 tensor of neighbor indices, nearest first (the point itself first, up to exact ties).
	"""
	points = config.check_wide(points, 3, 'points')
	n_points = points.shape[1]
	device = points.device

	n_outliers = 0 if outliers is None else len(outliers)
	exist_outliers = n_outliers > 0
	is_outlier = None
	if exist_outliers:
		outliers_host = np.asarray(outliers.cpu() if torch.is_tensor(outliers) else outliers)  # host indices
		outliers = torch.as_tensor(outliers, dtype=torch.long, device=device)
		is_outlier = torch.zeros(n_points, dtype=torch.bool, device=device)
		is_outlier[outliers] = True
	n_inliers = n_points - n_outliers

	k_neighbors = k_neighbors_ if (k_neighbors_ > 0 and k_neighbors_ < n_inliers) else n_inliers
	neighbors = torch.empty((n_points, k_neighbors), dtype=torch.long, device=device)

	if _use_bvh(n_inliers):
		# BVH over the inliers (as the C++ kd-tree over inlier_points), indices mapped back through idx_map
		if exist_outliers:
			idx_map = torch.as_tensor(np.setdiff1d(np.arange(n_points), outliers_host), dtype=torch.long, device=device)
			neighbors = idx_map[_bvh_knn(points, points[:, idx_map], k_neighbors)]
		else:
			neighbors = _bvh_knn(points, points, k_neighbors)
	else:
		points_sq = (points * points).sum(dim=0)
		for start, end in _chunk_ranges(n_points, _query_chunk_size(n_points)):
			neighbors[start:end] = _knn(points[:, start:end], points, k_neighbors, points_sq, exclude=is_outlier)

	if exist_outliers:
		neighbors = torch.where(is_outlier[:, None], -1, neighbors)

	return neighbors


def get_average_neighbor_distance(points: torch.Tensor, k_neighbors_: int) -> torch.Tensor:
	"""
	Average distance of each point to its k nearest neighbors (including itself), averaged over the point cloud.
	Returns a 0-d tensor (no host sync). Note: the C++ accumulates `total_distance` inside an OpenMP loop without a
	reduction clause (a data race); this computes the intended sum.
	"""
	points = config.check_wide(points, 3, 'points')
	n_points = points.shape[1]
	k_neighbors = k_neighbors_ if (k_neighbors_ > 0 and k_neighbors_ < n_points) else n_points

	total_distance = torch.zeros((), dtype=points.dtype, device=points.device)
	if _use_bvh(n_points):
		idx = _bvh_knn(points, points, k_neighbors)  # (|P|, k)
		total_distance = norm(points[:, :, None] - points[:, idx]).sum()
	else:
		points_sq = (points * points).sum(dim=0)
		for start, end in _chunk_ranges(n_points, _query_chunk_size(n_points)):
			q = points[:, start:end]
			idx = _knn(q, points, k_neighbors, points_sq)  # (Q, k)
			total_distance = total_distance + norm(q[:, :, None] - points[:, idx]).sum()

	return total_distance / (n_points * k_neighbors)


# ========================================================================
# TORUS SDF
# ========================================================================


def torus_signed_distance_gradient(
	q: torch.Tensor,
	center: torch.Tensor,
	axis3d: torch.Tensor,
	major_radius: torch.Tensor,
	minor_radius: torch.Tensor,
	want_gradient: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
	"""
	Signed distance (and gradient) of the SDF of tori, broadcasting over trailing axes.

	Args:
		q, center, axis3d: (3, ...) (axis3d: unit axis, i.e. rotation_to_cartesian(axis))
		major_radius, minor_radius: (...)

	Returns:
		phi: (...),  grad: (3, ...) or None
	"""
	r_vec = q - center
	v = norm(cross(r_vec, axis3d))
	d1 = v - major_radius
	d2 = dot(r_vec, axis3d)
	d_norm = torch.sqrt(d1 * d1 + d2 * d2)
	udf = d_norm - torch.abs(minor_radius)
	s = sign(minor_radius)
	phi = s * udf
	if not want_gradient:
		return phi, None

	ax, ay, az = axis3d[0], axis3d[1], axis3d[2]
	cx_qx, cy_qy, cz_qz = -r_vec[0], -r_vec[1], -r_vec[2]  # center - q

	# Entries of a 2x3 matrix
	G_11 = (ax * ay * cy_qy + ax * az * cz_qz - (ay * ay + az * az) * cx_qx) / v
	G_12 = (ax * ay * cx_qx + ay * az * cz_qz - (ax * ax + az * az) * cy_qy) / v
	G_13 = (ax * az * cx_qx + ay * az * cy_qy - (ax * ax + ay * ay) * cz_qz) / v
	# G_21, G_22, G_23 = ax, ay, az
	w_x = d1 / d_norm
	w_y = d2 / d_norm
	grad = s * torch.stack([w_x * G_11 + w_y * ax, w_x * G_12 + w_y * ay, w_x * G_13 + w_y * az], dim=0)  # (w^T * G)^T
	return phi, grad


# ========================================================================
# TORUS FITTING
# ========================================================================


def _principal_curvatures_from_forms(
	normals: torch.Tensor,
	coeffs: torch.Tensor,
	s_hat: torch.Tensor,
	t_hat: torch.Tensor,
	python_variant: bool,
):
	"""
	Shared body of get_torus_parameters / get_torus_parameters_python / get_principal_directions_at_point,
	vectorized over all points.

	The surface near p_i is the height field Q(s, t) = a00 + a10 s + a01 t + a20 s² + a11 s t + a02 t² over the tangent
	frame (s_hat, t_hat, n_i). At (0, 0): first fundamental form I = [[1 + a10², a10 a01], [a10 a01, 1 + a01²]],
	second fundamental form II = [[2 a20, a11], [a11, 2 a02]] / A, A = sqrt(1 + a01² + a10²). Principal directions
	are the generalized eigenvectors of (II, I); the closed-form eigenvectors (v1 ± sqrt(disc), v2) come from
	Mathematica; principal curvatures are the Rayleigh quotients κ = (eᵀ II e)/(eᵀ I e).

	python_variant=True reproduces get_torus_parameters_python (epsilon guards used in training).

	Args: normals (3, N), coeffs (6, N) = (a00, a01, a10, a11, a02, a20), s_hat, t_hat (3, N)
	Returns: a00, kappa_min, kappa_max (N,), v_min, v_max, normal (3, N)
	"""
	n_i = normals

	a00 = coeffs[0]
	a01 = coeffs[1]
	a10 = coeffs[2]
	a11 = coeffs[3]
	a02 = coeffs[4]
	a20 = coeffs[5]

	# Expressions from bashing Mathematica
	A = torch.sqrt(1.0 + a01 * a01 + a10 * a10)
	v1 = a20 * a01 * a01 - a02 * a10 * a10 - a02 + a20
	v2 = a11 * (a10 * a10 + 1.0) - 2.0 * a01 * a10 * a20
	term = a02 * (a10 * a10 + 1.0) - a01 * a10 * a11 + a20 * (a01 * a01 + 1.0)
	discriminant = (1.0 + a01 * a01 + a10 * a10) * (a11 * a11 - 4.0 * a02 * a20) + term * term

	if python_variant:
		discriminant = torch.sqrt(
			torch.clamp_min(discriminant, NAN_EPSILON)
		)  # reproduce Python function used in training

		ev_pos_0 = v1 + discriminant
		ev_pos_0 = sign(ev_pos_0) * torch.clamp_min(torch.abs(ev_pos_0), NAN_EPSILON)
		ev_neg_1 = v2
		ev_neg_1 = sign(ev_neg_1) * torch.clamp_min(torch.abs(ev_neg_1), NAN_EPSILON)

		ev_pos_x, ev_pos_y = ev_pos_0, v2
		ev_neg_x, ev_neg_y = v1 - discriminant, ev_neg_1

		ev_pos_norm = torch.sqrt(ev_pos_x * ev_pos_x + ev_pos_y * ev_pos_y)
		ev_neg_norm = torch.sqrt(ev_neg_x * ev_neg_x + ev_neg_y * ev_neg_y)
		ev_pos_x, ev_pos_y = ev_pos_x / ev_pos_norm, ev_pos_y / ev_pos_norm
		ev_neg_x, ev_neg_y = ev_neg_x / ev_neg_norm, ev_neg_y / ev_neg_norm
	else:
		discriminant = torch.sqrt(torch.clamp_min(discriminant, 0.0))
		ev_pos_x, ev_pos_y = v1 + discriminant, v2
		ev_neg_x, ev_neg_y = v1 - discriminant, v2

	dQstards = s_hat + a10 * n_i
	dQstardt = t_hat + a01 * n_i
	v_pos = ev_pos_x * dQstards + ev_pos_y * dQstardt
	v_neg = ev_neg_x * dQstards + ev_neg_y * dQstardt
	normal = (n_i - a10 * s_hat - a01 * t_hat) / A  # n^*(0,0)

	# Compute principal curvatures using Rayleigh quotient
	I_11 = 1.0 + a10 * a10
	I_12 = a10 * a01
	I_22 = 1.0 + a01 * a01
	II_11 = 2.0 * a20 / A
	II_12 = a11 / A
	II_22 = 2.0 * a02 / A

	def quadratic_form(m11, m12, m22, x, y):
		return x * (m11 * x + m12 * y) + y * (m12 * x + m22 * y)

	denom_pos = quadratic_form(I_11, I_12, I_22, ev_pos_x, ev_pos_y)
	denom_neg = quadratic_form(I_11, I_12, I_22, ev_neg_x, ev_neg_y)
	if python_variant:
		denom_pos = sign(denom_pos) * torch.clamp_min(torch.abs(denom_pos), NAN_EPSILON)
		denom_neg = sign(denom_neg) * torch.clamp_min(torch.abs(denom_neg), NAN_EPSILON)
	kappa_pos = quadratic_form(II_11, II_12, II_22, ev_pos_x, ev_pos_y) / denom_pos
	kappa_neg = quadratic_form(II_11, II_12, II_22, ev_neg_x, ev_neg_y) / denom_neg

	use_pos_as_min = torch.abs(kappa_pos) < torch.abs(kappa_neg)
	kappa_min = torch.where(use_pos_as_min, kappa_pos, kappa_neg)
	kappa_max = torch.where(use_pos_as_min, kappa_neg, kappa_pos)
	v_min = torch.where(use_pos_as_min, v_pos, v_neg)
	v_max = torch.where(use_pos_as_min, v_neg, v_pos)

	return a00, kappa_min, kappa_max, v_min, v_max, normal


# Return a00, kappa_min, kappa_max, v_min, normal at p_i
def get_torus_parameters(normals, coeffs, s_hat, t_hat):
	a00, kappa_min, kappa_max, v_min, _, normal = _principal_curvatures_from_forms(normals, coeffs, s_hat, t_hat, False)
	return a00, kappa_min, kappa_max, v_min, normal


# Return a00, kappa_min, kappa_max, v_min, normal at p_i
def get_torus_parameters_python(normals, coeffs, s_hat, t_hat):
	a00, kappa_min, kappa_max, v_min, _, normal = _principal_curvatures_from_forms(normals, coeffs, s_hat, t_hat, True)
	return a00, kappa_min, kappa_max, v_min, normal


def get_principal_directions_at_point(normals, coeffs, s_hat, t_hat):
	_, kappa_min, kappa_max, v_min, v_max, _ = _principal_curvatures_from_forms(normals, coeffs, s_hat, t_hat, False)
	return kappa_min, kappa_max, v_min, v_max


# Fit a torus at every point, directly from fundamental form coefficients
def fit_torus_at_point(
	points: torch.Tensor, normals: torch.Tensor, coeffs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Vectorized C++ fit_torus_at_point over all points.

	With κ_min, κ_max the principal curvatures (|κ_min| <= |κ_max|) and v_min the direction of κ_min:
		minor radius r = 1/|κ_max|,  major radius R = 1/|κ_min| - sgn(κ_max κ_min) r,
		orientation s = +1 if κ_max < 0 and κ_min != 0 ... (exactly: (κ_max<0 & κ_min<0) | (κ_max<0 & κ_min>0)), else -1
		center = p + a00 n - (s/|κ_min|) n*,   axis u = n* × v_min / |n* × v_min|
	Returns (center (3, N), axis (2, N), R (N,), s r (N,)).
	"""
	p_i = points
	n_i = normals

	s_hat, t_hat = orthonormal_basis(n_i)

	a00, kappa_min, kappa_max, v_min, normal = get_torus_parameters_python(n_i, coeffs, s_hat, t_hat)

	r = 1.0 / torch.abs(kappa_max)
	R = 1.0 / torch.abs(kappa_min) - sgn(kappa_max * kappa_min) * r
	s = torch.where(((kappa_max < 0.0) & (kappa_min < 0.0)) | ((kappa_max < 0) & (kappa_min > 0.0)), 1.0, -1.0).to(
		r.dtype
	)

	center = p_i + a00 * n_i - s / torch.abs(kappa_min) * normal
	u = cross(normal, v_min)
	u = u / torch.clamp_min(norm(u), NAN_EPSILON)  # reproduce Python results
	axis = cartesian_to_rotation(u)

	return center, axis, R, s * r


def fit_tori_from_forms(
	points: torch.Tensor, normals: torch.Tensor, coefficients: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Args: points, normals (3, |P|), coefficients (6, |P|) on the configured device.
	Returns: centers (3, |P|), axes (2, |P|), major_radii (|P|,), minor_radii (|P|,)
	"""
	points = config.check_wide(points, 3, 'points')
	normals = config.check_wide(normals, 3, 'normals')
	coefficients = config.check_wide(coefficients, 6, 'coefficients').to(points.dtype)
	return fit_torus_at_point(points, normals, coefficients)


# ========================================================================
# TORUS DISTANCE FIELD
# ========================================================================


class TorusDistanceField:
	def __init__(self, points: torch.Tensor, normals: torch.Tensor):
		"""
		Args:
			points, normals: (3, |P|) tensors on the configured device (use config.as_wide to import (|P|, 3) arrays)
		"""
		# input point cloud
		self.points = config.check_wide(points, 3, 'points')
		self.normals = config.check_wide(normals, 3, 'normals').to(self.points.dtype)
		self.n_points = self.points.shape[1]
		self.device = self.points.device
		self.dtype = self.points.dtype

		bbox_min, bbox_max = point_cloud_bounding_box(self.points)

		self.is_outlier = torch.zeros(self.n_points, dtype=torch.bool, device=self.device)

		# Set the bounding box of the SDF as approximately double the bounding box of the input point cloud (a
		# heuristic).
		diag = bbox_max - bbox_min
		self.bbox_min = bbox_min - 0.5 * diag
		self.bbox_max = bbox_max + 0.5 * diag
		self.diameter = torch.linalg.vector_norm(self.bbox_min - self.bbox_max)

		# Replaces the kd tree: squared norms reused by every brute-force neighbor search.
		self._points_sq = (self.points * self.points).sum(dim=0)

		#  SDF parameters
		self.dimension = 3
		self.k_evaluation = -1  # number of nearest neighbors to use when evaluating distance
		self.radius_evaluation = -1.0  # radius for radius-based evaluation (-1 = disabled, use k_evaluation)
		self.lambda_scale = 1.0
		self.exact_laplacian = False  # True: exact Δg of each torus in the Laplacian (the C++ uses (d-1)/|q - c|)
		self.use_areas = False

		# torus parameters
		self.centers = None
		self.axes = None  # two angles; axis always passes through the center
		self.major_radii = None
		self.minor_radii = None  # signs are encoded via minor_radii
		self._axes3d = None  # rotation_to_cartesian(axes), cached

	# Get the k closest neighbors to a given point in the input point cloud, including the point
	def get_neighbors(self, k_neighbors_: int, i: Optional[int] = None) -> torch.Tensor:
		"""(k,) neighbors of point i; with i=None, (|P|, k) neighbors of all points (get_all_neighbors)."""
		if i is None:
			return self.get_all_neighbors(k_neighbors_)
		k_neighbors = k_neighbors_ if (k_neighbors_ > 0 and k_neighbors_ < self.n_points) else self.n_points
		return _knn(self.points[:, i : i + 1], self.points, k_neighbors, self._points_sq)[0]

	def get_all_neighbors(self, k_neighbors_: int) -> torch.Tensor:
		"""(|P|, k)"""
		k_neighbors = k_neighbors_ if (k_neighbors_ > 0 and k_neighbors_ < self.n_points) else self.n_points
		if self._knn_bvh() and k_neighbors < self.n_points:
			return self._get_bvh().find_k_nearest(self.points, k_neighbors)[0]
		neighbors = torch.empty((self.n_points, k_neighbors), dtype=torch.long, device=self.device)
		for start, end in _chunk_ranges(self.n_points, _query_chunk_size(self.n_points)):
			neighbors[start:end] = _knn(self.points[:, start:end], self.points, k_neighbors, self._points_sq)
		return neighbors

	def get_all_neighbors_and_subsample(self, k_neighbors_: int, n_subsample_: int):
		"""
		(|P|, k) neighbors and (|P|, n_subsample) of them chosen by farthest-point sampling within each neighborhood
		(subsample_point_cloud, starting from the point itself). Note: the C++ shares one `neighborhood` buffer
		between OpenMP threads (a data race); this computes each neighborhood separately.
		"""
		k_neighbors = k_neighbors_ if (k_neighbors_ > 0 and k_neighbors_ < self.n_points) else self.n_points
		n_subsample = n_subsample_ if (n_subsample_ > 0 and n_subsample_ < self.n_points) else self.n_points
		neighbors = self.get_all_neighbors(k_neighbors)
		if n_subsample >= k_neighbors:
			return neighbors, neighbors[:, :n_subsample]
		# batched farthest-point sampling over all neighborhoods at once (same steps as subsample_point_cloud)
		nb = self.points[:, neighbors]  # (3, |P|, k)
		rows = torch.arange(self.n_points, device=self.device)
		selected = torch.zeros((self.n_points, n_subsample), dtype=torch.long, device=self.device)
		min_d = torch.full((self.n_points, k_neighbors), float('inf'), dtype=self.dtype, device=self.device)
		last = torch.zeros(self.n_points, dtype=torch.long, device=self.device)
		for i in range(1, n_subsample):
			diff = nb - nb[:, rows, last][:, :, None]
			min_d = torch.minimum(min_d, (diff * diff).sum(dim=0))
			last = torch.argmax(min_d, dim=1)
			selected[:, i] = last
			min_d[rows, last] = 0.0
		return neighbors, torch.gather(neighbors, 1, selected)

	# ===== SINGLE-TORUS QUERIES

	def torus_signed_distance(self, q, center, axis, major_radius, minor_radius) -> torch.Tensor:
		"""Signed distance of the SDF of a single given torus. q, center: (3,) or (3, ...); axis: (2,) angles."""
		axis3d = rotation_to_cartesian(self._as_tensor(axis))
		center = self._as_tensor(center)
		if center.ndim == 1 and q.ndim == 2:
			center, axis3d = center[:, None], axis3d[:, None]
		return torus_signed_distance_gradient(
			q, center, axis3d, self._as_tensor(major_radius), self._as_tensor(minor_radius), want_gradient=False
		)[0]

	def _as_tensor(self, x) -> torch.Tensor:
		"""Small per-torus parameters (host scalars / short arrays) -> tensors on the device (an upload, no sync)."""
		return torch.as_tensor(x, dtype=self.dtype, device=self.device)

	def torus_signed_distances(self, queries, center, axis, major_radius, minor_radius) -> torch.Tensor:
		"""queries (3, Q); center (3,); axis (2,) -> (Q,)"""
		queries = config.check_wide(queries, 3, 'queries')
		return self.torus_signed_distance(queries, center, axis, major_radius, minor_radius)

	def torus_signed_distances_gradients(self, queries, center, axis, major_radius, minor_radius):
		"""queries (3, Q); center (3,); axis (2,) -> (Q,), (3, Q)"""
		queries = config.check_wide(queries, 3, 'queries')
		center = self._as_tensor(center).reshape(3, 1)
		axis3d = rotation_to_cartesian(self._as_tensor(axis)).reshape(3, 1)
		return torus_signed_distance_gradient(
			queries, center, axis3d, self._as_tensor(major_radius), self._as_tensor(minor_radius)
		)

	# ===== BATCH QUERIES

	def _pair_areas(self, idx: Optional[torch.Tensor]):
		"""Areas of the neighbor pairs when use_areas: gathered (Q, K) for indices, broadcast (1, |P|) for None."""
		if not self.use_areas:
			return None
		areas = self.get_point_areas()
		return areas[None, :] if idx is None else areas[idx]

	def _knn_bvh(self) -> bool:
		return _use_bvh(self.n_points)

	def _get_bvh(self):
		"""
		BVH over all points (outliers included: they enter the neighbor search, not the sums), built once; a
		knn_slang.SlangKNN when config.KNN_BACKEND == 'slang' (rebuilt if the backend changes).
		"""
		backend = 'slang' if config.KNN_BACKEND == 'slang' else 'bvh'
		if getattr(self, '_bvh', None) is None or getattr(self, '_bvh_backend', None) != backend:
			self._bvh = _knn_searcher(self.points)
			self._bvh_backend = backend
		return self._bvh

	def _require_tori(self):
		if self.centers is None:
			raise RuntimeError('tori are not set: call set_tori() or fit_tori_from_forms() first')

	def _blend(
		self,
		q: torch.Tensor,
		rVec: torch.Tensor,
		d: torch.Tensor,
		centers: torch.Tensor,
		axes3d: torch.Tensor,
		major_radii: torch.Tensor,
		minor_radii: torch.Tensor,
		include: Optional[torch.Tensor],
		shift: torch.Tensor,
		lam: torch.Tensor,
		isovalue: float,
		want_gradient: bool,
		want_laplacian: bool,
		area: Optional[torch.Tensor] = None,
	):
		"""
		Body of evaluate_distance_single / evaluate_distance_gradient_laplacian_single for a chunk of Q queries.

		Args:
			q: (3, Q, 1);  rVec = q - p_i: (3, Q, K);  d = |rVec|: (Q, K)
			centers, axes3d: (3, Q, K) gathered or (3, 1, |P|) broadcast;  major_radii, minor_radii: (Q, K) or (1, |P|)
			include: bool (Q, K) / (1, |P|) — pairs that enter the sums (neighbors that are not outliers); None = all
			shift, lam: (Q,)
			area: point areas per pair (Q, K) / (1, |P|) when use_areas (weights w_i a_i), else None
		Returns:
			dist (Q,), grad (3, Q) or None, laplacian (Q,) or None
		"""

		def summed(x):
			# Pairs outside `include` are skipped (C++ `continue`): mask before summing so NaN/inf there cannot leak.
			if include is not None:
				x = torch.where(include, x, 0.0)
			return x.sum(dim=-1)

		lam_e = lam[:, None]
		weight = torch.exp(-lam_e * (d - shift[:, None]))  # (Q, K)
		if area is not None:
			weight = weight * area

		g_z, grad_g = torus_signed_distance_gradient(
			q, centers, axes3d, major_radii, minor_radii, want_gradient=want_gradient or want_laplacian
		)

		g = summed(weight * g_z)
		normalization = summed(weight)
		dist = g / normalization

		if not (want_gradient or want_laplacian):
			return dist - isovalue, None, None

		rHat = rVec / d
		grad_g_minus_lambda_n_g = summed(weight * (grad_g - lam_e * rHat * g_z))  # (3, Q)
		n = summed(weight * rHat)  # (3, Q)

		A = grad_g_minus_lambda_n_g / normalization
		B = lam * n / normalization
		grad = A + dist * B

		laplacian = None
		if want_laplacian:
			dm1 = float(self.dimension - 1)
			s = sign(minor_radii)
			if self.exact_laplacian:
				# Δg of the torus SDF: with ρ = |(q - c) × a|, z = (q - c)·a, u = √((ρ - R)² + z²), in cylindrical
				# coordinates Δu = u_ρρ + u_ρ/ρ + u_zz = 1/u + (ρ - R)/(u ρ)  (u_ρρ + u_zz = 1/u, u_ρ = (ρ - R)/u)
				r_c = q - centers
				rho = norm(cross(r_c, axes3d))
				z = dot(r_c, axes3d)
				u = torch.sqrt((rho - major_radii) ** 2 + z * z)
				term1 = s * (1.0 / u + (rho - major_radii) / (u * rho))
			else:
				d_to_center = norm(q - centers)
				term1 = s * dm1 / d_to_center  # TODO (as in the C++): Δg(z); exact for R = 0 (sphere): 2/|q - c|
			term2 = -2.0 * lam_e * dot(rHat, grad_g)
			term3 = lam_e * g_z * (lam_e - dm1 / d)
			numer1 = summed(weight * (term1 + term2 + term3))  # Δg(z) - 2λ< rHat, ∇g(x) > + g(z)( λ^2 - λ(d-1)/|x-z| )
			numer2 = summed(weight * (dm1 / d - lam_e))  # (d-1)/|x-z| - λ
			first_term = numer1 / normalization + dot(A, B)
			second_term = dot(grad, B) + dist * (lam * numer2 / normalization + dot(B, B))
			laplacian = first_term + second_term

		return dist - isovalue, grad, laplacian

	def _evaluate_chunk(self, q: torch.Tensor, isovalue: float, want_gradient: bool, want_laplacian: bool):
		"""q: (3, Q). Neighbor selection + blend; mirrors the three branches of the C++ batch evaluators."""
		n_points = self.n_points
		k_evaluation = self.k_evaluation
		q_e = q[:, :, None]  # (3, Q, 1)
		not_outlier = ~self.is_outlier

		def gathered(idx):
			# (Q, K) neighbor indices -> per-pair arrays
			rVec = q_e - self.points[:, idx]
			return (
				rVec,
				norm(rVec),
				self.centers[:, idx],
				self._axes3d[:, idx],
				self.major_radii[idx],
				self.minor_radii[idx],
				not_outlier[idx],
			)

		if self.radius_evaluation > 0:
			# Radius-based evaluation: dense over all points, masked to |q - p| < radius (nanoflann radiusSearch).
			radius = float(self.radius_evaluation)
			lam_radius_factor = radius / math.sqrt(3.0)
			rVec = q_e - self.points[:, None, :]  # (3, Q, |P|)
			d_sq = (rVec * rVec).sum(dim=0)
			d = torch.sqrt(d_sq)
			in_radius = d_sq < radius * radius  # (Q, |P|)
			n_found = in_radius.sum(dim=-1)  # (Q,)

			shift = 0.5 * torch.where(in_radius, d, 0.0).amax(dim=-1)
			lam = compute_lambda(shift) * self.lambda_scale
			# Scale lambda for radius-based truncation (see the C++ comment): only when not all points are neighbors.
			lam = torch.where(n_found < n_points, lam * lam_radius_factor, lam)
			result_radius = self._blend(
				q_e,
				rVec,
				d,
				self.centers[:, None, :],
				self._axes3d[:, None, :],
				self.major_radii[None, :],
				self.minor_radii[None, :],
				in_radius & not_outlier[None, :],
				shift,
				lam,
				isovalue,
				want_gradient,
				want_laplacian,
				self._pair_areas(None),
			)

			# Fallback to k-NN if no points found (computed for every query, selected per query: no host branch).
			k = k_evaluation if (k_evaluation > 0 and k_evaluation < n_points) else min(10, n_points)
			idx = torch.topk(d, k, dim=1, largest=False, sorted=True).indices
			rVec_k, d_k, c_k, a_k, R_k, r_k, inc_k = gathered(idx)
			shift_k = 0.5 * d_k.amax(dim=-1)
			lam_k = compute_lambda(shift_k) * self.lambda_scale
			if k < n_points:
				lam_k = lam_k * lam_radius_factor
			result_knn = self._blend(
				q_e,
				rVec_k,
				d_k,
				c_k,
				a_k,
				R_k,
				r_k,
				inc_k,
				shift_k,
				lam_k,
				isovalue,
				want_gradient,
				want_laplacian,
				self._pair_areas(idx),
			)

			found = n_found > 0
			return tuple(None if a is None else torch.where(found, a, b) for a, b in zip(result_radius, result_knn))

		elif k_evaluation > 0 and k_evaluation < n_points:
			# k-NN based evaluation
			if self._knn_bvh():
				idx = self._get_bvh().find_k_nearest(q, k_evaluation)[0]  # (Q, k)
			else:
				idx = _knn(q, self.points, k_evaluation, self._points_sq)  # (Q, k)
			rVec, d, c, a, R, r, include = gathered(idx)
			shift = 0.5 * d.amax(dim=-1)
			lam = compute_lambda(shift) * self.lambda_scale
			return self._blend(
				q_e,
				rVec,
				d,
				c,
				a,
				R,
				r,
				include,
				shift,
				lam,
				isovalue,
				want_gradient,
				want_laplacian,
				self._pair_areas(idx),
			)

		else:
			# Use all points (k_evaluation <= 0 or >= |P|). Deviation: the C++ returns NaN for k_evaluation == 0.
			rVec = q_e - self.points[:, None, :]  # (3, Q, |P|)
			d = norm(rVec)
			shift = 0.5 * d.amax(dim=-1)
			lam = compute_lambda(shift) * self.lambda_scale
			return self._blend(
				q_e,
				rVec,
				d,
				self.centers[:, None, :],
				self._axes3d[:, None, :],
				self.major_radii[None, :],
				self.minor_radii[None, :],
				not_outlier[None, :],
				shift,
				lam,
				isovalue,
				want_gradient,
				want_laplacian,
				self._pair_areas(None),
			)

	def _evaluate(self, queries: torch.Tensor, isovalue: float, want_gradient: bool, want_laplacian: bool):
		self._require_tori()
		queries = config.check_wide(queries, 3, 'queries').to(self.dtype)
		n_queries = queries.shape[1]

		distances = torch.empty(n_queries, dtype=self.dtype, device=self.device)
		gradients = torch.empty((3, n_queries), dtype=self.dtype, device=self.device) if want_gradient else None
		laplacians = torch.empty(n_queries, dtype=self.dtype, device=self.device) if want_laplacian else None

		k_mode = self.radius_evaluation <= 0 and 0 < self.k_evaluation < self.n_points
		if k_mode and self._knn_bvh():
			chunk = max(1, config.CHUNK_ELEMENTS // self.k_evaluation)  # (Q, k) pairs per chunk, not (Q, |P|)
		else:
			chunk = _query_chunk_size(self.n_points)
		for start, end in _chunk_ranges(n_queries, chunk):
			dist, grad, lap = self._evaluate_chunk(queries[:, start:end], isovalue, want_gradient, want_laplacian)
			distances[start:end] = dist
			if want_gradient:
				gradients[:, start:end] = grad
			if want_laplacian:
				laplacians[start:end] = lap

		return distances, gradients, laplacians

	def evaluate_distance(self, queries: torch.Tensor, isovalue: float = 0.0, parallelize: bool = True) -> torch.Tensor:
		"""queries (3, Q) or (3,) -> (Q,) (0-d for a single query). `parallelize` is accepted for API parity."""
		single = queries.ndim == 1
		distances, _, _ = self._evaluate(queries, isovalue, False, False)
		return distances[0] if single else distances

	def evaluate_gradient(self, queries: torch.Tensor, parallelize: bool = True) -> torch.Tensor:
		"""queries (3, Q) or (3,) -> (3, Q) (or (3,))"""
		single = queries.ndim == 1
		_, gradients, _ = self._evaluate(queries, 0.0, True, False)
		return gradients[:, 0] if single else gradients

	def evaluate_laplacian(self, queries: torch.Tensor, parallelize: bool = True) -> torch.Tensor:
		"""queries (3, Q) or (3,) -> (Q,)"""
		single = queries.ndim == 1
		_, _, laplacians = self._evaluate(queries, 0.0, True, True)
		return laplacians[0] if single else laplacians

	def evaluate_distance_gradient_laplacian(
		self, queries: torch.Tensor, isovalue: float = 0.0, parallelize: bool = True
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""queries (3, Q) -> distances (Q,), gradients (3, Q), laplacians (Q,)"""
		single = queries.ndim == 1
		distances, gradients, laplacians = self._evaluate(queries, isovalue, True, True)
		if single:
			return distances[0], gradients[:, 0], laplacians[0]
		return distances, gradients, laplacians

	def evaluate_distance_gradient(
		self, queries: torch.Tensor, isovalue: float = 0.0
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Added in the port: distances and gradients without the Laplacian terms (same values as the generic call)."""
		single = queries.ndim == 1
		distances, gradients, _ = self._evaluate(queries, isovalue, True, False)
		if single:
			return distances[0], gradients[:, 0]
		return distances, gradients

	# naive convolutional formula
	def log_sum_exp(self, queries: torch.Tensor, parallelize: bool = True) -> torch.Tensor:
		"""Compare against Madan & Levin: -log Σ_i exp(-λ (|q - p_i| - shift)) / λ + shift, λ = λ_scale, shift = 64/λ."""
		queries = config.check_wide(queries, 3, 'queries').to(self.dtype)
		lam = float(self.lambda_scale)
		shift = 64.0 / lam
		out = torch.empty(queries.shape[1], dtype=self.dtype, device=self.device)
		for start, end in _chunk_ranges(queries.shape[1], _query_chunk_size(self.n_points)):
			r = norm(queries[:, start:end, None] - self.points[:, None, :])
			w = torch.exp(-lam * (r - shift)).sum(dim=-1)
			out[start:end] = (-torch.log(torch.abs(w)) / lam) + shift
		return out

	# ===== NAIVE CONVOLUTIONAL FORMULAS AND WINDING NUMBERS (need point areas a_i)
	#
	# For a query q and point i: r_vec = q - p_i, r = |r_vec|, r̂ = r_vec / r, and over the k nearest points
	# (all points if k_evaluation <= 0 or >= |P|): shift = ½ max r, λ = 64 λ_scale / shift, e_i = exp(-λ (r_i - shift)).

	def _neighbor_pairs(self, q: torch.Tensor):
		"""(3, Q) -> r_vec (3, Q, K), r (Q, K), neighbor normals (3, Q, K), areas (Q, K), shift (Q,), λ (Q,)."""
		k = self.k_evaluation
		areas = self.get_point_areas()
		if 0 < k < self.n_points:
			if self._knn_bvh():
				idx = self._get_bvh().find_k_nearest(q, k)[0]
			else:
				idx = _knn(q, self.points, k, self._points_sq)
			r_vec = q[:, :, None] - self.points[:, idx]
			n_i, a_i = self.normals[:, idx], areas[idx]
		else:
			r_vec = q[:, :, None] - self.points[:, None, :]
			n_i, a_i = self.normals[:, None, :], areas[None, :]
		r = norm(r_vec)
		shift = 0.5 * r.amax(dim=-1)
		lam = compute_lambda(shift) * self.lambda_scale
		return r_vec, r, n_i, a_i, shift, lam

	def _pairwise(self, queries: torch.Tensor, fn, out_rows: int = 0) -> torch.Tensor:
		"""Apply fn(chunk (3, Q)) -> (Q,) or (3, Q) over query chunks."""
		queries = config.check_wide(queries, 3, 'queries').to(self.dtype)
		k = self.k_evaluation
		per_query = k if 0 < k < self.n_points else self.n_points
		chunk = max(1, config.CHUNK_ELEMENTS // per_query)
		shape = (queries.shape[1],) if out_rows == 0 else (out_rows, queries.shape[1])
		out = torch.empty(shape, dtype=self.dtype, device=self.device)
		for start, end in _chunk_ranges(queries.shape[1], chunk):
			out[..., start:end] = fn(queries[:, start:end])
		return out

	def signed_log_sum_exp(self, queries: torch.Tensor, parallelize: bool = True) -> torch.Tensor:
		"""
		w = Σ a_i h_i e_i,  h_i = (λ r + 1) (r_vec · n_i) / (2π r³);  returns sgn(w) (-log|w| / λ + shift).
		"""

		def fn(q):
			r_vec, r, n_i, a_i, shift, lam = self._neighbor_pairs(q)
			lam_e = lam[:, None]
			h = (lam_e * r + 1.0) * dot(r_vec, n_i) / (2.0 * math.pi * r * r * r)
			w = (a_i * h * torch.exp(-lam_e * (r - shift[:, None]))).sum(dim=-1)
			return sgn(w) * (-torch.log(torch.abs(w)) / lam + shift)

		return self._pairwise(queries, fn)

	def signed_log_sum_exp_gradient(self, queries: torch.Tensor, parallelize: bool = True) -> torch.Tensor:
		"""
		(3, Q): -(A - B) / (C λ),  A = Σ a ∇h e,  B = Σ a λ r̂ h e,  C = Σ a e h, with (as the C++)
		∇h = (λ r + 1) n / (2π r³) + λ r̂ (r_vec·n) / (2π r³) - (λ r + 1)(r_vec·n) 6π r r_vec / (4π² r⁶).
		"""

		def fn(q):
			r_vec, r, n_i, a_i, shift, lam = self._neighbor_pairs(q)
			lam_e = lam[:, None]
			r_hat = r_vec / r
			weight = torch.exp(-lam_e * (r - shift[:, None]))
			rn = dot(r_vec, n_i)
			h = (lam_e * r + 1.0) * rn / (2.0 * math.pi * r * r * r)
			grad_N = (lam_e * r + 1.0) * n_i + (lam_e * r_hat * rn)
			term1 = grad_N / (2.0 * math.pi * r * r * r)
			term2 = ((lam_e * r + 1.0) * rn * 6.0 * math.pi * r * r_vec) / (4.0 * math.pi * math.pi * r**6)
			grad_h = term1 - term2
			A = (a_i * grad_h * weight).sum(dim=-1)
			B = (a_i * lam_e * r_hat * h * weight).sum(dim=-1)
			C = (a_i * weight * h).sum(dim=-1)
			return -(A - B) / (C * lam)

		return self._pairwise(queries, fn, 3)

	def self_normalized_signed_log_sum_exp(self, queries: torch.Tensor, parallelize: bool = True) -> torch.Tensor:
		"""Σ a_i (r_vec · n_i) e_i / Σ a_i e_i."""

		def fn(q):
			r_vec, r, n_i, a_i, shift, lam = self._neighbor_pairs(q)
			weight = torch.exp(-lam[:, None] * (r - shift[:, None]))
			return (a_i * dot(r_vec, n_i) * weight).sum(dim=-1) / (a_i * weight).sum(dim=-1)

		return self._pairwise(queries, fn)

	def self_normalized_signed_log_sum_exp_gradient(self, queries: torch.Tensor, parallelize: bool = True):
		"""(3, Q): A / D + λ B C / D²,  A = Σ a (n - λ h r̂) e,  B = Σ a r̂ e,  C = Σ a h e,  D = Σ a e,  h = r_vec · n."""

		def fn(q):
			r_vec, r, n_i, a_i, shift, lam = self._neighbor_pairs(q)
			lam_e = lam[:, None]
			r_hat = r_vec / r
			weight = torch.exp(-lam_e * (r - shift[:, None]))
			h = dot(r_vec, n_i)
			A = (a_i * (n_i - lam_e * h * r_hat) * weight).sum(dim=-1)
			B = (a_i * r_hat * weight).sum(dim=-1)
			C = (a_i * h * weight).sum(dim=-1)
			D = (a_i * weight).sum(dim=-1)
			return A / D + lam * (B * C) / (D * D)

		return self._pairwise(queries, fn, 3)

	def winding_number(self, queries: torch.Tensor, parallelize: bool = True) -> torch.Tensor:
		"""
		Σ_i a_i n_i · (q - p_i) / (4π |q - p_i|³) over all points (the original's sign: q - p_i, i.e. -1 inside for
		outward normals; libigl and regularized_winding_number use p_i - q, +1 inside).
		"""
		areas = self.get_point_areas()

		def fn(q):
			r_vec = q[:, :, None] - self.points[:, None, :]
			r = norm(r_vec)
			P = dot(self.normals[:, None, :], r_vec) / (r * r * r) / (4.0 * math.pi)
			return (areas[None, :] * P).sum(dim=-1)

		return self._dense(queries, fn)

	def regularized_winding_number(self, queries: torch.Tensor, epsilon: float, parallelize: bool = True):
		"""
		Σ_i a_i S(r/ε) n_i · (p_i - q) / (4π r³),  S(t) = erf(t) - 2t/√π exp(-t²)  (all points).
		"""
		areas = self.get_point_areas()
		eps = float(epsilon)

		def fn(q):
			r_vec = self.points[:, None, :] - q[:, :, None]
			r = norm(r_vec)
			t = r / eps
			S = torch.erf(t) - 2.0 * t / math.sqrt(math.pi) * torch.exp(-t * t)
			P = dot(self.normals[:, None, :], r_vec) / (r * r * r) / (4.0 * math.pi)
			return (areas[None, :] * S * P).sum(dim=-1)

		return self._dense(queries, fn)

	def _dense(self, queries, fn):
		queries = config.check_wide(queries, 3, 'queries').to(self.dtype)
		out = torch.empty(queries.shape[1], dtype=self.dtype, device=self.device)
		for start, end in _chunk_ranges(queries.shape[1], _query_chunk_size(self.n_points)):
			out[start:end] = fn(queries[:, start:end])
		return out

	def compute_point_areas(self) -> None:
		"""Areas of the points: igl::copyleft::cgal::point_areas with each point's 20 nearest points (incl. itself)."""
		from . import igl

		I = get_neighbors(self.points, 20)  # noqa: E741 (libigl's name)
		self.areas = igl.point_areas(self.points, I, self.normals)

	def get_point_areas(self) -> torch.Tensor:
		if getattr(self, 'areas', None) is None:
			self.compute_point_areas()
		return self.areas

	def fast_winding_number_precompute(self) -> None:
		"""
		Octree of the points and order-2 expansions (igl::octree, igl::fast_winding_number precompute). As in the
		original, the expansions use areas from 21 neighbors, computed here and not stored as the point areas.
		"""
		from . import igl

		self._octree = igl.octree(self.points)
		A = igl.point_areas(self.points, get_neighbors(self.points, 21), self.normals)
		self._fwn = igl.fast_winding_number_precompute(self.points, self.normals, A, self._octree)

	def fast_winding_number(self, queries: torch.Tensor) -> torch.Tensor:
		"""
		Barnes-Hut winding number Σ a_i n_i · (p_i - q) / (4π |p_i - q|³) (libigl convention: +1 inside), β = 2.
		Direct terms use the point areas (get_point_areas, 20 neighbors), as the original passes `areas` to the query.
		Deviation: the original's query (n, 3) array becomes a (3, n) tensor; the original reads `areas` even when
		they were never computed (undefined behavior); here they are computed on demand.
		"""
		from . import igl

		if getattr(self, '_fwn', None) is None:
			raise RuntimeError('call fast_winding_number_precompute() first')
		queries = config.check_wide(queries, 3, 'queries').to(self.dtype)
		return igl.fast_winding_number(self.points, self.normals, self.get_point_areas(), self._fwn, queries, beta=2.0)

	# ===== SAMPLING

	def ray_intersection_single(self, ro, rd, diam: float, isovalue: float = 0.0):
		"""All intersections of one ray (see ray_intersections)."""
		return self.ray_intersections(ro[:, None], rd[:, None], isovalue, diam)

	def ray_intersections(self, ros: torch.Tensor, rds: torch.Tensor, isovalue: float = 0.0, diam=None):
		"""
		Sphere tracing of |φ - isovalue| along each ray from t = 0 to t = diameter (C++ ray_intersection_single):
		where |φ| < ε = 1e-4 the point is recorded and the ray steps by max(|φ|, ε) until |φ| ≥ ε; elsewhere it steps
		by |φ|. Returns (3, H) intersection points, in ray order and along each ray in t order.
		All rays march together; one host read per step (rays still marching, points recorded).
		"""
		ros = config.check_wide(ros, 3, 'ray origins').to(self.dtype)
		rds = config.check_wide(rds, 3, 'ray directions').to(self.dtype)
		tmax = float(self.diameter) if diam is None else float(diam)
		epsilon = 1e-4
		n = ros.shape[1]
		t = torch.zeros(n, dtype=self.dtype, device=self.device)
		inner = torch.zeros(n, dtype=torch.bool, device=self.device)
		ray_id = torch.arange(n, device=self.device)
		hits, hit_rays, hit_t = [], [], []
		while ray_id.numel():
			q = ros[:, ray_id] + t * rds[:, ray_id]
			d = torch.abs(self.evaluate_distance(q, isovalue))
			enter = ~inner & (d < epsilon)
			if bool(enter.any()):
				hits.append(q[:, enter])
				hit_rays.append(ray_id[enter])
				hit_t.append(t[enter])
			stay = d < epsilon  # in the band: step max(d, ε) (first step of the inner loop, or continuing it)
			t = t + torch.where(stay, torch.clamp(d, min=epsilon), d)
			inner = stay
			# the C++ checks t < tmax only in the outer loop: rays in the band keep marching
			alive = inner | (t < tmax)
			ray_id, t, inner = ray_id[alive], t[alive], inner[alive]
		if not hits:
			return torch.zeros((3, 0), dtype=self.dtype, device=self.device)
		pts, rays, ts = torch.cat(hits, dim=1), torch.cat(hit_rays), torch.cat(hit_t)
		order = torch.argsort(ts, stable=True)  # by t, then (stable) by ray: ray order, t order within a ray
		order = order[torch.argsort(rays[order], stable=True)]
		return pts[:, order]

	def sample_isosurface(self, n_samples: int, isovalue: float = 0.0, seed: int = 0, batch: int = 256):
		"""
		n_samples points of the level set and their normals (normalized gradients), by ray intersections of random
		lines through the bounding sphere (C++ sample_isosurface): direction uniform on the sphere (θ = 2πu,
		cos φ = 2u - 1), offset uniform on the disk of radius 1 orthogonal to it (θ_d = 2πu, r = √u), origin
		-d + r (cos θ_d s + sin θ_d t) with (s, t) = orthonormal_basis(d), length = diameter. Rays are cast in batches
		until enough points are found; the first n_samples (in ray order) are returned.
		Deviations: the C++ seeds a new std::mt19937 from std::random_device for every ray (not reproducible); here one
		std::mt19937(seed) stream is replayed with the same four draws per ray. The C++ evaluates the normals before
		filling the sample locations (on uninitialized memory); here they are the gradients at the samples.
		"""
		from . import cpp_random

		rng = cpp_random.MT19937(seed)
		uniform = cpp_random.UniformReal(0.0, 1.0)
		found = []
		count = 0
		while count < n_samples:
			u = torch.as_tensor(uniform(rng, 4 * batch).reshape(batch, 4).T, dtype=torch.float64, device=self.device)
			theta, cos_phi, theta_disk, r = (
				2.0 * math.pi * u[0],
				2.0 * u[1] - 1.0,
				2.0 * math.pi * u[2],
				torch.sqrt(u[3]),
			)
			sin_phi = torch.sqrt(1.0 - cos_phi * cos_phi)
			dirs = torch.stack([torch.cos(theta) * sin_phi, torch.sin(theta) * sin_phi, cos_phi]).to(self.dtype)
			s_hat, t_hat = orthonormal_basis(dirs, 0.0)
			r, theta_disk = r.to(self.dtype), theta_disk.to(self.dtype)
			origins = -dirs + r * (torch.cos(theta_disk) * s_hat + torch.sin(theta_disk) * t_hat)
			pts = self.ray_intersections(origins, dirs, isovalue)
			found.append(pts)
			count += pts.shape[1]
		locations = torch.cat(found, dim=1)[:, :n_samples]
		normals = self.evaluate_gradient(locations)
		nn = norm(normals)
		normals = torch.where(torch.isfinite(nn) & (nn > 0), normals / nn, 0.0)
		return locations, normals

	# torus fitting --- this function modifies the member variables controlling tori parameters
	def fit_tori_from_forms(self, coefficients: torch.Tensor):
		"""coefficients (6, |P|) -> sets and returns (centers (3, |P|), axes (2, |P|), major_radii, minor_radii)"""
		coefficients = config.check_wide(coefficients, 6, 'coefficients').to(self.dtype)
		self.set_tori(*fit_torus_at_point(self.points, self.normals, coefficients))
		return self.get_tori()

	# ===== APPLICATIONS

	def principal_curvatures(self, coefficients: torch.Tensor) -> torch.Tensor:
		"""(6, |P|) -> (2, |P|) rows (kappa_min, kappa_max)"""
		coefficients = config.check_wide(coefficients, 6, 'coefficients').to(self.dtype)
		s_hat, t_hat = orthonormal_basis(self.normals)
		_, kappa_min, kappa_max, _, _ = get_torus_parameters(self.normals, coefficients, s_hat, t_hat)
		return torch.stack([kappa_min, kappa_max], dim=0)

	def principal_directions(self, coefficients: torch.Tensor):
		"""(6, |P|) -> kappa_min (|P|,), kappa_max (|P|,), v_min (3, |P|), v_max (3, |P|)"""
		coefficients = config.check_wide(coefficients, 6, 'coefficients').to(self.dtype)
		s_hat, t_hat = orthonormal_basis(self.normals)
		return get_principal_directions_at_point(self.normals, coefficients, s_hat, t_hat)

	# ===== getters & setters

	def get_points(self) -> torch.Tensor:
		return self.points

	def get_points_and_normals(self) -> Tuple[torch.Tensor, torch.Tensor]:
		return self.points, self.normals

	def get_tori(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		return self.centers, self.axes, self.major_radii, self.minor_radii

	def set_tori(
		self, centers: torch.Tensor, axes: torch.Tensor, major_radii: torch.Tensor, minor_radii: torch.Tensor
	) -> None:
		"""centers (3, |P|), axes (2, |P|), major_radii (|P|,), minor_radii (|P|,) on the configured device"""
		for name, radii in (('major_radii', major_radii), ('minor_radii', minor_radii)):
			if not isinstance(radii, torch.Tensor) or radii.device != self.device or radii.numel() != self.n_points:
				raise ValueError(f'{name}: expected a ({self.n_points},) tensor on {self.device}')
		self.centers = config.check_wide(centers, 3, 'centers').to(self.dtype).contiguous()
		self.axes = config.check_wide(axes, 2, 'axes').to(self.dtype).contiguous()
		self.major_radii = major_radii.to(self.dtype).reshape(-1).contiguous()
		self.minor_radii = minor_radii.to(self.dtype).reshape(-1).contiguous()
		self._axes3d = rotation_to_cartesian(self.axes).contiguous()

	def set_use_areas(self, val: bool) -> None:
		"""Weight the blend by the point areas (computed on first use; the C++ requires get_point_areas() first)."""
		self.use_areas = bool(val)

	def set_k_evaluation(self, k: int) -> None:
		self.k_evaluation = int(k)

	def set_radius_evaluation(self, r: float) -> None:
		"""
		r must be a host float (selecting the evaluation mode is a host decision).
		Rounded through float32 because the C++ setter takes `float r` (reproduces its results exactly).
		"""
		self.radius_evaluation = float(np.float32(r))

	def set_outliers(self, indices) -> None:
		self.is_outlier = torch.zeros(self.n_points, dtype=torch.bool, device=self.device)
		if len(indices) > 0:
			self.is_outlier[torch.as_tensor(indices, dtype=torch.long, device=self.device)] = True

	def set_exact_laplacian(self, val: bool) -> None:
		"""Added in the port: use the exact Laplacian of each torus SDF in evaluate_laplacian (default: as the C++)."""
		self.exact_laplacian = bool(val)

	def set_lambda_scale(self, scale: float) -> None:
		"""Rounded through float32 because the C++ setter takes `float scale`."""
		self.lambda_scale = float(np.float32(scale))
