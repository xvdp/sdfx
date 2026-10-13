"""
Torch port of src/pointsastori/network.py (JAX / flax.nnx).

Function and class names are kept from the original for traceability, including the `_jax` suffixes.

Layout (see config.py): geometric arrays are structure-of-arrays with the component axis first, e.g. normals (3, ...),
axes (2, ...), coefficients (6, ...). Inside the transformer, activations keep the usual (batch, k, features) layout,
which is what nn.Linear / LayerNorm / scaled_dot_product_attention consume contiguously.

Equivalences with flax.nnx defaults (relevant when loading the pre-trained weights):
	nnx.Linear kernel (in, out)            -> nn.Linear weight (out, in) = kernel.T
	nnx.LayerNorm(epsilon=1e-6)            -> nn.LayerNorm(eps=1e-6)       (torch default is 1e-5)
	nnx.gelu == jax.nn.gelu(approximate=True) -> nn.GELU(approximate='tanh') (torch default is exact erf)
	nnx.MultiHeadAttention q/k/v kernels (in, heads, head_dim), out kernel (heads, head_dim, out);
	    attention softmax(q kᵀ / sqrt(head_dim)) v   -> MultiHeadAttention below (same parameters, flattened)
	jnp.median (mean of the two middle values for even k) -> _median below (torch.median returns the lower one)
"""

from __future__ import annotations

import gc
import math
import pickle
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import config
from .shape_2d import cross, dot, norm
from .shape_3d import export_PLY, export_PLY_with_colors, get_neighbors

NAN_EPSILON = 1e-10

MODELS_DIR = 'models/'

# ========================================================================
# HELPERS
# ========================================================================


def orthonormal_basis_with_rotation(n: torch.Tensor, rotation: float) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Compute an orthonormal basis (s, t) perpendicular to normal vector n, with a specified rotation of the axes.

	Args:
		n: (3, ...) array of normal vectors (will be normalized)
		rotation: scalar rotation angle about the normal

	Returns:
		s_hat: (3, ...) first basis vector
		t_hat: (3, ...) second basis vector
	"""
	# Normalize the normal vector
	n_norm = norm(n)[None]
	n_hat = n / torch.clamp_min(n_norm, NAN_EPSILON)

	# Choose initial vector u based on which component of n is smallest
	# If |n.x| > 0.9, use (0,1,0), otherwise use (1,0,0)
	abs_n_x = torch.abs(n_hat[0:1])
	zeros = torch.zeros_like(abs_n_x)
	ones = torch.ones_like(abs_n_x)
	u = torch.where(
		abs_n_x > 0.9,
		torch.cat([zeros, ones, zeros], dim=0),
		torch.cat([ones, zeros, zeros], dim=0),
	)

	# Apply rotation about n_hat using Rodrigues' formula
	# R = I + sin(θ) * K + (1 - cos(θ)) * K^2
	# where K is the skew-symmetric matrix for n_hat
	rotation = torch.as_tensor(rotation, dtype=n.dtype, device=n.device)
	cos_rot = torch.cos(rotation)
	sin_rot = torch.sin(rotation)

	# Compute u_rotated = cos(θ)*u + sin(θ)*(n × u) + (1-cos(θ))*(n·u)*n
	n_cross_u = cross(n_hat, u)
	n_dot_u = dot(n_hat, u)[None]
	u_rotated = cos_rot * u + sin_rot * n_cross_u + (1.0 - cos_rot) * n_dot_u * n_hat

	# Compute s = n × u (perpendicular to both)
	s = cross(n_hat, u_rotated)
	s_norm = norm(s)[None]
	s_hat = s / torch.clamp_min(s_norm, NAN_EPSILON)

	# Compute t = n × s (completes orthonormal basis)
	t_hat = cross(n_hat, s_hat)

	return s_hat, t_hat


# ========================================================================
# (AUTO-DIFFERENTIABLE) DISTANCE FUNCTIONS
# ========================================================================


def cartesian_to_rotation_jax(u: torch.Tensor) -> torch.Tensor:
	"""
	Convert 3D unit vector to spherical coordinates (theta, phi).

	Args:
		u: (3, ...) array of 3D vectors

	Returns:
		axes: (2, ...) array of (theta, phi) in radians
			theta: azimuthal angle in [0, 2π]
			phi: polar angle in [0, π]
	"""
	# Normalize input
	u_norm = norm(u)[None]
	u_hat = u / torch.clamp_min(u_norm, NAN_EPSILON)

	x = u_hat[0]
	y = u_hat[1]
	z = u_hat[2]

	phi = torch.arccos(torch.clamp(z, -1.0, 1.0))
	safe_x = torch.sign(x) * torch.clamp_min(torch.abs(x), NAN_EPSILON)
	theta = torch.arctan2(y, safe_x)
	return torch.stack([theta, phi], dim=0)


def rotation_to_cartesian_jax(axes: torch.Tensor) -> torch.Tensor:
	"""
	Convert spherical coordinates (theta, phi) to 3D unit vector.

	Args:
		axes: (2, ...) array of (theta, phi) in radians

	Returns:
		u: (3, ...) array of 3D unit vectors
	"""
	theta = axes[0]
	phi = axes[1]

	# Spherical to Cartesian conversion
	sin_phi = torch.sin(phi)
	cos_phi = torch.cos(phi)
	sin_theta = torch.sin(theta)
	cos_theta = torch.cos(theta)

	x = sin_phi * cos_theta
	y = sin_phi * sin_theta
	z = cos_phi

	return torch.stack([x, y, z], dim=0)


def _compute_shifts_from_distances(distances: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	"""
	Compute per-query shift from a pre-computed distance matrix.

	Args:
		distances: (batch_size, n_queries, max_points) distances from each query to each point
		mask: (max_points,) valid-point mask

	Returns:
		shifts: (batch_size, n_queries)
	"""
	k = 64
	max_points = distances.shape[-1]
	k_actual = min(k, max_points)

	m = mask[None, None, :]  # (1, 1, max_points)

	# top_k on negated distances: invalid points → -inf so they're never selected.
	neg_distances = torch.where(m, -distances, -torch.inf)
	neg_knn = torch.topk(
		neg_distances, k_actual, dim=-1, largest=True, sorted=True
	).values  # (batch, n_queries, k_actual)
	knn_dists = -neg_knn  # positive, ascending order (nearest first)

	# Shift = 0.5 * max distance among k nearest valid neighbors
	finite_knn = torch.where(torch.isfinite(knn_dists), knn_dists, 0.0)
	return 0.5 * finite_knn.amax(dim=-1)  # (batch, n_queries)


def compute_lambdas_jax(shifts: torch.Tensor) -> torch.Tensor:
	"""
	Compute lambda parameter from shift.

	Args:
		shift: (...,) shift values

	Returns:
		lam: (...,) lambda values
	"""
	return 64.0 / shifts


def torus_precompute_masked(
	mask: torch.Tensor,
	points: torch.Tensor,
	normals: torch.Tensor,
	coeffs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Only compute torii for valid points (where mask is True).
	(As in the original, every point is computed; `mask` is not used here — it is applied by the loss.)

	The per-point `solve_single` of the original (vmapped) is written directly over the point axis.

	Args:
		mask: (max_points,) - boolean mask indicating valid points
		points: (3, max_points) - may contain padding
		normals: (3, max_points) - may contain padding
		coeffs: (6, max_points) - may contain padding

	Returns:
		centers: (3, max_points)
		axes: (2, max_points)
		major_radii: (max_points,)
		minor_radii: (max_points,)
	"""
	# Center point info
	p_i = points
	n_i = normals

	# Compute orthonormal bases
	s_hat, t_hat = orthonormal_basis_with_rotation(n_i, 0.0)

	# Extract polynomial coefficients
	a00 = coeffs[0]
	a01 = coeffs[1]
	a10 = coeffs[2]
	a11 = coeffs[3]
	a02 = coeffs[4]
	a20 = coeffs[5]

	# Compute principal curvatures from polynomial fit
	A = torch.sqrt(1.0 + a01**2 + a10**2)
	# A = jnp.maximum(A, NAN_EPSILON)

	# Compute discriminant for eigenvector calculation
	v1 = a20 * a01**2 - a02 * a10**2 - a02 + a20
	v2 = a11 * (a10**2 + 1.0) - 2.0 * a01 * a10 * a20
	term = a02 * (a10**2 + 1.0) - a01 * a10 * a11 + a20 * (a01**2 + 1.0)
	discriminant = (1.0 + a01**2 + a10**2) * (a11**2 - 4.0 * a02 * a20) + term**2
	discriminant = torch.sqrt(torch.clamp_min(discriminant, NAN_EPSILON))  # sqrt not differentiable at 0

	ev_pos_0 = v1 + discriminant
	ev_pos_0 = torch.sign(ev_pos_0) * torch.clamp_min(torch.abs(ev_pos_0), NAN_EPSILON)
	ev_pos_x, ev_pos_y = ev_pos_0, v2

	ev_neg_1 = v2
	ev_neg_1 = torch.sign(ev_neg_1) * torch.clamp_min(torch.abs(ev_neg_1), NAN_EPSILON)
	ev_neg_x, ev_neg_y = v1 - discriminant, ev_neg_1

	# Even though normalization isn't necessary because we take a Rayleigh quotient, it helps numerically.
	ev_pos_norm = torch.sqrt(ev_pos_x**2 + ev_pos_y**2)
	ev_neg_norm = torch.sqrt(ev_neg_x**2 + ev_neg_y**2)
	ev_pos_x = ev_pos_x / ev_pos_norm
	ev_pos_y = ev_pos_y / ev_pos_norm
	ev_neg_x = ev_neg_x / ev_neg_norm
	ev_neg_y = ev_neg_y / ev_neg_norm

	# Convert to 3D eigenvectors
	dQstards = s_hat + a10 * n_i  # (3, max_points)
	dQstardt = t_hat + a01 * n_i  # (3, max_points)

	v_pos = ev_pos_x * dQstards + ev_pos_y * dQstardt
	v_neg = ev_neg_x * dQstards + ev_neg_y * dQstardt

	# Compute normal at fitted point
	normal = (n_i - a10 * s_hat - a01 * t_hat) / A

	# Compute curvatures using Rayleigh quotient
	I_11 = 1.0 + a10**2
	I_12 = a10 * a01
	I_22 = 1.0 + a01**2

	II_11 = 2.0 * a20 / A
	II_12 = a11 / A
	II_22 = 2.0 * a02 / A

	# Compute kappa = (ev^T @ II @ ev) / (ev^T @ I @ ev)
	# For positive eigenvector
	II_ev_pos_x = II_11 * ev_pos_x + II_12 * ev_pos_y
	II_ev_pos_y = II_12 * ev_pos_x + II_22 * ev_pos_y
	I_ev_pos_x = I_11 * ev_pos_x + I_12 * ev_pos_y
	I_ev_pos_y = I_12 * ev_pos_x + I_22 * ev_pos_y

	numer_pos = ev_pos_x * II_ev_pos_x + ev_pos_y * II_ev_pos_y
	denom_pos = ev_pos_x * I_ev_pos_x + ev_pos_y * I_ev_pos_y
	denom_pos = torch.sign(denom_pos) * torch.clamp_min(torch.abs(denom_pos), NAN_EPSILON)
	kappa_pos = numer_pos / denom_pos

	# For negative eigenvector
	II_ev_neg_x = II_11 * ev_neg_x + II_12 * ev_neg_y
	II_ev_neg_y = II_12 * ev_neg_x + II_22 * ev_neg_y
	I_ev_neg_x = I_11 * ev_neg_x + I_12 * ev_neg_y
	I_ev_neg_y = I_12 * ev_neg_x + I_22 * ev_neg_y

	numer_neg = ev_neg_x * II_ev_neg_x + ev_neg_y * II_ev_neg_y
	denom_neg = ev_neg_x * I_ev_neg_x + ev_neg_y * I_ev_neg_y
	denom_neg = torch.sign(denom_neg) * torch.clamp_min(torch.abs(denom_neg), NAN_EPSILON)
	kappa_neg = numer_neg / denom_neg

	# Identify min/max curvatures
	abs_kappa_pos = torch.abs(kappa_pos)
	abs_kappa_neg = torch.abs(kappa_neg)

	use_pos_as_min = abs_kappa_pos < abs_kappa_neg
	kappa_min = torch.where(use_pos_as_min, kappa_pos, kappa_neg)
	kappa_max = torch.where(use_pos_as_min, kappa_neg, kappa_pos)
	v_min = torch.where(use_pos_as_min, v_pos, v_neg)

	abs_kappa_max = torch.clamp_min(torch.abs(kappa_max), NAN_EPSILON)
	abs_kappa_min = torch.clamp_min(torch.abs(kappa_min), NAN_EPSILON)

	# Compute torus parameters
	r = 1.0 / abs_kappa_max
	R = 1.0 / abs_kappa_min - torch.sign(kappa_max * kappa_min) * r

	# Sign convention from C++
	s = torch.where(((kappa_max < 0) & (kappa_min < 0)) | ((kappa_max < 0) & (kappa_min > 0)), 1.0, -1.0).to(r.dtype)

	# Torus center
	center = p_i + a00 * n_i - (s / abs_kappa_min) * normal

	# Torus axis (u = normal × v_min, normalized)
	u = cross(normal, v_min)
	u_norm = norm(u)
	u = u / torch.clamp_min(u_norm, NAN_EPSILON)

	# Convert to rotation representation
	axis = cartesian_to_rotation_jax(u)  # (2, max_points)

	major_radius = R
	minor_radius = s * r

	return center, axis, major_radius, minor_radius


def torus_signed_distance(q, center, axis, major_radius, minor_radius):
	"""
	Compute the signed distance at q to a single (signed) torus.
	q, center: (3, ...), axis: (2, ...), radii: (...) — broadcasting over the trailing axes.
	"""
	axis3d = rotation_to_cartesian_jax(axis)

	# Distance from query to torus center
	r_vec = q - center

	# v = |r_vec × axis|
	cross_prod = cross(r_vec, axis3d)
	v = norm(cross_prod)

	# Distance components
	d1 = v - major_radius
	d2 = dot(r_vec, axis3d)

	# Combined distance to torus surface
	d = torch.sqrt(d1**2 + d2**2)

	s = torch.sign(minor_radius)
	g_z = s * (d - torch.abs(minor_radius))

	return g_z


def torus_distance_gradient_masked(
	mask: torch.Tensor,
	queries: torch.Tensor,
	points: torch.Tensor,
	centers: torch.Tensor,
	axes: torch.Tensor,
	major_radii: torch.Tensor,
	minor_radii: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Only sum over torii at valid points (where mask is True).

	Args:
		mask: (max_points,) - boolean mask indicating valid points
		queries: (3, batch_size, n_queries) array of query points
		points: (3, max_points) - may contain padding
		centers: (3, max_points) - torii centers
		axes: (2, max_points) - torii axes
		major_radii: (max_points,)
		minor_radii: (max_points,)

	Returns:
		distances: (batch_size, n_queries,) array of signed distances
		gradients: (3, batch_size, n_queries) array of signed distance gradients
	"""
	# Create validity mask for torii without NaNs --- in case there were still NaNs that escaped torus_precompute_masked()
	valid_torus_mask = (
		mask
		& torch.isfinite(centers).all(dim=0)
		& torch.isfinite(axes).all(dim=0)
		& torch.isfinite(major_radii)
		& torch.isfinite(minor_radii)
	)  # (max_points,)

	axis3d = rotation_to_cartesian_jax(axes)  # (3, max_points)

	# Compute distance and gradient for each query-torus pair

	# Reshape for broadcasting: queries (3, batch, n_queries, 1)
	q = queries[:, :, :, None]

	# Broadcast points, centers, etc. for all queries
	# (3, 1, 1, max_points)
	p_i = points[:, None, None, :]

	# Compute query-to-point distances once and reuse for both shift and weight computation.
	rVec = q - p_i  # (3, batch, n_queries, max_points)
	dist_to_point = norm(rVec)  # (batch, n_queries, max_points)

	shifts = _compute_shifts_from_distances(dist_to_point, valid_torus_mask)  # (batch, n_queries)
	lam = compute_lambdas_jax(shifts)  # (batch_size, n_queries)
	c_i = centers[:, None, None, :]
	axis_i = axis3d[:, None, None, :]
	R_i = major_radii[None, None, :]
	r_i = minor_radii[None, None, :]
	mask_i = valid_torus_mask[None, None, :]  # (1, 1, max_points)

	# Distance from query to torus center
	r_vec = q - c_i  # (3, batch, n_queries, max_points)

	# v = |r_vec × axis|
	cross_prod = cross(r_vec, axis_i)  # (3, batch, n_queries, max_points)
	v = norm(cross_prod)  # (batch, n_queries, max_points)

	# Distance components
	d1 = v - R_i  # (batch, n_queries, max_points)
	d2 = dot(r_vec, axis_i)  # (batch, n_queries, max_points)

	# Combined distance to torus surface
	d = torch.sqrt(d1**2 + d2**2)  # (batch, n_queries, max_points)

	s = torch.sign(r_i)  # (1, 1, max_points)
	g_z = s * (d - torch.abs(r_i))  # (batch, n_queries, max_points)
	g_z = torch.where(mask_i, g_z, 0.0)  # apply mask

	# Gradient computation (matrix G from C++ code)
	# G is a 2x3 matrix, w is a 2-vector, grad = s * (w^T @ G)^T
	v_safe = torch.clamp_min(v, NAN_EPSILON)

	ax, ay, az = axis_i[0], axis_i[1], axis_i[2]
	G_11 = (ax * ay * (c_i[1] - q[1]) + ax * az * (c_i[2] - q[2]) - (ay**2 + az**2) * (c_i[0] - q[0])) / v_safe
	G_12 = (ax * ay * (c_i[0] - q[0]) + ay * az * (c_i[2] - q[2]) - (ax**2 + az**2) * (c_i[1] - q[1])) / v_safe
	G_13 = (ax * az * (c_i[0] - q[0]) + ay * az * (c_i[1] - q[1]) - (ax**2 + ay**2) * (c_i[2] - q[2])) / v_safe

	G_21 = ax
	G_22 = ay
	G_23 = az

	# Weight vector w = (d1, d2) / ||(d1, d2)||
	d_safe = torch.clamp_min(d, NAN_EPSILON)
	w_x = d1 / d_safe
	w_y = d2 / d_safe

	# Gradient: grad = s * (w^T @ G)^T = s * G^T @ w
	grad_x = s * (w_x * G_11 + w_y * G_21)
	grad_y = s * (w_x * G_12 + w_y * G_22)
	grad_z = s * (w_x * G_13 + w_y * G_23)

	# Apply mask to gradients
	grad_x = torch.where(mask_i, grad_x, 0.0)
	grad_y = torch.where(mask_i, grad_y, 0.0)
	grad_z = torch.where(mask_i, grad_z, 0.0)

	grad_g = torch.stack([grad_x, grad_y, grad_z], dim=0)  # (3, batch, n_queries, max_points)

	# Compute exponential weights
	rHat = rVec / torch.clamp_min(dist_to_point, NAN_EPSILON)[None]

	# Exponential weight
	lam_expanded = lam[:, :, None]  # (batch, n_queries, 1)
	shift_expanded = shifts[:, :, None]  # (batch, n_queries, 1)
	weight = torch.exp(-lam_expanded * (dist_to_point - shift_expanded))  # (batch, n_queries, max_points)

	# Apply mask to weights
	weight = torch.where(mask_i, weight, 0.0)

	# Accumulate: grad_g_minus_lambda_n_g, g, n
	grad_g_minus_lambda_n_g = torch.sum(
		weight[None] * (grad_g - lam_expanded[None] * rHat * g_z[None]), dim=-1
	)  # (3, batch, n_queries)

	g = torch.sum(weight * g_z, dim=-1)  # (batch, n_queries)
	n = torch.sum(weight[None] * rHat, dim=-1)  # (3, batch, n_queries)

	normalization = torch.sum(weight, dim=-1)  # (batch, n_queries)
	normalization_safe = torch.clamp_min(normalization, NAN_EPSILON)

	# Compute final distance and gradient

	A = grad_g_minus_lambda_n_g / normalization_safe  # (3, batch, n_queries)
	B = lam * n / normalization_safe  # (3, batch, n_queries)
	dist = g / normalization_safe
	grad = A + dist * B  # (3, batch, n_queries)

	return dist, grad


# ========================================================================
# EXPORT FOR VISUALIZATION
# ========================================================================
# Export boundary: these functions take device tensors (SoA) or NumPy arrays, move them to the host once, and write files.


def _host(x) -> np.ndarray:
	return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def generate_torus_mesh(center: np.ndarray, axis: np.ndarray, major_radius: float, minor_radius: float):
	"""
	Generate quad mesh for a single torus.

	Args:
		center: (3,) center position
		axis: (2,) axis in spherical coordinates (theta, phi)
		major_radius: major radius R
		minor_radius: minor radius r (can be negative)

	Returns:
		vertices: (major_res * minor_res, 3)
		faces: (major_res * minor_res, 4) quad indices
	"""
	center = _host(center).astype(np.float64)
	axis = _host(axis).astype(np.float64)
	major_radius = float(major_radius)
	minor_radius = float(minor_radius)

	# Handle degenerate cases
	if np.abs(minor_radius) < 1e-10 or np.abs(major_radius) < 1e-10:
		# Degenerate torus - return empty mesh
		return np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int32)

	# major_res: number of segments around major circle
	# minor_res: number of segments around minor circle
	max_spacing = 0.1
	min_res = 16  # for small point clouds (a few thousand points), use 16
	max_res = 128
	major_res = min(max(int(major_radius / max_spacing), min_res), max_res)
	minor_res = min(max(int(minor_radius / max_spacing), min_res), max_res)

	# Get torus axis as 3D vector
	axis_3d = rotation_to_cartesian_jax(torch.from_numpy(axis))

	# Create two orthogonal vectors perpendicular to axis
	v1, v2 = orthonormal_basis_with_rotation(axis_3d, 0.0)
	axis_3d, v1, v2 = axis_3d.numpy(), v1.numpy(), v2.numpy()

	# Generate vertices (vectorized over the (major, minor) grid; same order as the original double loop)
	# Angle around major circle / minor circle
	theta = (2 * np.pi * np.arange(major_res) / major_res)[:, None, None]  # (major_res, 1, 1)
	phi = (2 * np.pi * np.arange(minor_res) / minor_res)[None, :, None]  # (1, minor_res, 1)

	# Normal direction at this point on major circle (pointing away from center)
	major_normal = np.cos(theta) * v1 + np.sin(theta) * v2  # (major_res, 1, 3)

	# Point on major circle
	major_circle_point = center + major_radius * major_normal

	# Offset from major circle
	# Note: minor_radius can be negative (changes which side of major circle)
	offset = minor_radius * (np.cos(phi) * major_normal + np.sin(phi) * axis_3d)  # (major_res, minor_res, 3)

	vertices = (major_circle_point + offset).reshape(-1, 3)

	# Generate quad faces
	i = np.arange(major_res)[:, None]
	j = np.arange(minor_res)[None, :]
	v0 = i * minor_res + j
	v1_ = ((i + 1) % major_res) * minor_res + j
	v2_ = ((i + 1) % major_res) * minor_res + (j + 1) % minor_res
	v3 = i * minor_res + (j + 1) % minor_res

	faces = np.stack([v1_, v2_, v3, v0], axis=-1).reshape(-1, 4).astype(np.int32)

	return vertices, faces


def export_torus(center, axis, major_radius: float, minor_radius: float, save_filepath: str):
	"""
	Export a single torus.

	Args:
		center: (3,)
		axis: (2,)
		major_radius: scalar
		minor_radius: scalar
	"""
	vertices, faces = generate_torus_mesh(center, axis, major_radius, minor_radius)
	# export_OBJ(vertices, faces, filepath=save_filepath)
	export_PLY(vertices, faces, filepath=save_filepath, binary=True)


def hsv_to_rgb(h, s, v):
	"""
	Convert HSV to RGB (matching the GLSL hsv2rgb function).

	Args:
		h, s, v: Hue [0,1], Saturation [0,1], Value [0,1]

	Returns:
		(r, g, b) tuple in [0, 1]
	"""
	c = v * s
	x = c * (1 - abs((h * 6) % 2 - 1))
	m = v - c

	if h < 1 / 6:
		r, g, b = c, x, 0
	elif h < 2 / 6:
		r, g, b = x, c, 0
	elif h < 3 / 6:
		r, g, b = 0, c, x
	elif h < 4 / 6:
		r, g, b = 0, x, c
	elif h < 5 / 6:
		r, g, b = x, 0, c
	else:
		r, g, b = c, 0, x

	return (r + m, g + m, b + m)


def torus_color(i):
	"""
	Compute the color of the i-th torus (matching shader logic).

	Args:
		i: Torus index

	Returns:
		(r, g, b) in [0, 255] as uint8
	"""
	incr = (1.0 + np.sqrt(5.0)) / 20.0
	h = (i * incr) % 1.0
	s = 0.5
	v = 1.0

	r, g, b = hsv_to_rgb(h, s, v)
	return np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)


def _tori_to_host(centers, axes, major_radii, minor_radii):
	"""(3, n), (2, n), (n,), (n,) device tensors -> (n, 3), (n, 2), (n,), (n,) NumPy arrays (export boundary)."""
	if isinstance(centers, torch.Tensor):
		return config.to_numpy(centers), config.to_numpy(axes), _host(major_radii), _host(minor_radii)
	return np.asarray(centers), np.asarray(axes), np.asarray(major_radii), np.asarray(minor_radii)


def export_tori(centers, axes, major_radii, minor_radii, save_filepath: str):
	"""
	Export a collection of torii in a single file.

	Args:
		centers: (3, n_torii) tensor (or (n_torii, 3) NumPy array)
		axes: (2, n_torii) tensor (or (n_torii, 2) NumPy array)
		major_radii: (n_torii, )
		minor_radii: (n_torii, )
	"""
	centers, axes, major_radii, minor_radii = _tori_to_host(centers, axes, major_radii, minor_radii)
	n_torii = len(centers)

	all_vertices = []
	all_faces = []
	all_colors = []
	vertex_offset = 0

	n_valid = 0

	for i in range(n_torii):
		# Check if torus is valid
		is_valid = (
			np.isfinite(centers[i]).all()
			and np.isfinite(axes[i]).all()
			and np.isfinite(major_radii[i])
			and np.isfinite(minor_radii[i])
			and np.abs(minor_radii[i]) > 1e-10
			and np.abs(major_radii[i]) > 1e-10
		)

		if not is_valid:
			continue

		# Generate mesh for this torus
		print(f'Generating torus mesh {i}...')
		vertices, faces = generate_torus_mesh(centers[i], axes[i], major_radii[i], minor_radii[i])

		if len(vertices) == 0:
			continue

		# Compute color for this torus (same color for all vertices of this torus)
		color = torus_color(i)
		vertex_colors = np.tile(color, (len(vertices), 1))

		# Add to combined mesh with offset
		all_vertices.append(vertices)
		all_faces.append(faces + vertex_offset)
		all_colors.append(vertex_colors)

		vertex_offset += len(vertices)
		n_valid += 1

	# Combine all meshes
	print('Combining all torus mehses...')
	if len(all_vertices) > 0:
		all_vertices = np.vstack(all_vertices)
		all_faces = np.vstack(all_faces)
		all_colors = np.vstack(all_colors)
		export_PLY_with_colors(all_vertices, all_faces, all_colors, save_filepath, binary=True)
		print(f'Exported {n_valid}/{n_torii} torii to {save_filepath}')
	else:
		print(f'Warning: No valid torii found, empty mesh exported to {save_filepath}')
		with open(save_filepath, 'w') as f:
			f.write('# Empty mesh (no valid torii)\n')


def _export_tori_single(
	centers: np.ndarray,
	axes: np.ndarray,
	major_radii: np.ndarray,
	minor_radii: np.ndarray,
	save_filepath: str,
	global_index_offset: int = 0,
	verbose: bool = True,
) -> int:
	"""
	Export a collection of torii to a single file.

	Args:
		centers: (n_torii, 3)  (host arrays; see export_tori_chunked)
		axes: (n_torii, 2)
		major_radii: (n_torii,)
		minor_radii: (n_torii,)
		save_filepath: output filepath
		global_index_offset: offset for global indexing (for consistent colors across chunks)
		verbose: print progress

	Returns:
		Number of valid torii exported
	"""
	n_torii = len(centers)

	all_vertices = []
	all_faces = []
	all_colors = []
	vertex_offset = 0
	n_valid = 0

	for i in range(n_torii):
		# Check if torus is valid
		is_valid = (
			np.isfinite(centers[i]).all()
			and np.isfinite(axes[i]).all()
			and np.isfinite(major_radii[i])
			and np.isfinite(minor_radii[i])
			and np.abs(minor_radii[i]) > 1e-10
			and np.abs(major_radii[i]) > 1e-10
		)

		if not is_valid:
			continue

		# Generate mesh for this torus
		vertices, faces = generate_torus_mesh(centers[i], axes[i], major_radii[i], minor_radii[i])

		if len(vertices) == 0:
			continue

		# Compute color using global index for consistency across chunks
		global_index = global_index_offset + i
		color = torus_color(global_index)
		vertex_colors = np.tile(color, (len(vertices), 1))

		# Add to combined mesh with offset
		all_vertices.append(vertices)
		all_faces.append(faces + vertex_offset)
		all_colors.append(vertex_colors)

		vertex_offset += len(vertices)
		n_valid += 1

	# Combine and export
	if len(all_vertices) > 0:
		all_vertices = np.vstack(all_vertices)
		all_faces = np.vstack(all_faces)
		all_colors = np.vstack(all_colors)
		export_PLY_with_colors(all_vertices, all_faces, all_colors, save_filepath, binary=True)
		if verbose:
			print(f'Exported {n_valid}/{n_torii} torii to {save_filepath}')
	else:
		if verbose:
			print(f'Warning: No valid torii found, skipping {save_filepath}')

	return n_valid


def export_tori_chunked(
	centers,
	axes,
	major_radii,
	minor_radii,
	save_filepath: str,
	chunk_size: int = 1_000_000,
	verbose: bool = True,
) -> List[str]:
	"""
	Export a collection of torii in chunks, with each chunk saved to a separate file.

	Args:
		centers: (3, n_torii) torus centers (tensor; or (n_torii, 3) NumPy array)
		axes: (2, n_torii) torus axes in spherical coordinates (tensor; or (n_torii, 2) NumPy array)
		major_radii: (n_torii,) torus major radii
		minor_radii: (n_torii,) torus minor radii
		save_filepath: base filepath for output (e.g., "output/torii.ply")
					   chunks will be saved as "output/torii_chunk0.ply", etc.
		chunk_size: number of torii per chunk (default 1M)
		verbose: print progress information

	Returns:
		List of output filepaths that were created
	"""
	soa = isinstance(centers, torch.Tensor)
	n_torii = centers.shape[1] if soa else len(centers)
	print(f'# torii: {n_torii}')

	# Parse base filepath
	if save_filepath.endswith('.ply'):
		base_path = save_filepath[:-4]
	else:
		base_path = save_filepath

	# Process in chunks
	n_chunks = (n_torii + chunk_size - 1) // chunk_size

	if verbose:
		print(f'Exporting {n_torii:,} torii in {n_chunks} chunks of up to {chunk_size:,} each...')

	output_files = []
	total_valid = 0

	for chunk_idx in range(n_chunks):
		start = chunk_idx * chunk_size
		end = min(start + chunk_size, n_torii)

		if verbose:
			progress = (chunk_idx + 1) / n_chunks * 100
			print(f'  Chunk {chunk_idx}/{n_chunks - 1} (torii {start:,}-{end - 1:,}, {progress:.1f}%)...')

		# Extract chunk data (device -> host per chunk)
		if soa:
			centers_chunk, axes_chunk, major_chunk, minor_chunk = _tori_to_host(
				centers[:, start:end], axes[:, start:end], major_radii[start:end], minor_radii[start:end]
			)
		else:
			centers_chunk = centers[start:end]
			axes_chunk = axes[start:end]
			major_chunk = major_radii[start:end]
			minor_chunk = minor_radii[start:end]

		# Export chunk with global index offset for consistent coloring
		filepath = f'{base_path}_chunk{chunk_idx}.ply'
		n_valid = _export_tori_single(
			centers_chunk, axes_chunk, major_chunk, minor_chunk, filepath, global_index_offset=start, verbose=False
		)

		if n_valid > 0:
			output_files.append(filepath)
			total_valid += n_valid
			if verbose:
				print(f'	Exported {n_valid:,} valid torii to {filepath}')
		else:
			if verbose:
				print(f'	No valid torii in chunk {chunk_idx}, skipping file creation')

		# Free memory
		del centers_chunk, axes_chunk, major_chunk, minor_chunk
		gc.collect()

	if verbose:
		print(f'  Done. Exported {total_valid:,}/{n_torii:,} valid torii to {len(output_files)} files.')

	return output_files


# ========================================================================
# NEURAL NETWORK ARCHITECTURE
# ========================================================================


def _lecun_normal_(weight: torch.Tensor, fan_in: int) -> None:
	"""
	flax default kernel init (lecun_normal = variance_scaling(1, 'fan_in', 'truncated_normal')):
	w = σ · TN(-2, 2), σ = sqrt(1 / fan_in) / 0.87962566103423978 (the std of a standard normal truncated to [-2, 2]).
	Matches the *distribution* of a freshly initialized flax model (relevant for training from scratch only).
	"""
	std = math.sqrt(1.0 / fan_in) / 0.87962566103423978
	nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


def _flax_linear(in_features: int, out_features: int) -> nn.Linear:
	layer = nn.Linear(in_features, out_features)
	_lecun_normal_(layer.weight, in_features)
	nn.init.zeros_(layer.bias)
	return layer


class MultiHeadAttention(nn.Module):
	"""
	Self-attention with the parameterization and math of flax.nnx.MultiHeadAttention (no mask, decode=False):
		q, k, v = x W_q + b_q, ...  split into heads;  y = softmax(q kᵀ / sqrt(head_dim)) v;  out = concat_h(y) W_o + b_o
	The flax kernels (in, heads, head_dim) / (heads, head_dim, out) are stored flattened as nn.Linear weights.
	"""

	def __init__(
		self, num_heads: int, in_features: int, qkv_features: int, out_features: int, dropout_rate: float = 0.0
	):
		super().__init__()
		self.num_heads = num_heads
		self.head_dim = qkv_features // num_heads
		self.query = _flax_linear(in_features, qkv_features)
		self.key = _flax_linear(in_features, qkv_features)
		self.value = _flax_linear(in_features, qkv_features)
		self.out = _flax_linear(qkv_features, out_features)
		self.dropout_rate = dropout_rate

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch, k, _ = x.shape

		def heads(t):
			# (batch, k, heads * head_dim) -> (batch, heads, k, head_dim): a view with unit stride on head_dim
			return t.view(batch, k, self.num_heads, self.head_dim).transpose(1, 2)

		y = F.scaled_dot_product_attention(
			heads(self.query(x)),
			heads(self.key(x)),
			heads(self.value(x)),
			dropout_p=self.dropout_rate if self.training else 0.0,
		)
		y = y.transpose(1, 2).reshape(batch, k, self.num_heads * self.head_dim)  # merge heads (one copy)
		return self.out(y)


class TransformerBlock(nn.Module):
	"""
	Transformer block (pre-norm), port of the flax.nnx block.
	"""

	def __init__(self, embed_dim: int, n_heads: int, mlp_dim: int, dropout_rate: float = 0.0):
		super().__init__()
		self.attention = MultiHeadAttention(
			num_heads=n_heads,
			in_features=embed_dim,
			qkv_features=embed_dim,
			out_features=embed_dim,
		)

		# feed-forward network (indices 0 and 3 match the flax checkpoint's ffn/layers/0 and ffn/layers/3)
		self.ffn = nn.Sequential(
			_flax_linear(embed_dim, mlp_dim),
			nn.GELU(approximate='tanh'),
			nn.Dropout(p=dropout_rate),
			_flax_linear(mlp_dim, embed_dim),
			nn.Dropout(p=dropout_rate),
		)

		# layer normalization
		self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
		self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
		self.dropout = nn.Dropout(p=dropout_rate)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# Self-attention block
		residual = x
		x = self.norm1(x)
		x = self.attention(x)
		x = self.dropout(x)
		x = residual + x

		# Feed-forward block
		residual = x
		x = self.norm2(x)
		x = self.ffn(x)
		x = residual + x

		return x


def _median(x: torch.Tensor) -> torch.Tensor:
	"""jnp.median along the last axis: mean of the two middle order statistics for an even count."""
	k = x.shape[-1]
	x_sorted = torch.sort(x, dim=-1).values
	return 0.5 * (x_sorted[..., (k - 1) // 2] + x_sorted[..., k // 2])


class InputFeatures(nn.Module):
	"""
	Turn a neighborhood (point positions & their normals) to the input features to be fed into the neural network.
	"""

	def __init__(self, output_dim: int, n_freq: int = 0):
		super().__init__()
		# Input features: local coordinates (3), normals in local coordinates (3)
		input_dim = 6

		self.n_freq = n_freq
		if n_freq > 0:
			pos_encoding_dim = 3 * (2 * n_freq + 1)
			input_dim = pos_encoding_dim + 3

		# A linear projection simply to match transformer dimension
		self.mlp = nn.Sequential(
			_flax_linear(input_dim, output_dim),
			nn.LayerNorm(output_dim, eps=1e-6),  # LayerNorm seems to really help here
		)

		# (the original also stores an unused jax.random.PRNGKey(57) here)

	def forward(
		self,
		positions: torch.Tensor,  # [3, batch, k]
		normals: torch.Tensor,  # [3, batch, k]
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Compute geometric features relative to first point (center).

		Returns:
			[batch, k, output_dim] geometric encodings, [batch] scale (median distance)
		"""
		norms = norm(normals)[None]  # [1, batch, k]
		normalized_normals = normals / torch.clamp_min(norms, NAN_EPSILON)  # [3, batch, k]

		# Center point (index 0)
		center_normal = normalized_normals[:, :, 0:1]  # [3, batch, 1]

		# Regularization: scale neighborhood according to median_j |p_i - p_j|
		diff_vectors = positions - positions[:, :, 0:1]  # [3, batch, k]
		distances = torch.sqrt(torch.sum(diff_vectors * diff_vectors, dim=0))  # [batch, k]
		median = _median(distances)  # [batch, ]
		normalized_positions = positions / median[None, :, None]  # [3, batch, k]

		# The following block of code is copied from local_coordinates() above, because we also need the orthonormal basis.
		center_point = normalized_positions[:, :, 0:1]  # [3, batch, 1]
		s_hat, t_hat = orthonormal_basis_with_rotation(center_normal, 0.0)

		# Encode relative positions in local frame as (s, t, Q(s, t)) tuples.
		diff_vectors = normalized_positions - center_point  # [3, batch, k]
		heights = dot(diff_vectors, center_normal)  # [batch, k]
		S = dot(diff_vectors, s_hat)  # [batch, k]
		T = dot(diff_vectors, t_hat)  # [batch, k]

		# Encode normals in local frame.
		S_n = dot(normalized_normals, s_hat)  # [batch, k]
		T_n = dot(normalized_normals, t_hat)  # [batch, k]
		N_n = dot(normalized_normals, center_normal)  # [batch, k]

		# [local coordinates (3), normals in local coordinates (3)]
		geometric_features = torch.stack([S, T, heights, S_n, T_n, N_n], dim=-1)  # [batch, k, input_dim]

		return self.mlp(geometric_features), median


class FundamentalFormPredictor(nn.Module):
	def __init__(
		self,
		embed_dim: int = 128,
		n_freq: int = 0,
		n_layers: int = 4,
		n_heads: int = 4,
		mlp_dim: int = 256,
		dropout_rate: float = 0.0,
	):
		"""
		Args: Parameters for each TransformerBlock.
			n_layers: Number of TransformerBlocks.
		"""
		super().__init__()
		self.hyperparameters = dict(
			embed_dim=embed_dim,
			n_freq=n_freq,
			n_layers=n_layers,
			n_heads=n_heads,
			mlp_dim=mlp_dim,
			dropout_rate=dropout_rate,
		)

		self.input_encoder = InputFeatures(embed_dim, n_freq=n_freq)

		self.blocks = nn.ModuleList(
			[TransformerBlock(embed_dim, n_heads, mlp_dim, dropout_rate) for _ in range(n_layers)]
		)

		self.final_norm = nn.LayerNorm(embed_dim, eps=1e-6)

		# Projection
		# (a00, a01, a10, a11, a02, a20)
		self.mlp = nn.Sequential(nn.Linear(embed_dim, 6))
		with torch.no_grad():
			nn.init.zeros_(self.mlp[0].weight)
			self.mlp[0].bias.copy_(torch.tensor([0.0, 0.0, 0.0, 0.0, -0.5, -0.5]))

	def forward(self, positions: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			positions, normals: [3, batch, k] representing a neighborhood of size k; the first point (index 0) is the center

		Returns:
			coefficients: [6, batch] (a00, a01, a10, a11, a02, a20)
		"""
		# input_features: [batch, k, embed_dim]
		# scale: [batch, ]
		input_features, scale = self.input_encoder(positions, normals)
		x = input_features

		# Apply transformer blocks
		for block in self.blocks:
			x = block(x)

		x = self.final_norm(x)  # [batch, k, embed_dim]

		# Extract center point
		x = x[:, 0, :]  # [batch, embed_dim]

		# Predict form coefficients, directly as [6, batch]: W xᵀ + b (same parameters as self.mlp)
		projection = self.mlp[0]
		coeffs = torch.addmm(projection.bias[:, None], projection.weight, x.T)  # [6, batch]

		# Scale back to original space
		coeffs = torch.stack(
			[
				coeffs[0] * scale,  # a00
				coeffs[1],  # a01
				coeffs[2],  # a10
				coeffs[3] / scale,  # a11
				coeffs[4] / scale,  # a02
				coeffs[5] / scale,  # a20
			],
			dim=0,
		)

		return coeffs

	def _network_dtype(self) -> torch.dtype:
		return self.final_norm.weight.dtype

	@torch.inference_mode()
	def precompute_coefficients(self, points, normals, k_neighbors, outliers=None):
		"""
		Args: points, normals (3, |P|) on the configured device
		Returns: (6, |P|) coefficients (geometry dtype)
		"""
		self.eval()
		# Generate neighborhoods
		indices = get_neighbors(points, k_neighbors, outliers)  # (|P|, k)

		# Run forward evaluation.
		dt = self._network_dtype()
		predicted_coefficients = self(points[:, indices].to(dt), normals[:, indices].to(dt))  # (6, |P|)

		return predicted_coefficients.to(points.dtype)

	@torch.inference_mode()
	def precompute_coefficients_in_chunks(self, points, normals, k_neighbors, chunk_size=50000, outliers=None):
		"""
		Process in chunks to avoid GPU memory issues.
		Results stay on the device (the original copied every chunk to host memory).

		Args: points, normals (3, |P|) on the configured device
		Returns: (6, |P|) coefficients (geometry dtype)
		"""
		self.eval()

		n_points = points.shape[1]
		print(f'Processing {n_points} points in chunks of {chunk_size}...')

		# Generate patches
		# (timings measure dispatch on asynchronous devices: no synchronization is inserted)
		t0 = time.time()
		indices = get_neighbors(points, k_neighbors, outliers)  # (|P|, k)
		t1 = time.time()
		print(f'  get_neighbors: {t1 - t0:.2f}s')

		# Allocate output array
		all_coefficients = torch.empty((6, n_points), dtype=points.dtype, device=points.device)
		dt = self._network_dtype()

		# Process in chunks
		n_chunks = (n_points + chunk_size - 1) // chunk_size
		t0 = time.time()

		for chunk_idx in range(n_chunks):
			start_idx = chunk_idx * chunk_size
			end_idx = min(start_idx + chunk_size, n_points)

			# Extract chunk
			chunk_indices = indices[start_idx:end_idx]  # (chunk_size, k)
			chunk_points = points[:, chunk_indices]  # (3, chunk_size, k)
			chunk_normals = normals[:, chunk_indices]  # (3, chunk_size, k)

			# Forward pass for this chunk
			all_coefficients[:, start_idx:end_idx] = self(chunk_points.to(dt), chunk_normals.to(dt))

			if (chunk_idx + 1) % 10 == 0:
				print(f'	{chunk_idx + 1}/{n_chunks} chunks processed')

		t1 = time.time()
		print(f'  Coefficient prediction (chunked): {t1 - t0:.2f}s')

		return all_coefficients

	def save_model(self, filepath: str, k_neighbors: int) -> None:
		"""Added in the port: save weights + hyperparameters as a torch checkpoint (loadable without JAX)."""
		state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
		torch.save({'state_dict': state, 'k_neighbors': int(k_neighbors), **self.hyperparameters}, filepath)

	@staticmethod
	def load_saved_model(filepath: str):
		"""
		Given the filepath where weights are saved, return the model and its parameters.

		Accepts a torch checkpoint (.pt, written by save_model / convert_checkpoint) or the original flax
		checkpoint (.pkl; converting it needs jax + flax installed). The model is created on the configured device.
		"""
		device = config.get_device()
		if filepath.endswith('.pkl'):
			saved_data = convert_flax_checkpoint(filepath)
		else:
			saved_data = torch.load(filepath, map_location=device, weights_only=True)

		n_freq = saved_data['n_freq'] if 'n_freq' in saved_data else 0

		with torch.device(device):
			loaded_model = FundamentalFormPredictor(
				embed_dim=saved_data['embed_dim'],
				n_freq=n_freq,
				n_heads=saved_data['n_heads'],
				n_layers=saved_data['n_layers'],
				mlp_dim=saved_data['mlp_dim'],
				dropout_rate=saved_data['dropout_rate'],  # not used in eval mode
			)

		# Load saved parameters into the model
		loaded_model.load_state_dict(saved_data['state_dict'])

		# Put in eval mode
		loaded_model.eval()

		print('Loaded FundamentalFormPredictor model:')
		print(f'  embed_dim: {saved_data["embed_dim"]}')
		print(f'  n_freq: {n_freq}')
		print(f'  n_heads: {saved_data["n_heads"]}')
		print(f'  n_layers: {saved_data["n_layers"]}')
		print(f'  mlp_dim: {saved_data["mlp_dim"]}')
		print(f'  dropout_rate: {saved_data["dropout_rate"]}')

		return loaded_model, saved_data['k_neighbors']


# ========================================================================
# FLAX CHECKPOINT CONVERSION
# ========================================================================


def convert_flax_params(flat_params: dict) -> dict:
	"""
	Map flat flax.nnx parameter paths ('blocks/0/attention/query/kernel', ...) to the torch state_dict.

		Linear kernel (in, out)                 -> weight (out, in)
		attention q/k/v kernel (in, H, D)       -> weight (H*D, in);  bias (H, D) -> (H*D,)
		attention out kernel (H, D, out)        -> weight (out, H*D)
		LayerNorm scale                         -> weight
		Sequential 'layers/i'                   -> 'i'
	"""
	state_dict = {}
	for path, value in flat_params.items():
		value = torch.as_tensor(np.array(value))
		parts = [p for p in path.split('/') if p != 'layers']
		leaf = parts[-1]
		module = '.'.join(parts[:-1])
		if leaf == 'kernel':
			if value.ndim == 2:
				value = value.T
			elif value.ndim == 3 and parts[-2] in ('query', 'key', 'value'):
				value = value.reshape(value.shape[0], -1).T
			elif value.ndim == 3 and parts[-2] == 'out':
				value = value.reshape(-1, value.shape[-1]).T
			else:
				raise ValueError(f'unexpected kernel {path} with shape {tuple(value.shape)}')
			leaf = 'weight'
		elif leaf == 'bias':
			value = value.reshape(-1)
		elif leaf == 'scale':
			leaf = 'weight'
		state_dict[f'{module}.{leaf}'] = value.contiguous()
	return state_dict


def convert_flax_checkpoint(filepath: str) -> dict:
	"""Read the original flax pickle (needs jax + flax) and return {'state_dict': ..., hyperparameters...}."""
	try:
		from flax import nnx
	except ImportError as e:
		raise ImportError('converting a flax .pkl checkpoint needs jax and flax; or load a converted .pt') from e

	with open(filepath, 'rb') as f:
		saved_data = pickle.load(f)

	flat = {}
	for path, variable in nnx.to_flat_state(saved_data['params']):
		value = variable.get_value() if hasattr(variable, 'get_value') else variable.value
		flat['/'.join(str(p) for p in path)] = np.asarray(value)

	converted = {k: v for k, v in saved_data.items() if k != 'params'}
	converted['state_dict'] = convert_flax_params(flat)
	return converted
