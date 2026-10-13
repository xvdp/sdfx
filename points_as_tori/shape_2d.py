"""
Torch port of src/cpp/shape_2d.h / shape_2d.cpp: shared geometric utilities (sgn, sign, frames, Fibonacci sphere,
farthest-point subsampling) and the 2D SDF primitives with their 3D revolutions / extrusions.

Layout: vectors are structure-of-arrays, (3, ...) / (2, ...) with the component axis first (see config.py).
Random draws that the C++ takes from std::mt19937 are replayed with points_as_tori.cpp_random (same seed, same
values); they are generated on the host and moved to the device once.
"""

from __future__ import annotations

import math
import sys
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from . import config
from . import cpp_random


def sgn(val: torch.Tensor) -> torch.Tensor:
	"""(T(0) < val) - (val < T(0)); can output 0 (and 0 for NaN, like the C++ template)."""
	return (val > 0).to(val.dtype) - (val < 0).to(val.dtype)


def sign(x: torch.Tensor) -> torch.Tensor:
	"""x >= 0.0 ? 1.0 : -1.0; force to pick a side (NaN -> -1, like the C++ function)."""
	return torch.where(x >= 0.0, 1.0, -1.0).to(x.dtype)


def cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	"""Cross product along the leading component axis of (3, ...) tensors (broadcasting over the trailing axes)."""
	ax, ay, az = a[0], a[1], a[2]
	bx, by, bz = b[0], b[1], b[2]
	return torch.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], dim=0)


def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	"""Dot product along the leading component axis: (c, ...) x (c, ...) -> (...)."""
	return (a * b).sum(dim=0)


def norm(a: torch.Tensor) -> torch.Tensor:
	"""Euclidean norm along the leading component axis: (c, ...) -> (...)."""
	return torch.sqrt((a * a).sum(dim=0))


# `rotation` is in [0, 2π], and controls the rotation of the local x-axis around the local z-axis, using an arbitrary
# reference vector.
def orthonormal_basis(n: torch.Tensor, rotation: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Args:
		n: (3, ...) normals (need not be unit length)
		rotation: scalar rotation angle about n

	Returns:
		s, t: (3, ...) such that (s, t, n_hat) is a right-handed orthonormal frame

	C++ semantics (no epsilon guards): n_hat = n / |n|; u = (0,1,0) if |n_hat.x| > 0.9 else (1,0,0);
	u <- AngleAxis(rotation, n_hat) u   (Rodrigues: u cosθ + (n̂ × u) sinθ + n̂ (n̂·u)(1 - cosθ));
	s = n̂ × u / |n̂ × u|;  t = n̂ × s.
	"""
	n_hat = n / norm(n)

	use_y = torch.abs(n_hat[0]) > 0.9
	zero = torch.zeros_like(n_hat[0])
	one = torch.ones_like(n_hat[0])
	u = torch.stack([torch.where(use_y, zero, one), torch.where(use_y, one, zero), zero], dim=0)

	if rotation != 0.0:
		cos_rot = torch.cos(torch.as_tensor(rotation, dtype=n.dtype, device=n.device))
		sin_rot = torch.sin(torch.as_tensor(rotation, dtype=n.dtype, device=n.device))
		u = cos_rot * u + sin_rot * cross(n_hat, u) + (1.0 - cos_rot) * dot(n_hat, u) * n_hat

	s = cross(n_hat, u)
	s = s / norm(s)
	t = cross(n_hat, s)
	return s, t


def orthonormal_basis_random(n: torch.Tensor, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	(3,) normal -> (s, t), s a random unit vector orthogonal to n̂ (Gram-Schmidt of three N(0, 1) draws of
	std::mt19937(seed)), t = n̂ × s. The C++ builds `Eigen::Vector3d s(dist(rng), dist(rng), dist(rng))`, whose
	arguments g++ evaluates right to left: s = (z3, z2, z1).
	"""
	rng = cpp_random.MT19937(seed)
	dist = cpp_random.Normal(0.0, 1.0)
	n_hat = n / norm(n)
	s = torch.as_tensor(dist(rng, 3)[::-1].copy(), dtype=n.dtype, device=n.device)
	s = s - dot(s, n_hat) * n_hat
	s = s / norm(s)
	return s, cross(n_hat, s)


def sample_sphere_positions(n_samples: int, rng: cpp_random.MT19937, dtype=None, device=None) -> torch.Tensor:
	"""
	(3, n) evenly spaced points on the unit sphere (Fibonacci sampling), rotated by a uniformly random rotation:
		φ_i = acos(1 - 2i/n), θ_i = 2πi / golden ratio,  p_i = (sin φ cos θ, sin φ sin θ, cos φ)
		quaternion (w, x, y, z) = (√(1-u1) sin 2πu2, √(1-u1) cos 2πu2, √u1 sin 2πu3, √u1 cos 2πu3)   (Shoemake 1992)
	`rng` is consumed like the C++ std::mt19937& (three uniform draws).
	"""
	dtype = config.get_dtype() if dtype is None else dtype
	device = config.get_device() if device is None else device
	i = torch.arange(n_samples, dtype=torch.float64)
	golden_ratio = 0.5 * (math.sqrt(5) + 1.0)
	phi = torch.acos(1.0 - 2.0 * i / n_samples)
	theta = 2.0 * i * math.pi / golden_ratio
	p = torch.stack([torch.sin(phi) * torch.cos(theta), torch.sin(phi) * torch.sin(theta), torch.cos(phi)], dim=0)

	u1, u2, u3 = cpp_random.UniformReal(0.0, 1.0)(rng, 3)
	w = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
	x = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
	y = math.sqrt(u1) * math.sin(2 * math.pi * u3)
	z = math.sqrt(u1) * math.cos(2 * math.pi * u3)
	# Eigen::Quaterniond(w, x, y, z).toRotationMatrix() (unit quaternion)
	R = torch.tensor(
		[
			[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
			[2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
			[2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
		],
		dtype=torch.float64,
	)
	return (R @ p).to(dtype=dtype, device=device).contiguous()


def point_cloud_bounding_box_2d(positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
	"""(2, n) -> (bbox_min, bbox_max), each (2,)."""
	return positions.amin(dim=1), positions.amax(dim=1)


def subsample_point_cloud(positions: torch.Tensor, n_points: int) -> torch.Tensor:
	"""
	Farthest-point sampling: indices (n_points,) of a subset of the (3, n) positions. Starts at index 0; each step
	adds argmax_j min_{s ∈ selected} |p_j - p_s|² (first index on ties, as the C++ strict comparison). If
	n ≤ n_points, all indices. The selection loop runs n_points steps on the device (no host reads).
	"""
	n_input = positions.shape[1]
	device = positions.device
	if n_input <= n_points:
		return torch.arange(n_input, device=device)
	selected = torch.zeros(n_points, dtype=torch.long, device=device)
	min_distances = torch.full((n_input,), float('inf'), dtype=positions.dtype, device=device)
	last = torch.zeros((), dtype=torch.long, device=device)
	for i in range(1, n_points):
		d = positions - positions[:, last][:, None]
		min_distances = torch.minimum(min_distances, (d * d).sum(dim=0))
		last = torch.argmax(min_distances)
		selected[i] = last
		min_distances[last] = 0.0  # mark as selected
	return selected


# ============================================================================
# 2D SDFs: class SDF2D and its subclasses (shape_2d.h / shape_2d.cpp)
# ============================================================================

# ----------------------------------------------------------------------------
# Utility functions (elementwise versions of the inline helpers of shape_2d.h)
# ----------------------------------------------------------------------------


def _std_min(a, b):
	"""std::min(a, b) = (b < a) ? b : a, elementwise (one of a, b may be a Python float)."""
	return torch.where(b < a, b, a)


def _std_max(a, b):
	"""std::max(a, b) = (a < b) ? b : a, elementwise."""
	return torch.where(a < b, b, a)


def clamp(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
	"""std::max(lo, std::min(hi, x)), elementwise (same NaN behavior as the C++ inline clamp)."""
	return _std_max(lo, _std_min(hi, x))


def _fmin(a: float, b: float) -> float:
	"""std::min on host doubles."""
	return b if b < a else a


def _fmax(a: float, b: float) -> float:
	"""std::max on host doubles."""
	return b if a < b else a


def _sign_host(x: float) -> float:
	return 1.0 if x >= 0.0 else -1.0


def _sqnorm(v: torch.Tensor) -> torch.Tensor:
	"""Eigen squaredNorm() of (2, ...) or (3, ...) vectors, summed in Eigen's order ((x² + y²) + z²)."""
	s = v[0] * v[0] + v[1] * v[1]
	if v.shape[0] == 3:
		s = s + v[2] * v[2]
	return s


def _normalized(v: torch.Tensor) -> torch.Tensor:
	"""Eigen normalized(): v / |v| when |v|² > 0, otherwise v unchanged (zero or NaN vectors are returned as is)."""
	z = _sqnorm(v)
	return torch.where(z > 0.0, v / torch.sqrt(z), v)


def _wide(q, n_components: int, name: str) -> Tuple[torch.Tensor, bool]:
	"""(c,) single vector or (c, n) batch on the configured device -> ((c, n) tensor, single flag). No copy."""
	single = isinstance(q, torch.Tensor) and q.ndim == 1
	return config.check_wide(q, n_components, name), single


def _host_vector(v, n: int) -> Tuple[float, ...]:
	"""Constructor / sampler argument (float sequence, NumPy array or tensor) -> tuple of n host doubles."""
	if isinstance(v, torch.Tensor):
		v = v.detach().cpu().numpy()  # import boundary (parameters are host doubles in the C++)
	v = np.asarray(v, dtype=np.float64).reshape(-1)
	if v.size != n:
		raise ValueError(f'expected {n} values, got {v.size}')
	return tuple(float(x) for x in v)


def _to_device(x: np.ndarray) -> torch.Tensor:
	"""Host float64 array -> configured device / dtype (the one host -> device transfer of a sampler stage)."""
	return torch.as_tensor(np.ascontiguousarray(x), dtype=config.get_dtype()).to(config.get_device())


# Autodiff-safe min/max helpers, as in shape_2d.h: the branch is chosen by value and only the chosen branch is
# differentiated (torch.where routes the gradient to the selected operand).
def ad_min(a, b):
	"""(a.value() < b.value()) ? a : b"""
	return torch.where(a < b, a, b)


def ad_max(a, b):
	"""(a.value() > b.value()) ? a : b"""
	return torch.where(a > b, a, b)


def ad_abs(a: torch.Tensor) -> torch.Tensor:
	"""If the value is negative, return -a (which correctly computes the derivative as -a.derivatives())."""
	return torch.where(a < 0.0, -a, a)


def _eigen_ad_max(a, b):
	"""Eigen's (max)(AutoDiffScalar, AutoDiffScalar) = (x.value() >= y.value() ? x : y), used by some templates."""
	return torch.where(a >= b, a, b)


# ----------------------------------------------------------------------------
# Random-number replay helpers for the samplers (host side)
# ----------------------------------------------------------------------------


def _log_glibc(x: np.ndarray) -> np.ndarray:
	"""Natural log with the C library (std::log); NumPy's SIMD log differs from it by 1 ulp for some inputs."""
	return np.fromiter(map(math.log, x.tolist()), dtype=np.float64, count=x.size)


def _polar_pairs(rng: cpp_random.MT19937, count: int) -> Tuple[np.ndarray, np.ndarray]:
	"""
	The next `count` accepted candidates of the Marsaglia polar method of libstdc++'s normal_distribution, taken in
	order from `rng`: x, y = 2u - 1 (two canonical draws each), rejected unless 0 < r² = x² + y² ≤ 1;
	m = √(-2 ln r² / r²). Returns (y·m, x·m) (the value returned now, the value saved for the next call).
	"""
	if count == 0:
		return np.empty(0), np.empty(0)
	m = int(count / 0.785 * 1.05) + 64  # candidates to examine (acceptance π/4)
	while True:
		raw = rng.peek(4 * m).astype(np.float64).reshape(m, 4)
		x = 2.0 * cpp_random._canonical(raw[:, 0], raw[:, 1]) - 1.0
		y = 2.0 * cpp_random._canonical(raw[:, 2], raw[:, 3]) - 1.0
		r2 = x * x + y * y
		ok = np.flatnonzero((r2 <= 1.0) & (r2 != 0.0))
		if ok.size >= count:
			break
		m *= 2
	use = ok[:count]
	rng.take(4 * (int(use[-1]) + 1))
	r2 = r2[use]
	mult = np.sqrt(-2.0 * _log_glibc(r2) / r2)
	return y[use] * mult, x[use] * mult


def _normal_draws(rng: cpp_random.MT19937, dists: Sequence[cpp_random.Normal], labels: np.ndarray) -> np.ndarray:
	"""
	Values of a program-ordered sequence of std::normal_distribution calls that share one std::mt19937:
	labels[j] = index (into `dists`) of the distribution object making the j-th call. Vectorized replay: each object
	alternates between generating a polar pair (returns y·m, saves x·m) and returning its saved value, so the calls
	that generate are known in advance and the k-th of them (in program order) takes the k-th accepted candidate of
	the stream. The objects' saved values are used and updated (cpp_random.Normal._saved, unscaled as in libstdc++).
	Returns (len(labels),) float64 values, value · stddev + mean of each call's object.
	"""
	labels = np.asarray(labels, dtype=np.int64)
	out = np.empty(labels.size)
	generates = np.zeros(labels.size, dtype=bool)
	per_dist = []
	for k, dist in enumerate(dists):
		pos = np.flatnonzero(labels == k)
		if pos.size > 0 and dist._saved is not None:
			out[pos[0]] = dist._saved
			dist._saved = None
			pos = pos[1:]
		generates[pos[0::2]] = True
		per_dist.append((pos[0::2], pos[1::2]))
	order = np.flatnonzero(generates)
	y_m, x_m = _polar_pairs(rng, order.size)
	rank = np.zeros(labels.size, dtype=np.int64)
	rank[order] = np.arange(order.size)
	out[order] = y_m
	for k, (gen_pos, saved_pos) in enumerate(per_dist):
		out[saved_pos] = x_m[rank[gen_pos[: saved_pos.size]]]
		if gen_pos.size > saved_pos.size:
			dists[k]._saved = float(x_m[rank[gen_pos[-1]]])
	stddev = np.array([d.stddev for d in dists] or [1.0])[labels]
	mean = np.array([d.mean for d in dists] or [0.0])[labels]
	return out * stddev + mean


def _cross_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
	return np.stack([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _norm_np(a: np.ndarray) -> np.ndarray:
	return np.sqrt((a[0] * a[0] + a[1] * a[1]) + a[2] * a[2])


def _orthonormal_basis_np(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""orthonormal_basis(n, 0) on (3, m) float64 host arrays (the rotation by 0 is the identity)."""
	n_hat = n / _norm_np(n)
	use_y = np.abs(n_hat[0]) > 0.9
	zero = np.zeros_like(n_hat[0])
	u = np.stack([np.where(use_y, 0.0, 1.0), np.where(use_y, 1.0, 0.0), zero])
	s = _cross_np(n_hat, u)
	s = s / _norm_np(s)
	return s, _cross_np(n_hat, s)


def _orthonormal_basis_random_np(n: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
	"""
	orthonormal_basis_random(n, seed) for (3, m) host directions, as compiled by g++: in
	`Eigen::Vector3d s(dist(rng), dist(rng), dist(rng))` the arguments are evaluated right to left, so the first N(0, 1)
	draw of std::mt19937(seed) is s.z (verified against the reference build). The same s is used for every n.
	(shape_2d.orthonormal_basis_random takes the draws left to right.)
	"""
	v = cpp_random.Normal(0.0, 1.0)(cpp_random.MT19937(seed), 3)
	s0 = np.array([v[2], v[1], v[0]])[:, None]
	n_hat = n / _norm_np(n)
	s = s0 - ((s0[0] * n_hat[0] + s0[1] * n_hat[1]) + s0[2] * n_hat[2]) * n_hat
	s = s / _norm_np(s)
	return s, _cross_np(n_hat, s)


def _sampler_rays(u: np.ndarray, center: Tuple[float, ...], diameter: float) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Rays of SDF2D::sample_revolution / sample_extrusion from (n, 4) uniform draws (u0..u3 per attempt), on the host
	in float64 with the C++ operation order:
		direction  d = (cos θ sin φ, sin θ sin φ, cos φ),  θ = 2π u0,  cos φ = 2 u1 - 1      (uniform on the sphere)
		origin     c - 0.6 D d + (√u3 D / 2)(cos θ' e1 + sin θ' e2),  θ' = 2π u2   (uniform on a disk of radius D/2
		           perpendicular to d, centered 0.6 D behind the bounding-box center c; D = bounding-box diagonal)
	"""
	theta = 2.0 * math.pi * u[:, 0]
	cos_phi = 2.0 * u[:, 1] - 1.0
	cos_theta = np.cos(theta)
	sin_theta = np.sin(theta)
	sin_phi = np.sqrt(1.0 - cos_phi * cos_phi)
	ray_dir = np.stack([cos_theta * sin_phi, sin_theta * sin_phi, cos_phi])
	theta_disk = 2.0 * math.pi * u[:, 2]
	r = np.sqrt(u[:, 3])
	onb_u, onb_v = _orthonormal_basis_np(ray_dir)
	center = np.asarray(center, dtype=np.float64)[:, None]
	ray_origin = (center - ray_dir * 0.6 * diameter) + (r * diameter * 0.5) * (
		np.cos(theta_disk) * onb_u + np.sin(theta_disk) * onb_v
	)
	return ray_origin, ray_dir


def _norm3_host(v: Sequence[float]) -> float:
	return math.sqrt((v[0] * v[0] + v[1] * v[1]) + v[2] * v[2])


# Number of sphere-tracing steps run on the device between two host checks in the 3D ray intersections.
_MARCH_BLOCK = 32


# ----------------------------------------------------------------------------
# Base SDF class
# ----------------------------------------------------------------------------


class SDF2D:
	"""
	Base class of the 2D signed distance functions φ: R² -> R (negative inside), their surfaces of revolution and
	extrusions in 3D, sphere tracing and the surface samplers used to generate training data.

	Port conventions:
	* Queries are wide tensors on the configured device: a single point (2,) or a batch (2, n) (3D: (3,) / (3, n)).
	  Every "single" method of the C++ also accepts a batch, and the *_batch methods are the batch forms.
	  Shape parameters are host doubles (Python floats), as in the C++.
	* The C++ branches (if / ?:) are evaluated for every query and selected with torch.where, with the same
	  comparisons, so the selected branch, and the result, are those of the C++ (std::min(a, b) = b < a ? b : a and
	  std::max(a, b) = a < b ? b : a are reproduced exactly, including ties).
	* Gradients: the analytic gradients of the C++ (evaluate_with_gradient) are ported. The Laplacian and
	  evaluate_gradient_autodiff use Eigen AutoDiff in the C++ (forward mode, second order for the Laplacian) on
	  `evaluate_autodiff_template`, a separate expression of each SDF. The port evaluates the same expression with
	  torch.autograd (reverse mode, create_graph for the Laplacian ∂²φ/∂x² + ∂²φ/∂y²). ad_min / ad_max / ad_abs pick
	  their branch by value, as in the C++, so the derivative is that of the same branch; only the rounding of the
	  derivative differs (forward vs reverse accumulation).
	* Laplacian of the original: SDF2D::evaluate_laplacian seeds the second-order AutoDiff with
	  derivatives()[i] = ADScalar(v, 2, i), i.e. a unit derivative of the first-order seed, so every input carries
	  ∂²x_k/∂x_i∂x_j = δ_ij and the chain rule adds Σ_k ∂φ/∂x_k δ_ij to the Hessian. The C++ therefore returns
	  Δφ + 2 (∂φ/∂x + ∂φ/∂y) (e.g. 2 inside a box, where Δφ = 0). The port reproduces this by default
	  (evaluate_laplacian(_batch)(..., exact=False)); exact=True returns Δφ. Circle2D overrides it with 1/‖q‖ (= Δφ).
	* No host synchronization in evaluation (distances, gradients, Laplacians, revolutions / extrusions, the single 2D
	  ray intersection): fixed iteration counts and masks. Variable-size outputs (ray_intersect_batch, the 3D ray
	  intersections and the samplers) read sizes back from the device; see each method.
	* Bounding boxes are returned as tensors on the device; the samplers use the host values (`_bounding_box`).

	3D shapes (https://iquilezles.org/articles/distfunctions/), with φ the 2D SDF:
		revolution about the y axis at distance o:   f(p) = φ(√(p_x² + p_z²) - o, p_y)
		extrusion along z with half height h:        w = (φ(p_x, p_y), |p_z| - h),
		                                             f(p) = min(max(w_x, w_y), 0) + |max(w, 0)|
	"""

	diameter = 4.0  # upper bound of diameter of a [-1,1]^2 box (C++: const float)
	epsilon = 1e-4  # epsilon used for ray-shape intersections
	max_steps = 1000  # max steps for ray intersections
	epsilon_dist = 1e-10  # epsilon used for determining whether two points are close

	def __init__(self):
		self.has_analytical_gradient = True

	# --- implemented by each shape --------------------------------------------------------------------------------

	def _evaluate_with_gradient(
		self, q: torch.Tensor, isovalue: float = 0.0, want_gradient: bool = True
	) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		"""(2, n) queries -> signed distance (n,) and analytic gradient (2, n) (None if not wanted)."""
		raise NotImplementedError

	def evaluate_autodiff_template(self, q: torch.Tensor, isovalue: float) -> torch.Tensor:
		"""The C++ evaluate_autodiff_template on (2, n) queries (differentiable torch expression)."""
		raise NotImplementedError

	def _bounding_box(self, isovalue: float = 0.0) -> Tuple[Tuple[float, float], Tuple[float, float]]:
		"""Host doubles ((min_x, min_y), (max_x, max_y))."""
		raise NotImplementedError

	# --- evaluation ------------------------------------------------------------------------------------------------

	# Evaluate signed distance and gradient at a single point
	def evaluate_with_gradient(self, q: torch.Tensor, isovalue: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
		q, single = _wide(q, 2, 'q')
		phi, grad = self._evaluate_with_gradient(q, isovalue)
		return (phi[0], grad[:, 0]) if single else (phi, grad)

	# Evaluate signed distance at a single point
	def evaluate(self, q: torch.Tensor, isovalue: float = 0.0) -> torch.Tensor:
		q, single = _wide(q, 2, 'q')
		phi = self._evaluate_with_gradient(q, isovalue, want_gradient=False)[0]
		return phi[0] if single else phi

	# Evaluate signed distance gradient at a single point
	def evaluate_gradient(self, q: torch.Tensor) -> torch.Tensor:
		q, single = _wide(q, 2, 'q')
		grad = self._evaluate_with_gradient(q, 0.0)[1]
		return grad[:, 0] if single else grad

	def evaluate_gradient_batch(self, queries: torch.Tensor) -> torch.Tensor:
		"""(2, n) -> (2, n) analytic gradients."""
		return self._evaluate_with_gradient(config.check_wide(queries, 2, 'queries'), 0.0)[1]

	# Batch evaluate signed distance at multiple points
	def evaluate_batch(self, queries: torch.Tensor, isovalue: float = 0.0) -> torch.Tensor:
		return self._evaluate_with_gradient(config.check_wide(queries, 2, 'queries'), isovalue, False)[0]

	# Batch evaluate signed distance and gradient
	def evaluate_with_gradient_batch(
		self, queries: torch.Tensor, isovalue: float = 0.0
	) -> Tuple[torch.Tensor, torch.Tensor]:
		return self._evaluate_with_gradient(config.check_wide(queries, 2, 'queries'), isovalue)

	def _autodiff(self, q: torch.Tensor, isovalue: float, laplacian: bool, exact: bool = False) -> torch.Tensor:
		"""
		Gradient (2, n) or Laplacian (n,) of evaluate_autodiff_template by torch.autograd (see the class notes).
		Laplacian: Δφ if exact, else Δφ + 2 (∂φ/∂x + ∂φ/∂y) as the C++ seeding gives.
		"""
		with torch.enable_grad():
			x = q.detach().clone().requires_grad_(True)
			phi = self.evaluate_autodiff_template(x, isovalue)
			(g,) = torch.autograd.grad(phi.sum(), x, create_graph=laplacian, allow_unused=True)
			if g is None:
				g = torch.zeros_like(x)
			if not laplacian:
				return g.detach()
			lap = torch.zeros_like(x[0])
			if g.requires_grad:  # otherwise the template is affine on its whole domain
				# the queries are independent, so ∂/∂x_j of Σ_i ∂φ_i/∂x_i is ∂²φ_j/∂x_j²
				gxx = torch.autograd.grad(g[0].sum(), x, retain_graph=True, allow_unused=True)[0]
				gyy = torch.autograd.grad(g[1].sum(), x, allow_unused=True)[0]
				if gxx is not None:
					lap = lap + gxx[0]
				if gyy is not None:
					lap = lap + gyy[1]
		g = g.detach()
		lap = lap.detach()
		return lap if exact else lap + 2.0 * (g[0] + g[1])

	def evaluate_gradient_autodiff(self, q: torch.Tensor, isovalue: float = 0.0) -> torch.Tensor:
		"""Gradient of evaluate_autodiff_template (C++: Eigen AutoDiff; here torch.autograd)."""
		q, single = _wide(q, 2, 'q')
		grad = self._autodiff(q, isovalue, laplacian=False)
		return grad[:, 0] if single else grad

	# Laplacian via autodiff
	def evaluate_laplacian(self, q: torch.Tensor, isovalue: float = 0.0, exact: bool = False) -> torch.Tensor:
		"""
		Laplacian of evaluate_autodiff_template at (2,) or (2, n) queries. exact=False (default) reproduces the
		original, Δφ + 2 (∂φ/∂x + ∂φ/∂y) (see the class notes); exact=True gives Δφ.
		"""
		q, single = _wide(q, 2, 'q')
		lap = self._autodiff(q, isovalue, laplacian=True, exact=exact)
		return lap[0] if single else lap

	def evaluate_laplacian_batch(self, queries: torch.Tensor, exact: bool = False) -> torch.Tensor:
		"""(2, n) -> (n,) Laplacians (isovalue 0); see evaluate_laplacian."""
		return self._autodiff(config.check_wide(queries, 2, 'queries'), 0.0, laplacian=True, exact=exact)

	def has_gradient(self) -> bool:
		return self.has_analytical_gradient

	# Bounding box of the isosurface
	# https://iquilezles.org/articles/bboxes2d/
	def bounding_box(self, isovalue: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
		"""(bbox_min, bbox_max), each a (2,) tensor on the device."""
		bmin, bmax = self._bounding_box(isovalue)
		kw = dict(dtype=config.get_dtype(), device=config.get_device())
		return torch.tensor(bmin, **kw), torch.tensor(bmax, **kw)

	# --- 3D: revolutions and extrusions ---------------------------------------------------------------------------

	def _evaluate_revolution(self, p: torch.Tensor, o: float) -> torch.Tensor:
		q = torch.stack([torch.sqrt(p[0] * p[0] + p[2] * p[2]) - o, p[1]])
		return self._evaluate_with_gradient(q, 0.0, want_gradient=False)[0]

	def _evaluate_extrusion(self, p: torch.Tensor, h: float) -> torch.Tensor:
		d = self._evaluate_with_gradient(p[:2], 0.0, want_gradient=False)[0]
		w_y = torch.abs(p[2]) - h
		w_max_x = _std_max(d, 0.0)
		w_max_y = _std_max(w_y, 0.0)
		return _std_min(_std_max(d, w_y), 0.0) + torch.sqrt(w_max_x * w_max_x + w_max_y * w_max_y)

	def _evaluate_revolution_gradient(self, p: torch.Tensor, o: float) -> torch.Tensor:
		"""∇f = (g_x p_x / r, g_y, g_x p_z / r), r = √(p_x² + p_z²), g = ∇φ(r - o, p_y)."""
		r = torch.sqrt(p[0] * p[0] + p[2] * p[2])
		# Query point in 2D SDF space
		grad_2d = self._evaluate_with_gradient(torch.stack([r - o, p[1]]), 0.0)[1]
		return torch.stack([grad_2d[0] * p[0] / r, grad_2d[1], grad_2d[0] * p[2] / r])

	def _evaluate_extrusion_gradient(self, p: torch.Tensor, h: float) -> torch.Tensor:
		"""
		Case analysis based on the extrusion formula, w = (φ(p_x, p_y), |p_z| - h), g = ∇φ, s = sign of p_z (+1 if > 0):
			w_x > 0 and w_y > 0 (outside both): ∇f = (g w_x, s w_y) / |w|
			else if w_x > w_y (closest to xy plane): ∇f = (g, 0)
			else (closest to z planes): ∇f = (0, 0, s)
		"""
		d, grad_2d = self._evaluate_with_gradient(p[:2], 0.0)
		w_x = d
		w_y = torch.abs(p[2]) - h
		outside = (w_x > 0.0) & (w_y > 0.0)
		closest_xy = w_x > w_y
		norm_w = torch.sqrt(w_x * w_x + w_y * w_y)
		s_z = torch.where(p[2] > 0.0, 1.0, -1.0).to(p.dtype)
		zero = torch.zeros_like(d)
		gx = torch.where(outside, grad_2d[0] * w_x / norm_w, torch.where(closest_xy, grad_2d[0], zero))
		gy = torch.where(outside, grad_2d[1] * w_x / norm_w, torch.where(closest_xy, grad_2d[1], zero))
		gz = torch.where(outside, s_z * w_y / norm_w, torch.where(closest_xy, zero, s_z))
		return torch.stack([gx, gy, gz])

	# https://iquilezles.org/articles/distfunctions/
	def evaluate_revolution_single(self, p: torch.Tensor, o: float) -> torch.Tensor:
		p, single = _wide(p, 3, 'p')
		d = self._evaluate_revolution(p, o)
		return d[0] if single else d

	def evaluate_extrusion_single(self, p: torch.Tensor, h: float) -> torch.Tensor:
		p, single = _wide(p, 3, 'p')
		d = self._evaluate_extrusion(p, h)
		return d[0] if single else d

	def evaluate_revolution_gradient_single(self, p: torch.Tensor, o: float) -> torch.Tensor:
		p, single = _wide(p, 3, 'p')
		g = self._evaluate_revolution_gradient(p, o)
		return g[:, 0] if single else g

	def evaluate_extrusion_gradient_single(self, p: torch.Tensor, h: float) -> torch.Tensor:
		p, single = _wide(p, 3, 'p')
		g = self._evaluate_extrusion_gradient(p, h)
		return g[:, 0] if single else g

	def evaluate_revolution(self, queries: torch.Tensor, o: float) -> torch.Tensor:
		"""(3, n) -> (n,)"""
		return self._evaluate_revolution(config.check_wide(queries, 3, 'queries'), o)

	def evaluate_extrusion(self, queries: torch.Tensor, h: float) -> torch.Tensor:
		"""(3, n) -> (n,)"""
		return self._evaluate_extrusion(config.check_wide(queries, 3, 'queries'), h)

	def evaluate_revolution_gradient(self, queries: torch.Tensor, o: float) -> torch.Tensor:
		"""(3, n) -> (3, n)"""
		return self._evaluate_revolution_gradient(config.check_wide(queries, 3, 'queries'), o)

	def evaluate_extrusion_gradient(self, queries: torch.Tensor, h: float) -> torch.Tensor:
		"""(3, n) -> (3, n)"""
		return self._evaluate_extrusion_gradient(config.check_wide(queries, 3, 'queries'), h)

	def evaluate_signed_distance_gradient(
		self, queries: torch.Tensor, is_extrusion: bool, SDF3D_param: float
	) -> torch.Tensor:
		"""Gradient (3, n) of the extrusion (half height SDF3D_param) or the revolution (offset SDF3D_param)."""
		queries = config.check_wide(queries, 3, 'queries')
		if is_extrusion:
			return self._evaluate_extrusion_gradient(queries, SDF3D_param)
		return self._evaluate_revolution_gradient(queries, SDF3D_param)

	def _bbox_revolution(self, o: float) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
		bbox_min_2d, bbox_max_2d = self._bounding_box()
		# For revolution: rotate the 2D bbox around the y-axis at offset o
		# The rotated bbox in xz has radius = max(|bbox.x|) + o
		max_radius = _fmax(abs(bbox_min_2d[0]), abs(bbox_max_2d[0])) + o
		return (-max_radius, bbox_min_2d[1], -max_radius), (max_radius, bbox_max_2d[1], max_radius)

	def _bbox_extrusion(self, h: float) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
		bbox_min_2d, bbox_max_2d = self._bounding_box()
		return (bbox_min_2d[0], bbox_min_2d[1], -h), (bbox_max_2d[0], bbox_max_2d[1], h)

	def bbox_revolution(self, o: float) -> Tuple[torch.Tensor, torch.Tensor]:
		"""(3,) bbox_min, bbox_max of the revolution: radius max(|x_min|, |x_max|) + o in xz, the 2D y range."""
		kw = dict(dtype=config.get_dtype(), device=config.get_device())
		return tuple(torch.tensor(b, **kw) for b in self._bbox_revolution(o))

	def bbox_extrusion(self, h: float) -> Tuple[torch.Tensor, torch.Tensor]:
		"""(3,) bbox_min, bbox_max of the extrusion: the 2D box times [-h, h]."""
		kw = dict(dtype=config.get_dtype(), device=config.get_device())
		return tuple(torch.tensor(b, **kw) for b in self._bbox_extrusion(h))

	# --- ray intersections ----------------------------------------------------------------------------------------

	def _ray_march_2d(self, ros: torch.Tensor, rds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Sphere tracing of (2, n) rays: t_0 = 0, t_{k+1} = t_k + |φ(o + t_k d)|, for at most max_steps steps while
		t < diameter; a hit is the first q = o + t d with |φ(q)| < epsilon, its normal ∇φ(q) normalized.
		Fixed max_steps iterations with masks (no host sync); finished rays are frozen.
		"""
		t = torch.zeros_like(ros[0])
		done = torch.zeros(ros.shape[1], dtype=torch.bool, device=ros.device)
		hit = torch.zeros_like(done)
		for _ in range(self.max_steps):
			d_abs = torch.abs(self._evaluate_with_gradient(ros + t * rds, 0.0, want_gradient=False)[0])
			active = ~done & (t < self.diameter)
			new_hit = active & (d_abs < self.epsilon)
			hit = hit | new_hit
			done = done | ~active | new_hit
			t = torch.where(active & ~new_hit, t + d_abs, t)
		q = ros + t * rds
		normal = _normalized(self._evaluate_with_gradient(q, 0.0)[1])
		zero = torch.zeros_like(q)
		return torch.where(hit, q, zero), torch.where(hit, normal, zero), hit

	# Ray intersection using sphere tracing
	# Return position and normal at intersection point.
	def ray_intersect(self, ro: torch.Tensor, rd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		(2,) ray -> (position (2,), normal (2,), hit (bool 0-d tensor)); zeros when there is no hit.
		Also accepts (2, n) batches (then (2, n), (2, n), (n,)). No host sync (fixed max_steps loop).
		"""
		ro, single = _wide(ro, 2, 'ro')
		rd = config.check_wide(rd, 2, 'rd')
		q, normal, hit = self._ray_march_2d(ro, rd)
		return (q[:, 0], normal[:, 0], hit[0]) if single else (q, normal, hit)

	def ray_intersect_batch(self, ros: torch.Tensor, rds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		(2, n) rays -> positions, normals (2, n_hits) of the rays that hit, in ray order. One host read (the number of
		hits). The C++ collects the hits of OpenMP threads (dynamic schedule) in nondeterministic order; the order here
		is the single-threaded one.
		"""
		q, normal, hit = self._ray_march_2d(config.check_wide(ros, 2, 'ros'), config.check_wide(rds, 2, 'rds'))
		return q[:, hit], normal[:, hit]

	def _ray_intersections_3d(
		self,
		ros: torch.Tensor,
		rds: torch.Tensor,
		is_extrusion: bool,
		param: float,
		isovalue: float = 0.0,
		first_only: bool = False,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		All intersections of (3, n) rays with the extrusion / revolution, as ray_intersection_extrusion /
		ray_intersection_revolution (per ray, in the C++ order):
			t = 0, tmax = diagonal of bbox_extrusion(h) / bbox_revolution(o), ε = 1e-4
			while t < tmax: d = |f(o + t d) - isovalue|
				if d < ε: record q; then step t += max(d, ε) while d < ε and t < tmax
				t += d
		Vectorized as a per-ray state machine (outer / inner loop), one evaluation per step. The C++ loop has no step
		limit, so the port runs blocks of _MARCH_BLOCK steps and reads the active-ray mask back after each block (host
		sync) to stop and to drop finished rays. `first_only`: stop a ray at its first intersection (enough when only
		the first one is used).

		Returns (ray index (k,), positions (3, k)) sorted by ray, then along the ray.
		"""
		bbox_min, bbox_max = self._bbox_extrusion(param) if is_extrusion else self._bbox_revolution(param)
		tmax = _norm3_host([bbox_max[i] - bbox_min[i] for i in range(3)])
		epsilon = 1e-4
		if is_extrusion:

			def sdf(q):
				return self._evaluate_extrusion(q, param)
		else:

			def sdf(q):
				return self._evaluate_revolution(q, param)

		device = ros.device
		n = ros.shape[1]
		ray = torch.arange(n, device=device)
		ro, rd = ros, rds
		t = torch.zeros(n, dtype=ros.dtype, device=device)
		inner = torch.zeros(n, dtype=torch.bool, device=device)
		alive = torch.ones(n, dtype=torch.bool, device=device)
		hit_ray, hit_pos = [], []
		while ray.numel() > 0:
			masks, qs = [], []
			for _ in range(_MARCH_BLOCK):
				q = ro + t * rd
				d = torch.abs(sdf(q) - isovalue)
				close = d < epsilon
				new_hit = alive & ~inner & close
				adv_eps = new_hit | (alive & inner & close & (t < tmax))  # t += max(d, ε) = ε
				adv_d = alive & ~adv_eps  # outer step without hit, or end of the inner loop: t += d
				t = torch.where(adv_eps, t + epsilon, torch.where(adv_d, t + d, t))
				inner = torch.where(alive, adv_eps, inner)
				alive = torch.where(adv_d, t < tmax, alive)
				if first_only:
					alive = alive & ~new_hit
				masks.append(new_hit)
				qs.append(q)
			step, local = torch.nonzero(torch.stack(masks), as_tuple=True)  # host sync (step-major order)
			if step.numel() > 0:
				hit_ray.append(ray[local])
				hit_pos.append(torch.stack(qs)[step, :, local].T)
			keep = torch.nonzero(alive).squeeze(1)
			ray, ro, rd, t, inner, alive = ray[keep], ro[:, keep], rd[:, keep], t[keep], inner[keep], alive[keep]
		if not hit_ray:
			return torch.zeros(0, dtype=torch.long, device=device), torch.zeros(3, 0, dtype=ros.dtype, device=device)
		rays = torch.cat(hit_ray)
		order = torch.sort(rays, stable=True).indices  # events are in step order: sort by ray keeps that order
		return rays[order], torch.cat(hit_pos, dim=1)[:, order].contiguous()

	def ray_intersection_revolution(
		self, ro: torch.Tensor, rd: torch.Tensor, o: float, isovalue: float = 0.0
	) -> torch.Tensor:
		"""All intersections (3, k) of the ray (ro, rd) ((3,) each) with the revolution, in order along the ray."""
		ro = config.check_wide(ro, 3, 'ro')
		rd = config.check_wide(rd, 3, 'rd')
		return self._ray_intersections_3d(ro, rd, False, o, isovalue)[1]

	def ray_intersection_extrusion(
		self, ro: torch.Tensor, rd: torch.Tensor, h: float, isovalue: float = 0.0
	) -> torch.Tensor:
		"""All intersections (3, k) of the ray (ro, rd) ((3,) each) with the extrusion, in order along the ray."""
		ro = config.check_wide(ro, 3, 'ro')
		rd = config.check_wide(rd, 3, 'rd')
		return self._ray_intersections_3d(ro, rd, True, h, isovalue)[1]

	# --- samplers -------------------------------------------------------------------------------------------------
	#
	# Surface sampling (sample_revolution / sample_extrusion): rays with directions uniform on the sphere and origins
	# uniform on a disk of radius D/2 perpendicular to the direction (see _sampler_rays) are intersected with the
	# surface (all intersections along each ray); the intersections inside the bounding box are kept in ray order
	# until n_samples are collected (at most 10 n_samples rays). The rays form an isotropic uniform line field over
	# the ball of radius D/2 around the box center, and by Crofton's formula the expected number of intersections
	# with a surface patch is proportional to its area: the samples are area-uniform on the surface inside the box,
	# except where the rays do not reach (a ray starts 0.6 D before the center and marches tmax = diagonal of the
	# shape's box, so surface farther than tmax - 0.6 D past the center plane, along the ray, is missed).
	#
	# Randomness: std::mt19937(seed) is replayed with cpp_random in the C++ order (4 uniform draws per ray, then the
	# position / normal noise of each sample), so a seed gives the same samples as the C++. Rays are generated in
	# batches on the host, intersected on the device, and the attempt that completes the sample count is read back
	# (one host read per batch; normally one or two batches) so that the stream continues where the C++ continues.

	def _sample_surface(
		self,
		n_samples: int,
		is_extrusion: bool,
		param: float,
		bbox_min_,
		bbox_max_,
		sigma_p: float,
		sigma_n: float,
		seed: int,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""sample_extrusion / sample_revolution (the two C++ functions differ only in the 3D SDF)."""
		n_samples = int(n_samples)
		sigma_p = float(sigma_p)
		sigma_n = float(sigma_n)
		bbox_min_ = _host_vector(bbox_min_, 3)
		bbox_max_ = _host_vector(bbox_max_, 3)
		device = config.get_device()

		# Determine bounding box
		if bbox_max_[0] - bbox_min_[0] < 0:
			bbox_min, bbox_max = self._bbox_extrusion(param) if is_extrusion else self._bbox_revolution(param)
		else:
			bbox_min, bbox_max = bbox_min_, bbox_max_
		bbox_center = tuple(0.5 * (bbox_min[i] + bbox_max[i]) for i in range(3))
		diameter = _norm3_host([bbox_max[i] - bbox_min[i] for i in range(3)])

		# Random generator
		rng = cpp_random.MT19937(seed)
		position_perturbation = cpp_random.Normal(0.0, sigma_p)
		normal_perturbation = cpp_random.Normal(0.0, sigma_n)

		epsilon = 1e-6
		max_tries = n_samples * 10
		attempt = 0
		n_found = 0
		int_locations = []
		batch = max(64, n_samples)
		while n_found < n_samples and attempt < max_tries:
			n_batch = min(batch, max_tries - attempt)
			# Sample a direction uniformly on the sphere, and an offset uniformly from a disk perpendicular to it
			raw = rng.peek(8 * n_batch).astype(np.float64).reshape(n_batch, 4, 2)
			ray_origin, ray_dir = _sampler_rays(cpp_random._canonical(raw[..., 0], raw[..., 1]), bbox_center, diameter)
			# Perform ray intersection
			ray, p = self._ray_intersections_3d(_to_device(ray_origin), _to_device(ray_dir), is_extrusion, param, 0.0)
			# Filter intersections to only include those within bounding box
			inside = (p[0] >= bbox_min[0]) & (p[0] <= bbox_max[0]) & (p[1] >= bbox_min[1]) & (p[1] <= bbox_max[1])
			inside = inside & (p[2] >= bbox_min[2]) & (p[2] <= bbox_max[2])
			ray, p = ray[inside], p[:, inside]  # host sync (number of samples)
			if n_found + ray.numel() >= n_samples:
				need = n_samples - n_found
				used = int(ray[need - 1]) + 1  # host read: the attempt that completed the samples
				int_locations.append(p[:, :need])
				n_found = n_samples
			else:
				used = n_batch
				int_locations.append(p)
				n_found += ray.numel()
			rng.take(8 * used)
			attempt += used
			batch = int(math.ceil((n_samples - n_found) / max(n_found / attempt, 0.05) * 1.2)) + 64

		dtype = config.get_dtype()
		locations = torch.cat(int_locations, dim=1) if int_locations else torch.zeros(3, 0, dtype=dtype, device=device)
		# Handle case where we didn't get enough samples
		n_final = min(locations.shape[1], n_samples)
		if n_final < n_samples:
			if n_final == 0:
				raise RuntimeError('SDF2D sampler: no intersection with the surface (the C++ divides by zero here)')
			# Duplicate samples to reach n_samples
			extra = torch.arange(n_samples - n_final, device=device) % n_final
			locations = torch.cat([locations, locations[:, extra]], dim=1)

		# Position / normal noise: per sample, three draws of each (x, y, z) from the same stream
		labels = ([0, 0, 0] if sigma_p > 0.0 else []) + ([1, 1, 1] if sigma_n > 0.0 else [])
		if labels:
			dists = [position_perturbation, normal_perturbation]
			draws = _normal_draws(rng, dists, np.tile(labels, n_samples)).reshape(n_samples, len(labels))
		# Add position noise
		sample_locations = locations + _to_device(draws[:, :3].T) if sigma_p > 0.0 else locations
		# Compute gradient/normal
		if is_extrusion:
			grad = self._evaluate_extrusion_gradient(locations, param)
		else:
			grad = self._evaluate_revolution_gradient(locations, param)
		# Add normal noise
		if sigma_n > 0.0:
			grad = grad + _to_device(draws[:, -3:].T)
		grad_norm = torch.sqrt(_sqnorm(grad))
		sample_normals = torch.where(grad_norm > epsilon, grad / grad_norm, torch.zeros_like(grad))
		return sample_locations.contiguous(), sample_normals.contiguous()

	def sample_revolution(
		self,
		n_samples: int,
		o: float,
		bbox_min=(1.0, 1.0, 1.0),
		bbox_max=(-1.0, -1.0, -1.0),
		sigma_p: float = 0.0,
		sigma_n: float = 0.0,
		seed: int = 0,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Uniformly sample the revolved SDF: (3, n_samples) positions and unit normals (∇f / |∇f|, zero if |∇f| ≤ 1e-6;
		normals are taken at the unperturbed points). bbox_min / bbox_max: sampling box, the default (empty) box
		meaning bbox_revolution(o). sigma_p / sigma_n: std of the Gaussian noise added to positions / gradients.
		"""
		return self._sample_surface(n_samples, False, o, bbox_min, bbox_max, sigma_p, sigma_n, seed)

	def sample_extrusion(
		self,
		n_samples: int,
		h: float,
		bbox_min=(1.0, 1.0, 1.0),
		bbox_max=(-1.0, -1.0, -1.0),
		sigma_p: float = 0.0,
		sigma_n: float = 0.0,
		seed: int = 0,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Uniformly sample the extruded SDF (see sample_revolution; default box bbox_extrusion(h))."""
		return self._sample_surface(n_samples, True, h, bbox_min, bbox_max, sigma_p, sigma_n, seed)

	def _sample_narrow_band(self, n_samples, is_extrusion, param, offset, bbox_min, bbox_max, seed) -> torch.Tensor:
		samples = self._sample_surface(n_samples, is_extrusion, param, bbox_min, bbox_max, 0.0, 0.0, seed)
		# perturb in the normal direction
		rng = cpp_random.MT19937(seed)  # a new generator with the same seed, as in the C++
		random_offset = cpp_random.UniformReal(-offset, offset)(rng, int(n_samples))
		return (samples[0] + _to_device(random_offset) * _normalized(samples[1])).contiguous()

	def sample_revolution_narrow_band(
		self, n_samples: int, o: float, offset: float, bbox_min=(1.0, 1.0, 1.0), bbox_max=(-1.0, -1.0, -1.0), seed=0
	) -> torch.Tensor:
		"""
		(3, n_samples) points p + u n, (p, n) = sample_revolution(..., seed) and u ~ U(-offset, offset), drawn from a
		fresh std::mt19937(seed) (so the offsets are the first draws of the stream that also generated the rays).
		"""
		return self._sample_narrow_band(n_samples, False, o, offset, bbox_min, bbox_max, seed)

	def sample_extrusion_narrow_band(
		self, n_samples: int, h: float, offset: float, bbox_min=(1.0, 1.0, 1.0), bbox_max=(-1.0, -1.0, -1.0), seed=0
	) -> torch.Tensor:
		"""Narrow band around the extrusion (see sample_revolution_narrow_band)."""
		return self._sample_narrow_band(n_samples, True, h, offset, bbox_min, bbox_max, seed)

	# Sample revolution or extrusion
	def sample_points_and_normals_uniformly(
		self, n_points: int, is_extrusion: bool, SDF3D_param: float, seed: int
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""sample_extrusion / sample_revolution with the default box and no noise."""
		return self._sample_surface(n_points, is_extrusion, SDF3D_param, (1, 1, 1), (-1, -1, -1), 0.0, 0.0, seed)

	def sample_farthest_point_cloud(
		self, n_points: int, is_extrusion: bool, SDF3D_param: float, seed: int
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""8 n_points uniform samples, subsampled to n_points by farthest-point sampling (subsample_point_cloud)."""
		points, normals = self._sample_surface(
			int(n_points) * 8, is_extrusion, SDF3D_param, (1, 1, 1), (-1, -1, -1), 0.0, 0.0, seed
		)
		# Subsample
		indices = subsample_point_cloud(points, int(n_points))
		return points[:, indices].contiguous(), normals[:, indices].contiguous()

	def sample_uneven_point_cloud(
		self,
		is_extrusion: bool,
		SDF3D_param: float,
		n_cameras: int,
		sensor_size: float,
		grid_spacing_min: float,
		grid_spacing_max: float,
		max_noise_level: float,
		max_normals_to_flip: float,
		max_points: int,
		seed: int,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Simulated scan ("copied from Mesh3D"): n_cameras on a sphere of radius D/2 around the bounding-box center
		(Fibonacci points, random rotation) look at the center; each casts a square grid of parallel rays (side
		sensor_size·D, spacing ~ U(grid_spacing_min, grid_spacing_max) per camera; the sensor's in-plane axes come from
		orthonormal_basis_random(direction, seed)). The first hit of each ray is kept, its normal is the 3D gradient.
		Then min(#hits, max_points) hits are drawn without replacement (shuffle), positions / normals get Gaussian
		noise with σ_p, σ_n ~ U(0, max_noise_level), normals are renormalized, and int(U(0, max_normals_to_flip) ·
		max_points) of them are flipped. Returns positions, normals (3, n).

		Reproduced from the C++:
		* the box is bbox_extrusion(SDF3D_param) in both cases;
		* the ray intersection is swapped: is_extrusion = True intersects the revolution and evaluates the extrusion
		  gradient at the hit, and vice versa;
		* the noise of each sample is drawn as Eigen::Vector3d(dist(rng), dist(rng), dist(rng)), whose arguments g++
		  evaluates right to left (the first draw is z).
		Deviations: the C++ draws the noise inside an OpenMP parallel loop (a data race on the generator when more
		than one thread runs); the port reproduces the single-threaded order. When int(...) · max_points exceeds the
		number of samples, the C++ reads past the end of the index vector; the port flips all samples.
		Host work: ray generation, the shuffles and the random draws; one device -> host read of the hit count.
		"""
		device = config.get_device()
		# Copied from Mesh3D
		rng = cpp_random.MT19937(seed)

		# Determine bounding box
		bbox_min, bbox_max = self._bbox_extrusion(SDF3D_param)
		bbox_center = np.array([0.5 * (bbox_min[i] + bbox_max[i]) for i in range(3)])
		diameter = _norm3_host([bbox_max[i] - bbox_min[i] for i in range(3)])

		# Get bounding sphere.
		centroid = bbox_center

		# Generate camera positions on sphere (looking inward)
		camera_positions = sample_sphere_positions(n_cameras, rng, dtype=torch.float64, device=torch.device('cpu'))
		camera_positions = camera_positions.numpy()
		camera_directions = -camera_positions

		# Compute plane of camera sensor
		s_axes, t_axes = _orthonormal_basis_random_np(camera_directions, seed)

		# Starting positions (offset backward along view direction)
		start_positions = camera_positions * 0.5 * diameter + centroid[:, None]

		# Generate all ray origins and directions.
		sensor_side = sensor_size * diameter
		grid_spacing = cpp_random.UniformReal(grid_spacing_min, grid_spacing_max)(rng, n_cameras)
		origins = [np.zeros((3, 0))]
		directions = [np.zeros((3, 0))]
		for cam in range(n_cameras):
			n_samples_per_row = int(math.ceil(sensor_side / grid_spacing[cam]))
			half_grid_length = 0.5 * (n_samples_per_row - 1) * grid_spacing[cam]
			# Create 2D grid for each camera: grid_coords[i * n_samples_per_row + j] = (x_i, y_j)
			coords = (np.arange(n_samples_per_row) + 0.5) * grid_spacing[cam] - half_grid_length
			x = np.repeat(coords, n_samples_per_row)
			y = np.tile(coords, n_samples_per_row)
			grid_position = x * s_axes[:, cam : cam + 1] + y * t_axes[:, cam : cam + 1]
			origins.append(start_positions[:, cam : cam + 1] + grid_position)
			directions.append(np.repeat(camera_directions[:, cam : cam + 1], x.size, axis=1))
		origins = np.concatenate(origins, axis=1)
		directions = np.concatenate(directions, axis=1)

		# Cast rays; take first hit (as in the C++, is_extrusion selects the *revolution* intersection, and vice versa)
		_, ray_hits = self._ray_intersections_3d(
			_to_device(origins), _to_device(directions), not is_extrusion, SDF3D_param, 0.0, first_only=True
		)
		if is_extrusion:
			ray_normals = self._evaluate_extrusion_gradient(ray_hits, SDF3D_param)
		else:
			ray_normals = self._evaluate_revolution_gradient(ray_hits, SDF3D_param)

		n_intersections = ray_hits.shape[1]
		print(f'\tOriginal # intersections: {n_intersections}', file=sys.stderr)

		# Uniformly subsample down to target size
		n_final = min(n_intersections, int(max_points))
		indices = cpp_random.shuffle(list(range(n_intersections)), rng)

		# Add noise.
		sigma_p = cpp_random.UniformReal(0.0, max_noise_level)(rng)
		sigma_n = cpp_random.UniformReal(0.0, max_noise_level)(rng)
		normals_to_flip = cpp_random.UniformReal(0.0, max_normals_to_flip)(rng)
		n_normals_to_flip = int(normals_to_flip * max_points)
		print(f'\t Flipping {n_normals_to_flip} normals', file=sys.stderr)

		position_perturbation = cpp_random.Normal(0.0, sigma_p)
		normal_perturbation = cpp_random.Normal(0.0, sigma_n)
		draws = _normal_draws(rng, [position_perturbation, normal_perturbation], np.tile([0, 0, 0, 1, 1, 1], n_final))
		draws = draws.reshape(n_final, 6)
		position_offset = draws[:, [2, 1, 0]].T  # right-to-left argument evaluation (g++)
		normal_offset = draws[:, [5, 4, 3]].T
		chosen = torch.as_tensor(indices[:n_final], dtype=torch.long).to(device)
		int_positions = ray_hits[:, chosen] + _to_device(position_offset)
		int_normals = _normalized(ray_normals[:, chosen] + _to_device(normal_offset))

		# Flip normals
		indices_to_flip = cpp_random.shuffle(list(range(n_final)), rng)
		flip = torch.as_tensor(indices_to_flip[: min(n_normals_to_flip, n_final)], dtype=torch.long).to(device)
		int_normals[:, flip] = -int_normals[:, flip]
		return int_positions.contiguous(), int_normals.contiguous()


# ============================================================================
# Individual SDFs
# ============================================================================
# Formulas: Inigo Quilez, https://iquilezles.org/articles/distfunctions2d/ (signed distances) and
# https://iquilezles.org/articles/bboxes2d/ (bounding boxes), with the original's changes. Notation: q = query,
# p = q folded by the shape's symmetry (|q_x| and/or |q_y|), s = sign(q) (+1 for 0), ‖·‖ Euclidean norm,
# clamp(x, a, b) = max(a, min(b, x)). The isovalue is subtracted from the distance (a positive isovalue inflates).


class Circle2D(SDF2D):
	"""
	Circle SDF: distance to circle centered at origin.  φ(q) = ‖q‖ - r,  ∇φ = q / ‖q‖ ((1, 0) if ‖q‖ ≤ 1e-10),
	Δφ = 1 / ‖q‖ (closed form, as the C++ override of evaluate_laplacian(_batch)).
	"""

	def __init__(self, r: float):
		super().__init__()
		self.r_ = float(r)

	def _evaluate_with_gradient(self, q, isovalue=0.0, want_gradient=True):
		length = torch.sqrt(q[0] * q[0] + q[1] * q[1])
		phi = length - self.r_
		grad = None
		if want_gradient:
			far = length > 1e-10
			grad = torch.stack([torch.where(far, q[0] / length, 1.0), torch.where(far, q[1] / length, 0.0)])
		return phi - isovalue, grad

	def _bounding_box(self, isovalue=0.0):
		r = self.r_
		return (-r - isovalue, -r - isovalue), (r + isovalue, r + isovalue)

	def evaluate_laplacian(self, q: torch.Tensor, isovalue: float = 0.0, exact: bool = False) -> torch.Tensor:
		q, single = _wide(q, 2, 'q')
		lap = 1.0 / torch.sqrt(q[0] * q[0] + q[1] * q[1])
		return lap[0] if single else lap

	def evaluate_laplacian_batch(self, queries: torch.Tensor, exact: bool = False) -> torch.Tensor:
		q = config.check_wide(queries, 2, 'queries')
		return 1.0 / torch.sqrt(q[0] * q[0] + q[1] * q[1])

	def evaluate_autodiff_template(self, q, isovalue):
		length = torch.sqrt(q[0] * q[0] + q[1] * q[1])
		return length - self.r_ - isovalue


class Pie2D(SDF2D):
	"""
	Pie SDF: pie/pacman shape (axis-aligned), symmetric about the y axis with half aperture t, (sin t, cos t) = c:
		p = (|q_x|, q_y),  n = ‖p‖ - r,  u = clamp(p·c, 0, r),  m = ‖p - u c‖ · sign(c_y p_x - c_x p_y),
		φ = max(n, m)  (n if n > m).
	Gradient: p / ‖p‖ on the arc; the straight edge normal (c_y, -c_x) when 1e-10 < u < r - 1e-10, else
	(p - u c) / m (vertex), x component times s_x. (Updated from Inigo's formula so gradients are computed to *S*DF,
	not UDF then signed.)
	"""

	def __init__(self, r: float, t: float):
		super().__init__()
		self.r_ = float(r)
		self.sc_ = (math.sin(float(t)), math.cos(float(t)))  # sin(t), cos(t)

	def _evaluate_with_gradient(self, query, isovalue=0.0, want_gradient=True):
		sc0, sc1 = self.sc_
		r = self.r_
		s = sign(query[0])
		px = torch.abs(query[0])
		py = query[1]
		length = torch.sqrt(px * px + py * py)
		n = length - r
		t = clamp(px * sc0 + py * sc1, 0.0, r)
		qx = px - sc0 * t
		qy = py - sc1 * t
		m = torch.sqrt(qx * qx + qy * qy) * sign(sc1 * px - sc0 * py)
		use_n = n > m
		d = torch.where(use_n, n, m)
		g = None
		if want_gradient:
			# closest to straight edge: -edge_normal = (sc_y, -sc_x); vertex: division is unavoidable
			edge = (torch.abs(t) > self.epsilon_dist) & (torch.abs(t - r) > self.epsilon_dist)
			gx = torch.where(use_n, px / length, torch.where(edge, sc1, qx / m))
			gy = torch.where(use_n, py / length, torch.where(edge, -sc0, qy / m))
			g = torch.stack([gx * s, gy])
		return d - isovalue, g

	def _bounding_box(self, isovalue=0.0):
		v, u = self.sc_
		r = self.r_
		k = (r * v, 0.0) if u > 0.0 else (r, r * u)
		return (-k[0] - isovalue, k[1] - isovalue), (k[0] + isovalue, r + isovalue)

	def evaluate_autodiff_template(self, q, isovalue):
		sc0, sc1 = self.sc_
		r = self.r_
		# Apply symmetry
		p0 = ad_abs(q[0])
		p1 = q[1]
		# Compute both distances
		n = torch.sqrt(p0 * p0 + p1 * p1) - r
		# Project onto sc direction and clamp
		clamped = ad_max(0.0, p0 * sc0 + p1 * sc1)
		clamped = torch.where(clamped > r, r, clamped)
		q0 = p0 - sc0 * clamped
		q1 = p1 - sc1 * clamped
		m_dist = torch.sqrt(q0 * q0 + q1 * q1)
		# Determine sign
		m = torch.where(sc1 * p0 - sc0 * p1 >= 0.0, m_dist, -m_dist)
		# Return minimum
		return torch.where(n > m, n - isovalue, m - isovalue)


class Arc2D(SDF2D):
	"""
	Arc SDF: arc with thickness. sc = (sin a, cos a) (half aperture a), radius ra, thickness rb; p = (|q_x|, q_y):
		in the wedge (sc_y p_x > sc_x p_y): φ = ‖p - ra sc‖ - rb, ∇φ = (s_x w_x, w_y) / ‖w‖, w = p - ra sc
		otherwise:                          φ = |‖q‖ - ra| - rb,  ∇φ = sign(‖q‖ - ra) q / ‖q‖
	"""

	def __init__(self, sc, ra: float, rb: float):
		super().__init__()
		self.sc_ = _host_vector(sc, 2)  # (sin(angle), cos(angle))
		self.ra_ = float(ra)  # radius
		self.rb_ = float(rb)  # thickness

	def _evaluate_with_gradient(self, q, isovalue=0.0, want_gradient=True):
		sc0, sc1 = self.sc_
		ra, rb = self.ra_, self.rb_
		s = sign(q[0])
		px = torch.abs(q[0])
		py = q[1]
		in_wedge = sc1 * px > sc0 * py
		# Distance to arc endpoint
		w1x = px - ra * sc0
		w1y = py - ra * sc1
		d1 = torch.sqrt(w1x * w1x + w1y * w1y)
		# Distance to ring
		l2 = torch.sqrt(q[0] * q[0] + q[1] * q[1])
		w2 = l2 - ra
		phi = torch.where(in_wedge, d1 - rb, torch.abs(w2) - rb)
		grad = None
		if want_gradient:
			sw = sign(w2)
			grad = torch.stack(
				[torch.where(in_wedge, s * w1x / d1, sw * q[0] / l2), torch.where(in_wedge, w1y / d1, sw * q[1] / l2)]
			)
		return phi - isovalue, grad

	def _bounding_box(self, isovalue=0.0):
		ra, rb = self.ra_, self.rb_
		outer_radius = ra + rb
		bmin = (-outer_radius, ra * self.sc_[1] - rb)
		bmax = (outer_radius, _fmax(self.sc_[1] * outer_radius, ra + rb))
		return (bmin[0] - isovalue, bmin[1] - isovalue), (bmax[0] + isovalue, bmax[1] + isovalue)

	def evaluate_autodiff_template(self, q, isovalue):
		sc0, sc1 = self.sc_
		ra, rb = self.ra_, self.rb_
		p0 = ad_abs(q[0])
		p1 = q[1]
		# Check if in wedge
		in_wedge = sc1 * p0 > sc0 * p1
		# Distance to arc endpoint
		w0 = p0 - ra * sc0
		w1 = p1 - ra * sc1
		d1 = torch.sqrt(w0 * w0 + w1 * w1)
		# Distance to ring
		w2 = torch.sqrt(q[0] * q[0] + q[1] * q[1]) - ra
		return torch.where(in_wedge, d1 - rb - isovalue, ad_abs(w2) - rb - isovalue)


class Pill2D(SDF2D):
	"""
	Pill/Segment SDF: segment [a, b] with radius r.  h = clamp((q - a)·(b - a) / ‖b - a‖², 0, 1),
	φ = ‖q - a - h (b - a)‖ - r,  ∇φ = (q - a - h (b - a)) / max(‖·‖, 1e-10).
	"""

	def __init__(self, a, b, r: float):
		super().__init__()
		self.a_ = _host_vector(a, 2)
		self.b_ = _host_vector(b, 2)
		self.r_ = float(r)

	def _evaluate_with_gradient(self, q, isovalue=0.0, want_gradient=True):
		(a0, a1), (b0, b1) = self.a_, self.b_
		ba0, ba1 = b0 - a0, b1 - a1
		pa0 = q[0] - a0
		pa1 = q[1] - a1
		h = clamp((pa0 * ba0 + pa1 * ba1) / (ba0 * ba0 + ba1 * ba1), 0.0, 1.0)
		q0 = pa0 - h * ba0
		q1 = pa1 - h * ba1
		d = torch.sqrt(q0 * q0 + q1 * q1)
		grad = None
		if want_gradient:
			dm = _std_max(d, 1e-10)
			grad = torch.stack([q0 / dm, q1 / dm])
		return d - self.r_ - isovalue, grad

	def _bounding_box(self, isovalue=0.0):
		(a0, a1), (b0, b1) = self.a_, self.b_
		r = self.r_
		bmin = (_fmin(a0, b0) - r, _fmin(a1, b1) - r)
		bmax = (_fmax(a0, b0) + r, _fmax(a1, b1) + r)
		return (bmin[0] - isovalue, bmin[1] - isovalue), (bmax[0] + isovalue, bmax[1] + isovalue)

	def evaluate_autodiff_template(self, q, isovalue):
		(a0, a1), (b0, b1) = self.a_, self.b_
		ba0, ba1 = b0 - a0, b1 - a1
		pa0 = q[0] - a0
		pa1 = q[1] - a1
		h = (pa0 * ba0 + pa1 * ba1) / (ba0 * ba0 + ba1 * ba1)
		h = ad_max(0.0, ad_min(1.0, h))
		q0 = pa0 - h * ba0
		q1 = pa1 - h * ba1
		return torch.sqrt(q0 * q0 + q1 * q1) - self.r_ - isovalue


class Vesica2D(SDF2D):
	"""
	Vesica SDF: intersection of two discs of radius r centered at (±d, 0); b = √(r² - d²). p = |q| (componentwise):
		if (p_y - b) d > p_x b:  φ = ‖p - (0, b)‖ · sign(d)  (distance to the tip)
		else:                    φ = ‖p + (d, 0)‖ - r       (distance to the circular arc)
	∇φ = s ⊙ (p - c) / φ resp. s ⊙ (p - c) / ‖p - c‖ for the corresponding center c.
	"""

	def __init__(self, r: float, d: float):
		super().__init__()
		self.r_ = float(r)
		self.d_ = float(d)
		self.b_ = math.sqrt(self.r_ * self.r_ - self.d_ * self.d_)

	def _evaluate_with_gradient(self, q, isovalue=0.0, want_gradient=True):
		r, d, b = self.r_, self.d_, self.b_
		s0, s1 = sign(q[0]), sign(q[1])
		px, py = torch.abs(q[0]), torch.abs(q[1])
		cond = (py - b) * d > px * b
		# Case 1: distance to the tip
		q1y = py - b
		l1 = torch.sqrt(px * px + q1y * q1y) * _sign_host(d)
		# Case 2: distance to the circular arc
		q2x = px + d
		l2 = torch.sqrt(q2x * q2x + py * py)
		phi = torch.where(cond, l1, l2 - r)
		grad = None
		if want_gradient:
			grad = torch.stack(
				[torch.where(cond, s0 * px / l1, s0 * q2x / l2), torch.where(cond, s1 * q1y / l1, s1 * py / l2)]
			)
		return phi - isovalue, grad

	def _bounding_box(self, isovalue=0.0):
		# Vesica is formed by two overlapping circles of radius r_, separated by distance d_
		# b := sqrt(r^2 - d^2) = height at inner intersection
		r, d = self.r_, self.d_
		height = _fmax(self.b_, r)
		return (-r + d - isovalue, -height - isovalue), (r - d + isovalue, height + isovalue)

	def evaluate_autodiff_template(self, q, isovalue):
		r, d, b = self.r_, self.d_, self.b_
		# Apply symmetry
		p0, p1 = ad_abs(q[0]), ad_abs(q[1])
		# Check condition
		cond = (p1 - b) * d > p0 * b
		# Distance to tip
		q1y = p1 - b
		r1 = torch.sqrt(p0 * p0 + q1y * q1y) * _sign_host(d) - isovalue
		# Distance to circular arc
		q2x = p0 + d
		r2 = torch.sqrt(q2x * q2x + p1 * p1) - r - isovalue
		return torch.where(cond, r1, r2)


class Box2D(SDF2D):
	"""
	Box SDF with half-extents b.  w = |q| - b,  φ = ‖max(w, 0)‖ if max(w_x, w_y) > 0, else max(w_x, w_y).
	∇φ = s ⊙ max(w, 0) / max(‖max(w, 0)‖, 1e-10) outside; inside the normal of the nearest face, (s_x, 0) if
	w_x > w_y else (0, s_y).
	"""

	def __init__(self, b):
		super().__init__()
		self.b_ = _host_vector(b, 2)  # half-extents

	def _evaluate_with_gradient(self, q, isovalue=0.0, want_gradient=True):
		b0, b1 = self.b_
		w0 = torch.abs(q[0]) - b0
		w1 = torch.abs(q[1]) - b1
		g = _std_max(w0, w1)
		q0 = _std_max(w0, 0.0)
		q1 = _std_max(w1, 0.0)
		length = torch.sqrt(q0 * q0 + q1 * q1)
		outside = g > 0.0
		phi = torch.where(outside, length, g)
		grad = None
		if want_gradient:
			s0, s1 = sign(q[0]), sign(q[1])
			lm = _std_max(length, 1e-10)
			# Inside: gradient points toward nearest face
			face_x = w0 > w1
			zero = torch.zeros_like(s0)
			grad = torch.stack(
				[
					torch.where(outside, s0 * q0 / lm, torch.where(face_x, s0, zero)),
					torch.where(outside, s1 * q1 / lm, torch.where(face_x, zero, s1)),
				]
			)
		return phi - isovalue, grad

	def _bounding_box(self, isovalue=0.0):
		b0, b1 = self.b_
		return (-b0 - isovalue, -b1 - isovalue), (b0 + isovalue, b1 + isovalue)

	def evaluate_autodiff_template(self, q, isovalue):
		b0, b1 = self.b_
		w0 = ad_abs(q[0]) - b0
		w1 = ad_abs(q[1]) - b1
		g = _eigen_ad_max(w0, w1)
		q0 = ad_max(w0, 0.0)
		q1 = ad_max(w1, 0.0)
		outside = g > 0.0
		# sqrt(0) inside: masked argument so that the unused branch does not give 0 · ∞ = NaN derivatives
		length = torch.sqrt(torch.where(outside, q0 * q0 + q1 * q1, 1.0))
		return torch.where(outside, length, g) - isovalue


class Cross2D(SDF2D):
	"""
	Cross SDF (b = (arm half-length, arm half-width)). p = |q| sorted so that p_x ≥ p_y, w = p - b, h = max(w):
		o = max((b_y - b_x - w_x, -w_y), 0) if h < 0 else max(w, 0)
		φ = sign(h) · (‖o‖ if h ≥ 0 or -w_x ≥ ‖o‖ else -w_x)
	∇φ = s ⊙ (o / ‖o‖ or (1, 0)), with the components swapped back.
	"""

	def __init__(self, b):
		super().__init__()
		self.b_ = _host_vector(b, 2)

	def _evaluate_with_gradient(self, q, isovalue=0.0, want_gradient=True):
		b0, b1 = self.b_
		p0, p1 = torch.abs(q[0]), torch.abs(q[1])
		swap = p1 > p0
		w0 = torch.where(swap, p1, p0) - b0
		w1 = torch.where(swap, p0, p1) - b1
		h = _std_max(w0, w1)
		neg = h < 0.0
		o0 = torch.where(neg, _std_max(b1 - b0 - w0, 0.0), _std_max(w0, 0.0))
		o1 = torch.where(neg, _std_max(-w1, 0.0), _std_max(w1, 0.0))
		length = torch.sqrt(o0 * o0 + o1 * o1)
		use_corner = (h >= 0.0) | (-w0 >= length)
		phi = torch.where(use_corner, length, -w0) * sign(h)
		grad = None
		if want_gradient:
			r0 = torch.where(use_corner, o0 / length, 1.0)
			r1 = torch.where(use_corner, o1 / length, 0.0)
			grad = torch.stack([sign(q[0]) * torch.where(swap, r1, r0), sign(q[1]) * torch.where(swap, r0, r1)])
		return phi - isovalue, grad

	def _bounding_box(self, isovalue=0.0):
		b0 = self.b_[0]
		return (-b0 - isovalue, -b0 - isovalue), (b0 + isovalue, b0 + isovalue)

	def evaluate_autodiff_template(self, q, isovalue):
		b0, b1 = self.b_
		# Apply symmetry
		p0, p1 = ad_abs(q[0]), ad_abs(q[1])
		# Swap if needed
		swap = p1 > p0
		w0 = torch.where(swap, p1, p0) - b0
		w1 = torch.where(swap, p0, p1) - b1
		h = ad_max(w0, w1)
		neg = h < 0.0
		o0 = torch.where(neg, ad_max(b1 - b0 - w0, 0.0), ad_max(w0, 0.0))
		o1 = torch.where(neg, ad_max(-w1, 0.0), ad_max(w1, 0.0))
		length = torch.sqrt(o0 * o0 + o1 * o1)
		use_corner = (h >= 0.0) | (-w0 >= length)
		sign_h = torch.where(h >= 0.0, 1.0, -1.0).to(q.dtype)
		return torch.where(use_corner, length * sign_h, -w0 * sign_h) - isovalue


class TriangleIsosceles2D(SDF2D):
	"""
	Isosceles Triangle SDF with apex at the origin and base corners (±q_x, q_y). p = (|x|, y):
		a = p - q clamp(p·q / q·q, 0, 1)    (slanted edge),   b = p - (q_x clamp(p_x / q_x, 0, 1), q_y)  (base)
		φ = √min(a·a, b·b) · sign(max(k (p_x q_y - p_y q_x), k (p_y - q_y))),  k = sign(q_y)
	Gradient: edge normal (-q_y, q_x)/‖q‖ (x times s_x) or (0, k) on the base when the projection is interior
	(1e-10 margin), otherwise (a or b)/‖·‖ (x times s_x) times the sign. (Updated from Inigo's formula so gradients
	are computed to *S*DF, not UDF then signed.)
	"""

	def __init__(self, q):
		super().__init__()
		self.q_ = _host_vector(q, 2)

	def _evaluate_with_gradient(self, query, isovalue=0.0, want_gradient=True):
		qx, qy = self.q_
		eps = self.epsilon_dist
		w = sign(query[0])
		px = torch.abs(query[0])
		py = query[1]
		t_a = clamp((px * qx + py * qy) / (qx * qx + qy * qy), 0.0, 1.0)
		ax = px - qx * t_a
		ay = py - qy * t_a
		t_b = clamp(px / qx, 0.0, 1.0)
		bx = px - qx * t_b
		by = py - qy * 1.0
		k = _sign_host(qy)
		l1 = ax * ax + ay * ay
		l2 = bx * bx + by * by
		use_a = l1 < l2
		d = torch.sqrt(torch.where(use_a, l1, l2))
		s = sign(_std_max(k * (px * qy - py * qx), k * (py - qy)))
		g = None
		if want_gradient:
			nrm = math.sqrt(qy * qy + qx * qx)
			edge_a = (torch.abs(t_a) > eps) & (torch.abs(t_a - 1.0) > eps)
			edge_b = (torch.abs(t_b) > eps) & (torch.abs(t_b - 1.0) > eps)
			na = torch.sqrt(ax * ax + ay * ay)
			nb = torch.sqrt(bx * bx + by * by)
			# closest to the slanted edge / its vertex
			gax = torch.where(edge_a, -qy / nrm * w, ax / na * w * s)
			gay = torch.where(edge_a, torch.full_like(s, qx / nrm), ay / na * s)
			# closest to base / its vertex
			gbx = torch.where(edge_b, torch.zeros_like(s), bx / nb * w * s)
			gby = torch.where(edge_b, torch.full_like(s, 1.0 * k), by / nb * s)
			g = torch.stack([torch.where(use_a, gax, gbx), torch.where(use_a, gay, gby)])
		return d * s - isovalue, g

	def _bounding_box(self, isovalue=0.0):
		qx, qy = self.q_
		return (-qx - isovalue, _fmin(0.0, qy) - isovalue), (qx + isovalue, _fmax(0.0, qy) + isovalue)

	def evaluate_autodiff_template(self, q, isovalue):
		qx, qy = self.q_
		p0 = ad_abs(q[0])
		p1 = q[1]
		# Project onto edge
		t = (p0 * qx + p1 * qy) / (qx * qx + qy * qy)
		t = ad_max(0.0, ad_min(1.0, t))
		a0 = p0 - qx * t
		a1 = p1 - qy * t
		# Project onto vertical edge
		tx = ad_max(0.0, ad_min(1.0, p0 / qx))
		b0 = p0 - qx * tx
		b1 = p1 - qy
		l1 = a0 * a0 + a1 * a1
		l2 = b0 * b0 + b1 * b1
		d = torch.sqrt(torch.where(l1 < l2, l1, l2))
		# Determine sign
		k = _sign_host(qy)
		s = ad_max(k * (p0 * qy - p1 * qx), k * (p1 - qy))
		return d * torch.where(s >= 0.0, 1.0, -1.0).to(q.dtype) - isovalue


class Triangle2D(SDF2D):
	"""
	General Triangle SDF with vertices v0, v1, v2. Per edge e_i = v_{i+1} - v_i, w = q - v_i:
		t = clamp(w·e / e·e, 0, 1),  d_i = ‖w - t e‖²,  s_i = g (w_x e_y - w_y e_x),  g = cross(v0 - v2, v1 - v0)
		φ = √min_i d_i · sign(max_i s_i)
	∇φ: of the nearest edge (first on ties), its normal (e_y, -e_x)/‖e‖ if 1e-10 < t < 1 - 1e-10, else
	(w - t e)/‖w - t e‖. (Updated from Inigo's formula so gradients are computed to *S*DF, not UDF then signed.)
	The autodiff template uses the sign of max_i (w_x e_y - w_y e_x), without g (as the C++).
	"""

	def __init__(self, v0, v1, v2):
		super().__init__()
		self.v_ = [_host_vector(v0, 2), _host_vector(v1, 2), _host_vector(v2, 2)]

	def _edge(self, p, i, j, want_gradient):
		vi, vj = self.v_[i], self.v_[j]
		ex, ey = vj[0] - vi[0], vj[1] - vi[1]
		wx = p[0] - vi[0]
		wy = p[1] - vi[1]
		t = clamp((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0, 1.0)
		qx = wx - ex * t
		qy = wy - ey * t
		d = qx * qx + qy * qy
		s = self._gs * (wx * ey - wy * ex)
		grad = None
		if want_gradient:
			eps = self.epsilon_dist
			edge = (torch.abs(t) > eps) & (torch.abs(t - 1.0) > eps)
			nrm = math.sqrt(ey * ey + ex * ex)
			nq = torch.sqrt(qx * qx + qy * qy)
			grad = torch.stack([torch.where(edge, ey / nrm, qx / nq), torch.where(edge, -ex / nrm, qy / nq)])
		return d, s, grad

	@property
	def _gs(self) -> float:
		v0, v1, v2 = self.v_
		return (v0[0] - v2[0]) * (v1[1] - v0[1]) - (v0[1] - v2[1]) * (v1[0] - v0[0])

	def _evaluate_with_gradient(self, p, isovalue=0.0, want_gradient=True):
		res_d, res_s, grad = self._edge(p, 0, 1, want_gradient)
		for i, j in ((1, 2), (2, 0)):
			d, s, g = self._edge(p, i, j, want_gradient)
			closer = d < res_d
			res_d = torch.where(closer, d, res_d)
			if want_gradient:
				grad = torch.where(closer, g, grad)
			res_s = _std_max(res_s, s)
		phi = torch.sqrt(res_d) * sign(res_s)
		return phi - isovalue, grad

	def _bounding_box(self, isovalue=0.0):
		v0, v1, v2 = self.v_
		bmin = (_fmin(v0[0], _fmin(v1[0], v2[0])), _fmin(v0[1], _fmin(v1[1], v2[1])))
		bmax = (_fmax(v0[0], _fmax(v1[0], v2[0])), _fmax(v0[1], _fmax(v1[1], v2[1])))
		return (bmin[0] - isovalue, bmin[1] - isovalue), (bmax[0] + isovalue, bmax[1] + isovalue)

	def evaluate_autodiff_template(self, q, isovalue):
		ds, ss = [], []
		for i, j in ((0, 1), (1, 2), (2, 0)):
			vi, vj = self.v_[i], self.v_[j]
			# Edge vectors, vectors from vertices to point
			ex, ey = vj[0] - vi[0], vj[1] - vi[1]
			wx = q[0] - vi[0]
			wy = q[1] - vi[1]
			# Project onto edges
			t = ad_max(0.0, ad_min(1.0, (wx * ex + wy * ey) / (ex * ex + ey * ey)))
			# Closest points, squared distances
			cx = wx - ex * t
			cy = wy - ey * t
			ds.append(cx * cx + cy * cy)
			# Sign using cross products
			ss.append(wx * ey - wy * ex)
		res_d = ad_min(ds[0], ad_min(ds[1], ds[2]))
		res_s = ad_max(ss[0], ad_max(ss[1], ss[2]))
		return torch.sqrt(res_d) * torch.where(res_s >= 0.0, 1.0, -1.0).to(q.dtype) - isovalue


class Hexagon2D(SDF2D):
	"""
	Hexagon SDF (regular, inradius r, flat sides at y = ±r). k = (-√3/2, 1/2, 1/√3), p = |q|:
		w = p·(k_x, k_y),  p -= 2 min(w, 0) (k_x, k_y),  t = clamp(p_x, -k_z r, k_z r),  p -= (t, r),
		φ = ‖p‖ sign(p_y)
	∇φ: vertex (t at a clamp bound): s ⊙ M p / φ (M = reflection matrix if w < 0); slanted edge (w < 0):
	s ⊙ (-k_x, k_y); horizontal edge: (0, sign(q_y)). (Updated from Inigo's formula so gradients are computed to
	*S*DF, not UDF then signed.)
	"""

	k_ = (-0.5 * math.sqrt(3), 0.5, 1.0 / math.sqrt(3))

	def __init__(self, r: float):
		super().__init__()
		self.r_ = float(r)

	def _evaluate_with_gradient(self, q, isovalue=0.0, want_gradient=True):
		kx, ky, kz = self.k_
		r = self.r_
		s0, s1 = sign(q[0]), sign(q[1])
		px, py = torch.abs(q[0]), torch.abs(q[1])
		w = px * kx + py * ky
		f = 2.0 * _std_min(w, 0.0)
		px = px - f * kx
		py = py - f * ky
		t = clamp(px, -kz * r, kz * r)
		px = px - t
		py = py - r
		d = torch.sqrt(px * px + py * py) * sign(py)
		grad = None
		if want_gradient:
			reflect = w < 0.0
			gx = torch.where(reflect, -ky * px + -kx * py, px)
			gy = torch.where(reflect, -kx * px + ky * py, py)
			eps = self.epsilon_dist
			vertex = (torch.abs(t - (-kz * r)) < eps) | (torch.abs(t - (kz * r)) < eps)
			# vertex / closest to angled edge / closest to horizontal edge
			grad = torch.stack(
				[
					torch.where(vertex, s0 * gx / d, torch.where(reflect, s0 * -kx, 0.0 * s1)),
					torch.where(vertex, s1 * gy / d, torch.where(reflect, s1 * ky, s1)),
				]
			)
		return d - isovalue, grad

	def _bounding_box(self, isovalue=0.0):
		half_width = 2.0 * self.r_ / math.sqrt(3.0)
		half_height = self.r_
		return (-half_width - isovalue, -half_height - isovalue), (half_width + isovalue, half_height + isovalue)

	def evaluate_autodiff_template(self, q, isovalue):
		kx, ky, kz = self.k_
		r = self.r_
		p0, p1 = ad_abs(q[0]), ad_abs(q[1])
		# w = p.x * k.x + p.y * k.y
		w = p0 * kx + p1 * ky
		# p -= 2 * ad_min(w, 0) * (k.x, k.y)
		factor = 2.0 * ad_min(w, 0.0)
		p0 = p0 - factor * kx
		p1 = p1 - factor * ky
		# p -= (clamp(p.x, -k.z * r, k.z * r), r)
		clamped_x = ad_min(ad_max(p0, -kz * r), kz * r)
		p0 = p0 - clamped_x
		p1 = p1 - r
		d = torch.sqrt(p0 * p0 + p1 * p1)
		return d * torch.where(p1 >= 0.0, 1.0, -1.0).to(q.dtype) - isovalue


class Ellipse2D(SDF2D):
	"""
	Ellipse SDF with semi-axes ab = (a, b). p = |q|; closest point (a cos ω, b sin ω) by 4 Newton steps
		ω += (p - u)·v / ((p - u)·u + v·v),  u = (a cos ω, b sin ω),  v = (-a sin ω, b cos ω)
	(a step is skipped when |denominator| ≤ 1e-10), started at ω = atan2(a p_y, b p_x) outside ((p_x/a)² + (p_y/b)² > 1)
	and at π/2 or 0 inside. φ = sgn (‖p - closest‖ - isovalue) (sgn = +1 outside, -1 inside: the isovalue enters
	with the sign, as in the C++), ∇φ = s ⊙ (cos ω / a, sin ω / b) normalized (the ellipse normal).
	Warning (original): not used and not verified yet.
	The autodiff template has no 1e-10 guard, starts inside at 1.570796327 and returns sgn ‖p - closest‖ - isovalue.
	"""

	def __init__(self, ab):
		super().__init__()
		self.ab_ = _host_vector(ab, 2)  # semi-axes

	def _evaluate_with_gradient(self, query, isovalue=0.0, want_gradient=True):
		a, b = self.ab_
		epsilon = 1e-10
		px, py = torch.abs(query[0]), torch.abs(query[1])
		pab0 = px / a
		pab1 = py / b
		s = pab0 * pab0 + pab1 * pab1 > 1.0
		# Initial angle estimate
		inside_w = torch.where(a * (px - a) < b * (py - b), torch.full_like(px, 0.5 * math.pi), torch.zeros_like(px))
		w = torch.where(s, torch.atan2(py * a, px * b), inside_w)
		# Newton iteration to find closest point parameter
		for _ in range(4):
			c = torch.cos(w)
			sn = torch.sin(w)
			ux = a * c
			uy = b * sn
			vx = a * -sn
			vy = b * c
			dx = px - ux
			dy = py - uy
			numerator = dx * vx + dy * vy
			denominator = (dx * ux + dy * uy) + (vx * vx + vy * vy)
			w = torch.where(torch.abs(denominator) > epsilon, w + numerator / denominator, w)
		# Closest point on ellipse
		c = torch.cos(w)
		sn = torch.sin(w)
		dx = px - a * c
		dy = py - b * sn
		d = torch.sqrt(dx * dx + dy * dy)
		sgn = torch.where(s, 1.0, -1.0).to(px.dtype)
		grad = None
		if want_gradient:
			# Gradient: use ellipse normal at parameter w
			# For ellipse (a*cos(w), b*sin(w)), the unnormalized outward normal is (cos(w)/a, sin(w)/b)
			nx = c / a
			ny = sn / b
			nrm = torch.sqrt(nx * nx + ny * ny)
			# Degenerate case: use radial direction
			radial = _normalized(torch.stack([dx, dy]))
			sp0, sp1 = sign(query[0]), sign(query[1])
			ok = nrm > epsilon
			grad = torch.stack([sp0 * torch.where(ok, nx / nrm, radial[0]), sp1 * torch.where(ok, ny / nrm, radial[1])])
		return sgn * (d - isovalue), grad

	def _bounding_box(self, isovalue=0.0):
		a, b = self.ab_
		return (-a - isovalue, -b - isovalue), (a + isovalue, b + isovalue)

	def evaluate_autodiff_template(self, q, isovalue):
		a, b = self.ab_
		p0, p1 = ad_abs(q[0]), ad_abs(q[1])
		normalized = (p0 * p0) / (a * a) + (p1 * p1) / (b * b)
		s = normalized > 1.0
		inside_w = torch.where(a * (p0 - a) < b * (p1 - b), torch.full_like(p0, 1.570796327), torch.zeros_like(p0))
		w = torch.where(s, torch.atan2(a * p1, b * p0), inside_w)
		for _ in range(4):
			c = torch.cos(w)
			sn = torch.sin(w)
			u0 = a * c
			u1 = b * sn
			v0 = -a * sn
			v1 = b * c
			d0 = p0 - u0
			d1 = p1 - u1
			w = w + (d0 * v0 + d1 * v1) / (d0 * u0 + d1 * u1 + v0 * v0 + v1 * v1)
		d0 = p0 - a * torch.cos(w)
		d1 = p1 - b * torch.sin(w)
		dist = torch.sqrt(d0 * d0 + d1 * d1)
		return torch.where(s, 1.0, -1.0).to(q.dtype) * dist - isovalue


class Heart2D(SDF2D):
	"""
	Heart SDF. p = (|q_x|, q_y), r = √2/4:
		if p_x + p_y > 1:  φ = ‖p - (1/4, 3/4)‖ - r                                    (upper part: circular arc)
		else:              φ = √min(‖p - (0, 1)‖², ‖p - ½ max(p_x + p_y, 0) (1, 1)‖²) · (1 if p_x > p_y else -1)
	∇φ = (s_x v_x, v_y) / max(‖v‖, 1e-10) for the vector v from the nearest feature (times the sign below).
	"""

	def _evaluate_with_gradient(self, q, isovalue=0.0, want_gradient=True):
		sx = sign(q[0])
		px = torch.abs(q[0])
		py = q[1]
		upper_cond = py + px > 1.0
		r = math.sqrt(2.0) / 4.0
		# Upper part: distance to circular arc
		q0x = px - 0.25
		q0y = py - 0.75
		l0 = torch.sqrt(q0x * q0x + q0y * q0y)
		# Lower part: distance to point (0,1) or diagonal line
		q1x = px - 0.0
		q1y = py - 1.0
		d1_sq = q1x * q1x + q1y * q1y
		h = _std_max(px + py, 0.0)
		q2x = px - 0.5 * h
		q2y = py - 0.5 * h
		d2_sq = q2x * q2x + q2y * q2y
		use_d1 = d1_sq < d2_sq
		phi_lower = torch.sqrt(torch.where(use_d1, d1_sq, d2_sq))
		sign_mult = torch.where(px > py, 1.0, -1.0).to(px.dtype)
		phi = torch.where(upper_cond, l0 - r, phi_lower * sign_mult)
		grad = None
		if want_gradient:
			l0m = _std_max(l0, 1e-10)
			plm = _std_max(phi_lower, 1e-10)
			gtx = torch.where(use_d1, q1x, q2x) / plm
			gty = torch.where(use_d1, q1y, q2y) / plm
			grad = torch.stack(
				[
					torch.where(upper_cond, sx * q0x / l0m, sx * gtx * sign_mult),
					torch.where(upper_cond, q0y / l0m, gty * sign_mult),
				]
			)
		return phi - isovalue, grad

	def _bounding_box(self, isovalue=0.0):
		r = math.sqrt(2.0) / 4.0
		return (-0.25 - r - isovalue, 0.0 - isovalue), (0.25 + r + isovalue, 0.75 + r + isovalue)

	def evaluate_autodiff_template(self, q, isovalue):
		r = math.sqrt(2.0) / 4.0
		p0 = ad_abs(q[0])
		p1 = q[1]
		upper_cond = p1 + p0 > 1.0
		q0x = p0 - 0.25
		q0y = p1 - 0.75
		upper = torch.sqrt(q0x * q0x + q0y * q0y) - r - isovalue
		q1x = p0 - 0.0
		q1y = p1 - 1.0
		d1_sq = q1x * q1x + q1y * q1y
		h = ad_max(p0 + p1, 0.0)
		q2x = p0 - h * 0.5
		q2y = p1 - h * 0.5
		d2_sq = q2x * q2x + q2y * q2y
		d = torch.sqrt(torch.where(d1_sq < d2_sq, d1_sq, d2_sq))
		lower = d * torch.where(p0 - p1 >= 0.0, 1.0, -1.0).to(q.dtype) - isovalue
		return torch.where(upper_cond, upper, lower)


class Pentagon2D(SDF2D):
	"""
	Pentagon SDF (regular, flat side at y = -r). m = (0.80901699, 0.58778525, 0.72654253) = (cos 36°, sin 36°,
	tan 36°), n = (m_x² - m_y², 2 m_x m_y); p = (|q_x|, q_y):
		w1 = p·(m_x, m_y),  w2 = p_x n_x - p_y n_y,  p -= 2 max(w1, 0) (m_x, m_y),  p -= 2 min(w2, 0) (m_x, -m_y),
		t = clamp(p_x, -r m_z, r m_z),  p -= (t, -r),  φ = ‖p‖ sign(-p_y)
	∇φ: vertex: M p / φ (x times s_x, M the reflection of the active fold); edges: (s_x m_y, m_x) if w2 < 0,
	(s_x n_y, -n_x) if w1 > 0, else (0, -1). (Updated from Inigo's formula so gradients are computed to *S*DF.)
	"""

	m_ = (0.80901699, 0.58778525, 0.72654253)
	n_ = (0.80901699 * 0.80901699 - 0.58778525 * 0.58778525, 2.0 * 0.80901699 * 0.58778525)

	def __init__(self, r: float):
		super().__init__()
		self.r_ = float(r)

	def _evaluate_with_gradient(self, q, isovalue=0.0, want_gradient=True):
		mx, my, mz = self.m_
		nx, ny = self.n_
		r = self.r_
		s = sign(q[0])
		px = torch.abs(q[0])
		py = q[1]
		w1 = px * mx + py * my
		w2 = px * nx - py * ny
		f1 = 2.0 * _std_max(w1, 0.0)
		px = px - f1 * mx
		py = py - f1 * my
		f2 = 2.0 * _std_min(w2, 0.0)
		px = px - f2 * mx
		py = py - f2 * -my
		t = clamp(px, -r * mz, r * mz)
		px = px - t
		py = py - -r
		d = torch.sqrt(px * px + py * py) * sign(-py)
		grad = None
		if want_gradient:
			fold2 = w2 < 0.0
			fold1 = w1 > 0.0
			gx = torch.where(fold2, -mx * px + -my * py, torch.where(fold1, -nx * px + -ny * py, px)) * s
			gy = torch.where(fold2, my * px + -mx * py, torch.where(fold1, -ny * px + nx * py, py))
			eps = self.epsilon_dist
			vertex = (torch.abs(t - (-r * mz)) < eps) | (torch.abs(t - (r * mz)) < eps)
			ex = torch.where(fold2, s * my, torch.where(fold1, s * ny, torch.zeros_like(s)))
			ey = torch.where(fold2, mx, torch.where(fold1, torch.full_like(px, -nx), torch.full_like(px, -1.0)))
			grad = torch.stack([torch.where(vertex, gx / d, ex), torch.where(vertex, gy / d, ey)])
		return d - isovalue, grad

	def _bounding_box(self, isovalue=0.0):
		r = self.r_
		R = math.sqrt(6.0 - 2.0 * math.sqrt(5)) * r
		diag = math.sqrt(1.0 - 2.0 / math.sqrt(5.0)) * (5.0 + math.sqrt(5.0)) * r
		return (-0.5 * diag - isovalue, -r - isovalue), (0.5 * diag + isovalue, R + isovalue)

	def evaluate_autodiff_template(self, q, isovalue):
		mx, my, mz = self.m_
		nx, ny = self.n_
		r = self.r_
		p0 = ad_abs(q[0])
		p1 = q[1]
		w1 = p0 * mx + p1 * my
		w2 = p0 * nx - p1 * ny
		factor1 = 2.0 * ad_max(w1, 0.0)
		p0 = p0 - factor1 * mx
		p1 = p1 - factor1 * my
		factor2 = 2.0 * ad_min(w2, 0.0)
		p0 = p0 - factor2 * mx
		p1 = p1 + factor2 * my
		clamped_x = ad_min(ad_max(p0, -r * mz), r * mz)
		p0 = p0 - clamped_x
		p1 = p1 + r
		d = torch.sqrt(p0 * p0 + p1 * p1)
		return d * torch.where(p1 >= 0.0, -1.0, 1.0).to(q.dtype) - isovalue


class Moon2D(SDF2D):
	"""
	Moon SDF: disc of radius ra at the origin minus the disc of radius rb at (d, 0). p = (q_x, |q_y|),
	a = (ra² - rb² + d²) / 2d, b = √max(ra² - a², 0) (the horn tip (a, b)):
		if d (p_x b - p_y a) > d² max(b - p_y, 0):  φ = ‖p - (a, b)‖    (distance to intersection point of two circles)
		else:  φ = max(‖p‖ - ra, rb - ‖p - (d, 0)‖)                      (distance to either circle)
	∇φ = (v_x, s_y v_y) / ‖v‖ for the corresponding vector v (negated for the inner circle).
	"""

	def __init__(self, d: float, ra: float, rb: float):
		super().__init__()
		self.d_, self.ra_, self.rb_ = float(d), float(ra), float(rb)

	def _ab(self):
		d, ra, rb = self.d_, self.ra_, self.rb_
		a = (ra * ra - rb * rb + d * d) / (2.0 * d)
		return a, math.sqrt(_fmax(ra * ra - a * a, 0.0))

	def _evaluate_with_gradient(self, q, isovalue=0.0, want_gradient=True):
		d, ra, rb = self.d_, self.ra_, self.rb_
		a, b = self._ab()
		s = sign(q[1])
		px = q[0]
		py = torch.abs(q[1])
		cond = d * (px * b - py * a) > d * d * _std_max(b - py, 0.0)
		# Distance to intersection point of two circles
		wx = px - a
		wy = py - b
		d_cond = torch.sqrt(wx * wx + wy * wy)
		# Distance to either circle
		l1 = torch.sqrt(px * px + py * py)
		d1 = l1 - ra
		w2x = px - d
		w2y = py - 0.0
		l2 = torch.sqrt(w2x * w2x + w2y * w2y)
		d2 = rb - l2
		use1 = d1 > d2
		phi = torch.where(cond, d_cond, torch.where(use1, d1, d2))
		grad = None
		if want_gradient:
			gx = torch.where(cond, wx / d_cond, torch.where(use1, px / l1, -w2x / l2))
			gy = torch.where(cond, wy * s / d_cond, torch.where(use1, py * s / l1, -w2y * s / l2))
			grad = torch.stack([gx, gy])
		return phi - isovalue, grad

	def _bounding_box(self, isovalue=0.0):
		ra = self.ra_
		a = self._ab()[0]
		return (-ra - isovalue, -ra - isovalue), (a + isovalue, ra + isovalue)

	def evaluate_autodiff_template(self, q, isovalue):
		d, ra, rb = self.d_, self.ra_, self.rb_
		a = (ra * ra - rb * rb + d * d) / (2.0 * d)
		b = ra * ra - a * a
		b = math.sqrt(b if b > 0.0 else 0.0)
		p0 = q[0]
		p1 = ad_abs(q[1])
		lhs = d * (p0 * b - p1 * a)
		rhs = d * d * ad_max(b - p1, 0.0)
		w0 = p0 - a
		w1 = p1 - b
		r_cond = torch.sqrt(w0 * w0 + w1 * w1) - isovalue
		d1 = torch.sqrt(p0 * p0 + p1 * p1) - ra
		w20 = p0 - d
		w21 = p1 - 0.0
		d2 = rb - torch.sqrt(w20 * w20 + w21 * w21)
		return torch.where(lhs > rhs, r_cond, torch.where(d1 > d2, d1, d2) - isovalue)


class Trapezoid2D(SDF2D):
	"""
	Isosceles Trapezoid SDF: bottom half width ra (y = -he), top half width rb (y = he). p = (|q_x|, q_y):
		caps:  c = (min(p_x, ra if p_y < 0 else rb), sign(p_y) he),  d = ‖p - c‖²,  s = |p_y| - he
		side:  k = (rb - ra, 2 he), h = clamp((p - (ra, -he))·k / k·k, 0, 1), d' = ‖p - (ra, -he) - h k‖²,
		       s' = cross(p - (ra, -he), k)
		φ = √min(d, d') · sign(max(s, s'))
	∇φ: cap (0, sign(q_y)); side edge normal; at a vertex (or when the cap projection is clamped) the normalized
	vector from the closest point times the sign. (Updated from Inigo's formula so gradients are computed to *S*DF,
	not UDF then signed.)
	"""

	def __init__(self, ra: float, rb: float, he: float):
		super().__init__()
		self.ra_, self.rb_, self.he_ = float(ra), float(rb), float(he)

	def _evaluate_with_gradient(self, q, isovalue=0.0, want_gradient=True):
		ra, rb, he = self.ra_, self.rb_, self.he_
		eps = self.epsilon_dist
		sx = sign(q[0])
		sy = sign(q[1])
		px = torch.abs(q[0])
		py = q[1]
		h_max = torch.where(py < 0.0, torch.full_like(px, ra), torch.full_like(px, rb))
		h = _std_min(px, h_max)
		qx = px - h
		qy = py - sy * he
		res_d = qx * qx + qy * qy
		res_s = torch.abs(py) - he
		res_qx, res_qy = qx, qy
		closest_vertex = (torch.abs(h) < eps) | (torch.abs(h - h_max) < eps)

		kx, ky = rb - ra, 2.0 * he
		wx = px - ra
		wy = py - -he
		h = clamp((wx * kx + wy * ky) / (kx * kx + ky * ky), 0.0, 1.0)
		qx = px - (ra + h * kx)
		qy = py - (-he + h * ky)
		d = qx * qx + qy * qy
		res_s = _std_max(res_s, wx * ky - wy * kx)
		closer = d < res_d
		res_d = torch.where(closer, d, res_d)
		phi = torch.sqrt(res_d) * sign(res_s)
		grad = None
		if want_gradient:
			res_qx = torch.where(closer, qx, res_qx)
			res_qy = torch.where(closer, qy, res_qy)
			side = (torch.abs(h) > eps) & (torch.abs(h - 1.0) > eps)
			closest_vertex = closest_vertex | (closer & ~side)
			# side edge: e = v0 - v3, v0 = (rb/2, he/2), v3 = (ra/2, -he/2)
			e0, e1 = 0.5 * rb - 0.5 * ra, 0.5 * he - -0.5 * he
			nrm = math.sqrt(e1 * e1 + e0 * e0)
			gx = torch.where(closer & side, e1 * sx / nrm, sy * 0.0)
			gy = torch.where(closer & side, torch.full_like(sy, -e0 / nrm), sy * 1.0)
			sr = sign(res_s)
			vx = sx * res_qx * sr
			vy = res_qy * sr
			vn = torch.sqrt(vx * vx + vy * vy)
			grad = torch.stack([torch.where(closest_vertex, vx / vn, gx), torch.where(closest_vertex, vy / vn, gy)])
		return phi - isovalue, grad

	def _bounding_box(self, isovalue=0.0):
		max_width = _fmax(self.ra_, self.rb_)
		he = self.he_
		return (-max_width - isovalue, -he - isovalue), (max_width + isovalue, he + isovalue)

	def evaluate_autodiff_template(self, q, isovalue):
		ra, rb, he = self.ra_, self.rb_, self.he_
		p0 = ad_abs(q[0])
		p1 = q[1]
		h = ad_min(p0, torch.where(p1 < 0.0, torch.full_like(p1, ra), torch.full_like(p1, rb)))
		sy = torch.where(p1 >= 0.0, 1.0, -1.0).to(q.dtype)
		q0 = p0 - h
		q1 = p1 - sy * he
		res_d = q0 * q0 + q1 * q1
		res_s = ad_abs(p1) - he
		k0, k1 = rb - ra, 2.0 * he
		w0 = p0 - ra
		w1 = p1 - -he
		h = (w0 * k0 + w1 * k1) / (k0 * k0 + k1 * k1)
		h = _eigen_ad_max(0.0, ad_min(1.0, h))
		q0 = p0 - (ra + k0 * h)
		q1 = p1 - (-he + k1 * h)
		d = q0 * q0 + q1 * q1
		s = w0 * k1 - w1 * k0
		res_d = torch.where(d < res_d, d, res_d)
		res_s = _eigen_ad_max(res_s, s)
		return torch.sqrt(res_d) * torch.where(res_s >= 0.0, 1.0, -1.0).to(q.dtype) - isovalue


class Quad2D(SDF2D):
	"""
	Quad SDF (simple quadrilateral v0..v3). Per edge e_i = v_{i+1} - v_i, w = q - v_i:
		h = clamp(w·e / e·e, 0, 1),  d_i = ‖w - h e‖²,  s_i = g cross(w, e),  g = cross(v0 - v3, v1 - v0)
		(global sign for winding order),  φ = √min_i d_i · sign(max_i s_i)
	∇φ as Triangle2D. (Updated from Inigo's formula so gradients are computed to *S*DF, not UDF then signed.)
	"""

	def __init__(self, v0, v1, v2, v3):
		super().__init__()
		self.v_ = [_host_vector(v, 2) for v in (v0, v1, v2, v3)]

	@property
	def _gs(self) -> float:
		v = self.v_
		return (v[0][0] - v[3][0]) * (v[1][1] - v[0][1]) - (v[0][1] - v[3][1]) * (v[1][0] - v[0][0])

	def _evaluate_with_gradient(self, p, isovalue=0.0, want_gradient=True):
		gs = self._gs
		res_d = torch.full_like(p[0], torch.finfo(p.dtype).max)  # DBL_MAX
		res_s = torch.full_like(p[0], -torch.finfo(p.dtype).max)
		grad = torch.zeros_like(p) if want_gradient else None
		eps = self.epsilon_dist
		# Process each edge
		for i in range(4):
			vi, vj = self.v_[i], self.v_[(i + 1) % 4]
			ex, ey = vj[0] - vi[0], vj[1] - vi[1]
			wx = p[0] - vi[0]
			wy = p[1] - vi[1]
			h = clamp((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0, 1.0)
			qx = wx - h * ex
			qy = wy - h * ey
			d = qx * qx + qy * qy
			closer = d < res_d
			res_d = torch.where(closer, d, res_d)
			if want_gradient:
				edge = (torch.abs(h) > eps) & (torch.abs(h - 1.0) > eps)
				nrm = math.sqrt(ey * ey + ex * ex)
				nq = torch.sqrt(qx * qx + qy * qy)
				g = torch.stack([torch.where(edge, ey / nrm, qx / nq), torch.where(edge, -ex / nrm, qy / nq)])
				grad = torch.where(closer, g, grad)
			res_s = _std_max(res_s, gs * (wx * ey - wy * ex))
		phi = torch.sqrt(res_d) * sign(res_s)
		return phi - isovalue, grad

	def _bounding_box(self, isovalue=0.0):
		bmin = list(self.v_[0])
		bmax = list(self.v_[0])
		for v in self.v_[1:]:
			bmin = [_fmin(bmin[0], v[0]), _fmin(bmin[1], v[1])]
			bmax = [_fmax(bmax[0], v[0]), _fmax(bmax[1], v[1])]
		return (bmin[0] - isovalue, bmin[1] - isovalue), (bmax[0] + isovalue, bmax[1] + isovalue)

	def evaluate_autodiff_template(self, q, isovalue):
		gs = self._gs
		res_d = None
		res_s = None
		for i in range(4):
			vi, vj = self.v_[i], self.v_[(i + 1) % 4]
			ex, ey = vj[0] - vi[0], vj[1] - vi[1]
			wx = q[0] - vi[0]
			wy = q[1] - vi[1]
			h = ad_max(0.0, ad_min(1.0, (wx * ex + wy * ey) / (ex * ex + ey * ey)))
			qx = wx - ex * h
			qy = wy - ey * h
			d = qx * qx + qy * qy
			s_curr = gs * (wx * ey - wy * ex)
			# res_d starts at 1e10 and res_s at -1e10 in the C++
			res_d = torch.where(d < 1e10, d, 1e10) if res_d is None else torch.where(d < res_d, d, res_d)
			res_s = (
				_eigen_ad_max(torch.full_like(s_curr, -1e10), s_curr) if res_s is None else _eigen_ad_max(res_s, s_curr)
			)
		return torch.sqrt(res_d) * torch.where(res_s >= 0.0, 1.0, -1.0).to(q.dtype) - isovalue
