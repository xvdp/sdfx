"""
Torch port of the small geometric utilities from src/cpp/shape_2d.h / shape_2d.cpp that the torus evaluator uses.
(The 2D SDF primitives of shape_2d.cpp are used to generate training data and are not ported yet.)

Layout: vectors are structure-of-arrays, (3, ...) with the component axis first (see config.py).
"""

from __future__ import annotations

from typing import Tuple

import torch


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
