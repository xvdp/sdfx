"""Sample any torch SDF on a regular grid (scene field type "grid"), in z-slabs so memory stays bounded."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch


def bake_grid(
	fn: Callable[[torch.Tensor], torch.Tensor],
	bounds_min: Sequence[float],
	bounds_max: Sequence[float],
	resolution,
	device=None,
	dtype=torch.float32,
	max_points: int = 1 << 20,
) -> np.ndarray:
	"""
	values[z, y, x] = fn(p) at p = min + (x, y, z) / (res - 1) · (max - min)   (grid vertices, x fastest)

	Args:
		fn: (3, n) tensor -> (n,) tensor (e.g. lambda q: sdf.signed_distance(q))
		resolution: int or (W, H, D) samples per axis (>= 2)
		device: of the queries (default: cuda if available, else cpu; pass fn's device)
		max_points: queries per call to fn (whole z-slices, at least one)
	Returns:
		(D, H, W) float32 NumPy array (host)
	"""
	if device is None:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
	res = (resolution,) * 3 if np.isscalar(resolution) else tuple(resolution)
	w, h, d = (int(r) for r in res)
	lo = torch.as_tensor(bounds_min, dtype=dtype, device=device)
	hi = torch.as_tensor(bounds_max, dtype=dtype, device=device)
	axes = [
		lo[i] + (hi[i] - lo[i]) * torch.linspace(0, 1, n, dtype=dtype, device=device) for i, n in enumerate((w, h, d))
	]
	out = np.empty((d, h, w), np.float32)
	slab = max(1, max_points // (w * h))
	for z0 in range(0, d, slab):
		z1 = min(d, z0 + slab)
		zz, yy, xx = torch.meshgrid(axes[2][z0:z1], axes[1], axes[0], indexing='ij')
		q = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=0)
		out[z0:z1] = fn(q).reshape(z1 - z0, h, w).detach().float().cpu().numpy()
	return out
