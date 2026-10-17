"""
What the network computes for a point, and from which neighbors (added in the port; no change to the fit).

The fit (infer.PointsAsTori, torus_distance.fit_torus_at_point): for point i, its k_network nearest inlier points
(64 for the bundled model: the training data's k_neighbors, ext/points-as-tori/training/mesh_data.py) are expressed in
the frame (s_hat, t_hat, n_i) of p_i (positions scaled by the median neighbor distance, normals in the same frame) and
fed as k tokens to a pre-norm transformer (network.FundamentalFormPredictor; the bundled model: 8 blocks of 8 heads,
width 128); the output token of the center
(token 0) is mapped linearly to 6 numbers, the coefficients of the height function over the tangent plane
	Q(s, t) = a00 + a10 s + a01 t + a20 s² + a11 s t + a02 t²       (order (a00, a01, a10, a11, a02, a20))
from which the principal curvatures and directions follow in closed form (torus_distance._principal_curvatures_from_
forms: fundamental forms I, II of the graph of Q at (0, 0), generalized eigenproblem II v = κ I v), and the torus.
The network is trained end to end on the blended signed distance (L1 + eikonal, training/train.py SDF_loss): the
curvatures are never supervised; they are whatever makes the blend fit the training shapes.

Functions here:
  curvature_frame         κ_min, κ_max, their unit directions, the normal n* and the foot point p* = p + a00 n of
                          every point, from the coefficients (the fit's own formulas)
  attention_rollout       which tokens (neighbors) the center token's output draws from, through all blocks
  gradient_attribution    ‖∂(κ_min, κ_max)/∂p_j‖: how much the curvatures move when neighbor j moves
  least_squares_forms     the classical alternative: Q fitted to the heights of the k nearest points (k >= 6: six
                          unknowns), for comparison with the network's Q
  explain_point           all of the above for one point
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from .shape_2d import orthonormal_basis
from .torus_distance import _principal_curvatures_from_forms


def _unit(v: torch.Tensor) -> torch.Tensor:
	return v / torch.linalg.vector_norm(v, dim=0, keepdim=True).clamp_min(1e-30)


def curvature_frame(points: torch.Tensor, normals: torch.Tensor, coefficients: torch.Tensor) -> Dict[str, torch.Tensor]:
	"""
	Args: points, normals (3, N), coefficients (6, N) (a00, a01, a10, a11, a02, a20).
	Returns: kappa_min, kappa_max (N,) with |κ_min| <= |κ_max| (sign: height along +n, so a sphere with outward normals
	has κ = -1/radius), dir_min, dir_max (3, N) unit tangent directions, normal (3, N) n* of the graph of Q at (0, 0),
	a00 (N,), foot (3, N) = p + a00 n. The same variant as fit_torus_at_point (python_variant: the training's guards).
	"""
	n = _unit(normals)
	s_hat, t_hat = orthonormal_basis(n)
	a00, k_min, k_max, v_min, v_max, n_star = _principal_curvatures_from_forms(n, coefficients, s_hat, t_hat, True)
	return {
		'kappa_min': k_min,
		'kappa_max': k_max,
		'dir_min': _unit(v_min),
		'dir_max': _unit(v_max),
		'normal': _unit(n_star),
		'a00': a00,
		'foot': points + a00 * n,
	}


def attention_rollout(attention: Sequence[torch.Tensor], residual: float = 0.5) -> torch.Tensor:
	"""
	Attention rollout (Abnar & Zuidema 2020, "Quantifying Attention Flow in Transformers"): with Ā_l the head-averaged
	attention of block l (rows sum to 1) and the residual connection x_l = x_(l-1) + Attn(LN(x_(l-1))) approximated by
	Â_l = residual I + (1 - residual) Ā_l, the mixing of input tokens into the outputs is R = Â_L ... Â_1 and row 0 of R
	says how much the center token's output draws from each input token (rows sum to 1).
	A heuristic: the value / output projections, the feed-forward blocks and the layer norms are left out.

	Args: list over blocks of (batch, heads, k, k). Returns: (batch, k).
	"""
	k = attention[0].shape[-1]
	eye = torch.eye(k, dtype=attention[0].dtype, device=attention[0].device)
	rollout = eye.expand(attention[0].shape[0], k, k)
	for a in attention:
		mixed = residual * eye + (1.0 - residual) * a.mean(dim=1)
		rollout = mixed @ rollout
	return rollout[:, 0, :]


def gradient_attribution(model, positions: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
	"""
	g_j = sqrt(|∂κ_min/∂p_j|² + |∂κ_max/∂p_j|²) for every neighbor j of every neighborhood (autograd through the
	network, the median scaling, the local frame and the curvature formulas; normals held fixed). Units 1/length².

	Args: positions, normals (3, batch, k), token 0 the center. Returns: (batch, k).
	"""
	dt = model._network_dtype()
	pos = positions.detach().to(dt).clone().requires_grad_(True)
	nrm = normals.detach().to(dt)
	with torch.enable_grad():
		coefficients = model(pos, nrm)
		center_n = _unit(nrm[:, :, 0])
		s_hat, t_hat = orthonormal_basis(center_n)
		_, k_min, k_max, _, _, _ = _principal_curvatures_from_forms(center_n, coefficients, s_hat, t_hat, True)
		(g_min,) = torch.autograd.grad(k_min.sum(), pos, retain_graph=True)
		(g_max,) = torch.autograd.grad(k_max.sum(), pos)
	return torch.sqrt((g_min * g_min).sum(0) + (g_max * g_max).sum(0)).detach()


def least_squares_forms(points: torch.Tensor, normals: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
	"""
	Classical fit: with (s_j, t_j, h_j) the neighbors' coordinates in the frame (s_hat, t_hat, n_i) of the center
	p_i = points[:, indices[b, 0]], minimize Σ_j (Q(s_j, t_j) - h_j)² over the 6 coefficients (exactly determined for 6
	points in general position; the center itself gives Q(0, 0) = a00 ≈ 0).

	Args: points, normals (3, N), indices (batch, k), k >= 6. Returns: (6, batch) (a00, a01, a10, a11, a02, a20).
	"""
	center = points[:, indices[:, 0]]  # (3, batch)
	n = _unit(normals[:, indices[:, 0]])
	s_hat, t_hat = orthonormal_basis(n)
	d = points[:, indices] - center[:, :, None]  # (3, batch, k)
	s = (d * s_hat[:, :, None]).sum(0)
	t = (d * t_hat[:, :, None]).sum(0)
	h = (d * n[:, :, None]).sum(0)
	one = torch.ones_like(s)
	design = torch.stack([one, t, s, s * t, t * t, s * s], dim=-1)  # columns: a00, a01, a10, a11, a02, a20
	return torch.linalg.lstsq(design, h[..., None]).solution[..., 0].T


def neighborhood(points: torch.Tensor, i: int, k: int, outlier: Optional[torch.Tensor] = None) -> torch.Tensor:
	"""The k nearest inlier points of point i, nearest first (point i itself first), as get_neighbors."""
	d = torch.linalg.vector_norm(points - points[:, i : i + 1], dim=0)
	if outlier is not None:
		d = torch.where(outlier, torch.full_like(d, float('inf')), d)
	k = min(k, int(torch.isfinite(d).sum()))
	order = torch.topk(d, k, largest=False).indices
	return order[torch.argsort(d[order], stable=True)]


def explain_point(model, points: torch.Tensor, normals: torch.Tensor, i: int, k: int, outlier=None) -> dict:
	"""
	For point i: its network neighborhood, the network's coefficients and curvatures, the attention rollout and the
	gradient attribution over the neighborhood, and least-squares curvatures from its 6 and its k nearest points.
	Tensors on the host, in neighborhood order (index 0 = point i).
	"""
	idx = neighborhood(points, i, k, outlier)
	dt = model._network_dtype()
	pos, nrm = points[:, idx][:, None, :], normals[:, idx][:, None, :]  # (3, 1, k)
	with torch.no_grad():
		coefficients, attention = model(pos.to(dt), nrm.to(dt), return_attention=True)
	coefficients = coefficients.to(points.dtype)
	frame = curvature_frame(points[:, i : i + 1], normals[:, i : i + 1], coefficients)
	out = {
		'indices': idx.cpu(),
		'coefficients': coefficients[:, 0].cpu(),
		'kappa_min': float(frame['kappa_min'][0]),
		'kappa_max': float(frame['kappa_max'][0]),
		'rollout': attention_rollout(attention)[0].float().cpu(),
		'gradient': gradient_attribution(model, pos, nrm)[0].float().cpu(),
		'least_squares': {},
	}
	for m in sorted({6, len(idx)}):
		lsq = least_squares_forms(points, normals, idx[None, :m])
		f = curvature_frame(points[:, i : i + 1], normals[:, i : i + 1], lsq)
		out['least_squares'][m] = (float(f['kappa_min'][0]), float(f['kappa_max'][0]))
	return out
