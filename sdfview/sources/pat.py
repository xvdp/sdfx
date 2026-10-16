"""
Points as Tori (points_as_tori.PointsAsTori) -> scene: points, tori, a BVH built on the device (sdfview.accel), the
oriented point cloud for display and, optionally, a grid bake of the field for fast sphere tracing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..accel.lbvh import build_lbvh
from ..scene import Scene, grid_field
from .grid import bake_grid


def _rows(x: torch.Tensor) -> np.ndarray:
	"""(c, n) device tensor -> (n, c) float32 host array (export boundary)."""
	return x.detach().T.contiguous().to(torch.float32).cpu().numpy()


def scene_from_pat(
	sdf,
	k: int = 32,
	lambda_scale: Optional[float] = None,
	leaf_size: int = 4,
	padding: float = 0.1,
	bake_resolution: Optional[int] = None,
	point_radius: Optional[float] = None,
	name: str = 'pat',
	meta: Optional[dict] = None,
) -> Scene:
	"""
	Args:
		sdf: points_as_tori.PointsAsTori (or a TorusDistanceField with tori set)
		k: neighbors of the blend (the torch default for accelerated evaluation is 32)
		lambda_scale: None -> the evaluator's value (1.0)
		padding: scene box = point bounding box grown by padding · diagonal on each side
		bake_resolution: if set, also sample the field (torch, k-NN mode with this k) on a grid of this many
			samples along the longest box side and add it as field '<name>_grid'
		point_radius: display radius; None -> 0.4 · mean nearest-neighbor spacing estimate
	"""
	tdf = getattr(sdf, '_tdf', sdf)
	if tdf.centers is None:
		raise ValueError('the field has no tori: fit them first')
	points, normals = tdf.points, tdf.normals
	n = points.shape[1]
	lam = float(tdf.lambda_scale if lambda_scale is None else lambda_scale)

	bvh = build_lbvh(points, leaf_size)
	p4 = torch.cat([points.T, tdf.is_outlier.to(points.dtype)[:, None]], dim=1)
	tori = torch.cat([tdf.centers.T, tdf.major_radii[:, None], tdf._axes3d.T, tdf.minor_radii[:, None]], dim=1)
	nrm = normals / torch.linalg.vector_norm(normals, dim=0, keepdim=True).clamp_min(1e-30)
	n4 = torch.cat([nrm.T, torch.zeros(n, 1, dtype=nrm.dtype, device=nrm.device)], dim=1)

	lo, hi = points.amin(dim=1), points.amax(dim=1)
	diag = torch.linalg.vector_norm(hi - lo)
	lo, hi = (lo - padding * diag).cpu().numpy(), (hi + padding * diag).cpu().numpy()
	if point_radius is None:
		# mean spacing of N points on a surface of area ~ the box's largest face: sqrt(A / N)
		ext = np.sort(hi - lo)
		point_radius = 0.4 * float(np.sqrt(ext[1] * ext[2] / max(n, 1)))

	scene = Scene(lo, hi, meta=dict(meta or {}, source='points_as_tori', n_points=int(n)))
	gpu = bvh.gpu_arrays()
	scene.add_array(f'{name}/points', p4.to(torch.float32).cpu().numpy())
	scene.add_array(f'{name}/normals', n4.to(torch.float32).cpu().numpy())
	scene.add_array(f'{name}/tori', tori.to(torch.float32).cpu().numpy())
	for key, a in gpu.items():
		scene.add_array(f'{name}/{key}', a)
	scene.add_field(
		{
			'name': name,
			'type': 'pat',
			'params': {
				'k': int(k),
				'lambda_scale': lam,
				'max_exp_arg': 64.0,
				'isovalue': 0.0,
				'bvh_depth': int(bvh.depth.max()),
				'bvh_leaf_size': int(leaf_size),
				# neighborhood the network predicted each torus from (tori inspector); absent for tori read from a file
				**({'network_k': int(sdf.network_k)} if getattr(sdf, 'network_k', None) else {}),
			},
			'arrays': {key: f'{name}/{key}' for key in ('points', 'tori', 'bvh_bounds', 'bvh_nodes', 'bvh_prims')},
		}
	)
	scene.add_pointcloud(f'{name}_points', f'{name}/points', f'{name}/normals', radius=point_radius)

	if bake_resolution:
		ext = hi - lo
		res = np.maximum(2, np.round(bake_resolution * ext / ext.max()).astype(int))
		k_saved, lam_saved = tdf.k_evaluation, tdf.lambda_scale
		tdf.set_k_evaluation(k)
		tdf.set_lambda_scale(lam)
		try:
			values = bake_grid(
				lambda q: tdf.evaluate_distance(q.to(points.dtype)),
				lo,
				hi,
				res,
				device=points.device,
				dtype=points.dtype,
			)
		finally:
			tdf.set_k_evaluation(k_saved)
			tdf.set_lambda_scale(lam_saved)
		grid_field(scene, f'{name}_grid', values, lo, hi)
	return scene
