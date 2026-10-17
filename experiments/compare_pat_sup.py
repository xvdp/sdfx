"""
Fit the same shapes with Points as Tori (PAT) and SuperFit (SUP) and write both fits into one scene per shape, to be
toggled or overlaid in the viewer; metrics optional (--metrics).

	# PAT fitted here + SuperFit assemblies (.pkl) fitted beforehand; --mesh and --suppkl pair in order
	python experiments/compare_pat_sup.py --mesh data/a.obj data/b.ply --suppkl fits/a/final_content.pkl fits/b/final_content.pkl -o out/
	# an assembly alone: the mesh is the one stored in it (the path SuperFit was given), if it exists here
	python experiments/compare_pat_sup.py --suppkl dataset/partobjaverse/superfrustum/<id>/primitive_assembly.pkl -o out/
	# a mesh alone: PAT and the ground truth (a warning prints how to fit SuperFit to it)
	python experiments/compare_pat_sup.py --mesh data/a.obj -o out/
	# view: G cycles the opaque model, V overlays the others, T makes them translucent
	python -m sdfview view out/<shape>.sdfx

SuperFit is not run here: fit it with its own script (examples/superfit/03_fit_superfit.sh: scripts/mesh_to_assembly.py
from ext/superfit, in an environment with SuperFit's dependencies and a CUDA GPU) and pass the result with --suppkl. A
shape is named after its mesh file (after its folder when the file is mesh.*, as in Toys4K).

Per shape, everything in SuperFit's normalized frame (the mesh read as SuperFit reads it, trimesh force='mesh',
process=False, then p_n = s (p - c), s = 1.8 / max bounding-box extent, c = bounding-box centre):
  pat  oriented point cloud of --pat-points samples, uniform by area on the normalized mesh with face normals
       (optionally with Gaussian position noise --pat-noise), then PointsAsTori (pre-trained network + torus fit);
       the scene holds its native field (tori + BVH, k = 32), a grid bake of it and the input points;
  sup  SuperFit's assembly (--suppkl) as superfit_sdf.SuperFitSDF (deterministic argmax reduction): generated GLSL
       (the composition of its primitives and each primitive alone; keys [ ] and U in the viewer) and a grid bake;
  gt   the input mesh's signed distance baked on a grid (--gt: cubvh 'raystab' as SuperFit, or the generalized winding
       number; 'none' = no gt model).
Output: <out>/<shape>.sdfx, the three scenes merged (sdfview.scene.merge_scenes; fields 'pat.pat', 'pat.pat_grid',
'sup.sup', 'sup.sup_grid', 'gt'; point cloud 'pat.pat_points'); the fit times and counts are in the scene meta
(python -m sdfview info).

--metrics also writes <out>/metrics.jsonl, one row per shape and model, with SuperFit's definitions
(algos/eval_tools.py) where they apply:
	iou@R          voxel IoU of {φ ≤ 0} on the R³ voxel centres (2i + 1)/R - 1 of [-1, 1]³ (SuperFit: R = 128)
	iou_gt_surface IoU of {φ ≤ 0} on 100k points p + n ⊙ u near the GT surface: p uniform by area, n its face normal,
	               u ~ U(-0.05, 0.05)³ elementwise (SuperFit's sample_surface_proximal_points)
	iou_pred_surface  the same around the model's surface (marching cubes of its R³ grid)
	bidir_surface_iou mean of the two
	cd_x100        Chamfer: 2048 points on each surface (GT mesh, model's marching-cubes mesh), mean squared nearest
	               distance in each direction, averaged, × 100 (SuperFit's CD_AVG@2048); its floor is the sampling:
	               the GT against its own 32³ marching-cubes surface gives 0.11 on knot.obj
	sdf_mae / sdf_median_ae  |φ - φ_gt| on the near-surface points
	normal_cos / normal_abs_cos  mean cos and |cos| between ∇φ (central differences, h = 1e-3) and the GT face normal
	               at GT surface samples
Differences from SuperFit's own numbers: SuperFit evaluates against its remeshed watertight target (cubvh + fast
sweeping + FlexiCubes), not the input mesh, and takes the best of ~100 Gumbel samples of the assembly; here the input
mesh and the argmax assembly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO)

import points_as_tori as pat  # noqa: E402
from points_as_tori import config  # noqa: E402
from points_as_tori.bvh import BVH  # noqa: E402
from points_as_tori.isosurface import marching_cubes  # noqa: E402
from points_as_tori.shape_3d import TriangleMesh  # noqa: E402
from points_as_tori.training.mesh_sampling import gwn_mesh  # noqa: E402

NEAR_SURFACE_POINTS = 100_000
NEAR_SURFACE_JITTER = 0.05
CHAMFER_POINTS = 2048
FD_STEP = 1e-3


# ----------------------------------------------------------------------------------------------- mesh, ground truth


def read_mesh(path: str):
	import trimesh

	mesh = trimesh.load(path, force='mesh', process=False)
	return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def normalize(vertices: np.ndarray):
	from superfit_sdf import mesh_normalization

	s, c = mesh_normalization(vertices)
	return (vertices - c) * s, s, c


class GroundTruth:
	"""Signed distance to the (normalized) input mesh: (3, Q) -> (Q,), negative inside."""

	def __init__(self, vertices: np.ndarray, faces: np.ndarray, method: str, device: torch.device):
		self.device = device
		if method == 'auto':
			try:
				import cubvh  # noqa: F401

				method = 'cubvh' if device.type == 'cuda' else 'gwn'
			except ImportError:
				method = 'gwn'
		self.method = method
		if method == 'cubvh':
			import cubvh

			self._bvh = cubvh.cuBVH(vertices.astype(np.float32), faces.astype(np.int32))
		else:
			self.V = config.as_wide(vertices, 3)
			self.F = config.as_wide(faces, 3, dtype=torch.long)
			self._bvh = BVH(self.V, self.F)
			self._mask = torch.ones(self.F.shape[1], dtype=torch.bool, device=self.F.device)

	def __call__(self, q: torch.Tensor) -> torch.Tensor:
		if self.method == 'cubvh':
			d, _, _ = self._bvh.signed_distance(q.T.contiguous().float(), return_uvw=False, mode='raystab')
			return torch.as_tensor(d, device=q.device).to(q.dtype)
		out = torch.empty(q.shape[1], dtype=q.dtype, device=q.device)
		chunk = max(1, config.CHUNK_ELEMENTS // max(1, self.F.shape[1]))
		_, dist, _ = self._bvh.closest_point_on_mesh(q)
		for s in range(0, q.shape[1], chunk):
			w = gwn_mesh(q[:, s : s + chunk], self.V, self.F, self._mask) / (4 * np.pi)
			out[s : s + chunk] = torch.where(w > 0.5, -dist[s : s + chunk], dist[s : s + chunk])
		return out


def surface_samples(vertices, faces, n: int, seed: int):
	"""vertices (V, 3), faces (F, 3) (NumPy, or torch on the device) -> (points (3, n), unit face normals (3, n)) on
	the device, uniform by area."""
	mesh = TriangleMesh(vertices, faces)
	p, nrm = mesh.sample_points_and_normals_uniformly(n, seed=seed)
	return p, nrm / nrm.norm(dim=0, keepdim=True).clamp_min(1e-12)


def near_surface(points, normals, seed: int):
	gen = torch.Generator(device='cpu').manual_seed(seed)
	u = (torch.rand(points.shape, generator=gen, dtype=points.dtype) * 2 - 1) * NEAR_SURFACE_JITTER
	return points + normals * u.to(points.device)


def voxel_centres(res: int, device, dtype):
	x = (2 * torch.arange(res, device=device, dtype=dtype) + 1) / res - 1
	X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
	return torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]), x


def evaluate_chunked(fn, q: torch.Tensor, chunk: int = 1 << 20) -> torch.Tensor:
	return torch.cat([fn(q[:, s : s + chunk]).reshape(-1).to(q.dtype) for s in range(0, q.shape[1], chunk)])


# ----------------------------------------------------------------------------------------------- metrics


def iou(a: torch.Tensor, b: torch.Tensor) -> float:
	union = (a | b).sum()
	return float((a & b).sum() / union) if int(union) > 0 else float('nan')


def chamfer_x100(a: torch.Tensor, b: torch.Tensor) -> float:
	"""a (3, n), b (3, m): (mean_a min_b |a - b|² + mean_b min_a |a - b|²) / 2 × 100."""
	d2 = torch.cdist(a.T.double(), b.T.double()) ** 2
	return float((d2.min(dim=1).values.mean() + d2.min(dim=0).values.mean()) / 2 * 100)


def gradient_fd(fn, q: torch.Tensor, h: float = FD_STEP) -> torch.Tensor:
	g = []
	for i in range(3):
		e = torch.zeros(3, 1, dtype=q.dtype, device=q.device)
		e[i] = h
		g.append((evaluate_chunked(fn, q + e) - evaluate_chunked(fn, q - e)) / (2 * h))
	return torch.stack(g)


def model_metrics(fn, gt, ctx: dict) -> dict:
	"""All metrics of one model `fn` ((3, Q) -> (Q,)) against the ground truth `gt` (see the module docstring)."""
	res, seed = ctx['res'], ctx['seed']
	out = {}
	t = time.perf_counter()
	values = evaluate_chunked(fn, ctx['grid'])
	if values.is_cuda:
		torch.cuda.synchronize(values.device)
	out['time_grid'] = time.perf_counter() - t
	occ = values <= 0
	out[f'iou@{res}'] = iou(occ, ctx['gt_occ'])

	near_phi = evaluate_chunked(fn, ctx['near'])
	out['iou_gt_surface'] = iou(near_phi <= 0, ctx['near_gt'] <= 0)
	err = (near_phi - ctx['near_gt']).abs()
	out['sdf_mae'], out['sdf_median_ae'] = float(err.mean()), float(err.median())

	p, n = ctx['gt_surface']
	g = gradient_fd(fn, p)
	cos = (g * n).sum(dim=0) / g.norm(dim=0).clamp_min(1e-12)
	out['normal_cos'], out['normal_abs_cos'] = float(cos.mean()), float(cos.abs().mean())

	if bool(occ.any()) and not bool(occ.all()):
		axis = ctx['axis']
		v, f = marching_cubes(values.reshape(res, res, res), axis[[0, 0, 0]], axis[[-1, -1, -1]], 0.0)  # (3, V), (3, F)
		out['triangles'] = int(f.shape[1])
		model_points, _ = surface_samples(v.T, f.T, CHAMFER_POINTS, seed + 1)  # (n, 3) views: stays on the device
		out['cd_x100'] = chamfer_x100(model_points, ctx['gt_chamfer'])
		pp, pn = surface_samples(v.T, f.T, NEAR_SURFACE_POINTS, seed + 3)
		pred_near = near_surface(pp, pn, seed + 4)
		out['iou_pred_surface'] = iou(evaluate_chunked(fn, pred_near) <= 0, evaluate_chunked(gt, pred_near) <= 0)
		out['bidir_surface_iou'] = (out['iou_gt_surface'] + out['iou_pred_surface']) / 2
	else:  # empty or full grid: no surface
		out['cd_x100'] = out['iou_pred_surface'] = out['bidir_surface_iou'] = float('nan')
	return out


# ----------------------------------------------------------------------------------------------- models


def run_pat(vertices, faces, n_points: int, seed: int, noise: float):
	p, n = surface_samples(vertices, faces, n_points, seed)
	if noise > 0:
		gen = torch.Generator(device='cpu').manual_seed(seed + 10)
		p = p + noise * torch.randn(p.shape, generator=gen, dtype=p.dtype).to(p.device)
	t = time.perf_counter()
	sdf = pat.PointsAsTori(p.T, n.T)  # (n, 3) views of the (3, n) device tensors: no copy, no host transfer
	if p.is_cuda:
		torch.cuda.synchronize(p.device)
	return sdf, time.perf_counter() - t


def stored_mesh(suppkl: str):
	"""(path, exists): the mesh a SuperFit assembly was fitted to, as stored in its pickle (input_mesh_file from
	SuperFit's dataset scripts, input_path from mesh_to_assembly.py)."""
	from superfit_sdf import load_assembly

	data = load_assembly(suppkl)
	stored = data.get('input_mesh_file') or data.get('input_path')
	return stored, bool(stored) and os.path.isfile(stored)


def superfit_hint(name: str, mesh: str) -> str:
	"""What to run to fit SuperFit to `mesh` and add it (printed when a mesh comes without --suppkl)."""
	pkl = os.path.join(REPO, 'examples', 'out', 'superfit', name, 'final_content.pkl')
	return (
		f'  SUP: no --suppkl for {name}: this scene holds PAT and the ground truth only. To fit SuperFit to this mesh '
		"(in an environment with SuperFit's dependencies, ext/superfit/env.yml, and a CUDA GPU):\n"
		f'    bash {os.path.join(REPO, "examples", "superfit", "03_fit_superfit.sh")} {mesh}\n'
		f'  then compare with it: --mesh {mesh} --suppkl {pkl}'
	)


# ----------------------------------------------------------------------------------------------- driver


def gt_scene(gt, res: int, device, dtype, meta: dict):
	from sdfview.scene import Scene, grid_field
	from sdfview.sources.grid import bake_grid

	lo, hi = [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]
	scene = Scene(bounds_min=lo, bounds_max=hi, meta=dict(meta, source='ground truth', method=gt.method))
	grid_field(scene, 'gt', bake_grid(gt, lo, hi, res, device=device, dtype=dtype), lo, hi)
	return scene


def metrics_context(gt, Vn, F, res: int, seed: int, device, dtype) -> dict:
	grid, axis = voxel_centres(res, device, dtype)
	gt_surface = surface_samples(Vn, F, NEAR_SURFACE_POINTS, seed)
	near = near_surface(*gt_surface, seed + 5)
	return {'res': res, 'seed': seed, 'grid': grid, 'axis': axis, 'gt_occ': evaluate_chunked(gt, grid) <= 0,
			'gt_surface': gt_surface, 'near': near, 'near_gt': evaluate_chunked(gt, near),
			'gt_chamfer': surface_samples(Vn, F, CHAMFER_POINTS, seed + 2)[0]}  # fmt: skip


def process(name, mesh_path, assembly, a, device, dtype, out_dir):
	"""Fit / load the models of one shape and write <out_dir>/<name>.sdfx; returns (scene path, metric rows)."""
	from sdfview.scene import merge_scenes

	V, F = read_mesh(mesh_path)
	Vn, s, c = normalize(V)
	gt = GroundTruth(Vn, F, a.gt, device) if a.gt != 'none' else None
	ctx = metrics_context(gt, Vn, F, a.res, a.seed, device, dtype) if a.metrics else None
	common = {'shape': name, 'mesh': mesh_path, 'faces': int(F.shape[0]), 'scale': s}
	scenes, rows = {}, []

	if not a.skip_pat:
		from sdfview.sources.pat import scene_from_pat

		sdf, t_fit = run_pat(Vn, F, a.pat_points, a.seed, a.pat_noise)
		print(f'  PAT: {a.pat_points} points, {t_fit:.2f} s', flush=True)
		meta = {'input': mesh_path, 'pat_points': a.pat_points, 'pat_noise': a.pat_noise, 'time_fit': t_fit}
		scenes['pat'] = scene_from_pat(sdf, bake_resolution=a.res, meta=meta)
		if ctx is not None:
			rows.append({**common, 'model': 'PAT', 'points': a.pat_points, 'time_fit': t_fit,
						 **model_metrics(sdf.signed_distance, gt, ctx)})  # fmt: skip

	if assembly is not None:
		from sdfview.sources.superfit import scene_from_superfit
		from superfit_sdf import SuperFitSDF

		sup = SuperFitSDF(assembly, program=a.program, device=device, dtype=dtype)
		fit_time = sup.data.get('total_time')
		fit_time = float(fit_time) if fit_time is not None else float('nan')
		print(f'  SUP: {sup.n_primitives} primitives ({sup.program}), fit {fit_time:.1f} s', flush=True)
		meta = {'input': assembly, 'mesh': mesh_path, 'time_fit': fit_time}
		scenes['sup'] = scene_from_superfit(sup, resolution=a.res, meta=meta, name='sup')  # GLSL + grid bake
		if ctx is not None:
			rows.append({**common, 'model': 'SUP', 'assembly': assembly, 'program': sup.program,
						 'primitives': sup.n_primitives, 'time_fit': fit_time, **model_metrics(sup, gt, ctx)})  # fmt: skip

	if gt is not None:
		scenes['gt'] = gt_scene(gt, a.res, device, dtype, {'input': mesh_path})
	if not scenes:
		return None, rows
	frame = {'frame': 'superfit_normalized', 'scale': float(s), 'center': [float(x) for x in c]}
	scene = merge_scenes(scenes, meta={'input': mesh_path, **frame})
	path = os.path.join(out_dir, f'{name}.sdfx')
	scene.save(path)
	return path, rows


def shape_name(mesh: str) -> str:
	"""The mesh's file name without extension; its folder's name when the file is mesh.* (Toys4K: <shape>/mesh.obj)."""
	stem = os.path.splitext(os.path.basename(mesh))[0]
	return os.path.basename(os.path.dirname(os.path.abspath(mesh))) if stem == 'mesh' else stem


def shapes(a):
	"""(name, mesh, suppkl or None) per shape: --mesh and --suppkl pair by position; with --suppkl alone, each
	assembly's stored mesh."""
	meshes, pkls = list(a.mesh or []), list(a.suppkl or [])
	if meshes and pkls and len(meshes) != len(pkls):
		raise SystemExit(f'--mesh ({len(meshes)}) and --suppkl ({len(pkls)}) pair one to one, in order')
	if not meshes:
		for pkl in pkls:
			stored, exists = stored_mesh(pkl)
			if not exists:
				raise SystemExit(
					f'{pkl}: the mesh it was fitted to ({stored}) is not on this machine; give it with --mesh'
				)
			meshes.append(stored)
	for mesh, pkl in list(zip(meshes, pkls or [None] * len(meshes)))[: a.limit or None]:
		yield shape_name(mesh), mesh, pkl


def main(argv=None):
	ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
	ap.add_argument('--mesh', nargs='+', help='input meshes (any format trimesh reads: obj, ply, glb, stl, off, ...)')
	ap.add_argument('--suppkl', nargs='+', help='SuperFit assemblies (.pkl), paired in order with --mesh')
	ap.add_argument('--program', default='best', help="assembly program: 'best', 'latest' or a key")
	ap.add_argument('-o', '--out', required=True, help='output directory')
	ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='PAT / evaluation device')
	ap.add_argument('--float64', action='store_true', help='float64 geometry (default float32)')
	ap.add_argument('--gt', default='auto', choices=['auto', 'cubvh', 'gwn', 'none'], help='ground-truth model')
	ap.add_argument('--res', type=int, default=128, help='grid resolution of the bakes (and of the metrics)')
	ap.add_argument('--pat-points', type=int, default=50_000, help='PAT input point cloud size')
	ap.add_argument('--pat-noise', type=float, default=0.0, help='Gaussian position noise of the PAT input')
	ap.add_argument('--skip-pat', action='store_true')
	ap.add_argument('--metrics', action='store_true', help='also compute metrics against the ground truth')
	ap.add_argument('--limit', type=int, default=0, help='first N shapes only')
	ap.add_argument('--seed', type=int, default=0)
	a = ap.parse_args(argv)
	if not (a.mesh or a.suppkl):
		ap.error('give --mesh and / or --suppkl')
	if a.metrics and a.gt == 'none':
		ap.error('--metrics needs a ground truth (--gt auto, cubvh or gwn)')

	dtype = torch.float64 if a.float64 else torch.float32
	pat.set_device(a.device, dtype)
	device = config.get_device()
	os.makedirs(a.out, exist_ok=True)
	results = os.path.join(a.out, 'metrics.jsonl')
	all_rows, written = [], []
	without_sup = []
	for name, mesh, assembly in shapes(a):
		print(f'--- {name}: {mesh}' + (f' | {assembly}' if assembly else ''), flush=True)
		if assembly is None:
			print(superfit_hint(name, mesh), flush=True)
			without_sup.append(name)
		try:
			path, rows = process(name, mesh, assembly, a, device, dtype, a.out)
		except Exception:  # noqa: BLE001  (one bad shape does not stop a dataset run)
			import traceback

			print(f'  FAILED {name}:', flush=True)
			traceback.print_exc()
			continue
		if path:
			written.append(path)
			print(f'  wrote {path}', flush=True)
		if rows:
			with open(results, 'a') as f:
				for row in rows:
					f.write(json.dumps(row) + '\n')
			all_rows += rows
			for row in rows:
				print('  ' + ', '.join(f'{k} {v:.4g}' if isinstance(v, float) else f'{k} {v}' for k, v in row.items()
										if k not in ('mesh', 'assembly', 'shape')), flush=True)  # fmt: skip
	if all_rows:
		summary(all_rows, a.res)
		print(f'wrote {results}')
	if written:
		print(f'view: python -m sdfview view {written[0]}   (G: opaque model, V: overlay, T: translucent)')
	if without_sup:
		print(f'without SuperFit (no --suppkl): {", ".join(without_sup)}')


def summary(rows, res):
	keys = [f'iou@{res}', 'bidir_surface_iou', 'cd_x100', 'sdf_mae', 'normal_abs_cos', 'time_fit', 'time_grid']
	for model in ('PAT', 'SUP'):
		sel = [r for r in rows if r['model'] == model]
		if not sel:
			continue
		means = {k: np.nanmean([r.get(k, np.nan) for r in sel]) for k in keys}
		print(f'{model} ({len(sel)} shapes, means): ' + ', '.join(f'{k} {v:.4g}' for k, v in means.items()))


if __name__ == '__main__':
	main()
