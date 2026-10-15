"""
Stage timings of the Points as Tori port on one device and dtype.

	python benchmarks/port_bench.py --device cuda:1                      # float32 (the GPU default)
	python benchmarks/port_bench.py --device cuda:1 --dtype float64      # the C++'s precision
	python benchmarks/port_bench.py --device cpu --threads 8

Stages (one JSON line each, seconds; `knn` = the k-NN backend config.KNN_BACKEND resolves to):
	inference   per point cloud (tooth.ply, 2048 points, and N points on a unit sphere): k-NN (k = 64, the network's
	            input), network coefficients (float32 network, chunks of 50k neighborhoods), torus fit, total
	evaluation  Q queries in the bounding box: distance (k = 32, PointsAsTori's accelerated mode), distance + gradient,
	            distance + gradient + Laplacian
	meshing     mesh_isocontour (grid evaluation + marching cubes) at the given resolutions
	training    one train_step of the full-size network (embed 128, 8 layers, 8 heads, mlp 512: the pre-trained model)
	            on a shape of 2048 points (max_points_per_pointcloud), k = 64, batch of 1024 neighborhoods × 120 queries
Every stage runs once untimed first (allocator, kernels, slangpy compilation) and reports the best of --reps runs;
CUDA is synchronized around each timed run. Library output (progress prints) is suppressed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
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
from points_as_tori.infer import DEFAULT_MODEL_PATH  # noqa: E402
from points_as_tori.isosurface import mesh_isocontour  # noqa: E402
from points_as_tori.network import FundamentalFormPredictor  # noqa: E402
from points_as_tori.shape_3d import fit_tori_from_forms  # noqa: E402

TOOTH = os.path.join(REPO, 'ext', 'points-as-tori', 'data', 'tooth.ply')


def sync(device):
	if device.type == 'cuda':
		torch.cuda.synchronize(device)


def timed(fn, device, reps):
	with contextlib.redirect_stdout(io.StringIO()):
		out = fn()  # warm-up
		sync(device)
		best = float('inf')
		for _ in range(reps):
			t = time.perf_counter()
			out = fn()
			sync(device)
			best = min(best, time.perf_counter() - t)
	return best, out


def sphere_cloud(n, seed=0):
	p = np.random.default_rng(seed).normal(size=(n, 3))
	p /= np.linalg.norm(p, axis=1, keepdims=True)
	return p, p.copy()


def emit(row):
	print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)


def bench_inference(name, points, normals, device, reps, queries, grids):
	from points_as_tori.torus_distance import _knn_backend

	with contextlib.redirect_stdout(io.StringIO()):
		model, k = FundamentalFormPredictor.load_saved_model(DEFAULT_MODEL_PATH)
	model.eval()
	P, N = pat.as_wide(points, 3), pat.as_wide(normals, 3)
	n = P.shape[1]
	row = {'stage': 'inference', 'cloud': name, 'N': n, 'knn': _knn_backend(n, device)}
	row['k-NN (k=64)'], _ = timed(lambda: pat.get_neighbors(P, k), device, reps)

	def coefficients():
		with torch.no_grad():
			return model.precompute_coefficients_in_chunks(P, N, k, chunk_size=50000)

	t_net, coeffs = timed(coefficients, device, reps)
	row['network incl. k-NN'] = t_net
	row['torus fit'], tori = timed(lambda: fit_tori_from_forms(P, N, coeffs), device, reps)
	row['total'] = t_net + row['torus fit']
	emit(row)

	tdf = pat.TorusDistanceField(P, N)
	tdf.set_tori(*tori)
	tdf.set_k_evaluation(32)
	lo, hi = P.amin(dim=1, keepdim=True), P.amax(dim=1, keepdim=True)
	pad = 0.1 * (hi - lo)
	for q_count in queries:
		gen = torch.Generator(device='cpu').manual_seed(1)
		u = torch.rand(3, q_count, generator=gen, dtype=P.dtype).to(device)
		q = (lo - pad) + u * (hi - lo + 2 * pad)
		row = {'stage': 'evaluation', 'cloud': name, 'N': n, 'Q': q_count, 'k': 32, 'knn': _knn_backend(n, device)}
		row['distance'], _ = timed(lambda: tdf.evaluate_distance(q), device, reps)
		row['+ gradient'], _ = timed(lambda: tdf.evaluate_distance_gradient(q), device, reps)
		row['+ Laplacian'], _ = timed(lambda: tdf.evaluate_distance_gradient_laplacian(q), device, reps)
		emit(row)
	for res in grids:
		t, (vertices, faces) = timed(lambda: mesh_isocontour(tdf, 0.0, res), device, 1)
		emit({'stage': 'meshing', 'cloud': name, 'N': n, 'grid': res, 'seconds': t, 'triangles': int(faces.shape[1])})


def bench_training(device, reps, n_points=2048, k=64, batch=1024, n_queries=120):
	from points_as_tori.training import train as ptrain

	with contextlib.redirect_stdout(io.StringIO()):
		model, optimizer, train_step = ptrain.FundamentalFormPredictorTrainer(
			embed_dim=128, n_freq=0, n_layers=8, n_heads=8, mlp_dim=512, learning_rate=1e-5, warmup_steps=0
		)
	points, normals = sphere_cloud(n_points, seed=2)
	P, N = pat.as_wide(points, 3), pat.as_wide(normals, 3)
	neighbors = pat.get_neighbors(P, k)
	mask = torch.ones(n_points, dtype=torch.bool, device=device)
	batch_mask = torch.ones(batch, dtype=torch.bool, device=device)
	gen = torch.Generator(device='cpu').manual_seed(3)
	queries = (torch.rand(3, batch, n_queries, generator=gen, dtype=P.dtype) * 2 - 1).to(device)
	true_distances = (queries.norm(dim=0) - 1.0).contiguous()
	args = (mask, batch_mask, P, N, neighbors, queries, true_distances)
	row = {'stage': 'training', 'N': n_points, 'k': k, 'batch': batch, 'queries': n_queries}
	try:
		row['train_step'], _ = timed(lambda: train_step(model, optimizer, *args), device, reps)
	except torch.OutOfMemoryError as e:
		row['train_step'] = f'out of memory: {str(e).splitlines()[0][:120]}'
	if device.type == 'cuda':
		row['peak memory (GB)'] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
	emit(row)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--device', default='cpu')
	ap.add_argument('--dtype', default='float32', choices=['float32', 'float64'])
	ap.add_argument('--points', type=int, nargs='+', default=[100_000], help='sizes of the sphere clouds')
	ap.add_argument('--queries', type=int, nargs='+', default=[100_000, 1_000_000])
	ap.add_argument('--grids', type=int, nargs='+', default=[64, 128])
	ap.add_argument('--reps', type=int, default=2)
	ap.add_argument('--threads', type=int, default=os.cpu_count())
	ap.add_argument('--skip-training', action='store_true')
	ap.add_argument(
		'--train-batch', type=int, default=1024, help='neighborhoods per training step (the original: 1024)'
	)
	a = ap.parse_args()
	torch.set_num_threads(a.threads)
	pat.set_device(a.device, getattr(torch, a.dtype))
	device = config.get_device()
	name = torch.cuda.get_device_name(device) if device.type == 'cuda' else f'CPU, {a.threads} threads'
	print(f'# torch {torch.__version__}, {device} ({name}), {a.dtype}, KNN_BACKEND={config.KNN_BACKEND!r}', flush=True)

	points, normals = pat.read_point_cloud(TOOTH)
	bench_inference('tooth', points, normals, device, a.reps, a.queries, a.grids)
	for n in a.points:
		points, normals = sphere_cloud(n)
		bench_inference('sphere', points, normals, device, a.reps, a.queries, a.grids)
	if not a.skip_training:
		bench_training(device, a.reps, batch=a.train_batch)


if __name__ == '__main__':
	main()
