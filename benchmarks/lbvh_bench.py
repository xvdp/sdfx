"""
LBVH (points_as_tori.bvh, torch) vs the original SAH BVH (C++, ext/points-as-tori/src/cpp/bvh.cpp):
build time, peak memory, tree size / depth, and k-NN traversal cost (nodes popped and points tested per query,
through the same algorithm as the shader). Also the torch brute-force k-NN the evaluator uses, for scale.

	python benchmarks/lbvh_bench.py                     # CPU (+ CUDA if available), N = 1e4, 1e5, 1e6
	python benchmarks/lbvh_bench.py --sizes 1e4 1e5 1e6 4e6 --device cuda
	PAT_REFERENCE_DIR=build/pat_reference python benchmarks/lbvh_bench.py   # adds the C++ SAH BVH columns

Points: N samples on the unit sphere (a surface, as point clouds are), queries near the surface.
CPU peak memory: high-water mark of the resident set (VmHWM) in a fresh subprocess, minus its value after the inputs
exist; CUDA: torch.cuda.max_memory_allocated() minus the inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO)

from points_as_tori.bvh import build_lbvh, from_cpp_layout, knn_reference  # noqa: E402


def sphere_points(n: int, seed: int = 0) -> np.ndarray:
	p = np.random.default_rng(seed).normal(size=(n, 3)).astype(np.float32)
	return p / np.linalg.norm(p, axis=1, keepdims=True)


def _vm_hwm_kb() -> int:
	with open('/proc/self/status') as f:
		for line in f:
			if line.startswith('VmHWM'):
				return int(line.split()[1])
	return -1


def _reset_hwm() -> None:
	try:
		with open('/proc/self/clear_refs', 'w') as f:
			f.write('5')  # resets the peak RSS counter (Linux >= 4.0)
	except OSError:
		pass


def child(n: int, device: str, threads: int) -> dict:
	"""One build in this process: time (median of 3) and peak memory."""
	torch.set_num_threads(threads)
	P = torch.from_numpy(sphere_points(n).T.copy()).to(device)
	build_lbvh(P[:, : min(n, 10000)])  # warm-up (kernels, allocator)
	if device.startswith('cuda'):
		torch.cuda.synchronize()
		torch.cuda.reset_peak_memory_stats()
		base = torch.cuda.memory_allocated()
	else:
		_reset_hwm()
		base = _vm_hwm_kb() * 1024
	times = []
	for _ in range(3):
		t = time.perf_counter()
		b = build_lbvh(P)
		if device.startswith('cuda'):
			torch.cuda.synchronize()
		times.append(time.perf_counter() - t)
	peak = (torch.cuda.max_memory_allocated() if device.startswith('cuda') else _vm_hwm_kb() * 1024) - base
	return {'time': float(np.median(times)), 'peak_mb': peak / 2**20, 'nodes': b.n_nodes, 'depth': int(b.depth.max())}


def brute_force_knn_time(n: int, device: str, n_queries: int = 10_000, k: int = 32) -> float:
	"""The torch evaluator's k-NN (GEMM + topk, chunked as in points_as_tori)."""
	from points_as_tori import config
	from points_as_tori.torus_distance import _chunk_ranges, _knn, _query_chunk_size

	P = torch.from_numpy(sphere_points(n).T.copy()).to(device)
	Q = P[:, torch.randint(0, n, (n_queries,), device=device)] * 1.01
	sq = (P * P).sum(0)
	t = time.perf_counter()
	for s, e in _chunk_ranges(n_queries, _query_chunk_size(n)):
		_knn(Q[:, s:e], P, k, sq)
	if device.startswith('cuda'):
		torch.cuda.synchronize()
	del config
	return time.perf_counter() - t


def traversal_cost(bvh, points: np.ndarray, k: int = 32, n_queries: int = 200, seed: int = 1) -> dict:
	rng = np.random.default_rng(seed)
	q = points[rng.integers(0, len(points), n_queries)] * (1 + 0.02 * rng.normal(size=(n_queries, 1)))
	stats, host = {}, {}
	for x in q:
		knn_reference(bvh, points, x.astype(np.float64), k, stats, host)
	return {key: v / n_queries for key, v in stats.items()}


def load_reference():
	ref_dir = os.environ.get('PAT_REFERENCE_DIR', os.path.join(REPO, 'build', 'pat_reference'))
	sys.path.insert(0, ref_dir)
	try:
		import pat_reference

		return pat_reference
	except ImportError:
		return None


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--sizes', type=float, nargs='+', default=[1e4, 1e5, 1e6])
	ap.add_argument('--device', nargs='+', default=None, help='default: cpu (+ cuda if available)')
	ap.add_argument('--threads', type=int, default=os.cpu_count())
	ap.add_argument('--no-traversal', action='store_true', help='skip the (slow, Python) traversal-cost columns')
	ap.add_argument('--child', nargs=2, help=argparse.SUPPRESS)
	a = ap.parse_args()
	if a.child:
		print(json.dumps(child(int(float(a.child[0])), a.child[1], a.threads)))
		return
	devices = a.device or (['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
	ref = load_reference()
	print(f'torch {torch.__version__}, {a.threads} CPU threads, devices {devices}, C++ reference: {bool(ref)}')
	for n in (int(float(s)) for s in a.sizes):
		row = {'N': n}
		for dev in devices:
			out = subprocess.run(
				[sys.executable, __file__, '--child', str(n), dev, '--threads', str(a.threads)],
				capture_output=True,
				text=True,
				check=True,
			)
			r = json.loads(out.stdout.strip().splitlines()[-1])
			row[f'lbvh_{dev}'] = f'{r["time"]:.3f} s, {r["peak_mb"]:.0f} MB'
			row['nodes'], row['depth'] = r['nodes'], r['depth']
			row[f'bruteforce_knn_10k_queries_{dev}'] = f'{brute_force_knn_time(n, dev):.2f} s'
		pts = sphere_points(n)
		if ref is not None:
			t = time.perf_counter()
			c = ref.BVH(np.asfortranarray(pts.T.astype(np.float64)))
			row['sah_cpp'] = f'{time.perf_counter() - t:.3f} s'
			nd, idx = c.get_gpu_data()
			sah = from_cpp_layout(nd, idx)
			row['sah_nodes'], row['sah_depth'] = sah.n_nodes, int(sah.depth.max())
		if not a.no_traversal:
			pts64 = pts.astype(np.float64)
			lb = build_lbvh(torch.from_numpy(pts64.T.copy()))
			row['knn32_cost_lbvh'] = traversal_cost(lb, pts64)
			if ref is not None:
				row['knn32_cost_sah'] = traversal_cost(sah, pts64)
		print(json.dumps(row))


if __name__ == '__main__':
	main()
