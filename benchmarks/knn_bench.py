"""
k-nearest-neighbor backends for Points as Tori, measured on the same data.

	python benchmarks/knn_bench.py                               # CPU
	python benchmarks/knn_bench.py --device cuda --sizes 1e4 1e5 1e6 4e6
	python benchmarks/knn_bench.py --device cuda:1 --backends torch-bvh slang   # one GPU, some backends
	PAT_REFERENCE_DIR=build/pat_reference python benchmarks/knn_bench.py   # adds the original C++ (nanoflann)

Two workloads, as PAT uses them:
	fit   every point's k = 64 nearest points (network input, torus fitting): Q = N
	eval  k = 32 nearest points of Q = 100k query points near the surface (SDF evaluation)

Backends (each runs only if importable; all exact except where noted):
	torch-brute   points_as_tori brute force: GEMM distances + torch.topk, chunked (the port's 'brute')
	torch-bvh     points_as_tori.bvh: linear BVH built with torch, batched stack traversal (the port's 'bvh'),
	              build time reported separately
	slang         points_as_tori.knn_slang: the same BVH, fused Slang kernel, one thread per query (the port's
	              'slang'; needs slangpy); upload of the BVH arrays reported as 'slang setup', kernel compiled in the
	              warm-up run. CUDA tensors: slangpy's CUDA device on torch's context; CPU tensors: a Vulkan device
	              with host copies (config.SLANG_DEVICE_TYPE / SLANG_ADAPTER select others)
	faiss         faiss IndexFlatL2 (exact brute force with fused k-selection; GPU: GpuIndexFlatL2, k ≤ 2048)
	keops         pykeops LazyTensor argKmin (exact brute force, reduction fused in registers, no Q×N matrix)
	pytorch3d     pytorch3d.ops.knn_points (exact brute-force CUDA kernel)
	scipy-kdtree  scipy.spatial.cKDTree, all CPU threads (CPU reference for a tree)
	cpp-nanoflann the original's get_neighbors (fit only)
Points: N samples of the unit sphere (a surface, as point clouds are); float32.
Exactness: the k-th neighbor distance of every query vs a float64 recomputation from the returned indices, compared
with the smallest k-th distance found by any backend (ties and float32 rounding allowed at 1e-6).
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
from points_as_tori.torus_distance import _chunk_ranges, _knn, _query_chunk_size  # noqa: E402


def sphere(n, seed):
	p = np.random.default_rng(seed).normal(size=(n, 3))
	return (p / np.linalg.norm(p, axis=1, keepdims=True)).astype(np.float32)


def sync(device):
	if device.type == 'cuda':
		torch.cuda.synchronize()


def timed(fn, device, reps=2):
	fn()  # warm-up (kernels, caches, JIT)
	sync(device)
	times = []
	out = None
	for _ in range(reps):
		t = time.perf_counter()
		out = fn()
		sync(device)
		times.append(time.perf_counter() - t)
	return min(times), out


def torch_brute(P, Q, k):
	sq = (P * P).sum(0)
	out = torch.empty((Q.shape[1], k), dtype=torch.long, device=P.device)
	for s, e in _chunk_ranges(Q.shape[1], _query_chunk_size(P.shape[1])):
		out[s:e] = _knn(Q[:, s:e], P, k, sq)
	return out


def backends(device):
	out = {'torch-brute': lambda P, Q, k: torch_brute(P, Q, k)}
	out['torch-bvh'] = None  # special-cased (build + query)
	from points_as_tori import knn_slang

	if knn_slang.available():
		out['slang'] = None  # special-cased (reuses the torch-bvh BVH)
	try:
		import faiss

		def run_faiss(P, Q, k):
			xb, xq = P.T.contiguous(), Q.T.contiguous()
			if device.type == 'cuda':
				from faiss.contrib import torch_utils  # noqa: F401  (torch tensors in search/add)

				res = faiss.StandardGpuResources()
				index = faiss.GpuIndexFlatL2(res, 3)
				index.add(xb)
				return index.search(xq, k)[1]
			index = faiss.IndexFlatL2(3)
			index.add(xb.cpu().numpy())
			return torch.from_numpy(index.search(xq.cpu().numpy(), k)[1])

		out['faiss'] = run_faiss
	except ImportError:
		pass
	try:
		from pykeops.torch import LazyTensor

		def run_keops(P, Q, k):
			x_i = LazyTensor(Q.T.contiguous()[:, None, :])
			y_j = LazyTensor(P.T.contiguous()[None, :, :])
			return ((x_i - y_j) ** 2).sum(-1).argKmin(k, dim=1)

		out['keops'] = run_keops
	except ImportError:
		pass
	try:
		from pytorch3d.ops import knn_points

		def run_p3d(P, Q, k):
			return knn_points(Q.T[None].contiguous(), P.T[None].contiguous(), K=k).idx[0]

		out['pytorch3d'] = run_p3d
	except ImportError:
		pass
	if device.type == 'cpu':
		try:
			from scipy.spatial import cKDTree

			def run_kdtree(P, Q, k):
				tree = cKDTree(P.T.numpy())
				return torch.from_numpy(tree.query(Q.T.numpy(), k=k, workers=-1)[1])

			out['scipy-kdtree'] = run_kdtree
		except ImportError:
			pass
	return out


def kth_distance(P, Q, idx):
	P64, Q64 = P.double(), Q.double()
	d = torch.sqrt(((Q64.T[:, None, :] - P64.T[idx.long().to(P.device)]) ** 2).sum(-1))
	return d.amax(dim=1)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--sizes', type=float, nargs='+', default=[2048, 2e4, 1e5, 1e6])
	ap.add_argument('--device', default='cpu')
	ap.add_argument('--queries', type=int, default=100_000)
	ap.add_argument('--threads', type=int, default=os.cpu_count())
	ap.add_argument('--max-brute', type=float, default=2e5, help='skip O(Q·N) backends above this N on CPU')
	ap.add_argument('--backends', nargs='+', default=None, help='run only these backends (default: all importable)')
	a = ap.parse_args()
	torch.set_num_threads(a.threads)
	pat.set_device(a.device, torch.float32)
	device = config.get_device()
	if device.type == 'cuda':
		torch.cuda.set_device(device)  # torch.cuda.synchronize() and slangpy's context follow this device
	bk = backends(device)
	if a.backends is not None:
		bk = {name: fn for name, fn in bk.items() if name in a.backends}
	ref_mod = None
	try:
		sys.path.insert(0, os.environ.get('PAT_REFERENCE_DIR', os.path.join(REPO, 'build', 'pat_reference')))
		import pat_reference as ref_mod
	except ImportError:
		pass
	print(f'torch {torch.__version__}, device {device}, {a.threads} threads, backends {list(bk)}, C++: {bool(ref_mod)}')
	for n in (int(float(s)) for s in a.sizes):
		P = pat.as_wide(sphere(n, 0), 3)
		Qe = pat.as_wide(sphere(a.queries, 1) * (1 + 0.02 * np.random.default_rng(2).normal(size=(a.queries, 1))), 3)
		for work, Q, k in (('fit', P, 64), ('eval', Qe, 32)):
			row = {'N': n, 'work': work, 'Q': Q.shape[1], 'k': k}
			kth = {}
			bvh = None
			for name, fn in bk.items():
				brute_like = name in ('torch-brute', 'faiss', 'keops', 'pytorch3d')
				if device.type == 'cpu' and brute_like and n > a.max_brute:
					continue
				try:
					if name == 'torch-bvh':
						tb, bvh = timed(lambda: BVH(P), device, reps=1)
						t, idx = timed(lambda: bvh.find_k_nearest(Q, k)[0], device, reps=1)
						row['torch-bvh build'] = round(tb, 4)
					elif name == 'slang':
						from points_as_tori.knn_slang import SlangKNN

						bvh = BVH(P) if bvh is None else bvh
						ts, sk = timed(lambda: SlangKNN(P, bvh=bvh), device, reps=1)
						t, idx = timed(lambda: sk.find_k_nearest(Q, k)[0], device, reps=2)
						row['slang setup'] = round(ts, 4)
					else:
						t, idx = timed(lambda: fn(P, Q, k), device, reps=1)
					row[name] = round(t, 4)
					kth[name] = kth_distance(P, Q, idx)
				except Exception as e:  # noqa: BLE001
					row[name] = f'error: {type(e).__name__}: {str(e)[:80]}'
			if ref_mod is not None and work == 'fit' and device.type == 'cpu':
				Pn = np.asfortranarray(P.cpu().numpy().astype(np.float64))
				t = time.perf_counter()
				idx = torch.from_numpy(ref_mod.get_neighbors(Pn, k, np.zeros(0, np.int32)).T.astype(np.int64))
				row['cpp-nanoflann'] = round(time.perf_counter() - t, 4)
				kth['cpp-nanoflann'] = kth_distance(P, Q, idx)
			best = torch.stack(list(kth.values())).amin(dim=0)
			row['exact'] = {name: bool(((v - best) <= 1e-6).all()) for name, v in kth.items()}
			print(json.dumps(row))


if __name__ == '__main__':
	main()
