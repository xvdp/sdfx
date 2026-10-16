"""
HOST — the fused Slang k-NN kernel through slangpy, called from torch (points_as_tori/knn_slang.py,
points_as_tori/slang/{common,bvh,knn}.slang): one GPU thread per query runs the whole BVH traversal.

	INPUT   random points (or --ply a point cloud)
	python examples/hosts/slang_knn.py --device cuda:0                    # slangpy CUDA device on torch's context
	python examples/hosts/slang_knn.py --device cuda:1 --type vulkan      # Vulkan device on the same GPU (interop)
	python examples/hosts/slang_knn.py --device cpu                       # CPU tensors: a Vulkan GPU, else lavapipe
	python examples/hosts/slang_knn.py --device cuda:0 --variant scan --k 64 --float64
	python -m points_as_tori.knn_slang                                    # diagnostics: devices, NVRTC, CUDA headers

Data movement (knn_slang.py "Data movement per call"): with the CUDA device type the kernel reads the queries and
writes the outputs in place (torch tensors), the BVH arrays are the torch BVH's; with Vulkan the BVH is uploaded once
and slangpy copies the per-call tensors through CUDA-interop buffers; with CPU tensors the queries are uploaded and
the results downloaded once per call. As the evaluator's backend: points_as_tori.config.KNN_BACKEND = 'slang' ('auto'
picks it on CUDA when slangpy runs).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch  # noqa: E402

import points_as_tori as pat  # noqa: E402
from points_as_tori import config  # noqa: E402


def timed(fn, device, reps=3):
	fn()  # first call: compiles the kernel for this (k, BVH depth)
	times = []
	for _ in range(reps):
		if device.type == 'cuda':
			torch.cuda.synchronize(device)
		t = time.perf_counter()
		out = fn()
		if device.type == 'cuda':
			torch.cuda.synchronize(device)
		times.append(time.perf_counter() - t)
	return min(times), out


def main():
	ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
	ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
	ap.add_argument('--type', default=None, choices=['cuda', 'vulkan'], help='slangpy device type (default by tensor)')
	ap.add_argument('--n', type=int, default=100_000, help='points')
	ap.add_argument('--q', type=int, default=100_000, help='queries')
	ap.add_argument('--k', type=int, default=32)
	ap.add_argument('--variant', default='heap', choices=['heap', 'scan'])
	ap.add_argument('--float64', action='store_true', help='float64 geometry; with --kernel64 the kernel too')
	ap.add_argument('--kernel64', action='store_true', help='config.SLANG_FLOAT64: the heap kernel in double')
	ap.add_argument('--ply', default=None, help='point cloud instead of random points')
	a = ap.parse_args()

	from points_as_tori.bvh import BVH
	from points_as_tori.knn_slang import SlangKNN

	dtype = torch.float64 if a.float64 else torch.float32
	pat.set_device(a.device, dtype)
	config.SLANG_DEVICE_TYPE = a.type
	config.SLANG_FLOAT64 = a.kernel64
	device = config.get_device()
	g = torch.Generator(device='cpu').manual_seed(0)
	if a.ply:
		P = pat.as_wide(pat.read_point_cloud(a.ply)[0], 3)
	else:
		P = torch.randn(3, a.n, generator=g, dtype=dtype).to(device)
	Q = torch.randn(3, a.q, generator=g, dtype=dtype).to(device)

	t = time.perf_counter()
	bvh = BVH(P)  # LBVH built with torch on the device (shared by both searches)
	print(f'BVH build: {time.perf_counter() - t:.3f} s, {P.shape[1]} points, depth {bvh.stack_size}')
	knn = SlangKNN(P, bvh=bvh, variant=a.variant)  # creates / reuses the slangpy device, uploads if needed
	print(f'slangpy device: {knn.device.info.adapter_name} ({knn.device.info.type}), kernel precision {knn.real}')
	ts, (idx, dist) = timed(lambda: knn.find_k_nearest(Q, a.k), device)
	tb, (idx_b, dist_b) = timed(lambda: bvh.find_k_nearest(Q, a.k), device, reps=1)
	same = (idx.sort(dim=1).values == idx_b.sort(dim=1).values).all(dim=1).float().mean()
	print(f'Slang {a.variant}: {ts * 1e3:.2f} ms   torch BVH: {tb * 1e3:.1f} ms   ({a.q} queries, k = {a.k})')
	print(f'same neighbor sets: {float(same):.4f}, max |d_slang - d_torch| = {float((dist - dist_b).abs().max()):.3g}')


if __name__ == '__main__':
	main()
