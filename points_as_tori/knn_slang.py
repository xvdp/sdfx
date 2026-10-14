"""
Fused k-nearest-neighbor search in Slang (through slangpy): one GPU thread per query runs the whole BVH traversal.

Why: the torch traversal (points_as_tori.bvh.BVH.find_k_nearest) advances every query by one node per iteration with
~90 small tensor operations, for 100-200 iterations per call, so on a GPU its cost is mostly kernel launches. Here the
traversal loop runs inside one kernel: one launch per chunk of queries, independent of the tree depth.

Kernel (slang/knn.slang, slang/bvh.slang): bvhKNearestNeighbors is a function-by-function mirror of
sdfview/shaders/glsl/bvh.glsl. Per query q:
  * depth-first traversal with a per-thread stack, nearer child first (children ordered by dist²(q, box));
  * a node is pruned when k points are held and dist²(q, box) > d_k², d_k = current k-th nearest distance
    (dist²(q, box) = |q - clamp(q, min, max)|², 0 inside);
  * the held set is an unsorted array of k (index, distance); a closer point replaces the farthest one;
  * afterwards the k results are insertion-sorted by (distance, index) and written to row q of the outputs.
The pruning is exact: every point in a pruned box is at distance ≥ the box distance > d_k. Distances are float32 and
computed from differences, |q - p|² = Σ (q_c - p_c)², without the |q|² + |p|² - 2 q·p cancellation of the GEMM brute
force, so the neighbor sets equal the float64 ones except where two candidate distances differ by less than float32
rounding (relative ~1e-7).

Data: the BVH of points_as_tori.bvh (built on the torch device) in the scene-format layout of sdfview/scene.py:
bounds (2M, 4) float32 rows (min, 0), (max, 0); nodes (M, 4) int32 (left, right, prim_start, prim_count);
prims (N, 1) int32; points (N, 4) float32. Uploaded once per SlangKNN.

Compile-time sizes are Slang link-time constants set per call: BVH_K_MAX = k rounded up to a power of two (≥ 8; the
per-thread result arrays) and BVH_STACK_SIZE = BVH depth + 2 rounded up to a multiple of 8 (a depth-first traversal
that pushes both children of each popped node holds at most depth + 1 entries). slangpy compiles one kernel per
distinct pair and caches it.

Devices (config.SLANG_DEVICE_TYPE, config.SLANG_ADAPTER):
  * CUDA tensors: spy.create_torch_device(type, torch_device=<the tensors' device>) — type 'cuda' (default) shares
    torch's CUDA context of that GPU, 'vulkan' creates a Vulkan device on the same GPU (sgl matches the adapter by
    the CUDA device UUID) with CUDA interop. Torch tensors are passed to the kernel and returned without copies;
    slangpy launches on torch's current stream. Without the slangpy-torch extension slangpy reads tensor metadata
    through Python (~µs per call; this module enables that fallback when the extension is missing).
  * CPU tensors: a device of type 'vulkan' (default) on the adapter whose name contains config.SLANG_ADAPTER
    (default: slangpy's choice, the first discrete GPU), e.g. Mesa's lavapipe on a machine without a GPU. Arrays
    are copied through NumPy. Meant for tests and for machines without CUDA. slangpy's 'cpu' device type does not
    run these calls (slangpy 0.43.1: "Device reports zero compute dispatch groups").
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch

from . import config

SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'slang')

K_LIMIT = 1024  # largest k run by the kernel (per-thread arrays of BVH_K_MAX ints and floats); above: torch BVH
QUERY_CHUNK = 1 << 20  # queries per dispatch, at most
QUERY_CHUNK_ELEMENTS = 1 << 24  # (query, neighbor) output entries per dispatch, at most

_DEVICES: dict = {}  # (device type, torch device or adapter) -> (spy.Device, spy.Module)


def available() -> bool:
	"""True if slangpy can be imported."""
	try:
		import slangpy  # noqa: F401
	except ImportError:
		return False
	return True


def _spy():
	try:
		import slangpy as spy
	except ImportError as e:
		raise ImportError(
			"config.KNN_BACKEND = 'slang' needs slangpy: pip install slangpy (see points_as_tori/PORT_NOTES.md)"
		) from e
	return spy


def _device_type_name(torch_device: torch.device) -> str:
	if config.SLANG_DEVICE_TYPE is not None:
		return config.SLANG_DEVICE_TYPE
	return 'cuda' if torch_device.type == 'cuda' else 'vulkan'


def get_device(torch_device: torch.device):
	"""(slangpy device, knn module) for tensors on `torch_device`; created once per (device type, device)."""
	spy = _spy()
	type_name = _device_type_name(torch_device)
	device_type = getattr(spy.DeviceType, type_name)
	if torch_device.type == 'cuda':
		index = torch_device.index if torch_device.index is not None else torch.cuda.current_device()
		torch_device = torch.device('cuda', index)
		key = (type_name, str(torch_device))
	else:
		key = (type_name, config.SLANG_ADAPTER)
	if key not in _DEVICES:
		if torch_device.type == 'cuda':
			if not spy.is_torch_bridge_available():
				spy.set_allow_torch_fallback(True)
			# create_torch_device takes the calling thread's current CUDA context (cuCtxGetCurrent); its own
			# `with torch.device(...)` only sets the default device of tensor factories, and torch creates contexts
			# lazily. Make this GPU current and its primary context exist (one allocation) before creating.
			with torch.cuda.device(torch_device):
				torch.empty(1, device=torch_device)
				device = spy.create_torch_device(
					type=device_type, torch_device=torch_device, include_paths=[SHADER_DIR]
				)
		else:
			adapter_luid = None
			if config.SLANG_ADAPTER is not None:
				adapters = [a for a in spy.Device.enumerate_adapters(device_type) if config.SLANG_ADAPTER in a.name]
				if not adapters:
					names = [a.name for a in spy.Device.enumerate_adapters(device_type)]
					raise RuntimeError(f'no {type_name} adapter name contains {config.SLANG_ADAPTER!r}; found {names}')
				adapter_luid = adapters[0].luid
			device = spy.create_device(type=device_type, include_paths=[SHADER_DIR], adapter_luid=adapter_luid)
		module = spy.Module.load_from_file(device, 'knn.slang')
		_DEVICES[key] = (device, module)
	return _DEVICES[key]


def _round_up_pow2(x: int, lo: int) -> int:
	return max(lo, 1 << (max(1, x) - 1).bit_length())


class SlangKNN:
	"""
	k nearest neighbors among points (3, N) through the fused Slang kernel; same interface as
	points_as_tori.bvh.BVH.find_k_nearest. `bvh`: a points_as_tori.bvh.BVH over the same points (built if None).
	"""

	def __init__(self, points: torch.Tensor, bvh=None):
		from .bvh import BVH

		self.points = points
		self.bvh = BVH(points) if bvh is None else bvh
		self.n_points = points.shape[1]
		self.torch_device = points.device
		self.device, self.module = get_device(points.device)
		self.on_host = points.device.type != 'cuda'
		self.stack_size = -(-(self.bvh.stack_size) // 8) * 8

		t = self.bvh.tree
		M = t.n_nodes
		bounds = torch.zeros((M, 8), dtype=torch.float32, device=points.device)
		bounds[:, 0:3] = t.bounds_min.T
		bounds[:, 4:7] = t.bounds_max.T
		pts = torch.zeros((self.n_points, 4), dtype=torch.float32, device=points.device)
		pts[:, 0:3] = points.T
		arrays = {
			'bvh_bounds': bounds.reshape(2 * M, 4),
			'bvh_nodes': torch.stack([t.left, t.right, t.prim_start, t.prim_count], dim=1).to(torch.int32),
			'bvh_prims': t.prim_indices.to(torch.int32)[:, None].contiguous(),
			'points': pts,
		}
		self.arrays = {name: self._upload(a) for name, a in arrays.items()}

	def _upload(self, a: torch.Tensor):
		if not self.on_host:
			return a.contiguous()
		spy = _spy()
		return spy.Tensor.from_numpy(self.device, np.ascontiguousarray(a.cpu().numpy()))

	def _run(self, q: torch.Tensor, k: int, k_max: int) -> Tuple[torch.Tensor, torch.Tensor]:
		"""(3, m) float32 queries -> (m, k) int32 indices, (m, k) float32 distances, on the torch device."""
		spy = _spy()
		m = q.shape[1]
		fn = self.module.knn.constants({'BVH_K_MAX': k_max, 'BVH_STACK_SIZE': self.stack_size})
		a = self.arrays
		if self.on_host:
			out_i = spy.Tensor.from_numpy(self.device, np.full((m, k), -1, dtype=np.int32))
			out_d = spy.Tensor.from_numpy(self.device, np.zeros((m, k), dtype=np.float32))
			qs = spy.Tensor.from_numpy(self.device, np.ascontiguousarray(q.T.cpu().numpy()))
		else:
			out_i = torch.full((m, k), -1, dtype=torch.int32, device=self.torch_device)
			out_d = torch.zeros((m, k), dtype=torch.float32, device=self.torch_device)
			qs = q.T.contiguous()
		args = (spy.call_id(), qs, a['bvh_bounds'], a['bvh_nodes'], a['bvh_prims'], a['points'], k, out_i, out_d)
		if self.on_host:
			fn(*args)
		else:
			with torch.cuda.device(self.torch_device):  # this GPU's context current during the call
				fn(*args)
		if self.on_host:
			out_i = torch.from_numpy(out_i.to_numpy()).to(self.torch_device)
			out_d = torch.from_numpy(out_d.to_numpy()).to(self.torch_device)
		return out_i, out_d

	def find_k_nearest(self, queries: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		(3, Q) queries (or one (3,) query) -> (indices (Q, k) int64, distances (Q, k) in the queries' dtype), nearest
		first, k = min(k, N). Distances in another dtype than float32 are recomputed in that dtype from the indices.
		"""
		single = queries.ndim == 1
		q_all = queries[:, None] if single else queries
		k = min(int(k), self.n_points)
		if k > K_LIMIT:
			idx, dist = self.bvh.find_k_nearest(q_all, k)
		else:
			k_max = _round_up_pow2(k, 8)
			n = q_all.shape[1]
			idx = torch.empty((n, k), dtype=torch.long, device=self.torch_device)
			dist = torch.empty((n, k), dtype=q_all.dtype, device=self.torch_device)
			chunk = max(1, min(QUERY_CHUNK, QUERY_CHUNK_ELEMENTS // k))
			for s in range(0, n, chunk):
				e = min(n, s + chunk)
				q = q_all[:, s:e]
				i32, d32 = self._run(q.to(torch.float32), k, k_max)
				idx[s:e] = i32.long()
				if q.dtype == torch.float32:
					dist[s:e] = d32
				else:
					diff = q[:, :, None] - self.points[:, idx[s:e]]
					dist[s:e] = torch.sqrt((diff * diff).sum(dim=0))
		if single:
			return idx[0], dist[0]
		return idx, dist

	def find_closest_point(self, queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		idx, dist = self.find_k_nearest(queries, 1)
		return idx[..., 0], dist[..., 0]


def knn(queries: torch.Tensor, points: torch.Tensor, k: int, searcher: Optional[SlangKNN] = None) -> torch.Tensor:
	"""(3, Q) queries -> (Q, k) indices of the k nearest points (3, N), nearest first."""
	searcher = SlangKNN(points) if searcher is None else searcher
	return searcher.find_k_nearest(queries, k)[0]


def diagnose(n_points: int = 20000, n_queries: int = 2000, k: int = 16) -> int:
	"""
	Print versions, adapters, and for each torch device (cpu, every cuda:i) and slangpy device type the outcome of
	creating the device and running a k-NN against the torch BVH, with full tracebacks. slang-rhi's own error lines
	(e.g. the reason behind "Failed to create device!") are printed by slangpy to the terminal as they occur.
	Returns the number of failed configurations.  Run: python -m points_as_tori.knn_slang
	"""
	import importlib.metadata
	import platform
	import time
	import traceback

	spy = _spy()

	def version(name):
		try:
			return importlib.metadata.version(name)
		except importlib.metadata.PackageNotFoundError:
			return 'not installed'

	print(f'python {platform.python_version()}, torch {torch.__version__} (CUDA {torch.version.cuda})')
	print(f'slangpy {version("slangpy")}, slangpy-torch {version("slangpy-torch")}')
	print(f'torch bridge available: {spy.is_torch_bridge_available()}', end='')
	if hasattr(spy, 'get_torch_bridge_fallback_reason') and not spy.is_torch_bridge_available():
		print(f' (fallback reason: {spy.get_torch_bridge_fallback_reason()})', end='')
	print()
	for i in range(torch.cuda.device_count()):
		print(f'torch cuda:{i}: {torch.cuda.get_device_name(i)}')
	for type_name in ('cuda', 'vulkan'):
		try:
			adapters = spy.Device.enumerate_adapters(getattr(spy.DeviceType, type_name))
			print(f'slangpy {type_name} adapters: {[a.name for a in adapters]}')
		except Exception as e:  # noqa: BLE001
			print(f'slangpy {type_name} adapters: error {type(e).__name__}: {e}')

	configs = [('cpu', None)]
	for i in range(torch.cuda.device_count()):
		configs += [(f'cuda:{i}', 'cuda'), (f'cuda:{i}', 'vulkan')]
	saved = config.SLANG_DEVICE_TYPE
	failures = 0
	for name, type_name in configs:
		label = f'{name} / slangpy {type_name or "vulkan (host copies)"}'
		print(f'--- {label}', flush=True)
		config.SLANG_DEVICE_TYPE = type_name
		try:
			device = torch.device(name)
			g = torch.Generator().manual_seed(0)
			P = torch.randn(3, n_points, generator=g)
			P = (P / P.norm(dim=0)).to(device)
			Q = torch.randn(3, n_queries, generator=g).to(device)
			t = time.perf_counter()
			s = SlangKNN(P)
			idx, dist = s.find_k_nearest(Q, k)
			if device.type == 'cuda':
				torch.cuda.synchronize(device)
			t = time.perf_counter() - t
			ref_idx, ref_dist = s.bvh.find_k_nearest(Q, k)
			same = (idx.sort(dim=1).values == ref_idx.sort(dim=1).values).all(dim=1).float().mean().item()
			err = (dist - ref_dist).abs().max().item()
			sdev = s.device.info
			print(
				f'OK on {sdev.adapter_name} ({sdev.api_name}): {t:.2f} s incl. compile, same sets {same:.4f}, '
				f'max |d - d_bvh| {err:.2e}'
			)
		except Exception:  # noqa: BLE001
			failures += 1
			print(f'FAILED {label}')
			traceback.print_exc()
	config.SLANG_DEVICE_TYPE = saved
	return failures


if __name__ == '__main__':
	raise SystemExit(diagnose())
