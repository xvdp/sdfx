"""
Fused k-nearest-neighbor search in Slang (through slangpy): one GPU thread per query runs the whole BVH traversal.

Why: the torch traversal (points_as_tori.bvh.BVH.find_k_nearest) advances every query by one node per iteration with
~90 small tensor operations, for 100-200 iterations per call, so on a GPU its cost is mostly kernel launches. Here the
traversal loop runs inside one kernel: one launch per chunk of queries, independent of the tree depth.

Kernel (slang/knn.slang, slang/bvh.slang): bvhKNearestNeighbors is a function-by-function mirror of
sdfview/shaders/glsl/bvh.glsl ('scan' variant); bvhKNearestNeighborsHeap keeps the held set as a max-heap like the
original C++ BVH ('heap' variant, the default). Per query q:
  * depth-first traversal with a per-thread stack, nearer child first (children ordered by dist²(q, box));
  * a node is pruned when k points are held and dist²(q, box) > d_k², d_k = current k-th nearest distance
    (dist²(q, box) = |q - clamp(q, min, max)|², 0 inside);
  * the held set: 'scan' an unsorted array of k (a closer point replaces the farthest, found by an O(k) scan, and
    d_k is recomputed in O(k)); 'heap' a max-heap on squared distance (d_k = root, O(1); replacement sifts down,
    O(log k));
  * afterwards the k results are sorted by (distance, index) ('scan': insertion sort, 'heap': heapsort) and written
    to row q of the outputs.
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

_DEVICES: dict = {}  # (device type, torch device or adapter) -> (spy.Device, spy.Module), in creation order
_CLOSE_REGISTERED = False


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
			if type_name != 'cuda':
				# A CUDA context created after a Vulkan device with CUDA interop exists was later unusable
				# (CUDA_ERROR_OPERATING_SYSTEM; measured on 2 GPUs, slangpy 0.43.0, see PORT_NOTES.md "Context
				# order"). Create torch's (primary) context on every GPU before the first interop device.
				for i in range(torch.cuda.device_count()):
					torch.empty(1, device=torch.device('cuda', i))
			# create_torch_device takes the calling thread's current CUDA context (cuCtxGetCurrent); its own
			# `with torch.device(...)` only sets the default device of tensor factories, and torch creates contexts
			# lazily. Make this GPU current and its primary context exist (one allocation) before creating.
			with torch.cuda.device(torch_device):
				torch.empty(1, device=torch_device)
				try:
					device = spy.create_torch_device(
						type=device_type, torch_device=torch_device, include_paths=[SHADER_DIR]
					)
				except RuntimeError as e:
					raise RuntimeError(
						f'{e}\nslangpy {type_name} device for {torch_device} not created. A CUDA device compiles '
						'kernels with NVRTC; an NVRTC newer than the driver gives CUDA_ERROR_UNSUPPORTED_PTX_VERSION. '
						'Details: python -m points_as_tori.knn_slang (see PORT_NOTES.md, "Fused Slang kernel")'
					) from e
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
		global _CLOSE_REGISTERED
		if not _CLOSE_REGISTERED:
			import atexit

			atexit.register(close_devices)  # registered after slangpy's own atexit handler, so it runs first
			_CLOSE_REGISTERED = True
	return _DEVICES[key]


def close_devices() -> None:
	"""
	Close the slangpy devices created here: synchronize torch on the GPUs they use, then close them in reverse
	creation order, each with its GPU current. Runs at interpreter exit (atexit, before slangpy closes the remaining
	devices in its own order); SlangKNN objects created before a call cannot be used after it.
	"""
	gpus = {where for _, where in _DEVICES if isinstance(where, str) and where.startswith('cuda')}
	for where in sorted(gpus):
		torch.cuda.synchronize(torch.device(where))
	for key in reversed(list(_DEVICES)):
		device, _ = _DEVICES.pop(key)
		where = key[1]
		if isinstance(where, str) and where.startswith('cuda'):
			with torch.cuda.device(torch.device(where)):
				device.close()
		else:
			device.close()


def _round_up_pow2(x: int, lo: int) -> int:
	return max(lo, 1 << (max(1, x) - 1).bit_length())


class SlangKNN:
	"""
	k nearest neighbors among points (3, N) through the fused Slang kernel; same interface as
	points_as_tori.bvh.BVH.find_k_nearest. `bvh`: a points_as_tori.bvh.BVH over the same points (built if None).
	`variant`: 'heap' (held set as a max-heap, as the original C++ BVH: O(log k) per accepted candidate) or 'scan'
	(unsorted array, as the GLSL / demo shader: O(k) per accepted candidate). Same neighbors up to ties.
	"""

	VARIANTS = {'heap': 'knn', 'scan': 'knn_scan'}

	def __init__(self, points: torch.Tensor, bvh=None, variant: str = 'heap'):
		from .bvh import BVH

		if variant not in self.VARIANTS:
			raise ValueError(f'variant must be one of {list(self.VARIANTS)}, got {variant!r}')
		self.variant = variant
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
		kernel = getattr(self.module, self.VARIANTS[self.variant])
		fn = kernel.constants({'BVH_K_MAX': k_max, 'BVH_STACK_SIZE': self.stack_size})
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


def _cuda_driver_version() -> Optional[Tuple[int, int]]:
	"""Highest CUDA version the installed driver supports (cuDriverGetVersion), or None without a driver."""
	import ctypes

	try:
		lib = ctypes.CDLL('libcuda.so.1')
	except OSError:
		return None
	v = ctypes.c_int()
	if lib.cuDriverGetVersion(ctypes.byref(v)) != 0:
		return None
	return v.value // 1000, (v.value % 1000) // 10


def _nvrtc_version(path: str) -> Optional[Tuple[int, int]]:
	"""(major, minor) of the NVRTC library at `path` (or loader name), None if it does not load."""
	import ctypes

	try:
		lib = ctypes.CDLL(path)
	except OSError:
		return None
	major, minor = ctypes.c_int(), ctypes.c_int()
	if lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)) != 0:
		return None
	return major.value, minor.value


def _rhi_nvrtc_dirs() -> list:
	"""
	Directories slang-rhi searches for libnvrtc.so on Linux, in order (slang-rhi src/cuda/cuda-nvrtc.cpp); the first
	holding a file named exactly libnvrtc.so is used for the device's internal kernels.
	"""
	dirs = []
	cuda_path = os.environ.get('CUDA_PATH')
	if cuda_path:
		dirs += [os.path.join(cuda_path, d) for d in ('lib64', 'lib', os.path.join('targets', 'x86_64-linux', 'lib'))]
	try:
		toolkits = sorted(
			(e.path for e in os.scandir('/usr/local') if e.is_dir() and e.name.startswith('cuda')), reverse=True
		)
	except OSError:
		toolkits = []
	for t in toolkits:
		for d in (os.path.join(t, 'lib64'), os.path.join(t, 'targets', 'x86_64-linux', 'lib')):
			if d not in dirs:
				dirs.append(d)
	dirs.append('/usr/lib/x86_64-linux-gnu/')
	return dirs


def _handle_path(lib) -> Optional[str]:
	"""File a ctypes.CDLL handle was loaded from (glibc dlinfo RTLD_DI_LINKMAP)."""
	import ctypes

	class LinkMap(ctypes.Structure):
		_fields_ = [('l_addr', ctypes.c_void_p), ('l_name', ctypes.c_char_p)]

	for libc_name in (None, 'libdl.so.2'):  # dlinfo is in libc from glibc 2.34, in libdl before
		try:
			dlinfo = ctypes.CDLL(libc_name).dlinfo
		except (OSError, AttributeError):
			continue
		link_map = ctypes.POINTER(LinkMap)()
		if dlinfo(ctypes.c_void_p(lib._handle), 2, ctypes.byref(link_map)) == 0 and link_map:
			return link_map.contents.l_name.decode()
	return None


def _rhi_cuda_include(nvrtc_path: str) -> Tuple[Optional[str], list]:
	"""
	slang-rhi's search for cuda_runtime.h next to the NVRTC it loaded (src/cuda/cuda-nvrtc.cpp): <dir>/../include,
	<dir>/../../include, /usr/include, with <dir> the directory of libnvrtc.so (symlinks not resolved).
	Returns (directory found or None, directories searched).
	"""
	lib_dir = os.path.dirname(nvrtc_path)
	dirs = [
		os.path.join(os.path.dirname(lib_dir), 'include'),
		os.path.join(os.path.dirname(os.path.dirname(lib_dir)), 'include'),
		'/usr/include',
	]
	for d in dirs:
		if os.path.isfile(os.path.join(d, 'cuda_runtime.h')):
			return d, dirs
	return None, dirs


def _report_nvrtc() -> None:
	"""
	Print the driver's CUDA version, the NVRTC each lookup resolves to and the CUDA headers slang-rhi finds; flag an
	NVRTC newer than the driver, missing headers, and stub libraries on the loader path.
	"""
	driver = _cuda_driver_version()
	print(f'CUDA driver: {"none" if driver is None else "supports CUDA %d.%d" % driver}')
	cuda_path = os.environ.get('CUDA_PATH')
	print(
		f'CUDA_PATH: {cuda_path or "unset"}   (CUDA_HOME: {os.environ.get("CUDA_HOME", "unset")}, not read by slang-rhi)'
	)

	def describe(path, version):
		if version is None:
			return f'{path} (loads, but nvrtcVersion fails: a stub or broken library)'
		text = f'{path}, NVRTC {version[0]}.{version[1]}'
		if driver is not None and version > driver:
			text += f'  <- newer than the driver (CUDA {driver[0]}.{driver[1]}): its PTX is rejected'
		return text

	found = None
	for d in _rhi_nvrtc_dirs():
		candidate = os.path.join(d, 'libnvrtc.so')
		if os.path.isfile(candidate):
			if found is None:
				found = candidate
				print(f'slang-rhi NVRTC (device creation): {describe(candidate, _nvrtc_version(candidate))}')
			else:
				print(f'  also available: {describe(candidate, _nvrtc_version(candidate))}')
	if found is None:
		print('slang-rhi NVRTC (device creation): no libnvrtc.so in its search directories')
	else:
		include, searched = _rhi_cuda_include(found)
		if include is not None:
			print(f'slang-rhi CUDA headers: {include}')
		else:
			print(f'slang-rhi CUDA headers: cuda_runtime.h not found in {searched}  <- device creation fails')
			if cuda_path:
				targets = os.path.join(cuda_path, 'targets', 'x86_64-linux')
				if os.path.isfile(os.path.join(targets, 'include', 'cuda_runtime.h')):
					if os.path.isfile(os.path.join(targets, 'lib', 'libnvrtc.so')):
						print(f'  conda layout: export CUDA_PATH={targets}  (lib/libnvrtc.so and include/ there)')
					else:
						print(f'  headers in {targets}/include, but no {targets}/lib/libnvrtc.so (cuda-nvrtc-dev)')

	stubs = [d for d in os.environ.get('LD_LIBRARY_PATH', '').split(':') if os.path.basename(d.rstrip('/')) == 'stubs']
	if stubs:
		print(f'LD_LIBRARY_PATH holds stub directories {stubs}  <- link-time only; remove them for running')
	import ctypes

	try:
		slang_nvrtc, error = _handle_path(ctypes.CDLL('libnvrtc.so')), None
	except OSError as e:
		slang_nvrtc, error = None, e
	if slang_nvrtc is None:
		print(f'Slang NVRTC (kernel compilation, dlopen libnvrtc.so): not loaded ({error})')
	else:
		print(
			f'Slang NVRTC (kernel compilation, dlopen libnvrtc.so): {describe(slang_nvrtc, _nvrtc_version(slang_nvrtc))}'
		)


def _environment_report() -> None:
	"""Versions, GPUs, slangpy adapters, NVRTC lookups (see diagnose)."""
	import importlib.metadata
	import platform

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
	_report_nvrtc()


def _cuda_health() -> list:
	"""[(i, None or error text)]: allocate, compute and synchronize with torch on every CUDA device."""
	out = []
	for i in range(torch.cuda.device_count()):
		try:
			torch.ones(1, device=torch.device('cuda', i)).add_(1)
			torch.cuda.synchronize(i)
			out.append((i, None))
		except Exception as e:  # noqa: BLE001
			out.append((i, f'{type(e).__name__}: {str(e).splitlines()[0]}'))
	return out


def _run_config(name: str, type_name: Optional[str], n_points: int, n_queries: int, k: int) -> bool:
	"""Create the device for torch device `name` / slangpy `type_name`, run a k-NN, compare with the torch BVH."""
	import time
	import traceback

	label = f'{name} / slangpy {type_name or "vulkan (host copies)"}'
	print(f'--- {label}', flush=True)
	saved = config.SLANG_DEVICE_TYPE
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
			f'max |d - d_bvh| {err:.2e}',
			flush=True,
		)
		return True
	except Exception:  # noqa: BLE001
		print(f'FAILED {label}')
		traceback.print_exc()
		return False
	finally:
		config.SLANG_DEVICE_TYPE = saved
		if torch.cuda.is_available():
			health = _cuda_health()
			print(
				'    torch afterwards: '
				+ ', '.join(f'cuda:{i} ' + ('ok' if e is None else f'BROKEN ({e})') for i, e in health)
			)
			if any(e is not None for _, e in health):
				return False  # noqa: B012  (a broken context counts as a failure of this step)


def diagnose(
	n_points: int = 20000, n_queries: int = 2000, k: int = 16, same_process: bool = False, pairs: bool = False
) -> int:
	"""
	Print versions, adapters and NVRTC lookups, then for each torch device (cpu, every cuda:i) and slangpy device
	type create the device, run a k-NN and compare it with the torch BVH, with full tracebacks. slang-rhi's own error
	lines are printed by slangpy to the terminal as they occur.

	Each configuration runs in its own Python process (also through interpreter exit, where the devices are closed),
	so a crash is attributed to its configuration; same_process=True runs them all in this process, as a program
	using several device types would; pairs=True runs every ordered pair of GPU configurations in a fresh process
	(to find which combination breaks a CUDA context). After each step torch allocates and synchronizes on every GPU
	("torch afterwards"). Returns the number of failed runs.
	Run: python -m points_as_tori.knn_slang [--same-process | --pairs]
	"""
	import signal
	import subprocess
	import sys

	_environment_report()
	configs = [('cpu', None)]
	for i in range(torch.cuda.device_count()):
		configs += [(f'cuda:{i}', 'cuda'), (f'cuda:{i}', 'vulkan')]
	if same_process:
		return sum(not _run_config(name, type_name, n_points, n_queries, k) for name, type_name in configs)

	def describe(sequence):
		return ' then '.join(f'{n}/{t or "host"}' for n, t in sequence)

	def run(sequence):
		cmd = [sys.executable, '-m', 'points_as_tori.knn_slang']
		for name, type_name in sequence:
			cmd += ['--config', name, type_name or 'host']
		cmd += ['--points', str(n_points), '--queries', str(n_queries), '-k', str(k)]
		rc = subprocess.run(cmd).returncode
		if rc < 0:
			outcome = f'killed by {signal.Signals(-rc).name}'
		else:
			outcome = 'ok' if rc == 0 else 'FAILED'
		print(f'    => {outcome}', flush=True)
		return outcome

	if pairs:
		gpu_configs = [c for c in configs if c[1] is not None]
		sequences = [(a, b) for a in gpu_configs for b in gpu_configs if a != b]
	else:
		sequences = [(c,) for c in configs]
	results = []
	for sequence in sequences:
		print(f'=== {describe(sequence)}', flush=True)
		results.append((sequence, run(sequence)))
	print('=== summary')
	for sequence, outcome in results:
		print(f'{describe(sequence):44s} {outcome}')
	return sum(outcome != 'ok' for _, outcome in results)


def main(argv=None) -> int:
	import argparse

	ap = argparse.ArgumentParser(description='Diagnose the slangpy k-NN backend (see diagnose()).')
	ap.add_argument('--same-process', action='store_true', help='run all configurations in one process')
	ap.add_argument('--pairs', action='store_true', help='every ordered pair of GPU configurations, one process each')
	ap.add_argument(
		'--config',
		nargs=2,
		action='append',
		metavar=('DEVICE', 'TYPE'),
		help='run these configurations in order in this process (TYPE: cuda|vulkan|host); repeatable',
	)
	ap.add_argument('--points', type=int, default=20000)
	ap.add_argument('--queries', type=int, default=2000)
	ap.add_argument('-k', type=int, default=16)
	a = ap.parse_args(argv)
	if a.config is not None:
		ok = [_run_config(n, None if t == 'host' else t, a.points, a.queries, a.k) for n, t in a.config]
		return 0 if all(ok) else 1
	return diagnose(a.points, a.queries, a.k, same_process=a.same_process, pairs=a.pairs)


if __name__ == '__main__':
	raise SystemExit(main())
