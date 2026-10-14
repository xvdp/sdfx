"""
Device / dtype configuration for the torch port of Points as Tori.

The device is set explicitly, once, by the caller (``set_device``) and every module reads it from here.
No function in this package moves data between devices on its own: inputs coming from NumPy are moved to the
configured device once at the import boundary (``as_wide``), and results leave the device only through the explicit
export helpers (``to_numpy``, ``save_tori``, ``export_*``). Inside, no operation needs a host sync
(no ``.item()``, ``.cpu()``, boolean-mask indexing, ``nonzero`` or value-dependent Python branching).

Layout ("wide arrays", structure-of-arrays): every array of 3D vectors is stored as a C-contiguous (3, n) tensor, so
each component row x = a[0], y = a[1], z = a[2] is itself a contiguous length-n vector and the elementwise geometry
runs over contiguous memory. Axes are (2, n), fundamental-form coefficients (6, n). This is the same shape convention as
the original C++ code (Eigen::Matrix<double, 3, Dynamic>), but row-major, i.e. component-major in memory.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch

_DEVICE: Optional[torch.device] = None
_DTYPE: torch.dtype = torch.float32

# Number of (query, point) pairs processed per chunk by the dense / brute-force kernels.
# Peak memory per chunk is roughly CHUNK_ELEMENTS * ~40 * sizeof(dtype) bytes.
CHUNK_ELEMENTS: int = 1 << 21

# k-nearest-neighbor search (replaces the C++ nanoflann kd-tree):
#   'brute'  distances to all points (one GEMM per chunk) + torch.topk: O(Q·N) work, few large kernels;
#   'bvh'    traversal of a linear BVH built on the device (points_as_tori.bvh): O(Q·visited nodes), many small kernels;
#   'slang'  the same BVH traversed by a fused Slang kernel, one GPU thread per query (points_as_tori.knn_slang; needs
#            slangpy); float32 distances;
#   'auto'   'bvh' when the point cloud has at least KNN_BVH_MIN_POINTS points, else 'brute'.
# All return the exact k nearest neighbors (up to ties; 'brute' and 'slang' in float32 up to float32 rounding). The
# threshold is from CPU measurements (PORT_NOTES.md); measure on a GPU with benchmarks/knn_bench.py.
KNN_BACKEND: str = 'auto'
KNN_BVH_MIN_POINTS: int = 32768

# slangpy device type for the 'slang' backend: None = 'cuda' for CUDA tensors (shares torch's CUDA context) and
# 'vulkan' for CPU tensors (arrays copied through the host; e.g. Mesa lavapipe when there is no GPU). 'vulkan' with
# CUDA tensors runs on the same GPU through CUDA interop. SLANG_ADAPTER: substring of the adapter name to use for
# devices not tied to a torch CUDA device (CPU tensors); None = slangpy's default (first discrete GPU).
SLANG_DEVICE_TYPE: Optional[str] = None
SLANG_ADAPTER: Optional[str] = None


def set_device(device: Union[str, torch.device], dtype: torch.dtype = torch.float32, force: bool = False) -> None:
	"""
	Set the torch device (and the floating point dtype used for geometry) for the whole package. Call once.

	Args:
		device: e.g. 'cuda', 'cuda:1', 'cpu', 'mps'
		dtype: floating point dtype for positions, normals, tori and SDF evaluation (the network always runs in the
			dtype of its weights, float32). The original C++ evaluated in float64.
		force: allow changing an already set device (tests); tensors created before the change stay where they are.
	"""
	global _DEVICE, _DTYPE
	device = torch.tensor(0, device=device).device  # force init. torch.device('cuda') returns cuda with no index
	if _DEVICE is not None and (_DEVICE != device or _DTYPE != dtype) and not force:
		raise RuntimeError(f'points_as_tori device already set to {_DEVICE} ({_DTYPE}); pass force=True to change it')
	_DEVICE = device
	_DTYPE = dtype


def get_device() -> torch.device:
	if _DEVICE is None:
		raise RuntimeError("points_as_tori: set the torch device first, e.g. points_as_tori.set_device('cuda')")
	return _DEVICE


def get_dtype() -> torch.dtype:
	return _DTYPE


def as_wide(x, n_components: int, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
	"""
	Import boundary: convert an array-of-structures (n, c) array (NumPy or torch, any device) into a C-contiguous
	structure-of-arrays (c, n) tensor on the configured device. A (c,) single vector becomes (c, 1).
	"""
	dtype = _DTYPE if dtype is None else dtype
	t = torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x)
	if t.ndim == 1:
		t = t[None, :]
	if t.ndim != 2 or t.shape[1] != n_components:
		raise ValueError(f'expected an (n, {n_components}) array, got shape {tuple(t.shape)}')
	return t.to(device=get_device(), dtype=dtype).T.contiguous()


def as_device_vector(x, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
	"""Import boundary for (n,) arrays."""
	dtype = _DTYPE if dtype is None else dtype
	t = torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x)
	return t.to(device=get_device(), dtype=dtype).contiguous()


def check_wide(x: torch.Tensor, n_components: int, name: str = 'array') -> torch.Tensor:
	"""
	Validate a (c, n) structure-of-arrays tensor already on the configured device (metadata only, no sync).
	A single (c,) vector is accepted and returned as (c, 1).
	"""
	if not isinstance(x, torch.Tensor):
		raise TypeError(f'{name}: expected a torch.Tensor of shape ({n_components}, n); use as_wide() to import arrays')
	if x.device != get_device():
		raise ValueError(f'{name} is on {x.device}, expected {get_device()} (no implicit transfers)')
	if x.ndim == 1:
		x = x[:, None]
	if x.ndim != 2 or x.shape[0] != n_components:
		raise ValueError(f'{name}: expected shape ({n_components}, n), got {tuple(x.shape)}')
	return x


def to_numpy(x: torch.Tensor) -> np.ndarray:
	"""
	Export boundary: (c, n) structure-of-arrays tensor -> (n, c) NumPy array (the original Python API layout);
	(n,) -> (n,). This is where the device -> host transfer (and sync) happens.
	"""
	x = x.detach()
	if x.ndim == 2:
		return x.T.contiguous().cpu().numpy()
	return x.cpu().numpy()
