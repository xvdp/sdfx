"""
SuperFit primitive assemblies (ext/superfit, "SUP") as torch signed distance functions.

Wrapper, not a port: the primitive SDFs, the expression classes and the evaluation are SuperFit's and geolipi's, imported
at run time; no SuperFit code is copied here. SuperFit is under the Adobe Research License (non-commercial research
use; derivative works carry the same restriction); geolipi and sysl3d are MIT. Using this module runs SuperFit code,
under SuperFit's license.

Assembly file (primitive_assembly.pkl, written by SuperFit's resfit through Stats.get_dict()): a flat dict with dotted
keys. Each program (e.g. 'iter_{k}.best_program') is a geolipi state {'expr_str', 'symbol_tensor_map', 'GLFunction'},
'expr_str' a Python expression with the parameters inlined, e.g.

	SmoothUnion(SmoothUnion(P1, P2, (k1,)), P3, (k2,))
	P = PrimitiveMarker(StochasticPrimitive(Translate3D(AxisAngleRotate3D(VarAxisSF(size, ...), axis_angle), t), logits))

Evaluation (SuperFit's deterministic reduction, fetch_singular_expr_eval with relaxed_eval=False, then geolipi's
recursive_evaluate):
	* a primitive is kept iff argmax(keep, drop logits) = keep; dropped primitives are removed from the union;
	* a variable-axis primitive becomes its X / Y / Z variant by argmax of its axis logits;
	* local coordinates q = R(a) (p - t), R(a) the Rodrigues rotation of the axis-angle a;
	* smooth union of two SDFs a, b with parameter k (geolipi sdf_smooth_union):
	      h = clamp(1/2 + (b - a) / (2 (k + 1e-9)), 0, 1),   d = b (1 - h) + a h - k h (1 - h)
	  nested left to right over the primitives.
SuperFit's own reconstruction metrics instead take the best of ~100 Gumbel samples of the keep/drop and axis choices
(prune.py), so the argmax assembly can differ from the one its metrics describe.

Frames: SuperFit fits in a normalized frame, the input mesh centred on its bounding box and scaled into [-0.9, 0.9]³,
	p_n = s (p - c),   s = 1.8 / max extent of the bounding box,   c = bounding-box centre,
and evaluates over [-1, 1]³. The transform is not stored in the assembly; given the input mesh, the SDF in the mesh's
frame is φ(p) = φ_n(s (p - c)) / s.

Loading: pickles can execute code. load_assembly unpickles with an allow-list (builtin containers and scalars, torch
tensors and storages, NumPy arrays and scalars); anything else raises pickle.UnpicklingError. Tensor storages, which
pickle as bytes reloaded by torch.load(weights_only=False), are reloaded with weights_only=True.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import io
import os
import pickle
import re
import sys
import types
import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
QUERY_CHUNK = 1 << 18  # queries per geolipi evaluation

_ALLOWED = {
	'builtins': {'dict', 'list', 'tuple', 'set', 'frozenset', 'int', 'float', 'complex', 'bool', 'str', 'bytes',
				 'bytearray', 'slice', 'range', 'NoneType'},
	'collections': {'OrderedDict', 'defaultdict'},
	'_codecs': {'encode'},
	'copyreg': {'_reconstructor'},
	'torch': {'Size', 'device', 'dtype', 'float16', 'float32', 'float64', 'bfloat16', 'int8', 'int16', 'int32', 'int64',
			  'uint8', 'bool'},
	'torch._utils': {'_rebuild_tensor', '_rebuild_tensor_v2', '_rebuild_parameter', '_rebuild_parameter_with_state'},
	'torch.storage': {'UntypedStorage', 'TypedStorage'},
	'numpy': {'ndarray', 'dtype'},
	'numpy.core.multiarray': {'_reconstruct', 'scalar'},
	'numpy._core.multiarray': {'_reconstruct', 'scalar'},
}  # fmt: skip
_STORAGE = re.compile(r'^[A-Za-z0-9]*Storage$')  # torch.FloatStorage, torch.DoubleStorage, ...


def _storage_from_bytes(b: bytes):
	"""Replaces torch.storage._load_from_bytes, which calls torch.load(..., weights_only=False) on the embedded bytes
	(arbitrary code); torch's weights-only unpickler accepts storages and tensors only."""
	return torch.load(io.BytesIO(b), weights_only=True)


class _AllowListUnpickler(pickle.Unpickler):
	def find_class(self, module, name):
		if module == 'torch.storage' and name == '_load_from_bytes':
			return _storage_from_bytes
		if name in _ALLOWED.get(module, ()) or (module == 'torch' and _STORAGE.match(name)):
			return super().find_class(module, name)
		if module in ('numpy.dtypes',) and name.endswith('DType'):
			return super().find_class(module, name)
		raise pickle.UnpicklingError(f'{module}.{name} is not allowed in a SuperFit assembly file')

	def persistent_load(self, pid):  # torch.save-style persistent ids are not used by SuperFit's pickle.dump
		raise pickle.UnpicklingError('persistent ids are not allowed in a SuperFit assembly file')


def load_assembly(path: str) -> dict:
	"""The flat dict SuperFit pickled (tensors on the CPU). Allow-list unpickling (see the module docstring)."""
	with open(path, 'rb') as f:
		data = _AllowListUnpickler(f).load()
	if not isinstance(data, dict):
		raise ValueError(f'{path}: expected a dict, got {type(data).__name__}')
	return data


def program_keys(data: dict) -> list:
	"""Keys of the stored programs (geolipi states), in file order."""
	return [k for k, v in data.items() if isinstance(v, dict) and v.get('GLFunction') and 'expr_str' in v]


def select_program(data: dict, which: str = 'best') -> Tuple[str, dict]:
	"""
	(key, state) of a stored program.
	  'best'    as SuperFit's get_best_expr (utils/io.py): iter_{n_iters - 1}.best_program, else .pruned_program;
	  'latest'  best_program of the highest recorded iteration (after an early stop SuperFit records iteration
	            n_iters, which get_best_expr does not read; best_program is the best so far, so this is at least as
	            good by SuperFit's objective);
	  a key     e.g. 'iter_2.pruned_program'.
	"""
	if which not in ('best', 'latest'):
		if which not in data:
			raise KeyError(f'{which!r} not in the assembly; programs: {program_keys(data)}')
		return which, data[which]
	if which == 'best' and 'n_iters' in data:
		n = int(data['n_iters']) - 1
		for name in ('best_program', 'pruned_program'):
			key = f'iter_{n}.{name}'
			if key in data:
				return key, data[key]
	iters = sorted({int(m.group(1)) for k in data if (m := re.match(r'iter_(\d+)\.best_program$', k))}, reverse=True)
	if not iters:
		raise KeyError(f'no iter_*.best_program in the assembly; programs: {program_keys(data)}')
	key = f'iter_{iters[0]}.best_program'
	return key, data[key]


def superfit_root() -> str:
	"""Directory of the `superfit` package: $SUPERFIT_ROOT, an installed package, or ext/superfit of this repo."""
	env = os.environ.get('SUPERFIT_ROOT')
	candidates = [env] if env else []
	spec = importlib.util.find_spec('superfit')  # locates without executing the package
	if spec is not None and spec.submodule_search_locations:
		candidates += list(spec.submodule_search_locations)
	candidates.append(os.path.join(REPO, 'ext', 'superfit', 'superfit'))
	for c in candidates:
		if c and os.path.isfile(os.path.join(c, 'symbolic', '__init__.py')):
			return c
	raise ImportError('SuperFit not found: set SUPERFIT_ROOT to its `superfit` package directory or init ext/superfit')


def import_superfit():
	"""
	Import SuperFit's expression classes and evaluation (superfit.symbolic, superfit.torch_compute, geolipi, sysl).
	superfit/__init__.py imports the whole fitting stack (cubvh, kaolin, fastsweep, drjit, cc3d, igl); evaluation needs
	none of it. If superfit is not already imported, the package is registered without running its __init__ and the
	two subpackages are imported directly. Returns (geolipi.symbolic, recursive_evaluate, Sketcher,
	fetch_singular_expr_eval, superfit.symbolic.primitives).
	"""
	try:
		import geolipi.symbolic as gls
		from geolipi.torch_compute import Sketcher, recursive_evaluate
		import sysl  # noqa: F401
	except ImportError as e:
		raise ImportError('SuperFit assemblies need geolipi and sysl3d: pip install -e ".[superfit]"') from e
	if 'superfit' not in sys.modules:
		root = superfit_root()
		package = types.ModuleType('superfit')
		package.__path__ = [root]
		package.__file__ = None
		# a package spec, so importlib.util.find_spec('superfit') keeps working (it raises on __spec__ = None)
		package.__spec__ = importlib.machinery.ModuleSpec('superfit', None, is_package=True)
		package.__spec__.submodule_search_locations = [root]
		sys.modules['superfit'] = package
	import superfit.symbolic
	import superfit.symbolic.primitives as primitives
	import superfit.torch_compute  # noqa: F401  (registers the primitives' torch evaluation with geolipi)
	from superfit.symbolic.utils import fetch_singular_expr_eval

	return gls, recursive_evaluate, Sketcher, fetch_singular_expr_eval, primitives


def mesh_normalization(vertices) -> Tuple[float, np.ndarray]:
	"""(s, c) of SuperFit's normalization p_n = s (p - c) for a mesh with these (n, 3) vertices (see module doc)."""
	v = np.asarray(vertices, dtype=np.float64)
	lo, hi = v.min(axis=0), v.max(axis=0)
	return 1.8 / float((hi - lo).max()), (lo + hi) / 2.0


def _param(node, index: int) -> float:
	"""First value of argument `index` of a geolipi node (tensors live in the node's lookup table, keyed by symbol)."""
	arg = node.args[index]
	value = node.lookup_table.get(arg, arg) if hasattr(node, 'lookup_table') else arg
	return float(torch.as_tensor(value).flatten()[0]) if not hasattr(value, 'free_symbols') else float(value)


def _primitive_nodes(expr, gls):
	if isinstance(expr, gls.Primitive3D):
		yield expr
		return
	for arg in getattr(expr, 'args', ()):
		if isinstance(arg, gls.GLFunction):
			yield from _primitive_nodes(arg, gls)


class SuperFitSDF:
	"""
	SDF of a SuperFit primitive assembly: __call__((3, Q) tensor) -> (Q,) tensor, on `device` in `dtype`.

	Args:
		assembly: path to primitive_assembly.pkl, its loaded dict, or a geolipi state {'expr_str', ...}
		program: 'best', 'latest' or a key (see select_program)
		mesh: input mesh (path, or (n, 3) vertices) to evaluate in its frame instead of SuperFit's normalized frame
		device, dtype: where and in which precision the expression is evaluated (device default: cuda if available,
			else cpu)
	"""

	def __init__(
		self,
		assembly: Union[str, dict],
		program: str = 'best',
		mesh=None,
		device: Optional[Union[str, torch.device]] = None,
		dtype: torch.dtype = torch.float32,
	):
		gls, self._evaluate, Sketcher, fetch_singular_expr_eval, primitives = import_superfit()
		if device is None:
			device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.device, self.dtype = torch.device(device), dtype
		data = load_assembly(assembly) if isinstance(assembly, str) else assembly
		if isinstance(data, dict) and 'expr_str' in data:
			self.data, self.program, state = {}, 'expr', data
		else:
			self.data = data
			self.program, state = select_program(data, program)
		expr = gls.GLFunction.from_state(state).tensor(dtype=dtype, device=self.device)
		self.expression = fetch_singular_expr_eval(expr, relaxed_eval=False, device=self.device)
		self._sketcher = Sketcher(device=self.device, dtype=dtype, n_dims=3, resolution=2)
		prims = list(_primitive_nodes(self.expression, gls))
		self.n_primitives = len(prims)
		self.primitive_types = [type(p).__name__ for p in prims]
		zero_bulge = [i for i, p in enumerate(prims) if isinstance(p, primitives.SuperFrustum) and _param(p, 4) == 0.0]
		if zero_bulge:
			warnings.warn(
				f"primitives {zero_bulge} have bulge = 0: SuperFit's torch SuperFrustum maps the local x coordinate to 0 "
				'for bulge exactly 0 (its GLSL skips the bulge map), so their SDF is wrong (see superfit_sdf/NOTES.md)',
				RuntimeWarning,
				stacklevel=2,
			)
		self.scale, self.center = 1.0, np.zeros(3)
		if mesh is not None:
			vertices = mesh if not isinstance(mesh, str) else _read_mesh_vertices(mesh)
			self.scale, self.center = mesh_normalization(vertices)
		self._center = torch.as_tensor(self.center, dtype=dtype, device=self.device)[:, None]

	def normalized(self, queries: torch.Tensor) -> torch.Tensor:
		"""SDF in SuperFit's normalized frame: (3, Q) -> (Q,)."""
		q = queries.to(device=self.device, dtype=self.dtype)
		out = torch.empty(q.shape[1], dtype=self.dtype, device=self.device)
		for s in range(0, q.shape[1], QUERY_CHUNK):
			e = min(q.shape[1], s + QUERY_CHUNK)
			out[s:e] = self._evaluate(self.expression, self._sketcher, coords=q[:, s:e].T.contiguous()).reshape(-1)
		return out

	def __call__(self, queries: torch.Tensor) -> torch.Tensor:
		"""SDF in the mesh frame if a mesh was given, else in the normalized frame: (3, Q) -> (Q,)."""
		if self.scale == 1.0 and not self.center.any():
			return self.normalized(queries)
		q = queries.to(device=self.device, dtype=self.dtype)
		return self.normalized((q - self._center) * self.scale) / self.scale

	def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
		"""SuperFit's evaluation domain [-1, 1]³, in the frame of __call__."""
		lo, hi = -np.ones(3) / self.scale + self.center, np.ones(3) / self.scale + self.center
		return lo, hi

	def metrics(self) -> dict:
		"""Scalar entries of the assembly whose key contains 'evaluation.' or is a timing / count, as floats."""
		out = {}
		for k, v in self.data.items():
			if 'evaluation.' in k or k in ('total_time', 'n_iters', 'best_iou', 'best_obj', 'best_recon_measure'):
				try:
					out[k] = float(v)
				except (TypeError, ValueError):
					pass
		return out

	def __repr__(self) -> str:
		frame = 'normalized' if self.scale == 1.0 else f'mesh (s = {self.scale:.4g})'
		return (
			f'SuperFitSDF({self.program}: {self.n_primitives} primitives, {frame} frame, {self.device}, {self.dtype})'
		)


def _read_mesh_vertices(path: str) -> np.ndarray:
	"""Vertices as SuperFit reads the input mesh (trimesh, force='mesh', process=False)."""
	try:
		import trimesh
	except ImportError as e:
		raise ImportError('reading the input mesh needs trimesh (pip install trimesh), or pass its vertices') from e
	return np.asarray(trimesh.load(path, force='mesh', process=False).vertices)


def sdf_from_assembly(
	path: str, program: str = 'best', mesh: Optional[Union[str, Sequence]] = None, device=None, dtype=torch.float32
) -> SuperFitSDF:
	"""Convenience constructor (see SuperFitSDF)."""
	return SuperFitSDF(path, program=program, mesh=mesh, device=device, dtype=dtype)
