"""
Scene format: one JSON manifest + one binary blob, in a single little-endian file (extension .sdfx), readable from
Python (this module) and JavaScript (sdfview/web/scene.js) without dependencies beyond NumPy / typed arrays.

File layout (all integers little-endian uint32):

	offset 0    magic  b'SDFX'
	offset 4    version (1)
	offset 8    J = byte length of the JSON chunk (padded with spaces to a multiple of 16)
	offset 12   B = byte length of the binary chunk
	offset 16   JSON manifest, UTF-8, J bytes
	offset 16+J binary chunk, B bytes; every array starts at a multiple of 16 bytes (typed-array / GPU friendly)

Manifest (JSON):

	{
	  "asset":  {"format": "sdfx-scene", "version": 1, "generator": "..."},
	  "bounds": {"min": [x, y, z], "max": [x, y, z]},           # region of interest: camera framing, cut plane
	  "arrays": {name: {"dtype": "float32" | "int32" | "uint32" | "uint8", "shape": [...],
	                    "byteOffset": int, "byteLength": int}},   # byteOffset relative to the binary chunk
	  "fields": [field, ...],                                     # signed distance fields, see FIELD TYPES
	  "pointclouds": [{"name": str, "positions": array name, "normals": array name or null,
	                   "colors": array name or null, "radius": float}],
	  "meta": {...}                                               # free-form provenance (not interpreted)
	}

Arrays are C-ordered, row = one GPU texel / struct when the last dimension is 4 (or a multiple of 4): they are stored
in the layout the shaders fetch, so a backend uploads them without reshuffling (an RGBA32F/RGBA32I texture in GL /
WebGL2, a storage buffer in WebGPU). The torch code works structure-of-arrays (3, n); the conversion to these
array-of-structures rows happens once, at export (sdfview.sources.*).

FIELD TYPES

	"pat" — Points as Tori: φ(q) = Σ_{i ∈ kNN(q)} w_i g_i(q) / Σ w_i  (see points_as_tori/torus_distance.py)
	    arrays: points     (N, 4) float32  [x, y, z, outlier flag (1 = excluded from the blend)]
	            tori       (N, 8) float32  [center x, y, z, R,  axis x, y, z, r]   (2 texels per torus; axis is the
	                                       unit 3D axis; sign(r) encodes orientation)
	            bvh_bounds (M, 8) float32  [min x, y, z, 0,  max x, y, z, 0]       (2 texels per node)
	            bvh_nodes  (M, 4) int32    [left, right, prim_start, prim_count]   (prim_count = 0: internal node)
	            bvh_prims  (N, 1) int32    point index of each leaf slot
	    params: k, lambda_scale, max_exp_arg (64), bvh_depth (max root-to-leaf edges), isovalue
	"grid" — samples of any SDF at the vertices of a regular grid spanning the field's "bounds"
	    arrays: values (D, H, W) float32, values[z, y, x] = φ(min + (x, y, z) / (W-1, H-1, D-1) · (max - min))
	    params: isovalue;  "bounds": {"min", "max"} of the grid
	    Any SDF evaluated with torch (PAT, SUP, a mesh SDF, an analytic shape) can be exported this way.

Optional keys of a field or point cloud:
	"model" — name of the model (fit) it belongs to: fields of one model (e.g. PAT's exact field and its grid bake)
	          share it, so viewers can toggle or overlay models (one field per model). Default: the field's name.

Unknown field types and keys must be ignored by readers (forward compatibility).
"""

from __future__ import annotations

import json
import struct
from typing import Dict, List, Optional

import numpy as np

MAGIC = b'SDFX'
VERSION = 1
ALIGN = 16
DTYPES = {'float32': np.float32, 'int32': np.int32, 'uint32': np.uint32, 'uint8': np.uint8}


def _pad(n: int, align: int = ALIGN) -> int:
	return (n + align - 1) // align * align


class Scene:
	"""In-memory scene: a manifest dict and named NumPy arrays (host memory; this is the export side of torch)."""

	def __init__(self, bounds_min=None, bounds_max=None, meta: Optional[dict] = None):
		self.bounds_min = None if bounds_min is None else [float(x) for x in bounds_min]
		self.bounds_max = None if bounds_max is None else [float(x) for x in bounds_max]
		self.arrays: Dict[str, np.ndarray] = {}
		self.fields: List[dict] = []
		self.pointclouds: List[dict] = []
		self.meta = dict(meta or {})

	# ---- building

	def add_array(self, name: str, array) -> str:
		a = np.ascontiguousarray(array)
		key = next((k for k, v in DTYPES.items() if a.dtype == v), None)
		if key is None:
			raise TypeError(f'{name}: unsupported dtype {a.dtype} (use one of {list(DTYPES)})')
		if name in self.arrays:
			raise KeyError(f'array {name!r} already exists')
		self.arrays[name] = a
		return name

	def add_field(self, field: dict) -> dict:
		if 'type' not in field or 'name' not in field:
			raise ValueError('a field needs "name" and "type"')
		for ref in field.get('arrays', {}).values():
			if ref not in self.arrays:
				raise KeyError(f'field {field["name"]!r} references missing array {ref!r}')
		self.fields.append(field)
		return field

	def add_pointcloud(
		self,
		name: str,
		positions: str,
		normals: Optional[str] = None,
		colors: Optional[str] = None,
		radius: float = 0.005,
	) -> dict:
		for ref in (positions, normals, colors):
			if ref is not None and ref not in self.arrays:
				raise KeyError(f'point cloud {name!r} references missing array {ref!r}')
		pc = {'name': name, 'positions': positions, 'normals': normals, 'colors': colors, 'radius': float(radius)}
		self.pointclouds.append(pc)
		return pc

	def field(self, name_or_type: str) -> Optional[dict]:
		"""First field with this name, else first field of this type, else None."""
		for f in self.fields:
			if f['name'] == name_or_type:
				return f
		return next((f for f in self.fields if f['type'] == name_or_type), None)

	def field_arrays(self, field: dict) -> Dict[str, np.ndarray]:
		return {k: self.arrays[v] for k, v in field.get('arrays', {}).items()}

	# ---- serialization

	def manifest(self, offsets: Optional[Dict[str, int]] = None) -> dict:
		arrays = {}
		for name, a in self.arrays.items():
			arrays[name] = {
				'dtype': next(k for k, v in DTYPES.items() if a.dtype == v),
				'shape': list(a.shape),
				'byteOffset': None if offsets is None else offsets[name],
				'byteLength': int(a.nbytes),
			}
		m = {
			'asset': {'format': 'sdfx-scene', 'version': VERSION, 'generator': 'sdfview'},
			'arrays': arrays,
			'fields': self.fields,
			'pointclouds': self.pointclouds,
			'meta': self.meta,
		}
		if self.bounds_min is not None:
			m['bounds'] = {'min': self.bounds_min, 'max': self.bounds_max}
		return m

	def to_bytes(self) -> bytes:
		offsets, pos = {}, 0
		for name, a in self.arrays.items():
			offsets[name] = pos
			pos = _pad(pos + a.nbytes)
		blob = bytearray(pos)
		for name, a in self.arrays.items():
			blob[offsets[name] : offsets[name] + a.nbytes] = a.astype(a.dtype.newbyteorder('<'), copy=False).tobytes()
		js = json.dumps(self.manifest(offsets), separators=(',', ':')).encode('utf-8')
		js += b' ' * (_pad(len(js)) - len(js))
		return MAGIC + struct.pack('<III', VERSION, len(js), len(blob)) + js + bytes(blob)

	def save(self, path: str) -> None:
		with open(path, 'wb') as f:
			f.write(self.to_bytes())

	@classmethod
	def from_bytes(cls, data: bytes) -> 'Scene':
		if data[:4] != MAGIC:
			raise ValueError('not an sdfx scene (bad magic)')
		version, json_len, bin_len = struct.unpack_from('<III', data, 4)
		if version > VERSION:
			raise ValueError(f'scene version {version} is newer than this reader ({VERSION})')
		m = json.loads(data[16 : 16 + json_len].decode('utf-8'))
		base = 16 + json_len
		scene = cls(m.get('bounds', {}).get('min'), m.get('bounds', {}).get('max'), m.get('meta'))
		for name, spec in m['arrays'].items():
			dt = np.dtype(DTYPES[spec['dtype']]).newbyteorder('<')
			count = spec['byteLength'] // dt.itemsize
			a = np.frombuffer(data, dtype=dt, count=count, offset=base + spec['byteOffset'])
			scene.arrays[name] = a.reshape(spec['shape']).astype(dt.newbyteorder('='), copy=False)
		scene.fields = m.get('fields', [])
		scene.pointclouds = m.get('pointclouds', [])
		return scene

	@classmethod
	def load(cls, path: str) -> 'Scene':
		with open(path, 'rb') as f:
			return cls.from_bytes(f.read())

	def __repr__(self) -> str:
		fields = ', '.join(f'{f["name"]}:{f["type"]}' for f in self.fields)
		pcs = ', '.join(f'{p["name"]}({self.arrays[p["positions"]].shape[0]})' for p in self.pointclouds)
		return f'Scene(fields=[{fields}], pointclouds=[{pcs}], bounds={self.bounds_min}..{self.bounds_max})'


def grid_field(scene: Scene, name: str, values: np.ndarray, bounds_min, bounds_max, isovalue: float = 0.0) -> dict:
	"""Add a "grid" field: values (D, H, W) = φ at vertex-aligned samples over [bounds_min, bounds_max] (x fastest)."""
	values = np.asarray(values, dtype=np.float32)
	if values.ndim != 3 or min(values.shape) < 2:
		raise ValueError(f'grid values must be (D, H, W) with every size >= 2, got {values.shape}')
	scene.add_array(f'{name}/values', values)
	return scene.add_field(
		{
			'name': name,
			'type': 'grid',
			'bounds': {'min': [float(x) for x in bounds_min], 'max': [float(x) for x in bounds_max]},
			'params': {'isovalue': float(isovalue)},
			'arrays': {'values': f'{name}/values'},
		}
	)


def model_of(item: dict) -> str:
	"""Model name of a field or point cloud ("model" key, else its name)."""
	return item.get('model', item['name'])


def merge_scenes(scenes: Dict[str, Scene], meta: Optional[dict] = None) -> Scene:
	"""
	One scene holding several models' scenes, e.g. {'pat': scene_a, 'sup': scene_b}: every array is renamed
	'<model>/<name>', every field and point cloud gets "model": <model> and the name '<model>' (if its scene has one
	field) or '<model>.<name>'; bounds = union of the scenes' bounds; meta = {'models': {model: scene meta}}.
	"""
	lo, hi = [], []
	merged_meta = dict(meta or {})
	merged_meta['models'] = {}
	out_arrays, out_fields, out_pcs = {}, [], []
	for model, scene in scenes.items():
		rename = {name: f'{model}/{name}' for name in scene.arrays}
		out_arrays.update({rename[n]: a for n, a in scene.arrays.items()})
		for f in scene.fields:
			g = json.loads(json.dumps(f))  # deep copy (plain JSON data)
			g['name'] = model if len(scene.fields) == 1 else f'{model}.{f["name"]}'
			g['model'] = model
			g['arrays'] = {k: rename[v] for k, v in f.get('arrays', {}).items()}
			out_fields.append(g)
		for pc in scene.pointclouds:
			q = dict(pc, name=f'{model}.{pc["name"]}', model=model)
			for key in ('positions', 'normals', 'colors'):
				if q.get(key):
					q[key] = rename[q[key]]
			out_pcs.append(q)
		if scene.bounds_min is not None:
			lo.append(scene.bounds_min)
			hi.append(scene.bounds_max)
		merged_meta['models'][model] = scene.meta
	out = Scene(np.min(lo, axis=0) if lo else None, np.max(hi, axis=0) if hi else None, merged_meta)
	for name, a in out_arrays.items():
		out.add_array(name, a)
	for f in out_fields:
		out.add_field(f)
	out.pointclouds = out_pcs
	return out
