"""
slangpy backend: the Slang modules of sdfview/shaders/slang (surface, probe, field, field_pat, field_grid, camera;
common and bvh from points_as_tori/slang, shared with the fused k-NN kernel) called as compute kernels through
slangpy: one thread per pixel (render) or per query (probe), no rasterization.

	r = Renderer(scene)                            # a Vulkan device by default; or Renderer(scene, device)
	rgba = r.render(camera, settings, (w, h))      # (h, w, 4) uint8, first row = top (as the moderngl Renderer)
	out = r.probe(queries, 'auto', settings)       # (n, 4) float32: (φ - isovalue, ∇φ)

Steps, as in the other hosts (sdfview/PARALLEL.md):
  1. context     spy.create_device(include_paths=include_paths()); spy.Module.load_from_file('surface.slang', ...)
  2. upload      every scene array as a slangpy tensor, once ((rows, 4) rows as the textures' texels; grid (d, h, w))
  3. programs    the generic functions surface<F : IField> and probe<F : IField>, specialized by the field argument
                 ({'_type': 'PatField' | 'GridField', ...}); PAT: BVH_K_MAX / BVH_STACK_SIZE link-time constants
  4. per frame   color (h, w, 4) and depth (h, w) tensors cleared, one surface call per pass over spy.grid((h, w)),
                 color read back once
Field types: 'pat', 'grid' ('glsl', the generated GLSL of a SuperFit assembly, has no Slang mirror: its grid bake is
drawn). Surfaces only: no cut plane, no points.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

from ... import view
from ...camera import OrbitCamera
from ...scene import Scene
from ...settings import ViewSettings
from ...shaders import SLANG_DIR

SUPPORTED = ('pat', 'grid')


def include_paths() -> list:
	"""Slang include paths of the modules: sdfview/shaders/slang, then points_as_tori/slang (common, bvh)."""
	import points_as_tori

	return [SLANG_DIR, os.path.join(os.path.dirname(points_as_tori.__file__), 'slang')]


def _host(a: np.ndarray, dtype=None) -> np.ndarray:
	"""C-contiguous and writable (spy.Tensor.from_numpy rejects read-only arrays, e.g. the scene's views of its file
	buffer): a copy only when needed."""
	return np.require(a, dtype, ['C', 'W'])


def _rows4(a: np.ndarray) -> np.ndarray:
	"""(n, 4 t) -> (n t, 4): t rows per element, as the texels of the GL textures (common.slang fetch_texel)."""
	a = _host(a)
	return a.reshape(-1, 4) if a.shape[-1] % 4 == 0 else a


class Renderer:
	SUPPORTED = SUPPORTED

	def __init__(self, scene: Scene, device=None, device_type: str = 'vulkan'):
		import slangpy as spy

		self.spy = spy
		self.scene = scene
		# a device given by the caller must have been created with include_paths=include_paths()
		self.device = device or spy.create_device(
			type=getattr(spy.DeviceType, device_type), include_paths=include_paths()
		)
		self.surface_module = spy.Module.load_from_file(self.device, 'surface.slang')
		self.probe_module = spy.Module.load_from_file(self.device, 'probe.slang')
		self.tensors: Dict[str, object] = {}
		for f in scene.fields:
			arrays = scene.field_arrays(f)
			if f['type'] == 'pat':
				for key in ('points', 'tori', 'bvh_bounds', 'bvh_nodes', 'bvh_prims'):
					self.tensors[f'{f["name"]}/{key}'] = spy.Tensor.from_numpy(self.device, _rows4(arrays[key]))
			elif f['type'] == 'grid':
				values = _host(arrays['values'], np.float32)
				self.tensors[f'{f["name"]}/values'] = spy.Tensor.from_numpy(self.device, values)
		self.box_min, self.box_max = view.scene_box(scene)
		self.diagonal = float(np.linalg.norm(self.box_max - self.box_min))
		self._functions: Dict[tuple, object] = {}

	def supports(self, field: dict) -> bool:
		return field['type'] in SUPPORTED

	# ---- field arguments (the GLSL's field uniforms and textures)

	@staticmethod
	def _k_max(field: dict) -> int:
		return max(32, int(field['params'].get('k', 32)))

	def _field(self, field: dict, s: ViewSettings) -> dict:
		"""The IField struct value of `field` ('_type' selects the struct, and so the specialization)."""
		spy, name, params = self.spy, field['name'], field.get('params', {})
		if field['type'] == 'pat':
			k = int(s.k if s.k is not None else params.get('k', 32))
			out = {
				key: self.tensors[f'{name}/{key}'] for key in ('points', 'tori', 'bvh_bounds', 'bvh_nodes', 'bvh_prims')
			}
			return {
				'_type': 'PatField',
				**out,
				'k': min(k, self._k_max(field)),
				'lambda_scale': float(s.lambda_scale or params.get('lambda_scale', 1.0)),
				'max_exp_arg': float(params.get('max_exp_arg', 64.0)),
			}
		if field['type'] == 'grid':
			d, h, w = self.scene.arrays[field['arrays']['values']].shape
			return {
				'_type': 'GridField',
				'values': self.tensors[f'{name}/values'],
				'res': spy.int3(w, h, d),
				'grid_min': spy.float3(*map(float, field['bounds']['min'])),
				'grid_max': spy.float3(*map(float, field['bounds']['max'])),
			}
		raise ValueError(f'field type {field["type"]!r} has no Slang mirror (supported: {SUPPORTED})')

	@staticmethod
	def _isovalue(field: dict, s: ViewSettings) -> float:
		return float(field.get('params', {}).get('isovalue', 0.0)) + s.isovalue

	def _function(self, module, name: str, field: dict):
		"""module.name with the field's link-time constants (PAT: the k-NN array and stack sizes), cached."""
		key = (name, field['name'])
		if key not in self._functions:
			fn = getattr(module, name)
			if field['type'] == 'pat':
				depth = int(field['params'].get('bvh_depth', 62))
				fn = fn.constants({'BVH_K_MAX': self._k_max(field), 'BVH_STACK_SIZE': max(64, depth + 2)})
			self._functions[key] = fn
		return self._functions[key]

	# ---- render

	def _camera(self, camera: OrbitCamera, w: int, h: int) -> dict:
		vp = camera.proj_matrix(w / max(h, 1)) @ camera.view_matrix()
		spy = self.spy
		return {
			'view_proj': spy.float4x4(vp.astype(np.float32)),
			'inv_view_proj': spy.float4x4(np.linalg.inv(vp).astype(np.float32)),
			'viewport': spy.float4(0.0, 0.0, float(w), float(h)),
		}

	def render(self, camera: OrbitCamera, settings: Optional[ViewSettings] = None, size=(800, 600)) -> np.ndarray:
		"""(height, width, 4) uint8 RGBA, first row = top of the image."""
		spy, s = self.spy, settings or ViewSettings()
		w, h = int(size[0]), int(size[1])
		clear = np.empty((h, w, 4), np.float32)
		clear[...] = (*s.background, 1.0)
		color = spy.Tensor.from_numpy(self.device, clear)
		depth = spy.Tensor.from_numpy(self.device, np.ones((h, w), np.float32))
		cam = self._camera(camera, w, h)
		eps = s.epsilon if s.epsilon is not None else 1e-4 * self.diagonal
		passes = view.surface_passes(self.scene, s, SUPPORTED)
		# opaque passes first, then the translucent ones blended over them (as the GL hosts)
		for field, front, back, opacity in sorted(passes, key=lambda p: p[3] < 1.0):
			trace = {
				'box_min': spy.float3(*self.box_min),
				'box_max': spy.float3(*self.box_max),
				'max_steps': int(s.max_steps),
				'epsilon': float(eps),
				'step_scale': float(s.step_scale),
				'clip': 0,
				'clip_plane': spy.float4(0.0, 0.0, 1.0, 0.0),
				'isovalue': self._isovalue(field, s),
			}
			shading = {
				'color_front': spy.float3(*front),
				'color_back': spy.float3(*back),
				'opacity': float(opacity),
				'color_parts': int(s.color_parts),
			}
			fn = self._function(self.surface_module, 'surface', field)
			fn(spy.grid((h, w)), self._field(field, s), trace, shading, cam, color, depth)
		rgba = np.round(np.clip(color.to_numpy(), 0.0, 1.0) * 255.0).astype(np.uint8)  # as GL's float -> unorm8
		return rgba[::-1].copy()  # rows bottom-up (window coordinates) -> top first

	# ---- probes

	def probe(self, queries: np.ndarray, field: str = 'auto', settings: Optional[ViewSettings] = None) -> np.ndarray:
		"""(n, 3) points -> (n, 4) float32: (φ - isovalue, ∇φ) computed by the Slang field."""
		s = settings or ViewSettings()
		f = view.resolve_field(self.scene, field, False, SUPPORTED)
		q = np.ascontiguousarray(np.asarray(queries, np.float32).reshape(-1, 3))
		fn = self._function(self.probe_module, 'probe', f)
		out = fn(self._field(f, s), self._isovalue(f, s), self.spy.Tensor.from_numpy(self.device, q), _result='numpy')
		return np.asarray(out, np.float32).reshape(-1, 4)

	def release(self) -> None:
		self.tensors, self._functions = {}, {}
