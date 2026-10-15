"""
moderngl backend: uploads a Scene to GL textures and draws it with the GLSL programs of sdfview/shaders/glsl.

Embedding: the Renderer only needs a moderngl.Context (OpenGL >= 3.3 core): a standalone/headless context, a
moderngl-window / pyglet / glfw / Qt window, or a context created by another application on the current thread
(moderngl.create_context()). `draw()` renders into the framebuffer bound by the caller; `render()` renders offscreen
and returns an image; `probe()` / `probe_knn()` evaluate the GLSL field / neighbor search at given points.

Passes (all with depth test): opaque surfaces (sphere tracing, writes gl_FragDepth), cut plane (rasterized
rectangle, field colormap), points (sprites), translucent surfaces (overlay of several models: blended, no depth write). Programs are compiled on first use per (stage pair, field), so the surface can
use a grid bake while the cut plane uses the exact field.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import moderngl
import numpy as np

from ...camera import OrbitCamera
from ...scene import Scene, model_of
from ...settings import ViewSettings, cut_plane
from ...shaders import glsl_source

TEXTURE_WIDTH = 4096  # texels per row of array textures (>= the WebGL2 minimum for MAX_TEXTURE_SIZE)

# surface colors of models (scenes with several models: sdfview.scene.merge_scenes)
MODEL_COLORS = {'pat': (0.86, 0.70, 0.45), 'sup': (0.42, 0.62, 0.90), 'gt': (0.72, 0.72, 0.72)}
EXTRA_COLORS = [(0.52, 0.78, 0.50), (0.80, 0.55, 0.82), (0.92, 0.55, 0.45), (0.45, 0.80, 0.80)]


def array_texture(ctx: moderngl.Context, array: np.ndarray, width: int = TEXTURE_WIDTH) -> moderngl.Texture:
	"""(n, 4·t) or (n, 1) array -> 2D texture of n·t RGBA (or R) texels, row-major, `width` texels per row."""
	a = np.ascontiguousarray(array)
	comps = 4 if a.shape[-1] % 4 == 0 else a.shape[-1]
	if comps not in (1, 4):
		raise ValueError(f'array texture rows must have 1 or a multiple of 4 values, got {a.shape}')
	texels = a.reshape(-1, comps)
	count = max(texels.shape[0], 1)
	w = min(width, count)
	h = (count + w - 1) // w
	padded = np.zeros((w * h, comps), dtype=a.dtype)
	padded[: texels.shape[0]] = texels
	dtype = {np.dtype(np.float32): 'f4', np.dtype(np.int32): 'i4', np.dtype(np.uint32): 'u4'}[a.dtype]
	tex = ctx.texture((w, h), comps, padded.tobytes(), dtype=dtype)
	tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
	return tex


def grid_texture(ctx: moderngl.Context, values: np.ndarray) -> moderngl.Texture3D:
	"""(D, H, W) float32 -> R32F 3D texture of size (W, H, D), linear filtering, clamped."""
	d, h, w = values.shape
	tex = ctx.texture3d((w, h, d), 1, np.ascontiguousarray(values, dtype=np.float32).tobytes(), dtype='f4')
	tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
	tex.repeat_x = tex.repeat_y = tex.repeat_z = False
	return tex


def _set(prog: moderngl.Program, name: str, value) -> None:
	"""Set a uniform if the compiled program kept it (unused uniforms are optimized out)."""
	if name not in prog:
		return
	u = prog[name]
	if isinstance(value, np.ndarray) and value.shape == (4, 4):
		u.write(np.ascontiguousarray(value.T, dtype='f4').tobytes())  # column-major
	elif isinstance(value, (tuple, list, np.ndarray)):
		u.value = tuple(float(x) for x in value)
	else:
		u.value = value


class Renderer:
	def __init__(self, ctx: moderngl.Context, scene: Scene, texture_width: int = TEXTURE_WIDTH):
		self.ctx = ctx
		self.scene = scene
		self.texture_width = min(texture_width, ctx.info['GL_MAX_TEXTURE_SIZE'])
		self._programs: Dict[Tuple, moderngl.Program] = {}
		self._vao: Dict[Tuple, moderngl.VertexArray] = {}
		self.textures: Dict[str, moderngl.Texture] = {}
		self._fields = {f['name']: f for f in scene.fields}
		for f in scene.fields:
			arrays = scene.field_arrays(f)
			if f['type'] == 'pat':
				for key in ('points', 'tori', 'bvh_bounds', 'bvh_nodes', 'bvh_prims'):
					self.textures[f'{f["name"]}/{key}'] = array_texture(ctx, arrays[key], self.texture_width)
			elif f['type'] == 'grid':
				self.textures[f'{f["name"]}/values'] = grid_texture(ctx, arrays['values'])
		for pc in scene.pointclouds:
			for key in ('positions', 'normals'):
				name = pc.get(key)
				if name and name not in self.textures:
					self.textures[name] = array_texture(ctx, scene.arrays[name], self.texture_width)
		if scene.bounds_min is not None:
			self.box_min, self.box_max = np.array(scene.bounds_min), np.array(scene.bounds_max)
		else:  # union of the fields' boxes, else the unit cube [-1, 1]³
			boxes = [(f['bounds']['min'], f['bounds']['max']) for f in scene.fields if 'bounds' in f]
			lo = np.min([b[0] for b in boxes], axis=0) if boxes else -np.ones(3)
			hi = np.max([b[1] for b in boxes], axis=0) if boxes else np.ones(3)
			self.box_min, self.box_max = np.asarray(lo, float), np.asarray(hi, float)
		self.diagonal = float(np.linalg.norm(self.box_max - self.box_min))

	# ---- programs

	def _field_defines(self, field: dict) -> dict:
		if field['type'] == 'pat':
			depth = int(field['params'].get('bvh_depth', 62))
			return {'FIELD_PAT': None, 'BVH_STACK_SIZE': max(64, depth + 2), 'BVH_K_MAX': self._k_max(field)}
		if field['type'] == 'grid':
			return {'FIELD_GRID': None}
		raise ValueError(f'unsupported field type {field["type"]!r}')

	@staticmethod
	def _k_max(field: dict) -> int:
		return max(32, int(field['params'].get('k', 32)))

	def program(self, vert: str, frag: str, field: Optional[dict] = None, extra: Optional[dict] = None):
		defines = dict(self._field_defines(field)) if field is not None else {}
		defines.update(extra or {})
		key = (vert, frag, field['name'] if field else None, tuple(sorted(defines.items(), key=str)))
		if key not in self._programs:
			self._programs[key] = self.ctx.program(
				vertex_shader=glsl_source(vert, 'gl330', defines), fragment_shader=glsl_source(frag, 'gl330', defines)
			)
			self._vao[key] = self.ctx.vertex_array(self._programs[key], [])
		return self._programs[key], self._vao[key]

	def _bind_field(self, prog: moderngl.Program, field: dict, settings: ViewSettings, unit: int = 0) -> int:
		"""Bind the field's textures and uniforms; returns the next free texture unit."""
		name, params = field['name'], field.get('params', {})
		_set(prog, 'field_isovalue', float(params.get('isovalue', 0.0)) + settings.isovalue)
		if field['type'] == 'pat':
			for key in ('points', 'tori', 'bvh_bounds', 'bvh_nodes', 'bvh_prims'):
				if f'pat_{key}' in prog:
					self.textures[f'{name}/{key}'].use(unit)
					prog[f'pat_{key}'].value = unit
					unit += 1
			k = int(settings.k if settings.k is not None else params.get('k', 32))
			_set(prog, 'pat_k', min(k, self._k_max(field)))
			_set(prog, 'pat_lambda_scale', float(settings.lambda_scale or params.get('lambda_scale', 1.0)))
			_set(prog, 'pat_max_exp_arg', float(params.get('max_exp_arg', 64.0)))
		elif field['type'] == 'grid':
			self.textures[f'{name}/values'].use(unit)
			_set(prog, 'grid_values', unit)
			_set(prog, 'grid_min', field['bounds']['min'])
			_set(prog, 'grid_max', field['bounds']['max'])
			unit += 1
		return unit

	def models(self) -> list:
		"""Model names of the fields, in field order (a field without a "model" key is its own model)."""
		out = []
		for f in self.scene.fields:
			if model_of(f) not in out:
				out.append(model_of(f))
		return out

	def model_field(self, model: str, for_surface: bool = True) -> Optional[dict]:
		"""The field drawn for a model: its grid for surfaces (fast to trace), its exact field for the cut plane."""
		fields = [f for f in self.scene.fields if model_of(f) == model]
		grids = [f for f in fields if f['type'] == 'grid']
		exact = [f for f in fields if f['type'] != 'grid']
		order = (grids + exact) if for_surface else (exact + grids)
		return order[0] if order else None

	def model_color(self, model: str, settings: ViewSettings):
		if settings.model_colors and model in settings.model_colors:
			return tuple(settings.model_colors[model])
		if model in MODEL_COLORS:
			return MODEL_COLORS[model]
		others = [m for m in self.models() if m not in MODEL_COLORS]
		return EXTRA_COLORS[others.index(model) % len(EXTRA_COLORS)] if model in others else EXTRA_COLORS[0]

	def resolve_field(self, which: str, for_surface: bool) -> Optional[dict]:
		if which != 'auto':
			f = self.scene.field(which)
			if f is None:
				raise KeyError(f'no field {which!r} in the scene')
			return f
		grids = [f for f in self.scene.fields if f['type'] == 'grid']
		exact = [f for f in self.scene.fields if f['type'] != 'grid']
		order = (grids + exact) if for_surface else (exact + grids)
		return order[0] if order else None

	# ---- drawing

	def _camera_uniforms(self, camera: OrbitCamera, viewport) -> dict:
		aspect = viewport[2] / max(viewport[3], 1)
		vp = camera.proj_matrix(aspect) @ camera.view_matrix()
		return {
			'u_view_proj': vp,
			'u_inv_view_proj': np.linalg.inv(vp),
			'u_viewport': tuple(float(x) for x in viewport),
			'u_proj11': 1.0 / math.tan(math.radians(camera.fov_y) / 2),
			'u_camera_dir': camera.direction(),
		}

	def draw(self, camera: OrbitCamera, settings: Optional[ViewSettings] = None, viewport=None) -> None:
		"""Draw into the currently bound framebuffer (clears color and depth inside `viewport`)."""
		s = settings or ViewSettings()
		ctx = self.ctx
		fbo = ctx.fbo
		viewport = tuple(viewport) if viewport is not None else fbo.viewport
		fbo.viewport = viewport
		fbo.clear(*s.background, 1.0, depth=1.0, viewport=viewport)
		ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)
		ctx.depth_func = '<'
		cam = self._camera_uniforms(camera, viewport)
		n, c, origin, u, v = cut_plane(s, self.box_min, self.box_max)
		eps = s.epsilon if s.epsilon is not None else 1e-4 * self.diagonal

		passes = self.surface_passes(s)
		for field, color_front, color_back, opacity in passes:
			if opacity >= 1.0:
				self._draw_surface(field, s, cam, (*n, c), eps, color_front, color_back, 1.0)

		cut_field = self.resolve_field(s.cut_field, False)
		if s.show_cutplane and cut_field is not None:
			prog, vao = self.program('cutplane.vert', 'cutplane.frag', cut_field)
			self._bind_field(prog, cut_field, s)
			for k, val in cam.items():
				_set(prog, k, val)
			_set(prog, 'u_plane_origin', origin)
			_set(prog, 'u_plane_u', u)
			_set(prog, 'u_plane_v', v)
			_set(prog, 'u_box_min', self.box_min)
			_set(prog, 'u_box_max', self.box_max)
			_set(prog, 'u_colormap_range', float(s.colormap_range or 0.5 * self.diagonal))
			_set(prog, 'u_isoband_frequency', float(s.isoband_frequency))
			_set(prog, 'u_opacity', float(s.cut_opacity))
			if s.cut_opacity < 1.0:
				ctx.enable(moderngl.BLEND)
				ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
			vao.render(moderngl.TRIANGLE_STRIP, vertices=4)
			ctx.disable(moderngl.BLEND)

		if s.show_points:
			for pc in self.scene.pointclouds:
				prog, vao = self.program('points.vert', 'points.frag')
				self.textures[pc['positions']].use(0)
				_set(prog, 'u_positions', 0)
				has_normals = bool(pc.get('normals'))
				if has_normals:
					self.textures[pc['normals']].use(1)
					_set(prog, 'u_normals', 1)
				_set(prog, 'u_has_normals', has_normals)
				for k, val in cam.items():
					_set(prog, k, val)
				_set(prog, 'u_point_radius', float(s.point_radius or pc.get('radius', 0.005)))
				_set(prog, 'u_point_color', s.point_color)
				vao.render(moderngl.POINTS, vertices=int(self.scene.arrays[pc['positions']].shape[0]))

		# translucent surfaces last, blended over everything drawn so far, without writing depth
		translucent = [p for p in passes if p[3] < 1.0]
		if translucent:
			ctx.enable(moderngl.BLEND)
			ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
			fbo.depth_mask = False
			for field, color_front, color_back, opacity in translucent:
				self._draw_surface(field, s, cam, (*n, c), eps, color_front, color_back, opacity)
			fbo.depth_mask = True
			ctx.disable(moderngl.BLEND)

	def surface_passes(self, s: ViewSettings) -> list:
		"""
		[(field, color_front, color_back, opacity)] of the surfaces to draw: surface_field alone ('single'), or one field
		per model ('overlay': surface_field's model opaque, the others with overlay_opacity). With several models in
		the scene, each surface takes its model's color.
		"""
		primary = self.resolve_field(s.surface_field, True)
		if not s.show_surface or primary is None:
			return []
		models = self.models()
		fields = [(primary, 1.0)]
		if s.surface_mode == 'overlay':
			fields += [(self.model_field(m), float(s.overlay_opacity)) for m in models if m != model_of(primary)]
		out = []
		for field, opacity in fields:
			if len(models) > 1:
				front = self.model_color(model_of(field), s)
				back = tuple(0.55 * x for x in front)
			else:
				front, back = tuple(s.color_front), tuple(s.color_back)
			out.append((field, front, back, opacity))
		return out

	def _draw_surface(self, field, s: ViewSettings, cam: dict, clip_plane, eps, color_front, color_back, opacity):
		prog, vao = self.program('fullscreen.vert', 'surface.frag', field)
		self._bind_field(prog, field, s)
		for k, val in cam.items():
			_set(prog, k, val)
		_set(prog, 'u_box_min', self.box_min)
		_set(prog, 'u_box_max', self.box_max)
		_set(prog, 'u_max_steps', int(s.max_steps))
		_set(prog, 'u_epsilon', float(eps))
		_set(prog, 'u_step_scale', float(s.step_scale))
		_set(prog, 'u_clip', bool(s.clip_surface and s.show_cutplane))
		_set(prog, 'u_clip_plane', clip_plane)
		_set(prog, 'u_color_front', color_front)
		_set(prog, 'u_color_back', color_back)
		_set(prog, 'u_opacity', float(opacity))
		vao.render(moderngl.TRIANGLES, vertices=3)

	def render(self, camera: OrbitCamera, settings: Optional[ViewSettings] = None, size=(800, 600)) -> np.ndarray:
		"""Offscreen render -> (height, width, 4) uint8 RGBA, first row = top of the image."""
		w, h = int(size[0]), int(size[1])
		fbo = self.ctx.framebuffer(self.ctx.renderbuffer((w, h), 4), self.ctx.depth_renderbuffer((w, h)))
		try:
			fbo.use()
			self.draw(camera, settings, (0, 0, w, h))
			data = fbo.read(components=4, alignment=1)
		finally:
			fbo.release()
		return np.frombuffer(data, np.uint8).reshape(h, w, 4)[::-1].copy()

	# ---- probes (GPU evaluation at given points)

	def _query_texture(self, queries: np.ndarray) -> moderngl.Texture:
		q = np.asarray(queries, dtype=np.float32).reshape(-1, 3)
		return array_texture(self.ctx, np.concatenate([q, np.zeros((len(q), 1), np.float32)], 1), self.texture_width)

	def probe(self, queries: np.ndarray, field: str = 'auto', settings: Optional[ViewSettings] = None) -> np.ndarray:
		"""(n, 3) points -> (n, 4) float32: (φ - isovalue, ∇φ) computed by the shader."""
		s = settings or ViewSettings()
		f = self.resolve_field(field, False)
		qtex = self._query_texture(queries)
		n = len(np.asarray(queries).reshape(-1, 3))
		out = self.ctx.texture(qtex.size, 4, dtype='f4')
		fbo = self.ctx.framebuffer([out])
		try:
			prog, vao = self.program('fullscreen.vert', 'probe.frag', f)
			unit = self._bind_field(prog, f, s)
			qtex.use(unit)
			_set(prog, 'u_queries', unit)
			_set(prog, 'u_count', n)
			fbo.use()
			self.ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
			vao.render(moderngl.TRIANGLES, vertices=3)
			data = np.frombuffer(fbo.read(components=4, dtype='f4', alignment=1), np.float32)
		finally:
			for o in (fbo, out, qtex):
				o.release()
		return data.reshape(-1, 4)[:n].copy()

	def probe_knn(self, queries: np.ndarray, k: int, field: str = 'auto') -> np.ndarray:
		"""(n, 3) points -> (n, k) int32 neighbor indices from the shader's BVH traversal, sorted by distance."""
		f = self.resolve_field(field, False)
		if f['type'] != 'pat':
			raise ValueError('probe_knn needs a "pat" field')
		qtex = self._query_texture(queries)
		n = len(np.asarray(queries).reshape(-1, 3))
		out = self.ctx.texture(qtex.size, 4, dtype='i4')
		fbo = self.ctx.framebuffer([out])
		result = np.empty((n, 4 * ((k + 3) // 4)), np.int32)
		try:
			defines = self._field_defines(f)
			defines['BVH_K_MAX'] = max(k, 4)
			prog, vao = self.program('fullscreen.vert', 'probe_knn.frag', None, defines)
			unit = 0
			for key in ('points', 'bvh_bounds', 'bvh_nodes', 'bvh_prims'):
				self.textures[f'{f["name"]}/{key}'].use(unit)
				_set(prog, f'pat_{key}', unit)
				unit += 1
			qtex.use(unit)
			_set(prog, 'u_queries', unit)
			_set(prog, 'u_count', n)
			_set(prog, 'u_k', k)
			fbo.use()
			self.ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
			for slot in range(result.shape[1] // 4):
				_set(prog, 'u_slot', slot)
				vao.render(moderngl.TRIANGLES, vertices=3)
				data = np.frombuffer(fbo.read(components=4, dtype='i4', alignment=1), np.int32).reshape(-1, 4)
				result[:, 4 * slot : 4 * slot + 4] = data[:n]
		finally:
			for o in (fbo, out, qtex):
				o.release()
		return result[:, :k]

	def release(self) -> None:
		for t in self.textures.values():
			t.release()
		for v in self._vao.values():
			v.release()
		for p in self._programs.values():
			p.release()
		self.textures, self._vao, self._programs = {}, {}, {}
