"""
Window app (moderngl-window). Renders only when something changed, into an offscreen buffer that is then scaled to
the window; while the mouse drags, at a reduced resolution (the exact PAT field costs a BVH k-NN search per
sphere-tracing step). Controls: HELP below.
"""

from __future__ import annotations

from typing import Optional

import moderngl
import moderngl_window as mglw

from ...camera import OrbitCamera
from ...scene import Scene
from ...settings import ViewSettings
from .renderer import Renderer

HELP = """\
  mouse   left drag: orbit · right/middle drag: pan · wheel: zoom
  keys    S surface · C cut plane · P points · X/Y/Z cut axis · ↑/↓ move cut plane (shift: faster)
          K clip surface at the plane · B cut plane opacity · G / F cycle the surface / cut-plane field
          V overlay all models (G picks the opaque one) · T overlay transparency
          [ / ] parts of an assembly (whole, all parts, part 0, 1, ...) · U color by part
          I / O isovalue ∓ · N / M neighbors k ∓ · 1-4 resolution while dragging (1 = full) · R reset view
          H help · Esc quit"""


class ViewerWindow(mglw.WindowConfig):
	gl_version = (3, 3)
	title = 'sdfview'
	window_size = (1280, 800)
	aspect_ratio = None
	resizable = True
	vsync = True

	# set by run()
	scene: Optional[Scene] = None
	settings: Optional[ViewSettings] = None
	up: str = 'y'

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.renderer = Renderer(self.ctx, self.scene)
		self.view = self.settings or ViewSettings()
		self.camera = OrbitCamera.fit(self.renderer.box_min, self.renderer.box_max, up=self.up)
		blit_prog, blit_vao = self.renderer.program('fullscreen.vert', 'blit.frag')
		self._blit = (blit_prog, blit_vao)
		self._offscreen = None
		self.drag_scale = 2  # resolution divisor while dragging
		self._dragging = False
		self._dirty = True
		names = [f['name'] for f in self.scene.fields]
		self._field_names = ['auto'] + names
		print(HELP)
		self._update_title()

	# ---- rendering

	def _target(self, w: int, h: int):
		if self._offscreen is None or self._offscreen[0].size != (w, h):
			if self._offscreen is not None:
				for o in self._offscreen:
					o.release()
			color = self.ctx.texture((w, h), 4)
			color.filter = (moderngl.LINEAR, moderngl.LINEAR)
			depth = self.ctx.depth_renderbuffer((w, h))
			self._offscreen = (color, depth, self.ctx.framebuffer([color], depth))
			self._dirty = True
		return self._offscreen

	def on_render(self, time: float, frame_time: float) -> None:
		bw, bh = self.wnd.buffer_size
		div = self.drag_scale if self._dragging else 1
		w, h = max(1, bw // div), max(1, bh // div)
		color, _, fbo = self._target(w, h)
		if self._dirty:
			fbo.use()
			self.renderer.draw(self.camera, self.view, (0, 0, w, h))
			self._dirty = False
		self.wnd.use()
		self.ctx.disable(moderngl.DEPTH_TEST)
		prog, vao = self._blit
		color.use(0)
		prog['u_image'].value = 0
		prog['u_viewport'].value = (0.0, 0.0, float(bw), float(bh))
		vao.render(moderngl.TRIANGLES, vertices=3)

	def on_resize(self, width: int, height: int) -> None:
		self._dirty = True

	# ---- input

	def on_mouse_press_event(self, x, y, button):
		self._dragging = True

	def on_mouse_release_event(self, x, y, button):
		self._dragging = False
		self._dirty = True

	def on_mouse_drag_event(self, x, y, dx, dy):
		states = self.wnd.mouse_states
		if states.left:
			self.camera.orbit(dx, dy)
		else:
			self.camera.pan(dx, dy, self.wnd.size[1])
		self._dirty = True

	def on_mouse_scroll_event(self, x_offset, y_offset):
		self.camera.zoom(y_offset)
		self._dirty = True

	def on_key_event(self, key, action, modifiers):
		keys = self.wnd.keys
		if action != keys.ACTION_PRESS:
			return
		v = self.view
		step = 0.1 if modifiers.shift else 0.02
		fields = self._field_names
		if key == keys.S:
			v.show_surface = not v.show_surface
		elif key == keys.C:
			v.show_cutplane = not v.show_cutplane
		elif key == keys.P:
			v.show_points = not v.show_points
		elif key in (keys.X, keys.Y, keys.Z):
			v.cut_axis, v.cut_normal = {keys.X: 0, keys.Y: 1, keys.Z: 2}[key], None
		elif key == keys.UP:
			v.cut_offset = min(1.0, v.cut_offset + step)
		elif key == keys.DOWN:
			v.cut_offset = max(-1.0, v.cut_offset - step)
		elif key == keys.K:
			v.clip_surface = not v.clip_surface
		elif key == keys.B:
			v.cut_opacity = 0.5 if v.cut_opacity == 1.0 else 1.0
		elif key == keys.G:
			v.surface_field = fields[(fields.index(v.surface_field) + 1) % len(fields)]
		elif key == keys.F:
			v.cut_field = fields[(fields.index(v.cut_field) + 1) % len(fields)]
		elif key == keys.V:
			v.surface_mode = 'single' if v.surface_mode == 'overlay' else 'overlay'
		elif key == keys.T:
			v.overlay_opacity = 0.45 if v.overlay_opacity >= 1.0 else 1.0
		elif key in (keys.LEFT_BRACKET, keys.RIGHT_BRACKET):
			# cycle -1 (the assembly), -2 (all parts), 0 .. n-1 (one part) over the scene's largest part count
			n = max([int(f.get('glsl', {}).get('n_parts', 0)) for f in self.renderer.scene.fields] + [0])
			order = [-1, -2] + list(range(n))
			i = order.index(v.part) if v.part in order else 0
			v.part = order[(i + (1 if key == keys.RIGHT_BRACKET else -1)) % len(order)]
		elif key == keys.U:
			v.color_parts = not v.color_parts
		elif key in (keys.I, keys.O):
			v.isovalue += (-1 if key == keys.I else 1) * 0.005 * self.renderer.diagonal * (5 if modifiers.shift else 1)
		elif key in (keys.N, keys.M):
			k = v.k or self._scene_k()
			v.k = max(1, k + (-1 if key == keys.N else 1) * (8 if modifiers.shift else 1))
		elif key in (keys.NUMBER_1, keys.NUMBER_2, keys.NUMBER_3, keys.NUMBER_4):
			self.drag_scale = {keys.NUMBER_1: 1, keys.NUMBER_2: 2, keys.NUMBER_3: 3, keys.NUMBER_4: 4}[key]
		elif key == keys.R:
			self.camera = OrbitCamera.fit(self.renderer.box_min, self.renderer.box_max, up=self.up)
		elif key == keys.H:
			print(HELP)
		else:
			return
		self._dirty = True
		self._update_title()

	def _scene_k(self) -> int:
		f = next((f for f in self.scene.fields if f['type'] == 'pat'), None)
		return int(f['params'].get('k', 32)) if f else 32

	def _update_title(self):
		v = self.view
		surf = self.renderer.resolve_field(v.surface_field, True)
		cut = self.renderer.resolve_field(v.cut_field, False)
		overlay = ''
		if v.surface_mode == 'overlay' and len(self.renderer.models()) > 1:
			overlay = f' + overlay{" (translucent)" if v.overlay_opacity < 1 else ""}'
		parts = [
			f'surface: {surf["name"] if (surf and v.show_surface) else "off"}{overlay}',
			f'cut: {cut["name"] if (cut and v.show_cutplane) else "off"} axis {"xyz"[v.cut_axis]} {v.cut_offset:+.2f}',
			f'iso {v.isovalue:+.4f}',
			f'k {v.k or self._scene_k()}',
		]
		assembly = next((f for f in self.scene.fields if f['type'] == 'glsl'), None)
		if assembly is not None:
			types = assembly['glsl'].get('part_types', [])
			shown = {-1: 'whole', -2: 'all parts'}.get(v.part)
			if shown is None:
				shown = f'part {v.part}' + (f' {types[v.part]}' if v.part < len(types) else '')
			parts.append(f'assembly: {shown}{" (colored)" if v.color_parts else ""}')
		self.wnd.title = 'sdfview — ' + ' · '.join(parts)


def run(scene: Scene, settings: Optional[ViewSettings] = None, up: str = 'y', size=(1280, 800), window: str = 'pyglet'):
	"""Open a window on `scene` (blocking). window: moderngl-window backend ('pyglet', 'glfw', ...)."""
	cls = type(
		'SdfViewWindow', (ViewerWindow,), {'scene': scene, 'settings': settings, 'up': up, 'window_size': tuple(size)}
	)
	mglw.run_window_config(cls, args=['--window', window])
