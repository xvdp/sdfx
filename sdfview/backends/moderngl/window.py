"""
Window app (moderngl-window). Renders only when something changed, into an offscreen buffer that is then scaled to
the window; while the mouse drags, at a reduced resolution (the exact PAT field costs a BVH k-NN search per
sphere-tracing step). Controls: HELP below.
"""

from __future__ import annotations

from typing import Optional

import moderngl
import moderngl_window as mglw

from ... import inspector
from ...camera import OrbitCamera
from ...scene import Scene
from ...settings import ViewSettings
from .renderer import Renderer

# keys and mouse; the same in the browser hosts (sdfview/web/core/app.js HELP)
HELP_LINES = [
	('mouse', 'left drag: orbit · right / middle drag: pan · wheel: zoom'),
	('S C P', 'surface · cut plane · points'),
	('X Y Z  Up Down', 'cut plane axis · move it (shift: faster)'),
	('K  B', 'clip the surface at the plane · cut plane opacity'),
	('G  F', 'next surface field · next cut plane field'),
	('V  T', 'overlay all models (G picks the opaque one) · overlay transparency'),
	('[ ]  U', 'parts of an assembly (whole, all parts, part 0, 1, ...) · color by part'),
	('I O  N M', 'isovalue -/+ (shift: ×5) · PAT neighbors k -/+ (shift: ×8)'),
	('click  , .', 'tori inspector at the clicked point · previous / next neighbor (by weight)'),
	('J  L  E', 'inspector: network neighborhood · torus · off'),
	('1-4', 'resolution while dragging (1 = full)'),
	('R  H  Esc', 'reset view · this help (on screen and here) · quit'),
]
HELP = '\n'.join(f'  {k:15s} {v}' for k, v in HELP_LINES)

# a translucent rectangle behind the help text: pixel rectangle -> NDC by gl_VertexID (triangle strip of 4)
_PANEL_VERT = """#version 330 core
uniform vec4 u_rect;      // x0, y0, x1, y1 in pixels, origin bottom left
uniform vec2 u_size;      // framebuffer size in pixels
void main() {
    vec2 c = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
    gl_Position = vec4(2.0 * mix(u_rect.xy, u_rect.zw, c) / u_size - 1.0, 0.0, 1.0);
}
"""
_PANEL_FRAG = """#version 330 core
out vec4 frag_color;
void main() { frag_color = vec4(0.08, 0.08, 0.08, 0.82); }
"""


class TextPanel:
	"""Lines of text over the window, in a corner ('top' left or 'bottom' left): moderngl-window's bitmap-font
	TextWriter2D (ASCII and Latin-1) on a translucent panel."""

	def __init__(self, ctx: moderngl.Context, lines=(), corner: str = 'top'):
		from moderngl_window.text.bitmapped import TextWriter2D

		self.ctx = ctx
		self.writer = TextWriter2D()
		self.panel = ctx.program(vertex_shader=_PANEL_VERT, fragment_shader=_PANEL_FRAG)
		self.vao = ctx.vertex_array(self.panel, [])
		self.lines = list(lines)
		self.corner = corner
		self.visible = True

	def draw(self, size, scale: float = 1.0) -> None:
		if not self.visible or not self.lines:
			return
		w, h = size
		px = 16.0 * scale  # character height in pixels
		cw = px * self.writer._meta.char_aspect_wh  # monospace: character width
		margin, step = 10.0 * scale, 1.35 * px
		width = max(len(line) for line in self.lines) * cw + 2 * margin
		height = len(self.lines) * step + 2 * margin
		x0 = 12.0 * scale
		y1 = h - 12.0 * scale if self.corner == 'top' else 12.0 * scale + height
		self.ctx.disable(moderngl.DEPTH_TEST)
		self.ctx.enable(moderngl.BLEND)
		self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
		self.panel['u_rect'].value = (x0, y1 - height, x0 + width, y1)
		self.panel['u_size'].value = (float(w), float(h))
		self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4)
		for i, line in enumerate(self.lines):
			self.writer.text = line
			self.writer.draw((x0 + margin + 0.5 * cw, y1 - margin - (i + 0.5) * step), size=px)
		self.ctx.disable(moderngl.BLEND)


def HelpOverlay(ctx: moderngl.Context) -> TextPanel:
	return TextPanel(ctx, [f'{k:15s} {v}' for k, v in HELP_LINES], 'top')


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
		try:
			self._help = HelpOverlay(self.ctx)
			self._info = TextPanel(self.ctx, corner='bottom')
		except Exception as e:  # noqa: BLE001  (font resources missing: the terminal copy only)
			print(f'(no on-screen text: {e})')
			self._help = self._info = None
		# tori inspector (sdfview/inspector.py): a left click without dragging inspects the clicked point
		self._inspection = None
		self._show_torus = self._show_network = True
		self._press = None  # (x, y, moved) of the pressed button
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
		if self._help is not None:
			scale = bw / max(self.wnd.size[0], 1)
			self._help.draw((bw, bh), scale)
			self._info.draw((bw, bh), scale)

	def on_resize(self, width: int, height: int) -> None:
		self._dirty = True

	# ---- input

	def on_mouse_press_event(self, x, y, button):
		self._dragging = True
		self._press = [x, y, False]

	def on_mouse_release_event(self, x, y, button):
		self._dragging = False
		self._dirty = True
		if button == 1 and self._press is not None and not self._press[2]:  # a click: no drag in between
			self.inspect_at(x, y)
		self._press = None

	def on_mouse_drag_event(self, x, y, dx, dy):
		if self._press is not None and abs(x - self._press[0]) + abs(y - self._press[1]) > 3:
			self._press[2] = True
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
			if self._inspection is not None:  # the blend at the same point with the new k
				rank = self._inspection.selected
				self._inspection = inspector.inspect(self.scene, self._inspection.x, k=v.k, lambda_scale=v.lambda_scale)
				inspector.select(self.scene, self._inspection, rank)
				self._show_inspection()
		elif key in (keys.NUMBER_1, keys.NUMBER_2, keys.NUMBER_3, keys.NUMBER_4):
			self.drag_scale = {keys.NUMBER_1: 1, keys.NUMBER_2: 2, keys.NUMBER_3: 3, keys.NUMBER_4: 4}[key]
		elif key in (keys.COMMA, keys.PERIOD) and self._inspection is not None:
			step_rank = -1 if key == keys.COMMA else 1
			inspector.select(self.scene, self._inspection, self._inspection.selected + step_rank)
			self._show_inspection()
		elif key == keys.J and self._inspection is not None:
			self._show_network = not self._show_network
			self._show_inspection()
		elif key == keys.L and self._inspection is not None:
			self._show_torus = not self._show_torus
			self._show_inspection()
		elif key == keys.E:
			self._inspection = None
			self._show_inspection()
		elif key == keys.R:
			self.camera = OrbitCamera.fit(self.renderer.box_min, self.renderer.box_max, up=self.up)
		elif key == keys.H:
			print(HELP)
			if self._help is not None:
				self._help.visible = not self._help.visible
		else:
			return
		self._dirty = True
		self._update_title()

	# ---- tori inspector

	def inspect_at(self, x: float, y: float) -> None:
		"""Inspect the point seen at window position (x, y) (top left origin, window units): pick it in the depth
		buffer of a frame drawn at the framebuffer's size, then sdfview.inspector.inspect."""
		bw, bh = self.wnd.buffer_size
		ww, wh = self.wnd.size
		px, py = x * bw / max(ww, 1), bh - 1 - y * bh / max(wh, 1)
		p = self.renderer.pick(self.camera, self.view, (bw, bh), (px, py))
		if p is None:
			return
		ins = inspector.inspect(self.scene, p, k=self.view.k, lambda_scale=self.view.lambda_scale)
		if ins is None:
			print('tori inspector: the scene has no pat field')
			return
		self._inspection = ins
		self._show_inspection()

	def _show_inspection(self) -> None:
		ins = self._inspection
		self.renderer.set_inspection(ins, show_torus=self._show_torus, show_network=self._show_network)
		lines = inspector.describe(ins) if ins is not None else []
		if lines:
			print('\n'.join(lines))
		if self._info is not None:
			self._info.lines = lines
		self._dirty = True

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
		self.wnd.title = 'sdfview — ' + ' · '.join(parts) + ' · H: keys'


def run(scene: Scene, settings: Optional[ViewSettings] = None, up: str = 'y', size=(1280, 800), window: str = 'pyglet'):
	"""Open a window on `scene` (blocking). window: moderngl-window backend ('pyglet', 'glfw', ...)."""
	cls = type(
		'SdfViewWindow', (ViewerWindow,), {'scene': scene, 'settings': settings, 'up': up, 'window_size': tuple(size)}
	)
	mglw.run_window_config(cls, args=['--window', window])
