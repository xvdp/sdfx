"""
Headless rendering (no window): an offscreen OpenGL 3.3 context. On Linux the EGL backend runs without a display
(GPU driver, or Mesa llvmpipe on CPU); elsewhere moderngl's default standalone context is used.
"""

from __future__ import annotations

import struct
import sys
import zlib
from typing import Optional

import moderngl
import numpy as np

from ...camera import OrbitCamera
from ...scene import Scene
from ...settings import ViewSettings
from .renderer import Renderer


def create_context(backend: Optional[str] = None) -> moderngl.Context:
	"""Standalone OpenGL >= 3.3 context. backend: None (EGL first on Linux, then the default) / 'egl' / 'default'."""
	tried = []
	order = [backend] if backend else (['egl', None] if sys.platform.startswith('linux') else [None])
	for b in order:
		b = None if b == 'default' else b
		try:
			kw = {'backend': b} if b else {}
			return moderngl.create_standalone_context(require=330, **kw)
		except Exception as e:  # noqa: BLE001 - report every backend's failure
			tried.append(f'{b or "default"}: {e}')
	raise RuntimeError('no OpenGL 3.3 context available:\n  ' + '\n  '.join(tried))


def write_png(path: str, rgba: np.ndarray) -> None:
	"""Minimal PNG writer (8-bit RGBA, no dependencies)."""
	h, w = rgba.shape[:2]
	raw = b''.join(b'\x00' + np.ascontiguousarray(rgba[y], np.uint8).tobytes() for y in range(h))

	def chunk(tag, data):
		return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF)

	with open(path, 'wb') as f:
		f.write(b'\x89PNG\r\n\x1a\n')
		f.write(chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)))
		f.write(chunk(b'IDAT', zlib.compress(raw, 6)))
		f.write(chunk(b'IEND', b''))


def render_image(
	scene: Scene,
	camera: Optional[OrbitCamera] = None,
	settings: Optional[ViewSettings] = None,
	size=(800, 600),
	up: str = 'y',
	ctx: Optional[moderngl.Context] = None,
) -> np.ndarray:
	"""Render a scene offscreen -> (height, width, 4) uint8 RGBA."""
	own = ctx is None
	ctx = ctx or create_context()
	renderer = Renderer(ctx, scene)
	try:
		camera = camera or OrbitCamera.fit(renderer.box_min, renderer.box_max, up=up)
		return renderer.render(camera, settings, size)
	finally:
		renderer.release()
		if own:
			ctx.release()
