"""
sdfview — viewer for signed distance fields produced with torch.

Layers (each usable on its own):
	scene.py            file format: JSON manifest + binary arrays (.sdfx), read by Python and JavaScript
	accel/              acceleration structures built with torch (LBVH)
	sources/            exporters from SDF producers (points_as_tori, any torch callable via a grid bake)
	shaders/glsl/       one GLSL code base (GL 3.3 core / WebGL2), modules mirrored later in Slang / WGSL
	backends/moderngl/  Renderer (embeddable in any moderngl context), headless rendering, window app
	web/                JavaScript side (scene reader now; WebGL2 / WebGPU / three.js later)

	python -m sdfview --help
"""

from .camera import OrbitCamera  # noqa: F401
from .scene import Scene, grid_field  # noqa: F401
from .settings import ViewSettings  # noqa: F401

__version__ = '0.1.0'


def show(scene, settings=None, **kw):
	"""Open a window on a Scene (blocking). Needs the `view` extra (moderngl-window)."""
	from .backends.moderngl.window import run

	return run(scene, settings, **kw)
