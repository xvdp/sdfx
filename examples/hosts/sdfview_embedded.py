"""
HOST — sdfview's Renderer inside your own moderngl application: the Renderer draws into whatever framebuffer is bound,
in a viewport you choose; your code owns the window, the camera and the other draws. Here: a moderngl-window app with
two viewports showing two fields of one scene side by side (e.g. PAT's exact field and its grid bake, or two models).

	INPUT   a scene (.sdfx) and two of its field names
	python examples/hosts/sdfview_embedded.py examples/out/pat/tooth.sdfx --left pat --right pat_grid
	python examples/hosts/sdfview_embedded.py examples/out/compare/knot.sdfx --left pat.pat_grid --right sup.sup
	(left drag: orbit both)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import moderngl_window as mglw  # noqa: E402

from sdfview import OrbitCamera, Scene, ViewSettings  # noqa: E402
from sdfview.backends.moderngl import Renderer  # noqa: E402


def main():
	ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
	ap.add_argument('scene')
	ap.add_argument('--left', required=True)
	ap.add_argument('--right', required=True)
	a = ap.parse_args()
	scene = Scene.load(a.scene)

	class App(mglw.WindowConfig):
		gl_version = (3, 3)
		title = f'{a.left} | {a.right}'
		window_size = (1280, 560)
		resizable = True

		def __init__(self, **kwargs):
			super().__init__(**kwargs)
			self.renderer = Renderer(self.ctx, scene)  # any current moderngl context
			self.camera = OrbitCamera.fit(self.renderer.box_min, self.renderer.box_max)
			self.views = [ViewSettings(surface_field=a.left, cut_field=a.left, show_cutplane=False),
						  ViewSettings(surface_field=a.right, cut_field=a.right, show_cutplane=False)]  # fmt: skip

		def on_render(self, t, dt):
			w, h = self.wnd.buffer_size
			self.wnd.use()
			for i, view in enumerate(self.views):  # draw() clears and draws inside its viewport only
				self.renderer.draw(self.camera, view, viewport=(i * w // 2, 0, w // 2, h))

		def on_mouse_drag_event(self, x, y, dx, dy):
			self.camera.orbit(dx, dy)

		def on_mouse_scroll_event(self, x_offset, y_offset):
			self.camera.zoom(y_offset)

	mglw.run_window_config(App, args=['--window', 'pyglet'])


if __name__ == '__main__':
	main()
