"""
HOST — sdfview without a window (EGL on Linux): render a scene to PNG images from several directions, and evaluate
the field on the GPU at given points (probe pass: the same shader code as the surfaces).

	INPUT   a scene (.sdfx)
	OUTPUT  <out>/<scene>_<yaw>.png per direction; printed: φ at a few points from the shader
	python examples/hosts/sdfview_headless.py examples/out/pat/tooth.sdfx --yaws 0 90 180
	python examples/hosts/sdfview_headless.py examples/out/compare/knot.sdfx --overlay --opacity 0.45
	python examples/hosts/sdfview_headless.py examples/out/pat/tooth.sdfx --no-cut --points
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np  # noqa: E402

from sdfview import OrbitCamera, Scene, ViewSettings  # noqa: E402
from sdfview.backends.moderngl import Renderer, create_context, write_png  # noqa: E402


def main():
	ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
	ap.add_argument('scene')
	ap.add_argument('--yaws', type=float, nargs='+', default=[35.0, 125.0, 215.0])
	ap.add_argument('--size', type=int, nargs=2, default=(800, 600))
	ap.add_argument('--overlay', action='store_true')
	ap.add_argument('--opacity', type=float, default=1.0)
	ap.add_argument('--no-cut', action='store_true', help='without the cut plane')
	ap.add_argument('--points', action='store_true', help='draw the point cloud')
	ap.add_argument(
		'-o', '--out', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'out', 'renders'))
	)
	a = ap.parse_args()

	scene = Scene.load(a.scene)
	ctx = create_context()  # EGL first on Linux: no display needed
	r = Renderer(ctx, scene)  # uploads the scene's arrays as textures once
	settings = ViewSettings(
		surface_mode='overlay' if a.overlay else 'single',
		overlay_opacity=a.opacity,
		show_cutplane=not a.no_cut,
		show_points=a.points,
	)
	os.makedirs(a.out, exist_ok=True)
	stem = os.path.splitext(os.path.basename(a.scene))[0]
	for yaw in a.yaws:
		cam = OrbitCamera.fit(r.box_min, r.box_max)
		cam.orbit(np.radians(yaw) / 0.01, 0.0)  # orbit() takes pixels at 0.01 rad per pixel
		rgba = r.render(cam, settings, size=a.size)  # (h, w, 4) uint8, first row = top
		path = os.path.join(a.out, f'{stem}_{int(yaw)}.png')
		write_png(path, rgba)
		print(f'wrote {path}')
	q = np.array([[0.0, 0.0, 0.0], r.box_min, r.box_max], np.float32)
	print('φ, ∇φ from the shader (probe):\n', r.probe(q, 'auto', settings))
	r.release()


if __name__ == '__main__':
	main()
