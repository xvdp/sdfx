"""
HOST — sdfview's Slang modules through slangpy (sdfview/backends/slangpy, sdfview/shaders/slang): the surface and the
probe as compute kernels, one thread per pixel / query, on a Vulkan device (a GPU, else lavapipe on the CPU).
Renders the scene with this host and, when moderngl runs, with the moderngl host, and compares images and probes.

	INPUT   a scene (.sdfx) with a 'pat' or 'grid' field ('glsl' fields have no Slang mirror: bake them, --bake N)
	OUTPUT  <out>/<scene>_slangpy.png, <scene>_moderngl.png; printed: timings, image and probe differences
	python examples/hosts/slangpy_render.py examples/out/pat/tooth.sdfx
	python examples/hosts/slangpy_render.py examples/out/pat/tooth.sdfx --size 320 240 --yaw 120
	python -m sdfview render examples/out/pat/tooth.sdfx -o tooth.png --backend slangpy    # the same from the CLI
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np  # noqa: E402

from sdfview import OrbitCamera, Scene, ViewSettings  # noqa: E402
from sdfview.backends.slangpy import Renderer  # noqa: E402
from sdfview.png import write_png  # noqa: E402
from sdfview.view import scene_box  # noqa: E402


def main():
	ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
	ap.add_argument('scene')
	ap.add_argument('--size', type=int, nargs=2, default=(320, 240))
	ap.add_argument('--yaw', type=float, default=35.0, help='degrees')
	ap.add_argument('--queries', type=int, default=10000, help='probe points in the scene box')
	ap.add_argument(
		'-o', '--out', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'out', 'renders'))
	)
	a = ap.parse_args()

	scene = Scene.load(a.scene)
	lo, hi = scene_box(scene)
	cam = OrbitCamera.fit(lo, hi)
	cam.yaw = np.radians(a.yaw)
	settings = ViewSettings(show_cutplane=False)  # the Slang host draws surfaces only
	stem = os.path.splitext(os.path.basename(a.scene))[0]
	os.makedirs(a.out, exist_ok=True)

	t = time.perf_counter()
	r = Renderer(scene)  # device, modules loaded, arrays uploaded once
	print(f'slangpy: device {r.device.info.adapter_name}, setup {time.perf_counter() - t:.2f} s')
	for i in range(2):  # the first call compiles the specialized kernels
		t = time.perf_counter()
		img = r.render(cam, settings, a.size)
		print(
			f'  render {a.size[0]}x{a.size[1]} ({"with" if i == 0 else "after"} compilation): {time.perf_counter() - t:.3f} s'
		)
	write_png(os.path.join(a.out, f'{stem}_slangpy.png'), img)
	q = np.random.default_rng(0).uniform(lo, hi, size=(a.queries, 3)).astype(np.float32)
	t = time.perf_counter()
	pr = r.probe(q)
	print(f'  probe {a.queries} points: {time.perf_counter() - t:.3f} s')

	try:
		from sdfview.backends.moderngl import Renderer as GLRenderer
		from sdfview.backends.moderngl import create_context
	except ImportError:
		print('moderngl not installed: no comparison')
		return
	g = GLRenderer(create_context(), scene)
	ref = g.render(cam, settings, a.size)
	write_png(os.path.join(a.out, f'{stem}_moderngl.png'), ref)
	d = np.abs(img[..., :3].astype(int) - ref[..., :3]).max(-1)
	print(f'image vs moderngl: max|Δ| {d.max()} / 255, pixels with |Δ| > 8: {(d > 8).mean():.4%}')
	gp = g.probe(q, 'auto', settings)
	print(
		f'probe vs moderngl: max|Δφ| {np.abs(pr[:, 0] - gp[:, 0]).max():.2e}, max|Δ∇φ| {np.abs(pr[:, 1:] - gp[:, 1:]).max():.2e}'
	)
	print(f'wrote {a.out}/{stem}_slangpy.png, {stem}_moderngl.png')


if __name__ == '__main__':
	main()
