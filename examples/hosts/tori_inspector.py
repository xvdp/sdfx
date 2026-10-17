"""
HOST — the tori inspector without a window: pick a point of a PAT scene (a pixel of the rendered image, or a point of
the cloud), inspect the blend there and draw it: the k blended points by weight (pale yellow -> orange), the torus of
one of them (translucent pink), its two principal circles and directions (cyan κ_min, magenta κ_max), the network's
neighborhood of that point (green; with --attribution gradient | rollout, shaded by how much each neighbor drives the
curvatures / feeds the network's output, points_as_tori/explain.py), and least-squares curvatures from 6 and 64 points
for comparison. In the window (python -m sdfview view scene.sdfx): click a point; , . cycle the neighbor whose torus
is shown; J network neighborhood; A its coloring; L torus; E clear.

	INPUT   a scene with a 'pat' field (examples/pat/01_fit.py, python -m sdfview pat)
	OUTPUT  <out>/<scene>_inspect.png (whole view) and <scene>_inspect_zoom.png (camera on the inspected point);
	        printed: the inspection (sdfview/inspector.py describe)
	python examples/hosts/tori_inspector.py examples/out/pat/tooth.sdfx                 # the surface at the image center
	python examples/hosts/tori_inspector.py examples/out/pat/tooth.sdfx --pixel 300 220 --rank 2
	python examples/hosts/tori_inspector.py examples/out/pat/tooth.sdfx --point 1234    # at point 1234 of the cloud
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from sdfview import OrbitCamera, Scene, ViewSettings  # noqa: E402
from sdfview.backends.moderngl import Renderer, create_context, write_png  # noqa: E402
from sdfview.inspector import describe, explain_network, inspect, pat_field  # noqa: E402


def main():
	ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
	ap.add_argument('scene')
	ap.add_argument('--size', type=int, nargs=2, default=(640, 480))
	ap.add_argument('--pixel', type=int, nargs=2, help='pixel to pick (from the top left); default: the center')
	ap.add_argument('--point', type=int, help='inspect at this point of the cloud instead of a picked pixel')
	ap.add_argument('--rank', type=int, default=0, help='neighbor (by weight) whose torus is shown')
	ap.add_argument('--cut', action='store_true', help='with the cut plane')
	ap.add_argument('--attribution', choices=['gradient', 'rollout'], help='shade the network neighborhood')
	ap.add_argument(
		'-o', '--out', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'out', 'renders'))
	)
	a = ap.parse_args()

	scene = Scene.load(a.scene)
	if pat_field(scene) is None:
		raise SystemExit('the scene has no pat field')
	r = Renderer(create_context(), scene)
	settings = ViewSettings(show_cutplane=a.cut)  # the cloud's own sprites off: the marks are the relevant points
	cam = OrbitCamera.fit(r.box_min, r.box_max)
	w, h = a.size
	if a.point is not None:
		x = scene.arrays[pat_field(scene)['arrays']['points']][a.point, :3].astype(float)
	else:
		px, py = a.pixel if a.pixel else (w // 2, h // 2)
		x = r.pick(cam, settings, (w, h), (px, h - 1 - py))  # pick() counts rows from the bottom
		if x is None:
			raise SystemExit(f'pixel ({px}, {py}) shows the background')
	ins = inspect(scene, x, selected=a.rank)
	explain_network(scene, ins)  # the network on point j's neighborhood (needs torch); None otherwise
	print('\n'.join(describe(ins)))
	r.set_inspection(ins, attribution=a.attribution)
	stem = os.path.splitext(os.path.basename(a.scene))[0]
	os.makedirs(a.out, exist_ok=True)
	write_png(os.path.join(a.out, f'{stem}_inspect.png'), r.render(cam, settings, (w, h)))
	zoom = OrbitCamera.fit(ins.x - ins.radius, ins.x + ins.radius)  # same direction, on the inspected point
	zoom.yaw, zoom.pitch = cam.yaw, cam.pitch
	write_png(os.path.join(a.out, f'{stem}_inspect_zoom.png'), r.render(zoom, settings, (w, h)))
	print(f'wrote {a.out}/{stem}_inspect.png, {stem}_inspect_zoom.png')
	r.release()


if __name__ == '__main__':
	main()
