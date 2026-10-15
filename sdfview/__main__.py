"""
Command line:

	python -m sdfview pat INPUT.ply -o SCENE.sdfx [--device cuda] [--k 32] [--bake 128] [--view]
	python -m sdfview view SCENE.sdfx [--up z] [--size 1280 800] [--window pyglet|glfw]
	python -m sdfview render SCENE.sdfx -o IMAGE.png [--size 800 600] [--surface pat|grid|NAME] [--points] ...
	python -m sdfview info SCENE.sdfx
"""

from __future__ import annotations

import argparse
import math
import time


def _settings(a):
	from .settings import ViewSettings

	return ViewSettings(
		show_surface=not a.no_surface,
		show_cutplane=not a.no_cut,
		show_points=a.points,
		surface_field=a.surface,
		cut_field=a.cut_field,
		cut_axis='xyz'.index(a.axis),
		cut_offset=a.offset,
		clip_surface=a.clip,
		isovalue=a.isovalue,
		k=a.k,
	)


def _view_args(p):
	p.add_argument('--up', default='y', choices=['x', 'y', 'z'], help='world up axis for the orbit camera')
	p.add_argument('--surface', default='auto', help="surface field: name, 'pat', 'grid' or 'auto' (grid if any)")
	p.add_argument('--cut-field', default='auto', help="cut plane field: name, 'pat', 'grid' or 'auto' (exact field)")
	p.add_argument('--no-surface', action='store_true')
	p.add_argument('--no-cut', action='store_true')
	p.add_argument('--points', action='store_true', help='draw the point cloud')
	p.add_argument('--axis', default='z', choices=['x', 'y', 'z'], help='cut plane normal axis')
	p.add_argument('--offset', type=float, default=0.0, help='cut plane position in [-1, 1] across the box')
	p.add_argument('--clip', action='store_true', help='hide the surface in front of the cut plane')
	p.add_argument('--isovalue', type=float, default=0.0)
	p.add_argument('--k', type=int, default=None, help='PAT neighbors (default: the scene value)')


def cmd_pat(a):
	import torch

	import points_as_tori as pat

	from .sources.pat import scene_from_pat

	pat.set_device(a.device, torch.float64 if a.float64 else torch.float32)
	points, normals = pat.read_point_cloud(a.input)
	t = time.perf_counter()
	sdf = pat.PointsAsTori(points, normals, model_path=a.model)
	t_fit = time.perf_counter() - t
	t = time.perf_counter()
	scene = scene_from_pat(sdf, k=a.k, bake_resolution=a.bake, meta={'input': a.input})
	t_scene = time.perf_counter() - t
	scene.save(a.output)
	print(f'{len(points)} points: tori {t_fit:.2f} s, BVH + export{" + bake" if a.bake else ""} {t_scene:.2f} s')
	print(f'wrote {a.output}: {scene}')
	if a.view:
		from . import show

		show(scene, up=a.up)


def cmd_superfit(a):
	import torch

	from superfit_sdf import SuperFitSDF

	from .sources.superfit import scene_from_superfit

	t = time.perf_counter()
	sdf = SuperFitSDF(
		a.input, program=a.program, mesh=a.mesh, device=a.device, dtype=torch.float64 if a.float64 else torch.float32
	)
	t_load = time.perf_counter() - t
	t = time.perf_counter()
	scene = scene_from_superfit(sdf, resolution=a.bake, meta={'input': a.input})
	print(f'{sdf}: load {t_load:.2f} s, bake {a.bake}³ {time.perf_counter() - t:.2f} s')
	scene.save(a.output)
	print(f'wrote {a.output}: {scene}')
	if a.view:
		from . import show

		show(scene, up=a.up)


def cmd_view(a):
	from . import show
	from .scene import Scene

	show(Scene.load(a.scene), _settings(a), up=a.up, size=a.size, window=a.window)


def cmd_render(a):
	from .backends.moderngl.headless import render_image, write_png
	from .camera import OrbitCamera
	from .scene import Scene

	scene = Scene.load(a.scene)
	cam = OrbitCamera.fit(scene.bounds_min, scene.bounds_max, up=a.up)
	cam.yaw, cam.pitch = math.radians(a.yaw), math.radians(a.pitch)
	cam.zoom(a.zoom)
	t = time.perf_counter()
	img = render_image(scene, cam, _settings(a), size=a.size)
	write_png(a.output, img)
	print(f'wrote {a.output} ({a.size[0]}x{a.size[1]}, {time.perf_counter() - t:.2f} s)')


def cmd_info(a):
	from .scene import Scene

	scene = Scene.load(a.scene)
	print(scene)
	for name, arr in scene.arrays.items():
		print(f'  {name:28s} {str(arr.dtype):8s} {tuple(arr.shape)}')
	for f in scene.fields:
		print(f'  field {f["name"]} ({f["type"]}): {f.get("params", {})}')


def main(argv=None):
	parser = argparse.ArgumentParser(prog='python -m sdfview', description='SDF scenes: build, view, render')
	sub = parser.add_subparsers(dest='cmd', required=True)

	p = sub.add_parser('pat', help='Points as Tori: point cloud (.ply) -> scene (.sdfx)')
	p.add_argument('input')
	p.add_argument('-o', '--output', required=True)
	p.add_argument('--device', default='cpu', help="torch device, e.g. 'cuda'")
	p.add_argument('--float64', action='store_true', help='float64 geometry (default float32)')
	p.add_argument('--model', default=None, help='network checkpoint (.pt); default: the bundled model')
	p.add_argument('--k', type=int, default=32, help='neighbors of the blend')
	p.add_argument('--bake', type=int, default=None, help='also store a grid bake with this many samples per side')
	p.add_argument('--view', action='store_true', help='open the viewer afterwards')
	p.add_argument('--up', default='y', choices=['x', 'y', 'z'])
	p.set_defaults(func=cmd_pat)

	p = sub.add_parser('superfit', help='SuperFit primitive assembly (.pkl) -> scene (.sdfx), baked on a grid')
	p.add_argument('input', help='primitive_assembly.pkl (loaded with an allow-list unpickler)')
	p.add_argument('-o', '--output', required=True)
	p.add_argument('--program', default='best', help="'best' (as SuperFit's get_best_expr), 'latest' or a key")
	p.add_argument(
		'--mesh', default=None, help="input mesh: evaluate in its frame (default: SuperFit's normalized frame)"
	)
	p.add_argument('--bake', type=int, default=192, help='grid samples per side')
	p.add_argument('--device', default='cpu', help="torch device, e.g. 'cuda'")
	p.add_argument('--float64', action='store_true')
	p.add_argument('--view', action='store_true', help='open the viewer afterwards')
	p.add_argument('--up', default='y', choices=['x', 'y', 'z'])
	p.set_defaults(func=cmd_superfit)

	p = sub.add_parser('view', help='open a window')
	p.add_argument('scene')
	p.add_argument('--size', type=int, nargs=2, default=(1280, 800))
	p.add_argument('--window', default='pyglet', help='moderngl-window backend: pyglet, glfw, ...')
	_view_args(p)
	p.set_defaults(func=cmd_view)

	p = sub.add_parser('render', help='render to a PNG without a window (EGL on Linux)')
	p.add_argument('scene')
	p.add_argument('-o', '--output', required=True)
	p.add_argument('--size', type=int, nargs=2, default=(800, 600))
	p.add_argument('--yaw', type=float, default=35.0, help='degrees about the up axis')
	p.add_argument('--pitch', type=float, default=25.0, help='degrees')
	p.add_argument('--zoom', type=float, default=0.0, help='zoom steps (each 10%% closer)')
	_view_args(p)
	p.set_defaults(func=cmd_render)

	p = sub.add_parser('info', help='print the manifest summary')
	p.add_argument('scene')
	p.set_defaults(func=cmd_info)

	a = parser.parse_args(argv)
	a.func(a)


if __name__ == '__main__':
	main()
