"""
Command line:

	python -m sdfview pat INPUT.ply -o SCENE.sdfx [--device cuda] [--k 32] [--bake 128] [--view]
	python -m sdfview view SCENE.sdfx [--up z] [--size 1280 800] [--window pyglet|glfw]
	python -m sdfview render SCENE.sdfx -o IMAGE.png [--size 800 600] [--surface pat|grid|NAME] [--points] ...
	python -m sdfview info SCENE.sdfx
	python -m sdfview merge A.sdfx B.sdfx -o AB.sdfx --names pat sup   # one scene, models toggled / overlaid
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
		surface_mode='overlay' if a.overlay else 'single',
		overlay_opacity=a.opacity,
		part=a.part,
		color_parts=a.color_parts,
	)


def _view_args(p):
	p.add_argument('--up', default='y', choices=['x', 'y', 'z'], help='world up axis for the orbit camera')
	p.add_argument(
		'--surface', default='auto', help="surface field: name or type, or 'auto' (the first model's cheapest to trace)"
	)
	p.add_argument('--cut-field', default='auto', help="cut plane field: name or type, or 'auto' (exact field)")
	p.add_argument('--overlay', action='store_true', help='draw every model of a merged scene (key V)')
	p.add_argument('--opacity', type=float, default=1.0, help='opacity of the other models in overlay (key T)')
	p.add_argument('--no-surface', action='store_true')
	p.add_argument('--no-cut', action='store_true')
	p.add_argument('--points', action='store_true', help='draw the point cloud')
	p.add_argument('--axis', default='z', choices=['x', 'y', 'z'], help='cut plane normal axis')
	p.add_argument('--offset', type=float, default=0.0, help='cut plane position in [-1, 1] across the box')
	p.add_argument('--clip', action='store_true', help='hide the surface in front of the cut plane')
	p.add_argument('--isovalue', type=float, default=0.0)
	p.add_argument('--k', type=int, default=None, help='PAT neighbors (default: the scene value)')
	p.add_argument('--part', type=int, default=-1, help='assembly: -1 whole, -2 all parts, i: part i alone (keys [ ])')
	p.add_argument('--color-parts', action='store_true', help='assembly: color by part (key U)')


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
	scene = scene_from_superfit(sdf, resolution=a.bake or None, meta={'input': a.input}, glsl=not a.no_glsl)
	baked = f', bake {a.bake}³' if a.bake else ''
	print(f'{sdf}: load {t_load:.2f} s, export{baked} {time.perf_counter() - t:.2f} s')
	if a.output:
		scene.save(a.output)
		print(f'wrote {a.output}: {scene}')
	if a.shadertoy:
		from superfit_sdf.glsl import assembly_shadertoy

		with open(a.shadertoy, 'w') as f:
			f.write(assembly_shadertoy(sdf))
		print(f'wrote {a.shadertoy}: Shadertoy-style fragment code (mainImage; #define PART / COLOR_PARTS)')
	if a.view:
		from . import show

		show(scene, _settings(a), up=a.up)


def cmd_view(a):
	from . import show
	from .scene import Scene

	show(Scene.load(a.scene), _settings(a), up=a.up, size=a.size, window=a.window)


def cmd_render(a):
	from .camera import OrbitCamera
	from .png import write_png
	from .scene import Scene
	from .view import scene_box

	scene = Scene.load(a.scene)
	cam = OrbitCamera.fit(*scene_box(scene), up=a.up)
	cam.yaw, cam.pitch = math.radians(a.yaw), math.radians(a.pitch)
	cam.zoom(a.zoom)
	t = time.perf_counter()
	if a.backend == 'slangpy':  # Slang compute kernels (surfaces only: no cut plane, no points)
		from .backends.slangpy import Renderer

		r = Renderer(scene)
		img = r.render(cam, _settings(a), size=a.size)
		r.release()
	else:
		from .backends.moderngl.headless import render_image

		img = render_image(scene, cam, _settings(a), size=a.size)
	write_png(a.output, img)
	print(f'wrote {a.output} ({a.size[0]}x{a.size[1]}, {a.backend}, {time.perf_counter() - t:.2f} s)')


def cmd_merge(a):
	from .scene import Scene, merge_scenes

	names = a.names or [os.path.splitext(os.path.basename(p))[0] for p in a.scenes]
	if len(names) != len(a.scenes) or len(set(names)) != len(names):
		raise SystemExit('--names: one distinct name per scene')
	scene = merge_scenes({n: Scene.load(p) for n, p in zip(names, a.scenes)})
	scene.save(a.output)
	print(f'wrote {a.output}: {scene}')


def cmd_serve(a):
	from .server import serve

	serve(a.scenes, port=a.port)


def cmd_info(a):
	from .scene import Scene

	scene = Scene.load(a.scene)
	print(scene)
	for name, arr in scene.arrays.items():
		print(f'  {name:28s} {str(arr.dtype):8s} {tuple(arr.shape)}')
	for f in scene.fields:
		model = f' model {f["model"]}' if 'model' in f else ''
		print(f'  field {f["name"]} ({f["type"]}{model}): {f.get("params", {})}')
	meta = dict(scene.meta)
	models = meta.pop('models', {})
	if meta:
		print(f'  meta {json.dumps(meta, default=str)}')
	for model, m in models.items():
		print(f'  meta of {model}: {json.dumps(m, default=str)}')


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

	p = sub.add_parser(
		'superfit', help='SuperFit primitive assembly (.pkl) -> scene (.sdfx): generated GLSL, optional grid bake'
	)
	p.add_argument('input', help='primitive_assembly.pkl (loaded with an allow-list unpickler)')
	p.add_argument('-o', '--output', default=None, help='scene file (.sdfx)')
	p.add_argument('--shadertoy', default=None, help='also write Shadertoy-style GLSL (mainImage) to this file')
	p.add_argument('--no-glsl', action='store_true', help='no generated GLSL field (then --bake is needed)')
	p.add_argument('--program', default='best', help="'best' (as SuperFit's get_best_expr), 'latest' or a key")
	p.add_argument(
		'--mesh', default=None, help="input mesh: evaluate in its frame (default: SuperFit's normalized frame)"
	)
	p.add_argument('--bake', type=int, default=0, help='also bake the torch evaluation on a grid of N per side')
	p.add_argument('--device', default='cpu', help="torch device, e.g. 'cuda'")
	p.add_argument('--float64', action='store_true')
	p.add_argument('--view', action='store_true', help='open the viewer afterwards')
	_view_args(p)
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
	p.add_argument(
		'--backend', choices=['moderngl', 'slangpy'], default='moderngl', help='slangpy: Slang kernels, surfaces only'
	)
	_view_args(p)
	p.set_defaults(func=cmd_render)

	p = sub.add_parser('merge', help='several scenes -> one scene with one model per input (toggle / overlay)')
	p.add_argument('scenes', nargs='+')
	p.add_argument('-o', '--output', required=True)
	p.add_argument('--names', nargs='+', help='model names (default: file names), e.g. pat sup gt')
	p.set_defaults(func=cmd_merge)

	p = sub.add_parser('serve', help='serve the browser hosts (WebGL2, WebGPU, three.js) and scenes on localhost')
	p.add_argument('scenes', nargs='*', help='.sdfx files, served at /scenes/<file name>')
	p.add_argument('--port', type=int, default=8000)
	p.set_defaults(func=cmd_serve)

	p = sub.add_parser('info', help='print the manifest summary')
	p.add_argument('scene')
	p.set_defaults(func=cmd_info)

	a = parser.parse_args(argv)
	a.func(a)


if __name__ == '__main__':
	main()
