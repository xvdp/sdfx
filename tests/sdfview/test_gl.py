"""
GLSL (moderngl backend) against torch: the shader's field and gradient (probe pass) against TorchDistanceField in
k-NN mode, the shader's BVH k-NN against brute force, grid sampling against trilinear interpolation, rendering
geometry against an analytic silhouette; plus WebGL2 compilation of the same sources, the window app and the CLI.
"""

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from points_as_tori import config
from sdfview import OrbitCamera, Scene, ViewSettings, grid_field
from sdfview.shaders import glsl_source
from sdfview.sources.pat import scene_from_pat

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def near_surface_queries(n, seed=0, spread=0.3, box=1.3):
	rng = np.random.default_rng(seed)
	u = rng.normal(size=(n, 3))
	u /= np.linalg.norm(u, axis=1, keepdims=True)
	q = u * (1 + spread * rng.uniform(-1, 1, size=(n, 1)))
	return np.concatenate([q, rng.uniform(-box, box, size=(n // 4, 3))]).astype(np.float32)


def test_probe_matches_torch(gl_ctx, sphere):
	from sdfview.backends.moderngl import Renderer

	q = near_surface_queries(3000)
	errors = {}
	for dtype in (torch.float64, torch.float32):
		tdf, _ = sphere(dtype, noise=0.08)
		tdf.set_k_evaluation(32)
		d, g = tdf.evaluate_distance_gradient(config.as_wide(q, 3))
		d, g = d.double().numpy(), g.double().numpy().T
		if dtype == torch.float32:
			r = Renderer(gl_ctx, scene_from_pat(tdf, k=32))
			out = r.probe(q, 'pat').astype(np.float64)
			r.release()
		errors[dtype] = (d, g)
	d64, g64 = errors[torch.float64]
	d32, g32 = errors[torch.float32]
	e_d32, e_g32 = np.abs(out[:, 0] - d32).max(), np.abs(out[:, 1:] - g32).max()
	e_d64, e_g64 = np.abs(out[:, 0] - d64).max(), np.abs(out[:, 1:] - g64).max()
	e_t = np.abs(d32 - d64).max(), np.abs(g32 - g64).max()
	print(f'\nGLSL vs torch float32: max|Δφ| {e_d32:.2e}, max|Δ∇φ| {e_g32:.2e}')
	print(f'GLSL vs torch float64: max|Δφ| {e_d64:.2e}, max|Δ∇φ| {e_g64:.2e}')
	print(f'torch float32 vs float64: max|Δφ| {e_t[0]:.2e}, max|Δ∇φ| {e_t[1]:.2e}  (|φ| up to {np.abs(d64).max():.2f})')
	assert e_d32 < 2e-5 and e_d64 < 2e-5
	assert e_g32 < 5e-3 and e_g64 < 5e-3


def test_probe_knn_matches_brute_force(gl_ctx, sphere):
	from sdfview.backends.moderngl import Renderer

	tdf, points = sphere(torch.float32, n=4000)
	r = Renderer(gl_ctx, scene_from_pat(tdf, k=32))
	q = near_surface_queries(1000, seed=1, spread=0.5, box=3.0)
	p32 = points.astype(np.float32).astype(np.float64)
	same = 0
	for k in (1, 7, 32):
		idx = r.probe_knn(q, k, 'pat')
		assert idx.shape == (len(q), k) and idx.min() >= 0
		bf = np.linalg.norm(q[:, None, :].astype(np.float64) - p32[None], axis=-1)
		ref = np.sort(bf, axis=1)[:, :k]
		got = np.take_along_axis(bf, idx, axis=1)
		np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6)  # same neighbors up to exact ties
		same = np.mean([set(a) == set(b) for a, b in zip(idx, np.argsort(bf, axis=1)[:, :k])])
		print(f'\nk={k}: identical neighbor sets for {100 * same:.1f}% of {len(q)} queries')
	r.release()


def test_shader_reads_the_original_bvh_layout(gl_ctx, sphere):
	"""The C++ SAH BVH (BVH::get_gpu_data) converted to the scene layout gives the same neighbors in the shader."""
	sys.path.insert(0, os.environ.get('PAT_REFERENCE_DIR', os.path.join(REPO, 'build', 'pat_reference')))
	ref = pytest.importorskip('pat_reference')
	from sdfview.accel import from_cpp_layout
	from sdfview.backends.moderngl import Renderer

	tdf, points = sphere(torch.float32, n=3000)
	scene = scene_from_pat(tdf, k=16)
	nd, idx = ref.BVH(np.asfortranarray(points.T)).get_gpu_data()
	gpu = from_cpp_layout(nd, idx).gpu_arrays()
	for key, a in gpu.items():
		scene.arrays[f'pat/{key}'] = a
	scene.field('pat')['params']['bvh_depth'] = int(from_cpp_layout(nd, idx).depth.max())
	r = Renderer(gl_ctx, scene)
	q = near_surface_queries(500, seed=2)
	got = r.probe_knn(q, 16, 'pat')
	bf = np.linalg.norm(q[:, None, :].astype(np.float64) - points.astype(np.float32)[None], axis=-1)
	np.testing.assert_allclose(np.take_along_axis(bf, got, 1), np.sort(bf, 1)[:, :16], atol=1e-6)
	r.release()


def test_grid_probe_matches_trilinear(gl_ctx):
	from sdfview.backends.moderngl import Renderer

	rng = np.random.default_rng(3)
	lo, hi = np.array([-1.0, -0.5, -2.0]), np.array([1.5, 1.0, 1.0])
	res = (13, 9, 7)  # W, H, D
	axes = [np.linspace(lo[i], hi[i], res[i]) for i in range(3)]
	zz, yy, xx = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
	values = (np.sqrt(xx**2 + yy**2 + zz**2) - 0.8 + 0.1 * np.sin(3 * xx)).astype(np.float32)  # (D, H, W)
	scene = Scene(lo, hi)
	grid_field(scene, 'g', values, lo, hi)
	r = Renderer(gl_ctx, scene)
	q = rng.uniform(lo, hi, size=(2000, 3)).astype(np.float32)
	got = r.probe(q, 'g')[:, 0]
	# torch reference: grid_sample on (N=1, C=1, D, H, W) with align_corners=True (samples on the grid vertices)
	coords = torch.from_numpy(2 * (q - lo) / (hi - lo) - 1).float()[None, None, None]  # (1, 1, 1, n, 3) = (x, y, z)
	ref = F.grid_sample(torch.from_numpy(values)[None, None], coords, align_corners=True)[0, 0, 0, 0].numpy()
	cell_range = float(values.max() - values.min()) / min(res)
	err = np.abs(got - ref).max()
	print(f'\ngrid: max|Δφ| GLSL vs trilinear {err:.2e} (texture-filter weights may be 8-bit fixed point on GPUs)')
	assert err < cell_range / 64
	r.release()


def _sphere_mask(camera, w, h):
	"""Pixels whose center ray meets the unit sphere (OpenGL conventions, first row = top)."""
	vp = camera.proj_matrix(w / h) @ camera.view_matrix()
	inv = np.linalg.inv(vp)
	xs = (np.arange(w) + 0.5) / w * 2 - 1
	ys = (np.arange(h) + 0.5) / h * 2 - 1
	X, Y = np.meshgrid(xs, ys[::-1])
	near = np.stack([X, Y, -np.ones_like(X), np.ones_like(X)], -1) @ inv.T
	far = np.stack([X, Y, np.ones_like(X), np.ones_like(X)], -1) @ inv.T
	o = near[..., :3] / near[..., 3:]
	d = far[..., :3] / far[..., 3:] - o
	d /= np.linalg.norm(d, axis=-1, keepdims=True)
	b = (o * d).sum(-1)
	return (b * b - ((o * o).sum(-1) - 1.0)) > 0


@pytest.mark.parametrize('surface', ['pat', 'pat_grid'])
def test_render_unit_sphere_silhouette(gl_ctx, sphere, surface):
	"""Tori fitted to the exact form of the unit sphere: the blended field is |x| - 1, so the sphere-traced surface
	must cover exactly the pixels whose rays meet the unit sphere (up to pixels on the boundary)."""
	from sdfview.backends.moderngl import Renderer

	tdf, _ = sphere(torch.float32, n=800, noise=0.0)
	scene = scene_from_pat(tdf, k=16, bake_resolution=48 if surface == 'pat_grid' else None, padding=0.3)
	r = Renderer(gl_ctx, scene)
	w, h = 96, 72
	cam = OrbitCamera.fit(r.box_min, r.box_max)
	img = r.render(cam, ViewSettings(surface_field=surface, show_cutplane=False), (w, h))
	mask = np.any(img[..., :3] != 255, axis=-1)
	ref = _sphere_mask(cam, w, h)
	iou = (mask & ref).sum() / (mask | ref).sum()
	print(f'\n{surface}: silhouette IoU {iou:.4f}, differing pixels {(mask ^ ref).sum()} of {ref.sum()}')
	assert iou > 0.97
	r.release()


def test_render_all_passes(gl_ctx, sphere, tmp_path):
	from sdfview.backends.moderngl import Renderer, write_png

	tdf, _ = sphere(torch.float32, n=1500, noise=0.08)
	r = Renderer(gl_ctx, scene_from_pat(tdf, k=32, bake_resolution=32))
	cam = OrbitCamera.fit(r.box_min, r.box_max)
	for name, s in {
		'all': ViewSettings(show_points=True),
		'clip': ViewSettings(clip_surface=True, cut_axis=0, cut_offset=0.2),
		'oblique': ViewSettings(cut_normal=(1, 1, 0), cut_opacity=0.5, surface_field='pat'),
	}.items():
		img = r.render(cam, s, (80, 60))
		assert img.shape == (60, 80, 4) and len(np.unique(img.reshape(-1, 4), axis=0)) > 20, name
		write_png(str(tmp_path / f'{name}.png'), img)
	r.release()


@pytest.mark.skipif(shutil.which('glslangValidator') is None, reason='needs glslangValidator (glslang-tools)')
@pytest.mark.parametrize('target', ['gl330', 'webgl2'])
def test_sources_compile_for_target(tmp_path, target):
	"""The same GLSL modules compile as GLSL 3.30 core and as GLSL ES 3.00 (WebGL2) with the Khronos validator."""
	programs = [
		('fullscreen.vert', None),
		('surface.frag', 'FIELD_PAT'),
		('surface.frag', 'FIELD_GRID'),
		('cutplane.vert', None),
		('cutplane.frag', 'FIELD_PAT'),
		('cutplane.frag', 'FIELD_GRID'),
		('points.vert', None),
		('points.frag', None),
		('probe.frag', 'FIELD_PAT'),
		('probe_knn.frag', None),
		('blit.frag', None),
	]
	for name, field in programs:
		stem, stage = name.split('.')
		path = tmp_path / f'{stem}_{field}.{stage}'
		path.write_text(glsl_source(name, target, {field: None} if field else None))
		res = subprocess.run(['glslangValidator', str(path)], capture_output=True, text=True)
		assert res.returncode == 0, f'{target} {name} {field}:\n{res.stdout}{res.stderr}'


WINDOW_SCRIPT = r"""
import sys, numpy as np, moderngl_window as mglw
from sdfview import Scene
from sdfview.backends.moderngl.window import ViewerWindow
scene = Scene.load(sys.argv[1])
cls = type('W', (ViewerWindow,), {'scene': scene, 'window_size': (160, 120)})
cfg = mglw.create_window_config_instance(cls, args=['--window', sys.argv[2]])
wnd = cfg.wnd
class Mods: shift = False; ctrl = False; alt = False
for key in ('P', 'X', 'UP', 'K', 'G'):
    wnd.clear(); wnd.render(0.0, 0.016); wnd.swap_buffers()
    cfg.on_key_event(getattr(wnd.keys, key), wnd.keys.ACTION_PRESS, Mods())
cfg.on_mouse_press_event(10, 10, 1); cfg.on_mouse_drag_event(12, 10, 2, 0); cfg.on_mouse_release_event(12, 10, 1)
cfg.on_mouse_scroll_event(0, 1)
wnd.clear(); wnd.render(0.1, 0.016)
cfg.on_mouse_press_event(80, 60, 1); cfg.on_mouse_release_event(80, 60, 1)  # a click: the tori inspector
for key in ('PERIOD', 'A', 'J', 'L'):
    cfg.on_key_event(getattr(wnd.keys, key), wnd.keys.ACTION_PRESS, Mods())
print('inspected:', cfg._inspection is not None, 'rank:', getattr(cfg._inspection, 'selected', None))
wnd.clear(); wnd.render(0.2, 0.016)
w, h = wnd.buffer_size
img = np.frombuffer(wnd.fbo.read(components=3), np.uint8).reshape(h, w, 3)
print('title:', wnd.title)
print('distinct colors:', len(np.unique(img.reshape(-1, 3), axis=0)))
wnd.close(); wnd.destroy()
"""


@pytest.mark.parametrize('backend', ['pyglet', 'glfw'])
def test_window(tmp_path, sphere, backend):
	pytest.importorskip('moderngl_window')
	pytest.importorskip(backend)
	if not os.environ.get('DISPLAY') and shutil.which('xvfb-run') is None:
		pytest.skip('needs a display (DISPLAY) or xvfb-run')
	tdf, _ = sphere(torch.float32, n=600)
	path = str(tmp_path / 's.sdfx')
	scene_from_pat(tdf, k=16, bake_resolution=24).save(path)
	script = tmp_path / 'w.py'
	script.write_text(WINDOW_SCRIPT)
	cmd = [sys.executable, str(script), path, backend]
	if not os.environ.get('DISPLAY'):
		cmd = ['xvfb-run', '-a', '-s', '-screen 0 640x480x24'] + cmd
	# the script runs from a temporary directory: make the repository importable whether or not it is pip-installed
	env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [REPO, os.environ.get('PYTHONPATH')])))
	res = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=REPO, env=env)
	assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
	colors = int(res.stdout.split('distinct colors:')[1].split()[0])
	print('\n' + [ln for ln in res.stdout.splitlines() if ln.startswith('title:')][0])
	assert colors > 10
	assert 'inspected: True rank: 1' in res.stdout, res.stdout[-2000:]


def test_cli(tmp_path, sphere, gl_ctx):
	tdf, _ = sphere(torch.float32, n=500)
	path = str(tmp_path / 's.sdfx')
	scene_from_pat(tdf, k=16, bake_resolution=16).save(path)
	env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [REPO, os.environ.get('PYTHONPATH')])))
	run = lambda *a: subprocess.run(  # noqa: E731
		[sys.executable, '-m', 'sdfview', *a], capture_output=True, text=True, cwd=REPO, env=env
	)
	res = run('info', path)
	assert res.returncode == 0 and 'pat/bvh_nodes' in res.stdout, res.stderr
	png = str(tmp_path / 'x.png')
	res = run('render', path, '-o', png, '--size', '64', '48', '--points', '--axis', 'x')
	assert res.returncode == 0 and os.path.getsize(png) > 100, res.stderr
	with open(png, 'rb') as f:
		assert f.read(8) == b'\x89PNG\r\n\x1a\n'


def test_overlay_of_models(gl_ctx):
	"""Two models in a merged scene (spheres of radius 0.6 at x = ∓0.35): 'single' draws the selected one in its
	model color, 'overlay' both (nearest wins), a translucent overlay blends the other one over the opaque one."""
	from sdfview.backends.moderngl import Renderer
	from sdfview.backends.moderngl.renderer import MODEL_COLORS
	from sdfview.scene import merge_scenes

	x = np.linspace(-1.2, 1.2, 40)
	Z, Y, X = np.meshgrid(x, x, x, indexing='ij')
	scenes = {}
	for model, cx in (('pat', -0.35), ('sup', 0.35)):
		s = Scene([-1.2] * 3, [1.2] * 3)
		grid_field(s, model, np.sqrt((X - cx) ** 2 + Y**2 + Z**2) - 0.6, [-1.2] * 3, [1.2] * 3)
		scenes[model] = s
	r = Renderer(gl_ctx, merge_scenes(scenes))
	assert r.models() == ['pat', 'sup']
	cam = OrbitCamera.fit(r.box_min, r.box_max, up='y')  # looks along -z: the two spheres side by side in x

	def hue_fraction(img, model):
		"""Fraction of drawn pixels whose dominant channel matches the model's color (pat: red, sup: blue)."""
		rgb = img[..., :3].astype(float)
		drawn = np.any(img[..., :3] != 255, axis=-1)
		channel = 0 if model == 'pat' else 2
		return float((np.argmax(rgb, axis=-1) == channel)[drawn].mean())

	single = r.render(cam, ViewSettings(surface_field='sup', show_cutplane=False), (96, 72))
	assert hue_fraction(single, 'sup') > 0.95
	overlay = r.render(cam, ViewSettings(surface_field='sup', surface_mode='overlay', show_cutplane=False), (96, 72))
	assert 0.2 < hue_fraction(overlay, 'pat') < 0.8 and 0.2 < hue_fraction(overlay, 'sup') < 0.8
	assert np.any(overlay[..., :3] != 255, axis=-1).sum() > np.any(single[..., :3] != 255, axis=-1).sum()
	ghost = r.render(
		cam,
		ViewSettings(surface_field='sup', surface_mode='overlay', overlay_opacity=0.4, show_cutplane=False),
		(96, 72),
	)
	assert not np.array_equal(ghost, overlay) and MODEL_COLORS['pat'][0] > MODEL_COLORS['pat'][2]
	r.release()
