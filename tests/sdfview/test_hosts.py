"""
Parity of the browser hosts (sdfview/web: raw WebGL2, raw WebGPU, three.js WebGL, three.js WebGPU) with torch:
each host evaluates the field at the same query points through its own probe path, in headless Chromium
(Playwright, SwiftShader without a GPU), on scenes served by sdfview.server. Tolerances as tests/sdfview/test_gl.py.
Skipped without node + playwright; the three.js hosts also need three.js (npm pack three@<THREE_VERSION>).
"""

import json
import os
import shutil
import subprocess
import tarfile

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from points_as_tori import config
from sdfview.scene import Scene, grid_field
from sdfview.server import ROOT, serve_in_thread
from sdfview.sources.pat import scene_from_pat
from conftest import sphere_field

THREE_VERSION = '0.186.1'
RUNNER = os.path.join(ROOT, 'web', 'test', 'run_probe.mjs')
# field types each host evaluates
HOSTS = {
	'webgl2': ['pat', 'grid', 'glsl'],
	'webgpu': ['pat', 'grid'],
	'three-webgl': ['pat', 'grid', 'glsl'],
	'three-webgpu': ['grid'],
	'three-tsl-webgl': ['grid'],  # the TSL host on WebGPURenderer's WebGL2 backend (forceWebGL)
}
N_QUERIES = 1500


def near_surface_queries(n, seed=0, spread=0.3, box=1.3):
	rng = np.random.default_rng(seed)
	u = rng.normal(size=(n, 3))
	u /= np.linalg.norm(u, axis=1, keepdims=True)
	q = u * (1 + spread * rng.uniform(-1, 1, size=(n, 1)))
	return np.concatenate([q, rng.uniform(-box, box, size=(n // 4, 3))]).astype(np.float32)


def grid_values():
	lo, hi = np.array([-1.0, -0.5, -2.0]), np.array([1.5, 1.0, 1.0])
	res = (13, 9, 7)  # W, H, D
	axes = [np.linspace(lo[i], hi[i], res[i]) for i in range(3)]
	zz, yy, xx = np.meshgrid(axes[2], axes[1], axes[0], indexing='ij')
	values = (np.sqrt(xx**2 + yy**2 + zz**2) - 0.8 + 0.1 * np.sin(3 * xx)).astype(np.float32)  # (D, H, W)
	return lo, hi, res, values


@pytest.fixture(scope='module')
def node_ok():
	if not shutil.which('node'):
		pytest.skip('node not found')
	res = subprocess.run(['node', RUNNER], input='{"jobs": []}', capture_output=True, text=True, timeout=120)
	if res.returncode != 0:
		pytest.skip(f'playwright / chromium not usable: {res.stderr[-300:]}')


@pytest.fixture(scope='module')
def three_dir(tmp_path_factory):
	"""A local copy of three.js (npm pack), served in place of the CDN; None if it cannot be fetched."""
	d = tmp_path_factory.mktemp('three')
	try:
		subprocess.run(['npm', 'pack', f'three@{THREE_VERSION}', '--pack-destination', str(d)], check=True,
					   capture_output=True, timeout=180)  # fmt: skip
		tgz = next(d.glob('three-*.tgz'))
		with tarfile.open(tgz) as t:
			t.extractall(d, filter='data')
		return str(d / 'package')
	except Exception:  # noqa: BLE001
		return None


@pytest.fixture(scope='module')
def scenes(tmp_path_factory):
	"""{kind: (scene path, field name, queries (n, 3), reference (n, 4) or None for the gradient)}."""
	d = tmp_path_factory.mktemp('scenes')
	out = {}
	config.set_device('cpu', torch.float32, force=True)
	tdf, _ = sphere_field(noise=0.08)
	tdf.set_k_evaluation(32)
	q = near_surface_queries(N_QUERIES)
	dist, grad = tdf.evaluate_distance_gradient(config.as_wide(q, 3))
	scene_from_pat(tdf, k=32).save(str(d / 'pat.sdfx'))
	out['pat'] = (str(d / 'pat.sdfx'), 'pat', q, np.concatenate([dist.numpy()[:, None], grad.numpy().T], 1))

	lo, hi, res, values = grid_values()
	scene = Scene(lo, hi)
	grid_field(scene, 'g', values, lo, hi)
	scene.save(str(d / 'grid.sdfx'))
	qg = np.random.default_rng(3).uniform(lo, hi, size=(N_QUERIES, 3)).astype(np.float32)
	coords = torch.from_numpy(2 * (qg - lo) / (hi - lo) - 1).float()[None, None, None]
	ref = F.grid_sample(torch.from_numpy(values)[None, None], coords, align_corners=True)[0, 0, 0, 0].numpy()
	out['grid'] = (str(d / 'grid.sdfx'), 'g', qg, ref[:, None])

	try:
		import sys

		from sdfview.sources.superfit import scene_from_superfit
		from superfit_sdf import SuperFitSDF

		sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'superfit'))
		from test_superfit_sdf import assembly_expr

		sup = SuperFitSDF(assembly_expr(drop_third=False).sympy().state(), dtype=torch.float64)
		scene_from_superfit(sup, resolution=None, name='sup').save(str(d / 'sup.sdfx'))
		qs = np.random.default_rng(4).uniform(-0.9, 0.9, size=(N_QUERIES, 3)).astype(np.float32)
		out['glsl'] = (str(d / 'sup.sdfx'), 'sup', qs, sup(torch.from_numpy(qs).double().T).numpy()[:, None])
	except Exception:  # noqa: BLE001  (no geolipi / sysl3d / ext/superfit)
		pass
	return out


@pytest.fixture(scope='module')
def results(node_ok, scenes, three_dir):
	"""Every (host, field kind) probe and one render per host, in one browser session per run."""
	server, base = serve_in_thread({os.path.basename(v[0]): v[0] for v in scenes.values()})
	jobs, keys = [], []
	only = [h for h in os.environ.get('SDFX_HOSTS', '').split(',') if h]  # e.g. SDFX_HOSTS=webgpu while developing
	for host, kinds in HOSTS.items():
		if only and host not in only:
			continue
		if host.startswith('three') and three_dir is None:
			continue
		for kind in kinds:
			if kind in scenes:
				path, field, q, _ = scenes[kind]
				url = f'/scenes/{os.path.basename(path)}'
				jobs.append({'host': host, 'scene': url, 'field': field, 'queries': q.reshape(-1).tolist()})
				keys.append((host, kind, 'probe'))
		# renders (tracing the exact PAT surface on a software GPU takes minutes: PAT only on the cut plane)
		for case, (kind, _) in RENDER_CASES.items():
			jobs.append(
				{
					'host': host,
					'scene': f'/scenes/{os.path.basename(scenes[kind][0])}',
					'kind': 'render',
					'settings': render_settings(case, host)[0],
				}
			)
			keys.append((host, case, 'render'))
	payload = json.dumps({'base': base, 'threeDir': three_dir, 'jobs': jobs})
	res = subprocess.run(['node', RUNNER], input=payload, capture_output=True, text=True, timeout=900)
	server.shutdown()
	assert res.returncode == 0, res.stderr[-3000:]
	out = dict(zip(keys, json.loads(res.stdout)['results']))
	for key, r in out.items():
		print(f'{key}: ' + (f'{r["seconds"]:.1f} s' if 'seconds' in r else r.get('error', '')[:120]))
	return out


def get(results, host, kind, what):
	if host.startswith('three') and not any(k[0] == host for k in results):
		pytest.skip('three.js not available (npm pack failed)')
	r = results.get((host, kind, what))
	if r is None:
		pytest.skip(f'{host} / {kind}: not run')
	assert 'error' not in r, f'{host} / {kind}: {r["error"]}\n{r.get("logs")}'
	return r['value']


@pytest.mark.parametrize('host,kind', [(h, k) for h, kinds in HOSTS.items() for k in kinds])
def test_probe_matches_torch(results, scenes, host, kind):
	if kind not in scenes:
		pytest.skip(f'no {kind} scene (optional dependency missing)')
	_, _, q, ref = scenes[kind]
	got = np.asarray(get(results, host, kind, 'probe'), np.float64).reshape(-1, 4)
	err_d = np.abs(got[:, 0] - ref[:, 0]).max()
	if kind == 'grid':
		_, _, res, values = grid_values()
		cell_range = float(values.max() - values.min()) / min(res)
		print(f'\n{host} grid: max|Δφ| {err_d:.2e} (texture-filter weights may be 8-bit on GPUs)')
		assert err_d < cell_range / 64
	elif kind == 'pat':
		err_g = np.abs(got[:, 1:] - ref[:, 1:]).max()
		print(f'\n{host} pat: max|Δφ| {err_d:.2e}, max|Δ∇φ| {err_g:.2e}')
		assert err_d < 2e-5 and err_g < 5e-3
	else:
		# GLSL ES leaves the precision of sin / cos / tan implementation-defined; SwiftShader's (Chromium without a GPU)
		# measured: sin, cos 1.9e-4, tan 7.6e-5, atan 1.6e-7 (relative to max(1, |v|)); the SuperFrustum's bulge map
		# uses them. Through moderngl / Mesa llvmpipe the same code is within 2.4e-6 (tests/superfit/test_glsl.py).
		print(f'\n{host} glsl (SuperFit): max|Δφ| {err_d:.2e}')
		assert err_d < 1e-3


RENDER_SIZE = (160, 120)  # the canvas of web/test/probe.html
# case -> (scene, settings as {JavaScript name: (value, ViewSettings name)})
RENDER_CASES = {
	'surface': ('grid', {'showCutplane': (False, 'show_cutplane')}),
	'clip': (
		'grid',
		{'clipSurface': (True, 'clip_surface'), 'cutAxis': (0, 'cut_axis'), 'cutOffset': (0.2, 'cut_offset')},
	),
	'cut+points': ('pat', {'showSurface': (False, 'show_surface'), 'showPoints': (True, 'show_points')}),
}


def render_settings(case, host):
	"""(JavaScript overrides, ViewSettings kwargs) of a render case on a host; a host without the scene's field type
	draws no cut plane (the reference neither)."""
	kind, settings = RENDER_CASES[case]
	js = {k: v for k, (v, _) in settings.items()}
	py = {name: v for v, name in settings.values()}
	if kind not in HOSTS[host]:
		js['showCutplane'], py['show_cutplane'] = False, False
	return js, py


@pytest.fixture(scope='module')
def moderngl_images(scenes):
	"""Reference images: the same scene, camera fit and settings through sdfview's moderngl Renderer, (h, w, 4), top
	row first; cached per (scene, settings)."""
	try:
		from sdfview import OrbitCamera, ViewSettings
		from sdfview.backends.moderngl import Renderer, create_context

		ctx = create_context()
	except Exception as e:  # noqa: BLE001  (no moderngl / no EGL)
		pytest.skip(f'moderngl unavailable: {e}')
	cache = {}

	def image(kind, settings):
		key = (kind, tuple(sorted(settings.items())))
		if key not in cache:
			r = Renderer(ctx, Scene.load(scenes[kind][0]))
			cache[key] = r.render(OrbitCamera.fit(r.box_min, r.box_max), ViewSettings(**settings), size=RENDER_SIZE)
			r.release()
		return cache[key]

	return image


def shift_tolerant_difference(img, ref):
	"""Per pixel: min over the 3x3 neighborhood of the reference of max_channel |img - ref|. Edges that land one pixel
	apart (rasterization rules, float rounding on a box face or a sprite border) give 0; different content does not."""
	a = img[..., :3].astype(np.int16)
	b = np.pad(ref[..., :3].astype(np.int16), ((1, 1), (1, 1), (0, 0)), mode='edge')
	h, w = a.shape[:2]
	shifts = [np.abs(a - b[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]).max(-1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
	return np.min(shifts, axis=0)


@pytest.mark.parametrize('host,case', [(h, c) for h in HOSTS for c in RENDER_CASES])
def test_render(results, moderngl_images, host, case):
	w, h = RENDER_SIZE
	kind = RENDER_CASES[case][0]
	ref = moderngl_images(kind, render_settings(case, host)[1])
	img = np.asarray(get(results, host, case, 'render'), np.uint8).reshape(h, w, 4)[::-1]  # bottom-up -> top first
	covered = (img[..., :3] < 250).any(-1).mean()
	diff = np.abs(img[..., :3].astype(np.int16) - ref[..., :3]).max(-1)
	tolerant = shift_tolerant_difference(img, ref)
	print(
		f'\n{host} {case}: {covered:.3f} of the pixels drawn; vs moderngl: |Δ| > 8 at {(diff > 8).mean():.4f} of the '
		f'pixels (max {diff.max()}), ±1 px tolerant: {(tolerant > 8).mean():.4f} (max {tolerant.max()})'
	)
	# measured (SwiftShader vs llvmpipe), pixels with |Δ| > 8 / ±1 px tolerant: surface 0 / 0; clip 0.0018 / 0.0011;
	# cut+points 0.0051 / 0.0022 (1-2 px sprites covered or not, overlapping sprites at equal depth, the zero contour's
	# fwidth: GLSL allows coarse or fine derivatives). Errors seen while writing the hosts: a missing cut plane 0.12,
	# surface depth not written 0.044.
	assert 0.002 < covered < 0.98
	assert (diff > 8).mean() < 0.03 and (tolerant > 8).mean() < 0.005
