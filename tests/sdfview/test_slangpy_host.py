"""
The slangpy host (sdfview/backends/slangpy: the Slang modules of sdfview/shaders/slang as compute kernels) against
torch (probes) and against the moderngl host (images: same camera, settings and field). Skipped without slangpy or
without a device (Vulkan; lavapipe on a machine without a GPU).
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from points_as_tori import config
from sdfview import OrbitCamera, ViewSettings
from sdfview.scene import Scene, grid_field, merge_scenes
from sdfview.sources.pat import scene_from_pat
from conftest import sphere_field
from test_hosts import grid_values, near_surface_queries

spy = pytest.importorskip('slangpy')


@pytest.fixture(scope='module')
def device():
	from sdfview.backends.slangpy import include_paths

	try:
		return spy.create_device(type=spy.DeviceType.vulkan, include_paths=include_paths())
	except Exception as e:  # noqa: BLE001
		pytest.skip(f'no Vulkan device: {e}')


@pytest.fixture(scope='module')
def pat_scene():
	config.set_device('cpu', torch.float32, force=True)
	tdf, _ = sphere_field(noise=0.08)
	tdf.set_k_evaluation(32)
	q = near_surface_queries(1500)
	dist, grad = tdf.evaluate_distance_gradient(config.as_wide(q, 3))
	ref = np.concatenate([dist.numpy()[:, None], grad.numpy().T], 1)
	return scene_from_pat(tdf, k=32), q, ref


@pytest.fixture(scope='module')
def grid_scene():
	lo, hi, res, values = grid_values()
	scene = Scene(lo, hi)
	grid_field(scene, 'g', values, lo, hi)
	q = np.random.default_rng(3).uniform(lo, hi, size=(1500, 3)).astype(np.float32)
	coords = torch.from_numpy(2 * (q - lo) / (hi - lo) - 1).float()[None, None, None]
	ref = F.grid_sample(torch.from_numpy(values)[None, None], coords, align_corners=True)[0, 0, 0, 0].numpy()
	return scene, q, ref


@pytest.fixture(scope='module')
def gl_renderer_factory():
	try:
		from sdfview.backends.moderngl import Renderer, create_context

		ctx = create_context()
	except Exception as e:  # noqa: BLE001
		pytest.skip(f'moderngl unavailable: {e}')
	return lambda scene: Renderer(ctx, scene)


def test_probe_pat(device, pat_scene):
	from sdfview.backends.slangpy import Renderer

	scene, q, ref = pat_scene
	got = Renderer(scene, device).probe(q)
	err_d, err_g = np.abs(got[:, 0] - ref[:, 0]).max(), np.abs(got[:, 1:] - ref[:, 1:]).max()
	print(f'\nslangpy pat: max|Δφ| {err_d:.2e}, max|Δ∇φ| {err_g:.2e}')
	assert err_d < 2e-5 and err_g < 5e-3


def test_probe_grid(device, grid_scene):
	from sdfview.backends.slangpy import Renderer

	scene, q, ref = grid_scene
	got = Renderer(scene, device).probe(q, 'g')
	err = np.abs(got[:, 0] - ref).max()
	print(f'\nslangpy grid: max|Δφ| {err:.2e}')
	assert err < 1e-5


def _compare(a, b):
	d = np.abs(a[..., :3].astype(np.int16) - b[..., :3]).max(-1)
	return float((a[..., :3] < 250).any(-1).mean()), int(d.max()), float((d > 8).mean())


@pytest.mark.parametrize('kind,size', [('grid', (160, 120)), ('pat', (64, 48))])
def test_render_matches_moderngl(device, gl_renderer_factory, pat_scene, grid_scene, kind, size):
	"""Same image as the moderngl Renderer without its cut plane (the Slang host draws surfaces only)."""
	from sdfview.backends.slangpy import Renderer

	scene = (pat_scene if kind == 'pat' else grid_scene)[0]
	r, g = Renderer(scene, device), gl_renderer_factory(scene)
	cam = OrbitCamera.fit(r.box_min, r.box_max)
	s = ViewSettings(show_cutplane=False)
	a, b = r.render(cam, s, size), g.render(cam, s, size)
	g.release()
	covered, dmax, frac = _compare(a, b)
	print(
		f'\nslangpy {kind}: {covered:.3f} of the pixels show the surface; vs moderngl max|Δ| {dmax}, |Δ| > 8: {frac:.4f}'
	)
	assert 0.02 < covered < 0.98 and frac < 0.01


def test_translucent_overlay_matches_moderngl(device, gl_renderer_factory):
	"""Two models (grids), the second one blended over the first (opacity 0.4): the blending written in surface.slang
	against GL's blend state."""
	from sdfview.backends.slangpy import Renderer

	x = np.linspace(-1.2, 1.2, 40)
	Z, Y, X = np.meshgrid(x, x, x, indexing='ij')
	scenes = {}
	for model, cx in (('pat', -0.35), ('sup', 0.35)):
		s = Scene([-1.2] * 3, [1.2] * 3)
		grid_field(s, model, (np.sqrt((X - cx) ** 2 + Y**2 + Z**2) - 0.6).astype(np.float32), [-1.2] * 3, [1.2] * 3)
		scenes[model] = s
	scene = merge_scenes(scenes)
	r, g = Renderer(scene, device), gl_renderer_factory(scene)
	cam = OrbitCamera.fit(r.box_min, r.box_max, up='y')
	s = ViewSettings(surface_field='sup', surface_mode='overlay', overlay_opacity=0.4, show_cutplane=False)
	a, b = r.render(cam, s, (96, 72)), g.render(cam, s, (96, 72))
	g.release()
	covered, dmax, frac = _compare(a, b)
	print(f'\nslangpy overlay: max|Δ| {dmax}, |Δ| > 8: {frac:.4f}')
	assert frac < 0.01
