"""
The tori inspector (sdfview/inspector.py): principal curvatures and directions recovered from the stored tori against
the fundamental forms they were fitted from (points_as_tori.torus_distance), the blend at a point against the torch
evaluator, the network neighborhood against points_as_tori.get_neighbors; picking and drawing in the moderngl host.
"""

import numpy as np
import pytest
import torch

import points_as_tori as pat
from points_as_tori import config
from points_as_tori.shape_2d import orthonormal_basis
from points_as_tori.torus_distance import get_neighbors, get_principal_directions_at_point
from sdfview import OrbitCamera, ViewSettings
from sdfview.inspector import inspect, select, torus_distance
from sdfview.sources.pat import scene_from_pat


def curved_forms(n=600, seed=1):
	"""Points on the unit sphere with radial normals and quadratic forms of both signs: convex, concave and saddle
	patches, non-zero slopes (a10, a01) and offsets a00 (coefficients (a00, a01, a10, a11, a02, a20))."""
	rng = np.random.default_rng(seed)
	p = rng.normal(size=(n, 3))
	p /= np.linalg.norm(p, axis=1, keepdims=True)
	c = np.zeros((n, 6))
	c[:, 0] = rng.uniform(-0.01, 0.01, n)
	c[:, 1:3] = rng.uniform(-0.2, 0.2, (n, 2))
	c[:, 3] = rng.uniform(-0.3, 0.3, n)
	c[:, 4] = rng.uniform(-1.5, 1.5, n)
	c[:, 5] = rng.uniform(-1.5, 1.5, n)
	keep = np.abs(np.abs(c[:, 4]) - np.abs(c[:, 5])) > 0.2  # distinct principal curvatures: directions defined
	keep &= np.minimum(np.abs(c[:, 4]), np.abs(c[:, 5])) > 0.1  # no near-flat direction: R stays moderate
	return p[keep], p[keep].copy(), c[keep]


@pytest.fixture(scope='module')
def fitted():
	config.set_device('cpu', torch.float64, force=True)
	points, normals, coeffs = curved_forms()
	tdf = pat.TorusDistanceField(config.as_wide(points, 3), config.as_wide(normals, 3))
	tdf.fit_tori_from_forms(config.as_wide(coeffs, 6))
	tdf.set_k_evaluation(32)
	scene = scene_from_pat(tdf, k=32)
	n = config.as_wide(normals, 3)
	s_hat, t_hat = orthonormal_basis(n)
	k_min, k_max, v_min, v_max = get_principal_directions_at_point(n, config.as_wide(coeffs, 6), s_hat, t_hat)
	ref = {
		'kappa_min': k_min.numpy(),
		'kappa_max': k_max.numpy(),
		'v_min': (v_min / torch.linalg.vector_norm(v_min, dim=0)).numpy().T,
		'v_max': (v_max / torch.linalg.vector_norm(v_max, dim=0)).numpy().T,
		'foot': points + coeffs[:, :1] * normals,  # p* = p + a00 n
	}
	yield tdf, scene, points, coeffs, ref
	config.set_device('cpu', torch.float32, force=True)


def test_curvatures_from_tori(fitted):
	"""κ_min, κ_max (values and signs), their directions and the foot point p*, from the stored (float32) tori."""
	tdf, scene, points, coeffs, ref = fitted
	ins = inspect(scene, points[0])
	errs = {'kappa': [], 'dir': [], 'foot': [], 'outer': []}
	for j in range(len(points)):
		ins.neighbors[0] = j
		t = select(scene, ins, 0).torus
		scale = max(abs(ref['kappa_max'][j]), 1.0)
		errs['kappa'].append(
			max(abs(t.kappa_min - ref['kappa_min'][j]), abs(t.kappa_max - ref['kappa_max'][j])) / scale
		)
		errs['dir'].append(max(1 - abs(t.dir_min @ ref['v_min'][j]), 1 - abs(t.dir_max @ ref['v_max'][j])))
		errs['foot'].append(np.linalg.norm(t.foot - ref['foot'][j]))
		errs['outer'].append(t.outer != (ref['kappa_min'][j] * ref['kappa_max'][j] > 0))
	saddles = int(np.sum(ref['kappa_min'] * ref['kappa_max'] < 0))
	print(
		f'\n{len(points)} points ({saddles} saddles): max relative |Δκ| {max(errs["kappa"]):.2e}, '
		f'max (1 - |cos|) of the directions {max(errs["dir"]):.2e}, max |p_t - p*| {max(errs["foot"]):.2e}, '
		f'equator side wrong: {sum(errs["outer"])}'
	)
	# measured: |Δκ| 1.0e-5 (float32 tori), directions 1 - |cos| 9.3e-6 (0.25°), |p_t - p*| 2.5e-3: p_t is the torus
	# point closest to p_j, p* = p_j + a00 n_j the one the network evaluated; n_j is not the torus normal (slopes
	# a10, a01), so they differ by ~|a00| · slope, and the frame at p_t turns by ~|p_t - p*| |κ| against the one at p*
	assert saddles > 0 and max(errs['kappa']) < 1e-4 and max(errs['dir']) < 1e-4 and sum(errs['outer']) == 0
	assert max(errs['foot']) < 0.01


def test_blend_matches_evaluator(fitted):
	tdf, scene, points, coeffs, ref = fitted
	q = np.random.default_rng(5).uniform(-1.3, 1.3, size=(20, 3))
	phi_ref = tdf.evaluate_distance(config.as_wide(q, 3)).numpy()
	phi = np.array([inspect(scene, x).phi for x in q])
	print(f'\nblend at 20 points: max |Δφ| {np.abs(phi - phi_ref).max():.2e}')
	assert np.abs(phi - phi_ref).max() < 1e-4  # float32 scene arrays
	ins = inspect(scene, q[0])
	assert abs(ins.weights.sum() - 1) < 1e-12 and np.all(np.diff(ins.weights) <= 0)
	t = ins.torus
	g = torus_distance(t.foot, t.center, t.axis, t.R, t.s * t.r)
	assert abs(g) < 1e-9  # the foot point is on the torus


def test_network_neighborhood(fitted):
	tdf, scene, points, coeffs, ref = fitted
	ins = inspect(scene, points[3], network_k=16)
	j = int(ins.neighbors[ins.selected])
	expect = get_neighbors(config.as_wide(points, 3), 16)[j].numpy()
	assert ins.network[0] == j and set(ins.network.tolist()) == set(expect.tolist())


@pytest.fixture(scope='module')
def gl():
	try:
		from sdfview.backends.moderngl import create_context

		return create_context()
	except Exception as e:  # noqa: BLE001
		pytest.skip(f'no OpenGL context: {e}')


def test_pick_and_draw(gl, fitted):
	"""Pick the surface at the image center (depth buffer), inspect there, draw the inspector."""
	from sdfview.backends.moderngl import Renderer

	tdf, scene, points, coeffs, ref = fitted
	r = Renderer(gl, scene)
	cam = OrbitCamera.fit(r.box_min, r.box_max)
	s = ViewSettings(show_cutplane=False)
	size = (96, 72)
	x = r.pick(cam, s, size, (48, 36))
	assert x is not None
	phi = r.probe(x[None].astype(np.float32), 'auto', s)[0, 0]
	print(f'\npicked {x}, φ there {phi:+.2e}')
	assert abs(phi) < 1e-3 * r.diagonal
	assert r.pick(cam, s, size, (0, 0)) is None  # background
	plain = r.render(cam, s, size)
	r.set_inspection(inspect(scene, x))
	shown = r.render(cam, s, size)
	changed = np.any(plain != shown, axis=-1).mean()
	print(f'inspector changes {changed:.3f} of the pixels')
	assert 0.005 < changed < 0.9
	r.set_inspection(None)
	assert np.array_equal(r.render(cam, s, size), plain)
	r.release()


@pytest.fixture(scope='module')
def network_scene():
	"""tooth.ply (every 4th point) fitted by the bundled network: the scene carries forms, curvature and frames."""
	import os

	from sdfview.scene import Scene

	config.set_device('cpu', torch.float32, force=True)
	p, n = pat.read_point_cloud(
		os.path.join(os.path.dirname(__file__), '..', '..', 'ext', 'points-as-tori', 'data', 'tooth.ply')
	)
	sdf = pat.PointsAsTori(p[::4], n[::4])
	scene = scene_from_pat(sdf, k=32)
	return Scene.from_bytes(scene.to_bytes()), sdf  # through the file format


def test_stored_curvatures(network_scene):
	"""The inspector shows the network's stored κ, directions and foot point; the torus recovery agrees with them."""
	scene, sdf = network_scene
	f = scene.field('pat')
	assert {'forms', 'curvature', 'frames'} <= set(f['arrays']) and f['params']['network_model'] == 'bundled'
	frame = sdf.curvatures()
	ins = inspect(scene, scene.arrays['pat/points'][0, :3])
	worst = 0.0
	for rank in range(len(ins.neighbors)):
		t = select(scene, ins, rank).torus
		j = int(ins.neighbors[rank])
		assert t.source == 'stored'
		assert abs(t.kappa_max - float(frame['kappa_max'][j])) < 1e-5 * max(1, abs(t.kappa_max))
		worst = max(
			worst,
			abs(t.recovered[1] - t.kappa_max) / abs(t.kappa_max),
			abs(t.recovered[0] - t.kappa_min) / abs(t.kappa_min),
		)
	print(f'\nstored vs recovered from the torus: max relative |Δκ| {worst:.2e}')
	assert worst < 1e-3


def test_explain_network(network_scene, gl):
	from sdfview.backends.moderngl import Renderer
	from sdfview.inspector import describe, explain_network, marks

	scene, sdf = network_scene
	ins = inspect(scene, scene.arrays['pat/points'][5, :3])
	ex = explain_network(scene, ins)
	assert ex is not None and len(ex['gradient']) == len(ex['rollout']) == len(ins.network) == sdf.network_k
	j = int(ins.neighbors[ins.selected])
	ref = sdf.explain(j)
	assert np.allclose(ex['gradient'], ref['gradient'].numpy(), rtol=1e-4, atol=1e-6)
	lines = describe(ins)
	assert any('least-squares' in ln for ln in lines) and any('gradient' in ln for ln in lines)
	flat = marks(scene, ins, 0.01)[2]
	colored = marks(scene, ins, 0.01, attribution='gradient')[2]
	assert len(flat) == len(colored) and not np.array_equal(flat, colored)
	r = Renderer(gl, scene)
	r.set_inspection(ins, attribution='rollout')
	img = r.render(OrbitCamera.fit(r.box_min, r.box_max), ViewSettings(show_cutplane=False), (64, 48))
	assert img.shape == (48, 64, 4)
	r.release()
