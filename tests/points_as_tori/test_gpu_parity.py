"""
The port on CUDA vs the port on CPU. The CPU path is validated against the original C++ and JAX in the other test
files; these tests check that the same code gives the same results on each visible CUDA device, and that the
inference path has no host synchronization on CUDA. Skipped without CUDA. Measured differences are printed
(pytest -s).

Sources of difference between devices, and the tolerances they set:
  * float64 geometry: different reduction orders and libm implementations (CUDA's exp / sqrt / atan2 are within 1-2
    ulp of the correctly rounded value): relative differences ~1e-15-1e-12, asserted ≤ 1e-9;
  * float32 geometry: the same in float32 (relative ~1e-7-1e-5), asserted ≤ 1e-4;
  * the network always runs in float32 and attention / matmul kernels differ (cuBLAS, fused attention vs MKL), so
    its coefficients differ at ~1e-6-1e-4 relative; the tori derived from them (radii are inverse curvatures) and the
    SDF inherit this, so the end-to-end check compares signs and robust statistics.
k-NN: the comparisons set config.KNN_BACKEND = 'bvh' on both devices (exact, same algorithm), so that the tests
measure arithmetic differences and not backend differences ('slang' vs 'bvh' is tested in test_knn_slang.py).
"""

import os

import numpy as np
import pytest
import torch

import points_as_tori as pat
from points_as_tori import config
from helpers import TORCH_MODEL, random_tori, read_ply, synthetic_dataset, torus_mesh

# PAT_PARITY_DEVICES=cuda:1 (comma-separated) restricts the devices; =cpu runs the comparisons CPU vs CPU, to check the
# tests themselves on a machine without a GPU.
GPUS = [d for d in os.environ.get('PAT_PARITY_DEVICES', '').split(',') if d] or [
	f'cuda:{i}' for i in range(torch.cuda.device_count())
]
pytestmark = pytest.mark.skipif(not GPUS, reason='needs CUDA')


@pytest.fixture(params=GPUS or ['cuda:0'])
def gpu(request):
	saved = config.KNN_BACKEND
	config.KNN_BACKEND = 'bvh'
	yield torch.device(request.param)
	config.KNN_BACKEND = saved
	config.set_device('cpu', torch.float64, force=True)


def on(device, dtype):
	config.set_device(device, dtype, force=True)


def rel_diff(a, b):
	"""max |a - b| / max |b| (CPU tensors)."""
	a, b = a.detach().double().cpu(), b.detach().double().cpu()
	return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def report(label, value):
	print(f'  {label}: {value:.3g}')
	return value


def evaluator(seed=9, n=3000):
	P, N, C = random_tori(n, seed=seed)
	tdf = pat.TorusDistanceField(pat.as_wide(P, 3), pat.as_wide(N, 3))
	tdf.fit_tori_from_forms(pat.as_wide(C, 6))
	q = pat.as_wide(np.random.default_rng(seed + 1).normal(size=(5000, 3)), 3)
	return tdf, q


@pytest.mark.parametrize('dtype, tol', [(torch.float64, 1e-9), (torch.float32, 1e-4)])
def test_evaluator_matches_cpu(gpu, dtype, tol):
	"""Distance, gradient, Laplacian in k-NN, all-point and radius modes, with outliers; tori fitted on device."""
	results = {}
	for device in ('cpu', gpu):
		on(device, dtype)
		tdf, q = evaluator()
		out = {'tori': tdf.get_tori()}
		for mode in ('knn32', 'all', 'radius', 'outliers'):
			tdf.set_k_evaluation(32 if mode != 'all' else -1)
			if mode == 'radius':
				tdf.set_radius_evaluation(0.1)
			if mode == 'outliers':
				tdf.set_radius_evaluation(-1)
				tdf.set_outliers(np.arange(0, 3000, 7))
			out[mode] = tdf.evaluate_distance_gradient_laplacian(q)
		results[str(device)] = out
	cpu, dev = results['cpu'], results[str(gpu)]
	print(f'\n{gpu} vs cpu, {dtype}:')
	for name in ('centers', 'axes', 'R', 'r'):
		i = ('centers', 'axes', 'R', 'r').index(name)
		assert report(f'tori {name}', rel_diff(dev['tori'][i], cpu['tori'][i])) <= tol
	for mode in ('knn32', 'all', 'radius', 'outliers'):
		for label, a, b in zip(('distance', 'gradient', 'laplacian'), dev[mode], cpu[mode]):
			assert report(f'{mode} {label}', rel_diff(a, b)) <= tol


def test_winding_numbers_match_cpu(gpu):
	"""Winding number (exact sum) and fast winding number (octree + expansions), float64."""
	results = {}
	for device in ('cpu', gpu):
		on(device, torch.float64)
		tdf, q = evaluator(n=1500)
		tdf.fast_winding_number_precompute()
		results[str(device)] = (tdf.winding_number(q), tdf.fast_winding_number(q))
	print(f'\n{gpu} vs cpu:')
	for label, a, b in zip(('winding number', 'fast winding number'), results[str(gpu)], results['cpu']):
		assert report(label, rel_diff(a, b)) <= 1e-9


def test_network_matches_cpu(gpu):
	"""Pre-trained network on tooth.ply neighborhoods (float32 network): coefficients CPU vs GPU."""
	from points_as_tori.network import FundamentalFormPredictor

	points, normals = read_ply('tooth.ply')
	results = {}
	for device in ('cpu', gpu):
		on(device, torch.float32)
		P, N = pat.as_wide(points, 3), pat.as_wide(normals, 3)
		model, k = FundamentalFormPredictor.load_saved_model(TORCH_MODEL)
		model.eval()
		nb = pat.get_neighbors(P, k)
		with torch.no_grad():
			results[str(device)] = (nb, model(P[:, nb], N[:, nb]))
	(nb_d, c_d), (nb_c, c_c) = results[str(gpu)], results['cpu']
	assert torch.equal(nb_d.cpu(), nb_c)
	print(f'\n{gpu} vs cpu:')
	assert report('network coefficients (relative)', rel_diff(c_d, c_c)) <= 1e-3


def test_inference_end_to_end(gpu):
	"""PointsAsTori on tooth.ply (float32): tori and SDF on 20k queries in the bounding box, CPU vs GPU."""
	points, normals = read_ply('tooth.ply')
	lo, hi = points.min(axis=0), points.max(axis=0)
	diag = float(np.linalg.norm(hi - lo))
	queries = np.random.default_rng(0).uniform(lo - 0.1 * diag, hi + 0.1 * diag, size=(20000, 3))
	results = {}
	for device in ('cpu', gpu):
		on(device, torch.float32)
		sdf = pat.PointsAsTori(points, normals, model_path=TORCH_MODEL)
		q = pat.as_wide(queries, 3)
		d, g = sdf.signed_distance_and_gradient(q)
		results[str(device)] = (sdf.get_tori(), d.cpu(), g.cpu())
	(t_d, d_d, g_d), (t_c, d_c, g_c) = results[str(gpu)], results['cpu']
	err = (d_d - d_c).abs() / diag
	print(f'\n{gpu} vs cpu (distances relative to the bounding-box diagonal):')
	report('tori centers (relative)', rel_diff(t_d[0], t_c[0]))
	report('median |Δφ|', float(err.median()))
	report('99th percentile |Δφ|', float(err.quantile(0.99)))
	report('max |Δφ|', float(err.max()))
	sign = float((torch.sign(d_d) == torch.sign(d_c)).double().mean())
	report('sign agreement', sign)
	assert sign >= 0.999
	assert float(err.quantile(0.99)) <= 1e-3


def test_train_step_matches_cpu(gpu):
	"""One training step of a small network from the same initial weights and inputs: loss, gradients, weights."""
	from points_as_tori.training import mesh_sampling as pms
	from points_as_tori.training import train as ptrain

	tiny = dict(embed_dim=16, n_freq=0, n_layers=1, n_heads=2, mlp_dim=32)
	on('cpu', torch.float32)  # inputs drawn on the CPU (device generators differ), then copied
	data = synthetic_dataset(n_shapes=1, n_points=60, k=16, pad_points=0, seed=5)
	keys = ('positions', 'normals', 'neighbors', 'masks', 'mesh_vertices', 'mesh_faces', 'face_masks')
	s = ptrain.upload_shape(tuple(data[k] for k in keys), 0)
	q, d = pms.generate_shape_features(
		s['positions'], s['neighbors'], s['mesh_vertices'], s['mesh_faces'], s['face_mask'],
		ptrain.shape_generator(1, 2, 3), n_queries=30,
	)  # fmt: skip
	results = {}
	for device in ('cpu', gpu):
		on(device, torch.float32)
		model, optimizer, train_step = ptrain.FundamentalFormPredictorTrainer(
			**tiny, learning_rate=1e-3, warmup_steps=0, seed=0
		)
		sd = {k: v.to(device) for k, v in s.items()}
		args = (sd['mask'], sd['mask'][:32], sd['positions'], sd['normals'], sd['neighbors'])
		loss, _ = train_step(model, optimizer, *args, q[:, :32].to(device), d[:32].to(device))
		grads = torch.cat([p.grad.flatten() for p in model.parameters()])
		weights = torch.cat([p.detach().flatten() for p in model.parameters()])
		results[str(device)] = (loss, grads, weights)
	(l_d, g_d, w_d), (l_c, g_c, w_c) = results[str(gpu)], results['cpu']
	print(f'\n{gpu} vs cpu:')
	assert report('loss (relative)', rel_diff(l_d, l_c)) <= 1e-4
	assert report('gradients (relative to max)', rel_diff(g_d, g_c)) <= 1e-3
	assert report('weights after the step (relative to max)', rel_diff(w_d, w_c)) <= 1e-5


def test_marching_cubes_matches_cpu(gpu):
	"""Same grid values on both devices: same triangles, vertices within rounding."""
	from points_as_tori.isosurface import marching_cubes

	on('cpu', torch.float64)
	x = torch.linspace(-1.2, 1.2, 41, dtype=torch.float64)
	X, Y, Z = torch.meshgrid(x, x, x, indexing='ij')
	values = torch.sqrt(X**2 + Y**2 + Z**2) - 1.0 + 0.05 * torch.sin(5 * X) * torch.cos(4 * Y)
	lo, hi = torch.full((3,), -1.2, dtype=torch.float64), torch.full((3,), 1.2, dtype=torch.float64)
	v_c, f_c = marching_cubes(values, lo, hi)
	on(gpu, torch.float64)
	v_d, f_d = marching_cubes(values.to(gpu), lo.to(gpu), hi.to(gpu))
	assert torch.equal(f_d.cpu(), f_c)
	assert report(f'\n{gpu} vs cpu marching cubes vertices (absolute)', float((v_d.cpu() - v_c).abs().max())) <= 1e-12


def test_bvh_queries_match_cpu(gpu):
	"""Torch BVH on both devices: k-NN sets, closest point on a mesh, ray hits (float64)."""
	from points_as_tori.bvh import BVH

	V, F = torus_mesh()
	rng = np.random.default_rng(3)
	P = rng.normal(size=(5000, 3))
	Q = rng.normal(size=(2000, 3))
	ro, rd = rng.normal(size=(500, 3)) * 2, rng.normal(size=(500, 3))
	results = {}
	for device in ('cpu', gpu):
		on(device, torch.float64)
		pts = BVH(pat.as_wide(P, 3))
		idx, dist = pts.find_k_nearest(pat.as_wide(Q, 3), 16)
		mesh = BVH(pat.as_wide(V, 3), pat.as_wide(F, 3, dtype=torch.long))
		closest = mesh.closest_point_on_mesh(pat.as_wide(Q, 3))
		hits = mesh.intersect_ray(pat.as_wide(ro, 3), pat.as_wide(rd, 3), 0.0, float('inf'), 0.0)
		results[str(device)] = (idx.cpu(), dist.cpu(), [c.cpu() for c in closest], [h.cpu() for h in hits])
	(i_d, d_d, c_d, h_d), (i_c, d_c, c_c, h_c) = results[str(gpu)], results['cpu']
	assert torch.equal(i_d.sort(dim=1).values, i_c.sort(dim=1).values)
	assert rel_diff(d_d, d_c) <= 1e-12
	for a, b in zip(c_d, c_c):
		assert (a.dtype == b.dtype and torch.equal(a, b)) if not a.is_floating_point() else rel_diff(a, b) <= 1e-12
	for a, b in zip(h_d, h_c):
		if a.is_floating_point():
			finite = torch.isfinite(b)
			assert torch.equal(torch.isfinite(a), finite) and rel_diff(a[finite], b[finite]) <= 1e-12
		else:
			assert torch.equal(a, b)


def test_inference_has_no_host_sync(gpu):
	"""Evaluation and the network forward on CUDA raise under torch.cuda.set_sync_debug_mode('error') if any
	operation waits for the device (brute-force k-NN; the BVH paths have documented host reads)."""
	from points_as_tori.network import FundamentalFormPredictor

	if gpu.type != 'cuda':
		pytest.skip('needs a CUDA device')
	config.KNN_BACKEND = 'brute'
	on(gpu, torch.float32)
	points, normals = read_ply('tooth.ply')
	P, N = pat.as_wide(points, 3), pat.as_wide(normals, 3)
	model, k = FundamentalFormPredictor.load_saved_model(TORCH_MODEL)
	model.eval()
	sdf = pat.PointsAsTori(points, normals, model_path=TORCH_MODEL)
	q = torch.rand(3, 50000, device=gpu) * 2 - 1
	torch.cuda.synchronize(gpu)
	torch.cuda.set_sync_debug_mode('error')
	try:
		nb = pat.get_neighbors(P, k)
		with torch.no_grad():
			coefficients = model(P[:, nb], N[:, nb])
		d, g = sdf.signed_distance_and_gradient(q)
		dist, grad, lap = sdf._tdf.evaluate_distance_gradient_laplacian(q)
	finally:
		torch.cuda.set_sync_debug_mode(0)
	assert coefficients.shape[1] == len(points) and d.shape == (50000,) and lap.shape == (50000,)
