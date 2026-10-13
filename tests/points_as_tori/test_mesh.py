"""
TriangleMesh (points_as_tori.shape_3d) vs the original C++ Mesh3D (pat_reference) and its Python wrapper.

Run:  PAT_REFERENCE_DIR=build/pat_reference_full python -m pytest tests/points_as_tori/test_mesh.py -q -s

Notes on what "the same" means here (measured values are printed):
  * FCPW (closest points, ray hits) computes in float32. With query_dtype=torch.float32 the port computes those in
    float32 with FCPW's operation order and matches bit for bit; with the default (float64) the port is exact and
    the C++ differs by float32 rounding.
  * The samplers replay the C++ random streams; with the same seed they return the same points and normals, bit for
    bit (the ray-casting ones with query_dtype=torch.float32). The meshes are built from the C++ center_and_scale
    output: the port's center_and_scale agrees to ~3e-16 only (Eigen's summation order of the centroid depends on
    the run-time alignment of its result).
  * The C++ samplers draw from a shared engine inside OpenMP loops in two places (sample_point_cloud's noise,
    sample_points_and_normals_uniformly_by_casting_rays' rays); the reference calls run with one thread.
"""

import contextlib
import os

import numpy as np
import pytest
import torch

from points_as_tori import config
from points_as_tori.shape_3d import TriangleMesh, sample_bounding_box
from helpers import PAT_DATA, torus_mesh

MESH_NAMES = ['knot', 'block', 'femur', 'torus']


def _np(t):
	return t.detach().cpu().numpy()


def _f(x):
	return np.asfortranarray(np.asarray(x, dtype=np.float64))


@contextlib.contextmanager
def _one_thread():
	"""The C++ and torch share libgomp: one OpenMP thread makes the C++ samplers deterministic."""
	n = torch.get_num_threads()
	torch.set_num_threads(1)
	try:
		yield
	finally:
		torch.set_num_threads(n)


def _mesh_arrays(name):
	"""(V, 3) float64, (F, 3) int arrays."""
	if name == 'torus':
		return torus_mesh()
	vertices, faces = [], []
	with open(os.path.join(PAT_DATA, f'{name}.obj')) as f:
		for line in f:
			parts = line.split()
			if parts and parts[0] == 'v':
				vertices.append([float(x) for x in parts[1:4]])
			elif parts and parts[0] == 'f':
				faces.append([int(p.split('/')[0]) - 1 for p in parts[1:]])
	return np.array(vertices), np.array(faces)


def _pair(reference, name, query_dtype=None):
	"""(C++ Mesh3D after center_and_scale, port mesh with the same vertices)."""
	V, F = _mesh_arrays(name)
	cpp = reference.Mesh3D(_f(V.T), np.asfortranarray(F.T.astype(np.int32))).center_and_scale()
	return cpp, TriangleMesh(cpp.get_vertices().T, F, query_dtype=query_dtype)


def _queries(n, seed=0, extent=1.2):
	return np.random.default_rng(seed).uniform(-extent, extent, size=(3, n))


def _compare_samples(label, port, cpp):
	p, n = _np(port[0]), _np(port[1])
	assert p.shape == cpp[0].shape, f'{label}: {p.shape} vs {cpp[0].shape}'
	dp = np.abs(p - cpp[0]).max() if p.size else 0.0
	dn = np.abs(n - cpp[1]).max() if n.size else 0.0
	print(
		f'  {label:42s} n={p.shape[1]:5d}  points max|Δ|={dp:.2e} (identical: {np.array_equal(p, cpp[0])})  normals max|Δ|={dn:.2e}'
	)
	assert np.array_equal(p, cpp[0]) and np.array_equal(n, cpp[1]), label


# ========================================================================
# Geometry, I/O
# ========================================================================


@pytest.mark.parametrize('name', MESH_NAMES)
def test_center_and_scale_and_bounding_box(reference, cpu64, name):
	V, F = _mesh_arrays(name)
	cpp = reference.Mesh3D(_f(V.T), np.asfortranarray(F.T.astype(np.int32)))
	mesh = TriangleMesh(V, F)
	bmin, bmax = mesh.bounding_box()
	assert np.array_equal(_np(bmin), cpp.bounding_box()[0]) and np.array_equal(_np(bmax), cpp.bounding_box()[1])
	cpp = cpp.center_and_scale()
	mesh.center_and_scale()
	d = np.abs(_np(mesh.get_vertices()) - cpp.get_vertices()).max()
	print(
		f'\n  {name}: center_and_scale max|Δ| = {d:.2e} (identical: {np.array_equal(_np(mesh.vertices), cpp.get_vertices())})'
	)
	assert d <= 2e-15
	assert np.array_equal(_np(mesh.get_faces()), cpp.get_faces())
	bmin, bmax = mesh.bounding_box()
	assert np.abs(_np(bmin) - cpp.bounding_box()[0]).max() <= 2e-15
	assert np.abs(_np(bmax) - cpp.bounding_box()[1]).max() <= 2e-15


def test_wrapper_io_and_vertex_normals(original, cpu64, tmp_path):
	"""read_OBJ / write_OBJ / vertex_normals vs the original Python wrapper (backed by the C++)."""
	path = os.path.join(PAT_DATA, 'knot.obj')
	o = original.shape_3d.TriangleMesh.read_OBJ(path)
	p = TriangleMesh.read_OBJ(path)
	assert np.array_equal(_np(p.vertices), o.vertices.T)
	d = np.abs(_np(p.vertex_normals) - o.vertex_normals.T).max()
	print(f'\n  vertex normals max|Δ| = {d:.2e}')
	assert d <= 1e-15
	o.center_and_scale()
	p.center_and_scale()
	o.write_OBJ(str(tmp_path / 'o.obj'))
	TriangleMesh(o.vertices, o.faces).write_OBJ(str(tmp_path / 'p.obj'))
	assert (tmp_path / 'o.obj').read_text() == (tmp_path / 'p.obj').read_text()
	q = TriangleMesh.read_OBJ(str(tmp_path / 'p.obj'))
	assert np.array_equal(_np(q.vertices), o.vertices.T) and np.array_equal(_np(q.faces), o.faces.T)
	p.export_PLY(str(tmp_path / 'p.ply'))
	from plyfile import PlyData

	ply = PlyData.read(str(tmp_path / 'p.ply'))
	assert len(ply['vertex']) == p.vertices.shape[1] and len(ply['face']) == p.faces.shape[1]


def test_remove_degenerate_faces(reference, cpu64):
	"""
	The filter itself (repeated index or area < 1e-10) on a torus with three bad faces added. Its result is unused
	by the C++ (and the C++ call reads the 3 x n matrices as n x 3, so it is not observable there); what is checked
	against the C++ is that the queries use all faces, degenerate ones included.
	"""
	V, F = torus_mesh()
	bad = np.array([[0, 0, 5], [3, 7, 3], [1, 2, len(V)]])  # repeated indices; a zero-area sliver (new vertex)
	V2 = np.concatenate([V, [(V[1] + V[2]) / 2]])
	F2 = np.concatenate([F, bad])
	mesh = TriangleMesh(V2, F2)
	assert mesh.cleaned_faces.shape[1] == len(F)
	assert np.array_equal(_np(mesh.cleaned_faces), F.T)
	assert torch.equal(mesh.cleaned_vertices, mesh.vertices)
	cpp = reference.Mesh3D(_f(V2.T), np.asfortranarray(F2.T.astype(np.int32)))
	q = _queries(300)
	d = np.abs(_np(mesh.evaluate_gwn(torch.tensor(q))) - cpp.evaluate_gwn(_f(q))).max()
	assert d <= 1e-12


# ========================================================================
# Queries
# ========================================================================


@pytest.mark.parametrize('name', MESH_NAMES)
def test_winding_numbers(reference, cpu64, name):
	cpp, mesh = _pair(reference, name)
	q = _queries(400 if name != 'femur' else 150, seed=1)
	gwn = _np(mesh.evaluate_gwn(torch.tensor(q)))
	gwn_ref = cpp.evaluate_gwn(_f(q))
	d = np.abs(gwn - gwn_ref).max() / (4 * np.pi)
	fast = _np(mesh.evaluate_fast_gwn(torch.tensor(q)))
	fast_ref = cpp.evaluate_fast_gwn(_f(q))
	d_fast = np.abs(fast - fast_ref).max()
	print(
		f'\n  {name}: evaluate_gwn max|Δ|/4π = {d:.2e};  evaluate_fast_gwn (exact) vs igl::fast_winding_number '
		f'(Barnes-Hut) max|Δ| = {d_fast:.2e}, inside/outside agree {np.mean((fast > 0.5) == (fast_ref > 0.5)):.4f}'
	)
	assert d <= 1e-12
	assert np.array_equal(fast, gwn / (4 * np.pi))
	assert d_fast <= 1e-2


@pytest.mark.parametrize('name', MESH_NAMES)
def test_signed_distance_and_gradient(reference, cpu64, name):
	cpp, mesh = _pair(reference, name)
	_, mesh32 = _pair(reference, name, query_dtype=torch.float32)
	n = 600 if name in ('knot', 'torus') else 150
	q = _queries(n, seed=2)
	qt = torch.tensor(q)
	d_ref = cpp.evaluate_signed_distance(_f(q), True)
	d_ref_seq = cpp.evaluate_signed_distance(_f(q), False)
	assert np.array_equal(d_ref, d_ref_seq)
	d32 = _np(mesh32.evaluate_signed_distance(qt))
	d64 = _np(mesh.evaluate_signed_distance(qt))
	dr, gr = cpp.evaluate_signed_distance_and_gradient(_f(q))
	d32b, g32 = (_np(x) for x in mesh32.evaluate_signed_distance_and_gradient(qt))
	d64b, g64 = (_np(x) for x in mesh.evaluate_signed_distance_and_gradient(qt))
	same_g = np.mean(np.all(g32 == gr, axis=0))
	single = [cpp.evaluate_signed_distance_single(q[:, i]) for i in range(20)]
	single32 = [float(mesh32.evaluate_signed_distance_single(qt[:, i])) for i in range(20)]
	print(
		f'\n  {name}: signed distance, float32 queries: max|Δ| = {np.abs(d32 - d_ref).max():.2e} '
		f'(identical: {np.array_equal(d32, d_ref)}); float64 queries: max|Δ| = {np.abs(d64 - d_ref).max():.2e}, '
		f'signs agree {np.mean(np.sign(d64) == np.sign(d_ref)):.4f}\n'
		f'      gradient, float32 queries: identical for {same_g:.4f} of the queries, max|Δ| = {np.abs(g32 - gr).max():.2e}; '
		f'float64: max|Δ| = {np.abs(g64 - gr).max():.2e}'
	)
	assert np.array_equal(d32, d_ref) and np.array_equal(d32b, dr)
	assert np.abs(d64 - d_ref).max() <= 3e-7 and np.array_equal(np.sign(d64), np.sign(d_ref))
	assert np.array_equal(np.array(single32), np.array(single))
	# gradients differ only where two faces (sharing an edge) are at exactly the same float32 distance: FCPW reports
	# the last one in its BVH order, the port the first, and their closest points differ by float32 rounding, which
	# the normalization of q - p amplifies by 1/|q - p|
	far = np.abs(dr) > 1e-2
	assert np.abs(g32 - gr)[:, far].max() <= 1e-5
	# float64 queries: the C++ closest points carry float32 rounding (of the query, the vertices, the arithmetic), and
	# near the medial axis (two far-apart closest points at nearly equal distance) the gradient is discontinuous
	assert np.quantile(np.abs(g64 - gr)[:, far].max(axis=0), 0.99) <= 1e-5


@pytest.mark.parametrize('name', ['knot', 'block', 'torus'])
def test_ray_intersections(reference, cpu64, name):
	cpp, mesh = _pair(reference, name)
	_, mesh32 = _pair(reference, name, query_dtype=torch.float32)
	rng = np.random.default_rng(3)
	ro = rng.normal(size=(3, 1500))
	ro = 2.5 * ro / np.linalg.norm(ro, axis=0)
	rd = -ro + rng.normal(scale=0.3, size=ro.shape)
	rd /= np.linalg.norm(rd, axis=0)
	p_ref, n_ref = cpp.ray_intersections(_f(ro), _f(rd))
	p32, n32 = (_np(x) for x in mesh32.ray_intersections(torch.tensor(ro), torch.tensor(rd)))
	p64, n64 = (_np(x) for x in mesh.ray_intersections(torch.tensor(ro), torch.tensor(rd)))
	print(
		f'\n  {name}: {p_ref.shape[1]} hits; float32 queries: points/normals identical {np.array_equal(p32, p_ref)}/'
		f'{np.array_equal(n32, n_ref)}; float64: {p64.shape[1]} hits, points max|Δ| = {np.abs(p64 - p_ref).max():.2e}, '
		f'normals max|Δ| = {np.abs(n64 - n_ref).max():.2e}'
	)
	assert np.array_equal(p32, p_ref) and np.array_equal(n32, n_ref)
	assert p64.shape == p_ref.shape
	assert np.abs(p64 - p_ref).max() <= 1e-5 and np.abs(n64 - n_ref).max() <= 1e-5


def test_queries_without_host_sync(cpu64):
	"""The evaluation paths run on the meta device (no data-dependent host operation)."""
	V, F = torus_mesh()
	config.set_device('meta', torch.float64, force=True)
	try:
		for query_dtype in (None, torch.float32):
			mesh = TriangleMesh(V, F, query_dtype=query_dtype)
			mesh.center_and_scale()
			q = torch.empty((3, 50), dtype=torch.float64, device='meta')
			assert mesh.evaluate_signed_distance(q).shape == (50,)
			d, g = mesh.evaluate_signed_distance_and_gradient(q)
			assert d.shape == (50,) and g.shape == (3, 50)
			assert mesh.evaluate_gwn(q).shape == (50,) and mesh.evaluate_fast_gwn(q).shape == (50,)
			assert mesh.evaluate_signed_distance_single(q[:, 0]).shape == ()
	finally:
		config.set_device('cpu', torch.float64, force=True)


# ========================================================================
# Samplers: same seed, same samples as the C++
# ========================================================================


@pytest.mark.parametrize('name', ['knot', 'torus'])
def test_surface_samplers(reference, cpu64, name):
	cpp, mesh = _pair(reference, name)
	one, minus = np.ones(3), -np.ones(3)
	bmin, bmax = np.array([-0.3, -0.5, -0.2]), np.array([0.4, 0.3, 0.5])
	print()
	with _one_thread():
		_compare_samples(
			'sample_points_and_normals_uniformly',
			mesh.sample_points_and_normals_uniformly(1000, 3),
			cpp.sample_points_and_normals_uniformly(1000, 3, one, minus, 0.0, 0.0),
		)
		_compare_samples(
			'  + noise',
			mesh.sample_points_and_normals_uniformly(1001, 4, sigma_p=0.01, sigma_n=0.02),
			cpp.sample_points_and_normals_uniformly(1001, 4, one, minus, 0.01, 0.02),
		)
		_compare_samples(
			'  + bounding box (original BVH leaves)',
			mesh.sample_points_and_normals_uniformly(500, 5, bmin, bmax, 0.01, 0.0),
			cpp.sample_points_and_normals_uniformly(500, 5, bmin, bmax, 0.01, 0.0),
		)
		p = _np(mesh.sample_uniformly(300, 7, bmin, bmax))
		assert np.array_equal(p, cpp.sample_uniformly(300, 7, bmin, bmax))
		p = _np(mesh.sample_narrow_band(300, 0.05, 8, bmin, bmax))
		assert np.array_equal(p, cpp.sample_narrow_band(300, 0.05, 8, bmin, bmax))
		p = _np(mesh.sample_narrow_band(300, 0.1, 9))
		assert np.array_equal(p, cpp.sample_narrow_band(300, 0.1, 9, one, minus))
		print('  sample_uniformly, sample_narrow_band (with / without box): identical')
		_compare_samples(
			'sample_uniform_point_cloud',
			mesh.sample_uniform_point_cloud(800, 0.02, 0.2, 11),
			cpp.sample_uniform_point_cloud(800, 0.02, 0.2, 11),
		)
		_compare_samples(
			'sample_farthest_point_cloud',
			mesh.sample_farthest_point_cloud(400, 0.02, 0.2, 12),
			cpp.sample_farthest_point_cloud(400, 0.02, 0.2, 12),
		)


@pytest.mark.parametrize('name', ['knot', 'torus'])
def test_ray_casting_samplers(reference, cpu64, name):
	"""Exact replay with FCPW's float32 queries; with float64 queries, the differences are printed."""
	cpp, mesh64 = _pair(reference, name)
	_, mesh = _pair(reference, name, query_dtype=torch.float32)
	print()
	with _one_thread():
		calls = [
			('sample_point_cloud', lambda m: m.sample_point_cloud(4, 1.0, 0.05, 13, 0.01, 0.02)),
			('sample_point_cloud (no noise)', lambda m: m.sample_point_cloud(3, 0.8, 0.04, 14, 0.0, 0.0)),
			# seed 14: farthest-point subsampling branch, seed 15: random subset branch
			(
				'sample_even_raycasted_point_cloud (fps)',
				lambda m: m.sample_even_raycasted_point_cloud(3, 700, 1.0, 0.03, 0.08, 14),
			),
			(
				'sample_even_raycasted_point_cloud (subset)',
				lambda m: m.sample_even_raycasted_point_cloud(3, 700, 1.0, 0.03, 0.08, 15),
			),
			(
				'sample_uneven_point_cloud',
				lambda m: m.sample_uneven_point_cloud(4, 1.0, 0.03, 0.08, 0.02, 0.2, 900, 18),
			),
		]
		for label, call in calls:
			ref = call(cpp)
			_compare_samples(label, call(mesh), ref)
			p, n = (_np(x) for x in call(mesh64))
			if p.shape == ref[0].shape:
				print(
					f'      float64 queries: points max|Δ| = {np.abs(p - ref[0]).max():.2e}, normals {np.abs(n - ref[1]).max():.2e}'
				)
			else:
				print(f'      float64 queries: {p.shape[1]} vs {ref[0].shape[1]} points')
	# the C++ sample_point_cloud adds the perturbed unit normal to the normal: |n| = 2 without noise
	_, n = mesh.sample_point_cloud(3, 0.8, 0.04, 14)
	assert torch.allclose(n.norm(dim=0), torch.full_like(n[0], 2.0))


def test_sample_by_casting_rays(cpu64):
	"""C++-only sampler (not bound in the original module): samples lie on the mesh, in the box, deterministic."""
	V, F = torus_mesh()
	mesh = TriangleMesh(V, F)
	bmin, bmax = np.array([-0.5, -0.5, -0.3]), np.array([0.6, 0.2, 0.3])
	p, n = mesh.sample_points_and_normals_uniformly_by_casting_rays(500, 3, bmin, bmax)
	assert p.shape == (3, 500) and n.shape == (3, 500)
	lo, hi = torch.tensor(bmin)[:, None], torch.tensor(bmax)[:, None]
	assert bool(((p >= lo) & (p <= hi)).all())
	assert float(mesh.evaluate_signed_distance(p).abs().max()) <= 1e-12
	assert float((n.norm(dim=0) - 1).abs().max()) <= 1e-12
	p2, _ = mesh.sample_points_and_normals_uniformly_by_casting_rays(500, 3, bmin, bmax)
	assert torch.equal(p, p2)
	p3, n3 = mesh.sample_points_and_normals_uniformly_by_casting_rays(200, 4, bmin, bmax, 0.01, 0.1)
	assert float((n3.norm(dim=0) - 1).abs().max()) <= 1e-12


def test_sample_bounding_box(cpu64):
	"""Declared but not defined in the original: uniform samples in the box, reproducible."""
	p = sample_bounding_box(20000, [-1.0, 0.0, 2.0], [1.0, 0.5, 3.0], 5)
	assert p.shape == (3, 20000)
	assert torch.equal(p, sample_bounding_box(20000, np.array([-1.0, 0.0, 2.0]), (1.0, 0.5, 3.0), 5))
	lo, hi = torch.tensor([-1.0, 0.0, 2.0])[:, None], torch.tensor([1.0, 0.5, 3.0])[:, None]
	assert bool(((p >= lo) & (p < hi)).all())
	u = (p - lo) / (hi - lo)
	assert float((u.mean(dim=1) - 0.5).abs().max()) < 0.01
