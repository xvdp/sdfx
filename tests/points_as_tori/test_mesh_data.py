"""
Dataset generation (points_as_tori.training.mesh_data) vs the original (training/mesh_data.py) and its dependencies.

Run:  PAT_REFERENCE_DIR=build/pat_reference_full python -m pytest tests/points_as_tori/test_mesh_data.py -q -s

Comparison-only packages (skipped when missing): noise (Perlin noise), trimesh + scipy (mesh checks), pyvista
(the original's grid / contouring), pyfqmr (simplification, needed by the blob generator itself), and the original
package + the C++ reference build (end-to-end comparison of the training files).
"""

import contextlib
import importlib.util
import os
import sys
import types

import numpy as np
import pytest
import torch

from points_as_tori import config
from points_as_tori.shape_3d import TriangleMesh
from points_as_tori.training import mesh_data as md
from points_as_tori.training import train
from helpers import PAT_DATA, REPO, torus_mesh

TRAINING_KEYS = ['positions', 'normals', 'neighbors', 'masks', 'mesh_vertices', 'mesh_faces', 'face_masks']
TRAINING_DTYPES = {
	'positions': np.float64,
	'normals': np.float64,
	'neighbors': np.int32,
	'masks': np.bool_,
	'mesh_vertices': np.float64,
	'mesh_faces': np.int32,
	'face_masks': np.bool_,
}


@contextlib.contextmanager
def _one_thread():
	n = torch.get_num_threads()
	torch.set_num_threads(1)
	try:
		yield
	finally:
		torch.set_num_threads(n)


def _read_obj(name):
	"""(V, 3) float64, (F, 3) int64 arrays of a data mesh."""
	vertices, faces = [], []
	with open(os.path.join(PAT_DATA, f'{name}.obj')) as f:
		for line in f:
			parts = line.split()
			if parts and parts[0] == 'v':
				vertices.append([float(x) for x in parts[1:4]])
			elif parts and parts[0] == 'f':
				faces.append([int(p.split('/')[0]) - 1 for p in parts[1:]])
	return np.array(vertices), np.array(faces)


def _wide(vertices, faces):
	return config.as_wide(vertices, 3), config.as_wide(faces, 3, dtype=torch.long)


def _check_training_file(path, n_shapes, max_vertices=md.max_vertices_per_mesh, max_faces=md.max_faces_per_mesh):
	data = np.load(path)
	assert sorted(data.files) == sorted(TRAINING_KEYS)
	for key in TRAINING_KEYS:
		assert data[key].dtype == TRAINING_DTYPES[key], key
	max_points = data['positions'].shape[1]
	assert data['positions'].shape == (n_shapes, max_points, 3)
	assert data['normals'].shape == (n_shapes, max_points, 3)
	assert data['neighbors'].shape == (n_shapes, max_points, min(md.k_neighbors, max_points))
	assert data['masks'].shape == (n_shapes, max_points)
	assert data['mesh_vertices'].shape == (n_shapes, max_vertices, 3)
	assert data['mesh_faces'].shape == (n_shapes, max_faces, 3)
	assert data['face_masks'].shape == (n_shapes, max_faces)
	loaded = train.load_mesh_training_data(path)  # the training port reads it
	for key, array in zip(TRAINING_KEYS, loaded):
		assert np.array_equal(array, data[key])
	return data


# ========================================================================
# Perlin noise
# ========================================================================


def test_pnoise3_matches_noise_package(cpu64):
	noise = pytest.importorskip('noise')
	rng = np.random.default_rng(0)
	points = rng.uniform(-6.0, 6.0, size=(3, 3000))
	points[:, :5] = [[0.0, -1.0, 2.5, 1023.7, -0.5], [0.0, 1.0, -2.5, 3.0, 1e-7], [0.0, 0.5, 7.0, -3.0, 2.0]]
	pt = torch.tensor(points)
	worst = 0.0
	# base > 0 reads past the end of the C permutation table (see mesh_data._PERM_EXTENDED)
	for octaves, persistence, lacunarity, base in [
		(1, 0.5, 2.0, 0),
		(4, 0.5, 2.0, 0),
		(3, 1.0, 2.0, 42),
		(4, 0.5, 2.0, 99),
		(2, 0.7, 1.9, 255),
		(1, 0.5, 2.0, 200),
	]:
		ref = np.array(
			[
				noise.pnoise3(*p, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=base)
				for p in points.T
			]
		)
		port = md.pnoise3(pt[0], pt[1], pt[2], octaves, persistence, lacunarity, base=base).numpy()
		worst = max(worst, np.abs(port - ref).max())
		assert np.array_equal(port, ref), (octaves, persistence, lacunarity, base)
	print(f'\n  pnoise3 vs noise.pnoise3 (6 parameter sets x 3000 points): max|Δ| = {worst:.1e}')
	with pytest.raises(ValueError):
		md.pnoise3(pt[0], pt[1], pt[2], octaves=0)
	# evaluate_perlin_noise_at_points: base = seed % 256, values times 2
	values = md.evaluate_perlin_noise_at_points(pt[:, :50], scale=1.7, octaves=3, persistence=0.5, seed=300).numpy()
	ref = [
		2.0 * noise.pnoise3(*(p * 1.7), octaves=3, persistence=0.5, lacunarity=2.0, base=300 % 256)
		for p in points.T[:50]
	]
	assert np.array_equal(values, np.array(ref))


# ========================================================================
# trimesh checks
# ========================================================================


def _check_cases():
	V, F = torus_mesh()
	cases = []
	for name in ['knot', 'block']:
		cases.append((name, *_read_obj(name)))
	cases.append(('torus', V, F))
	cases.append(('torus inverted', V, F[:, [0, 2, 1]]))
	F_flip = F.copy()
	F_flip[5] = F_flip[5, [0, 2, 1]]
	cases.append(('torus one face flipped', V, F_flip))
	cases.append(('torus open', V, F[:-3]))
	V_dup = np.concatenate([V, V[:10]])
	F_dup = F.copy()
	half = len(F) // 2
	F_dup[half:][F_dup[half:] < 10] += len(V)  # half of the faces use duplicated vertices
	cases.append(('torus duplicated vertices', V_dup, F_dup))
	cases.append(('torus unreferenced vertex', np.concatenate([V, [[5.0, 5.0, 5.0]]]), F))
	cases.append(('torus face twice', V, np.concatenate([F, F[:2]])))
	cases.append(
		('two tori, one inverted', np.concatenate([V, V + 3.0]), np.concatenate([F, F[:, [0, 2, 1]] + len(V)]))
	)
	return cases


@pytest.mark.parametrize('case', _check_cases(), ids=lambda c: c[0])
def test_mesh_checks_match_trimesh(cpu64, case):
	trimesh = pytest.importorskip('trimesh')
	from scipy.spatial import KDTree

	name, V, F = case
	tm = trimesh.Trimesh(vertices=V, faces=F)
	checks = md.trimesh_checks(*_wide(V, F))
	print(
		f'\n  {name}: trimesh watertight/winding/volume = {tm.is_watertight}/{tm.is_winding_consistent}/'
		f'{tm.is_volume} ({tm.volume:.6f}); port {checks.is_watertight}/{checks.is_winding_consistent}/'
		f'{checks.is_volume} ({checks.volume:.6f})'
	)
	assert checks.is_watertight == tm.is_watertight
	assert checks.is_winding_consistent == tm.is_winding_consistent
	assert checks.is_volume == tm.is_volume
	assert abs(checks.volume - tm.volume) <= 1e-12 * max(1.0, abs(tm.volume))

	# the orientation test of the original (KD-tree + trimesh vertex_faces / face_normals)
	query = np.array([2, 2, 2])
	_, closest_vertex_id = KDTree(V).query(query)
	cp = V[closest_vertex_id]
	vertex_faces = tm.vertex_faces[closest_vertex_id]
	vertex_faces = vertex_faces[vertex_faces >= 0]
	n = tm.face_normals[vertex_faces].mean(axis=0)
	n = n / np.linalg.norm(n)
	assert md.orientation_dot(*_wide(V, F)) == pytest.approx(np.dot(query - cp, n), abs=1e-14)


def test_empty_mesh_checks(cpu64):
	checks = md.trimesh_checks(torch.zeros((3, 3), dtype=torch.float64), torch.zeros((3, 0), dtype=torch.long))
	assert not (checks.is_watertight or checks.is_winding_consistent or checks.is_volume)


# ========================================================================
# Padding, saved features
# ========================================================================


def test_padding_and_saved_features(cpu64, tmp_path):
	V, F = torus_mesh()
	vp, fp, mask = md.pad_mesh(V, F)
	assert vp.shape == (md.max_vertices_per_mesh, 3) and vp.dtype == np.float64
	assert fp.shape == (md.max_faces_per_mesh, 3) and fp.dtype == np.int32
	assert np.array_equal(vp[: len(V)], V) and not vp[len(V) :].any()
	assert np.array_equal(fp[: len(F)], F) and (fp[len(F) :] == -1).all()
	assert mask.sum() == len(F) and mask[: len(F)].all()

	mesh = TriangleMesh(V, F)
	shapes = []
	for n_points, seed in [(300, 0), (40, 1)]:  # the second has fewer points than k_neighbors
		points, normals = mesh.sample_uniform_point_cloud(n_points, 0.0, 0.0, seed)
		neighbors = md.get_neighbors(points, md.k_neighbors)
		shapes.append(
			{
				'positions': config.to_numpy(points),
				'normals': config.to_numpy(normals),
				'neighbors': md._neighbors_numpy(neighbors),
				'vertices': V,
				'faces': F,
			}
		)
	assert shapes[1]['neighbors'].shape == (40, 40)
	neighbors_40 = shapes[1]['neighbors'].copy()
	features = md.generate_input_features_with_meshes(shapes, max_points=300)  # pads shapes[1]['neighbors'] in place
	positions, normals, neighbors, masks, mesh_vertices, mesh_faces, face_masks = features
	assert positions.shape == (2, 300, 3) and neighbors.shape == (2, 300, 64)
	assert masks[0].all() and masks[1].sum() == 40
	assert (neighbors[1, 40:] == -1).all() and (neighbors[1, :40, 40:] == -1).all()
	assert np.array_equal(neighbors[1, :40, :40], neighbors_40)
	assert shapes[1]['neighbors'].shape == (40, 64)  # as the original
	assert not positions[1, 40:].any() and not normals[1, 40:].any()
	path = str(tmp_path / 'features.npz')
	md.save_mesh_training_data(path, *features)
	data = _check_training_file(path, 2)
	assert np.array_equal(data['positions'], positions)

	# mesh datasets: keys, dtypes, metadata round trip
	meshes = [{'vertices': V, 'faces': F.astype(np.int32), 'file_id': 'torus', 'metadata': {'a': 1}}]
	md.save_mesh_dataset(meshes, str(tmp_path / 'meshes.npz'), metadata={'dataset': 'test'})
	loaded, meta = md.load_mesh_dataset(str(tmp_path / 'meshes.npz'))
	assert meta == {'dataset': 'test'} and loaded[0]['metadata'] == {'a': 1} and loaded[0]['file_id'] == 'torus'
	assert loaded[0]['vertices'].dtype == np.float64 and loaded[0]['faces'].dtype == np.int32
	assert sorted(np.load(str(tmp_path / 'meshes.npz')).files) == sorted(
		['n_meshes', 'metadata', 'vertices_0', 'faces_0', 'file_id_0', 'metadata_0']
	)


def test_add_point_cloud_noise(cpu64):
	"""Added (undefined in the original): only flips with max_noise_level = 0, as mode -2 calls it."""
	V, F = torus_mesh()
	points, normals = TriangleMesh(V, F).sample_uniform_point_cloud(500, 0.0, 0.0, 3)
	p, n = md.add_point_cloud_noise(points, normals, max_noise_level=0.0, max_normals_to_flip=0.5, seed=4)
	assert torch.equal(p, points)
	dots = (n * normals).sum(dim=0)
	assert bool(((dots - 1).abs() < 1e-12).logical_or((dots + 1).abs() < 1e-12).all()) and bool((dots < 0).any())
	p, n = md.add_point_cloud_noise(points, normals, max_noise_level=0.05, max_normals_to_flip=0.0, seed=5)
	assert float((p - points).abs().max()) > 0 and float((n.norm(dim=0) - 1).abs().max()) < 1e-12


# ========================================================================
# Blobs
# ========================================================================


@pytest.fixture(scope='module')
def original_mesh_data(original):
	"""training/mesh_data.py of the original, with the downloader / archive packages stubbed (not used here)."""
	for name in ['noise', 'pyvista', 'pyfqmr', 'trimesh']:
		pytest.importorskip(name)
	for name in ['thingi10k', 'py7zr', 'stl']:
		if name not in sys.modules and importlib.util.find_spec(name) is None:
			sys.modules[name] = types.ModuleType(name)
	if not hasattr(sys.modules['stl'], 'mesh'):
		sys.modules['stl'].mesh = types.ModuleType('stl.mesh')
	path = os.path.join(REPO, 'ext', 'points-as-tori', 'training', 'mesh_data.py')
	spec = importlib.util.spec_from_file_location('original_mesh_data', path)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def test_blob_implicit_surface_matches_original(cpu64, original_mesh_data):
	"""Grid and field values identical to the original's pv.ImageData; marching cubes vs PyVista's contour."""
	from scipy.spatial import cKDTree

	R = 24
	kw = dict(alpha=0.4, base_radius=1.0, grid_resolution=R, bbox_extent=1.68, scale=1.3, octaves=3, persistence=0.5)
	grid = original_mesh_data.generate_blob_implicit_surface(**kw, lacunarity=2.0, seed=45)
	values, lo, hi = md.generate_blob_implicit_surface(**kw, lacunarity=2.0, seed=45)
	ref_values = grid.point_data['values'].reshape(R, R, R, order='F')  # VTK point order: x fastest
	ref_points = grid.points.reshape(R, R, R, 3, order='F')
	assert np.array_equal(values.numpy(), ref_values)
	assert np.array_equal(lo.numpy(), ref_points[0, 0, 0]) and np.array_equal(hi.numpy(), ref_points[-1, -1, -1])

	contour = grid.contour(isosurfaces=[0.0], scalars='values')
	ref_v = np.asarray(contour.points)
	ref_f = contour.faces.reshape(-1, 4)[:, 1:]
	from points_as_tori.isosurface import marching_cubes

	v, f = marching_cubes(values, lo, hi)
	v = config.to_numpy(v)
	d = max(cKDTree(v).query(ref_v)[0].max(), cKDTree(ref_v).query(v)[0].max())
	checks = md.trimesh_checks(*_wide(v, config.to_numpy(f)))
	import trimesh

	ref_volume = trimesh.Trimesh(ref_v, ref_f[:, [0, 2, 1]]).volume
	print(
		f'\n  field values identical; marching cubes: {v.shape[0]} vertices / {f.shape[1]} faces vs PyVista '
		f'{ref_v.shape[0]} / {ref_f.shape[0]}, vertex sets within {d:.1e} (PyVista stores float32); '
		f'volume {checks.volume:.5f} vs {ref_volume:.5f} (triangulation of ambiguous cells differs)'
	)
	assert v.shape[0] == ref_v.shape[0] and f.shape[1] == ref_f.shape[0]
	assert d <= 1e-6
	assert checks.is_watertight and checks.is_volume
	assert abs(checks.volume - ref_volume) <= 1e-3 * ref_volume


def test_blob_dataset_end_to_end(cpu64, tmp_path, monkeypatch):
	"""A few low-resolution blobs -> mesh dataset -> scanned training files the training port reads."""
	pytest.importorskip('pyfqmr')
	monkeypatch.chdir(tmp_path)
	md.generate_blob_dataset(n_blobs=3, save_dir='datasets', grid_resolution=28, persistence=0.5, seed=42)
	meshes, meta = md.load_mesh_dataset('datasets/blobs_train_meshes.npz')
	test_meshes, _ = md.load_mesh_dataset('datasets/blobs_test_meshes.npz')
	assert len(meshes) + len(test_meshes) == 3 and meta['dataset'] == 'Procedural Blobs (Perlin)'
	for mesh in meshes:
		assert mesh['vertices'].dtype == np.float64 and mesh['faces'].dtype == np.int32
		assert len(mesh['faces']) <= md.max_faces_per_mesh
		assert np.abs(mesh['vertices']).max() == pytest.approx(1.0)  # center_and_scale
		checks = md.trimesh_checks(*_wide(mesh['vertices'], mesh['faces']))
		assert checks.is_watertight and checks.is_winding_consistent and checks.is_volume
		assert md.orientation_dot(*_wide(mesh['vertices'], mesh['faces'])) > 0

	monkeypatch.setattr(md, 'grid_spacing_min', 0.06)
	monkeypatch.setattr(md, 'grid_spacing_max', 0.1)
	for mode in [0, 3, -2]:
		md.generate_scanned_mesh_training_dataset(
			n_views=2,
			npz_filepath='datasets/blobs_train_meshes.npz',
			save_filepath='datasets/blobs_train.npz',
			meshes_per_npz=len(meshes),
			mode=mode,
			n_files=1,
		)
		data = _check_training_file(f'datasets/blobs_train_mode{mode}_0.npz', len(meshes))
		assert (data['masks'].sum(axis=1) > 10).all()
		assert os.path.exists(f'datasets/blobs_0_mode{mode}.obj')  # inspection files next to the mesh dataset


def test_scanned_dataset_matches_original(cpu64, tmp_path, monkeypatch, original_mesh_data):
	"""
	generate_scanned_mesh_training_dataset, port vs original (C++ samplers, trimesh, KD-tree), on the same mesh
	dataset (knot, torus): the same training files. Positions differ by the rounding of center_and_scale (~1e-16);
	the scans (modes 3, -1) are bit-identical through the float32 queries (mesh_data.query_dtype).
	"""
	monkeypatch.chdir(tmp_path)
	meshes = []
	for name in ['knot', 'torus']:
		V, F = torus_mesh() if name == 'torus' else _read_obj(name)
		meshes.append({'vertices': V, 'faces': F.astype(np.int32), 'file_id': name})
	for module in (md, original_mesh_data):
		monkeypatch.setattr(module, 'grid_spacing_min', 0.06)
		monkeypatch.setattr(module, 'grid_spacing_max', 0.1)
	print()
	for mode in [0, 1, 3, -1]:
		for directory in ['orig', 'port']:
			os.makedirs(directory, exist_ok=True)
			md.save_mesh_dataset(meshes, f'{directory}/shapes_train_meshes.npz')
		with _one_thread():
			original_mesh_data.generate_scanned_mesh_training_dataset(
				2, 'orig/shapes_train_meshes.npz', 'orig/shapes_train.npz', 2, mode, 1
			)
		md.generate_scanned_mesh_training_dataset(
			2, 'port/shapes_train_meshes.npz', 'port/shapes_train.npz', 2, mode, 1
		)
		ref = np.load(f'orig/shapes_train_mode{mode}_0.npz')
		port = _check_training_file(f'port/shapes_train_mode{mode}_0.npz', 2)
		assert sorted(ref.files) == sorted(port.files)
		diffs = {}
		for key in ref.files:
			assert ref[key].shape == port[key].shape and ref[key].dtype == port[key].dtype, key
			diffs[key] = np.abs(ref[key].astype(np.float64) - port[key].astype(np.float64)).max()
		# neighbors: identical up to the order of equidistant points (ray-cast samples lie on regular grids; the
		# original's kd-tree and the port's search break distance ties differently)
		same_rows = np.mean(np.all(ref['neighbors'] == port['neighbors'], axis=-1)[ref['masks']])
		for s in range(2):
			x = port['positions'][s]
			for nb in (ref['neighbors'][s], port['neighbors'][s]):
				assert ((nb >= 0) == ref['masks'][s][:, None]).all()
			m = ref['masks'][s]
			d_ref = np.linalg.norm(x[m][:, None] - x[ref['neighbors'][s][m]], axis=-1)
			d_port = np.linalg.norm(x[m][:, None] - x[port['neighbors'][s][m]], axis=-1)
			assert np.abs(d_ref - d_port).max() <= 1e-14
		print(
			f'  mode {mode:2d}: {int(ref["masks"].sum())} points; max|Δ| positions {diffs["positions"]:.1e}, normals '
			f'{diffs["normals"]:.1e}, mesh_vertices {diffs["mesh_vertices"]:.1e}; neighbor rows identical '
			f'{same_rows:.4f} (others: equidistant points swapped); masks, faces identical'
		)
		for key in ['masks', 'mesh_faces', 'face_masks']:
			assert diffs[key] == 0, key
		for key in ['positions', 'mesh_vertices']:
			assert diffs[key] <= 1e-15, key
		assert diffs['normals'] <= 1e-13


def test_pretraining_and_validation_datasets(cpu64, tmp_path, monkeypatch):
	"""The two fixed-path generators (with a torus as data/blob1024.obj and as the ABC test meshes), coarse scans."""
	monkeypatch.chdir(tmp_path)
	os.makedirs('data')
	os.makedirs('datasets')
	V, F = torus_mesh(n_major=48, n_minor=16)
	TriangleMesh(V, F).write_OBJ('data/blob1024.obj')
	monkeypatch.setattr(md, 'grid_spacings', [0.05])
	monkeypatch.setattr(md, 'max_points_per_pointcloud', 500)
	md.generate_pretraining_dataset()
	data = _check_training_file('datasets/pretraining_blob.npz', 1)
	assert data['masks'].sum() == 500
	assert np.allclose(np.linalg.norm(data['normals'][0], axis=1), 2.0)  # sample_point_cloud's normals (see C++)

	md.save_mesh_dataset([{'vertices': V, 'faces': F, 'file_id': 't'}] * 2, 'datasets/ABC7_components_test_meshes.npz')
	monkeypatch.setattr(md, 'grid_spacing_min', 0.06)
	monkeypatch.setattr(md, 'grid_spacing_max', 0.1)
	md.generate_validation_datasets()  # mode=3, n_files=1 (the original passes an unknown keyword)
	_check_training_file('datasets/ABC7_validation_mode3_0.npz', 2)


@pytest.mark.parametrize('module', ['thingi10k', 'py7zr'])
def test_optional_dependencies(cpu64, tmp_path, monkeypatch, module):
	"""Downloaders / archive readers are optional: a clear ImportError when missing (simulated here)."""
	monkeypatch.setitem(sys.modules, module, None)  # makes `import module` fail
	with pytest.raises(ImportError, match=f'pip install {module}'):
		if module == 'thingi10k':
			md.get_Thingi10k(str(tmp_path), 0.8)
		else:
			md.get_ABC(str(tmp_path), str(tmp_path), 0.8)
