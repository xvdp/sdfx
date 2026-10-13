# Process training/test datasets.
"""
Torch port of training/mesh_data.py: the mesh datasets (Thingi10K, ABC, procedural Perlin-noise blobs) and the
training / validation .npz files that training/train.py reads (simulated scans of the meshes, padded per shape).

The computation runs on the configured device (points_as_tori.set_device; the original's C++ ran in float64):
  * meshes, scan simulation and point samplers: shape_3d.TriangleMesh (port of the C++ Mesh3D);
  * k nearest neighbors: points_as_tori.get_neighbors;
  * Perlin noise: pnoise3, a bit-exact port of noise.pnoise3 (the `noise` package's C code, _perlin.c);
  * marching cubes: points_as_tori.isosurface.marching_cubes (the original contours with PyVista);
  * the mesh checks the original takes from trimesh (is_watertight, is_winding_consistent, is_volume, and the
    orientation test with a scipy KD-tree): trimesh_checks / orientation_dot.
External tools stay optional and are imported only by the functions that need them: thingi10k (get_Thingi10k),
py7zr and numpy-stl (get_ABC), pyfqmr (mesh simplification in generate_blob_mesh).

Layout: meshes and point clouds are (3, n) device tensors inside; everything stored (mesh datasets, training
features) is the original's host format: (n, 3) float64 vertices / positions / normals, int32 faces and neighbors.
Randomness: np.random exactly as the original (seed, rand, uniform, randint, permutation); the samplers replay the
C++ std::mt19937 streams (see TriangleMesh).

Fixes of the original (it cannot run as written there):
  * mode -2 of generate_scanned_mesh_training_dataset calls `add_point_cloud_noise`, defined nowhere: added here;
  * generate_pretraining_dataset calls `subsample_point_cloud`, not in the original's Python namespace: the C++
    farthest-point sampling (shape_2d.subsample_point_cloud);
  * generate_validation_datasets passes `only_save_one=True` (no such argument) and no `mode` / `n_files`: called
    with mode=3, n_files=1;
  * generate_blob_dataset imports save_mesh_dataset from a module named `mesh_data` (itself): the local function.
"""

from __future__ import annotations

import glob
import importlib
import json
import os
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .. import config
from .. import cpp_random
from ..isosurface import marching_cubes
from ..shape_2d import cross, dot, norm, subsample_point_cloud
from ..shape_3d import (
	TriangleMesh,
	_flip_signs,
	_normalized,
	_perturbations,
	_to_device,
	get_neighbors,
	write_point_cloud,
)

seed = 57

k_neighbors = 64
n_cameras = 4
sensor_size = 1.0
grid_spacing_min = 0.01
grid_spacing_max = 0.1
max_noise_level = 0.02
max_normals_to_flip = 0.1

# legacy
grid_spacings = [0.01]
sigmas_p = [0.0, 0.01]  # less noise, more noise
sigmas_n = [0.0, 0.01]
train_test_split = 0.8

max_faces_per_mesh = 2048  # maximum-sized mesh to consider in each dataset
max_vertices_per_mesh = max_faces_per_mesh  # training meshes should all be closed and manifold, meaning only max_faces_per_mesh is effectively active
max_points_per_pointcloud = max_faces_per_mesh  # maximum-sized point cloud
meshes_per_training_npz = 500  # number of shapes to include in each saved NPZ file encoding training features
n_inspect = 7  # number of meshes, point clouds to save per dataset to manually inspect

N_POINTS_SCALING = [512 * (4**k) for k in range(1, 7)]

# Precision of the meshes' closest-point and ray queries (TriangleMesh.query_dtype). The original's FCPW computes
# them in float32; with float32 here the simulated scans (modes -2, -1, 3, generate_pretraining_dataset) are the
# original's, bit for bit on the CPU. None: the configured dtype (exact hits; the farthest-point subsampling of the
# scans may then pick other points than the original where hit points are nearly equidistant).
query_dtype = torch.float32


def _optional_import(module: str, pip_name: str, purpose: str):
	"""External tools (dataset downloaders, archive / file formats, mesh simplification) are optional."""
	try:
		return importlib.import_module(module)
	except ImportError as e:
		raise ImportError(f'{purpose} needs the optional package `{pip_name}` (pip install {pip_name})') from e


def _vertices_numpy(shape: TriangleMesh) -> np.ndarray:
	"""Export: (n_vertices, 3) float64, as the original's TriangleMesh.get_vertices()."""
	return config.to_numpy(shape.get_vertices()).astype(np.float64)


def _faces_numpy(shape: TriangleMesh) -> np.ndarray:
	"""Export: (n_faces, 3) int32, as the original's TriangleMesh.get_faces() (Eigen int)."""
	return config.to_numpy(shape.get_faces()).astype(np.int32)


def _neighbors_numpy(neighbors: torch.Tensor) -> np.ndarray:
	"""Export: the (n_points, k) neighbor table of get_neighbors, as is (it is not a (3, n) wide array)."""
	return neighbors.detach().cpu().numpy()


# ========================================================================
# MESH CHECKS (trimesh in the original)
# ========================================================================


class MeshChecks:
	"""Results of trimesh_checks, the trimesh.Trimesh properties of the same names."""

	def __init__(self, is_watertight: bool, is_winding_consistent: bool, is_volume: bool, volume: float):
		self.is_watertight = is_watertight
		self.is_winding_consistent = is_winding_consistent
		self.is_volume = is_volume
		self.volume = volume


def _trimesh_process(vertices: torch.Tensor, faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	What trimesh.Trimesh(vertices, faces) does by default (process=True, trimesh 5.1): non-finite vertices and the
	faces using them are removed (remove_infinite_values), then merge_vertices: referenced vertices whose positions
	agree after rounding to tol.merge = 1e-8 (round(v · 10⁸) as int64) become one vertex, at the position of the first
	occurrence; unreferenced vertices are dropped.

	Args:
		vertices: (3, V), faces: (3, F) int64
	Returns:
		merged vertices (3, V'), faces (3, F') indexing them. Host reads: output sizes.
	"""
	finite = torch.isfinite(vertices).all(dim=0)
	faces = faces[:, finite[faces].all(dim=0)]
	n_vertices = vertices.shape[1]
	referenced = torch.zeros(n_vertices, dtype=torch.bool, device=vertices.device)
	referenced[faces.reshape(-1)] = True
	ref_idx = torch.nonzero(referenced)[:, 0]
	keys = torch.round(vertices[:, ref_idx] * 1e8).to(torch.int64)  # stacked = vertices * 10**digits_vertex
	_, group = torch.unique(keys, dim=1, return_inverse=True)
	n_groups = int(group.max()) + 1 if group.numel() else 0
	# first occurrence of each group
	first = torch.full((n_groups,), n_vertices, dtype=torch.long, device=vertices.device)
	first = first.scatter_reduce(0, group, ref_idx, reduce='amin')
	new_index = torch.full((n_vertices,), -1, dtype=torch.long, device=vertices.device)
	new_index[ref_idx] = group
	return vertices[:, first], new_index[faces]


def trimesh_checks(vertices: torch.Tensor, faces: torch.Tensor) -> MeshChecks:
	"""
	trimesh's Trimesh(vertices, faces).is_watertight / .is_winding_consistent / .is_volume (definitions of trimesh
	5.1, graph.is_watertight and triangles.mass_properties), on the processed mesh (_trimesh_process):
	  * edges = the (f0, f1), (f1, f2), (f2, f0) of every face; watertight iff every edge, as an unordered pair, occurs
	    exactly twice;
	  * winding consistent iff for every unordered edge occurring exactly twice the two occurrences have opposite
	    directions ((a, b) and (b, a));
	  * volume V = Σ_f c_f,x (x0 + x1 + x2)_f / 6 with c_f = (v1 - v0) × (v2 - v1) (divergence theorem), center of
	    mass Σ_f c_f ∘ (v0² + v1² + v0 v1 + v2 (v0 + v1 + v2)) / 24 / V; is_volume iff watertight, winding consistent,
	    finite center of mass and V > 0 (outward orientation).
	An empty mesh is neither. Host reads: a few sizes and the results.

	Args:
		vertices: (3, V), faces: (3, F) (device tensors)
	"""
	vertices, faces = _trimesh_process(vertices, faces.long())
	if vertices.shape[1] == 0 or faces.shape[1] == 0:
		return MeshChecks(False, False, False, 0.0)
	n_vertices = vertices.shape[1]
	edges = torch.cat([faces[[0, 1]], faces[[1, 2]], faces[[2, 0]]], dim=1)  # (2, 3F)
	lo, hi = torch.minimum(edges[0], edges[1]), torch.maximum(edges[0], edges[1])
	_, group, counts = torch.unique(lo * n_vertices + hi, return_inverse=True, return_counts=True)
	is_watertight = bool((counts == 2).all())
	# opposite directions: one occurrence with e0 < e1, the other with e0 > e1 (degenerate edges (a, a) pass)
	forward = torch.zeros(counts.shape[0], dtype=torch.long, device=edges.device)
	forward.index_add_(0, group, (edges[0] < edges[1]).long())
	degenerate = torch.zeros(counts.shape[0], dtype=torch.bool, device=edges.device)
	degenerate[group] = lo == hi
	is_winding_consistent = bool(((counts != 2) | (forward == 1) | degenerate).all())

	t0, t1, t2 = vertices[:, faces[0]], vertices[:, faces[1]], vertices[:, faces[2]]
	crosses = cross(t1 - t0, t2 - t1)
	f1 = t0 + t1 + t2
	f2 = t0**2 + t1**2 + t0 * t1 + t2 * f1
	volume = (crosses[0] * f1[0]).sum() / 6.0
	center_mass = (crosses * f2).sum(dim=1) / 24.0 / volume
	is_volume = bool(
		is_watertight and is_winding_consistent and bool(torch.isfinite(center_mass).all()) and float(volume) > 0.0
	)
	return MeshChecks(is_watertight, is_winding_consistent, is_volume, float(volume))


def orientation_dot(vertices: torch.Tensor, faces: torch.Tensor) -> float:
	"""
	The original's orientation check: with q = (2, 2, 2) (outside the [-1, 1]³ box and hence the mesh) and cp the
	mesh vertex nearest to q (a scipy KD-tree there; first index on ties here), returns (q - cp)·n, n the normalized
	mean of the unit normals of the faces around cp (trimesh face_normals: (v1 - v0) × (v2 - v1) normalized, zero if
	its norm is <= 1e-13). Negative means inward-facing. (trimesh merges duplicate vertices first, so the original's
	KD-tree index addresses the merged mesh; the same thing when the mesh has no duplicate or unreferenced vertices.)
	Host read: the result.

	Args:
		vertices: (3, V), faces: (3, F)
	"""
	query = torch.tensor([2.0, 2.0, 2.0], dtype=vertices.dtype, device=vertices.device)
	closest_vertex_id = torch.argmin(((vertices - query[:, None]) ** 2).sum(dim=0))
	cp = vertices[:, closest_vertex_id]
	# Get normal from adjacent faces
	t0, t1, t2 = vertices[:, faces[0]], vertices[:, faces[1]], vertices[:, faces[2]]
	c = cross(t1 - t0, t2 - t1)
	c_norm = norm(c)
	face_normals = torch.where(c_norm > 1e-13, c / c_norm, 0.0)
	vertex_faces = (faces == closest_vertex_id).any(dim=0)
	n = face_normals[:, vertex_faces].mean(dim=1)
	n = n / norm(n)
	return float(dot(query - cp, n))


# ========================================================================
# GET THE DATA
# ========================================================================


def save_mesh_dataset(meshes: List[Dict], filepath: str, metadata: Dict = None):
	"""
	Save a list of meshes to an NPZ file.

	Args:
		meshes: List of dicts with keys 'vertices', 'faces', 'file_id', 'metadata'
		filepath: Path to save the NPZ file
		metadata: Optional global metadata dictionary
	"""
	# Prepare arrays for storage
	# Store meshes as separate arrays since they have different sizes
	data = {
		'n_meshes': len(meshes),
	}

	# Add global metadata
	if metadata is not None:
		data['metadata'] = json.dumps(metadata)

	# Store each mesh separately with index
	for i, mesh_data in enumerate(meshes):
		data[f'vertices_{i}'] = mesh_data['vertices']
		data[f'faces_{i}'] = mesh_data['faces']
		data[f'file_id_{i}'] = mesh_data['file_id']

		# Store per-mesh metadata as JSON string
		if 'metadata' in mesh_data:
			data[f'metadata_{i}'] = json.dumps(mesh_data['metadata'])

	# Save with compression
	np.savez_compressed(filepath, **data)
	print(f'Saved {len(meshes)} meshes to {filepath}')


def load_mesh_dataset(filepath: str) -> Tuple[List[Dict], Dict]:
	"""
	Load meshes from an NPZ file.

	Args:
		filepath: Path to the NPZ file

	Returns:
		(meshes_list, global_metadata) where meshes_list contains dicts with
		'vertices', 'faces', 'file_id', 'metadata' keys
	"""
	data = np.load(filepath, allow_pickle=True)
	n_meshes = int(data['n_meshes'])

	# Load global metadata if present
	global_metadata = {}
	if 'metadata' in data:
		global_metadata = json.loads(str(data['metadata']))

	meshes = []
	for i in range(n_meshes):
		mesh_dict = {
			'vertices': data[f'vertices_{i}'],
			'faces': data[f'faces_{i}'],
			'file_id': str(data[f'file_id_{i}']),
		}

		# Load per-mesh metadata if present
		if f'metadata_{i}' in data:
			mesh_dict['metadata'] = json.loads(str(data[f'metadata_{i}']))

		meshes.append(mesh_dict)

	return meshes, global_metadata


def load_as_mesh3d(mesh_data: Dict):
	"""
	Convert loaded mesh data dict to TriangleMesh object.

	Args:
		mesh_data: Dict with 'vertices' and 'faces' keys

	Returns:
		TriangleMesh object (on the configured device)
	"""
	vertices = mesh_data['vertices']
	faces = mesh_data['faces']

	return TriangleMesh(vertices, faces, query_dtype=query_dtype)


def get_Thingi10k(save_dir: str, train_test_split: float):
	"""
	Get clean meshes from Thingi10k.

	Args:
		save_dir: Directory to save the train/test datasets
		train_test_split: Fraction of data to use for training (0.0 to 1.0)

	This function saves two files:
		- {save_dir}/thingi10k_train_meshes.npz
		- {save_dir}/thingi10k_test_meshes.npz

	(As the original, it also writes Thingi10k_{counter}.obj files of the meshes to the working directory.)
	"""
	thingi10k = _optional_import('thingi10k', 'thingi10k', 'get_Thingi10k (Thingi10K download)')

	# Create save directory
	os.makedirs(save_dir, exist_ok=True)

	thingi10k.init()

	# Collect valid meshes
	valid_meshes = []
	stats = {'total_processed': 0, 'valid': 0, 'errors': 0}

	print('\nProcessing meshes...')
	print('-' * 70)

	counter = 0
	min_n_faces = np.inf
	max_n_faces = 0
	mean_n_faces = 0
	for entry in thingi10k.dataset(
		closed=True, manifold=True, oriented=True, self_intersecting=False, solid=False, num_components=1
	):
		stats['total_processed'] += 1

		# Load mesh
		try:
			vertices, faces = thingi10k.load_file(entry['file_path'])
			min_n_faces = min(faces.shape[0], min_n_faces)
			max_n_faces = max(faces.shape[0], max_n_faces)
			mean_n_faces += faces.shape[0]

			# Center and scale data.
			shape = TriangleMesh(vertices, faces)
			shape.center_and_scale()
			vertices = _vertices_numpy(shape)
			faces = _faces_numpy(shape)

			# Check orientation
			if orientation_dot(shape.get_vertices(), shape.get_faces()) < 0.0:
				# Flip faces
				faces[:, [1, 2]] = faces[:, [2, 1]]
				shape = TriangleMesh(vertices, faces)

			# Save a few files to manually inspect
			if counter < n_inspect:
				shape.write_OBJ(f'Thingi10k_{counter}.obj')
			counter += 1

			shape.write_OBJ(f'Thingi10k_{counter}.obj')
			counter += 1

			# Store mesh with metadata
			mesh_data = {
				'vertices': vertices,
				'faces': faces,
				'file_id': entry['file_id'],
				'metadata': {
					'author': entry.get('author', 'unknown'),
					'license': entry.get('license', 'unknown'),
					'n_vertices': entry.get('n_vertices'),
					'n_faces': entry.get('n_faces'),
				},
			}
			valid_meshes.append(mesh_data)

		except Exception:
			stats['errors'] += 1

	if len(valid_meshes) == 0:
		raise ValueError('No valid meshes found in dataset!')

	print(f'Min # faces: {min_n_faces}')
	print(f'Max # faces: {max_n_faces}')
	print(f'Mean # faces: {mean_n_faces / len(valid_meshes)}')

	# Shuffle meshes for random split
	np.random.seed(seed)
	indices = np.random.permutation(len(valid_meshes))
	valid_meshes = [valid_meshes[i] for i in indices]

	# Split into train/test
	n_train = int(len(valid_meshes) * train_test_split)
	train_meshes = valid_meshes[:n_train]
	test_meshes = valid_meshes[n_train:]

	print('\nDataset Split:')
	print(f'  Training: {len(train_meshes)} meshes')
	print(f'  Test: {len(test_meshes)} meshes')

	# Save datasets
	train_path = os.path.join(save_dir, 'thingi10k_train_meshes.npz')
	test_path = os.path.join(save_dir, 'thingi10k_test_meshes.npz')

	global_metadata = {
		'dataset': 'Thingi10K',
		'train_test_split': train_test_split,
		'statistics': stats,
		'n_train': len(train_meshes),
		'n_test': len(test_meshes),
	}

	save_mesh_dataset(train_meshes, train_path, metadata=global_metadata)
	save_mesh_dataset(test_meshes, test_path, metadata=global_metadata)

	print('\nDatasets saved to:')
	print(f'  {train_path}')
	print(f'  {test_path}')


def save_train_test_meshes(valid_meshes: list, save_dir: str, name: str, train_test_split: float, seed: int):
	"""
	Given a list of dictionaries of mesh_data, split into training/test sets, and save.
	"""

	# Shuffle meshes for random split
	np.random.seed(seed)
	indices = np.random.permutation(len(valid_meshes))
	valid_meshes = [valid_meshes[i] for i in indices]

	# Split into train/test
	n_train = int(len(valid_meshes) * train_test_split)
	train_meshes = valid_meshes[:n_train]
	test_meshes = valid_meshes[n_train:]

	print('\nDataset Split:')
	print(f'  Training: {len(train_meshes)} meshes')
	print(f'  Test: {len(test_meshes)} meshes')

	# Save datasets
	train_path = os.path.join(save_dir, name + '_train_meshes.npz')
	test_path = os.path.join(save_dir, name + '_test_meshes.npz')

	save_mesh_dataset(train_meshes, train_path)
	save_mesh_dataset(test_meshes, test_path)

	print('\nDatasets saved to:')
	print(f'  {train_path}')
	print(f'  {test_path}')


# ========================================================================
# PERLIN NOISE (port of noise.pnoise3, noise 1.2.2, _perlin.c / _noise.h)
# ========================================================================

# fmt: off
_PERM_256 = (
	151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240,
	21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88,
	237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83,
	111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216,
	80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186,
	3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17,
	182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129,
	22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238,
	210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184,
	84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195,
	78, 66, 215, 61, 156, 180,
)

GRAD3 = (
	(1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
	(1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
	(0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
	(1, 0, -1), (-1, 0, -1), (0, -1, 1), (0, 1, 1),
)

GRAD4 = (
	(0, 1, 1, 1), (0, 1, 1, -1), (0, 1, -1, 1), (0, 1, -1, -1),
	(0, -1, 1, 1), (0, -1, 1, -1), (0, -1, -1, 1), (0, -1, -1, -1),
	(1, 0, 1, 1), (1, 0, 1, -1), (1, 0, -1, 1), (1, 0, -1, -1),
	(-1, 0, 1, 1), (-1, 0, 1, -1), (-1, 0, -1, 1), (-1, 0, -1, -1),
	(1, 1, 0, 1), (1, 1, 0, -1), (1, -1, 0, 1), (1, -1, 0, -1),
	(-1, 1, 0, 1), (-1, 1, 0, -1), (-1, -1, 0, 1), (-1, -1, 0, -1),
	(1, 1, 1, 0), (1, 1, -1, 0), (1, -1, 1, 0), (1, -1, -1, 0),
	(-1, 1, 1, 0), (-1, 1, -1, 0), (-1, -1, 1, 0), (-1, -1, -1, 0),
)
# fmt: on

# PERM is the 256-entry permutation twice (512 bytes). With base > 0 the C code indexes PERM up to 255 + 2·255 = 765,
# past its end (undefined behavior). In the extension as compiled by GCC on x86-64 Linux the bytes that follow PERM
# are those of the GRAD4 table (float32, little endian; constants laid out in reverse order of definition), checked
# against the installed module; the port reads the same bytes.
PERM = _PERM_256 * 2
_PERM_EXTENDED = PERM + tuple(np.array(GRAD4, dtype='<f4').tobytes()[:256])


def _noise3(x, y, z, repeatx: int, repeaty: int, repeatz: int, base: int, perm, grad3):
	"""noise3 of _perlin.c: Perlin's improved noise (2002) at float32 points, with its float32 arithmetic."""
	i = torch.floor(torch.fmod(x, float(repeatx))).to(torch.int32)
	j = torch.floor(torch.fmod(y, float(repeaty))).to(torch.int32)
	k = torch.floor(torch.fmod(z, float(repeatz))).to(torch.int32)
	ii = torch.fmod((i + 1).to(torch.float32), float(repeatx)).to(torch.int32)
	jj = torch.fmod((j + 1).to(torch.float32), float(repeaty)).to(torch.int32)
	kk = torch.fmod((k + 1).to(torch.float32), float(repeatz)).to(torch.int32)
	i = ((i & 255) + base).long()
	j = ((j & 255) + base).long()
	k = ((k & 255) + base).long()
	ii = ((ii & 255) + base).long()
	jj = ((jj & 255) + base).long()
	kk = ((kk & 255) + base).long()

	x = x - torch.floor(x)
	y = y - torch.floor(y)
	z = z - torch.floor(z)
	fx = x * x * x * (x * (x * 6 - 15) + 10)
	fy = y * y * y * (y * (y * 6 - 15) + 10)
	fz = z * z * z * (z * (z * 6 - 15) + 10)

	A = perm[i]
	AA = perm[A + j]
	AB = perm[A + jj]
	B = perm[ii]
	BA = perm[B + j]
	BB = perm[B + jj]

	def grad(hash_, gx, gy, gz):
		g = grad3[hash_ & 15]  # (..., 3)
		return gx * g[..., 0] + gy * g[..., 1] + gz * g[..., 2]

	def lerp(t, a, b):
		return a + t * (b - a)

	return lerp(
		fz,
		lerp(
			fy,
			lerp(fx, grad(perm[AA + k], x, y, z), grad(perm[BA + k], x - 1, y, z)),
			lerp(fx, grad(perm[AB + k], x, y - 1, z), grad(perm[BB + k], x - 1, y - 1, z)),
		),
		lerp(
			fy,
			lerp(fx, grad(perm[AA + kk], x, y, z - 1), grad(perm[BA + kk], x - 1, y, z - 1)),
			lerp(fx, grad(perm[AB + kk], x, y - 1, z - 1), grad(perm[BB + kk], x - 1, y - 1, z - 1)),
		),
	)


def pnoise3(
	x: torch.Tensor,
	y: torch.Tensor,
	z: torch.Tensor,
	octaves: int = 1,
	persistence: float = 0.5,
	lacunarity: float = 2.0,
	repeatx: int = 1024,
	repeaty: int = 1024,
	repeatz: int = 1024,
	base: int = 0,
) -> torch.Tensor:
	"""
	noise.pnoise3 (Perlin "improved" noise, fBm over octaves) for tensors of coordinates, bit-identical to the C
	extension (float32 arithmetic in the same order; the arguments are rounded to float32 as by its "f" parser):

		n(p) = Σ_o a^o noise3(l^o p) / Σ_o a^o,   a = persistence, l = lacunarity, o = 0 .. octaves - 1

	noise3 interpolates, with the quintic fade f(t) = t³ (t (6t - 15) + 10), the dot products of the 8 cell corners'
	pseudo-random gradients (PERM, GRAD3) with the offsets to the point. The lattice repeats every repeat·l^o cells;
	`base` offsets the permutation lookups (see _PERM_EXTENDED for base > 0).

	Returns:
		float64 tensor (the C double), shape of x
	"""
	if octaves < 1:
		raise ValueError('Expected octaves value > 0')
	device = x.device
	perm = torch.as_tensor(_PERM_EXTENDED, dtype=torch.long, device=device)
	grad3 = torch.as_tensor(GRAD3, dtype=torch.float32, device=device)
	x, y, z = x.to(torch.float32), y.to(torch.float32), z.to(torch.float32)
	base = int(base)
	if octaves == 1:
		# Single octave, return simple noise
		return _noise3(x, y, z, repeatx, repeaty, repeatz, base, perm, grad3).to(torch.float64)
	persistence = np.float32(persistence)
	lacunarity = np.float32(lacunarity)
	freq = np.float32(1.0)
	amp = np.float32(1.0)
	max_ = np.float32(0.0)
	total = torch.zeros_like(x)
	for _ in range(octaves):
		n = _noise3(
			x * freq,
			y * freq,
			z * freq,
			int(np.float32(repeatx) * freq),
			int(np.float32(repeaty) * freq),
			int(np.float32(repeatz) * freq),
			base,
			perm,
			grad3,
		)
		total = total + n * amp
		max_ = np.float32(max_ + amp)
		freq = np.float32(freq * lacunarity)
		amp = np.float32(amp * persistence)
	return (total / max_).to(torch.float64)


def evaluate_perlin_noise_at_points(
	points: torch.Tensor,
	scale: float = 1.0,
	octaves: int = 4,
	persistence: float = 0.5,
	lacunarity: float = 2.0,
	seed: Optional[int] = None,
) -> torch.Tensor:
	"""
	Evaluate 3D Perlin noise at given points (pnoise3, the port of the noise library).

	Args:
		points: (3, n) tensor of (x, y, z) coordinates
		scale: Frequency scale for the noise (higher = more frequent/smaller features)
		octaves: Number of noise layers to combine
		persistence: How much each octave contributes (0-1)
		lacunarity: Frequency multiplier between octaves (typically 2.0)
		seed: Random seed for reproducibility (0-255)

	Returns:
		(n,) tensor of noise values in range approximately [-1, 1] (dtype of points)
	"""
	# The noise library's pnoise3 expects base parameter (seed) in range 0-255
	if seed is not None:
		base = seed % 256
	else:
		base = 0

	# pnoise3 returns values in approximately [-0.5, 0.5]
	# Scale to [-1, 1]
	x, y, z = points[0], points[1], points[2]
	noise_values = pnoise3(
		x * scale,
		y * scale,
		z * scale,
		octaves=octaves,
		persistence=persistence,
		lacunarity=lacunarity,
		base=base,
	)
	return (noise_values * 2.0).to(points.dtype)


def generate_blob_implicit_surface(
	alpha: float,
	base_radius: float = 1.0,
	grid_resolution: int = 64,
	bbox_extent: float = 2.0,
	scale: float = 1.0,
	octaves: int = 4,
	persistence: float = 0.5,
	lacunarity: float = 2.0,
	seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Generate implicit surface values for a deformed sphere on a uniform grid.

	The implicit surface is defined by:
		F(x,y,z) = x^2 + y^2 + z^2 - (base_radius + alpha * f(x,y,z))^2 = 0
	with f the Perlin noise (evaluate_perlin_noise_at_points); F > 0 outside.

	The original returns a pyvista.ImageData with the samples as point data 'values'; the port returns the samples
	and the grid bounds, the input of isosurface.marching_cubes. Grid points are origin + i · spacing, as VTK's.

	Args:
		alpha: Deformation amplitude
		base_radius: Base radius of the sphere
		grid_resolution: Resolution of the marching cubes grid
		bbox_extent: Extent of bounding box for marching cubes
		scale: Perlin noise frequency scale (higher = smaller features)
		octaves: Number of Perlin noise octaves
		persistence: Perlin noise persistence
		lacunarity: Perlin noise lacunarity
		seed: Random seed

	Returns:
		values (grid_resolution,) * 3 with values[i, j, k] = F at (x_i, y_j, z_k), bounds_min, bounds_max (3,)
	"""
	dtype = config.get_dtype()
	device = config.get_device()
	# Create uniform grid
	spacing = (2 * bbox_extent) / (grid_resolution - 1)
	origin = -bbox_extent
	coords = torch.arange(grid_resolution, dtype=dtype, device=device) * spacing + origin
	x, y, z = torch.meshgrid(coords, coords, coords, indexing='ij')
	points = torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], dim=0)

	# Evaluate Perlin noise directly at all grid points
	f_values = evaluate_perlin_noise_at_points(
		points=points, scale=scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, seed=seed
	)

	# Compute implicit surface values
	# F(x,y,z) = x^2 + y^2 + z^2 - (base_radius + alpha * f(x,y,z))^2
	x, y, z = points[0], points[1], points[2]
	radius_squared = x**2 + y**2 + z**2
	deformed_radius = base_radius + alpha * f_values
	volume = radius_squared - deformed_radius**2

	bounds_min = coords[[0, 0, 0]]
	bounds_max = coords[[-1, -1, -1]]
	return volume.reshape(grid_resolution, grid_resolution, grid_resolution), bounds_min, bounds_max


def generate_blob_mesh(
	alpha: float,
	base_radius: float = 1.0,
	grid_resolution: int = 64,
	scale: float = 1.0,
	octaves: int = 4,
	persistence: float = 0.5,
	lacunarity: float = 2.0,
	seed: Optional[int] = None,
	validate: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Generate a single blob mesh using Perlin noise deformation: marching cubes of the implicit surface, simplified to
	max_faces_per_mesh faces with pyfqmr (optional dependency), validated with trimesh_checks.

	Args:
		alpha: Deformation amplitude (should be < 1.0 to avoid disconnected components)
		base_radius: Base radius of the sphere
		grid_resolution: Resolution for marching cubes (e.g., 64)
		scale: Perlin noise frequency scale (higher = smaller/more frequent features)
		octaves: Number of Perlin noise octaves
		persistence: Perlin noise persistence
		lacunarity: Perlin noise lacunarity (frequency multiplier)
		seed: Random seed for reproducibility
		validate: Whether to validate the mesh is closed, watertight, and manifold

	Returns:
		(vertices (n_vertices, 3) float64, faces (n_faces, 3) int32) host arrays for the mesh (pyfqmr's output)

	Raises:
		ValueError: If validation fails
	"""
	pyfqmr = _optional_import('pyfqmr', 'pyfqmr', 'generate_blob_mesh (mesh simplification)')

	# Generate implicit surface
	bbox_extent = base_radius * (1.0 + alpha) * 1.2  # Add 20% margin
	values, bounds_min, bounds_max = generate_blob_implicit_surface(
		alpha=alpha,
		base_radius=base_radius,
		grid_resolution=grid_resolution,
		bbox_extent=bbox_extent,
		scale=scale,
		octaves=octaves,
		persistence=persistence,
		lacunarity=lacunarity,
		seed=seed,
	)

	# Extract mesh with marching cubes (the original: PyVista's contour filter)
	try:
		vertices, faces = marching_cubes(values, bounds_min, bounds_max, isovalue=0.0)

		if vertices.shape[1] == 0 or faces.shape[1] == 0:
			raise ValueError('Marching cubes produced empty mesh')

	except Exception as e:
		raise ValueError(f'Marching cubes failed: {e}')

	# Extract vertices and faces (export to the host for the simplifier). marching_cubes orients the triangles
	# toward increasing values, i.e. outward here: the original's flip of PyVista's triangles
	# (faces[:, [1, 2]] = faces[:, [2, 1]]) is not needed.
	vertices = config.to_numpy(vertices).astype(np.float64)
	faces = config.to_numpy(faces)

	# Simplify
	mesh_simplifier = pyfqmr.Simplify()
	mesh_simplifier.setMesh(vertices, faces)
	mesh_simplifier.simplify_mesh(
		target_count=max_faces_per_mesh,
		aggressiveness=10,  # will always strive to hit target_count, at the cost of potentially a lot of geometric error
		preserve_border=True,
		verbose=0,
	)
	vertices, faces, _ = mesh_simplifier.getMesh()

	# Validate mesh if requested
	if validate:
		checks = trimesh_checks(config.as_wide(vertices, 3), config.as_wide(faces, 3, dtype=torch.long))

		if not checks.is_watertight:
			raise ValueError('Generated mesh is not watertight')

		if not checks.is_volume:
			raise ValueError('Generated mesh does not enclose a volume')

	return vertices, faces


def generate_blob_dataset(
	n_blobs: int,
	save_dir: str,
	name: str = 'blobs',
	alpha_range: Tuple[float, float] = (0.1, 0.7),
	scale_range: Tuple[float, float] = (0.5, 3.0),
	octaves_range: Tuple[int, int] = (1, 4),
	base_radius: float = 1.0,
	grid_resolution: int = 64,
	persistence: float = 1.0,
	lacunarity: float = 2.0,
	seed: int = 42,
	train_test_split: float = 0.8,
) -> None:
	"""
	Generate a batch of blob meshes with varying parameters.

	Args:
		n_blobs: Number of blobs to generate
		save_dir: Directory to save the meshes and NPZ files
		name: Root name for saved files (default: 'blobs')
		alpha_range: (min, max) range for deformation amplitude
		scale_range: (min, max) range for Perlin noise scale (higher = smaller features)
		octaves_range: (min, max) range for number of octaves (inclusive)
		base_radius: Base radius of the sphere
		grid_resolution: Marching cubes resolution
		persistence: Perlin noise persistence (fixed at 1.0)
		lacunarity: Perlin noise lacunarity (fixed at 2.0)
		seed: Random seed for reproducibility
		train_test_split: Fraction of data for training
	"""

	# Create save directory
	os.makedirs(save_dir, exist_ok=True)

	# Set random seed
	np.random.seed(seed)

	# Pre-sample more parameters than needed to handle failures
	# We'll generate 3x as many in case some fail
	n_to_sample = n_blobs * 3
	alphas = np.random.uniform(alpha_range[0], alpha_range[1], n_to_sample)
	scales = np.random.uniform(scale_range[0], scale_range[1], n_to_sample)
	octaves_list = np.random.randint(octaves_range[0], octaves_range[1] + 1, n_to_sample)

	valid_meshes = []
	stats = {'total_attempted': 0, 'valid': 0, 'errors': 0}

	print(f'\nGenerating {n_blobs} blob meshes...')
	print('Parameters:')
	print(f'  alpha: [{alpha_range[0]}, {alpha_range[1]}]')
	print(f'  scale: [{scale_range[0]}, {scale_range[1]}]')
	print(f'  octaves: [{octaves_range[0]}, {octaves_range[1]}]')
	print(f'  persistence: {persistence} (fixed)')
	print(f'  lacunarity: {lacunarity} (fixed)')
	print('-' * 70)

	t0_total = time.time()

	i = 0  # Index into pre-sampled parameters
	while len(valid_meshes) < n_blobs:
		try:
			# Check if we need to sample more parameters
			if i >= len(alphas):
				print(f'  Sampling additional parameters (current valid: {len(valid_meshes)}/{n_blobs})...')
				# Sample another batch
				additional_samples = n_blobs - len(valid_meshes) + 10  # Add a few extra
				new_alphas = np.random.uniform(alpha_range[0], alpha_range[1], additional_samples)
				new_scales = np.random.uniform(scale_range[0], scale_range[1], additional_samples)
				new_octaves = np.random.randint(octaves_range[0], octaves_range[1] + 1, additional_samples)

				alphas = np.concatenate([alphas, new_alphas])
				scales = np.concatenate([scales, new_scales])
				octaves_list = np.concatenate([octaves_list, new_octaves])

			t0 = time.time()

			# Generate unique seed for this blob
			blob_seed = seed + i

			# Sample parameters
			alpha = alphas[i]
			scale = scales[i]
			octaves = int(octaves_list[i])

			stats['total_attempted'] += 1

			# Generate mesh
			vertices, faces = generate_blob_mesh(
				alpha=alpha,
				base_radius=base_radius,
				grid_resolution=grid_resolution,
				scale=scale,
				octaves=octaves,
				persistence=persistence,
				lacunarity=lacunarity,
				seed=blob_seed,
				validate=True,
			)

			t1 = time.time()

			# Center and scale to unit sphere
			mesh = TriangleMesh(vertices, faces)
			mesh.center_and_scale()
			vertices = _vertices_numpy(mesh)
			faces = _faces_numpy(mesh)

			# # Save individual OBJ file
			# obj_path = os.path.join(save_dir, f'{name}_{len(valid_meshes):05d}.obj')
			# mesh.write_OBJ(obj_path)

			# Store mesh data
			mesh_data = {
				'vertices': vertices,
				'faces': faces,
				'file_id': f'{name}_{len(valid_meshes):05d}',
				'metadata': {
					'alpha': float(alpha),
					'scale': float(scale),
					'octaves': int(octaves),
					'base_radius': float(base_radius),
					'grid_resolution': grid_resolution,
					'persistence': float(persistence),
					'lacunarity': float(lacunarity),
					'seed': blob_seed,
					'n_vertices': len(vertices),
					'n_faces': len(faces),
					'generation_time': t1 - t0,
				},
			}

			valid_meshes.append(mesh_data)
			stats['valid'] += 1

			if len(valid_meshes) % 10 == 0:
				print(
					f'Generated {len(valid_meshes)}/{n_blobs} valid blobs ({t1 - t0:.2f}s) - last: alpha={alpha:.2f}, scale={scale:.2f}, octaves={octaves}'
				)

		except Exception as e:
			print(f'Error generating blob {i} (attempt {stats["total_attempted"]}): {e}')
			stats['errors'] += 1

		i += 1  # Move to next parameter set

	t1_total = time.time()

	print('\nGeneration complete!')
	print(f'  Total time: {t1_total - t0_total:.2f}s')
	print(f'  Average time per valid blob: {(t1_total - t0_total) / stats["valid"]:.2f}s')
	print(f'  Valid blobs: {stats["valid"]}')
	print(f'  Failed attempts: {stats["errors"]}')
	print(f'  Total attempts: {stats["total_attempted"]}')
	print(f'  Success rate: {100.0 * stats["valid"] / stats["total_attempted"]:.1f}%')

	if len(valid_meshes) == 0:
		raise ValueError('No valid meshes generated!')

	# Shuffle and split into train/test
	np.random.seed(seed)
	indices = np.random.permutation(len(valid_meshes))
	valid_meshes = [valid_meshes[i] for i in indices]

	n_train = int(len(valid_meshes) * train_test_split)
	train_meshes = valid_meshes[:n_train]
	test_meshes = valid_meshes[n_train:]

	print('\nDataset Split:')
	print(f'  Training: {len(train_meshes)} meshes')
	print(f'  Test: {len(test_meshes)} meshes')

	# Save NPZ files (the original imports save_mesh_dataset from `mesh_data`, i.e. this module)
	train_path = os.path.join(save_dir, f'{name}_train_meshes.npz')
	test_path = os.path.join(save_dir, f'{name}_test_meshes.npz')

	global_metadata = {
		'dataset': 'Procedural Blobs (Perlin)',
		'train_test_split': train_test_split,
		'statistics': stats,
		'n_train': len(train_meshes),
		'n_test': len(test_meshes),
		'generation_parameters': {
			'alpha_range': alpha_range,
			'scale_range': scale_range,
			'octaves_range': octaves_range,
			'base_radius': base_radius,
			'grid_resolution': grid_resolution,
			'persistence': persistence,
			'lacunarity': lacunarity,
			'seed': seed,
		},
	}

	save_mesh_dataset(train_meshes, train_path, metadata=global_metadata)
	save_mesh_dataset(test_meshes, test_path, metadata=global_metadata)

	print('\nDatasets saved to:')
	print(f'  {train_path}')
	print(f'  {test_path}')


def get_ABC(load_dir: str, save_dir: str, train_test_split: float):
	"""
	Get meshes from ABC.

	Args:
		load_dir: Directory in which .7z file(s) containing meshes representing whole CAD objects (stl2) are stored; downloaded from here: https://archive.nyu.edu/handle/2451/43778
		save_dir: Directory to save the train/test datasets
		train_test_split: Fraction of data to use for training (0.0 to 1.0)

	This function saves two files:
		- {save_dir}/ABC_train_meshes.npz
		- {save_dir}/ABC_test_meshes.npz

	"""
	py7zr = _optional_import('py7zr', 'py7zr', 'get_ABC (.7z archives)')
	stl_mesh = _optional_import('stl.mesh', 'numpy-stl', 'get_ABC (STL files)')

	# Create save directory
	os.makedirs(save_dir, exist_ok=True)

	print('\nProcessing meshes...')
	print('-' * 70)

	# Go through each 7z file
	chunks = sorted(glob.glob(os.path.join(load_dir, '*.7z')))

	if len(chunks) == 0:
		raise ValueError(f'No .7z files found in {load_dir}')

	print(f'Found {len(chunks)} .7z archive(s)')

	# Go through each 7z file
	for chunk_idx, chunk_path in enumerate(chunks):
		chunk_name = os.path.basename(chunk_path)
		print(f'\nProcessing archive {chunk_idx + 1}/{len(chunks)}: {chunk_name}')

		valid_meshes = []
		error_meshes = 0

		try:
			# Create a temporary directory for extraction
			with tempfile.TemporaryDirectory() as temp_dir:
				# Extract the 7z archive
				print('  Extracting...')
				with py7zr.SevenZipFile(chunk_path, mode='r') as archive:
					archive.extractall(path=temp_dir)

				# Find all STL files in the extracted directory
				stl_files = []
				for root, dirs, files in os.walk(temp_dir):
					for file in files:
						if file.lower().endswith('.stl'):
							stl_files.append(os.path.join(root, file))

				print(f'  Found {len(stl_files)} STL files')

				# For each STL file, load and save vertices and faces
				counter = 0
				min_n_faces = np.inf
				max_n_faces = 0
				mean_n_faces = 0
				for stl_path in stl_files:
					try:
						# Load STL file using numpy-stl
						mesh_data = stl_mesh.Mesh.from_file(stl_path)

						# STL stores each triangle as 3 vertices (redundant)
						# mesh_data.vectors is shape (n_triangles, 3, 3)
						# Reshape to (n_triangles * 3, 3) to get all vertices
						vertices = mesh_data.vectors.reshape(-1, 3)

						# Deduplicate vertices and create face indices
						unique_vertices, inverse_indices = np.unique(vertices, axis=0, return_inverse=True)
						faces = inverse_indices.reshape(-1, 3)

						# Convert to proper types
						vertices = unique_vertices.astype(np.float64)
						faces = faces.astype(np.int64)

						# Center and scale data
						shape = TriangleMesh(vertices, faces)
						shape.center_and_scale()
						vertices = _vertices_numpy(shape)
						faces = _faces_numpy(shape)

						min_n_faces = min(min_n_faces, faces.shape[0])
						max_n_faces = max(max_n_faces, faces.shape[0])
						mean_n_faces += faces.shape[0]

						# Save a few files to manually inspect
						if counter < n_inspect:
							shape.write_OBJ(f'ABC_{counter}.obj')
						counter += 1

						# Create file ID from relative path
						rel_path = os.path.relpath(stl_path, temp_dir)
						file_id = f'{chunk_name}_{rel_path}'.replace(os.sep, '_')

						mesh_data = {
							'vertices': vertices,
							'faces': faces,
							'file_id': file_id,
							'metadata': {
								'archive': chunk_name,
								'stl_path': rel_path,
								'n_vertices': len(vertices),
								'n_faces': len(faces),
							},
						}

						valid_meshes.append(mesh_data)

						if len(valid_meshes) % 100 == 0:
							print(f'  Processed {len(valid_meshes)} meshes...')

					except Exception:
						print(f"  Couldn't process STL file {stl_path}")
						error_meshes += 1

			print(f'Number of valid meshes: {len(valid_meshes)}')
			print(f'Meshes that could not be loaded: {error_meshes}')

			print(f'Min # faces: {min_n_faces}')
			print(f'Max # faces: {max_n_faces}')
			print(f'Mean # faces: {mean_n_faces / len(valid_meshes)}')

			save_train_test_meshes(
				valid_meshes, save_dir=save_dir, name=f'ABC{chunk_idx}', train_test_split=train_test_split, seed=seed
			)

		except Exception as e:
			print(f'  Error processing archive {chunk_name}: {e}')


# ========================================================================
# GENERATE DATASETS
# ========================================================================


def pad_mesh(vertices, faces, max_vertices=None, max_faces=None):
	"""
	Pad mesh to fixed size for batching.

	Args:
		vertices: (n_vertices, 3) array
		faces: (n_faces, 3) array
		max_vertices: Maximum number of vertices (default: max_vertices_per_mesh)
		max_faces: Maximum number of faces (default: max_faces_per_mesh)

	Returns:
		vertices_padded: (max_vertices, 3) padded with zeros
		faces_padded: (max_faces, 3) padded with -1
		face_mask: (max_faces,) boolean mask for valid faces
	"""
	if max_vertices is None:
		max_vertices = max_vertices_per_mesh
	if max_faces is None:
		max_faces = max_faces_per_mesh

	n_verts = vertices.shape[0]
	n_faces = faces.shape[0]

	# Pad vertices with zeros
	vertices_padded = np.zeros((max_vertices, 3), dtype=np.float64)
	vertices_padded[:n_verts] = vertices

	# Pad faces with -1 (invalid indices)
	faces_padded = np.full((max_faces, 3), -1, dtype=np.int32)
	faces_padded[:n_faces] = faces

	# Create boolean mask for valid faces
	face_mask = np.zeros(max_faces, dtype=bool)
	face_mask[:n_faces] = True

	return vertices_padded, faces_padded, face_mask


def generate_input_features_with_meshes(all_data, max_points):
	"""
	Args:
		all_data: List of dictionaries with keys (host arrays):
			{
				'positions': (n_points, 3),
				'normals': (n_points, 3),
				'neighbors': (n_points, k),
				'vertices': (n_vertices, 3),
				'faces': (n_faces, 3),
			}
		max_points: Maximum number of points to pad to

	Returns:
		Tuple of NumPy arrays:
			all_positions: (n_shapes, max_points, 3)
			all_normals: (n_shapes, max_points, 3)
			all_neighbors: (n_shapes, max_points, k)
			all_masks: (n_shapes, max_points)
			all_mesh_vertices: (n_shapes, max_vertices, 3)
			all_mesh_faces: (n_shapes, max_faces, 3)
			all_face_masks: (n_shapes, max_faces)
	"""
	all_padded_data = []

	for i in range(len(all_data)):
		n_points = all_data[i]['positions'].shape[0]
		pad_size = max_points - n_points

		# Pad point cloud data
		positions = np.concatenate([all_data[i]['positions'], np.zeros((pad_size, 3))], axis=0)
		normals = np.concatenate([all_data[i]['normals'], np.zeros((pad_size, 3))], axis=0)

		# Pad neighbors if |P| < k_neighbors
		if n_points < k_neighbors:
			pad_width = min(k_neighbors, max_points) - n_points
			all_data[i]['neighbors'] = np.pad(
				all_data[i]['neighbors'], ((0, 0), (0, pad_width)), mode='constant', constant_values=-1
			)

		neighbors = np.concatenate(
			[all_data[i]['neighbors'], -np.ones((pad_size, min(k_neighbors, max_points)), dtype=int)], axis=0
		)

		mask = np.concatenate([np.ones(n_points, dtype=bool), np.zeros(pad_size, dtype=bool)])

		# Pad mesh data
		vertices_padded, faces_padded, face_mask = pad_mesh(all_data[i]['vertices'], all_data[i]['faces'])

		all_padded_data.append(
			{
				'positions': positions,
				'normals': normals,
				'neighbors': neighbors,
				'mask': mask,
				'vertices': vertices_padded,
				'faces': faces_padded,
				'face_mask': face_mask,
			}
		)

	# Stack into arrays
	all_positions = np.stack([d['positions'] for d in all_padded_data])
	all_normals = np.stack([d['normals'] for d in all_padded_data])
	all_neighbors = np.stack([d['neighbors'] for d in all_padded_data])
	all_masks = np.stack([d['mask'] for d in all_padded_data])
	all_mesh_vertices = np.stack([d['vertices'] for d in all_padded_data])
	all_mesh_faces = np.stack([d['faces'] for d in all_padded_data])
	all_face_masks = np.stack([d['face_mask'] for d in all_padded_data])

	return (
		all_positions,
		all_normals,
		all_neighbors,
		all_masks,
		all_mesh_vertices,
		all_mesh_faces,
		all_face_masks,
	)


def save_mesh_training_data(
	filepath: str,
	all_positions,
	all_normals,
	all_neighbors,
	all_masks,
	all_mesh_vertices,
	all_mesh_faces,
	all_face_masks,
):
	"""
	Save training data with mesh information (no pre-generated queries).

	Args:
		filepath: Path to save file (should end in .npz)
		all_positions: (n_shapes, max_points, 3)
		all_normals: (n_shapes, max_points, 3)
		all_neighbors: (n_shapes, max_points, k)
		all_masks: (n_shapes, max_points)
		all_mesh_vertices: (n_shapes, max_vertices, 3)
		all_mesh_faces: (n_shapes, max_faces, 3)
		all_face_masks: (n_shapes, max_faces)
	"""
	np.savez_compressed(
		filepath,
		positions=np.array(all_positions),
		normals=np.array(all_normals),
		neighbors=np.array(all_neighbors, dtype=np.int32),
		masks=np.array(all_masks, dtype=np.bool_),
		mesh_vertices=np.array(all_mesh_vertices),
		mesh_faces=np.array(all_mesh_faces, dtype=np.int32),
		face_masks=np.array(all_face_masks, dtype=np.bool_),
	)

	print(f'Saved training data to {filepath}')
	print(f'  Shapes: {all_positions.shape[0]}')
	print(f'  Max points: {all_positions.shape[1]}')
	print(f'  k_neighbors: {all_neighbors.shape[2]}')
	print(f'  Max vertices: {all_mesh_vertices.shape[1]}')
	print(f'  Max faces: {all_mesh_faces.shape[1]}')


def add_point_cloud_noise(
	points: torch.Tensor,
	normals: torch.Tensor,
	max_noise_level: float = 0.01,
	max_normals_to_flip: float = 0.1,
	seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Not defined in the original: mode -2 of generate_scanned_mesh_training_dataset calls it (NameError there).
	Added with the noise model of Mesh3D::sample_uneven_point_cloud: from std::mt19937(seed), sigma_p, sigma_n
	~ U[0, max_noise_level) and f ~ U[0, max_normals_to_flip); points p + δp, normals normalize(n + δn),
	δp ~ N(0, sigma_p² I), δn ~ N(0, sigma_n² I); then the normals of int(f n) points (std::shuffle) negated.

	Args:
		points, normals: (3, n)
	Returns:
		points, normals: (3, n)
	"""
	n_points = points.shape[1]
	rng = cpp_random.MT19937(seed)
	sigma_p = cpp_random.UniformReal(0.0, max_noise_level)(rng)
	sigma_n = cpp_random.UniformReal(0.0, max_noise_level)(rng)
	normals_to_flip = cpp_random.UniformReal(0.0, max_normals_to_flip)(rng)
	n_normals_to_flip = int(normals_to_flip * n_points)
	position_offset, normal_offset = _perturbations(rng, n_points, sigma_p, sigma_n)
	points = points + _to_device(position_offset, points.dtype)
	normals = _normalized(normals + _to_device(normal_offset, normals.dtype))
	return points, normals * _to_device(_flip_signs(n_points, n_normals_to_flip, rng), normals.dtype)


def generate_scanned_mesh_training_dataset(
	n_views: int, npz_filepath: str, save_filepath: str, meshes_per_npz: int, mode: int, n_files: int
):
	"""
	Generate training dataset from triangle meshes with mesh data (no pre-generated queries).

	Given triangle meshes stored in NPZ format, sample point clouds using 3D scanning simulation, and store:
	{
		'positions': (n_shapes, max_points, 3),
		'normals': (n_shapes, max_points, 3),
		'neighbors': (n_shapes, max_points, k),
		'masks': (n_shapes, max_points),

		'mesh_vertices': (n_shapes, max_vertices, 3),
		'mesh_faces': (n_shapes, max_faces, 3),
		'face_masks': (n_shapes, max_faces),
	}

	Split into batches of size `meshes_per_npz`.

	Modes: 0 uniform or farthest-point samples (no noise); -1 ray-cast samples (holes), -2 the same with flipped
	normals; 1 and 2 uniform / farthest-point samples with noise (and flipped normals for farthest-point); 3 the full
	scan simulation (sample_uneven_point_cloud). Files: '{dir}/{root}_{suffix}_mode{mode}_{n}.npz', where dir is the
	first path component of save_filepath and root / suffix the first / last '_'-separated parts of its file name
	(as in the original). A few meshes and point clouds are written next to npz_filepath for inspection.

	Args:
		n_views: Number of cameras of the scan simulation
		npz_filepath: Path to input mesh NPZ file
		save_filepath: Path template for output NPZ files
		meshes_per_npz: Number of meshes per output file
		mode: Sampling mode (see above)
		n_files: Stop after saving this many files
	"""
	print('Generating mesh data from: %s' % npz_filepath)
	print('Saving data to: %s' % save_filepath)

	t0 = time.time()
	mesh_lists, _ = load_mesh_dataset(npz_filepath)

	n_npz = 0
	n_meshes_processed = 0
	all_data = []
	max_points = 0
	counter = 0
	n_training_examples = 0
	n_files_saved = 0

	# Set seed
	np.random.seed(seed)

	for i, mesh_data in enumerate(mesh_lists):
		print(f'Processing shape {i + 1}/{len(mesh_lists)}...')

		# Load mesh
		shape = load_as_mesh3d(mesh_data)
		shape.center_and_scale()
		n_faces = shape.get_faces().shape[1]
		n_vertices = shape.get_vertices().shape[1]

		# Skip meshes that are too large
		if n_faces > max_faces_per_mesh or n_vertices > max_vertices_per_mesh:
			print(
				f'  Skipping: {n_faces} faces, {n_vertices} vertices (max is {max_faces_per_mesh}, {max_vertices_per_mesh})'
			)
			# Inspect mesh
			# shape.write_OBJ(f'{save_dir}/{rootname}_{i}.obj')
			# write_point_cloud(points, normals, f'{save_dir}/{rootname}_{counter}_pointcloud.obj')
			continue

		try:
			# Check orientation
			if orientation_dot(shape.get_vertices(), shape.get_faces()) < 0.0:
				# Flip faces
				print(f'Shape {i}: not oriented correctly')
				continue
		except Exception:
			print(f'  Shape {i}: could not check orientation, skipping')
			continue

		mesh_checks = trimesh_checks(shape.get_vertices(), shape.get_faces())
		if not mesh_checks.is_watertight:
			print(f'  Shape {i}: not watertight')
			continue
		if not mesh_checks.is_volume:
			print(f'  Shape {i}: does not enclose volume, skipping')
			continue
		if not mesh_checks.is_winding_consistent:
			print(f'  Shape {i}: not winding consistent, skipping')
			continue

		vertices = _vertices_numpy(shape)  # (n_vertices, 3)
		faces = _faces_numpy(shape)  # (n_faces, 3)
		print(f'  {vertices.shape[0]} vertices, {faces.shape[0]} faces')

		n_meshes_processed += 1

		# Sample point cloud from mesh: (3, n) tensors
		points = None
		normals = None
		if mode == 0:
			# Easy: Nice samplings only: farthest point sampling of meshes, uniform sampling of meshes; no noise, no flipped normals
			# Choose at random whether to use uniform sampling or farthest point sampling (50-50)
			uniform = np.random.rand() < 0.5
			if uniform:
				points, normals = shape.sample_uniform_point_cloud(
					n_points=max_points_per_pointcloud, max_noise_level=0.0, max_normals_to_flip=0.0, seed=seed + i
				)
			else:
				points, normals = shape.sample_farthest_point_cloud(
					n_points=max_points_per_pointcloud, max_noise_level=0.0, max_normals_to_flip=0.0, seed=seed + i
				)
		elif mode == -1:
			# Still nice samplings: farthest-point sampling of meshes, uniform sampling of meshes; but possibly with holes from a scanning process
			points, normals = shape.sample_even_raycasted_point_cloud(
				n_cameras=n_views,
				n_points=max_points_per_pointcloud,
				sensor_size=sensor_size,
				grid_spacing_min=grid_spacing_min,
				grid_spacing_max=grid_spacing_max,
				seed=seed + i,
			)
		elif mode == -2:
			# Same as above, but flip some normals
			points, normals = shape.sample_even_raycasted_point_cloud(
				n_cameras=n_views,
				n_points=max_points_per_pointcloud,
				sensor_size=sensor_size,
				grid_spacing_min=grid_spacing_min,
				grid_spacing_max=grid_spacing_max,
				seed=seed + i,
			)
			points, normals = add_point_cloud_noise(
				points, normals, max_noise_level=0.0, max_normals_to_flip=max_normals_to_flip, seed=seed + i
			)

		elif mode == 1:
			# Medium: Still nice samplings: farthest-point sampling of meshes, uniform sampling of meshes; but with noise
			uniform = np.random.rand() < 0.5
			if uniform:
				points, normals = shape.sample_uniform_point_cloud(
					n_points=max_points_per_pointcloud,
					max_noise_level=max_noise_level,
					max_normals_to_flip=0.0,
					seed=seed + i,
				)
			else:
				points, normals = shape.sample_farthest_point_cloud(
					n_points=max_points_per_pointcloud,
					max_noise_level=max_noise_level,
					max_normals_to_flip=max_normals_to_flip,
					seed=seed + i,
				)

		elif mode == 2:
			# Still nice samplings: farthest-point sampling of meshes, uniform sampling of meshes; but with noise and flipped normals
			uniform = np.random.rand() < 0.5
			if uniform:
				points, normals = shape.sample_uniform_point_cloud(
					n_points=max_points_per_pointcloud,
					max_noise_level=max_noise_level,
					max_normals_to_flip=0.0,
					seed=seed + i,
				)
			else:
				points, normals = shape.sample_farthest_point_cloud(
					n_points=max_points_per_pointcloud,
					max_noise_level=max_noise_level,
					max_normals_to_flip=max_normals_to_flip,
					seed=seed + i,
				)
		elif mode == 3:
			# Hardest: Full sampling procedure (noise, flipped normals, holes, uneven sampling)
			points, normals = shape.sample_uneven_point_cloud(
				n_cameras=n_views,
				sensor_size=sensor_size,
				grid_spacing_min=grid_spacing_min,
				grid_spacing_max=grid_spacing_max,
				max_noise_level=max_noise_level,
				max_normals_to_flip=max_normals_to_flip,
				max_points=max_points_per_pointcloud,
				seed=seed + i,
			)

		print(f'  Number of points: {points.shape[1]}...', end='')

		# Skip point clouds that are too small
		if points.shape[1] < 10:
			continue

		# Compute neighbors for point cloud
		neighbors = get_neighbors(points, k_neighbors)  # (n_points, k)

		# Save a few shapes for inspection
		save_dir = npz_filepath.split('/')[0]
		filename = npz_filepath.split('/')[-1]
		rootname = filename.split('_')[0]

		if counter < n_inspect:
			shape.write_OBJ(f'{save_dir}/{rootname}_{i}_mode{mode}.obj')
			write_point_cloud(points, normals, f'{save_dir}/{rootname}_{i}_mode{mode}_pointcloud.obj')
			counter += 1

		n_points = points.shape[1]
		n_training_examples += n_points
		print(f'  Point cloud: {n_points} points')
		max_points = max(n_points, max_points)

		# Store data (no pre-generated queries/distances!), exported to the host in the original's layout
		all_data.append(
			{
				'positions': config.to_numpy(points).astype(np.float64),
				'normals': config.to_numpy(normals).astype(np.float64),
				'neighbors': _neighbors_numpy(neighbors),
				'vertices': vertices,
				'faces': faces,
			}
		)

		# Save batch when we've processed enough meshes
		if n_meshes_processed == meshes_per_npz or i == len(mesh_lists) - 1:
			print(f'Number of meshes: {n_meshes_processed}')
			print(f'Number of training examples: {n_training_examples}')
			print(f'Max points: {max_points}')

			# Stack into arrays
			(
				all_positions,
				all_normals,
				all_neighbors,
				all_masks,
				all_mesh_vertices,
				all_mesh_faces,
				all_face_masks,
			) = generate_input_features_with_meshes(all_data, max_points)

			# Save data
			save_dir = save_filepath.split('/')[0]
			filename = save_filepath.split('/')[-1]
			rootname = filename.split('_')[0]
			suffix = filename.split('_')[-1].split('.')[0]  # 'train', 'val'

			filepath = save_dir + '/' + rootname + f'_{suffix}_mode{mode}_{n_npz}.npz'
			save_mesh_training_data(
				filepath,
				all_positions,
				all_normals,
				all_neighbors,
				all_masks,
				all_mesh_vertices,
				all_mesh_faces,
				all_face_masks,
			)

			# Reset counters
			n_training_examples = 0
			n_meshes_processed = 0
			n_npz += 1
			all_data = []
			max_points = 0

			n_files_saved += 1
			if n_files_saved == n_files:
				break

	t1 = time.time()
	print(f'Generation complete: {t1 - t0:.2f}s')


def generate_all_training_datasets(n_views: int, mode: int):
	"""
	Generate training data, consisting of

	{
		'positions': (n_shapes, max_points, 3),
		'normals': (n_shapes, max_points, 3),
		'neighbors': (n_shapes, max_points, k),
		'masks': (n_shapes, max_points),

		'mesh_vertices': (n_shapes, max_vertices, 3),
		'mesh_faces': (n_shapes, max_faces, 3),
		'face_masks': (n_shapes, max_faces),
	}

	This assumes that mesh data has already been retrieved and compressed into local NPZ files,
	e.g. the functions get_Thingi10k(), get_ABC(), etc. have already been run.
	"""

	load_dir = 'datasets'

	# ABC meshes
	ABC_mesh_files = sorted(glob.glob(os.path.join('datasets/final_train_meshes', 'ABC*_train_meshes.npz')))
	for filepath in ABC_mesh_files:
		filename = filepath.split('/')[-1]
		rootname = filename.split('_')[0]
		t0 = time.time()
		generate_scanned_mesh_training_dataset(
			n_views=n_views,
			npz_filepath=filepath,
			save_filepath=load_dir + f'/{rootname}_train.npz',
			meshes_per_npz=meshes_per_training_npz,
			mode=mode,
			n_files=2000 // meshes_per_training_npz,
		)
		t1 = time.time()
		print(f'ABC processing time: {t1 - t0}s')

	# Blobs
	blob_mesh_files = sorted(glob.glob(os.path.join('datasets/final_train_meshes', 'blob*train_meshes.npz')))
	for filepath in blob_mesh_files:
		filename = filepath.split('/')[-1]
		rootname = filename.split('_')[0]
		t0 = time.time()
		generate_scanned_mesh_training_dataset(
			n_views=n_views,
			npz_filepath=filepath,
			save_filepath=load_dir + f'/{rootname}_train.npz',
			meshes_per_npz=meshes_per_training_npz,
			mode=mode,
			n_files=2000 // meshes_per_training_npz,
		)
		t1 = time.time()
		print(f'Blob processing time: {t1 - t0}s')


def generate_pretraining_dataset():
	"""
	Generate a training dataset of just a single, smooth blobby shape.
	"""
	t0 = time.time()

	shape = TriangleMesh.read_OBJ('data/blob1024.obj')
	shape.query_dtype = query_dtype
	shape.center_and_scale()
	vertices = _vertices_numpy(shape)
	faces = _faces_numpy(shape)

	# Sample point cloud from mesh
	points, normals = shape.sample_point_cloud(
		n_cameras=n_cameras,
		sensor_size=sensor_size,
		grid_spacing=grid_spacings[0],
		seed=seed,
		sigma_p=0.0,
		sigma_n=0.0,
	)

	print(f'  Original number of points: {points.shape[1]}...', end='')

	# Subsample to target number of points (the C++ farthest-point sampling; undefined name in the original).
	selected_indices = subsample_point_cloud(points, max_points_per_pointcloud)
	points = points[:, selected_indices]
	normals = normals[:, selected_indices]

	print(f'  Subsampled to: {points.shape[1]}')

	# Compute neighbors for point cloud
	neighbors = get_neighbors(points, k_neighbors)  # (n_points, k)

	n_points = points.shape[1]
	max_points = n_points

	# Store data
	all_data = [
		{
			'positions': config.to_numpy(points).astype(np.float64),
			'normals': config.to_numpy(normals).astype(np.float64),
			'neighbors': _neighbors_numpy(neighbors),
			'vertices': vertices,
			'faces': faces,
		}
	]

	# Stack into arrays
	(
		all_positions,
		all_normals,
		all_neighbors,
		all_masks,
		all_mesh_vertices,
		all_mesh_faces,
		all_face_masks,
	) = generate_input_features_with_meshes(all_data, max_points)

	# Save data
	filepath = 'datasets/pretraining_blob.npz'
	save_mesh_training_data(
		filepath,
		all_positions,
		all_normals,
		all_neighbors,
		all_masks,
		all_mesh_vertices,
		all_mesh_faces,
		all_face_masks,
	)

	t1 = time.time()
	print(f'Test training dataset processing time: {t1 - t0}s')


def generate_validation_datasets():
	"""
	(The original passes only_save_one=True, which generate_scanned_mesh_training_dataset does not take, and omits the
	required mode and n_files: fixed as mode=3, n_files=1. Output: datasets/ABC.._validation_mode3_0.npz.)
	"""
	n_validation = 100
	load_dir = 'datasets'
	ABC_mesh_files = sorted(glob.glob(os.path.join(load_dir, 'ABC*components_test_meshes.npz')))
	filepath = ABC_mesh_files[0]
	filename = filepath.split('/')[-1]
	rootname = filename.split('_')[0]
	t0 = time.time()
	generate_scanned_mesh_training_dataset(
		n_views=4,
		npz_filepath=filepath,
		save_filepath=load_dir + f'/{rootname}_validation.npz',
		meshes_per_npz=n_validation,
		mode=3,
		n_files=1,
	)
	t1 = time.time()
	print(f'ABC processing time: {t1 - t0}s')


if __name__ == '__main__':
	# The C++ evaluated in double; the device is the GPU if there is one.
	config.set_device('cuda' if torch.cuda.is_available() else 'cpu', torch.float64)

	# Generate blobby training data
	generate_blob_dataset(
		n_blobs=3000,
		save_dir='datasets',
		name='blobs',
		alpha_range=(0.1, 0.7),
		scale_range=(0.5, 2.0),
		octaves_range=(1, 4),
		base_radius=1.0,
		grid_resolution=64,
		persistence=0.5,
		lacunarity=2.0,
		seed=42,
		train_test_split=0.8,
	)

	generate_pretraining_dataset()

	generate_all_training_datasets(n_views=4, mode=0)
	generate_all_training_datasets(n_views=4, mode=-1)
