"""
Torch port of src/pointsastori/shape_3d.py.

In the original this module wraps the C++ bindings (pat_bindings), converting (n, 3) NumPy arrays to Fortran-ordered
(3, n) Eigen matrices on every call. In the port the torch implementation (torus_distance.py) already works on (3, n)
device tensors, so the wrappers below forward to it; NumPy arrays enter through the import helpers (config.as_wide)
and leave through the export helpers (config.to_numpy, export_*).

Ported: point-cloud I/O, PLY export, k-NN, torus fitting, TorusDistanceField.
Not ported yet (C++ mesh / 2D-SDF code used to generate training data): TriangleMesh, SDF2D and subclasses, SDF3D,
sample_SDF, BoundingVolumeHierarchy.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from . import config
from . import torus_distance as _td
from .torus_distance import TorusDistanceField  # noqa: F401  (re-export, as in the original module)

# ========================================================================
# POINT CLOUDS
# ========================================================================


def point_cloud_bounding_box(positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Args:
		positions: (3, |P|) tensor

	Returns:
		(bbox_min, bbox_max) where both bbox_min, bbox_max are (3,) tensors
	"""
	return _td.point_cloud_bounding_box(config.check_wide(positions, 3, 'positions'))


class PointCloud3D:
	"""Host-side (NumPy, (n, 3)) point cloud container used for file I/O, as in the original."""

	def __init__(self, points: np.ndarray, normals: np.ndarray, colors: np.ndarray = None) -> None:
		self.points = points
		self.normals = normals
		self.colors = colors
		self.size = len(self.points)

	def center_and_scale(self):
		centroid = np.mean(self.points, axis=0)
		centered_points = self.points - centroid
		max_range = np.max(np.ptp(centered_points, axis=0))
		self.points = centered_points / max_range

	def to_device(self) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Added in the port: import to the configured device as (3, n) points and normals."""
		return config.as_wide(self.points, 3), config.as_wide(self.normals, 3)

	@staticmethod
	def read(filepath: str) -> PointCloud3D:
		extension = filepath.split('.')[-1]
		if extension == 'obj' or extension == 'pc':
			points = []
			normals = []
			colors = []
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'v':
						p = list(map(float, parts[1:4]))
						points.append(p)
					elif parts[0] == 'vn':
						n = list(map(float, parts[1:4]))
						n = np.asarray(n)
						n /= np.linalg.norm(n)
						normals.append(n)
					elif parts[0] == 'vt' or parts[0] == 'vc':
						c = list(map(float, parts[1:4]))
						colors.append(c)
			if len(colors) > 0:
				return PointCloud3D(np.array(points), np.array(normals), np.array(colors))
			if len(normals) == 0:
				raise ValueError('No normals found in point cloud')
			return PointCloud3D(np.array(points), np.array(normals))
		elif extension == 'ply':
			return PointCloud3D.read_PLY(filepath)
		else:
			raise ValueError(f'Unsupported file format: {extension}')

	@staticmethod
	def read_PLY(filepath: str) -> PointCloud3D:
		"""
		The original reads PLY with open3d. The port uses plyfile (already a dependency of the original package) and
		falls back to open3d if plyfile is missing.
		"""
		try:
			from plyfile import PlyData
		except ImportError:
			import open3d as o3d

			pcd = o3d.io.read_point_cloud(filepath)
			points = np.asarray(pcd.points)
			colors = np.asarray(pcd.colors)
			normals = np.asarray(pcd.normals)
			return PointCloud3D(points, normals, colors)

		vertex = PlyData.read(filepath)['vertex']
		names = vertex.data.dtype.names
		points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1).astype(np.float64)
		normals = (
			np.stack([vertex['nx'], vertex['ny'], vertex['nz']], axis=-1).astype(np.float64)
			if 'nx' in names
			else np.zeros((0, 3))
		)
		colors = (
			np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=-1).astype(np.float64) / 255.0
			if 'red' in names
			else np.zeros((0, 3))
		)
		return PointCloud3D(points, normals, colors)

	def export(self, filepath: str, binary: bool = True):
		"""
		Export point cloud as PLY file (binary or ASCII).
		(Fixes vs. the original: color properties are declared in the vertex dtype, and the ASCII fallback refers to
		self.normals / self.colors.)

		Args:
			filepath: path to save PLY file
			binary: if True, use binary format; otherwise ASCII
		"""
		has_normals = self.normals is not None and len(self.normals) == len(self.points)
		has_colors = self.colors is not None and len(self.colors) == len(self.points)
		try:
			from plyfile import PlyData, PlyElement

			n_points = len(self.points)

			# Build vertex dtype based on available data
			dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

			if has_normals:
				dtype_list.extend([('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
			if has_colors:
				dtype_list.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

			# Create structured array
			vertex_data = np.empty(n_points, dtype=dtype_list)
			vertex_data['x'] = self.points[:, 0]
			vertex_data['y'] = self.points[:, 1]
			vertex_data['z'] = self.points[:, 2]

			if has_normals:
				vertex_data['nx'] = self.normals[:, 0]
				vertex_data['ny'] = self.normals[:, 1]
				vertex_data['nz'] = self.normals[:, 2]

			if has_colors:
				colors = self.colors * 255 if self.colors.max() <= 1.0 else self.colors
				vertex_data['red'] = colors[:, 0]
				vertex_data['green'] = colors[:, 1]
				vertex_data['blue'] = colors[:, 2]

			# Create PLY element
			vertex_element = PlyElement.describe(vertex_data, 'vertex')

			# Write PLY file
			ply_data = PlyData([vertex_element], text=not binary)
			ply_data.write(filepath)

		except ImportError:
			print('Warning: plyfile not available, writing PLY in ASCII format')

			with open(filepath, 'w') as f:
				# Header
				f.write('ply\n')
				f.write('format ascii 1.0\n')
				f.write(f'element vertex {len(self.points)}\n')
				f.write('property float x\n')
				f.write('property float y\n')
				f.write('property float z\n')

				if has_normals:
					f.write('property float nx\n')
					f.write('property float ny\n')
					f.write('property float nz\n')

				if has_colors:
					f.write('property uchar red\n')
					f.write('property uchar green\n')
					f.write('property uchar blue\n')

				f.write('end_header\n')

				# Vertices
				for i, p in enumerate(self.points):
					line = f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}'

					if has_normals:
						n = self.normals[i]
						line += f' {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}'

					if has_colors:
						c = self.colors[i]
						if c.max() <= 1.0:
							c = (c * 255).astype(int)
						line += f' {int(c[0])} {int(c[1])} {int(c[2])}'

					f.write(line + '\n')


def write_point_cloud(points: np.ndarray, normals: np.ndarray, filepath: str) -> None:
	"""
	Write point cloud as OBJ. Accepts (n, 3) NumPy arrays or (3, n) device tensors (exported here).
	"""
	if isinstance(points, torch.Tensor):
		points = config.to_numpy(points)
	if isinstance(normals, torch.Tensor):
		normals = config.to_numpy(normals)
	with open(filepath, 'w') as f:
		for p in points:
			f.write(f'v {p[0]} {p[1]} {p[2]}\n')
		for n in normals:
			f.write(f'vn {n[0]} {n[1]} {n[2]}\n')


# ========================================================================
# MESHES (export only; TriangleMesh / Mesh3D is not ported)
# ========================================================================


def export_PLY(vertices: np.ndarray, faces: np.ndarray, filepath: str, binary: bool = True):
	"""
	Free-function port of TriangleMesh.export_PLY (network.export_torus calls `export_PLY`, which the original module
	never defines).

	Args:
		vertices: (n_verts, 3)
		faces: (n_faces, 3) for triangles or (n_faces, 4) for quads
	"""
	from plyfile import PlyData, PlyElement

	vertices = np.asarray(vertices)
	faces = np.asarray(faces)
	n_verts_per_face = faces.shape[1]
	if n_verts_per_face not in (3, 4):
		raise ValueError(f'Faces must have 3 or 4 vertices, got {n_verts_per_face}')

	vertex_data = np.empty(len(vertices), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
	vertex_data['x'] = vertices[:, 0]
	vertex_data['y'] = vertices[:, 1]
	vertex_data['z'] = vertices[:, 2]
	face_data = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (n_verts_per_face,))])
	face_data['vertex_indices'] = faces

	ply_data = PlyData(
		[PlyElement.describe(vertex_data, 'vertex'), PlyElement.describe(face_data, 'face')], text=not binary
	)
	ply_data.write(filepath)


def export_PLY_with_colors(
	vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, filepath: str, binary: bool = True
):
	"""
	Export mesh with vertex colors to PLY.

	Args:
		vertices: (n_verts, 3)
		faces: (n_faces, 3) for triangles or (n_faces, 4) for quads
		colors: (n_verts, 3) RGB in [0, 255] or [0, 1]
		filepath: output path
		binary: use binary format
	"""
	try:
		from plyfile import PlyData, PlyElement

		# Ensure colors are uint8
		if colors.max() <= 1.0:
			colors = (colors * 255).astype(np.uint8)
		else:
			colors = colors.astype(np.uint8)

		# Create vertex element with colors
		vertex_data = np.empty(
			len(vertices), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
		)
		vertex_data['x'] = vertices[:, 0]
		vertex_data['y'] = vertices[:, 1]
		vertex_data['z'] = vertices[:, 2]
		vertex_data['red'] = colors[:, 0]
		vertex_data['green'] = colors[:, 1]
		vertex_data['blue'] = colors[:, 2]

		vertex_element = PlyElement.describe(vertex_data, 'vertex')

		# Create face element - handle both triangles and quads
		n_verts_per_face = faces.shape[1]

		if n_verts_per_face == 3:
			face_data = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (3,))])
		elif n_verts_per_face == 4:
			face_data = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (4,))])
		else:
			raise ValueError(f'Faces must have 3 or 4 vertices, got {n_verts_per_face}')

		face_data['vertex_indices'] = faces

		face_element = PlyElement.describe(face_data, 'face')

		# Write PLY
		ply_data = PlyData([vertex_element, face_element], text=not binary)
		ply_data.write(filepath)

	except ImportError:
		print('Warning: plyfile not available, writing ASCII PLY manually')

		n_verts_per_face = faces.shape[1]

		# Ensure colors are integers
		if colors.max() <= 1.0:
			colors = (colors * 255).astype(int)
		else:
			colors = colors.astype(int)

		with open(filepath, 'w') as f:
			f.write('ply\nformat ascii 1.0\n')
			f.write(f'element vertex {len(vertices)}\n')
			f.write('property float x\nproperty float y\nproperty float z\n')
			f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
			f.write(f'element face {len(faces)}\n')
			f.write('property list uchar int vertex_indices\n')
			f.write('end_header\n')

			for i, v in enumerate(vertices):
				c = colors[i]
				f.write(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n')

			for face in faces:
				f.write(f'{n_verts_per_face}')
				for idx in face:
					f.write(f' {idx}')
				f.write('\n')


# ========================================================================
# TORUS SDF EVALUATION
# ========================================================================


def get_neighbors(points: torch.Tensor, k_neighbors: int, outliers=None) -> torch.Tensor:
	"""Return a (|P|, k) tensor of k-nearest neighbor indices. points: (3, |P|)."""
	return _td.get_neighbors(points, k_neighbors, outliers)


def compute_optimal_radius(points: torch.Tensor, k_neighbors: int = 64) -> torch.Tensor:
	"""
	Returns a 0-d tensor (no host sync). set_radius_evaluation() needs a host float: converting this value with
	float(...) is a deliberate sync left to the caller.
	"""
	D = _td.get_average_neighbor_distance(points, k_neighbors)
	lam = 1e3 / D
	return 128.0 / lam


def fit_tori_from_forms(
	points: torch.Tensor,
	normals: torch.Tensor,
	coefficients: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Args: points, normals (3, |P|), coefficients (6, |P|)
	Returns: centers (3, |P|), axes (2, |P|), major_radii (|P|,), minor_radii (|P|,)
	"""
	return _td.fit_tori_from_forms(points, normals, coefficients)
