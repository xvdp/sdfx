"""
Torch port of src/pointsastori/shape_3d.py.

In the original this module wraps the C++ bindings (pat_bindings), converting (n, 3) NumPy arrays to Fortran-ordered
(3, n) Eigen matrices on every call. In the port the torch implementation (torus_distance.py) already works on (3, n)
device tensors, so the wrappers below forward to it; NumPy arrays enter through the import helpers (config.as_wide)
and leave through the export helpers (config.to_numpy, export_*).

Contents: point-cloud I/O, PLY export, TriangleMesh (with the C++ Mesh3D implementation), k-NN, torus fitting,
TorusDistanceField, the 2D SDF wrappers SDF2D + subclasses, load_SDF, sample_SDF, SDF3D (the torch classes they
wrap are in shape_2d.py), and BoundingVolumeHierarchy (bvh.py).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from . import config
from . import torus_distance as _td
from .torus_distance import TorusDistanceField  # noqa: F401  (re-export, as in the original module)
from .bvh import BoundingVolumeHierarchy  # noqa: F401  (defined in bvh.py; part of this module in the original)

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
# MESHES (export; TriangleMesh below)
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
# TRIANGLE MESHES (C++ Mesh3D, src/cpp/shape_3d.h/.cpp, merged with the Python wrapper TriangleMesh)
# ========================================================================

import sys as _sys  # noqa: E402

from . import cpp_random as _cpp_random  # noqa: E402
from . import shape_2d as _shape_2d  # noqa: E402
from .training import mesh_sampling as _mesh_sampling  # noqa: E402

_FLT_EPSILON = float(np.finfo(np.float32).eps)  # fcpw::epsilon = std::numeric_limits<float>::epsilon()
_UNIT_CUBE_DIAMETER = 2.0 * float(np.sqrt(3.0))  # the cameras use the bounding sphere of the unit cube


def _host_vector3(x) -> np.ndarray:
	"""Parameter vectors (box corners) are host values: (3,) float64 (a tensor argument is read once)."""
	if isinstance(x, torch.Tensor):
		x = x.detach().cpu()
	return np.asarray(x, dtype=np.float64).reshape(3)


def _to_device(x, dtype: torch.dtype = None) -> torch.Tensor:
	"""Host draws -> configured device (one transfer)."""
	dtype = config.get_dtype() if dtype is None else dtype
	return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=config.get_device())


def _eigen_squared_norm(v: torch.Tensor) -> torch.Tensor:
	"""
	Eigen's squaredNorm() of 3-vectors along the component axis of (3, ...), in its summation order (the order of
	the C++ library build, so that the float32 queries round as FCPW's): (x² + y²) + z² for double (an SSE packet,
	then the last term), x² + (y² + z²) for float (unrolled in halves). (Eigen's dot() is (x + y) + z for both.)
	"""
	sq = v * v
	if v.dtype == torch.float32:
		return sq[0] + (sq[1] + sq[2])
	return (sq[0] + sq[1]) + sq[2]


def _sqrt_rn(x: torch.Tensor) -> torch.Tensor:
	"""
	Correctly rounded √x in float64. torch's CPU sqrt is only faithful (measured with torch 2.14 on this CPU: 0.7% of
	random inputs one ulp off, float32 and float64), the C++ / NumPy sqrt is correctly rounded. One correction step:
	with s = sqrt(x), s² = p + e exactly (Dekker's product with Veltkamp's split, no FMA needed), the residual
	r = x - s² = (x - p) - e (x - p is exact, Sterbenz), and s moves to its neighbor s ± d when x lies beyond the
	midpoint, i.e. when r > s d + d²/4 (d = next(s) - s) or r < -(s d' - d'²/4) (d' = s - previous(s)). NaN / inf /
	zero inputs keep torch's value.
	"""
	s = torch.sqrt(x)
	c = s * 134217729.0  # 2^27 + 1
	hi = c - (c - s)
	lo = s - hi
	p = s * s
	e = ((hi * hi - p) + 2.0 * hi * lo) + lo * lo
	r = (x - p) - e
	up = torch.nextafter(s, torch.full_like(s, torch.inf))
	down = torch.nextafter(s, torch.zeros_like(s))
	d_up = up - s
	d_down = s - down
	s = torch.where(r > s * d_up + 0.25 * d_up * d_up, up, s)
	return torch.where(r < -(s * d_down - 0.25 * d_down * d_down), down, s)


def _sqrt(x: torch.Tensor) -> torch.Tensor:
	"""
	√x correctly rounded, as the C++: float64 by _sqrt_rn; float32 as sqrt in float64 rounded to float32 (exact for
	sqrt). torch's own sqrt is not always correctly rounded on the CPU (see _sqrt_rn); in float32 this would break the
	bit-for-bit agreement of the float32 queries with FCPW.
	"""
	if x.dtype == torch.float32:
		return _sqrt_rn(x.double()).float()
	if x.dtype == torch.float64:
		return _sqrt_rn(x)
	return torch.sqrt(x)


def _normalized(v: torch.Tensor) -> torch.Tensor:
	"""Eigen's normalized() along the component axis of (3, ...): v / |v| if |v|² > 0, else v unchanged."""
	z = _eigen_squared_norm(v)
	return torch.where(z > 0, v / _sqrt(z), v)


def sample_bounding_box(n_points: int, bbox_min, bbox_max, seed: int) -> torch.Tensor:
	"""
	(3, n_points) points uniformly distributed in the box [bbox_min, bbox_max].

	Declared in src/cpp/shape_3d.h but never defined (nor bound) in the original. Defined here with the evident
	semantics: std::mt19937(seed) and, per point, the x, y, z coordinates drawn in this order from
	std::uniform_real_distribution(bbox_min[c], bbox_max[c]).
	"""
	lo = _host_vector3(bbox_min)
	hi = _host_vector3(bbox_max)
	rng = _cpp_random.MT19937(seed)
	u = rng.canonical(3 * n_points).reshape(n_points, 3)
	return _to_device((u * (hi - lo) + lo).T)


def _perturbations(rng: _cpp_random.MT19937, n_points: int, sigma_p: float, sigma_n: float):
	"""
	Replay of the C++ noise loop

		std::normal_distribution position_perturbation{0.0, sigma_p};
		std::normal_distribution normal_perturbation{0.0, sigma_n};
		for (int i = 0; i < n_points; i++) {
			Eigen::Vector3d position_offset(position_perturbation(rng), position_perturbation(rng), position_perturbation(rng));
			Eigen::Vector3d normal_offset(normal_perturbation(rng), normal_perturbation(rng), normal_perturbation(rng));
			...

	libstdc++'s polar method makes values in pairs and each distribution object keeps the second value of its last
	pair, so the engine is consumed (in call order) by every other call of each object. GCC evaluates the three
	constructor arguments right to left: the first value drawn is the z component.

	Returns:
		position_offsets, normal_offsets: (3, n_points) float64 host arrays, N(0, sigma_p²), N(0, sigma_n²)
	"""
	calls = np.tile(np.array([0, 0, 0, 1, 1, 1]), n_points)  # distribution object of each call, in call order
	m = np.empty_like(calls)  # index of each call among the calls of its object
	for obj in (0, 1):
		mine = calls == obj
		m[mine] = np.arange(int(mine.sum()))
	generating = (m % 2) == 0  # the object has no saved value: a new pair is drawn from the engine
	rank = np.cumsum(generating) - 1  # engine order of the pair drawn by each generating call
	n_pairs = int(generating.sum())
	z = _cpp_random.Normal(0.0, 1.0)(rng, 2 * n_pairs).reshape(n_pairs, 2) if n_pairs else np.zeros((0, 2))
	offsets = []
	for obj, sigma in ((0, sigma_p), (1, sigma_n)):
		values = z[rank[(calls == obj) & generating]].reshape(-1)[: 3 * n_points]  # (y·m, x·m) of consecutive pairs
		values = values * float(sigma) + 0.0  # __ret * stddev + mean
		offsets.append(values.reshape(n_points, 3)[:, ::-1].T.copy())
	return offsets[0], offsets[1]


def _shuffled_indices(n: int, rng: _cpp_random.MT19937) -> list:
	"""std::iota(indices, 0) then std::shuffle(indices, rng)."""
	return _cpp_random.shuffle(list(range(n)), rng)


def _flip_signs(n: int, n_to_flip: int, rng: _cpp_random.MT19937) -> np.ndarray:
	"""The C++ flipping step: indices 0..n-1 shuffled, the normals at the first n_to_flip of them negated."""
	sign = np.ones(n)
	sign[_shuffled_indices(n, rng)[:n_to_flip]] = -1.0
	return sign


def _discrete_distribution(weights: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
	"""
	std::discrete_distribution<int>(weights) applied to canonical draws u (libstdc++, n >= 2 weights): p = w / Σw
	(Σ accumulated left to right), cp = partial sums of p with cp[-1] = 1, index = lower_bound(cp, u), i.e. the first
	i with cp[i] >= u: index i is drawn with probability w_i / Σw. On the device; torch.cumsum is the sequential sum
	of std::accumulate / std::partial_sum on the CPU.
	"""
	total = torch.cumsum(weights, dim=0)[-1]
	cp = torch.cumsum(weights / total, dim=0)
	cp = torch.cat([cp[:-1], torch.ones_like(cp[-1:])])
	return torch.searchsorted(cp, u.to(cp.dtype), side='left')


def _closest_point_on_triangle_fcpw(
	x: torch.Tensor, pa: torch.Tensor, pb: torch.Tensor, pc: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Closest point on triangle (pa, pb, pc) to x and its squared distance, broadcasting over the trailing axes of the
	(3, ...) arguments: FCPW's findClosestPointTriangle (Ericson, Real-Time Collision Detection, 5.1.5), with its
	operation order (FCPW returns the distance, the square root of this squared norm). The Voronoi region of x is tested in the order vertex a, b, c, edge ab, ac, bc, face, with
	d1 = ab·ax, d2 = ac·ax, d3 = ab·bx, d4 = ac·bx, d5 = ab·cx, d6 = ac·cx and va = d3 d6 - d5 d4, vb = d5 d2 - d1 d6,
	vc = d1 d4 - d3 d2; in the face region the point is a + ab v + ac w, v = vb / (va + vb + vc), w = vc / (va + vb + vc).

	Returns:
		closest points (3, ...), squared distances (...) (NaN for a degenerate triangle whose face region is selected)
	"""
	dot = _shape_2d.dot
	ab = pb - pa
	ac = pc - pa
	ax = x - pa
	d1 = dot(ab, ax)
	d2 = dot(ac, ax)
	bx = x - pb
	d3 = dot(ab, bx)
	d4 = dot(ac, bx)
	cx = x - pc
	d5 = dot(ab, cx)
	d6 = dot(ac, cx)
	vc = d1 * d4 - d3 * d2
	vb = d5 * d2 - d1 * d6
	va = d3 * d6 - d5 * d4

	in_a = (d1 <= 0.0) & (d2 <= 0.0)
	in_b = (d3 >= 0.0) & (d4 <= d3)
	in_c = (d6 >= 0.0) & (d5 <= d6)
	in_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
	in_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
	in_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
	at_vertex = in_a | in_b | in_c

	# point = (base + e1 * c1) + ac * c2, region by region (assigned in reverse order of the tests: the first test
	# that holds wins)
	denom = 1.0 / (va + vb + vc)
	c1 = vb * denom
	c2 = vc * denom
	c1 = torch.where(in_bc, (d4 - d3) / ((d4 - d3) + (d5 - d6)), c1)
	c1 = torch.where(in_ac, d2 / (d2 - d6), c1)
	c1 = torch.where(in_ab, d1 / (d1 - d3), c1)
	c1 = torch.where(at_vertex, 0.0, c1)
	c2 = torch.where(in_ab | in_ac | in_bc | at_vertex, 0.0, c2)
	e1 = torch.where(in_bc, pc - pb, ab)
	e1 = torch.where(in_ac, ac, e1)
	e1 = torch.where(in_ab | at_vertex, ab, e1)
	base = torch.where(in_bc, pb, pa)
	base = torch.where(in_ab | in_ac, pa, base)
	base = torch.where(in_c, pc, base)
	base = torch.where(in_b, pb, base)
	base = torch.where(in_a, pa, base)
	point = (base + e1 * c1) + ac * c2
	return point, _eigen_squared_norm(x - point)


class _BoxQueryBVH:
	"""
	Host port of the C++ triangle BVH (src/cpp/bvh.cpp), restricted to what Mesh3D uses: the top-down build
	(build_recursive: leaves of <= 4 triangles, 12-bin SAH split per axis, then the original's partition at the
	centroid of the primitive at the split position) and find_primitives_in_box (leaves overlapping a box, in
	depth-first left-to-right order). Mesh3D::sample_points_and_normals_uniformly with a box samples among the
	triangles of these leaves, in this order, so reproducing its draws needs this tree (points_as_tori.bvh.BVH builds
	a different one, an LBVH).
	"""

	def __init__(self, vertices: np.ndarray, faces: np.ndarray):
		tri = vertices[faces]  # (F, 3 corners, 3)
		self.centroids = ((tri[:, 0] + tri[:, 1]) + tri[:, 2]) / 3.0
		self.tri_min = tri.min(axis=1)
		self.tri_max = tri.max(axis=1)
		self.primitive_indices = list(range(len(faces)))
		self._centroid_axis = [self.centroids[:, a].tolist() for a in range(3)]
		self.nodes = []  # [aabb_min, aabb_max, left_child, right_child, prim_start, prim_count], depth-first order
		self._build(len(faces))

	def _build(self, n_primitives: int) -> None:
		# build_recursive(0, n), with an explicit stack: nodes are created in the same (pre-)order
		stack = [(0, n_primitives, -1, 0)]
		while stack:
			start, end, parent, side = stack.pop()
			node_idx = len(self.nodes)
			idx = self.primitive_indices[start:end]
			if idx:
				aabb = (self.tri_min[idx].min(axis=0), self.tri_max[idx].max(axis=0))
			else:
				aabb = (np.full(3, np.inf), np.full(3, -np.inf))
			node = [aabb[0], aabb[1], -1, -1, start, end - start]
			self.nodes.append(node)
			if parent >= 0:
				self.nodes[parent][2 + side] = node_idx

			# Create leaf if small enough
			if end - start <= 4:
				continue
			centroid_min = self.centroids[idx].min(axis=0)
			centroid_max = self.centroids[idx].max(axis=0)
			# Check if all centroids are at same point
			d = (centroid_max - centroid_min).tolist()
			if np.sqrt((d[0] * d[0] + d[1] * d[1]) + d[2] * d[2]) < 1e-10:
				continue

			axis, mid = self._find_best_split(start, end, centroid_min, centroid_max)
			# Ensure valid split
			mid = max(start + 1, min(mid, end - 1))
			# Partition primitives
			split_value = self._centroid_axis[axis][self.primitive_indices[mid]]
			mid = self._partition(start, end, axis, split_value)
			# Ensure we actually split
			if mid <= start or mid >= end:
				mid = (start + end) // 2
			node[4], node[5] = 0, 0
			stack.append((mid, end, node_idx, 1))
			stack.append((start, mid, node_idx, 0))

	@staticmethod
	def _surface_area(aabb_min, aabb_max) -> float:
		d = (aabb_max - aabb_min).tolist()
		return 2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])

	def _find_best_split(self, start: int, end: int, centroid_min, centroid_max):
		n_bins = 12
		best_cost = np.inf
		best_axis = 0
		best_split = start + 1
		idx = self.primitive_indices[start:end]
		for axis in range(3):
			if centroid_max[axis] - centroid_min[axis] < 1e-10:
				continue
			# Bin primitives
			value_range = float(centroid_max[axis] - centroid_min[axis])
			centroid_values = self.centroids[idx, axis]
			bins = np.minimum(
				((centroid_values - centroid_min[axis]) / value_range * n_bins).astype(np.int64), n_bins - 1
			)
			bin_counts = np.bincount(bins, minlength=n_bins)
			bin_min = np.full((n_bins, 3), np.inf)
			bin_max = np.full((n_bins, 3), -np.inf)
			np.minimum.at(bin_min, bins, self.tri_min[idx])
			np.maximum.at(bin_max, bins, self.tri_max[idx])
			# bins [0, s) and [s, n_bins) (empty bins hold +-inf and leave the min / max unchanged)
			left_min = np.minimum.accumulate(bin_min, axis=0)
			left_max = np.maximum.accumulate(bin_max, axis=0)
			right_min = np.minimum.accumulate(bin_min[::-1], axis=0)[::-1]
			right_max = np.maximum.accumulate(bin_max[::-1], axis=0)[::-1]
			left_counts = np.cumsum(bin_counts)
			# Evaluate SAH for each split
			for split_bin in range(1, n_bins):
				left_count = int(left_counts[split_bin - 1])
				right_count = len(idx) - left_count
				if left_count > 0 and right_count > 0:
					left_sa = self._surface_area(left_min[split_bin - 1], left_max[split_bin - 1])
					right_sa = self._surface_area(right_min[split_bin], right_max[split_bin])
					cost = 0.5 + (left_count * left_sa + right_count * right_sa)
					if cost < best_cost:
						best_cost = cost
						best_axis = axis
						# Find actual split point
						split_value = centroid_min[axis] + (split_bin / float(n_bins)) * value_range
						best_split = start + int(np.count_nonzero(centroid_values < split_value))
		return best_axis, best_split

	def _partition(self, start: int, end: int, axis: int, split_value: float) -> int:
		c = self._centroid_axis[axis]
		ind = self.primitive_indices
		left = start
		right = end - 1
		while left < right:
			while left < end and c[ind[left]] < split_value:
				left += 1
			while right >= start and c[ind[right]] >= split_value:
				right -= 1
			if left < right:
				ind[left], ind[right] = ind[right], ind[left]
				left += 1
				right -= 1
		return left

	def find_primitives_in_box(self, box_min: np.ndarray, box_max: np.ndarray) -> list:
		result = []
		stack = [0] if self.nodes else []
		while stack:
			aabb_min, aabb_max, left, right, prim_start, prim_count = self.nodes[stack.pop()]
			# Early exit if node's AABB doesn't intersect query box
			if not (np.all(aabb_min <= box_max) and np.all(aabb_max >= box_min)):
				continue
			# If leaf node, add all primitives
			if left == -1 and right == -1:
				result.extend(self.primitive_indices[prim_start : prim_start + prim_count])
				continue
			# Recurse on children (left first)
			if right != -1:
				stack.append(right)
			if left != -1:
				stack.append(left)
		return result


class TriangleMesh:
	"""
	Triangle mesh on the configured device: vertices (3, V) and faces (3, F) int64 (wide layout, see config.py).

	Port of the C++ Mesh3D (src/cpp/shape_3d.h/.cpp) and of the Python class TriangleMesh that wrapped it
	(src/pointsastori/shape_3d.py, the C++ object in `bound_object`): the wrapper's methods and arguments, the C++
	behavior. Layout rule of the port: (n, 3) arrays at construction / import (vertices, faces, read_OBJ); every
	method takes and returns (3, n) device tensors (queries, samples, normals); write_OBJ / export_PLY export files.
	Attributes vertices, faces, vertex_normals are the device tensors (the wrapper's were (n, 3) NumPy arrays).

	Queries, FCPW in the C++, are exact brute force over all faces, chunked over queries (config.CHUNK_ELEMENTS
	(query, face) pairs per chunk):
	  * closest point / unsigned distance: FCPW's closest point on a triangle for every pair, minimum over faces;
	  * ray casting: FCPW's Möller–Trumbore test for every pair, nearest hit;
	  * generalized winding number w(q) = Σ_f Ω_f(q), the solid angle of face f seen from q,
	        Ω_f = 2 atan2(det[a b c], |a||b||c| + (a·b)|c| + (b·c)|a| + (c·a)|b|),  a, b, c = face vertices - q
	    (Van Oosterom & Strackee 1983); w = 4π inside a closed outward-oriented mesh, 0 outside.
	FCPW stores the mesh and the queries in float32 (the C++ casts them). `query_dtype=torch.float32` reproduces
	this: closest points and ray hits are computed in float32 with FCPW's operation order (bit-identical to the C++
	on the CPU, up to which of two faces at exactly the same distance is reported). The default computes them in the
	configured dtype.

	Randomness: the samplers replay the C++ std::mt19937 streams with points_as_tori.cpp_random on the host (same
	draws, same order, one state per C++ distribution object) and move the draws to the device once. With the same
	seed and the same vertices they give the C++ samples bit for bit in float64 on the CPU (the ray-casting samplers
	with query_dtype=torch.float32). Square roots are correctly rounded as in the C++ (_sqrt). center_and_scale
	agrees with the C++ to ~3e-16 (see _centroid).

	Host reads (variable-size outputs): the number of ray hits (ray_intersections and the ray-casting samplers, per
	batch in sample_points_and_normals_uniformly_by_casting_rays), the mesh once for the original's BVH (samplers
	with a bounding box), the number of kept faces (cleaned_faces). The queries (evaluate_*) and center_and_scale
	read nothing.
	"""

	def __init__(self, vertices, faces, query_dtype: torch.dtype = None) -> None:
		"""
		Args:
			vertices: (n_vertices, 3) array (NumPy or torch), imported to the device as (3, n_vertices)
			faces: (n_faces, 3) integer array, imported as (3, n_faces) int64
			query_dtype: dtype of the closest-point and ray queries (FCPW in the C++); None: the configured dtype,
				torch.float32: FCPW's precision, as in the C++
		"""
		self.vertices = config.as_wide(vertices, 3)
		self.faces = config.as_wide(faces, 3, dtype=torch.long)
		self.query_dtype = query_dtype
		self.vertex_normals = self._compute_vertex_normals()
		self._bvh = None  # the C++ BVH for box queries, built on the host on first use

	def _compute_vertex_normals(self) -> torch.Tensor:
		"""Area-weighted vertex normals: Σ of the adjacent faces' (v1 - v0) × (v2 - v0), normalized; (3, V)."""
		n_vertices = self.vertices.shape[1]
		vertex_normals = torch.zeros((3, n_vertices), dtype=self.vertices.dtype, device=self.vertices.device)
		v0 = self.vertices[:, self.faces[0]]
		v1 = self.vertices[:, self.faces[1]]
		v2 = self.vertices[:, self.faces[2]]
		e1 = v1 - v0
		e2 = v2 - v0
		face_normals = _shape_2d.cross(e1, e2)
		vertex_normals.index_add_(1, self.faces[0], face_normals)
		vertex_normals.index_add_(1, self.faces[1], face_normals)
		vertex_normals.index_add_(1, self.faces[2], face_normals)
		norms = torch.clamp_min(_sqrt(_eigen_squared_norm(vertex_normals)), 1e-10)
		return vertex_normals / norms

	# ------------------------------------------------------------------
	# I/O
	# ------------------------------------------------------------------

	@staticmethod
	def read_OBJ(filepath: str) -> TriangleMesh:
		vertices = []
		faces = []
		with open(filepath, 'r') as file:
			for line in file:
				parts = line.strip().split()
				if not parts:
					continue
				elif parts[0] == 'v':
					p = list(map(float, parts[1:4]))
					vertices.append(p)
				elif parts[0] == 'f':
					idxs = [int(p.split('/')[0]) - 1 for p in parts[1:]]
					faces.append(idxs)
		return TriangleMesh(
			np.array(vertices, dtype=np.float64),
			np.array(faces, dtype=np.int64),
		)

	def write_OBJ(self, filepath: str):
		vertices = config.to_numpy(self.vertices)
		faces = config.to_numpy(self.faces)
		with open(filepath, 'w') as file:
			for vertex in vertices:
				file.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
			for face in faces:
				file.write(f'f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n')

	def export_PLY(self, filepath: str, binary: bool = True):
		"""(The original's ASCII fallback referred to undefined `vertices` / `faces`; here it writes the mesh.)"""
		vertices = config.to_numpy(self.vertices)
		faces = config.to_numpy(self.faces)
		try:
			export_PLY(vertices, faces, filepath, binary)
		except ImportError:
			# Fallback: write PLY manually
			print('Warning: plyfile not available, writing PLY in ASCII')

			with open(filepath, 'w') as f:
				# Header
				f.write('ply\n')
				f.write('format ascii 1.0\n')
				f.write(f'element vertex {len(vertices)}\n')
				f.write('property float x\n')
				f.write('property float y\n')
				f.write('property float z\n')
				f.write(f'element face {len(faces)}\n')
				f.write('property list uchar int vertex_indices\n')
				f.write('end_header\n')

				# Vertices
				for v in vertices:
					f.write(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')

				# Faces
				n_verts_per_face = faces.shape[1]
				for face in faces:
					f.write(f'{n_verts_per_face}')
					for idx in face:
						f.write(f' {idx}')
					f.write('\n')

	# ------------------------------------------------------------------
	# Geometry
	# ------------------------------------------------------------------

	def get_vertices(self) -> torch.Tensor:
		"""(3, n_vertices)"""
		return self.vertices

	def get_faces(self) -> torch.Tensor:
		"""(3, n_faces) int64"""
		return self.faces

	def bounding_box(self) -> Tuple[torch.Tensor, torch.Tensor]:
		"""(bbox_min, bbox_max), each (3,)"""
		return self.vertices.amin(dim=1), self.vertices.amax(dim=1)

	def center_and_scale(self):
		"""
		Center and scale mesh to [-1,1]^3 box: v <- (v - c) s with c the vertex centroid and s = 1 / max_i,k |v_ik - c_k|
		(s = 1 if that extent is <= 1e-10). In place, as the wrapper (the C++ returns a new Mesh3D).
		"""
		centroid = self._centroid()
		centered = self.vertices - centroid[:, None]
		max_extent = torch.abs(centered).amax()
		scale_factor = torch.where(max_extent > 1e-10, 1.0 / max_extent, 1.0)
		self.vertices = (centered * scale_factor).contiguous()
		self._bvh = None

	def remove_degenerate_faces(
		self, V: torch.Tensor = None, F: torch.Tensor = None, area_threshold: float = 1e-10
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Filter out degenerate (zero-area) faces so fast_winding_number behaves: drop faces with a repeated vertex index
		or with area 0.5 |(v1 - v0) × (v2 - v0)| < area_threshold. Returns (V unchanged, F_clean (3, F')).
		Host read: the number of kept faces (output size).

		The C++ constructor stores this result (cleaned_vertices / cleaned_faces) but no method uses it, so it does
		not affect any output. (The C++ also calls it with the 3 x n matrices where it expects n x 3 ones, so it
		reads 3 'faces' out of bounds; here the intended filter is computed.) Available as cleaned_vertices /
		cleaned_faces (computed on access).
		"""
		V = self.vertices if V is None else V
		F = self.faces if F is None else F
		v0, v1, v2 = V[:, F[0]], V[:, F[1]], V[:, F[2]]
		area = 0.5 * _sqrt(_eigen_squared_norm(_shape_2d.cross(v1 - v0, v2 - v0)))
		keep = (F[0] != F[1]) & (F[1] != F[2]) & (F[0] != F[2]) & ~(area < area_threshold)
		return V, F[:, keep]

	@property
	def cleaned_vertices(self) -> torch.Tensor:
		return self.remove_degenerate_faces()[0]

	@property
	def cleaned_faces(self) -> torch.Tensor:
		return self.remove_degenerate_faces()[1]

	# ------------------------------------------------------------------
	# Queries (FCPW / libigl in the C++): brute force over faces, chunked over queries
	# ------------------------------------------------------------------

	def _query_dtype(self) -> torch.dtype:
		return self.vertices.dtype if self.query_dtype is None else self.query_dtype

	def _queries_per_chunk(self) -> int:
		return max(1, config.CHUNK_ELEMENTS // max(1, self.faces.shape[1]))

	def _closest_points(self, queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		FCPW findClosestPoints: distance (Q,) from each query (3, Q) to the mesh and the closest point (3, Q),
		computed in the query dtype, returned in the dtype of `queries`. The nearest face minimizes the distance (the
		squared distance in float64: the square root is monotonic and is taken of the minimum only); NaN (degenerate
		triangles) is skipped as by FCPW's `d*d <= r²` test. Among faces at the same distance the first is reported
		(FCPW: the last one in its BVH order; their closest points can differ by rounding).
		"""
		qdtype = self._query_dtype()
		V = self.vertices.to(qdtype)
		pa = V[:, self.faces[0]][:, None, :]  # (3, 1, F)
		pb = V[:, self.faces[1]][:, None, :]
		pc = V[:, self.faces[2]][:, None, :]
		n_queries = queries.shape[1]
		distances = torch.empty(n_queries, dtype=queries.dtype, device=queries.device)
		points = torch.empty((3, n_queries), dtype=queries.dtype, device=queries.device)
		chunk = self._queries_per_chunk()
		for start in range(0, n_queries, chunk):
			x = queries[:, start : start + chunk].to(qdtype)[:, :, None]  # (3, Qc, 1)
			pt, d2 = _closest_point_on_triangle_fcpw(x, pa, pb, pc)  # (3, Qc, F), (Qc, F)
			d2 = torch.where(torch.isnan(d2), torch.inf, d2)
			# FCPW compares the float32 distances: faces at the same rounded distance tie
			key = _sqrt(d2) if qdtype == torch.float32 else d2
			idx = torch.argmin(key, dim=1)
			distances[start : start + chunk] = _sqrt(torch.gather(d2, 1, idx[:, None])[:, 0]).to(queries.dtype)
			pt = torch.gather(pt, 2, idx[None, :, None].expand(3, -1, 1))[..., 0]
			points[:, start : start + chunk] = pt.to(queries.dtype)
		return distances, points

	def _winding_numbers(self, queries: torch.Tensor) -> torch.Tensor:
		"""Σ_f Ω_f(q) (4π inside), all faces (training.mesh_sampling.gwn_mesh, the same formula as the C++)."""
		face_mask = torch.ones(self.faces.shape[1], dtype=torch.bool, device=self.faces.device)
		return _mesh_sampling.gwn_mesh(queries.to(self.vertices.dtype), self.vertices, self.faces, face_mask)

	def _cast_rays(self, origins: torch.Tensor, directions: torch.Tensor):
		"""
		FCPW Scene::intersect (nearest hit, no occlusion test), brute force, in the query dtype.

		Returns:
			hit (R,) bool, points (3, R), normals (3, R) (unit e1 × e2 of the hit face), directions as cast (3, R);
			points and normals in the configured dtype, undefined where hit is False
		"""
		dtype = origins.dtype
		qdtype = self._query_dtype()
		V = self.vertices.to(qdtype)
		pa = V[:, self.faces[0]]
		v1 = V[:, self.faces[1]] - pa
		v2 = V[:, self.faces[2]] - pa
		face_normals = _normalized(_shape_2d.cross(v1, v2))  # v1.cross(v2).normalized()
		pa_, v1_, v2_ = pa[:, None, :], v1[:, None, :], v2[:, None, :]
		dot, cross = _shape_2d.dot, _shape_2d.cross

		n_rays = origins.shape[1]
		hit = torch.empty(n_rays, dtype=torch.bool, device=origins.device)
		points = torch.empty((3, n_rays), dtype=dtype, device=origins.device)
		normals = torch.empty((3, n_rays), dtype=dtype, device=origins.device)
		chunk = self._queries_per_chunk()
		for start in range(0, n_rays, chunk):
			o = origins[:, start : start + chunk].to(qdtype)[:, :, None]  # (3, Rc, 1)
			d = directions[:, start : start + chunk].to(qdtype)[:, :, None]
			# Möller–Trumbore
			p = cross(d, v2_)
			det = dot(v1_, p)
			inv_det = 1.0 / det
			s = o - pa_
			v = dot(s, p) * inv_det
			q = cross(s, v1_)
			w = dot(d, q) * inv_det
			t = dot(v2_, q) * inv_det
			# ray and triangle are parallel if det is close to 0
			hits = ~(torch.abs(det) <= _FLT_EPSILON)
			hits &= ~((v < 0.0) | (v > 1.0))
			hits &= ~((w < 0.0) | ((v + w) > 1.0))
			hits &= t >= 0.0  # t <= tMax = inf
			t = torch.where(hits, t, torch.inf)
			idx = torch.argmin(t, dim=1)  # nearest hit
			hit[start : start + chunk] = hits.any(dim=1)
			v = torch.gather(v, 1, idx[:, None])[:, 0]
			w = torch.gather(w, 1, idx[:, None])[:, 0]
			points[:, start : start + chunk] = ((pa[:, idx] + v1[:, idx] * v) + v2[:, idx] * w).to(dtype)
			normals[:, start : start + chunk] = face_normals[:, idx].to(dtype)
		return hit, points, normals, directions.to(qdtype).to(dtype)

	def ray_intersections(self, ros: torch.Tensor, rds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Ray intersections. Returns (points, normals) for hits.

		Each ray o + t d (t >= 0) is tested against every face with Möller–Trumbore (FCPW's Triangle::intersect):
		e1 = v1 - v0, e2 = v2 - v0, p = d × e2, det = e1·p (parallel, no hit, if |det| <= FLT_EPSILON), s = o - v0,
		v = s·p / det, q = s × e1, w = d·q / det, t = e2·q / det; a hit needs 0 <= v, 0 <= w, v + w <= 1, t >= 0. The
		nearest hit is kept: point v0 + e1 v + e2 w, normal ±(e1 × e2) / |e1 × e2| with the sign making n·d < 0 (the
		sign of the normal aligns with the camera ray; relevant for inconsistently-oriented meshes).

		Args:
			ros, rds: (3, n_rays) ray origins and directions
		Returns:
			points, normals: (3, n_hits) in ray order. Host read: the number of hits (output size).
		"""
		ros = config.check_wide(ros, 3, 'ros')
		rds = config.check_wide(rds, 3, 'rds')
		hit, points, normals = self._ray_hits(ros, rds)
		return points[:, hit], normals[:, hit]

	def _ray_hits(self, ros: torch.Tensor, rds: torch.Tensor):
		"""ray_intersections before the compaction: hit (R,), points, normals (3, R) with oriented normals."""
		hit, points, normals, directions = self._cast_rays(ros, rds)
		s = torch.where(_shape_2d.dot(normals, directions) < 0.0, 1.0, -1.0).to(normals.dtype)
		return hit, points, normals * s

	def evaluate_signed_distance_single(self, query: torch.Tensor) -> torch.Tensor:
		"""
		Signed distance at one query (3,) -> 0-d tensor: the distance to the closest point, negative where the winding
		number Σ_f Ω_f > 2π (inside).
		"""
		query = config.check_wide(query, 3, 'query')[:, :1]
		gwn = self._winding_numbers(query)
		sgn = torch.where(gwn > 2.0 * np.pi, -1.0, 1.0).to(gwn.dtype)
		distances, _ = self._closest_points(query)
		return (distances * sgn)[0]

	def evaluate_signed_distance(self, q: torch.Tensor, parallelize: bool = True) -> torch.Tensor:
		"""
		Signed distance (n,) at queries q (3, n): distance to the closest point on the mesh, negative where the winding
		number W = Σ_f Ω_f / 4π > 0.5 (the C++ uses igl::fast_winding_number for W; here the exact sum, see
		evaluate_fast_gwn). `parallelize` only selected the C++ loop (same result).
		"""
		q = config.check_wide(q, 3, 'q')
		W = self.evaluate_fast_gwn(q)
		sgn = torch.where(W > 0.5, -1.0, 1.0).to(W.dtype)
		distances, _ = self._closest_points(q)
		return distances * sgn

	def evaluate_signed_distance_and_gradient(
		self, q: torch.Tensor, isovalue: float = 0.0
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Signed distance (n,) and gradient (3, n) at queries q (3, n): the gradient of the distance to the closest
		point p is s (q - p) / |q - p|, s = -1 inside (W > 0.5), +1 outside; it is undefined for points on the mesh,
		where Eigen's normalized() returns the zero vector. `isovalue` is unused (as in the original).
		"""
		q = config.check_wide(q, 3, 'q')
		print('Evaluating GWN...', file=_sys.stderr)
		W = self.evaluate_fast_gwn(q)
		print('Evaluating closest points...', file=_sys.stderr)
		distances, points = self._closest_points(q)
		print('Signing distances and computing gradients...', file=_sys.stderr)
		sgn = torch.where(W > 0.5, -1.0, 1.0).to(W.dtype)
		return distances * sgn, sgn * _normalized(q - points)

	def evaluate_signed_distance_gradient(self, q: torch.Tensor) -> torch.Tensor:
		_, grad = self.evaluate_signed_distance_and_gradient(q)
		return grad

	def evaluate_gwn(self, q: torch.Tensor) -> torch.Tensor:
		"""Generalized winding number as the sum of solid angles Σ_f Ω_f(q) (4π inside), q (3, n) -> (n,)."""
		return self._winding_numbers(config.check_wide(q, 3, 'q'))

	def evaluate_fast_gwn(self, q: torch.Tensor) -> torch.Tensor:
		"""
		Winding number normalized to 1 inside, W(q) = Σ_f Ω_f(q) / 4π, q (3, n) -> (n,).

		The C++ uses igl::fast_winding_number (Barill et al. 2018: Barnes–Hut tree, far faces approximated by dipole
		expansions); here W is the exact sum over all faces (the value that the approximation converges to).
		"""
		return self._winding_numbers(config.check_wide(q, 3, 'q')) / (4.0 * np.pi)

	# ------------------------------------------------------------------
	# Samplers
	# ------------------------------------------------------------------

	def _box_candidates(self, bbox_min: np.ndarray, bbox_max: np.ndarray) -> list:
		"""Faces of the C++ BVH leaves overlapping the box. Host read: the mesh, once (the tree is cached)."""
		if self._bvh is None:
			self._bvh = _BoxQueryBVH(config.to_numpy(self.vertices).astype(np.float64), config.to_numpy(self.faces))
		return self._bvh.find_primitives_in_box(bbox_min, bbox_max)

	def sample_points_and_normals_uniformly(
		self,
		n_points: int,
		seed: int = 0,
		bbox_min: np.ndarray = np.array([1, 1, 1]),
		bbox_max: np.ndarray = np.array([-1, -1, -1]),
		sigma_p: float = 0.0,
		sigma_n: float = 0.0,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Points distributed uniformly (w.r.t. area) on the mesh, with their face normals.

		A face is drawn with probability area / total area, then the point α v0 + β v1 + γ v2 with α = √u (1 - v),
		β = v √u, γ = 1 - α - β, u, v ~ U[0, 1), which is uniform on the triangle. If bbox_min <= bbox_max, the
		candidate faces are those of the leaves of the original's BVH that overlap the box (faces near the box; the
		samples are not clipped to it); otherwise (default: no bounding box) the whole mesh.

		Optional noise (if sigma_p > 0 or sigma_n > 0): p += δp, n += normalize(n + δn), δp ~ N(0, sigma_p² I),
		δn ~ N(0, sigma_n² I). As in the C++, the perturbed unit normal is added to the normal (|n| ≈ 2), not
		substituted for it.

		Returns:
			points, normals: (3, n_points)
		"""
		bbox_min = _host_vector3(bbox_min)
		bbox_max = _host_vector3(bbox_max)
		extent = bbox_max - bbox_min
		bbox_is_invalid = bool((extent < 0).any())

		rng = _cpp_random.MT19937(seed)

		# Compute face areas.
		if bbox_is_invalid:
			# No bounding box was provided; sample from entire mesh.
			face_indices = torch.arange(self.faces.shape[1], device=self.faces.device)
		else:
			# Identify faces within bounding box using BVH
			face_indices = _to_device(self._box_candidates(bbox_min, bbox_max), torch.long)
		n_faces = face_indices.shape[0]
		if n_faces == 0:
			raise ValueError('no faces to sample from (the C++ indexes an empty face list)')
		faces = self.faces[:, face_indices]
		v0 = self.vertices[:, faces[0]]
		v1 = self.vertices[:, faces[1]]
		v2 = self.vertices[:, faces[2]]
		area_normal = _shape_2d.cross(v1 - v0, v2 - v0)
		double_area = _sqrt(_eigen_squared_norm(area_normal))
		area_weights = 0.5 * double_area
		face_normals = area_normal / double_area  # normalized

		# Sample faces proportionally to area: per point, the face (one canonical draw; none if there is a single
		# face, which libstdc++'s discrete_distribution returns without drawing), then u, v.
		draws = rng.canonical((3 if n_faces > 1 else 2) * n_points).reshape(n_points, -1)
		if n_faces > 1:
			face_idx = _discrete_distribution(area_weights, _to_device(draws[:, 0], torch.float64))
		else:
			face_idx = torch.zeros(n_points, dtype=torch.long, device=self.faces.device)
		# Sample within that face (barycentric coordinates on the host: NumPy's sqrt is correctly rounded)
		u = draws[:, -2]
		v = draws[:, -1]
		sqrt_u = np.sqrt(u)
		alpha = sqrt_u * (1.0 - v)
		beta = v * sqrt_u
		gamma = 1.0 - alpha - beta
		alpha, beta, gamma = _to_device(alpha), _to_device(beta), _to_device(gamma)
		samples = alpha * v0[:, face_idx] + beta * v1[:, face_idx] + gamma * v2[:, face_idx]
		normals = face_normals[:, face_idx]

		# Optionally add noise.
		if sigma_n > 0.0 or sigma_p > 0.0:
			position_offset, normal_offset = _perturbations(rng, n_points, sigma_p, sigma_n)
			samples = samples + _to_device(position_offset)
			normals = normals + _normalized(normals + _to_device(normal_offset))

		return samples, normals

	def sample_uniformly(
		self,
		n_points: int,
		seed: int = 0,
		bbox_min: np.ndarray = np.array([1, 1, 1]),
		bbox_max: np.ndarray = np.array([-1, -1, -1]),
	) -> torch.Tensor:
		"""Points of sample_points_and_normals_uniformly (no noise), (3, n_points)."""
		return self.sample_points_and_normals_uniformly(n_points, seed, bbox_min, bbox_max)[0]

	def sample_narrow_band(
		self,
		n_points: int,
		offset: float = 0.1,
		seed: int = 0,
		bbox_min: np.ndarray = np.array([1, 1, 1]),
		bbox_max: np.ndarray = np.array([-1, -1, -1]),
	) -> torch.Tensor:
		"""
		Sample narrow band around boundary: uniform surface samples p (sample_points_and_normals_uniformly) moved along
		their face normal, p + τ n with τ ~ U[-offset, offset) drawn from std::mt19937(seed + 1). (3, n_points)
		"""
		samples, normals = self.sample_points_and_normals_uniformly(n_points, seed, bbox_min, bbox_max)
		rng = _cpp_random.MT19937(seed + 1)
		tau = _cpp_random.UniformReal(-offset, offset)(rng, n_points)
		return samples + _to_device(tau) * normals

	def sample_uniform_point_cloud(
		self,
		n_points: int,
		max_noise_level: float = 0.01,
		max_normals_to_flip: float = 0.1,
		seed: int = 0,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Uniform point cloud with random noise levels: sigma_p, sigma_n ~ U[0, max_noise_level) and a fraction
		f ~ U[0, max_normals_to_flip) drawn from std::mt19937(seed); the points of
		sample_points_and_normals_uniformly(n_points, seed, sigma_p=sigma_p, sigma_n=sigma_n), then the normals of
		int(f n_points) points (a std::shuffle of the indices) negated. Returns points, normals (3, n_points).
		"""
		# Sample noise parameters
		rng = _cpp_random.MT19937(seed)
		sigma_p = _cpp_random.UniformReal(0.0, max_noise_level)(rng)
		sigma_n = _cpp_random.UniformReal(0.0, max_noise_level)(rng)
		samples, normals = self.sample_points_and_normals_uniformly(n_points, seed, sigma_p=sigma_p, sigma_n=sigma_n)

		# Flip normals
		normals_to_flip = _cpp_random.UniformReal(0.0, max_normals_to_flip)(rng)
		n_normals_to_flip = int(normals_to_flip * n_points)
		print(f'\t Flipping {n_normals_to_flip} normals', file=_sys.stderr)
		return samples, normals * _to_device(_flip_signs(n_points, n_normals_to_flip, rng))

	def sample_farthest_point_cloud(
		self,
		n_points: int,
		max_noise_level: float = 0.01,
		max_normals_to_flip: float = 0.1,
		seed: int = 0,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		As sample_uniform_point_cloud, but samples 4 n_points points and keeps n_points of them by farthest-point
		sampling (shape_2d.subsample_point_cloud) before flipping normals. Returns points, normals (3, n_points).
		"""
		# Sample noise parameters
		rng = _cpp_random.MT19937(seed)
		sigma_p = _cpp_random.UniformReal(0.0, max_noise_level)(rng)
		sigma_n = _cpp_random.UniformReal(0.0, max_noise_level)(rng)

		# Sample way more points then we need, then subsample down using farthest point sampling.
		samples, normals = self.sample_points_and_normals_uniformly(
			n_points * 4, seed, sigma_p=sigma_p, sigma_n=sigma_n
		)
		indices = _shape_2d.subsample_point_cloud(samples, n_points)
		samples = samples[:, indices]
		normals = normals[:, indices]

		# Flip normals
		normals_to_flip = _cpp_random.UniformReal(0.0, max_normals_to_flip)(rng)
		n_normals_to_flip = int(normals_to_flip * n_points)
		print(f'\t Flipping {n_normals_to_flip} normals', file=_sys.stderr)
		return samples, normals * _to_device(_flip_signs(samples.shape[1], n_normals_to_flip, rng))

	def _centroid(self) -> torch.Tensor:
		"""
		vertices_.rowwise().mean(), (3,). Agrees with the C++ to rounding (~1e-16) but not bit for bit: Eigen splits
		the three rows into an SSE packet and a scalar according to the run-time alignment of the result, which
		changes the summation order (observed: rows 1-2 as a packet in Mesh3D::center_and_scale of the reference
		build, rows 0-1 in a standalone program).
		"""
		return self.vertices.sum(dim=1) / self.vertices.shape[1]

	def _camera_rays(
		self,
		n_cameras: int,
		sensor_size: float,
		rng: _cpp_random.MT19937,
		seed: int,
		grid_spacing: float = None,
		grid_spacing_range: Tuple[float, float] = None,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Rays of n_cameras orthographic cameras (the camera model of sample_point_cloud & co.).

		Cameras sit on the sphere of diameter D = 2√3 (bounding sphere of the unit cube) around the vertex centroid c:
		positions c + (D/2) p_k, p_k evenly spaced on the unit sphere (Fibonacci lattice under a random rotation,
		shape_2d.sample_sphere_positions, 3 draws of `rng`), looking inward along d_k = -p_k. The sensor is the square
		of side sensor_size D in the plane spanned by a random orthonormal basis (s_k, t_k) of d_k⊥ (seeded by `seed`,
		the same for all cameras), sampled by a grid of spacing h_k with n = ceil(sensor_size D / h_k) points per row:
		ray (i, j) starts at c + (D/2) p_k + x_i s_k + x_j t_k, x_i = (i + 1/2) h_k - (n - 1) h_k / 2, direction d_k.
		h_k = grid_spacing, or h_k ~ U[grid_spacing_range) drawn per camera after the camera positions.

		Returns:
			ray origins, directions (3, n_rays), camera after camera, row (i) major
		"""
		dtype = self.vertices.dtype
		device = self.vertices.device
		# Get bounding sphere.
		centroid = self._centroid()
		diameter = _UNIT_CUBE_DIAMETER  # use bounding sphere of unit cube

		# Generate camera positions on sphere (looking inward)
		camera_positions = _shape_2d.sample_sphere_positions(n_cameras, rng, dtype=dtype, device=device)
		camera_directions = -camera_positions

		# Compute plane of camera sensor
		s_axes, t_axes = [], []
		for i in range(n_cameras):
			s, t = _shape_2d.orthonormal_basis_random(camera_directions[:, i], seed)
			s_axes.append(s)
			t_axes.append(t)
		s_axes = torch.stack(s_axes, dim=1)
		t_axes = torch.stack(t_axes, dim=1)

		# Starting positions (offset backward along view direction)
		start_positions = camera_positions * 0.5 * diameter + centroid[:, None]

		# Generate all ray origins and directions: one square grid per camera
		sensor_side = sensor_size * diameter
		if grid_spacing_range is not None:
			grid_spacings = _cpp_random.UniformReal(*grid_spacing_range)(rng, n_cameras)
		else:
			grid_spacings = np.full(n_cameras, float(grid_spacing))
		cams, xs, ys = [], [], []
		for cam in range(n_cameras):
			h = float(grid_spacings[cam])
			n_samples_per_row = int(np.ceil(sensor_side / h))
			half_grid_length = 0.5 * (n_samples_per_row - 1) * h
			coords = (np.arange(n_samples_per_row) + 0.5) * h - half_grid_length
			xs.append(np.repeat(coords, n_samples_per_row))
			ys.append(np.tile(coords, n_samples_per_row))
			cams.append(np.full(n_samples_per_row * n_samples_per_row, cam))
		cam = _to_device(np.concatenate(cams), torch.long)
		x = _to_device(np.concatenate(xs), dtype)
		y = _to_device(np.concatenate(ys), dtype)
		ray_origins = start_positions[:, cam] + (x * s_axes[:, cam] + y * t_axes[:, cam])
		ray_directions = camera_directions[:, cam]
		return ray_origins, ray_directions

	def sample_point_cloud(
		self,
		n_cameras: int,
		sensor_size: float,
		grid_spacing: float,
		seed: int,
		sigma_p: float = 0.0,
		sigma_n: float = 0.0,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Generate a point cloud by sampling the positions of orthographic cameras on a bounding sphere, and shooting
		rays. Each camera is assumed to shoot rays arranged in a square grid, with grid spacing controlled by
		`grid_spacing`; the side length of the square is sensor_size times the diameter 2√3 (camera model:
		_camera_rays). Every hit is kept, with noise p += δp, n += normalize(n + δn), δp ~ N(0, sigma_p² I),
		δn ~ N(0, sigma_n² I): as in the C++ the perturbed unit normal is added to the normal, so |n| ≈ 2 (also for
		sigma_n = 0). The C++ draws the noise inside an OpenMP loop from a shared engine (with more than one thread
		the draws race); the port draws them in order, i.e. the one-thread C++ result.

		Returns:
			points, normals: (3, n_hits). Host read: the number of hits.
		"""
		rng = _cpp_random.MT19937(seed)
		ray_origins, ray_directions = self._camera_rays(n_cameras, sensor_size, rng, seed, grid_spacing=grid_spacing)

		# Cast all rays.
		samples, normals = self.ray_intersections(ray_origins, ray_directions)
		n_intersections = samples.shape[1]

		# Add noise.
		position_offset, normal_offset = _perturbations(rng, n_intersections, sigma_p, sigma_n)
		samples = samples + _to_device(position_offset)
		normals = normals + _normalized(normals + _to_device(normal_offset))
		return samples, normals

	def sample_even_raycasted_point_cloud(
		self,
		n_cameras,
		n_points: int,
		sensor_size: float,
		grid_spacing_min: float = 0.01,
		grid_spacing_max: float = 0.1,
		seed: int = 0,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Ray-cast point cloud (cameras as in sample_point_cloud, grid spacing h_k ~ U[grid_spacing_min,
		grid_spacing_max) per camera), no noise, reduced to min(#hits, n_points) points: with probability 1/2 by
		farthest-point sampling, otherwise by a uniform random subset (std::shuffle).

		Returns:
			points, normals: (3, min(#hits, n_points)). Host read: the number of hits.
		"""
		rng = _cpp_random.MT19937(seed)
		ray_origins, ray_directions = self._camera_rays(
			n_cameras, sensor_size, rng, seed, grid_spacing_range=(grid_spacing_min, grid_spacing_max)
		)

		# Cast all rays.
		samples, normals = self.ray_intersections(ray_origins, ray_directions)

		n_intersections = samples.shape[1]
		n_final = min(n_intersections, n_points)
		if _cpp_random.UniformReal(0.0, 1.0)(rng) < 0.5:
			# Subsample using farthest point sampling
			indices = _shape_2d.subsample_point_cloud(samples, n_final)
		else:
			# Subsample using uniform sampling
			indices = _to_device(_shuffled_indices(n_intersections, rng)[:n_final], torch.long)
		return samples[:, indices], normals[:, indices]

	def sample_uneven_point_cloud(
		self,
		n_cameras: int,
		sensor_size: float,
		grid_spacing_min: float = 0.01,
		grid_spacing_max: float = 0.1,
		max_noise_level: float = 0.01,
		max_normals_to_flip: float = 0.1,
		max_points: int = 2048,
		seed: int = 0,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Simulated scan: ray-cast hits (cameras as in sample_even_raycasted_point_cloud), a uniform random subset of
		min(#hits, max_points) of them, noise p + δp, normalize(n + δn) with δp ~ N(0, sigma_p² I),
		δn ~ N(0, sigma_n² I), sigma_p, sigma_n ~ U[0, max_noise_level), then the normals of int(f n) points negated,
		f ~ U[0, max_normals_to_flip).

		Returns:
			points, normals: (3, min(#hits, max_points)). Host read: the number of hits.
		"""
		rng = _cpp_random.MT19937(seed)
		ray_origins, ray_directions = self._camera_rays(
			n_cameras, sensor_size, rng, seed, grid_spacing_range=(grid_spacing_min, grid_spacing_max)
		)

		# Cast all rays.
		samples, normals = self.ray_intersections(ray_origins, ray_directions)

		n_intersections = samples.shape[1]
		print(f'\tOriginal # intersections: {n_intersections}', file=_sys.stderr)

		# Uniformly subsample down to target size
		n_final = min(n_intersections, max_points)
		indices = _to_device(_shuffled_indices(n_intersections, rng)[:n_final], torch.long)

		# Add noise.
		sigma_p = _cpp_random.UniformReal(0.0, max_noise_level)(rng)
		sigma_n = _cpp_random.UniformReal(0.0, max_noise_level)(rng)
		normals_to_flip = _cpp_random.UniformReal(0.0, max_normals_to_flip)(rng)
		n_normals_to_flip = int(normals_to_flip * n_final)
		print(f'\t Flipping {n_normals_to_flip} normals', file=_sys.stderr)

		position_offset, normal_offset = _perturbations(rng, n_final, sigma_p, sigma_n)
		int_positions = samples[:, indices] + _to_device(position_offset)
		int_normals = _normalized(normals[:, indices] + _to_device(normal_offset))

		# Flip normals
		return int_positions, int_normals * _to_device(_flip_signs(n_final, n_normals_to_flip, rng))

	def sample_points_and_normals_uniformly_by_casting_rays(
		self,
		n_points: int,
		seed: int,
		bbox_min,
		bbox_max,
		sigma_p: float = 0.0,
		sigma_n: float = 0.0,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Surface samples inside a box from random rays (C++ only; not bound in the original Python module).

		Batches of 4 n_points rays are cast until n_points hits inside the box (inclusive) are collected (the first
		n_points, in ray order, are kept). Ray k: direction ω uniform on the sphere (θ = 2π u1, cos φ = 2 u2 - 1),
		origin c - 0.6 D ω + (D/2) r (cos θ' a + sin θ' b), r = √u4, θ' = 2π u3, (a, b) = orthonormal_basis(ω),
		c the box center and D its diagonal: a uniform point of a disk ⊥ ω, so the rays form an isotropic family of
		lines (hits are uniform w.r.t. area times the number of crossings). Noise: p + δp, normalize(n + δn),
		δp ~ N(0, sigma_p² I), δn ~ N(0, sigma_n² I). The C++ draws the rays in an OpenMP loop from a shared engine
		(racing with more than one thread); the port draws them in order. Host reads: the number of hits per batch.

		Returns:
			points, normals: (3, n_points)
		"""
		dtype = self.vertices.dtype
		bbox_min = _host_vector3(bbox_min)
		bbox_max = _host_vector3(bbox_max)
		bbox_center = _to_device(0.5 * (bbox_min + bbox_max))
		bbox_extent = bbox_max - bbox_min
		diameter = float(np.sqrt((bbox_extent[0] ** 2 + bbox_extent[1] ** 2) + bbox_extent[2] ** 2))
		lo, hi = _to_device(bbox_min), _to_device(bbox_max)

		# Random generator
		rng = _cpp_random.MT19937(seed)
		n_rays = n_points * 4
		int_positions, int_normals = [], []
		n_collected = 0
		while n_collected < n_points:
			# Cast a batch of rays: per ray, theta, cos_phi, theta_disk, r
			u = _to_device(rng.canonical(4 * n_rays).reshape(n_rays, 4).T, dtype)
			# Sample direction uniformly on the sphere
			theta = 2.0 * np.pi * u[0]
			cos_phi = 2.0 * u[1] - 1.0
			sin_phi = torch.sqrt(1.0 - cos_phi * cos_phi)
			ray_dir = torch.stack([torch.cos(theta) * sin_phi, torch.sin(theta) * sin_phi, cos_phi], dim=0)
			# Sample an offset uniformly from a disk perpendicular to ray_dir
			theta_disk = 2.0 * np.pi * u[2]
			r = torch.sqrt(u[3])
			onb_u, onb_v = _shape_2d.orthonormal_basis(ray_dir)
			ray_origins = (bbox_center[:, None] - ray_dir * 0.6 * diameter) + (r * diameter * 0.5) * (
				torch.cos(theta_disk) * onb_u + torch.sin(theta_disk) * onb_v
			)

			# Cast rays; filter intersections to only include those within bounding box
			hit, points, normals = self._ray_hits(ray_origins, ray_dir)
			inside = hit & ((points >= lo[:, None]) & (points <= hi[:, None])).all(dim=0)
			points, normals = points[:, inside], normals[:, inside]  # host read: number of hits
			int_positions.append(points)
			int_normals.append(normals)
			n_collected += points.shape[1]

		# Subsample if we have too many
		samples = torch.cat(int_positions, dim=1)[:, :n_points]
		normals = torch.cat(int_normals, dim=1)[:, :n_points]

		# Add noise
		position_offset, normal_offset = _perturbations(rng, n_points, sigma_p, sigma_n)
		return samples + _to_device(position_offset), _normalized(normals + _to_device(normal_offset))


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


# ========================================================================
# 2D SDFS (port of the SDF2D / SDF3D part of src/pointsastori/shape_3d.py)
# ========================================================================
#
# In the original, SDF2D and its subclasses wrap the C++ classes (pat_bindings.Circle2D, ...): `bound_object` is the
# C++ object, and every call converts (n, 2) / (n, 3) NumPy arrays to Fortran-ordered (2, n) / (3, n) matrices and
# back. Here `bound_object` is the torch port of the C++ class (points_as_tori.shape_2d.Circle2D, ...), and queries
# and results are wide (2, n) / (3, n) tensors on the configured device, so the transposes disappear. Single points
# are (2,) / (3,) tensors. Fixes of crashes of the original are marked "(fix)".

import re as _re  # noqa: E402

from . import shape_2d as _sdf_cpp  # noqa: E402  (the original's `pat_bindings` for the 2D SDFs)

SDF_SHAPES = [
	'circle',
	'pie',
	'arc',
	'pill',
	'vesica',
	'box',
	'cross',
	'isosceles',
	'triangle',
	'hexagon',
	'ellipse',
	'heart',
	'pentagon',
	'moon',
	'trapezoid',
	'quad',
]


class SDF2D:
	"""
	2D SDF (Python API). Queries: (2,) or (2, n) tensors; 3D queries (3,) or (3, n); results on the device.
	See points_as_tori.shape_2d.SDF2D for the mathematics, the samplers and the deviations.
	"""

	def __init__(self, instance=None):
		if instance is None:
			# (fix) the original's default is the class utilsb.Circle2D itself, on which has_gradient() raises
			raise TypeError('SDF2D wraps a shape instance, e.g. SDF2D(shape_2d.Circle2D(0.5)); or use Circle2D(0.5)')
		self.bound_object = instance
		self._has_gradient = self.bound_object.has_gradient()

	def evaluate_signed_distance_and_gradient(self, q: torch.Tensor, isovalue: float = 0.0):
		"""(2,) -> (0-d distance, (2,) gradient); (2, n) -> ((n,), (2, n)). (fix: the original fails on (2,))"""
		if self._has_gradient:
			if q.ndim == 1:
				return self.bound_object.evaluate_with_gradient(q, isovalue)
			return self.bound_object.evaluate_with_gradient_batch(q, isovalue)

		return self.evaluate_signed_distance(q, isovalue), self.evaluate_signed_distance_gradient(q)

	def evaluate_signed_distance(self, q: torch.Tensor, isovalue: float = 0.0) -> torch.Tensor:
		"""
		A positive offset value means the shape gets inflated.
		"""
		if q.ndim == 1:
			return self.bound_object.evaluate(q, isovalue)
		return self.bound_object.evaluate_batch(q, isovalue)

	def evaluate_signed_distance_gradient_3D(self, q: torch.Tensor, is_extrusion: bool, param: float) -> torch.Tensor:
		"""(3, n) -> (3, n) gradient of the extrusion (half height param) or revolution (offset param)."""
		return self.bound_object.evaluate_signed_distance_gradient(q, is_extrusion, param)

	def evaluate_signed_distance_gradient(self, q: torch.Tensor) -> torch.Tensor:
		if self._has_gradient:
			_, grad = self.evaluate_signed_distance_and_gradient(q)
			return grad

		if q.ndim == 1:
			q = q[:, None]
		return self.bound_object.evaluate_gradient_batch(q)

	def evaluate_signed_distance_laplacian(self, q: torch.Tensor, exact: bool = False) -> torch.Tensor:
		"""
		(n,) Laplacians. exact=False reproduces the original (Δφ + 2 (∂φ/∂x + ∂φ/∂y), from the C++ autodiff seeding;
		see shape_2d.SDF2D); exact=True gives Δφ. (exact is an addition of the port.)
		"""
		if q.ndim == 1:
			q = q[:, None]

		return self.bound_object.evaluate_laplacian_batch(q, exact=exact)

	def evaluate_revolution(self, q: torch.Tensor, o: float) -> torch.Tensor:
		if q.ndim == 1:
			q = q[:, None]
		return self.bound_object.evaluate_revolution(q, o)

	def evaluate_extrusion(self, q: torch.Tensor, h: float) -> torch.Tensor:
		if q.ndim == 1:
			q = q[:, None]
		return self.bound_object.evaluate_extrusion(q, h)

	def sample_revolution(
		self,
		n_samples: int,
		o: float,
		bbox_min: np.ndarray = np.array([1, 1, 1]),
		bbox_max: np.ndarray = np.array([-1, -1, -1]),
		sigma_p: float = 0.0,
		sigma_n: float = 0.0,
		seed: int = 0,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""(3, n_samples) points and normals; see shape_2d.SDF2D.sample_revolution."""
		points, normals = self.bound_object.sample_revolution(n_samples, o, bbox_min, bbox_max, sigma_p, sigma_n, seed)
		return points, normals

	def sample_extrusion(
		self,
		n_samples: int,
		h: float,
		bbox_min: np.ndarray = np.array([1, 1, 1]),
		bbox_max: np.ndarray = np.array([-1, -1, -1]),
		sigma_p: float = 0.0,
		sigma_n: float = 0.0,
		seed: int = 0,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""(3, n_samples) points and normals; see shape_2d.SDF2D.sample_extrusion."""
		points, normals = self.bound_object.sample_extrusion(n_samples, h, bbox_min, bbox_max, sigma_p, sigma_n, seed)
		return points, normals

	def sample_revolution_narrow_band(
		self,
		n_samples: int,
		o: float,
		offset: float,
		bbox_min: np.ndarray = np.array([1, 1, 1]),
		bbox_max: np.ndarray = np.array([-1, -1, -1]),
		seed: int = 0,
	) -> torch.Tensor:
		samples = self.bound_object.sample_revolution_narrow_band(n_samples, o, offset, bbox_min, bbox_max, seed)
		return samples

	def sample_extrusion_narrow_band(
		self,
		n_samples: int,
		h: float,
		offset: float,
		bbox_min: np.ndarray = np.array([1, 1, 1]),
		bbox_max: np.ndarray = np.array([-1, -1, -1]),
		seed: int = 0,
	) -> torch.Tensor:
		samples = self.bound_object.sample_extrusion_narrow_band(n_samples, h, offset, bbox_min, bbox_max, seed)
		return samples

	def sample_farthest_point_cloud(
		self,
		n_points: int,
		is_extrusion: bool,
		SDF3D_param: float,
		seed: int,
	):
		points, normals = self.bound_object.sample_farthest_point_cloud(n_points, is_extrusion, SDF3D_param, seed)
		return points, normals

	def sample_points_and_normals_uniformly(self, n_points: int, is_extrusion: bool, SDF3D_param: float, seed: int):
		"""(fix) Called by SDF3D.sample_points_and_normals_uniformly but missing from the original wrapper."""
		return self.bound_object.sample_points_and_normals_uniformly(n_points, is_extrusion, SDF3D_param, seed)

	def sample_uneven_point_cloud(
		self,
		is_extrusion: bool,
		SDF3D_param: float,
		n_cameras: int,
		sensor_size: float,
		grid_spacing_min: float,
		grid_spacing_max: float,
		max_noise_level: float,
		max_normals_to_flip: float,
		max_points: int,
		seed: int,
	):
		"""(fix) Called by SDF3D.sample_uneven_point_cloud but missing from the original wrapper."""
		return self.bound_object.sample_uneven_point_cloud(
			is_extrusion,
			SDF3D_param,
			n_cameras,
			sensor_size,
			grid_spacing_min,
			grid_spacing_max,
			max_noise_level,
			max_normals_to_flip,
			max_points,
			seed,
		)

	def ray_intersections(self, ros: torch.Tensor, rds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""(2, n) rays -> hit positions, normals (2, n_hits) (ray order)."""
		return self.bound_object.ray_intersect_batch(ros, rds)

	@staticmethod
	def load_SDF(filepath):
		"""
		Shape file of the original format: the shape type is a word of the path (split at '_', '/', '-', '.'), the
		file has lines `<name> <values...>` (e.g. `r 0.5`, `sc 0.8 0.6`, `v -0.5 -0.5`). Returns None when the
		required line is missing (circle, box, cross, isosceles, hexagon, ellipse, pentagon), as the original.
		(fix) The original does not import `re` and calls Triangle2D(v=...) / Quad2D(v=...), which do not exist.
		"""
		parts = _re.split(r'_|/|-|\.', filepath)
		if 'circle' in parts:
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'r':
						radius = float(parts[1])
						return Circle2D(radius)
		elif 'pie' in parts:
			radius = None
			t = None
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'r':
						radius = float(parts[1])
					elif parts[0] == 't':
						t = float(parts[1])
			return Pie2D(radius, t)
		elif 'arc' in parts:
			sc = None
			ra = None
			rb = None
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'sc':
						sc = np.array(list(map(float, parts[1:3])))
					elif parts[0] == 'ra':
						ra = float(parts[1])
					elif parts[0] == 'rb':
						rb = float(parts[1])
			return Arc2D(sc, ra, rb)
		elif 'pill' in parts:
			a = None
			b = None
			r = None
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'a':
						a = np.array(list(map(float, parts[1:3])))
					elif parts[0] == 'b':
						b = np.array(list(map(float, parts[1:3])))
					elif parts[0] == 'r':
						r = float(parts[1])
			return Pill2D(a, b, r)
		elif 'vesica' in parts:
			r = None
			d = None
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'r':
						r = float(parts[1])
					elif parts[0] == 'd':
						d = float(parts[1])
			return Vesica2D(r, d)
		elif 'box' in parts:
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'b':
						b = np.array(list(map(float, parts[1:3])))
						return Box2D(b)
		elif 'cross' in parts:
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'b':
						b = np.array(list(map(float, parts[1:3])))
						return Cross2D(b)
		elif 'isosceles' in parts:
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'q':
						q = np.array(list(map(float, parts[1:3])))
						return TriangleIsosceles2D(q)
		elif 'triangle' in parts:
			v = []
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'v':
						v.append(list(map(float, parts[1:3])))
			return Triangle2D(*np.array(v))
		elif 'hexagon' in parts:
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'r':
						r = float(parts[1])
						return Hexagon2D(r)
		elif 'ellipse' in parts:
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'ab':
						ab = np.array(list(map(float, parts[1:3])))
						return Ellipse2D(ab)
		elif 'heart' in parts:
			return Heart2D()
		elif 'pentagon' in parts:
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'r':
						r = float(parts[1])
						return Pentagon2D(r)
		elif 'moon' in parts:
			d = None
			ra = None
			rb = None
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'd':
						d = float(parts[1])
					elif parts[0] == 'ra':
						ra = float(parts[1])
					elif parts[0] == 'rb':
						rb = float(parts[1])
			return Moon2D(d, ra, rb)
		elif 'trapezoid' in parts:
			ra = None
			rb = None
			he = None
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'ra':
						ra = float(parts[1])
					elif parts[0] == 'rb':
						rb = float(parts[1])
					elif parts[0] == 'he':
						he = float(parts[1])
			return Trapezoid2D(ra, rb, he)
		elif 'quad' in parts:
			v = []
			with open(filepath, 'r') as file:
				for line in file:
					parts = line.strip().split()
					if not parts:
						continue
					elif parts[0] == 'v':
						v.append(list(map(float, parts[1:3])))
			return Quad2D(*np.array(v))
		else:
			raise ValueError(f'Unknown SDF type in file: {filepath}')


load_SDF = SDF2D.load_SDF  # module-level alias (the original has only the static method)


class Circle2D(SDF2D):
	def __init__(self, radius: float):
		super().__init__(_sdf_cpp.Circle2D(radius))
		self.radius = radius


class Pie2D(SDF2D):
	def __init__(self, radius: float, t: float):
		super().__init__(_sdf_cpp.Pie2D(radius, t))
		self.radius = radius
		self.t = t


class Arc2D(SDF2D):
	def __init__(self, sc: np.ndarray, ra: float, rb: float):
		super().__init__(_sdf_cpp.Arc2D(sc, ra, rb))
		self.sc = sc
		self.ra = ra
		self.rb = rb


class Pill2D(SDF2D):
	def __init__(self, a: np.ndarray, b: np.ndarray, r: float):
		super().__init__(_sdf_cpp.Pill2D(a, b, r))
		self.a = a
		self.b = b
		self.r = r


class Vesica2D(SDF2D):
	def __init__(self, r: float, d: float):
		super().__init__(_sdf_cpp.Vesica2D(r, d))
		self.r = r
		self.d = d


class Box2D(SDF2D):
	def __init__(self, b: np.ndarray):
		super().__init__(_sdf_cpp.Box2D(b))
		self.b = b


class Cross2D(SDF2D):
	def __init__(self, b: np.ndarray):
		super().__init__(_sdf_cpp.Cross2D(b))
		self.b = b


class TriangleIsosceles2D(SDF2D):
	def __init__(self, q: np.ndarray):
		super().__init__(_sdf_cpp.TriangleIsosceles2D(q))
		self.q = q


class Triangle2D(SDF2D):
	def __init__(self, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray):
		super().__init__(_sdf_cpp.Triangle2D(v0, v1, v2))
		self.v = [v0, v1, v2]


class Hexagon2D(SDF2D):
	def __init__(self, r: float):
		super().__init__(_sdf_cpp.Hexagon2D(r))
		self.r = r


class Ellipse2D(SDF2D):
	def __init__(self, ab: np.ndarray):
		super().__init__(_sdf_cpp.Ellipse2D(ab))
		self.ab = ab


class Heart2D(SDF2D):
	def __init__(self):
		super().__init__(_sdf_cpp.Heart2D())


class Pentagon2D(SDF2D):
	def __init__(self, r: float):
		super().__init__(_sdf_cpp.Pentagon2D(r))
		self.r = r


class Moon2D(SDF2D):
	def __init__(self, d: float, ra: float, rb: float):
		super().__init__(_sdf_cpp.Moon2D(d, ra, rb))
		self.d = d
		self.ra = ra
		self.rb = rb


class Trapezoid2D(SDF2D):
	def __init__(self, ra: float, rb: float, he: float):
		super().__init__(_sdf_cpp.Trapezoid2D(ra, rb, he))
		self.ra = ra
		self.rb = rb
		self.he = he


class Quad2D(SDF2D):
	def __init__(self, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray):
		super().__init__(_sdf_cpp.Quad2D(v0, v1, v2, v3))
		self.v = [v0, v1, v2, v3]


def _sdf_split(key: int, num: int) -> list:
	"""Analog of jax.random.split for integer keys: `num` new keys drawn from a torch.Generator seeded with key."""
	g = torch.Generator().manual_seed(int(key))
	return [int(k) for k in torch.randint(0, 2**62, (num,), generator=g, dtype=torch.int64)]


def _sdf_uniform(key: int, minval: float = 0.0, maxval: float = 1.0, shape: tuple = ()):
	"""Analog of jax.random.uniform for integer keys: float64 values in [minval, maxval) (a float for shape ())."""
	g = torch.Generator().manual_seed(int(key))
	u = torch.rand(shape, generator=g, dtype=torch.float64).numpy() * (maxval - minval) + minval
	return float(u) if shape == () else u


def sample_SDF(shape_name: str = 'circle', key=None) -> SDF2D:
	"""
	Given a shape, define an SDF with random parameters

	Deviation (randomness): the original takes a JAX PRNG key (default jax.random.PRNGKey(42)). Here `key` is an
	integer seed (default 42) or a torch.Generator (a seed is drawn from it). The JAX calls are replaced by analogs
	with the same structure: split(key, n) draws n new integer keys from torch.Generator(key), uniform(key, ...)
	draws from torch.Generator(key). The parameters follow the original's distributions (and the 'quad' branch
	reuses subkey2 for h and the perturbation, as the original does), but they are not the JAX values, and they
	are float64 (JAX draws float32).
	"""
	if key is None:
		key = 42
	elif isinstance(key, torch.Generator):
		key = int(torch.randint(0, 2**62, (1,), generator=key, dtype=torch.int64))

	# Some guidelines so shapes stay roughly within [-1,1]^2.
	r_min = 0.1
	r_max = 0.9
	r_ratio_min = 0.1  # minimum ratio of smaller radius to larger radius
	r_ratio_max = 0.9  # maximum ratio of smaller radius to larger radius
	t_min = np.pi / 6
	t_max = 5 * np.pi / 6

	shape = None
	if shape_name == 'circle':
		# Can only choose half-height, which controls global scaling --- equivalent to varying the sampling density
		# per neighborhood. Nevertheless, choose random radius.
		r = _sdf_uniform(key, minval=r_min, maxval=r_max)
		shape = Circle2D(r)
	elif shape_name == 'pie':
		key, subkey1, subkey2 = _sdf_split(key, 3)
		r = _sdf_uniform(subkey1, minval=r_min, maxval=r_max)
		t = _sdf_uniform(subkey2, minval=t_min, maxval=t_max)
		shape = Pie2D(r, t)
	elif shape_name == 'arc':
		key, subkey1, subkey2, subkey3 = _sdf_split(key, 4)
		ra = _sdf_uniform(subkey1, minval=r_min, maxval=r_max)  # outer radius
		rb = _sdf_uniform(subkey2, minval=r_ratio_min, maxval=r_ratio_max) * ra  # inner radius
		angle = _sdf_uniform(subkey3, minval=t_min, maxval=t_max)  # angle
		sc = np.array([np.sin(angle), np.cos(angle)])
		shape = Arc2D(sc, ra, rb)
	elif shape_name == 'pill':
		# two endpoints and radius
		key, subkey1, subkey2 = _sdf_split(key, 3)
		halflength = _sdf_uniform(subkey1, minval=r_min, maxval=r_max)
		a = np.array([-halflength, 0])  # first endpoint
		b = np.array([halflength, 0])  # second endpoint
		r = _sdf_uniform(subkey2, minval=r_ratio_min, maxval=r_ratio_max) * halflength
		shape = Pill2D(a, b, r)
	elif shape_name == 'vesica':
		# radius, center offset
		key, subkey1, subkey2 = _sdf_split(key, 3)
		r = _sdf_uniform(subkey1, minval=r_min, maxval=r_max)
		d = _sdf_uniform(subkey2, minval=r_ratio_min, maxval=r_ratio_max) * (2.0 * r) - r
		shape = Vesica2D(r, d)
	elif shape_name == 'box':
		# half-extents
		key, subkey1, subkey2 = _sdf_split(key, 3)
		bx = _sdf_uniform(subkey1, minval=r_min, maxval=r_max)
		by = _sdf_uniform(subkey2, minval=r_min, maxval=r_max)
		b = np.array([bx, by])
		shape = Box2D(b)
	elif shape_name == 'cross':
		# (b.x, b.y) are horizontal/vertical arm dimensions
		key, subkey1, subkey2 = _sdf_split(key, 3)
		bx = _sdf_uniform(subkey1, minval=r_min, maxval=r_max)
		by = _sdf_uniform(subkey2, minval=r_min, maxval=r_max)
		b = np.array([bx, by])
		shape = Cross2D(b)
	elif shape_name == 'isosceles':
		# size
		key, subkey1, subkey2 = _sdf_split(key, 3)
		qx = _sdf_uniform(subkey1, minval=r_min, maxval=r_max)  # base length; must be positive
		qy = -_sdf_uniform(subkey2, minval=r_min, maxval=r_max)  # controls "height" above origin
		q = np.array([qx, qy])
		shape = TriangleIsosceles2D(q)
	elif shape_name == 'triangle':
		# Start with equilateral triangle, then perturb
		key, subkey1, subkey2 = _sdf_split(key, 3)
		h = _sdf_uniform(subkey1, minval=r_min, maxval=r_max)  # height
		perturbation_amt = r_ratio_max * h
		perturbation_vecs = _sdf_uniform(subkey2, shape=(3, 2))
		perturbation_vecs /= np.linalg.norm(perturbation_vecs, axis=1, keepdims=True)
		perturbation = perturbation_amt * perturbation_vecs
		b = 2.0 * h / np.sqrt(3.0)
		v0 = np.array([-b, -h]) + perturbation[0]
		v1 = np.array([b, -h]) + perturbation[1]
		v2 = np.array([0, h]) + perturbation[2]
		shape = Triangle2D(v0, v1, v2)
	elif shape_name == 'hexagon':
		r = _sdf_uniform(key, minval=r_min, maxval=r_max)
		shape = Hexagon2D(r)
	elif shape_name == 'ellipse':
		# semi-axes (a, b)
		key, subkey1, subkey2 = _sdf_split(key, 3)
		a = _sdf_uniform(subkey1, minval=r_min, maxval=r_max)
		b = _sdf_uniform(subkey2, minval=r_min, maxval=r_max)
		ab = np.array([a, b])
		shape = Ellipse2D(ab)
	elif shape_name == 'heart':
		return Heart2D()
	elif shape_name == 'pentagon':
		r = _sdf_uniform(key, minval=r_min, maxval=r_max)
		shape = Pentagon2D(r)
	elif shape_name == 'moon':
		# d is offset, ra and rb are circle radii
		key, subkey1, subkey2, subkey3 = _sdf_split(key, 4)
		ra = _sdf_uniform(subkey1, minval=r_min, maxval=r_max)  # outer radius
		rb = _sdf_uniform(subkey2, minval=r_ratio_min, maxval=r_ratio_max) * ra  # inner radius
		d = _sdf_uniform(subkey3, minval=(ra - rb) / r_ratio_max, maxval=(ra + rb) * r_ratio_max)
		shape = Moon2D(d, ra, rb)
	elif shape_name == 'trapezoid':
		key, subkey1, subkey2, subkey3 = _sdf_split(key, 4)
		ra = _sdf_uniform(subkey1, minval=r_min, maxval=r_max)
		rb = _sdf_uniform(subkey2, minval=r_min, maxval=r_max)
		he = 2.0 * _sdf_uniform(subkey3, minval=r_min, maxval=r_max)
		shape = Trapezoid2D(ra, rb, he)
	elif shape_name == 'quad':
		# Generate as slightly perturbed rectangle for validity
		key, subkey1, subkey2, subkey3 = _sdf_split(key, 4)
		w = _sdf_uniform(subkey1, minval=r_min, maxval=r_max)  # width
		h = _sdf_uniform(subkey2, minval=r_min, maxval=r_max)  # height
		perturbation_amt = min((r_ratio_max * h), (r_ratio_max * w))
		perturbation_vecs = _sdf_uniform(subkey2, shape=(4, 2))  # subkey2 reused, as in the original
		perturbation_vecs /= np.linalg.norm(perturbation_vecs, axis=1, keepdims=True)
		perturbation = perturbation_amt * perturbation_vecs
		v0 = np.array([-w, -h]) + perturbation[0]
		v1 = np.array([w, -h]) + perturbation[1]
		v2 = np.array([w, h]) + perturbation[2]
		v3 = np.array([-w, h]) + perturbation[3]
		shape = Quad2D(v0, v1, v2, v3)

	return shape


class SDF3D:
	"""Revolution (is_extrusion False, param = offset o) or extrusion (param = half height h) of a 2D SDF."""

	def __init__(self, sdf2d: SDF2D, is_extrusion: bool, param: float):
		self.SDF2D = sdf2d
		self.is_extrusion = is_extrusion
		self.param = param  # either o or h

	def evaluate_signed_distance(self, q: torch.Tensor) -> torch.Tensor:
		"""(3, n) -> (n,)"""
		if self.is_extrusion:
			return self.SDF2D.evaluate_extrusion(q, self.param)

		return self.SDF2D.evaluate_revolution(q, self.param)

	def evaluate_signed_distance_and_gradient(self, q: torch.Tensor):
		"""(3, n) -> distances (n,), gradients (3, n)"""
		distances = self.evaluate_signed_distance(q)
		gradients = self.SDF2D.evaluate_signed_distance_gradient_3D(q, self.is_extrusion, self.param)
		return distances, gradients

	def sample_farthest_point_cloud(
		self,
		n_points: int,
		seed: int,
	):
		return self.SDF2D.sample_farthest_point_cloud(n_points, self.is_extrusion, self.param, seed)

	def sample_uneven_point_cloud(
		self,
		is_extrusion: bool,
		SDF3D_param: float,
		n_cameras: int,
		sensor_size: float,
		grid_spacing_min: float,
		grid_spacing_max: float,
		max_noise_level: float,
		max_normals_to_flip: float,
		max_points: int,
		seed: int,
	):
		"""As the original, is_extrusion and SDF3D_param are arguments (not this object's)."""
		return self.SDF2D.sample_uneven_point_cloud(
			is_extrusion,
			SDF3D_param,
			n_cameras,
			sensor_size,
			grid_spacing_min,
			grid_spacing_max,
			max_noise_level,
			max_normals_to_flip,
			max_points,
			seed,
		)

	def sample_uniformly(
		self,
		n_points: int,
		bbox_min: np.ndarray = np.array([1, 1, 1]),
		bbox_max: np.ndarray = np.array([-1, -1, -1]),
		sigma_p: float = 0.0,
		sigma_n: float = 0.0,
		seed: int = 0,
	) -> torch.Tensor:
		"""
		Sample points uniformly on the surface.
		"""
		if self.is_extrusion:
			points, _ = self.SDF2D.sample_extrusion(n_points, self.param, bbox_min, bbox_max, sigma_p, sigma_n, seed)
			return points
		else:
			points, _ = self.SDF2D.sample_revolution(n_points, self.param, bbox_min, bbox_max, sigma_p, sigma_n, seed)
			return points

	def sample_points_and_normals_uniformly(self, n_points, seed):
		return self.SDF2D.sample_points_and_normals_uniformly(n_points, self.is_extrusion, self.param, seed)

	def sample_narrow_band(
		self,
		n_points: int,
		offset: float = 0.1,
		bbox_min: np.ndarray = np.array([1, 1, 1]),
		bbox_max: np.ndarray = np.array([-1, -1, -1]),
		seed: int = 0,
	) -> torch.Tensor:
		"""
		Sample points in a narrow band around the surface.
		"""
		if self.is_extrusion:
			return self.SDF2D.sample_extrusion_narrow_band(n_points, self.param, offset, bbox_min, bbox_max, seed)
		else:
			return self.SDF2D.sample_revolution_narrow_band(n_points, self.param, offset, bbox_min, bbox_max, seed)
