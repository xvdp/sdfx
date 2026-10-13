"""Shared test data helpers."""

import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PAT_SRC = os.path.join(REPO, 'ext', 'points-as-tori', 'src')
PAT_DATA = os.path.join(REPO, 'ext', 'points-as-tori', 'data')
FLAX_MODEL = os.path.join(PAT_SRC, 'pointsastori', 'models', 'FundamentalFormPredictor.pkl')
TORCH_MODEL = os.path.join(REPO, 'points_as_tori', 'models', 'FundamentalFormPredictor.pt')


def read_ply(name):
	from plyfile import PlyData

	v = PlyData.read(os.path.join(PAT_DATA, name))['vertex']
	points = np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float64)
	normals = np.stack([v['nx'], v['ny'], v['nz']], axis=-1).astype(np.float64)
	normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
	return points, normals


def random_tori(n, seed=0):
	"""Plausible random tori (fitted from random small curvature coefficients) on a unit sphere point cloud."""
	rng = np.random.default_rng(seed)
	points = rng.normal(size=(n, 3))
	points /= np.linalg.norm(points, axis=1, keepdims=True)
	normals = points + 0.1 * rng.normal(size=(n, 3))
	normals /= np.linalg.norm(normals, axis=1, keepdims=True)
	coeffs = rng.normal(scale=0.05, size=(n, 6))
	coeffs[:, 4:] += -0.5
	return points, normals, coeffs


def fortran(x):
	return np.asfortranarray(np.asarray(x, dtype=np.float64).T)


# ========================================================================
# Synthetic training data (same layout as mesh_data.generate_input_features_with_meshes / save_mesh_training_data)
# ========================================================================


def torus_mesh(R=0.5, r=0.2, n_major=24, n_minor=12, rotation=None, center=None):
	"""Closed, outward-oriented triangle mesh of a torus. Returns vertices (V, 3) float64, faces (F, 3) int."""
	u = 2 * np.pi * np.arange(n_major) / n_major
	v = 2 * np.pi * np.arange(n_minor) / n_minor
	uu, vv = np.meshgrid(u, v, indexing='ij')
	vertices = np.stack(
		[(R + r * np.cos(vv)) * np.cos(uu), (R + r * np.cos(vv)) * np.sin(uu), r * np.sin(vv)], axis=-1
	).reshape(-1, 3)
	i, j = np.meshgrid(np.arange(n_major), np.arange(n_minor), indexing='ij')
	a = i * n_minor + j
	b = ((i + 1) % n_major) * n_minor + j
	c = ((i + 1) % n_major) * n_minor + (j + 1) % n_minor
	d = i * n_minor + (j + 1) % n_minor
	faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
	# orient outward: the normal at a face centroid must point away from the tube center
	v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
	n = np.cross(v1 - v0, v2 - v0)
	cent = (v0 + v1 + v2) / 3
	ring = cent.copy()
	ring[:, 2] = 0
	ring = R * ring / np.linalg.norm(ring, axis=1, keepdims=True)
	if np.mean(np.sum(n * (cent - ring), axis=1) > 0) < 0.5:
		faces = faces[:, [0, 2, 1]]
	if rotation is not None:
		vertices = vertices @ rotation.T
	if center is not None:
		vertices = vertices + center
	return vertices, faces


def sample_mesh(vertices, faces, n_points, rng):
	"""Area-uniform samples with face normals."""
	v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
	cr = np.cross(v1 - v0, v2 - v0)
	area = 0.5 * np.linalg.norm(cr, axis=1)
	f = rng.choice(len(faces), size=n_points, p=area / area.sum())
	s, t = rng.random(n_points), rng.random(n_points)
	su = np.sqrt(s)
	alpha, beta = su * (1 - t), t * su
	points = alpha[:, None] * v0[f] + beta[:, None] * v1[f] + (1 - alpha - beta)[:, None] * v2[f]
	normals = cr[f] / np.linalg.norm(cr[f], axis=1, keepdims=True)
	return points, normals


def knn_numpy(points, k):
	d2 = np.sum((points[:, None] - points[None]) ** 2, axis=-1)
	return np.argsort(d2, axis=1, kind='stable')[:, :k]


def synthetic_dataset(n_shapes=2, n_points=200, k=16, pad_points=8, pad_faces=5, pad_vertices=3, seed=0):
	"""Tuple of arrays in the order returned by load_mesh_training_data."""
	rng = np.random.default_rng(seed)
	shapes = []
	for _ in range(n_shapes):
		q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
		vertices, faces = torus_mesh(
			R=rng.uniform(0.4, 0.6), r=rng.uniform(0.15, 0.25), rotation=q, center=rng.normal(scale=0.1, size=3)
		)
		points, normals = sample_mesh(vertices, faces, n_points, rng)
		shapes.append((points, normals, knn_numpy(points, k), vertices, faces))
	max_points = n_points + pad_points
	max_vertices = max(len(s[3]) for s in shapes) + pad_vertices
	max_faces = max(len(s[4]) for s in shapes) + pad_faces
	out = {
		'positions': np.zeros((n_shapes, max_points, 3)),
		'normals': np.zeros((n_shapes, max_points, 3)),
		'neighbors': -np.ones((n_shapes, max_points, k), dtype=np.int32),
		'masks': np.zeros((n_shapes, max_points), dtype=bool),
		'mesh_vertices': np.zeros((n_shapes, max_vertices, 3)),
		'mesh_faces': -np.ones((n_shapes, max_faces, 3), dtype=np.int32),
		'face_masks': np.zeros((n_shapes, max_faces), dtype=bool),
	}
	for i, (points, normals, nbrs, vertices, faces) in enumerate(shapes):
		out['positions'][i, :n_points] = points
		out['normals'][i, :n_points] = normals
		out['neighbors'][i, :n_points] = nbrs
		out['masks'][i, :n_points] = True
		out['mesh_vertices'][i, : len(vertices)] = vertices
		out['mesh_faces'][i, : len(faces)] = faces
		out['face_masks'][i, : len(faces)] = True
	return out


def save_synthetic_dataset(path, **kwargs):
	np.savez_compressed(path, **synthetic_dataset(**kwargs))
	return path
