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
