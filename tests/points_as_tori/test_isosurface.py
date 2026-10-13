"""Marching cubes (points_as_tori/isosurface.py) and mesh_isocontour (port of the demo's PyVista contouring)."""

from collections import Counter

import numpy as np
import pytest
import torch

import points_as_tori as pat
from points_as_tori.isosurface import marching_cubes, mesh_isocontour, triangle_table
from helpers import random_tori


def edge_counts(faces):
	F = faces.T.cpu().numpy()
	e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
	return Counter(map(tuple, e)), Counter(map(lambda x: tuple(sorted(x)), e))


def test_table():
	T = triangle_table()
	n = (T[:, :, 0] >= 0).sum(1)
	assert T.shape[0] == 256 and n[0] == 0 and n[255] == 0 and n.max() <= 5
	assert (n[1:255] > 0).all()


@pytest.mark.parametrize('seed', range(4))
def test_closed_and_oriented(seed):
	"""Random fields, positive on the boundary: every edge in exactly two triangles, each directed edge once."""
	g = np.random.default_rng(seed).normal(size=(24, 26, 28))
	g[[0, -1]] = g[:, [0, -1]] = g[:, :, [0, -1]] = 1.0
	V, F = marching_cubes(torch.from_numpy(g), [0, 0, 0], [1, 1, 1])
	directed, undirected = edge_counts(F)
	assert max(directed.values()) == 1 and set(undirected.values()) == {2}
	assert F.min() >= 0 and F.max() < V.shape[1]


def test_sphere():
	n = 40
	x = np.linspace(-1, 1, n)
	X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
	f = np.sqrt(X**2 + Y**2 + Z**2) - 0.7
	V, F = marching_cubes(torch.from_numpy(f), [-1, -1, -1], [1, 1, 1])
	assert float((V.norm(dim=0) - 0.7).abs().max()) < 2e-3
	a, b, c = V[:, F[0]], V[:, F[1]], V[:, F[2]]
	nrm = torch.linalg.cross(b - a, c - a, dim=0)
	assert bool(((nrm * (a + b + c)).sum(0) > 0).all())  # outward (toward increasing values)
	area = 0.5 * nrm.norm(dim=0).sum()
	skimage = pytest.importorskip('skimage.measure')
	sv, sf, _, _ = skimage.marching_cubes(f, 0.0, spacing=(2 / (n - 1),) * 3)
	s_area = 0.5 * np.linalg.norm(np.cross(sv[sf[:, 1]] - sv[sf[:, 0]], sv[sf[:, 2]] - sv[sf[:, 0]]), axis=1).sum()
	print(
		f'\nsphere: {V.shape[1]} vertices, {F.shape[1]} triangles (scikit-image: {len(sv)}, {len(sf)}); area {float(area):.7f} vs {s_area:.7f}'
	)
	assert V.shape[1] == len(sv) and F.shape[1] == len(sf)
	assert abs(float(area) - s_area) < 1e-5 * s_area


def test_mesh_isocontour(cpu64):
	"""Tori fitted to the unit sphere's exact form: the contoured blend is the unit sphere."""
	rng = np.random.default_rng(0)
	P = rng.normal(size=(800, 3))
	P /= np.linalg.norm(P, axis=1, keepdims=True)
	coeffs = np.zeros((800, 6))
	coeffs[:, 4:] = -0.5
	tdf = pat.TorusDistanceField(pat.as_wide(P, 3), pat.as_wide(P, 3))
	tdf.fit_tori_from_forms(pat.as_wide(coeffs, 6))
	tdf.set_k_evaluation(16)
	V, F = mesh_isocontour(tdf, 0.0, grid_resolution=32)
	assert float((V.norm(dim=0) - 1.0).abs().max()) < 5e-3
	_, undirected = edge_counts(F)
	assert set(undirected.values()) == {2}
	for method in ('GWN', 'SHC', 'SSPD', 'LogSumExp'):
		V2, F2 = mesh_isocontour(tdf, 0.0, grid_resolution=20, eval_method=method)
		assert V2.shape[0] == 3 and F2.shape[0] == 3


def test_random_tori_contour(cpu64):
	P, N, C = random_tori(600, seed=1)
	tdf = pat.TorusDistanceField(pat.as_wide(P, 3), pat.as_wide(N, 3))
	tdf.fit_tori_from_forms(pat.as_wide(C, 6))
	tdf.set_k_evaluation(32)
	V, F = mesh_isocontour(tdf, 0.0, grid_resolution=24, method='flying_edges')
	err = tdf.evaluate_distance(V).abs()
	cell = float((V.amax(1) - V.amin(1)).max()) / 20
	print(
		f'\nrandom tori, 24³ grid: |φ| at the vertices median {float(err.median()):.1e}, max {float(err.max()):.1e} (cell ~{cell:.2f})'
	)
	assert float(err.median()) < 0.1 * cell and float(err.max()) < 2 * cell  # linear interpolation of a non-SDF blend
