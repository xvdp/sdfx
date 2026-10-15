"""experiments/compare_pat_sup.py: metric definitions, checked on the ground truth against itself and a shifted copy."""

import importlib.util
import os

import numpy as np
import pytest
import torch

pytest.importorskip('trimesh')
pytest.importorskip('geolipi')

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
spec = importlib.util.spec_from_file_location(
	'compare_pat_sup', os.path.join(REPO, 'experiments', 'compare_pat_sup.py')
)
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)


@pytest.fixture
def cpu64():
	from points_as_tori import config

	config.set_device('cpu', torch.float64, force=True)
	yield


@pytest.fixture
def knot(cpu64):
	V, F = cmp.read_mesh(os.path.join(REPO, 'ext', 'points-as-tori', 'data', 'knot.obj'))
	Vn, s, c = cmp.normalize(V)
	assert np.allclose((Vn.max(0) - Vn.min(0)).max(), 1.8) and np.allclose((Vn.max(0) + Vn.min(0)) / 2, 0)
	gt = cmp.GroundTruth(Vn, F, 'gwn', torch.device('cpu'))
	res = 32
	grid, axis = cmp.voxel_centres(res, 'cpu', torch.float64)
	assert torch.allclose(axis, (2 * torch.arange(res, dtype=torch.float64) + 1) / res - 1)
	surface = cmp.surface_samples(Vn, F, 20000, 0)
	near = cmp.near_surface(*surface, 5)
	ctx = {'res': res, 'seed': 0, 'grid': grid, 'axis': axis, 'gt_occ': cmp.evaluate_chunked(gt, grid) <= 0,
		   'gt_surface': surface, 'near': near, 'near_gt': cmp.evaluate_chunked(gt, near),
		   'gt_chamfer': cmp.surface_samples(Vn, F, 2048, 2)[0]}  # fmt: skip
	return gt, ctx


def test_ground_truth_against_itself(knot, monkeypatch):
	monkeypatch.setattr(cmp, 'NEAR_SURFACE_POINTS', 20000)
	gt, ctx = knot
	m = cmp.model_metrics(gt, gt, ctx)
	print({k: round(v, 5) for k, v in m.items() if isinstance(v, float)})
	assert m['iou@32'] == 1.0 and m['iou_gt_surface'] == 1.0 and m['iou_pred_surface'] == 1.0
	assert m['sdf_mae'] == 0.0
	# two independent 2048-point samplings of (nearly) the same surface: the metric's floor, 0.11 measured on the knot
	assert m['cd_x100'] < 0.2
	assert m['normal_abs_cos'] > 0.9


def test_shifted_copy_scores_lower(knot, monkeypatch):
	monkeypatch.setattr(cmp, 'NEAR_SURFACE_POINTS', 20000)
	gt, ctx = knot
	shift = torch.tensor([[0.05], [0.0], [0.0]], dtype=torch.float64)
	m = cmp.model_metrics(lambda q: gt(q - shift), gt, ctx)
	assert m['iou@32'] < 0.95 and m['cd_x100'] > 0.05 and m['sdf_mae'] > 0.005


def test_chamfer_definition():
	a = torch.tensor([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
	b = torch.tensor([[0.0], [0.5], [0.0]])
	# a -> b: 0.25, 1.25 (mean 0.75); b -> a: 0.25; (0.75 + 0.25) / 2 × 100
	assert cmp.chamfer_x100(a, b) == pytest.approx(50.0)
