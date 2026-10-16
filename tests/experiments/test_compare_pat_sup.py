"""
experiments/compare_pat_sup.py: the merged scene of both fits (PAT fitted here, SUP from an assembly built with
SuperFit's classes), the SuperFit fitting command (subprocess stubbed), and the metric definitions, checked on the
ground truth against itself and a shifted copy.
"""

import importlib.util
import json
import os
import pickle

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


def superfit_test_helpers():
	"""tests/superfit/test_superfit_sdf.py (skips when SuperFit's checkout or sysl3d is missing)."""
	path = os.path.join(REPO, 'tests', 'superfit', 'test_superfit_sdf.py')
	spec = importlib.util.spec_from_file_location('superfit_test_helpers', path)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


def test_both_fits_in_one_scene(tmp_path, cpu64, monkeypatch):
	from sdfview.scene import Scene, model_of

	helpers = superfit_test_helpers()
	assembly = str(tmp_path / 'final_content.pkl')
	helpers.write_assembly(assembly, {'iter_0.best_program': helpers.assembly_expr()}, n_iters=1)
	knot = os.path.join(REPO, 'ext', 'points-as-tori', 'data', 'knot.obj')
	monkeypatch.setattr(cmp, 'NEAR_SURFACE_POINTS', 5000)
	out = tmp_path / 'out'
	cmp.main(['--mesh', knot, '--assembly', assembly, '-o', str(out), '--device', 'cpu', '--float64', '--res', '20',
			  '--pat-points', '3000', '--gt', 'gwn', '--metrics'])  # fmt: skip
	scene = Scene.load(str(out / 'knot.sdfx'))
	assert [f['name'] for f in scene.fields] == ['pat.pat', 'pat.pat_grid', 'sup.sup', 'sup.sup_grid', 'gt']
	assert [model_of(f) for f in scene.fields] == ['pat', 'pat', 'sup', 'sup', 'gt']
	assert scene.field('sup.sup')['type'] == 'glsl' and scene.field('sup.sup')['glsl']['n_parts'] == 2
	assert [pc['name'] for pc in scene.pointclouds] == ['pat.pat_points']
	assert scene.meta['frame'] == 'superfit_normalized' and set(scene.meta['models']) == {'pat', 'sup', 'gt'}
	assert scene.meta['models']['pat']['n_points'] == 3000 and scene.meta['models']['sup']['n_primitives'] == 2
	# the GT grid has its surface inside the normalized box: values change sign
	gt = scene.field_arrays(scene.field('gt'))['values']
	assert gt.shape == (20, 20, 20) and gt.min() < 0 < gt.max()
	rows = [json.loads(line) for line in open(out / 'metrics.jsonl')]
	assert [r['model'] for r in rows] == ['PAT', 'SUP']
	# PAT fitted to the knot's own samples: close to the GT; the SUP assembly is not a fit of the knot
	assert rows[0]['iou@20'] > 0.8 > rows[1]['iou@20']
	# without --metrics: scene only
	out2 = tmp_path / 'out2'
	cmp.main(['--mesh', knot, '--assembly', assembly, '-o', str(out2), '--device', 'cpu', '--float64', '--res', '16',
			  '--skip-pat', '--gt', 'none'])  # fmt: skip
	assert [f['name'] for f in Scene.load(str(out2 / 'knot.sdfx')).fields] == ['sup.sup', 'sup.sup_grid']
	assert not (out2 / 'metrics.jsonl').exists()


def test_fit_superfit_command(tmp_path, monkeypatch):
	"""mesh_to_assembly.py from ext/superfit with PYTHONPATH (no installation), GPU selection, config, reuse."""
	calls = []

	def run(cmd, cwd, env, check):
		calls.append((cmd, cwd, env))
		save_dir = cmd[cmd.index('--save_dir') + 1]
		with open(os.path.join(save_dir, 'final_content.pkl'), 'wb') as f:
			pickle.dump({'input_path': cmd[cmd.index('--input_path') + 1]}, f)

	monkeypatch.setattr(cmp.subprocess, 'run', run)
	monkeypatch.delenv('SUPERFIT_CONFIG', raising=False)
	monkeypatch.setenv('PYTHONPATH', '/elsewhere')
	monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
	if not os.path.isfile(os.path.join(cmp.SUPERFIT_DIR, 'scripts', 'mesh_to_assembly.py')):
		pytest.skip('ext/superfit not checked out')
	save_dir = str(tmp_path / 'superfit' / 'knot')
	path = cmp.fit_superfit('knot.obj', save_dir, '/env/bin/python', gpu=1, extra_args='--ablation 8')
	assert path == os.path.join(save_dir, 'final_content.pkl') and os.path.isfile(path)
	((cmd, cwd, env),) = calls
	assert cmd[:2] == ['/env/bin/python', os.path.join(cmp.SUPERFIT_DIR, 'scripts', 'mesh_to_assembly.py')]
	assert cmd[-3:] == ['--fastmode', '--ablation', '8']
	assert cmd[cmd.index('--input_path') + 1] == os.path.abspath('knot.obj')
	assert cwd == cmp.SUPERFIT_DIR
	assert env['PYTHONPATH'] == os.pathsep.join([cmp.SUPERFIT_DIR, '/elsewhere']) and env['CUDA_VISIBLE_DEVICES'] == '1'
	if not os.path.isfile(os.path.join(cmp.SUPERFIT_DIR, 'superfit_config.json')):
		cfg = json.load(open(env['SUPERFIT_CONFIG']))
		assert cfg['data_base'] == str(tmp_path / 'superfit' / 'data')
	# an existing result is reused; --refit runs again
	cmp.fit_superfit('knot.obj', save_dir, '/env/bin/python')
	assert len(calls) == 1
	cmp.fit_superfit('knot.obj', save_dir, '/env/bin/python', refit=True)
	assert len(calls) == 2 and 'CUDA_VISIBLE_DEVICES' not in calls[1][2]
	# --superfit-gpu N is this process's cuda:N
	monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3,2')
	cmp.fit_superfit('knot.obj', save_dir, '/env/bin/python', gpu=1, refit=True)
	assert calls[2][2]['CUDA_VISIBLE_DEVICES'] == '2'
