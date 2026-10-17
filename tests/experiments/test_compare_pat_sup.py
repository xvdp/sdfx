"""
experiments/compare_pat_sup.py: the merged scene of both fits (PAT fitted here, SUP from an assembly built with
SuperFit's classes), how --mesh and --suppkl pair (an assembly's stored mesh, the hint for a mesh without one), and the
metric definitions, checked on the ground truth against itself and a shifted copy.
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
	cmp.main(['--mesh', knot, '--suppkl', assembly, '-o', str(out), '--device', 'cpu', '--float64', '--res', '20',
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
	cmp.main(['--mesh', knot, '--suppkl', assembly, '-o', str(out2), '--device', 'cpu', '--float64', '--res', '16',
			  '--skip-pat', '--gt', 'none'])  # fmt: skip
	assert [f['name'] for f in Scene.load(str(out2 / 'knot.sdfx')).fields] == ['sup.sup', 'sup.sup_grid']
	assert not (out2 / 'metrics.jsonl').exists()


def test_inputs_pairing(tmp_path, cpu64, capsys):
	"""--mesh and --suppkl pair in order; --suppkl alone uses the mesh stored in the assembly (a missing one stops);
	a mesh without --suppkl gets the hint to fit SuperFit; shape names from the mesh file (its folder for mesh.*)."""
	from types import SimpleNamespace

	helpers = superfit_test_helpers()
	knot = os.path.join(REPO, 'ext', 'points-as-tori', 'data', 'knot.obj')
	stored = str(tmp_path / 'stored.pkl')  # SuperFit's dataset scripts store input_mesh_file: here the knot
	helpers.write_assembly(stored, {'iter_0.best_program': helpers.assembly_expr()}, n_iters=1)
	data = pickle.load(open(stored, 'rb'))
	data['input_mesh_file'] = knot
	pickle.dump(data, open(stored, 'wb'))
	missing = str(tmp_path / 'missing.pkl')  # written by write_assembly: input_mesh_file /data/mesh.obj
	helpers.write_assembly(missing, {'iter_0.best_program': helpers.assembly_expr()}, n_iters=1)

	def args(mesh=None, suppkl=None, limit=0):
		return SimpleNamespace(mesh=mesh, suppkl=suppkl, limit=limit)

	assert cmp.stored_mesh(stored) == (knot, True) and cmp.stored_mesh(missing) == ('/data/mesh.obj', False)
	assert list(cmp.shapes(args(suppkl=[stored]))) == [('knot', knot, stored)]
	assert list(cmp.shapes(args(mesh=['a/b.obj', 'c/mesh.ply'], suppkl=['x.pkl', 'y.pkl']))) == [
		('b', 'a/b.obj', 'x.pkl'),
		('c', 'c/mesh.ply', 'y.pkl'),
	]
	assert list(cmp.shapes(args(mesh=['a/b.obj', 'c/d.glb'], limit=1))) == [('b', 'a/b.obj', None)]
	with pytest.raises(SystemExit, match='not on this machine'):
		list(cmp.shapes(args(suppkl=[missing])))
	with pytest.raises(SystemExit, match='pair one to one'):
		list(cmp.shapes(args(mesh=['a.obj'], suppkl=['x.pkl', 'y.pkl'])))
	# the stored path may be wrong here and the mesh given: --mesh wins
	assert list(cmp.shapes(args(mesh=[knot], suppkl=[missing]))) == [('knot', knot, missing)]

	# a mesh without --suppkl: PAT and the ground truth, and the commands to fit SuperFit and compare with it
	out = tmp_path / 'out'
	cmp.main(
		[
			'--mesh',
			knot,
			'-o',
			str(out),
			'--device',
			'cpu',
			'--float64',
			'--res',
			'12',
			'--pat-points',
			'500',
			'--gt',
			'gwn',
		]
	)
	text = capsys.readouterr().out
	assert 'no --suppkl for knot' in text and f'03_fit_superfit.sh {knot}' in text
	assert f'--suppkl {os.path.join(REPO, "examples", "out", "superfit", "knot", "final_content.pkl")}' in text
	assert 'without SuperFit (no --suppkl): knot' in text
	from sdfview.scene import Scene

	assert [f['name'] for f in Scene.load(str(out / 'knot.sdfx')).fields] == ['pat.pat', 'pat.pat_grid', 'gt']
