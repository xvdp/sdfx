"""
superfit_sdf: SuperFit primitive assemblies as torch SDFs (wrapper around SuperFit's own expression classes and
evaluation). Assemblies are built here with SuperFit's classes and written in SuperFit's pickle layout (no released
assembly is needed). Skipped without geolipi / sysl3d or the ext/superfit checkout.
"""

import os
import pickle

import numpy as np
import pytest
import torch

pytest.importorskip('geolipi')
pytest.importorskip('sysl')

from superfit_sdf import SuperFitSDF, load_assembly, mesh_normalization, select_program  # noqa: E402
from superfit_sdf.assembly import import_superfit, superfit_root  # noqa: E402

try:
	superfit_root()
except ImportError:
	pytest.skip('ext/superfit not checked out', allow_module_level=True)


def primitive(size, roundness, dilate, taper, bulge, onion, axis_logits, axis_angle, translate, keep_drop):
	gls, _, _, _, _ = import_superfit()
	from superfit.symbolic.base import PrimitiveMarker, StochasticPrimitive
	from superfit.symbolic.var_axis import VarAxisSF

	p = VarAxisSF(tuple(size), (roundness,), (dilate,), (taper,), (bulge,), (onion,), tuple(axis_logits))
	return PrimitiveMarker(
		StochasticPrimitive(gls.Translate3D(gls.AxisAngleRotate3D(p, tuple(axis_angle)), tuple(translate)), keep_drop)
	)


def assembly_expr(drop_third=True):
	"""Optimization-stage form: three marked, stochastic, variable-axis SuperFrusta in a nested smooth union."""
	gls, _, _, _, _ = import_superfit()
	p1 = primitive(
		(0.6, 0.4, 0.5), 0.3, 0.0, 1.0, 0.1, 0.0, (2.0, 0.0, 0.0), (0.0, 0.3, 0.0), (0.2, 0.0, 0.0), (3.0, -3.0)
	)
	p2 = primitive(
		(0.3, 0.8, 0.3), 0.9, 0.02, 0.6, 0.2, 0.0, (0.0, 3.0, 0.0), (0.5, 0.0, 0.2), (-0.3, 0.2, 0.1), (2.0, -1.0)
	)
	keep3 = (-2.0, 2.0) if drop_third else (2.0, -2.0)
	p3 = primitive((0.2, 0.2, 0.2), 0.5, 0.0, 1.0, 0.05, 0.0, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, -0.6, 0.0), keep3)
	return gls.SmoothUnion(gls.SmoothUnion(p1, p2, (0.05,)), p3, (0.02,))


def write_assembly(path, programs, n_iters):
	"""Flat dict as SuperFit's Stats.get_dict() (programs as expr.sympy().state(); tensor and NumPy scalars)."""
	data = {'input_mesh_file': '/data/mesh.obj', 'n_iters': n_iters, 'success': True,
			'total_time': torch.tensor(12.5), 'evaluation.evaluation.iou@128': torch.tensor(0.91),
			'iter_0.pruned_n_prim': np.int64(2)}  # fmt: skip
	for key, expr in programs.items():
		data[key] = expr.sympy().state()
	with open(path, 'wb') as f:
		pickle.dump(data, f)


def reference(expr, q, dtype=torch.float64):
	"""SuperFit's deterministic evaluation of `expr` called directly: (N, 3) -> (N,)."""
	_, evaluate, Sketcher, fetch, _ = import_superfit()
	reduced = fetch(expr.tensor(device='cpu', dtype=dtype), relaxed_eval=False, device='cpu')
	return evaluate(reduced, Sketcher(device='cpu', dtype=dtype, n_dims=3, resolution=2), coords=q.to(dtype))


def test_program_selection_and_evaluation(tmp_path):
	gls, _, _, _, _ = import_superfit()
	path = os.path.join(tmp_path, 'primitive_assembly.pkl')
	expr = assembly_expr()
	other = gls.SmoothUnion(*assembly_expr().args[0].args[:2], (0.3,))  # a different program (larger blend)
	write_assembly(path, {'iter_0.best_program': other, 'iter_1.best_program': expr, 'iter_2.best_program': other}, 2)
	data = load_assembly(path)
	assert select_program(data, 'best')[0] == 'iter_1.best_program'  # get_best_expr: iteration n_iters - 1
	assert select_program(data, 'latest')[0] == 'iter_2.best_program'
	assert float(data['total_time']) == 12.5 and int(data['iter_0.pruned_n_prim']) == 2

	sdf = SuperFitSDF(path, device='cpu', dtype=torch.float64)
	assert sdf.program == 'iter_1.best_program'
	assert sdf.n_primitives == 2  # the third primitive's drop logit wins
	assert sorted(sdf.primitive_types) == ['SuperFrustumY', 'SuperFrustumZ']  # argmax of the axis logits
	assert sdf.metrics()['evaluation.evaluation.iou@128'] == pytest.approx(0.91)
	q = torch.rand(20000, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) * 2 - 1
	got = sdf(q.T.contiguous())
	assert got.shape == (20000,) and got.dtype == torch.float64
	assert torch.equal(got, reference(expr, q))
	# chunked evaluation gives the same values
	import superfit_sdf.assembly as sa

	saved = sa.QUERY_CHUNK
	try:
		sa.QUERY_CHUNK = 777
		assert torch.equal(sdf(q.T.contiguous()), got)
	finally:
		sa.QUERY_CHUNK = saved


def test_kept_primitive_changes_the_field(tmp_path):
	q = torch.tensor([[0.0, -0.6, 0.0]], dtype=torch.float64)  # centre of the third primitive
	dropped = SuperFitSDF(assembly_expr(drop_third=True).sympy().state(), dtype=torch.float64, device='cpu')
	kept = SuperFitSDF(assembly_expr(drop_third=False).sympy().state(), dtype=torch.float64, device='cpu')
	assert kept.n_primitives == 3 and dropped.n_primitives == 2
	assert float(kept(q.T)[0]) < 0 < float(dropped(q.T)[0])


def test_frames(tmp_path):
	"""Signs at known points; the mesh frame is the normalized frame mapped through p_n = s (p - c)."""
	state = assembly_expr().sympy().state()
	sdf = SuperFitSDF(state, dtype=torch.float64, device='cpu')
	inside = torch.tensor([[0.2, 0.0, 0.0], [-0.3, 0.2, 0.1]], dtype=torch.float64).T  # primitive centres
	outside = torch.tensor([[0.95, 0.95, 0.95], [-0.95, -0.95, 0.95]], dtype=torch.float64).T
	assert (sdf(inside) < 0).all() and (sdf(outside) > 0).all()
	lo, hi = sdf.bounds()
	assert np.allclose(lo, -1) and np.allclose(hi, 1)

	vertices = np.array(
		[[10.0, 2.0, -1.0], [14.0, 3.0, 1.0], [12.0, 6.0, 0.0]]
	)  # extent 4 (x and y), centre (12, 4, 0)
	s, c = mesh_normalization(vertices)
	assert s == pytest.approx(1.8 / 4.0) and np.allclose(c, [12.0, 4.0, 0.0])
	in_mesh = SuperFitSDF(state, mesh=vertices, dtype=torch.float64, device='cpu')
	p = torch.tensor(c)[:, None] + inside / s  # the primitive centres in the mesh frame
	assert torch.allclose(in_mesh(p), sdf(inside) / s, rtol=1e-12, atol=1e-12)
	lo, hi = in_mesh.bounds()
	assert np.allclose(lo, c - 1 / s) and np.allclose(hi, c + 1 / s)


def test_unsafe_pickle_is_rejected(tmp_path):
	path = os.path.join(tmp_path, 'evil.pkl')

	class Evil:
		def __reduce__(self):
			return (os.system, ('echo unsafe',))

	with open(path, 'wb') as f:
		pickle.dump({'iter_0.best_program': Evil()}, f)
	with pytest.raises(pickle.UnpicklingError):
		load_assembly(path)


def test_zero_bulge_warns():
	gls, _, _, _, _ = import_superfit()
	p = primitive(
		(0.5, 0.5, 0.5), 0.2, 0.0, 1.0, 0.0, 0.0, (2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (3.0, -3.0)
	)
	with pytest.warns(RuntimeWarning, match='bulge = 0'):
		SuperFitSDF(p.sympy().state(), device='cpu')


def test_scene_bake(tmp_path):
	from sdfview.scene import Scene
	from sdfview.sources.superfit import scene_from_superfit

	sdf = SuperFitSDF(assembly_expr().sympy().state(), device='cpu')
	scene = scene_from_superfit(sdf, resolution=24)
	path = os.path.join(tmp_path, 'sup.sdfx')
	scene.save(path)
	loaded = Scene.load(path)
	field = loaded.field('grid')
	values = loaded.field_arrays(field)['values']
	assert values.shape == (24, 24, 24) and values.min() < 0 < values.max()
	assert loaded.meta['n_primitives'] == 2
