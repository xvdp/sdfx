"""
SUP — a SuperFit assembly without fitting or downloading: three primitives built with SuperFit's own classes and
written in its pickle layout (the layout of primitive_assembly.pkl / final_content.pkl), to try the SuperFit side of
the pipeline (02_assembly_to_glsl.py, the viewer, the comparison script) before real fits exist.

	INPUT   none (SuperFit's classes from ext/superfit; geolipi and sysl3d: pip install -e ".[superfit]")
	OUTPUT  <out>/test_assembly.pkl
	python examples/superfit/01_test_assembly.py
	python -m sdfview superfit examples/out/superfit/test_assembly.pkl --view --color-parts

The program is SuperFit's optimization-stage form: marked, stochastic, variable-axis SuperFrusta in nested smooth
unions; superfit_sdf reduces it deterministically (keep / drop and axis by argmax), as SuperFit's evaluation.
"""

import argparse
import os
import pickle
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

import torch  # noqa: E402

from superfit_sdf import import_superfit  # noqa: E402


def primitive(size, roundness, dilate, taper, bulge, onion, axis_logits, axis_angle, translate, keep_drop):
	"""PrimitiveMarker(StochasticPrimitive(Translate3D(AxisAngleRotate3D(VarAxisSF(...), a), t), (keep, drop)))."""
	gls, _, _, _, _ = import_superfit()
	from superfit.symbolic.base import PrimitiveMarker, StochasticPrimitive
	from superfit.symbolic.var_axis import VarAxisSF

	p = VarAxisSF(tuple(size), (roundness,), (dilate,), (taper,), (bulge,), (onion,), tuple(axis_logits))
	return PrimitiveMarker(
		StochasticPrimitive(gls.Translate3D(gls.AxisAngleRotate3D(p, tuple(axis_angle)), tuple(translate)), keep_drop)
	)


def main():
	ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
	ap.add_argument('-o', '--out', default=os.path.join(REPO, 'examples', 'out', 'superfit'))
	a = ap.parse_args()
	gls, _, _, _, _ = import_superfit()
	# size (x, y, z), roundness, dilate, taper, bulge (!= 0, see superfit_sdf/NOTES.md), onion, axis logits (argmax
	# 0, 1, 2 -> SuperFrustumY, Z, X: SuperFit's VARAXIS_MAP),
	# axis-angle rotation, translation, (keep, drop) logits
	body = primitive((0.9, 0.5, 0.6), 0.3, 0.0, 0.8, 0.1, 0.0, (0, 3, 0), (0.0, 0.3, 0.0), (0.1, -0.1, 0.0), (3, -3))
	arm = primitive((0.25, 0.9, 0.25), 0.9, 0.02, 0.6, 0.2, 0.0, (0, 0, 3), (0.5, 0.0, 0.2), (-0.4, 0.4, 0.1), (3, -3))
	foot = primitive((0.3, 0.3, 0.3), 0.5, 0.0, 1.0, 0.05, 0.0, (3, 0, 0), (0.0, 0.0, 0.0), (0.3, -0.6, 0.0), (3, -3))
	program = gls.SmoothUnion(gls.SmoothUnion(body, arm, (0.06,)), foot, (0.03,))
	data = {'input_mesh_file': 'none', 'n_iters': 1, 'success': True, 'total_time': torch.tensor(0.0),
			'iter_0.best_program': program.sympy().state()}  # fmt: skip
	os.makedirs(a.out, exist_ok=True)
	path = os.path.join(a.out, 'test_assembly.pkl')
	with open(path, 'wb') as f:
		pickle.dump(data, f)
	print(f'wrote {path}\nnext: python examples/superfit/02_assembly_to_glsl.py {path}')


if __name__ == '__main__':
	main()
