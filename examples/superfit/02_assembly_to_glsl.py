"""
SUP — a SuperFit assembly from Python: torch evaluation, the GLSL of its composition and of each primitive, the parts
in torch, a scene (generated GLSL + optional grid bake) and a Shadertoy-style file.

	INPUT   an assembly: SuperFit's primitive_assembly.pkl / final_content.pkl (03_fit_superfit.sh, the Hugging Face
	        release) or examples/out/superfit/test_assembly.pkl (01_test_assembly.py)
	OUTPUT  <out>/<name>.sdfx    field '<name>' (type glsl: whole / parts) + '<name>_grid' (torch bake, --bake N)
	        <out>/<name>.glsl    Shadertoy-style code (mainImage; #define PART, COLOR_PARTS)
	        printed: the generated composition, φ from torch and per part
	python examples/superfit/02_assembly_to_glsl.py examples/out/superfit/test_assembly.pkl --bake 96
	python -m sdfview view examples/out/superfit/test_assembly.sdfx --color-parts      # [ ] parts, U colors
	python examples/hosts/shadertoy_moderngl.py examples/out/superfit/test_assembly.glsl --part -2

The same from the command line: python -m sdfview superfit A.pkl -o A.sdfx --shadertoy A.glsl [--bake N] [--view].
"""

import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

import torch  # noqa: E402

from superfit_sdf import SuperFitSDF, assembly_glsl, assembly_shadertoy, import_superfit, part_expressions  # noqa: E402


def main():
	ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
	ap.add_argument('assembly')
	ap.add_argument('--program', default='best', help="'best', 'latest' or a key of the pickle")
	ap.add_argument('--mesh', default=None, help='evaluate in the frame of the input mesh')
	ap.add_argument('--bake', type=int, default=0, help='also bake the torch evaluation on a grid')
	ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
	ap.add_argument('-o', '--out', default=os.path.join(REPO, 'examples', 'out', 'superfit'))
	a = ap.parse_args()
	name = os.path.splitext(os.path.basename(a.assembly))[0]

	# torch evaluation: SuperFit's own code (geolipi recursive_evaluate), deterministic reduction
	sdf = SuperFitSDF(a.assembly, program=a.program, mesh=a.mesh, device=a.device)
	print(sdf, sdf.primitive_types)
	q = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.9, 0.0]], device=sdf.device).T  # (3, Q)
	print('φ (torch):', sdf(q).tolist())

	# the composition as GLSL (primitives from the SuperFit checkout, composition written by superfit_sdf/glsl.py)
	g = assembly_glsl(sdf)
	code = g['code']
	print(f'\n{g["n_parts"]} parts {g["part_types"]}; entry points {g["functions"]}\n')
	print(code[code.index('float assemblyDistance') : code.index('float assemblyPartDistance')])

	# the parts in torch: each primitive under its transforms (what assembly_part<i> computes)
	_, evaluate, Sketcher, _, _ = import_superfit()
	sk = Sketcher(device=sdf.device, dtype=sdf.dtype, n_dims=3, resolution=2)
	center = torch.as_tensor(sdf.center, dtype=sdf.dtype, device=sdf.device)[:, None]
	qn = ((q - center) * sdf.scale).T.contiguous()  # SuperFit's normalized frame, (Q, 3) as geolipi takes it
	for i, e in enumerate(part_expressions(sdf)):
		phi_i = evaluate(e.tensor(dtype=sdf.dtype, device=sdf.device), sk, coords=qn) / sdf.scale
		print(f'part {i} ({g["part_types"][i]}) φ (torch):', phi_i.reshape(-1).tolist())

	from sdfview.sources.superfit import scene_from_superfit

	os.makedirs(a.out, exist_ok=True)
	scene = scene_from_superfit(sdf, resolution=a.bake or None, name=name, meta={'input': a.assembly})
	scene.save(os.path.join(a.out, f'{name}.sdfx'))
	with open(os.path.join(a.out, f'{name}.glsl'), 'w') as f:
		f.write(assembly_shadertoy(sdf))
	print(f'\nwrote {a.out}/{name}.sdfx ({[f["name"] for f in scene.fields]}) and {name}.glsl')


if __name__ == '__main__':
	main()
