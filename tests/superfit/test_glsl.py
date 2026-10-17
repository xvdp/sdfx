"""
superfit_sdf/glsl.py: SuperFit assemblies as generated GLSL (the composition, each primitive alone), evaluated by the
viewer's probe pass (sdfview Renderer.probe, OpenGL 3.3 through EGL) and compared with SuperFit's torch evaluation:
the assembly, each part, all parts as separate solids, the mesh frame; the WebGL2 (GLSL ES 3.00) compile; the
Shadertoy-style export. Skipped without geolipi / sysl3d, ext/superfit or an OpenGL context.
"""

import os
import shutil
import subprocess

import numpy as np
import pytest
import torch

pytest.importorskip('geolipi')
pytest.importorskip('sysl')

from superfit_sdf import SuperFitSDF  # noqa: E402
from superfit_sdf.assembly import import_superfit, superfit_root  # noqa: E402

try:
	superfit_root()
except ImportError:
	pytest.skip('ext/superfit not checked out', allow_module_level=True)

from test_superfit_sdf import assembly_expr, primitive  # noqa: E402

TOL = 1e-4  # float32 shader vs float64 torch, values of order 1


@pytest.fixture(scope='module')
def gl_ctx():
	try:
		from sdfview.backends.moderngl import create_context

		ctx = create_context()
	except Exception as e:  # noqa: BLE001
		pytest.skip(f'no OpenGL 3.3 context: {e}')
	yield ctx


def three_parts():
	"""Smooth union of three SuperFrusta (one with a zero rotation), all kept."""
	return SuperFitSDF(assembly_expr(drop_third=False).sympy().state(), dtype=torch.float64, device='cpu')


def queries(n=4000, seed=0, scale=1.0, center=(0.0, 0.0, 0.0)):
	q = np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n, 3)) * 0.9
	return q / scale + np.asarray(center)


def torch_parts(sdf, q):
	"""(n_parts, n) SuperFit's torch evaluation of each part (primitive under its transforms) in the SDF's frame."""
	from superfit_sdf.glsl import part_expressions

	_, evaluate, Sketcher, _, _ = import_superfit()
	sk = Sketcher(device='cpu', dtype=torch.float64, n_dims=3, resolution=2)
	qn = (torch.as_tensor(q) - torch.as_tensor(sdf.center)) * sdf.scale
	out = [evaluate(e.tensor(dtype=torch.float64, device='cpu'), sk, coords=qn).reshape(-1) / sdf.scale
		   for e in part_expressions(sdf)]  # fmt: skip
	return torch.stack(out).numpy()


def probe(ctx, sdf, q, part=-1):
	from sdfview import ViewSettings
	from sdfview.backends.moderngl import Renderer
	from sdfview.sources.superfit import scene_from_superfit

	scene = scene_from_superfit(sdf, resolution=None)
	r = Renderer(ctx, scene)
	try:
		return r.probe(q, 'superfit', ViewSettings(part=part))
	finally:
		r.release()


def test_assembly_and_parts_match_torch(gl_ctx):
	from superfit_sdf.glsl import assembly_glsl

	sdf = three_parts()
	g = assembly_glsl(sdf)
	assert g['n_parts'] == 3 and g['part_types'] == ['SuperFrustumY', 'SuperFrustumZ', 'SuperFrustumX']
	assert g['code'].count('= AxisAngleRotate3D(') == 4  # 2 in the assembly, 2 in the parts: the zero one dropped
	q = queries()
	ref = sdf(torch.as_tensor(q).T).numpy()
	whole = probe(gl_ctx, sdf, q)
	err = np.abs(whole[:, 0] - ref).max()
	print(f'assembly: max |glsl - torch| = {err:.2e} over {len(q)} points, range [{ref.min():.2f}, {ref.max():.2f}]')
	assert err < TOL
	parts = torch_parts(sdf, q)
	for i in range(3):
		got = probe(gl_ctx, sdf, q, part=i)[:, 0]
		e = np.abs(got - parts[i]).max()
		print(f'part {i}: max |glsl - torch| = {e:.2e}')
		assert e < TOL
	all_parts = probe(gl_ctx, sdf, q, part=-2)[:, 0]
	assert np.abs(all_parts - parts.min(axis=0)).max() < TOL
	# the gradient of the probe is the central difference of the same field: unit length away from junctions
	g_norm = np.linalg.norm(whole[:, 1:], axis=1)
	assert np.median(np.abs(g_norm - 1.0)) < 0.05


def test_mesh_frame(gl_ctx):
	"""φ(p) = φ_n(s (p - c)) / s in the generated code as in SuperFitSDF(mesh=...)."""
	rng = np.random.default_rng(3)
	vertices = rng.uniform(-1, 1, size=(50, 3)) * np.array([3.0, 1.0, 2.0]) + np.array([5.0, -2.0, 1.0])
	sdf = SuperFitSDF(assembly_expr().sympy().state(), mesh=vertices, dtype=torch.float64, device='cpu')
	q = queries(seed=4, scale=sdf.scale, center=sdf.center)
	ref = sdf(torch.as_tensor(q).T).numpy()
	got = probe(gl_ctx, sdf, q)[:, 0]
	assert np.abs(got - ref).max() < TOL / sdf.scale


def test_union_of_parts_and_single_primitive(gl_ctx):
	"""An assembly of one primitive (no combinator) and a plain Union."""
	gls, _, _, _, _ = import_superfit()
	p = primitive(
		(0.5, 0.3, 0.4), 0.2, 0.01, 0.8, 0.15, 0.0, (0.0, 2.0, 0.0), (0.1, 0.2, 0.3), (0.1, 0.0, 0.0), (3, -3)
	)
	r = primitive((0.2, 0.5, 0.2), 0.6, 0.0, 1.0, 0.3, 0.0, (2.0, 0.0, 0.0), (0.0, 0.0, 0.4), (0.0, 0.3, 0.0), (3, -3))
	q = queries(seed=5)
	for expr in (p, gls.Union(p, r)):
		sdf = SuperFitSDF(expr.sympy().state(), dtype=torch.float64, device='cpu')
		ref = sdf(torch.as_tensor(q).T).numpy()
		assert np.abs(probe(gl_ctx, sdf, q)[:, 0] - ref).max() < TOL


def test_webgl2_compile_and_shadertoy_export(tmp_path, gl_ctx):
	"""The generated code inside surface.frag compiles as GLSL ES 3.00 (glslangValidator, if installed); the
	Shadertoy-style export compiles as a fragment shader with the Shadertoy uniforms and renders a non-empty image."""
	from sdfview.shaders import glsl_source
	from superfit_sdf.glsl import assembly_glsl, assembly_shadertoy

	sdf = three_parts()
	validator = shutil.which('glslangValidator')
	if validator:
		for target in ('gl330', 'webgl2'):
			src = glsl_source('surface.frag', target, {'FIELD_GLSL': None}, assembly_glsl(sdf)['code'])
			path = os.path.join(tmp_path, f'surface_{target}.frag')
			with open(path, 'w') as f:
				f.write(src)
			res = subprocess.run([validator, path], capture_output=True, text=True)
			assert res.returncode == 0, res.stdout[-2000:]
	code = assembly_shadertoy(sdf)
	frag = (
		'#version 330 core\nuniform vec3 iResolution;\nuniform float iTime;\nuniform vec4 iMouse;\n'
		+ code
		+ '\nout vec4 outColor;\nvoid main() { mainImage(outColor, gl_FragCoord.xy); }\n'
	)
	vert = '#version 330 core\nvoid main() { vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));'
	vert += ' gl_Position = vec4(2.0 * p - 1.0, 0.0, 1.0); }\n'
	prog = gl_ctx.program(vertex_shader=vert, fragment_shader=frag)
	prog['iResolution'] = (96.0, 72.0, 1.0)
	fbo = gl_ctx.simple_framebuffer((96, 72))
	fbo.use()
	gl_ctx.vertex_array(prog, []).render(vertices=3)
	img = np.frombuffer(fbo.read(components=3), np.uint8).reshape(72, 96, 3)
	background = (img > 240).all(axis=2)
	assert 0.02 < 1 - background.mean() < 0.9  # the assembly covers part of the image
	fbo.release()
