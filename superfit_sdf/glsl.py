"""
SuperFit assembly -> GLSL, generated at run time from the user's SuperFit checkout: the assembly as a function
composing its primitives (transforms, nested smooth unions), and each primitive on its own.

Where the code comes from:
	* the primitives' GLSL (SuperFrustum, SPTaperedOnion, ...) is SuperFit's own (superfit/shader, Adobe Research
	  License). Importing superfit.shader registers it in sysl's shader-module registry (sysl.shader.shader_module.SMMap,
	  MIT); it is read from there at run time, with its dependencies and constants, and copied into the generated text
	  only, never into this repository. The generated GLSL is a derivative of SuperFit under its license.
	* the transforms' GLSL (Translate3D, AxisAngleRotate3D, EulerRotate3D) comes from the same registry (sysl, MIT);
	* written here (MIT): the composition of a reduced assembly (superfit_sdf.SuperFitSDF.expression, SuperFit's
	  deterministic reduction) into the functions below, and the smooth union.

Generated interface, prefix P (default 'assembly'):

	const int P_PART_COUNT                      number of primitives (parts)
	float P_part<i>(vec3 p)                     primitive i alone: its transforms from the root to it, then the primitive
	float PDistance(vec3 p)                     the assembly: the expression tree evaluated bottom-up
	float PPartDistance(int i, vec3 p)          part i alone
	float PParts(vec3 p, out int id)            the parts as separate solids: min over i of P_part<i>(p), id = argmin
	int PNearestPart(vec3 p)                    argmin over i of P_part<i>(p) (to color the assembly by part)

with p in the frame of the SDF (SuperFit's normalized frame, or the mesh frame: φ(p) = φ_n(s (p - c)) / s).

Operations (the expression's, in geolipi's semantics; checked against SuperFit's torch evaluation in
tests/superfit/test_glsl.py):
	Translate3D(e, t)         e(p - t)
	AxisAngleRotate3D(e, a)   e(p R(a)), R(a) the Rodrigues rotation of the axis-angle a (identity for a = 0, where
	                          the registry's GLSL would normalize a zero vector)
	EulerRotate3D(e, a)       the registry's GLSL
	SmoothUnion(a, b, k)      h = clamp(1/2 + (b - a) / (2 (k + 1e-9)), 0, 1),  d = b (1 - h) + a h - k h (1 - h)
	Union(a, b, ...)          min;  Intersection: max;  Difference(a, b): max(a, -b)
Parameters are inlined as float32 literals ('%.9g' round-trips float32). Every module and constant is wrapped in an
include guard (#ifndef SF_MODULE_<name>), so the code of several assemblies can share one program.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

SMOOTH_UNION = """\
#ifndef SF_MODULE_sdfxSmoothUnion
#define SF_MODULE_sdfxSmoothUnion
// geolipi sdf_smooth_union: h = clamp(1/2 + (b - a) / (2 (k + 1e-9)), 0, 1), d = mix(b, a, h) - k h (1 - h)
float sdfxSmoothUnion(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / (k + 1e-9), 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}
#endif
"""

SUPPORTED_TRANSFORMS = ('Translate3D', 'AxisAngleRotate3D', 'EulerRotate3D')


def _floats(arg) -> List[float]:
	values = list(arg) if hasattr(arg, '__iter__') else [arg]
	return [float(v) for v in values]


def _literal(values: List[float]) -> str:
	parts = ['%.9g' % v for v in values]
	parts = [s if any(c in s for c in '.e') or s in ('inf', '-inf', 'nan') else s + '.0' for s in parts]
	if len(parts) == 1:
		return parts[0]
	if len(parts) in (2, 3, 4):
		return f'vec{len(parts)}({", ".join(parts)})'
	raise ValueError(f'parameter of {len(parts)} values has no GLSL vector type')


def _registry():
	"""(sysl's shader-module registry with SuperFit's modules registered, sysl's constants)."""
	from .assembly import import_superfit

	import_superfit()
	import superfit.shader  # noqa: F401  (registers SuperFit's GLSL modules with sysl)
	from sysl.shader.shader_module import SMMap
	from sysl.shader.shader_templates.common import CONSTANTS

	return SMMap, CONSTANTS


def _children_and_params(node, gls) -> Tuple[list, list]:
	children, params = [], []
	for arg in node.args:
		(children if isinstance(arg, gls.GLBase) else params).append(arg)
	return children, params


class _Generator:
	def __init__(self, gls, smmap):
		self.gls = gls
		self.smmap = smmap
		self.modules: List[str] = []  # in first-use order, dependencies resolved later
		self.parts: List[Tuple[list, object]] = []  # (transforms root -> leaf, primitive node)
		self.lines: List[str] = []
		self.count = 0

	def _use(self, name: str) -> None:
		if name not in self.smmap:
			raise NotImplementedError(f'no GLSL module named {name!r} registered by sysl / SuperFit')
		if name not in self.modules:
			self.modules.append(name)

	def _var(self, kind: str) -> str:
		self.count += 1
		return f'{kind}{self.count}'

	def transform(self, node, pos: str) -> Optional[str]:
		"""Code line applying transform `node` to position variable `pos`; None for an identity rotation."""
		name = type(node).__name__
		if name not in SUPPORTED_TRANSFORMS:
			raise NotImplementedError(f'transform {name} is not supported (supported: {SUPPORTED_TRANSFORMS})')
		_, params = _children_and_params(node, self.gls)
		values = [_floats(p) for p in params]
		if name == 'AxisAngleRotate3D' and not any(values[0]):
			return None  # zero axis-angle: identity (geolipi clamps the angle; the GLSL would normalize 0)
		self._use(name)
		out = self._var('p')
		self.lines.append(f'    vec3 {out} = {name}({pos}, {", ".join(_literal(v) for v in values)});')
		return out

	def primitive_call(self, node, pos: str) -> str:
		name = type(node).__name__
		self._use(name)
		_, params = _children_and_params(node, self.gls)
		args = ', '.join(_literal(_floats(p)) for p in params)
		return f'{name}({pos}, {args})'

	def walk(self, node, pos: str, path: list) -> str:
		"""Code computing the distance of `node` at `pos` (its value: a float variable); collects the parts."""
		gls = self.gls
		children, params = _children_and_params(node, gls)
		if isinstance(node, gls.Primitive3D):
			index = len(self.parts)
			self.parts.append((list(path), node))
			out = self._var('d')
			self.lines.append(f'    float {out} = {self.primitive_call(node, pos)};  // part {index}')
			return out
		name = type(node).__name__
		if name in SUPPORTED_TRANSFORMS:
			new_pos = self.transform(node, pos)
			return self.walk(children[0], new_pos or pos, path + [node])
		values = [self.walk(c, pos, path) for c in children]
		out = self._var('d')
		if name == 'SmoothUnion':
			k = _literal(_floats(params[0]))
			self.lines.append(f'    float {out} = sdfxSmoothUnion({values[0]}, {values[1]}, {k});')
		elif name in ('Union', 'JoinUnion'):
			expr = values[0]
			for v in values[1:]:
				expr = f'min({expr}, {v})'
			self.lines.append(f'    float {out} = {expr};')
		elif name == 'Intersection':
			expr = values[0]
			for v in values[1:]:
				expr = f'max({expr}, {v})'
			self.lines.append(f'    float {out} = {expr};')
		elif name == 'Difference':
			self.lines.append(f'    float {out} = max({values[0]}, -{values[1]});')
		else:
			raise NotImplementedError(f'node {name} is not supported')
		return out


def _module_code(names: List[str], smmap, constants) -> str:
	"""Code of the modules and their dependencies, dependencies first, each in an include guard; constants first."""
	order, seen = [], set()

	def visit(name):
		if name in seen:
			return
		seen.add(name)
		module = smmap[name]()
		for dep in module.dependencies:
			visit(dep)
		order.append(module)

	for name in names:
		visit(name)
	out = []
	for var in sorted({v for m in order for v in m.vardeps}):
		if var not in constants:
			raise KeyError(f'constant {var!r} used by a module is not in sysl CONSTANTS')
		kind, value = constants[var]
		literal = _literal(_floats(value)) if kind.startswith(('float', 'vec')) else str(value)
		out.append(f'#ifndef SF_CONST_{var}\n#define SF_CONST_{var}\nconst {kind} {var} = {literal};\n#endif')
	for m in order:
		if not m.code:
			raise NotImplementedError(f'module {m.name} ({type(m).__name__}) generates its code from a template')
		out.append(f'#ifndef SF_MODULE_{m.name}\n#define SF_MODULE_{m.name}\n// Module: {m.name}\n{m.code}\n#endif')
	return '\n'.join(out) + '\n'


def assembly_glsl(sdf, prefix: str = 'assembly', header: bool = True) -> dict:
	"""
	GLSL of a superfit_sdf.SuperFitSDF (see the module doc). Returns {'code', 'prefix', 'n_parts', 'part_types',
	'functions': names of the generated entry points}.
	"""
	if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', prefix):
		raise ValueError(f'prefix {prefix!r} is not a GLSL identifier')
	smmap, constants = _registry()
	import geolipi.symbolic as gls

	expr = sdf.expression.sympy()
	gen = _Generator(gls, smmap)
	root = gen.walk(expr, 'q', [])
	tree_lines = gen.lines

	# parts alone: their transforms from the root, then the primitive
	part_functions = []
	for i, (path, node) in enumerate(gen.parts):
		gen.lines, pos = [], 'q'
		for t in path:
			pos = gen.transform(t, pos) or pos
		body = '\n'.join(gen.lines)
		call = gen.primitive_call(node, pos)
		part_functions.append(
			f'float {prefix}_part{i}(vec3 p) {{  // {type(node).__name__}\n'
			f'    vec3 q = {prefix}_frame(p);\n{body + chr(10) if body else ""}'
			f'    return {call} / {prefix}_FRAME_SCALE;\n}}'
		)
	n = len(gen.parts)
	scale = float(getattr(sdf, 'scale', 1.0))
	center = [float(c) for c in getattr(sdf, 'center', (0.0, 0.0, 0.0))]
	frame = (
		f'const float {prefix}_FRAME_SCALE = {_literal([scale])};  // phi(p) = phi_n(s (p - c)) / s\n'
		f'vec3 {prefix}_frame(vec3 p) {{ return {_literal([scale])} * (p - {_literal(center)}); }}'
	)
	select = '\n'.join(f'    if (i == {i}) return {prefix}_part{i}(p);' for i in range(n))
	parts_min = '\n'.join(f'    d = {prefix}_part{i}(p); if (d < best) {{ best = d; id = {i}; }}' for i in range(n))
	code = []
	if header:
		code.append(
			f'// SuperFit assembly ({sdf.program}, {n} parts: {", ".join(type(x[1]).__name__ for x in gen.parts)}),\n'
			'// generated by superfit_sdf/glsl.py from the SuperFit checkout at run time. The primitive modules are\n'
			"// SuperFit's (Adobe Research License: non-commercial research use); the composition below is sdfx's (MIT).\n"
		)
	code.append(_module_code(gen.modules, smmap, constants))
	code.append(SMOOTH_UNION)
	code.append(f'const int {prefix}_PART_COUNT = {n};\n{frame}\n')
	code.append('\n\n'.join(part_functions) + '\n')
	code.append(
		f'float {prefix}Distance(vec3 p) {{\n    vec3 q = {prefix}_frame(p);\n'
		+ '\n'.join(tree_lines)
		+ f'\n    return {root} / {prefix}_FRAME_SCALE;\n}}\n'
	)
	code.append(f'float {prefix}PartDistance(int i, vec3 p) {{\n{select}\n    return 1e30;\n}}\n')
	code.append(
		f'float {prefix}Parts(vec3 p, out int id) {{\n    float best = 1e30, d;\n    id = -1;\n{parts_min}\n'
		'    return best;\n}\n'
	)
	code.append(f'int {prefix}NearestPart(vec3 p) {{\n    int id;\n    {prefix}Parts(p, id);\n    return id;\n}}\n')
	return {
		'code': '\n'.join(code),
		'prefix': prefix,
		'n_parts': n,
		'part_types': [type(x[1]).__name__ for x in gen.parts],
		'functions': {
			'distance': f'{prefix}Distance',
			'part_distance': f'{prefix}PartDistance',
			'parts': f'{prefix}Parts',
			'nearest_part': f'{prefix}NearestPart',
			'part_count': f'{prefix}_PART_COUNT',
		},
	}


def part_expressions(sdf) -> list:
	"""
	The parts of a SuperFitSDF as geolipi expressions (each primitive under its transforms from the root), in the
	order of the generated <prefix>_part<i>: the torch counterpart of the per-part GLSL.
	"""
	import geolipi.symbolic as gls

	_registry()
	parts = []

	def walk(node, path):
		children, _ = _children_and_params(node, gls)
		if isinstance(node, gls.Primitive3D):
			e = node
			for t in reversed(path):
				_, params = _children_and_params(t, gls)
				e = type(t)(e, *params)
			parts.append(e)
			return
		if type(node).__name__ in SUPPORTED_TRANSFORMS:
			walk(children[0], path + [node])
			return
		for c in children:
			walk(c, path)

	walk(sdf.expression.sympy(), [])
	return parts


SHADERTOY_MAIN = """
// ---- viewer (Shadertoy conventions: iResolution, iTime, iMouse; mainImage) ------------------------------------------
// PART: -1 the assembly, -2 all parts as separate solids, i >= 0 part i alone. COLOR_PARTS: color by part.
#ifndef PART
#define PART -1
#endif
#ifndef COLOR_PARTS
#define COLOR_PARTS 1
#endif

float sceneDistance(vec3 p) {
    if (PART >= 0) return PREFIXPartDistance(PART, p);
    if (PART == -2) { int id; return PREFIXParts(p, id); }
    return PREFIXDistance(p);
}

vec3 sceneNormal(vec3 p) {
    const float h = 1e-3;
    vec2 e = vec2(1.0, -1.0);
    return normalize(e.xyy * sceneDistance(p + e.xyy * h) + e.yyx * sceneDistance(p + e.yyx * h)
                     + e.yxy * sceneDistance(p + e.yxy * h) + e.xxx * sceneDistance(p + e.xxx * h));
}

vec3 partColor(int i) {  // golden-ratio hues
    float hue = fract(0.13 + 0.618034 * float(i));
    vec3 k = clamp(abs(fract(hue + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
    return mix(vec3(1.0), k, 0.55);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 m = iMouse.z > 0.0 ? iMouse.xy / iResolution.xy : vec2(0.15 + 0.02 * iTime, 0.6);
    float yaw = 6.2831853 * m.x, pitch = 3.0 * (m.y - 0.5);
    vec3 dir = vec3(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw));
    vec3 ro = PREFIX_VIEW_CENTER + 3.2 * PREFIX_VIEW_RADIUS * dir;
    vec3 f = normalize(PREFIX_VIEW_CENTER - ro), r = normalize(cross(f, vec3(0.0, 1.0, 0.0))), u = cross(r, f);
    vec2 uv = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    vec3 rd = normalize(uv.x * r + uv.y * u + 2.0 * f);
    float t = 0.0;
    bool hit = false;
    for (int i = 0; i < 256; i++) {
        float d = sceneDistance(ro + t * rd);
        if (abs(d) < 1e-4 * PREFIX_VIEW_RADIUS) { hit = true; break; }
        t += 0.9 * d;
        if (t > 8.0 * PREFIX_VIEW_RADIUS) break;
    }
    vec3 col = vec3(0.96);
    if (hit) {
        vec3 p = ro + t * rd, n = sceneNormal(p);
        vec3 base = vec3(0.42, 0.62, 0.90);
        if (COLOR_PARTS != 0) base = partColor(PART >= 0 ? PART : PREFIXNearestPart(p));
        float diffuse = max(dot(n, normalize(vec3(0.4, 0.8, 0.3) - rd)), 0.0);
        col = base * (0.25 + 0.75 * diffuse);
    }
    fragColor = vec4(pow(col, vec3(0.4545)), 1.0);
}
"""


def assembly_shadertoy(sdf, prefix: str = 'assembly') -> str:
	"""
	Standalone Shadertoy-style fragment code: the generated assembly functions plus a mainImage that sphere traces
	them (paste into shadertoy.com, or prepend `#version` and the iResolution / iTime / iMouse uniforms and call
	mainImage from main, as examples/hosts/shadertoy_moderngl.py does). Set PART / COLOR_PARTS with #define.
	"""
	g = assembly_glsl(sdf, prefix)
	radius = float(1.0 / getattr(sdf, 'scale', 1.0))
	center = [float(c) for c in getattr(sdf, 'center', (0.0, 0.0, 0.0))]
	view = (
		f'const float {prefix}_VIEW_RADIUS = {_literal([radius])};  // half size of the scene (normalized frame: 1)\n'
		f'const vec3 {prefix}_VIEW_CENTER = {_literal(center)};  // the camera orbits this point\n'
	)
	return g['code'] + '\n' + view + SHADERTOY_MAIN.replace('PREFIX', prefix)
