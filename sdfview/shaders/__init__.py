"""
Shader sources, one code base per shading language, with the same module and function names in each:

	glsl/   GLSL, compiled for several targets by prepending a version header:
	            gl330   '#version 330 core'   desktop OpenGL 3.3+ (moderngl)
	            webgl2  '#version 300 es'     WebGL2 / OpenGL ES 3.0 (precision qualifiers added)
	            three   the webgl2 header without '#version' (three.js adds it to GLSL3 RawShaderMaterials)
	        modules joined with `#include "file.glsl"` (resolved here, each file once per program: GLSL has no include);
	        `#define`s and #ifdef are GLSL's own preprocessor.
	wgsl/   WGSL (WebGPU, three.js WebGPU). WGSL has no preprocessor: `#include`, `#ifdef`, `#ifndef`,
	        `#if defined(X)`, `#elif defined(X)`, `#else`, `#endif`, `#error` are resolved here (wgsl_source), and
	        valued defines become module constants (`const BVH_K_MAX = 64;`), usable as array sizes.
	slang/  Slang (slangpy): modules with `import`, field types as structs implementing an interface (no preprocessor
	        selection); compiled by slangpy.
The JavaScript hosts assemble GLSL and WGSL with sdfview/web/core/shaders.js, which follows the same rules (tested:
identical output).
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional

GLSL_DIR = os.path.join(os.path.dirname(__file__), 'glsl')
WGSL_DIR = os.path.join(os.path.dirname(__file__), 'wgsl')
SLANG_DIR = os.path.join(os.path.dirname(__file__), 'slang')

HEADERS = {
	'gl330': '#version 330 core\n',
	'webgl2': (
		'#version 300 es\n'
		'precision highp float;\nprecision highp int;\n'
		'precision highp sampler2D;\nprecision highp isampler2D;\nprecision highp sampler3D;\n'
	),
	# three.js RawShaderMaterial with glslVersion GLSL3 prepends '#version 300 es' (and its SHADER_TYPE / SHADER_NAME
	# defines) itself: the WebGL2 header without the version line
	'three': (
		'precision highp float;\nprecision highp int;\n'
		'precision highp sampler2D;\nprecision highp isampler2D;\nprecision highp sampler3D;\n'
	),
}

_INCLUDE = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)


def _expand(name: str, seen: set) -> str:
	if name in seen:
		return f'// (already included: {name})\n'
	seen.add(name)
	with open(os.path.join(GLSL_DIR, name)) as f:
		text = f.read()
	return _INCLUDE.sub(lambda m: _expand(m.group(1), seen), text)


def glsl_source(
	name: str, target: str = 'gl330', defines: Optional[Dict[str, object]] = None, prelude: Optional[str] = None
) -> str:
	"""
	Complete source of the program stage `name` (e.g. 'surface.frag') for `target`, with `#define`s, and `prelude`
	(e.g. the generated code of a "glsl" field) inserted after them, before the modules.
	"""
	if target not in HEADERS:
		raise ValueError(f'unknown target {target!r} (one of {list(HEADERS)})')
	lines = [HEADERS[target]]
	for key, value in (defines or {}).items():
		lines.append(f'#define {key}' if value is None or value is True else f'#define {key} {value}')
	if prelude:
		lines.append(prelude)
	return '\n'.join(lines) + '\n' + _expand(name, set())


_DIRECTIVE = re.compile(r'^\s*#\s*(include|ifdef|ifndef|if|elif|else|endif|error)\b\s*(.*?)\s*$')
_DEFINED = re.compile(r'^!?\s*defined\s*\(\s*(\w+)\s*\)$')


def _condition(expr: str, defines: Dict[str, object]) -> bool:
	"""`defined(X)`, `!defined(X)`, or a chain of them joined by `||` / `&&` (no parentheses, no mixing)."""
	for op, combine in (('||', any), ('&&', all)):
		if op in expr:
			return combine(_condition(e.strip(), defines) for e in expr.split(op))
	m = _DEFINED.match(expr)
	if not m:
		raise ValueError(f'unsupported #if expression {expr!r} (use defined(X), !defined(X), ||, &&)')
	return (m.group(1) in defines) != expr.startswith('!')


def preprocess(root: str, name: str, defines: Dict[str, object], seen: Optional[set] = None) -> str:
	"""
	Resolve #include (each file once) and conditional blocks of file `name` under `root`. Lines of directives are
	dropped; the rest is kept verbatim. Used for WGSL (which has no preprocessor) by wgsl_source.
	"""
	seen = set() if seen is None else seen
	if name in seen:
		return f'// (already included: {name})\n'
	seen.add(name)
	with open(os.path.join(root, name)) as f:
		lines = f.read().split('\n')
	out = []
	stack = []  # (active, branch_taken) per open #if
	active = True
	for line in lines:
		m = _DIRECTIVE.match(line)
		if not m:
			if active:
				out.append(line)
			continue
		kind, arg = m.group(1), m.group(2)
		if kind in ('ifdef', 'ifndef', 'if'):
			cond = {'ifdef': lambda: arg in defines, 'ifndef': lambda: arg not in defines}.get(
				kind, lambda: _condition(arg, defines)
			)()
			stack.append((active, active and cond))
			active = active and cond
		elif kind == 'elif':
			parent, taken = stack[-1]
			cond = parent and not taken and _condition(arg, defines)
			stack[-1] = (parent, taken or cond)
			active = cond
		elif kind == 'else':
			parent, taken = stack[-1]
			active = parent and not taken
			stack[-1] = (parent, True)
		elif kind == 'endif':
			active = stack.pop()[0]
		elif not active:
			continue
		elif kind == 'include':
			out.append(preprocess(root, arg.strip('"'), defines, seen).rstrip('\n'))
		elif kind == 'error':
			raise ValueError(f'{name}: #error {arg}')
	if stack:
		raise ValueError(f'{name}: unterminated #if')
	return '\n'.join(out) + '\n'


def wgsl_source(name: str, defines: Optional[Dict[str, object]] = None, prelude: Optional[str] = None) -> str:
	"""
	Complete WGSL of the program `name` (e.g. 'surface.wgsl'): valued defines as module constants
	(`const NAME = value;`), `prelude`, then the preprocessed modules.
	"""
	defines = dict(defines or {})
	lines = [f'const {k} = {v};' for k, v in defines.items() if v is not None and v is not True]
	if prelude:
		lines.append(prelude)
	return '\n'.join(lines) + '\n' + preprocess(WGSL_DIR, name, defines)
