"""
Shader sources. One GLSL code base (shaders/glsl) compiled for several targets by prepending a version header:

	gl330   '#version 330 core'   desktop OpenGL 3.3+ (moderngl)
	webgl2  '#version 300 es'     WebGL2 / OpenGL ES 3.0 (precision qualifiers added)

Modules are joined with `#include "file.glsl"` (resolved here, each file once per program: GLSL has no include).
Later targets keep the same module names in sibling folders (shaders/slang, shaders/wgsl) so the code compares
file by file.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional

GLSL_DIR = os.path.join(os.path.dirname(__file__), 'glsl')

HEADERS = {
	'gl330': '#version 330 core\n',
	'webgl2': (
		'#version 300 es\n'
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
