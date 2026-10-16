"""
The two shader assemblers produce the same programs: sdfview/shaders/__init__.py (Python hosts) and
sdfview/web/core/shaders.js (JavaScript hosts), for every GLSL stage x target and every WGSL program x field type.
The WGSL preprocessor is also checked on its own rules (conditionals, nesting, include once, #error).
Skipped without node.
"""

import json
import os
import shutil
import subprocess

import pytest

from sdfview import shaders
from sdfview.shaders import GLSL_DIR, WGSL_DIR, glsl_source, preprocess, wgsl_source

SHADERS = os.path.dirname(shaders.__file__)
JS = os.path.join(SHADERS, '..', 'web', 'core', 'shaders.js')

FIELD_DEFINES = [
	{'FIELD_PAT': None, 'BVH_STACK_SIZE': 64, 'BVH_K_MAX': 32},
	{'FIELD_GRID': None},
	{'FIELD_GLSL': None},
]
PRELUDE = 'float fieldGlsl(vec3 p) { return length(p) - 1.0; }'

NODE_SCRIPT = r"""
import fs from 'fs';
import path from 'path';
import { pathToFileURL } from 'url';
const { shadersDir, js, jobs } = JSON.parse(fs.readFileSync(0, 'utf8'));
const { glslSource, wgslSource } = await import(pathToFileURL(js).href);
const read = async (p) => fs.readFileSync(path.join(shadersDir, p), 'utf8');
const out = [];
for (const j of jobs) {
  try {
    out.push(j.lang === 'glsl' ? await glslSource(read, j.name, j.target, j.defines, j.prelude)
                               : await wgslSource(read, j.name, j.defines, j.prelude));
  } catch (e) {
    out.push({ error: String(e) });
  }
}
process.stdout.write(JSON.stringify(out));
"""


def _jobs():
	jobs = []
	stages = sorted(f for f in os.listdir(GLSL_DIR) if f.endswith(('.vert', '.frag')))
	for name in stages:
		for target in shaders.HEADERS:
			for defines in FIELD_DEFINES:
				prelude = PRELUDE if 'FIELD_GLSL' in defines else None
				jobs.append({'lang': 'glsl', 'name': name, 'target': target, 'defines': defines, 'prelude': prelude})
	for name in ('surface.wgsl', 'probe.wgsl'):
		for defines in FIELD_DEFINES[:2]:  # WGSL: pat, grid (a "glsl" field has no WGSL)
			jobs.append({'lang': 'wgsl', 'name': name, 'defines': defines, 'prelude': None})
	return jobs


def test_python_and_js_assemble_the_same():
	if shutil.which('node') is None:
		pytest.skip('node not found')
	jobs = _jobs()
	payload = json.dumps({'shadersDir': SHADERS, 'js': os.path.abspath(JS), 'jobs': jobs})
	res = subprocess.run(
		['node', '--input-type=module', '-e', NODE_SCRIPT], input=payload, capture_output=True, text=True, timeout=120
	)
	assert res.returncode == 0, res.stderr[-2000:]
	got = json.loads(res.stdout)
	for job, js in zip(jobs, got):
		if job['lang'] == 'glsl':
			py = glsl_source(job['name'], job['target'], job['defines'], job['prelude'])
		else:
			py = wgsl_source(job['name'], job['defines'], job['prelude'])
		assert js == py, f'{job["lang"]} {job["name"]} {job.get("target")} {job["defines"]}'
	print(f'\n{len(jobs)} programs identical')


def _tree(tmp_path, files):
	for name, text in files.items():
		(tmp_path / name).write_text(text)
	return str(tmp_path)


def test_preprocess_rules(tmp_path):
	root = _tree(
		tmp_path,
		{
			'a.wgsl': '#include "b.wgsl"\n#include "b.wgsl"\n'
			'#ifdef X\nx\n#elif defined(Y) || defined(Z)\nyz\n#else\nnone\n#endif\n'
			'#if defined(X) && !defined(Y)\n#ifndef W\nx_not_y\n#endif\n#endif\n',
			'b.wgsl': 'b\n',
			'e.wgsl': '#ifndef FIELD\n#error no field\n#endif\nok\n',
		},
	)
	lines = lambda d: [s for s in preprocess(root, 'a.wgsl', d).splitlines() if s and not s.startswith('//')]  # noqa: E731
	assert lines({}) == ['b', 'none']
	assert lines({'X': None}) == ['b', 'x', 'x_not_y']
	assert lines({'X': None, 'Y': None}) == ['b', 'x']
	assert lines({'Z': None}) == ['b', 'yz']
	assert lines({'X': None, 'W': 1}) == ['b', 'x']
	assert preprocess(root, 'e.wgsl', {'FIELD': None}).strip() == 'ok'
	with pytest.raises(Exception, match='no field'):
		preprocess(root, 'e.wgsl', {})


def test_wgsl_programs_resolve():
	"""Every WGSL program assembles without directives left over, for each field type it supports."""
	for name in ('surface.wgsl', 'probe.wgsl'):
		for defines in FIELD_DEFINES[:2]:
			src = wgsl_source(name, defines)
			left = [s for s in src.splitlines() if s.lstrip().startswith('#')]
			assert not left, f'{name} {defines}: {left[:3]}'
	with pytest.raises(Exception):
		wgsl_source('surface.wgsl', {})  # field.wgsl: #error without a field type
	assert os.path.isdir(WGSL_DIR)
