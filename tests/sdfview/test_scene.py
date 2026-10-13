"""Scene format: round trip, layout guarantees, and the JavaScript reader (Node) on the same bytes."""

import json
import os
import shutil
import struct
import subprocess

import numpy as np
import pytest

from sdfview.scene import ALIGN, Scene, grid_field

WEB = os.path.join(os.path.dirname(__file__), '..', '..', 'sdfview', 'web', 'scene.js')


def make_scene():
	rng = np.random.default_rng(0)
	s = Scene([-1, -2, -3], [1, 2, 3], meta={'note': 'test'})
	s.add_array('a/f', rng.normal(size=(7, 4)).astype(np.float32))
	s.add_array('a/i', rng.integers(-100, 100, size=(5, 4)).astype(np.int32))
	s.add_array('a/u8', rng.integers(0, 255, size=(3,)).astype(np.uint8))
	s.add_array('a/one', np.arange(9, dtype=np.int32)[:, None])
	s.add_field({'name': 'f', 'type': 'custom', 'params': {'x': 1}, 'arrays': {'v': 'a/f'}})
	grid_field(s, 'g', rng.normal(size=(3, 4, 5)), [-1, -1, -1], [1, 1, 1], isovalue=0.1)
	s.add_pointcloud('pc', 'a/f', None, radius=0.01)
	return s


def test_round_trip(tmp_path):
	s = make_scene()
	path = str(tmp_path / 'x.sdfx')
	s.save(path)
	t = Scene.load(path)
	assert t.bounds_min == s.bounds_min and t.bounds_max == s.bounds_max and t.meta == s.meta
	assert t.fields == s.fields and t.pointclouds == s.pointclouds
	assert list(t.arrays) == list(s.arrays)
	for name in s.arrays:
		assert t.arrays[name].dtype == s.arrays[name].dtype
		np.testing.assert_array_equal(t.arrays[name], s.arrays[name])
	assert t.field('g')['type'] == 'grid' and t.field('custom')['name'] == 'f' and t.field('nope') is None


def test_layout(tmp_path):
	data = make_scene().to_bytes()
	assert data[:4] == b'SDFX'
	version, json_len, bin_len = struct.unpack_from('<III', data, 4)
	assert version == 1 and json_len % ALIGN == 0 and len(data) == 16 + json_len + bin_len
	m = json.loads(data[16 : 16 + json_len])
	assert m['asset']['format'] == 'sdfx-scene'
	for spec in m['arrays'].values():
		assert spec['byteOffset'] % ALIGN == 0 and spec['byteOffset'] + spec['byteLength'] <= bin_len


def test_validation():
	s = Scene()
	with pytest.raises(TypeError):
		s.add_array('x', np.zeros(3, np.float64))
	with pytest.raises(KeyError):
		s.add_field({'name': 'f', 'type': 'pat', 'arrays': {'points': 'missing'}})
	with pytest.raises(ValueError):
		grid_field(s, 'g', np.zeros((1, 4, 4)), [0, 0, 0], [1, 1, 1])
	with pytest.raises(ValueError):
		Scene.from_bytes(b'NOPE' + bytes(12))


def test_forward_compatible_reader():
	"""Unknown manifest keys and field types are carried, not rejected."""
	s = make_scene()
	s.fields.append({'name': 'later', 'type': 'future-type', 'whatever': [1, 2]})
	t = Scene.from_bytes(s.to_bytes())
	assert t.field('later')['whatever'] == [1, 2]


@pytest.mark.skipif(shutil.which('node') is None, reason='needs Node.js')
def test_javascript_reader(tmp_path):
	s = make_scene()
	path = tmp_path / 'x.sdfx'
	s.save(str(path))
	script = tmp_path / 'read.mjs'
	script.write_text(
		f"import {{ readFileSync }} from 'node:fs';\n"
		f"import {{ parseScene }} from '{os.path.abspath(WEB)}';\n"
		f"const b = readFileSync('{path}');\n"
		'const s = parseScene(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));\n'
		'const out = {};\n'
		'for (const [k, a] of Object.entries(s.arrays)) out[k] = {shape: a.shape, values: Array.from(a.data)};\n'
		'console.log(JSON.stringify({fields: s.manifest.fields, arrays: out}));\n'
	)
	res = json.loads(subprocess.run(['node', str(script)], capture_output=True, text=True, check=True).stdout)
	assert res['fields'] == s.fields
	for name, a in s.arrays.items():
		assert res['arrays'][name]['shape'] == list(a.shape)
		np.testing.assert_array_equal(np.array(res['arrays'][name]['values'], dtype=a.dtype), a.ravel())
