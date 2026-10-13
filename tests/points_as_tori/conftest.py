"""
Fixtures for the points_as_tori port tests.

Parity references (tests skip when unavailable):
  * original JAX package:  ext/points-as-tori/src/pointsastori  (needs jax + flax)
  * original C++ evaluator: the `pat_reference` module built from tests/points_as_tori/reference
	(set PAT_REFERENCE_DIR to its build directory; default: build/pat_reference)
The original package imports `pat_bindings`; it is aliased to `pat_reference`, with placeholders for the
2D-shape / mesh bindings that the reference build leaves out.
"""

import os
import sys
import types

import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

sys.path.insert(0, REPO)

import points_as_tori as pat  # noqa: F401
from points_as_tori import config
from helpers import PAT_SRC, read_ply


def _load_reference():
	ref_dir = os.environ.get('PAT_REFERENCE_DIR', os.path.join(REPO, 'build', 'pat_reference'))
	if ref_dir not in sys.path:
		sys.path.insert(0, ref_dir)
	try:
		import pat_reference
	except ImportError:
		return None
	return pat_reference


def _load_original():
	ref = _load_reference()
	if ref is None:
		return None
	try:
		import jax  # noqa: F401
		import flax  # noqa: F401
	except ImportError:
		return None
	stub = types.ModuleType('pat_bindings')

	def __getattr__(name):
		if hasattr(ref, name):
			return getattr(ref, name)
		return type(name, (), {})  # placeholder for unported bindings (2D shapes, meshes, BVH)

	stub.__getattr__ = __getattr__
	sys.modules['pat_bindings'] = stub
	if PAT_SRC not in sys.path:
		sys.path.insert(0, PAT_SRC)
	import pointsastori

	return pointsastori


@pytest.fixture(scope='session')
def reference():
	ref = _load_reference()
	if ref is None:
		pytest.skip('pat_reference (C++ reference build) not available; see tests/points_as_tori/reference')
	return ref


def _load_original_training():
	"""training/train.py and training/mesh_sampling_jax.py of the original (mesh_data is stubbed: dataset generation
	needs thingi10k, pyvista, ... and is not used by the training loop)."""
	if _load_original() is None:
		return None
	try:
		import matplotlib  # noqa: F401  (imported at the top of the original train.py)
	except ImportError:
		return None
	sys.modules.setdefault('mesh_data', types.ModuleType('mesh_data'))
	training_dir = os.path.join(REPO, 'ext', 'points-as-tori', 'training')
	if training_dir not in sys.path:
		sys.path.insert(0, training_dir)
	import mesh_sampling_jax
	import train

	return types.SimpleNamespace(train=train, mesh_sampling=mesh_sampling_jax)


@pytest.fixture(scope='session')
def original_training():
	mod = _load_original_training()
	if mod is None:
		pytest.skip('original JAX training code not importable (needs jax, flax, optax, matplotlib, pat_reference)')
	return mod


@pytest.fixture
def jax_x64(original):
	import jax

	jax.config.update('jax_enable_x64', True)
	yield
	jax.config.update('jax_enable_x64', False)


@pytest.fixture(scope='session')
def original():
	mod = _load_original()
	if mod is None:
		pytest.skip('original JAX package not importable (needs jax, flax and the pat_reference build)')
	return mod


@pytest.fixture
def cpu64():
	"""Configured device for most tests: CPU, float64 geometry (the C++ evaluates in double)."""
	config.set_device('cpu', torch.float64, force=True)
	yield torch.device('cpu')


@pytest.fixture
def cpu32():
	config.set_device('cpu', torch.float32, force=True)
	yield torch.device('cpu')


@pytest.fixture(scope='session')
def tooth():
	return read_ply('tooth.ply')
