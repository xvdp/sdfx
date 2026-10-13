"""
Fixtures for the sdfview tests. GL tests need an OpenGL 3.3 context (EGL on Linux: a GPU driver or Mesa llvmpipe)
and skip without one; the window test also needs a display (DISPLAY, or xvfb-run on the PATH).
"""

import os
import sys

import numpy as np
import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

import points_as_tori as pat  # noqa: E402
from points_as_tori import config  # noqa: E402


def sphere_field(n=1500, seed=0, noise=0.05):
	"""TorusDistanceField on n points of the unit sphere: normals = radial, near-sphere tori from small random forms
	around the sphere's second fundamental form (a20 = a02 = -½, h = d·n convention). The field ≈ |x| - 1."""
	rng = np.random.default_rng(seed)
	points = rng.normal(size=(n, 3))
	points /= np.linalg.norm(points, axis=1, keepdims=True)
	normals = points.copy()
	coeffs = rng.normal(scale=noise, size=(n, 6))
	coeffs[:, :2] = 0.0
	coeffs[:, 4:] += -0.5
	tdf = pat.TorusDistanceField(config.as_wide(points, 3), config.as_wide(normals, 3))
	tdf.fit_tori_from_forms(config.as_wide(coeffs, 6))
	return tdf, points


@pytest.fixture
def sphere():
	"""Factory: sphere_field(...) on the CPU in the requested dtype."""

	def make(dtype=torch.float32, **kw):
		config.set_device('cpu', dtype, force=True)
		return sphere_field(**kw)

	yield make
	config.set_device('cpu', torch.float32, force=True)


@pytest.fixture
def cpu32():
	config.set_device('cpu', torch.float32, force=True)
	yield
	config.set_device('cpu', torch.float32, force=True)


@pytest.fixture
def cpu64():
	config.set_device('cpu', torch.float64, force=True)
	yield
	config.set_device('cpu', torch.float32, force=True)


@pytest.fixture(scope='session')
def gl_ctx():
	try:
		from sdfview.backends.moderngl import create_context

		ctx = create_context()
	except Exception as e:  # noqa: BLE001
		pytest.skip(f'no OpenGL 3.3 context: {e}')
	print(f'\nOpenGL: {ctx.info["GL_RENDERER"]} ({ctx.info["GL_VERSION"]})')
	yield ctx
	ctx.release()
