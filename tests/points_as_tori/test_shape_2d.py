"""
Parity of the 2D SDF port against the original, in float64:
  * points_as_tori.shape_2d.SDF2D and its 15 subclasses vs the C++ classes (pat_reference build of src/cpp/shape_2d.cpp):
    distances, analytic and autodiff gradients, Laplacians, bounding boxes, revolutions / extrusions and their 3D
    gradients, 2D and 3D ray intersections, and every sampler with the same seed (the std::mt19937 replay must be exact);
  * points_as_tori.shape_3d (SDF2D wrappers, load_SDF, sample_SDF, SDF3D) vs the original Python wrappers.
Metric: max |port - ref| / max |ref| (printed with -s).

The C++ draws the noise of sample_uneven_point_cloud inside an OpenMP loop (a data race with more than one thread)
and collects ray_intersect_batch hits from threads in nondeterministic order; those reference calls run with one
OpenMP thread (omp_set_num_threads through libgomp, shared with torch).
"""

import contextlib
import ctypes
import math

import numpy as np
import pytest
import torch

import points_as_tori as pat
from points_as_tori import config, cpp_random
from points_as_tori import shape_2d as S

TOL = 1e-12  # distances, gradients, positions (relative to max |value|); measured <= ~1e-14
TOL_NORMAL = 1e-10  # sampled normals: see test_samplers
NAMES = list(pat.SDF_SHAPES)
CLASSES = dict(
	circle='Circle2D',
	pie='Pie2D',
	arc='Arc2D',
	pill='Pill2D',
	vesica='Vesica2D',
	box='Box2D',
	cross='Cross2D',
	isosceles='TriangleIsosceles2D',
	triangle='Triangle2D',
	hexagon='Hexagon2D',
	ellipse='Ellipse2D',
	heart='Heart2D',
	pentagon='Pentagon2D',
	moon='Moon2D',
	trapezoid='Trapezoid2D',
	quad='Quad2D',
)


def rel_err(a, b):
	a = np.asarray(a, dtype=np.float64)
	b = np.asarray(b, dtype=np.float64)
	assert a.shape == b.shape, (a.shape, b.shape)
	assert np.array_equal(np.isfinite(a), np.isfinite(b)), 'non-finite patterns differ'
	m = np.isfinite(b)
	if not m.any():
		return 0.0
	return float(np.max(np.abs(a[m] - b[m])) / max(np.max(np.abs(b[m])), 1e-300))


def fortran(x):
	return np.asfortranarray(np.asarray(x, dtype=np.float64))


def wide(x):
	return torch.as_tensor(np.ascontiguousarray(x, dtype=np.float64), device=config.get_device())


@pytest.fixture(scope='module', autouse=True)
def single_threaded():
	"""
	Run this module with one intra-op thread: the tests run many small kernels, and an OpenMP fork/join costs
	milliseconds on small containers (torch parallelizes e.g. sqrt from ~2k elements). Results are unchanged.
	"""
	n = torch.get_num_threads()
	torch.set_num_threads(1)
	yield
	torch.set_num_threads(n)


@contextlib.contextmanager
def omp_single_thread():
	gomp = ctypes.CDLL('libgomp.so.1')
	n = gomp.omp_get_max_threads()
	gomp.omp_set_num_threads(1)
	try:
		yield
	finally:
		gomp.omp_set_num_threads(n)


def random_args(name, rng):
	"""Constructor arguments with the parameter ranges of sample_SDF (original)."""
	u = lambda lo=0.1, hi=0.9: float(rng.uniform(lo, hi))  # noqa: E731
	if name in ('circle', 'hexagon', 'pentagon'):
		return (u(),)
	if name == 'pie':
		return (u(), u(math.pi / 6, 5 * math.pi / 6))
	if name == 'arc':
		ra = u()
		rb = u() * ra
		angle = u(math.pi / 6, 5 * math.pi / 6)
		return (np.array([math.sin(angle), math.cos(angle)]), ra, rb)
	if name == 'pill':
		h = u()
		return (np.array([-h, 0.0]), np.array([h, 0.0]), u() * h)
	if name == 'vesica':
		r = u()
		return (r, u() * 2 * r - r)
	if name in ('box', 'cross', 'ellipse'):
		return (np.array([u(), u()]),)
	if name == 'isosceles':
		return (np.array([u(), -u()]),)
	if name == 'triangle':
		h = u()
		pv = rng.uniform(size=(3, 2))
		pv *= 0.9 * h / np.linalg.norm(pv, axis=1, keepdims=True)
		b = 2 * h / math.sqrt(3)
		return (np.array([-b, -h]) + pv[0], np.array([b, -h]) + pv[1], np.array([0.0, h]) + pv[2])
	if name == 'heart':
		return ()
	if name == 'moon':
		ra = u()
		rb = u() * ra
		return (u((ra - rb) / 0.9, (ra + rb) * 0.9), ra, rb)
	if name == 'trapezoid':
		return (u(), u(), 2 * u())
	if name == 'quad':
		w, h = u(), u()
		pv = rng.uniform(size=(4, 2))
		pv *= min(0.9 * h, 0.9 * w) / np.linalg.norm(pv, axis=1, keepdims=True)
		return tuple(np.array(c) + p for c, p in zip(([-w, -h], [w, -h], [w, h], [-w, h]), pv))
	raise ValueError(name)


# Extra parameters for branches that the sample_SDF ranges may miss
EXTRA_ARGS = dict(
	pie=[(0.6, 0.5), (0.6, 2.4)],  # cos t > 0 and < 0 (bounding-box branches)
	arc=[(np.array([math.sin(2.4), math.cos(2.4)]), 0.6, 0.2)],
	vesica=[(0.7, 0.3), (0.7, -0.3)],  # sign(d)
	isosceles=[(np.array([0.5, 0.6]),)],  # q_y > 0
	triangle=[(np.array([0.0, 0.6]), np.array([0.6, -0.5]), np.array([-0.5, -0.4]))],  # clockwise
	quad=[(np.array([-0.5, 0.5]), np.array([0.6, 0.4]), np.array([0.5, -0.6]), np.array([-0.4, -0.5]))],  # clockwise
)


def all_args(name):
	return [random_args(name, np.random.default_rng(seed)) for seed in (0, 1)] + EXTRA_ARGS.get(name, [])


def make(module, name, args):
	return getattr(module, CLASSES[name])(*args)


# ========================================================================
# 2D EVALUATION
# ========================================================================


@pytest.mark.parametrize('name', NAMES)
def test_evaluate_gradient_laplacian(cpu64, reference, name):
	rng = np.random.default_rng(10)
	Q = rng.uniform(-1.5, 1.5, size=(2, 3000))
	q = wide(Q)
	errs = {}

	def record(key, value):
		errs[key] = max(errs.get(key, 0.0), value)

	for args in all_args(name):
		ref, port = make(reference, name, args), make(S, name, args)
		for iso in (0.0, 0.05):
			record('distance', rel_err(port.evaluate_batch(q, iso).numpy(), ref.evaluate_batch(fortran(Q), iso)))
			d, g = ref.evaluate_with_gradient_batch(fortran(Q), iso)
			pd, pg = port.evaluate_with_gradient_batch(q, iso)
			record('distance', rel_err(pd.numpy(), d))
			record('gradient', rel_err(pg.numpy(), g))
			for k, v in zip(('min', 'max'), ref.bounding_box(iso)):
				record('bounding_box', rel_err(port.bounding_box(iso)[k == 'max'].numpy(), v))
		record('gradient', rel_err(port.evaluate_gradient_batch(q).numpy(), ref.evaluate_gradient_batch(fortran(Q))))
		record('laplacian', rel_err(port.evaluate_laplacian_batch(q).numpy(), ref.evaluate_laplacian_batch(fortran(Q))))
		# single points (C++ API: one point per call)
		for i in range(10):
			record('single', abs(float(port.evaluate(q[:, i], 0.03)) - ref.evaluate(Q[:, i], 0.03)))
			d, g = ref.evaluate_with_gradient(Q[:, i], 0.03)
			pd, pg = port.evaluate_with_gradient(q[:, i], 0.03)
			record('single', max(abs(float(pd) - d), np.max(np.abs(pg.numpy() - g))))
			record('single', np.max(np.abs(port.evaluate_gradient(q[:, i]).numpy() - ref.evaluate_gradient(Q[:, i]))))
		lap1 = port.evaluate_laplacian(q[:, 0])
		assert lap1.shape == () and abs(float(lap1) - ref.evaluate_laplacian_batch(fortran(Q[:, :1]))[0]) < 1e-10
		# Eigen AutoDiff gradient of the template vs torch.autograd of the same expression
		ga = np.stack([ref.evaluate_gradient_autodiff(Q[:, i], 0.02) for i in range(300)], axis=1)
		record('autodiff_gradient', rel_err(port.evaluate_gradient_autodiff(q[:, :300], 0.02).numpy(), ga))
	print(f'\n{name}: ' + ', '.join(f'{k} {v:.1e}' for k, v in errs.items()))
	for k, v in errs.items():
		assert v < (1e-11 if k == 'laplacian' else TOL), k


def test_laplacian_seeding_of_the_original(cpu64, reference):
	"""
	The C++ Laplacian is Δφ + 2 (∂φ/∂x + ∂φ/∂y) (second-order autodiff seeded with unit inner derivatives); the port
	reproduces it by default and gives Δφ with exact=True. Checked on Box2D, whose Laplacian is known in closed form:
	0 inside and in the edge regions, 1/φ in the corner regions (φ = distance to the corner).
	"""
	b = np.array([0.5, 0.3])
	ref, port = reference.Box2D(b), S.Box2D(b)
	Q = np.random.default_rng(3).uniform(-1.5, 1.5, size=(2, 2000))
	q = wide(Q)
	lap_ref = ref.evaluate_laplacian_batch(fortran(Q))
	g = port.evaluate_gradient_autodiff(q).numpy()
	exact = port.evaluate_laplacian_batch(q, exact=True).numpy()
	assert rel_err(exact + 2 * (g[0] + g[1]), lap_ref) < 1e-14
	w = np.abs(Q) - b[:, None]
	corner = (w[0] > 0) & (w[1] > 0)
	expected = np.where(corner, 1.0 / np.hypot(w[0], w[1]), 0.0)
	assert np.max(np.abs(exact - expected)) < 1e-12
	inside = (w[0] < 0) & (w[1] < 0)
	assert np.allclose(lap_ref[inside], 2 * np.where(w[0] > w[1], np.sign(Q[0]), np.sign(Q[1]))[inside])
	# Circle2D: the C++ overrides the Laplacian with 1/|q|, the autodiff template gives the same with exact=True
	c = S.Circle2D(0.4)
	lap_t = S.SDF2D.evaluate_laplacian_batch(c, q, exact=True).numpy()
	assert rel_err(lap_t, reference.Circle2D(0.4).evaluate_laplacian_batch(fortran(Q))) < 1e-14
	assert rel_err(pat.Circle2D(0.4).evaluate_signed_distance_laplacian(q, exact=True).numpy(), lap_t) < 1e-14


# ========================================================================
# 3D: REVOLUTIONS AND EXTRUSIONS
# ========================================================================


@pytest.mark.parametrize('name', NAMES)
def test_revolution_extrusion(cpu64, reference, name):
	Q = np.random.default_rng(11).uniform(-1.5, 1.5, size=(3, 2000))
	q = wide(Q)
	errs = {}
	for args in all_args(name)[:2]:
		ref, port = make(reference, name, args), make(S, name, args)
		for o in (0.0, 0.3):
			e = rel_err(port.evaluate_revolution(q, o).numpy(), ref.evaluate_revolution(fortran(Q), o))
			errs['revolution'] = max(errs.get('revolution', 0), e)
			g_ref = ref.evaluate_signed_distance_gradient(fortran(Q), False, o)
			e = rel_err(port.evaluate_signed_distance_gradient(q, False, o).numpy(), g_ref)
			errs['revolution_gradient'] = max(errs.get('revolution_gradient', 0), e)
		for h in (0.2, 0.5):
			e = rel_err(port.evaluate_extrusion(q, h).numpy(), ref.evaluate_extrusion(fortran(Q), h))
			errs['extrusion'] = max(errs.get('extrusion', 0), e)
			g_ref = ref.evaluate_signed_distance_gradient(fortran(Q), True, h)
			e = rel_err(port.evaluate_signed_distance_gradient(q, True, h).numpy(), g_ref)
			errs['extrusion_gradient'] = max(errs.get('extrusion_gradient', 0), e)
		# single-point forms
		assert port.evaluate_revolution_single(q[:, 0], 0.3).shape == ()
		assert port.evaluate_extrusion_gradient_single(q[:, 0], 0.3).shape == (3,)
		assert float(port.evaluate_extrusion_single(q[:, 5], 0.2)) == float(port.evaluate_extrusion(q, 0.2)[5])
		assert torch.equal(
			port.evaluate_revolution_gradient_single(q[:, 5], 0.3), port.evaluate_revolution_gradient(q, 0.3)[:, 5]
		)
		# 3D bounding boxes from the 2D one (not bound in the C++)
		bmin, bmax = ref.bounding_box(0.0)
		rad = max(abs(bmin[0]), abs(bmax[0])) + 0.3
		rmin, rmax = port.bbox_revolution(0.3)
		assert np.array_equal(rmin.numpy(), [-rad, bmin[1], -rad]) and np.array_equal(rmax.numpy(), [rad, bmax[1], rad])
		emin, emax = port.bbox_extrusion(0.4)
		assert np.array_equal(emin.numpy(), [bmin[0], bmin[1], -0.4]) and np.array_equal(emax.numpy(), [*bmax, 0.4])
	print(f'\n{name}: ' + ', '.join(f'{k} {v:.1e}' for k, v in errs.items()))
	for k, v in errs.items():
		assert v < TOL, k


# ========================================================================
# RAY INTERSECTIONS
# ========================================================================


@pytest.mark.parametrize('name', NAMES)
def test_ray_intersect_2d(cpu64, reference, name):
	rng = np.random.default_rng(12)
	n = 240
	ro = rng.uniform(-1.5, 1.5, size=(2, n))
	rd = rng.uniform(-0.5, 0.5, size=(2, n)) - ro
	rd[:, n // 2 :] = rng.normal(size=(2, n - n // 2))  # random directions: some misses
	rd /= np.linalg.norm(rd, axis=0)
	args = all_args(name)[0]
	ref, port = make(reference, name, args), make(S, name, args)

	pq, pn, ph = port.ray_intersect(wide(ro), wide(rd))
	rq, rn, rh = (np.array(x) for x in zip(*(ref.ray_intersect(ro[:, i], rd[:, i]) for i in range(n))))
	assert np.array_equal(ph.numpy(), rh), 'hit flags differ'
	errs = dict(position=rel_err(pq.numpy(), rq.T), normal=rel_err(pn.numpy(), rn.T))
	q1, n1, h1 = port.ray_intersect(wide(ro[:, 0]), wide(rd[:, 0]))
	assert q1.shape == (2,) and h1.shape == () and bool(h1) == rh[0]

	with omp_single_thread():
		bq, bn = ref.ray_intersect_batch(fortran(ro), fortran(rd))
	pbq, pbn = port.ray_intersect_batch(wide(ro), wide(rd))
	errs.update(batch_position=rel_err(pbq.numpy(), bq), batch_normal=rel_err(pbn.numpy(), bn))
	py_q, py_n = pat.SDF2D(port).ray_intersections(wide(ro), wide(rd))
	assert torch.equal(py_q, pbq) and torch.equal(py_n, pbn)
	print(f'\n{name}: {int(rh.sum())}/{n} hits, ' + ', '.join(f'{k} {v:.1e}' for k, v in errs.items()))
	for k, v in errs.items():
		assert v < TOL, k


def ray_intersections_3d_reference(ref, ro, rd, is_extrusion, param, tmax):
	"""Transcription of SDF2D::ray_intersection_revolution / _extrusion with the C++ 3D SDF (one point per call)."""
	f = ref.evaluate_extrusion if is_extrusion else ref.evaluate_revolution

	def dist(t):
		q = ro + t * rd
		return q, abs(f(fortran(q[:, None]), param)[0])

	t, eps, hits = 0.0, 1e-4, []
	while t < tmax:
		q, d = dist(t)
		if d < eps:
			hits.append(q)
			while d < eps and t < tmax:
				t += max(d, eps)
				q, d = dist(t)
		t += d
	return np.array(hits).reshape(-1, 3).T


@pytest.mark.parametrize('is_extrusion', [False, True])
def test_ray_intersections_3d(cpu64, reference, is_extrusion):
	"""All intersections along 3D rays (revolution / extrusion) vs the C++ loop driven by the C++ 3D SDF."""
	rng = np.random.default_rng(13)
	param = 0.35 if is_extrusion else 0.25
	n_hits = 0
	for name in ('circle', 'pie', 'cross', 'moon', 'quad'):
		args = all_args(name)[0]
		ref, port = make(reference, name, args), make(S, name, args)
		bmin, bmax = (port._bbox_extrusion if is_extrusion else port._bbox_revolution)(param)
		tmax = math.sqrt(sum((a - b) ** 2 for a, b in zip(bmax, bmin)))
		for _ in range(10):
			d = rng.normal(size=3)
			d /= np.linalg.norm(d)
			o = rng.uniform(-0.3, 0.3, size=3) - 0.6 * tmax * d
			with omp_single_thread():  # one-point calls: avoid the OpenMP fork per call
				expected = ray_intersections_3d_reference(ref, o, d, is_extrusion, param, tmax)
			f = port.ray_intersection_extrusion if is_extrusion else port.ray_intersection_revolution
			got = f(wide(o), wide(d), param).numpy()
			assert got.shape == expected.shape, (name, got.shape, expected.shape)
			if expected.size:
				assert np.max(np.abs(got - expected)) < 1e-13
			n_hits += expected.shape[1]
	print(f'\n{n_hits} intersections')
	assert n_hits > 40


# ========================================================================
# SAMPLERS (same seed -> same points as the C++)
# ========================================================================


@pytest.mark.parametrize('name', NAMES)
def test_samplers(cpu64, reference, name):
	"""
	Positions agree to rounding. Normals are compared with TOL_NORMAL: at some hits the C++ gradient is a vector of
	length |φ| < ε = 1e-4 normalized (e.g. Trapezoid2D, Heart2D, vertices), which amplifies position differences of
	1e-16 by up to 1e4 (measured <= 3e-12).
	"""
	args = all_args(name)[0]
	ref, port = make(reference, name, args), make(S, name, args)
	errs, errs_n = {}, {}
	B1, B0 = np.array([1.0, 1, 1]), np.array([-1.0, -1, -1])

	def compare(key, got, expected, normals=False):
		got = [g.numpy() for g in got] if isinstance(got, tuple) else [got.numpy()]
		expected = list(expected) if isinstance(expected, tuple) else [expected]
		errs[key] = rel_err(got[0], expected[0])
		if normals:
			errs_n[key] = rel_err(got[1], expected[1])

	compare(
		'revolution',
		port.sample_revolution(300, 0.3, B1, B0, 0.01, 0.02, 7),
		ref.sample_revolution(300, 0.3, B1, B0, 0.01, 0.02, 7),
		True,
	)
	bmin, bmax = (np.array(b) for b in port._bbox_extrusion(0.4))
	bmin[0] = 0.5 * (bmin[0] + bmax[0])  # a custom box: half of the default one
	compare(
		'extrusion',
		port.sample_extrusion(300, 0.4, bmin, bmax, 0.0, 0.05, 11),
		ref.sample_extrusion(300, 0.4, bmin, bmax, 0.0, 0.05, 11),
		True,
	)
	compare(
		'revolution_narrow_band',
		port.sample_revolution_narrow_band(200, 0.2, 0.05, B1, B0, 3),
		ref.sample_revolution_narrow_band(200, 0.2, 0.05, B1, B0, 3),
	)
	compare(
		'extrusion_narrow_band',
		port.sample_extrusion_narrow_band(200, 0.3, 0.05, B1, B0, 4),
		ref.sample_extrusion_narrow_band(200, 0.3, 0.05, B1, B0, 4),
	)
	for is_ext in (False, True):
		compare(
			f'uniform_{is_ext}',
			port.sample_points_and_normals_uniformly(200, is_ext, 0.3, 5),
			ref.sample_points_and_normals_uniformly(200, is_ext, 0.3, 5),
			True,
		)
	compare(
		'farthest',
		port.sample_farthest_point_cloud(40, True, 0.25, 9),
		ref.sample_farthest_point_cloud(40, True, 0.25, 9),
		True,
	)
	for is_ext in (False, True):
		uneven = (is_ext, 0.3, 4, 0.8, 0.05, 0.08, 0.01, 0.2, 200, 6)
		with omp_single_thread():
			expected = ref.sample_uneven_point_cloud(*uneven)
		compare(f'uneven_{is_ext}', port.sample_uneven_point_cloud(*uneven), expected, True)
	print(f'\n{name}: positions ' + ', '.join(f'{k} {v:.1e}' for k, v in errs.items()))
	print(f'{name}: normals ' + ', '.join(f'{k} {v:.1e}' for k, v in errs_n.items()))
	for k, v in errs.items():
		assert v < TOL, k
	for k, v in errs_n.items():
		assert v < TOL_NORMAL, k


def test_sampler_corner_cases(cpu64, reference):
	c = S.Circle2D(0.5)
	# no intersection: the C++ divides by zero (SIGFPE); the port raises
	with pytest.raises(RuntimeError, match='no intersection'):
		c.sample_extrusion(10, 0.3, (-0.3, -9, -9), (9, 9, 9), 0.0, 0.0, 0)
	# n_samples = 0
	p, n = c.sample_revolution(0, 0.3)
	assert p.shape == (3, 0) and n.shape == (3, 0)
	# fewer intersections than requested within max_tries: samples are duplicated, as in the C++
	B = (np.array([-0.05, -0.05, -0.05]), np.array([0.05, 0.6, 0.05]))
	got = c.sample_revolution(50, 0.3, *B, 0.0, 0.0, 2)
	expected = reference.Circle2D(0.5).sample_revolution(50, 0.3, *B, 0.0, 0.0, 2)
	assert rel_err(got[0].numpy(), expected[0]) < TOL and rel_err(got[1].numpy(), expected[1]) < TOL
	# more normals to flip than samples: the C++ reads past the end of its index vector; the port flips all
	pts, nrm = c.sample_uneven_point_cloud(True, 0.3, 2, 0.3, 0.2, 0.3, 0.0, 1.0, 1000, 1)
	assert pts.shape[1] == nrm.shape[1] < 1000


def test_normal_draws_replay():
	"""The vectorized replay of interleaved normal_distribution calls equals sequential calls (same stream use)."""
	labels = np.random.default_rng(0).integers(0, 3, size=301)
	rng_a, rng_b = cpp_random.MT19937(5), cpp_random.MT19937(5)
	dists_a = [cpp_random.Normal(0.0, s) for s in (1.0, 0.1, 2.0)]
	dists_b = [cpp_random.Normal(0.0, s) for s in (1.0, 0.1, 2.0)]
	dists_a[1](rng_a)  # start with a saved value in object 1
	dists_b[1](rng_b)
	got = S._normal_draws(rng_a, dists_a, labels)
	expected = np.array([dists_b[k](rng_b) for k in labels])
	assert np.max(np.abs(got - expected)) <= 1e-15 * np.max(np.abs(expected))  # np.log vs glibc log (1 ulp)
	assert np.array_equal(rng_a.peek(8), rng_b.peek(8))
	assert all((a._saved is None) == (b._saved is None) for a, b in zip(dists_a, dists_b))


# ========================================================================
# PYTHON API (shape_3d)
# ========================================================================

SHAPE_FILES = dict(
	circle=('r 0.5', ('radius', 0.5)),
	pie=('r 0.6\nt 1.2', ('t', 1.2)),
	arc=('sc 0.8 0.6\nra 0.5\nrb 0.1', ('ra', 0.5)),
	pill=('a -0.4 0\nb 0.4 0\nr 0.2', ('r', 0.2)),
	vesica=('r 0.6\nd 0.2', ('d', 0.2)),
	box=('b 0.5 0.3', ('b', [0.5, 0.3])),
	cross=('b 0.6 0.2', ('b', [0.6, 0.2])),
	isosceles=('q 0.4 -0.6', ('q', [0.4, -0.6])),
	triangle=('v -0.5 -0.4\nv 0.5 -0.4\n\nv 0 0.6', ('v', [[-0.5, -0.4], [0.5, -0.4], [0, 0.6]])),
	hexagon=('r 0.5', ('r', 0.5)),
	ellipse=('ab 0.6 0.3', ('ab', [0.6, 0.3])),
	heart=('', None),
	pentagon=('r 0.4', ('r', 0.4)),
	moon=('d 0.3\nra 0.6\nrb 0.4', ('rb', 0.4)),
	trapezoid=('ra 0.5\nrb 0.3\nhe 0.4', ('he', 0.4)),
	quad=(
		'v -0.5 -0.5\nv 0.5 -0.4\nv 0.6 0.5\nv -0.4 0.6',
		('v', [[-0.5, -0.5], [0.5, -0.4], [0.6, 0.5], [-0.4, 0.6]]),
	),
)


def test_load_SDF(cpu64, reference, tmp_path):
	Q = np.random.default_rng(14).uniform(-1, 1, size=(2, 500))
	for i, (name, (text, attribute)) in enumerate(SHAPE_FILES.items()):
		path = tmp_path / f'shape-{i}_{name}.txt'
		path.write_text(text + '\n')
		shape = pat.load_SDF(str(path))
		assert type(shape) is getattr(pat.shape_3d, CLASSES[name]), name
		assert isinstance(shape, pat.SDF2D) and pat.SDF2D.load_SDF is pat.load_SDF
		if attribute is not None:
			assert np.allclose(np.asarray(getattr(shape, attribute[0]), dtype=float), attribute[1]), name
		# same SDF as the C++ object with the parameters of the file
		ref = make(reference, name, ref_args(name, text))
		assert rel_err(shape.evaluate_signed_distance(wide(Q)).numpy(), ref.evaluate_batch(fortran(Q), 0.0)) < TOL
	empty = tmp_path / 'empty_box.txt'
	empty.write_text('r 0.3\n')
	assert pat.load_SDF(str(empty)) is None  # no `b` line: None, as the original
	unknown = tmp_path / 'blob_0.txt'
	unknown.write_text('r 0.3\n')
	with pytest.raises(ValueError, match='Unknown SDF type'):
		pat.load_SDF(str(unknown))


def ref_args(name, text):
	values = {}
	vs = []
	for line in text.splitlines():
		parts = line.split()
		if parts and parts[0] == 'v':
			vs.append(np.array(list(map(float, parts[1:3]))))
		elif parts:
			values[parts[0]] = np.array(list(map(float, parts[1:])))
	scalar = {k: float(v[0]) for k, v in values.items() if v.size == 1}
	return dict(
		circle=lambda: (scalar['r'],),
		pie=lambda: (scalar['r'], scalar['t']),
		arc=lambda: (values['sc'], scalar['ra'], scalar['rb']),
		pill=lambda: (values['a'], values['b'], scalar['r']),
		vesica=lambda: (scalar['r'], scalar['d']),
		box=lambda: (values['b'],),
		cross=lambda: (values['b'],),
		isosceles=lambda: (values['q'],),
		triangle=lambda: tuple(vs),
		hexagon=lambda: (scalar['r'],),
		ellipse=lambda: (values['ab'],),
		heart=lambda: (),
		pentagon=lambda: (scalar['r'],),
		moon=lambda: (scalar['d'], scalar['ra'], scalar['rb']),
		trapezoid=lambda: (scalar['ra'], scalar['rb'], scalar['he']),
		quad=lambda: tuple(vs),
	)[name]()


def test_sample_SDF(cpu64, reference):
	Q = wide(np.random.default_rng(15).uniform(-1, 1, size=(2, 300)))
	for name in NAMES:
		shapes = [pat.sample_SDF(name, key) for key in (None, 42, 7, torch.Generator().manual_seed(1))]
		assert all(type(s) is getattr(pat.shape_3d, CLASSES[name]) for s in shapes), name
		d = [s.evaluate_signed_distance(Q) for s in shapes]
		assert torch.equal(d[0], d[1]), name  # default key = 42
		if name != 'heart':
			assert not torch.equal(d[1], d[2]), name
		for s in shapes:  # the shape is valid: some queries inside, the bounding box contains the zero level set
			bmin, bmax = s.bound_object.bounding_box()
			assert (bmax > bmin).all()
		s = shapes[2]
		params = {k: v for k, v in vars(s).items() if k not in ('bound_object', '_has_gradient')}
		if name in ('circle', 'hexagon', 'pentagon'):
			assert 0.1 <= params['r' if name != 'circle' else 'radius'] <= 0.9
		if name == 'moon':
			assert (params['ra'] - params['rb']) / 0.9 <= params['d'] <= (params['ra'] + params['rb']) * 0.9
		if name == 'isosceles':
			assert params['q'][1] < 0
	assert pat.sample_SDF('blob') is None


def test_python_api_matches_original(cpu64, reference, original):
	o = original.shape_3d
	rng = np.random.default_rng(16)
	Q2 = rng.uniform(-1.2, 1.2, size=(200, 2))  # original layout (n, 2)
	Q3 = rng.uniform(-1.2, 1.2, size=(200, 3))
	B1, B0 = np.array([1, 1, 1]), np.array([-1, -1, -1])
	errs = {}
	for name in ('pie', 'vesica', 'triangle', 'moon'):
		args = all_args(name)[0]
		oc, pc = make(o, name, args), make(pat.shape_3d, name, args)
		q2, q3 = wide(Q2.T), wide(Q3.T)
		res = dict(
			distance=(pc.evaluate_signed_distance(q2, 0.1), oc.evaluate_signed_distance(Q2, 0.1)),
			gradient=(pc.evaluate_signed_distance_gradient(q2).T, oc.evaluate_signed_distance_gradient(Q2)),
			laplacian=(pc.evaluate_signed_distance_laplacian(q2), oc.evaluate_signed_distance_laplacian(Q2)),
			revolution=(pc.evaluate_revolution(q3, 0.2), oc.evaluate_revolution(Q3, 0.2)),
			extrusion=(pc.evaluate_extrusion(q3, 0.3), oc.evaluate_extrusion(Q3, 0.3)),
			gradient_3D=(
				pc.evaluate_signed_distance_gradient_3D(q3, True, 0.3).T,
				oc.evaluate_signed_distance_gradient_3D(fortran(Q3.T), True, 0.3),
			),
			sample_revolution=(pc.sample_revolution(100, 0.2, seed=3)[1].T, oc.sample_revolution(100, 0.2, seed=3)[1]),
			sample_extrusion=(
				pc.sample_extrusion(100, 0.3, B1, B0, 0.01, 0.0, 4)[0].T,
				oc.sample_extrusion(100, 0.3, B1, B0, 0.01, 0.0, 4)[0],
			),
			narrow_band=(
				pc.sample_revolution_narrow_band(100, 0.2, 0.1, seed=5).T,
				oc.sample_revolution_narrow_band(100, 0.2, 0.1, seed=5),
			),
			farthest=(
				pc.sample_farthest_point_cloud(30, False, 0.2, 6)[0].T,
				oc.sample_farthest_point_cloud(30, False, 0.2, 6)[0],
			),
		)
		d, g = pc.evaluate_signed_distance_and_gradient(q2)
		od, og = oc.evaluate_signed_distance_and_gradient(Q2)
		res.update(and_gradient=(torch.cat([d[None], g]).T, np.concatenate([od[:, None], og], axis=1)))
		ro, rd = Q2 * 1.5, -Q2
		with omp_single_thread():
			res['ray_intersections'] = (
				pc.ray_intersections(wide(ro.T), wide(rd.T))[1].T,
				oc.ray_intersections(ro, rd)[1],
			)

		for is_ext, param in ((False, 0.2), (True, 0.3)):
			o3, p3 = o.SDF3D(oc, is_ext, param), pat.SDF3D(pc, is_ext, param)
			d, g = p3.evaluate_signed_distance_and_gradient(q3)
			od, og = o3.evaluate_signed_distance_and_gradient(Q3)
			res[f'SDF3D_{is_ext}'] = (torch.cat([d[None], g]).T, np.concatenate([od[:, None], og], axis=1))
			res[f'SDF3D_uniformly_{is_ext}'] = (p3.sample_uniformly(80, seed=1).T, o3.sample_uniformly(80, seed=1))
			res[f'SDF3D_narrow_band_{is_ext}'] = (
				p3.sample_narrow_band(80, 0.05, seed=2).T,
				o3.sample_narrow_band(80, 0.05, seed=2),
			)
			res[f'SDF3D_farthest_{is_ext}'] = (
				p3.sample_farthest_point_cloud(20, 3)[1].T,
				o3.sample_farthest_point_cloud(20, 3)[1],
			)
			# missing in the original wrapper (AttributeError there): compared with the C++ directly
			got = p3.sample_points_and_normals_uniformly(80, 4)
			expected = oc.bound_object.sample_points_and_normals_uniformly(80, is_ext, param, 4)
			res[f'SDF3D_points_and_normals_{is_ext}'] = (got[1], expected[1])
			uneven = (is_ext, param, 3, 0.8, 0.06, 0.09, 0.01, 0.1, 100, 8)
			with omp_single_thread():
				expected = oc.bound_object.sample_uneven_point_cloud(*uneven)
			res[f'SDF3D_uneven_{is_ext}'] = (p3.sample_uneven_point_cloud(*uneven)[0], expected[0])
		for k, (got, expected) in res.items():
			errs[k] = max(errs.get(k, 0.0), rel_err(got.numpy(), expected))
	print('\n' + ', '.join(f'{k} {v:.1e}' for k, v in errs.items()))
	for k, v in errs.items():
		assert v < TOL, k


def test_original_python_wrapper_failures(original):
	"""Calls that fail in the original Python wrappers and work in the port (fixes listed in PORT_NOTES)."""
	o = original.shape_3d
	box = o.Box2D(np.array([0.5, 0.3]))
	with pytest.raises(AttributeError):  # single point: result[0].T on a float
		box.evaluate_signed_distance_and_gradient(np.array([0.1, 0.2]))
	with pytest.raises(NameError):  # `re` is not imported
		o.SDF2D.load_SDF('x_circle.txt')
	with pytest.raises(AttributeError):  # SDF2D has no sample_points_and_normals_uniformly
		o.SDF3D(box, True, 0.3).sample_points_and_normals_uniformly(10, 0)
	with pytest.raises(AttributeError):  # nor sample_uneven_point_cloud
		o.SDF3D(box, True, 0.3).sample_uneven_point_cloud(True, 0.3, 2, 1.0, 0.1, 0.2, 0.0, 0.0, 10, 0)
	with pytest.raises(TypeError):  # default instance is the class utilsb.Circle2D
		o.SDF2D()


# ========================================================================
# DEVICE / DTYPE
# ========================================================================


@pytest.fixture
def meta():
	config.set_device('meta', torch.float32, force=True)
	yield
	config.set_device('cpu', torch.float32, force=True)


def test_evaluation_on_meta_device(meta):
	"""Evaluation paths need no host sync (they run on the meta device, which has no data)."""
	q = torch.empty(2, 64, device='meta')
	q3 = torch.empty(3, 64, device='meta')
	for name in NAMES:
		s = make(S, name, all_args(name)[0])
		d, g = s.evaluate_with_gradient_batch(q)
		assert d.device.type == 'meta' and d.shape == (64,) and g.shape == (2, 64)
		assert s.evaluate_laplacian_batch(q).shape == (64,) and s.evaluate_laplacian(q, exact=True).shape == (64,)
		assert s.evaluate_gradient_autodiff(q).shape == (2, 64) and s.evaluate(q[:, 0]).shape == ()
		assert s.evaluate_revolution(q3, 0.3).shape == (64,) and s.evaluate_extrusion(q3, 0.3).shape == (64,)
		for is_ext in (False, True):
			assert s.evaluate_signed_distance_gradient(q3, is_ext, 0.3).shape == (3, 64)
		assert s.bounding_box()[0].device.type == 'meta'
	position, normal, hit = S.Circle2D(0.5).ray_intersect(q[:, :8], q[:, 8:16])  # fixed max_steps loop
	assert position.shape == (2, 8) and normal.shape == (2, 8) and hit.shape == (8,)


def test_float32(cpu32, reference):
	"""Default dtype: float32 tensors in and out, close to the float64 C++."""
	Q = np.random.default_rng(17).uniform(-1.2, 1.2, size=(2, 1000))
	q = torch.as_tensor(Q, dtype=torch.float32)
	for name in NAMES:
		args = all_args(name)[0]
		ref, port = make(reference, name, args), make(S, name, args)
		d, g = port.evaluate_with_gradient_batch(q)
		assert d.dtype == torch.float32 and g.dtype == torch.float32
		assert np.max(np.abs(d.numpy() - ref.evaluate_batch(fortran(Q), 0.0))) < 1e-5, name
	# samplers march in float32 (discrete decisions can differ from float64, so only check the samples' validity)
	box = S.Box2D([0.5, 0.3])
	points, normals = box.sample_extrusion(200, 0.3, seed=1)
	assert points.dtype == torch.float32 and points.shape == (3, 200)
	assert box.evaluate_extrusion(points, 0.3).abs().max() < 2e-4
	assert torch.allclose(normals.norm(dim=0), torch.ones(200))
