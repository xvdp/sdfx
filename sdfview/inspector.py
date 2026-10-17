"""
Tori inspector: what the Points-as-Tori field does at one point x of space (a clicked surface point, a point of the
cut plane), from the arrays of a scene (`pat` field: points, tori), on the host (NumPy).

1. The blend at x (field_pat.glsl, patDistanceGradient): the k nearest points p_i of x (brute force over all points;
   outliers take part in the search and the shift, not in the sums), shift = ½ max_i |x - p_i|,
   λ = λ_scale · max_exp_arg / shift, w_i = exp(-λ (|x - p_i| - shift)), ω_i = w_i / Σ w_j (the weights drawn),
   g_i(x) the signed distance of torus i, φ(x) = Σ ω_i g_i(x).

2. The torus of one neighbor j and its principal geometry. The fit (points_as_tori.torus_distance.fit_torus_at_point):
   the network predicts a quadratic height field Q(s, t) over p_j's tangent frame from p_j's k_network nearest points;
   at its foot point p* = p_j + a00 n_j (normal n*) the principal curvatures |κ_min| <= |κ_max| (direction v_min) give
       r = 1/|κ_max|,  R = 1/|κ_min| - sgn(κ_max κ_min) r,  c = p* - (s/|κ_min|) n*,  u = n* × v_min / |n* × v_min|,
   and the scene stores (c, R, u, s r). Since p* - c ∥ n* ⟂ u, p* lies on the torus' equator (height 0), at distance
   1/|κ_min| = R + sgn(κ_max κ_min) r from the axis: the outer equator for sgn = +1, the inner one for sgn = -1.
   So the curvatures and directions at p* are recovered from (c, R, u, r) and p_j, without the network:
       closest torus point to p_j:  w = p_j - c, h = w·u, e = (w - h u)/|w - h u|, m = c + R e (tube center),
                                    n_t = (p_j - m)/|p_j - m|, p_t = m + r n_t        (p_t = p* up to the a00 offset)
       principal directions:        t_par = u × e (along the parallel), t_mer = n_t × t_par (along the meridian)
       curvatures w.r.t. n_t:       κ_mer = 1/r,  κ_par = cos θ / (R + r cos θ),  cos θ = n_t·e
   With the solid's outward normal (s = +1: the tube is the solid, normal n_t; s = -1: its complement, normal -n_t),
   κ_1 = s κ_mer, κ_2 = s κ_par. The network's convention (height along +n, a sphere with outward normals has κ = -1)
   is the opposite sign: κ_max = -κ_1 (meridian, ∥ u), κ_min = -κ_2 (parallel, ∥ t_par), checked in
   tests/sdfview/test_inspector.py against the fundamental forms.
   The two principal circles through p_t lie on the torus: the meridian (center m, radius r, plane normal t_par) and
   the parallel (center c + (p_t - c)·u u, radius R + r cos θ, plane normal u); at p* (cos θ = ±1) they are the
   osculating circles of the fitted surface along v_max and v_min.

3. The network's neighborhood of p_j: the k_network nearest inlier points of p_j (points_as_tori.get_neighbors; the
   point itself first), the points the quadratic Q was predicted from. Not stored by the fit: recomputed here;
   k_network is the scene's 'network_k' parameter, else 64 (the bundled model's).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .scene import Scene

NETWORK_K_DEFAULT = 64  # k_neighbors of points_as_tori/models/FundamentalFormPredictor.pt


@dataclass
class TorusGeometry:
	index: int  # point j
	center: np.ndarray
	axis: np.ndarray  # unit u
	R: float
	r: float  # |minor radius|
	s: float  # +1: the tube is the solid, -1: its complement
	foot: np.ndarray  # p_t, the torus point closest to p_j (≈ p* = p_j + a00 n_j)
	offset: float  # |p_j - p_t| (≈ |a00|)
	normal: np.ndarray  # outward normal of the solid at p_t: s n_t
	dir_min: np.ndarray  # t_par: direction of κ_min (parallel)
	dir_max: np.ndarray  # t_mer: direction of κ_max (meridian)
	kappa_min: float  # network sign convention (see the module docstring)
	kappa_max: float
	outer: bool  # p_t on the outer equator (sgn(κ_max κ_min) = +1)
	circle_min: tuple  # (center, radius, plane normal) of the parallel through p_t
	circle_max: tuple  # (center, radius, plane normal) of the meridian through p_t
	# 'torus': all of the above recovered from the stored torus; 'stored': κ, directions, n*, foot point p* and the
	# osculating circles from the scene's per-point arrays (curvature, frames: the network's own values); then
	# `recovered` = (κ_min, κ_max) recovered from the torus, for comparison
	source: str = 'torus'
	recovered: Optional[tuple] = None


@dataclass
class Inspection:
	x: np.ndarray  # the inspected point
	neighbors: np.ndarray  # (k,) point indices, by decreasing weight
	weights: np.ndarray  # (k,) ω_i (0 for outliers)
	distances: np.ndarray  # (k,) torus signed distances g_i(x)
	phi: float  # Σ ω_i g_i(x)
	shift: float
	lam: float
	selected: int = 0  # rank of the neighbor whose torus is shown
	torus: Optional[TorusGeometry] = None
	network: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))  # k_network neighbors of p_j
	network_k: int = NETWORK_K_DEFAULT
	radius: float = 0.0  # size of the inspected region: drawing of the torus and circles is limited to it
	# points_as_tori.explain.explain_point for point j (explain_network): network neighborhood in order, coefficients,
	# curvatures, attention rollout and gradient attribution per neighbor, least-squares curvatures from 6 and k points
	explanation: Optional[dict] = None


def pat_field(scene: Scene) -> Optional[dict]:
	return next((f for f in scene.fields if f['type'] == 'pat'), None)


def _arrays(scene: Scene, field: dict):
	a = scene.field_arrays(field)
	pts = np.asarray(a['points'], np.float64)
	tori = np.asarray(a['tori'], np.float64)
	return pts[:, :3], pts[:, 3] > 0.5, tori[:, 0:3], tori[:, 3], tori[:, 4:7], tori[:, 7]


def torus_distance(q: np.ndarray, center, axis, R, r_signed) -> np.ndarray:
	"""g(q) = s (|(|(q - c) × u| - R, (q - c)·u)| - |r|), s = sign(r) (field_pat.glsl); q (..., 3)."""
	w = np.asarray(q, float) - center
	h = np.sum(w * axis, axis=-1)
	v = np.linalg.norm(np.cross(w, axis), axis=-1)
	s = np.where(np.asarray(r_signed) >= 0, 1.0, -1.0)
	return s * (np.hypot(v - R, h) - np.abs(r_signed))


def torus_geometry(index: int, point: np.ndarray, center, axis, R, r_signed) -> TorusGeometry:
	"""Principal geometry of torus `index` at the torus point closest to its point (module docstring, 2.)."""
	u = np.asarray(axis, float) / np.linalg.norm(axis)
	c = np.asarray(center, float)
	r, s = abs(float(r_signed)), (1.0 if r_signed >= 0 else -1.0)
	w = point - c
	h = float(w @ u)
	radial = w - h * u
	rho = float(np.linalg.norm(radial))
	if rho < 1e-12:  # on the axis: any direction perpendicular to u
		radial = np.cross(u, [1.0, 0.0, 0.0] if abs(u[0]) < 0.9 else [0.0, 1.0, 0.0])
		rho = float(np.linalg.norm(radial))
	e = radial / rho
	m = c + R * e
	d = point - m
	n_t = d / max(float(np.linalg.norm(d)), 1e-30)
	foot = m + r * n_t
	t_par = np.cross(u, e)
	t_mer = np.cross(n_t, t_par)
	cos_t = float(n_t @ e)
	rho_t = R + r * cos_t  # radius of the parallel through p_t
	k_mer, k_par = 1.0 / r, cos_t / rho_t
	# the network's sign convention: κ = -(curvature w.r.t. the solid's outward normal s n_t)
	kappa_max, kappa_min = -s * k_mer, -s * k_par
	if abs(kappa_min) > abs(kappa_max):  # |cos θ| / ρ_t > 1/r happens off the equator only
		kappa_min, kappa_max = kappa_max, kappa_min
		t_par, t_mer = t_mer, t_par
	axis_point = c + float((foot - c) @ u) * u
	return TorusGeometry(
		index=int(index), center=c, axis=u, R=float(R), r=r, s=s, foot=foot, offset=float(np.linalg.norm(point - foot)),
		normal=s * n_t, dir_min=t_par, dir_max=t_mer, kappa_min=float(kappa_min), kappa_max=float(kappa_max),
		outer=bool(cos_t > 0), circle_min=(axis_point, abs(rho_t), u), circle_max=(m, r, np.cross(e, u)),
	)  # fmt: skip


def inspect(
	scene: Scene,
	x,
	k: Optional[int] = None,
	lambda_scale: Optional[float] = None,
	selected: int = 0,
	network_k: Optional[int] = None,
) -> Optional[Inspection]:
	"""The blend of the scene's PAT field at x, the torus of the neighbor of rank `selected` (by weight) and its
	network neighborhood (module docstring). None if the scene has no PAT field."""
	f = pat_field(scene)
	if f is None:
		return None
	params = f.get('params', {})
	points, outlier, centers, R, axes, r = _arrays(scene, f)
	x = np.asarray(x, float).reshape(3)
	k = min(int(k or params.get('k', 32)), len(points))
	lam_scale = float(lambda_scale if lambda_scale is not None else params.get('lambda_scale', 1.0))
	d = np.linalg.norm(points - x, axis=1)
	nb = np.argpartition(d, k - 1)[:k]
	dist = d[nb]
	shift = 0.5 * float(dist.max())
	lam = lam_scale * float(params.get('max_exp_arg', 64.0)) / max(shift, 1e-30)
	w = np.where(outlier[nb], 0.0, np.exp(-lam * (dist - shift)))
	weights = w / max(float(w.sum()), 1e-300)
	g = torus_distance(x, centers[nb], axes[nb], R[nb], r[nb])
	order = np.argsort(-weights, kind='stable')
	ins = Inspection(
		x=x, neighbors=nb[order], weights=weights[order], distances=g[order], phi=float(weights @ g), shift=shift, lam=lam,
		network_k=int(network_k or params.get('network_k') or NETWORK_K_DEFAULT),
	)  # fmt: skip
	select(scene, ins, selected)
	return ins


def select(scene: Scene, ins: Inspection, rank: int) -> Inspection:
	"""Show the torus of the neighbor of rank `rank` (by weight, wrapped) and recompute its network neighborhood."""
	f = pat_field(scene)
	points, outlier, centers, R, axes, r = _arrays(scene, f)
	ins.selected = int(rank) % len(ins.neighbors)
	j = int(ins.neighbors[ins.selected])
	ins.torus = torus_geometry(j, points[j], centers[j], axes[j], R[j], r[j])
	_use_stored(scene, f, j, points[j], ins.torus)
	ins.explanation = None
	inliers = np.flatnonzero(~outlier)
	kk = min(ins.network_k, len(inliers))
	dj = np.linalg.norm(points[inliers] - points[j], axis=1)
	near = np.argpartition(dj, kk - 1)[:kk]
	near = near[np.argsort(dj[near], kind='stable')]
	ins.network = inliers[near]
	ins.radius = 1.5 * float(dj[near].max())  # the network patch's scale
	return ins


def _normals(scene: Scene, field: dict) -> Optional[np.ndarray]:
	"""Unit normals of the field's points: the point cloud sharing its points array (sdfview.sources.pat)."""
	pts = field['arrays']['points']
	pc = next((p for p in scene.pointclouds if p.get('positions') == pts and p.get('normals')), None)
	if pc is None:
		return None
	n = np.asarray(scene.arrays[pc['normals']], np.float64)[:, :3]
	return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)


def _use_stored(scene: Scene, field: dict, j: int, point: np.ndarray, t: TorusGeometry) -> None:
	"""Replace the recovered curvatures by the scene's stored ones (the network's values), when present: κ, directions,
	n*, the foot point p* = p_j + a00 n_j and the osculating circles at p* (center p* + n*/κ, radius 1/|κ|, in the plane
	of n* and the direction; κ > 0 bends toward +n*)."""
	arrays = field['arrays']
	normals = _normals(scene, field)
	if 'curvature' not in arrays or 'frames' not in arrays or normals is None:
		return
	k_min, k_max, a00 = (float(v) for v in scene.arrays[arrays['curvature']][j, :3])
	fr = np.asarray(scene.arrays[arrays['frames']][j], np.float64)
	d_min, d_max, n_star = fr[0:3], fr[4:7], fr[8:11]
	foot = point + a00 * normals[j]
	t.recovered = (t.kappa_min, t.kappa_max)
	t.kappa_min, t.kappa_max = k_min, k_max
	t.dir_min, t.dir_max, t.normal, t.foot = d_min, d_max, n_star, foot
	t.offset = abs(a00)
	t.circle_min = (foot + n_star / k_min, 1 / abs(k_min), np.cross(d_min, n_star)) if k_min != 0 else t.circle_min
	t.circle_max = (foot + n_star / k_max, 1 / abs(k_max), np.cross(d_max, n_star)) if k_max != 0 else t.circle_max
	t.source = 'stored'


_MODELS: dict = {}


def explain_network(scene: Scene, ins: Inspection, model=None) -> Optional[dict]:
	"""
	Run the network on point j's neighborhood (points_as_tori.explain.explain_point, CPU float32): the attribution of
	every neighbor (attention rollout, gradient of κ) and least-squares curvatures for comparison. The model: `model`,
	else the field's 'network_model' parameter ('bundled' or a checkpoint path), else the bundled one. Sets and returns
	ins.explanation and reorders ins.network as the network saw it; None without torch / points_as_tori / normals.
	"""
	try:
		import torch

		from points_as_tori import config
		from points_as_tori.explain import explain_point
		from points_as_tori.infer import DEFAULT_MODEL_PATH
		from points_as_tori.network import FundamentalFormPredictor
	except ImportError:
		return None
	f = pat_field(scene)
	normals = _normals(scene, f)
	if normals is None:
		return None
	if model is None:
		name = f.get('params', {}).get('network_model', 'bundled')
		path = DEFAULT_MODEL_PATH if name == 'bundled' or not os.path.isfile(str(name)) else name
		if path not in _MODELS:
			try:
				config.get_device()
			except RuntimeError:  # nothing configured in this process: the CPU (an existing choice is kept)
				config.set_device('cpu', torch.float32)
			m, _ = FundamentalFormPredictor.load_saved_model(path)
			_MODELS[path] = m.to('cpu', torch.float32).eval()
		model = _MODELS[path]
	points, outlier = _arrays(scene, f)[:2]
	j = int(ins.neighbors[ins.selected])
	P = torch.as_tensor(points.T, dtype=torch.float32)
	N = torch.as_tensor(normals.T, dtype=torch.float32)
	ex = explain_point(model, P, N, j, ins.network_k, torch.as_tensor(outlier))
	ex = {key: (v.numpy() if hasattr(v, 'numpy') else v) for key, v in ex.items()}
	ins.network = ex['indices'].astype(np.int64)
	ins.explanation = ex
	return ex


def describe(ins: Inspection) -> list:
	"""Text lines of an inspection (window panel, terminal)."""
	t = ins.torus
	x = ', '.join(f'{v:.4f}' for v in ins.x)
	j = int(ins.neighbors[ins.selected])
	solid = 'tube' if t.s > 0 else 'outside the tube'
	lines = [
		f'x = ({x})   phi(x) = {ins.phi:+.5f}   k = {len(ins.neighbors)}   lambda = {ins.lam:.4g}   shift = {ins.shift:.4g}',
		f'neighbor {ins.selected + 1}/{len(ins.neighbors)}: point {j}   weight {ins.weights[ins.selected]:.4f}   '
		f'g_j(x) = {ins.distances[ins.selected]:+.5f}',
		f'torus: R = {t.R:.4g}   r = {t.r:.4g}   solid: {solid}   {"outer" if t.outer else "inner"} equator   '
		f'|p_j - p_t| = {t.offset:.3g} (≈ |a00|)',
		f'kappa_min = {t.kappa_min:+.4g} (radius {1 / max(abs(t.kappa_min), 1e-30):.4g}, cyan)   '
		f'kappa_max = {t.kappa_max:+.4g} (radius {1 / max(abs(t.kappa_max), 1e-30):.4g}, magenta)   '
		+ (
			f'[network, stored; from the torus {t.recovered[0]:+.4g}, {t.recovered[1]:+.4g}]'
			if t.source == 'stored'
			else '[recovered from the torus]'
		),
	]
	ex = ins.explanation
	if ex is None:
		lines.append(f'network neighborhood of point {j}: {len(ins.network)} points (green)')
		return lines
	lsq = '   '.join(f'from {m} points {a:+.4g}, {b:+.4g}' for m, (a, b) in ex['least_squares'].items())
	lines.append(f'least-squares quadratic: kappa {lsq}')
	shares = []
	for name in ('gradient', 'rollout'):
		w = np.asarray(ex[name], np.float64)
		w = w / max(float(w.sum()), 1e-300)
		cum = np.cumsum(w)
		shares.append(
			f'{name}: point itself {w[0]:.2f}, nearest 6 {cum[min(5, len(w) - 1)]:.2f}, 90% within {int(np.searchsorted(cum, 0.9)) + 1}'
		)
	lines.append(f'network neighborhood: {len(ins.network)} points (green)   ' + '   '.join(shares))
	return lines


# ---- what a host draws (positions, sizes, colors; the same for every host)

COLOR_X = (1.0, 1.0, 1.0, 1.0)  # the inspected point
COLOR_SELECTED = (0.85, 0.05, 0.05, 1.0)  # point j (whose torus is shown)
COLOR_NETWORK = (0.15, 0.7, 0.3, 1.0)  # the network's neighborhood of p_j
COLOR_MIN = (0.0, 0.75, 0.85, 1.0)  # κ_min: the parallel circle, its direction
COLOR_MAX = (0.85, 0.15, 0.75, 1.0)  # κ_max: the meridian circle, its direction
COLOR_AXIS = (0.35, 0.35, 0.35, 1.0)
TORUS_FRONT, TORUS_BACK, TORUS_OPACITY = (0.95, 0.6, 0.8), (0.6, 0.35, 0.5), 0.35


def weight_color(w: float) -> tuple:
	"""Blend weight ω / max ω in [0, 1] -> pale yellow .. dark orange."""
	lo, hi = np.array([1.0, 0.95, 0.55]), np.array([0.85, 0.3, 0.0])
	return (*(lo + (hi - lo) * float(np.clip(w, 0, 1))), 1.0)


ATTRIBUTIONS = (None, 'gradient', 'rollout')  # how the network neighborhood is colored (window: key A)


def attribution_color(w: float) -> tuple:
	"""Attribution share / max share in [0, 1] -> pale green .. dark green."""
	lo, hi = np.array([0.8, 0.95, 0.8]), np.array([0.0, 0.4, 0.1])
	return (*(lo + (hi - lo) * float(np.clip(w, 0, 1)) ** 0.5), 1.0)


def marks(scene: Scene, ins: Inspection, point_radius: float, network: bool = True, attribution: Optional[str] = None):
	"""Sprites: (positions (n, 3), radii (n,), colors (n, 4)) float32, drawn in order (later on top): the network
	neighborhood (flat green, or by attribution 'gradient' / 'rollout' when ins.explanation is set: pale -> dark green,
	larger), the k blended points by weight, point j, the foot point, x."""
	f = pat_field(scene)
	points = _arrays(scene, f)[0]
	pos, rad, col = [], [], []

	def add(p, r, c):
		pos.append(p)
		rad.append(r)
		col.append(c)

	if network:
		ex = ins.explanation
		if attribution and ex is not None:
			w = np.asarray(ex[attribution], np.float64)
			# scaled by the strongest neighbor other than point j itself (j is drawn red; its gradient share is ~0.4)
			w = np.clip(w / max(float(w[1:].max() if len(w) > 1 else w.max()), 1e-300), 0, 1)
			for i, wi in sorted(zip(ins.network, w), key=lambda p: p[1]):  # strongest last: on top
				add(points[i], point_radius * (0.8 + 1.7 * wi), attribution_color(wi))
		else:
			for i in ins.network:
				add(points[i], 1.1 * point_radius, COLOR_NETWORK)
	wmax = max(float(ins.weights.max()), 1e-30)
	for i, w in zip(ins.neighbors[::-1], ins.weights[::-1]):  # heaviest last: on top
		add(points[i], point_radius * (1.0 + 1.5 * w / wmax), weight_color(w / wmax))
	j = int(ins.neighbors[ins.selected])
	add(points[j], 2.5 * point_radius, COLOR_SELECTED)
	add(ins.torus.foot, 1.2 * point_radius, COLOR_MAX)
	add(ins.x, 2.0 * point_radius, COLOR_X)
	return np.asarray(pos, np.float32), np.asarray(rad, np.float32), np.asarray(col, np.float32)


def _arc(center, radius, normal, through, half_angle, n=96):
	"""Points of the circle (center, radius, plane normal) from -half_angle to +half_angle around `through`."""
	a = through - center
	a = a - (a @ normal) * normal
	a = a / max(float(np.linalg.norm(a)), 1e-30)
	b = np.cross(normal, a)
	t = np.linspace(-half_angle, half_angle, n + 1)
	return center + radius * (np.cos(t)[:, None] * a + np.sin(t)[:, None] * b)


def lines(ins: Inspection, reach: float):
	"""Segments (positions (2m, 3), colors (2m, 4)) float32: the principal circles through p_t (whole, or the arcs
	within `reach` of p_t when larger), the principal directions and the normal at p_t, the torus axis, x -> p_j."""
	t = ins.torus
	segs, cols = [], []

	def polyline(p, c):
		for a, b in zip(p[:-1], p[1:]):
			segs.extend([a, b])
			cols.extend([c, c])

	for (center, radius, normal), c in ((t.circle_min, COLOR_MIN), (t.circle_max, COLOR_MAX)):
		# the arc of chord length <= reach on each side of p_t: half angle 2 asin(reach / (2 radius))
		half = np.pi if reach >= 2 * radius else 2 * np.arcsin(reach / (2 * radius))
		polyline(_arc(center, radius, normal, t.foot, half), c)
	arm = 0.5 * reach
	polyline(np.array([t.foot - arm * t.dir_min, t.foot + arm * t.dir_min]), COLOR_MIN)
	polyline(np.array([t.foot - arm * t.dir_max, t.foot + arm * t.dir_max]), COLOR_MAX)
	polyline(np.array([t.foot, t.foot + arm * t.normal]), COLOR_AXIS)
	# the axis line near the torus: its closest point to p_t ± (R + r)
	foot_on_axis = t.center + float((t.foot - t.center) @ t.axis) * t.axis
	half_axis = min(t.R + t.r, reach)
	polyline(np.array([foot_on_axis - half_axis * t.axis, foot_on_axis + half_axis * t.axis]), COLOR_AXIS)
	polyline(np.array([ins.x, t.foot]), (0.6, 0.6, 0.6, 1.0))
	return np.asarray(segs, np.float32), np.asarray(cols, np.float32)


def torus_box(ins: Inspection, limit: float):
	"""Box the torus is traced in: the whole torus if it fits in a cube of half-size `limit`, else the cube of
	half-size max(radius, ½ limit) around p_t (large R: near-flat directions)."""
	t = ins.torus
	full = t.R + t.r
	if full <= limit:
		return t.center - full, t.center + full
	h = max(ins.radius, 0.5 * limit)
	return t.foot - h, t.foot + h
