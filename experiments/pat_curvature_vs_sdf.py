"""
Does a classical curvature estimate give a good Points-as-Tori SDF?

Compares, by signed-distance error against ground truth, tori built from
  * the pre-trained PAT network (k = 64),
  * least-squares osculating jets (quadratic height field fitted to the k nearest neighbors in each point's frame;
    Cazals & Pouget 2005), optionally also fitting the neighbors' normals,
  * on an exact torus surface: the exact second fundamental form.
All produce the same 6 coefficients (a00, a01, a10, a11, a02, a20) of Q(s, t) = a00 + a10 s + a01 t + a20 s² + a11 s t
+ a02 t² and go through the same torus fit (fit_tori_from_forms) and blend (k = 32).

	python experiments/pat_curvature_vs_sdf.py        (CPU, a few minutes)
"""

import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO)

import points_as_tori as pat  # noqa: E402
from points_as_tori.infer import DEFAULT_MODEL_PATH as TORCH_MODEL  # noqa: E402
from points_as_tori.shape_2d import orthonormal_basis  # noqa: E402
from points_as_tori.torus_distance import get_torus_parameters_python  # noqa: E402
from points_as_tori.training import mesh_sampling as pms  # noqa: E402

DATA = os.path.join(REPO, 'ext', 'points-as-tori', 'data')


def sample_mesh(vertices, faces, n_points, rng):
	"""Area-uniform samples on a triangle mesh with face normals: α = √u (1 - v), β = v √u, γ = 1 - α - β."""
	v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
	cr = np.cross(v1 - v0, v2 - v0)
	area = 0.5 * np.linalg.norm(cr, axis=1)
	f = rng.choice(len(faces), size=n_points, p=area / area.sum())
	su, t = np.sqrt(rng.random(n_points)), rng.random(n_points)
	alpha, beta = su * (1 - t), t * su
	points = alpha[:, None] * v0[f] + beta[:, None] * v1[f] + (1 - alpha - beta)[:, None] * v2[f]
	return points, cr[f] / np.linalg.norm(cr[f], axis=1, keepdims=True)


def read_obj(path):
	V, F = [], []
	for line in open(path):
		p = line.split()
		if not p:
			continue
		if p[0] == 'v':
			V.append([float(x) for x in p[1:4]])
		elif p[0] == 'f':
			F.append([int(x.split('/')[0]) - 1 for x in p[1:4]])
	V, F = np.array(V), np.array(F)
	V = (V - (V.max(0) + V.min(0)) / 2) / (V.max(0) - V.min(0)).max()  # centered, max extent 1
	return V, F


def jet_fit(points, normals, k, use_normals=False):
	"""
	Least-squares quadratic height field per point: in the frame (s_hat, t_hat, n) of point i, neighbors j give rows
	[1, s, t, s², s t, t²] · (a00, a10, a01, a20, a11, a02) = h_j. With use_normals, each neighbor also gives the slope
	rows Q_s = -n_s/n_h, Q_t = -n_t/n_h (scaled by the patch radius to match units; skipped where n_h < 0.3).
	Returns (6, P) in PAT order (a00, a01, a10, a11, a02, a20).
	"""
	P, N = pat.as_wide(points, 3), pat.as_wide(normals, 3)
	idx = pat.get_neighbors(P, k)
	n = N / N.norm(dim=0, keepdim=True)
	s_hat, t_hat = orthonormal_basis(n)
	D = P[:, idx] - P[:, :, None]
	s, t, h = ((D * e[:, :, None]).sum(0) for e in (s_hat, t_hat, n))
	one, zero = torch.ones_like(s), torch.zeros_like(s)
	rows, rhs = [torch.stack([one, s, t, s * s, s * t, t * t], -1)], [h]
	if use_normals:
		Nj = n[:, idx]
		ns, nt, nh = ((Nj * e[:, :, None]).sum(0) for e in (s_hat, t_hat, n))
		w = (nh > 0.3).to(s.dtype) * D.norm(dim=0).median(dim=-1).values[:, None]
		rows += [w[..., None] * torch.stack([zero, one, zero, 2 * s, t, zero], -1)]
		rows += [w[..., None] * torch.stack([zero, zero, one, zero, s, 2 * t], -1)]
		rhs += [w * (-ns / nh.clamp_min(0.3)), w * (-nt / nh.clamp_min(0.3))]
	x = torch.linalg.lstsq(torch.cat(rows, 1), torch.cat(rhs, 1)[..., None]).solution[..., 0]
	a00, a10, a01, a20, a11, a02 = x.unbind(-1)
	return torch.stack([a00, a01, a10, a11, a02, a20], 0)


def sdf_errors(points, normals, coeffs, queries, gt):
	tdf = pat.TorusDistanceField(pat.as_wide(points, 3), pat.as_wide(normals, 3))
	tdf.fit_tori_from_forms(coeffs)
	tdf.set_k_evaluation(32)
	d = tdf.evaluate_distance(queries).numpy()
	e = np.abs(d - gt)
	return np.mean(e), np.percentile(e, 90), np.mean(np.sign(d) == np.sign(gt)), tdf


def meshes(model, k_model, rng):
	print(
		f'{"mesh / method":42s} {"band mean|e|":>12s} {"p90":>9s} {"sign":>7s} {"box mean|e|":>12s} {"p90":>9s} {"sign":>7s}'
	)
	for name in ('femur.obj', 'block.obj', 'knot.obj'):
		V, F = read_obj(os.path.join(DATA, name))
		tV, tF = pat.as_wide(V, 3), pat.as_wide(F, 3, dtype=torch.long)
		fmask = torch.ones(len(F), dtype=torch.bool)
		points, normals = sample_mesh(V, F, 4000, rng)
		sp, sn = sample_mesh(V, F, 3000, rng)
		queries = {
			'band': sp + rng.uniform(-0.05, 0.05, size=(3000, 1)) * sn,  # |offset| <= 0.05 along the normal
			'box': rng.uniform(V.min(0) - 0.1, V.max(0) + 0.1, size=(3000, 3)),
		}
		queries = {k: pat.as_wide(q, 3) for k, q in queries.items()}
		gts = {k: pms.signed_distance_to_mesh(q, tV, tF, fmask).numpy() for k, q in queries.items()}
		methods = {
			'network (pre-trained, k=64)': model.precompute_coefficients(
				pat.as_wide(points, 3), pat.as_wide(normals, 3), k_model
			)
		}
		for k in (16, 32, 64):
			methods[f'jet LS, k={k}'] = jet_fit(points, normals, k)
			methods[f'jet LS + normals, k={k}'] = jet_fit(points, normals, k, use_normals=True)
		print(f'{name}: {len(V)} vertices, {len(F)} faces, 4000 points')
		for mname, c in methods.items():
			b = sdf_errors(points, normals, c, queries['band'], gts['band'])
			x = sdf_errors(points, normals, c, queries['box'], gts['box'])
			print(f'  {mname:40s} {b[0]:12.2e} {b[1]:9.2e} {b[2]:7.3f} {x[0]:12.2e} {x[1]:9.2e} {x[2]:7.3f}')


def exact_torus(model, k_model, rng, R0=0.35, r0=0.12, n=3000):
	"""Samples of an exact torus: curvature accuracy vs SDF accuracy, including the exact second fundamental form."""
	u, v = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
	points = np.stack([(R0 + r0 * np.cos(v)) * np.cos(u), (R0 + r0 * np.cos(v)) * np.sin(u), r0 * np.sin(v)], -1)
	normals = np.stack([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)], -1)
	q = rng.uniform(-0.6, 0.6, size=(n, 3))
	gt = np.sqrt((np.linalg.norm(q[:, :2], axis=1) - R0) ** 2 + q[:, 2] ** 2) - r0
	P, N = pat.as_wide(points, 3), pat.as_wide(normals, 3)
	s_hat, t_hat = orthonormal_basis(N)

	# exact second fundamental form in each frame (h = d·n convention: convex directions are negative)
	e_u = np.stack([-np.sin(u), np.cos(u), np.zeros_like(u)], -1)
	e_v = np.stack([-np.sin(v) * np.cos(u), -np.sin(v) * np.sin(u), np.cos(v)], -1)
	k_u, k_v = -np.cos(v) / (R0 + r0 * np.cos(v)), -np.ones_like(v) / r0
	S, T = s_hat.numpy().T, t_hat.numpy().T

	def Q(a, b):
		return k_u * np.sum(a * e_u, 1) * np.sum(b * e_u, 1) + k_v * np.sum(a * e_v, 1) * np.sum(b * e_v, 1)

	z = np.zeros(n)
	exact = torch.from_numpy(np.stack([z, z, z, Q(S, T), 0.5 * Q(T, T), 0.5 * Q(S, S)], 0))
	k_true = np.sort(np.abs(np.stack([k_u, k_v], -1)), axis=1)

	print(f'\nexact torus R = {R0}, r = {r0}, {n} samples (analytic positions and normals)')
	print(
		f'  {"method":34s} {"|κ_min| err":>11s} {"|κ_max| err":>11s} {"SDF mean|e|":>12s} {"sign":>7s} {"fitted R med":>12s}'
	)
	for name, c in (
		('network (pre-trained, k=64)', model.precompute_coefficients(P, N, k_model)),
		('jet LS, k=32', jet_fit(points, normals, 32)),
		('exact second fundamental form', exact),
	):
		_, kmin, kmax, _, _ = get_torus_parameters_python(N, c, s_hat, t_hat)
		ek = [np.median(np.abs(np.abs(x.numpy()) - k_true[:, i]) / k_true[:, i]) for i, x in enumerate((kmin, kmax))]
		m, _, sign, tdf = sdf_errors(points, normals, c, pat.as_wide(q, 3), gt)
		R = tdf.major_radii
		print(f'  {name:34s} {ek[0]:11.2e} {ek[1]:11.2e} {m:12.2e} {sign:7.3f} {float(R.median()):12.4f}')
		if name.startswith('exact'):
			eq = torch.from_numpy(np.abs(np.cos(v)) > 0.95)
			print(
				f'    fitted major radius near the equators (|cos v| > 0.95): {float(R[eq].median()):.4f}; elsewhere: {float(R[~eq].median()):.4f}'
			)


if __name__ == '__main__':
	torch.set_num_threads(os.cpu_count() or 1)
	pat.set_device(pat.default_device(), torch.float64)
	model, k_model = pat.FundamentalFormPredictor.load_saved_model(TORCH_MODEL)
	rng = np.random.default_rng(0)
	meshes(model, k_model, rng)
	exact_torus(model, k_model, rng)
