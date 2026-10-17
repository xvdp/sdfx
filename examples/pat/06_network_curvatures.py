"""
FIT (inspection) — what the network's curvatures are, measured over a point cloud: the principal curvatures of the
network's quadratic Q (6 coefficients per point, predicted from the point's k_network = 64 nearest points) against a
classical least-squares quadratic fitted to the heights of the 6, 10, 16, 32, 64 nearest points; and how much each
neighbor contributes (points_as_tori/explain.py: gradient attribution ‖∂(κ_min, κ_max)/∂p_j‖ and attention rollout).

	INPUT   a point cloud with normals (.ply) or a scene (.sdfx with a pat field and normals)
	OUTPUT  printed: curvature statistics per method, agreement with the network, attribution shares by distance rank
	python examples/pat/06_network_curvatures.py ext/points-as-tori/data/tooth.ply
	python examples/pat/06_network_curvatures.py examples/out/pat/knot.sdfx --device cuda

Measured on tooth.ply (2048 points, CPU, 11 s): every network κ pair is convex (K > 0, H < 0 at 100% of the points)
and nearly umbilic (median |κ_min| 1.81, |κ_max| 1.97), while least squares from 64 points finds K < 0 at 43%;
on an exact sphere of radius 0.5 the network gives κ ≈ -1.76 / -1.91 (exact -2), least squares -2.00.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import points_as_tori as pat  # noqa: E402
from points_as_tori import config  # noqa: E402
from points_as_tori.explain import (  # noqa: E402
	attention_rollout,
	curvature_frame,
	gradient_attribution,
	least_squares_forms,
)
from points_as_tori.infer import DEFAULT_MODEL_PATH  # noqa: E402
from points_as_tori.network import FundamentalFormPredictor  # noqa: E402
from points_as_tori.torus_distance import get_neighbors  # noqa: E402


def load_cloud(path):
	if path.endswith('.sdfx'):
		from sdfview.inspector import _normals, pat_field
		from sdfview.scene import Scene

		scene = Scene.load(path)
		f = pat_field(scene)
		return scene.arrays[f['arrays']['points']][:, :3], _normals(scene, f)
	return pat.read_point_cloud(path)


def main():
	ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
	ap.add_argument('cloud')
	ap.add_argument('--model', default=DEFAULT_MODEL_PATH)
	ap.add_argument('--device', default='cpu')
	ap.add_argument('--batch', type=int, default=128, help='neighborhoods per network call (attention: k² per head)')
	a = ap.parse_args()

	config.set_device(a.device, torch.float32)
	p, n = load_cloud(a.cloud)
	P, N = config.as_wide(np.array(p, np.float32), 3), config.as_wide(np.array(n, np.float32), 3)  # copies: writable
	model, k = FundamentalFormPredictor.load_saved_model(a.model)
	model.eval()
	idx = get_neighbors(P, k)
	m = P.shape[1]
	coef = torch.empty(6, m, device=P.device)
	roll = torch.empty(m, k, device=P.device)
	grad = torch.empty(m, k, device=P.device)
	t0 = time.perf_counter()
	for s in range(0, m, a.batch):
		e = min(m, s + a.batch)
		pos, nrm = P[:, idx[s:e]], N[:, idx[s:e]]
		with torch.no_grad():
			c, att = model(pos, nrm, return_attention=True)
		coef[:, s:e], roll[s:e], grad[s:e] = c, attention_rollout(att), gradient_attribution(model, pos, nrm)
	print(f'{m} points, k_network = {k}: network + attributions {time.perf_counter() - t0:.1f} s')

	methods = {'network': coef}
	for j in (6, 10, 16, 32, 64):
		if j <= k:
			methods[f'lsq {j}'] = least_squares_forms(P, N, idx[:, :j])
	kappa = {}
	for name, c in methods.items():
		f = curvature_frame(P, N, c)
		kappa[name] = (f['kappa_min'].cpu().numpy(), f['kappa_max'].cpu().numpy())
	d = (P[:, idx] - P[:, idx[:, :1]]).norm(dim=0)
	print(
		f'neighborhood radius (median): 6 nearest {float(d[:, 5].median()):.4g}, {k} nearest {float(d[:, -1].median()):.4g}'
	)
	print(f'{"method":9s} {"median|κ_min|":>13s} {"median|κ_max|":>13s} {"K<0":>6s} {"H<0":>6s}   vs network: '
	      f'{"sign(H)":>7s} {"sign(K)":>7s} {"corr(H)":>7s} {"med rel|ΔH|":>11s}')  # fmt: skip
	H0 = 0.5 * (kappa['network'][0] + kappa['network'][1])
	K0 = kappa['network'][0] * kappa['network'][1]
	for name, (kmin, kmax) in kappa.items():
		H, K = 0.5 * (kmin + kmax), kmin * kmax
		row = f'{name:9s} {np.median(np.abs(kmin)):13.4g} {np.median(np.abs(kmax)):13.4g} {np.mean(K < 0):6.3f} {np.mean(H < 0):6.3f}'
		if name != 'network':
			rel = np.median(np.abs(H - H0) / np.maximum(np.abs(H0), 1e-12))
			row += (f'   {np.mean(np.sign(H) == np.sign(H0)):10.3f} {np.mean(np.sign(K) == np.sign(K0)):7.3f} '
			        f'{np.corrcoef(H, H0)[0, 1]:7.3f} {rel:11.3f}')  # fmt: skip
		print(row)
	for name, w in (('gradient', grad), ('rollout', roll)):
		w = (w / w.sum(1, keepdim=True)).cpu().numpy()
		cum = np.cumsum(w, axis=1)
		print(f'{name:8s} attribution by distance rank: the point itself {w[:, 0].mean():.3f}, nearest 6 '
		      f'{cum[:, 5].mean():.3f} (uniform {6 / k:.3f}), nearest 16 {cum[:, 15].mean():.3f}; '
		      f'neighbors holding 90%: median {int(np.median((cum < 0.9).sum(1) + 1))}')  # fmt: skip


if __name__ == '__main__':
	main()
