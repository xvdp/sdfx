"""
PAT — stage FIT with a trained checkpoint: fit the same point cloud with the bundled network and with yours, measure
how the two fields differ, and write both into one scene to toggle (G) or overlay (V, T) in the viewer.

	INPUT   --model: your checkpoint (from 03_train_original.py or 04_train_new.py), e.g. examples/out/models/mine.pt
	        a point cloud with normals (.ply), default ext/points-as-tori/data/tooth.ply
	OUTPUT  <out>/<name>_bundled_vs_<model>.sdfx   models 'bundled' and '<model>' (exact field, grid bake, points)
	        printed: |φ_trained - φ_bundled| at points near the surface and in the box; the tori's radii statistics

	python examples/pat/05_fit_trained.py --model examples/out/models/mine.pt --device cuda:0
	python -m sdfview view examples/out/pat/tooth_bundled_vs_mine.sdfx --overlay --opacity 0.45
"""

import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

import torch  # noqa: E402

import points_as_tori as pat  # noqa: E402
from points_as_tori import config  # noqa: E402


def main():
	ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
	ap.add_argument('input', nargs='?', default=os.path.join(REPO, 'ext/points-as-tori/data/tooth.ply'))
	ap.add_argument('--model', required=True, help='trained checkpoint (.pt)')
	ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
	ap.add_argument('--res', type=int, default=96, help='grid bake per model')
	ap.add_argument('-o', '--out', default=os.path.join(REPO, 'examples', 'out', 'pat'))
	a = ap.parse_args()

	from sdfview.scene import merge_scenes
	from sdfview.sources.pat import scene_from_pat

	pat.set_device(a.device, torch.float32)
	points, normals = pat.read_point_cloud(a.input)
	name = os.path.splitext(os.path.basename(a.input))[0]
	tag = os.path.splitext(os.path.basename(a.model))[0]
	fits = {'bundled': pat.PointsAsTori(points, normals), tag: pat.PointsAsTori(points, normals, model_path=a.model)}

	# where the fields differ: near the surface (points moved along their normals) and in the bounding box
	P, N = config.as_wide(points, 3), config.as_wide(normals, 3)
	g = torch.Generator(device='cpu').manual_seed(0)
	near = P + N * (torch.rand(P.shape[1], generator=g) * 0.1 - 0.05).to(P)
	lo, hi = P.amin(dim=1, keepdim=True), P.amax(dim=1, keepdim=True)
	box = lo + (hi - lo) * torch.rand(3, 50_000, generator=g).to(P)
	for label, q in (('near surface', near), ('bounding box', box)):
		d = (fits[tag].signed_distance(q) - fits['bundled'].signed_distance(q)).abs()
		print(f'{label:13s} |φ_{tag} - φ_bundled|: mean {float(d.mean()):.4g}, max {float(d.max()):.4g}')
	for key, sdf in fits.items():
		_, _, R, r = sdf.get_tori()
		print(f'{key:8s} tori: median |R| {float(R.abs().median()):.4g}, median |r| {float(r.abs().median()):.4g}')

	scenes = {key: scene_from_pat(sdf, bake_resolution=a.res, name='pat') for key, sdf in fits.items()}
	path = os.path.join(a.out, f'{name}_bundled_vs_{tag}.sdfx')
	os.makedirs(a.out, exist_ok=True)
	merge_scenes(scenes, meta={'input': a.input, 'models': {'bundled': 'bundled', tag: a.model}}).save(path)
	print(f'wrote {path}\nview: python -m sdfview view {path} --overlay --opacity 0.45   (G: opaque model, V, T)')


if __name__ == '__main__':
	main()
