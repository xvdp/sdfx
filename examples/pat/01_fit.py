"""
PAT — stage FIT (inference, no training): an oriented point cloud -> tori (pre-trained network + closed-form torus
fit) -> signed distance field; mesh of its zero level set; scene for the viewer.

	INPUT   a point cloud with normals (.ply, as the original reads it), default ext/points-as-tori/data/tooth.ply,
	        or a mesh (.obj / .ply / .stl via trimesh) sampled uniformly by area with --mesh
	MODEL   the bundled checkpoint points_as_tori/models/FundamentalFormPredictor.pt, or --model (e.g. a checkpoint
	        from examples/pat/04_train.py)
	OUTPUT  <out>/<name>_tori.pkl     the tori (the original's pickle format: centers, axes, R, r)
	        <out>/<name>_mesh.obj     marching cubes of φ = 0 (grid --res)
	        <out>/<name>.sdfx         scene: exact field (tori + BVH), grid bake, points  (python -m sdfview view)

	python examples/pat/01_fit.py                                         # tooth.ply on the default device
	python examples/pat/01_fit.py ext/points-as-tori/data/snail.ply --device cuda:0
	python examples/pat/01_fit.py ext/points-as-tori/data/knot.obj --mesh --points 20000 --device cuda:1
	python -m sdfview view examples/out/pat/tooth.sdfx                    # G: field, P: points, C: cut plane

Paths are relative to the repository root (run from there). Defaults: float32; --float64 for the C++ precision.
"""

import argparse
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

import torch  # noqa: E402

import points_as_tori as pat  # noqa: E402
from points_as_tori import config  # noqa: E402


def main():
	ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
	ap.add_argument('input', nargs='?', default=os.path.join(REPO, 'ext/points-as-tori/data/tooth.ply'))
	ap.add_argument('--mesh', action='store_true', help='input is a mesh: sample --points oriented points from it')
	ap.add_argument('--points', type=int, default=20000, help='samples of a mesh input')
	ap.add_argument('--model', default=None, help='network checkpoint (.pt); default: the bundled one')
	ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
	ap.add_argument('--float64', action='store_true')
	ap.add_argument('--res', type=int, default=128, help='grid of the mesh extraction and of the scene bake')
	ap.add_argument('-o', '--out', default=os.path.join(REPO, 'examples', 'out', 'pat'))
	a = ap.parse_args()

	pat.set_device(a.device, torch.float64 if a.float64 else torch.float32)  # once, for the whole package
	name = os.path.splitext(os.path.basename(a.input))[0]
	os.makedirs(a.out, exist_ok=True)

	# 1. input -> (n, 3) points and normals
	if a.mesh:
		import trimesh

		m = trimesh.load(a.input, force='mesh', process=False)
		mesh = pat.TriangleMesh(m.vertices, m.faces)  # on the device, (3, n) inside
		points, normals = mesh.sample_points_and_normals_uniformly(a.points, seed=0)  # (3, n) device tensors
		normals = normals / normals.norm(dim=0, keepdim=True)
		points, normals = points.T, normals.T  # (n, 3) views, no copy
	else:
		points, normals = pat.read_point_cloud(a.input)  # (n, 3) NumPy, as the original
	print(f'{name}: {points.shape[0]} points')

	# 2. FIT: network -> tori (the only stage with the network; done once per point cloud)
	t = time.perf_counter()
	sdf = pat.PointsAsTori(points, normals, model_path=a.model)
	if config.get_device().type == 'cuda':
		torch.cuda.synchronize()
	print(f'fit (network + torus fit): {time.perf_counter() - t:.2f} s')
	sdf.save_tori(os.path.join(a.out, f'{name}_tori.pkl'))  # reload: PointsAsTori(points, normals, tori_filepath=...)

	# 3. evaluate: φ and ∇φ at queries ((3, Q) tensors on the device)
	q = (torch.rand(3, 100_000, device=config.get_device(), dtype=config.get_dtype()) * 2 - 1) * 1.1
	t = time.perf_counter()
	phi, grad = sdf.signed_distance_and_gradient(q)
	print(
		f'evaluate 100k queries: {time.perf_counter() - t:.3f} s, φ in [{float(phi.min()):.3f}, {float(phi.max()):.3f}]'
	)

	# 4. mesh of φ = 0 (marching cubes in torch) -> OBJ
	V, F = pat.mesh_isocontour(sdf._tdf, 0.0, grid_resolution=a.res)  # (3, V), (3, F) on the device
	pat.TriangleMesh(V.T, F.T).write_OBJ(os.path.join(a.out, f'{name}_mesh.obj'))
	print(f'mesh: {V.shape[1]} vertices, {F.shape[1]} triangles')

	# 5. scene for the viewer: the exact field (tori + BVH), a grid bake (fast surfaces) and the input points
	from sdfview.sources.pat import scene_from_pat

	path = os.path.join(a.out, f'{name}.sdfx')
	scene_from_pat(sdf, bake_resolution=a.res, meta={'input': a.input, 'model': a.model or 'bundled'}).save(path)
	print(f'wrote {a.out}/{name}_tori.pkl, {name}_mesh.obj, {name}.sdfx\nview: python -m sdfview view {path}')


if __name__ == '__main__':
	main()
