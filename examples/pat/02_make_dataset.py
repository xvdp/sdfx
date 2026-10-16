"""
PAT — stage DATA (new data): your meshes (and/or procedural Perlin blobs) -> training files in the original's format,
as training/mesh_data.py builds them (the torch port: points_as_tori/training/mesh_data.py).

	INPUT   watertight meshes (.obj / .ply / .stl / ..., read with trimesh), e.g. ext/points-as-tori/data/*.obj;
	        --blobs N adds N procedural blobs (generate_blob_dataset, as the original's dataset)
	OUTPUT  <out>/datasets/<name>_train_meshes.npz, <name>_test_meshes.npz   meshes (split --split)
	        <out>/datasets/<name>_train_mode<m>_<i>.npz                     training features, one file per
	                                                                         --per-file shapes, per --modes m
	        a few <name>_<i>_mode<m>.obj / _pointcloud.obj next to them for inspection

	python examples/pat/02_make_dataset.py ext/points-as-tori/data/knot.obj ext/points-as-tori/data/block.obj \
		ext/points-as-tori/data/femur.obj --name mine --device cuda:0
	python examples/pat/02_make_dataset.py --blobs 40 --name blobsmall --device cuda:0    # procedural only

Steps per mesh (the requirements of generate_scanned_mesh_training_dataset, checked before writing, so no mesh is
dropped later): centre and scale to the unit cube (center_and_scale); if larger than --max-faces (the original's
limit, 2048), quadric simplification with pyfqmr, trying a few face targets (it can leave non-manifold edges: e.g.
femur.obj stays non-watertight at every target; such meshes are skipped, or pass --max-faces 8000); outward
orientation (faces flipped if the orientation test is negative); watertight, volume, consistent winding (trimesh
checks); meshes failing a check are reported and skipped. Point clouds: --points per shape (the original: 2048; a
training step runs the network on every neighborhood of a shape, so its memory grows with this number).
Training features (mode): 0 uniform or farthest-point samples (no noise), -1 ray-cast scan samples (holes), 3 the
full scan simulation (uneven, noise, flipped normals), as the original's modes. Each shape: <= 2048 points, normals,
its 64 nearest neighbors, and its mesh (for the SDF targets during training).

Use with 04_train_new.py --data '<out>/datasets/<name>_train_mode*_*.npz'. Paths relative to the repository root.
"""

import argparse
import glob
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import points_as_tori as pat  # noqa: E402
from points_as_tori.training import mesh_data as md  # noqa: E402


def _checked(V, F):
	"""TriangleMesh of (V, F) with outward orientation, or raise ValueError naming the failed check."""
	shape = pat.TriangleMesh(V, F)
	if md.orientation_dot(shape.get_vertices(), shape.get_faces()) < 0.0:
		shape = pat.TriangleMesh(V, F[:, [0, 2, 1]].copy())
	checks = md.trimesh_checks(shape.get_vertices(), shape.get_faces())
	for ok, what in ((checks.is_watertight, 'not watertight'), (checks.is_volume, 'no volume'),
					 (checks.is_winding_consistent, 'inconsistent winding')):  # fmt: skip
		if not ok:
			raise ValueError(what)
	return shape


def prepare_mesh(path: str):
	"""(vertices (n, 3) float64, faces (m, 3) int32) ready for training, or raise ValueError with the reason."""
	import trimesh

	m = trimesh.load(path, force='mesh', process=False)
	shape = pat.TriangleMesh(np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64))
	shape.center_and_scale()
	V, F = md._vertices_numpy(shape), md._faces_numpy(shape)
	if F.shape[0] <= md.max_faces_per_mesh and V.shape[0] <= md.max_vertices_per_mesh:
		shape = _checked(V, F)
		return md._vertices_numpy(shape), md._faces_numpy(shape)
	# too large: quadric simplification (pyfqmr, as the original's blobs); it can leave non-manifold edges, so a few
	# face targets are tried and the first watertight result is kept
	import pyfqmr

	reasons = []
	for target in (md.max_faces_per_mesh, int(0.9 * md.max_faces_per_mesh), int(0.75 * md.max_faces_per_mesh)):
		s = pyfqmr.Simplify()
		s.setMesh(V, F)
		s.simplify_mesh(target_count=target, aggressiveness=7, preserve_border=True, verbose=0)
		Vs, Fs, _ = s.getMesh()
		try:
			shape = _checked(Vs, Fs)
		except ValueError as e:
			reasons.append(f'{Fs.shape[0]} faces: {e}')
			continue
		shape.center_and_scale()
		return md._vertices_numpy(shape), md._faces_numpy(shape)
	raise ValueError(f'{F.shape[0]} faces > {md.max_faces_per_mesh}; simplified: ' + '; '.join(reasons)
					 + ' (raise --max-faces, or simplify the mesh yourself)')  # fmt: skip


def main():
	ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
	ap.add_argument('meshes', nargs='*', help='mesh files (globs allowed)')
	ap.add_argument('--blobs', type=int, default=0, help='also generate this many procedural blobs')
	ap.add_argument('--name', default='mine', help='dataset root name (letters and digits, no "_")')
	ap.add_argument('--modes', type=int, nargs='+', default=[0], help='sampling modes (0, -1, -2, 1, 2, 3)')
	ap.add_argument('--split', type=float, default=1.0, help='train fraction of the meshes')
	ap.add_argument('--per-file', type=int, default=500, help='shapes per training file')
	ap.add_argument('--max-faces', type=int, default=md.max_faces_per_mesh,
					help='mesh size limit (the original: 2048; larger meshes train slower)')  # fmt: skip
	ap.add_argument('--points', type=int, default=md.max_points_per_pointcloud,
					help="points per shape (the original: 2048; training memory grows with it)")  # fmt: skip
	ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
	ap.add_argument('-o', '--out', default=os.path.join(REPO, 'examples', 'out', 'pat_data'))
	a = ap.parse_args()
	if '_' in a.name:
		ap.error('--name must not contain "_" (the original derives file names from the first "_" field)')
	pat.set_device(a.device, torch.float64)  # the original's C++ worked in double
	md.max_faces_per_mesh = md.max_vertices_per_mesh = a.max_faces
	md.max_points_per_pointcloud = a.points

	paths = [os.path.abspath(p) for pattern in a.meshes for p in sorted(glob.glob(pattern))]
	os.makedirs(os.path.join(a.out, 'datasets'), exist_ok=True)
	os.chdir(a.out)  # the original's functions write relative to the working directory ('datasets/...')

	mesh_files = []
	if paths:
		meshes = []
		for p in paths:
			try:
				V, F = prepare_mesh(p)
			except ValueError as e:
				print(f'  skip {p}: {e}')
				continue
			print(f'  {os.path.basename(p)}: {V.shape[0]} vertices, {F.shape[0]} faces')
			meshes.append({'vertices': V, 'faces': F, 'file_id': os.path.basename(p), 'metadata': {'path': p}})
		if not meshes:
			raise SystemExit('no usable mesh')
		md.save_train_test_meshes(meshes, 'datasets', a.name, a.split, seed=md.seed)
		mesh_files.append((f'datasets/{a.name}_train_meshes.npz', a.name))
	if a.blobs:
		md.generate_blob_dataset(n_blobs=a.blobs, save_dir='datasets', name=f'{a.name}blobs', train_test_split=1.0)
		mesh_files.append((f'datasets/{a.name}blobs_train_meshes.npz', f'{a.name}blobs'))

	for npz, root in mesh_files:
		for mode in a.modes:
			n = len(md.load_mesh_dataset(npz)[0])
			md.generate_scanned_mesh_training_dataset(
				n_views=md.n_cameras, npz_filepath=npz, save_filepath=f'datasets/{root}_train.npz',
				meshes_per_npz=a.per_file, mode=mode, n_files=-(-n // a.per_file),
			)  # fmt: skip
	written = sorted(glob.glob(os.path.join(a.out, 'datasets', '*_train_mode*_*.npz')))
	print('training files:\n  ' + '\n  '.join(written))
	print(f"next: python examples/pat/04_train_new.py --data '{a.out}/datasets/*_train_mode*_*.npz'")


if __name__ == '__main__':
	main()
