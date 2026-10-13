# sdfx
Comparing models that output SDFs. Submodules in this repo. `ext/points-as-tori` (PAT) and `ext/superfit`.  `git submodule update --init --recursive`.

1. PAT: [`points-as-tori`](https://github.com/nzfeng/points-as-tori)
   - submodules: nanobind (with robin_map), libigl, nanoflann and fcpw (which pulls eigen and slang-rhi)
2. SUP: [`superfit`](https://github.com/BardOfCodes/superfit)

PAT trains once and then predicts. SUP optimizes per shape

## Gradient optimization
| | PAT | SUP |
|---|---|---|
| SGD | **Trains offline once.** `training/train.py:252` uses `nnx.value_and_grad` with `optax.adamw` and a warmup-cosine schedule. | **Per shape on inference.** `fast_opt.py` and `mat_opt/optim.py` use `torch.optim.Adam`/`AdamW`, `.backward()` and `torch.compile`. |
| Learned params | Transformer [4,4,128]  blocks, heads, embedding. Maps k-NN patches to 6 coeffs + closed form torus fit. | Params per primitive, with residual fitting and pruning. |
| Cost per SDF | c++ with OpenMP and nanoflann on CPU, analytic grad d with `torus_distance.cpp`. |  Stepwise SDF optimization w backprop. |
| Custom autograd | - | `custom_ops/varaxis_sf_cuda.py`: CUDA `torch.autograd` . |
| Dependencies | flax nnx, optax, nanoflann, libigl, fcpw | cubvh, kaolin, geolipi, sysl (generates GLSL) |

## Licenses
SUP is mostly Adobe Research License: non-commercial use only, and derivative works must carry the same restriction.

# PAT port

Signed distance fields from point clouds.

- `points_as_tori/`: torch port of [Points as Tori](ext/points-as-tori) (inference and training). Mapping to the
  original, deviations and measurements: `points_as_tori/PORT_NOTES.md`.
- `ext/`: upstream forks (git submodules). Needed for the sample data and the parity tests, not to run the port.

## Install (conda + pip)

```bash
git clone --recurse-submodules https://github.com/xvdp/sdfx
cd sdfx
conda create -n sdfx python=3.12 -y
conda activate sdfx

# 1. torch: choose the build that matches your NVIDIA driver (check with `nvidia-smi`)
pip install torch  # PyPI default as of 20160915 CUDA 13.0 build (torch 2.14.1+cu130), needs driver >= 580
# pip install torch --index-url https://download.pytorch.org/whl/cu128    # for older drivers (CUDA 12.8 build)

# 2. this package (editable) + optional extras: train = matplotlib plot, test = pytest
pip install -e ".[train,test]"
```

Requirements (`pyproject.toml`): Python >= 3.9, torch >= 2.1, numpy, plyfile. Tested combinations:

| python | torch | numpy | result |
|---|---|---|---|
| 3.13.16 | 2.14.1+cu130 | 2.5.4 | all 41 tests pass (with the JAX/C++ references) |
| 3.12.3 | 2.14.1+cu130 | 2.5.4 | non-editable `pip install .`, inference + training import from outside the repo |
| 3.11 | 2.1.2+cu121 | 1.26.4 | 28 pass; 13 reference tests skipped (JAX 0.8.2 needs Python >= 3.11) |


## Run

```python
import torch
import points_as_tori as pat

pat.set_device('cuda')  # once; float32 default (set_device('cuda', torch.float64) for float64 geometry)
points, normals = pat.read_point_cloud('ext/points-as-tori/data/tooth.ply')  # (N, 3) NumPy
sdf = pat.PointsAsTori(points, normals)  # fits tori with the bundled pre-trained network

# query points, (3, n) layout: row 0 = x, row 1 = y, row 2 = z (here 100k uniform random points in [-1, 1)^3)
queries = torch.rand(3, 100_000, device=pat.get_device()) * 2 - 1
# or a regular grid (e.g. for marching cubes): 128^3 points
xs = torch.linspace(-1, 1, 128, device=pat.get_device())
grid = torch.stack(torch.meshgrid(xs, xs, xs, indexing='ij')).reshape(3, -1)
d_grid = sdf.signed_distance(grid).reshape(128, 128, 128)
d, g = sdf.signed_distance_and_gradient(queries)  # (n,), (3, n) on the device
d_np = pat.to_numpy(d)  # export to host
sdf.save_tori('tori.pkl')  # same file format as the original
```

## Train

Training reads the `.npz` datasets of the original (pre-generated: link in `ext/points-as-tori/README.md`, 6.2 GB; or
generated with the original `training/mesh_data.py`, which needs JAX, its C++ extension, thingi10k, pyvista, ...).

```bash
# the original five-phase schedule; run where datasets/ is, writes models/*.pt
python -m points_as_tori.training.train --device cuda --dataset-dir datasets/nice --models-dir models
```

```python
import points_as_tori as pat
from points_as_tori.training.train import TrainFundamentalFormPredictor

pat.set_device('cuda')
model = TrainFundamentalFormPredictor(
	['datasets/pretraining_blob.npz'],
	num_epochs=64,
	learning_rate=1e-3,
	batch_size=32,
	warmup_steps=0,
	n_epochs_to_save=1,
	n_shapes_to_save=1,
	final_save_filepath='FundamentalFormPredictor_pretrained',
	models_dir='models',
	val_data_filename='datasets/ABC17_validation_3.npz',
)
points, normals = pat.read_point_cloud('ext/points-as-tori/data/tooth.ply')
sdf = pat.PointsAsTori(points, normals, model_path='models/FundamentalFormPredictor_pretrained.pt')
```

# Viewer (sdfview)

`sdfview/`: scene format (`.sdfx`), LBVH built with torch, GLSL field/render modules (GL 3.3 and WebGL2 from one source),
moderngl renderer (window, headless, embeddable). Notes and measurements: `sdfview/NOTES.md`.
```bash
pip install -e ".[view]"
# build a scene from a point cloud (tori on the device, LBVH, optional grid bake for a fast surface)
python -m sdfview pat ext/points-as-tori/data/tooth.ply -o tooth.sdfx --device cuda --bake 128 [--view]

python -m sdfview view tooth.sdfx                    # window
python -m sdfview view tooth.sdfx --surface pat      # surface from the exact field (slower)
python -m sdfview render tooth.sdfx -o tooth.png --size 1200 900 --points --axis x --clip   # no window
python -m sdfview info tooth.sdfx

python benchmarks/lbvh_bench.py --device cuda --sizes 1e5 1e6 4e6      # please run this on your GPU
PAT_REFERENCE_DIR=build/pat_reference pytest tests/sdfview -s           # prints the measured differences
```
From torch

```python
import points_as_tori as pat, sdfview
from sdfview.sources.pat import scene_from_pat
pat.set_device('cuda')
sdf = pat.PointsAsTori(*pat.read_point_cloud('ext/points-as-tori/data/tooth.ply'))
scene = scene_from_pat(sdf, k=32, bake_resolution=128)
sdfview.show(scene)                      # or scene.save('tooth.sdfx')

# embedded: any current moderngl context
import moderngl
from sdfview import OrbitCamera, ViewSettings
from sdfview.backends.moderngl import Renderer
r = Renderer(moderngl.create_context(), scene)
r.draw(OrbitCamera.fit(r.box_min, r.box_max), ViewSettings(show_points=True), viewport=(0, 0, w, h))
```

## Tests

```bash
pytest tests                       # port-only tests; reference tests are skipped

# with the references (original JAX code + original C++ built as a Python module); needs Python >= 3.11, cmake and a C++17 compiler with OpenMP
pip install -e ".[reference]"
cmake -S tests/points_as_tori/reference -B build/pat_reference \
      -Dnanobind_DIR=$(python -m nanobind --cmake_dir) -DPython_EXECUTABLE=$(which python)
cmake --build build/pat_reference
PAT_REFERENCE_DIR=build/pat_reference pytest tests -s     # -s prints the measured differences
```


