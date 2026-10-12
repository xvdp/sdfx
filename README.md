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



