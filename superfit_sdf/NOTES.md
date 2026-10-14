# superfit_sdf — SuperFit assemblies as torch SDFs

SuperFit ("SUP", `ext/superfit`, CVPR 2026) fits compact assemblies of SuperFrusta and other primitives to a
watertight mesh, per shape, by gradient descent in PyTorch. This package evaluates a fitted assembly as a signed
distance function `φ: (3, Q) tensor -> (Q,) tensor`, so it can be viewed (`sdfview`) and compared with Points as Tori.

## Port or wrap: wrap

| | Points as Tori | SuperFit |
|---|---|---|
| Original framework | JAX + C++ (nanobind) | PyTorch 2.9 (+ torch.compile, optional CUDA op) |
| Port to torch needed | yes (done: `points_as_tori/`) | no: already torch |
| License | permissive | Adobe Research License: non-commercial research only; derivative works carry the same restriction (MIT island: custom CUDA op, visualization) |

SuperFit needs no port. Reimplementing its primitive SDFs here would make that part of this MIT repository a
derivative work under the Adobe license, so this package **imports SuperFit at run time and copies none of its code**.
Using it runs SuperFit code, under SuperFit's license. geolipi and sysl3d (its expression and shader libraries) are MIT.

## Install and use

```bash
pip install -e ".[superfit]"            # geolipi, sysl3d, sympy, trimesh (MIT / BSD)
git submodule update --init ext/superfit   # SuperFit's code (or set SUPERFIT_ROOT to an installed superfit/ package)

python -m sdfview superfit primitive_assembly.pkl -o assembly.sdfx --bake 192 [--device cuda] [--view]
python -m sdfview superfit primitive_assembly.pkl -o assembly.sdfx --mesh input.obj   # in the input mesh's frame
```

```python
from superfit_sdf import SuperFitSDF

sdf = SuperFitSDF('primitive_assembly.pkl', program='best', device='cuda', dtype=torch.float32)
phi = sdf(queries)  # (3, Q) -> (Q,), SuperFit's normalized frame (or the mesh frame with mesh=...)
sdf.n_primitives, sdf.primitive_types, sdf.metrics()
```

Evaluation needs only SuperFit's `symbolic` and `torch_compute` subpackages, geolipi and sysl. SuperFit's
`superfit/__init__.py` imports its whole fitting stack (cubvh, kaolin, fastsweep, drjit, cc3d, igl); when `superfit`
is not already imported, `import_superfit` registers the package without running that `__init__` and imports the two
subpackages directly. Fitting new assemblies needs SuperFit's own installation (its README: conda env, cubvh,
kaolin) and runs on its scripts (`scripts/mesh_to_assembly.py`, `scripts/testset_fit_primitives.py`).

## Assembly file

`primitive_assembly.pkl` (SuperFit's `resfit` → `Stats.get_dict()` → `pickle.dump`): a flat dict with dotted keys —
`input_mesh_file`, `n_iters`, `success`, `total_time`, `evaluation.evaluation.<metric>` (e.g. `iou@128`,
`CD_AVG@2048`), and per iteration `iter_{k}.{init,opt,pruned,best,running}_program` plus counts and timings. A
program is a geolipi state `{'expr_str', 'symbol_tensor_map', 'GLFunction': True}`; `expr_str` is a Python
expression with the parameters inlined:

```
SmoothUnion(SmoothUnion(P1, P2, (k1,)), P3, (k2,))
P = PrimitiveMarker(StochasticPrimitive(Translate3D(AxisAngleRotate3D(VarAxisSF(size, (roundness,), (dilate,),
        (taper,), (bulge,), (onion,), axis_logits), axis_angle), t), (keep_logit, drop_logit)))
```

or, after pruning, `PrimitiveMarker(Translate3D(AxisAngleRotate3D(SuperFrustumX|Y|Z(...), axis_angle), t))`.

- **Program selection** (`select_program`): `'best'` = SuperFit's `get_best_expr` (`iter_{n_iters-1}.best_program`,
  else `.pruned_program`); `'latest'` = `best_program` of the highest recorded iteration; or any key. After an early
  stop SuperFit records iteration `n_iters`, which `get_best_expr` does not read; `best_program` is the best so far.
- **Loading** (`load_assembly`): pickles can run code; the file is read with an allow-list unpickler (builtin
  containers and scalars, torch tensors / storages, NumPy arrays / scalars). Tensor storages pickle as bytes that
  torch reloads with `torch.load(weights_only=False)`; here they are reloaded with `weights_only=True`. Anything else
  raises `pickle.UnpicklingError` (tested with a direct and a nested payload). Textured pickles (`*_textured.pkl`,
  `material_expr` with sympy symbols) are not supported.

## Evaluation (as SuperFit's deterministic path)

`fetch_singular_expr_eval(expr, relaxed_eval=False)` then geolipi's `recursive_evaluate`:

- a primitive is kept iff `argmax(keep, drop) = keep`; dropped primitives leave the union;
- a variable-axis primitive becomes its X / Y / Z variant by `argmax(axis_logits)`;
- local coordinates `q = R(a) (p − t)`, `R(a)` the Rodrigues rotation of the axis-angle `a`;
- smooth union (geolipi `sdf_smooth_union`), nested left to right:
  `h = clamp(1/2 + (b − a) / (2 (k + 1e-9)), 0, 1)`, `d = b (1 − h) + a h − k h (1 − h)`;
  k ∈ (1e-5, 0.4999) after optimization (initialized at 0.01).

SuperFit's reported metrics (`testset_eval.py`, `mesh_to_assembly.py`) instead take the best of ~100 Gumbel samples
of the keep/drop and axis choices, so the argmax assembly can differ from the one those metrics describe.

**Frames.** SuperFit centres the input mesh on its bounding box and scales it into [−0.9, 0.9]³,
`p_n = s (p − c)`, `s = 1.8 / max extent`, `c` = bounding-box centre, and evaluates over [−1, 1]³. The transform is
not saved; with the input mesh, `φ(p) = φ_n(s (p − c)) / s` (`SuperFitSDF(..., mesh=...)`; trimesh reads the mesh
as SuperFit does, `force='mesh', process=False`).

**Distance quality.** The fitted field is an SDF only approximately: SuperFrustum interior values are clipped at the
medial axis (sign correct, magnitude not a distance), and the smooth union lowers values near junctions by up to
k/4. Sphere tracing from outside is unaffected by the interior; the viewer traces the baked grid.

## Issues found in SuperFit (upstream, HEAD 42b2633)

1. **bulge = 0 in the torch SuperFrustum**: `map_arc_bulge` multiplies by `sign(bulge)`, so for bulge exactly 0 the
   local x coordinate becomes 0 (e.g. φ(5, 0, 0) = 0); the GLSL version skips the map when bulge = 0.
   `SuperFitSDF` warns when a primitive has bulge exactly 0. `SolidSF` forces bulge to 0.
2. **`varaxis_sg_eval` ignores its permutation**: it computes permuted coordinates and sizes and then evaluates the
   unpermuted ones (`torch_compute/super_geon.py`); the reduced X / Y / Z SuperGeon variants are correct.
3. **`mesh_to_assembly.py` key name**: it stores the input path as `input_path` and saves `final_content.pkl`, while
   `fit_texture.py` and `testset_eval.py` read `input_mesh_file`.
4. **`get_best_expr` reads iteration `n_iters − 1`**: after an early stop the last recorded iteration is `n_iters`.

## Validation (tests/superfit, CPU)

Assemblies built with SuperFit's own classes and written in its pickle layout: program selection (`best` / `latest`),
dropped primitives and axis choice, values bit-identical to SuperFit's evaluation called directly (float64),
chunked evaluation, signs at primitive centres and far points, the mesh-frame mapping (to 1e-12), the unsafe-pickle
rejection, the bulge = 0 warning, and the grid bake into an `.sdfx` scene. Not done here: real SuperFit fits
(need its CUDA stack) and the released assemblies (Hugging Face, blocked in this environment).

## Not done yet

- Native GLSL for the assembly in sdfview (SuperFit generates it through sysl; its shader modules are Adobe-licensed,
  so they would be generated at run time from the user's SuperFit checkout rather than stored here).
- Comparison tooling (Points as Tori vs SuperFit on the same meshes: IoU, Chamfer, normal consistency, timing),
  ideally with SuperFit's metric definitions (`algos/eval_tools.py`) for comparability.
