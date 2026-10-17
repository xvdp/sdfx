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

SuperFit is not installed: its code is read from the `ext/superfit` submodule (or `SUPERFIT_ROOT`, a directory holding
the `superfit/` package).

```bash
git submodule update --init ext/superfit
pip install -e ".[superfit]"            # evaluation dependencies only: geolipi, sysl3d, sympy, trimesh (MIT / BSD)

python -m sdfview superfit primitive_assembly.pkl -o assembly.sdfx --view [--color-parts]   # generated GLSL
python -m sdfview superfit primitive_assembly.pkl -o assembly.sdfx --bake 192 [--device cuda]  # + torch grid bake
python -m sdfview superfit primitive_assembly.pkl -o assembly.sdfx --mesh input.obj   # in the input mesh's frame
python -m sdfview superfit primitive_assembly.pkl --shadertoy assembly.glsl          # Shadertoy-style code
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
subpackages directly.

Fitting runs SuperFit's own script from the submodule, also without installing it:
`PYTHONPATH=ext/superfit python ext/superfit/scripts/mesh_to_assembly.py --input_path mesh.obj --save_dir out --fastmode`
(writes `out/final_content.pkl`; `examples/superfit/03_fit_superfit.sh mesh.obj` runs it and makes the scene). The
Python that runs it needs SuperFit's fitting dependencies (its `env.yml` without `-e .`, plus cubvh and kaolin built
from source as in its README) and a CUDA GPU. The input need not be watertight: SuperFit remeshes it (its signed
distance on a grid by cubvh ray stabbing, cleaned, marching cubes) and stops if that mesh is not watertight.

## The assembly as GLSL (`glsl.py`): composition and parts

`assembly_glsl(sdf)` writes the reduced assembly as GLSL functions, generated at run time:

```glsl
float assembly_part<i>(vec3 p)           // primitive i alone: its transforms from the root, then the primitive
float assemblyDistance(vec3 p)            // the assembly: transforms, primitives, nested smooth unions (bottom-up)
float assemblyPartDistance(int i, vec3 p) // part i alone
float assemblyParts(vec3 p, out int id)   // the parts as separate solids: min_i, id = argmin
int   assemblyNearestPart(vec3 p)         // to color the assembly by part
const int assembly_PART_COUNT;
```

with every parameter inlined (float32 literals), e.g.

```glsl
float assemblyDistance(vec3 p) {
    vec3 q = assembly_frame(p);
    vec3 p1 = Translate3D(q, vec3(0.200000003, 0.0, 0.0));
    vec3 p2 = AxisAngleRotate3D(p1, vec3(0.0, 0.300000012, 0.0));
    float d3 = SuperFrustumY(p2, vec3(0.600000024, 0.400000006, 0.5), 0.300000012, 0.0, 1.0, 0.100000001, 0.0);
    ...
    float d7 = sdfxSmoothUnion(d3, d6, 0.05);
    ...
}
```

- **Sources of the code.** The primitives' functions (`SuperFrustum{X,Y,Z}`, `SuperFrustum`, `SPTaperedOnion`, ...)
  are SuperFit's (`superfit/shader`, Adobe Research License): importing `superfit.shader` registers them in sysl's
  shader-module registry (`sysl.shader.shader_module.SMMap`), and `glsl.py` reads them from there with their
  dependencies and constants. They are copied into the generated text (scenes, `--shadertoy` files), not into this
  repository; that text is a derivative of SuperFit under its license. The transforms come from sysl (MIT); the
  composition and the smooth union are written in `glsl.py` (MIT).
- **Semantics** as SuperFit's torch evaluation (geolipi): `Translate3D(e, t)` = e(p − t); `AxisAngleRotate3D(e, a)` =
  e(p R(a)) (the zero rotation is dropped: the registry's GLSL would normalize a zero vector, geolipi clamps the angle
  to 1e-8, i.e. the identity); smooth union h = clamp(½ + (b − a)/(2(k + 1e-9)), 0, 1), d = b(1 − h) + ah − kh(1 − h);
  `Union` min, `Intersection` max, `Difference` max(a, −b). Other node types raise `NotImplementedError`.
- **Validation** (`tests/superfit/test_glsl.py`, Mesa llvmpipe): the assembly, each part, all parts and the mesh frame
  against SuperFit's torch evaluation at 4000 points: max |GLSL − torch| = 2.4e-6 (float32 shader, float64 torch);
  the generated code inside `surface.frag` compiles as GLSL 3.30 and as GLSL ES 3.00 (WebGL2, glslangValidator); the
  Shadertoy export compiles and renders.
- **Viewer.** Scene field type `glsl` (`sdfview/scene.py`): the code is stored in the scene (JSON), the viewer inserts it
  before its field module. Keys `[` / `]` cycle whole → all parts → part 0, 1, ...; `U` colors by part (`--part`,
  `--color-parts`). As the field is exact and cheap (one call per primitive), surfaces trace it rather than a bake.
- **Shadertoy.** `assembly_shadertoy(sdf)` (`--shadertoy FILE`): the functions plus a `mainImage` sphere tracer with
  `iResolution`, `iTime`, `iMouse`; `#define PART` / `COLOR_PARTS` select what is drawn. Paste into shadertoy.com, or
  load with `examples/hosts/shadertoy_moderngl.py`.
- **Upstream.** sysl's `AxisAngleRotate3D` GLSL normalizes the axis-angle vector, undefined for a zero rotation
  (NaN on most drivers); SuperFit's HTML export goes through it.

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

## Comparison with Points as Tori (`experiments/compare_pat_sup.py`)

Both pipelines fit the same meshes and the fits go into one scene per mesh, `<out>/<shape>.sdfx`, to be toggled or
overlaid in the viewer (`python -m sdfview view <shape>.sdfx`: G picks the opaque model, V overlays the others, T makes
them translucent). Everything is in SuperFit's normalized frame. Models:

- `pat`: PointsAsTori on an oriented point cloud sampled from the normalized mesh (uniform by area, face normals,
  `--pat-points`, optional `--pat-noise`); native field (tori + BVH), grid bake and the input points;
- `sup`: a SuperFit assembly fitted beforehand (`--suppkl`, paired in order with `--mesh`; alone, the mesh path
  stored in the assembly is used), baked on a grid; a mesh without one gets PAT and `gt` and the commands to fit
  SuperFit to it (`examples/superfit/03_fit_superfit.sh`);
- `gt`: the mesh's signed distance baked on a grid (cubvh 'raystab' as SuperFit, or the generalized winding number;
  `--gt none` to leave it out).

Fit times and counts are in the scene meta (`python -m sdfview info`). `--metrics` adds IoU@R, surface IoU,
Chamfer@2048, SDF error and normal agreement with SuperFit's definitions where they apply (`<out>/metrics.jsonl`);
validated on knot.obj (ground truth against itself: IoU 1, SDF error 0, Chamfer floor 0.11 from the 2048-point
sampling). The two inputs differ (PAT: an oriented point cloud; SuperFit: a watertight mesh, which it remeshes), so
the metrics describe each fit against the mesh, not one method under the other's conditions.

## Not done yet

- Primitive families other than the SuperFrustum ones are generated only if their sysl module is a plain code module
  (template-generated modules raise `NotImplementedError`); tested here: SuperFrustum X / Y / Z.
