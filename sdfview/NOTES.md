# sdfview — notes

Viewer for signed distance fields produced with torch. Field types: Points as Tori (`pat`: points, tori, BVH), any SDF
sampled on a grid (`grid`), and generated GLSL (`glsl`: a SuperFit assembly as the composition of its primitives).

## Layout

```
sdfview/
  scene.py                 file format (.sdfx): JSON manifest + binary arrays, Python reader/writer
  camera.py, settings.py   orbit camera and view settings (NumPy / plain data, backend independent)
  view.py                  field choice per model, surface passes, model colors, scene box (all Python hosts)
  png.py                   minimal PNG writer
  accel/lbvh.py            re-export of the LBVH builder in points_as_tori/bvh.py (port of the original bvh.cpp)
  sources/pat.py           PointsAsTori -> scene (points, tori, BVH, point cloud, optional grid bake)
  sources/grid.py          any torch SDF -> regular grid, in z-slabs
  sources/superfit.py      SuperFit primitive assembly (superfit_sdf) -> generated GLSL field (+ optional grid bake)
  shaders/__init__.py      assembly: GLSL (version header per target gl330 | webgl2 | three, #include), WGSL
                           (preprocessor: include, ifdef / if defined / elif / else, valued defines as consts)
  shaders/glsl/            modules: common, bvh, field_pat, field_grid, field, camera, colormap
                           programs: fullscreen.vert, surface.frag, cutplane.{vert,frag}, points.{vert,frag},
                           probe.frag, probe_knn.frag, blit.frag
  shaders/wgsl/            the same modules in WGSL; programs surface.wgsl, probe.wgsl (compute)
  shaders/slang/           field (interface IField), field_pat (PatField), field_grid (GridField), camera, surface,
                           probe; common and bvh are points_as_tori/slang/ (shared with the fused k-NN kernel)
  backends/moderngl/       Renderer (any moderngl context), headless (EGL), window (moderngl-window)
  backends/slangpy/        Renderer: the Slang modules as compute kernels (render, probe), surfaces only
  web/scene.js             scene reader for the browser / Node
  web/core/                shaders.js (the Python assembly in JS, identical output), camera.js, view.js, app.js
  web/webgl2/, web/webgpu/ raw API hosts (renderer.js + index.html)
  web/three/               three.js hosts: webgl_renderer.js (GLSL in RawShaderMaterials, sdfMesh for three.js
                           scenes), webgpu_renderer.js (TSL); pages webgl.html, webgpu.html, compose.html
  web/test/                probe.html + run_probe.mjs: headless Chromium runner of tests/sdfview/test_hosts.py
  server.py                static server for the web hosts (python -m sdfview serve)
  __main__.py              CLI: pat | superfit | view | render (--backend moderngl | slangpy) | merge | serve | info
```

The hosts side by side (steps, names, layout, deviations, measured parity): `PARALLEL.md`.

## Scene format

Specified in `scene.py` (docstring). One file: `SDFX`, version, JSON length, binary length, JSON (padded to 16 B),
binary chunk (each array 16 B aligned). Arrays are stored in the layout the shaders fetch (one texel per row of
4 values), so backends upload them without reshuffling; the torch side is structure-of-arrays (3, n) and converts
once at export. Field types: `pat` (points, tori, BVH), `grid` (vertex-aligned samples) and `glsl` (generated code
stored in the manifest, no arrays; the viewer inserts it before `field.glsl`, compiled with `FIELD_GLSL`: entry points
`assemblyDistance`, `assemblyPartDistance`, `assemblyParts`, `assemblyNearestPart`, uniform `field_part` = −1 whole,
−2 all parts, i part i; see superfit_sdf/NOTES.md).

## LBVH (points_as_tori/bvh.py, re-exported as sdfview.accel.lbvh)

Morton codes (21 bits per axis, 63-bit keys) → sort → Karras 2012 radix tree (per internal node: direction, range
end and split by binary search over common-prefix lengths of (code, index) keys) → leaf collapse (ranges ≤ 4
points become leaves, as the original's leaf size) → depth-first pre-order by sorting (range start ↑, size ↓) →
boxes by a sparse table over the sorted points, one level at a time → depth by pointer jumping.

The prefix comparison δ(i, j) > δ(i, r) is evaluated without counting leading zeros:
with x = k_i ⊕ k_r, it holds iff (k_i ⊕ k_j) >> (bitlen(x) - 1) = 0 (keys differ), or k_j = k_i and
(i ⊕ j) >> (bitlen(i ⊕ r) - 1) = 0 (keys equal). Checked equal to the δ-based form in the tests.

Host syncs: two (number of kept nodes; node counts per sparse-table level). Everything else is queued work on the
device.

Measurements (CPU, 4 threads, torch 2.14.1; points on the unit sphere; `benchmarks/lbvh_bench.py`):

| N | LBVH build (torch) | peak memory | C++ SAH build | nodes LBVH / SAH | depth LBVH / SAH | k-NN (k=32) nodes / points per query, LBVH vs SAH | torch brute-force k-NN, 10k queries |
|---|---|---|---|---|---|---|---|
| 10⁴ | 0.039 s | (below resolution) | 0.115 s | 7,477 / 7,513 | 18 / 32 | 80 / 64 vs 124 / 97 | 0.29 s |
| 10⁵ | 0.128 s | 41 MB | 1.40 s | 74,881 / 75,731 | 23 / 33 | 89 / 65 vs 124 / 95 | 2.33 s |
| 10⁶ | 1.91 s | 420 MB | 19.9 s | 751,811 / 754,313 | 29 / 46 | 112 / 77 vs 188 / 139 | 59.1 s |

Peak memory on CPU is the growth of the process high-water mark (VmHWM) during the build: ~420 bytes per point,
an upper estimate (allocator caching included). On CUDA the benchmark reports `torch.cuda.max_memory_allocated`.
The largest consumers are the int64 index arrays over the 2N - 1 radix-tree nodes and the binary-search
temporaries; int32 indices would roughly halve it (not done: int32 advanced indexing is not uniform across the
supported torch versions).

The traversal counts are for the shader's algorithm (nearer child first, prune by the current k-th distance); the
SAH cost model targets ray casting, the Morton order groups points by proximity, which is what k-NN pruning uses.
GPU timings: not measured here (no GPU in the container); `--device cuda` in the benchmark.

## Field evaluation in GLSL (shaders/glsl/field_pat.glsl)

Same quantities as the torch evaluator in k-NN mode (`TorusDistanceField._evaluate_chunk`, `_blend`):
φ = Σ w_i g_i / Σ w_i, w_i = exp(-λ(|q - p_i| - shift)), shift = ½ max_i |q - p_i|, λ = 64 λ_scale / shift,
gradient with λ and shift frozen. Mapping to the original demo shader (`ext/points-as-tori/demo/shaders_3d.py`):

| sdfview | original |
|---|---|
| `fetch_texel(tex, i)` (width from `textureSize`) | `fetch_texel(i, tex)` with a `texture_size` uniform |
| `fetchBVHNode` (2 float texels + 1 int texel) | `fetchBVHNode` (3 float texels, integers as floats) |
| `bvhKNearestNeighbors` (k ≤ `BVH_K_MAX`, stack guarded) | `bvhKNearestNeighbors` (stack[64], unguarded) |
| `accumulateTorusPoint`, `TorusAccumulator` | same names; tori texels (center, R), (unit axis, r) instead of (center), (θ, φ, R, r) |
| `patDistanceGradient` (k-NN branch) | `blendedTorusDistanceGradient` (radius branch with k-NN fallback; brute force when k ≤ 0) |
| `colormap_SDF`, `darken_color`, isobands | same; plus a zero-contour line |

Measured parity (tests/sdfview/test_gl.py, Mesa llvmpipe, float32 shaders; printed with `pytest -s`):
GLSL vs torch float32 / float64 distance max |Δφ| ≈ 1e-7–1e-6 (|φ| up to ~1), gradient max |Δ∇φ| ≈ 1e-4
(the λ = 64/shift factor amplifies rounding in the gradient); k-NN sets identical to brute force; GLSL trilinear vs
`torch.nn.functional.grid_sample(align_corners=True)` ≈ 1e-7 on llvmpipe (GPU texture units use fixed-point filter
weights, typically 8 bits: errors up to ~1/256 of a cell's value range); sphere-traced silhouette of |x| - 1 vs
the analytic unit sphere: IoU > 0.97 at 96×72.

## Several models in one scene (toggle / overlay)

`sdfview.scene.merge_scenes({'pat': a, 'sup': b, 'gt': c})` (CLI `python -m sdfview merge a.sdfx b.sdfx c.sdfx -o
abc.sdfx --names pat sup gt`) puts several scenes in one: arrays renamed `<model>/<name>`, fields and point clouds
tagged `"model"` (an optional key; older readers ignore it), bounds = union. The renderer then:

- **single** (default): draws `surface_field` only, in its model's color (pat tan, sup blue, gt grey, others from a
  palette; `ViewSettings.model_colors` overrides); cycling the surface field (key G) toggles between models;
- **overlay** (key V, `--overlay`): draws one field per model (its grid if it has one, else its exact field), the
  selected model opaque; with opacity 1 every surface writes depth and the nearest one shows per pixel (where one model
  lies outside the other, its color appears); with `overlay_opacity` < 1 (key T, `--opacity`) the other models are
  drawn last, blended, without writing depth, so a surface inside another stays visible.

Coincident surfaces in an opaque overlay alternate pixel by pixel (depth fighting); the translucent mode shows them
without it.

Field choice per model (`view.model_field`, `web/core/view.js`; 'auto' = the first model's): surfaces take the cheapest field to
trace (`glsl`, then `grid`, then `pat`), the cut plane the exact one (`glsl` / `pat`, then `grid`).

Fields with parts (`glsl`): keys `[` / `]` cycle the whole assembly → all parts as separate solids → part 0, 1, ...
(`ViewSettings.part`, `--part`); `U` colors the surface by part (`color_parts`, `--color-parts`; golden-ratio hues).

## Rendering

- Surface: sphere tracing t ← t + κ|φ| (κ = 0.9), hit at |φ| < ε (1e-4 of the box diagonal) or at a sign change,
  refined by 10 bisection steps; shading from ∇φ (analytic for PAT, central differences for the grid);
  `gl_FragDepth` written so points and the cut plane composite by depth.
- Cut plane: a rectangle rasterized through the box, field evaluated per fragment; colormap of the original demo.
- Points: sprites read from the points texture by `gl_VertexID` (no vertex buffers).
- Cost: the exact PAT field costs one BVH k-NN search per evaluation; sphere tracing needs tens of evaluations per
  pixel. The grid bake (torch, `--bake N`) gives one texture fetch per evaluation; by default the surface uses the
  bake and the cut plane uses the exact field. The window renders only on change, at reduced resolution while
  dragging.

## Not done yet

- Cut plane and points in the web and slangpy hosts; PAT in TSL (three.js WebGPURenderer); the SuperFit "glsl"
  field in WGSL / Slang (the hosts without it draw its grid bake).
- Radius-mode evaluation and point areas in the shader (the original's `bvhAccumulateInRadius`, `usePointAreas`).
- Tori, normals and ground-truth mesh overlays of the original demo; GUI widgets (keyboard only).
