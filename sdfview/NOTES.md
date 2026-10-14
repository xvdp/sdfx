# sdfview — notes

Viewer for signed distance fields produced with torch. First field type: Points as Tori (`points_as_tori`);
second: any SDF sampled on a grid (the route for SUP and for comparisons).

## Layout

```
sdfview/
  scene.py                 file format (.sdfx): JSON manifest + binary arrays, Python reader/writer
  camera.py, settings.py   orbit camera and view settings (NumPy / plain data, backend independent)
  accel/lbvh.py            re-export of the LBVH builder in points_as_tori/bvh.py (port of the original bvh.cpp)
  sources/pat.py           PointsAsTori -> scene (points, tori, BVH, point cloud, optional grid bake)
  sources/grid.py          any torch SDF -> regular grid, in z-slabs
  sources/superfit.py      SuperFit primitive assembly (superfit_sdf) -> grid-baked scene
  shaders/__init__.py      GLSL assembly: version header per target (gl330 | webgl2) + #include resolution
  shaders/glsl/            modules: common, bvh, field_pat, field_grid, field, camera, colormap
                           programs: fullscreen.vert, surface.frag, cutplane.{vert,frag}, points.{vert,frag},
                           probe.frag, probe_knn.frag, blit.frag
  backends/moderngl/       Renderer (any moderngl context), headless (EGL), window (moderngl-window)
  web/scene.js             scene reader for the browser / Node
  __main__.py              CLI: pat | superfit | view | render | info
```

Second pass (planned, same module names so files compare one to one):
`shaders/slang/*.slang` (didactic parallel implementation; the BVH module exists as `points_as_tori/slang/bvh.slang`,
a function-by-function mirror of `shaders/glsl/bvh.glsl` run as a torch k-NN kernel), `shaders/wgsl/*.wgsl` (WebGPU),
`web/webgl2/`, `web/webgpu/`, `web/three/` (three.js adapter: a ShaderMaterial on a full-screen quad / plane fed
with the same textures).

## Scene format

Specified in `scene.py` (docstring). One file: `SDFX`, version, JSON length, binary length, JSON (padded to 16 B),
binary chunk (each array 16 B aligned). Arrays are stored in the layout the shaders fetch (one texel per row of
4 values), so backends upload them without reshuffling; the torch side is structure-of-arrays (3, n) and converts
once at export. Field types: `pat` (points, tori, BVH) and `grid` (vertex-aligned samples).

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

- WebGL2 / WebGPU / three.js front ends (`web/` has the scene reader only; the GLSL compiles as ES 3.00).
- Slang mirrors of the render modules (common and bvh exist in `points_as_tori/slang/`, with tensors in place of
  textures); WGSL mirrors.
- Radius-mode evaluation and point areas in the shader (the original's `bvhAccumulateInRadius`, `usePointAreas`).
- Tori, normals and ground-truth mesh overlays of the original demo; GUI widgets (keyboard only).
