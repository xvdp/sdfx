# points_as_tori: torch port of Points as Tori (PAT)

Source: `ext/points-as-tori` (JAX/flax.nnx network + C++/nanobind evaluator), MIT license.

## File mapping

| port (`points_as_tori/`) | original | content |
|---|---|---|
| `config.py` | — | device/dtype set once (`set_device`), import (`as_wide`) / export (`to_numpy`) boundaries, chunk size |
| `shape_2d.py` | `src/cpp/shape_2d.h/.cpp` | `sgn`, `sign`, `orthonormal_basis(_random)`, `sample_sphere_positions`, `subsample_point_cloud` (+ SoA `cross`/`dot`/`norm`); `SDF2D` and its 15 subclasses `Circle2D` … `Quad2D`: distances, analytic/autodiff gradients, Laplacians, bounding boxes, revolutions/extrusions and their 3D gradients, 2D/3D ray intersections, all samplers |
| `shape_3d.py` (section "2D SDFS") | `src/pointsastori/shape_3d.py` | Python wrappers `SDF2D` + subclasses (`bound_object` = the `shape_2d` class), `load_SDF`, `sample_SDF`, `SDF3D`, `SDF_SHAPES` |
| `torus_distance.py` | `src/cpp/torus_distance.h/.cpp` | `TorusDistanceField` (evaluation, log-sum-exp variants, winding numbers, point areas, isosurface sampling), `get_neighbors`, `get_average_neighbor_distance`, torus fitting, torus SDF/gradient |
| `bvh.py` | `src/cpp/bvh.h/.cpp`, `BoundingVolumeHierarchy` of `src/pointsastori/shape_3d.py` | `BVH` over points or triangles (LBVH build in torch, batched queries), `BoundingVolumeHierarchy` |
| `knn_slang.py`, `slang/common.slang`, `slang/bvh.slang`, `slang/knn.slang` | `bvhKNearestNeighbors` of `demo/shaders_3d.py` (via `sdfview/shaders/glsl/bvh.glsl`) | `SlangKNN`: fused k-NN kernel in Slang, one thread per query, called through slangpy (the `'slang'` k-NN backend) |
| `igl.py` | libigl `copyleft/cgal/point_areas`, `octree`, `fast_winding_number` (`deps/libigl`) | point areas, octree, Barnes-Hut winding number |
| `isosurface.py` | `mesh_isocontour` of `demo/demo_3d.py` (PyVista) | marching cubes in torch, `mesh_isocontour` |
| `cpp_random.py` | `std::mt19937` + libstdc++ distributions | bit-exact replay of the C++ random draws |
| `shape_3d.py` | `src/pointsastori/shape_3d.py` | Python API layer: point-cloud I/O, PLY export, `get_neighbors`, `compute_optimal_radius`, `fit_tori_from_forms`, `TorusDistanceField` |
| `network.py` | `src/pointsastori/network.py` | helpers, training-side torus math (`torus_precompute_masked`, `torus_distance_gradient_masked`), export, transformer |
| `infer.py` | `src/pointsastori/infer.py` | `PointsAsTori`, `read_point_cloud` |
| `convert_checkpoint.py` | — | flax `.pkl` -> torch `.pt` |
| `training/train.py` | `training/train.py` | losses, schedule, train step, epoch loop, validation, checkpoints, the five-phase `train()` |
| `training/mesh_sampling.py` | `training/mesh_sampling_jax.py` | on-the-fly query sampling and ground-truth signed distances |
| `shape_3d.py` (`TriangleMesh`, `sample_bounding_box`) | `src/cpp/shape_3d.h/.cpp` (`Mesh3D`) + `src/pointsastori/shape_3d.py` (`TriangleMesh`) | mesh I/O, signed distance, winding numbers, ray casting, surface samplers, simulated scans |
| `training/mesh_data.py` | `training/mesh_data.py` | mesh datasets, Perlin blobs, scan simulation, padded training `.npz` |
| `models/FundamentalFormPredictor.pt` | `src/pointsastori/models/FundamentalFormPredictor.pkl` | converted pre-trained weights (embed 128, 8 heads, 8 layers, mlp 512, k = 64; 1,588,358 parameters) |

Function names are unchanged, including the `_jax` suffixes.

## Conventions

**Device.** `points_as_tori.set_device('cuda')` once; until it is called, `get_device()` raises, and calling it again with a different device also raises (unless `force=True`). Nothing moves tensors implicitly: inputs on another device are rejected.
- Import: `as_wide`, and the `PointsAsTori(points, normals, tori=...)` constructor, which takes the original (n, 3) layout.
- Export: `to_numpy`, `save_tori`, `export_tori*`, `write_point_cloud`, `PointCloud3D.export`.

**No host sync** between import and export on the inference path (network, torus fit, evaluation with the brute-force k-NN). That code contains no `.item()`, `.cpu()`, `nonzero`, boolean-mask indexing or branching on tensor values. Data-dependent choices are made per element with `torch.where`, for example the radius-mode k-NN fallback, outliers and the min/max curvature selection. This is verified by running the whole pipeline on the `meta` device (`tests/points_as_tori/test_device.py`). On CUDA it can also be checked with `torch.cuda.set_sync_debug_mode('error')`.

Host reads that remain, all of a size or a termination flag: the BVH build (2 reads) and traversal (one "any query still active?" read every 8 steps), marching cubes (number of crossed cubes), the octree (one per level), the fast-winding-number traversal (one per level), sphere tracing in `ray_intersections` (one per step), point areas (the count of near-degenerate cases, decided exactly on the host), and samplers with variable-size output. The C++ random draws are replayed on the host (`cpp_random.py`) and moved to the device once.

**Layout (wide arrays, structure-of-arrays).**
- Vectors are C-contiguous `(3, n)`, so `x = a[0]` is a contiguous length-n row; axes are `(2, n)`, coefficients `(6, n)`. These are the C++ Eigen shapes, stored row-major.
- Per-pair quantities are `(Q, K)` and per-pair vectors `(3, Q, K)`; reductions run over the last axis.
- Inside the transformer, activations use the standard `(batch, k, features)` layout.
- The coefficient head writes `(6, batch)` directly as `W xᵀ + b`.

**dtype.** Geometry runs in the dtype passed to `set_device` (the C++ used float64). The network always runs in float32, as JAX did with x64 disabled.

## Mathematics (as implemented, unchanged from the original)

Per query q with neighbors N(q), oriented points p_i, and tori (c_i, a_i, R_i, r_i):

- **Torus SDF:** g_i(q) = s_i (√(d₁² + d₂²) − |r_i|), where d₁ = |(q − c_i) × a_i| − R_i, d₂ = (q − c_i)·a_i and s_i = sign(r_i).
- **Blend:** φ(q) = Σ w_i g_i / Σ w_i, where w_i = exp(−λ(|q − p_i| − shift)), shift = ½ max_j |q − p_j| and λ = 64/shift · λ_scale.
- **Gradient (returned):** ∇φ = Σ w_i(∇g_i − λ r̂_i g_i)/W + φ λ Σ w_i r̂_i / W, with **λ held fixed**.
  - Because λ depends on q, the exact gradient adds (∂φ/∂λ)∇λ, where ∂φ/∂λ = −Cov_w(|q−p|, g) and ∇λ = −(λ/shift)·½ r̂_far.
  - On a random 300-torus configuration (all-point mode), the frozen-λ gradient differs from central differences by 1.3e-2 (max over components, relative to max |∇φ|). With the λ term added, the difference is 4.9e-10.
  - The port returns the original's frozen-λ gradient; `test_gradient_is_the_frozen_lambda_gradient` documents the decomposition.
- **Torus fit from the fundamental-form coefficients (a00, a01, a10, a11, a02, a20):**
  - I = [[1+a10², a10 a01], [a10 a01, 1+a01²]] and II = [[2a20, a11], [a11, 2a02]]/√(1+a01²+a10²).
  - The principal curvatures are Rayleigh quotients of the closed-form eigenvectors.
  - The torus is r = 1/|κ_max| and R = 1/|κ_min| − sgn(κ_max κ_min) r, with center p + a00 n − (s/|κ_min|) n* and axis n* × v_min.

## Deviations from the original

1. **Nearest neighbors:** the nanoflann kd-tree is replaced by one of three exact searches, `config.KNN_BACKEND`:
   - `'brute'`: one GEMM for squared distances, then `topk` per query chunk; O(|Q||P|).
   - `'bvh'`: the BVH of `bvh.py`, traversed by all queries at once; O(|Q| · visited nodes).
   - `'slang'`: the same BVH traversed by a fused Slang kernel (`knn_slang.py`), one GPU thread per query; needs slangpy. Neighbors selected with float32 distances.
   - `'auto'` (default): `'bvh'` from `config.KNN_BVH_MIN_POINTS` = 32768 points on (a CPU crossover; see "k-NN backends").
   Distances that enter the SDF are recomputed exactly from the gathered differences. `'brute'` and `'bvh'` return the C++ neighbors (tested) and give bitwise-identical evaluator output; `'slang'` returns the exact neighbors up to float32 near-ties (tested against float64 k-NN distances) and evaluator output equal to `'bvh'` within 1e-9 relative.
2. **Radius evaluation** is dense over all points, masked to |q − p| < r. The k-NN fallback is computed for every query and selected per query.
3. **`set_radius_evaluation` / `set_lambda_scale`** round their argument through float32, because the C++ setters take `float`. Without this, results differ at the 1e-9 level; with it they agree to 1e-15.
4. **`cartesian_to_rotation`** (C++ version) clamps z to [−1, 1] before `acos`; the original returns NaN when rounding pushes |z| above 1.
5. **`k_evaluation == 0`** means all points; the C++ returns NaN.
6. **`get_average_neighbor_distance`** computes the intended sum. The C++ accumulates inside an OpenMP loop without a reduction clause (a data race).
7. **`signed_distance_and_gradient`** skips the Laplacian terms that the original computes and discards. Values are identical.
8. **I/O:**
   - `PointCloud3D.read_PLY` uses plyfile, falling back to open3d.
   - `PointCloud3D.export` declares the color fields; the original would raise.
   - `export_PLY` exists as a free function; `network.export_torus` in the original calls an undefined `export_PLY`.
   - `save_tori` writes the original file format. Files are interchangeable in both directions (tested).
9. **Initialization:** flax `lecun_normal` (truncated normal) is reproduced for the Linear layers. Only the distribution matches, which matters only for training from scratch.
10. **BVH construction:** a linear BVH (Karras 2012; Morton codes, radix tree, leaves of ≤ 4 primitives) built with torch, instead of the original's top-down binned SAH. Same node layout (`get_gpu_data`) and query semantics; a different tree, so the order of equal-distance results differs. Queries are batched over many queries.
11. **Point-BVH ray queries** (`BVH.intersect_ray` on points) return the closest sphere hit. The original tests boxes of the sphere centers (not grown by the radius) and compares slab distances (units of `ray_dir`) with sphere distances (unit direction), so it can skip the closest sphere: on the test cloud it agrees with the closest hit for 54% of the rays.
12. **`sample_isosurface`:** the original seeds a new `std::mt19937` from `std::random_device` for every ray (not reproducible); the port replays one `std::mt19937(seed)` stream with the same four draws per ray. The original computes the normals before filling the sample locations, i.e. on uninitialized memory: measured, its normals agree with the gradient at its own samples for 0% of the samples (cos > 0.99). The port returns the normalized gradients at the samples.
13. **`fast_winding_number`:** the original passes the member `areas` to the query even if they were never computed (undefined behavior); the port computes them on demand. As in the original, the expansions use areas from 21 neighbors and the direct terms the 20-neighbor point areas. The octree depth is capped at 64 (libigl: 30000; only matters for coincident points).
14. **`get_all_neighbors_and_subsample`:** the C++ shares one neighborhood buffer between OpenMP threads (a data race); the port computes each neighborhood's farthest-point subsample separately (equal to the C++ run single-threaded).
15. **Laplacian (addition):** `set_exact_laplacian(True)` uses the exact Laplacian of each torus SDF, Δg = s (1/u + (ρ − R)/(u ρ)) with ρ = |(q − c) × a|, u = √((ρ − R)² + z²) (cylindrical coordinates: u_ρρ + u_zz = 1/u). The default reproduces the C++ term (d − 1)/|q − c|, which is exact only for R = 0 (marked TODO in the original). Against central differences of the blend (λ, shift frozen), relative max error: 1.8e-2 (original term) vs 5.5e-6 (exact).
16. **Isosurface extraction:** marching cubes in torch replaces PyVista's contour filter (`flying_edges` / `marching_cubes`, which give the same surface). The case table is generated from a face-consistent rule (ambiguous faces separate the positive corners), so meshes are closed and consistently oriented; triangulations of ambiguous cases may differ from VTK's. Triangle normals point toward increasing values (outward for the SDF); the original swaps PyVista's vertex order to the same effect.

## Not ported

- **External tools** stay optional dependencies of the dataset generation (imported only where used): `thingi10k` and archive/STL readers (`py7zr`, `zipfile`, numpy-stl) for downloading meshes, `pyfqmr` for mesh simplification.
- **The original demo** (`demo/demo_3d.py`, pyglet + imgui): replaced by `sdfview`.

## k-NN backends (measurements)

`benchmarks/knn_bench.py`, CPU (4 threads), float32, N points on the unit sphere. Workloads as PAT uses them: **fit** = every point's 64 nearest points (network input); **eval** = 32 nearest points of 100k queries near the surface (SDF evaluation). Seconds; "×" = returned a wrong k-th neighbor for some queries.

| N | work | torch brute | torch BVH (build) | Slang kernel (lavapipe) | faiss flat | KeOps | scipy kd-tree | C++ nanoflann |
|---|---|---|---|---|---|---|---|---|
| 2,048 | fit | 0.012 | 0.43 (0.008) | 0.040 | 0.021 | 0.016 | 0.010 | 0.010 |
| 2,048 | eval | 0.49 | 2.70 (0.007) | 0.43 | 0.48 | 0.32 | 0.094 | — |
| 20,000 | fit | 0.65 | 1.39 (0.030) | 0.39 | 0.48 | 0.44 | 0.055 | 0.045 |
| 20,000 | eval | 2.78 | 2.74 (0.031) | 0.35 | 2.15 × | 1.24 | 0.110 | — |
| 100,000 | fit | 13.3 × | 10.2 (0.059) | 2.19 | 14.9 × | 5.85 | 0.29 | 0.28 |
| 100,000 | eval | 13.8 × | 3.32 (0.077) | 0.44 | 14.1 × | 4.93 | 0.22 | — |
| 1,000,000 | fit | — | 144 (0.90) | 24.7 | — | — | 5.69 | 9.27 |
| 1,000,000 | eval | — | 6.17 (1.21) | 0.64 | — | — | 1.35 | — |

- **Slang kernel column:** the same CPU, through Mesa's lavapipe (software Vulkan, LLVM-compiled SIMD) since there is no GPU here: a measurement of the kernel's work on this CPU, not of GPU speed. The BVH upload ("setup") was ≤ 0.09 s, the BVH build is the torch BVH's. Run-to-run variation on this machine ~25%. A variant with the held set kept sorted by insertion (d_k = last entry, no final sort) measured on lavapipe at N = 100k: fit −10%, eval (k = 32) +54%, k = 8 +26%; not adopted (GPU not measured). The column is the 'scan' kernel (held set as an unsorted array, as the GLSL shader); the 'heap' kernel (max-heap as the original C++, now the default) measured on lavapipe: N = 20k fit 0.26 s (scan 0.45), eval 0.65 (scan 0.45); N = 100k fit 1.25 (scan 2.40), eval 0.57 (scan 0.52): about 2× faster at k = 64, 10-40% slower at k = 32. GPU: `--backends slang slang-scan`.
- **Exactness.** Brute force and faiss's flat index compute |q|² + |p|² − 2 q·p in float32 (cancellation); measured separately (5000 queries, k = 32): a wrong neighbor set for 1 query at N = 100k and 58 queries (1.2%) at N = 1M, k-th distance too large by up to 8e-6 (mean k-th distance 2.1e-2). The BVH, KeOps (differences) and kd-trees returned the exact sets. In the SDF the k-th neighbor's weight is ~e^-64 of the nearest, so the effect on φ is below float32 resolution.
- **Shape of the BVH traversal.** 112-200 iterations of 88 torch ops each (N = 2k-1M, 100k queries), i.e. 10-18k small kernels per call, about independent of N. On a GPU this is launch-bound (~10 µs per launch: ~0.1-0.2 s per call) rather than work-bound; not measured here (no GPU).
- **On CPU** compiled trees (scipy, nanoflann) are 10-100× faster than any torch method: the torch traversal pays Python-level iteration and per-op overhead. The `'auto'` threshold (32768 points) is the measured CPU crossover between torch brute force and the torch BVH.
- **GPU:** run `python benchmarks/knn_bench.py --device cuda` (`--device cuda:1 --backends torch-bvh slang` for one GPU and some backends).

### Fused Slang kernel (`knn_slang.py`, `slang/*.slang`)

The torch traversal is launch-bound on a GPU (above). The kernel runs the whole traversal of one query in one thread:
one dispatch per chunk of ≤ 2²⁰ queries, independent of the tree depth.

- **Algorithm** (`slang/bvh.slang`): depth-first, nearer child first; a node is pruned when k points are held and
  dist²(q, box) > d_k², d_k the current k-th distance. Held set, two variants (`SlangKNN(..., variant=)`):
  `'scan'` (`bvhKNearestNeighbors`, function-by-function mirror of `sdfview/shaders/glsl/bvh.glsl`): an unsorted
  array of k, a closer point replaces the farthest found by an O(k) scan, d_k recomputed in O(k), insertion sort at
  the end (O(k²)); `'heap'` (`bvhKNearestNeighborsHeap`, default, the original C++ `std::priority_queue`): a
  max-heap on squared distance, d_k = root, replacement sifts down in O(log k), heapsort at the end (O(k log k)).
  Per accepted candidate at k = 64: ~2k = 128 operations vs ~2 log₂k = 12. `slang/knn.slang` writes row q ordered by
  (distance, index). Exact: every point of a pruned box is at
  distance ≥ dist(q, box) > d_k.
- **Precision.** float32, distances from differences Σ (q_c − p_c)² (no |q|² + |p|² − 2 q·p cancellation). The
  returned set can differ from the float64 one only where two candidate distances differ by less than float32
  rounding of the coordinates. For float64 queries the returned distances are recomputed in float64 from the indices.
- **Sizes.** Link-time constants per call: `BVH_K_MAX` = k rounded up to a power of two (≥ 8; per-thread arrays),
  `BVH_STACK_SIZE` = BVH depth + 2 rounded up to 8 (a depth-first traversal that pushes both children of each
  popped node holds ≤ depth + 1 entries, so the stack guard never drops a node). One compiled kernel per pair, cached
  by slangpy. k > 1024 falls back to the torch traversal.
- **Devices.** CUDA tensors: `spy.create_torch_device(type, torch_device=<tensors' device>)`; `'cuda'` (default)
  shares torch's CUDA context on that GPU (no copies, torch's current stream); `config.SLANG_DEVICE_TYPE = 'vulkan'`
  creates a Vulkan device on the same GPU (sgl matches the Vulkan adapter to the CUDA device UUID) with CUDA interop.
  CPU tensors: a Vulkan device (`config.SLANG_ADAPTER` = adapter-name substring, else slangpy's default, the first
  discrete GPU) with copies through NumPy; on a machine without a GPU, Mesa's lavapipe (software Vulkan). slangpy's
  `cpu` device type does not run these calls (slangpy 0.43.1: "Device reports zero compute dispatch groups"), and
  binding torch CPU tensors to a Vulkan device crashed (segfault) in a test with slangpy 0.43.1, hence the copies.
- **NVRTC (CUDA device type only).** slangpy's CUDA device compiles PTX at run time with NVRTC, looked up twice:
  slang-rhi (device creation, its internal kernels) takes the first `libnvrtc.so` in `$CUDA_PATH/{lib64, lib,
  targets/x86_64-linux/lib}`, then `/usr/local/cuda*` (newest first), then `/usr/lib/x86_64-linux-gnu`; Slang (the
  k-NN kernel) loads `libnvrtc.so` through the dynamic loader (`LD_LIBRARY_PATH`, ld.so cache). PTX from an NVRTC
  newer than the CUDA version the driver supports is rejected (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, 222): point
  both lookups at a toolkit no newer than the driver, `export CUDA_PATH=/usr/local/cuda-X.Y` and
  `export LD_LIBRARY_PATH=/usr/local/cuda-X.Y/lib64:$LD_LIBRARY_PATH` before starting Python, or update the driver.
  slang-rhi also needs `cuda_runtime.h` in `<dir of libnvrtc.so>/../include`, `<dir>/../../include` or
  `/usr/include`; `CUDA_HOME` (nvcc, torch extensions) is not read. Conda CUDA packages keep headers and libraries
  under `$CONDA_PREFIX/targets/x86_64-linux/{include, lib}` (with `lib/libnvrtc.so` from `cuda-nvrtc-dev`), so there
  `CUDA_PATH=$CONDA_PREFIX/targets/x86_64-linux`. `lib/stubs` directories hold link-time stubs and must not be on
  `LD_LIBRARY_PATH`. The Vulkan device type compiles to SPIR-V without NVRTC. `python -m points_as_tori.knn_slang`
  prints the driver's CUDA version, the NVRTC each lookup resolves to, the headers slang-rhi finds, and stub
  directories on the loader path.
- **Context order (Vulkan interop, several GPUs).** Measured on a TITAN RTX + RTX A6000, slangpy 0.43.0: a GPU's
  CUDA context created after a Vulkan device with CUDA interop exists in the process (on another GPU) was later
  unusable, `CUDA_ERROR_OPERATING_SYSTEM` (304): slang-rhi's `cuCtxSynchronize` aborted the process when a CUDA
  device using that context was closed at exit, and in a longer run (pytest) every torch call on that GPU failed.
  All 12 ordered pairs of {CUDA, Vulkan} × {GPU 0, GPU 1} (`python -m points_as_tori.knn_slang --pairs`) and the
  runs in one process fit this rule; contexts created before the interop device were unaffected, and same-GPU
  combinations passed. Mechanism not established (driver or slangpy). `get_device` therefore creates torch's
  context on every GPU before the first Vulkan-interop device. `knn_slang.close_devices` (atexit, before slangpy's
  own handler) synchronizes torch on the GPUs used and closes the devices in reverse creation order. The diagnostic
  runs each configuration in its own process by default; `--same-process` and `--pairs` test combinations, and
  after each step torch allocates and synchronizes on every GPU ("torch afterwards").
- **Multi-GPU.** No masking needed: the slangpy device follows the torch device of the points (one slangpy device per
  torch device, cached). For Vulkan without torch, slang-rhi's default adapter is the first discrete GPU in Vulkan
  enumeration order, which need not be CUDA's device 0 (CUDA orders fastest first unless
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`; `CUDA_VISIBLE_DEVICES` does not affect Vulkan).

## 2D SDFs (`shape_2d.SDF2D`, `shape_3d.SDF2D` / `SDF3D`)

**Layout.** Queries are (2,)/(2, n) tensors (3D: (3,)/(3, n)); every single-point C++ method also takes a batch. Shape parameters are host doubles. The Python layer keeps the original two levels (`shape_3d.Circle2D(r).bound_object` is a `shape_2d.Circle2D`), without the transposes.

**Mathematics.**
- 2D SDFs: Inigo Quilez's formulas (https://iquilezles.org/articles/distfunctions2d/) as modified by the original; each shape's formula is in its docstring.
- Revolution: f(p) = φ(√(x² + z²) − o, y). Extrusion: f(p) = min(max(w), 0) + |max(w, 0)|, w = (φ(x, y), |z| − h).
- Branches: every C++ branch is evaluated and selected with `torch.where` using the same comparisons, including the tie semantics of `std::min`/`std::max` and of Eigen's AutoDiff `max`.
- Gradients: analytic gradients are ported as is. Eigen AutoDiff (forward mode, second order for the Laplacian) becomes `torch.autograd` (reverse mode, `create_graph` for the Laplacian) on the same template expression; `ad_min`/`ad_max`/`ad_abs` pick the branch by value, so the derivative is that of the same branch. Box2D's interior sqrt(0) is masked so that 0·∞ cannot leak into the derivative.
- Samplers: ray directions uniform on the sphere, origins uniform on a disk of radius D/2 placed 0.6·D before the box center; all intersections inside the box are kept, in ray order, from at most 10·n rays (by Crofton's formula, area-uniform where the rays reach).

**Randomness.** `std::mt19937` is replayed with `cpp_random` in the C++ order, so the same seed gives the same samples as the C++. g++ evaluates `Eigen::Vector3d(dist(rng), dist(rng), dist(rng))` right to left (the first draw is z); this affects `orthonormal_basis_random` and the noise of `sample_uneven_point_cloud` (tested against the verbatim C++ function).

**Host reads:** `ray_intersect_batch` (hit count), 3D ray intersections (one active-mask read per 32 steps; the C++ loop has no step limit), samplers (per ray batch). Distances, gradients, Laplacians, 3D functions and single `ray_intersect` (fixed 1000-step loop with masks) run on `meta`.

**Behavior of the original, reproduced:**
1. **Laplacian = Δφ + 2(∂φ/∂x + ∂φ/∂y):** the second-order AutoDiff seeds are `ADScalar(v, 2, i)`, a unit inner derivative instead of zero (inside a box the C++ returns 2 instead of 0). `evaluate_laplacian(..., exact=True)` gives Δφ. Circle2D's closed form 1/‖q‖ is exact.
2. **`sample_uneven_point_cloud`:** the box is always `bbox_extrusion`; `is_extrusion=True` intersects the revolution and takes the extrusion gradient (and vice versa).
3. **Ellipse2D:** sgn·(d − isovalue) (commented "not used and not verified" in the original).
4. **Triangle2D autodiff template:** no winding sign (autodiff gradient/Laplacian of −φ for clockwise triangles).
5. **Trapezoid2D gradient:** `closest_vertex` stays set from the cap test (right direction, ill-conditioned near the surface).
6. **`sample_SDF('quad')`** reuses `subkey2` for `h` and the perturbation.
7. **Sampler ray length:** rays march only the shape-box diagonal from 0.6·D before the box center.

**Deviations:**
1. **No intersection in a sampler:** the C++ computes `i % 0` (SIGFPE); the port raises `RuntimeError`.
2. **More normals to flip than samples** (`sample_uneven_point_cloud`): the C++ reads out of bounds; the port flips all samples.
3. **OpenMP:** `sample_uneven_point_cloud` draws its noise inside an OpenMP loop (a data race with more than one thread: two identical calls gave different results with 4 threads), and `ray_intersect_batch` merges hits in thread order. The port reproduces the single-thread order; the reference calls in the tests run with one thread.
4. **Python wrapper crashes fixed** (each confirmed to raise in the original): missing `import re` in `load_SDF`; `Triangle2D(v=)` / `Quad2D(v=)` vs the constructors; single-point `evaluate_signed_distance_and_gradient` calls `.T` on a float; the Python `SDF2D` lacks `sample_points_and_normals_uniformly` / `sample_uneven_point_cloud` called by `SDF3D`; the default `SDF2D()` holds the class instead of an instance.
5. **`sample_SDF`:** an int seed or `torch.Generator` instead of a JAX key (same key structure through split/uniform analogs); the values are not the JAX values.
6. **Additions:** the Laplacian `exact` option, a module-level `load_SDF`.
7. **float32 (default):** samplers march in float32, so hit and box decisions within ~1e-7 of a threshold can differ from the C++.

| 2D SDF check (float64, 16 shapes) | measured (max \|port − C++\| / max \|C++\|) |
|---|---|
| distance, analytic gradient, autodiff gradient | ≤ 3.4e-16 |
| Laplacian (with the original's seeding term) | ≤ 3.6e-13 |
| bounding boxes | identical |
| revolution / extrusion distances; 3D gradients | ≤ 3.6e-16; ≤ 1.1e-15 |
| 2D ray intersection: hit flags; positions and normals | identical; ≤ 1.3e-16 |
| 3D ray intersections vs the C++ loop (not bound; transcribed, driven by the C++ SDF) | ≤ 1e-13 absolute, same hits |
| all samplers, same seed: positions; normals | ≤ 6.4e-15; ≤ 8e-13 (normalizing gradients of length ~ε amplifies 1e-16 by up to 1e4) |
| Python API vs the original wrappers | ≤ 1.1e-15 |

## Triangle meshes and dataset generation (`shape_3d.TriangleMesh`, `training/mesh_data.py`)

**Layout.** `TriangleMesh(vertices (V,3), faces (F,3), query_dtype=None)` holds `vertices (3,V)`, `faces (3,F)` int64 and `vertex_normals (3,V)` on the device. Every method takes and returns `(3, n)` tensors; `write_OBJ` / `export_PLY` export. `center_and_scale` works in place, as in the wrapper.

**Queries.** FCPW becomes brute force over all faces, chunked over queries with `config.CHUNK_ELEMENTS`.
- Closest point: Ericson's region test (FCPW's `findClosestPointTriangle`); the minimum over faces, NaN skipped.
- Rays: Möller-Trumbore (FCPW's `Triangle::intersect`, no hit if |det| ≤ FLT_EPSILON); nearest hit, normal oriented so that n·d < 0.
- Winding number: w(q) = Σ_f 2 atan2(det[a b c], |a||b||c| + (a·b)|c| + (b·c)|a| + (c·a)|b|), 4π inside. The signed distance is negative where w/4π > 0.5.
- `evaluate_fast_gwn` is the exact w/4π, not libigl's Barnes-Hut approximation for meshes.
- FCPW computes in float32. `query_dtype=torch.float32` reproduces it bit for bit (FCPW's operation order, Eigen's float `squaredNorm` = x² + (y² + z²), correctly rounded sqrt). `mesh_data.query_dtype` defaults to float32, so scans match the original datasets.

**Sampling densities and camera model.**
- Uniform samples: face chosen with probability area / total area (`std::discrete_distribution`), then α = √u(1−v), β = v√u, γ = 1−α−β.
- With a box, the candidate faces are the leaves of the original (SAH) BVH that overlap the box, in tree order; the host builds that tree (`_BoxQueryBVH`), because the candidate set depends on the leaves.
- Cameras sit at c + √3 p_k, p_k a rotated Fibonacci lattice, looking along −p_k. The sensor grid has spacing h and n = ⌈2√3·sensor_size/h⌉ points per row; ray (i, j) starts at x_i s + x_j t with x_i = (i+½)h − (n−1)h/2.

**Randomness.** `cpp_random` replays std::mt19937 with libstdc++'s distributions, one state per C++ distribution object (the two interleaved `normal_distribution` objects keep their saved values; g++ evaluates `Vector3d(d(rng), d(rng), d(rng))` right to left; a `discrete_distribution` with fewer than two weights draws nothing). The OpenMP loops that race on the shared engine are replayed in order (the one-thread result).

**Host reads.** The ray hit count (per batch in `..._by_casting_rays`), the mesh once for the box BVH, and the count of kept faces in `cleaned_faces`. The `evaluate_*` methods and `center_and_scale` run on `meta`.

**Dataset generation.**
- Perlin noise: `pnoise3`, bit-exact with `noise` 1.2.2, including its read past the 512-entry permutation table for `base > 0` (indices up to 765; with GCC on x86-64 the following bytes are the GRAD4 table, reproduced).
- Contouring: `isosurface.marching_cubes` instead of PyVista (same vertices; ambiguous cells triangulated differently).
- trimesh's `is_watertight`, `is_winding_consistent` and `is_volume` (including its vertex merging at 1e-8) and the KD-tree orientation test are reimplemented in torch (`MeshChecks`, `trimesh_checks`, `orientation_dot`).
- `thingi10k`, `py7zr`, `numpy-stl` (`get_Thingi10k`, `get_ABC`) and `pyfqmr` (`generate_blob_mesh`) are optional imports.
- Run: `python -m points_as_tori.training.mesh_data` (CUDA if available, float64).

**Deviations.**
1. **FCPW becomes brute force** (above). Ties: when two faces sharing an edge are at the same float32 distance, the port reports the first face and FCPW the last in its BVH order; their closest points differ by rounding, but the gradient divides by |q − p|.
2. **Centroid rounding:** Eigen's summation order for the centroid depends on the run-time alignment of its result; the port's plain sum agrees to ~3e-16 (`center_and_scale`).
3. **`evaluate_fast_gwn`** is exact (above).
4. **Square roots:** torch's CPU sqrt is not always correctly rounded (~0.7% of random inputs one ulp off); `_sqrt_rn` corrects with one step (exact on 2M inputs) where bit parity with FCPW needs it.
5. **OpenMP races:** the C++ draws normals from a shared engine inside two OpenMP loops (`sample_point_cloud` noise, the ray loop of `..._by_casting_rays`); with more than one thread its result changes between runs (two 4-thread runs differed by 0.05). The port draws in order (the one-thread result).
6. **`remove_degenerate_faces`** computes the intended filter (bug 2 below).
7. **`generate_blob_implicit_surface`** returns `(values (R,R,R), bounds_min, bounds_max)` instead of a `pv.ImageData`; the original's face flip for PyVista is not needed. Simplified blob meshes therefore differ from the original's.
8. **`orientation_dot`** indexes the given vertices; trimesh merges duplicate vertices first. They agree without duplicate or unreferenced vertices.
9. **`get_faces`** returns int64; `mesh_data` exports int32 as the original.

**Original bugs found.**
1. **Normals of length ≈ 2:** `sample_points_and_normals_uniformly` (with noise) and `sample_point_cloud` (always) do `normals.col(i) += (normals.col(i) + noise).normalized()` (adds instead of replacing). Training data of modes 1 and 2 and the pretraining dataset have |n| ≈ 2. Kept.
2. **`remove_degenerate_faces`** is called with 3×n matrices where it expects n×3 (reads out of bounds); its result is never used.
3. **Race on the shared engine** in two OpenMP loops (deviation 5).
4. **`sample_bounding_box`** is declared in the header but not defined; defined here as x, y, z drawn in order from uniform distributions seeded by `seed`.
5. **The `noise` package** reads past its permutation table for `base > 0` (reproduced, above).
6. **`mesh_data.py` cannot run as written;** minimal fixes: `add_point_cloud_noise` undefined (mode −2): defined with the uneven sampler's noise model; `subsample_point_cloud` undefined in `generate_pretraining_dataset`: the C++ farthest-point sampling port; `generate_validation_datasets` passes `only_save_one=True` and omits `mode` / `n_files`: now `mode=3, n_files=1`; `generate_blob_dataset` imports itself as `mesh_data`: uses the local function.
7. **Kept as in the original:** `get_Thingi10k` writes an OBJ for every mesh into the working directory; inspection paths use `split('/')[0]` (an absolute path writes into `/`); the last partial batch is not saved if the last mesh is skipped.

| mesh / dataset check (float64, CPU; the port mesh built from the C++ `center_and_scale` output; C++ samplers with one OpenMP thread) | measured |
|---|---|
| `center_and_scale` vs C++ (knot, block, femur, torus) | ≤ 3.3e-16 |
| bounding box, faces; `vertex_normals`, `write_OBJ` text | identical |
| `evaluate_gwn` | ≤ 5.2e-15 (relative to 4π) |
| `evaluate_fast_gwn` (exact) vs `igl::fast_winding_number` (Barnes-Hut) | 9e-4 to 2e-3; inside/outside 100% |
| signed distance, `query_dtype=float32` | bit-identical |
| signed distance, float64 queries | ≤ 1.4e-7 (FCPW's float32 rounding); signs 100% |
| gradient, float32 queries | 87-97% of queries bit-identical, the rest ≤ 4e-7 (ties) |
| gradient, float64 queries | 99% ≤ 1e-5; max 1.4e-4 near the medial axis |
| `ray_intersections`, float32 / float64 | bit-identical / same hits, points ≤ 1.9e-6 |
| every sampler, same seed (ray-cast ones with float32 queries) | points and normals bit-identical |
| `pnoise3` vs `noise.pnoise3` (6 settings × 3000 points, `base` up to 255) | bit-identical |
| `trimesh_checks` / `orientation_dot` vs trimesh + KD-tree (11 cases) | identical booleans; volume ≤ 1e-12 relative |
| marching cubes vs PyVista contour (blob) | same vertex set (≤ 6.5e-8, PyVista stores float32); volume 4.38567 vs 4.38732 (ambiguous cells) |
| blob accept/reject (pyfqmr + checks) vs original pipeline | 12/12 at resolution 64, 11/12 at 28 (pyfqmr on multi-component blobs) |
| `generate_scanned_mesh_training_dataset` vs the original module (driven by the C++), modes 0, 1, 2 | positions ≤ 4.4e-16, normals ≤ 2e-15, neighbors identical |
| same, modes 3 and −1 | positions, normals identical; neighbors identical up to the order of equidistant points (99.87% of rows) |
| output `.npz` | keys, dtypes, shapes identical; read by `train.load_mesh_training_data` |

**CPU timing** (2048-face mesh, 4 threads): signed distance of 10k queries C++ 0.04 s, port 7.3 s (float64) / 4.3 s (float32 queries; GWN 2.4 s of it); ray casting ~20 ns per ray-face pair (worst-case scan, 480k rays, ~20 s per shape; C++ ~0.1 s); farthest point cloud of 2048 points port 0.16 s, C++ 0.28 s. The brute-force queries could use `bvh.BVH` (triangles) for large meshes; not done, to keep FCPW's float32 bit parity.

## Training (`points_as_tori/training/`)

The loop follows `training/train.py`: per epoch, shapes are shuffled; per shape, queries and ground truth are sampled on the device; per batch, the network predicts the coefficients of **all** points, the tori are fitted, and the loss over the batch's queries is backpropagated; then AdamW updates. Printed logs, checkpoint cadence, validation and the five training phases (`train()`) are the same.

**Loss.** L = mean_valid |φ(q) − d_mesh(q)| + mean_valid |1 − |∇φ(q)||, where φ is the torus blend (`torus_distance_gradient_masked`).
- Queries per neighborhood: 40 on the mesh, 40 in a ±0.2 band along face normals, 40 in the extended neighborhood box.
- Ground truth: the brute-force distance to the mesh. The sign comes from the generalized winding number, Σ_f Ω_f > π meaning inside.

**Correspondences.**
- `nnx.value_and_grad` + `zero_out_nans` + `optax.adamw(warmup_cosine_decay_schedule)` → `backward()`, NaN gradient entries set to 0, `torch.optim.AdamW` with `lr = schedule(count)` set before each update. The Adam formulas are identical.
- `jax.lax.cond` → `torch.where` with a safe denominator.
- `jax.lax.dynamic_slice_in_dim` → start index clamped, so the last batch of a shape overlaps the previous one, as in JAX.
- JAX keys `fold_in(epoch, shape)` → a `torch.Generator` seeded from (seed, epoch, shape).
- `jax.random.choice(p=...)` → the same inverse-CDF arithmetic: `cumsum`, `r = c[-1](1 − u)`, `searchsorted`.

**Layout and device.**
- Queries are `(3, P, n_queries)` and ground truth `(P, n_queries)`. Meshes are `vertices (3, V)` and `faces (3, F)`.
- Each shape is uploaded once per epoch (pinned, non-blocking on CUDA).
- Losses accumulate on the device and are read back only where the original prints.
- Validation replaces the `if jnp.sum(batch_mask) > 0` host branches with weights.
- The whole step runs on `meta`: sampling, ground truth, loss, backward and AdamW.

**Behavior inherited from the original (measured identically in JAX):**
1. **Padded shapes give an all-NaN gradient.**
   - Padded points have neighbor rows of −1, i.e. neighborhoods of identical zero points. Their median neighbor distance is 0, so the input features are 0/0.
   - The loss masks these points, but backpropagating a zero cotangent through NaN activations gives 0·NaN = NaN, and every weight gradient sums over all points.
   - Result: all 1,588,358 gradient entries are NaN (JAX and torch), and `zero_out_nans` turns the step into a zero-gradient AdamW step. Parameters still move by the momentum of earlier steps.
   - Any dataset whose shapes have fewer points than `max_points` is affected.
   - **Fixed by default in the port** (see Deviations); `fix_padding=False` / `--original-padding` reproduces this.
2. **The distance loss multiplies the error by the mask** (`error * valid_mask`) instead of selecting, so a NaN prediction makes the loss NaN even when its query is marked invalid.
3. **`save_model` switches the model to eval mode** and training continues in it. This has no effect with `dropout_rate = 0`.

**Deviations.**
- **Padding fix (default on):** `fill_padding` replaces each padded point's position, normal and neighbor row by those of the first valid point before the network; the mask is unchanged. Measured: the loss is unchanged, the gradient equals the unpadded shape's to 3.3e-18 (float64), one AdamW step matches the unpadded step to 1e-14, validation values are unchanged. On a dataset whose shapes are all padded, one epoch with the original behavior changes no weight (max change 0.0) vs 5.9e-3 with the fix. Option: `TrainFundamentalFormPredictor(..., fix_padding=False)`, `train(fix_padding=False)`, CLI `--original-padding`.
- **Learning-rate schedule:** computed in float64. optax computes it in float32, which cancels digits early in the warmup and near the end of a cosine. The maximum relative difference over the original's five phases is 1.97e-5.
- **Checkpoints:** `.pt` instead of `.pkl`. Warm starts accept either.
- **Initialization:** same distributions as flax (lecun-normal truncated), but different random values. The same torch seed also gives different values in different torch versions (2.1.2 vs 2.14.1).
- **API:** architecture, sampling and path parameters of `TrainFundamentalFormPredictor` are keyword arguments, with the original values as defaults. It returns the trained model. The plot is skipped when matplotlib is missing.

**A float64 detail of the original.** `flax.nnx.MultiHeadAttention` (dropout 0) calls `jax.nn.dot_product_attention`, whose XLA path casts the logits to float32 before the softmax (`jax/_src/nn/functions.py`). In float64 the original therefore agrees with an exact float64 attention only to about 1.2e-7. The torch attention agrees with a NumPy float64 reference to 6e-16. In float32, the setting used for training, this makes no difference. The float64 parity tests give the JAX model an exact attention.

| training check | measured |
|---|---|
| closest point, solid angle, face properties, signed/unsigned distance, GWN vs JAX (float64) | ≤ 5e-16 (asserted ≤ 1e-13) |
| `generate_shape_features` with replayed JAX random draws | float64: queries 2.4e-16, distances 3.9e-16; float32: 1.7e-7, 2.9e-7 |
| SDF loss and all 1.59M parameter gradients, pre-trained model (float64) | loss ≤ 1e-12 relative; gradients max\|Δ\|/max\|g\| = 9.1e-14 |
| 4 AdamW steps vs the original trainer (float64) | steps 0–1: loss 1.5e-15, params 4.8e-11; steps 2–3: loss ≤ 7.3e-9, params ≤ 1.8e-8 |
| 4 AdamW steps, original unmodified (float32) | loss relative difference ≤ 5.8e-6 |
| schedule vs optax | ≤ 2e-5 relative (float32 cancellation in optax) |
| backward passes, torch 2.1.2 vs 2.14.1 (float64, same weights) | ≤ 6.3e-15 |
| end-to-end run on synthetic torus meshes | 2 epochs + validation, checkpoints, warm start, inference from the trained checkpoint |

About the 4-step float64 drift: Adam's update g/(|g| + ε) has derivative ε/(|g| + ε)², up to 1/(4ε) at |g| ≈ ε. This amplifies rounding-level gradient differences once moments accumulate.

Training was run on CPU only; GPU throughput was not measured.

## Validation of the inference port (CPU)

The references are the original C++ built unmodified as `pat_reference` (`tests/points_as_tori/reference`: all original sources and bindings; with CGAL when it is installed, otherwise the point-area header is stubbed and those tests skip) and the original JAX package.

| check | measured |
|---|---|
| fit_tori_from_forms vs C++ (float64, 3000 tori) | ≤ 1.6e-15 (relative to max) |
| get_neighbors vs C++ (with/without outliers) | identical indices |
| evaluate distance/gradient/Laplacian vs C++ (float64; k = 32, all points, outliers, λ_scale = 0.5, radius with k-NN fallback) | ≤ 5.4e-14 (relative to max) |
| JAX functions vs port, float64 (`torus_precompute_masked`, `torus_distance_gradient_masked`, bases, angles) | ≤ 1.7e-13 |
| network, converted weights, float32 (tooth.ply, 2048 patches of 64) | ≤ 2.6e-6 (relative to max, per coefficient) |
| end to end, original (JAX + C++) vs port, tooth / snail | max \|Δφ\| ≤ 6.0e-8 (φ range ≈ 1.2), sign agreement 100%; with identical tori ≤ 1.1e-15 |
| whole pipeline on `meta` device | runs (no data-dependent host ops) |
| point areas vs CGAL `point_areas` (float64; tooth, 1500-point sphere) | ≤ 6.2e-17 absolute (areas up to 2.6e-2) |
| point areas, regular 30×30 grid (cocircular neighborhoods) | interior identical (1.4e-17); 58 border points differ: tied 20th-nearest distances, broken differently by igl::knn |
| `signed_log_sum_exp(_gradient)`, `self_normalized_*`, `log_sum_exp` (k = 32 and all points, λ_scale 1 and 0.5) | ≤ 1.7e-13 (relative to max) |
| `winding_number`, `regularized_winding_number` | ≤ 3.3e-15 |
| `fast_winding_number` vs libigl (same octree and expansions) | ≤ 3.4e-14; Barnes-Hut vs exact sum ≤ 0.15 near the surface (β = 2) |
| `get_all_neighbors_and_subsample` vs C++ (one thread) | identical |
| BVH k-NN / closest point / radius vs the C++ BVH | identical sets, distances ≤ 1e-14 |
| Slang k-NN (lavapipe) vs float64 k-NN (k = 1 … 100, clustered cloud) | sorted exact distances of the returned sets equal the exact k-NN distances within float32 rounding; evaluator vs `'bvh'` ≤ 1e-9 relative |
| BVH triangle rays (hits, t, barycentrics) / closest point on mesh | identical, ≤ 1e-12 / distances ≤ 1e-14 |
| `cpp_random` vs g++ 13 libstdc++ (uniform, normal, int, shuffle, discrete) | bit-identical |
| marching cubes vs scikit-image (sphere, 40³) | same vertex and triangle counts, area within 1e-7 |

The end-to-end difference comes from float32 network rounding (about 1e-6), which the torus fit amplifies through 1/κ for nearly flat patches (R up to about 130).

**CPU timings** (4 threads, 100k queries, k = 32; brute-force k-NN, before the BVH backend). The port is slower than the C++ kd-tree path on CPU because brute-force selection dominates. GPU was not available to measure. See "k-NN backends" for the BVH.

| cloud | C++ (dist + grad + lap) | port float64 (dist + grad) | port float32 (dist + grad) |
|---|---|---|---|
| snail, 4096 points | 0.45 s | 2.27 s | 1.54 s |
| holey-bunny, 12917 points | 0.50 s | 5.96 s | 4.35 s |

With all points as neighbors (10k queries, snail, float32) the times are comparable: C++ 3.22 s, port 3.31 s.

## Running the tests

See the repository README (Install, Tests). Tests that compare against the original JAX code (jax 0.8.2 / flax 0.12.2, Python >= 3.11) or the original C++ (`tests/points_as_tori/reference`, CGAL optional) are skipped when those are not available.
