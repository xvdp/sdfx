# points_as_tori: torch port of Points as Tori (PAT)

Source: `ext/points-as-tori` (JAX/flax.nnx network + C++/nanobind evaluator), MIT license.

## File mapping

| port (`points_as_tori/`) | original | content |
|---|---|---|
| `config.py` | — | device/dtype set once (`set_device`), import (`as_wide`) / export (`to_numpy`) boundaries, chunk size |
| `shape_2d.py` | `src/cpp/shape_2d.h/.cpp` | `sgn`, `sign`, `orthonormal_basis` (+ SoA `cross`/`dot`/`norm`) |
| `torus_distance.py` | `src/cpp/torus_distance.h/.cpp` | `TorusDistanceField`, `get_neighbors`, `get_average_neighbor_distance`, torus fitting, torus SDF/gradient |
| `shape_3d.py` | `src/pointsastori/shape_3d.py` | Python API layer: point-cloud I/O, PLY export, `get_neighbors`, `compute_optimal_radius`, `fit_tori_from_forms`, `TorusDistanceField` |
| `network.py` | `src/pointsastori/network.py` | helpers, training-side torus math (`torus_precompute_masked`, `torus_distance_gradient_masked`), export, transformer |
| `infer.py` | `src/pointsastori/infer.py` | `PointsAsTori`, `read_point_cloud` |
| `convert_checkpoint.py` | — | flax `.pkl` -> torch `.pt` |
| `models/FundamentalFormPredictor.pt` | `src/pointsastori/models/FundamentalFormPredictor.pkl` | converted pre-trained weights (embed 128, 8 heads, 8 layers, mlp 512, k = 64; 1,588,358 parameters) |

Function names are unchanged, including the `_jax` suffixes.

## Conventions

**Device.** `points_as_tori.set_device('cuda')` once; until it is called, `get_device()` raises, and calling it again with a different device also raises (unless `force=True`). Nothing moves tensors implicitly: inputs on another device are rejected.
- Import: `as_wide`, and the `PointsAsTori(points, normals, tori=...)` constructor, which takes the original (n, 3) layout.
- Export: `to_numpy`, `save_tori`, `export_tori*`, `write_point_cloud`, `PointCloud3D.export`.

**No host sync** between import and export. The code contains no `.item()`, `.cpu()`, `nonzero`, boolean-mask indexing or branching on tensor values. Data-dependent choices are made per element with `torch.where`, for example the radius-mode k-NN fallback, outliers and the min/max curvature selection. This is verified by running the whole pipeline on the `meta` device (`tests/points_as_tori/test_device.py`). On CUDA it can also be checked with `torch.cuda.set_sync_debug_mode('error')`.

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

1. **Nearest neighbors:** exact brute force (one GEMM for squared distances, then `topk` per query chunk) instead of the nanoflann kd-tree. Distances that enter the SDF are recomputed exactly from the gathered differences. The cost is O(|Q||P|); see the timings below.
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

## Not ported yet

- **Point-area based functions** (CGAL `point_areas`): `signed_log_sum_exp(_gradient)`, `self_normalized_*`, `winding_number`, `fast_winding_number*`, `regularized_winding_number`. These raise `NotImplementedError`.
- **Training-data generation:** `TriangleMesh`, `SDF2D`/`SDF3D`, `BoundingVolumeHierarchy`, `get_all_neighbors_and_subsample`, `sample_isosurface`.
- **The training loop** (`training/*.py`). Its loss functions `torus_precompute_masked` and `torus_distance_gradient_masked` are ported and tested.

## Validation (`tests/points_as_tori`, 25 tests, CPU)

The references are the original C++ built unmodified as `pat_reference` (`tests/points_as_tori/reference`; the CGAL point-area header is stubbed) and the original JAX package.

| check | measured |
|---|---|
| fit_tori_from_forms vs C++ (float64, 3000 tori) | ≤ 1.6e-15 (relative to max) |
| get_neighbors vs C++ (with/without outliers) | identical indices |
| evaluate distance/gradient/Laplacian vs C++ (float64; k = 32, all points, outliers, λ_scale = 0.5, radius with k-NN fallback) | ≤ 5.4e-14 (relative to max) |
| JAX functions vs port, float64 (`torus_precompute_masked`, `torus_distance_gradient_masked`, bases, angles) | ≤ 1.7e-13 |
| network, converted weights, float32 (tooth.ply, 2048 patches of 64) | ≤ 2.6e-6 (relative to max, per coefficient) |
| end to end, original (JAX + C++) vs port, tooth / snail | max \|Δφ\| ≤ 6.0e-8 (φ range ≈ 1.2), sign agreement 100%; with identical tori ≤ 1.1e-15 |
| whole pipeline on `meta` device | runs (no data-dependent host ops) |

The end-to-end difference comes from float32 network rounding (about 1e-6), which the torus fit amplifies through 1/κ for nearly flat patches (R up to about 130).

**CPU timings** (4 threads, 100k queries, k = 32). The port is slower than the C++ kd-tree path on CPU because brute-force selection dominates. GPU was not available to measure.

| cloud | C++ (dist + grad + lap) | port float64 (dist + grad) | port float32 (dist + grad) |
|---|---|---|---|
| snail, 4096 points | 0.45 s | 2.27 s | 1.54 s |
| holey-bunny, 12917 points | 0.50 s | 5.96 s | 4.35 s |

With all points as neighbors (10k queries, snail, float32) the times are comparable: C++ 3.22 s, port 3.31 s.

## Running the tests

```
pip install torch numpy plyfile pytest            # port
pip install jax==0.8.2 flax==0.12.2 optax nanobind # references (optional)
cmake -S tests/points_as_tori/reference -B build/pat_reference -G Ninja \
      -Dnanobind_DIR=$(python -m nanobind --cmake_dir) -DPython_EXECUTABLE=$(which python)
cmake --build build/pat_reference
PAT_REFERENCE_DIR=build/pat_reference python -m pytest tests/points_as_tori -s
```

Tests that need the references are skipped when the references are unavailable.
