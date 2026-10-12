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
| `training/train.py` | `training/train.py` | losses, schedule, train step, epoch loop, validation, checkpoints, the five-phase `train()` |
| `training/mesh_sampling.py` | `training/mesh_sampling_jax.py` | on-the-fly query sampling and ground-truth signed distances |
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
- **Training-data generation** (`training/mesh_data.py`; uses the C++ `Mesh3D` sampler, thingi10k, pyvista, pyfqmr, ...): `TriangleMesh`, `SDF2D`/`SDF3D`, `BoundingVolumeHierarchy`, `get_all_neighbors_and_subsample`, `sample_isosurface`. The training port reads the `.npz` files it produces (or the pre-generated datasets linked in the original README).

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

See the repository README (Install, Tests). 41 tests: 28 need only the port; 13 compare against the original JAX
code (jax 0.8.2 / flax 0.12.2, Python >= 3.11) and the original C++ (`tests/points_as_tori/reference`), and are skipped
when those are not available.
