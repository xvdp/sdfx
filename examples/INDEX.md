# examples

Runnable examples of every stage of the repository, run from the repository root (this file is their index, kept out
of the top-level README). Each file starts with its stage, its inputs and outputs, and the commands to run it.
Outputs go to `examples/out/` (not tracked). Timings in the notes were measured on the CPU of the machine the
examples were written on (4 threads, no GPU); GPU paths were not run there.

| file | stage | input → output |
|---|---|---|
| `00_install.sh` | install | pip extras, Slang / CUDA environment, reference build, SuperFit environment |
| **Points as Tori (PAT)** | | |
| `pat/01_fit.py` | FIT (inference) | point cloud `.ply` (or a mesh, sampled) → tori `.pkl`, mesh `.obj`, scene `.sdfx` |
| `pat/02_make_dataset.py` | DATA (new data) | your meshes, procedural blobs → training `.npz` (the original's format) |
| `pat/03_train_original.py` | TRAIN (original data) | the original's pre-generated datasets → the five-phase schedule, checkpoints |
| `pat/04_train_new.py` | TRAIN (new data) | training `.npz` → checkpoint (fine-tune the bundled network, or from scratch) |
| `pat/05_fit_trained.py` | FIT (your network) | point cloud + your checkpoint → scene with both networks' fits, differences |
| **SuperFit (SUP)** | | |
| `superfit/01_test_assembly.py` | — | nothing → a small assembly in SuperFit's pickle layout (to try the SUP path) |
| `superfit/02_assembly_to_glsl.py` | export | assembly `.pkl` → torch values, generated GLSL (whole + parts), `.sdfx`, Shadertoy `.glsl` |
| `superfit/03_fit_superfit.sh` | FIT (SuperFit) | watertight mesh → SuperFit's optimization (from `ext/superfit`, not installed) → `.sdfx` |
| **Both** | | |
| `compare/compare.sh` | FIT both | meshes → one scene per mesh: models `pat`, `sup`, `gt` (toggle / overlay); `--metrics` |
| **Hosts (viewers)** | | |
| `hosts/shadertoy_moderngl.py` | view | Shadertoy file or any `.sdfx` field (`pat`, `grid`, `glsl`) in a single-file moderngl host |
| `hosts/sdfview_window.py` | view | `.sdfx` → sdfview's window (keys: H) |
| `hosts/sdfview_headless.py` | view | `.sdfx` → PNG images and GPU probes, without a display (EGL) |
| `hosts/sdfview_embedded.py` | view | sdfview's Renderer inside your own moderngl app (two viewports) |
| `hosts/slang_knn.py` | k-NN | the fused Slang kernel from torch (CUDA, Vulkan, CPU tensors) vs the torch BVH |
| **Measurements** | | |
| `benchmarks/run_benchmarks.sh` | measure | k-NN backends (GPU; CPU with scipy and the original C++), LBVH build, port stages, Slang |
| `benchmarks/run_tests.sh` | test | CPU tests; with the reference build; GPU parity and Slang on every GPU |

## Order

1. `00_install.sh` (once).
2. PAT inference: `pat/01_fit.py`, view with `hosts/sdfview_window.py` or `python -m sdfview view`.
3. PAT training on new data: `pat/02_make_dataset.py` → `pat/04_train_new.py` → `pat/05_fit_trained.py`;
   on the original data: download it (Google Drive link in `ext/points-as-tori/README.md`, 6.2 GB) →
   `pat/03_train_original.py --check`, then the schedule.
4. SuperFit: `superfit/01_test_assembly.py` → `superfit/02_assembly_to_glsl.py`; real fits with
   `superfit/03_fit_superfit.sh` (SuperFit's environment) or the released assemblies.
5. Both: `compare/compare.sh`.
6. Measurements: `benchmarks/run_benchmarks.sh`, `benchmarks/run_tests.sh`.

## Notes measured while writing them (CPU)

- `pat/01_fit.py` on tooth.ply (2048 points): fit 6.3 s, 100k evaluations 1.2 s; knot.obj sampled at 8000 points:
  fit 22.6 s.
- `pat/02_make_dataset.py`: knot.obj (480 faces) and block.obj (4272 faces, simplified to 2048) pass the checks;
  femur.obj (7798 faces) stays non-watertight after pyfqmr at every target tried (non-manifold edges), so it is skipped
  unless `--max-faces 8000`.
- Training memory: a step runs the network on every neighborhood of a shape. Peak memory 3.1 GB at 256 points per
  shape, 5.5 GB at 512 (CPU, `--batch 128`); the original's 2048 points: ~20 GB by linear extrapolation (not
  measured). On this 15 GB machine 2048-point shapes ran out of memory; `02_make_dataset.py --points 256` makes a small
  dataset for a local smoke run (2 epochs on 2 shapes: 51 s).
- `compare/compare.sh` (knot.obj, test assembly, grid 96, 20k PAT points): 2 min 47 s.

