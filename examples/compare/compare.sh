#!/usr/bin/env bash
# PAT + SUP — both pipelines on the same meshes, one scene per mesh with models pat, sup and gt (toggle G, overlay V,
# translucency T; SUP parts [ ] and U). experiments/compare_pat_sup.py (python ... --help for every option).
#
#   INPUT    meshes (--mesh) and / or SuperFit assemblies (--suppkl, fitted beforehand, e.g. with
#            examples/superfit/03_fit_superfit.sh); the two pair in order; --suppkl alone: each assembly's stored mesh
#   OUTPUT   $OUT/<shape>.sdfx  (fields pat.pat, pat.pat_grid, sup.sup, sup.sup_grid, gt);  --metrics: metrics.jsonl
#   ENV      DEVICE (default cuda if available, else cpu), OUT (examples/out/compare)
#
#   bash examples/compare/compare.sh            # knot.obj with the test assembly (runs anywhere, CPU ok)
set -euo pipefail
cd "$(dirname "$0")/../.."
DEVICE=${DEVICE:-$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")}
OUT=${OUT:-examples/out/compare}
MESH=ext/points-as-tori/data/knot.obj

# 1. PAT fit (stage FIT, bundled network) + a given assembly + ground truth (here: the test assembly, not a fit of the
#    knot; examples/superfit/01_test_assembly.py)
[ -f examples/out/superfit/test_assembly.pkl ] || python examples/superfit/01_test_assembly.py
python experiments/compare_pat_sup.py --mesh $MESH --suppkl examples/out/superfit/test_assembly.pkl \
	--device "$DEVICE" --res 96 --pat-points 20000 -o "$OUT"
echo "view: python -m sdfview view $OUT/knot.sdfx --overlay --opacity 0.45"

# 2. other forms (uncomment):
# SuperFit fitted to the mesh first (its environment, a CUDA GPU), then both:
# bash examples/superfit/03_fit_superfit.sh a.obj
# python experiments/compare_pat_sup.py --mesh a.obj --suppkl examples/out/superfit/a/final_content.pkl -o "$OUT"
# a released assembly with the mesh it was fitted to (the stored path is the authors' machine's: give the mesh):
# python experiments/compare_pat_sup.py --mesh PartObjaverse-Tiny/<id>.glb \
# 	--suppkl dataset/partobjaverse/superfrustum/<id>/primitive_assembly.pkl -o "$OUT"
# a mesh without an assembly: PAT and the ground truth only (prints how to fit SuperFit to it):
# python experiments/compare_pat_sup.py --mesh a.obj -o "$OUT"
# metrics against the ground truth (IoU, Chamfer, SDF error, normals; SuperFit's definitions where they apply):
# python experiments/compare_pat_sup.py --mesh $MESH --suppkl A.pkl --metrics -o "$OUT"
