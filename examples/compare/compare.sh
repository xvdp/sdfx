#!/usr/bin/env bash
# PAT + SUP — both pipelines on the same meshes, one scene per mesh with models pat, sup and gt (toggle G, overlay V,
# translucency T; SUP parts [ ] and U). experiments/compare_pat_sup.py (python ... --help for every option).
#
#   INPUT    meshes; SUP from an assembly (--assembly), a directory of assemblies (--assembly-dir, e.g. the Hugging Face
#            release with --mesh-root to the dataset), or fitted here (--fit-superfit, needs SuperFit's environment)
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
python experiments/compare_pat_sup.py --mesh $MESH --assembly examples/out/superfit/test_assembly.pkl \
	--device "$DEVICE" --res 96 --pat-points 20000 -o "$OUT"
echo "view: python -m sdfview view $OUT/knot.sdfx --overlay --opacity 0.45"

# 2. other forms (uncomment):
# both fitted here (SuperFit through its own environment; GPU index as torch numbers them):
# python experiments/compare_pat_sup.py --mesh a.obj b.obj --fit-superfit \
# 	--superfit-python ~/conda/envs/superfit/bin/python --superfit-gpu 1 --device cuda:0 -o "$OUT"
# released assemblies with their meshes (paths re-rooted under --mesh-root):
# python experiments/compare_pat_sup.py --assembly-dir superfit-release/dataset/toys4k/superfrustum \
# 	--mesh-root /data/toys4k --device cuda:0 -o "$OUT" --limit 20
# metrics against the ground truth (IoU, Chamfer, SDF error, normals; SuperFit's definitions where they apply):
# python experiments/compare_pat_sup.py --mesh $MESH --assembly A.pkl --metrics --device cuda:0 -o "$OUT"
