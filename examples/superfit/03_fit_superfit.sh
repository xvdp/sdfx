#!/usr/bin/env bash
# SUP — stage FIT: SuperFit's own per-shape optimization (scripts/mesh_to_assembly.py) run from ext/superfit without
# installing SuperFit, then the result as a scene (generated GLSL: whole assembly and parts) and a Shadertoy file.
#
#   INPUT    $1 = a mesh (default ext/points-as-tori/data/knot.obj; any format trimesh reads). SuperFit remeshes it: its
#            signed distance on a grid (cubvh ray stabbing), cleaned, then marching cubes; the fit stops if that mesh is
#            not watertight
#   OUTPUT   $OUT/<name>/final_content.pkl, config.json   SuperFit's result (its own format)
#            $OUT/<name>.sdfx, $OUT/<name>.glsl           in the input mesh's frame (--mesh)
#            <name> = the mesh file's name without extension (its folder's name when the file is mesh.*, as in Toys4K)
#   REQUIRES the active python with SuperFit's fitting dependencies (ext/superfit/env.yml without its "-e ." line: torch
#            2.9 cu128, geolipi, sysl3d, cc3d, libigl, fastsweep, drjit, ...; cubvh and kaolin built as in
#            ext/superfit/README.md) and this repository's (pip install -e ".[superfit]"); a CUDA GPU: SuperFit runs on
#            "cuda", the first visible one (choose another with CUDA_VISIBLE_DEVICES)
#   ENV      OUT (examples/out/superfit), EXTRA (more mesh_to_assembly.py arguments, e.g. "--ablation 8": CUDA op)
#
#   bash examples/superfit/03_fit_superfit.sh ext/points-as-tori/data/knot.obj
#   CUDA_VISIBLE_DEVICES=1 bash examples/superfit/03_fit_superfit.sh my_mesh.glb
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD
MESH=$(realpath "${1:-ext/points-as-tori/data/knot.obj}")
OUT=$(realpath -m "${OUT:-examples/out/superfit}")
EXTRA=${EXTRA:-}
name=$(basename "${MESH%.*}")
[ "$name" = mesh ] && name=$(basename "$(dirname "$MESH")")
mkdir -p "$OUT/$name"

# SuperFit reads its paths from superfit_config.json (or $SUPERFIT_CONFIG); its default data_base is inside its
# repository (--fastmode stores torch.compile artifacts at <data_base>/project_sf/aot): keep them under $OUT instead
if [ -z "${SUPERFIT_CONFIG:-}" ] && [ ! -f ext/superfit/superfit_config.json ]; then
	printf '{"data_base": "%s/data", "outputs_base": "%s/outputs"}\n' "$OUT" "$OUT" > "$OUT/superfit_config.json"
	export SUPERFIT_CONFIG="$OUT/superfit_config.json"
fi

# 1. FIT (SuperFit): PYTHONPATH makes `import superfit` resolve to the submodule
( cd ext/superfit && PYTHONPATH="$REPO/ext/superfit${PYTHONPATH:+:$PYTHONPATH}" \
	python scripts/mesh_to_assembly.py --input_path "$MESH" --save_dir "$OUT/$name" --fastmode $EXTRA )

# 2. scene + Shadertoy file (evaluation needs only geolipi / sysl3d: pip install -e ".[superfit]")
python -m sdfview superfit "$OUT/$name/final_content.pkl" --mesh "$MESH" -o "$OUT/$name.sdfx" --shadertoy "$OUT/$name.glsl"
echo "view:   python -m sdfview view $OUT/$name.sdfx --color-parts      ([ ] parts, U colors)"
echo "        python examples/hosts/shadertoy_moderngl.py $OUT/$name.glsl --part -2"
