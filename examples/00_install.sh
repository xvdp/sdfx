#!/usr/bin/env bash
# Installation: every command used in the conversation, in order. Read before running; each block is independent.
# Tested here: Python 3.13, torch 2.11 (CPU), Mesa (llvmpipe / lavapipe). Yours: torch 2.11+cu128, TITAN RTX + A6000.
set -euo pipefail

# 0. the repository and its submodules (ext/points-as-tori with its own submodules, ext/superfit)
# git clone --recurse-submodules -b claude/gallant-dijkstra-50fc4k https://github.com/xvdp/sdfx && cd sdfx
cd "$(dirname "$0")/.."
git submodule update --init --recursive

# 1. torch for your CUDA (example: CUDA 12.8 wheels); CPU-only: pip install torch
# pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2. the three packages (points_as_tori, sdfview, superfit_sdf) and their optional parts
pip install -e ".[view,test,train]"     # viewer (moderngl, moderngl-window, pyglet), pytest, training plots
pip install -e ".[superfit]"            # SuperFit assemblies: geolipi, sysl3d, sympy, trimesh (SuperFit itself: ext/)
pip install -e ".[slang]"               # fused Slang k-NN kernel: slangpy (pip install slangpy-torch: faster binding)
pip install -e ".[dataset]"             # training-data generation: thingi10k, py7zr, numpy-stl, pyfqmr
# pip install glfw                       # second window backend (--window glfw)
# sudo apt install libegl1 libegl-mesa0 libvulkan1 mesa-vulkan-drivers xvfb glslang-tools   # headless GL, Vulkan, tests

# 3. Slang on CUDA (slangpy's CUDA device compiles with NVRTC at run time): NVRTC and CUDA headers no newer than the
#    driver. With conda's CUDA packages (headers and libnvrtc under targets/x86_64-linux):
# export CUDA_PATH=$CONDA_PREFIX/targets/x86_64-linux
# export LD_LIBRARY_PATH=$CUDA_PATH/lib:$LD_LIBRARY_PATH     # never a .../lib/stubs directory
# python -m points_as_tori.knn_slang                         # what slangpy finds, and one run per configuration

# 4. the original C++ / JAX as references for the parity tests (cmake, C++17 + OpenMP; CGAL optional)
# pip install -e ".[reference]"          # jax[cuda12] for a GPU jax; CPU jax is enough for the tests
# cmake -S tests/points_as_tori/reference -B build/pat_reference \
# 	-Dnanobind_DIR=$(python -m nanobind --cmake_dir) -DPython_EXECUTABLE=$(which python)
# cmake --build build/pat_reference -j

# 5. SuperFit fitting (only to fit new assemblies; evaluation needs step 2 only): its own environment
# conda env create -f ext/superfit/env.yml   # then remove its "-e ." line, or keep: sdfx imports ext/superfit anyway
# + cubvh and kaolin built as in ext/superfit/README.md; run with PYTHONPATH=ext/superfit (examples/superfit/03_*.sh)
echo "installed; next: python examples/pat/01_fit.py"
