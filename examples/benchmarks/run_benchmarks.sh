#!/usr/bin/env bash
# Benchmarks (the commands given in the conversation, in one place). Each section prints JSON rows / tables.
#
#   bash examples/benchmarks/run_benchmarks.sh knn      # k-NN backends on DEVICE (and on the CPU with scipy / C++)
#   bash examples/benchmarks/run_benchmarks.sh lbvh     # BVH build (torch LBVH vs the original C++ SAH)
#   bash examples/benchmarks/run_benchmarks.sh port     # stages of the port: inference, evaluation, meshing, training
#   bash examples/benchmarks/run_benchmarks.sh slang    # Slang kernel: diagnostics, configurations, one timing
#   bash examples/benchmarks/run_benchmarks.sh all
#   ENV  DEVICE (default cuda:0 if available, else cpu); PAT_REFERENCE_DIR (the original C++ build, for the C++ column)
set -euo pipefail
cd "$(dirname "$0")/../.."
DEVICE=${DEVICE:-$(python -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'cpu')")}
what=${1:-all}

if [[ $what == knn || $what == all ]]; then
	# GPU: torch brute force, torch BVH, Slang heap / scan kernels (fit = every point's 64 NN; eval = 32 NN of 100k)
	python benchmarks/knn_bench.py --device "$DEVICE" --backends torch-brute torch-bvh slang slang-scan
	# CPU references: scipy cKDTree and, with PAT_REFERENCE_DIR, the original C++ (nanoflann, fit only)
	PAT_REFERENCE_DIR=${PAT_REFERENCE_DIR:-build/pat_reference} \
		python benchmarks/knn_bench.py --device cpu --backends scipy-kdtree torch-bvh
fi
if [[ $what == lbvh || $what == all ]]; then
	python benchmarks/lbvh_bench.py --device "$DEVICE" --sizes 1e5 1e6 4e6 --no-traversal
fi
if [[ $what == port || $what == all ]]; then
	python benchmarks/port_bench.py --device "$DEVICE"                     # float32
	python benchmarks/port_bench.py --device "$DEVICE" --dtype float64     # the C++ precision
	# a training step at the original size needs ~20 GB (extrapolated): --train-batch / --skip-training if it fails
fi
if [[ $what == slang || $what == all ]]; then
	python -m points_as_tori.knn_slang                    # devices, NVRTC / CUDA headers found, one run per config
	python -m points_as_tori.knn_slang --pairs            # every ordered pair of GPU configurations (multi-GPU)
	python examples/hosts/slang_knn.py --device "$DEVICE" --k 32
	python examples/hosts/slang_knn.py --device "$DEVICE" --k 64 --variant scan
fi
