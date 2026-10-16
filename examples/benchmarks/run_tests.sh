#!/usr/bin/env bash
# Tests (pytest). Optional dependencies are skipped when missing; -s prints the measured differences.
#
#   bash examples/benchmarks/run_tests.sh            # port-only tests (CPU), viewer (EGL), SuperFit, experiments
#   bash examples/benchmarks/run_tests.sh reference  # + parity with the original C++ / JAX (needs the reference build)
#   bash examples/benchmarks/run_tests.sh gpu        # GPU parity (each GPU vs CPU) and the Slang kernel on every GPU
set -euo pipefail
cd "$(dirname "$0")/../.."
what=${1:-cpu}

if [[ $what == cpu ]]; then
	pytest tests -q
fi
if [[ $what == reference ]]; then
	# the original's C++ as a Python module (cmake, a C++17 compiler with OpenMP; CGAL optional for point areas)
	pip install -e ".[reference]"
	cmake -S tests/points_as_tori/reference -B build/pat_reference \
		-Dnanobind_DIR="$(python -m nanobind --cmake_dir)" -DPython_EXECUTABLE="$(which python)"
	cmake --build build/pat_reference -j
	PAT_REFERENCE_DIR=build/pat_reference pytest tests -q -s
fi
if [[ $what == gpu ]]; then
	PAT_PARITY_DEVICES=${PAT_PARITY_DEVICES:-} pytest tests/points_as_tori/test_gpu_parity.py -v -s
	pytest tests/points_as_tori/test_knn_slang.py -v      # cpu, cuda:i, cuda:i/vulkan for every GPU
fi
