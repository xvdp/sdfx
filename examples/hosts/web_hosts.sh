#!/usr/bin/env bash
# HOST — the browser hosts (sdfview/web): raw WebGL2, raw WebGPU, three.js WebGLRenderer (GLSL modules), three.js
# WebGPURenderer (TSL; also on its WebGL2 backend), and an SDF inside an ordinary three.js scene (compose.html).
# Every host implements the same contract as the moderngl Renderer (draw / render / probe / release); see
# sdfview/PARALLEL.md for the step-by-step correspondence.
#
#	INPUT   one or more scenes (.sdfx)
#	OUTPUT  a local server; open the printed URLs (one per host and scene) in Chrome or Firefox
#	bash examples/hosts/web_hosts.sh examples/out/pat/tooth.sdfx
#	bash examples/hosts/web_hosts.sh examples/out/compare/knot.sdfx --port 8001
#
# Field types per host: WebGL2, three.js WebGL: pat, grid, glsl; WebGPU: pat, grid; three.js WebGPU (TSL): grid
# (bake PAT / SuperFit scenes with --bake N to see them there). Every page draws surfaces, the cut plane and points. The three.js pages load three@0.186.1 from the
# jsDelivr CDN (import maps in the pages). WebGPU needs a browser exposing navigator.gpu; per
# https://github.com/gpuweb/gpuweb/wiki/Implementation-Status (edited 2026-05): Chrome on Windows / macOS / ChromeOS;
# on Linux Chrome 144+ (Intel Gen12+) and 147+ (NVIDIA, Wayland), other GPUs behind chrome://flags/#enable-unsafe-webgpu;
# Firefox 141+ on Windows, 145+ / 147+ on Apple Silicon macOS, Linux in Nightly only. The WebGL2 hosts run everywhere.
# Keys in the viewer pages: the moderngl window's (H shows / hides them on the page): S surface, C cut plane,
# P points, X Y Z cut axis, arrows move the cut, K clip, B cut opacity, G / F surface / cut field, V overlay models,
# T translucency, [ ] parts, U part colors, I / O isovalue, N / M neighbors, R reset; left drag orbit, right drag pan,
# wheel zoom; a .sdfx file can be dropped on the page.
#
# Tests of the same hosts, headless (node + playwright; three.js fetched with npm pack):
#	pytest tests/sdfview/test_hosts.py tests/sdfview/test_shader_assembly.py -s
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ $# -lt 1 ]; then echo "usage: $0 scene.sdfx [scene2.sdfx ...] [--port N]"; exit 1; fi
python -m sdfview serve "$@"
