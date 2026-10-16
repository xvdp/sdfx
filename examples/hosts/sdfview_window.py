"""
HOST — sdfview's own window (moderngl-window): open any .sdfx scene with view settings.

	INPUT   a scene: examples/pat/01_fit.py, python -m sdfview pat|superfit|merge, experiments/compare_pat_sup.py
	python examples/hosts/sdfview_window.py examples/out/pat/tooth.sdfx
	python examples/hosts/sdfview_window.py examples/out/compare/knot.sdfx --overlay --opacity 0.45 --color-parts

Keys (H prints them): S surface, C cut plane, P points, X/Y/Z cut axis, arrows move the cut, K clip, G/F cycle the
surface / cut-plane field, V overlay all models, T translucency, [ ] assembly parts, U color by part, I/O isovalue,
N/M neighbors k, R reset. The same as: python -m sdfview view <scene> [options].
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import sdfview  # noqa: E402
from sdfview import Scene, ViewSettings  # noqa: E402


def main():
	ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
	ap.add_argument('scene')
	ap.add_argument('--overlay', action='store_true')
	ap.add_argument('--opacity', type=float, default=1.0)
	ap.add_argument('--color-parts', action='store_true')
	ap.add_argument('--points', action='store_true')
	ap.add_argument('--up', default='y', choices=['x', 'y', 'z'])
	a = ap.parse_args()
	settings = ViewSettings(surface_mode='overlay' if a.overlay else 'single', overlay_opacity=a.opacity,
							color_parts=a.color_parts, show_points=a.points)  # fmt: skip
	sdfview.show(Scene.load(a.scene), settings, up=a.up)  # blocking


if __name__ == '__main__':
	main()
