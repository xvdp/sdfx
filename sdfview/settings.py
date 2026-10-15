"""View settings shared by all backends (plain data; defaults that depend on the scene are resolved by `resolve`)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np


@dataclass
class ViewSettings:
	# what to draw
	show_surface: bool = True
	show_cutplane: bool = True
	show_points: bool = False
	surface_field: str = 'auto'  # field name or type; 'auto': a grid if the scene has one (fast), else the first field
	cut_field: str = 'auto'  # 'auto': the first non-grid field (exact), else the grid
	# several models in one scene (fields with a "model" key, sdfview.scene.merge_scenes):
	#   'single'   the surface of surface_field only (toggle between models by changing it)
	#   'overlay'  every model's surface (one field per model, its grid if it has one), each in its model's color;
	#              surface_field's model is drawn opaque, the others with overlay_opacity
	surface_mode: str = 'single'
	overlay_opacity: float = 1.0  # < 1: the other models blend over (no depth write), so surfaces behind stay visible
	model_colors: Optional[dict] = None  # {model: (r, g, b)}; default: palette (pat tan, sup blue, gt grey, ...)

	# level set / field
	isovalue: float = 0.0
	k: Optional[int] = None  # PAT neighbors; None: the scene's value
	lambda_scale: Optional[float] = None  # PAT λ scale; None: the scene's value

	# cut plane: n·x = c through the box; normal = axis unless cut_normal is given; offset ∈ [-1, 1] across the box
	cut_axis: int = 2
	cut_normal: Optional[Sequence[float]] = None
	cut_offset: float = 0.0
	clip_surface: bool = False  # hide the surface on the side n·x > c
	colormap_range: Optional[float] = None  # None: ½ of the box diagonal (the original: 2.0 for shapes in [-1, 1]³)
	isoband_frequency: float = 60.0  # the original demo's default (radians per unit distance)
	cut_opacity: float = 1.0

	# sphere tracing
	max_steps: int = 128
	epsilon: Optional[float] = None  # None: 1e-4 of the box diagonal
	step_scale: float = 0.9

	# points
	point_radius: Optional[float] = None  # None: the scene's value
	point_color: Sequence[float] = (0.15, 0.35, 0.75)

	# colors
	background: Sequence[float] = (1.0, 1.0, 1.0)
	color_front: Sequence[float] = (0.85, 0.78, 0.62)
	color_back: Sequence[float] = (0.55, 0.62, 0.85)

	def copy(self, **changes) -> 'ViewSettings':
		return replace(self, **changes)


def cut_plane(settings: ViewSettings, box_min, box_max):
	"""(unit normal n, offset c, rectangle origin, half-axes u, v) of the cut plane n·x = c through the box."""
	lo, hi = np.asarray(box_min, float), np.asarray(box_max, float)
	center, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
	if settings.cut_normal is not None:
		n = np.asarray(settings.cut_normal, float)
		n = n / np.linalg.norm(n)
	else:
		n = np.eye(3)[settings.cut_axis]
	reach = float(np.abs(n) @ half)  # support of the box along n, from its center
	c = float(n @ center + settings.cut_offset * reach)
	origin = center + (c - n @ center) * n
	# in-plane axes; the rectangle covers the box's projection (fragments outside the box are discarded)
	a = np.eye(3)[int(np.argmin(np.abs(n)))]
	u = np.cross(n, a)
	u /= np.linalg.norm(u)
	v = np.cross(n, u)
	r = float(np.linalg.norm(half))
	return n, c, origin, u * r, v * r
