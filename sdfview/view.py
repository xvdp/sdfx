"""
Host-independent view logic shared by the Python hosts (backends/moderngl, backends/slangpy): the scene box, which
field each model draws, the surface passes of a frame, model colors. web/core/view.js is the same for the JavaScript
hosts (same names in camelCase, same rules).

`supported`: the field types a host can evaluate (None: all); a model whose fields are all unsupported is skipped.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .scene import Scene, model_of
from .settings import ViewSettings

# surface colors of models (scenes with several models: sdfview.scene.merge_scenes)
MODEL_COLORS = {'pat': (0.86, 0.70, 0.45), 'sup': (0.42, 0.62, 0.90), 'gt': (0.72, 0.72, 0.72)}
EXTRA_COLORS = [(0.52, 0.78, 0.50), (0.80, 0.55, 0.82), (0.92, 0.55, 0.45), (0.45, 0.80, 0.80)]

# preference among a model's fields: surfaces are traced with many evaluations per pixel (cheap fields first:
# generated GLSL, then grids, then PAT's exact k-NN blend); the cut plane and probes evaluate once per point (exact
# first)
SURFACE_ORDER = {'glsl': 0, 'grid': 1}
CUT_ORDER = {'grid': 1}


def scene_box(scene: Scene):
	"""(lo, hi) float64 (3,): the scene's bounds, else the union of its fields' boxes, else [-1, 1]³."""
	if scene.bounds_min is not None:
		return np.array(scene.bounds_min, float), np.array(scene.bounds_max, float)
	boxes = [(f['bounds']['min'], f['bounds']['max']) for f in scene.fields if 'bounds' in f]
	if not boxes:
		return -np.ones(3), np.ones(3)
	return np.min([b[0] for b in boxes], axis=0).astype(float), np.max([b[1] for b in boxes], axis=0).astype(float)


def models(scene: Scene) -> list:
	"""Model names of the fields, in field order (a field without a "model" key is its own model)."""
	out = []
	for f in scene.fields:
		if model_of(f) not in out:
			out.append(model_of(f))
	return out


def preferred(fields: list, for_surface: bool, supported: Optional[Sequence[str]] = None) -> Optional[dict]:
	usable = [f for f in fields if supported is None or f['type'] in supported]
	rank = SURFACE_ORDER if for_surface else CUT_ORDER
	return min(usable, key=lambda f: rank.get(f['type'], 2 if for_surface else 0)) if usable else None


def model_field(
	scene: Scene, model: str, for_surface: bool = True, supported: Optional[Sequence[str]] = None
) -> Optional[dict]:
	"""The field drawn for a model: for surfaces its cheapest field to trace, for the cut plane its exact field."""
	return preferred([f for f in scene.fields if model_of(f) == model], for_surface, supported)


def resolve_field(
	scene: Scene, which: str, for_surface: bool, supported: Optional[Sequence[str]] = None
) -> Optional[dict]:
	"""A field by name (or type); 'auto': the first model with a supported field, its preferred one."""
	if which != 'auto':
		f = scene.field(which)
		if f is None:
			raise KeyError(f'no field {which!r} in the scene')
		return f
	for m in models(scene):
		f = model_field(scene, m, for_surface, supported)
		if f is not None:
			return f
	return None


def model_color(scene: Scene, model: str, settings: ViewSettings):
	if settings.model_colors and model in settings.model_colors:
		return tuple(settings.model_colors[model])
	if model in MODEL_COLORS:
		return MODEL_COLORS[model]
	others = [m for m in models(scene) if m not in MODEL_COLORS]
	return EXTRA_COLORS[others.index(model) % len(EXTRA_COLORS)] if model in others else EXTRA_COLORS[0]


def surface_passes(scene: Scene, s: ViewSettings, supported: Optional[Sequence[str]] = None) -> list:
	"""
	[(field, color_front, color_back, opacity)] of the surfaces to draw: surface_field alone ('single'), or one field
	per model ('overlay': surface_field's model opaque, the others with overlay_opacity). With several models in
	the scene, each surface takes its model's color.
	"""
	primary = resolve_field(scene, s.surface_field, True, supported)
	if not s.show_surface or primary is None:
		return []
	ms = models(scene)
	fields = [(primary, 1.0)]
	if s.surface_mode == 'overlay':
		for m in ms:
			f = model_field(scene, m, True, supported) if m != model_of(primary) else None
			if f is not None:
				fields.append((f, float(s.overlay_opacity)))
	out = []
	for field, opacity in fields:
		if len(ms) > 1:
			front = model_color(scene, model_of(field), s)
			back = tuple(0.55 * x for x in front)
		else:
			front, back = tuple(s.color_front), tuple(s.color_back)
		out.append((field, front, back, opacity))
	return out


def field_part(field: dict, settings: ViewSettings) -> int:
	"""Part shown for a field with parts: settings.part if in range (-2 all parts, i >= 0 part i), else -1."""
	n = int(field.get('glsl', {}).get('n_parts', 0))
	return settings.part if -2 <= settings.part < n else -1
