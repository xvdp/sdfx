"""
SuperFit primitive assembly (superfit_sdf.SuperFitSDF) -> scene:
	* field type "glsl" (default): the assembly as GLSL generated at run time from the user's SuperFit checkout
	  (superfit_sdf/glsl.py): the composition of its primitives, and each primitive alone; exact, evaluated by the
	  viewer per pixel. The primitives' GLSL is SuperFit's (Adobe Research License), stored in the scene, not here.
	* field type "grid" (resolution given): the torch evaluation baked on a grid.
"""

from __future__ import annotations

from typing import Optional

from ..scene import Scene, glsl_field, grid_field
from .grid import bake_grid


def scene_from_superfit(
	sdf, resolution: Optional[int] = 192, meta: Optional[dict] = None, name: str = 'superfit', glsl: bool = True
) -> Scene:
	"""
	Scene of `sdf` (superfit_sdf.SuperFitSDF) over its evaluation domain (SuperFit's [-1, 1]³, in the frame of the
	SDF): field `name` (type "glsl") if glsl, and `name`_grid (or `name` without glsl) baked on a resolution³
	vertex-aligned grid if resolution.
	"""
	if not glsl and not resolution:
		raise ValueError('nothing to export: glsl=False and no grid resolution')
	lo, hi = sdf.bounds()
	meta = dict(meta or {})
	meta.update({'source': 'superfit', 'program': sdf.program, 'n_primitives': sdf.n_primitives,
				 'primitive_types': sdf.primitive_types, 'scale': float(sdf.scale),
				 'center': [float(c) for c in sdf.center]})  # fmt: skip
	scene = Scene(bounds_min=lo, bounds_max=hi, meta=meta)
	if glsl:
		from superfit_sdf.glsl import assembly_glsl

		glsl_field(scene, name, assembly_glsl(sdf), lo, hi)
	if resolution:
		values = bake_grid(lambda q: sdf(q), lo, hi, int(resolution), device=sdf.device, dtype=sdf.dtype)
		grid_field(scene, f'{name}_grid' if glsl else name, values, lo, hi)
	return scene
