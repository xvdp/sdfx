"""
SuperFit primitive assembly (superfit_sdf.SuperFitSDF) -> scene: the assembly's SDF baked on a grid (field type
"grid"). The viewer has no native field type for SuperFit's primitives; their GLSL (SuperFit's shader modules,
Adobe Research License) is not copied into this repository.
"""

from __future__ import annotations

from typing import Optional

from ..scene import Scene, grid_field
from .grid import bake_grid


def scene_from_superfit(sdf, resolution: int = 192, meta: Optional[dict] = None, name: str = 'superfit') -> Scene:
	"""
	Bake `sdf` (superfit_sdf.SuperFitSDF) over its evaluation domain (SuperFit's [-1, 1]³, in the frame of the SDF) on a
	resolution³ vertex-aligned grid.
	"""
	lo, hi = sdf.bounds()
	meta = dict(meta or {})
	meta.update({'source': 'superfit', 'program': sdf.program, 'n_primitives': sdf.n_primitives,
				 'primitive_types': sdf.primitive_types, 'scale': float(sdf.scale),
				 'center': [float(c) for c in sdf.center]})  # fmt: skip
	scene = Scene(bounds_min=lo, bounds_max=hi, meta=meta)
	values = bake_grid(lambda q: sdf(q), lo, hi, int(resolution), device=sdf.device, dtype=sdf.dtype)
	grid_field(scene, name, values, lo, hi)
	return scene
