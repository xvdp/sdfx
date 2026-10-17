"""
points_as_tori — torch port of Points as Tori (Feng, Gkioulekas, Crane, SIGGRAPH 2026):
fast signed distance from oriented point clouds. Original (JAX + C++): ext/points-as-tori.

Quickstart::

    import torch
    import points_as_tori as pat

    pat.set_device('cuda')                      # once (default: cuda if available, else cpu); everything below stays on it

    points, normals = pat.read_point_cloud('input.ply')   # (N, 3) NumPy arrays
    sdf = pat.PointsAsTori(points, normals)               # tori fitted on the device

    queries = torch.rand(3, 1000, device=pat.get_device()) * 2 - 1   # (3, n) wide layout
    distances = sdf.signed_distance(queries)    # (1000,) tensor on the device
    gradients = sdf.sdf_gradient(queries)       # (3, 1000) tensor on the device

    distances_np = pat.to_numpy(distances)      # export (device -> host)
"""

from .config import set_device, get_device, get_dtype, default_device, as_wide, to_numpy  # noqa: F401
from .network import FundamentalFormPredictor  # noqa: F401
from .infer import PointsAsTori, read_point_cloud  # noqa: F401
from .shape_3d import *
from .bvh import BVH  # noqa: F401
from .isosurface import marching_cubes, mesh_isocontour  # noqa: F401
