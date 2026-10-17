from __future__ import annotations

import os
import pickle
from typing import Tuple

import numpy as np
import torch

from . import config
from .shape_3d import TorusDistanceField, PointCloud3D, fit_tori_from_forms
from .network import FundamentalFormPredictor

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'FundamentalFormPredictor.pt')
K_NEIGHBORS_ACCELERATION = 32


class PointsAsTori:
	"""
	Object that stores an oriented point cloud and its (pre-)fitted tori, on the configured torch device.

	Usage:

		points_as_tori.set_device('cuda')
		pat = PointsAsTori(points, normals, model_path='path/to/model.pt')
		distances = pat.signed_distance(queries)   # queries (3, n_queries) tensor -> (n_queries, ) tensor
		gradients = pat.sdf_gradient(queries)	  # (3, n_queries) tensor
		distances_np = points_as_tori.to_numpy(distances)   # export

	Inputs given at construction (points, normals, tori) use the original (n, 3) layout and are moved to the device
	once; queries and results are (3, n) device tensors (see config.py).
	"""

	def __init__(
		self,
		points,
		normals,
		tori: dict = None,
		tori_filepath: str = None,
		model_path: str = None,
	) -> None:
		"""
		Args:
			points: (|P|, 3) array of point positions (NumPy or torch)
			normals: (|P|, 3) array of point normals
			tori: dictionary of tori, has format
				 {centers: (|P|, 3) array, axes: (|P|, 2) array, major_radii: (|P|,) array, minor_radii: (|P|,) array}
			tori_filepath: filepath to precomputed tori
			model_path: path to a .pt model file (or the original flax .pkl, which needs jax + flax to convert)
				representing a trained neural network. Defaults to the included pre-trained model.
		"""
		points = config.as_wide(points, 3)  # (3, |P|) on the configured device
		normals = config.as_wide(normals, 3)
		self._tdf = TorusDistanceField(points, normals)
		# added in the port, set when the tori come from the network: its neighborhood size, the model and its path,
		# the predicted coefficients (6, |P|) (curvatures(), explain(), sdfview.sources.pat writes them to scenes)
		self.network_k = None
		self.model = None
		self.model_path = None
		self.coefficients = None

		if tori_filepath is not None:
			tori = self._load_tori(tori_filepath)
			self._set_tori(tori)

		if tori is None:
			if model_path is None:
				model_path = DEFAULT_MODEL_PATH
			if not os.path.isfile(model_path):
				raise FileNotFoundError(f'Model file not found: {model_path}')

			model, k_nb = PointsAsTori._load_model(model_path)
			self.network_k = int(k_nb)
			self.model, self.model_path = model, model_path

			coeffs = model.precompute_coefficients_in_chunks(points, normals, k_nb, chunk_size=50000)
			self.coefficients = coeffs
			centers, axes, major_radii, minor_radii = fit_tori_from_forms(points, normals, coeffs)

			self._tdf.set_tori(centers, axes, major_radii, minor_radii)

		else:
			self._set_tori(tori)

	def _set_tori(self, tori: dict) -> None:
		# import boundary: original (n, 3) / (n, 2) / (n,) layout -> (3, n) / (2, n) / (n,) on the device
		self._tdf.set_tori(
			config.as_wide(tori['centers'], 3),
			config.as_wide(tori['axes'], 2),
			config.as_device_vector(tori['major_radii']),
			config.as_device_vector(tori['minor_radii']),
		)

	def signed_distance(self, queries: torch.Tensor, accelerate: bool = True) -> torch.Tensor:
		"""
		Evaluate signed distance at the given query points.

		Args:
			queries: query locations, shape (3, n_queries), on the configured device

		Returns:
			(n_queries, ) tensor of distances
		"""
		self._tdf.set_k_evaluation(K_NEIGHBORS_ACCELERATION if accelerate else -1)
		return self._tdf.evaluate_distance(queries)

	def sdf_gradient(self, queries: torch.Tensor, accelerate: bool = True) -> torch.Tensor:
		"""
		Evaluate the gradient of signed distance at the given query points.

		Args:
			queries: query locations, shape (3, n_queries)

		Returns:
			(3, n_queries) tensor
		"""
		self._tdf.set_k_evaluation(K_NEIGHBORS_ACCELERATION if accelerate else -1)
		return self._tdf.evaluate_gradient(queries)

	def signed_distance_and_gradient(
		self, queries: torch.Tensor, accelerate: bool = True
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Return (distances (n_queries,), gradients (3, n_queries)).
		(The original computes the Laplacian too and discards it; the port skips those terms. Values are identical.)
		"""
		self._tdf.set_k_evaluation(K_NEIGHBORS_ACCELERATION if accelerate else -1)
		return self._tdf.evaluate_distance_gradient(queries)

	def curvatures(self) -> dict:
		"""
		Added in the port: per point, from the network's coefficients (points_as_tori.explain.curvature_frame):
		kappa_min, kappa_max (|P|,), dir_min, dir_max, normal (3, |P|), a00 (|P|,), foot (3, |P|) = p + a00 n.
		Only for tori predicted by the network in this object (not for tori read from a file).
		"""
		from .explain import curvature_frame

		if self.coefficients is None:
			raise ValueError('no network coefficients: the tori were not predicted here')
		return curvature_frame(self._tdf.points, self._tdf.normals, self.coefficients)

	def explain(self, i: int) -> dict:
		"""Added in the port: points_as_tori.explain.explain_point for point i (neighborhood, coefficients,
		curvatures, attention rollout, gradient attribution, least-squares curvatures from 6 and k points)."""
		from .explain import explain_point

		if self.model is None:
			raise ValueError('no network: the tori were not predicted here')
		tdf = self._tdf
		return explain_point(self.model, tdf.points, tdf.normals, i, self.network_k, tdf.is_outlier)

	def get_tori(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""Added in the port: tori on the device, (3, |P|), (2, |P|), (|P|,), (|P|,)."""
		return self._tdf.get_tori()

	@staticmethod
	def _load_model(model_path: str):
		model, k_neighbors = FundamentalFormPredictor.load_saved_model(model_path)
		return model, k_neighbors

	def save_tori(self, filepath: str = 'tori/tori.pkl') -> None:
		"""
		Save tori as binary file (export: device -> host). Same file format as the original:
		a pickled tuple (centers (|P|, 3), axes (|P|, 2), major_radii (|P|,), minor_radii (|P|,)) of NumPy arrays.
		"""
		dirpath = os.path.dirname(filepath)
		if dirpath:
			os.makedirs(dirpath, exist_ok=True)
		centers, axes, major_radii, minor_radii = self._tdf.get_tori()
		tori = (
			config.to_numpy(centers),
			config.to_numpy(axes),
			config.to_numpy(major_radii),
			config.to_numpy(minor_radii),
		)
		with open(filepath, 'wb') as f:
			pickle.dump(tori, f)

	def _load_tori(self, filepath: str = 'tori/tori.pkl') -> dict:
		"""
		Load tori from a binary file saved by save_tori() (of this port or of the original).

		Args:
			filepath: path to the tori file

		Returns:
			dictionary with keys 'centers', 'axes', 'major_radii', 'minor_radii' (host arrays, original layout)
		"""
		with open(filepath, 'rb') as f:
			tori = pickle.load(f)  # tuple
			return {'centers': tori[0], 'axes': tori[1], 'major_radii': tori[2], 'minor_radii': tori[3]}


def read_point_cloud(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Load an oriented point cloud from a PLY or OBJ file.

	Args:
		filepath: location of point cloud file

	Returns:
		points: (N, 3) array
		normals: (N, 3) array
	"""
	pc = PointCloud3D.read(filepath)
	return pc.points, pc.normals
