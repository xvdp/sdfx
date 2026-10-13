"""
Orbit camera (NumPy, backend independent). Matrices follow the OpenGL conventions: right-handed view space looking
down -z, clip space z ∈ [-w, w]; row-major NumPy arrays (transpose when writing column-major GPU uniforms).

	eye  = target + distance · (cos(pitch) (sin(yaw) e1 + cos(yaw) e2) + sin(pitch) up),  (e1, e2, up) orthonormal
	view = lookAt(eye, target, up)
	proj = perspective(fov_y, aspect, near, far):  P[0,0] = f / aspect, P[1,1] = f, f = cot(fov_y / 2),
	       P[2,2] = (far + near) / (near - far), P[2,3] = 2 far near / (near - far), P[3,2] = -1
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

AXES = {'x': (1.0, 0.0, 0.0), 'y': (0.0, 1.0, 0.0), 'z': (0.0, 0.0, 1.0)}


def look_at(eye, target, up) -> np.ndarray:
	f = np.asarray(target, float) - eye
	f /= np.linalg.norm(f)
	s = np.cross(f, up)
	if np.linalg.norm(s) < 1e-9:  # looking along up
		s = np.cross(f, np.roll(up, 1))
	s /= np.linalg.norm(s)
	u = np.cross(s, f)
	m = np.eye(4)
	m[0, :3], m[1, :3], m[2, :3] = s, u, -f
	m[:3, 3] = -m[:3, :3] @ eye
	return m


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
	f = 1.0 / math.tan(math.radians(fov_y_deg) / 2)
	m = np.zeros((4, 4))
	m[0, 0], m[1, 1] = f / aspect, f
	m[2, 2], m[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
	m[3, 2] = -1.0
	return m


@dataclass
class OrbitCamera:
	target: np.ndarray = field(default_factory=lambda: np.zeros(3))
	distance: float = 3.0
	yaw: float = 0.6  # radians, about `up`
	pitch: float = 0.4  # radians, elevation above the plane normal to `up`
	fov_y: float = 40.0  # degrees
	up: np.ndarray = field(default_factory=lambda: np.array(AXES['y']))
	radius: float = 1.0  # scene radius, for the clip planes

	@classmethod
	def fit(cls, bounds_min, bounds_max, up='y', **kw) -> 'OrbitCamera':
		lo, hi = np.asarray(bounds_min, float), np.asarray(bounds_max, float)
		radius = 0.5 * float(np.linalg.norm(hi - lo))
		cam = cls(
			target=0.5 * (lo + hi), up=np.array(AXES[up] if isinstance(up, str) else up, float), radius=radius, **kw
		)
		cam.distance = radius / math.sin(math.radians(cam.fov_y) / 2)  # bounding sphere just inside the view cone
		return cam

	def basis(self):
		up = self.up / np.linalg.norm(self.up)
		e1 = np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 1.0, 0.0])
		e1 /= np.linalg.norm(e1)
		e2 = np.cross(e1, up)
		return e1, e2, up

	def eye(self) -> np.ndarray:
		e1, e2, up = self.basis()
		cp = math.cos(self.pitch)
		d = cp * (math.sin(self.yaw) * e1 + math.cos(self.yaw) * e2) + math.sin(self.pitch) * up
		return self.target + self.distance * d

	def view_matrix(self) -> np.ndarray:
		return look_at(self.eye(), self.target, self.up)

	def proj_matrix(self, aspect: float) -> np.ndarray:
		near = max(self.distance - 2.0 * self.radius, 1e-3 * self.distance)
		far = self.distance + 2.0 * self.radius
		return perspective(self.fov_y, aspect, near, far)

	def direction(self) -> np.ndarray:
		"""Unit view direction (eye -> target)."""
		d = self.target - self.eye()
		return d / np.linalg.norm(d)

	# ---- interaction (pixels -> angles / distances)

	def orbit(self, dx: float, dy: float, speed: float = 0.01) -> None:
		self.yaw -= dx * speed
		self.pitch = float(np.clip(self.pitch + dy * speed, -1.55, 1.55))

	def zoom(self, steps: float) -> None:
		self.distance *= 0.9**steps

	def pan(self, dx: float, dy: float, viewport_height: float) -> None:
		view = self.view_matrix()
		right, up = view[0, :3], view[1, :3]
		scale = 2.0 * self.distance * math.tan(math.radians(self.fov_y) / 2) / max(viewport_height, 1.0)
		self.target = self.target - scale * (dx * right - dy * up)
