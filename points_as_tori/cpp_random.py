"""
Replay of the C++ random streams the original uses (std::mt19937 with the libstdc++ distributions), so that the
ported samplers can reproduce the original's draws exactly when given the same seed.

Implemented as in libstdc++ (GCC 13, bits/random.h, bits/random.tcc, bits/uniform_int_dist.h, bits/stl_algo.h):

	std::mt19937(seed)                 init_genrand(seed); raw 32-bit outputs (numpy MT19937 with legacy seeding)
	generate_canonical<double, 53>     two raw draws g1, g2:  (g1 + g2 · 2³²) / 2⁶⁴ in double, clamped below 1
	uniform_real_distribution(a, b)    canonical · (b - a) + a
	normal_distribution(μ, σ)          Marsaglia polar: x, y = 2u - 1 until 0 < r² = x² + y² ≤ 1; m = √(-2 ln r² / r²);
	                                   returns y·m and keeps x·m for the next call (per distribution object);
	                                   ln from libm (math.log), as the C++
	uniform_int_distribution(a, b)     Lemire's nearly divisionless method with 32-bit draws:
	                                   p = g · R (64-bit), R = b - a + 1; reject while (p mod 2³²) < (2³² - R) mod R;
	                                   a + p >> 32
	std::shuffle                       for n ≤ 65535: swaps generated two at a time from one integer draw
	                                   x ∈ [0, b0·b1 - 1] -> (x / b1, x mod b1); otherwise one draw per swap
	discrete_distribution(w)           cumulative p = partial sums of w / Σw (last set to 1); lower_bound of a canonical draw

These run on the host (the original's sampling is sequential in the stream); results move to the device once.

Note on C++ evaluation order: g++ evaluates function and constructor arguments right to left, so
`Eigen::Vector3d s(dist(rng), dist(rng), dist(rng))` gets the third draw in s.x (s = (z3, z2, z1)). Draws inside one
expression must be replayed in that order.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

TWO32 = 4294967296.0
TWO64 = 18446744073709551616.0
_ONE_BELOW = np.nextafter(1.0, 0.0)


class MT19937:
	"""std::mt19937 raw stream with lookahead, so vectorized consumers can take exactly what they use."""

	def __init__(self, seed: int):
		self._bg = np.random.MT19937()
		self._bg._legacy_seeding(int(seed) & 0xFFFFFFFF)
		self._buf = np.empty(0, dtype=np.uint64)

	def peek(self, n: int) -> np.ndarray:
		if self._buf.size < n:
			more = max(n - self._buf.size, 4096)
			self._buf = np.concatenate([self._buf, self._bg.random_raw(more).astype(np.uint64)])
		return self._buf[:n]

	def take(self, n: int) -> np.ndarray:
		out = self.peek(n).copy()
		self._buf = self._buf[n:]
		return out

	def __call__(self) -> int:
		return int(self.take(1)[0])

	# generate_canonical<double, 53>: two draws per value
	def canonical(self, n: int) -> np.ndarray:
		g = self.take(2 * n).astype(np.float64).reshape(n, 2)
		return _canonical(g[:, 0], g[:, 1])


def _canonical(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
	s = g1 + g2 * TWO32  # rounded like the C++ double accumulation (sum += g * tmp)
	r = s / TWO64
	return np.where(r >= 1.0, _ONE_BELOW, r)


class UniformReal:
	"""std::uniform_real_distribution<double>(a, b)."""

	def __init__(self, a: float = 0.0, b: float = 1.0):
		self.a, self.b = float(a), float(b)

	def __call__(self, rng: MT19937, n: int = None):
		u = rng.canonical(1 if n is None else n) * (self.b - self.a) + self.a
		return float(u[0]) if n is None else u


class Normal:
	"""std::normal_distribution<double>(mean, stddev), with the saved second value of the polar method."""

	def __init__(self, mean: float = 0.0, stddev: float = 1.0):
		self.mean, self.stddev = float(mean), float(stddev)
		self._saved = None

	def __call__(self, rng: MT19937, n: int = None):
		count = 1 if n is None else int(n)
		out = np.empty(count)
		filled = 0
		if self._saved is not None and count > 0:
			out[0] = self._saved
			self._saved = None
			filled = 1
		while filled < count:
			need = count - filled  # values still needed; each accepted candidate gives two (y·m now, x·m saved)
			m = int(math.ceil(need / 2 * 1.3)) + 4  # candidates to examine (acceptance π/4)
			raw = rng.peek(4 * m).astype(np.float64).reshape(m, 4)
			x = 2.0 * _canonical(raw[:, 0], raw[:, 1]) - 1.0
			y = 2.0 * _canonical(raw[:, 2], raw[:, 3]) - 1.0
			r2 = x * x + y * y
			ok = np.flatnonzero((r2 <= 1.0) & (r2 != 0.0))
			# use accepted candidates in order until `need` values are produced
			use = ok[: (need + 1) // 2]
			# log of libm (glibc), as the C++: numpy's vectorized log differs from it by 1 ulp for ~0.3% of inputs
			log_r2 = np.array([math.log(v) for v in r2[use]], dtype=np.float64)
			mult = np.sqrt(-2.0 * log_r2 / r2[use])
			vals = np.stack([y[use] * mult, x[use] * mult], axis=1).reshape(-1)
			if len(use) == 0:
				rng.take(4 * m)
				continue
			rng.take(4 * (use[-1] + 1))
			k = min(need, vals.size)
			out[filled : filled + k] = vals[:k]
			filled += k
			if vals.size > k:  # odd count: the last candidate's x·m stays saved
				self._saved = vals[k]
		out = out * self.stddev + self.mean
		return float(out[0]) if n is None else out


def _lemire32(rng: MT19937, urange_plus_1: int) -> int:
	r = int(urange_plus_1) & 0xFFFFFFFF
	product = rng() * r
	low = product & 0xFFFFFFFF
	if low < r:
		threshold = (0x100000000 - r) % r
		while low < threshold:
			product = rng() * r
			low = product & 0xFFFFFFFF
	return product >> 32


class UniformInt:
	"""std::uniform_int_distribution(a, b) with a 32-bit engine (range < 2³² - 1)."""

	def __init__(self, a: int, b: int):
		self.a, self.b = int(a), int(b)

	def __call__(self, rng: MT19937) -> int:
		urange = self.b - self.a
		if urange >= 0xFFFFFFFF:
			raise NotImplementedError('ranges of 2^32 or more are not used by the original')
		return self.a + _lemire32(rng, urange + 1)


def shuffle(values: list, rng: MT19937) -> list:
	"""std::shuffle (in place, returns the list)."""
	n = len(values)
	if n == 0:
		return values
	urngrange = 0xFFFFFFFF
	if urngrange // n >= n:
		i = 1
		if n % 2 == 0:
			j = UniformInt(0, 1)(rng)
			values[i], values[j] = values[j], values[i]
			i += 1
		while i != n:
			swap_range = i + 1
			x = UniformInt(0, swap_range * (swap_range + 1) - 1)(rng)
			p0, p1 = x // (swap_range + 1), x % (swap_range + 1)
			values[i], values[p0] = values[p0], values[i]
			i += 1
			values[i], values[p1] = values[p1], values[i]
			i += 1
		return values
	for i in range(1, n):
		j = UniformInt(0, i)(rng)
		values[i], values[j] = values[j], values[i]
	return values


class Discrete:
	"""std::discrete_distribution<int>(weights)."""

	def __init__(self, weights: Sequence[float]):
		w = np.asarray(weights, dtype=np.float64)
		total = 0.0
		for x in w:  # std::accumulate order
			total += x
		p = w / total
		self.cp = np.cumsum(p)  # std::partial_sum (left to right)
		self.cp[-1] = 1.0

	def __call__(self, rng: MT19937, n: int = None):
		u = rng.canonical(1 if n is None else n)
		idx = np.searchsorted(self.cp, u, side='left')  # std::lower_bound
		return int(idx[0]) if n is None else idx
