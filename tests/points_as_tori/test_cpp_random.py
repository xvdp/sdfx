"""points_as_tori.cpp_random replays std::mt19937 + libstdc++ distributions bit-exactly (g++ reference program)."""

import os
import shutil
import subprocess

import numpy as np
import pytest

from points_as_tori import cpp_random as cr

SRC = os.path.join(os.path.dirname(__file__), 'reference', 'rng_reference.cpp')


def port_lines():
	rng = cr.MT19937(42)
	u01, uab = cr.UniformReal(0, 1), cr.UniformReal(-2, 3)
	n1, n2 = cr.Normal(0, 1), cr.Normal(1, 0.5)
	lines = [['U'] + list(u01(rng, 5)), ['A'] + list(uab(rng, 7)), ['N'] + list(n1(rng, 9))]
	m = []
	for _ in range(4):
		m += [n2(rng), n1(rng)]
	lines.append(['M'] + m)
	ints = [cr.UniformInt(0, 9)(rng) for _ in range(10)] + [cr.UniformInt(0, 1)(rng) for _ in range(5)]
	ints += [cr.UniformInt(0, 100000)(rng) for _ in range(5)]
	lines.append(['I'] + ints)
	for n in (10, 7, 1, 2, 70000):
		v = cr.shuffle(list(range(n)), rng)
		lines.append(['S', n, sum(x * (i + 1) for i, x in enumerate(v))] + v[:10])
	lines.append(['D'] + list(cr.Discrete([0.5, 2.0, 0.0, 1.25, 3.0, 1e-3])(rng, 20)))
	lines.append(['U'] + list(u01(rng, 3)))
	return lines


@pytest.mark.skipif(shutil.which('g++') is None, reason='needs g++ (libstdc++)')
def test_matches_libstdcxx(tmp_path):
	exe = str(tmp_path / 'rng')
	subprocess.run(['g++', '-O2', '-std=c++17', SRC, '-o', exe], check=True)
	ref = subprocess.run([exe], capture_output=True, text=True, check=True).stdout.strip().split('\n')
	ours = port_lines()
	assert len(ref) == len(ours)
	for a, b in zip(ours, ref):
		bt = b.split()
		assert a[0] == bt[0]
		if a[0] in 'UANM':
			assert np.array_equal(np.array(a[1:], float), np.array(bt[1:], float)), a[0]
		else:
			assert [str(int(x)) for x in a[1:]] == bt[1:], a[0]


def test_mt19937_standard_value():
	"""C++ standard [rand.predef]: the 10000th output of a default-constructed mt19937 (seed 5489) is 4123659995."""
	assert int(cr.MT19937(5489).take(10000)[-1]) == 4123659995


def test_normal_saved_value_per_object():
	"""Two normal_distribution objects on one engine keep separate saved values (as libstdc++)."""
	a, b = cr.MT19937(1), cr.MT19937(1)
	n1, n2 = cr.Normal(), cr.Normal()
	x = [n1(a), n2(a), n1(a), n2(a)]
	m = cr.Normal()
	y = m(b, 2)
	assert x[0] == y[0] and x[2] == y[1]  # n1 returns y·m then its saved x·m


EIGEN = os.path.join(os.path.dirname(__file__), '..', '..', 'ext', 'points-as-tori', 'deps', 'fcpw', 'deps', 'eigen')

BASIS_SRC = r"""
#include <random>
#include <cstdio>
#include <Eigen/Core>
#include <Eigen/Geometry>
// verbatim from src/cpp/shape_2d.cpp
std::pair<Eigen::Vector3d, Eigen::Vector3d> orthonormal_basis_random(const Eigen::Vector3d& n, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution dist{0.0, 1.0};
    Eigen::Vector3d n_hat = n / n.norm();
    Eigen::Vector3d s(dist(rng), dist(rng), dist(rng));
    s -= s.dot(n_hat) * n_hat;
    s /= s.norm();
    Eigen::Vector3d t = n_hat.cross(s);
    return {s, t};
}
int main() {
    for (int seed = 0; seed < 5; seed++) {
        auto st = orthonormal_basis_random(Eigen::Vector3d(0.3, -0.2, 0.9), seed);
        printf("%.17g %.17g %.17g %.17g %.17g %.17g\n", st.first.x(), st.first.y(), st.first.z(),
               st.second.x(), st.second.y(), st.second.z());
    }
}
"""


@pytest.mark.skipif(shutil.which('g++') is None or not os.path.isdir(EIGEN), reason='needs g++ and the Eigen submodule')
def test_orthonormal_basis_random_draw_order(tmp_path):
	"""g++ evaluates the three dist(rng) constructor arguments right to left: s = (z3, z2, z1)."""
	import torch

	import points_as_tori as pat
	from points_as_tori import config
	from points_as_tori.shape_2d import orthonormal_basis_random

	src = tmp_path / 'basis.cpp'
	src.write_text(BASIS_SRC)
	exe = str(tmp_path / 'basis')
	subprocess.run(['g++', '-O2', '-std=c++17', f'-I{EIGEN}', str(src), '-o', exe], check=True)
	ref = np.array(
		[
			[float(v) for v in line.split()]
			for line in subprocess.run([exe], capture_output=True, text=True).stdout.split('\n')
			if line
		]
	)
	config.set_device('cpu', torch.float64, force=True)
	try:
		n = torch.tensor([0.3, -0.2, 0.9], dtype=torch.float64)
		for seed in range(5):
			s, t = orthonormal_basis_random(n, seed)
			np.testing.assert_allclose(torch.cat([s, t]).numpy(), ref[seed], rtol=0, atol=1e-15)
	finally:
		config.set_device('cpu', torch.float32, force=True)
	assert pat is not None
