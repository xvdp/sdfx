"""
points_as_tori.explain and the network's attention outputs: same coefficients with and without attention, attention
and rollout rows summing to 1, the network's local frame = the fit's, least squares exact on quadratic data,
gradient attribution = finite differences, PointsAsTori.curvatures / explain consistent with its tori.
"""

import numpy as np
import pytest
import torch

import points_as_tori as pat
from points_as_tori import config
from points_as_tori.explain import (
	attention_rollout,
	gradient_attribution,
	least_squares_forms,
	neighborhood,
)
from points_as_tori.network import FundamentalFormPredictor, orthonormal_basis_with_rotation
from points_as_tori.shape_2d import orthonormal_basis
from points_as_tori.torus_distance import _principal_curvatures_from_forms, get_neighbors
from helpers import TORCH_MODEL, read_ply


@pytest.fixture(scope='module')
def tooth():
	config.set_device('cpu', torch.float32, force=True)
	p, n = read_ply('tooth.ply')
	model, k = FundamentalFormPredictor.load_saved_model(TORCH_MODEL)
	return config.as_wide(p, 3), config.as_wide(n, 3), model.eval(), k


def test_attention_outputs(tooth):
	P, N, model, k = tooth
	idx = get_neighbors(P, k)[:64]
	with torch.no_grad():
		plain = model(P[:, idx], N[:, idx])
		coeffs, attention = model(P[:, idx], N[:, idx], return_attention=True)
	assert len(attention) == len(model.blocks) and attention[0].shape == (64, model.blocks[0].attention.num_heads, k, k)
	assert (plain - coeffs).abs().max() < 1e-5  # explicit softmax vs scaled_dot_product_attention: float32 rounding
	assert (attention[0].sum(-1) - 1).abs().max() < 1e-5
	rollout = attention_rollout(attention)
	assert rollout.shape == (64, k) and (rollout.sum(-1) - 1).abs().max() < 1e-5 and (rollout >= 0).all()


def test_network_frame_is_the_fit_frame(tooth):
	P, N, model, k = tooth
	n = N / N.norm(dim=0)
	s1, t1 = orthonormal_basis(n)
	s2, t2 = orthonormal_basis_with_rotation(n, 0.0)
	assert torch.equal(s1, s2) and torch.equal(t1, t2)


def test_least_squares_exact_on_a_quadratic():
	"""Points on the graph of a known Q over the tangent frame of a tilted normal: the fit returns Q."""
	rng = np.random.default_rng(0)
	c_true = torch.tensor([0.0, 0.1, -0.2, 0.3, 0.8, -0.5], dtype=torch.float64)  # (a00, a01, a10, a11, a02, a20)
	n0 = torch.tensor([0.2, 0.3, 0.93], dtype=torch.float64)
	n0 = n0 / n0.norm()
	s_hat, t_hat = (v[:, 0] for v in orthonormal_basis(n0[:, None]))
	st = torch.tensor(rng.uniform(-0.3, 0.3, (64, 2)))
	st[0] = 0
	s, t = st[:, 0], st[:, 1]
	q = c_true[0] + c_true[2] * s + c_true[1] * t + c_true[5] * s * s + c_true[3] * s * t + c_true[4] * t * t
	X = (s[:, None] * s_hat + t[:, None] * t_hat + q[:, None] * n0).T
	N = n0[:, None].expand(3, 64)
	for m in (6, 64):
		est = least_squares_forms(X, N, torch.arange(m)[None])[:, 0]
		assert (est - c_true).abs().max() < 1e-10, m


def test_gradient_attribution_matches_finite_differences(tooth):
	P, N, model, k = tooth
	m64 = FundamentalFormPredictor.load_saved_model(TORCH_MODEL)[0].double().eval()
	idx = neighborhood(P, 100, k)
	pos, nrm = P[:, idx][:, None, :].double(), N[:, idx][:, None, :].double()
	g = gradient_attribution(m64, pos, nrm)[0]
	assert torch.isfinite(g).all()

	def kappas(p):
		with torch.no_grad():
			c = m64(p, nrm)
			n = nrm[:, :, 0] / nrm[:, :, 0].norm(dim=0)
			s, t = orthonormal_basis(n)
			_, a, b, _, _, _ = _principal_curvatures_from_forms(n, c, s, t, True)
		return torch.stack([a[0], b[0]])

	h = 1e-6
	for j in (0, 1, 7, 40):
		jac = []
		for axis in range(3):
			pp, pm = pos.clone(), pos.clone()
			pp[axis, 0, j] += h
			pm[axis, 0, j] -= h
			jac.append((kappas(pp) - kappas(pm)) / (2 * h))
		fd = torch.stack(jac).norm()
		assert abs(g[j] - fd) < 1e-5 * max(1.0, float(fd)), (j, float(g[j]), float(fd))


def test_points_as_tori_curvatures_and_explain(tooth):
	P, N, model, k = tooth
	sub = torch.arange(0, P.shape[1], 4)  # 512 points: a sparser cloud, still fitted by the same network
	sdf = pat.PointsAsTori(P[:, sub].T, N[:, sub].T)
	frame = sdf.curvatures()
	c, ax, R, r = sdf.get_tori()
	# the fit's tori: r = 1/|κ_max|, R = 1/|κ_min| - sgn(κ_max κ_min) r
	k_min, k_max = frame['kappa_min'], frame['kappa_max']
	assert torch.allclose(r.abs(), 1 / k_max.abs(), rtol=1e-5)
	assert torch.allclose(R, 1 / k_min.abs() - torch.sign(k_min * k_max) * r.abs(), rtol=1e-4, atol=1e-6)
	ex = sdf.explain(10)
	assert ex['indices'][0] == 10 and len(ex['indices']) == sdf.network_k
	expect = get_neighbors(sdf._tdf.points, sdf.network_k)[10]
	assert set(ex['indices'].tolist()) == set(expect.tolist())
	assert abs(ex['kappa_min'] - float(k_min[10])) < 1e-4 * max(1, abs(float(k_min[10])))
	assert set(ex['least_squares']) == {6, sdf.network_k}
