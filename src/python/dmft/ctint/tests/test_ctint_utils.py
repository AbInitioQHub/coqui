"""
CT-INT's own helpers (coqui.dmft.ctint.ctint_utils): the D0 channel order, the signed alpha tensor and
the M(tau) route to G.
Serial: python -m pytest <this file> -v     (from outside the repository)
MPI:    mpirun -n 2 python -m pytest <this file> -v
"""
import numpy as np
import pytest

pytest.importorskip("triqs", reason="triqs is required for dmft tests")

from triqs.gfs import Gf, BlockGf, MeshImTime, MeshDLRImFreq
from triqs.operators import n
from coqui.dmft.ctint.ctint_utils import d0_channel_colors, signed_alpha, g_from_m_tau, resolve_tau_grids

_GF2 = [("up_0", 1), ("up_1", 1), ("down_0", 1), ("down_1", 1)]     # two orbitals, size-1 blocks


def test_d0_channel_colors():
  """d0_channel_colors lists the D0 vertices in triqs_ctint's label order, blocks then positions,
  row-major, and refuses blocks of unequal size.

  Setup: _GF2; a gf_struct mixing sizes 1 and 2.

  Cases:
    1. _GF2 gives the 16 colour pairs (b1, b2), b1 outer.
    2. Size-2 blocks [("up", 2), ("down", 2)]: (block b1, position a) -> colour 2 b1 + a, so the pairs
       are (0,0), (0,1), (1,0), (1,1), (0,2), ... in (b1, b2, a, a') order.
    3. Unequal sizes raise ValueError naming "equal size".
  """
  assert d0_channel_colors(_GF2) == [(b1, b2) for b1 in range(4) for b2 in range(4)]
  pairs = d0_channel_colors([("up", 2), ("down", 2)])
  assert pairs == [(2 * b1 + a, 2 * b2 + ap) for b1 in range(2) for b2 in range(2) for a in range(2) for ap in range(2)]
  with pytest.raises(ValueError, match="equal size"):
    d0_channel_colors([("up_0", 1), ("up_1", 2), ("down_0", 1), ("down_1", 2)])


def test_signed_alpha():
  """signed_alpha places one shift pair per vertex in triqs_ctint's layout, opposite directions for a
  repulsive coupling, the same for an attractive one.

  Setup: _GF2; h_int = 2 n_up0 n_dn0 + 3 n_up1 n_dn1 - 1 n_up0 n_dn1 (two repulsive and one attractive
  density-density term); D(tau) channels: all 16 colour pairs of _GF2, each a constant -0.5
  (attractive, as a retarded GW+EDMFT coupling is).

  Cases:
    1. center 0.5, delta 0.51: shape (3 + 16, 2, 2, 2); the cross entries [., 0, 1] and [., 1, 0] are
       zero; for a repulsive term alpha[l, 0, 0, s] = 0.5 - sgn_s 0.51 and alpha[l, 1, 1, s] =
       0.5 + sgn_s 0.51 (opposite directions, sgn_s = +1, -1 for s = 0, 1); for the attractive term
       and for an attractive D0 channel both entries move in the same direction, 0.5 + sgn_s 0.51;
       the s-average of every entry is the centre 0.5 (what the library's compensation reads off).
    2. A D(tau) channel that changes sign, [0.9, -0.5, -0.5, -0.5, -0.5] (a non-causal Weiss field):
       "dominant" means the largest |D(tau)|, here positive, so the channel is treated as repulsive
       (opposite directions) although its mean is negative; a warning, not an error. A channel that
       is numerically zero counts as attractive.
    3. A non-density-density term (c+_up0 c+_dn0 c_dn1 c_up1) raises ValueError.
  """
  pairs = d0_channel_colors(_GF2)
  h_int = (2.0 * n("up_0", 0) * n("down_0", 0) + 3.0 * n("up_1", 0) * n("down_1", 0)
           - 1.0 * n("up_0", 0) * n("down_1", 0))
  channels = [np.full(5, -0.5) for _ in pairs]
  alpha = signed_alpha(h_int=h_int, d0_channels=channels, center=0.5, delta=0.51)
  assert alpha.shape == (3 + 16, 2, 2, 2)
  assert np.all(alpha[:, 0, 1, :] == 0.0) and np.all(alpha[:, 1, 0, :] == 0.0)
  coefs = [float(np.real(coef)) for _, coef in h_int]
  repulsive = [l for l, coef in enumerate(coefs) if coef > 0]
  attractive = [l for l, coef in enumerate(coefs) if coef < 0]
  assert len(repulsive) == 2 and len(attractive) == 1
  for s, sgn in enumerate((1.0, -1.0)):
    assert np.allclose(alpha[repulsive, 0, 0, s], 0.5 - sgn * 0.51)       # repulsive: opposite directions
    assert np.allclose(alpha[repulsive, 1, 1, s], 0.5 + sgn * 0.51)
    assert np.allclose(alpha[attractive, 0, 0, s], 0.5 + sgn * 0.51)      # attractive: same direction
    assert np.allclose(alpha[attractive, 1, 1, s], 0.5 + sgn * 0.51)
    assert np.allclose(alpha[3:, 0, 0, s], 0.5 + sgn * 0.51)              # attractive D0: same direction
    assert np.allclose(alpha[3:, 1, 1, s], 0.5 + sgn * 0.51)
  assert np.allclose(alpha.mean(axis=-1)[:, 0, 0], 0.5) and np.allclose(alpha.mean(axis=-1)[:, 1, 1], 0.5)

  mixed = list(channels)
  mixed[3] = np.array([0.9, -0.5, -0.5, -0.5, -0.5])
  alpha_mixed = signed_alpha(h_int=h_int, d0_channels=mixed, center=0.5, delta=0.51)
  assert np.allclose(alpha_mixed[3 + 3, 0, 0, 0], 0.5 - 0.51) and np.allclose(alpha_mixed[3 + 3, 1, 1, 0], 0.5 + 0.51)
  zero = list(channels)
  zero[0] = np.zeros(5)
  alpha_zero = signed_alpha(h_int=h_int, d0_channels=zero, center=0.5, delta=0.51)
  assert np.allclose(alpha_zero[3, 0, 0, 0], 0.5 + 0.51)           # zero channel: attractive direction

  from triqs.operators import c_dag, c
  with pytest.raises(ValueError, match="density-density"):
    signed_alpha(h_int=c_dag("up_0", 0) * c_dag("down_0", 0) * c("down_1", 0) * c("up_1", 0),
                 d0_channels=channels, center=0.5, delta=0.51)


def test_resolve_tau_grids():
  """resolve_tau_grids sizes the G0 and D0 tau grids from the DLR mesh unless overridden.

  Setup: closed forms.

  Cases:
    1. beta 500, dlr_wmax 20: max(10001, 2 500 20 + 1) = 20001 for both grids.
    2. beta 20, dlr_wmax 20: the floor 10001 for both.
    3. Overrides: n_tau 4001 alone sets both; n_tau_bosonic 2001 with it keeps 4001 / 2001.
  """
  assert resolve_tau_grids(beta=500.0, dlr_wmax=20.0) == (20001, 20001)
  assert resolve_tau_grids(beta=20.0, dlr_wmax=20.0) == (10001, 10001)
  assert resolve_tau_grids(beta=20.0, dlr_wmax=20.0, n_tau=4001) == (4001, 4001)
  assert resolve_tau_grids(beta=20.0, dlr_wmax=20.0, n_tau=4001, n_tau_bosonic=2001) == (4001, 2001)


class Solver:
  """Stand-in for a solved triqs_ctint.Solver: the three members g_from_m_tau reads."""

  def __init__(self, M_tau, M_hartree, G0_shift_iw):
    self.M_tau = M_tau
    self.M_hartree = M_hartree
    self.G0_shift_iw = G0_shift_iw


def test_g_from_m_tau():
  """g_from_m_tau builds G = G0s + G0s (M + M_hartree) G0s per block from the DLR fit of M(tau).

  Setup: beta 10, one block "up_0" of size 1, G0s = 1/(iw - 0.3) on the fermionic DLR mesh
  (w_max 5, eps 1e-10), M(tau) on 201 uniform points; Matsubara integers n = [-3, -1, 1, 3, 21].

  Cases:
    1. M(tau) = 0, M_hartree = 0.7: G = G0s + 0.7 G0s^2 at every point to 1e-10 (the fit of a zero
       function is zero).
    2. M(tau) = -exp(-0.5 tau)/(1 + exp(-0.5 beta)) (a one-pole fermionic function, M(iw) =
       1/(iw - 0.5)), M_hartree = 0: G = G0s + G0s M(iw) G0s to 1e-8 (the DLR fit of the pole).
  """
  beta, gf_struct = 10.0, [("up_0", 1)]
  mesh_f = MeshDLRImFreq(beta=beta, statistic="Fermion", w_max=5.0, eps=1e-10, symmetrize=True)
  g0s = Gf(mesh=mesh_f, target_shape=(1, 1))
  g0s.data[:, 0, 0] = 1.0 / (1j * (2 * np.array([w.index for w in mesh_f]) + 1) * np.pi / beta - 0.3)
  G0_shift_iw = BlockGf(name_list=["up_0"], block_list=[g0s])
  mesh_t = MeshImTime(beta=beta, statistic="Fermion", n_tau=201)
  tau = np.array([float(t) for t in mesh_t])
  n_f = np.array([-3, -1, 1, 3, 21])
  iw = 1j * np.pi * n_f / beta
  g0 = 1.0 / (iw - 0.3)

  m_zero = Gf(mesh=mesh_t, target_shape=(1, 1))
  m_zero.data[:] = 0.0
  solver = Solver(M_tau=BlockGf(name_list=["up_0"], block_list=[m_zero]), M_hartree=[np.array([[0.7]])],
                  G0_shift_iw=G0_shift_iw)
  got = g_from_m_tau(S=solver, gf_struct=gf_struct, n_f=n_f, beta=beta, w_max=5.0, eps=1e-10)
  assert np.allclose(got[0][:, 0, 0], g0 + 0.7 * g0**2, atol=1e-10)

  m_pole = Gf(mesh=mesh_t, target_shape=(1, 1))
  m_pole.data[:, 0, 0] = -np.exp(-0.5 * tau) / (1.0 + np.exp(-0.5 * beta))
  solver = Solver(M_tau=BlockGf(name_list=["up_0"], block_list=[m_pole]), M_hartree=[np.zeros((1, 1))],
                  G0_shift_iw=G0_shift_iw)
  got = g_from_m_tau(S=solver, gf_struct=gf_struct, n_f=n_f, beta=beta, w_max=5.0, eps=1e-10)
  assert np.allclose(got[0][:, 0, 0], g0 + g0 * (1.0 / (iw - 0.5)) * g0, atol=1e-8)
