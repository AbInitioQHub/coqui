"""
Shared QMC post-processing helpers (coqui.dmft.qmc_postproc) on small hand-built inputs and on the
Hubbard–Holstein dimer's retarded interaction.
Serial: python -m pytest <this file> -v     (from outside the repository)
MPI:    mpirun -n 2 python -m pytest <this file> -v
"""
import numpy as np
import pytest

pytest.importorskip("triqs", reason="triqs is required for dmft tests")
pytest.importorskip("coqui._lib.iaft_module", reason="the DLR IAFT binding is required")

from triqs.gfs import Gf, MeshImFreq, MeshImTime, iOmega_n, inverse
from triqs.operators import n
from coqui.dmft.qmc_postproc import (color_table, block_offsets, color_groups, static_color_matrix,
                                     sigma_infty_hf, pi_w_from_chi, connected_by_plateau, average_blocks,
                                     average_color_matrix, eval_uniform_then_dlr, color_matrix_at_points,
                                     d0_tau_colors, fit_tau_correlator)
from coqui.dmft.triqs_utils import problem_to_triqs
from coqui.dmft.tests.hubbard_holstein_dimer import DimerParams, make_iaft, make_problem

_GF1 = [("up_0", 1), ("down_0", 1)]                                  # one orbital, size-1 blocks
_GF2 = [("up_0", 1), ("up_1", 1), ("down_0", 1), ("down_1", 1)]     # two orbitals, size-1 blocks
_GF2B = [("up", 2), ("down", 2)]                                     # two orbitals, size-2 blocks


def test_colors():
  """The colour helpers number the colours in gf_struct order, spin by block name, orbital per spin.

  Setup: _GF2 (colours 0..3 = up_0, up_1, down_0, down_1).

  Cases:
    1. color_table: block [0,1,2,3], inner [0,0,0,0], spin [0,0,1,1], orb [0,1,0,1].
    2. block_offsets: [0,1,2,3]; on _GF2B [0, 2].
    3. color_groups([[0, 2], [1, 3]]) == [[0, 2], [1, 3]]; color_groups(None) == []; on _GF2B the group
       [[0, 1]] expands to the four colours [0, 1, 2, 3].
  """
  block, inner, spin, orb = color_table(_GF2)
  assert block.tolist() == [0, 1, 2, 3] and inner.tolist() == [0, 0, 0, 0]
  assert spin.tolist() == [0, 0, 1, 1] and orb.tolist() == [0, 1, 0, 1]
  assert block_offsets(_GF2) == [0, 1, 2, 3] and block_offsets(_GF2B) == [0, 2]
  assert color_groups(_GF2, [[0, 2], [1, 3]]) == [[0, 2], [1, 3]] and color_groups(_GF2, None) == []
  assert color_groups(_GF2B, [[0, 1]]) == [[0, 1, 2, 3]]


def test_sigma_infty_hf():
  """sigma_infty_hf is the Hartree-Fock decoupling of h_int with the block density matrices plus the
  D(0) Hartree term on the diagonal.

  Setup: closed forms with rho[a, a'] = <c+_a' c_a>.

  Cases:
    1. Cross-block term U n_up n_dn (U = 2) on _GF1 with rho_up = 0.3, rho_dn = 0.7 and D(0) = -0.4 for
       every colour pair: U n_sigmabar + D(0) (n_up + n_dn), i.e. up 2*0.7 - 0.4 = 1.0 and down
       2*0.3 - 0.4 = 0.2 (the dimer's reference formula). Only the Hartree terms are live: the Fock
       contractions vanish across blocks.
    2. Intra-block term U' n_0 n_1 (U' = 1.5) in the size-2 "up" block of _GF2B with an asymmetric
       rho_up = [[0.3, 0.1], [0.2, 0.4]] and D(0) = 0: Hartree U' rho[1,1] and U' rho[0,0] on the
       diagonal, Fock -U' <c+_1 c_0> = -U' rho[0,1] at [0,1] and -U' rho[1,0] at [1,0]; the "down"
       block is zero. Pins the Fock sign and the transposition convention of rho.
  """
  h_int = 2.0 * n("up_0", 0) * n("down_0", 0)
  rho = [np.array([[0.3]], dtype=complex), np.array([[0.7]], dtype=complex)]
  sig = sigma_infty_hf(rho_blocks=rho, h_int=h_int, d_nu0=np.full((2, 2), -0.4), gf_struct=_GF1)
  assert np.allclose(sig[0], [[1.0]]) and np.allclose(sig[1], [[0.2]])

  up = 1.5 * n("up", 0) * n("up", 1)
  rho2 = [np.array([[0.3, 0.1], [0.2, 0.4]], dtype=complex), np.zeros((2, 2), dtype=complex)]
  sig2 = sigma_infty_hf(rho_blocks=rho2, h_int=up, d_nu0=np.zeros((4, 4)), gf_struct=_GF2B)
  assert np.allclose(sig2[0], 1.5 * np.array([[0.4, -0.1], [-0.2, 0.3]]))
  assert np.allclose(sig2[1], 0.0)


def test_average_blocks():
  """average_blocks replaces every block of a group by the group mean and always returns copies.

  Setup: three (1, 1) arrays a, b, c.

  Cases:
    1. groups [[0, 2]]: blocks 0 and 2 become (a + c)/2, block 1 is b.
    2. groups None: the values are unchanged and the outputs are not the input objects (the callers
       subtract from them in place).
  """
  arrays = [np.array([[1.0]]), np.array([[2.0]]), np.array([[5.0]])]
  out = average_blocks(arrays=arrays, groups=[[0, 2]])
  assert np.allclose(out[0], 3.0) and np.allclose(out[1], 2.0) and np.allclose(out[2], 3.0)
  same = average_blocks(arrays=arrays, groups=None)
  assert all(np.allclose(x, y) and x is not y for x, y in zip(same, arrays))


def test_average_color_matrix():
  """average_color_matrix mixes only the colour pairs of one class (same spin?, same orbital?, same
  block?, positions in the blocks) inside a group.

  Setup: _GF2B (size-2 blocks, so position in block, orbital and block are three different things),
  M[c1, c2] = 4 c1 + c2, one group of all four colours.

  Cases:
    1. Every class has two pairs, (c1, c2) and its spin partner (c1 + 2, c2 + 2) modulo 4: the diagonals
       [0,0]/[2,2] and [1,1]/[3,3] are averaged separately (different positions), [0,1] with [2,3],
       [0,2] with [2,0] (opposite spins, same orbital), [0,3] with [2,1]; pairs of different classes
       keep different values.
  """
  block, inner, spin, orb = color_table(_GF2B)
  M = np.arange(16, dtype=float).reshape(4, 4)
  avg = average_color_matrix(M=M, groups=[[0, 1, 2, 3]], spin=spin, orb=orb, block=block, inner=inner)
  for (c1, c2), (d1, d2) in (((0, 0), (2, 2)), ((1, 1), (3, 3)), ((0, 1), (2, 3)), ((1, 0), (3, 2)),
                             ((0, 2), (2, 0)), ((1, 3), (3, 1)), ((0, 3), (2, 1)), ((1, 2), (3, 0))):
    assert avg[c1, c2] == avg[d1, d2] == np.mean([M[c1, c2], M[d1, d2]])
  assert avg[0, 0] != avg[1, 1] and avg[0, 1] != avg[1, 0] and avg[0, 2] != avg[0, 3]


def test_connected_by_plateau():
  """connected_by_plateau subtracts the mean over the middle half of tau.

  Setup: a tau array of 8 points whose middle half is the constant 0.21 and whose ends are 0.5.

  Cases:
    1. The plateau is 0.21; the connected part is 0.29 at the ends and 0 in the middle.
  """
  q = np.full((8, 1, 1), 0.5)
  q[2:6] = 0.21
  conn, plateau = connected_by_plateau(Q_tau=q, beta=10.0)
  assert np.allclose(plateau, 0.21) and np.allclose(conn[0], 0.29) and np.allclose(conn[3], 0.0)


def test_fit_tau_correlator():
  """fit_tau_correlator limits the DLR cutoff to what the tau grid resolves, w_max = min(dlr_wmax,
  pi / (4 dtau)), and returns the fit of the correlator at that cutoff.

  Setup: beta 20, a single-pole bosonic correlator a cosh(w (tau - beta/2)) / sinh(w beta / 2) with
  a = 0.1, w = 2 (transform 2 a w / (w^2 + nu^2)), on 201 uniform points (dtau = 0.1, cap 7.85).

  Cases:
    1. dlr_wmax 20: the cap binds, the returned w_max is pi / (4 dtau), and the fit still reproduces the
       pole at nu = 0 and nu = 2 pi / beta to 1e-6 (w = 2 is well inside the cap).
    2. dlr_wmax 5: below the cap, the returned w_max is dlr_wmax.
  """
  beta, a, w = 20.0, 0.1, 2.0
  mesh = MeshImTime(beta=beta, statistic="Boson", n_tau=201)
  tau = np.array([float(t) for t in mesh])
  q = Gf(mesh=mesh, target_shape=(1, 1))
  q.data[:, 0, 0] = a * np.cosh(w * (tau - beta / 2)) / np.sinh(w * beta / 2)
  dtau = beta / 200
  q_dlr, w_max = fit_tau_correlator(g_tau=q, dlr_wmax=20.0, dlr_eps=1e-10)
  assert w_max == np.pi / (4 * dtau)
  from coqui.dmft.qmc_postproc import eval_on_points
  got = eval_on_points(g=q_dlr, n=np.array([0, 2]), stat="Boson", beta=beta)[:, 0, 0].real
  nu = np.pi * np.array([0, 2]) / beta
  assert np.allclose(got, 2 * a * w / (w**2 + nu**2), atol=1e-6)
  _, w_max5 = fit_tau_correlator(g_tau=q, dlr_wmax=5.0, dlr_eps=1e-10)
  assert w_max5 == 5.0


def test_pi_w_from_chi():
  """pi_w_from_chi gives Pi = chi (U chi - 1)^-1 and W = U - U chi U - V in the density channel, with
  U = V + D(nu), from the spin-summed, symmetrized connected colour correlator.

  Setup: static U from static_color_matrix of a density-density h_int (the colour matrix of the
  n_c n_c' coefficients), D(nu) constant in nu on three bosonic points.

  Cases:
    1. One orbital (_GF1, U = 2, D = -0.4), chi_c = c = 0.1 on all four colour pairs: chi_dd = 4c,
       U_dd = 1.6, Pi = 4c/(1.6 4c - 1), W = 1.6 - 1.6 4c 1.6 - 2 at every nu; max_cond is 1
       (a scalar). Scalar algebra only.
    2. Two orbitals (_GF2, U = 2, U' = 1.5 on every other pair, D = -0.4): chi_c[c1, c2] =
       A[orb c1, orb c2]/4 + 0.02 (c1 - c2) with A = [[0.1, 0.03], [0.03, 0.08]]; the antisymmetric
       part is removed by the symmetrization, so chi_dd = A, and Pi, W follow the 2 x 2 closed forms
       with U_dd = U' + D off the diagonal and U + D on it (solved with np.linalg.solve in the test);
       max_cond is the condition number of the pair-basis U chi - 1, i.e. of U_dd chi_dd - 1 padded
       with the unit singular values of the inert exchange rows. Pins the product order and the
       symmetrization, which a scalar cannot.
    3. Case 2 with degenerate_blk [[0, 1, 2, 3]] (every colour equivalent): chi_dd becomes the
       class average, diagonal (A00 + A11)/2 and off-diagonal A01, before the Dyson step.
  """
  h1 = 2.0 * n("up_0", 0) * n("down_0", 0)
  U1 = static_color_matrix(h_int=h1, gf_struct=_GF1)
  c = 0.1
  out = pi_w_from_chi(chi_c=np.full((3, 2, 2), c), U_c=U1, D_c=np.full((3, 2, 2), -0.4), gf_struct=_GF1,
                      degenerate_blk=None)
  chi_dd, u_full = 4 * c, 2.0 - 0.4
  assert np.allclose(out["chi_dd"][:, 0, 0], chi_dd)
  assert np.allclose(out["Pi_dd"][:, 0, 0], chi_dd / (u_full * chi_dd - 1.0))
  assert np.allclose(out["W_dd"][:, 0, 0], u_full - u_full * chi_dd * u_full - 2.0)
  assert np.isclose(out["max_cond"], 1.0)

  U, Up = 2.0, 1.5
  h2 = (U * n("up_0", 0) * n("down_0", 0) + U * n("up_1", 0) * n("down_1", 0)
        + Up * (n("up_0", 0) * n("down_1", 0) + n("up_1", 0) * n("down_0", 0)
                + n("up_0", 0) * n("up_1", 0) + n("down_0", 0) * n("down_1", 0)))
  U2 = static_color_matrix(h_int=h2, gf_struct=_GF2)
  _, _, _, orb = color_table(_GF2)
  A = np.array([[0.1, 0.03], [0.03, 0.08]])
  chi_c = np.zeros((3, 4, 4))
  for c1 in range(4):
    for c2 in range(4):
      chi_c[:, c1, c2] = A[orb[c1], orb[c2]] / 4 + 0.02 * (c1 - c2)
  d = -0.4
  out2 = pi_w_from_chi(chi_c=chi_c, U_c=U2, D_c=np.full((3, 4, 4), d), gf_struct=_GF2, degenerate_blk=None)
  U_dd = np.array([[U, Up], [Up, U]]) + d
  denom = U_dd @ A - np.eye(2)
  Pi = np.linalg.solve(denom.T, A.T).T                                # A denom^-1
  W = U_dd - U_dd @ A @ U_dd - np.array([[U, Up], [Up, U]])
  for w in range(3):
    assert np.allclose(out2["chi_dd"][w], A)
    assert np.allclose(out2["Pi_dd"][w], Pi) and np.allclose(out2["W_dd"][w], W)
  singular = np.concatenate([np.linalg.svd(denom, compute_uv=False), [1.0, 1.0]])
  assert np.isclose(out2["max_cond"], singular.max() / singular.min())

  out3 = pi_w_from_chi(chi_c=chi_c, U_c=U2, D_c=np.full((3, 4, 4), d), gf_struct=_GF2,
                       degenerate_blk=[[0, 1, 2, 3]])
  A_avg = np.array([[0.09, 0.03], [0.03, 0.09]])
  Pi_avg = np.linalg.solve((U_dd @ A_avg - np.eye(2)).T, A_avg.T).T
  assert np.allclose(out3["chi_dd"][0], A_avg) and np.allclose(out3["Pi_dd"][0], Pi_avg)


def test_eval_uniform_then_dlr():
  """eval_uniform_then_dlr reads the stored values inside the uniform mesh and the DLR of g beyond it.

  Setup: one-pole G = 1/(iw - 0.3) on a uniform mesh of n_iw = 40 (beta 10).

  Cases:
    1. Points inside the mesh (negative ones included) are the stored values exactly.
    2. Points beyond it come from the DLR of G (w_max 2, whose nodes the mesh must cover) and match the
       pole within 1e-8.
  """
  mesh = MeshImFreq(beta=10.0, statistic="Fermion", n_iw=40)
  g = Gf(mesh=mesh, target_shape=(1, 1))
  g << inverse(iOmega_n - 0.3)
  n_inside = np.array([-5, -1, 1, 5, 79])                      # iw = n pi / beta, odd n; 79 -> k = 39 (last point)
  n_beyond = np.array([81, 101, 121])
  got_in = eval_uniform_then_dlr(g=g, n=n_inside, stat="Fermion", beta=10.0, w_max=2.0, eps=1e-10)
  for i, nn in enumerate(n_inside):
    assert got_in[i, 0, 0] == g.data[(nn - 1) // 2 + 40, 0, 0]
  got_out = eval_uniform_then_dlr(g=g, n=n_beyond, stat="Fermion", beta=10.0, w_max=2.0, eps=1e-10)
  exact = 1.0 / (1j * np.pi * n_beyond / 10.0 - 0.3)
  assert np.allclose(got_out[:, 0, 0], exact, atol=1e-8)


@pytest.fixture(scope="module")
def dimer_u():
  """The dimer's retarded interaction as the Block2Gf problem_to_triqs builds, with its parameters."""
  params = DimerParams()
  iaft = make_iaft(params, wmax=20.0, eps=1e-10)
  problem = make_problem(params, iaft, dlr_wmax=20.0, dlr_eps=1e-10, degenerate_blk=((0, 1),))
  inputs = problem_to_triqs(problem, h_int_kind="density", screen_j=False, real_hamiltonian=True)
  return params, iaft, problem, inputs


def test_color_matrix_at_points(dimer_u):
  """color_matrix_at_points evaluates a density-density Block2Gf on the bosonic DLR mesh at the given
  bosonic Matsubara integers, iv = n pi / beta (even n).

  Setup: the dimer's u(iv) = -2 g^2 w0 / (w0^2 + v^2), the same for every colour pair.

  Cases:
    1. At the positive half of the IAFT bosonic mesh every (c1, c2) entry equals the closed form to
       1e-8 (the DLR of the single pole is exact to its eps).
  """
  params, iaft, problem, inputs = dimer_u
  wn_b = iaft.wn_mesh('b', positive_only=True)
  nu = np.pi * wn_b / params.beta
  exact = -2.0 * params.g**2 * params.omega0 / (params.omega0**2 + nu**2)
  D_c = color_matrix_at_points(block2gf=inputs.u_weiss_iw, gf_struct=problem.gf_struct, n_b=wn_b,
                               beta=params.beta)
  assert D_c.shape == (len(wn_b), 2, 2)
  for c1 in range(2):
    for c2 in range(2):
      assert np.allclose(D_c[:, c1, c2], exact, atol=1e-8)


def test_d0_tau_colors(dimer_u):
  """d0_tau_colors returns D(tau) of a density-density Block2Gf on a uniform [0, beta] grid.

  Setup: the dimer's single boson, D(tau) = -g^2 cosh(w0 (tau - beta/2)) / sinh(w0 beta / 2).

  Cases:
    1. On 1001 points every colour pair equals the closed form to 1e-8.
  """
  params, iaft, problem, inputs = dimer_u
  tau = np.linspace(0.0, params.beta, 1001)
  exact = -params.g**2 * np.cosh(params.omega0 * (tau - params.beta / 2)) / np.sinh(params.omega0 * params.beta / 2)
  d_tau = d0_tau_colors(block2gf=inputs.u_weiss_iw, gf_struct=problem.gf_struct, n_tau=1001)
  assert d_tau.shape == (1001, 2, 2)
  for c1 in range(2):
    for c2 in range(2):
      assert np.allclose(d_tau[:, c1, c2], exact, atol=1e-8)
