"""
CT-HYB backend with matrix-valued blocks (coqui.dmft.cthyb.CthybSolver, offdiag_delta): h0 and
Delta off-diagonal inside a block, checked against the exactly solvable two-orbital Kanamori–Holstein
impurity.
Serial: python -m pytest <this file> -v     (from outside the repository)
MPI:    mpirun -n 2 python -m pytest <this file> -v
"""
import numpy as np
import pytest

pytest.importorskip("triqs", reason="triqs is required for dmft tests")
pytest.importorskip("triqs_cthyb", reason="triqs_cthyb is required for this backend")
pytest.importorskip("coqui._lib.iaft_module", reason="the DLR IAFT binding is required")

from triqs.utility import mpi
from coqui.dmft.cthyb import CthybSolver
from coqui.dmft.cthyb.cthyb_solver import _warn_offdiag_h0_without_delta
from coqui.dmft.cthyb.cthyb_utils import density_from_g_tau, density_matrix_sector_mask
from coqui.dmft.triqs_utils import problem_to_triqs
from coqui.dmft.tests.kanamori_holstein_dimer import (KanamoriParams, KanamoriED, make_iaft, make_problem,
                                                      GF_STRUCT, GF_STRUCT_BLOCKS, COLOR_SPIN)

# Monte Carlo budget (n_cycles is the total over ranks; ~19 s per production solve on 4 ranks,
# expansion order ~36, sign 0.8) and the tolerances. Measured once on 4 ranks at this budget over
# the base seeds 11 and 12 (a third seed, 13, reproduced seed 12 to three digits in every quantity,
# so it counts as the same sample): density off by -0.012 and +0.033; |G_00(iw0)| by 0.3 % and 0.3 %; the
# off-diagonal G_01(iw0) = 0.072 by 0.006 and 0.0025; Sigma_dyn_00(iw0) by 0.025 and 0.006;
# Sigma_dyn_01(iw0) = 0.025 by 0.065 and 0.031, i.e. not resolved at this budget (the Dyson step
# amplifies the noise of G_01 by G^-2) and bounded only; Sigma_infty_00 by 0.003, Sigma_infty_01 =
# 0.054 by 0.0016 (from the solver's density matrix, which holds the off-diagonal element because the
# hopping t mixes the symmetry sectors of h_loc); W within 33 % and 23 % of ED at iv0, iv1 and Pi off
# by more than 100 % (the density correlator of a Kanamori h_loc is measured by the direct sweep,
# at sign 0.8; Pi is only checked for shape and finiteness). The tolerances are at least twice the
# larger of the two samples; the test passes at 1 and 2 ranks with them (different samples, since
# n_cycles is the total over ranks and the seed is rank-dependent).
_SOLVER = {"n_cycles": 200000, "n_warmup_cycles": 5000, "length_cycle": 50, "random_seed": 11,
           "n_tau_bosonic": 10001, "dyn_n_l": 100, "h_int": "kanamori"}
_TOL = {"density": 0.07, "g00_rel": 0.01, "g01": 0.015, "sigma00": 0.05, "sigma01_bound": 0.13,
        "sigma_infty00": 0.01, "sigma_infty01": 5e-3, "w_rel": 0.7}
_CHEAP = {"n_cycles": 20000, "n_warmup_cycles": 1000, "length_cycle": 20, "random_seed": 11,
          "n_tau_bosonic": 10001, "dyn_n_l": 100, "h_int": "kanamori"}


def _i0(wn):
  """Position of the lowest positive frequency."""
  return int(np.argmin(np.where(wn > 0, wn, np.inf)))


def test_cthyb_offdiag_delta(monkeypatch):
  """CthybSolver.solve with one 2x2 block per spin reproduces the exact solution of an impurity whose
  h0 and Delta are both off-diagonal, and warns about a block whose h0 couples what Delta does not.

  Setup: the two-orbital Kanamori–Holstein impurity (beta 20, U 2, U' 0.6, J 0.7, V 1, one retarded
  pole) with the inter-orbital hopping t = 0.15 and the cross hybridization V_01 = 0.15, in the
  layout GF_STRUCT_BLOCKS [("up_0", 2), ("down_0", 2)] with degenerate_blk [[0, 1]]; the budget
  _SOLVER; CthybSolver instances for comm_size = the MPI size.

  Cases (2 and 3 are independent of 1):
    1. Production solve, h_int "kanamori": validate(problem); the density, the 2x2 matrices G(iw0),
       Sigma_dyn(iw0) and Sigma_infty of the up block within _TOL of the ED colour matrices (see the
       header: G_01 and Sigma_infty_01 are the off-diagonal checks, a transposed, dropped or
       unmeasured element would miss by their full size, 0.07 and 0.054; Sigma_dyn_01 is bounded
       only); the down block equal to the up block (degenerate_blk); G symmetric (G_10 = G_01 to
       1e-12: the Hamiltonian is real, and make_hermitian keeps that); W at iv0 and iv1 within
       _TOL["w_rel"] of ED; Pi of shape (nw_b, 2, 2) and finite. Two things this case cannot see:
       the density matrix holds every element here (t mixes the sectors of h_loc), so the
       G(tau = 0-) fallback of solve is not taken (case 4); and the retarded interaction of
       make_problem is the same for every colour pair, so a permuted D0_tau fill would give the same
       W (test_fill_d0_tau).
    2. _warn_offdiag_h0_without_delta (app_log replaced by a recorder) on the converted inputs
       logs nothing for a block with h0 and Delta both diagonal (t = V_01 = 0), for h0 and Delta
       both off-diagonal (the problem of case 1), and for size-1 blocks whatever h0 does between
       them (t = 0.15 in GF_STRUCT); for t = 0.15 and V_01 = 0 in the block layout it logs one
       [WARNING] per spin block naming "Delta is diagonal" (the hybridization expansion cannot
       measure the off-diagonal G of such a block, so Sigma would be wrong).
    3. The fallback of solve: the impurity with t = 0 and V_01 = 0.3 (c+_0 c_1 changes the sector
       of h_loc, so the density matrix lacks the off-diagonal element) solved at the cheap budget
       _CHEAP (no accuracy claim): solver_data["rho_blocks"] holds, in every block, the diagonal of
       the solver's density matrix (orbital_occupations) and the off-diagonal of
       density_from_g_tau(solver_data["G_tau"]) to 1e-12, i.e. the elements the sector mask marks
       missing come from G(tau = 0-) and the others do not.
  """
  params = KanamoriParams(t=0.15, V_01=0.15)
  iaft = make_iaft(params, wmax=20.0, eps=1e-10)
  ref = KanamoriED(params).reference(iaft, n_bose=2)
  problem = make_problem(params, iaft, dlr_wmax=20.0, dlr_eps=1e-10, degenerate_blk=((0, 1),),
                         gf_struct=GF_STRUCT_BLOCKS)
  i0 = _i0(ref["wn_f"])
  up = [c for c in range(4) if COLOR_SPIN[c] == 0]
  g_ex = ref["G_iw_full"][i0][np.ix_(up, up)]
  sigma_ex = ref["Sigma_dyn_iw_full"][i0][np.ix_(up, up)]
  sigma_infty_ex = ref["Sigma_infty_full"][np.ix_(up, up)]
  assert abs(g_ex[0, 1]) > 0.05 and abs(sigma_infty_ex[0, 1]) > 0.05

  result = CthybSolver(dict(_SOLVER), comm_size=mpi.size).solve(problem, occupations_only=False,
                                                              chkpt=None)
  result.validate(problem)
  assert abs(result.density - ref["occupations"].sum()) < _TOL["density"]
  g = result.G_iw_data[0][i0]
  assert abs(abs(g[0, 0]) - abs(g_ex[0, 0])) / abs(g_ex[0, 0]) < _TOL["g00_rel"]
  assert abs(g[0, 1] - g_ex[0, 1]) < _TOL["g01"]
  assert abs(g[1, 0] - g[0, 1]) < 1e-12
  sigma = result.Sigma_iw_data[0][i0]
  assert abs(sigma[0, 0] - sigma_ex[0, 0]) < _TOL["sigma00"]
  assert abs(sigma[0, 1] - sigma_ex[0, 1]) < _TOL["sigma01_bound"]
  sigma_infty = result.Sigma_infty[0]
  assert abs(sigma_infty[0, 0] - sigma_infty_ex[0, 0]) < _TOL["sigma_infty00"]
  assert abs(sigma_infty[0, 1] - sigma_infty_ex[0, 1]) < _TOL["sigma_infty01"]
  assert np.allclose(result.G_iw_data[1], result.G_iw_data[0])
  assert np.allclose(result.Sigma_iw_data[1], result.Sigma_iw_data[0])
  assert np.allclose(result.Sigma_infty[1], result.Sigma_infty[0])
  w = result.W_iw_data[0]
  for index in (0, 1):
    assert np.abs(w[index] - ref["W_iw"][index]).max() / np.abs(ref["W_iw"][index]).max() < _TOL["w_rel"]
  pi = result.Pi_iw_data[0]
  assert pi.shape == (len(iaft.wn_mesh('b', positive_only=True)), 2, 2) and np.isfinite(pi).all()

  def inputs_of(t, v_01, gf_struct):
    prob = make_problem(KanamoriParams(t=t, V_01=v_01), iaft, dlr_wmax=20.0, dlr_eps=1e-10,
                        degenerate_blk=None, gf_struct=gf_struct)
    return problem_to_triqs(prob, h_int_kind="kanamori", screen_j=False, real_hamiltonian=True)
  logged = []
  monkeypatch.setattr("coqui.dmft.cthyb.cthyb_solver.app_log", lambda level, msg: logged.append(msg))
  for t, v_01, gf_struct in ((0.0, 0.0, GF_STRUCT_BLOCKS), (0.15, 0.15, GF_STRUCT_BLOCKS),
                             (0.15, 0.0, GF_STRUCT)):
    _warn_offdiag_h0_without_delta(inputs_of(t, v_01, gf_struct), gf_struct, h0_tol=1e-6, delta_rel_tol=1e-8)
  assert logged == []
  _warn_offdiag_h0_without_delta(inputs_of(0.15, 0.0, GF_STRUCT_BLOCKS), GF_STRUCT_BLOCKS, h0_tol=1e-6,
                                 delta_rel_tol=1e-8)
  assert len(logged) == 2 and all("[WARNING]" in msg and "Delta is diagonal" in msg for msg in logged)

  from triqs_cthyb.util import orbital_occupations
  sector = make_problem(KanamoriParams(t=0.0, V_01=0.3), iaft, dlr_wmax=20.0, dlr_eps=1e-10,
                        degenerate_blk=((0, 1),), gf_struct=GF_STRUCT_BLOCKS)
  cheap = CthybSolver(dict(_CHEAP), comm_size=mpi.size).solve(sector, occupations_only=False, chkpt=None)
  rho_g = density_from_g_tau(G_tau=cheap.solver_data["G_tau"], gf_struct=GF_STRUCT_BLOCKS, dlr_wmax=20.0,
                             dlr_eps=1e-10)
  for blk, (name, _) in enumerate(GF_STRUCT_BLOCKS):
    rho = cheap.solver_data["rho_blocks"][blk]
    assert np.abs(np.diag(rho) - cheap.orbital_occupations[name]).max() < 1e-12
    off = ~np.eye(2, dtype=bool)
    assert np.abs(rho[off] - rho_g[blk][off]).max() < 1e-12
    assert np.abs(rho_g[blk][off]).max() > 1e-4            # the fallback carries a nonzero element
