"""
CT-HYB backend helpers (coqui.dmft.cthyb.cthyb_utils): the Lang-Firsov kernel check against a closed
form, the Dyson self-energy from an exact G(tau), the tail-fit window and the density correlator.
Serial: python -m pytest <this file> -v     (from outside the repository)
MPI:    mpirun -n 2 python -m pytest <this file> -v
"""
import numpy as np
import pytest

pytest.importorskip("triqs", reason="triqs is required for dmft tests")
pytest.importorskip("coqui._lib.iaft_module", reason="the DLR IAFT binding is required")

from triqs.gfs import Gf, BlockGf, Block2Gf, MeshImFreq, MeshImTime, MeshDLRImFreq, make_gf_from_fourier
from coqui.dmft.cthyb.cthyb_utils import (legendre_kernel_error, kernel_error_from_block2gf, sigma_from_g_tau,
                                          density_from_g_tau, density_matrix_sector_mask, fill_d0_tau,
                                          g_iw_from_g_tau,
                                          resolve_tail_fit_window, chi_from_q_tau, kernel_mirror_mismatch,
                                          legendre_coeffs_of_interpolant, kernel_from_d, resolve_meshes)
from coqui.dmft.qmc_postproc import d0_tau_colors
from coqui.dmft.triqs_utils import problem_to_triqs
from coqui.dmft.tests.hubbard_holstein_dimer import DimerParams, DimerED, make_iaft, make_problem
from coqui.dmft.tests import kanamori_holstein_dimer as kanamori


def test_legendre_kernel_error():
  """legendre_kernel_error measures the Lang-Firsov kernel's truncation error against the exact kernel.

  Setup: a single-pole retarded D(tau) = -c cosh(w (tau - beta/2)) / sinh(w beta / 2) on 10001 points
  (beta 20, w 1, c 0.98; its Fourier transform at nu = 0 is -2c/w = -1.96), one colour pair; the exact
  kernel is integrated from D on 400001 points.

  Cases:
    1. With n_l = 300 the max kernel error is below 1e-6 Ha and the static shift
       K'(0) = -(2/beta) int (beta - tau) D is matched to 1e-6 (the truncation is converged); k_max
       is positive.
    2. With n_l = 4 the error exceeds 1e-3 (the check sees a truncated kernel).
  """
  beta, w, c = 20.0, 1.0, 0.98
  tau = np.linspace(0.0, beta, 10001)
  D_tau = (-c * np.cosh(w * (tau - beta / 2)) / np.sinh(w * beta / 2))[:, None, None]
  tau_f = np.linspace(0.0, beta, 400001)
  D_fine = (-c * np.cosh(w * (tau_f - beta / 2)) / np.sinh(w * beta / 2))[:, None, None]
  err, k_max, dkp0 = legendre_kernel_error(D_tau=D_tau, beta=beta, n_l=300, D_fine=D_fine)
  assert k_max > 0 and err.max() < 1e-6 and np.abs(dkp0).max() < 1e-6
  err4, _, _ = legendre_kernel_error(D_tau=D_tau, beta=beta, n_l=4, D_fine=D_fine)
  assert err4.max() > 1e-3


def test_kernel_mirror_mismatch():
  """kernel_mirror_mismatch is the largest difference between the solver's Legendre kernel
  coefficients and the Python mirror kernel_from_d(legendre_coeffs_of_interpolant(D, n_l), beta).

  Setup: the single-pole D(tau) of test_legendre_kernel_error on 1001 points, two colours with the
  same D, n_l = 20; K_n built from the mirror itself.

  Cases:
    1. K_n equal to the mirror for every pair: 0.
    2. One pair's coefficients shifted by 1e-3: 1e-3 (to 1e-12).
    3. K_n None or empty: None (the solver did not expose a kernel).
  """
  beta, w, c, n_l = 20.0, 1.0, 0.98, 20
  tau = np.linspace(0.0, beta, 1001)
  d = -c * np.cosh(w * (tau - beta / 2)) / np.sinh(w * beta / 2)
  d_tau = np.repeat(np.repeat(d[:, None, None], 2, axis=1), 2, axis=2)
  mirror = kernel_from_d(legendre_coeffs_of_interpolant(d, n_l), beta)
  K_n = [[mirror.tolist(), mirror.tolist()], [mirror.tolist(), mirror.tolist()]]
  assert kernel_mirror_mismatch(K_n=K_n, d_tau=d_tau, beta=beta, n_l=n_l) == 0.0
  K_n[1][0] = (mirror + 1e-3).tolist()
  assert abs(kernel_mirror_mismatch(K_n=K_n, d_tau=d_tau, beta=beta, n_l=n_l) - 1e-3) < 1e-12
  assert kernel_mirror_mismatch(K_n=None, d_tau=d_tau, beta=beta, n_l=n_l) is None
  assert kernel_mirror_mismatch(K_n=[], d_tau=d_tau, beta=beta, n_l=n_l) is None


@pytest.fixture(scope="module")
def dimer():
  """The dimer, its ED reference on the IAFT mesh, its ImpurityProblem and the converted TRIQS inputs."""
  params = DimerParams()
  iaft = make_iaft(params, wmax=20.0, eps=1e-10)
  ed = DimerED(params)
  ref = ed.reference(iaft)
  problem = make_problem(params, iaft, dlr_wmax=20.0, dlr_eps=1e-10, degenerate_blk=((0, 1),))
  inputs = problem_to_triqs(problem, h_int_kind="density", screen_j=False, real_hamiltonian=True)
  return params, ed, ref, problem, inputs


def test_kernel_error_from_block2gf(dimer):
  """kernel_error_from_block2gf gives the dense result of legendre_kernel_error pair by pair.

  Setup: the dimer's retarded interaction (Block2Gf from problem_to_triqs), n_tau_bosonic 10001,
  dyn_n_l 100, the fine grid 4 (n - 1) + 1.

  Cases:
    1. Max error, max |K| and shift error equal those of legendre_kernel_error on the dense D(tau) of
       d0_tau_colors with the same fine grid, to 1e-12; it works pair by pair, so its memory does not
       grow with the square of the colour count.
    2. Matrix blocks: the two-orbital Kanamori impurity (no boson in the ED, the retarded part is the
       one-pole u(nu) of make_problem) in the one-block-per-spin layout, every (2, 2) block pair
       entry scaled by 1 + 0.1 (2 i1 + i2) so the pairs differ; the same three numbers equal the
       dense ones to 1e-12 (a dropped or swapped inner index would skip or double-count a pair).
  """
  params, ed, ref, problem, inputs = dimer
  n_tb = 10001
  dense = d0_tau_colors(block2gf=inputs.u_weiss_iw, gf_struct=problem.gf_struct, n_tau=n_tb)
  fine = d0_tau_colors(block2gf=inputs.u_weiss_iw, gf_struct=problem.gf_struct, n_tau=4 * (n_tb - 1) + 1)
  err_d, kmax_d, dkp0_d = legendre_kernel_error(D_tau=dense, beta=params.beta, n_l=100, D_fine=fine)
  err_p, kmax_p, shift_p = kernel_error_from_block2gf(
    block2gf=inputs.u_weiss_iw, gf_struct=problem.gf_struct, beta=params.beta, n_l=100, n_tau_bosonic=n_tb,
    n_fine=4 * (n_tb - 1) + 1)
  assert abs(err_p - err_d.max()) < 1e-12 and abs(kmax_p - kmax_d) < 1e-12 and abs(shift_p - np.abs(dkp0_d).max()) < 1e-12

  kp = kanamori.KanamoriParams(g=0.0, n_boson=0)
  k_iaft = kanamori.make_iaft(kp, wmax=20.0, eps=1e-10)
  k_problem = kanamori.make_problem(kp, k_iaft, dlr_wmax=20.0, dlr_eps=1e-10, degenerate_blk=None,
                                    gf_struct=kanamori.GF_STRUCT_BLOCKS)
  k_inputs = problem_to_triqs(k_problem, h_int_kind="kanamori", screen_j=False, real_hamiltonian=True)
  u_blocks = k_inputs.u_weiss_iw
  for n1, _ in kanamori.GF_STRUCT_BLOCKS:
    for n2, _ in kanamori.GF_STRUCT_BLOCKS:
      for i1 in range(2):
        for i2 in range(2):
          u_blocks[n1, n2].data[:, i1, i2] *= 1.0 + 0.1 * (2 * i1 + i2)
  dense = d0_tau_colors(block2gf=u_blocks, gf_struct=kanamori.GF_STRUCT_BLOCKS, n_tau=n_tb)
  fine = d0_tau_colors(block2gf=u_blocks, gf_struct=kanamori.GF_STRUCT_BLOCKS, n_tau=4 * (n_tb - 1) + 1)
  err_d, kmax_d, dkp0_d = legendre_kernel_error(D_tau=dense, beta=kp.beta, n_l=100, D_fine=fine)
  err_p, kmax_p, shift_p = kernel_error_from_block2gf(
    block2gf=u_blocks, gf_struct=kanamori.GF_STRUCT_BLOCKS, beta=kp.beta, n_l=100, n_tau_bosonic=n_tb,
    n_fine=4 * (n_tb - 1) + 1)
  assert abs(err_p - err_d.max()) < 1e-12 and abs(kmax_p - kmax_d) < 1e-12 and abs(shift_p - np.abs(dkp0_d).max()) < 1e-12


def _dimer_g_tau(ed, params, gf_struct, n_iw, noise=0.0, seed=0):
  """The dimer's exact G(iw) on a uniform mesh of n_iw, Fourier-transformed to G(tau) on 6 n_iw + 1
  points, optionally with white noise of the given amplitude on the tau data (the same on every rank)."""
  names = [name for name, _ in gf_struct]
  mesh = MeshImFreq(beta=params.beta, statistic="Fermion", n_iw=n_iw)
  n_uni = 2 * np.arange(-n_iw, n_iw) + 1
  g_exact = ed.green(n_uni)                                     # (2 n_iw, 2)
  G_iw = BlockGf(mesh=mesh, gf_struct=gf_struct)
  for spin, name in enumerate(names):
    G_iw[name].data[:, 0, 0] = g_exact[:, spin]
  G_tau = BlockGf(name_list=names, block_list=[make_gf_from_fourier(G_iw[name], 6 * n_iw + 1) for name in names])
  if noise:
    rng = np.random.default_rng(seed)
    for _, g in G_tau:
      g.data[:] += noise * rng.standard_normal(g.data.shape)
  return G_tau


def _dimer_delta_iw(params, gf_struct):
  """The dimer's Delta(iw) = V^2 / (iw - eps_b) on the fermionic DLR mesh (w_max 20, eps 1e-10)."""
  names = [name for name, _ in gf_struct]
  mesh_f = MeshDLRImFreq(beta=params.beta, statistic="Fermion", w_max=20.0, eps=1e-10, symmetrize=True)
  idx = np.array([x.index for x in mesh_f])
  delta_iw = BlockGf(name_list=names, block_list=[Gf(mesh=mesh_f, target_shape=(1, 1)) for _ in names])
  for name in names:
    delta_iw[name].data[:, 0, 0] = params.V**2 / (1j * (2 * idx + 1) * np.pi / params.beta - params.eps_b)
  return delta_iw


def test_sigma_from_g_tau(dimer):
  """sigma_from_g_tau reproduces the dimer's self-energy from its exact G(tau), and the tail fit
  replaces the noisy high-frequency Sigma by a fit with Sigma_infty as the known zeroth moment.

  Setup: the dimer's exact G(iw) on a uniform mesh of n_iw = 400 (beta 20), Fourier-transformed to
  G(tau) on 2401 points; Delta(iw) on the DLR mesh; h0 = eps per block; Sigma_infty from ED.

  Cases:
    1. Without a tail fit: Sigma_dyn(iw0) = Sigma(iw0) - Sigma_infty and G(iw0) within 1e-5 of ED (the
       G(iw) -> G(tau) -> G(iw) round trip at n_tau = 6 n_iw + 1 is accurate to 1e-6, measured; the
       tolerance is ten times that); the two spin blocks equal after averaging over degenerate_blk
       [[0, 1]].
    2. With the tail fit (fit_min_w 2.5, fit_max_w 10, fit_max_moment 7, Sigma_infty known): the
       result at iw0 is unchanged within 1e-5 (the window starts far above iw0) and G is the Dyson G
       of the fitted Sigma (G = (G0^-1 - Sigma)^-1 to 1e-12).
    3. White noise of 1e-4 on G(tau): without the fit, Sigma at the top of the mesh is off the exact
       Sigma by more than 1e-2 (the noise is amplified by G^-1 ~ iw); with the fit it is within 2e-3 of
       the exact value at the top of the mesh, and within 1e-3 of Sigma_infty + m1/iw with the exact
       first moment. This is what the fit is for; the clean case 2 cannot see it.
    4. Matrix blocks (reuses the uniform mesh integers of case 3): the two-orbital Kanamori impurity
       with the hopping t = 0.15 and the cross hybridization V_01 = 0.3 and no boson (the Dyson step
       does not see it) in the one-block-per-spin layout (h0, Delta and G all off-diagonal), its
       exact G(iw) on the uniform mesh to G(tau) as in case 1, Delta and h0 from problem_to_triqs;
       without a tail fit, Sigma(iw0) and G(iw0) equal the ED's 2x2 Sigma_iw_full and G_iw_full,
       off-diagonal element included (0.02 in size, so a transposed or dropped index, or a dropped
       off-diagonal h0, would show), within 1e-4 (the round trip of case 1 is 6e-5 on this G,
       measured), and the two spin blocks agree. With the tail fit of case 2 and the ED's (2, 2)
       Sigma_infty_full as the known moment, the fitted Sigma at the top of the mesh equals
       Sigma_infty_full + m1 / iw with the exact first moment from the top of the exact Sigma, to
       1e-3 in every element (the matrix known moment reaches the fit).
  """
  params, ed, ref, problem, inputs = dimer
  gf_struct = problem.gf_struct
  n_iw = 400
  G_tau = _dimer_g_tau(ed, params, gf_struct, n_iw)
  delta_iw = _dimer_delta_iw(params, gf_struct)
  h0_blocks = [np.array([[params.eps]]), np.array([[params.eps]])]
  sigma_inf = np.array(ref["Sigma_infty"], dtype=float)
  sigma_inf_blocks = [np.array([[x]]) for x in sigma_inf]
  i0 = int(np.argmin(np.where(ref["wn_f"] > 0, ref["wn_f"], np.inf)))
  k0 = (int(ref["wn_f"][i0]) - 1) // 2 + n_iw
  Sigma_iw, G_iw, G0_iw = sigma_from_g_tau(G_tau=G_tau, delta_iw=delta_iw, h0_blocks=h0_blocks, sigma_infty_blocks=sigma_inf_blocks,
                                           gf_struct=gf_struct, n_iw=n_iw, degenerate_blk=[[0, 1]],
                                           tail_fit_params=None)
  assert abs(Sigma_iw["up_0"].data[k0, 0, 0] - sigma_inf[0] - ref["Sigma_dyn_iw"][i0, 0]) < 1e-5
  assert abs(G_iw["up_0"].data[k0, 0, 0] - ref["G_iw"][i0, 0]) < 1e-5
  assert np.allclose(Sigma_iw["up_0"].data, Sigma_iw["down_0"].data)

  fit = {"fit_min_n": None, "fit_max_n": None, "fit_min_w": 2.5, "fit_max_w": 10.0, "fit_max_moment": 7}
  Sigma_fit, G_fit, G0_fit = sigma_from_g_tau(G_tau=G_tau, delta_iw=delta_iw, h0_blocks=h0_blocks, sigma_infty_blocks=sigma_inf_blocks,
                                              gf_struct=gf_struct, n_iw=n_iw, degenerate_blk=[[0, 1]],
                                              tail_fit_params=fit)
  assert abs(Sigma_fit["up_0"].data[k0, 0, 0] - Sigma_iw["up_0"].data[k0, 0, 0]) < 1e-5
  dyson = 1.0 / (1.0 / G0_fit["up_0"].data[:, 0, 0] - Sigma_fit["up_0"].data[:, 0, 0])
  assert np.allclose(G_fit["up_0"].data[:, 0, 0], dyson, atol=1e-12)

  # the exact Sigma on the uniform mesh, for the tail
  n_uni = 2 * np.arange(-n_iw, n_iw) + 1
  iw = 1j * np.pi * n_uni / params.beta
  g0_exact = 1.0 / (iw - params.eps - params.V**2 / (iw - params.eps_b))
  sigma_exact = 1.0 / g0_exact - 1.0 / ed.green(n_uni)[:, 0]
  m1 = ((sigma_exact[-1] - sigma_inf[0]) * iw[-1]).real          # first moment from the exact tail
  G_noisy = _dimer_g_tau(ed, params, gf_struct, n_iw, noise=1e-4, seed=0)
  Sigma_raw, _, _ = sigma_from_g_tau(G_tau=G_noisy, delta_iw=delta_iw, h0_blocks=h0_blocks, sigma_infty_blocks=sigma_inf_blocks,
                                     gf_struct=gf_struct, n_iw=n_iw, degenerate_blk=[[0, 1]],
                                     tail_fit_params=None)
  Sigma_fitn, _, _ = sigma_from_g_tau(G_tau=G_noisy, delta_iw=delta_iw, h0_blocks=h0_blocks, sigma_infty_blocks=sigma_inf_blocks,
                                      gf_struct=gf_struct, n_iw=n_iw, degenerate_blk=[[0, 1]],
                                      tail_fit_params=fit)
  top = slice(2 * n_iw - 20, 2 * n_iw)
  assert np.abs(Sigma_raw["up_0"].data[top, 0, 0] - sigma_exact[top]).max() > 1e-2
  assert np.abs(Sigma_fitn["up_0"].data[top, 0, 0] - sigma_exact[top]).max() < 2e-3
  assert np.abs(Sigma_fitn["up_0"].data[top, 0, 0] - sigma_inf[0] - m1 / iw[top]).max() < 1e-3

  kp = kanamori.KanamoriParams(t=0.15, V_01=0.3, g=0.0, n_boson=0)
  k_iaft = kanamori.make_iaft(kp, wmax=20.0, eps=1e-10)
  k_ed = kanamori.KanamoriED(kp)
  k_ref = k_ed.reference(k_iaft, n_bose=2)
  k_problem = kanamori.make_problem(kp, k_iaft, dlr_wmax=20.0, dlr_eps=1e-10, degenerate_blk=None,
                                    gf_struct=kanamori.GF_STRUCT_BLOCKS)
  k_inputs = problem_to_triqs(k_problem, h_int_kind="kanamori", screen_j=False, real_hamiltonian=True)
  k_mesh = MeshImFreq(beta=kp.beta, statistic="Fermion", n_iw=n_iw)
  k_full = k_ed.green_matrix(n_uni)                                 # (2 n_iw, 4, 4) colour matrix
  k_names = [name for name, _ in kanamori.GF_STRUCT_BLOCKS]
  k_G_iw = BlockGf(mesh=k_mesh, gf_struct=kanamori.GF_STRUCT_BLOCKS)
  for spin, name in enumerate(k_names):
    cols = [c for c in range(4) if kanamori.COLOR_SPIN[c] == spin]
    k_G_iw[name].data[:] = k_full[np.ix_(np.arange(2 * n_iw), cols, cols)]
  k_G_tau = BlockGf(name_list=k_names, block_list=[make_gf_from_fourier(k_G_iw[name], 6 * n_iw + 1) for name in k_names])
  k_sigma_inf = [k_ref["Sigma_infty_full"][:2, :2], k_ref["Sigma_infty_full"][2:, 2:]]
  k_i0 = int(np.argmin(np.where(k_ref["wn_f"] > 0, k_ref["wn_f"], np.inf)))
  k_k0 = (int(k_ref["wn_f"][k_i0]) - 1) // 2 + n_iw
  k_Sigma, k_G, _ = sigma_from_g_tau(G_tau=k_G_tau, delta_iw=k_inputs.delta_iw, h0_blocks=k_inputs.h0_blocks,
                                     sigma_infty_blocks=k_sigma_inf, gf_struct=kanamori.GF_STRUCT_BLOCKS,
                                     n_iw=n_iw, degenerate_blk=None, tail_fit_params=None)
  assert abs(k_ref["Sigma_iw_full"][k_i0, 0, 1]) > 0.01
  assert np.abs(k_Sigma["up_0"].data[k_k0] - k_ref["Sigma_iw_full"][k_i0][:2, :2]).max() < 1e-4
  assert np.abs(k_G["up_0"].data[k_k0] - k_ref["G_iw_full"][k_i0][:2, :2]).max() < 1e-4
  assert np.allclose(k_Sigma["up_0"].data, k_Sigma["down_0"].data)
  k_Sigma_fit, _, _ = sigma_from_g_tau(G_tau=k_G_tau, delta_iw=k_inputs.delta_iw, h0_blocks=k_inputs.h0_blocks,
                                       sigma_infty_blocks=k_sigma_inf, gf_struct=kanamori.GF_STRUCT_BLOCKS,
                                       n_iw=n_iw, degenerate_blk=None, tail_fit_params=fit)
  # the exact Sigma of the up block on the uniform mesh: G0^-1 - G^-1 with Delta = V_mat V_mat^T / (iw - eps_b)
  v_mat = np.array([[kp.V, kp.V_01], [kp.V_01, kp.V]])
  k_delta_uni = (v_mat @ v_mat.T)[None] / (iw - kp.eps_b)[:, None, None]
  k_g0_inv = iw[:, None, None] * np.eye(2) - np.asarray(k_inputs.h0_blocks[0])[None] - k_delta_uni
  k_sigma_exact = k_g0_inv - np.linalg.inv(k_full[np.ix_(np.arange(2 * n_iw), [0, 1], [0, 1])])
  k_m1 = ((k_sigma_exact[-1] - k_sigma_inf[0]) * iw[-1]).real
  assert np.abs(k_Sigma_fit["up_0"].data[top] - k_sigma_inf[0][None] - k_m1[None] / iw[top, None, None]).max() < 1e-3


def test_resolve_tail_fit_window():
  """resolve_tail_fit_window converts the Hartree window to mesh indices as TRIQS's tail_fit does,
  n = int(0.5 (w beta / pi - 1)), clipped to the mesh, explicit indices winning.

  Setup: closed forms.

  Cases:
    1. beta 1000, n_iw 1025, window 2.5-10 Ha: fit_min_n 397, fit_max_n clipped to 1025 (TRIQS returns
       NaN beyond the mesh instead of clipping), fit_max_moment passed through.
    2. Explicit fit_min_n 10 / fit_max_n 50 win over the Hartree form.
    3. Neither form given: the default window [0.8 n_iw, n_iw) = [80, 100) at n_iw 100.
    4. A window that starts beyond the mesh (fit_min_w 40 Ha = index 126 at n_iw 100, beta 20) raises
       ValueError naming fit_min; an empty window (fit_min_n 50, fit_max_n 50) raises naming "empty".
  """
  win = resolve_tail_fit_window(tail_fit_params={"fit_min_w": 2.5, "fit_max_w": 10.0, "fit_min_n": None,
                                                 "fit_max_n": None, "fit_max_moment": 7}, beta=1000.0, n_iw=1025)
  assert (win["fit_min_n"], win["fit_max_n"], win["fit_max_moment"]) == (397, 1025, 7)
  win = resolve_tail_fit_window(tail_fit_params={"fit_min_w": 2.5, "fit_max_w": 10.0, "fit_min_n": 10,
                                                 "fit_max_n": 50, "fit_max_moment": 3}, beta=1000.0, n_iw=1025)
  assert (win["fit_min_n"], win["fit_max_n"]) == (10, 50)
  win = resolve_tail_fit_window(tail_fit_params={"fit_min_w": None, "fit_max_w": None, "fit_min_n": None,
                                                 "fit_max_n": None, "fit_max_moment": 7}, beta=20.0, n_iw=100)
  assert (win["fit_min_n"], win["fit_max_n"]) == (80, 100)
  with pytest.raises(ValueError, match="fit_min"):
    resolve_tail_fit_window(tail_fit_params={"fit_min_w": 40.0, "fit_max_w": 60.0, "fit_min_n": None,
                                             "fit_max_n": None, "fit_max_moment": 7}, beta=20.0, n_iw=100)
  with pytest.raises(ValueError, match="empty"):
    resolve_tail_fit_window(tail_fit_params={"fit_min_w": None, "fit_max_w": None, "fit_min_n": 50,
                                             "fit_max_n": 50, "fit_max_moment": 7}, beta=20.0, n_iw=100)


def test_resolve_meshes():
  """resolve_meshes sizes the solver meshes from the DLR mesh and the tail-fit window, and keeps an
  override except where the DLR range or the Fourier transform needs more.

  Setup: beta 20, dlr_wmax 20, largest DLR index 100; closed forms.

  Cases:
    1. No override, fit_max_w 10 Ha: n_iw = max(140, int(0.5 (10 20/pi - 1)) + 1 = 32, 101) = 140,
       n_tau = max(10001, 6 140 + 1) = 10001, n_tau_bosonic = max(100001, 2 20 20 + 1) = 100001.
    2. fit_max_w 60 Ha dominates: n_iw = int(0.5 (60 20/pi - 1)) + 1 = 191; without a fit (None) the
       DLR rule alone, 140.
    3. Overrides: n_iw 50 is raised to the DLR range, 101; n_iw 300 is kept; n_tau 10 is raised to
       6 n_iw + 1; n_tau_bosonic 2001 is kept; n_tau_bosonic_min 1001 lowers the floor to 1001;
       n_iw 2000 raises n_tau to 12001 above the floor.
  """
  n_iw, n_tau, n_tb = resolve_meshes(beta=20.0, dlr_wmax=20.0, max_dlr_index=100, fit_max_w=10.0)
  assert (n_iw, n_tau, n_tb) == (140, 10001, 100001)
  n_iw, _, _ = resolve_meshes(beta=20.0, dlr_wmax=20.0, max_dlr_index=100, fit_max_w=60.0)
  assert n_iw == int(0.5 * (60.0 * 20.0 / np.pi - 1.0)) + 1 == 191
  n_iw, _, _ = resolve_meshes(beta=20.0, dlr_wmax=20.0, max_dlr_index=100, fit_max_w=None)
  assert n_iw == 140
  n_iw, n_tau, n_tb = resolve_meshes(beta=20.0, dlr_wmax=20.0, max_dlr_index=100, fit_max_w=10.0, n_iw=50,
                                     n_tau=10, n_tau_bosonic=2001)
  assert (n_iw, n_tau, n_tb) == (101, 6 * 101 + 1, 2001)
  n_iw, _, n_tb = resolve_meshes(beta=20.0, dlr_wmax=20.0, max_dlr_index=100, fit_max_w=10.0, n_iw=300,
                                 n_tau_bosonic_min=1001)
  assert (n_iw, n_tb) == (300, 1001)
  _, n_tau, _ = resolve_meshes(beta=20.0, dlr_wmax=20.0, max_dlr_index=100, fit_max_w=10.0, n_iw=2000)
  assert n_tau == 12001


def test_g_iw_from_g_tau():
  """g_iw_from_g_tau recovers G(iw) of a matrix block from G(tau) over the whole uniform mesh.

  Setup: the two-orbital Kanamori impurity with t = 0.15, V_01 = 0.3 and no boson in the
  one-block-per-spin layout; its exact G(iw) on a uniform mesh of n_iw = 400 (beta 20)
  Fourier-transformed to G(tau) on 6 n_iw + 1 points.

  Cases:
    1. Without degenerate blocks: every element of the up block within 1e-4 of the exact G(iw) at
       every frequency of the mesh (measured 8e-6; the 1/iw moment of a matrix block is the
       identity, and a wrong moment shows at the top of the mesh, 1.5e-3, not at iw0); the down
       block equal to the exact down block likewise.
    2. With degenerate_blk [[0, 1]]: the two blocks equal (the problem is spin-symmetric, so the
       average changes nothing beyond 1e-12).
  """
  kp = kanamori.KanamoriParams(t=0.15, V_01=0.3, g=0.0, n_boson=0)
  ed = kanamori.KanamoriED(kp)
  n_iw = 400
  n_uni = 2 * np.arange(-n_iw, n_iw) + 1
  full = ed.green_matrix(n_uni)
  names = [name for name, _ in kanamori.GF_STRUCT_BLOCKS]
  G_iw = BlockGf(mesh=MeshImFreq(beta=kp.beta, statistic="Fermion", n_iw=n_iw), gf_struct=kanamori.GF_STRUCT_BLOCKS)
  exact = {}
  for spin, name in enumerate(names):
    cols = [c for c in range(4) if kanamori.COLOR_SPIN[c] == spin]
    exact[name] = full[np.ix_(np.arange(2 * n_iw), cols, cols)]
    G_iw[name].data[:] = exact[name]
  G_tau = BlockGf(name_list=names, block_list=[make_gf_from_fourier(G_iw[name], 6 * n_iw + 1) for name in names])
  back = g_iw_from_g_tau(G_tau=G_tau, gf_struct=kanamori.GF_STRUCT_BLOCKS, n_iw=n_iw, degenerate_blk=None)
  for name in names:
    assert np.abs(back[name].data - exact[name]).max() < 1e-4
  averaged = g_iw_from_g_tau(G_tau=G_tau, gf_struct=kanamori.GF_STRUCT_BLOCKS, n_iw=n_iw, degenerate_blk=[[0, 1]])
  assert np.abs(averaged["up_0"].data - averaged["down_0"].data).max() < 1e-12


def test_density_matrix_sector_mask():
  """density_matrix_sector_mask marks the elements <c+_a c_a'> of a block that c+_a c_a' keeps inside
  every symmetry sector of h_loc, which are the ones the solver's density matrix holds.

  Setup: the two-orbital Kanamori impurity (no boson) in the one-block-per-spin layout, h_loc =
  h0 + h_int from problem_to_triqs diagonalized with triqs.atom_diag.AtomDiag on the block's
  operators.

  Cases:
    1. h0 diagonal (t = 0): the Kanamori atom conserves the orbital occupations modulo the pair
       hopping, so c+_0 c_1 changes sector; the mask is the identity in both blocks.
    2. h0 with the inter-orbital hopping t = 0.15: every element is within the sectors; the mask is
       all True.
  """
  from triqs.atom_diag import AtomDiag
  fops = [(name, i) for name, size in kanamori.GF_STRUCT_BLOCKS for i in range(size)]
  for t, expected in ((0.0, np.eye(2, dtype=bool)), (0.15, np.ones((2, 2), dtype=bool))):
    kp = kanamori.KanamoriParams(t=t, g=0.0, n_boson=0)
    iaft = kanamori.make_iaft(kp, wmax=20.0, eps=1e-10)
    problem = kanamori.make_problem(kp, iaft, dlr_wmax=20.0, dlr_eps=1e-10, degenerate_blk=None,
                                    gf_struct=kanamori.GF_STRUCT_BLOCKS)
    inputs = problem_to_triqs(problem, h_int_kind="kanamori", screen_j=False, real_hamiltonian=True)
    h_loc_diag = AtomDiag(inputs.h_int + inputs.h0_op, fops)
    mask = density_matrix_sector_mask(h_loc_diag=h_loc_diag, gf_struct=kanamori.GF_STRUCT_BLOCKS)
    assert len(mask) == 2
    for blk in mask:
      assert np.array_equal(blk, expected), (t, blk)


def test_density_from_g_tau():
  """density_from_g_tau recovers G(tau = 0-) = <c+_a' c_a> of a matrix block from a noisy G(tau) on a
  uniform grid.

  Setup: the two-orbital Kanamori impurity with V_01 = 0.3 and no boson in the one-block-per-spin
  layout; its exact G(iw) on a uniform mesh of n_iw = 400 (beta 20) Fourier-transformed to G(tau) on
  10001 points; the ED density matrix as the reference (rho_01 = 1.1e-3, rho_00 = 0.493).

  Cases:
    1. Exact G(tau): every element of both blocks within 1e-5 of the ED density matrix (the DLR fit
       at w_max 20, eps 1e-10 is exact to the Fourier round trip, measured 1e-7).
    2. White noise of 1e-3 on G(tau): every element within 2e-3 (measured up to 1.4e-3 over the two
       blocks with this seed; the last grid point alone is off by 1.6e-3 and a tail fit of G(iw) by
       more than 1). The off-diagonal element, 1.1e-3, is below this noise and is resolved by case 1
       only.
  """
  kp = kanamori.KanamoriParams(V_01=0.3, g=0.0, n_boson=0)
  ed = kanamori.KanamoriED(kp)
  n_iw = 400
  n_uni = 2 * np.arange(-n_iw, n_iw) + 1
  full = ed.green_matrix(n_uni)
  rho_ed = ed.density_matrix()
  names = [name for name, _ in kanamori.GF_STRUCT_BLOCKS]
  G_iw = BlockGf(mesh=MeshImFreq(beta=kp.beta, statistic="Fermion", n_iw=n_iw), gf_struct=kanamori.GF_STRUCT_BLOCKS)
  for spin, name in enumerate(names):
    cols = [c for c in range(4) if kanamori.COLOR_SPIN[c] == spin]
    G_iw[name].data[:] = full[np.ix_(np.arange(2 * n_iw), cols, cols)]
  G_tau = BlockGf(name_list=names, block_list=[make_gf_from_fourier(G_iw[name], 10001) for name in names])
  rho = density_from_g_tau(G_tau=G_tau, gf_struct=kanamori.GF_STRUCT_BLOCKS, dlr_wmax=20.0, dlr_eps=1e-10)
  assert np.abs(rho[0] - rho_ed[:2, :2]).max() < 1e-5
  assert np.abs(rho[1] - rho_ed[2:, 2:]).max() < 1e-5
  rng = np.random.default_rng(1)
  for _, g in G_tau:
    g.data[:] += 1e-3 * rng.standard_normal(g.data.shape)
  rho_n = density_from_g_tau(G_tau=G_tau, gf_struct=kanamori.GF_STRUCT_BLOCKS, dlr_wmax=20.0, dlr_eps=1e-10)
  assert np.abs(rho_n[0] - rho_ed[:2, :2]).max() < 2e-3
  assert np.abs(rho_n[1] - rho_ed[2:, 2:]).max() < 2e-3


def test_chi_from_q_tau():
  """chi_from_q_tau subtracts the plateau of the measured Q(tau), fits the connected part within what the
  tau grid resolves and evaluates it at the bosonic IAFT integers.

  Setup: beta 20, two colours; Q_ab(tau) = n^2 + a cosh(w (tau - beta/2)) / sinh(w beta / 2) with
  n^2 = 0.36, a = 0.1, w = 2, the same for every pair, on a uniform grid of 2001 points.

  Cases:
    1. At nu = 0, 2, 4 (integers 0, 2, 4) every pair equals the single-pole transform 2 a w / (w^2 + nu^2)
       to 1e-4 (the plateau of the cosh over the middle half of tau is 1e-5 of its peak at w = 2 and
       enters nu = 0 alone, times beta); the returned w_max is min(dlr_wmax, pi / (4 dtau)) = 20 here.
    2. One block of size 2 with the amplitude a (1 + 0.1 (2 i1 + i2)) per entry: chi_c[:, i1, i2] equals
       that entry's transform to 1e-4 (a swapped or dropped inner index would pick the wrong amplitude).
  """
  beta, nsq, a, w = 20.0, 0.36, 0.1, 2.0
  gf_struct = [("up_0", 1), ("down_0", 1)]
  names = [name for name, _ in gf_struct]
  mesh = MeshImTime(beta=beta, statistic="Boson", n_tau=2001)
  tau = np.array([float(t) for t in mesh])
  blocks = []
  for _ in names:
    row = []
    for _ in names:
      q = Gf(mesh=mesh, target_shape=(1, 1))
      q.data[:, 0, 0] = nsq + a * np.cosh(w * (tau - beta / 2)) / np.sinh(w * beta / 2)
      row.append(q)
    blocks.append(row)
  Q_tau = Block2Gf(names, names, blocks)
  wn_b = np.array([0, 2, 4])
  chi_c, w_max_fit = chi_from_q_tau(Q_tau=Q_tau, gf_struct=gf_struct, wn_b=wn_b, beta=beta, dlr_wmax=20.0,
                                    dlr_eps=1e-10)
  nu = np.pi * wn_b / beta
  exact = 2 * a * w / (w**2 + nu**2)
  assert chi_c.shape == (3, 2, 2)
  for c1 in range(2):
    for c2 in range(2):
      assert np.allclose(chi_c[:, c1, c2], exact, atol=1e-4)
  assert w_max_fit == min(20.0, np.pi / (4 * beta / 2000))

  q = Gf(mesh=mesh, target_shape=(2, 2))
  for i1 in range(2):
    for i2 in range(2):
      q.data[:, i1, i2] = nsq + a * (1.0 + 0.1 * (2 * i1 + i2)) * np.cosh(w * (tau - beta / 2)) / np.sinh(w * beta / 2)
  Q_blk = Block2Gf(["up_0"], ["up_0"], [[q]])
  chi_blk, _ = chi_from_q_tau(Q_tau=Q_blk, gf_struct=[("up_0", 2)], wn_b=wn_b, beta=beta, dlr_wmax=20.0,
                              dlr_eps=1e-10)
  assert chi_blk.shape == (3, 2, 2)
  for i1 in range(2):
    for i2 in range(2):
      assert np.allclose(chi_blk[:, i1, i2], exact * (1.0 + 0.1 * (2 * i1 + i2)), atol=1e-4)


def test_fill_d0_tau():
  """fill_d0_tau writes the colour matrix D(tau) into the solver's Block2Gf block pair by block pair,
  with the colour offsets of gf_struct.

  Setup: gf_struct [("up_0", 2), ("down_0", 1)] (three colours, blocks of different sizes), a bosonic
  tau mesh of 11 points, d_tau[t, c1, c2] = t + 10 c1 + 100 c2 (every entry distinct).

  Cases:
    1. After the fill, block pair (b1, b2) entry [i1, i2] equals d_tau[:, offset(b1) + i1,
       offset(b2) + i2] for every pair and entry (exact).
  """
  gf_struct = [("up_0", 2), ("down_0", 1)]
  names = [name for name, _ in gf_struct]
  mesh = MeshImTime(beta=20.0, statistic="Boson", n_tau=11)
  blocks = [[Gf(mesh=mesh, target_shape=(d1, d2)) for _, d2 in gf_struct] for _, d1 in gf_struct]
  D0_tau = Block2Gf(names, names, blocks)
  t = np.arange(11)[:, None, None]
  d_tau = t + 10 * np.arange(3)[None, :, None] + 100 * np.arange(3)[None, None, :]
  fill_d0_tau(D0_tau=D0_tau, d_tau=d_tau.astype(float), gf_struct=gf_struct)
  offs = [0, 2]
  for b1, (n1, d1) in enumerate(gf_struct):
    for b2, (n2, d2) in enumerate(gf_struct):
      for i1 in range(d1):
        for i2 in range(d2):
          assert np.array_equal(D0_tau[n1, n2].data[:, i1, i2].real, d_tau[:, offs[b1] + i1, offs[b2] + i2])
