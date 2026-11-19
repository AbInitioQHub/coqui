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
                                          resolve_tail_fit_window, chi_from_q_tau, kernel_mirror_mismatch,
                                          legendre_coeffs_of_interpolant, kernel_from_d, resolve_meshes)
from coqui.dmft.qmc_postproc import d0_tau_colors
from coqui.dmft.triqs_utils import problem_to_triqs
from coqui.dmft.tests.hubbard_holstein_dimer import DimerParams, DimerED, make_iaft, make_problem


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
  G(tau) on 2401 points; Delta(iw) on the DLR mesh; h0 = eps; Sigma_infty from ED.

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
  """
  params, ed, ref, problem, inputs = dimer
  gf_struct = problem.gf_struct
  n_iw = 400
  G_tau = _dimer_g_tau(ed, params, gf_struct, n_iw)
  delta_iw = _dimer_delta_iw(params, gf_struct)
  h0_c = np.array([params.eps, params.eps])
  sigma_inf = np.array(ref["Sigma_infty"], dtype=float)
  i0 = int(np.argmin(np.where(ref["wn_f"] > 0, ref["wn_f"], np.inf)))
  k0 = (int(ref["wn_f"][i0]) - 1) // 2 + n_iw
  Sigma_iw, G_iw, G0_iw = sigma_from_g_tau(G_tau=G_tau, delta_iw=delta_iw, h0_c=h0_c, sigma_inf_c=sigma_inf,
                                           gf_struct=gf_struct, n_iw=n_iw, degenerate_blk=[[0, 1]],
                                           tail_fit_params=None)
  assert abs(Sigma_iw["up_0"].data[k0, 0, 0] - sigma_inf[0] - ref["Sigma_dyn_iw"][i0, 0]) < 1e-5
  assert abs(G_iw["up_0"].data[k0, 0, 0] - ref["G_iw"][i0, 0]) < 1e-5
  assert np.allclose(Sigma_iw["up_0"].data, Sigma_iw["down_0"].data)

  fit = {"fit_min_n": None, "fit_max_n": None, "fit_min_w": 2.5, "fit_max_w": 10.0, "fit_max_moment": 7}
  Sigma_fit, G_fit, G0_fit = sigma_from_g_tau(G_tau=G_tau, delta_iw=delta_iw, h0_c=h0_c, sigma_inf_c=sigma_inf,
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
  Sigma_raw, _, _ = sigma_from_g_tau(G_tau=G_noisy, delta_iw=delta_iw, h0_c=h0_c, sigma_inf_c=sigma_inf,
                                     gf_struct=gf_struct, n_iw=n_iw, degenerate_blk=[[0, 1]],
                                     tail_fit_params=None)
  Sigma_fitn, _, _ = sigma_from_g_tau(G_tau=G_noisy, delta_iw=delta_iw, h0_c=h0_c, sigma_inf_c=sigma_inf,
                                      gf_struct=gf_struct, n_iw=n_iw, degenerate_blk=[[0, 1]],
                                      tail_fit_params=fit)
  top = slice(2 * n_iw - 20, 2 * n_iw)
  assert np.abs(Sigma_raw["up_0"].data[top, 0, 0] - sigma_exact[top]).max() > 1e-2
  assert np.abs(Sigma_fitn["up_0"].data[top, 0, 0] - sigma_exact[top]).max() < 2e-3
  assert np.abs(Sigma_fitn["up_0"].data[top, 0, 0] - sigma_inf[0] - m1 / iw[top]).max() < 1e-3


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


def test_chi_from_q_tau():
  """chi_from_q_tau subtracts the plateau of the measured Q(tau), fits the connected part within what the
  tau grid resolves and evaluates it at the bosonic IAFT integers.

  Setup: beta 20, two colours; Q_ab(tau) = n^2 + a cosh(w (tau - beta/2)) / sinh(w beta / 2) with
  n^2 = 0.36, a = 0.1, w = 2, the same for every pair, on a uniform grid of 2001 points.

  Cases:
    1. At nu = 0, 2, 4 (integers 0, 2, 4) every pair equals the single-pole transform 2 a w / (w^2 + nu^2)
       to 1e-4 (the plateau of the cosh over the middle half of tau is 1e-5 of its peak at w = 2 and
       enters nu = 0 alone, times beta); the returned w_max is min(dlr_wmax, pi / (4 dtau)) = 20 here.
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
