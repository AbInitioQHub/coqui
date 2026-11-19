"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==========================================================================
"""
"""
Helpers of the CT-HYB backend: the Lang-Firsov kernel check, the Dyson self-energy from the measured
G(tau) with its tail fit, and the density correlator from the measured Q(tau). The shared numerics
are in ``qmc_postproc``.
"""
import numpy as np
from numpy.polynomial import legendre
import triqs_modest as modest
from triqs.gfs import BlockGf, MeshImFreq, make_gf_dlr, make_gf_imfreq, make_hermitian, inverse, iOmega_n
from triqs.gfs.tools import make_zero_tail
from triqs.solver_utils import tail_fit

from triqs.gfs import make_gf_imtime
from coqui.dmft.qmc_postproc import eval_on_points, connected_by_plateau, fit_tau_correlator


# ---------------------------------------------------------------- Lang-Firsov kernel

def legendre_coeffs_of_interpolant(f, N):
  """
  Bosonic Legendre coefficients ``d_n = (2n+1)/2 int_{-1}^{1} D P_n dx``, ``n < N``, of the
  piecewise-linear interpolant of the samples ``f`` on a uniform grid over [0, beta] (both ends
  included), computed as ``triqs_cthyb``'s ``fit_legendre_coeffs`` does: two integrations by parts
  against the second antiderivative of ``P_n`` leave a sum over the slope jumps at the interior
  nodes.
  """
  f = np.asarray(f, dtype=float)
  n_pt = len(f)
  h = 2.0 / (n_pt - 1)
  x = -1.0 + h * np.arange(1, n_pt - 1)
  jump = ((f[2:] - f[1:-1]) - (f[1:-1] - f[:-2])) / h
  d = np.zeros(N)
  d[0] = (0.5 * (f[0] + f[-1]) + f[1:-1].sum()) * h
  if N > 1:
    d[1] = 2.0 / 3.0 * (f[-1] - f[-2]) / h + (((x**3 / 3 - x) / 2 - 1.0 / 3.0) * jump).sum()
  P = [np.ones_like(x), x.copy()]                                     # P[j] = P_{n-2+j} at the interior nodes
  for l in range(1, 4):
    P.append(((2 * l + 1) * x * P[l] - l * P[l - 1]) / (l + 1))
  for nn in range(2, N):
    d[nn] = (((P[4] - P[2]) / (2 * nn + 3) - (P[2] - P[0]) / (2 * nn - 1)) * jump).sum() / (2 * nn + 1)
    l = nn + 2
    P = P[1:] + [((2 * l + 1) * x * P[4] - l * P[3]) / (l + 1)]
  return d * (2 * np.arange(N) + 1) / 2.0


def kernel_from_d(d, beta):
  """Legendre coefficients of ``K`` (``K'' = D``, ``K(0) = K(beta) = 0``) from those of ``D``, as
  ``triqs_cthyb``'s ``build_M_matrix`` does."""
  N = len(d)
  M = np.zeros((N, N))
  b2 = beta * beta
  M[0, 0] = -b2 / 12.0
  if N > 1:
    M[1, 1] = -b2 / 60.0
  for q in range(N):
    if q >= 2:
      M[q, q] = -b2 / (2.0 * (2 * q - 1) * (2 * q + 3))
    if q + 2 < N:
      M[q + 2, q] = b2 / (4.0 * (2 * q + 1) * (2 * q + 3))
    if q - 2 >= 0:
      M[q - 2, q] = b2 / (4.0 * (2 * q - 1) * (2 * q + 1))
  return M @ d


def legendre_kernel_error(D_tau, beta, n_l, D_fine=None):
  """
  The error of the Lang-Firsov kernel ``K_ab(tau)`` as ``triqs_cthyb`` builds it.

  The solver projects the piecewise-linear interpolant of ``D`` on its ``n_tau_bosonic`` grid onto
  ``n_l`` Legendre polynomials and maps them to ``K`` (``K'' = D``, ``K(0) = K(beta) = 0``). This
  compares that kernel with the exact one from ``D_fine`` (``D`` on a finer grid; default ``D_tau``
  itself, which then omits the grid floor). Two errors: the truncation (a GW ``D`` decays over a
  fraction of a Ha^-1) and the grid floor ``(w dtau)^2 / 12`` for a boson of frequency ``w``.

  Parameters
  ----------
  D_tau : np.ndarray (n_tau, n_color, n_color)
      ``D`` on the solver's uniform grid over [0, beta].
  beta : float
  n_l : int
      ``dyn_n_l``.
  D_fine : np.ndarray (n_fine, n_color, n_color), optional
      ``D`` on a finer uniform grid.

  Returns
  -------
  err : np.ndarray (n_color, n_color)
      max |K_solver - K| per pair, in Ha.
  k_max : float
      max |K| over all pairs.
  dkp0 : np.ndarray (n_color, n_color)
      ``K'(0)_solver - K'(0)`` per pair; ``K'(0) = -beta (d_0 - d_1/3)`` is the solver's static
      shift (``h_loc -= K'(0)/2 n_a n_b`` per ordered vertex).
  """
  D = np.asarray(D_tau).real
  Df = D if D_fine is None else np.asarray(D_fine).real
  tf = np.linspace(0.0, beta, Df.shape[0])
  dt = tf[1] - tf[0]
  Kp = np.concatenate([np.zeros((1,) + Df.shape[1:]), np.cumsum(0.5 * (Df[1:] + Df[:-1]) * dt, axis=0)])
  K = np.concatenate([np.zeros((1,) + Df.shape[1:]), np.cumsum(0.5 * (Kp[1:] + Kp[:-1]) * dt, axis=0)])
  K -= (tf / beta)[:, None, None] * K[-1]                        # K(beta) = 0
  w = np.full(len(tf), dt)
  w[[0, -1]] *= 0.5
  kp0 = -2.0 / beta * np.einsum("t,tab->ab", w * (beta - tf), Df)  # -(2/beta) int (beta - tau) D
  n_end = int(np.searchsorted(tf, 0.04 * beta))                  # every point near the ends (K varies fastest there)
  sel = np.unique(np.concatenate([np.arange(n_end), len(tf) - 1 - np.arange(n_end),
                                  np.linspace(0, len(tf) - 1, 20001).astype(int)]))
  xs = 2.0 * tf[sel] / beta - 1.0
  err = np.zeros(D.shape[1:])
  dkp0 = np.zeros(D.shape[1:])
  done = {}
  for a in range(D.shape[1]):
    for b in range(D.shape[2]):
      key = (D[:, a, b].tobytes(), Df[:, a, b].tobytes())
      if key not in done:                                        # density channels often repeat the same D
        d = legendre_coeffs_of_interpolant(D[:, a, b], n_l)
        k = kernel_from_d(d, beta)
        done[key] = (np.abs(legendre.legval(xs, k) - K[sel, a, b]).max(),
                     -beta * (d[0] - (d[1] / 3.0 if n_l > 1 else 0.0)) - kp0[a, b])
      err[a, b], dkp0[a, b] = done[key]
  return err, float(np.abs(K).max()), dkp0


def kernel_error_from_block2gf(block2gf, gf_struct, beta, n_l, n_tau_bosonic, n_fine):
  """
  ``legendre_kernel_error`` of a retarded density-density ``Block2Gf`` (the bosonic DLR mesh),
  pair by pair over the distinct colour pairs, so the memory is one fine 1-D array at a time
  rather than ``(n_fine, n_color, n_color)``.

  Parameters
  ----------
  block2gf : triqs.gfs.Block2Gf
      One ``(1, 1)`` block per colour pair (size-1 blocks).
  gf_struct : list of (str, int)
  beta : float
  n_l : int
      ``dyn_n_l``.
  n_tau_bosonic : int
      The solver's grid of ``D0_tau``.
  n_fine : int
      The finer grid of the exact kernel.

  Returns
  -------
  err_max : float
      max over pairs of max |K_solver - K|, in Ha.
  k_max : float
      max |K| over all pairs.
  shift_max : float
      max over pairs of |K'(0)_solver - K'(0)|.
  """
  names = [name for name, _ in gf_struct]
  done = {}
  err_max = k_max = shift_max = 0.0
  for n1 in names:
    for n2 in names:
      block = block2gf[n1, n2]
      key = np.asarray(block.data).tobytes()
      if key in done:
        continue
      dlr = make_gf_dlr(block)
      d = np.asarray(make_gf_imtime(dlr, int(n_tau_bosonic)).data).real[:, 0, 0]
      d_fine = np.asarray(make_gf_imtime(dlr, int(n_fine)).data).real[:, 0, 0]
      err, kmax, dkp0 = legendre_kernel_error(d[:, None, None], beta, n_l, D_fine=d_fine[:, None, None])
      done[key] = True
      err_max, k_max, shift_max = max(err_max, float(err.max())), max(k_max, kmax), max(shift_max, float(np.abs(dkp0).max()))
  return err_max, k_max, shift_max


def kernel_mirror_mismatch(K_n, d_tau, beta, n_l):
  """
  The largest difference between the Legendre coefficients of the Lang-Firsov kernel the solver
  built (``Solver.K_n[a][b][n]``, over the colour pairs) and the Python mirror
  (``kernel_from_d(legendre_coeffs_of_interpolant(D_ab, n_l), beta)``) that ``legendre_kernel_error``
  judges. A nonzero value means the solver's kernel fit no longer matches the mirror, so the reported
  kernel error describes a kernel the Monte Carlo did not use.

  Parameters
  ----------
  K_n : nested list (n_color, n_color, n_l) or None
      The solver's coefficients; ``None`` (not exposed, or not built) gives ``None``.
  d_tau : np.ndarray (n_tau, n_color, n_color)
      ``D(tau)`` on the solver's grid, as handed to it.
  beta : float
  n_l : int

  Returns
  -------
  float or None
  """
  if K_n is None or len(K_n) == 0:
    return None
  n_color = d_tau.shape[1]
  worst = 0.0
  for a in range(n_color):
    for b in range(n_color):
      lib = np.asarray(K_n[a][b], dtype=float)
      if lib.size == 0:
        continue
      mirror = kernel_from_d(legendre_coeffs_of_interpolant(d_tau[:, a, b], n_l), beta)
      n_cmp = min(lib.size, mirror.size)
      worst = max(worst, float(np.abs(lib[:n_cmp] - mirror[:n_cmp]).max()))
  return worst


# ---------------------------------------------------------------- meshes

def resolve_meshes(*, beta, dlr_wmax, max_dlr_index, fit_max_w, n_iw=None, n_tau=None, n_tau_bosonic=None,
                   n_tau_min=10001, n_tau_bosonic_min=100001):
  """
  The solver's mesh sizes from the impurity DLR mesh, unless overridden.

  ``n_iw = max(1.4 max_dlr_index, index(fit_max_w) + 1, max_dlr_index + 1)`` so that every IAFT point
  and the whole tail-fit window lie on the uniform mesh; ``n_tau = max(n_tau_min, 6 n_iw + 1)`` (the
  G(tau) measurement bins; the Fourier transform of a mesh of n_iw needs 6 n_iw + 1);
  ``n_tau_bosonic = max(n_tau_bosonic_min,
  2 beta dlr_wmax + 1)``, the grid the Lang-Firsov kernel is built on (its floor is
  ``(w dtau)^2 / 12``). An override is taken as given, except that ``n_iw`` is still raised to cover
  the DLR range and ``n_tau`` to ``6 n_iw + 1``.

  Returns
  -------
  n_iw, n_tau, n_tau_bosonic : int
  """
  n_iw_dlr = int(max_dlr_index) + 1
  if n_iw is None:
    n_iw = int(1.4 * max_dlr_index)
    if fit_max_w is not None:
      n_iw = max(n_iw, int(0.5 * (float(fit_max_w) * beta / np.pi - 1.0)) + 1)
  n_iw = max(int(n_iw), n_iw_dlr)
  n_tau = max(int(n_tau) if n_tau is not None else int(n_tau_min), 6 * n_iw + 1)
  if n_tau_bosonic is None:
    n_tau_bosonic = max(int(n_tau_bosonic_min), int(2 * beta * dlr_wmax) + 1)
  return n_iw, n_tau, int(n_tau_bosonic)


# ---------------------------------------------------------------- Sigma from G(tau)

def resolve_tail_fit_window(tail_fit_params, beta, n_iw):
  """
  The tail-fit window as mesh indices for a uniform mesh of ``n_iw`` positive frequencies.

  The Hartree form is converted as ``triqs.solver_utils.tail_fit`` does,
  ``n = int(0.5 (w beta / pi - 1))``; explicit ``fit_min_n`` / ``fit_max_n`` win; without either form
  the window is ``[0.8 n_iw, n_iw)``. ``fit_max_n`` is
  clipped to ``n_iw`` (``tail_fit`` returns NaN beyond the mesh instead of clipping), and a window
  that starts at or beyond the mesh raises, before any solve.

  Parameters
  ----------
  tail_fit_params : dict
      ``fit_min_w``, ``fit_max_w``, ``fit_min_n``, ``fit_max_n``, ``fit_max_moment``.
  beta : float
  n_iw : int

  Returns
  -------
  dict
      ``fit_min_n``, ``fit_max_n`` (resolved), ``fit_max_moment``.

  Raises
  ------
  ValueError
      ``fit_min_n`` at or beyond ``n_iw``.
  """
  def to_n(w):
    return int(0.5 * (float(w) * beta / np.pi - 1.0))
  fit_min_n = tail_fit_params.get("fit_min_n")
  fit_max_n = tail_fit_params.get("fit_max_n")
  if fit_min_n is None:
    fit_min_n = to_n(tail_fit_params["fit_min_w"]) if tail_fit_params.get("fit_min_w") is not None else int(0.8 * n_iw)
  if fit_max_n is None:
    fit_max_n = to_n(tail_fit_params["fit_max_w"]) if tail_fit_params.get("fit_max_w") is not None else n_iw
  fit_min_n, fit_max_n = int(fit_min_n), min(int(fit_max_n), int(n_iw))
  if fit_min_n >= n_iw:
    raise ValueError(f"cthyb_utils::resolve_tail_fit_window: the tail-fit window starts at mesh index "
                     f"{fit_min_n} (fit_min_w / fit_min_n) but the uniform mesh has n_iw = {n_iw}; lower "
                     f"fit_min_w or raise n_iw")
  if fit_min_n >= fit_max_n:
    raise ValueError(f"cthyb_utils::resolve_tail_fit_window: empty tail-fit window, indices "
                     f"[{fit_min_n}, {fit_max_n}) on a mesh of n_iw = {n_iw}")
  return {"fit_min_n": fit_min_n, "fit_max_n": fit_max_n, "fit_max_moment": tail_fit_params.get("fit_max_moment")}


def sigma_from_g_tau(G_tau, delta_iw, h0_c, sigma_inf_c, gf_struct, n_iw, degenerate_blk, tail_fit_params):
  """
  The Dyson self-energy on the uniform Matsubara mesh from the measured ``G(tau)``.

  ``G(iw)`` from ``G(tau)`` with the known ``1/iw`` moment, hermitized and averaged over the
  degenerate blocks before the Dyson step; ``G0 = (iw - h0 - Delta)^-1`` with the exact DLR
  ``Delta``; ``Sigma = G0^-1 - G^-1``; with ``tail_fit_params`` the tail of ``Sigma`` is fitted
  (``triqs.solver_utils.tail_fit``) with ``sigma_inf_c`` as the known zeroth moment and ``G`` is
  recomputed from the Dyson equation, so the two stay consistent.

  Parameters
  ----------
  G_tau : triqs.gfs.BlockGf
      The measured ``G(tau)``, size-1 blocks.
  delta_iw : triqs.gfs.BlockGf
      ``Delta(iw)`` on the fermionic DLR mesh.
  h0_c : array (n_color,)
      The level of every colour (``mu_shift`` included).
  sigma_inf_c : array (n_color,)
      The static self-energy of every colour.
  gf_struct : list of (str, int)
  n_iw : int
      The uniform mesh.
  degenerate_blk : list of index groups or None
  tail_fit_params : dict or None
      ``fit_min_n``, ``fit_max_n``, ``fit_min_w``, ``fit_max_w``, ``fit_max_moment``; resolved to
      mesh indices by ``resolve_tail_fit_window`` (clipped to the mesh).

  Returns
  -------
  Sigma_iw, G_iw, G0_iw : triqs.gfs.BlockGf on ``MeshImFreq(n_iw)``
      The full self-energy (static part included), ``G`` and ``G0``.
  """
  beta = G_tau.mesh.beta
  mesh = MeshImFreq(beta=beta, statistic="Fermion", n_iw=n_iw)
  G_iw = BlockGf(mesh=mesh, gf_struct=gf_struct)
  for name, g in G_iw:
    km = make_zero_tail(g, 2)
    km[1] = np.eye(g.target_shape[0])
    g.set_from_fourier(G_tau[name], km)
  G_iw = make_hermitian(G_iw)
  if degenerate_blk:
    G_iw = modest.symmetrize(G_iw, degenerate_blk)
  G0_iw = G_iw.copy()
  for c, (name, g) in enumerate(G0_iw):
    delta_uni = make_gf_imfreq(make_gf_dlr(delta_iw[name]), n_iw)
    g << inverse(iOmega_n - float(np.real(h0_c[c])) - delta_uni)
  Sigma_iw = inverse(G0_iw) - inverse(G_iw)
  if tail_fit_params is not None:
    window = resolve_tail_fit_window(tail_fit_params, beta, n_iw)
    known = {name: np.array([[[sigma_inf_c[c]]]], dtype=complex) for c, (name, _) in enumerate(gf_struct)}
    Sigma_iw = tail_fit(Sigma_iw, fit_min_n=window["fit_min_n"], fit_max_n=window["fit_max_n"],
                        fit_max_moment=window["fit_max_moment"], fit_known_moments=known)
    G_iw = inverse(inverse(G0_iw) - Sigma_iw)
  return Sigma_iw, G_iw, G0_iw


# ---------------------------------------------------------------- chi from Q(tau)

def chi_from_q_tau(Q_tau, gf_struct, wn_b, beta, dlr_wmax, dlr_eps):
  """
  The plateau-connected colour correlator at the bosonic Matsubara integers ``wn_b`` from the
  measured ``<n_a(tau) n_b(0)>`` (``Block2Gf``, one ``(1, 1)`` block per colour pair on a uniform
  tau grid, equal-time part included: the fork's ``Q_tau`` or ``nn_tau``).

  Returns
  -------
  chi_c : np.ndarray (len(wn_b), n_color, n_color), real
  w_max_fit : float
      The DLR cutoff the fit used (``fit_tau_correlator``).
  """
  names = [name for name, _ in gf_struct]
  n_color = len(names)
  chi_c = np.zeros((len(wn_b), n_color, n_color))
  w_max_fit = dlr_wmax
  for c1, n1 in enumerate(names):
    for c2, n2 in enumerate(names):
      q = Q_tau[n1, n2].copy()
      conn, _ = connected_by_plateau(np.asarray(Q_tau[n1, n2].data).real, beta)
      q.data[:] = conn
      q_dlr, w_max_fit = fit_tau_correlator(q, dlr_wmax, dlr_eps)
      chi_c[:, c1, c2] = eval_on_points(q_dlr, wn_b, "Boson", beta)[:, 0, 0].real
  return chi_c, w_max_fit
