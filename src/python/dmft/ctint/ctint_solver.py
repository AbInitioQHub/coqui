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
TRIQS/ctint behind the impurity-solver interface.
"""
import textwrap

import numpy as np
from triqs.operators import n as n_op
from triqs_ctint import Solver

from coqui import app_log
from coqui.dmft.solver_api import (ImpuritySolver, ImpurityResult, SolverCapabilities,
                                   qmc_cycle_params, resolve_mu_search_params)
from coqui.dmft.triqs_utils import problem_to_triqs
from coqui.dmft.ctint import ctint_utils as cu

# keys of the author's benchmark runner that this version does not implement: rejected by name so a
# block written for the runner fails loudly instead of being forwarded to triqs_ctint
_NOT_IMPLEMENTED_KEYS = ("alpha_mode", "alpha_edge", "alpha_recenter", "alpha_min_sign",
                         "alpha_delta_offdiag", "alpha_cross", "alpha_delta_cross", "alpha_center_from",
                         "measure_chi", "chi_connected", "reuse_config", "restart_n_warmup_cycles")


def _mesh_indices(mesh):
  return np.array([w.index for w in mesh], dtype=np.int64)


def _solve_ctint(inputs, problem, params, *, occupations_only, alpha_center, alpha_delta, n_tau,
                 n_tau_bosonic, n_tau_chi2):
  """
  One TRIQS/ctint solve of the converted problem.

  Builds the solver on the problem's DLR controls, fills ``G0_iw = (iw - h0 - Delta)^-1`` per block
  and ``D0_iw = D/2`` (CT-INT's action carries no 1/2), builds the signed alpha tensor, and runs
  ``Solver.solve`` with the density, ``M(tau)`` and (production only) ``chiAB`` measurements.

  Parameters
  ----------
  inputs : TriqsInputs
      From ``problem_to_triqs``.
  problem : ImpurityProblem
  params : dict
      The QMC cycle controls (per-rank ``n_cycles``, ``n_warmup_cycles``, ``length_cycle``), an
      optional ``random_seed`` (base; made rank-dependent here) and anything forwarded to
      ``Solver.solve``.
  occupations_only : bool
      ``measure_M_tau=False``, ``post_process=False``, no ``chiAB``.
  alpha_center, alpha_delta : float
  n_tau, n_tau_bosonic, n_tau_chi2 : int

  Returns
  -------
  triqs_ctint.Solver
      After the solve; the caller reads its measured objects.

  Raises
  ------
  RuntimeError
      The solver's DLR meshes differ from the conversion's.
  """
  gf_struct = problem.gf_struct
  beta = problem.beta
  S = Solver(beta=beta, gf_struct=gf_struct, n_tau=int(n_tau), dlr_wmax=problem.dlr_wmax,
             dlr_eps=problem.dlr_eps, use_D=True, n_tau_dynamical_interactions=int(n_tau_bosonic))
  name0 = gf_struct[0][0]
  if not np.array_equal(_mesh_indices(S.G0_iw.mesh), _mesh_indices(inputs.mesh_f)):
    raise RuntimeError("ctint_solver::_solve_ctint: triqs_ctint's fermionic DLR mesh differs from the "
                       "one the inputs were built on")
  if not np.array_equal(_mesh_indices(S.D0_iw[name0, name0].mesh), _mesh_indices(inputs.mesh_b)):
    raise RuntimeError("ctint_solver::_solve_ctint: triqs_ctint's bosonic DLR mesh differs from the "
                       "one the inputs were built on")
  # G0 = (iw - h0 - Delta)^-1, a full matrix per block at every DLR node
  iw = 1j * (2 * _mesh_indices(inputs.mesh_f) + 1) * np.pi / beta
  for (name, size), h0 in zip(gf_struct, inputs.h0_blocks):
    delta = np.asarray(inputs.delta_iw[name].data)
    S.G0_iw[name].data[:] = np.linalg.inv(iw[:, None, None] * np.eye(size) - np.asarray(h0)[None] - delta)
  # D0 = D/2
  for name1, _ in gf_struct:
    for name2, _ in gf_struct:
      S.D0_iw[name1, name2].data[:] = 0.5 * inputs.u_weiss_iw[name1, name2].data
  # the alpha tensor: one shift pair per h_int term and per D0 channel
  d_tau = cu.d0_tau_colors(S.D0_iw, gf_struct, int(n_tau_bosonic))
  channels = [d_tau[:, ca, cb] for ca, cb in cu.d0_channel_colors(gf_struct)]
  alpha = cu.signed_alpha(inputs.h_int, channels, center=alpha_center, delta=alpha_delta)
  app_log(2, f"ctint signed alpha: centre {alpha_center}, delta {alpha_delta},\n"
             f"  {len(list(inputs.h_int))} h_int vertices + {len(channels)} D0 channels")

  solve_params = dict(params)
  # CoQuí-level keys that Solver.solve does not accept
  suppress_output = solve_params.pop('suppress_solver_output', False)
  output_file = solve_params.pop('solver_output_file', None)
  if solve_params.get('random_seed') is not None:
    solve_params['random_seed'] = cu.seed_for_rank(solve_params['random_seed'])
  solve_params.update(h_int=inputs.h_int, alpha=alpha, n_s=2, measure_density=True)
  if occupations_only:
    solve_params.update(measure_M_tau=False, post_process=False, measure_chiAB_tau=False)
  else:
    chi_ops = [n_op(name, i) for name, size in gf_struct for i in range(size)]      # every colour, in order
    solve_params.update(measure_M_tau=True, post_process=True, measure_chiAB_tau=True,
                        chi_A_vec=chi_ops, chi_B_vec=chi_ops, n_tau_chi2=int(n_tau_chi2))
  with cu.redirect_solver_output(suppress_output, output_file):
    S.solve(**solve_params)
  return S


class CtintSolver(ImpuritySolver):
  """
  TRIQS/ctint behind the impurity-solver interface (minimal version).

  Density-density ``h_int`` with the signed alpha tensor; the density correlator is measured on
  every production solve, so the polarization is always returned; no configuration reuse.

  Solver-block keys: the QMC controls (``n_cycles`` total over ranks, ``n_warmup_cycles``,
  ``length_cycle``, ``random_seed``), ``screen_j`` (default False), ``n_tau_chi2`` (1001, the
  tau points of the density-correlator measurement, which dominates the cost), ``mu_search``.
  The tau grids of ``G0`` and ``D0`` follow the impurity DLR mesh, ``max(10001, 2 beta wmax + 1)``
  points; the signed alpha shifts are ``0.5 +- 0.51`` and the measured ``M(tau)`` is DLR-fitted at
  ``1e-6`` up to the DLR ``wmax``. These may be overridden by name (``n_tau``, ``n_tau_bosonic``,
  ``alpha_center``, ``alpha_delta``, ``m_fit_eps``, ``m_fit_wmax``); ``h_int`` accepts
  ``"density"`` only; every other key is forwarded to ``triqs_ctint.Solver.solve``. The
  alpha-mode, chi and configuration-reuse keys of a fuller CT-INT setup are rejected by name.
  """
  name = "ctint"
  capabilities = SolverCapabilities(
    occupations_only_solve=True, general_h_int=False, offdiag_delta=True,
    complex_hamiltonian=False, retarded_interaction="density", polarization="always")

  def __init__(self, solver_params, *, comm_size):
    params = dict(solver_params)
    rejected = [key for key in _NOT_IMPLEMENTED_KEYS if key in params]
    rejected += [f"mu_search.{key}" for key in _NOT_IMPLEMENTED_KEYS if key in params.get('mu_search', {})]
    if rejected:
      raise ValueError(f"ctint_solver::CtintSolver: {rejected} are not implemented in this version "
                       f"of the CT-INT backend (density-density h_int, signed alpha, chi always "
                       f"measured, no configuration reuse)")
    self.h_int_kind = params.pop('h_int', 'density')
    if self.h_int_kind != 'density':
      raise ValueError(f"ctint_solver::CtintSolver: this version accepts h_int = \"density\" only "
                       f"(signed alpha), got {self.h_int_kind!r}")
    self.screen_j = bool(params.pop('screen_j', False))
    self.alpha_center = float(params.pop('alpha_center', 0.5))
    self.alpha_delta = float(params.pop('alpha_delta', 0.51))
    self.n_tau_override = params.pop('n_tau', None)                   # None: from the DLR mesh
    self.n_tau_bosonic_override = params.pop('n_tau_bosonic', None)
    self.n_tau_chi2 = int(params.pop('n_tau_chi2', 1001))
    self.m_fit_eps = float(params.pop('m_fit_eps', 1e-6))
    self.m_fit_wmax = params.pop('m_fit_wmax', None)          # None: the problem's dlr_wmax
    self.comm_size = comm_size
    mu_search = params.pop('mu_search', {})
    self.production_params = qmc_cycle_params(solver_params=params, comm_size=comm_size)
    self.mu_search_params = resolve_mu_search_params(
      production_params=params, mu_search=mu_search, comm_size=comm_size)

  def describe(self):
    caps = self.capabilities
    params = ', '.join(f'{k} = {self.production_params[k]!r}' for k in sorted(self.production_params))
    lines = [f"Impurity solver: CT-INT (triqs_ctint), h_int = density, screen_j = {self.screen_j}",
             f"  capabilities: occupations-only solve {caps.occupations_only_solve}, general h_int "
             f"{caps.general_h_int}, off-diagonal Delta {caps.offdiag_delta},\n"
             f"    complex Hamiltonian {caps.complex_hamiltonian}, retarded interaction "
             f"{caps.retarded_interaction}, polarization {caps.polarization}",
             f"  signed alpha: centre {self.alpha_center}, delta {self.alpha_delta}; tau grids from the "
             f"DLR mesh"
             + "".join(f", {name} {value}" for name, value in (("n_tau", self.n_tau_override),
                                                              ("n_tau_bosonic", self.n_tau_bosonic_override))
                       if value is not None)
             + f", n_tau_chi2 {self.n_tau_chi2}; M(tau) fit eps {self.m_fit_eps}",
             f"  cycles per rank: {self.production_params['n_cycles']} "
             f"(x {self.comm_size} ranks), warm-up {self.production_params['n_warmup_cycles']}, "
             f"cycle length {self.production_params['length_cycle']}",
             textwrap.fill("parameters: " + params, width=88, initial_indent="  ",
                           subsequent_indent="    ")]
    differs = {k: v for k, v in self.mu_search_params.items()
               if k not in self.production_params or self.production_params[k] != v}
    lines.append("  chemical-potential search, if requested: as production except "
                 + ', '.join(f'{k} = {differs[k]!r}' for k in sorted(differs)))
    return "\n".join(lines)

  def solve(self, problem, *, occupations_only=False, chkpt=None):
    """
    Solve one impurity problem with TRIQS/ctint. Collective over the MPI ranks.

    Steps:

    1. Convert the problem with ``problem_to_triqs`` (density-density, real, blocks as given).
    2. Reject blocks of unequal size (CT-INT's ``D0`` labels assume one size).
    3. Run the TRIQS solve (``_solve_ctint``): ``G0 = (iw - h0 - Delta)^-1``, ``D0 = D/2``, the
       signed alpha, the density and (production) ``M(tau)`` and ``chiAB`` measurements.
    4. Production: the backend's post-processing (``ctint_utils``): occupations from the density
       matrices; Hartree-Fock ``Sigma_infty`` with the D(0) Hartree term; G from the own DLR fit of
       ``M(tau)`` at the IAFT points, averaged over the degenerate blocks; ``Sigma = G0^-1 - G^-1``;
       chi from ``chiAB`` (plateau-connected, capped DLR fit) to Pi and W.
    5. Assemble the ``ImpurityResult``.

    Parameters
    ----------
    problem : ImpurityProblem
    occupations_only : bool, optional
        The cheap solve of the chemical-potential search. Default ``False``.
    chkpt : dict or None, optional
        Accepted for the interface; CT-INT has no configuration reuse, so it is ignored.

    Returns
    -------
    ImpurityResult
    """
    gf_struct = problem.gf_struct
    cu.d0_channel_colors(gf_struct)            # raises on unequal block sizes, before any solver is built
    inputs = problem_to_triqs(problem, h_int_kind="density", screen_j=self.screen_j,
                              real_hamiltonian=True)
    params = dict(self.mu_search_params if occupations_only else self.production_params)
    n_tau, n_tau_bosonic = cu.resolve_tau_grids(
      beta=problem.beta, dlr_wmax=problem.dlr_wmax, n_tau=self.n_tau_override,
      n_tau_bosonic=self.n_tau_bosonic_override)
    app_log(2, f"ctint tau grids from the DLR mesh: n_tau = {n_tau}, n_tau_bosonic = {n_tau_bosonic}, "
               f"n_tau_chi2 = {self.n_tau_chi2}")
    S = _solve_ctint(inputs, problem, params, occupations_only=occupations_only,
                     alpha_center=self.alpha_center, alpha_delta=self.alpha_delta, n_tau=n_tau,
                     n_tau_bosonic=n_tau_bosonic, n_tau_chi2=self.n_tau_chi2)
    rho = [0.5 * (np.asarray(S.density[b], dtype=complex) + np.asarray(S.density[b], dtype=complex).conj().T)
           for b in range(len(gf_struct))]
    occupations = {name: np.diag(rho[b]).real.copy() for b, (name, _) in enumerate(gf_struct)}
    dens_c = np.concatenate([np.diag(r).real for r in rho])
    sign = float(np.real(S.average_sign))
    diagnostics = {"average_sign": sign, "average_order": float(S.average_k)}
    if abs(sign) < 0.1:
      app_log(1, f" [WARNING] average sign {sign:.3f} < 0.1; the CT-INT results are unreliable\n"
                 f"  (try a larger alpha_delta, or a smaller problem)\n")
    if occupations_only:
      return ImpurityResult(orbital_occupations=occupations, density=float(dens_c.sum()),
                            diagnostics=diagnostics)

    iaft, beta = problem.iaft, problem.beta
    wn_f = iaft.wn_mesh('f')
    wn_b = iaft.wn_mesh('b', positive_only=True)
    D_c = cu.color_matrix_at_points(inputs.u_weiss_iw, gf_struct, wn_b, beta)       # D(nu), full D
    U_c = cu.static_color_matrix(inputs.h_int, gf_struct)
    groups = problem.degenerate_blk
    sigma_infty = cu.average_blocks(cu.sigma_infty_hf(rho, inputs.h_int, D_c[0], gf_struct), groups)
    w_max = problem.dlr_wmax if self.m_fit_wmax is None else float(self.m_fit_wmax)
    G = cu.g_from_m_tau(S, gf_struct, wn_f, beta, w_max, self.m_fit_eps)
    i0 = len(wn_f) // 2
    G_lib = cu.eval_on_points(S.G_iw, wn_f[i0:i0 + 1], "Fermion", beta)
    lib_dev = max(float(np.abs(gl[0] - g[i0]).max() / np.abs(g[i0]).max()) for gl, g in zip(G_lib, G))
    app_log(2, f"ctint: G from the DLR fit of M(tau) (w_max {w_max:g}, eps {self.m_fit_eps:g});\n"
               f"  the library's G (fit at dlr_eps {problem.dlr_eps:g}) differs by {lib_dev:.2e} at iw_0")
    G = cu.average_blocks(G, groups)
    g0 = cu.eval_on_points(S.G0_iw, wn_f, "Fermion", beta)
    sigma_full = [np.linalg.inv(a) - np.linalg.inv(b) for a, b in zip(g0, G)]
    sigma_dyn = cu.average_blocks([s - sigma_infty[b] for b, s in enumerate(sigma_full)], groups)
    # chi from chiAB(tau): connected by the plateau, DLR-fitted within what the grid resolves
    chi_tau = S.chiAB_tau.copy()
    conn, _ = cu.connected_by_plateau(np.asarray(S.chiAB_tau.data).real, beta)
    chi_tau.data[:] = conn
    chi_dlr, wmax_fit = cu.fit_tau_correlator(chi_tau, problem.dlr_wmax, problem.dlr_eps)
    chi_c = cu.eval_on_points(chi_dlr, wn_b, "Boson", beta).real
    piw = cu.pi_w_from_chi(chi_c, U_c, D_c, gf_struct, groups)
    diagnostics["max_cond"] = float(piw["max_cond"])
    app_log(2, f"ctint: chiAB DLR-fitted with w_max {wmax_fit:.3g}; Pi/W Dyson step max condition "
               f"number {piw['max_cond']:.1f}")
    return ImpurityResult(
      orbital_occupations=occupations, density=float(dens_c.sum()),
      Sigma_infty=sigma_infty, Sigma_iw_data=sigma_dyn, G_iw_data=G,
      Pi_iw_data=[piw["Pi_dd"].astype(complex)], W_iw_data=[piw["W_dd"].astype(complex)],
      diagnostics=diagnostics,
      solver_data={"M_tau": S.M_tau, "G_iw": S.G_iw,
                   "density": [np.asarray(S.density[b]) for b in range(len(gf_struct))],
                   "chiAB_tau": S.chiAB_tau})
