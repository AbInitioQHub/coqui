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
TRIQS/cthyb (the 4.0.x fork) behind the impurity-solver interface.
"""
import textwrap

import numpy as np
from triqs.gfs import make_gf_dlr, make_gf_imtime
from triqs_cthyb import Solver
from triqs_cthyb.util import orbital_occupations

from coqui import app_log
from coqui.dmft.solver_api import (ImpuritySolver, ImpurityResult, SolverCapabilities,
                                   qmc_cycle_params, resolve_mu_search_params)
from coqui.dmft.triqs_utils import problem_to_triqs, h_int_density_density, symmetrize_h_int_op
from coqui.dmft import qmc_postproc as postproc
from coqui.dmft.cthyb import cthyb_utils as hu

HARTREE_EV = 27.211386245988
_H_INT_KINDS = ("density", "kanamori", "full")
# keys of the other backends that cthyb does not implement: rejected by name so a block written for
# CT-SEG or CT-INT fails loudly instead of being forwarded to the fork
_NOT_IMPLEMENTED_KEYS = ("reuse_config", "restart_n_warmup_cycles", "measure_chi", "chi_connected")
# former keys of this backend with no user-facing meaning any more: rejected by name so an old block
# fails loudly instead of being forwarded to the fork
_REMOVED_KEYS = ("lang_firsov", "n_l", "fit_min_n", "fit_max_n", "store_q_tau")
_N_L = 50                    # Legendre coefficients of G_l, which is never measured
_DYN_N_L = 300               # Legendre coefficients of the Lang-Firsov kernel
_N_TAU_BOSONIC_MIN = 100001  # floor of the tau grid the kernel is built on
_KERNEL_WARN = 1e-4          # Ha; above it the Monte Carlo weights see a distorted kernel


def _mesh_indices(mesh):
  return np.array([w.index for w in mesh], dtype=np.int64)


def _solve_cthyb(inputs, problem, params, *, occupations_only, lang_firsov, dyn_n_l, n_l,
                 n_iw, n_tau, n_tau_bosonic, d_tau):
  """
  One TRIQS/cthyb solve of the converted problem through the fork's ``delta_interface``.

  Parameters
  ----------
  inputs : TriqsInputs
  problem : ImpurityProblem
  params : dict
      The QMC cycle controls (per-rank ``n_cycles``, ``n_warmup_cycles``, ``length_cycle``), an
      optional ``random_seed`` (made rank-dependent here), the CoQuí output keys, and anything
      forwarded to ``Solver.solve``.
  occupations_only : bool
      ``measure_G_tau=False``, no ``D0_corr``, no perturbation order; the density matrix only.
  lang_firsov, dyn_n_l, n_l, n_iw, n_tau, n_tau_bosonic : the solver controls, resolved
  d_tau : np.ndarray (n_tau_bosonic, nc, nc)
      ``D(tau)`` as built (no 1/2: cthyb's action carries it).

  Returns
  -------
  triqs_cthyb.Solver
      After the solve.
  """
  gf_struct = problem.gf_struct
  S = Solver(beta=problem.beta, gf_struct=gf_struct, n_iw=int(n_iw), n_tau=int(n_tau), n_l=int(n_l),
             delta_interface=True, n_tau_bosonic=int(n_tau_bosonic))
  for name, g in S.Delta_tau:
    g << make_gf_imtime(make_gf_dlr(inputs.delta_iw[name]), int(n_tau))
  for c1, (n1, _) in enumerate(gf_struct):
    for c2, (n2, _) in enumerate(gf_struct):
      S.D0_tau[n1, n2].data[:, 0, 0] = d_tau[:, c1, c2]
  solve_params = dict(params)
  suppress_output = solve_params.pop('suppress_solver_output', False)
  output_file = solve_params.pop('solver_output_file', None)
  if solve_params.get('random_seed') is not None:
    solve_params['random_seed'] = postproc.seed_for_rank(solve_params['random_seed'])
  solve_params.update(h_int=inputs.h_int, h_loc0=inputs.h0_op, lang_firsov=bool(lang_firsov),
                      dyn_n_l=int(dyn_n_l), measure_density_matrix=True, use_norm_as_weight=True,
                      perform_post_proc=False)
  if occupations_only:
    solve_params.update(measure_G_tau=False, measure_D0_corr=False, measure_pert_order=False)
  else:
    # the density correlator <n_a(tau) n_b(0)> of every colour pair: from the occupation kinks when
    # every density commutes with h_loc (a density-density h_int), by the direct sweep otherwise;
    # the fork delivers both as nn_tau on n_tau_bosonic points, equal-time part included
    solve_params.update(measure_G_tau=True, measure_D0_corr=True, measure_nn_tau=True,
                        dlr_w_max=float(problem.dlr_wmax), dlr_eps=float(problem.dlr_eps),
                        measure_pert_order=True)
  with postproc.redirect_solver_output(suppress_output, output_file):
    S.solve(**solve_params)
  return S


class CthybSolver(ImpuritySolver):
  """
  TRIQS/cthyb (the 4.0.x fork) behind the impurity-solver interface.

  Any static ``h_int`` (``"density"``, ``"kanamori"``, ``"full"``); the retarded density-density
  interaction through the Lang-Firsov route; the impurity polarization from the measured density
  correlator on every production solve (from the occupation kinks for a density-density ``h_int``,
  by a direct measurement otherwise); size-1 blocks; no configuration reuse.

  Solver-block keys: the QMC controls (``n_cycles`` total over ranks, ``n_warmup_cycles``,
  ``length_cycle``, ``random_seed``), ``h_int`` ("density"), ``screen_j`` (False), the tail fit of
  the Dyson self-energy, ``perform_tail_fit`` (True) with its window ``fit_min_w`` / ``fit_max_w``
  (2.5 / 10.0 Ha; the one numerically critical choice: CT-HYB's Dyson Sigma turns to noise a few
  eV out, so the window depends on the problem and the budget) and ``fit_max_moment`` (7), and
  ``mu_search``. The meshes follow the impurity DLR mesh: ``n_iw = max(1.4 x largest DLR index,
  the index of fit_max_w + 1)``, ``n_tau = max(10001, 6 n_iw + 1)``, ``n_tau_bosonic = max(100001,
  2 beta wmax + 1)``; the Lang-Firsov kernel uses ``dyn_n_l = 300`` Legendre coefficients. These
  and the fork's own solve parameters may be overridden by name (``n_iw``, ``n_tau``,
  ``n_tau_bosonic``, ``dyn_n_l``; anything else is forwarded to ``triqs_cthyb.Solver.solve``).
  The configuration-reuse and chi keys of the other backends and the former ``lang_firsov``,
  ``n_l``, ``fit_min_n``, ``fit_max_n``, ``store_q_tau`` are rejected by name.
  """
  name = "cthyb"
  capabilities = SolverCapabilities(
    occupations_only_solve=True, general_h_int=True, offdiag_delta=False,
    complex_hamiltonian=False, retarded_interaction="density", polarization="always")

  def __init__(self, solver_params, *, comm_size):
    params = dict(solver_params)
    rejected = [key for key in _NOT_IMPLEMENTED_KEYS if key in params]
    rejected += [f"mu_search.{key}" for key in _NOT_IMPLEMENTED_KEYS if key in params.get('mu_search', {})]
    if rejected:
      raise ValueError(f"cthyb_solver::CthybSolver: {rejected} are not implemented in the CT-HYB backend "
                       f"(no configuration reuse; the polarization is measured whenever h_int is "
                       f"density-density)")
    removed = [key for key in _REMOVED_KEYS if key in params]
    if removed:
      raise ValueError(f"cthyb_solver::CthybSolver: {removed} are no longer solver keys (the Lang-Firsov "
                       f"route is always used, G_l is not measured, the tail-fit window is given in Ha "
                       f"as fit_min_w / fit_max_w, Q_tau is not stored)")
    self.h_int_kind = params.pop('h_int', 'density')
    if self.h_int_kind not in _H_INT_KINDS:
      raise ValueError(f"cthyb_solver::CthybSolver: h_int must be one of {_H_INT_KINDS}, got "
                       f"{self.h_int_kind!r}")
    self.screen_j = bool(params.pop('screen_j', False))
    # expert overrides of the derived meshes and the kernel expansion; None = derive in solve
    self.dyn_n_l = int(params.pop('dyn_n_l', _DYN_N_L))
    self.n_iw_override = params.pop('n_iw', None)
    self.n_tau_override = params.pop('n_tau', None)
    self.n_tau_bosonic_override = params.pop('n_tau_bosonic', None)
    perform_tail_fit = bool(params.pop('perform_tail_fit', True))
    fit = {"fit_max_moment": int(params.pop('fit_max_moment', 7)),
           "fit_min_w": float(params.pop('fit_min_w', 2.5)), "fit_max_w": float(params.pop('fit_max_w', 10.0)),
           "fit_min_n": None, "fit_max_n": None}
    self.tail_fit_params = fit if perform_tail_fit else None
    self.comm_size = comm_size
    mu_search = params.pop('mu_search', {})
    self.production_params = qmc_cycle_params(solver_params=params, comm_size=comm_size)
    self.mu_search_params = resolve_mu_search_params(
      production_params=params, mu_search=mu_search, comm_size=comm_size)

  def describe(self):
    caps = self.capabilities
    params = ', '.join(f'{k} = {self.production_params[k]!r}' for k in sorted(self.production_params))
    fit = ("off" if self.tail_fit_params is None else
           f"window {self.tail_fit_params['fit_min_w']}-{self.tail_fit_params['fit_max_w']} Ha, "
           f"max moment {self.tail_fit_params['fit_max_moment']}")
    lines = [f"Impurity solver: CT-HYB (triqs_cthyb 4.0.x fork), h_int = {self.h_int_kind}, "
             f"screen_j = {self.screen_j}",
             f"  capabilities: occupations-only solve {caps.occupations_only_solve}, general h_int "
             f"{caps.general_h_int}, off-diagonal Delta {caps.offdiag_delta},\n"
             f"    complex Hamiltonian {caps.complex_hamiltonian}, retarded interaction "
             f"{caps.retarded_interaction}, polarization {caps.polarization}",
             f"  Lang-Firsov kernel with dyn_n_l {self.dyn_n_l}; meshes from the DLR mesh"
             + "".join(f", {name} {value}" for name, value in (("n_iw", self.n_iw_override),
                                                              ("n_tau", self.n_tau_override),
                                                              ("n_tau_bosonic", self.n_tau_bosonic_override))
                       if value is not None)
             + f"; Sigma tail fit {fit}",
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
    Solve one impurity problem with TRIQS/cthyb. Collective over the MPI ranks.

    Steps:

    1. Convert the problem with ``problem_to_triqs`` (``h_int`` as configured, real, size-1 blocks).
    2. Resolve the meshes: ``n_iw`` at least the largest DLR fermionic index + 1 (so every IAFT point
       is read off the uniform mesh), ``n_tau`` at least ``6 n_iw + 1``.
    3. Build ``D(tau)`` on ``n_tau_bosonic`` points and, for a production solve with
       ``lang_firsov``, check the Lang-Firsov kernel pair by pair (``kernel_error_from_block2gf``),
       warning above 1e-4 Ha; the tail-fit window is resolved to mesh indices here, before the
       solve, so an empty window raises before any Monte Carlo.
    4. Run the TRIQS solve (``_solve_cthyb``): ``Delta_tau`` from the exact DLR Delta, ``D0_tau`` as
       built, the density matrix, and (production) ``G_tau`` and the density correlator ``nn_tau``.
    5. Production: occupations from the density matrix; ``Sigma_infty`` by Hartree-Fock with the
       D(0) Hartree term; ``G`` and ``Sigma`` on the uniform mesh (``sigma_from_g_tau``, tail fit),
       read at the IAFT points; ``chi`` from ``nn_tau`` to ``Pi`` and ``W`` with the density-density
       part of the problem's static interaction.
    6. Assemble the ``ImpurityResult``.

    Parameters
    ----------
    problem : ImpurityProblem
    occupations_only : bool, optional
        The cheap solve of the chemical-potential search. Default ``False``.
    chkpt : dict or None, optional
        Accepted for the interface; CT-HYB has no configuration reuse here, so it is ignored.

    Returns
    -------
    ImpurityResult
    """
    gf_struct = problem.gf_struct
    if any(size != 1 for _, size in gf_struct):
      raise ValueError(f"cthyb_solver::CthybSolver.solve: this backend takes blocks of size 1 only, "
                       f"got {gf_struct}")
    inputs = problem_to_triqs(problem, h_int_kind=self.h_int_kind, screen_j=self.screen_j,
                              real_hamiltonian=True)
    beta, iaft = problem.beta, problem.iaft
    n_iw, n_tau, n_tau_bosonic = hu.resolve_meshes(
      beta=beta, dlr_wmax=problem.dlr_wmax, max_dlr_index=int(np.abs(_mesh_indices(inputs.mesh_f)).max()),
      fit_max_w=None if self.tail_fit_params is None else self.tail_fit_params["fit_max_w"],
      n_iw=self.n_iw_override, n_tau=self.n_tau_override, n_tau_bosonic=self.n_tau_bosonic_override,
      n_tau_bosonic_min=_N_TAU_BOSONIC_MIN)
    app_log(2, f"cthyb meshes: n_iw = {n_iw} (uniform, covering the DLR range and the tail-fit window;\n"
               f"  G and Sigma are read at the IAFT points), n_tau = {n_tau}, n_tau_bosonic = {n_tau_bosonic},\n"
               f"  dyn_n_l = {self.dyn_n_l}")
    if self.tail_fit_params is not None:
      window = hu.resolve_tail_fit_window(self.tail_fit_params, beta, n_iw)       # raises before the solve
      lo, hi = self.tail_fit_params["fit_min_w"], self.tail_fit_params["fit_max_w"]
      app_log(2, f"cthyb: Sigma tail fit over mesh indices [{window['fit_min_n']}, {window['fit_max_n']})\n"
                 f"  (from {lo}-{hi} Ha = {float(lo) * HARTREE_EV:.0f}-{float(hi) * HARTREE_EV:.0f} eV when "
                 f"given that way, clipped to the mesh;\n  the window is problem- and budget-dependent), "
                 f"max moment {self.tail_fit_params['fit_max_moment']},\n  Sigma_infty as the known moment")
    d_tau = postproc.d0_tau_colors(inputs.u_weiss_iw, gf_struct, n_tau_bosonic)
    kernel_error = shift_error = None
    if not occupations_only:
      # the kernel the Monte Carlo weights see; pair by pair (memory), once per production solve
      # (D depends on u_weiss only, which the chemical-potential search does not change)
      kernel_error, k_max, shift_error = hu.kernel_error_from_block2gf(
        inputs.u_weiss_iw, gf_struct, beta, self.dyn_n_l, n_tau_bosonic, 4 * (n_tau_bosonic - 1) + 1)
      app_log(1, f"Lang-Firsov kernel K(tau) from dyn_n_l = {self.dyn_n_l} Legendre coefficients of D0\n"
                 f"  on n_tau_bosonic = {n_tau_bosonic} points: max error {kernel_error:.1e} Ha "
                 f"({100 * kernel_error / max(k_max, 1e-300):.3f} % of\n  max |K| = {k_max:.3f} Ha), "
                 f"static shift K'(0) off by {shift_error:.1e} Ha")
      if kernel_error > _KERNEL_WARN or shift_error > _KERNEL_WARN:
        app_log(1, f" [WARNING] the Lang-Firsov kernel error exceeds {_KERNEL_WARN:.0e} Ha; the Monte Carlo\n"
                   f"  weights see a distorted kernel. Raise dyn_n_l (truncation) and/or n_tau_bosonic\n"
                   f"  (grid floor); both cost set-up time only.\n")
    params = dict(self.mu_search_params if occupations_only else self.production_params)
    S = _solve_cthyb(inputs, problem, params, occupations_only=occupations_only,
                     lang_firsov=True, dyn_n_l=self.dyn_n_l, n_l=_N_L, n_iw=n_iw,
                     n_tau=n_tau, n_tau_bosonic=n_tau_bosonic, d_tau=d_tau)
    mirror_mismatch = None
    if kernel_error is not None:
      # the kernel the fork built, pinned to the Python mirror that produced kernel_error
      mirror_mismatch = hu.kernel_mirror_mismatch(getattr(S, "K_n", None), d_tau, beta, self.dyn_n_l)
      if mirror_mismatch is not None:
        app_log(2, f"cthyb: the fork's Lang-Firsov kernel K_n differs from the Python mirror by "
                   f"{mirror_mismatch:.1e}")
        if mirror_mismatch > 1e-8:
          app_log(1, f" [WARNING] the fork's Lang-Firsov kernel K_n differs from the Python mirror by\n"
                     f"  {mirror_mismatch:.1e} (Legendre coefficients): the reported kernel_error describes\n"
                     f"  a kernel the Monte Carlo did not use; the fork's kernel fit has changed.\n")
    occ_mat = orbital_occupations(S.density_matrix, gf_struct, S.h_loc_diagonalization)
    rho = [np.asarray(occ_mat[name], dtype=complex) for name, _ in gf_struct]
    occupations = {name: np.diag(rho[b]).real.copy() for b, (name, _) in enumerate(gf_struct)}
    dens_c = np.concatenate([np.diag(r).real for r in rho])
    diagnostics = {"average_sign": float(np.real(S.average_sign)), "n_iw": int(n_iw)}
    if kernel_error is not None:
      diagnostics.update({"kernel_error": kernel_error, "kernel_shift_error": shift_error})
      if mirror_mismatch is not None:
        diagnostics["kernel_mirror_mismatch"] = mirror_mismatch
    if occupations_only:
      return ImpurityResult(orbital_occupations=occupations, density=float(dens_c.sum()),
                            diagnostics=diagnostics)
    diagnostics.update({"average_order": float(S.average_order),
                        "auto_corr_time": float(np.max(np.asarray(S.auto_corr_time)))})
    dyn_ops = getattr(S, "dyn_vertex_operators", None)
    diagnostics["n_stochastic_dyn_vertex_types"] = (
      int(len(dyn_ops)) if getattr(S, "perturbation_order_dyn", None) is not None and dyn_ops is not None else 0)
    app_log(2, f"cthyb: {diagnostics['n_stochastic_dyn_vertex_types']} stochastic dynamical vertex types\n"
               f"  (the rest went through the Lang-Firsov route)")

    wn_f = iaft.wn_mesh('f')
    wn_b = iaft.wn_mesh('b', positive_only=True)
    groups = problem.degenerate_blk
    D_c = postproc.color_matrix_at_points(inputs.u_weiss_iw, gf_struct, wn_b, beta)
    sigma_infty = postproc.average_blocks(postproc.sigma_infty_hf(rho, inputs.h_int, D_c[0], gf_struct), groups)
    sigma_inf_c = np.array([float(np.real(s[0, 0])) for s in sigma_infty])
    h0_c = np.array([float(np.real(h[0, 0])) for h in inputs.h0_blocks])
    Sigma_iw, G_iw, G0_iw = hu.sigma_from_g_tau(S.G_tau, inputs.delta_iw, h0_c, sigma_inf_c, gf_struct, n_iw,
                                                groups, self.tail_fit_params)
    Sigma_dyn_iw = Sigma_iw.copy()
    for c, (name, g) in enumerate(Sigma_dyn_iw):
      g.data[:] -= sigma_inf_c[c]
    G = postproc.eval_uniform_then_dlr(G_iw, wn_f, "Fermion", beta, problem.dlr_wmax, problem.dlr_eps)
    sigma_dyn = postproc.eval_uniform_then_dlr(Sigma_dyn_iw, wn_f, "Fermion", beta, problem.dlr_wmax, problem.dlr_eps)
    G = postproc.average_blocks(G, groups)
    sigma_dyn = postproc.average_blocks(sigma_dyn, groups)
    solver_data = {"G_tau": S.G_tau, "Sigma_iw": Sigma_iw,
                   "density_matrix": [np.asarray(m) for m in S.density_matrix]}
    nn_tau = S.nn_tau
    q0 = np.array([float(np.real(nn_tau[name, name].data[0, 0, 0])) for name, _ in gf_struct])
    diagnostics["q0_mismatch"] = float(np.abs(q0 - dens_c).max())
    app_log(2, f"cthyb: nn_tau(0) diagonal vs occupations differ by at most {diagnostics['q0_mismatch']:.2e}")
    chi_c, wmax_fit = hu.chi_from_q_tau(nn_tau, gf_struct, wn_b, beta, problem.dlr_wmax, problem.dlr_eps)
    # the Dyson step of Pi and W acts in the density-density channel: the static colour matrix of
    # the density-density part of the problem's interaction (symmetrized over the degenerate blocks
    # as the solved operator is), whatever h_int the solver sampled
    h_int_dd = h_int_density_density(np.real(problem.v_abcd), gf_struct)
    if groups:
      h_int_dd = symmetrize_h_int_op(h_int_dd, groups, gf_struct)
    U_c = postproc.static_color_matrix(h_int_dd, gf_struct)
    piw = postproc.pi_w_from_chi(chi_c, U_c, D_c, gf_struct, groups)
    diagnostics["max_cond"] = float(piw["max_cond"])
    app_log(2, f"cthyb: nn_tau DLR-fitted with w_max {wmax_fit:.3g} Ha; Pi/W Dyson step max condition number "
               f"{piw['max_cond']:.1f}")
    pi_data, w_data = [piw["Pi_dd"].astype(complex)], [piw["W_dd"].astype(complex)]
    return ImpurityResult(
      orbital_occupations=occupations, density=float(dens_c.sum()),
      Sigma_infty=sigma_infty, Sigma_iw_data=sigma_dyn, G_iw_data=G,
      Pi_iw_data=pi_data, W_iw_data=w_data, diagnostics=diagnostics, solver_data=solver_data)
