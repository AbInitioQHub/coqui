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

import triqs.utility.mpi as mpi
import numpy as np
import os
import sys
from contextlib import contextmanager
from triqs.gfs import MeshImFreq, Gf, BlockGf, Block2Gf, MeshDLRImFreq, MeshImFreq, make_gf_from_fourier, fit_hermitian_tail, make_hermitian
from triqs.gfs import make_gf_imtime
from triqs.gfs.tools import make_zero_tail
from triqs.operators import c_dag, c
from triqs.operators.util.extractors import block_matrix_from_op

from triqs_ctseg import Solver
import coqui.dmft.ctseg.ctseg_utils as ctseg_utils
import coqui.dmft.ctseg.config_reuse as config_reuse
from coqui.dmft.solver_api import (ImpuritySolver, ImpurityResult, SolverCapabilities,
                                   qmc_cycle_params, resolve_mu_search_params)
from coqui.dmft.triqs_utils import problem_to_triqs


@contextmanager
def _redirect_solver_output(suppress_output=False, output_file=None):
    """Temporarily redirect native stdout/stderr during external solver calls."""
    if not suppress_output:
        # does nothing, output will be printed as usual
        yield
        return

    target_path = output_file if output_file else os.devnull
    target = open(target_path, "a", buffering=1)
    saved_stdout_fd = None
    saved_stderr_fd = None
    try:
        # avoids mixing already-buffered Python output with redirected output.
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        # save current stdout/stderr
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        # Redirect stdout/stderr to the target
        os.dup2(target.fileno(), 1)
        os.dup2(target.fileno(), 2)
        # run the code block with output redirected
        yield
    finally:
        try:
            # restore original stdout/stderr and close the backup descriptors
            if saved_stdout_fd is not None:
                os.dup2(saved_stdout_fd, 1)
                os.close(saved_stdout_fd)
            if saved_stderr_fd is not None:
                os.dup2(saved_stderr_fd, 2)
                os.close(saved_stderr_fd)
        finally:
            # close the target output file
            target.close()


class SmartList(list):
    def __init__(self, **kwargs): #self._data = { key : val for key, val in kwargs.items() }
        super().__init__(kwargs)


    def __getitem__(self, key):
        if isinstance(key, int): return self.__values()[key]
        #if isinstance(key, str): return self._data[key]
        if isinstance(key, str): return super().__getitem__(key)

    def __getattr__(self, key):
        try: return self._data[key]
        except KeyError: raise AttributeError(f"SmartList has no property {key}")

    def __iter__(self): return iter(self.__values())

    def __repr__(self): return repr(self._data)
    __str__ = __repr__

    def to_vector(self): return self.__values()

class SolverResults(dict):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
    def __getattr__(self, key): 
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"SolverResults has no property {key}")
    def __repr__(self):
        return "\n".join([f"{key:<12}" for key in self.keys() ])
        
    __str__ = __repr__

def matrix_to_many_body_operator(h_loc_matrix, gf_struct):
    h_loc0_mat = {block : h_loc_matrix[ibl].real for ibl, (block, _) in enumerate(gf_struct) }
    c_dag_vec  = {block : np.matrix([[c_dag(block, o) for o in range(bl_size)]]) for block, bl_size in gf_struct }
    c_vec      = {block : np.matrix([[c(block, o) for o in range(bl_size)]]) for block, bl_size in gf_struct }
    return sum(c_dag_vec[block]*h_loc0_mat[block]*c_vec[block] for block, bl_size in gf_struct)[0,0]


def _solve_dynamic_dlr_mesh(Delta_iw, h_loc0, D0_iw, h_int, chkpt=None, **solver_interface_params):
    """
    Solve the impurity with dynamic interactions

    Delta_iw : triqs.BlockGf
    h_loc0: triqs.operator
    D0_iw : triqs.Block2Gf
    h_int: triqs.operator
    chkpt: dict, optional
        {"h5", "impurity", "iteration", "save"}: where configuration reuse (``reuse_config``)
        reads the previous iteration's CT-SEG configuration and saves this one
    """
    # CoQuí-level keys that Solver.solve does not accept; a given random_seed is made rank-dependent
    reuse_config = solver_interface_params.pop('reuse_config', False)
    restart_n_warmup_cycles = solver_interface_params.pop('restart_n_warmup_cycles', None)
    if solver_interface_params.get('random_seed') is not None:
        solver_interface_params['random_seed'] = \
            ctseg_utils.seed_for_rank(solver_interface_params['random_seed'])
    gf_struct = [(bl, gf.target_shape[0]) for (bl, gf) in Delta_iw]
    beta, wmax, eps  = Delta_iw.mesh.beta, Delta_iw.mesh.w_max, Delta_iw.mesh.eps 

    # extract the largest DLR frequency index to determine the minimum required n_iw
    mesh_dlr_idx = np.array([iw.index for iw in Delta_iw.mesh])
    max_dlr_idx = max(abs(mesh_dlr_idx[0]), abs(mesh_dlr_idx[-1]))

    # check bosonic DLR mesh and update max_dlr_idx if necessary
    dlr_mesh_boson = next(iter(D0_iw))[1].mesh
    mesh_dlr_idx = np.array([iw.index for iw in dlr_mesh_boson])
    max_dlr_idx = max(abs(mesh_dlr_idx[0]), abs(mesh_dlr_idx[-1]), max_dlr_idx)

    # This is used only in post-processing, and should be large enough to cover the DLR frequencies. 
    # It should not affect the solver accuracy though (in principle!?). 
    n_iw = solver_interface_params.pop('n_iw', int(1.4*max_dlr_idx))
    # Use very extreme default values for high resolution in the measured G(tau)/F(tau). 
    # This should be fine since all measurements of G(tau)/F(tau) scales very mildly with n_tau (~O(1)!?).  
    n_tau = solver_interface_params.pop('n_tau', max(500001, int(2*beta*wmax)+1))
    n_tau_bosonic = solver_interface_params.pop('n_tau_bosonic', n_tau)
    solver_interface_params.setdefault('dlr_omega_max', wmax)
    solver_interface_params.setdefault('dlr_epsilon', eps)

    # check if n_iw is compatible with dlr mesh
    if max_dlr_idx > n_iw:
        mpi.report(f"WARNING: n_iw = {n_iw} is smaller than the maximum DLR frequency index"
                   f" ({max_dlr_idx}). Setting n_iw to {max_dlr_idx+1}.")
        n_iw = max_dlr_idx+1

    S = Solver(gf_struct=gf_struct, beta=beta, n_tau=n_tau, n_tau_bosonic=n_tau_bosonic)

    # initialization for post-processing
    post_proc_params = {}
    post_proc_params['perform_tail_fit'] = solver_interface_params.pop('perform_tail_fit', False)
    post_proc_params['fit_max_moment']   = solver_interface_params.pop('fit_max_moment', 3)
    post_proc_params['fit_min_w']        = solver_interface_params.pop('fit_min_w', None)
    post_proc_params['fit_max_w']        = solver_interface_params.pop('fit_max_w', None)
    post_proc_params['fit_min_n']        = solver_interface_params.pop('fit_min_n', None)
    post_proc_params['fit_max_n']        = solver_interface_params.pop('fit_max_n', None)
    post_proc_params['analytic_hf']      = solver_interface_params.pop('analytic_hf', True)
    post_proc_params['degenerate_blk']   = solver_interface_params.pop('degenerate_blk', None)
    post_proc_params['truncate_uchi']    = solver_interface_params.pop('truncate_uchi', False)
    S.n_iw, S.beta, S.gf_struct = n_iw, beta, gf_struct     # useful and necessary for post-processing
    S.h_int = h_int
    S.h_loc0_mat = block_matrix_from_op(h_loc0, gf_struct)

    # prepare Delta_tau
    S.Delta_tau << make_hermitian(make_gf_imtime(Delta_iw, n_tau))

    # prepare D0_tau
    for name1, name2 in D0_iw.indices:
        S.D0_tau[name1, name2] << make_hermitian(make_gf_imtime(D0_iw[name1, name2], n_tau))

    # call solver
    solver_interface_params['measure_densities'] = True
    solver_interface_params['measure_F_tau']     = True
    solver_interface_params['measure_nn_nu_dlr']     = True
    # configuration reuse: start from the previous iteration's configuration when it is usable
    initial_config, config_source_iter = config_reuse.read_initial_config(
        chkpt, reuse_config, beta, gf_struct, label="impurity solve")
    if initial_config is not None:
        solver_interface_params['n_warmup_cycles'] = restart_n_warmup_cycles
        S.initial_config = initial_config
    S.solve(h_loc0=h_loc0, h_int=h_int, **solver_interface_params)
    if initial_config is not None and not S.results.restarted_from_config:
        config_reuse.warn_restart_rejected(chkpt, "impurity solve", restart_n_warmup_cycles)
    if reuse_config and chkpt is not None and chkpt.get('save', False):
        config_reuse.save_to_chkpt(chkpt['h5'], chkpt['iteration'], chkpt['impurity'],
                                   S.final_config)

    # post process
    ctseg_utils.post_process(S, **post_proc_params)

    return SolverResults(# required
        G_iw  = ctseg_utils.fill_dlr_imfreq_gf(S.G_iw, wmax, eps),
        Sigma_infty = list(S.Sigma_infty.values()),
        Sigma_iw = ctseg_utils.fill_dlr_imfreq_gf(S.Sigma_iw, wmax, eps),
        Pi_iw = ctseg_utils.fill_dlr_imfreq_gf(S.Pi_iw, wmax, eps),
        W_iw  = ctseg_utils.fill_dlr_imfreq_gf(S.W_iw, wmax, eps),
        Chi_iw = ctseg_utils.fill_dlr_imfreq_gf(S.Chi_iw, wmax, eps),

        # optional
        orbital_occupations = S.results.densities,
        average_order = S.results.average_order_Delta,
        average_sign = S.results.average_sign,
        # configuration reuse (the defaults keep CT-SEG builds without the restart API working)
        restarted_from_config = bool(getattr(S.results, 'restarted_from_config', False)),
        warmup_trace = getattr(S.results, 'warmup_trace', None),
        config_source_iter = config_source_iter,
    )
    

def _solve_density_dynamic_u_dlr_mesh(Delta_iw, h_loc0, D0_iw, h_int, chkpt=None, **solver_interface_params):
    """
    Solve the impurity with dynamic interactions

    Delta_iw : triqs.BlockGf
    h_loc0: triqs.operator
    D0_iw : triqs.Block2Gf
    h_int: triqs.operator
    chkpt: dict, optional
        {"h5", "impurity", "iteration", "save"}: where configuration reuse (``reuse_config``)
        reads the previous iteration's CT-SEG configuration and saves this one
    """
    # CoQuí-level keys that Solver.solve does not accept; a given random_seed is made rank-dependent
    reuse_config = solver_interface_params.pop('reuse_config', False)
    restart_n_warmup_cycles = solver_interface_params.pop('restart_n_warmup_cycles', None)
    if solver_interface_params.get('random_seed') is not None:
        solver_interface_params['random_seed'] = \
            ctseg_utils.seed_for_rank(solver_interface_params['random_seed'])
    gf_struct = [(bl, gf.target_shape[0]) for (bl, gf) in Delta_iw]
    beta, wmax, eps  = Delta_iw.mesh.beta, Delta_iw.mesh.w_max, Delta_iw.mesh.eps
    n_tau = solver_interface_params.pop('n_tau', max(500001, int(2*beta*wmax)+1))
    n_tau_bosonic = solver_interface_params.pop('n_tau_bosonic', n_tau)
    suppress_solver_output = solver_interface_params.pop('suppress_solver_output', False)
    solver_output_file = solver_interface_params.pop('solver_output_file', None)

    S = Solver(gf_struct=gf_struct, beta=beta, n_tau=n_tau, n_tau_bosonic=n_tau_bosonic)

    # prepare Delta_tau
    S.Delta_tau << make_gf_imtime(Delta_iw, n_tau)

    # prepare D0_tau
    for name1, name2 in D0_iw.indices:
        S.D0_tau[name1, name2] << make_gf_imtime(D0_iw[name1, name2], n_tau)

    # remove unnecessary parameters
    solver_interface_params.pop('n_iw', 1025)
    solver_interface_params.pop('perform_tail_fit', False)
    solver_interface_params.pop('fit_max_moment', 3)
    solver_interface_params.pop('fit_min_w', None)
    solver_interface_params.pop('fit_max_w', None)
    solver_interface_params.pop('fit_min_n', None)
    solver_interface_params.pop('fit_max_n', None)
    solver_interface_params.pop('analytic_hf', False)
    solver_interface_params.pop('degenerate_blk', None)
    solver_interface_params.pop("truncate_uchi", None)

    # call solver
    solver_interface_params['measure_densities'] = True
    solver_interface_params['measure_G_tau']     = False
    solver_interface_params['measure_F_tau']     = False
    solver_interface_params['measure_nn_tau']    = False
    # configuration reuse: the chemical-potential search starts from the previous iteration's
    # configuration when it is usable, and never saves
    initial_config, config_source_iter = config_reuse.read_initial_config(
        chkpt, reuse_config, beta, gf_struct, label="mu search", log_level=2)
    if initial_config is not None:
        solver_interface_params['n_warmup_cycles'] = restart_n_warmup_cycles
        S.initial_config = initial_config
    with _redirect_solver_output(suppress_solver_output, solver_output_file):
        S.solve(h_loc0=h_loc0, h_int=h_int, **solver_interface_params)
    if initial_config is not None and not S.results.restarted_from_config:
        config_reuse.warn_restart_rejected(chkpt, "mu search", restart_n_warmup_cycles)

    return SolverResults(# required
        orbital_occupations = S.results.densities,
        average_order = S.results.average_order_Delta,
        average_sign = S.results.average_sign,
        # configuration reuse (the defaults keep CT-SEG builds without the restart API working)
        restarted_from_config = bool(getattr(S.results, 'restarted_from_config', False)),
        warmup_trace = getattr(S.results, 'warmup_trace', None),
        config_source_iter = config_source_iter,
    )


def _check_config_reuse(production, mu_search):
    """
    Validate, in place, the configuration-reuse keys of the production and the chemical-potential
    search parameter sets of a solver block with ``reuse_config`` on. Warm-ups are per rank and
    are not divided by the rank count.
    """
    def _restart_warmup(value, key):
        if value is None or int(value) < 1:
            raise ValueError(
                f"ctseg_solver::CtsegSolver: reuse_config = True requires {key} >= 1 "
                f"(warm-up cycles of a solve that starts from a stored configuration); "
                f"got {value!r}.")
        return int(value)

    production['restart_n_warmup_cycles'] = _restart_warmup(
        production.get('restart_n_warmup_cycles'), 'restart_n_warmup_cycles')
    mu_search['restart_n_warmup_cycles'] = _restart_warmup(
        mu_search.get('restart_n_warmup_cycles', production['restart_n_warmup_cycles']),
        'solver.mu_search.restart_n_warmup_cycles')
    try:
        import triqs_ctseg
        has_reuse = hasattr(triqs_ctseg, 'ConfigDataT')
    except ImportError:
        has_reuse = False
    if not has_reuse:
        raise ImportError(
            "ctseg_solver::CtsegSolver: reuse_config = True requires a TRIQS/ctseg build "
            "that can restart from a saved configuration (triqs_ctseg.ConfigDataT).")


class CtsegSolver(ImpuritySolver):
    """
    TRIQS/CT-SEG behind the impurity-solver interface.

    Solver-block keys: the QMC controls (``n_cycles`` total over ranks, ``n_warmup_cycles``,
    ``length_cycle``, ``random_seed``, ...), ``screen_j`` (default False), ``h_int`` (``"density"``
    only), ``reuse_config`` / ``restart_n_warmup_cycles``, the mesh sizes ``n_iw``, ``n_tau``,
    ``n_tau_bosonic``, the post-processing keys ``perform_tail_fit``, ``fit_max_moment``,
    ``fit_min_w``, ``fit_max_w``, ``fit_min_n``, ``fit_max_n``, ``analytic_hf``, ``truncate_uchi``;
    every other key is forwarded to ``triqs_ctseg.Solver.solve``.
    """
    name = "ctseg"
    capabilities = SolverCapabilities(
        occupations_only_solve=True, general_h_int=False, offdiag_delta=False,
        complex_hamiltonian=False, retarded_interaction="density", polarization="always")

    def __init__(self, solver_params, *, comm_size):
        params = dict(solver_params)
        self.h_int_kind = params.pop('h_int', 'density')
        if self.h_int_kind != 'density':
            raise ValueError(f"ctseg_solver::CtsegSolver: CT-SEG accepts h_int = \"density\" only, "
                             f"got {self.h_int_kind!r}")
        self.screen_j = bool(params.pop('screen_j', False))
        self.comm_size = comm_size
        # the production parameters, and those of the occupations-only solves of the
        # chemical-potential search: the production ones with different QMC cycle controls
        mu_search = params.pop('mu_search', {})
        self.production_params = qmc_cycle_params(solver_params=params, comm_size=comm_size)
        self.mu_search_params = resolve_mu_search_params(
            production_params=params, mu_search=mu_search, comm_size=comm_size)

        if self.production_params.get('reuse_config', False):
            _check_config_reuse(self.production_params, self.mu_search_params)

    def describe(self):
        import textwrap
        caps = self.capabilities
        params = ', '.join(f'{k} = {self.production_params[k]!r}' for k in sorted(self.production_params))
        lines = [f"Impurity solver: CT-SEG (triqs_ctseg), h_int = density, screen_j = {self.screen_j}",
                 f"  capabilities: occupations-only solve {caps.occupations_only_solve}, general h_int "
                 f"{caps.general_h_int}, off-diagonal Delta {caps.offdiag_delta},\n"
                 f"    complex Hamiltonian {caps.complex_hamiltonian}, retarded interaction "
                 f"{caps.retarded_interaction}, polarization {caps.polarization}",
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
        Solve one impurity problem with TRIQS/CT-SEG. Collective over the MPI ranks.

        Steps:

        1. Convert the problem to TRIQS containers with ``problem_to_triqs`` (density-density
           interactions, real Hamiltonian, blocks of size 1: CT-SEG's capabilities).
        2. Call the TRIQS solver: production run or occupation-only run. 
        3. For a production solve, bring the TRIQS results back onto the IAFT sampling meshes
           and assemble the ``ImpurityResult``.  

        Parameters
        ----------
        problem : ImpurityProblem
            The problem, unreduced.
        occupations_only : bool, optional
            Run the cheap occupations-only solve of the chemical-potential search. Default
            ``False``.
        chkpt : dict or None, optional
            ``{"h5", "impurity", "iteration", "save"}``: where the configuration reuse reads
            the previous iteration's CT-SEG configuration and saves this one. Default ``None``.

        Returns
        -------
        ImpurityResult
            With ``Sigma_infty``, ``Sigma_iw_data``, ``G_iw_data``, ``Pi_iw_data`` and
            ``W_iw_data`` for a production solve; occupations, density and diagnostics only
            for an occupations-only solve.
        """
        inputs = problem_to_triqs(problem, h_int_kind=self.h_int_kind, screen_j=self.screen_j,
                                  real_hamiltonian=not self.capabilities.complex_hamiltonian)
        diagnostic_keys = ('average_order', 'average_sign', 'restarted_from_config',
                           'warmup_trace', 'config_source_iter')

        if occupations_only:
            # ct-seg to measure only the occupations. 
            params = dict(self.mu_search_params)
            params['degenerate_blk'] = problem.degenerate_blk   # popped and ignored by the density solve
            res = _solve_density_dynamic_u_dlr_mesh(
                inputs.delta_iw, inputs.h0_op, inputs.u_weiss_iw, inputs.h_int,
                chkpt=chkpt, **params)
            occupations = {name: np.asarray(res['orbital_occupations'][name], dtype=float)
                           for name, _ in problem.gf_struct}
            return ImpurityResult(
                orbital_occupations=occupations,
                density=float(sum(occ.sum() for occ in occupations.values())),
                diagnostics={key: res[key] for key in diagnostic_keys if key in res})
        
        # degenerate_blk is problem data, but CT-SEG's post-processing symmetrizes G, Sigma and Pi
        # with it (post_proc_params['degenerate_blk']), exactly as the driver passed it before
        params = dict(self.production_params)
        params['degenerate_blk'] = problem.degenerate_blk
        res = _solve_dynamic_dlr_mesh(
            inputs.delta_iw, inputs.h0_op, inputs.u_weiss_iw, inputs.h_int,
            chkpt=chkpt, **params)
        raw = ctseg_utils.imp_results_to_raw_data(
            res['G_iw'], res['Sigma_iw'], res['W_iw'], res['Pi_iw'], problem.iaft)
        occupations = {name: np.asarray(res['orbital_occupations'][name], dtype=float)
                       for name, _ in problem.gf_struct}
        return ImpurityResult(
            orbital_occupations=occupations,
            density=float(sum(occ.sum() for occ in occupations.values())),
            Sigma_infty=res['Sigma_infty'],
            Sigma_iw_data=raw['Sigma_iw_data'], G_iw_data=raw['G_iw_data'],
            Pi_iw_data=raw['Pi_iw_data'], W_iw_data=raw['W_iw_data'],
            diagnostics={key: res[key] for key in diagnostic_keys if key in res},
            solver_data={key: res[key] for key in ('G_iw', 'Sigma_iw', 'Pi_iw', 'W_iw', 'Chi_iw')})
