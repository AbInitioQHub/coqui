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


def solve_dynamic_full_mesh(Delta_iw, h_loc0, D0_iw, h_int, **solver_interface_params):
    """
    Solve the impurity with dynamic interactions 

    Delta_iw : triqs.BlockGf
    h_loc0: triqs.operator
    D0_iw : triqs.Block2Gf 
    h_int: triqs.operator
    """
    gf_struct = [(bl, gf.target_shape[0]) for (bl, gf) in Delta_iw]
    beta  = Delta_iw.mesh.beta
    n_iw = Delta_iw.mesh.n_iw
    solver_interface_params.pop('n_iw', None)
    n_tau = solver_interface_params.pop('n_tau', 10001)
    n_tau_bosonic = solver_interface_params.pop('n_tau_bosonic', n_tau)
    
    S = Solver(gf_struct=gf_struct, beta=beta, n_tau=n_tau, n_tau_bosonic=n_tau_bosonic)

    # initialization for post proccessing 
    post_proc_params = {}
    post_proc_params['perform_tail_fit'] = solver_interface_params.pop('perform_tail_fit', False)
    post_proc_params['fit_max_moment']   = solver_interface_params.pop('fit_max_moment', 3)
    post_proc_params['fit_min_w']        = solver_interface_params.pop('fit_min_w', None)
    post_proc_params['fit_max_w']        = solver_interface_params.pop('fit_max_w', None)
    post_proc_params['fit_min_n']        = solver_interface_params.pop('fit_min_n', None)
    post_proc_params['fit_max_n']        = solver_interface_params.pop('fit_max_n', None)
    post_proc_params['analytic_hf']      = solver_interface_params.pop('analytic_hf', False)
    post_proc_params['degenerate_blk']   = solver_interface_params.pop('degenerate_blk', None)
    S.n_iw, S.beta, S.gf_struct = n_iw, beta, gf_struct     # useful and necessary for post-processing
    S.h_int = h_int
    S.h_loc0_mat = block_matrix_from_op(h_loc0, gf_struct)

    # prepare Delta_tau
    for block, delta in S.Delta_tau:
        delta_tau = make_gf_from_fourier(
            Delta_iw[block],                                          # Δ(iω)
            S.Delta_tau[block].mesh,                                  # time mesh
            fit_hermitian_tail(Delta_iw[block], make_zero_tail(Delta_iw[block], 1))[0] # tail
        )
        S.Delta_tau[block] << make_hermitian(delta_tau)
        
    # prepare D0_tau
    for name1, name2 in D0_iw.indices:
        Dt = make_gf_from_fourier(
            D0_iw[name1, name2],                                          # D0(iω)
            S.D0_tau[name1, name2].mesh,                                  # time mesh
            fit_hermitian_tail(D0_iw[name1, name2], make_zero_tail(D0_iw[name1, name2], n_moments=2))[0] # tail
        )
        S.D0_tau[name1, name2] << make_hermitian(Dt)
        
    # call solver
    solver_interface_params['measure_densities'] = True
    solver_interface_params['measure_F_tau']     = True
    solver_interface_params['measure_nn_tau']    = True
    S.solve(h_loc0=h_loc0, h_int=h_int, **solver_interface_params)
    
    # post process
    ctseg_utils.post_process(S, **post_proc_params)
    
    return SolverResults(# required
        G_iw  = S.G_iw,
        Sigma_infty = list(S.Sigma_infty.values()),
        Sigma_iw = S.Sigma_iw,
        Pi_iw = S.Pi_iw,
        W_iw = S.W_iw,
        # TODO return raw Sigma as well

        # optional
        G_tau = S.results.G_tau,
        orbital_occupations = S.results.densities,
        average_order = S.results.average_order_Delta,
        average_sign = S.results.average_sign
    )

def solve_dynamic_dlr_mesh(Delta_iw, h_loc0, D0_iw, h_int, **solver_interface_params):
    """
    Solve the impurity with dynamic interactions

    Delta_iw : triqs.BlockGf
    h_loc0: triqs.operator
    D0_iw : triqs.Block2Gf
    h_int: triqs.operator
    """
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
    S.solve(h_loc0=h_loc0, h_int=h_int, **solver_interface_params)

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
        average_sign = S.results.average_sign
    )
    

def solve_dynamic_imp(Delta_iw, h_loc0, U_iw, h_int, **solver_params):
    if isinstance(Delta_iw.mesh, MeshImFreq):
        return solve_dynamic_full_mesh(Delta_iw, h_loc0, U_iw, h_int, **solver_params)
    elif isinstance(Delta_iw.mesh, MeshDLRImFreq):
        return solve_dynamic_dlr_mesh(Delta_iw, h_loc0, U_iw, h_int, **solver_params)
    else:
        raise NotImplemented


def solve_density_dynamic_u_dlr_mesh(Delta_iw, h_loc0, D0_iw, h_int, **solver_interface_params):
    """
    Solve the impurity with dynamic interactions

    Delta_iw : triqs.BlockGf
    h_loc0: triqs.operator
    D0_iw : triqs.Block2Gf
    h_int: triqs.operator
    """
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
    with _redirect_solver_output(suppress_solver_output, solver_output_file):
        S.solve(h_loc0=h_loc0, h_int=h_int, **solver_interface_params)

    return SolverResults(# required
        orbital_occupations = S.results.densities,
        average_order = S.results.average_order_Delta,
        average_sign = S.results.average_sign
    )


def solve_density_dynamic_u(Delta_iw, h_loc0, U_iw, h_int, **solver_params):
    if isinstance(Delta_iw.mesh, MeshImFreq):
        raise NotImplemented
    elif isinstance(Delta_iw.mesh, MeshDLRImFreq):
        return solve_density_dynamic_u_dlr_mesh(Delta_iw, h_loc0, U_iw, h_int, **solver_params)
    else:
        raise NotImplemented