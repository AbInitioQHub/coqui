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
from itertools import product

from triqs.gfs import (iOmega_n, MeshImFreq, MeshDLRImFreq, Gf, BlockGf,
                      make_gf_dlr, make_gf_imfreq, make_hermitian) 
from triqs.gfs.gf_fnt import fit_hermitian_tail_on_window, replace_by_tail
from triqs.gfs.tools import inverse, make_zero_tail

import triqs_modest as modest
from triqs_ctseg import Solver as CTSEG_Solver

def post_process(solver, **post_proc_params):
    
    post_process_sigma(solver, **post_proc_params)
    
    if solver.D0_tau is not None and solver.results.nn_nu_dlr is not None:
        if post_proc_params['degenerate_blk']:
            deg_blk_2e = []
            for blks in post_proc_params['degenerate_blk']:
                n_half = len(blks) // 2
                deg_blk_2e.append(np.array(blks[:n_half]))
        else:
            deg_blk_2e = None
        post_process_pi(solver, deg_blk_2e,
                        truncate_uchi=post_proc_params['truncate_uchi'])


def post_process_pi(solver, degenerate_blk=None, output_in_4idx=False,
                    truncate_uchi=False):
    #mpi.report("Charge susceptibility is measured for a impurity with dynamic interactions")
    #mpi.report('--> Post-processing the density-density susceptibility to obtain the impurity polarizability.\n')

    n_color = 0
    for _, blk_dim in solver.gf_struct:
        n_color += blk_dim
    assert n_color % 2 == 0, "Oh oh... n_color is an odd number." 
    n_orb = n_color// 2 

    block_name       = []
    index_in_block   = []
    color_to_orbital = []  # mapping from color to orbital
    for color in range(n_color):
        block_name.append(find_block_name(color, solver.gf_struct))
        index_in_block.append(find_index_in_block(color, solver.gf_struct))
        color_to_orbital.append(find_orbital_index(color, solver.gf_struct))
    
    # initialization
    tau_mesh = solver.D0_tau[block_name[0], block_name[0]].mesh
    iw_mesh_dlr = solver.results.nn_nu_dlr[block_name[0], block_name[0]].mesh

    D0_tau = Gf(mesh=tau_mesh, target_shape=(n_color, n_color))
    nn_iw_dlr = Gf(mesh=iw_mesh_dlr, target_shape=(n_color, n_color))
    for c1, c2 in product(range(n_color), repeat=2):
        nn_iw_dlr.data[:, c1, c2] = (
            solver.results.nn_nu_dlr[block_name[c1], block_name[c2]].data[:, index_in_block[c1], index_in_block[c2]]
        )
        D0_tau.data[:, c1, c2] = (
            solver.D0_tau[block_name[c1], block_name[c2]].data[:, index_in_block[c1], index_in_block[c2]]
        )

    
    # density for the constant part of chi
    densities = np.zeros(n_color, dtype=float)
    for c1 in range(n_color):
        densities[c1] = solver.results.densities[block_name[c1]][index_in_block[c1]]
    #mpi.report(f"Average of time-dependent occupations: {densities}")

    #mpi.report("Subtracting the constant component, and then symmetrizing the density-density susceptibility: \n"
    #           "  1. nn(t).imag = 0.0\n"
    #           "  2. nn(i, j) = nn(j, i)\n")

    for c1, c2 in product(range(n_color), repeat=2):
        iw_idx = np.array([ iw.index for iw in nn_iw_dlr[c1,c2].mesh ])
        w0_idx = np.where(iw_idx==0)[0][0]
        if c1 >= c2:
            # remove the constant part
            nn_iw_dlr[c1,c2].data[w0_idx] -= iw_mesh_dlr.beta * (densities[c1] * densities[c2])
            # symmetrization
            nn_iw_dlr[c1,c2].data.imag = 0.0
            if c1 != c2:
                nn_iw_dlr[c2,c1].data[w0_idx] -= iw_mesh_dlr.beta * (densities[c2] * densities[c1])
                nn_iw_dlr[c1,c2].data[:] += nn_iw_dlr[c2,c1].data[:].real
                nn_iw_dlr[c1,c2].data[:] /= 2.0
                nn_iw_dlr[c2,c1] << nn_iw_dlr[c1,c2]

    nn_iw = make_gf_imfreq(make_gf_dlr(nn_iw_dlr), n_iw = solver.n_iw)

    D0_iw = nn_iw.copy()
    D0_iw << 0.0
    u_known_moments = make_zero_tail(D0_iw, n_moments=2)
    D0_iw.set_from_fourier(D0_tau, u_known_moments)

    # convert to density-density basis
    nn_iw_dd = Gf(mesh=nn_iw.mesh, target_shape=[n_orb, n_orb])
    for c1, c2 in product(range(n_color), repeat=2):
        nn_iw_dd[color_to_orbital[c1], color_to_orbital[c2]].data[:] += nn_iw[c1, c2].data[:]
    if degenerate_blk is not None:
        nn_iw_dd << modest.symmetrize(nn_iw_dd, degenerate_blk)
        
    # Convert to a product basis 
    nn_iw_pb = Gf(mesh=nn_iw.mesh, target_shape=[n_orb*n_orb, n_orb*n_orb])
    for i, j in product(range(n_orb), repeat=2):
        if i >= j:
            # only the real part
            nn_iw_pb[i*n_orb+i, j*n_orb+j] << nn_iw_dd[i,j].real
            if i != j:
                nn_iw_pb[j*n_orb+j, i*n_orb+i] << nn_iw_pb[i*n_orb+i, j*n_orb+j]


    # bosonic Weiss field in the product basis
    U_iw_pb = Gf(mesh=nn_iw_pb.mesh, target_shape=nn_iw_pb.target_shape)

    # U(iw) in density-density basis
    D0_iijj = D0_iw[:n_orb, n_orb:2*n_orb]
    # screening J if non-zero
    D0_ijij = D0_iijj - D0_iw[:n_orb, :n_orb]

    # Vijkl is the full 4-index tensor without the block structure in TRIQS's notation
    Vijkl = extract_u_tensor_from_h_int(h_int=solver.h_int, gf_struct=solver.gf_struct, return_4idx=True)

    for i, j in product(range(n_orb), repeat=2):
        if i == j:
            # intra-orbital density-density term
            U_iw_pb[i*n_orb+i, i*n_orb+i] << D0_iijj[i, i].real
            U_iw_pb[i*n_orb+i, i*n_orb+i].data[:] += Vijkl[i, i, i, i]
        if i > j:
            # inter-orbital density-density term
            U_iw_pb[i*n_orb+i, j*n_orb+j] << D0_iijj[i, j].real
            U_iw_pb[i*n_orb+i, j*n_orb+j].data[:] += Vijkl[i, j, i, j]
            U_iw_pb[j*n_orb+j, i*n_orb+i] << U_iw_pb[i*n_orb+i, j*n_orb+j]
            # Hund's J: Spin-flip (i, j, j, i)
            U_iw_pb[i*n_orb+j, i*n_orb+j] << D0_ijij[i, j].real
            U_iw_pb[i*n_orb+j, i*n_orb+j].data[:] += Vijkl[i, j, j, i]
            U_iw_pb[j*n_orb+i, j*n_orb+i] << U_iw_pb[i*n_orb+j, i*n_orb+j]
            # Hund's J: Pair hopping (i, j, i, j)
            U_iw_pb[i*n_orb+j, j*n_orb+i] << D0_ijij[i, j].real
            U_iw_pb[i*n_orb+j, j*n_orb+i].data[:] += Vijkl[i, i, j, j]
            U_iw_pb[j*n_orb+i, i*n_orb+j] << U_iw_pb[i*n_orb+j, j*n_orb+i]
    

    # Dyson equation: Pi(w) = Chi(w) * [U(w)*Chi(w) - I]^-1
    Pi_iw_pb = Gf(mesh=nn_iw_pb.mesh, target_shape=nn_iw_pb.target_shape)
    ones = np.eye(n_orb*n_orb, dtype=complex)
    for iwn in nn_iw_pb.mesh:
        UX = U_iw_pb[iwn] @ nn_iw_pb[iwn]
        UX = check_spectrum(UX, truncation=truncate_uchi)
        denom = UX - ones
        cond = np.linalg.cond(denom)
        if cond > 100: 
            mpi.report(f"WARNING: Large condition number for [U(w) * Chi(w) - I] = {cond} at n = {iwn.index}.")        
        Pi_iw_pb[iwn] = nn_iw_pb[iwn] @ np.linalg.pinv(denom)
        # explicit set Pi(iw).imag = 0.0
        Pi_iw_pb[iwn].imag = 0.0


    # Screened interaction W(w) = U(w) - U(w) * Chi(w) * U(w)
    W_iw_pb = Gf(mesh=nn_iw_pb.mesh, target_shape=nn_iw_pb.target_shape)
    for iwn in nn_iw_pb.mesh:
        W_iw_pb[iwn] = U_iw_pb[iwn] - U_iw_pb[iwn] @ nn_iw_pb[iwn] @ U_iw_pb[iwn]
    
    # Remove the static part 
    for i, j in product(range(n_orb), repeat=2):
        if i == j:
            # intra-orbital density-density term
            W_iw_pb[i*n_orb+i, i*n_orb+i].data[:] -= Vijkl[i, i, i, i]
        if i > j:
            # inter-orbital density-density term
            W_iw_pb[i*n_orb+i, j*n_orb+j].data[:] -= Vijkl[i, j, i, j]
            W_iw_pb[j*n_orb+j, i*n_orb+i] << W_iw_pb[i*n_orb+i, j*n_orb+j]
            # Hund's J: Spin-flip (i, j, j, i)
            W_iw_pb[j*n_orb+i, j*n_orb+i].data[:] -= Vijkl[i, j, j, i]
            W_iw_pb[i*n_orb+j, i*n_orb+j] << W_iw_pb[j*n_orb+i, j*n_orb+i]
            # Hund's J: Pair hopping (i, j, i, j)
            W_iw_pb[j*n_orb+i, i*n_orb+j].data[:] -= Vijkl[i, i, j, j]
            W_iw_pb[i*n_orb+j, j*n_orb+i] << W_iw_pb[j*n_orb+i, i*n_orb+j]

    # transform back to density-density basis
    # TODO is W(iw) density-density only for density-density interaction?
    if not output_in_4idx:
        # transform back to the density-density basis
        solver.Chi_iw = nn_iw_dd
        solver.Pi_iw  = Gf(mesh=nn_iw_pb.mesh, target_shape=(n_orb, n_orb))
        solver.W_iw   = solver.Pi_iw.copy()
        for i, j in product(range(n_orb), repeat=2):
            solver.Pi_iw[i,j] << Pi_iw_pb[i*n_orb+i, j*n_orb+j]
            solver.W_iw[i,j] << W_iw_pb[i*n_orb+i, j*n_orb+j]
        if degenerate_blk is not None:
            solver.Pi_iw << modest.symmetrize(solver.Pi_iw, degenerate_blk)
            solver.W_iw  << modest.symmetrize(solver.W_iw, degenerate_blk)
    else:
        # transform back to 4-index tensor
        solver.chi_iw = nn_iw_pb
        solver.Pi_iw  = Gf(mesh=nn_iw_pb.mesh, target_shape=(n_orb, n_orb, n_orb, n_orb))
        solver.W_iw  = solver.Pi_iw.copy()
        for i, j, k, l in product(range(n_orb), repeat=4):
            solver.Pi_iw[i,j,k,l] << Pi_iw_pb[i*n_orb+j, k*n_orb+l]
            solver.W_iw[i,j,k,l] << W_iw_pb[i*n_orb+j, k*n_orb+l]


# TODO return raw Sigma to solver.Sigma_iw_raw
def post_process_sigma(solver, **post_proc_params):
    
    # initialization
    mesh = MeshImFreq(beta = solver.beta, statistic = "Fermion", n_iw = solver.n_iw)
    solver.Sigma_iw = BlockGf(mesh = mesh, gf_struct = solver.gf_struct)
    solver.Sigma_iw.zero()
    solver.Sigma_moments = None
    solver.G_iw = solver.Sigma_iw.copy()
    solver.G0_iw = solver.Sigma_iw.copy()
    
    # Fourier transform G(tau) to G(iw)
    Gf_known_moments = make_zero_tail(solver.G_iw, n_moments=2)
    for i, bl in enumerate(solver.G_iw.indices):
        Gf_known_moments[i][1] = np.eye(solver.G_iw[bl].target_shape[0])
        solver.G_iw[bl].set_from_fourier(solver.results.G_tau[bl], Gf_known_moments[i])
    solver.G_iw << make_hermitian(solver.G_iw)
    if post_proc_params['degenerate_blk']:
        solver.G_iw << modest.symmetrize(solver.G_iw, post_proc_params['degenerate_blk'])

    # compute fermionic Weiss field g(iw)
    Delta_iw = BlockGf(mesh = mesh, gf_struct = solver.gf_struct)
    Delta_known_moments = make_zero_tail(Delta_iw, n_moments=1)
    for i, bl in enumerate(solver.Delta_tau.indices):
        Delta_iw[bl].set_from_fourier(solver.Delta_tau[bl], Delta_known_moments[i])
        solver.G0_iw[bl] << inverse(iOmega_n - solver.h_loc0_mat[i] - Delta_iw[bl])
    solver.G0_iw << make_hermitian(solver.G0_iw)
    if post_proc_params['degenerate_blk']:
        solver.G0_iw << modest.symmetrize(solver.G0_iw, post_proc_params['degenerate_blk'])

    # Compute the HF self-energy as the first moment of the self-energy
    if post_proc_params['analytic_hf']:
        solver.Sigma_infty = compute_sigma_infty(solver, post_proc_params['degenerate_blk'])
        solver.Sigma_moments = {}
        for blk_name, hf_val in solver.Sigma_infty.items():
            mpi.report(f"Σ_HF {blk_name}:")
            mpi.report(f"    {hf_val}")
            solver.Sigma_moments[blk_name] = np.array([hf_val], dtype=complex)
        mpi.report("")
    else:
        solver.Sigma_infty = None
        solver.Sigma_moments = None

    # Compute Self-energy
    if solver.results.F_tau is None:
        mpi.report("Compute the self-energy via the Dyson equation.\n")
        solver.Sigma_iw = inverse(solver.G0_iw) - inverse(solver.G_iw)
    else:
        mpi.report("Compute the self-energy via the improved estimator.\n")
        F_iw = solver.G_iw.copy()
        F_iw << 0.0
        F_known_moments = make_zero_tail(F_iw, n_moments=1)
        for i, bl in enumerate(F_iw.indices):
            F_iw[bl].set_from_fourier(solver.results.F_tau[bl], F_known_moments[i])
        F_iw << make_hermitian(F_iw)
        if post_proc_params['degenerate_blk']:
            F_iw << modest.symmetrize(F_iw, post_proc_params['degenerate_blk'])

        for block, fw in F_iw:
            for iw in fw.mesh:
                solver.Sigma_iw[block][iw] = fw[iw] / solver.G_iw[block][iw]
    solver.Sigma_iw << make_hermitian(solver.Sigma_iw)
    if post_proc_params['degenerate_blk']:
        solver.Sigma_iw << modest.symmetrize(solver.Sigma_iw, post_proc_params['degenerate_blk'])

    if post_proc_params['perform_tail_fit']:
        solver.Sigma_iw = tail_fit(
            solver.Sigma_iw,
            fit_min_n=post_proc_params['fit_min_n'],
            fit_max_n=post_proc_params['fit_max_n'],
            fit_min_w=post_proc_params['fit_min_w'],
            fit_max_w=post_proc_params['fit_max_w'],
            fit_max_moment=post_proc_params['fit_max_moment'],
            fit_known_moments=solver.Sigma_moments
        )
    solver.Sigma_iw << make_hermitian(solver.Sigma_iw)
    if post_proc_params['degenerate_blk']:
        solver.Sigma_iw << modest.symmetrize(solver.Sigma_iw, post_proc_params['degenerate_blk'])

    # update G(iw) with the fitted self-energy
    solver.G_iw << inverse( inverse(solver.G0_iw) - solver.Sigma_iw )
    solver.G_iw << make_hermitian(solver.G_iw)
    if post_proc_params['degenerate_blk']:
        solver.G_iw << modest.symmetrize(solver.G_iw, post_proc_params['degenerate_blk'])

    if solver.Sigma_infty is None:
        solver.Sigma_infty = {}
        for block, gf in solver.Sigma_iw:
            tail, err = gf.fit_hermitian_tail()
            solver.Sigma_infty[block] = tail[0]
        mpi.report("Extracting the static self-energy via tail fitting:")
        for blk_name, hf_val in solver.Sigma_infty.items():
            mpi.report(f"Σ_HF {blk_name}:")
            mpi.report(f"    {hf_val}")
        mpi.report("")

    # remove the static part of the self-energy
    for blk, sigma_infty_value in solver.Sigma_infty.items():
        solver.Sigma_iw[blk] = solver.Sigma_iw[blk] - sigma_infty_value

    
def tail_fit(Sigma_iw,
             fit_min_n=None, fit_max_n=None, fit_min_w=None, fit_max_w=None,
             fit_max_moment=None, fit_known_moments=None):

    Sigma_iw_fit = Sigma_iw.copy()

    # Resolve defaults
    beta = Sigma_iw.mesh.beta
    if fit_min_w is not None: fit_min_n = int(0.5*(fit_min_w*beta/np.pi - 1.0))
    if fit_max_w is not None: fit_max_n = int(0.5*(fit_max_w*beta/np.pi - 1.0))
    if fit_min_n is None: fit_min_n = int(0.8*len(Sigma_iw.mesh)/2)
    if fit_max_n is None: fit_max_n = int(len(Sigma_iw.mesh)/2)
    if fit_max_moment is None: fit_max_moment = 3

    w_min = (2*fit_min_n + 1) * np.pi / beta
    w_max = (2*fit_max_n + 1) * np.pi / beta
    mpi.report("Tail fit for Σ(iω):")
    mpi.report(f"  fit window : ω = [{w_min:.4f}, {w_max:.4f}] a.u.")
    mpi.report(f"  max moment : {fit_max_moment}")
    mpi.report(f"  known moments provided : {'yes' if fit_known_moments is not None else 'no (using zero)'}")
    mpi.report("")

    if fit_known_moments is None:
        fit_known_moments = {}
        for name, sig in Sigma_iw_fit:
            shape = [0] + list(sig.target_shape)
            fit_known_moments[name] = np.zeros(shape, dtype=complex) # no known moments

    # Now fit the tails of Sigma_iw and replace the high frequency part with the tail expansion
    for name, sig in Sigma_iw_fit:

        tail, err = fit_hermitian_tail_on_window(
            sig,
            n_min = fit_min_n,
            n_max = fit_max_n,
            known_moments = fit_known_moments[name],
            # set max number of pts used in fit larger than mesh size, to use all data in fit
            n_tail_max = 10 * len(sig.mesh), 
            expansion_order = fit_max_moment
            )
        
        replace_by_tail(sig, tail, n_min=fit_min_n)        

    return Sigma_iw_fit


def extract_u_tensor_from_h_int(h_int, gf_struct, return_4idx=False):
    """
    Return Uijkl tensor in TRIQS's notation from a Coulomb many-body operator h_int
    """
    from triqs.operators.util.extractors import extract_U_dict2, dict_to_matrix

    # extract the density-density Coulomb matrix U_dd
    U_dd = dict_to_matrix(extract_U_dict2(h_int), gf_struct=gf_struct)
    if not return_4idx:
        return U_dd

    n_orb = U_dd.shape[0]//2

    # extract Uijij (inter- and intra-orbital Coulomb) and Uijji (Hund's coupling) terms
    # a) For static impurity problem, Us are the static screened interactions
    # b) For dynamic impurity problem, Us are the bare interactions
    Uijij = U_dd[:n_orb, n_orb:2*n_orb]
    Uijji = Uijij - U_dd[:n_orb, :n_orb]
    
    # construct full Uijkl tensor for static interaction
    Uijkl = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=complex)

    # assuming Uijji = Uiijj
    for i, j, k, l in product(range(n_orb), repeat=4):
        if i == j == k == l:  # Uiiii
            Uijkl[i, i, i, i] = Uijij[i, j]
        elif i == k and j == l:  # Uijij
            Uijkl[i, j, i, j] = Uijij[i, j]
        elif i == l and j == k:  # Uijji
            Uijkl[i, j, j, i] = Uijji[i, j]
        elif i == j and k == l:  # Uiijj
            Uijkl[i, i, k, k] = Uijji[i, k]
    
    return Uijkl


def extract_screen_matrix_from_D0_tau(blk2_D0_tau, gf_struct, return_4idx=False):
    n_color = 0
    for _, blk_dim in gf_struct:
        n_color += blk_dim

    block_name = []
    index_in_block = []
    for color in range(n_color):
        block_name.append(find_block_name(color, gf_struct))
        index_in_block.append(find_index_in_block(color, gf_struct))

    mesh = blk2_D0_tau[block_name[0], block_name[0]].mesh
    D0_tau = Gf(mesh=mesh, target_shape=(n_color, n_color))
    for c1 in range(n_color):
        for c2 in range(n_color):
            D0_tau.data[:, c1, c2] = (
                blk2_D0_tau[block_name[c1], block_name[c2]].data[:, index_in_block[c1], index_in_block[c2]]
            )

    w0_mesh = MeshImFreq(beta = D0_tau.mesh.beta, statistic = "Boson", n_iw = 1)
    D0_iw = Gf(mesh=w0_mesh, target_shape=D0_tau.target_shape)
    D0_iw.set_from_fourier(D0_tau, make_zero_tail(D0_iw, n_moments=2))
    Dw0_dd = D0_iw.data[0].real

    if not return_4idx:
        return Dw0_dd

    n_orb = Dw0_dd.shape[0]//2
    Dw0_iijj = Dw0_dd[:n_orb, n_orb:2*n_orb]
    Dw0_ijij = Dw0_iijj - Dw0_dd[:n_orb, :n_orb]

    # construct full Uijkl tensor for static interaction
    Dw0_ijkl = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=complex)

    for i, j, k, l in product(range(n_orb), repeat=4):
        if i == j == k == l:  # iiii
            Dw0_ijkl[i, i, i, i] = Dw0_iijj[i, j]
        elif i == k and j == l:  # ijij
            Dw0_ijkl[i, j, i, j] = Dw0_iijj[i, j]
        elif i == l and j == k:  # ijji
            Dw0_ijkl[i, j, j, i] = Dw0_ijij[i, j]
        elif i == j and k == l:  # iijj
            Dw0_ijkl[i, i, k, k] = Dw0_ijij[i, k]

    return Dw0_ijkl


def compute_sigma_infty(solver, degenerate_blk=None):
    mpi.report('\nEvaluating static impurity self-energy analytically in density-density basis:')
    Sigma_infty = {}

    # initialization
    n_color = 0
    for _, blk_dim in solver.gf_struct:
        n_color += blk_dim

    block_name       = []
    index_in_block   = []
    for color in range(n_color):
        block_name.append(find_block_name(color, solver.gf_struct))
        index_in_block.append(find_index_in_block(color, solver.gf_struct))

    # Bare Coulomb in density-density basis
    V = extract_u_tensor_from_h_int(h_int=solver.h_int, gf_struct=solver.gf_struct)
    # Add screening function at iw=0 in density-density basis
    U_w0 = V + extract_screen_matrix_from_D0_tau(blk2_D0_tau=solver.D0_tau, gf_struct=solver.gf_struct)

    # impurity density
    densities = np.zeros(n_color, dtype=float)
    for c1 in range(n_color):
        densities[c1] = solver.results.densities[block_name[c1]][index_in_block[c1]]

    # sigma_infty
    c1 = 0
    for blk_name, blk_dim in solver.gf_struct:
        Sigma_infty[blk_name] = np.zeros((blk_dim, blk_dim), dtype=float)

        for iorb in range(blk_dim):
            for c2 in range(n_color):
                Sigma_infty[blk_name][iorb, iorb] += densities[c2] * U_w0[c1, c2].real
            c1 += 1

    if degenerate_blk:
        Sigma_infty_list = modest.symmetrize(list(Sigma_infty.values()), degenerate_blk)
        for i, blk_name in enumerate(Sigma_infty.keys()):
            Sigma_infty[blk_name] = Sigma_infty_list[i]

    return Sigma_infty


def compute_sigma_infty_in_4idx(solver, degenerate_blk=None):
    mpi.report('\nEvaluating static impurity self-energy analytically using 4-index Coulomb tensors:')
    Sigma_infty = {}

    # Full 4-index tensors without the block structure in TRIQS notation
    # bare Coulomb
    Vijkl = extract_u_tensor_from_h_int(h_int=solver.h_int, gf_struct=solver.gf_struct, return_4idx=True)
    # screened Coulomb at w=0
    Uw0_ijkl = Vijkl + extract_screen_matrix_from_D0_tau(
        blk2_D0_tau=solver.D0_tau, gf_struct=solver.gf_struct, return_4idx=True)
    norb = Vijkl.shape[0]

    # compute density matrix without the block structure
    density_matrix = {"up": np.zeros((norb, norb)), "down": np.zeros((norb,norb))}
    o1 = [0, 0]
    for blk_name, blk_dim in solver.gf_struct:
        spin = "up" if blk_name[:2] == "up" else "down"
        s1 = 0 if blk_name[:2] == "up" else 1
        for iorb in range(blk_dim):
            density_matrix[spin][o1[s1]+iorb, o1[s1]+iorb] = solver.results.densities[blk_name][iorb]
        o1[s1] += blk_dim
        assert o1[s1] <= norb, "orbital offset exceeds band range"

    # compute HF self-energy
    o1 = [0, 0]
    for blk_name, blk_dim in solver.gf_struct:
        Sigma_infty[blk_name] = np.zeros((blk_dim, blk_dim), dtype=float)
        s1 = 0 if blk_name[:2] == "up" else 1

        # Sigma_HF_{ij} = \sum_{a,b} n_{ab} \left( 2 Uw0_{i a j b} - U_{i a b j} \right)
        for iorb, jorb in product(range(blk_dim), repeat=2):
            # inner loop needs to run over the entire norb and "spin"
            for inner in range(norb*2):
                spin_idx, orb_idx = inner//norb, inner%norb
                spin_inner = "up" if spin_idx == 0 else "down"
                # exchange diagram K coming only from the "same-spin" channel
                if s1 == spin_idx:
                    Sigma_infty[blk_name][iorb, jorb] -= (
                        density_matrix[spin_inner][orb_idx, orb_idx].real
                        * Vijkl[o1[s1]+iorb, orb_idx, orb_idx, o1[s1]+jorb].real
                    )
                # Hartree (Coulomb) diagram J
                Sigma_infty[blk_name][iorb, jorb] += (
                    density_matrix[spin_inner][orb_idx, orb_idx].real
                    * Uw0_ijkl[o1[s1]+iorb, orb_idx, o1[s1]+jorb, orb_idx].real
                )

        o1[s1] += blk_dim
        assert o1[s1] <= norb, "orbital offset exceeds band range"

    if degenerate_blk:
        Sigma_infty_list = modest.symmetrize(list(Sigma_infty.values()), degenerate_blk)
        for i, blk_name in enumerate(Sigma_infty.keys()):
            Sigma_infty[blk_name] = Sigma_infty_list[i]

    return Sigma_infty



def find_block_name(color, gf_struct):
    bl, colors_so_far = 0, 0
    for blk_name, blk_dim in gf_struct:
        colors_so_far += blk_dim
        if color < colors_so_far: 
            return blk_name
        bl+=1
    raise ValueError(f"Color index {color} out of bounds for gf_struct of total size {colors_so_far}")


def find_index_in_block(color, gf_struct):
    colors_so_far = 0
    for blk_name, blk_dim in gf_struct:
        colors_so_far += blk_dim
        if color < colors_so_far:
            return color - (colors_so_far - blk_dim)
    raise ValueError(f"Color index {color} out of bounds for gf_struct of total size {colors_so_far}")


def find_orbital_index(color, gf_struct):
    n_color = 0
    for _, blk_dim in gf_struct:
        n_color += blk_dim
    n_orb = n_color // 2
    
    colors_so_far = 0
    o_up, o_dn = 0, 0
    for blk_name, blk_dim in gf_struct:
        colors_so_far += blk_dim
        if blk_name[:2] == "up":
            if color < colors_so_far:
                return o_up + ( color - (colors_so_far - blk_dim) )
            else:
                o_up += blk_dim
            assert o_up <= n_orb, f"Spin up orbital index {o_up} excceds band range {n_orb}"
        else:
            if color < colors_so_far:
                return o_dn + ( color - (colors_so_far - blk_dim) )
            else:
                o_dn += blk_dim
            assert o_dn <= n_orb, f"Spin down orbital index {o_dn} excceds band range {n_orb}"    

    raise ValueError(f"Color index {color} out of bounds for gf_struct of total size {colors_so_far}")


def fill_dlr_imfreq_gf(g_iw, wmax, eps):

    assert isinstance(g_iw.mesh, MeshImFreq), (
        "fill_dlr_imfreq_gf: input Green's function should live on MeshImFreq."
    )

    mesh_dlr_iw = MeshDLRImFreq(
        beta=g_iw.mesh.beta, statistic=g_iw.mesh.statistic,
        w_max=wmax, eps=eps, symmetrize=True
    )

    mesh_dlr_idx = np.array([iw.index for iw in mesh_dlr_iw])
    max_dlr_idx = max(abs(mesh_dlr_idx[0]), abs(mesh_dlr_idx[-1]))
    assert max_dlr_idx <= g_iw.mesh.n_iw, (
        f"fill_dlr_imfreq_gf: g_iw.mesh.n_iw = {g_iw.mesh.n_iw} < maximum DLRImFreq index ({max_dlr_idx})."
    )

    if isinstance(g_iw, BlockGf):
        gf_struct = [(bl, gf.target_shape[0]) for (bl, gf) in g_iw]
        g_dlr_iw = BlockGf(mesh=mesh_dlr_iw, gf_struct=gf_struct)
        for w in g_dlr_iw.mesh:
            for block, gf in g_dlr_iw:
                gf[w] = g_iw[block](w)
        return g_dlr_iw

    elif isinstance(g_iw, Gf):
        g_dlr_iw = Gf(mesh=mesh_dlr_iw, target_shape=g_iw.target_shape)
        for w in g_dlr_iw.mesh:
            g_dlr_iw[w] = g_iw(w)
        return g_dlr_iw

    else:
        raise NotImplemented


def check_spectrum(A, radius=1.0, tol=0.0, truncation=False):
    """
    1) Compute eigenvalues of A
    2) Remove eigenvalues with |lambda| > radius (+ tol)
    3) Reconstruct A from the remaining eigenpairs using a pseudoinverse

    Parameters
    ----------
    A : (n, n) array_like
        Square matrix.
    radius : float
        Eigenvalue cutoff.
    tol : float
        Extra tolerance.
    truncation : bool
        If True, truncate unqualified eigenvalues and reconstruct A.

    Returns
    -------
    A_trunc : (n,n) ndarray if truncation is True else A
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")

    w, V = np.linalg.eig(A)

    # Qualification mask
    keep = np.abs(w) < (radius + tol)
    drop = ~keep

    w_keep = w[keep]
    w_drop = w[drop]
    V_keep = V[:, keep]

    if w_drop.size > 0:
        mpi.report(f"Unqualified eigenvalues (|λ| > {radius}):")
        for lam in w_drop:
            mpi.report(f"  {lam}")
        mpi.report("")

    if not truncation or w_keep.size == w.size:
        return A
    else:
        mpi.report(f"Reconstructing matrix with eigenvalues |λ| <= {radius} only.\n")
        n = A.shape[0]
        if w_keep.size == 0:
            return np.zeros((n, n), dtype=np.result_type(A, 1j))

        # least-squares reconstruction
        A_trunc = V_keep @ np.diag(w_keep) @ np.linalg.pinv(V_keep)
        return A_trunc