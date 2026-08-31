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
Utility functions for EDMFT.
"""
from h5 import HDFArchive
import numpy as np
import itertools

import coqui.dmft as coqui_dmft
from coqui import app_log


def make_h5_sumk_format(mlwf_h5, orb_list=None):
    l, SO = 2, 0
    n_inequiv_shells = 1
    with HDFArchive(mlwf_h5, 'a') as ar:
        dft_inp = ar["dft_input"]

        # modify proj_mat and band_window
        if orb_list is not None:
            proj_mat = dft_inp["proj_mat"]
            dft_inp["proj_mat"] = proj_mat[:, :, :, orb_list]

            shell = dft_inp["corr_shells"]
            shell[0]["dim"] = len(orb_list)
            dft_inp["corr_shells"] = shell

            shell = dft_inp["shells"]
            shell[0]["dim"] = len(orb_list)
            dft_inp["shells"] = shell

        # dummy values that are unused but required by SumkDFT
        kpts_w90 = dft_inp['kpts']
        dft_inp["n_k"] = kpts_w90.shape[0]
        dft_inp["k_dep_projection"] = 0
        dft_inp["n_shells"] = n_inequiv_shells
        dft_inp["n_corr_shells"] = n_inequiv_shells
        dft_inp["n_inequiv_shells"] = n_inequiv_shells
        dft_inp["corr_to_inequiv"] = [0]
        dft_inp["inequiv_to_corr"] = [0]

        dft_inp["n_reps"] = [1 for i in range(n_inequiv_shells)]
        dft_inp["dim_reps"] = [0 for i in range(n_inequiv_shells)]

        ll = 2 * l + 1
        lmax = ll * (SO + 1)
        dft_inp["T"] = [np.zeros([lmax, lmax], dtype=complex) for i in range(n_inequiv_shells)]


def get_proj_info(modest_proj):
    return {'proj_mat': modest_proj.P_k[:,:,np.newaxis],
            'band_window': modest_proj.band_window[:1],
            'kpts_w90': modest_proj.kpts}


def set_n_iw(iaft):
    iw_idx_f = iaft.wn_mesh('fermion', phys_notation=True)
    iw_idx_b = iaft.wn_mesh('boson', phys_notation=True)
    max_idx = max(abs(iw_idx_f[0]), abs(iw_idx_f[-1]), abs(iw_idx_b[0]), abs(iw_idx_b[-1]))
    return int(max_idx + 1)


def chemistry_to_product_basis(A_abcd_chem):
    if A_abcd_chem.ndim < 4:
        raise ValueError("chemistry_to_product_basis: "
                         "A_abcd must have at least 4 dimensions (..., norb, norb, norb, norb).")
    n1, n2, n3, n4 = A_abcd_chem.shape[-4:]
    if not (n1 == n2 == n3 == n4):
        raise ValueError("The last four axes must have equal length (nbnd).")

    A_abcd_pb = np.swapaxes(A_abcd_chem, -3, -4)
    return A_abcd_pb


def product_basis_to_density_density(A_abcd):
    if A_abcd.ndim < 4:
        raise ValueError("A must have at least 4 dimensions.")
    n1, n2, n3, n4 = A_abcd.shape[-4:]
    if not (n1 == n2 == n3 == n4):
        raise ValueError("The last four axes must have equal length (nbnd).")

    lead_shape = A_abcd.shape[:-4]
    out_shape = lead_shape + (n1, n1)
    A_ab = np.zeros(out_shape, dtype=A_abcd.dtype)

    for a, b in itertools.product(range(n1), repeat=2):
        if lead_shape:  # with leading dims, e.g. (nw, norb, norb, norb, norb)
            A_ab[..., a, b] = A_abcd[..., a, a, b, b]
        else:           # no leading dims, just (norb, norb, norb, norb)
            A_ab[a, b] = A_abcd[a, a, b, b]

    return A_ab


def density_density_to_product_basis(A_ab):
    if A_ab.ndim < 2:
        raise ValueError("A_ab must have at least 2 dimensions (..., norb, norb).")

    n1, n2 = A_ab.shape[-2:]
    if n1 != n2:
        raise ValueError("Last two dimensions of A_ab must be equal (norb, norb).")

    lead_shape = A_ab.shape[:-2]
    out_shape = lead_shape + (n1, n1, n1, n1)
    A_abcd = np.zeros(out_shape, dtype=A_ab.dtype)

    # Iterate over orbital indices
    for a, b in itertools.product(range(n1), repeat=2):
        if lead_shape:  # with leading dims, e.g. (nw, norb, norb)
            A_abcd[..., a, a, b, b] = A_ab[..., a, b]
        else:           # no leading dims, just (norb, norb)
            A_abcd[a, a, b, b] = A_ab[a, b]

    return A_abcd


def dynamic_4idx_u_to_dd_basis(u_wabcd_pb, nspin=2, screen_j=False):
    """
    :param u_wabcd_pb: bosonic Weiss field in the product basis
    :param nspin:
    :param screen_j:
    :return:
    """
    nw, nbnd = u_wabcd_pb.shape[:2]
    U  = np.zeros((nw,nbnd,nbnd), dtype=u_wabcd_pb.dtype)      # matrix for same spin
    Uprime = np.zeros((nw,nbnd,nbnd), dtype=u_wabcd_pb.dtype)  # matrix for opposite spin
    for m in range(nbnd):
        for mp in range(nbnd):
            U[:,m,mp]  = u_wabcd_pb[:,m,m,mp,mp] - u_wabcd_pb[:,m,mp,mp,m]
            Uprime[:,m,mp] = u_wabcd_pb[:,m,m,mp,mp]

    if not screen_j:
        # set same-spin U = opposite-spin U
        U[:] = Uprime[:]
    else:
        # fill diagonals of same-spin U
        for m in range(nbnd):
            U[:,m,m] = Uprime[:,m,m]

    U_s1s2 = np.empty((nw, nspin, nspin, nbnd, nbnd), dtype=u_wabcd_pb.dtype)
    U_s1s2[:,0,0], U_s1s2[:,1,1] = U, U
    U_s1s2[:,0,1], U_s1s2[:,1,0] = Uprime, Uprime

    return U_s1s2


def estimate_zero_moment(Aw, iw_mesh):
    """
    Estimate the zeroth moment (high-frequency constant term) of A(iω). 

    Assumes that at large Matsubara frequencies the function behaves as

        A(iω) ≈ t + B / (iω) + O(1/ω²),

    where `t` is the zeroth moment (constant term). The estimate uses the values
    of A(iω) at the last two frequencies in `iw_mesh` to extrapolate to the
    ω → ∞ limit.

    Parameters
    ----------
    Aw : array of complex, shape (nw, ...)
        Values of A(iω) on the Matsubara frequency mesh. Must be ordered such
        that the last entries correspond to the largest |ω|.
    iw_mesh : array of complex, shape (nw)
        Matsubara frequency mesh corresponding to `Aw`. The last two entries are
        used for the extrapolation.

    Returns
    -------
    t : complex
        Estimated zeroth moment (constant term) of A(iω) in the high-frequency
        expansion.

    Notes
    -----
    - Accuracy depends on how far into the asymptotic regime the last two
      frequencies are. If the frequencies are not sufficiently large, the
      estimate may be biased.
    """
    iw_m1 = iw_mesh[-1]
    iw_m2 = iw_mesh[-2]
    t = Aw[-1].real - (Aw[-1] - Aw[-2]).real * iw_m2 ** 2 / (
           iw_m2 ** 2 - iw_m1 ** 2)
    t = t.astype(complex)

    return t


def extract_h0_and_delta(g_weiss_wsab, iaft, high_freq_multiplier=10, make_hermitian=True):
    """
    Estimate the static one-body term h₀ (as t_sIab) and the hybridization function Δ(iω)
    from a Weiss Green's function G₀(iω) sampled on a fermionic Matsubara mesh.

    The method:
      1) Interpolate G₀(iω) to a few very large Matsubara frequencies (scaled by
         `high_freq_multiplier`) to probe the asymptotic regime.
      2) Construct W(iω) = iω·I - [G₀(iω)]⁻¹ and estimate its zeroth moment
         t_sIab = lim_{|ω|→∞} W(iω) via `estimate_zero_moment`.
      3) Build Δ(iω) from the Dyson-like relation:
            Δ(iω) = iω·I - t_sIab - [G₀(iω)]⁻¹.

    Parameters
    ----------
    g_weiss_wsab : ndarray, complex, shape (nw, nspin, nbnd, nbnd)
        Weiss Green's function G₀(iωₙ) on the fermionic Matsubara mesh returned by `iaft`.
        The leading dimension is frequency index; s,a,b are spin and orbital indices.
    iaft : IAFT object
    high_freq_multiplier : float, default 10
        Multiplier applied to the last few (three) IR fermionic frequencies (in IR notation)
        before converting to physical Matsubara frequencies, to push evaluation deep into
        the asymptotic region for a more stable moment estimate.

    Returns
    -------
    t_sIab_estimate : ndarray, complex, shape (nspin, nbnd, nbnd)
        Estimate of the static one-body term (zeroth moment) per spin block.
    delta_estimate : ndarray, complex, shape (nw, nspin, nbnd, nbnd)
        Estimated hybridization function Δ(iωₙ) on the original fermionic mesh.

    Notes
    -----
    - Accuracy of `t_sIab_estimate` depends on how large the interpolated frequencies are.
    """
    nspin = g_weiss_wsab.shape[1]
    
    # 1) Interpolate G0 to very high fermionic frequencies to improve the accuracy of high-frequency fitting
    iwn_interp = iaft.wn_mesh('f', phys_notation=True)[-3:] * high_freq_multiplier
    g_weiss_interp = iaft.w_interpolate(g_weiss_wsab, iwn_interp, 'f', phys_notation=True)
    iwn_interp = (2*iwn_interp.astype(float) + 1) * np.pi / iaft.beta
    weiss_tmp = np.zeros(g_weiss_interp.shape, dtype=complex)
    for n, g in enumerate(g_weiss_interp):
        for s in range(nspin):
            weiss_tmp[n, s] = 1j * iwn_interp[n] - np.linalg.inv(g[s])

    # 2) Fitting the zeroth moment as the non-interacting Hamiltonian
    t_sIab_estimate = estimate_zero_moment(weiss_tmp, iwn_interp)

    # 3) Construct Δ(iω) = iω·I - t_sIab - [G0(iω)]^{-1} on the original mesh
    iwn_mesh_imp = iaft.wn_mesh('f') * np.pi / iaft.beta
    delta_estimate = np.zeros(g_weiss_wsab.shape, dtype=complex)
    nbnd = t_sIab_estimate.shape[-1]
    for n in range(delta_estimate.shape[0]):
        for s in range(nspin):
            g_weiss_inv = np.linalg.inv(g_weiss_wsab[n, s])
            delta_estimate[n, s] = 1j * iwn_mesh_imp[n] * np.eye(nbnd) - t_sIab_estimate[s] - g_weiss_inv

    if make_hermitian:
        # enforce hermiticity of Δ(iω)
        nw = delta_estimate.shape[0]
        for n_neg in range(nw//2+nw%2):
            n_pos = nw - 1 - n_neg
            if n_pos == n_neg:
                continue
            for s in range(nspin):
                delta_estimate[n_pos, s] = 0.5 * (delta_estimate[n_pos, s] + delta_estimate[n_neg, s].conj().T)
                delta_estimate[n_neg, s] = delta_estimate[n_pos, s].conj().T

    # 4) checking the leakage of the resulting Δ(iω)
    iaft.check_leakage(delta_estimate, 'f', 'delta', w_input=True)
    app_log(1, "")

    return t_sIab_estimate, delta_estimate

# TODO merge call compute_weiss_fields_w internally to avoid code duplication
def init_weiss_fields_w(*, iaft, local_gf, init_imp_results="dc", density_only=False):
    # checking inputs
    missing = {"Gloc_t", "Wloc_t", "Vloc"} - local_gf.keys()
    if missing:
        raise ValueError(f"Missing keys in local_gf: {missing}")

    if init_imp_results not in {"dc", "zero"}:
        raise ValueError(f"init_imp_results must be 'dc' or 'zero', got {init_imp_results}")

    # bosonic first
    if init_imp_results == "dc":
        app_log(2, "Evaluate the bosonic Weiss field at the RPA level.")
        pi_imp_w = iaft.tau_to_w_phsym(eval_pi_rpa(local_gf["Gloc_t"], density_only=density_only), stats='b')
        if len(pi_imp_w.shape) == 3:
            pi_imp_w = density_density_to_product_basis(pi_imp_w)
    else:
        app_log(2, "Evaluate the bosonic Weiss fields using zero impurity polarizability.")
        pi_imp_w = None

    u_weiss_w = compute_weiss_boson_w(
        local_gf["Vloc"],
        iaft.tau_to_w_phsym(local_gf["Wloc_t"], stats='b'),
        pi_imp_w
    )
    iaft.check_leakage_phsym(u_weiss_w, 'b', 'u_weiss', w_input=True)
    app_log(2, "")

    # fermionic
    if init_imp_results == "dc":
        app_log(2, "Evaluate the fermionic Weiss field using the local GW self-energy.")
        vhf_imp = eval_hf_dc(
            -iaft.tau_interpolate(local_gf["Gloc_t"], iaft.beta, stats='f')[0],
            local_gf["Vloc"],
            u_weiss_w[0]+local_gf["Vloc"]
        )
        sigma_imp_w = iaft.tau_to_w(eval_gw_dc_t(local_gf["Gloc_t"], local_gf["Wloc_t"]), stats='f')
    else:
        app_log(2, "Evaluate the fermionic Weiss field using zero impurity self-energy.")
        vhf_imp, sigma_imp_w = None, None

    g_weiss_w = compute_weiss_fermion_w(
        iaft.tau_to_w(local_gf["Gloc_t"], stats='f'),
        vhf_imp, sigma_imp_w
    )
    iaft.check_leakage(g_weiss_w, 'f', 'g_weiss', w_input=True)
    app_log(2, "")

    return g_weiss_w, u_weiss_w



def compute_weiss_fields_w(*, iaft, local_gf, impurity_selfenergies, density_only=False):
    # checking inputs 
    missing = {"Gloc_t", "Wloc_t", "Vloc"} - local_gf.keys()
    if missing:
        raise ValueError(f"Missing keys in local_gf: {missing}")

    missing = {"Vhf_imp", "Sigma_imp_w", "Pi_imp_w"} - impurity_selfenergies.keys()
    if missing:
        raise ValueError(f"Missing keys in impurity_selfenergies: {missing}")

    # bosonic first 
    app_log(2, "Evaluate the bosonic Weiss field using the provided impurity polarizability.")
    pi_imp_w = impurity_selfenergies["Pi_imp_w"]
    # check if pi_imp_w contains only density-density
    if len(pi_imp_w.shape) == 3:
        pi_imp_w_pb = density_density_to_product_basis(pi_imp_w)
    else:
        pi_imp_w_pb = pi_imp_w
        
    u_weiss_w = compute_weiss_boson_w(
        local_gf["Vloc"],
        iaft.tau_to_w_phsym(local_gf["Wloc_t"], stats='b'),
        pi_imp_w_pb
    )
    iaft.check_leakage_phsym(u_weiss_w, 'b', 'u_weiss', w_input=True)
    app_log(2, "")

    # fermionic 
    app_log(2, "Evaluate the fermionic Weiss field using the provided impurity self-energy.")
    vhf_imp = impurity_selfenergies["Vhf_imp"]
    sigma_imp_w = impurity_selfenergies["Sigma_imp_w"]

    g_weiss_w = compute_weiss_fermion_w(
        iaft.tau_to_w(local_gf["Gloc_t"], stats='f'),
        vhf_imp, sigma_imp_w
    )

    iaft.check_leakage(g_weiss_w, 'f', 'g_weiss', w_input=True)
    app_log(2, "")

    return g_weiss_w, u_weiss_w


def compute_weiss_fermion_w(g_wsab, vhf_sab, sigma_wsab):
    if vhf_sab is None:
        assert sigma_wsab is None, (
            "compute_weiss_fermion_w: sigma_wsab should be None if vhf_sab is None"
        )
        return g_wsab.copy()

    nw, nspin, nbnd = g_wsab.shape[:3]
    g_weiss_wsab = np.zeros(g_wsab.shape, dtype=g_wsab.dtype)

    #  g(w) = [G(w)^{-1} + Sigma_imp]^{-1]
    for ws in range(nw*nspin):
        w = ws // nspin
        s = ws % nspin
        X_inv = np.linalg.solve(g_wsab[w, s], np.eye(nbnd)) + vhf_sab[s] + sigma_wsab[w, s]
        g_weiss_wsab[w, s] = np.linalg.solve(X_inv, np.eye(nbnd))

    return g_weiss_wsab


def compute_weiss_boson_w(V_abcd, W_wabcd, Pi_wabcd):
    if Pi_wabcd is None:
        return W_wabcd

    nbnd = V_abcd.shape[0]
    nbnd2 = nbnd*nbnd
    Wfull_pb = (W_wabcd + V_abcd[np.newaxis, ...]).reshape(-1,nbnd2,nbnd2)
    Pi_pb = Pi_wabcd.reshape(-1,nbnd2,nbnd2)

    # U(w) = W(w)[I + Pi(w)*W(w)]^{-1}
    U_pb = np.zeros(Wfull_pb.shape, dtype=Wfull_pb.dtype)
    for n, W in enumerate(Wfull_pb):
        X = np.eye(nbnd2) + Pi_pb[n] @ W
        cond = np.linalg.cond(X)
        if cond > 100: 
            app_log(1, f"WARNING: Large condition number for [I + Pi(w)*W(w)] = {cond} at n = {n}.")    
        U_pb[n] = W @ np.linalg.pinv(X)

    return U_pb.reshape(W_wabcd.shape) - V_abcd


def eval_pi_rpa(G_tsab, *, density_only=False):
    nts, nspin, nbnd = G_tsab.shape[:3]
    nts_half = nts//2 if nts%2==0 else nts//2 + 1
    spin_factor = -2.0 if nspin == 1 else -1.0
    if not density_only:
        pi_t = np.zeros((nts_half, nbnd, nbnd, nbnd, nbnd), dtype=complex)
        for t in range(nts_half):
            mt = nts-t-1
            pi_t[t] += spin_factor * np.einsum(
                'sbd,sca->abcd',
                G_tsab[t], G_tsab[mt]
            )
    else:
        pi_t = np.zeros((nts_half, nbnd, nbnd), dtype=complex)
        for t in range(nts_half):
            mt = nts-t-1
            for s in range(nspin):
                for a, b in itertools.product(range(nbnd), repeat=2):
                    pi_t[t, a, b] += spin_factor * G_tsab[t, s, a, b] * G_tsab[mt, s, b, a]
    return pi_t


def eval_hf_dc(Dm_sab, V_abcd, U0_abcd):
    """
    Evaluate the Hartree and exchange contributions to the self-energy.

    Parameters:
    - Dm_sab: Density matrix in dimensions (spin, band, band).
    - V_abcd: Bare interaction on a product basis (band, band, band, band)
    - U0_abcd: Static screened interaction on a product basis (band, band, band, band)

    Returns:
    - Hartree-Fock potential for an impurity with dynamic interactions.
    """

    Vhf_sab = np.zeros(Dm_sab.shape, dtype=Dm_sab.dtype)

    # Hartree contribution
    spin_factor = 2.0 if Dm_sab.shape[0] == 1 else 1.0
    for s in range(Dm_sab.shape[0]):
        Vhf_sab += spin_factor * np.einsum('dc,bacd->ab', Dm_sab[s], U0_abcd)

    # Exchange contribution
    Vhf_sab -= np.einsum('sab,aibj->sij', Dm_sab, V_abcd)

    return Vhf_sab


def eval_gw_dc_t(G_tsab, W_tabcd):
    """
    Evaluate the GW self-energy contribution to the impurity Green's function.

    Parameters:
    - G_tsab: Impurity Green's function in time, spin, and band indices.
    - W_tabcd: Dynamic interaction on a product basis.

    Returns:
    - GW self-energy contribution to the impurity Green's function.
    """

    nts, nts_half = G_tsab.shape[0], W_tabcd.shape[0]
    sigma_tsab = np.zeros(G_tsab.shape, dtype=G_tsab.dtype)
    for t in range(nts_half):
        mt = nts-t-1
        sigma_tsab[t] = -1 * np.einsum('sab,aibj->sij', G_tsab[t], W_tabcd[t])
        if mt != t:
            sigma_tsab[mt] = -1 * np.einsum('sab,aibj->sij', G_tsab[mt], W_tabcd[t])

    return sigma_tsab


def solve_gw_dc(G_t, V, W_t, u_weiss_iw, iaft, density_only=True,
                *, gf_struct=None):
    # TODO density-density approximation to V and W_t
    dm = -iaft.tau_interpolate(G_t, iaft.beta, stats='f')[0]
    vhf_dc = eval_hf_dc(dm, V, u_weiss_iw[0]+V)
    
    sigma_dc_iw = iaft.tau_to_w(eval_gw_dc_t(G_t, W_t), stats='f')

    # (niw, norb, norb) if density_only else (niw, norb, norb, norb, norb)
    pi_dc_iw = iaft.tau_to_w_phsym(eval_pi_rpa(G_t, density_only=density_only), stats='b')

    if gf_struct is not None:
        # convert to block matrix format
        vhf_dc = coqui_dmft.arr_to_blk_arr(vhf_dc, gf_struct)
        sigma_dc_iw = coqui_dmft.arr_to_blk_arr(sigma_dc_iw, gf_struct)
        pi_dc_iw = [pi_dc_iw]

    return {
        "Sigma_infty_dc": vhf_dc,
        "Sigma_iw_dc_data": sigma_dc_iw,
        "Pi_iw_dc_data": pi_dc_iw
    }


def embed_impurities(embeding_1e, embeding_2e, solver_results, spin_average=False):
    # A list of 3D arrays (w, i, j) with the length of the list = number of spins
    Sigma_imp_embed = embeding_1e.embed([ Res['Sigma_iw_data'] for Res in solver_results ])
    Vhf_imp_embed   = embeding_1e.embed([ Res['Sigma_infty'] for Res in solver_results ])
    Pi_imp_embed    = embeding_2e.embed([ Res['Pi_iw_data'] for Res in solver_results ])

    # The same applied to the DC terms
    Sigma_dc_embed = embeding_1e.embed([ Res['Sigma_iw_dc_data'] for Res in solver_results ])
    Vhf_dc_embed   = embeding_1e.embed([ Res['Sigma_infty_dc'] for Res in solver_results ])
    Pi_dc_embed    = embeding_2e.embed([ Res['Pi_iw_dc_data'] for Res in solver_results ])

    #combine spins to a single array and add auxiliary impurity index
    Sigma_imp_embed = np.stack(Sigma_imp_embed, axis=1)[:,:,None]
    Sigma_dc_embed  = np.stack(Sigma_dc_embed, axis=1)[:,:,None]
    Vhf_imp_embed   = np.stack(Vhf_imp_embed, axis=0)[:,None]
    Vhf_dc_embed    = np.stack(Vhf_dc_embed, axis=0)[:,None]
    Pi_imp_embed    = Pi_imp_embed[0]
    Pi_dc_embed     = Pi_dc_embed[0]

    if spin_average:
        # Average over spin
        Sigma_imp_embed = np.sum(Sigma_imp_embed, axis=1)[:, None] * 0.5
        Sigma_dc_embed = np.sum(Sigma_dc_embed, axis=1)[:, None] * 0.5
        Vhf_imp_embed = np.sum(Vhf_imp_embed, axis=0)[None, :] * 0.5
        Vhf_dc_embed = np.sum(Vhf_dc_embed, axis=0)[None, :] * 0.5

    local_sigma_w = {'imp': Sigma_imp_embed, 'dc': Sigma_dc_embed}
    local_hf      = {'imp': Vhf_imp_embed, 'dc': Vhf_dc_embed}
    local_pi_w    = {
        'imp': density_density_to_product_basis(Pi_imp_embed),
        'dc': density_density_to_product_basis(Pi_dc_embed)
    }

    return local_sigma_w, local_hf, local_pi_w


def hubbard_kanamori_coulomb(V_abcd):
    n_orb = V_abcd.shape[0]
    U, Up, J_pair, J_spin = 0.0, 0.0, 0.0, 0.0

    # intra-orbital U
    for i in range(n_orb):
        U += V_abcd[i, i, i, i]
    U /= n_orb

    if n_orb == 1:
        return U, Up, J_pair, J_spin

    # inter-orbital U
    for i in range(n_orb):
        for j in range(i+1, n_orb):
            if i != j:
                Up += V_abcd[i, i, j, j]
    Up /= (n_orb * (n_orb - 1)) / 2

    # Hund's coupling J: pair-hopping
    for i in range(n_orb):
        for j in range(i+1, n_orb):
            if i != j:
                J_pair += V_abcd[i, j, i, j]
    J_pair /= (n_orb * (n_orb - 1)) / 2

    # Hund's coupling J: spin-flip
    for i in range(n_orb):
        for j in range(i+1, n_orb):
            if i != j:
                J_spin += V_abcd[i, j, j, i]
    J_spin /= (n_orb * (n_orb - 1)) / 2

    if U.imag > 1e-8:
        app_log(1, f"Warning: complex value encountered in intra-orbital U.imag = {U.imag}.")
    if Up.imag > 1e-8:
        app_log(1, f"Warning: complex value encountered in inter-orbital U.imag = {Up.imag}.")
    if J_pair.imag > 1e-8:
        app_log(1, f"Warning: complex value encountered in pair-hopping J.imag = {J_pair.imag}.")
    if J_spin.imag > 1e-8:
        app_log(1, f"Warning: complex value encountered in spin-flip J.imag = {J_spin.imag}.")

    return U.real, Up.real, J_pair.real, J_spin.real


def make_hermitian(g_wIij):
    if isinstance(g_wIij, list):
        return [_make_hermitian(g) for g in g_wIij]
    else:
        return _make_hermitian(g_wIij)

def _make_hermitian(g_wIij):
    # This function assume g_wIij live on the Matsubara frequencies defined in IAFT. 
    # This is dangerous without proper checks. 
    g_wIij_herm = g_wIij.copy()
    if g_wIij.ndim == 1:
        g_wIij_herm = g_wIij_herm[:, None, None]
    elif g_wIij.ndim == 2:
        g_wIij_herm = g_wIij_herm[:, :, None, None]
    elif g_wIij.ndim == 3:
        g_wIij_herm = g_wIij_herm[:, None]
    else:
        raise ValueError("make_hermitian: g_wIij must have 1, 2, or 3 dimensions.")

    nw, nbnd = g_wIij_herm.shape[0], g_wIij_herm.shape[2]
    if g_wIij_herm.shape[3] != nbnd:
        raise ValueError("make_hermitian: The last two dimensions of g_wIij must be square (nbnd, nbnd).")

    for n_neg in range(nw//2+nw%2):
        n_pos = nw - 1 - n_neg
        if n_pos == n_neg:
            continue
        for I in range(g_wIij_herm.shape[1]):
            g_wIij_herm[n_pos,I] = 0.5 * (g_wIij_herm[n_pos,I] + g_wIij_herm[n_neg,I].conj().T)
            g_wIij_herm[n_neg,I] = g_wIij_herm[n_pos,I].conj().T
    
    return g_wIij_herm.reshape(g_wIij.shape)