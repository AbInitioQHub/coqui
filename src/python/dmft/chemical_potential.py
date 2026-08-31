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
Utility functions for impurity chemical potential
"""
import numpy as np

from triqs.gfs import inverse, iOmega_n
from triqs.operators.util.extractors import block_matrix_from_op
import coqui.dmft as coqui_dmft
from coqui import app_log


def compute_nelec(h0, delta_iw, sigma_infty, sigma_iw, mu):
    g_iw = sigma_iw.copy()
    for blk_idx, (blk_name, g) in enumerate(g_iw):
        g << inverse(iOmega_n + mu - h0[blk_idx] - sigma_infty[blk_idx] - delta_iw[blk_name] - sigma_iw[blk_name])

    nelec = 0.0
    for blk_name, occ in g_iw.density().items():
        nelec += np.diag(occ).sum()
    if nelec.imag >= 1e-6:
        app_log(1, f"WARNING: nelec.imag = {nelec.imag} >= 1e-6")

    return nelec.real


def compute_nelec_from_solver(gf_struct, h0_sab, delta_iw, u_weiss_iw, h_int, mu, **solver_params):
    # update h0 = h0 - mu_imp
    h0_mat_shifted = np.array([ h0_mat - np.eye(h0_mat.shape[0])*mu for h0_mat in h0_sab ])
    h0_op = coqui_dmft.h0_operator(h0_mat_shifted, gf_struct, force_real=True)

    density_res = coqui_dmft.ctseg.solve_density_dynamic_u(delta_iw, h0_op, u_weiss_iw, h_int, **solver_params)
    imp_density = 0.0
    for blk_name, occ in density_res['orbital_occupations'].items():
        imp_density += occ.sum()

    return imp_density


def compute_mu_impurity(nelec_target, compute_nelec_fcn, tolerance=1e-2, mu0=0):

    mu, delta = mu0, 0.2
    nelec = compute_nelec_fcn(mu=mu0)

    # Header
    app_log(1, "")
    app_log(1, "Impurity chemical potential search")
    app_log(1, " ----------------------------------")
    app_log(1, "")
    app_log(1, f"  target number of electrons (nelec) = {nelec_target:.6f}")
    app_log(1, f"  tolerance                         = {tolerance}")
    app_log(1, "")
    app_log(1, f"  {'iter':<4}  {'mu_imp':>8}  {'nelec':>11}  {'|nelec-target|':>20}")
    app_log(1, f"  {'-'*48}")

    def _log_iter(i, mu_val, nelec_val):
        app_log(1, f"  {i:<4d}  {mu_val:10.6f}  {nelec_val:12.6f}  {abs(nelec_val-nelec_target):12.6f}")

    it = 0
    _log_iter(it, mu0, nelec)
    if abs(nelec - nelec_target) < tolerance:
        app_log(1, "")
        app_log(1, f"  converged: mu_imp = {mu0:.6f}, nelec = {nelec:.6f}")
        return mu0, nelec

    if nelec >= nelec_target:
        mu1, mu2 = mu0-delta, mu0
        nelec1 = compute_nelec_fcn(mu=mu1)
        while nelec1 > nelec_target:
            mu1 -= delta
            nelec1 = compute_nelec_fcn(mu=mu1)
        it += 1
        _log_iter(it, mu1, nelec1)
        nelec2 = nelec
    else:
        mu1, mu2 = mu0, mu0+delta
        nelec2 = compute_nelec_fcn(mu=mu2)
        while nelec2 < nelec_target:
            mu2 += delta
            nelec2 = compute_nelec_fcn(mu=mu2)
        it += 1
        _log_iter(it, mu2, nelec2)
        nelec1 = nelec

    # Binary search
    while True:
        mu_mid = 0.5 * (mu1 + mu2)
        nelec_mid = compute_nelec_fcn(mu=mu_mid)
        it += 1
        _log_iter(it, mu_mid, nelec_mid)
        if abs(nelec_mid - nelec_target) < tolerance:
            mu = mu_mid
            nelec = nelec_mid
            break
        if nelec_mid >= nelec_target:
            mu2 = mu_mid
        else:
            mu1 = mu_mid

    app_log(1, "")
    app_log(1, f"  converged: mu_imp = {mu:.6f}, nelec = {nelec:.6f}")
    return mu, nelec
