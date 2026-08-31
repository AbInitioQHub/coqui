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

import sys
import numpy as np


def bath_fit(delta_wsab, iw_mesh, nbath : int, verbal:bool = True):
    """
    Bath fitting using adapol package (https://github.com/flatironinstitute/adapol)
    :param delta_wsab: numpy.ndarray(dim=4)
        Hybridization function on the Matsubara axis (niw, nspin, nImpOrbs, nImpOrbs).
    :param iw_mesh: numpy.ndarray(dim=1)
        Matsubara frequencies (niw).
        Note: iw_mesh = 1j*n*pi/beta
    :param nbath: int
        Number of bath orbital per impurity orbital.
    :param verbal: bool
        verbal output for adapol or not

    :returns:
        - bath_energy: numpy.ndarray(dim=2)
          Energy of bath orbitals (nspin, nbath*nImpOrbs).
        - bath_hyb: numpy.ndarray(dim=3)
          Hybridization between the bath and impurity orbitals (nspin, nbath*nImpOrbs, nImpOrbs).
        - delta_fit: numpy.ndarray(dim=4)
          Fitted hybridization function (niw, nspin, nImpOrbs, nImpOrbs).
    """
    from adapol import hybfit
    if nbath % 2 != 1:
        raise ValueError("nbath must be an odd number due to the constraint from \"adapol\".")

    ns, nImpOrbs = delta_wsab.shape[1], delta_wsab.shape[2]

    print("Bath fitting with nspin = {}, nImpOrbs = {}, nbath/impurity orbital = {}.".format(
        ns, nImpOrbs, nbath))

    bath_energy = np.zeros((ns, nImpOrbs*nbath), dtype=complex)
    bath_hyb = np.zeros((ns, nImpOrbs*nbath, nImpOrbs), dtype=complex)
    delta_fit = np.zeros(delta_wsab.shape, dtype=complex)
    for s in range(ns):
        bath_energy[s], bath_hyb[s], final_error, func = hybfit(delta_wsab[:, s], iw_mesh, Np=nbath,
                                                                solver='sdp', verbose=verbal)
        delta_fit_spin = func(iw_mesh)
        delta_fit[:, s] = delta_fit_spin
        print("Bath fitting error at spin {}: {}".format(s, final_error))

    print("bath energy:")
    print("{}".format(bath_energy))
    print("bath hybridization:")
    print("{}".format(bath_hyb))
    print("Maximum imaginary part of hybridization: {}\n".format(np.max(np.abs(bath_hyb.imag))))
    sys.stdout.flush()

    return bath_energy, bath_hyb, delta_fit


def hybridization(w_mesh, bath_energy, bath_hyb):
    """
    Reconstruct hybridization from the bath Hamiltonian, assuming no SOC.
    :param w_mesh: numpy.ndarray(dim=1)
        Target frequencies array. This could be real or Matsubara frequencies.
    :param bath_energy: numpy.ndarray(dim=2)
        Energy of bath orbitals (nspin, nbath).
    :param bath_hyb: numpy.ndarray(dim=3)
        Hybridization between the bath and impurity orbitals (nspin, nbath, nImpOrbs).
    :return: Hybridization: numpy.ndarray(dim=4)
        Hybridization function (nw, nspin, nImpOrbs, nImpOrbs)
    """
    nw = w_mesh.shape[0]
    nspin, nbath, nImpOrbs = bath_hyb.shape
    delta_fit_wsab = np.zeros((nw, nspin, nImpOrbs, nImpOrbs), dtype=complex)
    # delta(n, s, i, j) = sum_{b} [V_sbi x (V_sbj)* ] / [iw - eps_sb]
    for n, w in enumerate(w_mesh):
        for s in range(nspin):
            for b, eps_b in enumerate(bath_energy[s]):
                for i in range(nImpOrbs):
                    for j in range(nImpOrbs):
                        # delta_fit_wsab should be symmetric in the absence of SOC
                        delta_fit_wsab[n,s,i,j] += bath_hyb[s,b,i].real * bath_hyb[s,b,j].real / (w - eps_b.real)
    return delta_fit_wsab


def estimate_zero_moment(Aw, iw_mesh, data: str = None):
    print("Estimating the 0th moment of {} via high-frequency expansion.".format(data))
    sys.stdout.flush()
    iw_m1 = iw_mesh[-1]
    iw_m2 = iw_mesh[-2]
    t = Aw[-1].real - (Aw[-1] - Aw[-2]).real * iw_m2 ** 2 / (
           iw_m2 ** 2 - iw_m1 ** 2)
    t = t.astype(complex)

    return t


def extract_t_and_delta(g_weiss_wsIab, iaft_imp):
    iwn_mesh_imp = iaft_imp.wn_mesh('f') * np.pi / iaft_imp.beta

    ns, nImp = g_weiss_wsIab.shape[1:3]
    weiss_tmp = np.zeros(g_weiss_wsIab.shape, dtype=complex)
    for n, g in enumerate(g_weiss_wsIab):
        for s in range(ns):
            for I in range(nImp):
                weiss_tmp[n, s, I] = 1j * iwn_mesh_imp[n] - np.linalg.inv(g[s, I])

    t_sIab_estimate = estimate_zero_moment(weiss_tmp, iwn_mesh_imp, "hybridization")
    print("Non-interacting Hamiltonian for the impurity:")
    print("Spin 0")
    print(t_sIab_estimate[0, 0])
    if t_sIab_estimate.shape[0] == 2:
        print("Spin 1")
        print("{}\n".format(t_sIab_estimate[1, 0]))
    else:
        print("")

    # construct delta: delta(iw) = iw - t - g^{-1}(iw)
    delta_estimate = np.zeros(g_weiss_wsIab.shape, dtype=complex)
    nbnd = t_sIab_estimate.shape[-1]
    for n in range(delta_estimate.shape[0]):
        for s in range(ns):
            for I in range(nImp):
                g_weiss_inv = np.linalg.inv(g_weiss_wsIab[n, 0, 0])
                delta_estimate[n, s, I] = 1j * iwn_mesh_imp[n] * np.eye(nbnd) - t_sIab_estimate[s, I] - g_weiss_inv
    iaft_imp.check_leakage(delta_estimate, 'f', 'delta_estimate', w_input=True)

    return t_sIab_estimate, delta_estimate

