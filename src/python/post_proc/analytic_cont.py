"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team

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
from h5 import HDFArchive
from triqs.gf import *

""" 
Analytical continuation utilities based on TRIQS application  
"""


def sum_of_gaussians(x_array, centers={0.0}, exponents={0.1}):
    """
    Returns a 1D array representing the sum of Gaussian functions over a range.

    Parameters:
        x_array (array of float): List of x points
        centers (list of float): List of Gaussian centers
        exponents (list of float): List of exponents (a in exp(-a*(x-c)^2))

    Returns:
        y (np.ndarray): Sum of Gaussians evaluated on x
    """
    y = np.zeros_like(x_array)

    for c, a in zip(centers, exponents):
        y += np.exp(-a * (x_array - c)**2)

    return y


def _maxent_run(Sigma_iw, ft, n_iw,
                w_min, w_max, nw_maxent, a_min, a_max, na, error,
                omega_mesh="hyperbolic",
                A_init_func=None, verbose=True):
    """

    :param Sigma_iw:
        Self-energy on the Matsubara frequency
    :param analyzer: string
        MaxEnt analyzer: 'LineFitAnalyzer', 'Chi2CurvatureAnalyzer', 'ClassicAnalyzer',
        'EntropyAnalyzer', 'BryanAnalyzer'
    :param w_min:
    :param w_max:
    :param nw:
    :param a_min:
    :param a_max:
    :param na:
    :param error:
    :return:
    """
    try:
        from triqs_maxent import PoormanMaxEnt
        from triqs_maxent.sigma_continuator import DirectSigmaContinuator
        from triqs_maxent.default_models import DataDefaultModel
        from triqs_maxent.omega_meshes import HyperbolicOmegaMesh, LorentzianOmegaMesh
        from triqs_maxent.alpha_meshes import LogAlphaMesh
        from triqs_maxent.logtaker import VerbosityFlags
    except ImportError:
        raise ImportError("Fails to import triqs/maxent (https://github.com/TRIQS/maxent)! \n"
                          "Please ensure that it is installed.")

    iw_mesh_uni = MeshImFreq(beta=ft.beta, S='Fermion', n_iw=n_iw)
    Sigma_iw_uni = Gf(mesh=iw_mesh_uni, target_shape=[1, 1])
    iw_idx = np.array([iw.index for iw in iw_mesh_uni])
    Sigma_iw_uni[0, 0].data[:] = ft.w_interpolate(Sigma_iw, iw_idx, 'f', ir_notation=False)

    print("Setup maxent solver...")
    sys.stdout.flush()

    if omega_mesh == "hyperbolic":
        omega = HyperbolicOmegaMesh(omega_min=w_min, omega_max=w_max, n_points=nw_maxent)
    elif omega_mesh == "lorentzian":
        omega = LorentzianOmegaMesh(omega_min=w_min, omega_max=w_max, n_points=nw_maxent)
    else:
        assert False, "unsupported omega_mesh type!"

    maxent_solver = PoormanMaxEnt()
    maxent_solver.set_G_iw(Sigma_iw_uni)
    maxent_solver.omega = omega
    maxent_solver.alpha_mesh = LogAlphaMesh(alpha_min=a_min, alpha_max=a_max, n_points=na)
    maxent_solver.set_error(error)
    if A_init_func is not None:
        w_array = np.asarray(list(maxent_solver.omega))
        d_model = DataDefaultModel(A_init_func(w_array)/omega.delta, omega)
        maxent_solver.maxent_diagonal.D = d_model

    if not verbose:
        maxent_solver.maxent_diagonal.logtaker.verbose = VerbosityFlags.Quiet
        maxent_solver.maxent_offdiagonal.logtaker.verbose = VerbosityFlags.Quiet

    print("Maxent starts...")
    sys.stdout.flush()
    maxent_results = maxent_solver.run()

    return maxent_results


def _maxent_sigma(Sigma_iw, ft, n_iw, analyzer,
                  w_min, w_max, nw_maxent, a_min, a_max, na, error,
                  nw_final, nw_interp = None, A_init_func= None, verbose=True):
    """

    :param Sigma_iw:
        Self-energy on the Matsubara frequency
    :param analyzer: string
        MaxEnt analyzer: 'LineFitAnalyzer', 'Chi2CurvatureAnalyzer', 'ClassicAnalyzer',
        'EntropyAnalyzer', 'BryanAnalyzer'
    :param w_min:
    :param w_max:
    :param nw:
    :param a_min:
    :param a_max:
    :param na:
    :param error:
    :return:
    """
    try:
        from triqs_maxent import PoormanMaxEnt
        from triqs_maxent.sigma_continuator import DirectSigmaContinuator
        from triqs_maxent.default_models import DataDefaultModel
        from triqs_maxent.omega_meshes import HyperbolicOmegaMesh, LorentzianOmegaMesh
        from triqs_maxent.alpha_meshes import LogAlphaMesh
        from triqs_maxent.logtaker import VerbosityFlags
    except ImportError:
        raise ImportError("Fails to import triqs/maxent (https://github.com/TRIQS/maxent)! \n"
                          "Please ensure that it is installed.")

    iw_mesh_uni = MeshImFreq(beta=ft.beta, S='Fermion', n_iw=n_iw)
    Sigma_iw_uni = Gf(mesh=iw_mesh_uni, target_shape=[1, 1])
    iw_idx = np.array([iw.index for iw in iw_mesh_uni])
    Sigma_iw_uni[0, 0].data[:] = ft.w_interpolate(Sigma_iw, iw_idx, 'f', ir_notation=False)

    continuators = DirectSigmaContinuator(Sigma_iw_uni)

    print("Setup maxent solver...")
    sys.stdout.flush()

    omega = HyperbolicOmegaMesh(omega_min=w_min, omega_max=w_max, n_points=nw_maxent)

    maxent_solver = PoormanMaxEnt()
    maxent_solver.set_G_iw(Sigma_iw_uni)
    maxent_solver.omega = omega
    maxent_solver.alpha_mesh = LogAlphaMesh(alpha_min=a_min, alpha_max=a_max, n_points=na)
    maxent_solver.set_error(error)
    if A_init_func is not None:
        w_array = np.asarray(list(maxent_solver.omega))
        d_model = DataDefaultModel(A_init_func(w_array)/omega.delta, omega)
        maxent_solver.maxent_diagonal.D = d_model

    if not verbose:
        maxent_solver.maxent_diagonal.logtaker.verbose = VerbosityFlags.Quiet
        maxent_solver.maxent_offdiagonal.logtaker.verbose = VerbosityFlags.Quiet

    print("Maxent starts...")
    sys.stdout.flush()
    maxent_results = maxent_solver.run()

    # Kramers-Kronig
    print("Calculate Kramers-Kronig...")
    sys.stdout.flush()
    continuators.set_Gaux_w_from_Aaux_w(maxent_results.get_A_out(analyzer), maxent_results.omega,
                                        np_interp_A=nw_interp,
                                        np_omega=nw_final, w_min=w_min, w_max=w_max)

    return maxent_results, continuators.Gaux_w


def maxent_sigma(aimbes, iteration=-1, analyzer="LineFitAnalyzer",
                 w_min=-0.2, w_max=0.2, nw_out=2000,
                 nw_maxent=200, nw_interp=2000, n_iw_maxent=400,
                 a_min=1e-6, a_max=1e2, na=50, error=0.001,
                 A_init_func=None):
    with HDFArchive(aimbes.aimbes_h5, 'r') as ar:
        if iteration == -1:
            iteration = ar['embed/final_iter']
        Sigma_imp = ar[f"downfold_1e/iter{iteration}/Sigma_imp_wsIab"]

    niw, ns, nImp, nImpOrb, nImpOrb2 = Sigma_imp.shape
    if nImp != 1:
        raise NotImplementedError(f"nImp ({nImp}) has to be 1 at this moment!")

    print("Maxent for diagonals of the impurity self-energy")
    print("------------------------------------------------")
    print("aimbes h5  = {}".format(aimbes.aimbes_h5))
    print("iteration  = {}".format(iteration))
    print("output     = embed/iter{}/ac/Sigma_imp_wsa".format(iteration))
    print("wmin, wmax = {}, {}".format(w_min, w_max))
    print("nw_out     = {}".format(nw_out))
    print("nspin      = {}".format(ns))
    print("nbnd       = {}\n".format(nImpOrb))
    sys.stdout.flush()

    Simp_wsa = np.zeros((nw_out, ns, nImpOrb), dtype=complex)
    w_mesh_data = np.zeros(nw_out, dtype=float)
    maxents = []
    for s in range(ns):
        for i in range(nImpOrb):
            maxent_results, S_w_triqs = _maxent_sigma(Sigma_imp[:,s,0,i,i], aimbes.iaft, n_iw=n_iw_maxent,
                                                      analyzer=analyzer, w_min=w_min, w_max=w_max, nw_maxent=nw_maxent,
                                                      a_min=a_min, a_max=a_max, na=na, error=error,
                                                      nw_final=nw_out, nw_interp=nw_interp,
                                                      A_init_func=A_init_func)
            maxents.append(maxent_results)
            Simp_wsa[:, s, i] = S_w_triqs[0, 0].data[:]
            if s == 0 and i == 0:
                w_mesh_data = np.array([w.value for w in S_w_triqs.mesh])

    print("Maxent done. \n")
    print("Writing results to {}\n".format(aimbes.aimbes_h5))
    sys.stdout.flush()

    with HDFArchive(aimbes.aimbes_h5, 'a') as ar:
        if "ac" not in ar[f"embed/iter{iteration}"]:
            ar[f"embed/iter{iteration}"].create_group("ac")
        if "Sigma_imp_wsa" not in ar[f"embed/iter{iteration}/ac"]:
            ar[f"embed/iter{iteration}/ac"].create_group("Sigma_imp_wsa")
        S_grp = ar[f"embed/iter{iteration}/ac/Sigma_imp_wsa"]
        S_grp["output"] = Simp_wsa
        S_grp["w_mesh"] = w_mesh_data
        S_grp["alg"] = "maxent"
        for si in range(ns * nImpOrb):
            S_grp[f"maxent_results_sa{si}"] = maxents[si].data


def maxent_sigma_k(aimbes, iteration=-1, analyzer="LineFitAnalyzer",
                   w_min=-0.2, w_max=0.2, nw_out=2000,
                   nw_maxent=200, nw_interp=2000, n_iw_maxent=400,
                   a_min=1e-6, a_max=1e2, na=50, error=0.001,
                   G_input=False):
    try:
        import triqs.utility.mpi as mpi
    except ImportError:
        raise ImportError("Fails to import triqs.mpi! \n"
                          "The utility functions in the AC module requires triqs package. \n"
                          "(https://github.com/TRIQS/triqs). Please ensure that it is installed. ")

    with HDFArchive(aimbes.aimbes_h5, 'r') as ar:
        if iteration == -1:
            iteration = ar['embed/final_iter']
        if not G_input:
            Sigma_k = ar[f"embed/iter{iteration}/wannier_inter/Sigma_tskab"]
            Sigma_wskab = aimbes.iaft.tau_to_w(Sigma_k, stats='f')
        else:
            Sigma_wskab = ar[f"embed/iter{iteration}/wannier_inter/G_wskab"]

    niw, ns, nkpts, nbnd, nbnd2 = Sigma_wskab.shape

    if mpi.is_master_node():
        print("Maxent for diagonals of the impurity self-energy")
        print("------------------------------------------------")
        print("aimbes h5  = {}".format(aimbes.aimbes_h5))
        print("iteration  = {}".format(iteration))
        print("output     = embed/iter{}/ac/Sigma_imp_wsa".format(iteration))
        print("wmin, wmax = {}, {}".format(w_min, w_max))
        print("nw_out     = {}".format(nw_out))
        print("nkpts      = {}".format(nkpts))
        print("nspin      = {}".format(ns))
        print("nbnd       = {}\n".format(nbnd))
    mpi.barrier()

    Simp_wska = np.zeros((nw_out, ns, nkpts, nbnd), dtype=complex)
    w_mesh_data = np.zeros(nw_out, dtype=float)
    for ska in mpi.slice_array(np.arange(ns*nkpts*nbnd)):
        # ska = s*nkpts*nbnd + k*nbnd + a
        s = ska // (nkpts*nbnd)
        k = (ska // nbnd) % nkpts
        a = ska % nbnd
        mpi.report(f"ska = {ska}, s = {s}, k = {k}, a = {a}")
        maxent_results, S_w_triqs = _maxent_sigma(
                    Sigma_wskab[:, s, k, a, a], aimbes.iaft, n_iw=n_iw_maxent,
                    analyzer=analyzer, w_min=w_min, w_max=w_max,
                    nw_maxent=nw_maxent,
                    a_min=a_min, a_max=a_max, na=na, error=error,
                    nw_final=nw_out, nw_interp=nw_interp, verbose=True if mpi.is_master_node() else False)
        Simp_wska[:, s, k, a] = S_w_triqs[0, 0].data[:]
        if s+k+a == 0:
            w_mesh_data = np.array([w.value for w in S_w_triqs.mesh])

    mpi.all_reduce(Simp_wska)
    mpi.all_reduce(w_mesh_data)

    if mpi.is_master_node():
        print("Maxent done. \n")
        print("Writing results to {}\n".format(aimbes.aimbes_h5))

        grp_name = "Sigma_wska" if not G_input else "G_wska"

        with HDFArchive(aimbes.aimbes_h5, 'a') as ar:
            if "ac" not in ar[f"embed/iter{iteration}"]:
                ar[f"embed/iter{iteration}"].create_group("ac")
            if grp_name not in ar[f"embed/iter{iteration}/ac"]:
                ar[f"embed/iter{iteration}/ac"].create_group(grp_name)
            S_grp = ar[f"embed/iter{iteration}/ac/{grp_name}"]
            S_grp["output"] = Simp_wska
            S_grp["w_mesh"] = w_mesh_data
            S_grp["alg"] = "maxent"

    mpi.barrier()
