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
from h5 import HDFArchive
from triqs.gfs import *

""" 
Analytical continuation utilities based on TRIQS application  
"""


def pade_triqs(g_iw, iaft, stats, wmin, wmax, Nw, *, Nfit=100, eta=None, ph_sym=False):
    # Convert g_iw to a TRIQS Green's function
    if stats in ['fermion', 'f']:
        statistic = "Fermion"
    elif stats in ['boson', 'b']:
        statistic = "Boson"
    else:
        raise ValueError("Invalid statistic. Use 'fermion' or 'boson'.")
    nw_max = iaft.wn_mesh(stats=stats)[-1]
    iw_mesh = MeshImFreq(beta=iaft.beta, statistic=statistic, n_iw=nw_max)

    g_iw = np.asarray(g_iw)
    target_shape = g_iw.shape[1:]
    if g_iw.ndim  == 1:
        g_iw = g_iw[:, None, None]
    elif g_iw.ndim == 2:
        g_iw = np.array([ np.diag(x) for x in g_iw])
    elif g_iw.ndim > 3:
        raise ValueError("Input g_iw can only have 1, 2, or 3 dimensions.")

    gf_imfreq = Gf(mesh=iw_mesh, target_shape=g_iw.shape[1:])
    mesh_iw_idx = np.array([iwn.index for iwn in iw_mesh])
    gf_imfreq.data[:] = iaft.w_interpolate(g_iw, mesh_iw_idx, stats=stats, phys_notation=True, ph_sym=ph_sym)

    # Create a real-frequency Gf using set_from_pade() function 
    eta = np.pi / iaft.beta if eta is None else float(eta)
    w_mesh = MeshReFreq(window=(wmin, wmax), n_w=Nw)
    gf_refreq = Gf(mesh=w_mesh, target_shape=g_iw.shape[1:])
    gf_refreq.set_from_pade(gf_imfreq, n_points=Nfit, freq_offset=eta)

    g_w = gf_refreq.data[:].reshape((Nw,) + target_shape)

    return g_w, w_mesh.values()


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


def maxent_triqs(G_iw, iaft, wmin, wmax, Nw, *, 
                 alpha_min=1e-6, alpha_max=1e2, num_alpha=30, error=1e-2, 
                 n_iw=None, omega_mesh="lorentzian", A_init_func=None, verbose=True):
    """
    Run the TRIQS maximum-entropy (MaxEnt) analytic continuation on a single
    scalar imaginary-frequency function and return the raw MaxEnt result.

    The input is interpolated from CoQuí's (non-uniform) IR/DLR Matsubara grid onto a
    uniform Matsubara mesh of ``n_iw`` positive frequencies, wrapped in a 1x1 TRIQS Gf,
    and continued with the ``PoormanMaxEnt`` solver. The returned object still carries
    the full alpha-scan; the caller selects an analyzer afterwards, e.g.
    ``maxent_results.get_A_out("LineFitAnalyzer")`` to obtain the spectral function
    A(w) on ``maxent_results.omega``.

    Despite the ``G_iw`` name this routine is generic: any fermionic scalar Matsubara
    function (self-energy, local Green's function, ...) can be continued. 

    :param G_iw: array
        Scalar function sampled on CoQuí's positive IR/DLR Matsubara frequencies.
    :param iaft: IAFT
        Imaginary-axis Fourier-transform kernel (provides ``beta`` and the
        ``w_interpolate`` used to remap onto the uniform Matsubara mesh).
    :param wmin: float
        Lower bound of the real-frequency (omega) output mesh.
    :param wmax: float
        Upper bound of the real-frequency (omega) output mesh.
    :param Nw: int
        Number of real-frequency points in the MaxEnt omega mesh.
    :param alpha_min: float
        Smallest entropy weight alpha in the (log-spaced) alpha scan.
    :param alpha_max: float
        Largest entropy weight alpha in the (log-spaced) alpha scan.
    :param num_alpha: int
        Number of alpha points in the scan.
    :param error: float
        Assumed (uniform) error/noise level on the input data; sets the data covariance.
    :param n_iw: int or None
        Number of positive Matsubara frequencies on the uniform mesh fed to MaxEnt.
        If None, it is inferred from the IAFT fermionic Matsubara mesh so that the
        uniform mesh spans the full range of ``iaft``'s frequencies.
    :param omega_mesh: str
        Real-frequency mesh type: "hyperbolic" (dense near 0) or "lorentzian".
    :param A_init_func: callable or None
        Optional default-model generator A(w); called on the omega array and normalized
        by the mesh spacing to build a ``DataDefaultModel``. If None, the flat default model
        is used. See ``sum_of_gaussians`` for a convenient builder.
    :param verbose: bool
        If False, silence the TRIQS MaxEnt logtaker output.
    :return: triqs_maxent.MaxEntResult
        The full MaxEnt result (alpha scan); call ``get_A_out(analyzer)`` for A(w).
    """
    try:
        from triqs_maxent import PoormanMaxEnt
        from triqs_maxent.default_models import DataDefaultModel
        from triqs_maxent.omega_meshes import HyperbolicOmegaMesh, LorentzianOmegaMesh
        from triqs_maxent.alpha_meshes import LogAlphaMesh
        from triqs_maxent.logtaker import VerbosityFlags
    except ImportError:
        raise ImportError("Fails to import triqs/maxent (https://github.com/TRIQS/maxent)! \n"
                          "Ensure that it is installed.")

    # set default n_iw 
    if n_iw is None:
        iw_idx_f = iaft.wn_mesh('fermion', phys_notation=True)
        n_iw = max(abs(iw_idx_f[0]), abs(iw_idx_f[-1])) + 1

    iw_mesh_uni = MeshImFreq(beta=iaft.beta, statistic='Fermion', n_iw=n_iw)
    G_iw_uni = Gf(mesh=iw_mesh_uni, target_shape=[1, 1])
    iw_idx = np.array([iw.index for iw in iw_mesh_uni])
    G_iw_uni[0, 0].data[:] = iaft.w_interpolate(G_iw, iw_idx, 'fermion', phys_notation=True)

    if omega_mesh == "hyperbolic":
        omega = HyperbolicOmegaMesh(omega_min=wmin, omega_max=wmax, n_points=Nw)
    elif omega_mesh == "lorentzian":
        omega = LorentzianOmegaMesh(omega_min=wmin, omega_max=wmax, n_points=Nw)
    else:
        assert False, "unsupported omega_mesh type!"

    maxent_solver = PoormanMaxEnt()
    maxent_solver.set_G_iw(G_iw_uni)
    maxent_solver.omega = omega
    maxent_solver.alpha_mesh = LogAlphaMesh(alpha_min=alpha_min, alpha_max=alpha_max, n_points=num_alpha)
    maxent_solver.set_error(error)
    if A_init_func is not None:
        w_array = np.asarray(list(maxent_solver.omega))
        d_model = DataDefaultModel(A_init_func(w_array)/omega.delta, omega)
        maxent_solver.maxent_diagonal.D = d_model

    if not verbose:
        maxent_solver.maxent_diagonal.logtaker.verbose = VerbosityFlags.Quiet
        maxent_solver.maxent_offdiagonal.logtaker.verbose = VerbosityFlags.Quiet

    maxent_results = maxent_solver.run()

    return maxent_results


def _maxent_sigma(Sigma_iw, ft, n_iw, analyzer,
                  w_min, w_max, nw_maxent, a_min, a_max, na, error,
                  nw_final, nw_interp = None, A_init_func= None, verbose=True):
    """
    MaxEnt analytic continuation of a single scalar self-energy onto the real axis,
    including the Kramers-Kronig reconstruction of the full (complex) self-energy.

    Unlike ``maxent_triqs`` (which only returns the spectral function), this routine wraps
    the input in a ``DirectSigmaContinuator``: MaxEnt continues the spectral function of
    an auxiliary Green's function, then a Kramers-Kronig transform produces the retarded
    self-energy Sigma(w) (real and imaginary parts) on a uniform real-frequency mesh. The
    omega mesh used inside MaxEnt is always "hyperbolic" here.

    :param Sigma_iw: array
        Scalar self-energy sampled on CoQuí's positive IR/DLR Matsubara frequencies.
    :param ft: IAFT
        Imaginary-axis Fourier-transform kernel (provides ``beta`` and ``w_interpolate``).
    :param n_iw: int
        Number of positive Matsubara frequencies on the uniform mesh fed to MaxEnt.
    :param analyzer: str
        MaxEnt analyzer used to pick A(w) from the alpha scan before Kramers-Kronig:
        'LineFitAnalyzer', 'Chi2CurvatureAnalyzer', 'ClassicAnalyzer',
        'EntropyAnalyzer', or 'BryanAnalyzer'.
    :param w_min: float
        Lower bound of the real-frequency output mesh.
    :param w_max: float
        Upper bound of the real-frequency output mesh.
    :param nw_maxent: int
        Number of real-frequency points in the MaxEnt omega mesh.
    :param a_min: float
        Smallest entropy weight alpha in the (log-spaced) alpha scan.
    :param a_max: float
        Largest entropy weight alpha in the (log-spaced) alpha scan.
    :param na: int
        Number of alpha points in the scan.
    :param error: float
        Assumed (uniform) error/noise level on the input data.
    :param nw_final: int
        Number of real-frequency points in the final Kramers-Kronig output Sigma(w).
    :param nw_interp: int or None
        Number of points used to interpolate A(w) before the Kramers-Kronig transform.
        If None, no intermediate interpolation is applied.
    :param A_init_func: callable or None
        Optional default-model generator A(w) (see ``maxent_triqs``). If None, the flat
        default model is used.
    :param verbose: bool
        If False, silence the TRIQS MaxEnt logtaker output.
    :return: (triqs_maxent.MaxEntResult, triqs.gf.Gf)
        The MaxEnt result (alpha scan) and the Kramers-Kronig-reconstructed retarded
        self-energy Sigma(w) as a 1x1 real-frequency Gf (``continuators.Gaux_w``).
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
    Sigma_iw_uni[0, 0].data[:] = ft.w_interpolate(Sigma_iw, iw_idx, 'f', phys_notation=True)

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
    """
    Driver: MaxEnt-continue the diagonal of the impurity self-energy from a CoQuí
    checkpoint and write the real-frequency result back into the same HDF5 archive.

    Reads ``downfold_1e/iter{iteration}/Sigma_imp_wsIab`` (a single impurity, ``nImp == 1``),
    loops over spin and orbital, continues each diagonal element ``Sigma_imp[:, s, 0, i, i]``
    via ``_maxent_sigma`` (MaxEnt + Kramers-Kronig), and stores Sigma(w) under
    ``embed/iter{iteration}/ac/Sigma_imp_wsa`` together with the omega mesh and the raw
    per-(spin, orbital) MaxEnt scans.

    :param aimbes: CoQuí AIMBES handle
        Provides ``aimbes_h5`` (checkpoint path) and ``iaft`` (the IAFT kernel).
    :param iteration: int
        Embedding iteration to read; -1 selects ``embed/final_iter``.
    :param analyzer: str
        MaxEnt analyzer used to pick A(w); see ``_maxent_sigma``.
    :param w_min: float
        Lower bound of the real-frequency output mesh.
    :param w_max: float
        Upper bound of the real-frequency output mesh.
    :param nw_out: int
        Number of real-frequency points in the final Sigma(w) output.
    :param nw_maxent: int
        Number of real-frequency points in the MaxEnt omega mesh.
    :param nw_interp: int
        Number of points used to interpolate A(w) before Kramers-Kronig.
    :param n_iw_maxent: int
        Number of positive Matsubara frequencies on the uniform mesh fed to MaxEnt.
    :param a_min: float
        Smallest entropy weight alpha in the (log-spaced) alpha scan.
    :param a_max: float
        Largest entropy weight alpha in the (log-spaced) alpha scan.
    :param na: int
        Number of alpha points in the scan.
    :param error: float
        Assumed (uniform) error/noise level on the input data.
    :param A_init_func: callable or None
        Optional default-model generator A(w); see ``maxent_triqs``.
    :return: None
        Results are written in-place to ``embed/iter{iteration}/ac/Sigma_imp_wsa`` in
        ``aimbes.aimbes_h5``.
    """
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
    """
    Driver: MPI-parallel MaxEnt continuation of the k-resolved (Wannier-interpolated)
    self-energy or Green's function from a CoQuí checkpoint, written back to HDF5.

    With ``G_input=False`` it reads the imaginary-time self-energy
    ``embed/iter{iteration}/wannier_inter/Sigma_tskab`` and transforms it to Matsubara
    frequency; with ``G_input=True`` it reads ``.../G_wskab`` directly. The diagonal band
    elements ``[:, s, k, a, a]`` are distributed over MPI ranks (via ``mpi.slice_array``),
    each continued with ``_maxent_sigma`` (MaxEnt + Kramers-Kronig), then gathered. The
    real-frequency result Sigma(w)/G(w) is stored under ``embed/iter{iteration}/ac/Sigma_wska``
    (or ``.../G_wska``) together with the omega mesh.

    :param aimbes: CoQuí AIMBES handle
        Provides ``aimbes_h5`` (checkpoint path) and ``iaft`` (the IAFT kernel).
    :param iteration: int
        Embedding iteration to read; -1 selects ``embed/final_iter``.
    :param analyzer: str
        MaxEnt analyzer used to pick A(w); see ``_maxent_sigma``.
    :param w_min: float
        Lower bound of the real-frequency output mesh.
    :param w_max: float
        Upper bound of the real-frequency output mesh.
    :param nw_out: int
        Number of real-frequency points in the final output.
    :param nw_maxent: int
        Number of real-frequency points in the MaxEnt omega mesh.
    :param nw_interp: int
        Number of points used to interpolate A(w) before Kramers-Kronig.
    :param n_iw_maxent: int
        Number of positive Matsubara frequencies on the uniform mesh fed to MaxEnt.
    :param a_min: float
        Smallest entropy weight alpha in the (log-spaced) alpha scan.
    :param a_max: float
        Largest entropy weight alpha in the (log-spaced) alpha scan.
    :param na: int
        Number of alpha points in the scan.
    :param error: float
        Assumed (uniform) error/noise level on the input data.
    :param G_input: bool
        If True, read a k-resolved Green's function (``G_wskab``) instead of the
        self-energy and write under ``.../ac/G_wska``; otherwise continue the self-energy.
    :return: None
        Results are written in-place (on the master rank) to
        ``embed/iter{iteration}/ac/{Sigma_wska,G_wska}`` in ``aimbes.aimbes_h5``.
    """
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
