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
Unit tests for the Python layer of the linearized quasiparticle approximation
(post_proc.linearized_qp): how linearized_qp_from_arrays and
linearized_qp_from_checkpoint hand their arguments, arrays and IAFT to the C++ kernel and
which result they return or write. 
"""

from mpi4py import MPI
import os
import numpy as np
import pytest
from h5 import HDFArchive

from coqui import IAFT
from coqui.utils.tests.test_coqui_env import mpi
from coqui.post_proc import (
    linearized_qp_from_arrays,
    linearized_qp_from_checkpoint,
    LinearizedQPResult,
    LinearizedQPLadder,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _scalar_pole_sigma_tau(iaft, norb, e, g):
    """Sigma(tau) of Sigma(iw) = g I / (iw - e), on the fermionic tau mesh of iaft."""
    iw = 1j * np.pi / iaft.beta * iaft.wn_mesh("fermion")
    Sigma_w = (g / (iw - e))[:, None, None] * np.eye(norb, dtype=np.complex128)
    return iaft.w_to_tau(Sigma_w, "fermion")


def _random_hermitian(norb, seed):
    rng = np.random.RandomState(seed)
    matrix = rng.normal(size=(norb, norb)) + 1j * rng.normal(size=(norb, norb))
    return matrix + matrix.conj().T


# ---------------------------------------------------------------------------
# linearized_qp_from_arrays
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("basis", ["dlr", "ir"])
def test_linearized_qp_from_arrays(mpi, basis):
    """linearized_qp_from_arrays hands its arrays, mu, IAFT and fit controls to the kernel
    and returns the single fit or the ladder it asked for.

    Setup: a scalar-pole Sigma (e = 7, g = 3) on a 4x4 real-symmetric F, mu = 0.3, sampled with
    the Python IAFT of the parametrized basis. 

    Cases:
      1. Single fit at n_fit = 6.
         - Returns a LinearizedQPResult with n_fit = 6 and fit_order = 11 (from -1).
         - Zqp and Z equal z = (1 + g/e^2)^-1, and E the absolute z (eig F - g/e - mu) + mu:
           a wrong mu, IAFT or array handover would break them.
      2. converge=True with n_fit_max = 4.
         - Returns a LinearizedQPLadder whose last fit is a LinearizedQPResult and whose cap is
           n_fit_max (below the mesh ceiling), so n_fit_max reaches the kernel.
      3. Inputs the bindings cannot wrap as they are: they accept only C-ordered complex128
         buffers (wrapped without copying) and raise TypeError for anything else, so the
         Python layer must convert.
         - Fortran-ordered F and Sigma give the result of case 1.
         - A real-typed F gives the result of case 1, where the same F is passed as complex.
      4. "ir" only: an IAFT with a custom eps.
         - Raises NotImplementedError: the C++ IR backend has no basis for it.
    """
    if basis == "ir":
        pytest.importorskip("sparse_ir")
    iaft = IAFT(beta=100.0, wmax=40.0, prec="high", basis=basis, verbose=False)
    norb, e, g, mu = 4, 7.0, 3.0, 0.3
    Sigma_tsk = _scalar_pole_sigma_tau(iaft, norb, e, g)[:, None, None]
    F = _random_hermitian(norb, 17).real[None, None]
    z = 1.0 / (1.0 + g / e**2)

    # 1.
    res = linearized_qp_from_arrays(mpi, F.astype(np.complex128), Sigma_tsk, mu, iaft,
                                    converge=False, n_fit=6, fit_order=-1)
    assert isinstance(res, LinearizedQPResult)
    assert res.n_fit == 6 and res.fit_order == 11
    np.testing.assert_allclose(res.Zqp_ska[0, 0], np.full(norb, z), atol=1e-10)
    np.testing.assert_allclose(res.Z_skab[0, 0], z * np.eye(norb), atol=1e-10)
    E_exact = (np.linalg.eigvalsh(F[0, 0]) - g / e - mu) * z + mu
    np.testing.assert_allclose(np.sort(res.E_ska[0, 0]), np.sort(E_exact), atol=1e-9)

    # 2.
    lad = linearized_qp_from_arrays(mpi, F.astype(np.complex128), Sigma_tsk, mu, iaft,
                                    converge=True, n_fit_max=4)
    assert isinstance(lad, LinearizedQPLadder) and isinstance(lad.last, LinearizedQPResult)
    assert lad.n_fit_cap == 4 and lad.last.n_fit == 4

    # 3.
    res_fortran = linearized_qp_from_arrays(mpi, np.asfortranarray(F.astype(np.complex128)),
                                            np.asfortranarray(Sigma_tsk), mu, iaft,
                                            converge=False, n_fit=6, fit_order=-1)
    np.testing.assert_allclose(res_fortran.E_ska, res.E_ska, atol=1e-12)
    np.testing.assert_allclose(res_fortran.Z_skab, res.Z_skab, atol=1e-12)
    res_real = linearized_qp_from_arrays(mpi, F, Sigma_tsk, mu, iaft,
                                         converge=False, n_fit=6, fit_order=-1)
    np.testing.assert_allclose(res_real.E_ska, res.E_ska, atol=1e-12)

    # 4.
    if basis == "ir":
        custom = IAFT(beta=100.0, wmax=40.0, prec="custom", eps=1e-12, basis="ir", verbose=False)
        Sigma_tsk_custom = _scalar_pole_sigma_tau(custom, norb, e, g)[:, None, None]
        with pytest.raises(NotImplementedError):
            linearized_qp_from_arrays(mpi, F, Sigma_tsk_custom, mu, custom, converge=False,
                                      n_fit=6)


# ---------------------------------------------------------------------------
# linearized_qp_from_checkpoint
# ---------------------------------------------------------------------------

def _write_synthetic_checkpoint(path, iaft):
    """Diagonal scalar-pole self-energies (e, g) = (10, 9) and (12, 25) in a KS-basis
    checkpoint, and the same arrays under wannier_inter/ with H0 folded into F. Returns the
    analytic Z per orbital."""
    params = [(10.0, 9.0), (12.0, 25.0)]
    ns, nk, norb = 1, 1, len(params)
    iw = 1j * np.pi / iaft.beta * iaft.wn_mesh("fermion")
    Sigma_w = np.zeros((len(iw), ns, nk, norb, norb), dtype=np.complex128)
    for a, (e, g) in enumerate(params):
        Sigma_w[:, 0, 0, a, a] = g / (iw - e)
    Sigma_t = iaft.w_to_tau(Sigma_w, "fermion")
    H0 = np.zeros((ns, nk, norb, norb), dtype=np.complex128); H0[0, 0] = np.diag([0.3, 0.7])
    F = np.zeros_like(H0); F[0, 0] = np.diag([-0.1, 0.05])
    # plain groups, as CoQui writes them: a group stored from a Python dict is tagged as one and
    # no longer reads back once the routine under test adds subgroups to it
    with HDFArchive(path, "w") as ar:
        iaft.save(ar)
        ar.create_group("system"); ar["system"]["H0_skij"] = H0
        ar.create_group("scf"); scf = ar["scf"]
        scf["final_iter"] = 0
        scf.create_group("iter0"); it0 = scf["iter0"]
        it0["Sigma_tskij"] = Sigma_t; it0["F_skij"] = F; it0["mu"] = 0.2
        it0.create_group("wannier_inter"); winter = it0["wannier_inter"]
        winter["F_skab"] = H0 + F; winter["Sigma_tskab"] = Sigma_t
    return np.array([1.0 / (1.0 + g / e**2) for e, g in params])


def _read_datasets(path, group, keys):
    """The datasets `keys` of `group` (a '/'-separated path) as values, None when absent. Every
    h5 handle is released on return: an open one would keep the file open and stop the next
    writer."""
    with HDFArchive(path, "r") as ar:
        grp = ar
        for part in group.split("/"):
            grp = grp[part]
        return {key: (grp[key] if key in grp else None) for key in keys}


def test_linearized_qp_from_checkpoint(mpi, tmp_path):
    """linearized_qp_from_checkpoint hands its arguments to the C++ writer unchanged and
    returns None.

    Setup: one synthetic checkpoint with diagonal scalar poles. 

    Cases (2-3 continue on the file left by the case before, so they run in order):
      1. source = "mesh", single fit with n_fit = 5, fit_order = 7, fit_resid_tol = 1e-7.
         - Returns None.
         - qp_approx/lqp/ stores exactly these controls.
         - qp_approx/Zqp_ska is the analytic Z.
      2. source = "mesh", converge=True with n_fit_max = 4.
         - qp_approx/lqp/ladder/ stores n_fit_cap = 4.
      3. source = "interpolated".
         - The result is in wannier_inter/qp_approx/, with the analytic Z.
    """
    comm = MPI.COMM_WORLD
    chkpt = os.path.join(comm.bcast(str(tmp_path), root=0), "synthetic.mbpt.h5")
    iaft = IAFT(beta=100.0, wmax=40.0, prec="high", basis="dlr", verbose=False)
    Z_exact = _write_synthetic_checkpoint(chkpt, iaft) if comm.rank == 0 else None
    Z_exact = comm.bcast(Z_exact, root=0)
    comm.Barrier()

    # 1.
    linearized_qp_from_checkpoint(mpi, chkpt, grp_name="scf", iteration=-1, source="mesh",
                                  converge=False, n_fit=5, fit_order=7,
                                  symmetric_window=True, fit_resid_tol=1e-7)
    controls = _read_datasets(chkpt, "scf/iter0/qp_approx/lqp",
                              ("n_fit", "fit_order", "symmetric_window", "fit_resid_tol"))
    assert controls["n_fit"] == 5 and controls["fit_order"] == 7
    assert controls["symmetric_window"] and controls["fit_resid_tol"] == 1e-7
    Zqp = _read_datasets(chkpt, "scf/iter0/qp_approx", ("Zqp_ska",))["Zqp_ska"]
    np.testing.assert_allclose(np.sort(Zqp[0, 0]), np.sort(Z_exact), rtol=1e-5)
    comm.Barrier()

    # 2.
    linearized_qp_from_checkpoint(mpi, chkpt, grp_name="scf", iteration=-1, source="mesh",
                                  converge=True, n_fit_max=4)
    lad = _read_datasets(chkpt, "scf/iter0/qp_approx/lqp/ladder", ("n_fit_cap",))
    assert lad["n_fit_cap"] == 4
    comm.Barrier()

    # 3.
    linearized_qp_from_checkpoint(mpi, chkpt, grp_name="scf", iteration=-1, source="interpolated",
                                  converge=False)
    Zqp = _read_datasets(chkpt, "scf/iter0/wannier_inter/qp_approx", ("Zqp_ska",))["Zqp_ska"]
    np.testing.assert_allclose(np.sort(Zqp[0, 0]), np.sort(Z_exact), rtol=1e-5)
    comm.Barrier()
