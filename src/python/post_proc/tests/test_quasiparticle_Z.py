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

Unit tests for the LQSGW quasiparticle renormalization factor Z.

The tests use a single-pole correlation self-energy

    Σ(iω) = g² / (iω − ε),

which has a proper spectral representation (a δ peak of weight g² at ε, so it
is reproduced by the IAFT compact basis) and a known low-frequency behaviour

    Im Σ(iω) = −g² ω / (ω² + ε²)  ⟹  ∂Im Σ/∂ω|_{ω→0} = −g²/ε²,

giving the analytic renormalization factor

    Z = 1 / (1 + g²/ε²).
"""

from mpi4py import MPI
import pytest
import os
import tempfile
import numpy as np
from h5 import HDFArchive

from coqui import IAFT
from coqui.post_proc import (
    matsubara_renormalization_factor,
    quasiparticle_renormalization_factor,
)


def _single_pole_sigma_w(iaft, eps, g):
    """Σ(iω_n) = g²/(iω_n − ε) on the IAFT fermionic Matsubara sampling mesh."""
    iw = 1j * iaft.wn_mesh("fermion") * np.pi / iaft.beta
    return g**2 / (iw - eps)


def _Z_exact(eps, g):
    return 1.0 / (1.0 + g**2 / eps**2)


@pytest.mark.parametrize("eps, g", [(10.0, 3.0), (-8.0, 2.0), (12.0, 5.0)])
def test_single_pole_high_order_is_exact(eps, g):
    """With enough fit orders the single-pole Z is recovered to basis precision."""
    iaft = IAFT(beta=100.0, wmax=40.0, prec="high", basis="dlr", verbose=False)
    Sigma_w = _single_pole_sigma_w(iaft, eps, g)
    Z = matsubara_renormalization_factor(Sigma_w, iaft, n_fit=6, fit_order=5)
    assert np.isrealobj(Z)
    assert Z.shape == ()
    assert float(Z) == pytest.approx(_Z_exact(eps, g), rel=1e-6)


def test_fit_order_convergence():
    """Higher odd fit order reduces the truncation error of the slope."""
    iaft = IAFT(beta=50.0, wmax=30.0, prec="high", basis="dlr", verbose=False)
    eps, g = 8.0, 2.0
    Sigma_w = _single_pole_sigma_w(iaft, eps, g)
    Z_exact = _Z_exact(eps, g)

    err1 = abs(matsubara_renormalization_factor(Sigma_w, iaft, n_fit=6, fit_order=1) - Z_exact)
    err3 = abs(matsubara_renormalization_factor(Sigma_w, iaft, n_fit=6, fit_order=3) - Z_exact)
    assert err3 < err1
    assert err3 < 1e-4


def test_zero_self_energy_gives_unit_Z():
    """A vanishing self-energy has zero slope, so Z = 1 exactly."""
    iaft = IAFT(beta=50.0, wmax=30.0, prec="high", basis="dlr", verbose=False)
    Sigma_w = np.zeros(len(iaft.wn_mesh("fermion")), dtype=np.complex128)
    Z = matsubara_renormalization_factor(Sigma_w, iaft)
    assert float(Z) == pytest.approx(1.0, abs=1e-12)


def test_trailing_axes_are_independent():
    """Stacked poles are fitted element-wise and match per-element scalar calls."""
    iaft = IAFT(beta=100.0, wmax=40.0, prec="high", basis="dlr", verbose=False)
    params = [(10.0, 3.0), (-8.0, 2.0), (12.0, 5.0)]
    Sigma_w = np.stack([_single_pole_sigma_w(iaft, eps, g) for eps, g in params], axis=-1)

    Z = matsubara_renormalization_factor(Sigma_w, iaft, n_fit=6, fit_order=5)
    assert Z.shape == (len(params),)
    Z_expected = np.array([_Z_exact(eps, g) for eps, g in params])
    np.testing.assert_allclose(Z, Z_expected, rtol=1e-6)

    # element-wise consistency with scalar invocations
    for i, (eps, g) in enumerate(params):
        Zi = matsubara_renormalization_factor(Sigma_w[:, i], iaft, n_fit=6, fit_order=5)
        assert float(Zi) == pytest.approx(Z[i], rel=1e-12)


@pytest.mark.parametrize("kwargs", [
    {"fit_order": 2},   # even
    {"fit_order": 0},   # non-positive
    {"n_fit": 1, "fit_order": 3},   # n_fit < number of fit params
])
def test_invalid_parameters_raise(kwargs):
    iaft = IAFT(beta=50.0, wmax=30.0, prec="high", basis="dlr", verbose=False)
    Sigma_w = _single_pole_sigma_w(iaft, 8.0, 2.0)
    with pytest.raises(ValueError):
        matsubara_renormalization_factor(Sigma_w, iaft, **kwargs)


def test_checkpoint_roundtrip():
    """End-to-end: write a synthetic Sigma_tskij checkpoint and read Z back.

    Builds a diagonal (per-orbital) single-pole self-energy on the Matsubara
    axis, transforms it to imaginary time, stores it in a CoQuí-style checkpoint
    together with the IAFT metadata, and checks that the orbital-resolved Z from
    ``quasiparticle_renormalization_factor`` matches the analytic values.
    """
    iaft = IAFT(beta=100.0, wmax=40.0, prec="high", basis="dlr", verbose=False)
    nw = len(iaft.wn_mesh("fermion"))
    nt = len(iaft.tau_mesh("fermion"))
    params = [(10.0, 3.0), (12.0, 5.0)]
    ns, nk, norb = 1, 1, len(params)

    # diagonal matrix-valued self-energy on the Matsubara axis
    Sigma_wskij = np.zeros((nw, ns, nk, norb, norb), dtype=np.complex128)
    for a, (eps, g) in enumerate(params):
        Sigma_wskij[:, 0, 0, a, a] = _single_pole_sigma_w(iaft, eps, g)
    Sigma_tskij = iaft.w_to_tau(Sigma_wskij, "fermion")
    assert Sigma_tskij.shape[0] == nt

    Z_expected = np.array([_Z_exact(eps, g) for eps, g in params])

    with tempfile.TemporaryDirectory() as tmpdir:
        chkpt = os.path.join(tmpdir, "synthetic.mbpt.h5")
        with HDFArchive(chkpt, "w") as ar:
            iaft.save(ar)
            ar["scf"] = {"final_iter": 0, "iter0": {"Sigma_tskij": Sigma_tskij}}

        Z = quasiparticle_renormalization_factor(chkpt, n_fit=6, fit_order=5)

    assert Z.shape == (ns, nk, norb)
    np.testing.assert_allclose(Z[0, 0, :], Z_expected, rtol=1e-5)
