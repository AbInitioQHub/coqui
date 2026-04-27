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

from array import array
import os

import numpy as np
import pytest

pytest.importorskip("triqs", reason="triqs is required for dmft tests")

import coqui
from coqui.utils.tests.test_coqui_env import mpi
from coqui.dmft.bath_fit import causal_projection_boson
from coqui.dmft.weiss import extract_h0_and_delta
from coqui.mean_field.tests.test_qe import construct_qe_mf
from coqui.utils.imag_axes_ft import IAFT


@pytest.fixture(scope="module", params=["ir", "dlr"])
def downfold_inputs(mpi, request):
    iaft_basis = request.param
    if iaft_basis == "dlr":
        pytest.importorskip("coqui._lib.iaft_module")
    prefix = f"svo_{iaft_basis}"
    chkpt_h5 = f"./{prefix}.mbpt.h5"
    wan_h5 = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/mlwf/svo.mlwf.h5"

    mf = construct_qe_mf(mpi, "qe_svo222_sym")
    eri_params = {
        "storage": "incore",
        "thresh": 1e-4,
        "ecut": mf.ecutwfc() * 1.5,
        "init": True,
    }
    thc = coqui.make_thc_coulomb(mf, eri_params)

    gw_params = {
        "restart": False,
        "output": prefix,
        "niter": 1,
        "beta": 100,
        "iaft": {"prec": "high", "basis": iaft_basis},
        "iter_alg": {"alg": "damping", "mixing": 0.7},
    }
    coqui.run_gw(gw_params, h_int=thc)
    mpi.barrier()

    ir_kernel = IAFT.from_coqui_chkpt(chkpt_h5, verbose=False)

    gloc_params = {
        "outdir": "./",
        "prefix": prefix,
        "greens_func_source": "scf",
        "greens_func_iteration": 1,
        "wannier_file": wan_h5,
        "force_real": True,
    }
    gloc_t = coqui.downfold_local_gf(mf, gloc_params)

    wloc_params = {
        "outdir": "./",
        "prefix": prefix,
        "screen_type": "gw_edmft_density",
        "greens_func_source": "scf",
        "greens_func_iteration": 1,
        "wannier_file": wan_h5,
        "output_in_tau": True,
    }
    _, wloc_t = coqui.downfold_coulomb(thc, wloc_params)

    yield {
        "basis": iaft_basis,
        "iaft": ir_kernel,
        "gloc_t": gloc_t,
        "wloc_t": wloc_t,
    }

    if mpi.root() and os.path.isfile(chkpt_h5):
        os.remove(chkpt_h5)
    mpi.barrier()


def test_extract_h0_and_delta_with_real_downfold_gloc(downfold_inputs):
    assert downfold_inputs["iaft"].basis == downfold_inputs["basis"]

    ir_kernel = downfold_inputs["iaft"]
    gloc_t = downfold_inputs["gloc_t"]
    g_iw = ir_kernel.tau_to_w(gloc_t, stats="f")

    h0_sab, delta_iw = extract_h0_and_delta(g_iw, ir_kernel)

    H0_SAB_REF = np.array([[
        [ 1.98632457e-01, -3.39822832e-05,  3.27477287e-05], 
        [-3.39822832e-05,  1.98823108e-01, -9.76175950e-05],
        [ 3.27477287e-05, -9.76175950e-05,  1.98808918e-01]
    ]]) 
    assert h0_sab.shape == g_iw.shape[1:]
    assert h0_sab.shape == H0_SAB_REF.shape
    assert delta_iw.shape == g_iw.shape
    assert np.allclose(h0_sab.real, H0_SAB_REF, atol=1e-8)
    assert np.allclose(h0_sab.imag, 0.0, atol=1e-12)
    # Causality check
    wzero_idx = delta_iw.shape[0] // 2
    delta_imag_diag = np.diagonal(delta_iw[wzero_idx:,0].imag, axis1=1, axis2=2)
    assert np.all(delta_imag_diag < 1e-12)
    assert np.all(np.isfinite(delta_iw))


def test_causal_projection_boson_with_real_downfold_coulomb(downfold_inputs):
    pytest.importorskip("adapol")

    assert downfold_inputs["iaft"].basis == downfold_inputs["basis"]

    ir_kernel = downfold_inputs["iaft"]
    wloc_t = downfold_inputs["wloc_t"]
    wloc_iw = ir_kernel.tau_to_w_phsym(wloc_t, stats="b")
    nbnd = wloc_iw.shape[-1]
    wloc_dd_iw = wloc_iw.reshape(-1, nbnd * nbnd, nbnd * nbnd)

    wloc_fit_iw = causal_projection_boson(
        wloc_dd_iw,
        ir_kernel,
        {"nbath_per_orbital": 2, "exclude_w0": False},
        ph_symmetry=True,
        target_name="Wloc downfolded",
    )

    assert wloc_fit_iw.shape == wloc_dd_iw.shape
    assert np.allclose(wloc_fit_iw.imag, 0.0, atol=1e-12)

