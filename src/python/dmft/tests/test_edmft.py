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
    # The ISDF interpolating points depend on the number of MPI ranks, so every quantity below
    # differs between rank counts within the ISDF accuracy (h0 by ~5e-5 at thresh = 1e-5 with the
    # default block size). 
    # chol_block_size = 1 makes the point selection rank-independent up to ties and rounding:
    eri_params = {
        "storage": "incore",
        "thresh": 1e-5,
        "ecut": mf.ecutwfc() * 1.5,
        "chol_block_size": 1,
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

    # generated with thresh = 1e-5, chol_block_size = 1; atol is the ISDF accuracy, see the fixture
    H0_SAB_REF = np.array([[
        [ 1.985756442e-01,  9.557958753e-06, -1.224950026e-05],
        [ 9.557958753e-06,  1.985735773e-01,  4.318933304e-06],
        [-1.224950026e-05,  4.318933304e-06,  1.985829596e-01]
    ]])
    assert h0_sab.shape == g_iw.shape[1:]
    assert h0_sab.shape == H0_SAB_REF.shape
    assert delta_iw.shape == g_iw.shape
    assert np.allclose(h0_sab.real, H0_SAB_REF, atol=1e-4)
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
        2,
        ph_symmetry=True,
        n_exclude_low_freq=0,
        target_name="Wloc downfolded",
    )

    assert wloc_fit_iw.shape == wloc_dd_iw.shape
    assert np.allclose(wloc_fit_iw.imag, 0.0, atol=1e-12)



def test_run_gw_edmft_one_iteration(downfold_inputs, mpi, tmp_path):
    """run_gw_edmft runs one EDMFT iteration with CT-SEG through the nested solver layout and
    through the legacy flat layout, and writes the new checkpoint layout.

    Setup: the SrVO3 2x2x2 fixture (GW checkpoint and MLWF file), a tiny Monte Carlo budget
    (n_cycles 4000 total, warm-up 200, cycle length 20), screen_type "gw_edmft". The DLR
    fixture only (the IR one differs just in the IAFT and would double the run time).

    Cases (each runs on its own copy of the fixture's GW checkpoint in tmp_path, so 2 reuses
    nothing from 1 and the fixture file is left untouched):
      1. Nested layout: the run completes; dmft/iter0/impurity_0/results holds Sigma_iw_data,
         Pi_iw_data, mu_imp and density; impurity_results holds G_iw, Sigma_iw, Pi_iw, W_iw,
         Chi_iw; inputs holds degenerate_blk, occupations_sab and screen_type.
      2. Legacy flat layout (the same keys without the solver table): the run completes and
         writes the same result keys (the shim nested them).
    """
    if downfold_inputs["basis"] != "dlr":
        pytest.skip("one basis is enough for the end-to-end check")
    pytest.importorskip("triqs_ctseg")
    import shutil
    import triqs_modest as modest
    from triqs.utility import mpi as triqs_mpi
    from h5 import HDFArchive
    import coqui.dmft as coqui_dmft
    wan_h5 = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/mlwf/svo.mlwf.h5"
    obe = modest.make_one_body_elements_gw(wan_h5)
    embedding = modest.make_embedding(obe.C_space)
    proj_info = coqui_dmft.get_proj_info(obe.P)
    mf = construct_qe_mf(mpi, "qe_svo222_sym")
    thc = coqui.make_thc_coulomb(mf, {"storage": "incore", "thresh": 1e-5,
                                      "ecut": mf.ecutwfc() * 1.5, "chol_block_size": 1, "init": True})
    out = triqs_mpi.bcast(str(tmp_path))          # tmp_path is per rank under MPI
    solver = {"type": "ctseg", "n_cycles": 4000, "n_warmup_cycles": 200, "length_cycle": 20,
              "random_seed": 5, "perform_tail_fit": False}
    base = {"niter": 1, "gw_iter_per_loop": 1, "edmft_iter_per_loop": 1,
            "prefix": "svo_dlr", "screen_type": "gw_edmft", "wannier_file": wan_h5,
            "iter_alg": {"alg": "damping", "mixing": 0.3}}
    for label, impurity in (("nested", {"init_imp_results": "dc", "degenerate_blk": [[0, 1, 2, 3, 4, 5]],
                                        "solver": solver}),
                            ("legacy", {"init_imp_results": "dc", "degenerate_blk": [[0, 1, 2, 3, 4, 5]],
                                        **{k: v for k, v in solver.items() if k != "type"}})):
        outdir = f"{out}/{label}"
        chkpt = f"{outdir}/svo.{label}.dmft.h5"
        if mpi.root():
            os.makedirs(outdir, exist_ok=True)
            shutil.copy("./svo_dlr.mbpt.h5", f"{outdir}/svo_dlr.mbpt.h5")
        mpi.barrier()
        params = dict(base, outdir=outdir, edmft={"chkpt_h5": chkpt, "impurity": impurity})
        coqui_dmft.run_gw_edmft(thc, embedding, proj_info=proj_info, params=params)
        mpi.barrier()
        if mpi.root():
            with HDFArchive(chkpt, 'r') as ar:
                imp = ar["dmft/iter0/impurity_0"]
                for key in ("Sigma_iw_data", "Pi_iw_data", "mu_imp", "density", "average_sign"):
                    assert key in imp["results"].keys(), (label, key)
                assert set(imp["solver_results"].keys()) == {"G_iw", "Sigma_iw", "Pi_iw", "W_iw", "Chi_iw"}
                for key in ("degenerate_blk", "occupations_sab", "screen_type"):
                    assert key in imp["inputs"].keys(), (label, key)
        mpi.barrier()
