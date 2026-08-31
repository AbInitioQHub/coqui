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

from mpi4py import MPI
import os
import pytest
import numpy as np

import coqui
from coqui.utils.tests.test_coqui_env import mpi
from coqui.mean_field.tests.test_qe import construct_qe_mf


def test_downfold(mpi):
    mf = construct_qe_mf(mpi, "qe_lih222_sym")
    eri_params = {
        "storage": "incore",
        "nIpts": mf.nbnd() * 10,
        "thresh": 1e-10,
        "ecut": mf.ecutrho(),
        "chol_block_size": 1,
        "init": True
    }
    thc = coqui.make_thc_coulomb(mf, eri_params)

    gw_params = {
        "restart": False, 
        "output": "gw", 
        "niter": 1,
        "div_treatment": "gygi",
        "beta": 100, 
        "iaft": {
            "prec": "medium",
            "basis": "dlr"
        },
        "iter_alg": {
            "alg": "damping", 
            "mixing": 0.7
        }
    }
    coqui.run_gw(gw_params, h_int=thc)
    mpi.barrier()

    proj_info = coqui.read_proj_info(mf.outdir()+"/lih_wan.h5")

    # downfold the local Green's function
    gloc_params = {
        "outdir": "./", 
        "prefix": "gw",
        "greens_func_source": "scf", 
        "greens_func_iteration": 1,
        "wannier_file": mf.outdir() + "/lih_wan.h5",
        "force_real": True
    }
    Gloc_t = coqui.downfold_local_gf(mf, gloc_params)

    assert np.allclose(Gloc_t.imag, 0.0, atol=1e-10), "Imaginary part of Gloc(t) is not negligible"
    assert Gloc_t[-1,0,0,0] == pytest.approx(-0.990216810442369, abs=1e-10)
    assert Gloc_t[-1,0,1,0] == pytest.approx(-2.0127730737525417e-05, abs=1e-10)
    assert Gloc_t[-1,0,1,1] == pytest.approx(-0.9671840734138286, abs=1e-10)

    # downfold the local screened interaction
    wloc_params = {
        "outdir": "./", 
        "prefix": "gw",
        "screen_type": "gw_edmft_density",
        "greens_func_source": "scf", 
        "greens_func_iteration": 1,
        "wannier_file": mf.outdir() + "/lih_wan.h5",
        "div_treatment": "gygi_smallest_q",
        "permut_symm": True, 
        "force_real": True, 
        "output_in_tau": True
    }
    Vloc, Wloc_t = coqui.downfold_coulomb(thc, wloc_params)

    assert np.allclose(Vloc.imag, 0.0, atol=1e-12), "Imaginary part of Vloc is not negligible"
    assert Vloc[0,0,0,0] == pytest.approx(1.4160723518754155, abs=1e-10)
    assert Vloc[0,0,1,1] == pytest.approx(0.25499347676713535, abs=1e-10)
    assert Vloc[0,1,0,1] == pytest.approx(4.286546964169289e-05, abs=1e-10)
    assert Vloc[1,1,1,1] == pytest.approx(0.5557140951494038, abs=1e-10)

    assert np.allclose(Wloc_t.imag, 0.0, atol=1e-12), "Imaginary part of Wloc(t) is not negligible"
    assert Wloc_t[0,0,0,0,0] == pytest.approx(-0.2206215541852932, abs=1e-10)
    assert Wloc_t[0,0,0,1,1] == pytest.approx(-0.04908133825297427, abs=1e-10)
    assert Wloc_t[0,0,1,0,1] == pytest.approx(-6.915384193746768e-06, abs=1e-10)
    assert Wloc_t[0,1,1,1,1] == pytest.approx(-0.10210426895593258, abs=1e-10)

    # downfold the cRPA local screened interaction
    wloc_params["screen_type"] = "crpa"
    Vloc, Uloc_t = coqui.downfold_coulomb(thc, wloc_params, projector_info=proj_info)

    assert np.allclose(Uloc_t.imag, 0.0, atol=1e-12), "Imaginary part of Uloc(t) is not negligible"
    assert Uloc_t[0,0,0,0,0] == pytest.approx(-0.21483386189174042, abs=1e-10)
    assert Uloc_t[0,0,0,1,1] == pytest.approx(-0.04774705924289125, abs=1e-10)
    assert Uloc_t[0,0,1,0,1] == pytest.approx(-6.819360600413984e-06, abs=1e-10)
    assert Uloc_t[0,1,1,1,1] == pytest.approx(-0.09674740361073561, abs=1e-10)

    if mpi.root():
        os.remove("./gw.mbpt.h5")
    mpi.barrier()

def test_local_coulomb_from_mf(mpi):
    mf = construct_qe_mf(mpi, "qe_lih222_sym")
    eri_params = {
        "storage": "incore",
        "nIpts": mf.nbnd() * 10,
        "thresh": 1e-10,
        "ecut": mf.ecutrho(),
        "chol_block_size": 1,
        "init": True
    }
    thc = coqui.make_thc_coulomb(mf, eri_params)

    # downfold the local screened interaction
    wloc_params = {
        "outdir": "./", 
        "prefix": "crpa",
        "beta": 100, 
        "iaft": {
            "prec": "medium",
            "basis": "dlr"
        },
        "screen_type": "crpa",
        "greens_func_source": "mf",
        "wannier_file": mf.outdir() + "/lih_wan.h5",
        "div_treatment": "gygi",
        "permut_symm": True, 
        "force_real": True, 
        "output_in_tau": True
    }
    Vloc, Wloc_t = coqui.downfold_coulomb(thc, wloc_params)

    assert np.allclose(Vloc.imag, 0.0, atol=1e-12), "Imaginary part of Vloc is not negligible"
    assert Vloc[0,0,0,0] == pytest.approx(1.4160723518754155, abs=1e-10)
    assert Vloc[0,0,1,1] == pytest.approx(0.25499347676713535, abs=1e-10)
    assert Vloc[0,1,0,1] == pytest.approx(4.286546964169289e-05, abs=1e-10)
    assert Vloc[1,1,1,1] == pytest.approx(0.5557140951494038, abs=1e-10)

    assert np.allclose(Wloc_t.imag, 0.0, atol=1e-12), "Imaginary part of Wloc(t) is not negligible"
    assert Wloc_t[0,0,0,0,0] == pytest.approx(-0.2047780731982674, abs=1e-10)
    assert Wloc_t[0,0,0,1,1] == pytest.approx(-0.04195005851002771, abs=1e-10)
    assert Wloc_t[0,0,1,0,1] == pytest.approx(-6.82550064519015e-06, abs=1e-10)
    assert Wloc_t[0,1,1,1,1] == pytest.approx(-0.08664882474619286, abs=1e-10)

    if mpi.root():
        os.remove("./crpa.mbpt.h5")
    mpi.barrier()
