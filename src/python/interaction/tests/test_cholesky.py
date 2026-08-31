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

import coqui
from coqui.utils.tests.test_coqui_env import mpi
from coqui.mean_field.tests.test_qe import construct_qe_mf

def test_cholesky_eri(mpi):
    mf = construct_qe_mf(mpi, "qe_lih222")
    chol_params = {
        "tol": 1e-3,
        "storage": "outcore",
        "path": "./",
        "chol_block_size": 32
    }
    chol = coqui.make_chol_coulomb(mf, chol_params)
    assert isinstance(chol, coqui._lib.eri_module.CholCoulomb)
    assert mpi == chol.mpi()
    mpi.barrier()

    if mpi.root():
        os.remove("./chol_info.h5")
        for iq in range(mf.nqpts_ibz()):
            os.remove(f"./Vq{iq}.h5")
    mpi.barrier()

