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
import pytest

import coqui
from coqui.utils.tests.test_coqui_env import mpi
from coqui._lib.utils_module import utest_filename


def construct_qe_mf(mpi_context, system_id: str):
    outdir, prefix = utest_filename(system_id)
    mf_params = {
        "prefix": prefix,
        "outdir": outdir,
        "filetype": "h5"
    }
    qe_mf = coqui.make_mf(mpi_context, mf_params, "qe")
    assert isinstance(qe_mf, coqui._lib.mf_module.Mf)
    if mpi_context.root():
        print(qe_mf)

    return qe_mf


def test_qe_mf(mpi):
    mf1 = construct_qe_mf(mpi, "qe_lih223")
    mf2 = construct_qe_mf(mpi, "qe_lih223_sym")
    mf3 = construct_qe_mf(mpi, "qe_lih223_inv")

    assert mpi == mf1.mpi()
    assert mpi == mf2.mpi()
    assert mpi == mf3.mpi()