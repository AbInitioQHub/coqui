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

import json

from coqui._lib.mf_module import Mf


def make_mf(mpi, params, mf_type):
    """
    Read a pre-computed mean-field (DFT) solution and construct a CoQuí Mf object.

    CoQuí does not run DFT itself. Instead, it reads the output of an external
    DFT code — selected by ``mf_type`` — and wraps it in a read-only ``Mf`` object
    that subsequent CoQuí steps (interaction construction, MBPT) operate on.

    Supported DFT codes (``mf_type`` values):

    - ``"qe"``    — Quantum ESPRESSO, via the pw2coqui post-processing step
    - ``"pyscf"`` — PySCF
    - ``"bdft"``  — CoQuí's internal mean-field format
    - ``"model"`` — model Hamiltonian (no external DFT code required)

    Parameters
    ----------
    mpi : MpiHandler
        MPI communicator handle obtained from ``coqui.utils.MpiHandler()``.
    mf_type : str
        Selects which DFT code's output to read. One of ``"qe"``, ``"pyscf"``,
        ``"bdft"``, or ``"model"``. 
    params : dict
        File location and read options for the chosen DFT code.
        Keys common to all ``mf_type`` values:

        - ``prefix`` *(str, required)* — file name prefix of the DFT output
        - ``outdir`` *(str, optional, default ``"./"``)*  — directory containing
          the DFT output files.

        Additional keys for each ``mf_type``:

        **"qe"** — reads Quantum ESPRESSO output converted by pw2coqui:

        - ``nbnd`` *(int, optional, default ``-1``)* — number of bands to read;
          ``-1`` reads all bands present in the QE output. 
        - ``ecut`` *(float, optional, default ``0.0``, units: Hartree)* — plane-wave
          kinetic-energy cutoff for the charge-density grid. ``0`` or negative keeps
          the value written by QE; a positive value requests a new FFT grid at that
          cutoff.
        - ``filetype`` *(str, optional, default ``"h5"``)* — input file format.
          ``"h5"`` for pw2coqui HDF5 output (recommended); ``"xml"`` for the legacy
          pw2bgw XML format.

        **"bdft"** — reads a CoQuí BDFT HDF5 file:

        - ``nbnd`` *(int, optional, default ``-1``)* — number of bands to read;
          ``-1`` reads all bands.
        - ``ecut`` *(float, optional, default ``0.0``, units: Hartree)* — same
          semantics as for ``"qe"``.

        **"model"** — reads a model Hamiltonian HDF5 file:

        - ``nbnd`` *(int, optional, default ``-1``)* — number of bands to read;
          ``-1`` reads all bands.

        **"pyscf"** — reads a PySCF HDF5 file; only ``prefix`` and ``outdir`` apply.

    Returns
    -------
    Mf
        A read-only mean-field object that can be passed to interaction and MBPT
        functions such as ``make_thc_coulomb`` and ``run_hf``.

    Examples
    --------
    Quantum ESPRESSO::

        from coqui.utils import MpiHandler
        from coqui.mean_field import make_mf

        mpi = MpiHandler()
        mf = make_mf(mpi,
                     {"prefix": "svo", "outdir": "qe_output/", "nbnd": 40},
                     "qe")

    PySCF::

        mf = make_mf(mpi, {"prefix": "h2o", "outdir": "pyscf_output/"}, "pyscf")
    """
    return Mf(mpi, json.dumps(params), mf_type)
