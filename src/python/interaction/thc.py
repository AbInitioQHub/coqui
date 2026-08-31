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

from coqui._lib.eri_module import ThcCoulomb
from coqui._lib.eri_module import run_isdf as isdf_cxx


def make_thc_coulomb(mf, params):
    """
    Compute or read THC-decomposed Coulomb integrals and return a ``ThcCoulomb`` handler.

    The ``ThcCoulomb`` object returned contains a THC representation of the 
    two-electron Coulomb integrals that are either computed during the function call 
    or read from a pre-existing HDF5. (``save`` key in ``params``). 

    The resulting object can be passed to electronic structure methods such as ``run_hf`` 
    and ``downfold_coulomb``. 

    Parameters
    ----------
    mf : Mf
        Mean-field object for the target system, obtained from ``make_mf``.
    params : dict
        THC construction options. Supported keys:

        - ``nIpts`` *(int, optional, default ``0``)* — number of THC interpolation
          points. Primary stopping criterion when set to a positive value. If ``0``,
          ``thresh`` controls termination instead.
        - ``thresh`` *(float, optional, default ``1e-5``)* — convergence threshold 
          for the THC auxiliary basis construction. 
          Defaults to ``1e-5`` when ``nIpts=0`` or to ``1e-13`` when ``nIpts > 0``. 
          When both ``nIpts`` and ``thresh`` are given explicitly, the algorithm stops 
          as soon as either criterion is satisfied.
        - ``ecut`` *(float, optional, default ``1.4 * mf.ecutwfc()``)* - kinetic-energy
          cutoff for evaluating Coulomb matrix elements. For backends without a
          wavefunction grid (e.g. PySCF, model), the default falls back to
          ``0.4 * mf.ecutrho()``.
        - ``storage`` *(str, optional, default ``"incore"``)* — how integrals are
          stored after construction. ``"incore"`` keeps them in memory;
          ``"outcore"`` reads them from the HDF5 file on demand.
        - ``save`` *(str, optional, default ``""``)* — path to an HDF5 file for
          saving (or loading) the THC integrals. An empty string disables file I/O. 
          If the file exists, the THC integrals are automatically loaded. 
        - ``cd_dir`` *(str, optional, default ``""``)* — directory containing
          pre-computed Cholesky-decomposed Coulomb integrals. When provided, a
          least-squares THC fit is performed instead of ISDF.
        - ``chol_block_size`` *(int, optional, default ``8``)* — block size for
          the internal Cholesky step.
        - ``init`` *(bool, optional, default ``True``)* — if ``True``, runs the
          full THC computation immediately at construction. Set to ``False`` to
          defer until ``.init()`` is called explicitly.

    Returns
    -------
    ThcCoulomb
        A THC Coulomb interaction object that can be passed to MBPT functions
        such as ``run_hf`` and ``run_gw``.

    Examples
    --------
    ::

        from coqui.interaction import make_thc_coulomb

        thc = make_thc_coulomb(mf, {"save": "svo_isdf.h5", "storage": "incore"})
    """
    return ThcCoulomb(mf, json.dumps(params))


def run_isdf(mf, params):
    """
    Run the Interpolative Separable Density Fitting (ISDF) decomposition for 
    pair densities and save the result to an HDF5 file. 

    Parameters
    ----------
    mf : Mf
        Mean-field object for the target system, obtained from ``make_mf``.
    params : dict
        ISDF computation options. Supported keys:

        - ``thresh`` *(float, optional, default ``1e-10``)* — threshold for
          constructing the ISDF auxiliary basis. Either ``thresh > 0`` or
          ``nIpts > 0`` must be set.
        - ``nIpts`` *(int, optional, default ``0``)* — fixed number of
          interpolating points. When positive, overrides threshold-based selection;
          ``thresh`` then defaults to ``1e-13``.
        - ``ecut`` *(float, optional, default matches ``mf`` value)*
          — plane-wave kinetic-energy cutoff for evaluating Coulomb matrix elements.
        - ``save`` *(str, optional, default ``"isdf.h5"``)* — HDF5 file where the
          ISDF result (interpolating points and pair-density values) is written.
        - ``write_zeta_on_fft_mesh`` *(bool, optional, default ``False``)* — if
          ``True``, saves the ISDF basis functions on the full FFT mesh.
        - ``check_accuracy`` *(bool, optional, default ``False``)* — if ``True``,
          perform accuracy diagnostics for the decomposition.
        - ``chol_block_size`` *(int, optional, default ``8``)* — block size for
          the pivoted-Cholesky algorithm. Larger values are faster but use more
          memory; typical range: 1–12.
        - ``matrix_block_size`` *(int, optional, default ``1024``)* — block size
          for distributed array operations.
        - ``memory_frac`` *(float, optional, default ``0.75``)* — fraction of
          available node memory to budget for intermediate arrays.

    Examples
    --------
    ::

        from coqui.interaction import run_isdf

        run_isdf(mf, {"thresh": 1e-4, "save": "svo_isdf.h5"})
    """
    isdf_cxx(mf, json.dumps(params))

