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
import os
from coqui._lib.eri_module import CholCoulomb


def make_chol_coulomb(mf, params):
    """
    Compute or load Cholesky-decomposed Coulomb integrals and return a ``CholCoulomb`` handler.

    The `CholCoulomb`` object returned contains a Cholesky representation of the 
    two-electron Coulomb integrals that are either computed during the function call 
    or read from pre-existing HDF5 files under the specified directory 
    (``path`` key in ``params``).

    The resulting object can be passed to MBPT functions such as ``run_hf`` and ``run_gw``.

    Parameters
    ----------
    mf : Mf
        Mean-field object for the target system, obtained from ``make_mf``.
    params : dict
        Cholesky construction options. Supported keys:

        - ``path`` *(str, optional, default ``"./"``)*  — directory where
          Cholesky integral files are written to or read from. Created
          automatically if it does not exist.
        - ``tol`` *(float, optional, default ``1e-4``)* — convergence tolerance
          for the Cholesky decomposition of the ERI tensor.
        - ``ecut`` *(float, optional, default ``mf.ecutrho()``)* - kinetic-energy 
          cutoff for evaluating Coulomb matrix elements.
        - ``chol_block_size`` *(int, optional, default ``32``)* — block size for
          the decomposition algorithm. Larger values are faster but more
          memory-intensive.
        - ``output`` *(str, optional, default ``"chol_info.h5"``)* — name of the
          HDF5 metadata file. When ``write_type="multi"``, tensor data is stored
          in separate ``Vq*.h5`` files alongside this metadata file.
        - ``write_type`` *(str, optional, default ``"multi"``)* — how Cholesky 
          integrals are written. 
          ``"multi"`` uses one file per q-point (``Vq*.h5``);
          ``"single"`` writes everything into one file.
        - ``read_type`` *(str, optional, default ``"all"``)* — how stored vectors
          are read back. ``"all"`` loads all k-points for a given q-point at once
          (higher memory); ``"single"`` reads each (k, k-q) pair separately
          (lower memory).
        - ``overwrite`` *(bool, optional, default ``False``)* — if ``False``,
          existing Cholesky files on disk are reused. Set to ``True`` to force
          recomputation.
        - ``storage`` *(str, optional, default ``"outcore"``)* — currently only
          ``"outcore"`` is supported (integrals are read from disk as needed).

    Returns
    -------
    CholCoulomb
        A Cholesky Coulomb interaction object that can be passed to MBPT functions
        such as ``run_hf`` and ``run_gw``.

    Examples
    --------
    ::

        from coqui.interaction import make_chol_coulomb

        chol = make_chol_coulomb(mf, {"path": "chol_dir/", "tol": 1e-4})
    """
    # Create output directory if it does not exist
    path = os.path.abspath(os.path.expanduser(params.get("path", "./")))
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            raise RuntimeError(f"make_chol_coulomb: Failed to create directory '{path}': {e}") from e
    elif not os.path.isdir(path):
        raise RuntimeError(f"make_chol_coulomb: params['path'] exists but is not a directory: {path}")

    return CholCoulomb(mf, json.dumps(params))

