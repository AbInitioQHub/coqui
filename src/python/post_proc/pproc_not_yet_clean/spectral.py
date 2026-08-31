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

import sys
import numpy as np

try:
    from h5 import HDFArchive
except ImportError:
    raise ImportError("Fails to import triqs/h5! \n"
                      "The utility functions in the spectral module requires the C++ HDF5 from triqs/h5 \n"
                      "(https://github.com/TRIQS/h5). Please ensure that it is installed. ")


def spectral_kpath(aimbes_h5, aimb_iter, Simp_wsa, w_mesh,
                   eta=0.001, verbal=True):
    with HDFArchive(aimbes_h5, "r") as ar:
        mu = ar[f'embed/iter{aimb_iter}/mu']
        F_skab_kpath = ar[f'embed/iter{aimb_iter}/wannier_inter/F_skab']
        kpath = ar[f'embed/iter{aimb_iter}/wannier_inter/kpts']

    nw, ns, nWanOrb = Simp_wsa.shape
    nkpts = kpath.shape[0]

    if verbal:
        print("\nComputing DMFT spectral function along the high-symmetry kpath")
        print("----------------------------------------------------------------")
        print(f"  - nw = {nw}")
        print(f"  - nspin = {ns}")
        print(f"  - nkpts = {nkpts}")
        print(f"  - # of Wannier orbitals = {nWanOrb}")
        print(f"  - eta = {eta}\n")
        sys.stdout.flush()

    A_wska = np.zeros((nw, ns, nkpts, nWanOrb), dtype=float)
    for n, wn in enumerate(w_mesh):
        for sa in range(ns*nWanOrb):
            s = sa // nWanOrb
            a = sa % nWanOrb
            Ginv_k = wn + mu - F_skab_kpath[s,:,a,a] - Simp_wsa[n,s,a] + 1j*eta
            A_wska[n,s,:,a] = -1.0/np.pi * (1.0/Ginv_k).imag

    return A_wska

def spectral_from_maxent(coqui_h5, coqui_iter, ft, coqui_grp="embed", **kwargs):
    import py2aimb.dmft.pproc.analytic_cont as AC
    with HDFArchive(coqui_h5, 'r') as ar:
        Gwkab_ir = ar[f"{coqui_grp}/iter{coqui_iter}/wannier_inter/G_wskab"][:, 0]
        Gwka_ir = np.diagonal(Gwkab_ir, axis1=-2, axis2=-1)
        Gwk_ir = np.sum(Gwka_ir, axis=2)

    # loop over k-path
    A_kpath = []
    for ik in range(Gwk_ir.shape[1]):
        print(f"Processing kpoint {ik}...")
        maxent_results = AC._maxent_run(Gwk_ir[:, ik], ft, **kwargs)
        A_kpath.append(maxent_results.get_A_out("LineFitAnalyzer")[0, 0])

    return A_kpath, maxent_results.omega
