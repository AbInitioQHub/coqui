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

import os
import numpy as np
import h5._h5py as h5

from pyscf import lib
from pyscf.pbc import gto, df
import pyscf.df as mol_df


def test_cell():
    cell = gto.M(
        a='''4.0655,    0.0,    0.0
             0.0,    4.0655,    0.0
             0.0,    0.0,    4.0655''',
        atom='''H -0.25 -0.25 -0.25
                H  0.25  0.25  0.25''',
        unit = 'A',
        basis = 'sto-3g',
        verbose=4
    )
    return cell


def wrap_to_1stBZ(k):
    while k < 0 :
        k = 1 + k
    while (k - 9.9999999999e-1) > 0.0 :
        k = k - 1
    return k


def compute_qpts(kpts=None):
    if kpts is None:
        raise ValueError('compute_qpts: missing kpts argument')

    qpts = kpts[:, :] - kpts[0, :]
    return qpts


def compute_qk_to_kmq(cell, kpts, qpts=None):
    """
    Map from (iq, ik) -> ikk such that
    qk_to_kmq[iq, ik] = ikk, where Q(iq) = K(ik) - K(ikk) + G for some reciprocal vector G
    """
    if qpts is None:
        qpts = compute_qpts(kpts)
    if kpts.shape != qpts.shape:
        raise ValueError("compute_qk_to_kmq: inconsistent dimensions for qpts and kpts")

    nqpts, nkpts = qpts.shape[0], kpts.shape[0]
    qk_to_kmq = np.empty((nqpts, nkpts), dtype=int)
    qk_to_kmq.fill(-1)

    for iq in range(nqpts):
        for ik in range(nkpts):
            for ikk in range(nkpts):
                # Given (iq, ik) pair,
                # find ikk such that K(ikk) = K(ik) - Q(iq) + G
                dk = kpts[ik] - qpts[iq] - kpts[ikk]
                # Wrap back to the first Brillioun zone
                dk_scaled = cell.get_scaled_kpts(dk)
                dk_scaled = [wrap_to_1stBZ(i) for i in dk_scaled]
                # check if dk == n * G
                if np.linalg.norm(dk_scaled) < 1e-6:
                    if qk_to_kmq[iq, ik] != -1:
                        raise ValueError("compute_qk_to_kmq: \n"
                                         "problems defining k2 = q - k1 mapping")
                    qk_to_kmq[iq, ik] = ikk
                    break
            if qk_to_kmq[iq, ik] == -1:
                raise ValueError("compute_qk_to_kmq: \n"
                                 "problems defining k2 = q - k1 mapping")
    return qk_to_kmq


def gdf_dump_to_h5(gdf, mo=False, outdir=None, qpts=None, qk_to_kmq=None):
    if not isinstance(gdf, df.RSGDF):
        raise ValueError("gdf_dump_to_h5: DF obejct has to be a RSGDF instance")
    if outdir is None:
        outdir = './'
    if qpts is None:
        qpts = compute_qpts(gdf.kpts)
    if qk_to_kmq is None:
        qk_to_kmq = compute_qk_to_kmq(gdf.cell, gdf.kpts, qpts)
    if mo:
        raise ValueError("gdf_dump_to_h5: MO is not supported yet!")

    if not os.path.exists(gdf._cderi_to_save):
        gdf.build()
    if gdf.auxcell is None:
        raise ValueError("gdf_dump_to_h5: gdf.auxcell is not initialized")

    print("# Dump GDF-ERIs from pyscf (\"{}\") to \"{}\"".format(gdf._cderi_to_save, outdir))

    cell = gdf.cell
    nkpts = gdf.kpts.shape[0]
    nqpts = qpts.shape[0]
    nbnd  = cell.nao_nr()
    Naux  = gdf.auxcell.nao_nr()

    if not os.path.exists(outdir):
        os.system("mkdir " + outdir)
    filename = outdir + '/chol_info.h5'
    f = h5.File(filename, 'w')
    g = h5.Group(f).create_group("Interaction")
    h5.h5_write(g, 'Np', np.int32(Naux))
    h5.h5_write(g, "tol", -1.0)
    h5.h5_write(g, 'nkpts', np.int32(nkpts))
    h5.h5_write(g, 'nbnd', np.int32(nbnd))
    h5.h5_write(g, 'kpts', gdf.kpts)
    h5.h5_write(g, 'qpts', qpts)
    h5.h5_write(g, 'qk_to_kmq', qk_to_kmq.astype(np.int32))
    del f

    # loop over q-points: extract V(k, k-q)(Q, i, j)
    V_Qskij = np.zeros((Naux, 1, nkpts, nbnd, nbnd), dtype=complex)
    for iq in range(nqpts):
        filename = outdir + "/Vq{}.h5".format(iq)
        f = h5.File(filename, 'w')
        g = h5.Group(f).create_group("Interaction")
        h5.h5_write(g, "Np", np.int32(Naux))
        for ik in range(nkpts):
            ikk = qk_to_kmq[iq, ik]
            k, kmq = gdf.kpts[ik], gdf.kpts[ikk]
            Q_loc = 0
            for bufferR, bufferI, sign in gdf.sr_loop((k, kmq), compact=False):
                X = (bufferR + bufferI*1j)
                X = X.reshape(-1, nbnd, nbnd)
                Qsize = X.shape[0]
                V_Qskij[Q_loc:Q_loc+Qsize,0,ik] = X
                Q_loc += Qsize
        h5.h5_write(g, 'Vq{}'.format(iq), V_Qskij)
        V_Qskij[:] = 0.0
        del f


def mol_gdf_dump_to_h5(gdf, mo=False, outdir=None):
    if not isinstance(gdf, mol_df.DF):
        raise ValueError("mol_gdf_dump_to_h5: DF obejct has to be a DF instance")
    if outdir is None:
        outdir = './'
    if mo:
        raise ValueError("mol_gdf_dump_to_h5: MO is not supported yet!")
    if gdf.auxmol is None:
        gdf.build()

    print("# Dump GDF-ERIs from pyscf (\"{}\") to \"{}\"".format(gdf._cderi_to_save, outdir))

    mol   = gdf.mol
    nkpts, nqpts = 1, 1
    nbnd  = mol.nao_nr()
    Naux  = gdf.auxmol.nao_nr()
    kpts  = np.zeros((1,3), dtype=float)
    qpts  = np.zeros((1,3), dtype=float)
    qk_to_kmq = np.zeros((1,1), dtype=int)

    if not os.path.exists(outdir):
        os.system("mkdir " + outdir)
    filename = outdir + '/chol_info.h5'
    f = h5.File(filename, 'w')
    g = h5.Group(f).create_group("Interaction")
    h5.h5_write(g, 'Np', np.int32(Naux))
    h5.h5_write(g, "tol", -1.0)
    h5.h5_write(g, 'nkpts', np.int32(nkpts))
    h5.h5_write(g, 'nbnd', np.int32(nbnd))
    h5.h5_write(g, 'kpts', kpts)
    h5.h5_write(g, 'qpts', qpts)
    h5.h5_write(g, 'qk_to_kmq', qk_to_kmq.astype(np.int32))
    del f

    # loop over q-points: extract V(k, k-q)(Q, i, j)
    V_Qskij = np.zeros((Naux, 1, nkpts, nbnd, nbnd), dtype=complex)
    filename = outdir + "/Vq0.h5"
    f = h5.File(filename, 'w')
    g = h5.Group(f).create_group("Interaction")
    h5.h5_write(g, "Np", np.int32(Naux))
    Q_loc = 0
    for L in gdf.loop():
        # L = (Naux, nbnd*(nbnd+1)//2)
        print("Shape of eri = ", L.shape)
        L_unpack = lib.unpack_tril(L, lib.SYMMETRIC, axis=1)
        print("Shape of unpack eri = ", L_unpack.shape)
        Q_size = L_unpack.shape[0]
        V_Qskij[Q_loc:Q_loc+Q_size,0,0] = L_unpack
        Q_loc += Q_size
    h5.h5_write(g, 'Vq0', V_Qskij)
    V_Qskij[:] = 0.0
    del f



class bdft_DF(object):
    """
    A DF interface class that wraps the RSGDF class from pyscf
    mydf = bdft_GDF(cell, kpts, auxbasis)
    mydf.build()
    mydf.dump_to_bdft_format()

    mydf = bdft_GDF() // An example system is constructed
    mydf.build()
    mydf.dump_to_bdft_format()
    """
    def __init__(self, df=None, cell=None, kpts=None):
        ''' Initialization '''
        # public instance variables
        self.df = None            # instance of DF object
        self.cell = None          # system
        self.kpts = None          # k-points
        self.qpts = None          # q-points
        self.auxbasis = None      # auxiliary basis for DF
        self.outdir = "df_eri"    # output directory of DF ERIs in the bdft format

        ''' Setup '''
        if self.df is not None:
            self.df = df
        if cell is None:
            self.cell = test_cell()
        else:
            self.cell = cell
        if kpts is None:
            self.kpts = self.cell.make_kpts([2,2,2])
        else:
            self.kpts = kpts
        self.qpts = compute_qpts(self.kpts)
        self.qk_to_kmq = compute_qk_to_kmq(self.cell, self.qpts)

        print(self)

    def __str__(self):
        return "\n"\
               "**** ERI interface between PySCF and coqui ****\n"

    def RSGDF_build(self, auxbasis=None):
        self.df = df.RSGDF(self.cell, kpts=self.kpts)
        if auxbasis is not None:
            self.df.auxbasis = auxbasis
        self.df._cderi_to_save = "cderi.h5"
        self.df.build()
        #if os.path.exists("cderi.h5"):
        #    self.df._cderi = "cderi.h5"
        #else:
        #    self.df._cderi_to_save = "cderi.h5"
        #    self.df.build()

    def RSGDF_dump(self):
        gdf_dump_to_h5(self.df, False, self.outdir, self.qpts, self.qk_to_kmq)

    def FFTDF_build(self, mesh=None):
        self.df = df.FFTDF(self.cell, kpts=self.kpts)
        if mesh is not None:
            self.df.mesh = mesh
        self.df.build()

    def FFTDF_eri(self, ik, ikk, iq):
        nbnd = self.cell.nao_nr()
        ikmq = self.qk_to_kmq[iq, ik]
        ikkmq = self.qk_to_kmq[iq, ikk]
        ikpts = [ik, ikmq, ikkmq, ikk]
        kpts = self.kpts[ikpts]
        eri = self.df.get_eri(kpts, compact=False)
        eri = eri.reshape(nbnd, nbnd, nbnd, nbnd)
        return eri


if __name__ == '__main__':
    mydf = bdft_DF()
    mydf.build()
    mydf.dump_to_bdft_format()

    print("kpts: \n{}".format(mydf.kpts))
    scaled_kpts = mydf.cell.get_scaled_kpts(mydf.kpts)
    print("scaled kpts: \n{}".format(scaled_kpts))

    qk_to_kmq = mydf.compute_qk_to_kmq()
    print(qk_to_kmq.shape)
    print(qk_to_kmq[0,0])




