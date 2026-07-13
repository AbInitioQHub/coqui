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
import h5

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
    with h5.HDFArchive(filename, 'w') as root:
        root.create_group("Interaction")
        g = root["Interaction"]
        g['Np'] = np.int32(Naux)
        g["tol"] = -1.0
        g['nkpts'] = np.int32(nkpts)
        g['nbnd'] = np.int32(nbnd)
        g['kpts'] = gdf.kpts
        g['qpts'] = qpts
        g['qk_to_kmq'] = qk_to_kmq.astype(np.int32)

    # loop over q-points: extract V(k, k-q)(Q, i, j)
    V_Qskij = np.zeros((Naux, 1, nkpts, nbnd, nbnd), dtype=complex)
    for iq in range(nqpts):
        filename = outdir + "/Vq{}.h5".format(iq)
        with h5.HDFArchive(filename, 'w') as root:
            root.create_group("Interaction")
            g = root["Interaction"]
            g["Np"] = np.int32(Naux)
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
            g['Vq{}'.format(iq)] = V_Qskij
        V_Qskij[:] = 0.0


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
    with h5.HDFArchive(filename, 'w') as root:
        root.create_group("Interaction")
        g = root["Interaction"]
        g['Np'] = np.int32(Naux)
        g["tol"] = -1.0
        g['nkpts'] = np.int32(nkpts)
        g['nbnd'] = np.int32(nbnd)
        g['kpts'] = kpts
        g['qpts'] = qpts
        g['qk_to_kmq'] = qk_to_kmq.astype(np.int32)

    # loop over q-points: extract V(k, k-q)(Q, i, j)
    V_Qskij = np.zeros((Naux, 1, nkpts, nbnd, nbnd), dtype=complex)
    filename = outdir + "/Vq0.h5"
    with h5.HDFArchive(filename, 'w') as root:
        root.create_group("Interaction")
        g = root["Interaction"]
        g["Np"] = np.int32(Naux)
        Q_loc = 0
        for L in gdf.loop():
            # L = (Naux, nbnd*(nbnd+1)//2)
            print("Shape of eri = ", L.shape)
            L_unpack = lib.unpack_tril(L, lib.SYMMETRIC, axis=1)
            print("Shape of unpack eri = ", L_unpack.shape)
            Q_size = L_unpack.shape[0]
            V_Qskij[Q_loc:Q_loc+Q_size,0,0] = L_unpack
            Q_loc += Q_size
        g['Vq0'] = V_Qskij
    V_Qskij[:] = 0.0


def mol_gdf_grad_dump_to_h5(gdf, auxbasis_response=True, mo=False, outdir=None):

    if not isinstance(gdf, mol_df.DF):
        raise ValueError('mol_gdf_dump_to_h5: DF obejct has to be a DF instance')
    if outdir is None:
        outdir = './'
    if mo:
        raise ValueError('mol_gdf_dump_to_h5: MO is not supported yet!')
    if gdf.auxmol is None:
        gdf.build()

    print('# Dump GDF-ERI gradients from pyscf')

    mol   = gdf.mol
    natoms = mol.natm
    nkpts, nqpts = 1, 1
    nbnd  = mol.nao_nr()
    naux  = gdf.auxmol.nao_nr()
    kpts  = np.zeros((1,3), dtype=float)
    qpts  = np.zeros((1,3), dtype=float)
    qk_to_kmq = np.zeros((1,1), dtype=int)

    # imformation of atoms and AO slices
    slice = gdf.mol.aoslice_by_atom()
    mol_ao_slice_by_atom = np.zeros((natoms, 2), dtype=int)
    for a in range(natoms):
        mol_ao_slice_by_atom[a, :] = slice[a, 2:4]
    slice = gdf.auxmol.aoslice_by_atom()
    auxmol_ao_slice_by_atom = np.zeros((natoms, 2), dtype=int)
    for a in range(natoms):
        auxmol_ao_slice_by_atom[a, :] = slice[a, 2:4]
    print()
    print('mol_ao_slice_by_atom')
    print(mol_ao_slice_by_atom)
    print('auxmol_ao_slice_by_atom')
    print(auxmol_ao_slice_by_atom)
    print()

    if not os.path.exists(outdir):
        os.system('mkdir ' + outdir)
    filename = outdir + '/chol_info.h5'
    with h5.HDFArchive(filename, 'w') as root:
        root.create_group("Interaction")
        g = root["Interaction"]
        g['Np'] = np.int32(naux)
        g['tol'] = -1.0
        g['nkpts'] = np.int32(nkpts)
        g['nbnd'] = np.int32(nbnd)
        g['kpts'] = kpts
        g['qpts'] = qpts
        g['qk_to_kmq'] = qk_to_kmq.astype(np.int32)

    filename = outdir + '/Vq0.h5'
    with h5.HDFArchive(filename, 'w') as root:
        root.create_group("Interaction")
        g['natoms'] = np.int32(natoms)
        g['Np'] = np.int32(naux)
        g['nbnd'] = np.int32(nbnd)
        g['mol_ao_slice_by_atom'] = mol_ao_slice_by_atom
        g['auxmol_ao_slice_by_atom'] = auxmol_ao_slice_by_atom


        # integral derivatives

        V_a3ijQ_dijQ = np.zeros((natoms, 3, nbnd, nbnd, naux), dtype=complex)

        # ( d/dX i, j | Q )
        int3c_e1 = mol_df.incore.aux_e2(gdf.mol, gdf.auxmol, intor='int3c2e_ip1', aosym='s1', comp=3)
        print('shape of int3c_e1 = ', int3c_e1.shape)
        V_3ijQ_di = -int3c_e1.astype(complex)
        del int3c_e1

        for a in range(natoms):
            for d in range(3):
                for Q in range(naux):
                    V_a3ijQ_dijQ[a, d, mol_ao_slice_by_atom[a][0]:mol_ao_slice_by_atom[a][1], :, Q] \
                        += V_3ijQ_di[d, mol_ao_slice_by_atom[a][0]:mol_ao_slice_by_atom[a][1], :, Q]
                    V_a3ijQ_dijQ[a, d, :, mol_ao_slice_by_atom[a][0]:mol_ao_slice_by_atom[a][1], Q] \
                        += V_3ijQ_di[d, mol_ao_slice_by_atom[a][0]:mol_ao_slice_by_atom[a][1], :, Q].T
        del V_3ijQ_di

        if auxbasis_response:

            # ( i, j | d/dX Q )
            int3c_e2 = mol_df.incore.aux_e2(gdf.mol, gdf.auxmol, intor='int3c2e_ip2', aosym='s1', comp=3)
            print('shape of int3c_e2 = ', int3c_e2.shape)
            V_3ijQ_dQ = -int3c_e2.astype(complex)
            del int3c_e2

            for a in range(natoms):
                for d in range(3):
                    V_a3ijQ_dijQ[a, d, :, :, auxmol_ao_slice_by_atom[a][0]:auxmol_ao_slice_by_atom[a][1]] \
                        += V_3ijQ_dQ[d, :, :, auxmol_ao_slice_by_atom[a][0]:auxmol_ao_slice_by_atom[a][1]]
            del V_3ijQ_dQ

        # ( i, j | Q )
        int3c = mol_df.incore.aux_e2(gdf.mol, gdf.auxmol, intor='int3c2e', aosym='s1', comp=1)
        print('shape of int3c = ', int3c.shape)
        V_ijQ = int3c.astype(complex)
        del int3c

        # ( P | Q ) and ( P | Q )^{-1}

        int2c = gdf.auxmol.intor('int2c2e', aosym='s1', comp=1)
        print('shape of int2c = ', int2c.shape)
        int2c_inv = np.linalg.inv(int2c)
        eig, U = np.linalg.eigh(int2c)
        int2c_inv_sqrt = np.zeros((naux, naux))
        for Q in range(naux):
            int2c_inv_sqrt[Q, Q] = 1 / np.sqrt(eig[Q])
        int2c_inv_sqrt = U @ int2c_inv_sqrt @ U.T.conj()
        V_PQ = int2c.astype(complex)
        V_PQ_inv = int2c_inv.astype(complex)
        V_PQ_inv_sqrt = int2c_inv_sqrt.astype(complex)
        del int2c
        del int2c_inv
        del int2c_inv_sqrt

        V_ijQ = V_ijQ.reshape((nbnd*nbnd, naux)) @ V_PQ_inv_sqrt

        V_tilde_a3ijQ = np.zeros((natoms, 3, nbnd*nbnd, naux), dtype=complex)

        for a in range(natoms):
            for d in range(3):
                V_tilde_a3ijQ[a, d] += V_a3ijQ_dijQ[a, d].reshape(nbnd*nbnd, naux) @ V_PQ_inv_sqrt
        del V_a3ijQ_dijQ

        if auxbasis_response:

            V_a3PQ_dP = np.zeros((natoms, 3, naux, naux), dtype=complex)

            # ( d/dX P | Q )
            int2c_e1 = gdf.auxmol.intor('int2c2e_ip1', aosym='s1', comp=3)
            print('shape of int2c_e1 = ', int2c_e1.shape)
            V_3PQ_dP = -int2c_e1.astype(complex)
            del int2c_e1

            for a in range(natoms):
                for d in range(3):
                    V_a3PQ_dP[a, d, auxmol_ao_slice_by_atom[a][0]:auxmol_ao_slice_by_atom[a][1], :] \
                        += V_3PQ_dP[d, auxmol_ao_slice_by_atom[a][0]:auxmol_ao_slice_by_atom[a][1], :]
            del V_3PQ_dP

            for a in range(natoms):
                for d in range(3):
                    V_tilde_a3ijQ[a, d] -= V_ijQ @ V_PQ_inv_sqrt @ V_a3PQ_dP[a, d] @ V_PQ_inv_sqrt
            del V_a3PQ_dP

        V_Qskij = np.zeros((naux, 1, 1, nbnd, nbnd), dtype=complex)
        V_ijQ = V_ijQ.reshape((nbnd, nbnd, naux))
        for Q in range(naux):
            for i in range(nbnd):
                for j in range(nbnd):
                    V_Qskij[Q, 0, 0, i, j] = V_ijQ[i, j, Q]

        V_tilde_a3Qskij = np.zeros((natoms, 3, naux, 1, 1, nbnd, nbnd), dtype=complex)
        V_tilde_a3ijQ = V_tilde_a3ijQ.reshape((natoms, 3, nbnd, nbnd, naux))
        for a in range(natoms):
            for d in range(3):
                for Q in range(naux):
                    for i in range(nbnd):
                        for j in range(nbnd):
                            V_tilde_a3Qskij[a, d, Q, 0, 0, i, j] = V_tilde_a3ijQ[a, d, i, j, Q]

        g['Vq0'] = V_Qskij
        g['dVq0'] = V_tilde_a3Qskij

        del V_PQ
        del V_PQ_inv
        del V_PQ_inv_sqrt
        del V_ijQ
        del V_Qskij
        del V_tilde_a3ijQ
        del V_tilde_a3Qskij

        del slice
        del mol_ao_slice_by_atom
        del auxmol_ao_slice_by_atom


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




