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
from functools import reduce
import numpy as np
import h5._h5py as h5

from pyscf import lib
from pyscf.pbc.df import ft_ao
from pyscf.pbc import gto, tools
from pyscf.pbc.dft.krks import KRKS
from pyscf.pbc.scf.khf import KRHF
from pyscf.pbc.dft.kuks import KUKS
from pyscf.pbc.scf.kuhf import KUHF

def dump_to_h5(mf, super_lattice_kpts, MO=False, outdir='./', prefix='pyscf', nbnd_out=None):
  '''
  Dump the mean-field information from a pyscf calculation (HF/DFT) into a hdf5 file
  :param mf: [INPUT] pyscf pbc mean-field object, i.e.
                      pyscf.pbc.dft.KRKS, pyscf.pbc.dft.KUKS,
                      pyscf.pbc.scf.KRHF, and pyscf.pbc.scf.KUHF
  :param super_lattice_kpts: [INPUT] k-points inside k-patch
  :param MO: [INPUT] MO or AO basis
  :param outdir: [INPUT] output directory
  :param prefix: [INPUT] prefix for the output hdf5 file
  :param nbnd_out: [INPUT] dimension of the output orbital space. FOR DEBUG PURPOSE!
  :return: 0
  '''
  os.system("sync")
  if not os.path.exists(outdir):
    os.system("mkdir " + outdir)
  filename = outdir + prefix + '.h5'
  print("# Dump the mean-field information from a pyscf calculation (HF/DFT) into {}".format(filename))

  nkpts, nbnd = len(mf.kpts), mf.cell.nao_nr()
  S, H0, dm = mf.get_ovlp(), mf.get_hcore(), mf.make_rdm1()
  F = mf.get_fock(H0, S, None, dm)
  mo_coeff, mo_occ, mo_energy = np.asarray(mf.mo_coeff), np.asarray(mf.mo_occ), np.asarray(mf.mo_energy)
  # compute super lattice quantities
  super_lattice_nkpts = super_lattice_kpts.shape[0]
  full_kpts = []
  for ik, k in enumerate(super_lattice_kpts):
    for iK, K in enumerate(mf.kpts):
      full_kpts.append(k + K)
  full_kpts = np.array(full_kpts)
  S_lattice, H0_lattice =  mf.get_ovlp(kpts=full_kpts), mf.get_hcore(kpts=full_kpts)
  if isinstance(mf, KRHF) or isinstance(mf, KRKS):
    nspin = 1
    S = S.reshape(nspin, nkpts, nbnd, nbnd).astype(complex)
    H0 = H0.reshape(nspin, nkpts, nbnd, nbnd).astype(complex)
    S_lattice = S_lattice.reshape(super_lattice_nkpts, nspin, nkpts, nbnd, nbnd).astype(complex)
    H0_lattice = H0_lattice.reshape(super_lattice_nkpts, nspin, nkpts, nbnd, nbnd).astype(complex)
    dm = dm.reshape(nspin, nkpts, nbnd, nbnd).astype(complex)
    F  = F.reshape(nspin, nkpts, nbnd, nbnd).astype(complex)
    mo_coeff = mo_coeff.reshape(nspin, nkpts, nbnd, nbnd)
    mo_occ = mo_occ.reshape(nspin, nkpts, nbnd).astype(float)
    mo_occ /= 2.0
    mo_energy = mo_energy.reshape(nspin, nkpts, nbnd)
  elif isinstance(mf, KUHF) or isinstance(mf, KUKS):
    nspin = 2
    mo_occ = mo_occ.astype(float)
    S, H0 = np.array((S, S)), np.array((H0, H0))
    S_lattice, H0_lattice = np.array((S_lattice, S_lattice)), np.array((H0_lattice, H0_lattice))
    S_lattice = np.einsum("skKij->ksKij", S_lattice).astype(complex)
    H0_lattice = np.einsum("skKij->ksKij", H0_lattice).astype(complex)
  else:
    raise ValueError("Incorrect type of mf object.")

  if MO:
    for s in range(nspin):
      for k in range(nkpts):
        H0[s, k] = reduce(np.dot, (mo_coeff[s, k].T.conj(), H0[s, k], mo_coeff[s, k]))
        F[s, k]  = reduce(np.dot, (mo_coeff[s, k].T.conj(), F[s, k], mo_coeff[s, k]))
        no_c = np.linalg.inv(mo_coeff[s, k].conj().T)
        dm[s, k] = reduce(np.dot, (no_c.conj().T, dm[s, k], no_c))
        S[s, k] = np.eye(nbnd, dtype=complex)

  natoms  = mf.cell.natm
  species = np.asarray(mf.cell.elements)
  species = np.unique(species)
  nspecies = len(species)
  species = str(species)
  charges = np.asarray(mf.cell.atom_charges(), dtype=int)
  coords  = np.asarray(mf.cell.atom_coords('ANG'), dtype=float)

  # Fourier coefficient of Bloch atomic orbitals
  mesh = np.asarray(mf.cell.mesh)
  ecut = tools.pbc.mesh_to_cutoff(mf.cell.lattice_vectors(), mf.cell.mesh)
  ecut = max(ecut)
  ''' Gv: fft mesh; Gv_scaled: scaled fft mesh; weights: 1/cell.vol '''
  Gv, Gvbase, weights = mf.cell.get_Gv_weights(mf.cell.mesh)
  Gv_scaled = lib.cartesian_prod(Gvbase)
  Gv_scaled = np.asarray(Gv_scaled, dtype=np.int32)
  b = mf.cell.reciprocal_vectors(2*np.pi)
  gxyz = lib.cartesian_prod([np.arange(len(x)) for x in Gvbase])
  shls_slice = (0, mf.cell.nbas)
  Orb_G = []
  for s in range(nspin):
    k = 0
    for kpt in mf.kpts:
      # (nGv, nao)
      Orb_kG = ft_ao.ft_ao(mf.cell, Gv, shls_slice, b, gxyz, Gvbase, kpt)
      if MO:
        # (nGv, nao) * (nao, nao)
        Orb_kG = np.dot(Orb_kG, mo_coeff[s, k])
      # (nao, nGv)
      Orb_G.append(Orb_kG.T)
      k+=1
  Orb_G = np.asarray(Orb_G)
  Orb_G *= np.sqrt(weights)

  if nbnd_out is None:
    nbnd_out = nbnd

  f = h5.File(filename, 'w')
  g = h5.Group(f)

  h5.h5_write(g, 'nkpts', np.int32(nkpts))
  h5.h5_write(g, 'nbnd', np.int32(nbnd_out))
  h5.h5_write(g, 'nspin', np.int32(nspin))
  h5.h5_write(g, 'natoms', np.int32(natoms))
  h5.h5_write(g, 'species', species)
  h5.h5_write(g, 'nspecies', np.int32(nspecies))
  h5.h5_write(g, 'nelec', np.float64(mf.cell.nelectron) )
  h5.h5_write(g, 'at_ids', charges.astype(np.int32))
  h5.h5_write(g, 'at_pos', coords)
  h5.h5_write(g, 'latt', mf.cell.lattice_vectors())
  h5.h5_write(g, 'recv', b) # a*b = 2*pi
  h5.h5_write(g, 'kpts', mf.kpts) # in absolute unit
  h5.h5_write(g, 'k_weight', np.ones(len(mf.kpts), dtype=float))
  h5.h5_write(g, 'madelung', tools.pbc.madelung(mf.cell, full_kpts))
  h5.h5_write(g, 'enuc', mf.cell.energy_nuc())

  fft_g = g.create_group("FFT")
  h5.h5_write(fft_g, 'ecut', ecut)
  h5.h5_write(fft_g, 'fft_mesh', mesh.astype(np.int32))

  scf_g = g.create_group("SCF")
  h5.h5_write(scf_g, 'ovlp', S[:,:,:nbnd_out,:nbnd_out])
  h5.h5_write(scf_g, 'H0', H0[:,:,:nbnd_out,:nbnd_out])
  h5.h5_write(scf_g, 'Fock', F[:,:,:nbnd_out,:nbnd_out])
  h5.h5_write(scf_g, 'dm', dm[:,:,:nbnd_out,:nbnd_out])
  h5.h5_write(scf_g, 'mo_coeff', mo_coeff[:,:,:nbnd_out,:nbnd_out])
  h5.h5_write(scf_g, 'eigval', mo_energy[:,:,:nbnd_out])
  h5.h5_write(scf_g, 'occ', mo_occ[:,:,:nbnd_out])
  h5.h5_write(scf_g, 'S_lattice', S_lattice[:,:,:,:nbnd_out,:nbnd_out])
  h5.h5_write(scf_g, 'H0_lattice', H0_lattice[:,:,:,:nbnd_out,:nbnd_out])

  del f

  dump_orb_fft(outdir+"/Orb_fft", Orb_G[:,:nbnd_out])

  return 0


def dump_orb_fft(outdir, Orb_G):
  os.system("sync")
  if os.path.exists(outdir):
    os.system("rm -r " + outdir)
    os.system("sync")
  os.system("mkdir " + outdir)

  nsk, nao, nG = Orb_G.shape[:3]
  for isk in range(nsk):
    f = h5.File(outdir+"/Orb_"+str(isk)+".h5", 'w')
    grp = h5.Group(f)
    h5.h5_write(grp, "Orb_"+str(isk), Orb_G[isk])
    del f









