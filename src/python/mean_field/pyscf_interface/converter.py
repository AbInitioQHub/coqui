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
from pyscf.pbc import scf, dft, tools

from pyscf import dft as mol_dft
from pyscf import scf as mol_scf


def dump_to_h5(mf, mo=False, outdir='./', prefix='pyscf', nbnd_out=None):
  """
  Dump the mean-field information from a pyscf calculation (HF/DFT) into a hdf5 file
  :param mf: [INPUT] pyscf pbc mean-field object, i.e.
                      pyscf.pbc.dft.KRKS, pyscf.pbc.dft.KUKS,
                      pyscf.pbc.scf.KRHF, and pyscf.pbc.scf.KUHF
  :param mo: [INPUT] MO or AO basis
  :param outdir: [INPUT] output directory
  :param prefix: [INPUT] prefix for the output hdf5 file
  :param nbnd_out: [INPUT] dimension of the output orbital space. FOR DEBUG PURPOSE!
  :return: 0
  """
  os.system("sync")
  if not os.path.exists(outdir):
    os.system("mkdir " + outdir)
  filename = outdir + prefix + '.h5'
  print("# Dump the mean-field information from a pyscf calculation (HF/DFT) into {}".format(filename))

  kp_grid = tools.get_monkhorst_pack_size(mf.cell, mf.kpts)
  kpts = mf.cell.make_kpts(kp_grid)
  assert np.array_equal(kpts, mf.kpts), "kpts mismatch!"

  nkpts, nbnd = len(mf.kpts), mf.cell.nao_nr()
  S, H0, dm = mf.get_ovlp(), mf.get_hcore(), mf.make_rdm1()
  F = mf.get_fock(H0, S, None, dm)
  mo_coeff, mo_occ, mo_energy = np.asarray(mf.mo_coeff), np.asarray(mf.mo_occ), np.asarray(mf.mo_energy)
  if isinstance(mf, scf.krhf.KRHF) or isinstance(mf, dft.krks.KRKS):
    nspin = 1
    nspin_in_basis = 1
    S = S.reshape(nspin, nkpts, nbnd, nbnd).astype(complex)
    H0 = H0.reshape(nspin, nkpts, nbnd, nbnd).astype(complex)
    dm = dm.reshape(nspin, nkpts, nbnd, nbnd).astype(complex)
    F  = F.reshape(nspin, nkpts, nbnd, nbnd).astype(complex)
    mo_coeff = mo_coeff.reshape(nspin, nkpts, nbnd, nbnd).astype(complex)
    mo_occ = mo_occ.reshape(nspin, nkpts, nbnd).astype(float)
    mo_occ /= 2.0
    mo_energy = mo_energy.reshape(nspin, nkpts, nbnd).astype(float)
  elif isinstance(mf, scf.kuhf.KUHF) or isinstance(mf, dft.kuks.KUKS):
    nspin = 2
    nspin_in_basis = 2 if mo else 1
    mo_occ = mo_occ.astype(float)
    S, H0 = np.array((S, S)), np.array((H0, H0))
  else:
    raise ValueError("Incorrect type of mf object.")

  print("Shape of S = {}".format(S.shape))

  if mo:
    orth_error = -1.0
    for s in range(nspin):
      for k in range(nkpts):
        H0[s, k] = reduce(np.dot, (mo_coeff[s, k].T.conj(), H0[s, k], mo_coeff[s, k]))
        F[s, k]  = reduce(np.dot, (mo_coeff[s, k].T.conj(), F[s, k], mo_coeff[s, k]))
        S[s, k]  = reduce(np.dot, (mo_coeff[s, k].T.conj(), S[s, k], mo_coeff[s, k]))
        no_c = np.linalg.inv(mo_coeff[s, k].conj().T)
        dm[s, k] = reduce(np.dot, (no_c.conj().T, dm[s, k], no_c))
        # Check orthogonality
        diff = S[s, k] - np.eye(nbnd)
        orth_error = max(orth_error, np.max(np.abs(diff)))
    print("Maximum error of orthogonalization: {}".format(orth_error))
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
  for s in range(nspin_in_basis):
    k = 0
    for kpt in mf.kpts:
      # (nGv, nao)
      Orb_kG = ft_ao.ft_ao(mf.cell, Gv, shls_slice, b, gxyz, Gvbase, kpt)
      if mo:
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
  h5.h5_write(g, 'nspin_in_basis', np.int32(nspin_in_basis))
  h5.h5_write(g, 'natoms', np.int32(natoms))
  h5.h5_write(g, 'species', species)
  h5.h5_write(g, 'nspecies', np.int32(nspecies))
  h5.h5_write(g, 'nelec', np.float64(mf.cell.nelectron) )
  h5.h5_write(g, 'at_ids', charges.astype(np.int32))
  h5.h5_write(g, 'at_pos', coords)
  h5.h5_write(g, 'latt', mf.cell.lattice_vectors() )
  h5.h5_write(g, 'recv', b) # a*b = 2*pi
  h5.h5_write(g, 'kp_grid', kp_grid.astype(np.int32) )
  h5.h5_write(g, 'kpts', mf.kpts) # in absolute unit
  h5.h5_write(g, 'k_weight', np.ones(len(mf.kpts), dtype=float)/len(mf.kpts) )
  h5.h5_write(g, 'madelung', tools.pbc.madelung(mf.cell, mf.kpts))
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

  del f

  dump_orb(outdir+"/Orb_fft", Orb_G[:,:nbnd_out])

  return 0


def mol_dump_to_h5(mf, becke_grid_level=3, mo=False, outdir='./', prefix='pyscf', r_grid=None, nbnd_out=None):
  '''
  Dump the mean-field information from a pyscf calculation (HF/DFT) into a hdf5 file
  :param mf: [INPUT] pyscf mean-field object, i.e.
                     pyscf.dft.RKS, pyscf.dft.UKS,
                     pyscf.scf.RHF, and pyscf.scf.UHF
  :param becke_grid_level: [INPUT] To control the number of radial and angular grids. Large number
                                   leads to large mesh grids. Grids level 0 - 9.  Big number
                                   indicates dense grids.
  :param mo: [INPUT] MO or AO basis
  :param outdir: [INPUT] output directory
  :param prefix: [INPUT] prefix for the output hdf5 file
  :param r_grid: [OPTIONAL INPUT] initial real space grid for ISDF
  :param nbnd_out: [INPUT] dimension of the output orbital space. FOR DEBUG PURPOSE!
  :return: 0
  '''
  os.system("sync")
  if not os.path.exists(outdir):
    os.system("mkdir " + outdir)
  filename = outdir + prefix + '.h5'
  print("# Dump the mean-field information from a pyscf calculation (HF/DFT) into {}".format(filename))

  nkpts, nbnd = 1, mf.mol.nao_nr()
  if isinstance(mf, mol_scf.hf.RHF) or isinstance(mf, mol_dft.rks.RKS):
    nspin = 1
    nspin_in_basis = 1
    S = np.zeros((nspin, nkpts, nbnd, nbnd), dtype=complex)
    H0 = np.zeros((nspin, nkpts, nbnd, nbnd), dtype=complex)
    dm = np.zeros((nspin, nkpts, nbnd, nbnd), dtype=complex)
    F = np.zeros((nspin, nkpts, nbnd, nbnd), dtype=complex)
    mo_coeff = np.zeros((nspin, nkpts, nbnd, nbnd), dtype=complex)
    mo_occ = np.zeros((nspin, nkpts, nbnd), dtype=float)
    mo_energy = np.zeros((nspin, nkpts, nbnd), dtype=float)

    S[0,0], H0[0,0], dm[0,0] = mf.get_ovlp(), mf.get_hcore(), mf.make_rdm1()
    F[0,0] = mf.get_fock()
    mo_coeff[0,0], mo_occ[0,0], mo_energy[0,0] = np.asarray(mf.mo_coeff), np.asarray(mf.mo_occ), np.asarray(mf.mo_energy)
  elif isinstance(mf, mol_scf.uhf.UHF) or isinstance(mf, mol_dft.uks.UKS):
    nspin = 2
    nspin_in_basis = 2 if mo else 1
    S = np.zeros((nspin, nkpts, nbnd, nbnd), dtype=complex)
    H0 = np.zeros((nspin, nkpts, nbnd, nbnd), dtype=complex)
    dm = np.zeros((nspin, nkpts, nbnd, nbnd), dtype=complex)
    F = np.zeros((nspin, nkpts, nbnd, nbnd), dtype=complex)
    mo_coeff = np.zeros((nspin, nkpts, nbnd, nbnd), dtype=complex)
    mo_occ = np.zeros((nspin, nkpts, nbnd), dtype=float)
    mo_energy = np.zeros((nspin, nkpts, nbnd), dtype=float)

    S[0,0], H0[0,0], dm[:,0] = mf.get_ovlp(), mf.get_hcore(), mf.make_rdm1()
    S[1] = S[0]
    H0[1] = H0[0]
    F[:,0] = mf.get_fock()
    mo_coeff[:,0], mo_occ[:,0], mo_energy[:,0] = np.asarray(mf.mo_coeff), np.asarray(mf.mo_occ), np.asarray(mf.mo_energy)
  else:
    raise ValueError("Incorrect type of mf object.")

  if mo:
    orth_error = -1.0
    for s in range(nspin_in_basis):
      for k in range(nkpts):
        H0[s, k] = reduce(np.dot, (mo_coeff[s, k].T.conj(), H0[s, k], mo_coeff[s, k]))
        F[s, k]  = reduce(np.dot, (mo_coeff[s, k].T.conj(), F[s, k], mo_coeff[s, k]))
        S[s, k]  = reduce(np.dot, (mo_coeff[s, k].T.conj(), S[s, k], mo_coeff[s, k]))
        no_c = np.linalg.inv(mo_coeff[s, k].conj().T)
        dm[s, k] = reduce(np.dot, (no_c.conj().T, dm[s, k], no_c))
        # Check orthogonality
        diff = S[s, k] - np.eye(nbnd)
        orth_error = max(orth_error, np.max(np.abs(diff)))
    print("Maximum error of orthogonalization: {}".format(orth_error))
  natoms  = mf.mol.natm
  species = np.asarray(mf.mol.elements)
  species = np.unique(species)
  nspecies = len(species)
  species = str(species)
  charges = np.asarray(mf.mol.atom_charges(), dtype=int)
  coords  = np.asarray(mf.mol.atom_coords('ANG'), dtype=float)

  if r_grid is None:
    # Atomic orbitals on Becke grids
    grids = mol_dft.gen_grid.Grids(mf.mol)
    grids.level = becke_grid_level
    grids.build()
    r_grid = grids.coords
    becke_weights = grids.weights
  else:
    becke_weights = np.ones(r_grid.shape[0], dtype=float)
  Orb_r = np.zeros((nspin_in_basis, nbnd, r_grid.shape[0]), dtype=complex)
  Orb_ao = mol_dft.numint.eval_ao(mf.mol, r_grid)
  #Orb_r[0] = mol_dft.numint.eval_ao(mf.mol, r_grid).T
  for s in range(nspin_in_basis):
    if mo:
      # (nr, nao) * (nao, nao)
      Orb_r[s] = np.dot(Orb_ao, mo_coeff[s,0]).T
    else:
      Orb_r[s] = Orb_ao.T

  if nbnd_out is None:
    nbnd_out = nbnd

  f = h5.File(filename, 'w')
  g = h5.Group(f)

  h5.h5_write(g, 'nkpts', np.int32(nkpts))
  h5.h5_write(g, 'nbnd', np.int32(nbnd_out))
  h5.h5_write(g, 'nspin', np.int32(nspin))
  h5.h5_write(g, 'nspin_in_basis', np.int32(nspin_in_basis))
  h5.h5_write(g, 'natoms', np.int32(natoms))
  h5.h5_write(g, 'species', species)
  h5.h5_write(g, 'nspecies', np.int32(nspecies))
  h5.h5_write(g, 'nelec', np.float64(mf.mol.nelectron) )
  h5.h5_write(g, 'at_ids', charges.astype(np.int32))
  h5.h5_write(g, 'at_pos', coords)
  h5.h5_write(g, 'latt', np.eye(3, dtype=float))
  h5.h5_write(g, 'recv', 2*np.pi*np.eye(3, dtype=float)) # a*b = 2*pi
  h5.h5_write(g, 'kpts', np.zeros((1,3), dtype=float)) # in absolute unit
  h5.h5_write(g, 'k_weight', np.ones(1, dtype=float))
  h5.h5_write(g, 'madelung', 0.0)
  h5.h5_write(g, 'enuc', mf.mol.energy_nuc())

  becke_g = g.create_group("BECKE")
  h5.h5_write(becke_g, 'r_grid', r_grid)
  h5.h5_write(becke_g, 'weight', becke_weights)
  h5.h5_write(becke_g, 'number_of_rpoints', np.int32(becke_weights.shape[0]) )

  scf_g = g.create_group("SCF")
  h5.h5_write(scf_g, 'ovlp', S[:,:,:nbnd_out,:nbnd_out])
  h5.h5_write(scf_g, 'H0', H0[:,:,:nbnd_out,:nbnd_out])
  h5.h5_write(scf_g, 'Fock', F[:,:,:nbnd_out,:nbnd_out])
  h5.h5_write(scf_g, 'dm', dm[:,:,:nbnd_out,:nbnd_out])
  h5.h5_write(scf_g, 'mo_coeff', mo_coeff[:,:,:nbnd_out,:nbnd_out])
  h5.h5_write(scf_g, 'eigval', mo_energy[:,:,:nbnd_out])
  h5.h5_write(scf_g, 'occ', mo_occ[:,:,:nbnd_out])
  del f

  dump_orb(outdir+"/Orb_r", Orb_r[:,:nbnd_out])

  return 0


def dump_orb(outdir, Orb_G):
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









