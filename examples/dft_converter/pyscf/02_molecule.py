#!/usr/bin/env python
"""
Convert a PySCF mean-field (molecule) to CoQuí HDF5 format.

What this does
--------------
- Runs a minimal PySCF SCF calculation
- Exports the mean-field data and orbital values on a Becke grid to HDF5
  using CoQuí's converter

Outputs
-------
- <outdir>/<prefix>.h5         : metadata and MF results
- <outdir>/Orb_r/              : orbital values on the Becke grid

Notes
-----
- `becke_grid_level` controls the atom-centered Lebedev/Becke grid density.
  Higher = more accurate but larger files.
- Our converter uses the Python interface for hdf5 from
  nda (https://github.com/TRIQS/nda), rather than h5py.

!! Important !!!
- The orbital dump on the Becke grid itself cannot be used to construct THC
  Coulomb integrals directly. Instead, additional Gaussian density-fitting (GDF) integrals
  need to be constructed first using PySCF and then the Least-Squares THC procedure (LS-THC)
  procedure must be applied.
- See the example: 
  1. interaction/01_pyscf_gdf_coulomb.py 
  2. interaction/01_ls_thc_coulomb.py
"""

from pyscf import gto, scf, dft

from coqui.mean_field.pyscf_interface import mol_dump_to_h5

mol = gto.M(
    atom = '''O 0 0 0 
              H 0 1 0  
              H 0 0 1''',
    basis = 'ccpvdz',
    verbose = 7
)

mf = scf.RHF(mol)
mf.kernel()
mf.analyze()

# --- Export to CoQuí format
mol_dump_to_h5(mf, becke_grid_level=8, outdir='./', prefix='pyscf')

