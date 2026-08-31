"""
Example: PySCF GDF Coulomb Hamiltonian

This example demonstrates how to prepare Gaussian density-fitting (GDF)
integrals in PySCF and export them for use in CoQuí.

The exported GDF integrals are written in the same format as the Cholesky
Coulomb integrals produced internally by CoQuí.

Why GDF?
--------
1. The default ISDF-THC algorithm in CoQuí relies on FFT grids, which are
   insufficient for all-electron molecular calculations.
2. The workaround is to start from a pre-computed GDF integrals (e.g. from PySCF),
   and then apply the Least-Squares THC (LS-THC) procedure in CoQuí to compress
   them into a THC form.

See also
--------
- Cholesky-decomposed Coulomb Hamiltonian: interaction/01_cholesky_coulomb.py
- ISDF-based THC Coulomb Hamiltonian: interaction/01_isdf_thc_coulomb.py
- Least-Squares THC: interactions/01_ls_thc_coulomb.py
"""

from pyscf import gto, df, scf
import coqui
from coqui.interaction.pyscf_interface import mol_gdf_dump_to_h5

mol = gto.M(
    atom = '''O 0 0 0 
              H 0 1 0  
              H 0 0 1''',
    basis = 'ccpvdz',
    verbose = 7
)

mydf = df.DF(mol)
mydf._cderi_to_save = "cderi.h5"
mydf.build()

# --- Export to CoQuí format, identical to that of Cholesky integrals
mol_gdf_dump_to_h5(mydf, outdir="gdf_coulomb")

# --- The exported integrals can now be read in CoQuí as a Cholesky Coulomb Hamiltonian:
# first we need a mean-field object for the same system
from mpi4py import MPI
from coqui.mean_field.pyscf_interface import mol_dump_to_h5
mf = scf.RHF(mol)
mf.kernel()

mol_dump_to_h5(mf, becke_grid_level=8, outdir='./', prefix='pyscf')

# mpi handler
mpi = coqui.MpiHandler()
coqui.set_verbosity(mpi, output_level=1)

# --- Build Mf object from PySCF results
mf_params = {
    "prefix": "pyscf",      # prefix for pyscf converter
    "outdir": "./"          # directory for pyscf results
}
h2o_mf = coqui.make_mf(mpi, params=mf_params, mf_type="pyscf")

# --- Read the GDF integrals as Cholesky Coulomb Hamiltonian
gdf_params = {
    "path": "gdf_coulomb"  # directory we wrote with mol_gdf_dump_to_h5
}
h2o_gdf = coqui.make_chol_coulomb(mf=h2o_mf, params=gdf_params)
