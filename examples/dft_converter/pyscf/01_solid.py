#!/usr/bin/env python
"""
Converter for a PySCF mean-field object with k-points

Our converter uses the Python interface for hdf5 from
nda (https://github.com/TRIQS/nda), rather than h5py.
"""

from pyscf.pbc import gto, scf, dft

from coqui.mean_field.pyscf_interface import dump_to_h5

cell = gto.M(
    a = '''0.0,  2.715, 2.715
           2.715, 0.0,  2.715
           2.715, 2.715, 0.0''',
    atom = '''Si  0.      0.      0.
              Si 1.3575 1.3575 1.3575''',
    basis = 'gth-dzvp-molopt-sr',
    pseudo = 'gth-pbe',
    verbose = 7,
)

kmf = scf.KRKS(cell, cell.make_kpts([2,2,2]))
kmf.xc = 'pbe'
kmf.kernel()

#
# dump_to_h5() function extracts necessary information form kmf into h5 files
# that is readable by CoQuí.
#
# When "mo=True", all quantities in the AO basis will be transformed to MOs.
#
# The outputs include: 
#   a) a h5 file for metadata: "outdir/prefix.h5"
#   b) a folder for wfc on a FFT mesh: "outdir/Orb_fft"
if dump_to_h5(mf=kmf, mo=False, outdir='./', prefix='pyscf') != 0:
    raise ValueError("dump_to_h5() fail!")
