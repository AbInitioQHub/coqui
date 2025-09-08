"""
Control the number of basis states read from DFT.

This example demonstrates how to tailor the single-particle basis size
(bands/states) loaded from DFT outputs, allowing users to investigate
the basis set convergence without rerunning the DFT calculations.
"""

from mpi4py import MPI
import coqui

qe_dir = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/out"

# mpi handler 
mpi = coqui.MpiHandler()
coqui.set_verbosity(mpi, output_level=1)

# The number of basis functions to be read from the input DFT data
# can be controlled via the "nbnd" parameter.
# By default, all the basis functions from the input DFT data will be read.
mf_params = {
  "prefix": "svo",      # prefix of the QE scf/nscf
  "outdir": qe_dir,     # outdir of the QE scf/nscf
  "nbnd": 20            # number of basis functions read
}
svo_mf = coqui.make_mf(mpi, params=mf_params, mf_type="qe")
assert svo_mf.nbnd() == 20
