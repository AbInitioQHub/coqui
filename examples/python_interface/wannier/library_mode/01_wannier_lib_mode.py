"""
Example: Wannier90 MLWF Construction via CoQui (Library Mode)

This example demonstrates how to construct maximally localized Wannier
functions (MLWFs) using CoQui's Python interface to Wannier90 in *library mode*.
In contrast to invoking the standalone `wannier90.x` executable, Wannier90 is
called directly as a linked library.

In CoQui, Wannierization relies on an existing mean-field (MF) object, which
encodes the Bloch wavefunctions obtained from a prior electronic-structure
calculation. These Bloch states define the single-particle Hilbert space and
form the fundamental basis for the construction of MLWFs.

Requirements:
  - Wannier90 must be compiled with library support and linked against CoQui.
  - A `seedname.win` file must be provided to specify Wannier90 options relevant
    to Wannierization, such as projections, disentanglement windows.
    Crystal structure and k-point metadata are not required, as they are already
    contained in the MF object. See `svo.win` for an example.

Notes:
    - help(coqui.wannier90) for a comprehensive list of parameters
"""
from mpi4py import MPI
import coqui

qe_dir = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/out"

# mpi handler and verbosity
mpi = coqui.MpiHandler()
coqui.set_verbosity(mpi, output_level=1)

# Constructing a MF object
mf_params = {
    "prefix": "svo",
    "outdir": qe_dir,
    "nbnd": 40
}
svo_mf = coqui.make_mf(mpi, params=mf_params, mf_type="qe")

# construct MLWFs via calling Wannier90 in the library mode
w90_params = {
    "outdir": "./",
    "prefix": "svo",     # equivalent to wannier90's seedname
}
coqui.wannier90(mf=svo_mf, params=w90_params)
