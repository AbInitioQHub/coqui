"""
Example: Append Crystal Metadata to a Wannier90 .win File via CoQui

This example demonstrates how to append crystal metadata from a CoQui
mean-field (MF) object to an existing `seedname.win` file.

This functionality is intended for use with the *standalone* Wannier90
workflow. In this mode, Wannier90 requires crystal metadata to be explicitly
present in the `.win` file. CoQui provides a convenient way to populate this
information directly from a MF object, ensuring consistency between the
underlying electronic-structure calculation and the Wannier90 input.

By default, the following crystal metadata are appended:
  - lattice vectors
  - atomic positions
  - k-point information

Notes:
  - This example assumes that a partial `seedname.win` file already exists,
    containing Wannierization-specific options (e.g., projections or
    disentanglement settings), to which the crystal metadata will be appended.
  - help(coqui.append_wannier90_win) for a comprehensive list of parameters

See also
--------
- Wannier90 *libray mode* through CoQui: 01_wannier_lib_mode.py
"""
from mpi4py import MPI
import coqui

qe_dir = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/out"

# mpi handler and verbosity
mpi = coqui.MpiHandler()
coqui.set_verbosity(mpi, output_level=2)

# Constructing a MF object
mf_params = {
    "prefix": "svo",
    "outdir": qe_dir,
    "nbnd": 40
}
svo_mf = coqui.make_mf(mpi, params=mf_params, mf_type="qe")

# append Wannier90 win file
w90_params = {
    "outdir": "./",
  "prefix": "svo",     # equivalent to wannier90's seedname
}
coqui.append_wannier90_win(mf=svo_mf, params=w90_params)
