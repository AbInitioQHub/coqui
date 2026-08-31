"""
Example: Read Wannier90 Standalone Output and Generate MLWF HDF5

This example demonstrates how to read output files from a Wannier90 standalone
calculation and convert them into CoQuí's HDF5 format for maximally localized
Wannier functions (MLWFs).

After running Wannier90 in standalone mode, the following files are generated:
  - `seedname_u.mat`: Unitary rotation matrix (U)
  - `seedname_u_dis.mat`: Disentanglement matrix (U_dis) [if disentanglement was used]
  - `seedname_centres.xyz`: Wannier function centres
  - `seedname.eig`: Band eigenvalues [optional]
  - `seedname.win`: Input file containing num_wann, num_bands, exclude_bands

This function reads these files and converts them into CoQuí's HDF5 format,
which can then be used for embedding or other many-body calculations.

Notes:
  - The `.win` file is required to determine the number of Wannier functions,
    bands, and any excluded bands.
  - If `seedname.eig` is not present, eigenvalues will be computed from the
    mean-field object.
  - Disentanglement is auto-detected based on the presence of `seedname_u_dis.mat`.
  - help(coqui.mlwf_h5_from_wannier90_output) for a comprehensive list of parameters

See also
--------
- Append `seedname.win` with crystal metadata: 01_append_wannier90_win.py
- Generate Wannier90 inputs (amn/mmn): 02_coqui2wannier.py
- Wannier90 *library mode* through CoQui: ../01_wannier_lib_mode.py
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

# Read Wannier90 standalone output and generate MLWF HDF5
w90_params = {
  "outdir": "./",
  "prefix": "svo",              # equivalent to wannier90's seedname
  "h5_filename": "svo.mlwf.h5", # output HDF5 filename 
}
coqui.mlwf_h5_from_wannier90_output(mf=svo_mf, params=w90_params)


# For the generated HDF5 file to be interfaced with TRIQS/Modest, one can further partition
# the MLWFs into shells based on their angular momentum character. This requires specifying the `shells` parameter
# in the `w90_params` dictionary: 
w90_params = {
  "outdir": "./",
  "prefix": "svo",              # equivalent to wannier90's seedname
  "h5_filename": "svo.mlwf.h5", # output HDF5 filename 
  "shells": {        
    "atoms": [0, 1, 2],
    "sort": [0, 1, 2],
    "l": [2, 2, 2], 
    "dim": [1, 1, 1], 
    "SO": [0, 0, 0],
    "irep": [0, 0, 0] 
  }
}
coqui.mlwf_h5_from_wannier90_output(mf=svo_mf, params=w90_params)
