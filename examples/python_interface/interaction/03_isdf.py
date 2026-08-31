"""

Example: Interpolative separable density fitting from QE mean-field object

This example demonstrates how to perform interpolative separable density fitting (ISDF)
for the pair densities and dump the results to hdf5.
"""

from mpi4py import MPI
import coqui

qe_dir = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/out"

mpi = coqui.MpiHandler()
coqui.set_verbosity(mpi, output_level=1)

# --- Build Mf object from QE results
mf_params = {
    "prefix": "svo",      # prefix of the QE scf/nscf
    "outdir": qe_dir      # outdir of the QE scf/nscf
}
svo_mf = coqui.make_mf(mpi, params=mf_params, mf_type="qe")

# --- ISDF for pair densities and dump to hdf5
isdf_params = {
    "ecut": 40,
    "thresh": 1e-4,
    "save": "svo_isdf.h5",
    "write_zeta_on_fft_mesh": True
}
coqui.run_isdf(mf=svo_mf, params=isdf_params)
