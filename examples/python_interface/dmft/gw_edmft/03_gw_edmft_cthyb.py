"""
GW+EDMFT on SrVO3 with the CT-HYB impurity solver (TRIQS/cthyb, the 4.0.x fork) and the full
Kanamori interaction.

CT-HYB handles a general static impurity interaction (density-density, Kanamori, full) with
the retarded density-density Weiss field resummed exactly (Lang-Firsov), and measures the impurity
polarization for any of them. The price: the Dyson self-energy of CT-HYB is noisier than CT-SEG's
improved estimator, which makes the tail-fit window the one choice to watch, and the density
correlator of a non-density interaction is measured directly rather than from the occupation
kinks. With h_int = "density" this script runs the same GW+EDMFT as 02_gw_edmft_ctseg.py with
CT-HYB in place of CT-SEG.

    mpirun -n N python 03_gw_edmft_cthyb.py
"""
import triqs_modest as modest

import coqui
import coqui.dmft as coqui_dmft

qe_dir = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/out"
wan_h5 = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/mlwf/svo.mlwf.h5"

coqui_mpi = coqui.MpiHandler()
coqui.set_verbosity(coqui_mpi, output_level=2)

# Define correlated subspace via ModEST
obe = modest.make_one_body_elements_gw(wan_h5)
E1 = modest.make_embedding(obe.C_space)

# Update the rotation matrix
proj_info = coqui_dmft.get_proj_info(obe.P)

mf_params = {"prefix": "svo", "outdir": qe_dir, "nbnd": 40}
mf = coqui.make_mf(coqui_mpi, params=mf_params, mf_type='qe')

thc_params = {"thresh": 1e-5, "ecut": 60, "save": "thc.coulomb.h5"}
thc = coqui.make_thc_coulomb(mf=mf, params=thc_params)

# GW+EDMFT parameters
gw_edmft_params = {
    "outdir": "./",
    "prefix": "svo",
    "niter": 1,
    "gw_iter_per_loop": 1,
    "edmft_iter_per_loop": 1,
    "screen_type": "gw_edmft",      # screening model: gw_edmft (impurity polarization) or rpa
    "div_treatment": "gygi_metal",
    "iter_alg": {"alg": "damping", "mixing": 0.15},

    "edmft": {
        "chkpt_h5": "svo_cthyb.dmft.h5",

        "impurity": [{
            "init_imp_results": "dc",
            "degenerate_blk": [[0, 1, 2, 3, 4, 5]],
            "chemical_potential": {"tolerance": 0.1},

            # CT-HYB: the meshes and the Lang-Firsov kernel follow the impurity DLR mesh;
            # the tail-fit window of the Dyson self-energy is the numerically critical choice
            "solver": {
                "type": "cthyb",
                "h_int": "kanamori",           # "density", "kanamori" or "full"
                "n_cycles": 2000000,           # total over all MPI ranks
                "n_warmup_cycles": 6000,
                "length_cycle": 60,
                "perform_tail_fit": True,
                "fit_max_moment": 7,
                "fit_min_w": 1.8,              # Ha; where the Dyson Sigma is still above its noise
                "fit_max_w": 4.5,              # Ha
                "mu_search": {"n_cycles": 100000},
            },
        }],
    },
}

# GW+EDMFT driver
coqui_dmft.run_gw_edmft(thc, E1, proj_info=proj_info, params=gw_edmft_params)
