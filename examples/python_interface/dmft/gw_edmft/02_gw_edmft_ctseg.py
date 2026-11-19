"""
GW+EDMFT on SrVO3 with the CT-SEG impurity solver (TRIQS/ctseg).

CT-SEG is the recommended solver for a density-density impurity interaction with a retarded
(frequency-dependent) density-density Weiss field: it samples the retarded interaction exactly at
a cost that grows mildly with the number of orbitals, and it measures the impurity polarization
the GW+EDMFT screening needs. The GW starting point, svo.mbpt.h5, comes from 01_gw.py.

    mpirun -n N python 02_gw_edmft_ctseg.py
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
    "niter": 1,                     # number of GW+EDMFT cycles
    "gw_iter_per_loop": 1,          # GW updates per cycle
    "edmft_iter_per_loop": 1,       # EDMFT updates per cycle
    "screen_type": "gw_edmft",      # screening model: gw_edmft (impurity polarization) or rpa
    "div_treatment": "gygi_metal",  # q -> 0 divergence treatment
    "iter_alg": {"alg": "damping", "mixing": 0.15},

    "edmft": {
        "chkpt_h5": "svo_ctseg.dmft.h5",       # EDMFT checkpoint of this run
        # "iaft": {"wmax": 5, "eps": 1e-10},   # DLR mesh of the impurity problem (default: the GW one)

        # one block per impurity: the workflow's keys and the solver table
        "impurity": [{
            "init_imp_results": "dc",          # initial impurity self-energy/polarization: dc or zero
            "degenerate_blk": [[0, 1, 2, 3, 4, 5]],   # the t2g spin-orbitals are equivalent
            "chemical_potential": {"tolerance": 0.1},  # impurity chemical-potential search

            # CT-SEG: `type` names the backend; the other keys are the solver's
            "solver": {
                "type": "ctseg",
                "n_cycles": 2000000,           # total over all MPI ranks
                "n_warmup_cycles": 6000,
                "length_cycle": 60,
                "measure_pert_order": True,
                "perform_tail_fit": True,      # tail fit of the self-energy
                "fit_max_moment": 12,
                "fit_min_w": 1.8,              # Ha
                "fit_max_w": 4.5,              # Ha
                # the occupations-only solves of the chemical-potential search: as production
                # except these overrides (n_cycles defaults to 5 % of the production total)
                "mu_search": {"n_cycles": 100000},
            },
        }],
    },
}

# GW+EDMFT driver
coqui_dmft.run_gw_edmft(thc, E1, proj_info=proj_info, params=gw_edmft_params)
