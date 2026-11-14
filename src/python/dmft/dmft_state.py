"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==========================================================================
"""

import os
import numpy as np

import triqs.utility.mpi as mpi
from h5 import HDFArchive

from coqui.utils.imag_axes_ft import IAFT
from coqui.dmft.weiss import embed_impurities
import coqui.dmft.io as dmft_io
from coqui import app_log

"""
Data structure for DMFT state 
"""

# TODO use coqui MPIHandler? or triqs mpi wrapper?

def _mix_into(curr, prev, mixing_factor=0.7):
    """In-place: handles ndarray OR list-of-ndarrays; falls back to object algebra."""
    # list of arrays
    if isinstance(curr, list) and isinstance(prev, list):
        if len(curr) != len(prev):
            raise ValueError(f"List length mismatch: {len(curr)} vs {len(prev)}")
        for i, (c, p) in enumerate(zip(curr, prev)):
            if isinstance(c, np.ndarray) and isinstance(p, np.ndarray):
                if c.shape != p.shape:
                    raise ValueError(f"Shape mismatch at list[{i}]: {c.shape} vs {p.shape}")
                c *= mixing_factor
                c += (1 - mixing_factor) * p
            else:
                # fall back to reassignment (TRIQS objects or scalars)
                curr[i] = mixing_factor * c + (1 - mixing_factor) * p
        return

    # plain ndarray
    if isinstance(curr, np.ndarray) and isinstance(prev, np.ndarray):
        if curr.shape != prev.shape:
            raise ValueError(f"Shape mismatch: {curr.shape} vs {prev.shape}")
        curr *= mixing_factor
        curr += (1 - mixing_factor) * prev
        return

    # fallback: rely on object algebra (e.g., TRIQS Gf/BlockGf support)
    curr *= mixing_factor
    curr += (1 - mixing_factor) * prev
    return


def save_impurity_results(impurity_results, dmft_chkpt, iteration=-1, impurity_index=-1, *,
                          chkpt_impurity_index=None, solver_data=None, mark_final_iter=False):
    """
    Write impurity solver results to ``dmft/iter{iteration}/impurity_{i}/results`` of the solver
    checkpoint (master only; collective through the closing barrier).

    Parameters
    ----------
    impurity_results : list of dict
        Solver results, one entry per impurity.
    dmft_chkpt : str
        Solver checkpoint file.
    iteration : int, optional
        DMFT iteration to write; ``-1`` writes ``final_iter + 1``. Default ``-1``.
    impurity_index : int, optional
        Entry of ``impurity_results`` to write; ``-1`` writes every entry. Default ``-1``.
    chkpt_impurity_index : int, optional
        Impurity number ``i`` of the checkpoint group the entry ``impurity_index`` is written to,
        for a ``impurity_results`` that holds only a subset of the impurities (as in
        ``solve_impurities_from_chkpt`` with ``imp_indices``). Default ``None``: the group number
        is ``impurity_index`` itself.
    solver_data : list of dict, optional
        Backend-native objects, one entry per impurity like ``impurity_results``, written to
        ``impurity_{i}/solver_results`` (an empty entry removes the group, ``None`` skips it).
    mark_final_iter : bool, optional
        Also set the restart marker ``dmft/final_iter`` to the written iteration. Only the
        GW+EDMFT loop (``DMFTState``) passes ``True``; a tool rewriting a stored iteration,
        such as ``solve_impurities_from_chkpt``, leaves the marker alone. Default ``False``.
    """
    if mpi.is_master_node():
        with HDFArchive(dmft_chkpt, 'a') as ar:
            if "dmft" not in ar.keys():
                ar.create_group("dmft")
            written = dmft_io.save_impurities(
                ar["dmft"], impurity_results = impurity_results, solver_data = solver_data,
                impurity_index = impurity_index, iteration = iteration,
                chkpt_impurity_index = chkpt_impurity_index
            )
            if mark_final_iter:
                ar["dmft"]["final_iter"] = written
    mpi.barrier()


def save_impurity_inputs(impurity_inputs, dmft_chkpt, iteration=-1, impurity_index=-1, *, iaft=None,
                         mark_final_iter=False):
    """
    Write impurity inputs to ``dmft/iter{iteration}/impurity_{i}/inputs`` of the solver checkpoint
    (master only; collective through the closing barrier); ``iaft`` is saved alongside when given.
    ``mark_final_iter`` as in ``save_impurity_results``.
    """
    if mpi.is_master_node():
        with HDFArchive(dmft_chkpt, 'a') as ar:
            if iaft:
                iaft.save(ar)
            if "dmft" not in ar.keys():
                ar.create_group("dmft")
            written = dmft_io.save_impurities(
                ar["dmft"], impurity_inputs = impurity_inputs,
                impurity_index = impurity_index, iteration = iteration
            )
            if mark_final_iter:
                ar["dmft"]["final_iter"] = written
    mpi.barrier()


class DMFTState(object):
    """
    The state of the GW+EDMFT loop. 

    Per impurity, 
    1) ``impurity_inputs[i]`` holds what the solver is handed (the local Green's
    function and interactions, the Weiss fields, ``h0``, ``delta_iw``, ``degenerate_blk``,
    ``occupations_sab``), 
    
    2) ``impurity_results[i]`` what the loop keeps of the solve (the arrays of
    ``ImpurityResult.solver_agnostic_results()``, ``mu_imp``, the double-counting terms,
    ``convergence``) 
    
    3) `solver_data[i]`` the backend-native objects of the last solve. 
    """
    def __init__(self, embedding_1e, embedding_2e,
                 dlr_wmax, dlr_eps, iaft_crystal,
                 spin_average=False, screen_type='gw_edmft',
                 verbal=False):
        self.iteration = 0
        self.spin_average = spin_average
        self.embedding = {'1e': embedding_1e, '2e': embedding_2e}
        self.iaft = iaft_crystal
        # DLR mesh controls of the impurity problems (the backends build their own TRIQS meshes)
        self.dlr_wmax = dlr_wmax
        self.dlr_eps = dlr_eps

        self.local_sigma_w = None
        self.local_sigma_infty = None
        self.local_pi_w = None
        assert screen_type in {"rpa", "gw_edmft", "gw_edmft_density"}, (
            f"DMFTState: incompatible screen_type = {screen_type} detected. "
            f"Available choices are \"rpa\", \"gw_edmft\" and \"gw_edmft_density\"."
        )

        self.impurity_inputs = [ {
            'gf_struct': self.embedding['1e'].imp_block_structure[imp],
            'Gloc_t': None,
            'Wloc_t': None,
            'Vloc': None,
            'u_weiss_iw': None,
            'g_weiss_iw': None,
            'delta_iw': None,
            'h0': None,
            'screen_type': screen_type
        } for imp in range(self.embedding['1e'].n_impurities) ]

        self.impurity_results = [ {
            'gf_struct': self.embedding['1e'].imp_block_structure[imp],
            'mu_imp': 0.0,
            'Sigma_infty': None,
            'Sigma_iw_data': None, # numpy array on IR mesh
            'Pi_iw_data': None,    # numpy array on IR mesh
            'W_iw_data': None,     # numpy array on IR mesh
            'Sigma_infty_dc': None,
            'Sigma_iw_dc_data': None,
            'Pi_iw_dc_data': None
        } for imp in range(self.embedding['1e'].n_impurities) ]
        # backend-native objects of the last solve, written to solver_results/ (never loaded back)
        self.solver_data = [ {} for _ in range(self.embedding['1e'].n_impurities) ]

        #if verbal:
        #    mpi.report(self)
        #    mpi.barrier()

    def __str__(self):
        return ("DMFT State                        \n" \
                "----------------------------------\n" )

    def __eq__(self, other):
        if not isinstance(other, DMFTState):
            return NotImplemented

        return (
                self.embedding['1e'] == other.embedding['1e'] and
                self.embedding['2e'] == other.embedding['2e']
        )

    # TODO make_dmft_state without an existing coqui_h5. Directly take the IR parameters as input. 
    @classmethod
    def make_dmft_state(cls, coqui_h5, embedding_1e, embedding_2e,
                        wmax_imp=None, eps_imp=None, spin_average=False,
                        screen_type='gw_edmft', verbal=False):
        """
        A fresh state (iteration 0, nothing solved) on the IAFT of the CoQuí checkpoint
        ``coqui_h5``; ``wmax_imp`` / ``eps_imp`` set the DLR controls of the impurity problems,
        defaulting to the IAFT's. Reads only the IAFT parameters, no impurity data.
        """
        iaft = IAFT.from_coqui_chkpt(coqui_h5, verbose=verbal)
        dlr_wmax = iaft.wmax if wmax_imp is None else wmax_imp
        dlr_eps = iaft.eps if eps_imp is None else eps_imp
        return cls(embedding_1e, embedding_2e, dlr_wmax, dlr_eps,
                   iaft, spin_average, screen_type, verbal)


    def load(self, dmft_chkpt):
        """
        Resume from the DMFT checkpoint ``dmft_chkpt``: 
        1. load ``impurity_results`` of every impurity from its last iteration (``dmft/final_iter``), 
        2. set ``iteration`` to the next one and 
        3. rebuild the embedded local quantities. 
        
        ``impurity_inputs`` and ``solver_data`` are not loaded. 
        
        A missing file leaves the state fresh.
        """
        if not os.path.isfile(dmft_chkpt):
            app_log(2, "No solver checkpoint file found. Will skip loading impurity results.\n")
            return
            

        # TODO for "each" impurity, load the previous results if existing, otherwise initialize to empty
        self.iteration = dmft_io.update_impurity_results_from_chkpt(self.impurity_results, dmft_chkpt) + 1

        # check dimensions of loaded impurity results
        nw_b_half = self.iaft.nw_b//2 if self.iaft.nw_b%2==0 else self.iaft.nw_b//2 + 1
        for imp_idx, res in enumerate(self.impurity_results):
            for blk_idx, (blk_name, blk_dim) in enumerate(res['gf_struct']):
                assert (res['Sigma_infty'][blk_idx].shape[0] == blk_dim and
                        res['Sigma_infty'][blk_idx].shape[1] == blk_dim), (
                    "Incompatible block dimension for the loaded impurity Sigma"
                )
                assert res['Sigma_iw_data'][blk_idx].shape[0] == self.iaft.nw_f, (
                    "Incompatible fermionic Matsubara mesh for the loaded impurity Sigma"
                )
                assert (res['Sigma_iw_data'][blk_idx].shape[1] == blk_dim and
                        res['Sigma_iw_data'][blk_idx].shape[2] == blk_dim), (
                    "Incompatible block dimension for the loaded impurity Sigma"
                )
            gf_struct_2e = self.embedding['2e'].imp_block_structure[imp_idx]
            for blk_idx, (blk_name, blk_dim) in enumerate(gf_struct_2e):
                assert res['Pi_iw_data'][blk_idx].shape[0] == nw_b_half, (
                    "Incompatible bosonic Matsubara mesh for the loaded impurity Pi"
                )
                assert (res['Pi_iw_data'][blk_idx].shape[1] == blk_dim and
                        res['Pi_iw_data'][blk_idx].shape[2] == blk_dim), (
                    "Incompatible block dimension for the loaded impurity Pi"
                )

        self.local_sigma_w, self.local_sigma_infty, self.local_pi_w = embed_impurities(
            self.embedding['1e'], self.embedding['2e'], self.impurity_results, self.spin_average
        )
        self.iaft.check_leakage(self.local_sigma_w["imp"], stats='f', name='Sigma_imp', w_input=True)
        self.iaft.check_leakage(self.local_sigma_w["dc"], stats='f', name='Sigma_dc', w_input=True)
        self.iaft.check_leakage_phsym(self.local_pi_w["imp"], stats='b', name='Pi_imp', w_input=True)
        self.iaft.check_leakage_phsym(self.local_pi_w["dc"], stats='b', name='Pi_dc', w_input=True)
        app_log(1, "")


    # The loop's writers: every one of them advances the restart marker dmft/final_iter to
    # the current iteration (mark_final_iter), which the module-level functions do not.
    def save_impurity_inputs(self, dmft_chkpt, impurity_index):
        """
        Write ``impurity_inputs[impurity_index]`` to ``dmft/iter{iteration}/impurity_{i}/inputs``
        of ``dmft_chkpt``. 
        """
        save_impurity_inputs(
            self.impurity_inputs, dmft_chkpt, self.iteration, impurity_index, iaft=self.iaft,
            mark_final_iter=True
        )

    def save_impurity_results(self, dmft_chkpt, impurity_index):
        """
        Write ``impurity_results[impurity_index]`` to ``.../impurity_{i}/results`` and
        ``solver_data[impurity_index]`` to ``.../impurity_{i}/solver_results`` of
        ``dmft/iter{iteration}`` of ``dmft_chkpt``. 
        """
        save_impurity_results(
            self.impurity_results, dmft_chkpt, self.iteration, impurity_index,
            solver_data=self.solver_data, mark_final_iter=True
        )

    def embed_impurity_results(self):
        # pi impurities are embedded even if screen_type == "rpa".
        # This is okay since coqui.run_gw() and coqui.downfold_wloc()
        # automatically omit pi_imp if screen_type="rpa"
        local_sigma_w, local_sigma_infty, local_pi_w = embed_impurities(
            self.embedding['1e'], self.embedding['2e'], self.impurity_results, self.spin_average
        )

        # check convergence:
        if self.local_sigma_w and self.local_sigma_infty and self.local_pi_w:
            max_diff_sigma_infty = np.max(
                np.abs(local_sigma_infty["imp"] - local_sigma_infty["dc"]
                   - self.local_sigma_infty["imp"] + self.local_sigma_infty["dc"])
            )
            # check convergence on the imaginary-time axis
            max_diff_sigma_t = np.max(np.abs(
                self.iaft.w_to_tau(
                    local_sigma_w["imp"] - local_sigma_w["dc"] 
                        - self.local_sigma_w["imp"] + self.local_sigma_w["dc"], 
                    stats='f'
                )
            ))
            max_diff_pi_t = np.max(np.abs(
                self.iaft.w_to_tau_phsym(
                    local_pi_w["imp"] - local_pi_w["dc"]
                        - self.local_pi_w["imp"] + self.local_pi_w["dc"],
                    stats='b'
                )
            ))
            app_log(1, f"Max difference in embedded impurity results: \n"
                       f"|Delta Sigma_t|     = {max_diff_sigma_t}, \n"
                       f"|Delta Sigma_infty| = {max_diff_sigma_infty}, \n"
                       f"|Delta Pi_t|        = {max_diff_pi_t}\n")

        self.local_sigma_w     = local_sigma_w
        self.local_sigma_infty = local_sigma_infty
        self.local_pi_w        = local_pi_w

        self.iaft.check_leakage(self.local_sigma_w["imp"], stats='f', name='Sigma_imp', w_input=True)
        self.iaft.check_leakage(self.local_sigma_w["dc"], stats='f', name='Sigma_dc', w_input=True)
        self.iaft.check_leakage_phsym(self.local_pi_w["imp"], stats='b', name='Pi_imp', w_input=True)
        self.iaft.check_leakage_phsym(self.local_pi_w["dc"], stats='b', name='Pi_dc', w_input=True)
        app_log(1, "")


    def damp_impurity_results(self, dmft_chkpt, mixing=0.7, *, impurity_indices=None, 
                              mix_in_first_iter=False):
        """
        Damp ``impurity_results`` of ``impurity_indices`` (default all) in place:
        ``mixing * current + (1 - mixing) * previous``, where the previous results are read
        from ``dmft/iter{iteration - 1}`` of ``dmft_chkpt``. In iteration 0 there is no previous
        iteration: the impurity terms are mixed with their double-counting terms when
        ``mix_in_first_iter`` is set, else left untouched. 
        """
        assert self.iteration >= 0, "damp_impurity_results: Current iteration must be a non-negative integer."

        if impurity_indices is not None:
            assert isinstance(impurity_indices, list), "impurity_indices should be a list of integers."
        else:
            impurity_indices = np.arange(len(self.impurity_results))

        if self.iteration == 0 and not mix_in_first_iter: # no damping in the first iteration
            app_log(2, "Skipping damping the impurity results in the first iteration.\n")
            return

        if self.iteration == 0 and mix_in_first_iter:
            # first iteration: mix impurity results with the dc terms in the first iteration 
            app_log(2, f"Mixing impurity results with DC terms for impurities {impurity_indices}\n")
            for idx, imp_idx in enumerate(impurity_indices):
                res = self.impurity_results[imp_idx]
                imp_key = ['Sigma_infty', 'Sigma_iw_data', 'Pi_iw_data']
                dc_key  = ['Sigma_infty_dc', 'Sigma_iw_dc_data', 'Pi_iw_dc_data']
                for (imp, dc) in zip(imp_key, dc_key):
                    if imp not in res:
                        raise KeyError(f"Missing key '{imp}' for impurity {imp_idx} during damping.")
                    if dc not in res:
                        raise KeyError(f"Missing key '{dc}' for impurity {imp_idx} during damping.")
                    _mix_into(res[imp], res[dc], mixing)
            return
        
        app_log(2, f"Mixing impurity results with the previous iteration for impurities {impurity_indices}\n")
        impurity_results_prev = dmft_io.read_impurity_chkpt(
            dmft_chkpt, self.iteration-1, read="results", impurity_indices=impurity_indices
        )

        keys_to_damp = ['Sigma_infty', 'Sigma_infty_dc',
                        'Sigma_iw_data', 'Sigma_iw_dc_data',
                        'Pi_iw_data', 'Pi_iw_dc_data']
        for idx, imp_idx in enumerate(impurity_indices):
            res = self.impurity_results[imp_idx]
            res_prev = impurity_results_prev[idx]
            for key in keys_to_damp:
                if key not in res or key not in res_prev:
                    raise KeyError(f"Missing key '{key}' for impurity {imp_idx} during damping.")
                _mix_into(res[key], res_prev[key], mixing)
