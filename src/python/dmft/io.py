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
from copy import deepcopy
import triqs.utility.mpi as mpi
from h5 import HDFArchive
import numpy as np
from coqui import app_log


def convert_gw_edmft_params(gw_edmft_params: dict):
    gw_edmft_group = deepcopy(gw_edmft_params)

    # Copy top-level gw_edmft settings
    gw_edmft_params = {}
    niter = gw_edmft_group.get('niter')
    gw_iter_per_loop = gw_edmft_group.get('gw_iter_per_loop', 1)
    edmft_iter_per_loop = gw_edmft_group.get('edmft_iter_per_loop', 1)
    if gw_edmft_group.get('wannier_file'):
        gw_edmft_params['wannier_file'] = gw_edmft_group.get('wannier_file')
    if niter is None:
        raise KeyError("Missing 'niter' parameter to specify the total number of GW+EDMFT cycles.")
    if not isinstance(niter, int) or niter <= 0:
        raise ValueError("'niter' must be a positive integer.")
    if not isinstance(gw_iter_per_loop, int) or gw_iter_per_loop < 0:
        raise ValueError("'gw_iter_per_loop' must be a non-negative integer.")
    if not isinstance(edmft_iter_per_loop, int) or edmft_iter_per_loop < 0:
        raise ValueError("'edmft_iter_per_loop' must be a non-negative integer.")
    if edmft_iter_per_loop == 0 and gw_iter_per_loop == 1:
        # Current GW+EDMFT SC logic does not upfold the GW solution and write new `embed` data 
        # if `gw_iter_per_loop` is 1 in order to save some memory in the checkpoint hdf5. 
        # Here, we swap `niter` and `gw_iter_per_loop` to make sure `embed` data is updated in the checkpoint.  
        gw_iter_per_loop = niter
        niter = 1
        if gw_iter_per_loop == 1:
            raise ValueError("Invalid combination of iteration parameters:\n" \
            "When fixing the EDMFT part of the workflow by setting `edmft_iter_per_loop` to 0, " \
            "`gw_iter_per_loop` or `niter` must be greater than 1.")
    
    gw_edmft_params['niter'] = niter
    gw_edmft_params['gw_iter_per_loop'] = gw_iter_per_loop
    gw_edmft_params['edmft_iter_per_loop'] = edmft_iter_per_loop
    
    screen_type = gw_edmft_group.get('screen_type', 'gw_edmft')
    div_treatment = gw_edmft_group.get('div_treatment', 'gygi')
    outdir = gw_edmft_group.get('outdir', './')
    prefix = gw_edmft_group.get('prefix', 'coqui')
    # restart = False implies the workflow will start from a fresh checkpoint where both GW and EDMFT parts start from scratch;
    # this is not available yet in the current implementation as the GW part is always assumed to restart from a previous checkpoint.
    restart = gw_edmft_group.get('restart', True)
    if restart is False:
        raise NotImplementedError(
            "Starting the GW+EDMFT workflow from scratch (restart=False) is not implemented yet. " \
            "Set restart=True (i.e. current default) to start from a previous checkpoint."
        )
    coqui_chkpt_h5 = outdir + "/" + prefix + ".mbpt.h5"
    if not os.path.exists(coqui_chkpt_h5):
        raise FileNotFoundError(
            f"The GW checkpoint {coqui_chkpt_h5} is missing. A valid GW checkpoint file is necessary to initialize the GW+EDMFT workflow."
        )

    iter_params = gw_edmft_group.pop('iter_alg', {"alg": "damping", "mixing": 0.3})
    if iter_params['alg'] != 'damping':
        raise NotImplementedError(
            f"Iterative algorithm '{iter_params['alg']}' is not supported in the GW+EDMFT workflow at the moment. "
        )
    gw_mixing = iter_params.pop('gw_mixing', iter_params.get('mixing', 0.3))
    edmft_mixing = iter_params.pop('edmft_mixing', iter_params.get('mixing', 0.3))
    edmft_mix_in_first_iter = iter_params.pop('edmft_mix_in_first_iter', True)
    gw_iter_params = iter_params.copy()
    gw_iter_params['mixing'] = gw_mixing
    edmft_iter_params = iter_params.copy()
    edmft_iter_params['mixing'] = edmft_mixing
    edmft_iter_params['mix_in_first_iter'] = edmft_mix_in_first_iter

    # GW parameters
    if gw_iter_per_loop > 0:
        gw_edmft_params['gw'] = { 
            'outdir': outdir, 
            'prefix': prefix,
            'restart': True, 
            'screen_type': screen_type,
            'niter': 1,
            'div_treatment': div_treatment,
            'iter_alg': gw_iter_params
        }

    # wloc parameters
    gw_edmft_params['wloc'] = {'outdir': outdir, 'prefix': prefix, 'screen_type': screen_type, 
                               'div_treatment': div_treatment, 'output_in_tau': True}
    # gloc parameters
    gw_edmft_params['gloc'] = {'outdir': outdir, 'prefix': prefix}
    # embed parameters
    gw_edmft_params['dmft_embed'] = {
        'outdir': outdir, 'prefix': prefix, 
        'corr_only': gw_edmft_group.get('corr_only', True), 
        # disable iterative solver for the embedding step. The iterative solve happens somewhere else. 
        'iter_alg': {'enable': False} 
    }

    # Convert 'edmft' section to 'impurity' (with nested 'impurity' -> 'solver')
    if 'edmft' not in gw_edmft_group:
        raise KeyError("Missing 'edmft' section in 'gw_edmft' parameters to specify the EDMFT part of the workflow.")
    edmft_section = gw_edmft_group.pop('edmft')
    
    gw_edmft_params['impurity'] = {
        'restart': restart, 
        'iter_alg': edmft_iter_params,
        'chkpt_h5': edmft_section.pop('chkpt_h5', outdir+"/"+prefix+".mbpt.h5")
    }
  
    # Convert 'impurity' (new) to 'solver' (old)
    if 'impurity' in edmft_section:
        solver_list = edmft_section.pop('impurity')
        if not isinstance(solver_list, list):
            solver_list = [solver_list]
    
        for solver_params in solver_list:
            if solver_params.get('degenerate_blk'):
                solver_params['degenerate_blk'] = [np.array(x) for x in solver_params['degenerate_blk']]
        
        gw_edmft_params['impurity']['solver'] = solver_list

    # Copy any remaining keys
    for key, value in edmft_section.items():
        gw_edmft_params['impurity'][key] = value

    return gw_edmft_params


# Need to be careful with non-ct-qmc solver
def _normalize_solver_params_list(solver_params_list, comm_size):
    """
    Scale Monte-Carlo cycle counts by MPI communicator size.
    """
    assert isinstance(solver_params_list, list), "solver_params_list must be a list of dictionaries."
    normalized_list = []
    for solver_params in solver_params_list:
        solver_params_norm = deepcopy(solver_params)

        mu_params = solver_params_norm.get('chemical_potential')
        if mu_params: 
            mu_params.setdefault('n_cycles', int(solver_params_norm['n_cycles']*0.05))
            mu_params['n_cycles'] = int(mu_params['n_cycles']/comm_size) 
            mu_params.setdefault('n_warmup_cycles', int(solver_params_norm['n_warmup_cycles']))
            mu_params.setdefault('length_cycle', int(solver_params_norm['length_cycle']))

        solver_params_norm['n_cycles'] = int(solver_params_norm['n_cycles']/comm_size)
        solver_params_norm['n_warmup_cycles'] = int(solver_params_norm['n_warmup_cycles'])
        solver_params_norm['length_cycle'] = int(solver_params_norm['length_cycle'])

        normalized_list.append(solver_params_norm)

    return normalized_list


def print_title_box(name, box_width=19):
    top_left = '╔'
    top_right = '╗'
    bottom_left = '╚'
    bottom_right = '╝'
    horizontal = '═'
    vertical = '║'

    title = f"{vertical}{name.center(box_width - 2)}{vertical}"
    top_border = f"{top_left}{horizontal * (box_width - 2)}{top_right}"
    bottom_border = f"{bottom_left}{horizontal * (box_width - 2)}{bottom_right}"

    app_log(1, "\n"+top_border)
    app_log(1, title)
    app_log(1, bottom_border+"\n")


def print_degenerate_blks(deg_blks, gf_struct):
    app_log(1, f"Degenerate blocks for impurity")
    app_log(1, "-------------------------------")
    for i, blks in enumerate(deg_blks):
        subset = [gf_struct[b] for b in blks]
        app_log(1, f"Shell {i}: {subset}\n")


def save_impurities(dmft_grp, *, solver_results=None, solver_inputs=None, impurity_index=-1, iteration=-1):
    """
    Save impurity solver results and/or inputs to an HDF5 group.

    Parameters
    ----------
    dmft_grp : h5py.Group
        The HDF5 group where the DMFT data will be stored.
    solver_results : list, optional
        A list of dictionaries containing solver results for each impurity.
    solver_inputs : list, optional
        A list of dictionaries containing solver inputs for each impurity.
    impurity_index : int, optional
        The index of the impurity to save. If -1, save all impurities. Default is -1.
    iteration : int, optional
        The DMFT iteration number. If -1, determine the iteration automatically. Default is -1.

    Raises
    ------
    ValueError
        If neither `solver_results` nor `solver_inputs` is provided.
        If `solver_results` and `solver_inputs` are provided but have different lengths.
        If `impurity_index` is out of range.

    Notes
    -----
    - If `iteration` is -1, the function attempts to determine the iteration number
      from the `dmft_grp` keys. If no iteration information is found, it defaults to 1.
    - The function creates a new group for the specified iteration and stores the
      impurity data under subgroups named `impurity_<index>`.
    - Backward compatibility is maintained for older keys like "final_iteration".
    """
    if solver_results is None and solver_inputs is None:
        raise ValueError("Either solver_results or solver_inputs must be provided.")

    if iteration == -1:
        if "final_iter" in dmft_grp.keys():
            iteration = dmft_grp["final_iter"] + 1
        elif "final_iteration" in dmft_grp.keys():
            # backward compatibility
            # TODO remove it
            iteration = dmft_grp["final_iteration"]
        else:
            iteration = 1

    if solver_results is not None and solver_inputs is not None:
        if len(solver_results) != len(solver_inputs):
            raise ValueError("solver_results and solver_inputs must have the same length.")
        num_impurities = len(solver_results)
    elif solver_results is not None:
        num_impurities = len(solver_results)
    else:
        num_impurities = len(solver_inputs)

    if impurity_index == -1:
        impurity_list = np.arange(num_impurities)
    else:
        if impurity_index < 0 or impurity_index >= num_impurities:
            raise ValueError(f"impurity_index {impurity_index} is out of range [-1, {num_impurities}).")
        impurity_list = [impurity_index]

    if f"iter{iteration}" not in dmft_grp.keys():
        dmft_grp.create_group(f"iter{iteration}")
    iter_grp = dmft_grp[f"iter{iteration}"]

    for imp_i in impurity_list:
        if f"impurity_{imp_i}" not in iter_grp.keys():
            iter_grp.create_group(f"impurity_{imp_i}")
        imp_grp = iter_grp[f"impurity_{imp_i}"]
        if solver_results is not None:
            _write_impurity_results(imp_grp, solver_results[imp_i])
        if solver_inputs is not None:
            _write_impurity_inputs(imp_grp, solver_inputs[imp_i])

    # FIXME this is beyond the save_impurities responsibility since this will affect the restart behavior of the DMFT SCF loop. 
    dmft_grp["final_iter"] = iteration


def _write_impurity_results(h5_grp, impurity_results):
    """
    Write impurity solver results to an HDF5 group.

    Parameters
    ----------
    h5_grp : h5py.Group
        The HDF5 group where the impurity results will be stored.
    impurity_results : dict
        A dictionary containing the impurity solver results. It should include
        both optional and mandatory keys.

    Notes
    -----
    - The function creates a subgroup named "results" within `h5_grp` if it does not exist.
    - Optional keys (e.g., raw numpy arrays on IR mesh) are stored if present in `impurity_results`.
    - Mandatory keys (e.g., TRIQS objects) are always expected to be present in `impurity_results`.
    """
    if "results" not in h5_grp.keys():
        h5_grp.create_group("results")
    res_grp = h5_grp["results"]

    # ----- optional keys (raw numpy arrays on IR mesh) -----
    # "_data" appendices imply raw numpy arrays on IR mesh
    optional_keys = [
        'gf_struct', 'mu_imp', 'convergence', 'density',
        'G_iw_data', 'Sigma_iw_data',
        'Pi_iw_data', 'W_iw_data',
        'Sigma_infty_dc', 'Sigma_iw_dc_data', 'Pi_iw_dc_data'
    ]
    for key in optional_keys:
        if key in impurity_results:
            res_grp[key] = impurity_results[key]

    # ----- mandatory TRIQS objects directly from the solver -----
    mandatory_keys = [
        'Sigma_infty', 'G_iw', 'Sigma_iw',
        'Pi_iw', 'W_iw', 'Chi_iw',
        'orbital_occupations', 'average_order', 'average_sign'
    ]
    for key in mandatory_keys:
        res_grp[key] = impurity_results[key]


def _write_impurity_inputs(h5_grp, impurity_inputs):
    if "inputs" not in h5_grp.keys():
        h5_grp.create_group("inputs")
    inp_grp = h5_grp["inputs"]
    inp_grp['gf_struct'] = impurity_inputs['gf_struct']
    inp_grp['Gloc_t'] = impurity_inputs['Gloc_t']
    inp_grp['Wloc_t'] = impurity_inputs['Wloc_t']
    inp_grp['Vloc'] = impurity_inputs['Vloc']
    inp_grp['u_weiss_iw'] = impurity_inputs['u_weiss_iw']
    inp_grp['g_weiss_iw'] = impurity_inputs['g_weiss_iw']
    inp_grp['delta_iw'] = impurity_inputs['delta_iw']
    inp_grp['h0'] = impurity_inputs['h0']


def read_impurity_chkpt(h5_filename, iteration=-1, *, read="both", impurity_indices=None):
    """
    Read impurity checkpoint file from a DMFT run.

    Parameters
    ----------
    h5_filename : str
        Path to the checkpoint file.
    iteration : int, optional
        Iteration index to read. If -1, read the final iteration.
    read : {"results", "inputs", "both"}, optional
        Which part of the checkpoint to read.
    impurity_indices : list of int, optional
        Specific impurity indices to read. If None, read all.

    Returns
    -------
    Depending on `read`:
        - "results" → list of solver_results
        - "inputs"  → list of solver_inputs
        - "both"    → (solver_results, solver_inputs)
    """
    app_log(2, f"Reading impurity checkpoint file: {h5_filename}")
    assert read in {"results", "inputs", "both"}, \
        "Argument 'read' must be one of {'results', 'inputs', 'both'}."
    solver_results, solver_inputs = [], []
    res_tmp, inp_tmp = {}, {}
    num_impurities = -1

    if mpi.is_master_node():
        with HDFArchive(h5_filename, 'r') as ar:
            if iteration == -1:
                if "final_iter" in ar["dmft"].keys():
                    iteration = ar["dmft"]["final_iter"]
                else:
                    # backward compatibility
                    # TODO remove it
                    iteration = ar["dmft"]["final_iteration"]

            iter_grp = ar[f"dmft/iter{iteration}"]
            num_impurities = len([k for k in iter_grp.keys() if k.startswith("impurity_")])
            mpi.bcast(num_impurities)

            # Determine which impurities to read
            if impurity_indices is None:
                impurity_indices = np.arange(num_impurities)
            else:
                assert isinstance(impurity_indices, list), "impurity_indices must be a list of integers."
            app_log(2, f"impurity list = {impurity_indices}\n")

            for i in impurity_indices:
                if read in {"results", "both"}:
                    res_grp = iter_grp[f"impurity_{i}/results"]
                    res_tmp = read_all_keys(res_grp)
                    mpi.bcast(res_tmp)
                    solver_results.append(res_tmp)

                if read in {"inputs", "both"}:
                    inp_grp = iter_grp[f"impurity_{i}/inputs"]
                    inp_tmp = read_all_keys(inp_grp)
                    mpi.bcast(inp_tmp)
                    solver_inputs.append(inp_tmp)
    else:
        num_impurities = mpi.bcast(num_impurities)

        # Determine which impurities to read
        if impurity_indices is None:
            impurity_indices = np.arange(num_impurities)
        else:
            assert isinstance(impurity_indices, list), "impurity_indices must be a list of integers."

        for _ in impurity_indices:
            if read in {"results", "both"}:
                res_tmp = mpi.bcast(res_tmp)
                solver_results.append(res_tmp)
            if read in {"inputs", "both"}:
                inp_tmp = mpi.bcast(inp_tmp)
                solver_inputs.append(inp_tmp)

    mpi.barrier()

    if read == "results":
        return solver_results
    elif read == "inputs":
        return solver_inputs
    else:
        return solver_results, solver_inputs


def update_impurity_results_from_chkpt(solver_results, h5_filename, iteration=-1):
    """
    Update impurity solver results from a checkpoint file.

    Parameters
    ----------
    solver_results : list of dict
        A list of dictionaries containing the current solver results for each impurity.
        This will be updated with the data read from the checkpoint file.
    h5_filename : str
        Path to the HDF5 checkpoint file.
    iteration : int, optional
        The DMFT iteration to read from the checkpoint file. If -1, the function
        attempts to determine the most recent iteration automatically. Default is -1.

    Returns
    -------
    int
        The iteration number from which the results were updated.

    Raises
    ------
    RuntimeError
        If no valid DMFT iteration is found in the checkpoint file.

    Notes
    -----
    - If the specified iteration does not exist, the function automatically falls back
      to the previous iteration.
    - The function ensures that the checkpoint file is complete and not corrupted.
    """
    res_tmp = {}
    if mpi.is_master_node():
        with HDFArchive(h5_filename, 'r') as ar:
            if iteration == -1:
                if "final_iter" in ar["dmft"].keys():
                    iteration = ar["dmft"]["final_iter"]
                else:
                    # backward compatibility
                    iteration = ar["dmft"]["final_iteration"]
            try:
                _ = ar[f'dmft/iter{iteration}/impurity_0/results']
            except KeyError:
                # automatically go to the previous iteration if the current one does not exist
                app_log(1, 
                    f"[Warning] Path 'dmft/iter{iteration}/impurity_0/results' "
                    f"not found in checkpoint file '{h5_filename}'.\n"
                    f"→ Falling back to the previous iteration: iter{iteration-1}.\n"
                )
                iteration -= 1
            if iteration < 0:
                raise RuntimeError(
                    f"[Error] No valid DMFT iteration found in '{h5_filename}'.\n"
                    f"Expected path 'dmft/iter*/impurity_0/results' but none exist.\n"
                    f"Ensure the checkpoint file is complete and not corrupted."
                )

            app_log(1, 
                f"Loading impurity results from checkpoint '{h5_filename}' "
                f"at DMFT iteration {iteration}.\n"
            )
            for imp_index, res in enumerate(solver_results):
                imp_grp = ar[f'dmft/iter{iteration}/impurity_{imp_index}/results']
                res_tmp.update(read_all_keys(imp_grp))
                res_tmp = mpi.bcast(res_tmp)
                res.update(res_tmp)
    else:
        for imp_index, res in enumerate(solver_results):
            res_tmp = mpi.bcast(res_tmp)
            res.update(res_tmp)

    iteration = mpi.bcast(iteration)

    return iteration


def read_all_keys(h5_grp):
    output = {}
    for key in h5_grp.keys():
        output[key] = h5_grp[key]
    return output

