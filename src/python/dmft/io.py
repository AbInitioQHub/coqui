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


# Keys of an impurity block that are agnostic to the solver type; they stay at the top level of
# the block, every other key is the solver's and goes under its 'solver' table.
SOLVER_AGNOSTIC_KEYS = frozenset({
  'solver', 'init_imp_results', 'set_sigma_infty_to_dc', 'degenerate_blk',
  'degenerate_blk_thresh', 'causal_projection', 'chemical_potential'})
# The same for the 'chemical_potential' table: every other key there is a solver-specific
# parameter of the search and goes under 'solver.mu_search'.
CHEMICAL_POTENTIAL_AGNOSTIC_KEYS = frozenset({'tolerance'})


def convert_legacy_impurity_block(imp_block, index):
  """
  Convert an impurity block in a legacy layout to the current one for backward compatibility.

  Legacy layout (flat CT-SEG block; the search's solver keys inside ``chemical_potential``)::

      {
        "init_imp_results": "dc",
        "degenerate_blk": [[0, 1]],
        "n_cycles": 2000000,
        "n_warmup_cycles": 6000,
        "length_cycle": 60,
        "screen_j": True,
        "chemical_potential": {"tolerance": 0.1,
                               "n_cycles": 100000}
      }

  Current layout::

      {
        "init_imp_results": "dc",
        "degenerate_blk": [[0, 1]],
        "solver": {
          "type": "ctseg",
          "n_cycles": 2000000,
          "n_warmup_cycles": 6000,
          "length_cycle": 60,
          "screen_j": True,
          "mu_search": {
            "n_cycles": 100000}
          },
        "chemical_potential": {"tolerance": 0.1}
      }

  The current layout follows three rules: 
  1) only solver-agnostic parameters (``SOLVER_AGNOSTIC_KEYS``) stay at the top level of the block; 
  2) every solver-specific parameter goes under the ``solver`` table, whose ``type`` names the backend; 
  3) solver-specific parameters of the chemical-potential search go under the ``mu_search`` table 
     inside ``solver``, so that ``chemical_potential`` holds solver-agnostic parameters only
     (``CHEMICAL_POTENTIAL_AGNOSTIC_KEYS``, i.e. ``tolerance``).

  The legacy layout is converted, always assuming the solver type is CT-SEG. 
  """
  block = dict(imp_block)
  # solver-specific keys at the top level of the block: the legacy flat layout, or a mistake
  solver_keys = [key for key in block if key not in SOLVER_AGNOSTIC_KEYS]
  if 'solver' not in block:
    moved = {key: block.pop(key) for key in solver_keys}
    block['solver'] = {'type': 'ctseg', **moved}
    app_log(1, f" [WARNING] impurity block {index} has no 'solver' table (legacy layout):\n"
               f"  treating {sorted(moved)} as CT-SEG solver parameters. Move them under the\n"
               f"  impurity's 'solver' table with type = \"ctseg\".\n")
  elif solver_keys:
    raise KeyError(f"io::convert_legacy_impurity_block: impurity block {index} has key(s) {sorted(solver_keys)} "
                   f"beside its 'solver' table; solver-agnostic keys are "
                   f"{sorted(SOLVER_AGNOSTIC_KEYS)}, solver parameters go under 'solver'")
  if not isinstance(block['solver'], dict) or 'type' not in block['solver']:
    raise KeyError(f"io::convert_legacy_impurity_block: impurity block {index}: the 'solver' table needs a "
                   f"'type' (e.g. \"ctseg\")")
  mu_block = block.get('chemical_potential')
  if isinstance(mu_block, dict):
    # solver-specific keys inside the table: the legacy layout of the search parameters
    search = {key: value for key, value in mu_block.items()
              if key not in CHEMICAL_POTENTIAL_AGNOSTIC_KEYS}
    if search:
      if 'mu_search' in block['solver']:
        raise KeyError(f"io::convert_legacy_impurity_block: impurity block {index} has solver parameters of "
                       f"the chemical-potential search both in 'chemical_potential' "
                       f"({sorted(search)}, legacy layout) and in 'solver.mu_search'; keep "
                       f"'solver.mu_search' only")
      block['chemical_potential'] = {key: value for key, value in mu_block.items()
                                     if key in CHEMICAL_POTENTIAL_AGNOSTIC_KEYS}
      block['solver'] = dict(block['solver'], mu_search=search)
      app_log(1, f" [WARNING] impurity block {index}: 'chemical_potential' carries solver "
                 f"parameters\n  {sorted(search)} (legacy layout); treating them as the "
                 f"'solver.mu_search' table.\n  'chemical_potential' holds 'tolerance' only; "
                 f"move the others under 'solver.mu_search'.\n")
  return block


def convert_gw_edmft_params(gw_edmft_params: dict):
    """
    Parse the input parameters of the GW+EDMFT workflow and organize them into 
    parameters of each stage.

    Input example::

        gw_edmft_params ={
          "niter": 2,
          "gw_iter_per_loop": 1,
          "edmft_iter_per_loop": 1,
          "outdir": "./",
          "prefix": "svo",
          "screen_type": "gw_edmft",
          "div_treatment": "gygi",
          "wannier_file": "svo.mlwf.h5",
          "iter_alg": {"alg": "damping", "mixing": 0.4, "edmft_mixing": 0.2},
          "edmft": {
            "chkpt_h5": "./svo.dmft.h5",
            "impurity": {
              "init_imp_results": "dc",
              "solver": {
                "type": "ctseg", "n_cycles": 2000000, ...
              }
            }
          }
        }

    Output structure, for that input::

        {
          "niter": 2,
          "gw_iter_per_loop": 1,
          "edmft_iter_per_loop": 1,
          "wannier_file": "svo.mlwf.h5",
          
          # The GW stage
          "gw": {"outdir": "./", "prefix": "svo", "restart": True, "niter": 1,
                 "screen_type": "gw_edmft", "div_treatment": "gygi",
                 "iter_alg": {"alg": "damping", "mixing": 0.4}},
          
          # The local W stage
          "wloc": {"outdir": "./", "prefix": "svo", "screen_type": "gw_edmft",
                   "div_treatment": "gygi", "output_in_tau": True},
          
          # The local G stage
          "gloc": {"outdir": "./", "prefix": "svo"},
          
          # The embedding (upfolding) stage
          "dmft_embed": {"outdir": "./", "prefix": "svo", "corr_only": True,
                         "iter_alg": {"enable": False}},
          
          # The EDMFT stage
          "edmft": {
            "restart": True,
            "chkpt_h5": "./svo.dmft.h5",
            "iter_alg": {"alg": "damping", "mixing": 0.2, "mix_in_first_iter": True},
            "impurity": [
              {
                "init_imp_results": "dc",
                "solver": {"type": "ctseg", "n_cycles": 2000000, ...}
              }
            ]
          }
        }

    - ``gw`` is absent when ``gw_iter_per_loop == 0``. A top-level ``iaft`` lands in ``edmft``
      (it is the impurity mesh); ``edmft.impurity`` is always a list, one block per impurity;
      every other key of ``edmft`` and of the top level is copied through unchanged.
    
    - With ``edmft_iter_per_loop == 0`` and ``gw_iter_per_loop == 1``, ``niter`` and
      ``gw_iter_per_loop`` are swapped so that the GW-only run still writes the ``embed`` data.
    
    - ``restart = False`` is not implemented and the GW checkpoint ``{outdir}/{prefix}.mbpt.h5``
      must exist; both are errors here, before anything runs.
    """
    gw_edmft_group = deepcopy(gw_edmft_params)

    # Copy top-level gw_edmft settings
    gw_edmft_params = {}
    if 'niter' not in gw_edmft_group:
        raise KeyError("'niter' is required: specify the total number of GW+EDMFT cycles as a positive integer.")
    niter = gw_edmft_group.pop('niter')
    gw_iter_per_loop = gw_edmft_group.pop('gw_iter_per_loop', 1)
    edmft_iter_per_loop = gw_edmft_group.pop('edmft_iter_per_loop', 1)
    if 'wannier_file' in gw_edmft_group:
        gw_edmft_params['wannier_file'] = gw_edmft_group.pop('wannier_file')
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
    
    screen_type = gw_edmft_group.pop('screen_type', 'gw_edmft')
    div_treatment = gw_edmft_group.pop('div_treatment', 'gygi')
    outdir = gw_edmft_group.pop('outdir', './')
    prefix = gw_edmft_group.pop('prefix', 'coqui')
    # restart = False implies the workflow will start from a fresh checkpoint where both GW and EDMFT parts start from scratch;
    # this is not available yet in the current implementation as the GW part is always assumed to restart from a previous checkpoint.
    restart = gw_edmft_group.pop('restart', True)
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
        'corr_only': gw_edmft_group.pop('corr_only', True),
        # disable iterative solver for the embedding step. The iterative solve happens somewhere else. 
        'iter_alg': {'enable': False} 
    }

    if 'edmft' not in gw_edmft_group:
        raise KeyError("Missing 'edmft' section in 'gw_edmft' parameters to specify the EDMFT part of the workflow.")
    edmft_section = gw_edmft_group.pop('edmft')
    gw_edmft_params['edmft'] = {
        'restart': restart,
        'iter_alg': edmft_iter_params,
        'chkpt_h5': edmft_section.pop('chkpt_h5', outdir+"/"+prefix+".mbpt.h5")
    }

    # a top-level 'iaft' is the impurity mesh, so it belongs to the EDMFT section
    if 'iaft' in gw_edmft_group:
        gw_edmft_params['edmft']['iaft'] = gw_edmft_group.pop('iaft')

    # One block per impurity (a single dict is one impurity), in the nested layout
    if 'impurity' in edmft_section:
        impurity_blocks = edmft_section.pop('impurity')
        if not isinstance(impurity_blocks, list):
            impurity_blocks = [impurity_blocks]
        impurity_blocks = [convert_legacy_impurity_block(imp_block=block, index=index)
                           for index, block in enumerate(impurity_blocks)]
        for block in impurity_blocks:
            if 'degenerate_blk' in block:
                block['degenerate_blk'] = [np.array(x) for x in block['degenerate_blk']]
        gw_edmft_params['edmft']['impurity'] = impurity_blocks

    # Copy any remaining keys in the edmft section
    for key, value in edmft_section.items():
        gw_edmft_params['edmft'][key] = value

    # Copy any remaining keys in the top-level gw_edmft section
    for key, value in gw_edmft_group.items():
        gw_edmft_params[key] = value

    return gw_edmft_params


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


def save_impurities(dmft_grp, *, impurity_results=None, impurity_inputs=None, solver_data=None,
                    impurity_index=-1, iteration=-1, chkpt_impurity_index=None):
    """
    Write the impurity groups of one DMFT iteration, ``dmft_grp/iter{iteration}/impurity_{i}/``.

    Each impurity group holds up to three subgroups, one per kind of data:

    - ``inputs/``: what the workflow handed to the impurity solver (the local Green's function
      and interactions, the Weiss fields, ``h0``, ``delta_iw``, ...); written from
      ``impurity_inputs``.
    - ``results/``: the workflow's results of the impurity, the arrays of the solver contract
      (``Sigma_infty``, ``Sigma_iw_data``, ``G_iw_data``, ``Pi_iw_data``, ``W_iw_data``,
      ``orbital_occupations``, ``density``) and what the loop adds to them (``mu_imp``, the
      double-counting terms, ``convergence``); written from ``impurity_results`` and loaded back
      on restart.
    - ``solver_results/``: the backend-native objects of the solve (for CT-SEG, the TRIQS
      containers); written from ``solver_data``, replaced as a whole on every write, and never
      loaded back.

    The three lists are indexed alike, one entry per impurity, and `impurity_index` selects the
    same entry of each.

    Parameters
    ----------
    dmft_grp : h5 group
        The ``dmft`` group of the checkpoint.
    impurity_results : list of dict, optional
        One dict per impurity, for ``results/``.
    impurity_inputs : list of dict, optional
        One dict per impurity, for ``inputs/``.
    solver_data : list of dict, optional
        One dict per impurity, for ``solver_results/``. An entry that is an empty dict removes
        the subgroup of its impurity; an entry that is ``None`` leaves it untouched.
    impurity_index : int, optional
        The entry of the lists (`impurity_results`, `impurity_inputs`, `solver_data`) to save.
        If -1, save all impurities. Default is -1.
    iteration : int, optional
        The DMFT iteration number. If -1, determine the iteration automatically. Default is -1.
    chkpt_impurity_index : int, optional
        Impurity number of the checkpoint group ``impurity_<chkpt_impurity_index>`` that list
        entry ``impurity_index`` is written to, for lists that hold only a subset of the
        impurities.
        Default ``None``: the group number is ``impurity_index`` itself.

    Raises
    ------
    ValueError
        If neither `impurity_results` nor `impurity_inputs` is provided.
        If the given lists have different lengths.
        If `impurity_index` is out of range.
        If `chkpt_impurity_index` is given together with `impurity_index = -1`, or is negative.

    Notes
    -----
    - If `iteration` is -1, the iteration is ``final_iter + 1`` read from `dmft_grp` (the legacy
      key ``final_iteration`` is accepted); without either key it defaults to 1.
    - The function creates the group of the iteration if needed and stores the impurity data
      under subgroups named `impurity_<index>`.
    - The restart marker ``dmft_grp/final_iter`` is read but never written here: it records
      the progress of the GW+EDMFT loop and is written by the loop's own writers in
      ``dmft_state``, so rewriting the impurities of an earlier iteration, as
      ``solve_impurities_from_chkpt`` does, does not change where a later restart resumes.
    """
    if impurity_results is None and impurity_inputs is None:
        raise ValueError("Either impurity_results or impurity_inputs must be provided.")

    if iteration == -1:
        if "final_iter" in dmft_grp.keys():
            iteration = dmft_grp["final_iter"] + 1
        elif "final_iteration" in dmft_grp.keys():
            # backward compatibility
            # TODO remove it
            iteration = dmft_grp["final_iteration"]
        else:
            iteration = 1

    given = {name: lst for name, lst in (('impurity_results', impurity_results),
                                         ('impurity_inputs', impurity_inputs),
                                         ('solver_data', solver_data)) if lst is not None}
    lengths = {name: len(lst) for name, lst in given.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"io::save_impurities: the lists must have the same length, one entry "
                         f"per impurity; got {lengths}.")
    num_impurities = next(iter(lengths.values()))

    if impurity_index == -1:
        impurity_list = np.arange(num_impurities)
    else:
        if impurity_index < 0 or impurity_index >= num_impurities:
            raise ValueError(f"impurity_index {impurity_index} is out of range [-1, {num_impurities}).")
        impurity_list = [impurity_index]
    if chkpt_impurity_index is not None and (impurity_index == -1 or chkpt_impurity_index < 0):
        raise ValueError(f"io::save_impurities: chkpt_impurity_index {chkpt_impurity_index} needs "
                         f"a single impurity_index >= 0 and must be non-negative "
                         f"(got impurity_index {impurity_index}).")
    if f"iter{iteration}" not in dmft_grp.keys():
        dmft_grp.create_group(f"iter{iteration}")
    iter_grp = dmft_grp[f"iter{iteration}"]

    for imp_i in impurity_list:
        grp_i = imp_i if chkpt_impurity_index is None else chkpt_impurity_index
        if f"impurity_{grp_i}" not in iter_grp.keys():
            iter_grp.create_group(f"impurity_{grp_i}")
        imp_grp = iter_grp[f"impurity_{grp_i}"]
        if impurity_results is not None:
            _write_impurity_results(imp_grp, impurity_results[imp_i])
        if impurity_inputs is not None:
            _write_impurity_inputs(imp_grp, impurity_inputs[imp_i])
        if solver_data is not None and solver_data[imp_i] is not None:
            _write_solver_results(imp_grp, solver_data[imp_i])

    return iteration


_REQUIRED_RESULT_KEYS = ('Sigma_infty', 'Sigma_iw_data', 'G_iw_data', 'orbital_occupations',
                         'density')


def _write_impurity_results(h5_grp, impurity_results):
    """
    Write impurity results to ``results/`` of ``h5_grp``: every key whose value is not None
    (h5 cannot store None). The arrays of the solver contract are required.
    """
    if "results" not in h5_grp.keys():
        h5_grp.create_group("results")
    res_grp = h5_grp["results"]
    missing = [key for key in _REQUIRED_RESULT_KEYS if impurity_results.get(key) is None]
    if missing:
        raise KeyError(f"io::_write_impurity_results: missing required result key(s) {missing}")
    for key, value in impurity_results.items():
        if value is not None:
            res_grp[key] = value


def _write_solver_results(h5_grp, solver_data):
    """
    Replace ``solver_results/`` of ``h5_grp`` (an impurity group) with the backend-native
    objects of ``solver_data``; an empty dict removes the group.
    """
    if "solver_results" in h5_grp.keys():
        del h5_grp["solver_results"]
    if not solver_data:
        return
    h5_grp.create_group("solver_results")
    grp = h5_grp["solver_results"]
    for key, value in solver_data.items():
        grp[key] = value


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
    for key in ('screen_type', 'degenerate_blk', 'occupations_sab'):
        if impurity_inputs.get(key) is not None:
            inp_grp[key] = impurity_inputs[key]


def resolve_dmft_iteration(h5_filename, iteration):
    """``iteration`` itself, or the last stored DMFT iteration of ``h5_filename`` for ``-1``. Collective."""
    if iteration >= 0:
        return iteration
    last = None
    if mpi.is_master_node():
        with HDFArchive(h5_filename, 'r') as ar:
            dmft_grp = ar["dmft"]
            # "final_iteration": backward compatibility, as in read_impurity_chkpt
            last = dmft_grp["final_iter"] if "final_iter" in dmft_grp else dmft_grp["final_iteration"]
            del dmft_grp  # an open subgroup keeps the file open
    return int(mpi.bcast(last))


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
        - "results" → list of impurity_results
        - "inputs"  → list of impurity_inputs
        - "both"    → (impurity_results, impurity_inputs)
    """
    app_log(2, f"Reading impurity checkpoint file: {h5_filename}")
    assert read in {"results", "inputs", "both"}, \
        "Argument 'read' must be one of {'results', 'inputs', 'both'}."
    impurity_results, impurity_inputs = [], []
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
                    impurity_results.append(res_tmp)

                if read in {"inputs", "both"}:
                    inp_grp = iter_grp[f"impurity_{i}/inputs"]
                    inp_tmp = read_all_keys(inp_grp)
                    mpi.bcast(inp_tmp)
                    impurity_inputs.append(inp_tmp)
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
                impurity_results.append(res_tmp)
            if read in {"inputs", "both"}:
                inp_tmp = mpi.bcast(inp_tmp)
                impurity_inputs.append(inp_tmp)

    mpi.barrier()

    if read == "results":
        return impurity_results
    elif read == "inputs":
        return impurity_inputs
    else:
        return impurity_results, impurity_inputs


# TRIQS solver objects that CT-SEG wrote under results/ before the impurity-solver interface;
# they belong to solver_results/ now and are never loaded back into Res.
_LEGACY_SOLVER_RESULT_KEYS = ('G_iw', 'Sigma_iw', 'Pi_iw', 'W_iw', 'Chi_iw')


def update_impurity_results_from_chkpt(impurity_results, h5_filename, iteration=-1):
    """
    Update impurity solver results from a checkpoint file.

    Parameters
    ----------
    impurity_results : list of dict
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
            for imp_index, res in enumerate(impurity_results):
                imp_grp = ar[f'dmft/iter{iteration}/impurity_{imp_index}/results']
                loaded = read_all_keys(imp_grp)
                legacy = sorted(set(loaded) & set(_LEGACY_SOLVER_RESULT_KEYS))
                if legacy:
                    app_log(1, f"Checkpoint impurity {imp_index}: the solver objects {legacy} under "
                               f"results/ belong to solver_results/ since the impurity-solver\n"
                               f"  interface; they are not loaded (the native arrays are).\n")
                res_tmp.update({key: value for key, value in loaded.items() if key not in legacy})
                res_tmp = mpi.bcast(res_tmp)
                res.update(res_tmp)
    else:
        for imp_index, res in enumerate(impurity_results):
            res_tmp = mpi.bcast(res_tmp)
            res.update(res_tmp)

    iteration = mpi.bcast(iteration)

    return iteration


def read_all_keys(h5_grp):
    output = {}
    for key in h5_grp.keys():
        output[key] = h5_grp[key]
    return output

