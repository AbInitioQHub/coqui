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
from h5 import HDFArchive

import triqs_modest as modest

import coqui
from coqui.utils.imag_axes_ft import IAFT
import coqui.dmft as coqui_dmft
from coqui.dmft.io import convert_gw_edmft_params, resolve_dmft_iteration
from coqui.dmft.solver_api import (ImpurityProblem, make_solver, 
                                   check_solver_capabilities_for_gw_edmft,
                                   check_solver_capabilities_for_problem)

Hartree_eV = 27.211386245988


def run_gw_edmft(h_int, embedding, *, proj_info=None, params: dict):
    """
    Run the GW+EDMFT self-consistency workflow.

    Parameters are passed as a dictionary via ``params``.  

    Parameters
    ----------
    h_int : ThcCoulomb
        Coulomb interaction object for the full system, used in GW and
        downfolding/upfolding steps. Obtained from ``make_thc_coulomb``.
    embedding : triqs_modest.embedding
        Embedding object from TRIQS/ModEST
        (https://github.com/TRIQS/modest) defining mappings between local MLWF
        orbitals and impurity subspaces.
    proj_info : dict, optional
        Projector metadata for the correlated subspace. If ``None``,
        ``wannier_file`` must be provided in ``params``.
    params : dict
        GW+EDMFT control parameter dictionary.

        Required top-level keys
        ~~~~~~~~~~~~~~~~~~~~~~~
            - ``niter`` (int): total number of outer GW+EDMFT cycles. Must be a positive integer.
            - ``edmft`` (dict): EDMFT subsection containing impurity controls.
            - ``edmft.impurity`` (dict or list[dict]): one block per impurity.
            - ``wannier_file`` (str), only when ``proj_info is None``:
                Wannier90 HDF5 file used to build projectors.

        Optional top-level keys and defaults
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            - ``gw_iter_per_loop`` (int, default ``1``): number of GW updates per
                GW+EDMFT cycle.
            - ``edmft_iter_per_loop`` (int, default ``1``): number of EDMFT updates
                per GW+EDMFT cycle.
            - ``outdir`` (str, default ``"./"``): output/checkpoint directory.
            - ``prefix`` (str, default ``"coqui"``): checkpoint file prefix.
            - ``restart`` (bool, default ``True``; ``False`` is not implemented):
                continue from an existing GW checkpoint.
            - ``screen_type`` (``"rpa"``, ``"gw_edmft"``; default ``"gw_edmft"``): 
                screening prescription for the lattice irreducible polarization. 
            - ``div_treatment`` (str, default ``"gygi"``): treatment for the
                ``q -> 0`` divergence.
            - ``corr_only`` (bool, default ``True``): embmeding dynamic part of 
                the self-energy only. 

        ``iter_alg`` section (optional): Iterative controls for GW and EDMFT inner loops.
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ``iter_alg`` defaults to ``{"alg": "damping", "mixing": 0.3}``.
        Currently, only ``alg="damping"`` is supported for GW+EDMFT.

            - ``mixing`` (float, default ``0.3``): fallback damping value.
            - ``gw_mixing`` (float, default ``mixing``): GW-specific damping.
            - ``edmft_mixing`` (float, default ``mixing``): EDMFT-specific damping.
            - ``edmft_mix_in_first_iter`` (bool, default ``True``): whether to mix 
                the EDMFT solution in the first iteration.

        ``edmft`` section
        ~~~~~~~~~~~~~~~~~
            - ``chkpt_h5`` (str, default ``{outdir}/{prefix}.mbpt.h5``): impurity
                solver checkpoint file path. If the file exists and ``restart=True``,
                previous EDMFT impurity results are loaded as the initial guess.
            - ``iaft`` (dict, optional): impurity DLR mesh controls.
                - ``wmax`` (float, default GW ``wmax``): impurity DLR frequency cutoff.
                - ``eps`` (float, default GW ``eps``): impurity DLR precision.
            - ``impurity`` (dict or list[dict], required): per-impurity solver
                configuration. A single dict applies to one impurity; use a list for
                multiple impurities.

                Per-impurity block (``edmft.impurity``)
                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                Workflow keys, beside a nested ``solver`` table (required):

                    - ``solver`` (dict): ``type`` is the registered name of the backend
                        (``"ctseg"``, or a name added with ``coqui.dmft.register_solver`` for
                        an ``ImpuritySolver`` subclass outside the package); every other key is
                        the backend's. Common to the QMC backends: ``n_cycles``
                        (total over MPI ranks), ``n_warmup_cycles`` and ``length_cycle`` (per
                        rank), ``random_seed`` (base seed; rank ``r`` uses
                        ``(random_seed + 928374 r) mod (2^31 - 1)``), ``screen_j`` (default
                        ``False``: include the screened Hund's coupling in the density-density
                        retarded interaction), ``h_int`` (``"density"``; the other kinds are
                        reserved for backends with a general interaction). CT-SEG keys: ``n_iw``
                        (default ``int(1.4 * max_dlr_idx)``), ``n_tau`` (default
                        ``max(500001, 2 beta wmax + 1)``), ``n_tau_bosonic`` (default ``n_tau``),
                        ``perform_tail_fit`` (default ``False``), ``fit_max_moment`` (``3``),
                        ``fit_min_w`` / ``fit_max_w`` / ``fit_min_n`` / ``fit_max_n``
                        (``None``), ``analytic_hf`` (``True``), ``truncate_uchi`` (``False``),
                        ``reuse_config`` (``False``) with ``restart_n_warmup_cycles`` (int >= 1,
                        required with ``reuse_config``); unknown keys are forwarded to CT-SEG's
                        ``solve``.
                        CT-INT keys (``type = "ctint"``; density-density ``h_int`` only, the
                        polarization always measured, no configuration reuse): ``n_tau_chi2``
                        (``1001``, the tau points of the density-correlator measurement); the
                        tau grids follow the impurity DLR mesh; unknown keys are forwarded to
                        ``triqs_ctint.Solver.solve``.
                        CT-HYB keys (``type = "cthyb"``; ``h_int`` = ``"density"``, ``"kanamori"``
                        or ``"full"``; matrix-valued blocks; no configuration reuse): ``perform_tail_fit`` (``True``) with the window
                        ``fit_min_w`` / ``fit_max_w`` (``2.5`` / ``10.0`` Ha, problem- and
                        budget-dependent) and ``fit_max_moment`` (``7``); the meshes and the
                        Lang-Firsov kernel follow the impurity DLR mesh; unknown keys are
                        forwarded to ``triqs_cthyb.Solver.solve``.
                    - ``init_imp_results`` (str, default ``"dc"``): initialization strategy
                        for impurity self-energies (``"dc"`` or ``"zero"``).
                    - ``degenerate_blk`` (list[list[int]], default ``None``): explicit
                        orbital-degeneracy groups for symmetry enforcement.
                    - ``degenerate_blk_thresh`` (float, default ``None``): infer
                        ``degenerate_blk`` from hybridization when explicit blocks are absent.
                     - ``causal_projection`` (dict, default ``None``): options for bosonic
                         causal projection. A bath-fitting step is skipped when its
                         ``nbath_per_orbital_*`` key is absent or set to ``-1``; the
                         ``w0_treatment_*`` regularizations are applied whenever they are
                         set, with or without the fit.
                         Keys:

                         - ``nbath_per_orbital_wloc`` (int, default ``-1``): bath orbitals for
                           local screened interaction ``Wloc`` fitting.
                         - ``nbath_per_orbital_u_weiss`` (int, default ``-1``): bath orbitals for
                           bosonic Weiss field ``U_weiss`` fitting.
                         - ``nbath_per_orbital_impurity`` (int, default ``-1``): bath orbitals for
                           impurity ``Pi_iw`` / ``W_iw`` fitting.
                         - ``n_exclude_low_freq_wloc`` / ``n_exclude_low_freq_u_weiss`` /
                           ``n_exclude_low_freq_impurity`` (int, default ``0`` = no exclusion):
                           number of lowest-frequency Matsubara shells to drop from that
                           target's fit input (the fitted function is still evaluated on the
                           full mesh). ``1`` excludes only ``w=0``.
                         - ``w0_treatment_for_pi`` / ``w0_treatment_for_w`` / ``w0_treatment_for_weiss``
                           (str, optional): regularization applied to ``A(iw=0)`` before the
                           bath fit, and on its own when that target is not fitted
                           (``"flatten"`` or ``"extrapolate_order_<n>"``).

                         Example — fit only ``Wloc`` and ``U_weiss`` with 3 poles each::

                             {"nbath_per_orbital_wloc": 3, "nbath_per_orbital_u_weiss": 3}
                    - ``chemical_potential`` (dict, optional): requests the impurity
                        chemical-potential search; ``tolerance`` (default ``0.1``) is its only
                        key. The solver parameters of the search's occupations-only solves are
                        the production ones with the QMC cycle controls of the ``mu_search``
                        table of the ``solver`` block (``n_cycles``, default 5 % of the
                        production total, ``n_warmup_cycles``, ``length_cycle``,
                        ``restart_n_warmup_cycles``) and its solver-output keys
                        (``suppress_solver_output``, default ``True``, ``solver_output_file``);
                        no other key may be set there.

                A block without ``solver`` is read as the former flat CT-SEG layout for backward compatibility. 

    Returns
    -------
    None
        Results are written to checkpoint files (``{outdir}/{prefix}.mbpt.h5`` 
        and the impurity ``chkpt_h5``).

    Examples
    --------
    Minimal flat ``params`` dictionary:

    .. code-block:: python

        params = {
            "niter": 2,
            "gw_iter_per_loop": 1,
            "edmft_iter_per_loop": 1,
            "outdir": "./",
            "prefix": "svo",
            "screen_type": "gw_edmft",
            "div_treatment": "gygi",
            "wannier_file": "svo.mlwf.h5",
            "iter_alg": {
                "alg": "damping",
                "mixing": 0.4,
                "edmft_mixing": 0.2,
                "edmft_mix_in_first_iter": True,
            },
            "edmft": {
                "chkpt_h5": "./svo.dmft.h5",
                "iaft": {"wmax": 5.0, "eps": 1e-10},
                "impurity": {
                    "init_imp_results": "dc",
                    "solver": {
                        "type": "ctseg",
                        "n_iw": 3000,
                        "n_tau": 96001,
                        "length_cycle": 60,
                        "n_warmup_cycles": 6000,
                        "n_cycles": 2000000,
                        "mu_search": {"n_cycles": 100000},
                    },
                    "chemical_potential": {"tolerance": 0.1},
                },
            },
        }
        run_gw_edmft(h_int, embedding, params=params)
    """
    # Convert parameters to the internal format and set defaults
    params = convert_gw_edmft_params(params)
    coqui_mpi = h_int.mpi()
    mf = h_int.mf()

    if not isinstance(embedding, modest.embedding.Embedding):
        raise TypeError(f"Expected a modest.embedding object for 'embedding', got {type(embedding)}")

    if proj_info is None:
        obe = modest.make_one_body_elements_gw(params.pop("wannier_file"))
        proj_info = coqui_dmft.get_proj_info(obe.P)

    # Convert to the internal format
    niter, gw_iter_per_loop, edmft_iter_per_loop = (
        params.pop('niter'), params.pop('gw_iter_per_loop'), params.pop('edmft_iter_per_loop')
    )

    # http://patorjk.com/software/taag/#p=display&f=Calvin+S&t=COQUI+GW%2BEDMFT&x=none&v=4&h=4&w=80&we=false
    coqui.app_log(1, "╔═╗╔═╗╔═╗ ╦ ╦╦  ╔═╗┬ ┬╔═╗┌┬┐┌┬┐┌─┐┌┬┐\n"
                     "║  ║ ║║═╬╗║ ║║  ║ ╦│││║╣  │││││├┤  │ \n"
                     "╚═╝╚═╝╚═╝╚╚═╝╩  ╚═╝└┴┘╚═╝─┴┘┴ ┴└   ┴ \n")
    coqui.app_log(1, f"  Total GW+EDMFT cycles (niter)       = {niter}")
    coqui.app_log(1, f"  GW iterations per GW+EDMFT cycle    = {gw_iter_per_loop}")
    coqui.app_log(1, f"  EDMFT iterations per GW+EDMFT cycle = {edmft_iter_per_loop}\n")

    embedding_2e = embedding.merge_embed_block_by_imp.slice_sigma
    coqui.app_log(1, embedding.description(True))

    try:
        gw_params        = params.pop('gw', None)
        wloc_params      = params.pop('wloc')
        gloc_params      = params.pop('gloc')
        embed_params     = params.pop('dmft_embed')
        edmft_params     = params.pop('edmft')
        imp_iaft_params  = edmft_params.pop('iaft', {})
        iterative_params = edmft_params.pop('iter_alg', None)
    except KeyError as e:
        raise KeyError(f"run_gw_edmft: Missing required params key: {e.args[0]}")

    
    # one block per impurity: the workflow keys beside the backend's 'solver' table
    impurity_params_list = edmft_params['impurity']
    solvers = _make_solvers(impurity_params_list, coqui_mpi.comm_size(), wloc_params['screen_type'])

    coqui_chkpt_h5 = embed_params['outdir']+"/"+embed_params['prefix']+".mbpt.h5"
    if not os.path.exists(coqui_chkpt_h5):
        raise FileNotFoundError(
            f"scf_driver::run_gw_edmft: the GW checkpoint {coqui_chkpt_h5} is missing; the GW+EDMFT "
            f"workflow starts from a converged GW checkpoint.")
    dmft_chkpt_h5 = edmft_params.pop('chkpt_h5', coqui_chkpt_h5)

    # DMFT state container
    dmft_state = coqui_dmft.DMFTState.make_dmft_state(
        coqui_chkpt_h5, embedding, embedding_2e, 
        wmax_imp=imp_iaft_params.get('wmax', None),
        eps_imp=imp_iaft_params.get('eps', None),
        spin_average=mf.nspin()==1,
        screen_type=wloc_params['screen_type'],
        verbal=coqui_mpi.root()
    )
    if edmft_params.pop('restart', True):
        dmft_state.load(dmft_chkpt_h5)
    
    coqui_mpi.barrier()

    for iteration in range(niter):

        if gw_params is not None and gw_iter_per_loop >= 1:
            # update GW solution with fixed impurity self-energies and polarizabilities
            _gw_loop(
                mf, h_int, proj_info, 
                dmft_state, coqui_chkpt_h5, 
                gw_params, embed_params, gw_iter_per_loop
            )

        if edmft_iter_per_loop >= 1:
            # Set the Green's function for the non-local RPA polarizability
            with HDFArchive(coqui_chkpt_h5, 'r') as ar:
                mbpt_final_iter = ar["scf/final_iter"]
                try:
                    gf_for_wloc_source = ar[f"scf/iter{mbpt_final_iter}/greens_func_source"]
                    gf_for_wloc_iteration = ar[f"scf/iter{mbpt_final_iter}/greens_func_iteration"]
                except KeyError:
                    gf_for_wloc_source = ar[f"scf/iter{mbpt_final_iter}/input_grp"]
                    gf_for_wloc_iteration = ar[f"scf/iter{mbpt_final_iter}/input_iter"]
            wloc_params["greens_func_source"] = gf_for_wloc_source
            wloc_params["greens_func_iteration"] = gf_for_wloc_iteration
            coqui_mpi.barrier()

            # inner EDMFT loop
            _edmft_loop(
                mf, h_int, proj_info, dmft_state, dmft_chkpt_h5, coqui_chkpt_h5, 
                gloc_params, wloc_params, solvers, impurity_params_list, embed_params,
                iterative_params, edmft_iter_per_loop
            )


def _gw_loop(mf, h_int, proj_info, 
             dmft_state, coqui_chkpt_h5, 
             gw_params, embed_params, gw_iter_per_loop):
    if gw_iter_per_loop < 1:
        return

    coqui_mpi = h_int.mpi()
    for gw_iteration in range(gw_iter_per_loop):
        with HDFArchive(coqui_chkpt_h5, 'r') as ar:
            greens_func_source = "embed" if "embed" in ar.keys() else "scf"
            greens_func_iteration = ar[f"{greens_func_source}/final_iter"]
        coqui_mpi.barrier()

        # GW if gw_params presents
        gw_params["greens_func_source"] = greens_func_source
        gw_params["greens_func_iteration"] = greens_func_iteration
        coqui.run_gw(
            gw_params, h_int = h_int, projector_info = proj_info,
            local_polarizabilities = dmft_state.local_pi_w
        )
        coqui_mpi.barrier()

        # Don't upfold the results if gw_iter_per_loop==1. 
        # Not sure if this is the best choice, but it allows us to skip one upfolding in the common case of 
        # doing just one GW iteration per GW+EDMFT loop, which can save some disk space in the checkpoint h5.
        if gw_iter_per_loop > 1: 
            # Updates GW+EDMFT solution with the latest GW results while keeping the impurity solutions fixed.
            # Upfolding
            coqui.dmft_embed(
                mf, embed_params,
                projector_info = proj_info,
                local_hf_potentials = dmft_state.local_sigma_infty,
                local_sigma_dynamic = dmft_state.local_sigma_w
            )
            coqui_mpi.barrier()


def _edmft_loop(mf, h_int, proj_info, dmft_state, dmft_chkpt_h5, coqui_chkpt_h5, 
               gloc_params, wloc_params, solvers, impurity_params_list, embed_params,
               iterative_params, num_iter):

    coqui_mpi = mf.mpi()

    for iteration in range(num_iter):
        with HDFArchive(coqui_chkpt_h5, 'r') as ar:
            greens_func_source = "embed" if "embed" in ar.keys() else "scf"
            greens_func_iteration = ar[f"{greens_func_source}/final_iter"]

        # downfold for W_loc
        # greens_func_source and greens_func_iteration should be fixed during the inner loop
        Vloc, Wloc_t = coqui.downfold_coulomb(
            h_int, wloc_params,
            projector_info=proj_info,
            local_polarizabilities=dmft_state.local_pi_w
        )

        # downfold for G_loc
        gloc_params["greens_func_source"] = greens_func_source
        gloc_params["greens_func_iteration"] = greens_func_iteration
        Gloc_t = coqui.downfold_local_gf(mf, gloc_params, projector_info=proj_info)

        if coqui_mpi.root():
            dmft_state.iaft.check_leakage(Gloc_t, stats='f', name='Gloc in the full MLWF space')
            dmft_state.iaft.check_leakage_phsym(Wloc_t, stats='b', name='Wloc in the full MLWF space')

        # Convert spin axis → list of length nspin
        Vloc, Wloc_t = [Vloc], [Wloc_t]
        Gloc_t = [Gloc_t[:, s] for s in range(Gloc_t.shape[1])]
        if mf.nspin() == 1:
            Gloc_t = [Gloc_t[0], Gloc_t[0].copy()]
        
        # Extract local Green's function and screened interactions for each impurity
        Gloc_C    = dmft_state.embedding['1e'].extract(Gloc_t)   # block matrix
        Vloc_C    = dmft_state.embedding['2e'].extract(Vloc)
        Wloc_C    = dmft_state.embedding['2e'].extract(Wloc_t)
        Vloc_C    = [ V[0] for V in Vloc_C ]      # spinless
        Wloc_C    = [ W_t[0] for W_t in Wloc_C ]  # spinless

        for imp_index, (G_t, W_t, V) in enumerate(zip(Gloc_C, Wloc_C, Vloc_C)):
            coqui_dmft.print_title_box(f"IMPURITY {imp_index}")

            imp_params = impurity_params_list[imp_index]
            solver = solvers[imp_index]
            Res, Input = dmft_state.impurity_results[imp_index], dmft_state.impurity_inputs[imp_index]
            Input['Gloc_t'] = G_t
            Input['Wloc_t'] = coqui_dmft.chemistry_to_product_basis(W_t)
            Input['Vloc'] = coqui_dmft.chemistry_to_product_basis(V)

            coqui_dmft.fit_local_results_boson(
                Input, dmft_state.iaft, imp_params.get("causal_projection")
            )

            # Save previous Weiss fields before updating (for convergence check)
            prev_g_weiss_iw = Input.get('g_weiss_iw')
            prev_u_weiss_iw = Input.get('u_weiss_iw')

            # Fermionic and bosonic Weiss fields
            Input['g_weiss_iw'], Input['u_weiss_iw'] = _compute_weiss_fields(
                coqui_mpi, Res, Input, imp_params, dmft_state.iaft
            )
            Input['u_weiss_iw'] = coqui_dmft.fit_u_weiss(
                Input['u_weiss_iw'], dmft_state.iaft, imp_params.get("causal_projection")
            )

            # h0: (nspin, norb, norb), delta_iw: (niw, nspin, norb, norb)
            Input['h0'], Input['delta_iw'] = coqui_dmft.extract_h0_and_delta(
                Input['g_weiss_iw'], dmft_state.iaft
            )

            Ub, Ubp, Jb_spin, Jb_pair = coqui_dmft.hubbard_kanamori_coulomb(Input['Vloc'])
            U, Up, J_spin, J_pair = coqui_dmft.hubbard_kanamori_coulomb(Input['Vloc']+Input['u_weiss_iw'][0])
            gloc_t_arr = coqui_dmft.blk_arr_to_arr(Input['Gloc_t'], Input["gf_struct"])
            dm = -dmft_state.iaft.tau_interpolate(gloc_t_arr, dmft_state.iaft.beta, 'f')[0]
            g_beta_half = -dmft_state.iaft.tau_interpolate(gloc_t_arr, dmft_state.iaft.beta/2, 'f')[0]
            Input['density'] = (np.diag(dm[0]).sum() + np.diag(dm[1]).sum()).real
            Input['occupations_sab'] = np.real(dm)
            coqui.app_log(1, "Bare/static orbital-averaged interactions for the impurity")
            coqui.app_log(1, "----------------------------------------------------------")
            coqui.app_log(1, f"  intra-orbital                  = {Ub*Hartree_eV:.4f}, {U*Hartree_eV:.4f} eV")
            coqui.app_log(1, f"  inter-orbital                  = {Ubp*Hartree_eV:.4f}, {Up*Hartree_eV:.4f} eV")
            coqui.app_log(1, f"  Hund's coupling (spin-flip)    = {Jb_spin*Hartree_eV:.4f}, {J_spin*Hartree_eV:.4f} eV")
            coqui.app_log(1, f"  Hund's coupling (pair-hopping) = {Jb_pair*Hartree_eV:.4f}, {J_pair*Hartree_eV:.4f} eV\n")

            # For metals, beta * G(beta/2) saturates with increasing beta
            # For insulators, beta * G(beta/2) decays exponentially to zero with increasing beta
            coqui.app_log(1, "Spectral weight proxy at Fermi level (eV^-1): A(w) ~ -beta/pi * G_loc(tau=beta/2)")
            coqui.app_log(1, "--------------------------------------------------------------")
            coqui.app_log(1, f"  Spin up:   {(dmft_state.iaft.beta/np.pi) * np.diag(g_beta_half[0]).real / Hartree_eV}")
            coqui.app_log(1, f"  Spin down: {(dmft_state.iaft.beta/np.pi) * np.diag(g_beta_half[1]).real / Hartree_eV}\n")

            coqui.app_log(1, "Local densities ")
            coqui.app_log(1, "-------------------")
            coqui.app_log(1, f"  Total: {Input['density']:.4f}")
            coqui.app_log(1, f"  Spin up: {np.diag(dm[0]).real}")
            coqui.app_log(1, f"  Spin down: {np.diag(dm[1]).real}\n")

            # degenerate groups of blocks: fixed on the first hybridization of this impurity
            if Input.get('degenerate_blk') is None:
                Input['degenerate_blk'] = _degenerate_blocks(
                    Input, imp_params, dmft_state.iaft, dmft_state.dlr_wmax, dmft_state.dlr_eps,
                    solver)
            # impurity problem in CoQui's data format. 
            problem = ImpurityProblem(
                gf_struct=Res['gf_struct'], beta=dmft_state.iaft.beta, iaft=dmft_state.iaft,
                dlr_wmax=dmft_state.dlr_wmax, dlr_eps=dmft_state.dlr_eps,
                h0_sab=Input['h0'], delta_wsab=Input['delta_iw'], v_abcd=Input['Vloc'],
                u_weiss_wabcd=Input['u_weiss_iw'], target_density=Input['density'],
                occupations_sab=Input['occupations_sab'], degenerate_blk=Input['degenerate_blk'])
            check_solver_capabilities_for_problem(solver=solver, problem=problem)
            dmft_state.save_impurity_inputs(dmft_chkpt_h5, imp_index)

            # Call impurity solver, and store sigma_imp, vhf_imp, and pi_imp in "Res"
            chkpt = {'h5': dmft_chkpt_h5, 'impurity': imp_index,
                     'iteration': dmft_state.iteration, 'save': True}
            result = _solve_impurity(solver, problem, imp_params, chkpt)
            # Update the solver-agnostic results in Res. 
            Res.update(result.solver_agnostic_results())
            Res['mu_imp'] = problem.mu_shift
            # Backend-specific data goes here
            dmft_state.solver_data[imp_index] = result.solver_data
            # Causal projection
            coqui_dmft.fit_impurity_results_boson(
                Res, dmft_state.iaft, imp_params.get("causal_projection"))

            conv_metrics = _edmft_convergence_check(coqui_mpi, imp_index, Input, Res, dmft_state.iaft,
                                                    prev_g_weiss_iw, prev_u_weiss_iw)

            # Store convergence metrics array: [U_w0, A_w0, diff_g, diff_g_weiss, diff_u_weiss, diff_w, Sigma_w1]
            if coqui_mpi.root():
                A_w0  = float(np.mean((dmft_state.iaft.beta / np.pi) * np.diag(g_beta_half[0]+g_beta_half[1]).real)) * 0.5

                w1_idx = np.where(dmft_state.iaft.wn_mesh(stats='fermion') == 1)[0][0]
                sigma_w0 = 0.0
                norb_tot = 0
                for sigma_blk in Res['Sigma_iw_data']:
                    sigma_w0 += np.sum(np.diag(sigma_blk[w1_idx]).imag)
                    norb_tot += sigma_blk.shape[1]  # number of orbitals in the block
                sigma_w0 = sigma_w0 / norb_tot  # average over all orbitals

                Sigma_imp_iw0 = Res['Sigma_iw_data']
                Res['convergence'] = np.array([
                    U, A_w0,
                    conv_metrics['diff_g'], conv_metrics['diff_g_weiss'],
                    conv_metrics['diff_u_weiss'], conv_metrics['diff_w'], 
                    sigma_w0
                ])

            # GW double counting contributions (current implementation uses Gloc/Wloc inputs)
            Res.update(
                coqui_dmft.solve_gw_dc(
                    coqui_dmft.blk_arr_to_arr(Input['Gloc_t'], Res['gf_struct']),
                    Input['Vloc'], Input['Wloc_t'], Input['u_weiss_iw'],
                    dmft_state.iaft, density_only=True,
                    gf_struct=Res['gf_struct']
                )
            )

            # mixing impurity and dc solutions to facilitate convergence
            dmft_state.damp_impurity_results(
                dmft_chkpt_h5, mixing=iterative_params.get('mixing', 0.7), impurity_indices=[imp_index],
                mix_in_first_iter=iterative_params.get('mix_in_first_iter', True)
            )

            # save solver results for current impurity
            dmft_state.save_impurity_results(dmft_chkpt_h5, imp_index)
            
        # Embed impurity results
        dmft_state.embed_impurity_results()

        # Upfolding
        coqui.dmft_embed(
            mf, embed_params,
            projector_info = proj_info,
            local_hf_potentials = dmft_state.local_sigma_infty,
            local_sigma_dynamic = dmft_state.local_sigma_w
        )
        dmft_state.iteration += 1
        coqui_mpi.barrier()


def _compute_weiss_fields(coqui_mpi, imp_results, imp_inputs, imp_params, iaft):

    gloc_t_mat = coqui_dmft.blk_arr_to_arr(imp_inputs['Gloc_t'], imp_inputs['gf_struct'])

    if imp_results['Sigma_iw_data'] is not None:
        if imp_params.get('set_sigma_infty_to_dc', False):
            vhf_imp =  imp_results['Sigma_infty_dc']
        else:
            vhf_imp =  imp_results['Sigma_infty']

        if imp_inputs['screen_type'] == 'rpa':
            coqui.app_log(2, "screen_type = \"rpa\" -> "
                             "Set impurity polarizability to RPA for bosonic Weiss field.\n")
            # eval Pi_dc using the current Gloc
            pi_imp = iaft.tau_to_w_phsym(
                coqui_dmft.eval_pi_rpa(gloc_t_mat, density_only=True), stats='b'
            )
        else:
            if imp_results.get('Pi_iw_data') is None:
                raise RuntimeError(f"scf_driver::_compute_weiss_fields: screen_type = "
                                   f"{imp_inputs['screen_type']!r} needs the impurity polarization, "
                                   f"but the impurity results carry none (the backend did not "
                                   f"measure the density correlator); use screen_type = \"rpa\" "
                                   f"or a backend that returns it")
            pi_imp = imp_results['Pi_iw_data'][0]

        return (
            coqui_dmft.compute_weiss_fields_w(
                iaft = iaft,
                local_gf = {
                    "Gloc_t": gloc_t_mat,
                    "Wloc_t": imp_inputs['Wloc_t'],
                    "Vloc": imp_inputs['Vloc']
                },
                impurity_selfenergies = {
                    "Vhf_imp": coqui_dmft.blk_arr_to_arr(vhf_imp, imp_results['gf_struct']),
                    "Sigma_imp_w": coqui_dmft.blk_arr_to_arr(imp_results['Sigma_iw_data'], imp_results['gf_struct']),
                    "Pi_imp_w": pi_imp
                },
                density_only=True
            )
        )
    else:
        return (
            coqui_dmft.init_weiss_fields_w(
                iaft = iaft,
                local_gf = {
                    "Gloc_t": gloc_t_mat,
                    "Wloc_t": imp_inputs['Wloc_t'],
                    "Vloc": imp_inputs['Vloc']
                },
                init_imp_results = imp_params.get('init_imp_results', 'dc'),
                density_only=True
            )
        )


def _make_solvers(impurity_params_list, comm_size, screen_type):
    """
    Construct one impurity-solver backend per impurity block, once for the whole run.

    Each backend is built from the block's ``solver`` table, checked against what the GW+EDMFT
    setup needs (``check_solver_capabilities_for_gw_edmft``), and described in the log.

    Parameters
    ----------
    impurity_params_list : list of dict
        The ``edmft.impurity`` blocks, one per impurity, each with a ``solver`` table.
    comm_size : int
        Number of MPI ranks, for the backends that divide a Monte Carlo budget over ranks.
    screen_type : str
        The screening prescription of the run; anything but ``"rpa"`` needs backends that
        return the impurity polarization.

    Returns
    -------
    list of ImpuritySolver
        The backends, in the order of ``impurity_params_list``.

    Raises
    ------
    ValueError, ImportError, TypeError, KeyError
        From ``make_solver`` (an unknown or unimportable backend, an invalid block) or from
        the capability check (a search or a screening the backend cannot provide).
    """
    solvers = []
    for index, imp_params in enumerate(impurity_params_list):
        solver = make_solver(imp_params['solver'], comm_size=comm_size)
        check_solver_capabilities_for_gw_edmft(solver=solver, imp_params=imp_params, screen_type=screen_type)
        coqui.app_log(1, f"Impurity {index}: {solver.describe()}\n")
        solvers.append(solver)
    return solvers


def _degenerate_blocks(Input, imp_params, iaft, dlr_wmax, dlr_eps, solver):
    """
    Determine the degenerate groups of ``gf_struct`` blocks of one impurity. 

    The groups are either 
    1) read from ``imp_params['degenerate_blk']``, if present, or
    2) inferred from the hybridization ``Input['delta_iw']`` if 
        ``degenerate_blk_thresh`` is set, or
    3) none, otherwise. 

    Parameters
    ----------
    Input : dict
        The impurity inputs; ``delta_iw`` and ``gf_struct`` are read.
    imp_params : dict
        The impurity block; ``degenerate_blk`` and ``degenerate_blk_thresh`` are read.
    iaft : IAFT
        The imaginary-axis transform of the crystal.
    dlr_wmax, dlr_eps : float
        DLR controls of the mesh the inference samples the hybridization on.
    solver : ImpuritySolver
        The impurity's backend; its ``complex_hamiltonian`` capability decides whether the
        inference sees a real-forced hybridization, as the backend will.

    Returns
    -------
    list of index groups or None
    """
    degenerate_blk = imp_params.get('degenerate_blk')
    if degenerate_blk is None and imp_params.get('degenerate_blk_thresh'):
        coqui.app_log(2, "Analyzing block symmetries via the hybridization function...\n")
        degenerate_blk = coqui_dmft.analyze_degenerate_blocks(
            delta_iw=Input['delta_iw'], iaft=iaft, gf_struct=Input['gf_struct'],
            dlr_wmax=dlr_wmax, dlr_eps=dlr_eps, threshold=imp_params['degenerate_blk_thresh'],
            force_real=not solver.capabilities.complex_hamiltonian)
    if degenerate_blk is not None:
        coqui_dmft.print_degenerate_blks(degenerate_blk, Input['gf_struct'])
    return degenerate_blk


def _solve_impurity(solver, problem, imp_params, chkpt):
    """
    Impurity chemical-potential search (optional) followed by the production impurity solve.

    Parameters
    ----------
    solver : ImpuritySolver
        The backend of this impurity, built by ``_make_solvers``.
    problem : ImpurityProblem
        The problem of this iteration, built by the loop from its ``Input``. Its ``mu_shift``
        is updated in place by the search.
    imp_params : dict
        The impurity block; its ``chemical_potential`` entry (``tolerance``, default 0.1)
        requests the search, absent means no search.
    chkpt : dict or None
        ``{"h5", "impurity", "iteration", "save"}`` handed to ``solver.solve``; the search
        solves get a copy with ``save = False``.

    Returns
    -------
    ImpurityResult
        The production result
    """
    mu_params = imp_params.get('chemical_potential')
    if mu_params is not None:
        search_chkpt = None if chkpt is None else dict(chkpt, save=False)

        def compute_nelec_fcn(mu):
            problem.mu_shift = mu
            return solver.solve(problem, occupations_only=True, chkpt=search_chkpt).density

        problem.mu_shift, _ = coqui_dmft.compute_mu_impurity(
            problem.target_density, compute_nelec_fcn,
            tolerance=mu_params.get('tolerance', 0.1),
            mu0=0.0, # always start from mu=0 s.t. mu_imp falls back to 0 at convergence
        )

    result = solver.solve(problem, chkpt=chkpt)
    # validate the result dimensions against the problem.
    result.validate(problem)
    coqui.app_log(1, f"Total impurity densities = {result.density}")
    coqui.app_log(1, f"Convergence of impurity density: {result.density - problem.target_density}\n")
    return result


def _edmft_convergence_check(coqui_mpi, imp_index, Input, Res, iaft,
                              prev_g_weiss_iw=None, prev_u_weiss_iw=None):
    """Compute EDMFT self-consistency metrics and return them as a dict.

    Returns
    -------
    dict with keys:
        diff_g       : |Gloc_tau - Gimp_tau| (always computed)
        diff_g_weiss : |g_weiss_tau - g_weiss_prev_tau| (-1.0 if prev not available)
        diff_w       : |Wloc_tau - Wimp_tau| dd (-1.0 if W_iw_data is None)
        diff_u_weiss : |u_weiss_tau - u_weiss_prev_tau| dd (-1.0 if prev not available)
    """
    metrics = {'diff_g': -1.0, 'diff_g_weiss': -1.0, 'diff_w': -1.0, 'diff_u_weiss': -1.0}

    if not coqui_mpi.root():
        return metrics

    gf_struct = Res['gf_struct']

    # |Gloc - Gimp| on the imaginary-time axis
    gloc_t      = coqui_dmft.blk_arr_to_arr(Input['Gloc_t'], gf_struct)
    gimp_iw_mat = coqui_dmft.blk_arr_to_arr(Res['G_iw_data'], gf_struct)
    gimp_t      = iaft.w_to_tau(gimp_iw_mat, stats='f')
    norm_grid   = np.linalg.norm(gloc_t - gimp_t, axis=tuple(range(2, gloc_t.ndim)))
    metrics['diff_g'] = float(np.max(np.abs(norm_grid)))

    coqui.app_log(1, f"EDMFT self-consistency check for impurity {imp_index}:")
    coqui.app_log(1, f"  |Gloc_tau - Gimp_tau|                   = {metrics['diff_g']}")

    if prev_g_weiss_iw is not None:
        g_weiss_t      = iaft.w_to_tau(Input['g_weiss_iw'], stats='f')
        g_weiss_prev_t = iaft.w_to_tau(prev_g_weiss_iw, stats='f')
        norm_grid_gw   = np.linalg.norm(g_weiss_t - g_weiss_prev_t,
                                        axis=tuple(range(2, g_weiss_t.ndim)))
        metrics['diff_g_weiss'] = float(np.max(np.abs(norm_grid_gw)))
        coqui.app_log(1, f"  |g_weiss_tau - g_weiss_prev_tau|        = {metrics['diff_g_weiss']}")

    if Res.get('W_iw_data') is not None:
        # |Wloc - Wimp| restricted to density-density components (always computed)
        wloc_dd  = coqui_dmft.product_basis_to_density_density(Input['Wloc_t'])
        wimp_raw = iaft.w_to_tau_phsym(Res["W_iw_data"][0], stats='b')
        wimp_dd  = wimp_raw if wimp_raw.ndim == 3 else coqui_dmft.product_basis_to_density_density(wimp_raw)
        norm_grid_w = np.linalg.norm(wloc_dd - wimp_dd, axis=tuple(range(1, wloc_dd.ndim)))
        metrics['diff_w'] = float(np.max(np.abs(norm_grid_w)))
        # Only print when screen_type != 'rpa' since diff_w is not expected to converge there
        if Input['screen_type'] != 'rpa':
            coqui.app_log(1, f"  |Wloc_tau - Wimp_tau| (density-density) = {metrics['diff_w']}")

    if prev_u_weiss_iw is not None:
        u_weiss_t      = iaft.w_to_tau_phsym(Input['u_weiss_iw'], stats='b')
        u_weiss_prev_t = iaft.w_to_tau_phsym(prev_u_weiss_iw, stats='b')
        if u_weiss_t.ndim == 3:
            u_weiss_dd      = u_weiss_t
            u_weiss_prev_dd = u_weiss_prev_t
        else:
            u_weiss_dd      = coqui_dmft.product_basis_to_density_density(u_weiss_t)
            u_weiss_prev_dd = coqui_dmft.product_basis_to_density_density(u_weiss_prev_t)
        norm_grid_uw = np.linalg.norm(u_weiss_dd - u_weiss_prev_dd, axis=tuple(range(1, u_weiss_dd.ndim)))
        metrics['diff_u_weiss'] = float(np.max(np.abs(norm_grid_uw)))
        coqui.app_log(1, f"  |u_weiss_tau - u_weiss_prev_tau| (dd)   = {metrics['diff_u_weiss']}")

    coqui.app_log(1, "")
    return metrics


def solve_impurities_from_chkpt(coqui_mpi, *, dmft_iteration=-1, imp_indices=None, 
                                params: dict, save_results=False):
    """
    Re-solve EDMFT impurity problems from a saved checkpoint.

    This helper reads previously stored impurity inputs from ``chkpt_h5`` and
    reruns only the impurity-solver stage, without performing GW updates,
    downfolding, or embedding/upfolding. It accepts the same flat GW+EDMFT
    parameter layout as :func:`run_gw_edmft`, but only the EDMFT/impurity
    subsection is used.

    Parameters
    ----------
    coqui_mpi : coqui.MpiHandler
        MPI handler used for communicator size, barriers, and root-only output.
    dmft_iteration : int, optional
        DMFT iteration to read from the impurity checkpoint.

        - ``-1``: use the latest stored iteration.
        - ``>= 0``: read the specified iteration explicitly.

        Default is ``-1``.
    imp_indices : list[int], optional
        Subset of impurity indices to solve. If ``None``, all impurities stored
        in the checkpoint are processed.
    params : dict
        GW+EDMFT parameter dictionary in the same format as :func:`run_gw_edmft`.
        For the ``edmft`` subsection, including ``edmft.impurity`` solver
        options, refer directly to the docstring of :func:`run_gw_edmft`.

        In this helper, only the EDMFT/impurity part is used. In particular, 
        ``edmft.chkpt_h5`` selects the impurity checkpoint to read, ``edmft.iaft`` 
        may override the impurity DLR mesh, and ``edmft.impurity`` provides 
        the solver settings for the re-solve.

    Returns
    -------
    list[dict]
        List of impurity solver result dictionaries, one per processed impurity,
        containing TRIQS solver outputs together with raw-data conversions such
        as ``Sigma_iw_data``, ``Pi_iw_data``, and ``W_iw_data``.
    """

    # only the edmft stage is used; niter is irrelevant to a re-solve
    params = convert_gw_edmft_params({'niter': 1, **params})
    
    edmft_params = params.pop('edmft')
    imp_iaft_params = edmft_params.pop('iaft', {})

    # one block per impurity: the workflow keys beside the backend's 'solver' table
    impurity_params_list = edmft_params['impurity']
    dmft_chkpt_h5 = edmft_params['chkpt_h5']
    iaft = IAFT.from_coqui_chkpt(dmft_chkpt_h5, verbose=coqui_mpi.root())
    dlr_wmax = imp_iaft_params.get('wmax', iaft.wmax)
    dlr_eps = imp_iaft_params.get('eps', iaft.eps)

    # explicit iteration index: results (and solver state, e.g. CT-SEG configurations) are
    # written back to it
    dmft_iteration = resolve_dmft_iteration(dmft_chkpt_h5, dmft_iteration)

    impurity_inputs = coqui_dmft.read_impurity_chkpt(
        dmft_chkpt_h5, dmft_iteration, read="inputs", impurity_indices=imp_indices
    )
    impurity_results, solver_data = [], []
    for pos, inputs in enumerate(impurity_inputs):
        # read_impurity_chkpt returns only the selected impurities, in the order of imp_indices
        imp_index = imp_indices[pos] if imp_indices is not None else pos
        coqui_dmft.print_title_box(f"IMPURITY {imp_index}")
        imp_params = impurity_params_list[imp_index]
        Input = impurity_inputs[pos]
        # absent in checkpoints written before the solver interface: require the polarization
        screen_type = Input.get('screen_type', 'gw_edmft')
        solver = make_solver(imp_params['solver'], comm_size=coqui_mpi.comm_size())
        check_solver_capabilities_for_gw_edmft(solver=solver, imp_params=imp_params, screen_type=screen_type)
        coqui.app_log(1, f"Impurity {imp_index}: {solver.describe()}\n")

        Input['u_weiss_iw'] = coqui_dmft.fit_u_weiss(
            Input['u_weiss_iw'], iaft, imp_params.get("causal_projection")
        )

        Ub, Ubp, Jb_spin, Jb_pair = coqui_dmft.hubbard_kanamori_coulomb(Input['Vloc'])
        U, Up, J_spin, J_pair = coqui_dmft.hubbard_kanamori_coulomb(Input['Vloc']+Input['u_weiss_iw'][0])
        dm = -iaft.tau_interpolate(
            coqui_dmft.blk_arr_to_arr(Input['Gloc_t'], Input["gf_struct"]),
            iaft.beta, 'f')[0]
        Input['density'] = (np.diag(dm[0]).sum() + np.diag(dm[1]).sum()).real
        Input['occupations_sab'] = np.real(dm)
        coqui.app_log(1, "Bare/static orbital-averaged interactions for the impurity")
        coqui.app_log(1, "----------------------------------------------------------")
        coqui.app_log(1, f"  intra-orbital                  = {Ub*Hartree_eV:.4f}, {U*Hartree_eV:.4f} eV")
        coqui.app_log(1, f"  inter-orbital                  = {Ubp*Hartree_eV:.4f}, {Up*Hartree_eV:.4f} eV")
        coqui.app_log(1, f"  Hund's coupling (spin-flip)    = {Jb_spin*Hartree_eV:.4f}, {J_spin*Hartree_eV:.4f} eV")
        coqui.app_log(1, f"  Hund's coupling (pair-hopping) = {Jb_pair*Hartree_eV:.4f}, {J_pair*Hartree_eV:.4f} eV\n")

        coqui.app_log(1, "Local densities ")
        coqui.app_log(1, "-------------------")
        coqui.app_log(1, f"Total: {Input['density']:.4f}")
        coqui.app_log(1, f"Spin up: {np.diag(dm[0]).real}")
        coqui.app_log(1, f"Spin down: {np.diag(dm[1]).real}\n")

        if coqui_mpi.root():
            iaft.check_leakage(Input['delta_iw'], 'f', 'delta', w_input=True)
            iaft.check_leakage_phsym(Input['u_weiss_iw'], 'b', 'u_weiss', w_input=True)

        # degenerate groups of blocks: the checkpoint's when it stores them, else as the loop
        if Input.get('degenerate_blk') is None:
            Input['degenerate_blk'] = _degenerate_blocks(Input, imp_params, iaft, dlr_wmax, dlr_eps,
                                                         solver)
        problem = ImpurityProblem(
            gf_struct=Input['gf_struct'], beta=iaft.beta, iaft=iaft, dlr_wmax=dlr_wmax,
            dlr_eps=dlr_eps, h0_sab=Input['h0'], delta_wsab=Input['delta_iw'], v_abcd=Input['Vloc'],
            u_weiss_wabcd=Input['u_weiss_iw'], target_density=Input['density'],
            occupations_sab=Input['occupations_sab'], degenerate_blk=Input['degenerate_blk'])
        check_solver_capabilities_for_problem(solver=solver, problem=problem)
        chkpt = {'h5': dmft_chkpt_h5, 'impurity': imp_index, 'iteration': dmft_iteration,
                 'save': save_results}
        result = _solve_impurity(solver, problem, imp_params, chkpt)
        Res = {'gf_struct': Input['gf_struct']}
        Res.update(result.solver_agnostic_results())
        Res['mu_imp'] = problem.mu_shift

        # Causal projection
        coqui_dmft.fit_impurity_results_boson(
            Res, iaft, imp_params.get("causal_projection"))

        impurity_results.append(Res)
        solver_data.append(result.solver_data)

        if save_results:
            # the lists hold only the selected impurities: entry pos goes to impurity_{imp_index}
            coqui_dmft.save_impurity_results(
                impurity_results, dmft_chkpt_h5, iteration=dmft_iteration, impurity_index=pos,
                chkpt_impurity_index=imp_index, solver_data=solver_data
            )

    return impurity_results
