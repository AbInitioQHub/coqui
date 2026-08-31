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

import json

from coqui._lib.mbpt_module import mbpt as mbpt_cxx


def _run_mbpt(solver_type, params, h_int,
             h_int_hf = None, h_int_hartree = None, h_int_exchange = None,
             *, projector_info = None, local_polarizabilities = None):
    args = [solver_type, json.dumps(params), h_int]

    if projector_info is not None:
        ## GW+EDMFT interface with optional local polarizabilities
        if local_polarizabilities is not None:
            required_keys = {"imp", "dc"}
            missing = required_keys - local_polarizabilities.keys()
            if missing:
                raise ValueError(f"Missing keys: {missing}")

        proj_mat = projector_info.get("proj_mat")
        band_window = projector_info.get("band_window")
        kpts_w90 = projector_info.get("kpts_w90")
        mbpt_cxx(*args, proj_mat, band_window, kpts_w90, local_polarizabilities)
    else:
        # Pure MBPT interface without projector info
        if h_int_hf is not None:
            args.append(h_int_hf)
        elif h_int_hartree is not None and h_int_exchange is not None:
            args.extend([h_int_hartree, h_int_exchange])
        elif h_int_hf is None and (h_int_hartree is not None or h_int_exchange is not None):
            raise ValueError("Invalid mbpt input: hartree_eri and exchange_eri must be both provided, or neither.")
        mbpt_cxx(*args)


def run_hf(params, h_int, h_int_exchange=None):
    """
    Run a self-consistent Hartree-Fock (HF) calculation. 

    Results are written to an HDF5 checkpoint file at ``outdir/prefix.mbpt.h5``, 
    which can be used for restarting or post-processing. 

    Parameters
    ----------
    params : dict
        Calculation options. Supported keys:

        - ``outdir`` *(str, optional, default ``"./"``)*  — output directory for the
          HDF5 checkpoint file.
        - ``prefix`` *(str, required)* — prefix for the checkpoint filename 
          (written to ``outdir/prefix.mbpt.h5``).
        - ``beta`` *(float, optional, default ``1000.0``, units: 1/Hartree)* —
          inverse temperature. Large values approximate the zero-temperature limit.
        - ``niter`` *(int, optional, default ``1``)* — maximum number of SCF
          iterations.
        - ``conv_thr`` *(float, optional, default ``1e-8``)* — convergence threshold
          on the self-consistency between iterations.
        - ``restart`` *(bool, optional, default ``False``)* — if ``True``, resumes
          from an existing checkpoint file.
        - ``const_mu`` *(bool, optional, default ``False``)* — if ``True``, fixes the
          chemical potential throughout the SCF loop.
        - ``mu_tolerance`` *(float, optional, default ``1e-9``)* — tolerance for the
          chemical potential search.
        - ``div_treatment`` *(str, optional, default ``"gygi"``)* — treatment of the
          Coulomb kernel divergence at q→0. Common choices:

          - ``"gygi"`` *(recommended)* — polynomial extrapolation of ε⁻¹ to q=0
            along the reciprocal lattice directions. The default maximum extrapolation 
            order is 10. Append ``"_order_N"`` to set the polynomial fit order manually 
            (e.g. ``"gygi_order_4"``).
          - ``"gygi_smallest_q"`` — approximates q=0 using the smallest finite
            |q| point; this was the behavior of ``"gygi"`` in earlier versions.
          - ``"ignore_g0"`` — identical to ``"gygi_smallest_q"``; used
            automatically when only one q-point is available.

          Suffix modifiers (combinable, e.g. ``"gygi_2d"`` or ``"gygi_metal"``):

          - ``"_2d"`` — restricts the extrapolation to the xy-plane; use for 2D
            materials.
          - ``"_metal"`` — enforces ε⁻¹(q=0, ω=0) → 0; use for metallic systems.

        - ``hf_div_treatment`` *(str, optional, default ``"gygi"``)* — divergence
          treatment for the HF Coulomb kernel specifically. Accepts the same values
          and suffix modifiers as ``div_treatment``.
        - ``greens_func_source`` *(str, optional, default ``"scf"``)* — source of the
          initial Green's function. ``"mf"`` uses the mean-field Green's function;
          ``"scf"`` reads from the checkpoint.
        - ``greens_func_iteration`` *(int, optional, default ``-1``)* — checkpoint
          iteration to read the Green's function from. ``-1`` selects the latest.
        - ``iaft`` *(dict, optional)* — imaginary-axis frequency-grid settings:

          - ``wmax`` *(float, optional, units: Hartree)* — maximum frequency in Hartree. If omitted,
            estimated automatically from the mean-field bandwidth.
          - ``prec`` *(str, optional, default ``"medium"``)* — target numerical
            precision: ``"high"`` (~1e-15), ``"medium"`` (~1e-10), ``"low"`` (~1e-6).
          - ``basis`` *(str, optional, default ``"dlr"``)* — frequency basis.
            ``"dlr"`` (Discrete Lehmann Representation) or ``"ir"`` (intermediate
            representation).
          - ``eps`` *(float, optional)* — explicit accuracy target; alternative to
            ``prec``.

        - ``iter_alg`` *(dict, optional)* — SCF mixing algorithm settings:

          - ``alg`` *(str, optional, default ``"diis"``)* — mixing algorithm.
            ``"damping"`` (linear mixing) or ``"diis"`` (Pulay DIIS).
          - ``mixing`` *(float, optional, default ``0.7``)* — mixing fraction in
            ``[0, 1]`` for damping; smaller values give more conservative updates.
          - ``max_subsp_size`` *(int, optional, default ``5``)* — maximum DIIS
            subspace size (ignored for ``"damping"``).
          - ``residual_type`` *(str, optional, default ``"commutator"``)* — residual
            definition for DIIS. ``"commutator"`` or ``"vector_diff"``.

    h_int : ThcCoulomb or CholCoulomb
        Coulomb interaction object used for both the Hartree and exchange terms.
        Obtained from ``make_thc_coulomb`` or ``make_chol_coulomb``.
    h_int_exchange : ThcCoulomb or CholCoulomb, optional
        If provided, used for the exchange term only; ``h_int`` is then used for
        the Hartree term only. Useful when Hartree and exchange are computed with
        different interaction objects 

    Returns
    -------
    None
        Results are written to ``outdir/prefix.mbpt.h5``.

    Examples
    --------
    ::

        from coqui.mbpt import run_hf

        run_hf(
            {"beta": 300, "niter": 10, "prefix": "svo.hf",
             "iaft": {"prec": "medium"},
             "iter_alg": {"alg": "damping", "mixing": 0.7}},
            h_int=thc,
        )
    """
    args = ["hf", json.dumps(params), h_int]
    if h_int_exchange is not None:
        args.append(h_int_exchange)
    mbpt_cxx(*args)


def run_gw(params, h_int,
           h_int_hf = None, h_int_hartree = None, h_int_exchange = None,
           *, projector_info = None, local_polarizabilities = None):
    """
    Run a full-frequency self-consistent GW calculation.

    Computes the GW self-energy using the screened Coulomb interaction W and
    writes results to ``outdir/prefix.mbpt.h5``. Set ``niter=1`` for G0W0.

    Parameters
    ----------
    params : dict
        Accepts all keys documented in ``run_hf``, plus:

        - ``screen_type`` *(str, optional, default ``"rpa"``)* — approximation used
          to build the screened interaction W. Common choices:

          - ``"rpa"`` — random-phase approximation; standard choice for GW.
          - ``"crpa"`` — constrained RPA; excludes selected (e.g. correlated) bands
            from the polarizability, typically used when constructing low-energy
            downfolded models. This requires ``projector_info`` to be provided for 
            defining the correlated subspace. 
          - ``"gw_edmft"`` — GW+EDMFT; requires ``projector_info`` and
            ``local_polarizabilities`` to be provided so that the local EDMFT
            polarizability is added on top of the RPA contribution within the 
            correlated subspace.
        - ``dump_w_to_h5`` *(bool, optional, default ``False``)* — if ``True``,
          saves the screened interaction W to the checkpoint file.

    h_int : ThcCoulomb or CholCoulomb
        Primary Coulomb interaction object, used for both the polarizability and
        the GW self-energy when no split ERI arguments are provided.
    h_int_hf : ThcCoulomb or CholCoulomb, optional
        If provided, used for the HF (Hartree + exchange) part of the self-energy
        instead of ``h_int``. Mutually exclusive with ``h_int_hartree`` /
        ``h_int_exchange``.
    h_int_hartree : ThcCoulomb or CholCoulomb, optional
        Coulomb object for the Hartree channel. Must be paired with
        ``h_int_exchange``; cannot be combined with ``h_int_hf``.
    h_int_exchange : ThcCoulomb or CholCoulomb, optional
        Coulomb object for the exchange channel. Must be paired with
        ``h_int_hartree``; cannot be combined with ``h_int_hf``.
    projector_info : dict, optional
        Wannier projector data for defining corrleated subspace. Required keys:
        ``"proj_mat"``, ``"band_window"``, ``"kpts_w90"``. 
    local_polarizabilities : dict, optional
        Local polarizabilities for local EDMFT corrections. 
        Required keys: ``"imp"`` (impurity) and ``"dc"`` (double-counting). 
        Only used together with ``projector_info``.

    Returns
    -------
    None
        Results are written to ``outdir/prefix.mbpt.h5``.

    Examples
    --------
    ::

        from coqui.mbpt import run_gw

        run_gw(
            {"beta": 200, "niter": 1, "prefix": "svo.gw",
             "iaft": {"prec": "medium"},
             "screen_type": "rpa",
             "div_treatment": "gygi"},
            h_int=thc,
        )
    """
    _run_mbpt("gw", params, h_int,
              h_int_hf = h_int_hf, h_int_hartree = h_int_hartree, h_int_exchange = h_int_exchange,
              projector_info = projector_info, local_polarizabilities = local_polarizabilities)


def run_evgw(params, h_int,
               h_int_hf = None, h_int_hartree = None, h_int_exchange = None):
    """
    Run an eigenvalue-only self-consistent GW (evGW or evGW0) calculation.

    Updates quasiparticle energies self-consistently while keeping the
    wavefunctions fixed at the DFT level. 

    Parameters
    ----------
    params : dict
        Accepts all keys documented in ``run_hf``, plus:

        - ``keep_scr_coulomb_fixed`` *(bool, optional, default ``False``)* — if
          ``True``, the screened Coulomb interaction W is held fixed at the 
          first iteration (evGW0); if ``False``, W is updated each iteration (evGW).
        - ``ac_alg`` *(str, optional, default ``"pade"``)* — analytic continuation
          algorithm used to extract quasiparticle energies from the Matsubara
          self-energy. Currently only ``"pade"`` is implemented. 
        - ``eta`` *(float, optional, default ``π/beta``, units: Hartree)* —
          broadening for the analytic continuation.
        - ``Nfit`` *(int, optional, default ``18``)* — number of Matsubara
          frequencies used for the Padé fit. Set ``-1`` to use all Matsubara frequencies 
          on a DLR mesh. 

    h_int : ThcCoulomb or CholCoulomb
        Primary Coulomb interaction object.
    h_int_hf : ThcCoulomb or CholCoulomb, optional
        If provided, used for the HF part instead of ``h_int``. Mutually
        exclusive with ``h_int_hartree`` / ``h_int_exchange``.
    h_int_hartree : ThcCoulomb or CholCoulomb, optional
        Coulomb object for the Hartree channel. Must be paired with
        ``h_int_exchange``.
    h_int_exchange : ThcCoulomb or CholCoulomb, optional
        Coulomb object for the exchange channel. Must be paired with
        ``h_int_hartree``.

    Returns
    -------
    None
        Results are written to ``outdir/prefix.mbpt.h5``.

    Examples
    --------
    ::

        from coqui.mbpt import run_evgw

        run_evgw(
            {"beta": 200, "niter": 5, "prefix": "svo.evgw",
             "iaft": {"prec": "medium"},
             "keep_scr_coulomb_fixed": False},
            h_int=thc,
        )
    """
    _run_mbpt("evgw", params, h_int,
              h_int_hf = h_int_hf, h_int_hartree = h_int_hartree, h_int_exchange = h_int_exchange,
              projector_info = None, local_polarizabilities = None)


def run_qpgw(params, h_int,
               h_int_hf = None, h_int_hartree = None, h_int_exchange = None):
    """
    Run a quasiparticle self-consistent GW (qpGW) calculation.

    Updates both quasiparticle energies and wavefunctions self-consistently by
    constructing a Hermitian, energy-independent effective Hamiltonian from the
    GW self-energy at each iteration.

    Parameters
    ----------
    params : dict
        Accepts all keys documented in ``run_hf``, plus:

        - ``ac_alg`` *(str, optional, default ``"pade"``)* — analytic continuation
          algorithm used to extract quasiparticle energies from the Matsubara
          self-energy.
        - ``eta`` *(float, optional, default ``π/beta``, units: Hartree)* —
          broadening for the analytic continuation.
        - ``Nfit`` *(int, optional, default ``18``)* — number of Matsubara
          frequencies used for the Padé fit. Set ``-1`` to use all Matsubara 
          frequencies on a DLR mesh. 
        - ``off_diag_mode`` *(str, optional, default ``"fermi"``)* — frequency at
          which off-diagonal self-energy matrix elements are evaluated when
          constructing the qpGW Hamiltonian. ``"fermi"`` evaluates at the Fermi
          level; ``"qp_energy"`` evaluates at the quasiparticle energy of each
          state.

    h_int : ThcCoulomb or CholCoulomb
        Primary Coulomb interaction object.
    h_int_hf : ThcCoulomb or CholCoulomb, optional
        If provided, used for the HF part instead of ``h_int``. Mutually
        exclusive with ``h_int_hartree`` / ``h_int_exchange``.
    h_int_hartree : ThcCoulomb or CholCoulomb, optional
        Coulomb object for the Hartree channel. Must be paired with
        ``h_int_exchange``.
    h_int_exchange : ThcCoulomb or CholCoulomb, optional
        Coulomb object for the exchange channel. Must be paired with
        ``h_int_hartree``.

    Returns
    -------
    None
        Results are written to ``outdir/prefix.mbpt.h5``.

    Examples
    --------
    ::

        from coqui.mbpt import run_qpgw

        run_qpgw(
            {"beta": 200, "niter": 5, "prefix": "svo.qpgw",
             "iaft": {"prec": "medium"},
             "off_diag_mode": "fermi"},
            h_int=thc,
        )
    """
    _run_mbpt("qpgw", params, h_int,
              h_int_hf = h_int_hf, h_int_hartree = h_int_hartree, h_int_exchange = h_int_exchange,
              projector_info = None, local_polarizabilities = None)
