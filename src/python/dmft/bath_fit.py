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

"""
Utility functions for bath fitting 
"""
import triqs.utility.mpi as mpi
import numpy as np
from coqui import app_log


def bath_fitting(A_wsab, iw_mesh, statistics, Np=5,
                 *, name="", iw_mesh_out=None):
    app_log(2, f"Causal projection for {name} with nbath/orbital = {Np} and statistics = {statistics}")
    try:
        from adapol import hybfit as adapol_hybfit
    except ImportError:
        raise ImportError("bath fitting requires the adapol package (https://github.com/flatironinstitute/adapol). \n"
                          "Please ensure that it is installed. ")
    statistics = 'fermion' if statistics == 'f' else statistics
    statistics = 'boson' if statistics == 'b' else statistics
    if statistics not in ("fermion", "boson"):
        raise ValueError(f"Invalid statistics: {statistics!r}. Use 'fermion' or 'boson'.")
    if A_wsab.ndim < 3:
        raise ValueError("A_wsab must be at least 3D: (Nw, norb, norb) or (Nw, nspin, norb, norb).")
    if A_wsab.shape[-1] != A_wsab.shape[-2]:
        raise ValueError("The last two dimensions of A_wsab must be equal (square matrices).")
    if A_wsab.shape[0] != iw_mesh.shape[0]:
        raise ValueError("Mismatch: A_wsab.shape[0] (Nw) != iw_mesh.shape[0].")

    if iw_mesh_out is None:
        iw_mesh_out = iw_mesh

    original_shape = None
    if len(A_wsab.shape) != 4:
        original_shape = A_wsab.shape
        A_wsab = A_wsab.reshape(A_wsab.shape[0], -1, A_wsab.shape[-2], A_wsab.shape[-1])

    nspin, error = A_wsab.shape[1], -1
    A_out = np.zeros((iw_mesh_out.shape[0],) + A_wsab.shape[1:], dtype=A_wsab.dtype)
    for s in np.arange(nspin):
        _, __, fit_error, func = adapol_hybfit(
            A_wsab[:, s], iw_mesh, Np=Np, solver='sdp', verbose=False, statistics=statistics
        )
        A_out[:, s] = func(iw_mesh_out)
        error = max(error, abs(fit_error))
    app_log(2, f"Causal projection error =  {error}\n")

    if original_shape is not None:
        A_wsab = A_wsab.reshape(original_shape)
        A_out = A_out.reshape((iw_mesh_out.shape[0],) + original_shape[1:])

    return A_out


def causal_projection_boson(A_iw, iaft, causal_params, 
                            ph_symmetry=False, w0_regularization=None, target_name=""):
    """
    Perform causal projection for bosonic Green's functions.

    This function fits the input bosonic Green's function `A_iw` to a causal
    representation using bath fitting. It supports optional particle-hole
    symmetry enforcement and allows for customization of the fitting process
    through `causal_params`.

    Parameters:
        A_iw (numpy.ndarray): Input bosonic Green's function data.
        iaft: Kernel object providing the Matsubara frequency mesh and
                   other transformations.
        causal_params (dict): Parameters for the causal projection, including:
                              - 'nbath_per_orbital': Number of bath orbitals.
                              - 'exclude_w0' (optional): Whether to exclude
                                the zero-frequency component.
        ph_symmetry (bool): If True, enforces particle-hole symmetry by
                            setting the imaginary part to zero.
        w0_regularization (optional): Regularization to A(iw=0) before causal projection.
        target_name (str, optional): Name of the target for reporting purposes.
    
    Note: If `causal_params` is None or `causal_params["nbath_per_orbital"]` <= 0, 
          the function returns the input `A_iw` unchanged.

    Returns:
        numpy.ndarray: The fitted bosonic Green's function in causal form.
    """
    if causal_params is None:
        return A_iw

    iw_mesh_b = iaft.wn_mesh('b', positive_only=ph_symmetry) * np.pi / iaft.beta

    if w0_regularization is not None:
        A_iw = apply_w0_regularization(A_iw, iw_mesh_b, w0_regularization, target_name)

    if causal_params["nbath_per_orbital"] <= 0:
        app_log(2, f"Skipping causal projection for {target_name} as nbath_per_orbital <= 0")
        return A_iw

    exclude_w0 = causal_params.get('exclude_w0', False)
    zero_index = np.where(iw_mesh_b == 0.0)[0][0]
    iw_inputs = np.delete(iw_mesh_b, zero_index) if exclude_w0 else iw_mesh_b
    A_iw_input = np.delete(A_iw, zero_index, axis=0) if exclude_w0 else A_iw

    A_iw_fit = bath_fitting(
        A_iw_input, 1j*iw_inputs,
        statistics="boson", name=target_name,
        Np=causal_params["nbath_per_orbital"],
        iw_mesh_out=1j*iw_mesh_b
    )
    if ph_symmetry is True:
        A_iw_fit[:].imag = 0.0

    return A_iw_fit


def fit_impurity_results_boson(imp_res, iaft, causal_params):
    if causal_params is None or causal_params.get("target", "both")=="local":
        return

    fit_res = {}
    if mpi.is_master_node():
        fit_res = {
            "Pi_iw_data": causal_projection_boson(
                imp_res["Pi_iw_data"][0], iaft, causal_params, 
                ph_symmetry=True, w0_regularization=causal_params.get("w0_treatment_for_pi", None), 
                target_name="impurity polarizability"
            ),
            "W_iw_data": causal_projection_boson(
                imp_res["W_iw_data"][0], iaft, causal_params, 
                ph_symmetry=True, w0_regularization=causal_params.get("w0_treatment_for_w", None), 
                target_name="impurity screened interaction"
            )
        }
    fit_res = mpi.bcast(fit_res)
    imp_res["Pi_iw_data"][0] = fit_res["Pi_iw_data"]
    imp_res["W_iw_data"][0] = fit_res["W_iw_data"]


def fit_local_results_boson(local_res, iaft, causal_params):
    if causal_params is None or causal_params.get("target", "both")=="impurity":
        return

    wloc_t_fit = None
    if mpi.is_master_node():
        wloc_iw = iaft.tau_to_w_phsym(local_res["Wloc_t"], 'b')
        nbnd = wloc_iw.shape[-1]
        wloc_iw_fit = causal_projection_boson(
            wloc_iw.reshape(-1, nbnd*nbnd, nbnd*nbnd), iaft, causal_params,
            ph_symmetry=True, w0_regularization=causal_params.get("w0_treatment_for_w", None), 
            target_name="local screened interaction"
        )
        wloc_t_fit = iaft.w_to_tau_phsym(
            wloc_iw_fit.reshape(-1, nbnd, nbnd, nbnd, nbnd), 'b'
        )

    wloc_t_fit = mpi.bcast(wloc_t_fit)
    local_res["Wloc_t"] = wloc_t_fit


def fit_u_weiss(u_weiss_iw, iaft, causal_params):
    if causal_params is None or causal_params.get("target", "both")=="impurity":
        return u_weiss_iw

    u_iw_fit = None
    if mpi.is_master_node():
        nbnd = u_weiss_iw.shape[-1]
        u_iw_fit = causal_projection_boson(
            u_weiss_iw.reshape(-1, nbnd*nbnd, nbnd*nbnd), iaft, causal_params,
            ph_symmetry=True, w0_regularization=causal_params.get("w0_treatment_for_weiss", None), 
            target_name="bosonic Weiss field"
        )
        u_iw_fit = u_iw_fit.reshape(-1, nbnd, nbnd, nbnd, nbnd)

    u_iw_fit = mpi.bcast(u_iw_fit)
    return u_iw_fit


def poly_lstsq_extrapolate(x_data, y_data, fit_order=1, x_target=0.0):
    """
    Fit y(x) with a polynomial of arbitrary order using least squares,
    then evaluate the fitted polynomial at x_target.

    Parameters:
        x_data (array-like): 1D sample locations with shape (nsample,).
        y_data (numpy.ndarray): Sample values with shape (nsample, ...).
                                The first axis must match x_data length.
        fit_order (int): Polynomial order.
        x_target (float): Extrapolation/evaluation point.

    Returns:
        numpy.ndarray: Extrapolated value at x_target with shape y_data.shape[1:].
    """
    x_data = np.asarray(x_data, dtype=np.float64)
    y_data = np.asarray(y_data)

    if x_data.ndim != 1:
        raise ValueError("x_data must be a 1D array.")
    if y_data.ndim < 1:
        raise ValueError("y_data must have at least 1 dimension.")
    if y_data.shape[0] != x_data.shape[0]:
        raise ValueError("y_data.shape[0] must match x_data.shape[0].")
    if fit_order < 0:
        raise ValueError("fit_order must be non-negative.")
    if x_data.shape[0] < fit_order + 1:
        raise ValueError(
            f"Need at least fit_order+1={fit_order+1} samples, got {x_data.shape[0]}."
        )

    # Vandermonde matrix with ascending powers: [1, x, x^2, ...]
    vandermonde = np.vander(x_data, N=fit_order + 1, increasing=True)

    # Flatten all trailing dimensions, solve independent least-squares fits in batch
    y_matrix = y_data.reshape(y_data.shape[0], -1)
    coeffs, *_ = np.linalg.lstsq(vandermonde, y_matrix, rcond=None)

    # Evaluate polynomial at x_target
    x_powers = np.power(float(x_target), np.arange(fit_order + 1, dtype=np.float64))
    y_target = (x_powers[:, None] * coeffs).sum(axis=0)

    return y_target.reshape(y_data.shape[1:])


def apply_w0_regularization(A_iw, iw_mesh_b, w0_regularization, target_name):
    """
    Apply regularization to the zero-frequency component of the bosonic Green's function.

    This function modifies the bosonic Green's function `A_iw` at iw = 0 based on the
    specified `w0_regularization` method. It supports two types of regularization:
    "flatten" and "linear_extrapolate". 
    
    For "flatten", it flattens the value at iw = 0 by setting it equal to the value 
    at the first positive frequency. 
    
    For "linear_extrapolate", it performs a linear extrapolation using the first two positive frequencies 
    to estimate the value at iw = 0.

    Parameters:
        A_iw (numpy.ndarray): Input bosonic Green's function data.
        iw_mesh_b (numpy.ndarray): Bosonic Matsubara frequency mesh.
        w0_regularization (string): Type of regularization to apply ("flatten" or "linear_extrapolate").
        target_name (str): Name of the target for reporting purposes.          

    Returns:
        numpy.ndarray: The modified bosonic Green's function with regularized iw = 0 component.
    """
    app_log(2, f"Applying w=0 regularization for {target_name} before causal projection:")
    
    if w0_regularization == "flatten":
        
        # For insulating case, flatten at iw=0 by setting A(iw=0) = A(iw1)
        app_log(2, f"  --> Flattening {target_name} at w=0.")
        zero_index = np.where(iw_mesh_b == 0.0)[0][0]
        A_iw[zero_index] = A_iw[zero_index + 1]
    
    elif w0_regularization[:18] == "extrapolate_order_":
   
        try:
            order = int(w0_regularization[18:])
        except ValueError:
            raise ValueError(
                f"Invalid extrapolate_order value: {w0_regularization[18:]}. Must be an integer."
            )
        app_log(2, f"  --> Extrapolating {target_name} at w = 0 as an O(w²) polynomial of order {order}.")
        zero_index = np.where(iw_mesh_b == 0.0)[0][0]
        available_points = iw_mesh_b[zero_index+1:].shape[0]
        if order+1 > available_points:
            raise ValueError(
                f"extrapolate_order_{order} requires at least {order+1} positive Matsubara points after w=0 for {target_name}."
            )

        x_data = np.array([np.abs(iw)**2 for iw in iw_mesh_b[zero_index + 1:zero_index + 2 + order]])
        y_data = A_iw[zero_index + 1:zero_index + 2 + order]
        A_iw[zero_index] = poly_lstsq_extrapolate(x_data, y_data, fit_order=order, x_target=0.0)

    else:
        raise ValueError(
            f"Invalid w0_regularization option: {w0_regularization}. "
            "Use 'flatten' or 'extrapolate_order_<n>'."
        )

    return A_iw
