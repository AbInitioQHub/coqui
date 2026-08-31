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

from .iaft_sparse_ir import _IAFTIRAdapter

try:
    from coqui._lib.iaft_module import IAFT as IAFTCpp
    from coqui._lib.iaft_module import build_g_tau_ref, build_g_iw_ref
    from coqui._lib.iaft_module import string_to_basis_enum
    _IAFT_CPP_IMPORT_ERROR = None
except ImportError as exc:
    IAFTCpp = None
    build_g_tau_ref = None
    build_g_iw_ref = None
    string_to_basis_enum = None
    _IAFT_CPP_IMPORT_ERROR = exc

"""
Kernel for Fourier transform on the imaginary axis, supporting both IR and DLR basis.
"""


def _validate_accuracy_args(basis: str, prec, eps):
    if basis not in ("dlr", "ir"):
        raise ValueError("basis must be 'dlr' or 'ir', got '{}'".format(basis))

    if prec is not None:
        _validate_prec_string(prec)

    if eps is not None:
        if isinstance(eps, bool) or not isinstance(eps, (int, float, np.integer, np.floating)):
            raise TypeError("'eps' must be a floating-point value when provided.")
        if float(eps) <= 0.0:
            raise ValueError("'eps' must be positive.")

    if prec is None and eps is None:
        raise ValueError("IAFT requires at least one of 'prec' or 'eps'.")

    if prec == "custom" and eps is None:
        raise ValueError("`eps` must be provided when prec='custom'.")


def _validate_prec_string(precision):
    if not isinstance(precision, str):
        raise TypeError("Precision must be a string.")
    if precision not in ("high", "medium", "low", "custom"):
        raise ValueError("Unknown precision value: {}. ".format(precision))

def _normalize_stats(stats_input: str) -> str:
    """
    Normalize stats input to C++ string form.
    Accepts both short forms ('f', 'b') and long forms ('fermion', 'boson').

    :param stats_input: str
    Statistics string ('f', 'fermion', 'b', or 'boson')
    :return: str
    Normalized C++ form ('fermion' or 'boson')
    :raises ValueError: if stats_input is not a recognized statistics format
    """
    if stats_input == 'f' or stats_input == 'fermion':
        return 'fermion'
    elif stats_input == 'b' or stats_input == 'boson':
        return 'boson'
    else:
        raise ValueError(
            "Unknown statistics '{}'. Acceptable options are 'f'/'fermion' for fermions "
            "and 'b'/'boson' for bosons.".format(stats_input))


class _IAFTCppAdapter(object):
    """
    Internal implementation class wrapping the C++ ``imag_axes_ft::IAFT`` binding.

    Supports both DLR (Discrete Lehmann Representation) and IR (Intermediate
    Representation) backends via the C++ library. Use ``IAFT(basis='dlr')`` for
    the public-facing API; do not instantiate this class directly. 

    Accuracy policy:
        - If only ``eps`` is given (prec is None or 'custom'), the basis is
          built with that explicit accuracy target.
        - If only ``prec`` is given, the preset eps is used.
        - If both are given and prec != 'custom', prec takes priority.

    Attributes
    ----------
    beta : float
        Inverse temperature
    wmax : float
        Frequency cutoff
    lmbda : float
        Dimensionless parameter lmbda = beta * wmax.
    prec : str
        Active precision label returned by the C++ backend.
    eps : float
        Active accuracy target returned by the C++ backend.
    nt_f, nw_f : int
        Number of fermionic tau / Matsubara sampling points.
    nt_b, nw_b : int
        Number of bosonic tau / Matsubara sampling points.
    """
    def __init__(self, beta: float, wmax: float, *, prec=None, eps=None, verbose: bool=True, basis: str="dlr"):
        """
        :param beta: float
            Inverse temperature (a.u.)
        :param wmax: float
            Frequency cutoff (a.u.)
        :param prec: str or None
            Precision label for basis
        :param eps: float or None
            Explicit DLR accuracy target
        :param verbose: bool
            Print metadata (currently unused; kept for API compatibility)
        :param basis: str
            C++ backend basis, one of {"dlr", "ir"}
        """

        if IAFTCpp is None or string_to_basis_enum is None:
            raise ImportError(
                "DLR backend requires IAFTCpp binding from CoQuí C++ library. "
                "Please rebuild CoQuí with 'COQUI_UPDATE_PYTHON_BINDINGS=ON'. "
            ) from _IAFT_CPP_IMPORT_ERROR

        self.basis = basis
        self.beta = beta
        self.lmbda = wmax * beta
        self.wmax = wmax

        # Create C++ IAFT instance with selected C++ basis backend
        build_with_eps = (eps is not None) and (prec is None or prec == "custom")
        if build_with_eps:
            self._iaft_cpp = IAFTCpp(
                self.beta, self.wmax, string_to_basis_enum(self.basis),
                float(eps), verbose
            )
        else:
            _validate_prec_string(prec)
            self._iaft_cpp = IAFTCpp(
                self.beta, self.wmax, string_to_basis_enum(self.basis),
                prec, verbose
            )

        self.prec = self._iaft_cpp.prec()
        self.eps = self._iaft_cpp.eps()

        # Cache mesh properties from C++ backend
        self._tau_mesh_f = self._iaft_cpp.tau_mesh_f()
        self._tau_mesh_b = self._iaft_cpp.tau_mesh_b()
        self._wn_mesh_f = self._iaft_cpp.wn_mesh_f()
        self._wn_mesh_b = self._iaft_cpp.wn_mesh_b()
        self.nt_f, self.nw_f = self._tau_mesh_f.shape[0], self._wn_mesh_f.shape[0]
        self.nt_b, self.nw_b = self._tau_mesh_b.shape[0], self._wn_mesh_b.shape[0]

    # Expose C++ backend matrices as new numpy arrays
    @property
    def Ttw_ff(self):
        """ndarray, shape (nt_f, nw_f): fermionic frequency-to-tau transform matrix."""
        return np.array(self._iaft_cpp.Ttw_ff())

    @property
    def Twt_ff(self):
        """ndarray, shape (nw_f, nt_f): fermionic tau-to-frequency transform matrix."""
        return np.array(self._iaft_cpp.Twt_ff())

    @property
    def Ttw_bb(self):
        """ndarray, shape (nt_b, nw_b): bosonic frequency-to-tau transform matrix."""
        return np.array(self._iaft_cpp.Ttw_bb())

    @property
    def Twt_bb(self):
        """ndarray, shape (nw_b, nt_b): bosonic tau-to-frequency transform matrix."""
        return np.array(self._iaft_cpp.Twt_bb())

    @property
    def Ttt_bf(self):
        """ndarray: tau-to-tau transform matrix from fermionic to bosonic mesh."""
        return np.array(self._iaft_cpp.Ttt_bf())

    @property
    def Ttt_fb(self):
        """ndarray: tau-to-tau transform matrix from bosonic to fermionic mesh."""
        return np.array(self._iaft_cpp.Ttt_fb())

    @property
    def T_beta_t_ff(self):
        """ndarray, shape (1, nt_f): row vector evaluating the basis at tau=beta (fermionic)."""
        return np.array(self._iaft_cpp.T_beta_t_ff())

    @property
    def T_zero_t_ff(self):
        """ndarray, shape (1, nt_f): row vector evaluating the basis at tau=0 (fermionic)."""
        return np.array(self._iaft_cpp.T_zero_t_ff())

    @property
    def Tct_ff(self):
        """ndarray: fermionic tau-to-coefficient transform matrix."""
        return np.array(self._iaft_cpp.Tct_ff())

    @property
    def Tct_bb(self):
        """ndarray: bosonic tau-to-coefficient transform matrix."""
        return np.array(self._iaft_cpp.Tct_bb())

    def save(self, h5_grp):
        """
        Save basis parameters to an HDF5 group under the key
        ``'imaginary_fourier_transform'``. No-op if the key already exists.

        :param h5_grp: h5py/HDFArchive group
            Target HDF5 group to write into.
        """
        if 'imaginary_fourier_transform' in h5_grp:
            return
        h5_grp.create_group('imaginary_fourier_transform')
        h5_grp['imaginary_fourier_transform']['beta'] = self.beta
        h5_grp['imaginary_fourier_transform']['wmax'] = self.wmax
        h5_grp['imaginary_fourier_transform']['prec'] = self.prec
        h5_grp['imaginary_fourier_transform']['eps'] = self.eps
        h5_grp['imaginary_fourier_transform']['basis'] = self.basis

    def __str__(self):
        self._iaft_cpp.metadata_log()
        return ""

    def __eq__(self, other):
        return (
            isinstance(other, _IAFTCppAdapter) and
            self.beta == other.beta and
            self.wmax == other.wmax and
            self.lmbda == other.lmbda and
            self.eps == other.eps and
            self.basis == other.basis
        )

    def tau_mesh(self, stats: str, rel_notation: bool=False):
        """
        Return imaginary-time sampling points.

        :param stats: str
            Statistics: 'f'/'fermion' for fermions, 'b'/'boson' for bosons.
        :param rel_notation: bool
            If True, return points in the rescaled coordinate t ∈ [−1, 1].
            If False (default), return physical imaginary time τ ∈ [0, β].
        :return: numpy.ndarray of float, shape (nt,)
        """
        stats = _normalize_stats(stats)
        if rel_notation:
            return np.array(self._tau_mesh_f, dtype=float) if stats=="fermion" else np.array(self._tau_mesh_b, dtype=float)
        else:
            tau_mesh = np.array(self._tau_mesh_f, dtype=float) if stats=="fermion" else np.array(self._tau_mesh_b, dtype=float)
            return (tau_mesh + 1.0) * self.beta / 2.0

    def wn_mesh(self, stats: str, *, phys_notation: bool=False, positive_only=False):
        """
        Return Matsubara-frequency sampling indices.

        :param stats: str
            Statistics: 'f'/'fermion' for fermions, 'b'/'boson' for bosons.
        :param phys_notation: bool, keyword-only
            If False (default), use the CoQuí convention iω_n = n·π/β (n odd for
            fermions, even for bosons). If True, use the physical convention
            iω_n = (2n+1)·π/β for fermions and 2n·π/β for bosons.
        :param positive_only: bool, keyword-only
            If True, return only the non-negative half of the mesh.
        :return: numpy.ndarray of int, shape (nw,) or (nw//2,)
        """
        stats = _normalize_stats(stats)
        wn_mesh = np.array(self._wn_mesh_f, dtype=int) if stats=="fermion" else np.array(self._wn_mesh_b, dtype=int)
        if phys_notation:
            wn_mesh = (wn_mesh-1)//2 if stats=="fermion" else wn_mesh//2
        if positive_only:
            nw_half = wn_mesh.shape[0]//2
            return wn_mesh[nw_half:]
        else:
            return wn_mesh

    def tau_to_w(self, Ot, stats: str):
        """
        Fourier transform from imaginary-time axis to Matsubara-frequency axis.

        :param Ot: numpy.ndarray, shape (nt, ...)
            Array on the imaginary-time mesh.
        :param stats: str
            Statistics: 'f'/'fermion' or 'b'/'boson'.
        :return: numpy.ndarray, shape (nw, ...)
        """
        stats = _normalize_stats(stats)
        Ot_shape = Ot.shape
        Ot_2d = Ot.reshape(Ot.shape[0], -1)
        Ow_2d = self._iaft_cpp.tau_to_w_2d(Ot_2d, stats)
        Ow = Ow_2d.reshape((Ow_2d.shape[0],) + Ot_shape[1:])
        return Ow

    def w_to_tau(self, Ow, stats: str):
        """
        Fourier transform from Matsubara-frequency axis to imaginary-time axis.

        :param Ow: numpy.ndarray, shape (nw, ...)
            Array on the Matsubara-frequency mesh.
        :param stats: str
            Statistics: 'f'/'fermion' or 'b'/'boson'.
        :return: numpy.ndarray, shape (nt, ...)
        """
        stats = _normalize_stats(stats)
        Ow_shape = Ow.shape
        Ow_2d = Ow.reshape(Ow.shape[0], -1)
        Ot_2d = self._iaft_cpp.w_to_tau_2d(Ow_2d, stats)
        Ot = Ot_2d.reshape((Ot_2d.shape[0],) + Ow_shape[1:])
        return Ot

    def tau_to_w_phsym(self, Ot, stats: str):
        """
        Particle-hole-symmetric Fourier transform from imaginary-time to Matsubara-frequency.
        Bosons only; input is the first half of the tau mesh.

        :param Ot: numpy.ndarray, shape (nt//2, ...)
        :param stats: str
            Must be 'b' or 'boson'.
        :return: numpy.ndarray, shape (nw//2, ...)
        :raises ValueError: if stats is not bosonic.
        """
        stats = _normalize_stats(stats)
        if stats != 'boson':
            raise ValueError("FT w/ particle-hole symmetry only support bosonic correlation functions")
        Ot_shape = Ot.shape
        Ot_2d = Ot.reshape(Ot.shape[0], -1)
        Ow_2d = self._iaft_cpp.tau_to_w_phsym_2d(Ot_2d)
        Ow = Ow_2d.reshape((Ow_2d.shape[0],) + Ot_shape[1:])
        return Ow

    def w_to_tau_phsym(self, Ow, stats: str):
        """
        Particle-hole-symmetric inverse Fourier transform from Matsubara-frequency to imaginary-time.
        Bosons only; input covers positive Matsubara frequencies only.

        :param Ow: numpy.ndarray, shape (nw//2, ...)
        :param stats: str
            Must be 'b' or 'boson'.
        :return: numpy.ndarray, shape (nt//2, ...)
        :raises ValueError: if stats is not bosonic.
        """
        stats = _normalize_stats(stats)
        if stats != 'boson':
            raise ValueError("FT w/ particle-hole symmetry only support bosonic correlation functions")
        Ow_shape = Ow.shape
        Ow_2d = Ow.reshape(Ow.shape[0], -1)
        Ot_2d = self._iaft_cpp.w_to_tau_phsym_2d(Ow_2d)
        Ot = Ot_2d.reshape((Ot_2d.shape[0],) + Ow_shape[1:])
        return Ot

    def w_interpolate(self, Ow, target, stats: str, *, phys_notation: bool=False, ph_sym: bool=False):
        """
        Interpolate a Matsubara-frequency object to arbitrary frequency indices.

        :param Ow: numpy.ndarray, shape (nw, ...)
            Array on the native Matsubara-frequency mesh (or positive half when ph_sym=True).
        :param target: array-like of int or _IAFTCppAdapter
            Target Matsubara indices. If an adapter instance is passed, its wn_mesh is used.
        :param stats: str
            Statistics: 'f'/'fermion' or 'b'/'boson'.
        :param phys_notation: bool, keyword-only
            Index convention for target (see wn_mesh). Default False.
        :param ph_sym: bool, keyword-only
            Apply particle-hole-symmetric interpolation (bosons only). Default False.
        :return: numpy.ndarray, shape (len(target), ...)
        """
        if isinstance(target, _IAFTCppAdapter):
            wn_mesh = target.wn_mesh(stats, phys_notation=phys_notation, positive_only=ph_sym)
            return self._w_interpolate(Ow, wn_mesh, stats, ph_sym=ph_sym, phys_notation=phys_notation)
        else:
            return self._w_interpolate(Ow, target, stats, ph_sym=ph_sym, phys_notation=phys_notation)

    def w_interpolate_phsym(self, Ow, target, stats: str, *, phys_notation: bool=False):
        """
        Particle-hole-symmetric Matsubara interpolation. Delegates to w_interpolate with ph_sym=True.

        :param Ow: numpy.ndarray, shape (nw//2, ...)
        :param target: array-like of int or _IAFTCppAdapter
        :param stats: str
        :param phys_notation: bool, keyword-only
        :return: numpy.ndarray, shape (len(target), ...)
        """
        return self.w_interpolate(Ow, target, stats, phys_notation=phys_notation, ph_sym=True)

    def _w_interpolate(self, Ow, wn_mesh_interp, stats: str, *, ph_sym: bool=False, phys_notation: bool=False):
        if isinstance(wn_mesh_interp, int):
            wn_mesh_interp = np.array([wn_mesh_interp], dtype=int)

        stats = _normalize_stats(stats)

        # convert to ir notation if needed
        if phys_notation:
            wn_mesh_interp = (2*wn_mesh_interp+1) if stats=="fermion" else (2*wn_mesh_interp)

        T_ww = self._iaft_cpp.construct_w_interpolate_matrix(wn_mesh_interp, stats, ph_sym=ph_sym)
        if Ow.shape[0] != T_ww.shape[1]:
            raise ValueError(
                "w_interpolate: Number of w points are inconsistent: {} and {}".format(
                Ow.shape[0], T_ww.shape[1]))

        Ow_shape = Ow.shape
        Ow_2d = Ow.reshape(Ow.shape[0], -1)
        Ow_interp = np.dot(T_ww, Ow_2d)
        Ow_interp = Ow_interp.reshape((Ow_interp.shape[0],) + Ow_shape[1:])
        return Ow_interp

    def tau_interpolate(self, Ot, target, stats: str, *, rel_notation: bool=False, ph_sym: bool=False):
        """
        Interpolate an imaginary-time object to arbitrary tau points.

        :param Ot: numpy.ndarray, shape (nt, ...)
            Array on the native imaginary-time mesh (or first half when ph_sym=True).
        :param target: float, array-like of float, or _IAFTCppAdapter
            Target tau points. If an adapter instance is passed, its tau_mesh is used.
        :param stats: str
            Statistics: 'f'/'fermion' or 'b'/'boson'.
        :param rel_notation: bool, keyword-only
            Coordinate convention for target (see tau_mesh). Default False.
        :param ph_sym: bool, keyword-only
            Apply particle-hole-symmetric interpolation (bosons only). Default False.
        :return: numpy.ndarray, shape (len(target), ...)
        """
        if isinstance(target, _IAFTCppAdapter):
            tau_mesh = target.tau_mesh(stats, rel_notation=rel_notation)  # get tau mesh in current notation
            if ph_sym:
                nt_half = tau_mesh.shape[0] // 2 + tau_mesh.shape[0] % 2
                tau_mesh = tau_mesh[:nt_half]
            return self._tau_interpolate(Ot, tau_mesh, rel_notation=rel_notation, ph_sym=ph_sym)
        else:
            return self._tau_interpolate(Ot, target, rel_notation=rel_notation, ph_sym=ph_sym)

    def tau_interpolate_phsym(self, Ot, target, stats: str, *, rel_notation: bool=False):
        """
        Particle-hole-symmetric tau interpolation. Delegates to tau_interpolate with ph_sym=True.

        :param Ot: numpy.ndarray, shape (nt//2, ...)
        :param target: float, array-like of float, or _IAFTCppAdapter
        :param stats: str
        :param rel_notation: bool, keyword-only
        :return: numpy.ndarray, shape (len(target), ...)
        """
        return self.tau_interpolate(Ot, target, stats, rel_notation=rel_notation, ph_sym=True)

    def _tau_interpolate(self, Ot, tau_mesh_interp, *, rel_notation: bool=False, ph_sym: bool=False):
        if np.isscalar(tau_mesh_interp):
            try:
                tau_mesh_interp = np.array([tau_mesh_interp], dtype=float)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "tau_mesh_interp must be a number, a list of numbers, or a 1D numpy array of numbers."
                ) from exc
        elif isinstance(tau_mesh_interp, (list, tuple, np.ndarray)):
            try:
                tau_mesh_interp = np.asarray(tau_mesh_interp, dtype=float)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "tau_mesh_interp must be a number, a list of numbers, or a 1D numpy array of numbers."
                ) from exc
        else:
            raise TypeError(
                "tau_mesh_interp must be a number, a list of numbers, or a 1D numpy array of numbers."
            )

        if tau_mesh_interp.ndim != 1:
            raise ValueError(
                "tau_mesh_interp must be one-dimensional, got shape {}.".format(tau_mesh_interp.shape)
            )

        if not rel_notation:
            # convert to [-1, 1] notation for IAFTCpp
            tau_mesh_interp = tau_mesh_interp * 2.0 / self.beta - 1.0

        T_tt = self._iaft_cpp.construct_tau_interpolate_matrix(tau_mesh_interp, ph_sym=ph_sym)
        if Ot.shape[0] != T_tt.shape[1]:
            raise ValueError(
                "tau_interpolate: Number of tau points are inconsistent: {} and {}".format(Ot.shape[0], T_tt.shape[1]))

        Ot_shape = Ot.shape
        Ot_2d = Ot.reshape(Ot.shape[0], -1)
        Ot_interp = np.dot(T_tt, Ot_2d)

        Ot_interp = Ot_interp.reshape((Ot_interp.shape[0],) + Ot_shape[1:])
        return Ot_interp

    def check_leakage(self, Ot, stats: str, name: str="", w_input: bool=False):
        """
        Check decay of the C++-backend coefficients to assess basis quality.
        Intentionally no operation until C++ leakage diagnostics in DLR are implemented.
        """
        return None

    def check_leakage_phsym(self, Ot, stats: str, name: str="", w_input: bool=False):
        """
        Check decay of the C++-backend coefficients w/ particle-hole symmetry.
        Intentionally no operation until C++ leakage diagnostics in DLR are implemented.
        """
        return None


class IAFT(object):
    """
    Imaginary-axis Fourier transform (IAFT) for dynamical correlation functions.

    The `IAFT` class provides controlled-accuracy transformations between the 
    imaginary-time axis τ ∈ [0, β] and the Matsubara-frequency axis iω_n for 
    fermionic and bosonic correlation functions. The transformation are performed 
    using compact basis representations, either the Intermediate Representation (IR) 
    or the Discrete Lehmann Representation (DLR) basis.

    Parameters
    ----------
    beta : float
        Inverse temperature
    wmax : float
        Frequency cutoff. Sets the energy scale of the basis.
    prec : {'high', 'medium', 'low', 'custom'} or None
        Named accuracy preset. Approximate epsilon values:

        - ``'high'``   → eps ≈ 1e-15
        - ``'medium'`` → eps ≈ 1e-10
        - ``'low'``    → eps ≈ 1e-6
        - ``'custom'`` → use the value given in ``eps``

        Must be provided when ``eps`` is None.
    eps : float or None
        Explicit accuracy target (positive float). Required when ``prec='custom'``.
        If both ``prec`` and ``eps`` are given and ``prec != 'custom'``, ``prec``
        takes priority.
    verbose : bool, optional
        Print basis metadata on initialization. Default ``True``.
    basis : {'dlr', 'ir'}
        Compressed-basis backend:

        - ``'dlr'`` – Discrete Lehmann Representation via the C++ ``IAFTCpp``
          binding (default).
        - ``'ir'``  – Intermediate Representation via the ``sparse_ir`` library.

    Raises
    ------
    ValueError
        If ``basis`` is not ``'ir'`` or ``'dlr'``, or if the precision arguments
        are inconsistent.
    ImportError
        If the required backend library is not available.

    Attributes
    ----------
    basis : str
        Active backend: ``'ir'`` or ``'dlr'``.
    beta : float
        Inverse temperature
    wmax : float
        Frequency cutoff.
    lmbda : float
        Dimensionless parameter lmbda = beta * wmax.
    prec : str
        Active precision label.
    eps : float
        Active accuracy target.
    nt_f : int
        Number of fermionic imaginary-time sampling points.
    nt_b : int
        Number of bosonic imaginary-time sampling points.
    nw_f : int
        Number of fermionic Matsubara-frequency sampling points.
    nw_b : int
        Number of bosonic Matsubara-frequency sampling points.
    Ttw_ff : numpy.ndarray, shape (nt_f, nw_f)
        Transformation matrix mapping fermionic Matsubara frequencies to imaginary time.
    Twt_ff : numpy.ndarray, shape (nw_f, nt_f)
        Transformation matrix mapping fermionic imaginary time to Matsubara frequencies.
    Ttw_bb : numpy.ndarray, shape (nt_b, nw_b)
        Transformation matrix mapping bosonic Matsubara frequencies to imaginary time.
    Twt_bb : numpy.ndarray, shape (nw_b, nt_b)
        Transformation matrix mapping bosonic imaginary time to Matsubara frequencies.

    Notes
    -----
    **Frequency-index conventions**

    Matsubara indices are controlled by the ``phys_notation`` flag:

    - ``phys_notation=False`` (default): CoQuí/IR convention where iω_n = n·π/β
      for both statistics.  For fermions n is odd; for bosons n is even.
    - ``phys_notation=True``: physical convention where iω_n = (2n+1)·π/β for
      fermions and iω_n = 2n·π/β for bosons.

    **Imaginary-time conventions**

    Imaginary-time values are controlled by the ``rel_notation`` flag:

    - ``rel_notation=False`` (default): physical time τ ∈ [0, β].
    - ``rel_notation=True``: rescaled coordinate t = 2τ/β − 1 ∈ [−1, 1].

    **Particle-hole symmetry**

    Bosonic correlation functions that satisfy O(β − τ) = O(τ) need only half the
    tau / frequency mesh. Methods ending in ``_phsym`` (or the ``ph_sym=True``
    keyword) exploit this symmetry to halve the storage and computation.

    **Statistics strings**
    The methods accept statistics strings in both short and long forms:
    - 'fermion' or 'f' for fermions
    - 'boson' or 'b' for bosons

    Examples
    --------
    >>> from coqui import IAFT
    >>> import numpy as np
    >>> ft = IAFT(beta=100.0, wmax=10.0, prec='medium', basis='dlr')
    >>> tau = ft.tau_mesh('f')                         # shape (nt_f,)
    >>> wn  = ft.wn_mesh('f', phys_notation=True)       # physical Matsubara indices
    >>> Gt  = np.zeros((ft.nt_f, 3, 3))                # dummy G(τ), (nt_f, norb, norb)
    >>> Gw  = ft.tau_to_w(Gt, 'fermion')               # shape (nw_f, 3, 3)
    >>> Gt2 = ft.w_to_tau(Gw, 'fermion')               # back to imaginary time
    >>> # Interpolate to arbitrary tau points
    >>> tau_pts = np.linspace(0, ft.beta, 50)
    >>> Gt_interp = ft.tau_interpolate(Gt, tau_pts, 'fermion')  # shape (50, 3, 3)
    """
    def __init__(self, beta: float, wmax: float, *, prec=None, eps=None,
                 verbose: bool=True, basis: str="dlr"):
        """
        Initialize IAFT with the specified basis backend.

        :param beta: float
            Inverse temperature.
        :param wmax: float
            Frequency cutoff.
        :param prec: {'high', 'medium', 'low', 'custom'} or None
            Named accuracy preset. Must be provided when ``eps`` is None.
        :param eps: float or None
            Explicit accuracy target (positive float). Required when ``prec='custom'``.
        :param verbose: bool
            Print basis metadata on initialization. Default True.
        :param basis: {'dlr', 'ir'}
            Backend basis. Default ``'dlr'`` (C++ cppdlr).
        :raises ValueError: if basis is not ``'dlr'`` or ``'ir'``,
            or if precision arguments are inconsistent.
        :raises ImportError: if the required backend library is not available.
        """
        _validate_accuracy_args(basis, prec, eps)

        # Check backend availability and create adapter
        if basis == "ir":
            try:
                import sparse_ir
            except ImportError:
                raise ImportError(
                    "IR backend requires sparse_ir library. "
                    "Please install with: pip install sparse-ir[xprec]==1.1.7"
                )
            self._impl = _IAFTIRAdapter(beta, wmax, prec=prec, eps=eps, verbose=verbose)
        else:  # basis == "dlr"
            self._impl = _IAFTCppAdapter(beta, wmax, prec=prec, eps=eps, verbose=verbose, basis="dlr")

        self.basis = basis

    # Delegate properties to implementation
    @property
    def beta(self):
        """float: Inverse temperature."""
        return self._impl.beta

    @property
    def wmax(self):
        """float: Frequency cutoff."""
        return self._impl.wmax

    @property
    def lmbda(self):
        """float: Dimensionless parameter lmbda = beta * wmax."""
        return self._impl.lmbda

    @property
    def prec(self):
        """str: Active precision label (e.g. 'high', 'medium', 'low', 'custom')."""
        return self._impl.prec

    @property
    def eps(self):
        """float: Active accuracy target used to build the basis."""
        return self._impl.eps

    @property
    def nt_f(self):
        """int: Number of fermionic imaginary-time sampling points."""
        return self._impl.nt_f

    @property
    def nt_b(self):
        """int: Number of bosonic imaginary-time sampling points."""
        return self._impl.nt_b

    @property
    def nw_f(self):
        """int: Number of fermionic Matsubara-frequency sampling points."""
        return self._impl.nw_f

    @property
    def nw_b(self):
        """int: Number of bosonic Matsubara-frequency sampling points."""
        return self._impl.nw_b

    @property
    def Ttw_ff(self):
        """ndarray, shape (nt_f, nw_f): fermionic frequency-to-tau transform matrix."""
        return self._impl.Ttw_ff

    @property
    def Twt_ff(self):
        """ndarray, shape (nw_f, nt_f): fermionic tau-to-frequency transform matrix."""
        return self._impl.Twt_ff

    @property
    def Ttw_bb(self):
        """ndarray, shape (nt_b, nw_b): bosonic frequency-to-tau transform matrix."""
        return self._impl.Ttw_bb

    @property
    def Twt_bb(self):
        """ndarray, shape (nw_b, nt_b): bosonic tau-to-frequency transform matrix."""
        return self._impl.Twt_bb

    @classmethod
    def from_coqui_chkpt(cls, chkpt_h5, verbose: bool=True):
        """
        Reconstruct an IAFT instance from a CoQuí HDF5 checkpoint file.

        Reads the ``'imaginary_fourier_transform'`` group written by :meth:`save`.
        Defaults to the IR basis when the checkpoint does not record a basis field
        (backward compatibility with older checkpoints).

        :param chkpt_h5: str
            Path to the HDF5 checkpoint file.
        :param verbose: bool
            Print basis metadata on initialization. Default True.
        :return: IAFT
            Initialized with the parameters stored in the checkpoint.
        """
        from h5 import HDFArchive
        
        if not os.path.exists(chkpt_h5):
            raise FileNotFoundError(f"Checkpoint file to initialize IAFT not found: {chkpt_h5}")
        
        with HDFArchive(chkpt_h5, 'r') as ar:
            iaft_grp = ar['imaginary_fourier_transform']
            beta = iaft_grp['beta']
            prec = iaft_grp['prec'] if 'prec' in iaft_grp else None
            eps = iaft_grp['eps'] if 'eps' in iaft_grp else None
            if 'wmax' in iaft_grp:
                wmax = iaft_grp['wmax']
            else:
                ir_lambda = iaft_grp['lambda']
                wmax = ir_lambda / beta
            # Prefer basis metadata; fall back to source for backward compatibility.
            basis = iaft_grp['basis'] if 'basis' in iaft_grp else (iaft_grp['source'] if 'source' in iaft_grp else "ir")

        return cls(beta, wmax, prec=prec, eps=eps, verbose=verbose, basis=basis)

    def save(self, h5_grp):
        """
        Save IAFT parameters to an HDF5 group under the key
        ``'imaginary_fourier_transform'``.

        Stores ``beta``, ``wmax``, ``prec``, ``eps``, and ``basis`` so that the
        instance can be reconstructed with :meth:`from_coqui_chkpt`. No-op if
        the key already exists in the group.

        :param h5_grp: h5py/HDFArchive group
            Target HDF5 group to write into.
        """
        self._impl.save(h5_grp)

    def __str__(self):
        return self._impl.__str__()

    def __eq__(self, other):
        return (isinstance(other, IAFT) and self._impl == other._impl)

    # Delegate all transform methods
    def wn_mesh(self, stats: str, *, phys_notation: bool=False, positive_only=False):
        """
        Return the Matsubara-frequency sampling indices.

        :param stats: str
            Particle statistics: [``'f'``, ``'fermion'``] for fermions,
            [``'b'``, ``'boson'``] for bosons.
        :param phys_notation: bool, keyword-only
            Frequency-index convention:

            - ``False`` (default): CoQuí/IR convention, iω_n = n·π/β for all
              statistics (n odd for fermions, even for bosons).
            - ``True``: physical convention, iω_n = (2n+1)·π/β for fermions
              and iω_n = 2n·π/β for bosons.
        :param positive_only: bool, keyword-only
            If True, return only the non-negative half of the mesh. Default False.
        :return: numpy.ndarray of int, shape (nw,) or (nw//2,)
            Matsubara frequency indices in the requested convention.
        """
        return self._impl.wn_mesh(stats, phys_notation=phys_notation, positive_only=positive_only)

    def tau_mesh(self, stats: str, rel_notation: bool=False):
        """
        Return the imaginary-time sampling points.

        :param stats: str
            Particle statistics: [``'f'``, ``'fermion'``] for fermions,
            [``'b'``, ``'boson'``] for bosons.
        :param rel_notation: bool, optional
            Coordinate convention:

            - ``False`` (default): physical imaginary time τ ∈ [0, β].
            - ``True``: rescaled coordinate t = 2τ/β − 1 ∈ [−1, 1].
        :return: numpy.ndarray of float, shape (nt_f,) or (nt_b,)
            Imaginary-time sampling points.
        """
        return self._impl.tau_mesh(stats, rel_notation=rel_notation)

    def tau_to_w(self, Ot, stats: str, ph_sym: bool=False):
        """
        Fourier transform from imaginary-time to Matsubara-frequency axis.

        :param Ot: numpy.ndarray, shape (nt, ...)
            Array sampled on the imaginary-time mesh returned by :meth:`tau_mesh`.
            The leading axis must match ``nt_f`` (fermions) or ``nt_b`` (bosons),
            or the particle-hole half-size when ``ph_sym=True``.
        :param stats: str
            Particle statistics: [``'f'``, ``'fermion'``] for fermions,
            [``'b'``, ``'boson'``] for bosons.
        :param ph_sym: bool, optional
            If True, use the particle-hole-symmetric transform (bosons only).
            Input must be the first half of the tau mesh; output covers only
            positive Matsubara frequencies. Default False.
        :return: numpy.ndarray, shape (nw, ...)
            Array sampled on the Matsubara-frequency mesh returned by :meth:`wn_mesh`.
        """
        return self._impl.tau_to_w(Ot, stats) if not ph_sym else self._impl.tau_to_w_phsym(Ot, stats)

    def w_to_tau(self, Ow, stats: str, ph_sym: bool=False):
        """
        Fourier transform from Matsubara-frequency to imaginary-time axis.

        :param Ow: numpy.ndarray, shape (nw, ...)
            Array sampled on the Matsubara-frequency mesh returned by :meth:`wn_mesh`.
            When ``ph_sym=True``, must cover only the positive half-mesh.
        :param stats: str
            Particle statistics: ``'f'``/``'fermion'`` or ``'b'``/``'boson'``.
        :param ph_sym: bool, optional
            If True, use the particle-hole-symmetric transform (bosons only).
            Output is the first half of the imaginary-time mesh. Default False.
        :return: numpy.ndarray, shape (nt, ...)
            Array sampled on the imaginary-time mesh returned by :meth:`tau_mesh`.
        """
        return self._impl.w_to_tau(Ow, stats) if not ph_sym else self._impl.w_to_tau_phsym(Ow, stats)

    def tau_to_w_phsym(self, Ot, stats: str):
        """
        Particle-hole-symmetric Fourier transform from imaginary-time to Matsubara-frequency.

        Equivalent to ``tau_to_w(Ot, stats, ph_sym=True)``. Bosons only.
        Input must be the first half of the tau mesh (``nt_b // 2 + nt_b % 2``
        points); output covers positive Matsubara frequencies only.

        :param Ot: numpy.ndarray, shape (nt_b//2 + nt_b%2, ...)
            Bosonic imaginary-time array on the particle-hole half-mesh.
        :param stats: str
            Must be ``'b'`` or ``'boson'``.
        :return: numpy.ndarray, shape (nw_b//2 + nw_b%2, ...)
            Bosonic Matsubara-frequency array at positive frequencies.
        :raises ValueError: if stats is not bosonic.
        """
        return self._impl.tau_to_w_phsym(Ot, stats)

    def w_to_tau_phsym(self, Ow, stats: str):
        """
        Particle-hole-symmetric inverse Fourier transform from Matsubara-frequency to imaginary-time.

        Equivalent to ``w_to_tau(Ow, stats, ph_sym=True)``. Bosons only.
        Input must cover positive Matsubara frequencies (``nw_b // 2 + nw_b % 2``
        points); output is the first half of the imaginary-time mesh.

        :param Ow: numpy.ndarray, shape (nw_b//2 + nw_b%2, ...)
            Bosonic Matsubara-frequency array at positive frequencies.
        :param stats: str
            Must be ``'b'`` or ``'boson'``.
        :return: numpy.ndarray, shape (nt_b//2 + nt_b%2, ...)
            Bosonic imaginary-time array on the particle-hole half-mesh.
        :raises ValueError: if stats is not bosonic.
        """
        return self._impl.w_to_tau_phsym(Ow, stats)

    def w_interpolate(self, Ow, target, stats: str, *, phys_notation: bool=False, ph_sym: bool=False):
        """
        Interpolate a Matsubara-frequency object to arbitrary frequency indices.

        Uses the compact basis representation to evaluate the object at arbitrary
        Matsubara indices.

        :param Ow: numpy.ndarray, shape (nw, ...)
            Array sampled on the native Matsubara-frequency mesh (or the positive
            half when ``ph_sym=True``).
        :param target: array-like of int or IAFT
            Target Matsubara indices in the convention set by ``phys_notation``.
            If an :class:`IAFT` instance is passed, its ``wn_mesh(stats)`` is used.
        :param stats: str
            Particle statistics: ``'f'``/``'fermion'`` or ``'b'``/``'boson'``.
        :param phys_notation: bool, keyword-only
            Convention for target indices (see :meth:`wn_mesh`). Default False.
        :param ph_sym: bool, keyword-only
            Apply particle-hole-symmetric interpolation (bosons only). ``Ow``
            must cover the positive half-mesh. Default False.
        :return: numpy.ndarray, shape (len(target), ...)
            Interpolated values at the target Matsubara indices.
        """
        if isinstance(target, IAFT):
            return self._impl.w_interpolate(Ow, target._impl, stats, phys_notation=phys_notation, ph_sym=ph_sym)
        else:
            return self._impl.w_interpolate(Ow, target, stats, phys_notation=phys_notation, ph_sym=ph_sym)

    def w_interpolate_phsym(self, Ow, target, stats: str, *, phys_notation: bool=False):
        """
        Particle-hole-symmetric Matsubara interpolation.

        Equivalent to ``w_interpolate(Ow, target, stats, phys_notation=phys_notation, ph_sym=True)``.
        Input ``Ow`` must cover only the positive Matsubara half-mesh. Bosons only.

        :param Ow: numpy.ndarray, shape (nw//2, ...)
            Bosonic Matsubara-frequency array at positive frequencies.
        :param target: array-like of int or IAFT
            Target Matsubara indices (positive half only).
        :param stats: str
            Must be ``'b'`` or ``'boson'``.
        :param phys_notation: bool, keyword-only
            Convention for target indices. Default False.
        :return: numpy.ndarray, shape (len(target), ...)
            Interpolated values at the target indices.
        """
        return self.w_interpolate(Ow, target, stats, phys_notation=phys_notation, ph_sym=True)

    def tau_interpolate(self, Ot, target, stats: str, *, rel_notation: bool=False, ph_sym: bool=False):
        """
        Interpolate an imaginary-time object to arbitrary tau points.

        Uses the compact basis representation to evaluate the object at 
        arbitrary τ ∈ [0, β]. 

        :param Ot: numpy.ndarray, shape (nt, ...)
            Array sampled on the native imaginary-time mesh (or the first half
            when ``ph_sym=True``).
        :param target: float, array-like of float, or IAFT
            Target imaginary-time points in the convention set by ``rel_notation``.
            If an :class:`IAFT` instance is passed, its ``tau_mesh(stats)`` is used.
        :param stats: str
            Particle statistics: ``'f'``/``'fermion'`` or ``'b'``/``'boson'``.
        :param rel_notation: bool, keyword-only
            Coordinate convention for target points (see :meth:`tau_mesh`).
            Default False (physical τ ∈ [0, β]).
        :param ph_sym: bool, keyword-only
            Apply particle-hole-symmetric interpolation (bosons only). ``Ot``
            must cover the first half of the tau mesh. Default False.
        :return: numpy.ndarray, shape (len(target), ...)
            Interpolated values at the target imaginary-time points.
        """
        if isinstance(target, IAFT):
            return self._impl.tau_interpolate(Ot, target._impl, stats, rel_notation=rel_notation, ph_sym=ph_sym)
        else:
            return self._impl.tau_interpolate(Ot, target, stats, rel_notation=rel_notation, ph_sym=ph_sym)

    def tau_interpolate_phsym(self, Ot, target, stats: str, *, rel_notation: bool=False):
        """
        Particle-hole-symmetric imaginary-time interpolation.

        Equivalent to ``tau_interpolate(Ot, target, stats, rel_notation=rel_notation, ph_sym=True)``.
        Input ``Ot`` must cover only the first half of the tau mesh. Bosons only.

        :param Ot: numpy.ndarray, shape (nt//2, ...)
            Bosonic imaginary-time array on the particle-hole half-mesh.
        :param target: float, array-like of float, or IAFT
            Target imaginary-time points.
        :param stats: str
            Must be ``'b'`` or ``'boson'``.
        :param rel_notation: bool, keyword-only
            Coordinate convention for target points. Default False (physical [0, β]).
        :return: numpy.ndarray, shape (len(target), ...)
            Interpolated values at the target imaginary-time points.
        """
        return self.tau_interpolate(Ot, target, stats, rel_notation=rel_notation, ph_sym=True)

    def check_leakage(self, Ot, stats: str, name: str="", w_input: bool=False):
        """
        Diagnose basis truncation by checking the decay of compressed-basis coefficients.

        Projects ``Ot`` onto the basis representation and compares the magnitude of 
        the last two coefficients with the first. If ``coeff_last/coeff_first`` is 
        too large, the basis is not sufficiently converged for the input function, 
        and the user should consider increasing ``wmax`` or decreasing ``eps``.

        Note: For the DLR backend this method is a no-op until C++ leakage
        diagnostics are implemented.

        :param Ot: numpy.ndarray, shape (nt, ...) or (nw, ...)
            Input array on the imaginary-time mesh, or on the Matsubara-frequency
            mesh when ``w_input=True``.
        :param stats: str
            Particle statistics: ``'f'``/``'fermion'`` or ``'b'``/``'boson'``.
        :param name: str, optional
            Label printed in the diagnostic output. Default ``""``.
        :param w_input: bool, optional
            If True, ``Ot`` is on the Matsubara-frequency mesh and is transformed
            to imaginary time before the leakage check. Default False.
        """
        return self._impl.check_leakage(Ot, stats, name, w_input)

    def check_leakage_phsym(self, Ot, stats: str, name: str="", w_input: bool=False):
        """
        Particle-hole-symmetric variant of :meth:`check_leakage`. Bosons only.

        Input ``Ot`` must cover the first half of the imaginary-time mesh (or, when
        ``w_input=True``, the positive Matsubara half-mesh).

        Note: For the DLR backend this method is a no-op until C++ leakage
        diagnostics are implemented.

        :param Ot: numpy.ndarray, shape (nt//2, ...) or (nw//2, ...)
            Bosonic half-mesh input array.
        :param stats: str
            Must be ``'b'`` or ``'boson'``.
        :param name: str, optional
            Label printed in the diagnostic output. Default ``""``.
        :param w_input: bool, optional
            If True, ``Ot`` is on the positive Matsubara half-mesh. Default False.
        :raises ValueError: if stats is not bosonic.
        """
        return self._impl.check_leakage_phsym(Ot, stats, name, w_input)

    def build_g_tau_ref(self, norb: int, stats: str, ph_sym: bool = False) -> np.ndarray:
        """
        Build a reference Green's function on the imaginary time mesh with a known analytical form.

        The analytical form is the same as used internally for testing IAFT accuracy.
        The tau convention is [-1, 1] mapping to physical imaginary time [0, beta].

        :param norb: number of orbitals (output shape will be (nt, norb, norb)).
        :param stats: statistics, 'f'/'fermion' for fermions or 'b'/'boson' for bosons.
        :param ph_sym: if True, only the first half of tau points is constructed (particle-hole symmetry).
        :return: numpy array of shape (nt, norb, norb).
        """
        if build_g_tau_ref is None:
            raise ImportError(
                "build_g_tau_ref requires IAFTCpp binding from CoQuí C++ library."
            ) from _IAFT_CPP_IMPORT_ERROR
        return build_g_tau_ref(self.tau_mesh(stats, rel_notation=True), self.beta, norb, ph_sym)


    def build_g_iw_ref(self, norb: int, stats: str, ph_sym: bool = False) -> np.ndarray:
        """
        Build a reference Green's function on the Matsubara frequency mesh with a known analytical form.

        The analytical form is the same as used internally for testing IAFT accuracy.
        The Matsubara index convention is n = (2k+1) for fermions and n = 2k for bosons.

        :param norb: number of orbitals (output shape will be (nw, norb, norb)).
        :param stats: statistics, 'f'/'fermion' for fermions or 'b'/'boson' for bosons.
        :param ph_sym: if True, only the positive Matsubara frequencies are constructed (particle-hole symmetry).
        :return: numpy array of shape (nw, norb, norb).
        """
        stats = _normalize_stats(stats)
        if build_g_iw_ref is None:
            raise ImportError(
                "build_g_iw_ref requires IAFTCpp binding from CoQuí C++ library."
            ) from _IAFT_CPP_IMPORT_ERROR
        return build_g_iw_ref(self.wn_mesh(stats), self.beta, stats, norb, ph_sym)


if __name__ == '__main__':
    # Initialize IAFT object for given inverse temperature, lambda and precision
    ft = IAFT(1000.0, 10.0, prec="low", basis='ir')

    print(ft.wn_mesh('f', phys_notation=False))

    Gt = np.zeros((ft.nt_f, 2, 2, 2))
    Gw = ft.tau_to_w(Gt, 'f')
    print(Gw.shape)

    # Interpolate to arbitrary tau point
    tau_interp = np.array([0.0, ft.beta])
    Gt_interp = ft.tau_interpolate(Gt, tau_interp, 'f')
    print(Gt_interp.shape)

    # wn in spare_ir notation
    w_interp = np.array([-1,1,3,5], dtype=int)
    Gw_interp = ft.w_interpolate(Gw, w_interp, 'f', phys_notation=False)
    print(Gw_interp.shape)

    # wn in physical notation
    w_interp = np.array([-1,0,1,2,3,4], dtype=int)
    Gw_interp = ft.w_interpolate(Gw, w_interp, 'f', phys_notation=True)
    print(Gw_interp.shape)

    Gt2 = ft.w_to_tau(Gw, 'f')
    print(Gt2.shape)
