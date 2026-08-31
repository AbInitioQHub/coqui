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

import sys
import numpy as np
from coqui._lib.utils_module import app_log, app_warning


def set_epsilon(prec_str: str):
    if isinstance(prec_str, str):
        if prec_str == "high":
            return 1e-15
        elif prec_str == "medium":
            return 1e-10
        elif prec_str == "low":
            return 1e-6
        else:
            raise ValueError(f"Unknown precision string: {prec_str}. Acceptable options are 'high', 'medium', and 'low'.")
    return -1.0


def set_lambda(lmbda, coqui_cxx_style=False):
    if coqui_cxx_style:
        if lmbda > 0 and lmbda <= 100:
            return 100
        elif lmbda > 100 and lmbda <= 1000:
            return 1000
        elif lmbda > 1000 and lmbda <= 10000:
            return 10000
        elif lmbda > 10000 and lmbda <= 100000:
            return 100000
        elif lmbda > 100000 and lmbda <= 1000000:
            return 1000000
        else:
            raise ValueError(f"Invalid lambda value: {lmbda}. Acceptable range is (0, 1000000]")
    return lmbda


def _normalize_stats(stats_input: str) -> str:
    """
    Normalize stats input to short form ('f' or 'b').
    Accepts both short forms ('f', 'b') and long forms ('fermion', 'boson').

    :param stats_input: str
    Statistics string ('f', 'fermion', 'b', or 'boson')
    :return: str
    Normalized short form ('f' or 'b')
    :raises ValueError: if stats_input is not a recognized statistics format
    """
    if stats_input == 'f' or stats_input == 'fermion':
        return 'f'
    elif stats_input == 'b' or stats_input == 'boson':
        return 'b'
    else:
        raise ValueError(
            "Unknown statistics '{}'. Acceptable options are 'f'/'fermion' for fermions "
            "and 'b'/'boson' for bosons.".format(stats_input)
        )


class _IAFTIRAdapter(object):
    """
    Internal adapter implementing IAFT via the ``sparse_ir`` Intermediate 
    Representation library.

    Use ``IAFT(basis='ir')`` for the public-facing API; do not instantiate this 
    class directly. 

    Transforms are carried out by pre-computing projection matrices between the 
    IR coefficient space and the sparse sampling meshes:

    - ``Tlt_ff`` / ``Tlt_bb`` – tau-point-to-IR-coefficients (pseudo-inverse of Ttl)
    - ``Tlw_ff`` / ``Tlw_bb`` – freq-point-to-IR-coefficients (pseudo-inverse of Twl)
    - ``Ttw_ff = Ttl * Tlw`` – complete freq-to-tau transform matrix (fermionic)
    - ``Twt_ff = Twl * Tlt`` – complete tau-to-freq transform matrix (fermionic)
    - Analogous ``_bb`` matrices for bosons

    Imaginary-time points are stored internally in the rescaled coordinate
    t = 2τ/β − 1 ∈ [−1, 1]; the ``tau_mesh`` method converts to physical τ
    unless ``rel_notation=True``.

    Dependencies
    ------------
    Requires ``sparse-ir`` version 1.1.7 with xprec support::

        pip install sparse-ir[xprec]==1.1.7

    Versions 2.0+ have not been tested and may be incompatible.

    Attributes
    ----------
    beta : float
        Inverse temperature in a.u.
    wmax : float
        Frequency cutoff in a.u. (may be rounded up to the nearest IR grid value).
    lmbda : float
        Dimensionless parameter lmbda = beta * wmax.
    prec : str
        Active precision label.
    eps : float
        Active accuracy target used to construct the IR basis.
    nt_f, nw_f : int
        Number of fermionic tau / Matsubara sampling points.
    nt_b, nw_b : int
        Number of bosonic tau / Matsubara sampling points.
    Ttw_ff : numpy.ndarray, shape (nt_f, nw_f)
        Fermionic frequency-to-tau transform matrix.
    Twt_ff : numpy.ndarray, shape (nw_f, nt_f)
        Fermionic tau-to-frequency transform matrix.
    Ttw_bb : numpy.ndarray, shape (nt_b, nw_b)
        Bosonic frequency-to-tau transform matrix.
    Twt_bb : numpy.ndarray, shape (nw_b, nt_b)
        Bosonic tau-to-frequency transform matrix.
    """
    def __init__(self, beta: float, wmax: float, *, prec: str=None, eps: float=None, verbose: bool=True):
        """
        :param beta: float
            Inverse temperature. 
        :param wmax: float
            Frequency cutoff. Rounded up to the nearest IR grid value
            unless ``prec='custom'`` or ``prec=None`` (no rounding in that case).
        :param prec: str or None
            Precision label. If provided (and not 'custom'), ``eps`` is derived
            from the label and ``wmax`` is rounded to the nearest IR grid value.
            Must be provided when ``eps`` is None.
        :param eps: float or None
            Explicit accuracy target. Required when ``prec='custom'``.
        :param verbose: bool
            Print basis metadata on initialization. Default True.
        """
        import sparse_ir

        self.beta = beta

        if eps is not None and (prec is None or prec == "custom"):
            if not isinstance(eps, (float, int)):
                raise TypeError("eps should be a float or integer value, got type {}".format(type(eps)))
            self.prec = "custom"
            self.eps = float(eps)
        else:
            if prec is None:
                raise ValueError("prec must be provided when eps is not provided")
            self.prec = prec
            self.eps = set_epsilon(prec)

        if prec == "custom" or prec is None:
            self.lmbda = wmax * beta
        else:
            self.lmbda = set_lambda(wmax * beta, coqui_cxx_style=True)
        self.wmax = self.lmbda / self.beta

        self.statistics = {'f', 'b'}

        self._bases = sparse_ir.FiniteTempBasisSet(beta=self.beta, wmax=self.wmax, eps=self.eps)
        self._tau_mesh_f = self._bases.smpl_tau_f.sampling_points
        self._tau_mesh_b = self._bases.smpl_tau_b.sampling_points
        self._wn_mesh_f = self._bases.smpl_wn_f.sampling_points
        self._wn_mesh_b = self._bases.smpl_wn_b.sampling_points
        self.nt_f, self.nw_f = self._tau_mesh_f.shape[0], self._wn_mesh_f.shape[0]
        self.nt_b, self.nw_b = self._tau_mesh_b.shape[0], self._wn_mesh_b.shape[0]

        Ttl_ff = self._bases.basis_f.u(self._tau_mesh_f).T
        Twl_ff = self._bases.basis_f.uhat(self._wn_mesh_f).T
        Ttl_bb = self._bases.basis_b.u(self._tau_mesh_b).T
        Twl_bb = self._bases.basis_b.uhat(self._wn_mesh_b).T

        self.Tlt_ff = np.linalg.pinv(Ttl_ff)
        self.Tlt_bb = np.linalg.pinv(Ttl_bb)
        self.Tlw_ff = np.linalg.pinv(Twl_ff)
        self.Tlw_bb = np.linalg.pinv(Twl_bb)

        # Ttw_ff = Ttl_ff * [Twl_ff]^{-1}
        self.Ttw_ff = np.dot(Ttl_ff, self.Tlw_ff)
        self.Twt_ff = np.dot(Twl_ff, self.Tlt_ff)
        self.Ttw_bb = np.dot(Ttl_bb, self.Tlw_bb)
        self.Twt_bb = np.dot(Twl_bb, self.Tlt_bb)

        # convert tau_mesh to [-1, 1] notation
        self._tau_mesh_f = 2.0 * (self._tau_mesh_f) / self.beta - 1.0
        self._tau_mesh_b = 2.0 * (self._tau_mesh_b) / self.beta - 1.0

        if verbose:
            app_log(1, str(self))

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
        h5_grp['imaginary_fourier_transform']['basis'] = "ir"

    def __str__(self):
        return ("Mesh details on the imaginary axis\n"
                "----------------------------------\n"
                "Intermediate Representation\n"
                "epsilon = {}\n"
                "beta = {}\n"
                "frequency cutoff = {}\n"
                "lambda = {}\n"
                "nt_f, nw_f = {}, {}\n"
                "nt_b, nw_b = {}, {}\n".format(
            self.eps, self.beta, self.wmax, self.lmbda,
            self.nt_f, self.nw_f, self.nt_b, self.nw_b))

    def __eq__(self, other):
        return (
            isinstance(other, _IAFTIRAdapter) and
            self.beta == other.beta and
            self.wmax == other.wmax and
            self.lmbda == other.lmbda and
            self.eps == other.eps
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
        if stats not in self.statistics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))
        tau_mesh = np.array(self._tau_mesh_f, dtype=float) if stats == 'f' else np.array(self._tau_mesh_b, dtype=float)
        if not rel_notation:
            tau_mesh = (tau_mesh + 1.0) * self.beta / 2.0
        return tau_mesh

    def wn_mesh(self, stats: str, *, phys_notation: bool=False, positive_only=False):
        """
        Return Matsubara-frequency sampling indices.

        :param stats: str
            Statistics: 'f'/'fermion' for fermions, 'b'/'boson' for bosons.
        :param phys_notation: bool, keyword-only
            If False (default), use the CoQuí/IR convention iω_n = n·π/β (n odd
            for fermions, even for bosons). If True, use the physical convention
            iω_n = (2n+1)·π/β for fermions and 2n·π/β for bosons.
        :param positive_only: bool, keyword-only
            If True, return only the non-negative half of the mesh.
        :return: numpy.ndarray of int, shape (nw,) or (nw//2,)
        """
        stats = _normalize_stats(stats)
        if stats not in self.statistics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))
        wn_mesh = np.array(self._wn_mesh_f, dtype=int) if stats == 'f' else np.array(self._wn_mesh_b, dtype=int)
        if phys_notation:
            wn_mesh = (wn_mesh - 1) // 2 if stats == 'f' else wn_mesh // 2

        if positive_only:
            nw_half = wn_mesh.shape[0] // 2
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
        if stats not in self.statistics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))
        Twt = self.Twt_ff if stats == 'f' else self.Twt_bb
        if Ot.shape[0] != Twt.shape[1]:
            raise ValueError(
                "tau_to_w: Number of tau points are inconsistent: {} and {}".format(Ot.shape[0], Twt.shape[1]))

        Ot_shape = Ot.shape
        Ot = Ot.reshape(Ot.shape[0], -1)
        Ow = np.dot(Twt, Ot)

        Ot = Ot.reshape(Ot_shape)
        Ow = Ow.reshape((Twt.shape[0],) + Ot_shape[1:])
        return Ow

    def tau_to_w_phsym(self, Ot, stats: str):
        """
        Particle-hole-symmetric Fourier transform from imaginary-time to Matsubara-frequency.
        Bosons only; input is the first half of the tau mesh.

        Constructs a reduced transform matrix that folds the bosonic symmetry
        O(β−τ) = O(τ) before projecting onto positive Matsubara frequencies.

        :param Ot: numpy.ndarray, shape (nt_b//2 + nt_b%2, ...)
        :param stats: str
            Must be 'b' or 'boson'.
        :return: numpy.ndarray, shape (nw_b//2 + nw_b%2, ...)
        :raises ValueError: if stats is not bosonic.
        """
        stats = _normalize_stats(stats)
        if stats != 'b':
            raise ValueError("FT w/ particle-hole symmetry only support bosonic correlation functions")

        nw_half = self.nw_b // 2 if self.nw_b % 2 == 0 else self.nw_b // 2 + 1
        nt_half = self.nt_b // 2 if self.nt_b % 2 == 0 else self.nt_b // 2 + 1
        if Ot.shape[0] != nt_half:
            raise ValueError(
                "tau_to_w_phsym: Number of tau points are inconsistent: {} and {}".format(Ot.shape[0], nt_half))

        Twt_pos = np.zeros((nw_half, nt_half), dtype=self.Twt_bb.dtype)
        for n in range(nw_half):
            iw = self.nw_b // 2 + n
            for it in range(nt_half):
                imt = self.nt_b - it - 1
                Twt_pos[n, it] = self.Twt_bb[iw, it] if it == imt else self.Twt_bb[iw, it] + self.Twt_bb[iw, imt]

        Ot_shape = Ot.shape
        Ot = Ot.reshape(Ot.shape[0], -1)
        Ow = np.dot(Twt_pos, Ot)

        Ot = Ot.reshape(Ot_shape)
        Ow = Ow.reshape((Twt_pos.shape[0],) + Ot_shape[1:])
        return Ow

    def w_to_tau(self, Ow, stats):
        """
        Fourier transform from Matsubara-frequency axis to imaginary-time axis.

        :param Ow: numpy.ndarray, shape (nw, ...)
            Array on the Matsubara-frequency mesh.
        :param stats: str
            Statistics: 'f'/'fermion' or 'b'/'boson'.
        :return: numpy.ndarray, shape (nt, ...)
        """
        stats = _normalize_stats(stats)
        if stats not in self.statistics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))
        Ttw = self.Ttw_ff if stats == 'f' else self.Ttw_bb
        if Ow.shape[0] != Ttw.shape[1]:
            raise ValueError(
                "w_to_tau: Number of w points are inconsistent: {} and {}".format(Ow.shape[0], Ttw.shape[1]))

        Ow_shape = Ow.shape
        Ow = Ow.reshape(Ow.shape[0], -1)
        Ot = np.dot(Ttw, Ow)

        Ow = Ow.reshape(Ow_shape)
        Ot = Ot.reshape((Ttw.shape[0],) + Ow_shape[1:])
        return Ot

    def w_to_tau_phsym(self, Ow, stats):
        """
        Particle-hole-symmetric inverse Fourier transform from Matsubara-frequency to imaginary-time.
        Bosons only; input covers positive Matsubara frequencies only.

        Constructs a reduced transform matrix that exploits O(β−τ) = O(τ) to
        produce only the first half of the tau mesh.

        :param Ow: numpy.ndarray, shape (nw_b//2 + nw_b%2, ...)
        :param stats: str
            Must be 'b' or 'boson'.
        :return: numpy.ndarray, shape (nt_b//2 + nt_b%2, ...)
        :raises ValueError: if stats is not bosonic.
        """
        stats = _normalize_stats(stats)
        if stats != 'b':
            raise ValueError("FT w/ particle-hole symmetry only support bosonic correlation functions")

        nw_half = self.nw_b // 2 if self.nw_b % 2 == 0 else self.nw_b // 2 + 1
        nt_half = self.nt_b // 2 if self.nt_b % 2 == 0 else self.nt_b // 2 + 1
        if Ow.shape[0] != nw_half:
            raise ValueError(
                "w_to_tau_phsym: Number of w points are inconsistent: {} and {}".format(Ow.shape[0], nw_half))

        Ttw_pos = np.zeros((nt_half, nw_half), dtype=self.Ttw_bb.dtype)
        for it in range(nt_half):
            for n in range(nw_half):
                iw = self.nw_b // 2 + n
                imw = self.nw_b // 2 - n
                Ttw_pos[it, n] = self.Ttw_bb[it, iw] if iw == imw else self.Ttw_bb[it, iw] + self.Ttw_bb[it, imw]

        Ow_shape = Ow.shape
        Ow = Ow.reshape(Ow.shape[0], -1)
        Ot = np.dot(Ttw_pos, Ow)

        Ow = Ow.reshape(Ow_shape)
        Ot = Ot.reshape((Ttw_pos.shape[0],) + Ow_shape[1:])
        return Ot

    def w_interpolate(self, Ow, target, stats: str, *, phys_notation: bool=False, ph_sym: bool=False):
        """
        Interpolate a Matsubara-frequency object to arbitrary frequency indices.

        :param Ow: numpy.ndarray, shape (nw, ...)
            Array on the native Matsubara-frequency mesh (or positive half when ph_sym=True).
        :param target: array-like of int or _IAFTIRAdapter
            Target Matsubara indices. If an adapter instance is passed, its wn_mesh is used.
        :param stats: str
            Statistics: 'f'/'fermion' or 'b'/'boson'.
        :param phys_notation: bool, keyword-only
            Index convention for target (see wn_mesh). Default False.
        :param ph_sym: bool, keyword-only
            Apply particle-hole-symmetric interpolation (bosons only). Default False.
        :return: numpy.ndarray, shape (len(target), ...)
        """
        if isinstance(target, _IAFTIRAdapter):
            return self._w_interpolate(Ow, target.wn_mesh(stats, phys_notation=phys_notation, positive_only=ph_sym),
                                       stats, phys_notation=phys_notation, ph_sym=ph_sym)
        else:
            return self._w_interpolate(Ow, target, stats, phys_notation=phys_notation, ph_sym=ph_sym)

    def w_interpolate_phsym(self, Ow, target, stats: str, *, phys_notation: bool=False):
        """
        Particle-hole-symmetric Matsubara interpolation. Delegates to w_interpolate with ph_sym=True.

        :param Ow: numpy.ndarray, shape (nw//2, ...)
        :param target: array-like of int or _IAFTIRAdapter
        :param stats: str
        :param phys_notation: bool, keyword-only
        :return: numpy.ndarray, shape (len(target), ...)
        """
        return self.w_interpolate(Ow, target, stats, phys_notation=phys_notation, ph_sym=True)

    def _w_interpolate(self, Ow, wn_mesh_interp, stats: str, *, phys_notation: bool=False, ph_sym: bool=False):
        stats = _normalize_stats(stats)
        if stats not in self.statistics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))

        if isinstance(wn_mesh_interp, int):
            wn_mesh_interp = np.array([wn_mesh_interp], dtype=int)

        if not phys_notation:
            wn_indices = np.asarray(wn_mesh_interp, dtype=int)
        if phys_notation:
            wn_indices = np.array([2 * n + 1 if stats == 'f' else 2 * n for n in wn_mesh_interp], dtype=int)

        Tlw = self.Tlw_ff if stats == 'f' else self.Tlw_bb

        if ph_sym:
            nw_half = self.nw_b // 2 if self.nw_b % 2 == 0 else self.nw_b // 2 + 1
            nw_zero_offset = self.nw_b // 2
            Tlw_pos = np.zeros((Tlw.shape[0], nw_half), dtype=Tlw.dtype)
            for l in range(Tlw.shape[0]):
                for n in range(nw_half):
                    iw = nw_zero_offset + n
                    imw = nw_zero_offset - n
                    Tlw_pos[l, n] = Tlw[l, iw] if iw == imw else Tlw[l, iw] + Tlw[l, imw]
            Tlw = Tlw_pos

        if Ow.shape[0] != Tlw.shape[1]:
            raise ValueError(
                "w_interpolate: Number of w points are inconsistent: {} and {}".format(Ow.shape[0], Tlw.shape[1]))

        Twl_interp = self._bases.basis_f.uhat(wn_indices).T if stats == 'f' else self._bases.basis_b.uhat(wn_indices).T
        Tww = np.dot(Twl_interp, Tlw)

        Ow_shape = Ow.shape
        Ow = Ow.reshape(Ow.shape[0], -1)
        Ow_interp = np.dot(Tww, Ow)

        Ow = Ow.reshape(Ow_shape)
        Ow_interp = Ow_interp.reshape((wn_indices.shape[0],) + Ow_shape[1:])
        return Ow_interp

    def tau_interpolate(self, Ot, target, stats: str, *, rel_notation: bool=False, ph_sym: bool=False):
        """
        Interpolate an imaginary-time object to arbitrary tau points.

        :param Ot: numpy.ndarray, shape (nt, ...)
            Array on the native imaginary-time mesh (or first half when ph_sym=True).
        :param target: float, array-like of float, or _IAFTIRAdapter
            Target tau points. If an adapter instance is passed, its tau_mesh is used.
        :param stats: str
            Statistics: 'f'/'fermion' or 'b'/'boson'.
        :param rel_notation: bool, keyword-only
            Coordinate convention for target (see tau_mesh). Default False.
        :param ph_sym: bool, keyword-only
            Apply particle-hole-symmetric interpolation (bosons only). Default False.
        :return: numpy.ndarray, shape (len(target), ...)
        """
        stats = _normalize_stats(stats)
        if isinstance(target, _IAFTIRAdapter):
            #tau_mesh = target._tau_mesh_f if stats == 'f' else target._tau_mesh_b
            tau_mesh = target.tau_mesh(stats, rel_notation=rel_notation)
            if ph_sym:
                nt_half = tau_mesh.shape[0] // 2 + tau_mesh.shape[0] % 2
                tau_mesh = tau_mesh[:nt_half]
            return self._tau_interpolate(Ot, tau_mesh, stats, rel_notation=rel_notation, ph_sym=ph_sym)
        else:
            return self._tau_interpolate(Ot, target, stats, rel_notation=rel_notation, ph_sym=ph_sym)

    def tau_interpolate_phsym(self, Ot, target, stats: str, *, rel_notation: bool=False):
        """
        Particle-hole-symmetric tau interpolation. Delegates to tau_interpolate with ph_sym=True.

        :param Ot: numpy.ndarray, shape (nt//2, ...)
        :param target: float, array-like of float, or _IAFTIRAdapter
        :param stats: str
        :param rel_notation: bool, keyword-only
        :return: numpy.ndarray, shape (len(target), ...)
        """
        return self.tau_interpolate(Ot, target, stats, rel_notation=rel_notation, ph_sym=True)

    def _tau_interpolate(self, Ot, tau_mesh_interp, stats: str, *, rel_notation: bool=False, ph_sym: bool=False):
        stats = _normalize_stats(stats)
        if stats not in self.statistics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))

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

        if rel_notation:
            # convert to [0, beta] notation for sparse_ir library
            tau_mesh_interp = 0.5 * (tau_mesh_interp + 1.0) * self.beta

        Tlt = self.Tlt_ff if stats == 'f' else self.Tlt_bb
        nt = self.nt_f if stats == 'f' else self.nt_b
        Ttl_interp = self._bases.basis_f.u(tau_mesh_interp).T if stats == 'f' else self._bases.basis_b.u(tau_mesh_interp).T

        if ph_sym:
            nt_half = nt // 2 + nt % 2
            Tlt_pos = np.zeros((Tlt.shape[0], nt_half), dtype=Tlt.dtype)
            for l in range(Tlt.shape[0]):
                for it in range(nt_half):
                    imt = nt - it - 1
                    Tlt_pos[l, it] = Tlt[l, it] if it == imt else Tlt[l, it] + Tlt[l, imt]

        Ttt = np.dot(Ttl_interp, Tlt) if not ph_sym else np.dot(Ttl_interp, Tlt_pos)
        if Ot.shape[0] != Ttt.shape[1]:
            raise ValueError(
                "t_interpolate: Number of tau points are inconsistent: {} and {}".format(Ot.shape[0], Ttt.shape[1]))

        Ot_shape = Ot.shape
        Ot = Ot.reshape(Ot.shape[0], -1)
        Ot_interp = np.dot(Ttt, Ot)

        Ot = Ot.reshape(Ot_shape)
        Ot_interp = Ot_interp.reshape((np.shape(tau_mesh_interp)[0],) + Ot_shape[1:])
        return Ot_interp

    def check_leakage(self, Ot, stats: str, name: str="", w_input: bool=False):
        """
        Diagnose basis truncation by checking the decay of IR coefficients.

        Projects ``Ot`` onto the IR basis and compares the magnitude of the last
        two coefficients with the first. If ``coeff_last/coeff_first >= 1e-5``,
        a warning is printed. A well-represented function should show rapid
        coefficient decay.

        :param Ot: numpy.ndarray, shape (nt, ...) or (nw, ...)
            Input array on the imaginary-time mesh, or on the Matsubara-frequency
            mesh when ``w_input=True``.
        :param stats: str
            Statistics: 'f'/'fermion' or 'b'/'boson'.
        :param name: str, optional
            Label printed in the output. Default "".
        :param w_input: bool, optional
            If True, ``Ot`` is on the Matsubara-frequency mesh and is transformed
            to imaginary time before checking. Default False.
        """
        stats = _normalize_stats(stats)
        if w_input:
            Ot_ = self.w_to_tau(Ot, stats)
            self.check_leakage(Ot_, stats, name, w_input=False)
            return

        if stats not in self.statistics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))
        nts = self.nt_f if stats == 'f' else self.nt_b
        Tlt = self.Tlt_ff if stats == 'f' else self.Tlt_bb
        if nts != Ot.shape[0]:
            raise ValueError("Inconsistency between nts = {} and Ot.shape[0] = {}".format(nts, Ot.shape[0]))

        O_l0_i = np.einsum('t,ti->i', Tlt[0], Ot.reshape(nts, -1))
        coeff_first = np.max(np.abs(O_l0_i))

        O_lm2_t = np.einsum('lt,ti->li', Tlt[-2:], Ot.reshape(nts, -1))
        coeff_last = np.max(np.abs(O_lm2_t))

        leakage = coeff_last / coeff_first
        app_log(1, "IAFT leakage of {}: {}".format(name, leakage))
        if leakage >= 1e-5:
            app_log(1, "[WARNING] check_leakage: coeff_last/coeff_first = {} >= 1e-5; " \
                    "coeff_last = {}, coeff_first = {}".format(leakage, coeff_last, coeff_first))
        sys.stdout.flush()

    def check_leakage_phsym(self, Ot, stats: str, name: str="", w_input: bool=False):
        """
        Particle-hole-symmetric variant of check_leakage. Bosons only.

        Input ``Ot`` must cover the first half of the imaginary-time mesh (or, when
        ``w_input=True``, the positive Matsubara half-mesh).

        :param Ot: numpy.ndarray, shape (nt_b//2, ...) or (nw_b//2, ...)
            Bosonic half-mesh input array.
        :param stats: str
            Must be 'b' or 'boson'.
        :param name: str, optional
            Label printed in the output. Default "".
        :param w_input: bool, optional
            If True, ``Ot`` is on the positive Matsubara half-mesh. Default False.
        :raises ValueError: if stats is not bosonic.
        """
        stats = _normalize_stats(stats)
        if stats != 'b':
            raise ValueError("FT w/ particle-hole symmetry only support bosonic correlation functions")

        if w_input:
            Ot_ = self.w_to_tau_phsym(Ot, stats)
            self.check_leakage_phsym(Ot_, stats, name, w_input=False)
            return

        if stats not in self.statistics:
            raise ValueError("Unknown statistics '{}'. "
                             "Acceptable options are 'f' for fermion and 'b' for bosons.".format(stats))

        nts = self.nt_b
        nt_half = self.nt_b // 2 if self.nt_b % 2 == 0 else self.nt_b // 2 + 1
        Tlt = self.Tlt_bb
        if nt_half != Ot.shape[0]:
            raise ValueError("Inconsistency between nts_half = {} and Ot.shape[0] = {}".format(nt_half, Ot.shape[0]))

        Tl0_t_pos = np.zeros(nt_half, dtype=complex)
        for it in range(nt_half):
            imt = nts - it - 1
            Tl0_t_pos[it] = Tlt[0, it] if it == imt else Tlt[0, it] + Tlt[0, imt]
        O_l0_i = np.einsum('t,ti->i', Tl0_t_pos, Ot.reshape(nt_half, -1))
        coeff_first = np.max(np.abs(O_l0_i))

        Tlm2_t_pos = np.zeros((2, nt_half), dtype=complex)
        nl = Tlt.shape[0]
        for it in range(nt_half):
            imt = nts - it - 1
            Tlm2_t_pos[0, it] = Tlt[nl - 2, it] if it == imt else Tlt[nl - 2, it] + Tlt[nl - 2, imt]
            Tlm2_t_pos[1, it] = Tlt[nl - 1, it] if it == imt else Tlt[nl - 1, it] + Tlt[nl - 1, imt]
        O_lm2_t = np.einsum('lt,ti->li', Tlm2_t_pos, Ot.reshape(nt_half, -1))
        coeff_last = np.max(np.abs(O_lm2_t))

        leakage = coeff_last / coeff_first
        app_log(1, "IAFT leakage of {}: {}".format(name, leakage))
        if leakage >= 1e-5:
            app_log(1, "[WARNING] check_leakage_phsym: coeff_last/coeff_first = {} >= 1e-5; " \
                       "coeff_last = {}, coeff_first = {}".format(leakage, coeff_last, coeff_first))
