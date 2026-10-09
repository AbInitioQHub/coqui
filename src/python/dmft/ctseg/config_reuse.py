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
Reuse of CT-SEG Monte Carlo configurations across the DMFT self-consistency. 

With ``reuse_config`` on, every CT-SEG call of DMFT iteration N (chemical-potential search and
production solve) starts its warm-up from the final configuration of iteration N-1, read from 
``dmft/iter{N-1}/impurity_{i}/ctseg_configs`` of the solver checkpoint. 

The configurations are per MPI rank (every rank runs its own Markov chain) and are written by
``triqs_ctseg.save_configs``. A ``triqs_ctseg.ConfigDataT`` is the position in imaginary time
of every operator of a chain:

- ``beta``, ``block_sizes``: the inverse temperature the times refer to and the ``gf_struct``
  block sizes; a solve with a different value of either cannot use it.
- ``seglists``: one ``uint64`` array of shape ``(n_segments, 4)`` per color (the ``gf_struct``
  orbitals in order), columns ``tau_c, tau_cdag, J_c, J_cdag``. A row is an occupied segment
  from ``tau_cdag`` to ``tau_c``, of length ``(tau_c - tau_cdag) mod beta``; it wraps through
  ``beta -> 0`` when ``tau_c < tau_cdag``, and ``[beta, 0]`` is a line occupied the whole time.
  Times are exact integers on a 2^64 grid, ``tau = beta * n / (2^64 - 1)``. ``J_c``/``J_cdag``
  flag operators of a spin-flip line.
- ``Jperp_lines``: ``(n, 2)`` times ``tau_Sminus, tau_Splus`` of the spin-flip (J_perp) lines;
  empty in CoQuí, which passes no J_perp.
- ``is_copy``: set by ``load_configs`` when this rank received a configuration that another
  rank also starts from; such a solve needs a warm-up.

Requires a TRIQS/ctseg build providing ``ConfigDataT``. ``triqs_ctseg`` is imported lazily, so
this module loads with any CT-SEG build.
"""
import os
import textwrap

import numpy as np
from h5 import HDFArchive
import triqs.utility.mpi as mpi
from coqui import app_log

__all__ = ['read_initial_config', 'warn_restart_rejected', 'save_to_chkpt', 'load_from_chkpt']

_CONFIG_GROUP = "ctseg_configs"


def _compatible(config, beta, gf_struct):
    """
    Check the compatibility of a given config with the current impurity problem. 

    Whether ``config`` can start a solve at ``beta`` with ``gf_struct``, as far as can be told
    before CT-SEG rebuilds the determinants. 
    
    """
    sizes = [int(size) for _, size in gf_struct]
    if (abs(config.beta - beta) > 1e-12 * abs(beta)
            or [int(size) for size in config.block_sizes] != sizes
            or len(config.seglists) != sum(sizes)):
        return False
    if len(config.Jperp_lines) > 0 or any(np.any(seg[:, 2:]) for seg in config.seglists):
        return False
    return True


def _warning_box(text, level=1):
    """Log ``text`` as a boxed ``[ WARNING ]`` (78 columns inside the box)."""
    width = 78
    body = [" [ WARNING ] CT-SEG configuration reuse"] + \
           [" " + line for line in textwrap.wrap(text, width - 2)]
    box = (["╔" + "═" * width + "╗"] + [f"║{line:<{width}}║" for line in body]
           + ["╚" + "═" * width + "╝"])
    app_log(level, "\n" + "\n".join(box) + "\n")


def read_initial_config(chkpt, reuse_config, beta, gf_struct, *, label, log_level=1):
    """
    Read the configuration of the previous iteration from the checkpoint, if it exists. 

    The configuration is dropped if it is incompatible with the current impurity problem. 

    Parameters
    ----------
    chkpt : dict or None
        ``{"h5": solver checkpoint, "impurity": impurity index, "iteration": DMFT iteration,
        "save": bool}``.
    reuse_config : bool
        Configuration reuse requested for this impurity.
    beta : float
        Inverse temperature of the solve.
    gf_struct : list of (str, int)
        Block structure of the solve.
    label : str
        Names the solve in log messages (e.g. ``"impurity solve"``).
    log_level : int
        ``app_log`` level of the "starting from scratch" warnings (2 for the solves of the
        chemical-potential search, which would otherwise repeat them).

    Returns
    -------
    (triqs_ctseg.ConfigDataT or None, int)
        CT-SEG configuration (``None``: start from scratch) and the iteration it
        came from (``-1`` for none).
    """
    if not reuse_config or chkpt is None:
        return None, -1

    imp, source_iter = chkpt['impurity'], chkpt['iteration'] - 1
    config = load_from_chkpt(chkpt['h5'], source_iter, imp) if source_iter >= 0 else None
    if config is None:
        app_log(log_level, f" No CT-SEG configuration of impurity {imp} at iteration "
                           f"{source_iter};\n  {label} starts from scratch with the full "
                           f"n_warmup_cycles\n")
        return None, -1
    n_incompatible = mpi.all_reduce(int(not _compatible(config, beta, gf_struct)))
    if n_incompatible > 0:
        _warning_box(f"Impurity {imp}, {label}: the CT-SEG configuration of iteration "
                     f"{source_iter} is incompatible with this solve on {n_incompatible} of "
                     f"{mpi.size} MPI ranks (a different beta, block structure or number of "
                     f"colors, or spin-flip operators). Falling back to a cold warm-up: every "
                     f"rank starts from scratch with the full n_warmup_cycles.", log_level)
        return None, -1
    app_log(1, f"Restart: impurity {imp}, {label}, from iteration {source_iter}")
    return config, source_iter


def warn_restart_rejected(chkpt, label, restart_n_warmup_cycles):
    """
    Log a boxed warning: CT-SEG did not use the initial configuration on at least one MPI rank
    (``results.restarted_from_config`` is False). Such a rank found its hybridization determinant
    singular or non-finite under the new hybridization and ran from the empty configuration with
    only ``restart_n_warmup_cycles``, so its contribution to this solve is under-thermalized.
    """
    imp, source_iter = chkpt['impurity'], chkpt['iteration'] - 1
    _warning_box(f"Impurity {imp}, {label}: CT-SEG did not use the configuration of iteration "
                 f"{source_iter} on at least one MPI rank (its hybridization determinant is "
                 f"singular or not finite under the new hybridization; CT-SEG's own warning names "
                 f"the rank and block). That rank started from the empty configuration with only "
                 f"restart_n_warmup_cycles = {restart_n_warmup_cycles}, so its measurements are "
                 f"under-thermalized. To start this impurity from scratch, set "
                 f"reuse_config = False or delete "
                 f"dmft/iter{source_iter}/impurity_{imp}/ctseg_configs.")


def _impurity_path(iteration, impurity_index):
    return ['dmft', f'iter{iteration}', f'impurity_{impurity_index}']


def _open_impurity_group(ar, iteration, impurity_index):
    """The impurity group of ``iteration`` in an archive opened for writing; created if missing."""
    grp = ar
    for key in _impurity_path(iteration, impurity_index):
        if key not in grp:
            grp.create_group(key)
        grp = grp[key]
    return grp


def save_to_chkpt(solver_chkpt, iteration, impurity_index, final_config):
    """
    Write the configurations of all ranks to the checkpoint. 
    
    Configurations are written to 
    ``dmft/iter{iteration}/impurity_{impurity_index}/ctseg_configs`` 
    under the h5 checkpoint. Existing configurations are overwritten. 

    ``triqs_ctseg.save_configs`` is a collective call in which every rank passes 
    its configuration while only the master is passed a valid h5 group to write to.
    """
    from triqs_ctseg import save_configs
    path = "/".join(_impurity_path(iteration, impurity_index) + [_CONFIG_GROUP])
    app_log(2, f"Writing CT-SEG configurations to {solver_chkpt}:{path}")
    if not mpi.is_master_node():
        save_configs(None, _CONFIG_GROUP, final_config)
    else:
        ar, grp = None, None
        try:
            ar = HDFArchive(solver_chkpt, 'a')
            grp = _open_impurity_group(ar, iteration, impurity_index)
            if _CONFIG_GROUP in grp:
                del grp[_CONFIG_GROUP]
        except Exception as e:
            app_log(1, f" [WARNING] cannot prepare {solver_chkpt}:{path} for writing: {e}\n")
            grp = None
        try:
            save_configs(grp, _CONFIG_GROUP, final_config)   
        finally:
            # an open subgroup keeps the file open, which blocks other ranks' opens
            del grp, ar
    mpi.barrier()


def load_from_chkpt(solver_chkpt, iteration, impurity_index):
    """
    Read the stored configuration from the checkpoint. 

    Configurations are read from 
    ``dmft/iter{iteration}/impurity_{impurity_index}/ctseg_configs`` 
    under the h5 checkpoint or ``None`` if the file or the group does not exist. 

    In cases where the checkpoint was written with a different rank count,
    the read is handled by ``triqs_ctseg.load_configs``. With N saved configurations and
    M ranks, rank r receives configuration ``r`` (M = N), ``floor(r N / M)`` (M < N) or
    ``r mod N`` (M > N); in the last case ranks r >= N receive copies, flagged ``is_copy``.
    """
    from triqs_ctseg import load_configs
    keys = _impurity_path(iteration, impurity_index) + [_CONFIG_GROUP]
    path = "/".join(keys)
    app_log(2, f"Checking for CT-SEG configurations at {solver_chkpt}:{path}")
    found, error = False, None
    if mpi.is_master_node() and os.path.isfile(solver_chkpt):
        try:
            with HDFArchive(solver_chkpt, 'r') as ar:
                grp, found = ar, True
                for key in keys:
                    if key not in grp:
                        found = False
                        break
                    grp = grp[key]
                del grp  # an open subgroup keeps the file open, which blocks other ranks' opens
        except Exception as e:
            error = str(e)
    found, error = mpi.bcast((found, error))
    # The master's read-only open and key lookup fail only if the file is unreadable: not HDF5 or
    # truncated (e.g. a job killed while writing it), an I/O or permission error, or an HDF5 lock
    # held by a writer. Every caller has just read this same checkpoint (DMFTState.load,
    # read_impurity_chkpt), so this is a real I/O problem; abort rather than start from scratch,
    # which would hide it and then write to the same file.
    if error:
        raise RuntimeError(f"config_reuse::load_from_chkpt: {error}")
    app_log(2, f"  CT-SEG configurations {'found' if found else 'not found'}")
    if not found:
        return None
    with HDFArchive(solver_chkpt, 'r') as ar:
        config = load_configs(ar["/".join(keys[:-1])], _CONFIG_GROUP)
    mpi.barrier()
    return config
