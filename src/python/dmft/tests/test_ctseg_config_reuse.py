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
CT-SEG configuration reuse in the DMFT package (coqui.dmft.ctseg.config_reuse).

Requires a TRIQS/ctseg build with configuration reuse (one that provides `triqs_ctseg.ConfigDataT`);
the whole file is skipped otherwise.

Serial:  python -m pytest <this file> -v                      (from outside the repository)
MPI:     mpirun -n 2 python -m pytest <this file> -v
"""
import sys
import types

import numpy as np
import pytest

pytest.importorskip("triqs", reason="triqs is required for dmft tests")
triqs_ctseg = pytest.importorskip("triqs_ctseg", reason="triqs_ctseg is required for these tests")
if not hasattr(triqs_ctseg, "ConfigDataT"):
    pytest.skip("requires a TRIQS/ctseg build with configuration reuse (ConfigDataT)",
                allow_module_level=True)

from triqs.utility import mpi
from h5 import HDFArchive
from coqui.dmft.ctseg import seed_for_rank, ctseg_solver, config_reuse
from coqui.dmft.ctseg import save_to_chkpt, load_from_chkpt, read_initial_config
from coqui.dmft.io import _normalize_solver_params_list

_GF_STRUCT = [('up_0', 1), ('up_1', 1), ('down_0', 1), ('down_1', 1)]


def _shared_path(tmp_path, name):
    """Pytest's tmp_path is per-rank under MPI: broadcast the master's tmp_path to all ranks."""
    return mpi.bcast(str(tmp_path)) + "/" + name


# ---------- read_initial_config, with load_from_chkpt replaced ----------

_ONE_SEGMENT = np.array([[2**63, 2**62, 0, 0]], dtype=np.uint64)


def _fake_config(beta=10.0, block_sizes=(1, 1, 1, 1), seglists=None, jperp=0):
    """Stand-in for a ConfigDataT with the fields read_initial_config checks."""
    n_colors = sum(block_sizes)
    return types.SimpleNamespace(
        beta=beta, block_sizes=list(block_sizes),
        seglists=seglists if seglists is not None else [_ONE_SEGMENT] * n_colors,
        Jperp_lines=np.zeros((jperp, 2), dtype=np.uint64))


_CHKPT = {"h5": "unused.h5", "impurity": 2, "iteration": 5, "save": True}


def test_read_initial_config_first_iteration(monkeypatch):
    """At iteration 0 there is no iteration N-1, so the solve starts cold.

    load_from_chkpt is replaced by one that returns a compatible configuration for every
    iteration, so (None, -1) can only come from read_initial_config's own iteration-0 check,
    not from a configuration that is missing.
    """
    monkeypatch.setattr(config_reuse, 'load_from_chkpt',
                        lambda solver_chkpt, iteration, impurity_index: _fake_config())
    config, source_iter = read_initial_config(chkpt=dict(_CHKPT, iteration=0), reuse_config=True,
                                              beta=10.0, gf_struct=_GF_STRUCT,
                                              label="test", log_level=1)
    assert (config, source_iter) == (None, -1)


@pytest.mark.parametrize("cfg", [
    _fake_config(beta=12.0),                                        # other beta
    _fake_config(block_sizes=(2, 2)),                               # other block structure
    _fake_config(seglists=[_ONE_SEGMENT] * 3),                      # other number of colors
    _fake_config(jperp=1),                                          # spin-flip lines
    _fake_config(seglists=[np.array([[2**63, 2**62, 1, 0]], dtype=np.uint64)] * 4),  # J flags
])
def test_read_initial_config_incompatible(monkeypatch, cfg):
    """A stored configuration that does not fit the current impurity problem is not reused.

    Each case breaks one check of config_reuse._compatible: beta, block structure, number of
    colors, spin-flip (Jperp) lines, or J flags on a segment. load_from_chkpt is replaced by
    one that returns the configuration for every iteration, so (None, -1) can only come from
    the rejection, not from a configuration that is missing.
    """
    monkeypatch.setattr(config_reuse, 'load_from_chkpt',
                        lambda solver_chkpt, iteration, impurity_index: cfg)
    config, source_iter = read_initial_config(chkpt=_CHKPT, reuse_config=True,
                                              beta=10.0, gf_struct=_GF_STRUCT,
                                              label="test", log_level=1)
    assert (config, source_iter) == (None, -1)


# ---------- toy impurity problem (DLR mesh, static U, D0 = 0) ----------

_SOLVE = dict(n_cycles=2000, length_cycle=50, n_tau=2001)


def _toy_problem():
    beta = 10.0
    from triqs.gfs import MeshDLRImFreq, Gf, BlockGf, Block2Gf, iOmega_n, inverse
    from triqs.operators import n
    names = [b for b, _ in _GF_STRUCT]
    mesh_f = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=10.0, eps=1e-8, symmetrize=True)
    mesh_b = MeshDLRImFreq(beta=beta, statistic='Boson', w_max=10.0, eps=1e-8, symmetrize=True)
    delta = BlockGf(name_list=names, block_list=[Gf(mesh=mesh_f, target_shape=[1, 1]) for _ in names])
    for _, d in delta:
        d << 1.0 * inverse(iOmega_n - 0.3)
    D0 = Block2Gf(names, names, [[Gf(mesh=mesh_b, target_shape=[1, 1]) for _ in names] for _ in names])
    h_int = 2.0 * (n('up_0', 0) * n('down_0', 0) + n('up_1', 0) * n('down_1', 0))
    h0 = -1.0 * sum(n(b, 0) for b in names)
    return dict(delta=delta, h0=h0, D0=D0, h_int=h_int, beta=beta)


def _cold_config(seed=11):
    """Solve the toy problem and access its final configuration"""
    from triqs.gfs import make_gf_imtime
    p, n_tau = _toy_problem(), _SOLVE['n_tau']
    S = triqs_ctseg.Solver(gf_struct=_GF_STRUCT, beta=p['beta'], n_tau=n_tau, n_tau_bosonic=n_tau)
    S.Delta_tau << make_gf_imtime(p['delta'], n_tau)
    for name1, name2 in p['D0'].indices:
        S.D0_tau[name1, name2] << make_gf_imtime(p['D0'][name1, name2], n_tau)
    S.solve(h_loc0=p['h0'], h_int=p['h_int'], n_cycles=_SOLVE['n_cycles'],
            length_cycle=_SOLVE['length_cycle'], n_warmup_cycles=200,
            random_seed=seed_for_rank(base=seed))
    return S.final_config


def _same_configuration(a, b):
    return (a.beta == b.beta and list(a.block_sizes) == list(b.block_sizes)
            and len(a.seglists) == len(b.seglists)
            and all(np.array_equal(x, y) for x, y in zip(a.seglists, b.seglists))
            and np.array_equal(a.Jperp_lines, b.Jperp_lines))


# ---------- solve functions, with real CT-SEG solves and a real checkpoint ----------

def _chkpt_holding_iteration_4(tmp_path):
    """Create a checkpoint holding a configuration for iteration 4, impurity 0,
    and return the checkpoint information dictionary"""
    h5 = _shared_path(tmp_path=tmp_path, name="solve.h5")
    save_to_chkpt(solver_chkpt=h5, iteration=4, impurity_index=0, final_config=_cold_config())
    return {"h5": h5, "impurity": 0, "iteration": 5, "save": True}


def test_density_solve_restarts_and_never_saves(tmp_path):
    """
    Runs solve_density_dynamic_u_dlr_mesh at iteration 5 with a configuration stored from
    iteration 4.

    CT-SEG should restart from iteration 4. Since this is a density-only solve, it should
    not save the resulting configuration.
    """
    chkpt = _chkpt_holding_iteration_4(tmp_path)
    p = _toy_problem()
    res = ctseg_solver.solve_density_dynamic_u_dlr_mesh(
        Delta_iw=p['delta'], h_loc0=p['h0'], D0_iw=p['D0'], h_int=p['h_int'], chkpt=chkpt, **_SOLVE,
        n_warmup_cycles=500, reuse_config=True, restart_n_warmup_cycles=5)
    assert res['restarted_from_config'] is True and res['config_source_iter'] == 4
    assert load_from_chkpt(solver_chkpt=chkpt['h5'], iteration=5, impurity_index=0) is None


def test_production_solve_save_conditions(tmp_path):
    """
    Runs solve_dynamic_dlr_mesh at iteration 5, with a configuration stored as iteration 4,
    for different combinations of reuse_config, chkpt and chkpt["save"]:

    (1) chkpt["save"] False, as in solve_impurities_from_chkpt(save_results=False): restarts,
        saves nothing;
    (2) reuse_config off: starts cold, saves nothing;
    (3) no checkpoint: starts cold;
    (4) reuse_config on, chkpt["save"] True: restarts from iteration 4 and saves its final
        configuration, which differs from iteration 4's, as iteration 5.
    Case (4) runs last because cases (1) and (2) check that nothing is stored as iteration 5.
    """
    chkpt = _chkpt_holding_iteration_4(tmp_path)
    p = _toy_problem()
    params = dict(_SOLVE, n_warmup_cycles=500, restart_n_warmup_cycles=5)

    res = ctseg_solver.solve_dynamic_dlr_mesh(
        Delta_iw=p['delta'], h_loc0=p['h0'], D0_iw=p['D0'], h_int=p['h_int'],
        chkpt=dict(chkpt, save=False), reuse_config=True, **params)
    assert res['restarted_from_config'] is True
    assert load_from_chkpt(solver_chkpt=chkpt['h5'], iteration=5, impurity_index=0) is None

    res = ctseg_solver.solve_dynamic_dlr_mesh(
        Delta_iw=p['delta'], h_loc0=p['h0'], D0_iw=p['D0'], h_int=p['h_int'],
        chkpt=chkpt, reuse_config=False, **params)
    assert res['restarted_from_config'] is False
    assert load_from_chkpt(solver_chkpt=chkpt['h5'], iteration=5, impurity_index=0) is None

    res = ctseg_solver.solve_dynamic_dlr_mesh(
        Delta_iw=p['delta'], h_loc0=p['h0'], D0_iw=p['D0'], h_int=p['h_int'],
        chkpt=None, reuse_config=True, **params)
    assert res['restarted_from_config'] is False

    res = ctseg_solver.solve_dynamic_dlr_mesh(
        Delta_iw=p['delta'], h_loc0=p['h0'], D0_iw=p['D0'], h_int=p['h_int'],
        chkpt=chkpt, reuse_config=True, **params)
    assert res['restarted_from_config'] is True and res['config_source_iter'] == 4
    saved = load_from_chkpt(solver_chkpt=chkpt['h5'], iteration=5, impurity_index=0)
    previous = load_from_chkpt(solver_chkpt=chkpt['h5'], iteration=4, impurity_index=0)
    assert saved is not None and not _same_configuration(saved, previous)


# ---------- checkpoint ----------

def test_chkpt_round_trip(tmp_path):
    """A real ct-seg configuration survives save_to_chkpt / load_from_chkpt through real HDF5.

    The configuration is saved as iteration 3, impurity 0, and read back unchanged. Another
    impurity and a missing file give None. Saving iteration 3 again, as a re-solve does,
    replaces the stored configuration. Under MPI every rank saves and reads back its own
    configuration.
    """
    chkpt = _shared_path(tmp_path=tmp_path, name="chkpt.h5")
    cfg = _cold_config()
    save_to_chkpt(solver_chkpt=chkpt, iteration=3, impurity_index=0, final_config=cfg)
    loaded = load_from_chkpt(solver_chkpt=chkpt, iteration=3, impurity_index=0)
    assert _same_configuration(loaded, cfg)
    assert load_from_chkpt(solver_chkpt=chkpt, iteration=3, impurity_index=1) is None
    missing = _shared_path(tmp_path=tmp_path, name="missing.h5")
    assert load_from_chkpt(solver_chkpt=missing, iteration=3, impurity_index=0) is None
    cfg2 = _cold_config(seed=12)
    # a re-solve replaces the group
    save_to_chkpt(solver_chkpt=chkpt, iteration=3, impurity_index=0, final_config=cfg2)
    loaded = load_from_chkpt(solver_chkpt=chkpt, iteration=3, impurity_index=0)
    assert _same_configuration(loaded, cfg2)


def test_save_to_chkpt_master_error_is_collective(tmp_path):
    """A save that fails on the master raises on every rank instead of hanging the others.

    The checkpoint holds a dataset where save_to_chkpt expects the "dmft" group, so the
    master fails while preparing the group, and every rank must raise RuntimeError. The
    collective part is only exercised under mpirun -n 2.
    """
    chkpt = _shared_path(tmp_path=tmp_path, name="bad_checkpoint.h5")
    cfg = _cold_config()
    if mpi.is_master_node():
        with HDFArchive(chkpt, 'w') as ar:
            ar['dmft'] = 1  # write a dataset, not a group
    mpi.barrier()
    with pytest.raises(RuntimeError):
        save_to_chkpt(solver_chkpt=chkpt, iteration=0, impurity_index=0, final_config=cfg)


def test_load_from_chkpt_master_error_is_collective(tmp_path):
    """A load that fails on the master raises on every rank instead of hanging the others.

    The checkpoint is a text file, so the master cannot open it to check for the
    configuration, and every rank must raise RuntimeError naming load_from_chkpt. The
    collective part is only exercised under mpirun -n 2.
    """
    chkpt = _shared_path(tmp_path=tmp_path, name="not_hdf5.txt")
    if mpi.is_master_node():
        with open(chkpt, 'w') as f:
            f.write("This is not an HDF5 file\n")
    mpi.barrier()
    with pytest.raises(RuntimeError, match="load_from_chkpt"):
        load_from_chkpt(solver_chkpt=chkpt, iteration=0, impurity_index=0)


# ---------- input validation ----------

_BASE = dict(n_cycles=1000, n_warmup_cycles=100, length_cycle=50)


def test_reuse_requires_restart_warmup():
    """reuse_config = True requires restart_n_warmup_cycles to be an integer >= 1.

    A missing, zero, negative or fractional value is rejected with a ValueError naming the key.
    """
    for bad in (None, 0, -3, 0.5):
        p = dict(_BASE, reuse_config=True)
        if bad is not None:
            p['restart_n_warmup_cycles'] = bad
        with pytest.raises(ValueError, match="restart_n_warmup_cycles"):
            _normalize_solver_params_list(solver_params_list=[p], comm_size=1)


def test_restart_warmup_cast_and_defaults():
    """How input normalization treats restart_n_warmup_cycles.

    An integral float (1e4) is cast to int; the chemical-potential search inherits the value
    when it sets none of its own; and, unlike n_cycles, which is divided over the ranks
    (2 here), warm-up counts are per rank and are not divided.
    """
    p = dict(_BASE, reuse_config=True, restart_n_warmup_cycles=1e4,
             chemical_potential={'tolerance': 0.1})
    out = _normalize_solver_params_list(solver_params_list=[p], comm_size=2)[0]
    assert out['restart_n_warmup_cycles'] == 10000
    assert isinstance(out['restart_n_warmup_cycles'], int)
    assert out['chemical_potential']['restart_n_warmup_cycles'] == 10000
    assert out['n_cycles'] == 500          # n_cycles is split over ranks, warm-ups are not


def test_reuse_needs_ctseg_with_config_reuse(monkeypatch):
    """reuse_config = True with a ct-seg build that cannot restart fails at input time.

    triqs_ctseg is replaced by an empty module (no ConfigDataT). Normalizing the input must
    raise ImportError naming ConfigDataT, before any solve starts.
    """
    monkeypatch.setitem(sys.modules, 'triqs_ctseg', types.ModuleType('triqs_ctseg'))
    with pytest.raises(ImportError, match="ConfigDataT"):
        _normalize_solver_params_list(
            solver_params_list=[dict(_BASE, reuse_config=True, restart_n_warmup_cycles=10)],
            comm_size=1)
