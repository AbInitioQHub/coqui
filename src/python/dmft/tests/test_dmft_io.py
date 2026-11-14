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
Impurity checkpoint I/O of the DMFT package (coqui.dmft.io). Serial; run from outside the
repository: python -m pytest <this file> -v
"""
import pytest

pytest.importorskip("triqs", reason="triqs is required for dmft tests")

import numpy as np
from h5 import HDFArchive
from coqui.dmft.io import (save_impurities, _write_impurity_results, convert_legacy_impurity_block,
                           _write_solver_results, SOLVER_AGNOSTIC_KEYS)

_REQUIRED = ['Sigma_infty', 'Sigma_iw_data', 'G_iw_data', 'orbital_occupations', 'density']


def _results(marker):
    """Minimal results dict; `average_sign` carries a marker to identify where it was written."""
    res = {k: np.zeros(1) for k in _REQUIRED}
    res['density'] = 1.0
    res['orbital_occupations'] = {'up_0': np.array([0.5])}
    res['average_sign'] = float(marker)
    return res


def test_save_subset_entry_to_its_own_group(tmp_path):
    """save_impurities writes each result to the group named by chkpt_impurity_index.

    Two impurities are saved at iteration 3; then impurity 1 is re-solved alone and saved as a
    one-entry list (impurity_index=0, chkpt_impurity_index=1), as solve_impurities_from_chkpt
    does for a subset of impurities. The new result must land in impurity_1, not in the
    impurity_0 its list position would suggest, and impurity_0 must be left untouched.
    """
    chkpt = str(tmp_path / "chkpt.h5")
    with HDFArchive(chkpt, 'w') as ar:
        ar.create_group('dmft')
        save_impurities(dmft_grp=ar['dmft'], impurity_results=[_results(10), _results(11)],
                        iteration=3)
    with HDFArchive(chkpt, 'a') as ar:
        save_impurities(dmft_grp=ar['dmft'], impurity_results=[_results(21)], iteration=3,
                        impurity_index=0, chkpt_impurity_index=1)
    with HDFArchive(chkpt, 'r') as ar:
        iter_grp = ar['dmft']['iter3']
        assert iter_grp['impurity_0']['results']['average_sign'] == 10.0
        assert iter_grp['impurity_1']['results']['average_sign'] == 21.0


def test_final_iter_marker_is_not_moved_by_save_impurities(tmp_path):
    """save_impurities never writes the restart marker dmft/final_iter (the loop's writers do),
    so rewriting a stored iteration does not change where the loop restarts.

    Setup: an empty dmft group in a file in tmp_path.

    Cases (in order; each builds on the file state of the previous one):
      1. save_impurities at iteration 3 leaves the file without final_iter.
      2. The loop sets final_iter = 3.
      3. save_impurities at iteration 1 (as solve_impurities_from_chkpt rewriting an earlier
         iteration) leaves final_iter at 3.
      4. save_impurities with iteration=-1 resolves the iteration from the marker: it writes
         iter4 and returns 4, still without moving final_iter.
    """
    chkpt = str(tmp_path / "chkpt.h5")
    with HDFArchive(chkpt, 'w') as ar:
        ar.create_group('dmft')
        save_impurities(dmft_grp=ar['dmft'], impurity_results=[_results(1)], iteration=3)
    with HDFArchive(chkpt, 'r') as ar:
        assert 'final_iter' not in ar['dmft'].keys()
    with HDFArchive(chkpt, 'a') as ar:
        ar['dmft']['final_iter'] = 3
        save_impurities(dmft_grp=ar['dmft'], impurity_results=[_results(2)], iteration=1)
        written = save_impurities(dmft_grp=ar['dmft'], impurity_results=[_results(3)], iteration=-1)
    assert written == 4
    with HDFArchive(chkpt, 'r') as ar:
        assert ar['dmft']['final_iter'] == 3
        assert set(k for k in ar['dmft'].keys() if k.startswith('iter')) == {'iter1', 'iter3', 'iter4'}


def test_results_writer_skips_none_optional_keys(tmp_path):
    """h5 cannot store None: an optional key whose value is None is not written."""
    res = _results(0)
    res.update(config_source_iter=-1, warmup_trace=None)
    with HDFArchive(str(tmp_path / "res.h5"), 'w') as ar:
        _write_impurity_results(h5_grp=ar, impurity_results=res)
    with HDFArchive(str(tmp_path / "res.h5"), 'r') as ar:
        grp = ar['results']
        assert 'warmup_trace' not in grp
        assert grp['config_source_iter'] == -1


def test_convert_legacy_impurity_block():
    """convert_legacy_impurity_block converts a legacy flat impurity block to the nested solver layout.

    Setup: dict blocks as convert_gw_edmft_params receives them.

    Cases:
      1. Flat block: every key that is not solver-agnostic moves under 'solver' with type "ctseg"; solver-agnostic keys
         stay; the input dict is not modified.
      2. Flat block with screen_j and reuse_config: both land under 'solver' (they are solver
         keys), not rejected.
      3. Nested block with a stray key ('n_cycles' beside 'solver'): KeyError naming it.
      4. Nested block without 'type' in solver: KeyError naming 'type'.
      5. Nested block: returned equal to the input.
      6. Legacy chemical_potential carrying solver keys beside tolerance (flat and nested block):
         those keys move into solver.mu_search, tolerance stays alone, the input is not modified;
         a block that has both solver.mu_search and such keys raises KeyError naming both.
    """
    flat = {'init_imp_results': 'dc', 'degenerate_blk': [[0, 1]], 'n_cycles': 10,
            'perform_tail_fit': True, 'chemical_potential': {'tolerance': 0.1}}
    nested = convert_legacy_impurity_block(imp_block=flat, index=0)
    assert nested['solver'] == {'type': 'ctseg', 'n_cycles': 10, 'perform_tail_fit': True}
    assert set(nested) == {'init_imp_results', 'degenerate_blk', 'chemical_potential', 'solver'}
    assert 'solver' not in flat and 'n_cycles' in flat
    nested = convert_legacy_impurity_block(imp_block={'n_cycles': 1, 'screen_j': True, 'reuse_config': True}, index=1)
    assert nested['solver'] == {'type': 'ctseg', 'n_cycles': 1, 'screen_j': True, 'reuse_config': True}
    with pytest.raises(KeyError, match="n_cycles"):
        convert_legacy_impurity_block(imp_block={'solver': {'type': 'ctseg'}, 'n_cycles': 1}, index=0)
    with pytest.raises(KeyError, match="type"):
        convert_legacy_impurity_block(imp_block={'solver': {'n_cycles': 1}}, index=0)
    good = {'solver': {'type': 'ctseg', 'n_cycles': 1}, 'init_imp_results': 'zero'}
    assert convert_legacy_impurity_block(imp_block=good, index=0) == good
    legacy_mu = {'tolerance': 0.05, 'n_cycles': 100, 'n_warmup_cycles': 10}
    for block in ({'n_cycles': 1, 'chemical_potential': dict(legacy_mu)},
                  {'solver': {'type': 'ctseg', 'n_cycles': 1}, 'chemical_potential': dict(legacy_mu)}):
        nested = convert_legacy_impurity_block(imp_block=block, index=0)
        assert nested['chemical_potential'] == {'tolerance': 0.05}
        assert nested['solver'] == {'type': 'ctseg', 'n_cycles': 1,
                                    'mu_search': {'n_cycles': 100, 'n_warmup_cycles': 10}}
        assert block['chemical_potential'] == legacy_mu and 'mu_search' not in block.get('solver', {})
    with pytest.raises(KeyError, match="mu_search"):
        convert_legacy_impurity_block(imp_block={'solver': {'type': 'ctseg', 'mu_search': {'n_cycles': 5}},
                                       'chemical_potential': dict(legacy_mu)}, index=0)
    assert 'solver' in SOLVER_AGNOSTIC_KEYS and 'screen_j' not in SOLVER_AGNOSTIC_KEYS


def test_write_impurity_results_and_solver_results(tmp_path):
    """The results writer stores every non-None key, requires the contract's arrays, and
    _write_solver_results replaces the solver_results group as a whole.

    Setup: a results dict with a diagnostic and a None entry; solver_data dicts.

    Cases:
      1. Every non-None key of the dict is under results/; the None key is absent.
      2. A dict without 'G_iw_data' raises KeyError naming it.
      3. _write_solver_results writes solver_results/ next to results/; a second call with other
         keys leaves none of the first call's keys; an empty dict removes the group. results/ is
         untouched, so a file with TRIQS objects under results/ (an old checkpoint) still reads.
      4. save_impurities takes solver_data as a list over impurities like impurity_results:
         with impurity_index=1 only impurity_1 gets its results/ and solver_results/; with
         impurity_index=-1 every impurity is written, a None entry leaving solver_results/
         absent; a list of another length raises ValueError.
    """
    res = _results(3)
    res['warmup_trace'] = None
    with HDFArchive(str(tmp_path / "res.h5"), 'w') as ar:
        _write_impurity_results(h5_grp=ar, impurity_results=res)
        _write_solver_results(h5_grp=ar, solver_data={'G_iw': np.ones(2), 'note': 'first'})
    with HDFArchive(str(tmp_path / "res.h5"), 'r') as ar:
        assert ar['results']['average_sign'] == 3.0 and 'warmup_trace' not in ar['results']
        assert set(ar['solver_results'].keys()) == {'G_iw', 'note'}
    with HDFArchive(str(tmp_path / "res.h5"), 'a') as ar:
        _write_solver_results(h5_grp=ar, solver_data={'M_tau': np.ones(3)})
    with HDFArchive(str(tmp_path / "res.h5"), 'r') as ar:
        assert set(ar['solver_results'].keys()) == {'M_tau'} and 'average_sign' in ar['results']
    with HDFArchive(str(tmp_path / "res.h5"), 'a') as ar:
        _write_solver_results(h5_grp=ar, solver_data={})
    with HDFArchive(str(tmp_path / "res.h5"), 'r') as ar:
        assert 'solver_results' not in ar.keys()
    bad = _results(0)
    del bad['G_iw_data']
    with HDFArchive(str(tmp_path / "bad.h5"), 'w') as ar:
        with pytest.raises(KeyError, match="G_iw_data"):
            _write_impurity_results(h5_grp=ar, impurity_results=bad)
    with HDFArchive(str(tmp_path / "one.h5"), 'w') as ar:
        ar.create_group('dmft')
        save_impurities(dmft_grp=ar['dmft'], impurity_results=[_results(1), _results(2)],
                        solver_data=[{'M_tau': np.ones(3)}, {'G_iw': np.ones(2)}],
                        impurity_index=1, iteration=4)
        with pytest.raises(ValueError, match="same length"):
            save_impurities(dmft_grp=ar['dmft'], impurity_results=[_results(1), _results(2)],
                            solver_data=[{'G_iw': np.ones(2)}], impurity_index=-1, iteration=4)
    # no subgroup reference may outlive the 'with' block: it would keep the file open read-only
    with HDFArchive(str(tmp_path / "one.h5"), 'r') as ar:
        assert set(ar['dmft/iter4/impurity_1/solver_results'].keys()) == {'G_iw'}
        assert ar['dmft/iter4/impurity_1/results/average_sign'] == 2.0
        assert 'impurity_0' not in ar['dmft/iter4'].keys()
    with HDFArchive(str(tmp_path / "one.h5"), 'a') as ar:
        save_impurities(dmft_grp=ar['dmft'], impurity_results=[_results(1), _results(2)],
                        solver_data=[None, {'M_tau': np.ones(3)}], impurity_index=-1, iteration=5)
    with HDFArchive(str(tmp_path / "one.h5"), 'r') as ar:
        assert 'solver_results' not in ar['dmft/iter5/impurity_0'].keys()
        assert set(ar['dmft/iter5/impurity_1/solver_results'].keys()) == {'M_tau'}


def test_update_impurity_results_from_chkpt_drops_legacy_solver_objects(tmp_path):
    """Restarting from a checkpoint written before the solver interface does not carry its TRIQS
    solver objects into the next iteration's results.

    Setup: a checkpoint whose results/ holds the native arrays plus the legacy keys G_iw,
    Sigma_iw, Pi_iw, W_iw, Chi_iw (dummies), as CT-SEG wrote them before solver_results/ existed.

    Cases:
      1. update_impurity_results_from_chkpt loads the native keys (mu_imp, density, ...) and none
         of the legacy keys.
      2. Writing the updated dict with _write_impurity_results therefore produces a results/
         group without the legacy keys (they belong to solver_results/ now).
    """
    from coqui.dmft.io import update_impurity_results_from_chkpt
    chkpt = str(tmp_path / "old.h5")
    old = _results(7)
    old.update(mu_imp=0.25, G_iw=np.ones(2), Sigma_iw=np.ones(2), Pi_iw=np.ones(2), W_iw=np.ones(2),
               Chi_iw=np.ones(2))
    with HDFArchive(chkpt, 'w') as ar:
        ar.create_group('dmft')
        save_impurities(dmft_grp=ar['dmft'], impurity_results=[old], iteration=2)
        ar['dmft']['final_iter'] = 2
    res = {'gf_struct': [('up_0', 1)], 'mu_imp': 0.0, 'Sigma_infty': None}
    iteration = update_impurity_results_from_chkpt(impurity_results=[res], h5_filename=chkpt, iteration=-1)
    assert iteration == 2 and res['mu_imp'] == 0.25 and res['average_sign'] == 7.0
    assert not ({'G_iw', 'Sigma_iw', 'Pi_iw', 'W_iw', 'Chi_iw'} & set(res))
    with HDFArchive(str(tmp_path / "new.h5"), 'w') as ar:
        _write_impurity_results(h5_grp=ar, impurity_results=res)
    with HDFArchive(str(tmp_path / "new.h5"), 'r') as ar:
        assert not ({'G_iw', 'Sigma_iw', 'Pi_iw', 'W_iw', 'Chi_iw'} & set(ar['results'].keys()))
        assert ar['results']['mu_imp'] == 0.25
