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

from h5 import HDFArchive
from coqui.dmft.io import save_impurities, _write_impurity_results

_MANDATORY = ['Sigma_infty', 'G_iw', 'Sigma_iw', 'Pi_iw', 'W_iw', 'Chi_iw',
              'orbital_occupations', 'average_order', 'average_sign']


def _results(marker):
    """Minimal results dict; `average_sign` carries a marker to identify where it was written."""
    res = {k: 0.0 for k in _MANDATORY}
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
        save_impurities(dmft_grp=ar['dmft'], solver_results=[_results(10), _results(11)],
                        iteration=3)
    with HDFArchive(chkpt, 'a') as ar:
        save_impurities(dmft_grp=ar['dmft'], solver_results=[_results(21)], iteration=3,
                        impurity_index=0, chkpt_impurity_index=1)
    with HDFArchive(chkpt, 'r') as ar:
        iter_grp = ar['dmft']['iter3']
        assert iter_grp['impurity_0']['results']['average_sign'] == 10.0
        assert iter_grp['impurity_1']['results']['average_sign'] == 21.0


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
