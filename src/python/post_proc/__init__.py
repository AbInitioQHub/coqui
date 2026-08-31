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

from .post_proc import (
  band_interpolation,
  spectral_interpolation,
  local_dos,
  unfold_bz,
  dump_vxc,
  dump_hartree,
  pade,
  aaa_adapol_on_mesh,
  aaa_adapol,
  aaa_adapol_imag,
  minipole,
)
from .plot_utils import band_plot, spectral_plot

_TRIQS_AVAILABLE = False
_TRIQS_IMPORT_ERROR = None

try:
  from .analytic_cont import (
    pade_triqs,
    maxent_triqs,
    maxent_sigma,
    maxent_sigma_k,
  )
  _TRIQS_AVAILABLE = True
except ImportError as _e:
  _TRIQS_IMPORT_ERROR = _e

# Names from analytic_cont that require TRIQS
_TRIQS_AC_NAMES = frozenset(["maxent_sigma", "maxent_sigma_k", "pade_triqs", "maxent_triqs"])

def __getattr__(name):
  if name in _TRIQS_AC_NAMES:
    raise ImportError(
      f"'{name}' is not available because it requires TRIQS. "
      f"Ensure TRIQS is installed (https://triqs.github.io). "
      f"Original error: {_TRIQS_IMPORT_ERROR}"
    )
  raise AttributeError(f"module 'coqui.post_proc' has no attribute '{name}'")

__all__ = [
  "band_interpolation", "spectral_interpolation",
  "local_dos", "unfold_bz", "dump_vxc", "dump_hartree",
  "pade", "aaa_adapol_on_mesh", "aaa_adapol", "aaa_adapol_imag", "minipole",
  "band_plot", "spectral_plot",
]

if _TRIQS_AVAILABLE:
  __all__.extend(["maxent_sigma", "maxent_sigma_k", "pade_triqs", "maxent_triqs"])
