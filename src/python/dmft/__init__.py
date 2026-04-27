"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team

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

_TRIQS_AVAILABLE = False
_TRIQS_CTSEG_AVAILABLE = False
_TRIQS_IMPORT_ERROR = None

try:
  from .scf_driver import *
  from .dmft_state import *
  from .triqs_utils import *
  from .chemical_potential import *
  from .weiss import *
  from .bath_fit import *
  from .io import *
  _TRIQS_AVAILABLE = True
except ImportError as _e:
  _TRIQS_IMPORT_ERROR = _e

if _TRIQS_AVAILABLE:
  try:
    from . import ctseg
    _TRIQS_CTSEG_AVAILABLE = True
  except ImportError:
    pass

def __getattr__(name):
  if not _TRIQS_AVAILABLE:
    raise ImportError(
      f"'{name}' is not available because the coqui.dmft submodule requires TRIQS core library and TRIQS/ModEST. "
      f"Ensure TRIQS and TRIQS/ModEST are installed (https://github.com/triqs/triqs and https://github.com/TRIQS/modest). "
      f"Original error: {_TRIQS_IMPORT_ERROR}"
    )
  if name == "ctseg" and not _TRIQS_CTSEG_AVAILABLE:
    raise ImportError(
      f"'ctseg' is not available because it requires TRIQS/CTSEG. "
      f"Please ensure TRIQS/CTSEG is installed (https://github.com/triqs/ctseg)."
    )
  raise AttributeError(f"module 'coqui.dmft' has no attribute '{name}'")
