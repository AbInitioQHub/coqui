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

r"""
 ---------------------------------
     ____ ___   ___  _   _ ___
    / ___/ _ \ / _ \| | | |_ _|
   | |  | | | | | | | | | || |
   | |__| |_| | |_| | |_| || |
    \____\___/ \__\_\\___/|___|
  --------------------------------
 |  Correlated Quantum Interface  |
  --------------------------------

CoQuí python is a lightweight python interface to CoQuí, a software project designed for
ab initio electronic structure calculations beyond the scope of density functional theory (DFT)
for quantum materials.

This python package provides a user-friendly interface to the CoQui library, with parameters directly
parsed to the C++ code and allows for easy integration with other python libraries.

For the pure C++ interface, please refer to the CoQuí C++ documentation [link here].

"""

from .version import *

# Direct api for important routines
from .utils import set_verbosity, MpiHandler, IAFT, app_log, app_debug, app_warning, app_error
from .mean_field import make_mf
from .interaction import make_thc_coulomb, make_chol_coulomb, run_isdf
from .mbpt import run_hf, run_gw, run_evgw, run_qpgw
from .embed import read_proj_info, downfold_1e, downfold_2e, downfold_local_gf, downfold_coulomb, dmft_embed
from .wannier import wannier90, coqui2wannier90, append_wannier90_win, mlwf_h5_from_wannier90_output

# Make submodules available
from . import utils
from . import mean_field
from . import interaction
from . import mbpt
from . import embed
from . import wannier
from . import post_proc
# remove dmft submodule here since it has a heavy dependency on TRIQS.
#from . import dmft
