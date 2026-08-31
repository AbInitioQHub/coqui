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

import numpy as np


def diis_solve(res_list):
    space_size = len(res_list)

    B = np.zeros((space_size, space_size), dtype=float)
    for i in range(space_size):
        for j in range(space_size):
            B[i, j] = np.dot(res_list[i], res_list[j]).real

    coeffs = np.linalg.solve(B, np.ones(space_size))
    # renormalization factor
    N = np.sum(coeffs)

    return coeffs / N

