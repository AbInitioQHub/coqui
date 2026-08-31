/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */


#ifndef HAMILTONIAN_ADD_KINETIC_H
#define HAMILTONIAN_ADD_KINETIC_H

#include "configuration.hpp"
#include "grids/g_grids.hpp"
#include "nda/concepts.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "mean_field/MF.hpp"

namespace hamilt
{

/**
 * Adds the kinetic contribution to hpsi: hpsi(s,k,a,G) += 0.5*|G+k|^2 * psi(s,k,b,G)
 * Assumes psi and hpsi are in the 'w' grid.
 *
 * @param npol  - [INPUT] # of polarizations
 * @param kpts  - [INPUT] k-points in the Brillouin zones
 * @param wfc_g - [INPUT] Handler for "Wavefunction" plane-wave grid
 * @param psi   - [INPUT] Single-particle basis (s,k,a,g) where g is in the 'w' grid
 * @param hpsi  - [INPUT] \hat{h} * psi where h is an arbitrary local operator
 *                [OUTPUT] (\hat{h} + 0.5*|G+k|^2) * psi
 */
void add_kinetic(int npol,
                 nda::ArrayOfRank<2> auto const& kpts,
                 grids::truncated_g_grid const& wfc_g,
                 math::nda::DistributedArrayOfRank<4> auto const& psi,
                 math::nda::DistributedArrayOfRank<4> auto & hpsi);

}

#endif
