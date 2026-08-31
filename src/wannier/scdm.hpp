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


#ifndef WANNIER_SCDM_H 
#define WANNIER_SCDM_H 

#include "configuration.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/mpi_context.h"
#include "nda/nda.hpp"
#include "mean_field/MF.hpp"

namespace wannier 
{

  /**
   * Calculates the expansion coefficients of the localized orbitals, A(k,m,n) = <psi^{k}_m | g^{k}_n>. 
   *
   * @param nwann     - [INPUT] Number of localized functions. 
   * @param band_list  - [INPUT] List of orbitals included in calculation.
   * @param fi        - [INPUT] Weight of each orbital. 
   *
   */
//  template<MEMORY_SPACE MEM = HOST_MEMORY>
  auto scdm(utils::mpi_context_t<mpi3::communicator> &mpi, mf::MF &mf, ptree const& pt, int nwann,
          nda::array<int, 1> const& kp_map, nda::array<double,2> const& wann_kp,
          nda::array<int,1> const& band_list, bool transpose, bool write_to_file);

}

#include "wannier/scdm.icc"

#endif
