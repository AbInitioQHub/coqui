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


#ifndef HAMILTONIANS_H
#define HAMILTONIANS_H

#include "configuration.hpp"

#include "mpi3/communicator.hpp"
#include "utilities/mpi_context.h"
#include "IO/ptree/ptree_utilities.hpp"

#include "mean_field/MF.hpp"

#include "methods/ERI/thc_reader_t.hpp"
#include "methods/ERI/chol_reader_t.hpp"

namespace methods 
{

template<MEMORY_SPACE MEM = HOST_MEMORY>
void add_core_hamiltonian(mf::MF &mf, ptree const& pt);

template<MEMORY_SPACE MEM = HOST_MEMORY>
void add_thc_hamiltonian_components(mf::MF &mf,
                          thc_reader_t& thc, ptree const& pt);

template<MEMORY_SPACE MEM = HOST_MEMORY>
void add_cholesky_hamiltonian_components(mf::MF &mf,
                          chol_reader_t& chol, ptree const& pt);

}

#endif
