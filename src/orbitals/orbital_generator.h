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


#ifndef ORBITAL_GENERATOR_H
#define ORBITAL_GENERATOR_H

#include "configuration.hpp"

#include "mpi3/communicator.hpp"
#include "IO/ptree/ptree_utilities.hpp"

#include "numerics/distributed_array/nda.hpp"
#include "mean_field/MF.hpp"

#include "orbitals/pgto.h"

namespace orbitals
{

template<MEMORY_SPACE MEM = HOST_MEMORY>
mf::MF add_pgto(mf::MF& mf, std::string fn, std::string basis, std::string type,
                int b0 = -1, bool diag_F = false, bool orthonormalize = false,
                double thresh = 1e-8, bool orthonormalize_by_shell = true); 

mf::MF eigenstate_selection(mf::MF& mf, std::string fn,
                            std::string grid_type, long n0, long nblk);

template<MEMORY_SPACE MEM = HOST_MEMORY>
void orbital_factory(mf::MF &mf, ptree const& pt);

}

#endif
