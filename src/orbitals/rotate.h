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


#ifndef ORBITALS_ROTATE_H
#define ORBITALS_ROTATE_H

#include "configuration.hpp"

#include "mpi3/communicator.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "mean_field/MF.hpp"

namespace orbitals
{


template<MEMORY_SPACE MEM = HOST_MEMORY, utils::Communicator comm_t>
void orthonormalize(memory::darray_t<memory::array<MEM, ComplexType, 4>, comm_t>& psi,
                    double cutoff = 1e-8); 

template<MEMORY_SPACE MEM = HOST_MEMORY, utils::Communicator comm_t>
void orthonormalize(nda::array<int,2> const& ranges,
                    memory::darray_t<memory::array<MEM, ComplexType, 4>, comm_t>& psi);
                                        

template<MEMORY_SPACE MEM = HOST_MEMORY, utils::Communicator comm_t>
auto canonicalize_diagonal_basis(mf::MF& mf, 
                    memory::darray_t<memory::array<MEM, ComplexType, 4>, comm_t>& psi)
  -> nda::array<ComplexType, 3>;


//void diagonalize() {};

}

#endif
