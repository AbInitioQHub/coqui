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


#ifndef COQUI_MODEL_HAMILTONIAN_UTILS_HPP
#define COQUI_MODEL_HAMILTONIAN_UTILS_HPP

#include "configuration.hpp"

#include <h5/h5.hpp>
#include <nda/h5.hpp>

#include "IO/app_loggers.h"
#include "utilities/concepts.hpp"
#include "utilities/mpi_context.h"

#include "mean_field/model_hamiltonian/model_system.hpp"

namespace mf {
namespace model {

// creates a model_readonly object with (empty) matrices at gamma point kpoint
template<utils::Communicator comm_t>
auto make_dummy_model(std::shared_ptr<utils::mpi_context_t<comm_t>> mpi, int nb, double nelec,
                      std::string outdir = "./", std::string prefix = "__dummy_model__")
{

  int ns = 1;
  int nk = 1;
  int npol = 1;

  auto symm = mf::bz_symm::gamma_point_instance();
  auto h = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});
  auto s = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});
  auto d = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});
  auto f = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});

  mf::model::model_system m(std::move(mpi),outdir,prefix,symm,ns,npol,nelec,h,s,d,f);
  return model_readonly(std::move(m));

}

} // model
} // mf
#endif
