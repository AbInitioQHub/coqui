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


#ifndef MEANFIELD_QE_QE_INTERFACE_H
#define MEANFIELD_QE_QE_INTERFACE_H

#include <map>
#include "utilities/mpi_context.h"
#include "nda/h5.hpp"
#include "qe_system.hpp"
#include "IO/app_loggers.h"
#include "IO/ptree/InputParser.hpp"
#include "IO/AppAbort.hpp"
#include "utilities/check.hpp"
#include "utilities/concepts.hpp"

namespace mf 
{
namespace qe 
{

/**
 * Read QE inputs from a xml file
 * @param comm - MPI communicator
 * @param outdir - QE output directory
 * @param prefix - QE prefix
 * @param nbnd_overwrite - Customized number of KS orbitals.
 *                         Default is -1, i.e. using all the available orbitals.
 * @param no_q_sym - Disable q-point symmetries
 * @return Coqui handler of QE inputs
 */
qe_system read_xml(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
                   std::string outdir, std::string prefix,
                   int nbnd_overwrite = -1,
                   bool no_q_sym = false); 

/**
 * Read QE inputs from a H5 file
 * @param comm - MPI communicator
 * @param outdir - QE output directory
 * @param prefix - QE prefix
 * @param nbnd_overwrite - Customized number of KS orbitals.
 *                         Default is -1, i.e. using all the available orbitals.
 * @return Coqui handler of QE inputs
 */
qe_system read_h5(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
                  std::string outdir, std::string prefix,
                  int nbnd_overwrite = -1);

} // qe
} // mf

#endif
