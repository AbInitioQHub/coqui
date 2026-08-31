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


#ifndef AIMBES_METHODS_PPROC_WAVEFUNCTION_UTILS_HPP
#define AIMBES_METHODS_PPROC_WAVEFUNCTION_UTILS_HPP

#include "configuration.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "h5/h5.hpp"
#include "mean_field/MF.hpp"

namespace methods
{

  void add_wavefunction(h5::group & grp, mf::MF &mf, ptree const& pt);

}

#endif
