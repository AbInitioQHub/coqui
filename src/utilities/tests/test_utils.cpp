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


#undef NDEBUG

#include <complex>

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "utilities/test_common.hpp"
#include "utilities/Timer.hpp"
#include "utilities/interpolation_utils.hpp"

namespace bdft_tests
{

using utils::VALUE_EQUAL;

TEST_CASE("interpolation", "[utilities]")
{
  // not really checking anything right now, just making sure they run without errors
  {
    std::vector<nda::array<double,2>> kp(4);
    kp[0] = nda::array<double,2>{ {0.0,0.0,0.0}, {0.5,0.0,0.0} };
    kp[1] = nda::array<double,2>{ {0.5,0.0,0.0}, {0.5,0.5,0.0} };
    kp[2] = nda::array<double,2>{ {0.5,0.5,0.0}, {0.5,0.5,0.5} };
    kp[3] = nda::array<double,2>{ {0.5,0.5,0.5}, {0.0,0.0,0.0} };
    nda::array<double,2> recv = {{0.0,6.0,8.0},{4.0,0.0,8.0},{4.0,6.0,0.0}};
    std::vector<std::string> id = {"K","G","G","M","M","R","R","G"};
    auto [kpath,idx] = utils::generate_kpath(recv,kp,id,10);
  }

  {
    nda::array<double,2> recv = {{0.0,6.0,8.0},{4.0,0.0,8.0},{4.0,6.0,0.0}};
    nda::array<long,1> mesh = {4,4,4};
    auto ws = utils::WS_rgrid(recv,mesh);
  }
}


}

