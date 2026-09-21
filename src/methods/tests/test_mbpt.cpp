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

#include "catch2/catch.hpp"

#include <cstdio>

#include "utilities/test_common.hpp"
#include "mean_field/default_MF.hpp"
#include "mean_field/mf_utils.hpp"
#include "methods/MBPT_drivers.h"
#include "numerics/imag_axes_ft/iaft_utils.hpp"

namespace bdft_tests {

  TEST_CASE("iaft_wmax_from_mf", "[methods][mbpt][iaft]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    
    auto run_wmax_check = [&](std::string const& mf_name) {
      
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, mf_name));

      ptree pt;
      pt.put("beta", 1000.0);
#ifdef ENABLE_DLR
      pt.put("iaft.basis", "dlr");
#else
      pt.put("iaft.basis", "ir"); 
      pt.put("iaft.prec", "high");
#endif

      auto ft_ref = imag_axes_ft::IAFT(pt, true, mf::wmax_from_mf(*mf));
      auto wmax_ref = ft_ref.wmax();

      auto in_pt2 = pt;
      in_pt2.put("iaft.wmax", wmax_ref);
      auto ft = imag_axes_ft::IAFT(in_pt2, true, mf::wmax_from_mf(*mf) * 100.0);

      REQUIRE(ft.wmax() == Approx(ft_ref.wmax()));
      REQUIRE(ft.nt_f() == ft_ref.nt_f());
      REQUIRE(ft.nw_f() == ft_ref.nw_f());
    };

    SECTION("qe_solid") {
      run_wmax_check("qe_lih222_sym");
    }

    SECTION("pyscf_solid") {
      run_wmax_check("pyscf_si222");
    }

    SECTION("pyscf_molecule") {
      run_wmax_check("pyscf_h2o_mol");
    }

  }

  TEST_CASE("wmax_from_mf_formula", "[methods][mbpt][iaft]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));

    auto ev = mf->eigval();
    auto [it_min, it_max] = std::minmax_element(ev.data(), ev.data() + ev.size());
    double emin = *it_min, emax = *it_max, ef = mf->efermi();
    double span  = emax - emin;
    double halfw = std::max(emax - ef, ef - emin);

    // Sized for Sigma = G*W: the whole scale (span + halfw) times a fixed padding.
    // Asserts arithmetic, not accuracy.
    REQUIRE(mf::wmax_from_mf(*mf, 1.5) == Approx(1.5 * (span + halfw)));
    REQUIRE(mf::wmax_from_mf(*mf, 2.0) == Approx(2.0 * (span + halfw)));
  }

} // bdft_tests
