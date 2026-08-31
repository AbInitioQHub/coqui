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
#include "configuration.hpp"
#include "nda/nda.hpp"

#include "utilities/test_common.hpp"
#include "numerics/imag_axes_ft/ir/ir_driver.hpp"

namespace bdft_tests {

  using utils::ARRAY_EQUAL;
  template<int N>
  using shape_t = std::array<long,N>;

  TEST_CASE("ir_read", "[iaft][ir]") {
    double beta = 1000;
    double lambda = 1.2e4;
    imag_axes_ft::ir::IR myir(beta, lambda/beta, "high");

    REQUIRE(myir.lambda == 1e5);
    REQUIRE(myir.nt_f == 137);
    REQUIRE(myir.nw_f == 138);
    REQUIRE(myir.nt_b == 137);
    REQUIRE(myir.nw_b == 137);

    REQUIRE(myir.Ttw_ff.shape() == shape_t<2>{myir.nt_f, myir.nw_f});
    REQUIRE(myir.Twt_ff.shape() == shape_t<2>{myir.nw_f, myir.nt_f});
    REQUIRE(myir.Ttt_bf.shape() == shape_t<2>{myir.nt_b, myir.nt_f});
    REQUIRE(myir.Ttw_bb.shape() == shape_t<2>{myir.nt_b, myir.nw_b});
    REQUIRE(myir.Twt_bb.shape() == shape_t<2>{myir.nw_b, myir.nt_b});
    REQUIRE(myir.Ttt_fb.shape() == shape_t<2>{myir.nt_f, myir.nt_b});
    REQUIRE(myir.T_beta_t_ff.shape() == shape_t<1>{myir.nt_f});

    auto eye1 = myir.Ttw_ff * myir.Twt_ff;
    auto eye2 = myir.Ttt_bf * myir.Ttt_fb;
    ARRAY_EQUAL(eye1, nda::eye<ComplexType>(myir.nt_f), 1e-10);
    ARRAY_EQUAL(eye2, nda::eye<RealType>(myir.nt_b), 1e-10);
  }

} // bdft_tests
