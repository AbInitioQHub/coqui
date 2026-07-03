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

#include <iostream>
#include <iomanip>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"
#include "mean_field/default_MF.hpp"

#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"
#include "numerics/iter_scf/iter_scf_utils.hpp"

namespace bdft_tests {

  using namespace methods;
  using utils::VALUE_EQUAL;
 
  TEST_CASE("thermodynamics_gw_qe", "[methods_scf][thermodynamics][gw][qe]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::ir_basis, "high");
    std::string output = "coqui";

    // Reference values generated from the current implementation with the following 
    // setup: GW, 1 iteration, wmax=1.2, ir_basis "high", THC alpha=20, eps=1e-10, 
    //        and no symmetry.
    // Regenerate (see the [REFVAL] print below) if the setup or the
    // thermodynamics implementation changes.
    const double omega_ref   = -4.67075760352667;  // grand potential       [a.u.]
    const double afe_ref     = -4.34301273810908;  // Helmholtz free energy [a.u.]
    const double entropy_ref =  4.59329984057444;  // entropy               [a.u.]
    const double nelec_ref   =  4.0;               
    const double tol = 1e-5;                       // THC accuracy at alpha=20

    auto solve_and_check = [&](std::shared_ptr<mf::MF> &mf, const std::string &label) {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);

      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 20, "", "incore", "", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      iter_scf::iter_scf_t iter_sol("damping");

      // Single GW iteration with the thermodynamics evaluation enabled (the
      // trailing eval_thermodynamics = true).
      scf_loop(mb_state, dyson, eri, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
               1, false, 1e-9, false, "scf", -1, true);

      REQUIRE(mb_state.thermodynamics.has_value());
      auto& th = mb_state.thermodynamics.value();

      // Print the observables (for reference-value generation).
      app_log(1, "[REFVAL][{}] grand_potential={} helmholtz_free_energy={} entropy={} n_electron={}",
              label, th.grand_potential, th.helmholtz_free_energy, th.entropy, th.n_electron);

      VALUE_EQUAL(th.grand_potential,       omega_ref,   tol);
      VALUE_EQUAL(th.helmholtz_free_energy, afe_ref,     tol);
      VALUE_EQUAL(th.entropy,               entropy_ref, tol);
      VALUE_EQUAL(th.n_electron,            nelec_ref,   tol);

      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) {
        remove((output+".mbpt.h5").c_str());
      }
      mpi_context->comm.barrier();
    };

    SECTION("nosym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
      solve_and_check(mf, "nosym");
    }
    SECTION("sym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222_sym"));
      solve_and_check(mf, "sym");
    }
  }

} // namespace bdft_tests
