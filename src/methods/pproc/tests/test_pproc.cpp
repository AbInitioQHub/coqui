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
#include "nda/h5.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/iter_scf/iter_scf_utils.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/scf_driver.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/qp/linearized_qp.hpp"
#include "methods/pproc/pproc_t.h"

namespace bdft_tests {

  TEST_CASE("pproc_lqp_on_ibz_kmesh", "[methods][mbpt][pproc][lqp]") {
    using utils::VALUE_EQUAL;
    using utils::ARRAY_EQUAL;
    using namespace methods;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(100.0, 7.0, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    std::string output = "pproc_lqp_test";

    // 1. one Dyson G0W0 iteration -> scf/iter1/{F_skij, Sigma_tskij, mu}
    {
      simple_dyson dyson(mf.get(), &ft);
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*8, "", "incore", "", output,
                                                 1e-8, mf->ecutrho(), 1, 512));
      auto eri = mb_eri_t(thc, thc);
      iter_scf::iter_scf_t damp(iter_scf::damp_t(1.0));
      MBState mb_state(mpi_context, ft, output);
      scf_loop(mb_state, dyson, eri, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &damp,
               1, false, 1e-8, true);
      mpi_context->comm.barrier();
    }

    // 2. LQSGW QP energies on the IBZ mesh
    qp_params_t qp; qp.qp_approx = "lqp"; qp.lqp.n_fit = 4;
    pproc_t pp(*mpi_context, output, "./");
    pp.compute_qp_on_ibz_kmesh(*mf, qp, "scf", 1);
    mpi_context->comm.barrier();

    // 3. compare with the kernel called directly on the same arrays
    nda::array<ComplexType, 4> F, H0; nda::array<ComplexType, 5> Sigma;
    nda::array<double, 3> E_file, Z_file; nda::array<ComplexType, 4> Heff_file;
    double mu; std::string scheme;
    {
      h5::file file(output + ".mbpt.h5", 'r');
      auto root = h5::group(file);
      nda::h5_read(root.open_group("system"), "H0_skij", H0);
      auto it = root.open_group("scf/iter1");
      nda::h5_read(it, "F_skij", F); nda::h5_read(it, "Sigma_tskij", Sigma); h5::h5_read(it, "mu", mu);
      auto qg = it.open_group("qp_approx");
      nda::h5_read(qg, "E_ska", E_file); nda::h5_read(qg, "Z_ska", Z_file);
      nda::h5_read(qg, "Heff_skij", Heff_file); h5::h5_read(qg, "scheme", scheme);
    }
    CHECK(scheme == "lqp");
    F += H0;
    lqp::fit_params_t p; p.n_fit = 4;
    auto res = lqp::linearized_qp_solve(mpi_context->comm, F, Sigma, mu, ft, p);
    long ns = F.shape(0), nk = F.shape(1), nb = F.shape(2);
    for (long s = 0; s < ns; ++s)
      for (long k = 0; k < nk; ++k) {
        for (long a = 0; a < nb; ++a) {
          VALUE_EQUAL(E_file(s, k, a), res.E_ska(s, k, a), 1e-10, 1e-10);   // both absolute
          VALUE_EQUAL(Z_file(s, k, a), res.Zqp_ska(s, k, a), 1e-10, 1e-10);
          CHECK(Z_file(s, k, a) > 0.0); CHECK(Z_file(s, k, a) <= 1.0 + 1e-10);
        }
        ARRAY_EQUAL(Heff_file(s, k, nda::ellipsis{}), res.Hqp_skab(s, k, nda::ellipsis{}), 1e-10);
      }
    mpi_context->comm.barrier();
    if (mpi_context->comm.root()) std::remove((output + ".mbpt.h5").c_str());
    mpi_context->comm.barrier();
  }

} // bdft_tests
