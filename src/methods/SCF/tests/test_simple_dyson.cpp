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
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"

#include "utilities/test_common.hpp"

#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "mean_field/mf_utils.hpp"
#include "methods/SCF/scf_common.hpp"
#include "methods/SCF/simple_dyson.h"
#include "hamiltonian/pseudo/pseudopot.h"

namespace bdft_tests {

  using utils::mpi_context_t;
  using utils::VALUE_EQUAL;
  using utils::ARRAY_EQUAL;
  namespace mpi3 = boost::mpi3;
  using namespace methods;

  TEST_CASE("dyson_init", "[methods_scf]") {
    auto& context = utils::make_unit_test_mpi_context();
    std::string source_path = PROJECT_SOURCE_DIR;
    std::string filepath = source_path + "/tests/unit_test_files/pyscf/si_kp222_krhf/";

    double beta = 1000;
    double wmax = 12.0;
    auto mf = mf::make_MF(context, mf::pyscf_source, filepath, "pyscf");
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::ir_basis, "high");
    simple_dyson dyson(std::addressof(mf), std::addressof(ft), 1e-9, "bisection");
  }

  TEST_CASE("dyson", "[methods_scf]") {
    auto& context = utils::make_unit_test_mpi_context();
    std::string source_path = PROJECT_SOURCE_DIR;
    std::string filepath = source_path + "/tests/unit_test_files/pyscf/si_kp222_krhf/";
    double beta = 1000;
    double wmax = 1.2;
    auto mf = mf::make_MF(context, mf::pyscf_source, filepath, "pyscf");
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::ir_basis, "high");
    hamilt::pseudopot psp(mf);
    sArray_t<Array_view_4D_t> F(math::shm::make_shared_array<Array_view_4D_t>(
        *context, {mf.nspin(), mf.nkpts(), mf.nbnd(), mf.nbnd()}));
    sArray_t<Array_view_4D_t> Dm(math::shm::make_shared_array<Array_view_4D_t>(
        *context, {mf.nspin(), mf.nkpts(), mf.nbnd(), mf.nbnd()}));
    sArray_t<Array_view_5D_t> G(math::shm::make_shared_array<Array_view_5D_t>(
        *context, {ft.nt_f(), mf.nspin(), mf.nkpts(), mf.nbnd(), mf.nbnd()}));
    sArray_t<Array_view_5D_t> Sigma(math::shm::make_shared_array<Array_view_5D_t>(
        *context, {ft.nt_f(), mf.nspin(), mf.nkpts(), mf.nbnd(), mf.nbnd()}));
    hamilt::set_fock(mf, std::addressof(psp), F, true);

    simple_dyson dyson(std::addressof(mf), std::addressof(ft), 1e-9, "bisection");
    context->comm.barrier();

    double mu = update_mu(0.0, dyson, mf, ft, F, Sigma);
    CHECK(mu == Approx(0.175));
    context->comm.barrier();

    update_G(dyson, mf, ft, Dm, G, F, Sigma, mu, true);// keep mu constant
    context->comm.barrier();

    /**
     * References are obtained from PySCF in the zero-temperature formalism.
     * The accuracy is roughly 1e-8 when comparing with Dyson equation at beta = 1000
     **/
    {
      std::string source_path_ = PROJECT_SOURCE_DIR;
      std::string filename = source_path_ + "/tests/unit_test_files/pyscf/si_kp222_krhf/hf_Gw_Gt_beta1000_wmax1.2_high.h5";
      h5::file file(filename, 'r');
      h5::group grp(file);

      nda::array<std::complex<double>, 5> G_tskij_ref;
      nda::array<std::complex<double>, 4> Dm_skij_ref;
      nda::h5_read(grp, "Gt", G_tskij_ref);
      nda::h5_read(grp, "Dm", Dm_skij_ref);

      ARRAY_EQUAL(G.local(), G_tskij_ref, 1e-8);
      ARRAY_EQUAL(Dm.local(), Dm_skij_ref, 1e-8);
    }
    auto k_weight = mf.k_weight();
    auto [e_1e, e_hf] = eval_hf_energy(Dm, F, dyson.sH0_skij(), k_weight, false);
    VALUE_EQUAL(e_1e+e_hf, 0.8730537612681228, 1e-8);

    double e_corr = eval_corr_energy(context->comm, ft, G, Sigma, k_weight);
    VALUE_EQUAL(e_corr, 0.0, 1e-12);
    context->comm.barrier();
  }


} // bdft_tests
