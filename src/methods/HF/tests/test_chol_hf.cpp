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

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"

#include "nda/h5.hpp"

#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "mean_field/default_MF.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/mb_solver_t.h"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"
#include "methods/HF/hf_t.h"
#include "methods/HF/cholesky_hf.icc"


namespace bdft_tests {

  using namespace methods;

  TEST_CASE("chol_hf_qe", "[methods][chol][hf][qe]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::ir_basis);
    solvers::hf_t hf;

    /**
     * References are obtained from chol-hf with Cholesky tol = 1e-10
     **/
    chol_reader_t chol_reader(mf, methods::make_chol_reader_ptree(1e-10, mf->ecutrho(), 32, "./"));
    chol_reader.set_read_type() = chol_reading_type_e::each_q;
    auto eri = mb_eri_t(chol_reader, chol_reader);
    simple_dyson dyson(mf.get(), &ft);
    iter_scf::iter_scf_t iter_sol("damping");
    MBState mb_state(mpi_context, ft, "bdft");
    auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                   solvers::mb_solver_t(&hf), &iter_sol,
                                   1, false, 1e-9, false);
    VALUE_EQUAL(e_hf, -4.2818278244126935, 1e-8);
    mpi_context->comm.barrier();

    if (mpi_context->comm.root()) {
      remove("./chol_info.h5");
      for (size_t ik = 0; ik < mf->nqpts(); ++ik) {
        std::string fname = "Vq"+std::to_string(ik)+".h5";
        remove(fname.c_str());
      }
      remove("./bdft.mbpt.h5");
    }
  }

  TEST_CASE("chol_hf_pyscf", "[methods][chol][hf][pyscf]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_si222"));
    using Array_4D_t = nda::array<ComplexType, 4>;
    std::array<long, 4> shape = {mf->nspin(), mf->nkpts(), mf->nbnd(), mf->nbnd()};
    auto sDm_skij = math::shm::make_shared_array<Array_4D_t>(*mpi_context,shape);
    auto sF_skij = math::shm::make_shared_array<Array_4D_t>(*mpi_context,shape);
    auto sS_skij = math::shm::make_shared_array<Array_4D_t>(*mpi_context,shape);
    auto sH0_skij = math::shm::make_shared_array<Array_4D_t>(*mpi_context,shape);

    auto sJ_skij_ref = math::shm::make_shared_array<Array_4D_t>(*mpi_context,shape);
    auto sK_skij_ref = math::shm::make_shared_array<Array_4D_t>(*mpi_context,shape);
    auto sF_skij_ref = math::shm::make_shared_array<Array_4D_t>(*mpi_context,shape);
    /**
     * References are obtained from PySCF in the zero-temperature formalism.
     * The accuracy is roughly 1e-8, comparing with Dyson equation at beta = 1000
     **/
    if (mpi_context->node_comm.root()) {
      auto [outdir, prefix] = utils::utest_filename("pyscf_si222");
      h5::file file(outdir + "/" + prefix + ".h5", 'r');
      h5::group grp(file);
      h5::group scf_grp = grp.open_group("SCF");
      auto sDloc = sDm_skij.local();
      auto sJloc = sJ_skij_ref.local();
      auto sKloc = sK_skij_ref.local();
      auto sFloc = sF_skij_ref.local();
      auto sSloc = sS_skij.local();
      auto sH0loc = sH0_skij.local();

      nda::h5_read(scf_grp, "dm", sDloc);
      nda::h5_read(scf_grp, "J", sJloc);
      nda::h5_read(scf_grp, "K", sKloc);
      nda::h5_read(scf_grp, "Fock", sFloc);
      nda::h5_read(scf_grp, "ovlp", sSloc);
      nda::h5_read(scf_grp, "H0", sH0loc);
      sDm_skij.local() *= 0.5;
    }
    mpi_context->comm.barrier();

    solvers::hf_t hf;
    chol_reader_t chol_reader(mf, methods::make_chol_reader_ptree(1e-12, mf->ecutrho(), 32, "./"));
    chol_reader.set_read_type() = chol_reading_type_e::each_q;

    {
      hf.add_J(sF_skij, sDm_skij.local(), chol_reader);
      ARRAY_EQUAL(sF_skij.local(), sJ_skij_ref.local(), 1e-8);
      mpi_context->comm.barrier();
      sF_skij.set_zero();

      hf.add_K(sF_skij, sDm_skij.local(), chol_reader, sS_skij.local());
      ARRAY_EQUAL(sF_skij.local(), sK_skij_ref.local(), 1e-8);
      mpi_context->comm.barrier();
      sF_skij.set_zero();
    }

    hf.evaluate(sF_skij, sDm_skij.local(), chol_reader, sS_skij.local());
    if (mpi_context->node_comm.root())
      sF_skij.local() += sH0_skij.local();
    mpi_context->comm.barrier();
    ARRAY_EQUAL(sF_skij.local(), sF_skij_ref.local(), 1e-8);
    if (mpi_context->comm.root()) {
      remove("chol_info.h5");
      for (size_t ik = 0; ik < mf->nqpts(); ++ik) {
        std::string fname = "Vq"+std::to_string(ik)+".h5";
        remove(fname.c_str());
      }
    }
    mpi_context->comm.barrier();
  }
} // bdft_tests
