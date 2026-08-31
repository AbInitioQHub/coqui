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

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"
#include "utilities/mpi_context.h"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"

#include "mean_field/MF.hpp"
#include "mean_field/mf_utils.hpp"
#include "mean_field/default_MF.hpp"
#include "methods/ERI/cholesky.h"
#include "methods/ERI/chol_reader_t.hpp"
#include "methods/ERI/eri_utils.hpp"

namespace bdft_tests
{
  using namespace methods;

  TEST_CASE("chol_reader", "[methods]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, mf::pyscf_source));

    chol_reader_t chol(mf, methods::make_chol_reader_ptree(1e-6, mf->ecutrho(), 32, "./", "chol_info.h5"));

    auto V = chol.V(0, 0, 0);
    REQUIRE(V.shape() == shape_t<3>{(long)chol.Np(), (long)chol.nbnd(), (long)chol.nbnd()});
    std::cout << "Reading type = " << chol.chol_read_type() << std::endl;
    std::cout << "Writing type = " << chol.chol_write_type() << std::endl;
    mpi->comm.barrier();

    chol.set_read_type() = methods::chol_reading_type_e::each_q;
    [[maybe_unused]] auto Vq = chol.V(0, 0, 1);
    REQUIRE(V.shape() == shape_t<3>{(long)chol.Np(), (long)chol.nbnd(), (long)chol.nbnd()});
    std::cout << "Reading type = " << chol.chol_read_type() << std::endl;
    mpi->comm.barrier();

    if(mpi->comm.root()) {
      remove("chol_info.h5");
      for (size_t iq = 0; iq < mf->nkpts(); ++iq) remove(("Vq"+std::to_string(iq)+".h5").c_str());
    }
  }

  TEST_CASE("chol_reader_single_write", "[methods]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, mf::qe_source));

    chol_reader_t chol(mf, methods::make_chol_reader_ptree(1e-6, mf->ecutrho(), 32, "./",
                                                           "chol_info.h5", each_q, single_file));

    auto V = chol.V(0, 0, 0);
    REQUIRE(V.shape() == shape_t<3>{(long)chol.Np(), (long)chol.nbnd(), (long)chol.nbnd()});
    std::cout << "Reading type = " << chol.chol_read_type() << std::endl;
    std::cout << "Writing type = " << chol.chol_write_type() << std::endl;

    chol.set_read_type() = methods::chol_reading_type_e::each_q;
    [[maybe_unused]] auto Vq = chol.V(0, 0, 1);
    REQUIRE(V.shape() == shape_t<3>{(long)chol.Np(), (long)chol.nbnd(), (long)chol.nbnd()});
    std::cout << "Reading type = " << chol.chol_read_type() << std::endl;

    if(mpi->comm.root())
      remove("chol_info.h5");
  }

  TEST_CASE("make_cholesky", "[methods]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, mf::qe_source));
    ptree pt;
    auto chol = make_cholesky(mf, pt);

    auto V = chol.V(0, 0, 0);
    REQUIRE(V.shape() == shape_t<3>{(long)chol.Np(), (long)chol.nbnd(), (long)chol.nbnd()});
    std::cout << "Reading type = " << chol.chol_read_type() << std::endl;
    mpi->comm.barrier();

    chol.set_read_type() = methods::chol_reading_type_e::each_q;
    [[maybe_unused]] auto Vq = chol.V(0, 0, 1);
    REQUIRE(V.shape() == shape_t<3>{(long)chol.Np(), (long)chol.nbnd(), (long)chol.nbnd()});
    std::cout << "Reading type = " << chol.chol_read_type() << std::endl;
    
    mpi->comm.barrier();
    if(mpi->comm.root()) {
      remove("chol_info.h5");
      for (size_t iq = 0; iq < mf->nkpts(); ++iq) remove(("Vq"+std::to_string(iq)+".h5").c_str());
    }
  }

} // bdft_tests
