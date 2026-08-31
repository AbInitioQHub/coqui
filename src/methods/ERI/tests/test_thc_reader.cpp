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

#include <map>

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"
#include "utilities/mpi_context.h"

#include "utilities/test_common.hpp"
#include "utilities/symmetry.hpp"
#include "numerics/sparse/sparse.hpp"
#include "methods/tests/test_common.hpp"

#include "mean_field/MF.hpp"
#include "mean_field/mf_utils.hpp"
#include "mean_field/default_MF.hpp"
#include "methods/ERI/eri_utils.hpp"
#include "methods/ERI/thc_reader_t.hpp"

namespace bdft_tests
{
  using namespace methods;
  using methods::make_thc_reader_ptree;

  TEST_CASE("thc_incore", "[methods]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, mf::pyscf_source));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*15, "", "incore", "", "bdft", 1e-10, mf->ecutrho(),
                   1, 1024));

    int is = 0;
    int ik = 0;
    int iq = 0;
    REQUIRE(thc.X(is, 0, ik).shape() == shape_t<2>{thc.Np(), thc.nbnd()});
    REQUIRE(thc.Z(iq).shape() == shape_t<2>{thc.Np(), thc.Np()});
  }

#if defined(ENABLE_DEVICE)
  TEST_CASE("thc_incore_device", "[methods]") {
    auto& mpi = utils::make_unit_test_mpi_context();
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, mf::qe_source));

    std::map<std::string,std::string> opt = {{"nIpts",std::to_string(mf->nbnd()*15)},
              {"ecut",std::to_string(0.3*mf->ecutrho())},
              {"compute","gpu"}, 
              {"thresh",std::to_string(1e-6)}};

    thc_reader_t thc(mf, io::make_ptree(opt));

    int is = 0;
    int ik = 0;
    int iq = 0;
    REQUIRE(thc.X(is, 0, ik).shape() == shape_t<2>{thc.Np(), thc.nbnd()});
    REQUIRE(thc.Z(iq).shape() == shape_t<2>{thc.Np(), thc.Np()});
  }
#endif

  TEST_CASE("make_thc", "[methods]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, mf::qe_source));
    ptree pt; 
    pt.put("nIpts","16");     
    pt.put("metric","ov");     
    pt.put("alg","cholesky");     
    pt.put("storage","incore");     
    pt.put("save","./thc_eri.h5");     
    pt.put("format","bdft");     
    auto eri = make_thc(mf, pt);

    int is = 0;
    int ik = 0;
    int iq = 0;
    REQUIRE(eri.X(is, 0, ik).shape() == shape_t<2>{eri.Np(), eri.nbnd()});
    REQUIRE(eri.Z(iq).shape() == shape_t<2>{eri.Np(), eri.Np()});
    mpi->comm.barrier();
    if(mpi->comm.root()) remove("./thc_eri.h5");
  }  

  TEST_CASE("thc_outcore", "[methods]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, mf::pyscf_source));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*15, "", "outcore", "./thc_eri.h5", "bdft", 1e-10,
                     mf->ecutrho(), 1, 1024));

    int is = 0;
    int ik = 0;
    int iq = 0;
    REQUIRE(thc.X(is, 0, ik).shape() == shape_t<2>{thc.Np(), thc.nbnd()});
    REQUIRE(thc.Z(iq).shape() == shape_t<2>{thc.Np(), thc.Np()});
    mpi->comm.barrier();
    if(mpi->comm.root()) remove("./thc_eri.h5");
  }

  TEST_CASE("thc_ls", "[methods]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "pyscf_h2o_mol"));
    auto [outd, pref] = utils::utest_filename("pyscf_h2o_mol");
    std::string gdf_dir = outd + "/gdf_eri/";
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*15, gdf_dir, "incore", "",
                                               "bdft", 1e-10, mf->ecutrho(),
                                               1, 1024, 1, 0.2, 0.75, false));

    int is = 0;
    int ik = 0;
    int iq = 0;
    REQUIRE(thc.X(is, 0, ik).shape() == shape_t<2>{thc.Np(), thc.nbnd()});
    REQUIRE(thc.Z(iq).shape() == shape_t<2>{thc.Np(), thc.Np()});
    mpi->comm.barrier();
    if(mpi->comm.root()) remove("./thc_eri.h5");
    if(mpi->comm.root()) {
      remove("chol_info.h5");
      remove("Vq0.h5");
    }
  }
} // bdft_tests

