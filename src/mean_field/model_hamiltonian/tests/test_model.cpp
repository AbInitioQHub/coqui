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

#include "configuration.hpp"
#include "nda/nda.hpp"

#include "utilities/test_common.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "mean_field/default_MF.hpp"
#include "mean_field/distributed_orbital_readers.hpp"
#include "mean_field/qe/qe_readonly.hpp"
#include "mean_field/model_hamiltonian/model_readonly.hpp"
#include "mean_field/symmetry/bz_symmetry.hpp"

namespace model_tests {

  using utils::ARRAY_EQUAL;
  template<int N>
  using shape_t = std::array<long,N>;

  TEST_CASE("model_system", "[mean_field_model]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    std::string prefix = "dummy.model";
    int ns = 1;
    int nk = 1;
    int nb = 4;
    double nel = 2.0;
    if(mpi->comm.root()) {
      h5::file file("dummy_symm.h5", 'w');
      h5::group grp(file);
      h5::group sgrp = grp.create_group("System");
      mf::bz_symm::gamma_point_h5(sgrp);
      mf::bz_symm symm("dummy_symm.h5");

      auto h = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});
      auto s = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});
      auto d = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});
      auto f = nda::array<ComplexType,4>::zeros({ns,nk,nb,nb});
    
      mf::model::model_system m(mpi,"./",prefix,symm,ns,1,nel,h,s,d,f); 
      m.save(prefix+".h5");
      remove("dummy_symm.h5");
    }
    mpi->comm.barrier();

    mf::model::model_system m(mpi,"./",prefix,-1); 
    
    REQUIRE(m.nspin == ns);
    REQUIRE(m.nbnd == nb);
    REQUIRE(m.nelec == nel);
    mpi->comm.barrier();

    if(mpi->comm.root())
      remove("dummy.model.h5");
    mpi->comm.barrier();
  }

/*
  TEST_CASE("model_h5", "[mean_field_model]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    auto [outdir,prefix] = utils::utest_filename(mf::model_source);
    mf::model::model_readonly mf(mpi_context, outdir, prefix);

    // unit cell info
    REQUIRE(mf.get_sys().nelec > 0);

    // BZ info
    auto nkpts = mf.nkpts();
    REQUIRE(nkpts > 0);
    REQUIRE( mf.get_sys().bz().kp_grid.shape() == shape_t<1>{3} );
    REQUIRE( mf.kpts().shape() == shape_t<2>{nkpts, 3} );
    REQUIRE( mf.get_sys().bz().Qpts.shape() == shape_t<2>{nkpts, 3} );
    REQUIRE( mf.get_sys().bz().qk_to_k2.shape() == shape_t<2>{nkpts, nkpts} );
    REQUIRE(mf.get_sys().bz().qminus.shape()[0] == nkpts);

    // basis info
    auto nbnd = mf.get_sys().nbnd;
    REQUIRE(nbnd > 0);

  }
*/
} // model_tests
