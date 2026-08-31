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
#include "mean_field/bdft/bdft_readonly.hpp"

namespace bdft_tests {

  using utils::ARRAY_EQUAL;
  template<int N>
  using shape_t = std::array<long,N>;

  TEST_CASE("bdft_from_qe", "[mean_field_bdft]") {

    auto& mpi_context = utils::make_unit_test_mpi_context();
    auto qe_mf = mf::default_MF(mpi_context, mf::qe_source);

    using larray = nda::array<ComplexType,5>;
    auto psi = mf::read_distributed_orbital_set_ibz<larray>(qe_mf,mpi_context->comm,'w');
    mpi_context->comm.barrier();
    app_log(2, "Done with read_distributed_orbital_set_ibz");

// MAM: finish
//    mf::bdft::bdft_readonly mf(mpi_context, qe_mf, "bdft_new.h5", psi);

//    if(mpi_context->comm.root()) remove("bdft_new.h5");

  }

  TEST_CASE("bdft_h5", "[mean_field_bdft]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    auto [outdir,prefix] = utils::utest_filename(mf::bdft_source);
    mf::bdft::bdft_readonly mf(mpi_context, outdir, prefix);

    // unit cell info
    REQUIRE(mf.get_sys().nelec > 0);
    auto a = mf.lattice();
    auto b = mf.recv();
    REQUIRE( a.shape() == shape_t<2>{3,3});
    REQUIRE( b.shape() == shape_t<2>{3,3});

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
    auto fft_mesh = mf.fft_grid_dim();
    REQUIRE(nbnd > 0);
    REQUIRE(mf.fft_grid_size() == fft_mesh(0)*fft_mesh(1)*fft_mesh(2) );

  }
} // bdft_tests
