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

#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/h5.hpp"
#include "utilities/test_common.hpp"

#include "mean_field/default_MF.hpp"
#include "hamiltonian/one_body_hamiltonian.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "orbitals/orbital_generator.h"
#include "utilities/mpi_context.h"

namespace bdft_tests
{

using namespace math::nda;
template <int Rank> using shape_t = std::array<long, Rank>;

/*
TEST_CASE("add_pgto", "[orbit]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  auto [outdir,prefix] = utils::utest_filename(mf::qe_source);
  auto qe_mf = mf::default_MF(mpi,mf::qe_source);

  // basic (no ortho or n0)
  {
    auto bdft_mf = orbitals::add_pgto(qe_mf,"dummy.h5",
                                    outdir+"basis.txt","nwchem",-1,false,false,0.0,false);
    auto ov = hamilt::ovlp(bdft_mf, mpi->comm, nda::range(bdft_mf.nkpts_ibz()),nda::range(bdft_mf.nbnd()));
  }
  mpi->comm.barrier();
  if(mpi->comm.root()) remove("dummy.h5");
  mpi->comm.barrier();

  // orthogonal with n0
  {
    auto bdft_mf = orbitals::add_pgto(qe_mf,"dummy.h5",
                                    outdir+"basis.txt","nwchem",-1,false,true,0.0,false);
    auto ov = hamilt::ovlp(bdft_mf, mpi->comm, nda::range(bdft_mf.nkpts_ibz()), nda::range(bdft_mf.nbnd()));
    auto ov_loc = ov.local();
    double e=0.0;
    for( auto [is,s] : itertools::enumerate(ov.local_range(0)) )   
      for( auto [ik,k] : itertools::enumerate(ov.local_range(1)) )   
        for( auto [ia,a] : itertools::enumerate(ov.local_range(2)) )   
          for( auto [ib,b] : itertools::enumerate(ov.local_range(3)) )  { 
            e += std::abs(ov_loc(is,ik,ia,ib)-(a==b?1.0:0.0));
          }
    auto e_sum = mpi->comm.all_reduce_value(e);
    utils::VALUE_EQUAL(e_sum,0.0); 
  }
  mpi->comm.barrier();
  if(mpi->comm.root()) remove("dummy.h5");
}
*/

TEST_CASE("eig_select", "[orbit]")
{ 
  auto& mpi = utils::make_unit_test_mpi_context();
  auto qe_mf = mf::default_MF(mpi,mf::qe_source);

  int n0 = int(qe_mf.nelec()/2.0);
  if(const char* env_p = std::getenv("N0")) n0 = std::atoi(env_p);
  int nblk = (qe_mf.nbnd()-n0)/3;
  if(const char* env_p = std::getenv("NBLK")) nblk = std::atoi(env_p);
  
  auto bdft_mf = orbitals::eigenstate_selection(qe_mf,"dummy2.h5","linear",n0,nblk);
  auto ov = hamilt::ovlp(bdft_mf, mpi->comm, nda::range(bdft_mf.nkpts_ibz()), nda::range(bdft_mf.nbnd()));

  if(mpi->comm.root()) remove("dummy2.h5");
}

}

