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
#include <complex>
#include "stdio.h"

#include "configuration.hpp"
#include "utilities/mpi_context.h"

#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/test_common.hpp"

#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/h5.hpp"

#include "mean_field/MF.hpp"
#include "mean_field/default_MF.hpp"
#include "mean_field/qe/qe_readonly.hpp"
#include "grids/g_grids.hpp"
#include "methods/ERI/cholesky.h"
#include "methods/ERI/eri_utils.hpp"

// erase
#include "nda/blas.hpp"
#include "grids/g_grids.hpp"
#include "hamiltonian/potentials.hpp"

namespace bdft_tests
{

using utils::VALUE_EQUAL;
using utils::ARRAY_EQUAL;
namespace mpi3 = boost::mpi3;
using methods::make_chol_ptree;

TEST_CASE("cholesky_seq", "[methods]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  auto mf = mf::default_MF(mpi, mf::qe_source);

  // sequential variant
  methods::cholesky chol(std::addressof(mf),*mpi,make_chol_ptree(1e-4,mf.ecutrho(),0));
  auto L = chol.evaluate<HOST_MEMORY>(0);
 
  chol.print_timers();

#if 0 //defined(ENABLE_DEVICE)
  chol.reset_timers();
  auto Ld = chol.evaluate<DEVICE_MEMORY>(0);
  chol.print_timers();
  chol.reset_timers();
  auto Lu = chol.evaluate<UNIFIED_MEMORY>(0);
  chol.print_timers();
#endif
}

TEST_CASE("cholesky_seq_ecut", "[methods]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  auto mf = mf::default_MF(mpi, mf::qe_source);

  // sequential variant
  methods::cholesky chol(std::addressof(mf),*mpi,make_chol_ptree(1e-4,0.4*mf.ecutrho(),0));
  auto L = chol.evaluate<HOST_MEMORY>(0);

  chol.print_timers();

#if 0//defined(ENABLE_DEVICE)
  chol.reset_timers();
  auto Ld = chol.evaluate<DEVICE_MEMORY>(0);
  chol.print_timers();
  chol.reset_timers();
  auto Lu = chol.evaluate<UNIFIED_MEMORY>(0);
  chol.print_timers();
#endif
}

TEST_CASE("cholesky_blocked", "[methods]")
{ 
  auto& mpi = utils::make_unit_test_mpi_context();
  
  auto mf = mf::default_MF(mpi, mf::qe_source);
  
  // standard partitioned alg 
  methods::cholesky chol(std::addressof(mf),*mpi,make_chol_ptree(1e-4,mf.ecutrho()));
  auto L = chol.evaluate<HOST_MEMORY>(0);
  chol.print_timers();

#if 0//defined(ENABLE_DEVICE)
  chol.reset_timers();
  auto Ld = chol.evaluate<DEVICE_MEMORY>(0);
  chol.print_timers();
  chol.reset_timers();
  auto Lu = chol.evaluate<UNIFIED_MEMORY>(0);
  chol.print_timers();
#endif
}

TEST_CASE("cholesky_blocked_ecut", "[methods]")
{ 
  auto& mpi = utils::make_unit_test_mpi_context();
  
  auto mf = mf::default_MF(mpi, mf::qe_source);
  
  // standard partitioned alg 
  methods::cholesky chol(std::addressof(mf),*mpi,make_chol_ptree(1e-4,0.4*mf.ecutrho()));
  auto L = chol.evaluate<HOST_MEMORY>(0);
  chol.print_timers();

#if 0//defined(ENABLE_DEVICE)
  chol.reset_timers();
  auto Ld = chol.evaluate<DEVICE_MEMORY>(0);
  chol.print_timers();
  chol.reset_timers();
  auto Lu = chol.evaluate<UNIFIED_MEMORY>(0);
  chol.print_timers();
#endif
}

TEST_CASE("cholesky_diagkk", "[methods]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  auto mf = mf::default_MF(mpi, mf::qe_source);

  // standard partitioned alg 
  methods::cholesky chol(std::addressof(mf),*mpi,make_chol_ptree(1e-4,mf.ecutrho()));
  int nqpts = mf.Qpts().shape()[0];
  for(int q=0; q<nqpts; q++)
  { 
    auto L = chol.evaluate<HOST_MEMORY>(q,nda::range(-1,-1),nda::range(-1,-1),true);
  }
//  chol.print_timers();
}

TEST_CASE("cholesky_range", "[methods]")
{ 
  auto& mpi = utils::make_unit_test_mpi_context();
  
  auto mf = mf::default_MF(mpi, mf::qe_source);
  
  {
    // standard partitioned alg 
    methods::cholesky chol(std::addressof(mf),*mpi,make_chol_ptree(1e-4,mf.ecutrho(),0));
    // MAM: will fail if nbnd < 5
    auto Lf = chol.evaluate<HOST_MEMORY>(0,nda::range(2,mf.nbnd()-2),nda::range(1,mf.nbnd()-1));
    //auto Lr = chol.evaluate<HOST_MEMORY>(0);
  }

  {
    // standard partitioned alg 
    methods::cholesky chol(std::addressof(mf),*mpi,make_chol_ptree(1e-4,mf.ecutrho()));
    // MAM: will fail if nbnd < 5
    auto Lf = chol.evaluate<HOST_MEMORY>(0,nda::range(2,mf.nbnd()-2),nda::range(1,mf.nbnd()-1));
    //auto Lr = chol.evaluate<HOST_MEMORY>(0);
  }
}

TEST_CASE("cholesky_EHF", "[methods]")
{ 
  // serially for now
  auto& mpi = utils::make_unit_test_mpi_context();
  
  auto mf = mf::default_MF(mpi, mf::qe_source);
  int nqpts = mf.Qpts().shape()[0];
  int nspins = mf.nspin();
  int nkpts = mf.nkpts();
  int nbnd = mf.nbnd();
  auto occ = mf.occ(); 
  
  utils::check( not mf.noncolin(), "Error: No noncollinear yet."); 
  utils::check( mf.nkpts() == mf.nkpts_ibz(), "Error: No symmetry on cholesky_EHF yet.");

  // sequential variant
  methods::cholesky chol(std::addressof(mf),*mpi,make_chol_ptree(1e-4,mf.ecutrho()));

  nda::array<ComplexType,2> eri;
  if(mpi->comm.root())
    eri = nda::array<ComplexType,2>::zeros({nspins*nkpts*nbnd*nbnd,nspins*nkpts*nbnd*nbnd});
  ComplexType EH(0.0), EX(0.0);
  nda::array<ComplexType,5> L;

  for( auto iq : nda::range(nqpts) ) {
  
    auto dL = chol.evaluate<HOST_MEMORY>(iq);
    if(mpi->comm.root()) L = nda::array<ComplexType,5>(dL.global_shape()[0],nspins,nkpts,nbnd,nbnd);
    math::nda::gather(0,dL,std::addressof(L));
    if(mpi->comm.root()) {
      auto L2D = nda::reshape(L, std::array<long,2>{L.shape()[0],nspins*nkpts*nbnd*nbnd});
      L2D = nda::conj(L2D);
      nda::blas::gemm(ComplexType(1.0),nda::dagger(L2D),L2D,ComplexType(0.0),eri);

      if(iq==0) {
        for( auto isa : nda::range(nspins) ) 
          for( auto ika : nda::range(nkpts) ) 
            for( auto ia : nda::range(nbnd) ) {  
              if( std::abs(occ(isa,ika,ia)) < 1e-6 ) continue; 
              ComplexType e_(0.0);
              long skaa = ((isa*nkpts + ika)*nbnd + ia )*nbnd + ia;
              for( auto isb : nda::range(nspins) ) 
                for( auto ikb : nda::range(nkpts) ) 
                  for( auto ib : nda::range(nbnd) ) { 
                    long skbb = ((isb*nkpts + ikb)*nbnd + ib )*nbnd + ib;
                    e_ += occ(isb,ikb,ib) * eri(skaa,skbb);
                  }
              EH += e_* occ(isa,ika,ia);
            }
      }  

      for( auto is : nda::range(nspins) )
        for( auto ik : nda::range(nkpts) )
          for( auto ia : nda::range(nbnd) ) 
            for( auto ib : nda::range(nbnd) ) {
              if( std::abs(occ(is,ik,ia))*std::abs(occ(is,ik,ib)) < 1e-6 ) continue;
              long skab = ((is*nkpts + ik)*nbnd + ia )*nbnd + ib;
              EX -= occ(is,ik,ib) * eri(skab,skab) * occ(is,ik,ia);
            }
    }
  }

  app_log(0," EH: {}", 0.5*EH/nkpts*(3.0-nspins)*(3.0-nspins));
  app_log(0," EX: {}", 0.5*EX/nkpts*(3.0-nspins));
}

TEST_CASE("cholesky_io", "[methods]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  auto mf = mf::default_MF(mpi, mf::qe_source);

  // standard partitioned alg 
  methods::cholesky chol(std::addressof(mf),*mpi,make_chol_ptree(1e-4,mf.ecutrho()));

  // evaluate L
  auto L = chol.evaluate<HOST_MEMORY>(0);

  using T = typename decltype(L)::value_type;
  nda::array<T,5> Lcopy = L.local();

  // write
  {
    if(mpi->comm.root()) {
      h5::file fh5("dummy.h5",'w');
      h5::group top{fh5};
      auto gh5 = top.create_group("hamiltonian").create_group("cholesky");
      chol.write_meta_data(gh5);
      chol.write(gh5,0,L);
    } else {
      h5::group L0{};
      chol.write(L0,0,L);
    }
    mpi->comm.barrier();
  }

  L.local() = 0.0; 

  // read
  {
    h5::file fh5("dummy.h5",'r');
    h5::group top{fh5};
    auto gh5 = top.open_group("hamiltonian").open_group("cholesky");
    math::nda::h5_read(gh5, "L0", L);
    REQUIRE( L.local() == Lcopy );
    mpi->comm.barrier();
  }
  
  mpi->comm.barrier();
  if(mpi->comm.root()) remove("dummy.h5");
}

TEST_CASE("cholesky_pyscf", "[methods]") {
  auto& mpi = utils::make_unit_test_mpi_context();

  auto [outdir,prefix] = utils::utest_filename(mf::pyscf_source);
  auto mf = mf::default_MF(mpi, mf::pyscf_source);
  methods::cholesky chol(std::addressof(mf), *mpi, make_chol_ptree(1e-12, mf.ecutrho()));

  size_t iq = 2;
  auto L_d = chol.evaluate<HOST_MEMORY>(iq);
//  auto L_loc = L.local();
  mpi->comm.barrier();
  // MAM: very lazy, but easy. These tests are small!
  nda::array<ComplexType, 5> L_loc(L_d.global_shape());
  math::nda::gather(0,L_d,std::addressof(L_loc));
  mpi->comm.broadcast_n(L_loc.data(),L_loc.size(),0);

  nda::array<ComplexType, 3> U_pyscf(mf.nkpts(), mf.nbnd()*mf.nbnd(), mf.nbnd()*mf.nbnd());
  nda::array<ComplexType, 3> U_chol(mf.nkpts(), mf.nbnd()*mf.nbnd(), mf.nbnd()*mf.nbnd());
  {
    // ERIs obtained using FFTDF from PySCF. Converged to 1e-14.
    std::string pyscf_eri = outdir + "/fftdf_eri.h5";
    std::string dataset = "eri_q" + std::to_string(iq);
    h5::file file(pyscf_eri,'r');
    h5::group grp(file);
    nda::h5_read(grp, dataset, U_pyscf);
    U_pyscf *= (1.0/mf.nkpts());
  }

  {
    for (size_t ik = 0; ik < mf.nkpts(); ++ik) {
      auto Lqk = nda::make_regular(L_loc(nda::range::all, 0, ik, nda::range::all, nda::range::all) );
      nda::array<ComplexType, 3> LL(Lqk.shape());

      auto L_2D = nda::reshape(Lqk, std::array<long,2>{Lqk.shape(0), Lqk.shape(1)*Lqk.shape(2)});
      auto LL_2D = nda::reshape(LL, std::array<long,2>{LL.shape(0), LL.shape(1)*LL.shape(2)});
      for (size_t P = 0; P < Lqk.shape(0); ++P) {
        auto LP = Lqk(P, nda::ellipsis{});
        auto LP_conj = nda::make_regular(nda::conj(LP));
        LL(P, nda::ellipsis{}) = nda::transpose(LP_conj);
      }
      nda::blas::gemm(1.0, nda::transpose(L_2D), LL_2D, 0.0, U_chol(ik, nda::ellipsis{}));
    }
  }
  mpi->comm.barrier();

  ARRAY_EQUAL(U_pyscf, U_chol, 1e-12);
  mpi->comm.barrier();
}

} // bdft_tests
