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
#include <complex>

#include "utilities/mpi_context.h"

#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/test_common.hpp"
#include "hamiltonian/potentials.hpp"

#include "mean_field/qe/qe_interface.h"
#include "mean_field/qe/qe_readonly.hpp"

#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/fft/nda.hpp"

#include "grids/g_grids.hpp"

namespace bdft_tests
{

using utils::VALUE_EQUAL;
using utils::ARRAY_EQUAL;

TEST_CASE("qe_dmat", "[mean_field_qe]")
{
 // auto [outdir,prefix] = utils::utest_filename("qe_lih222_sym");
  auto& mpi = utils::make_unit_test_mpi_context();
  auto [outdir,prefix] = utils::utest_filename(mf::qe_source);
  mf::qe::qe_readonly mf(mpi, outdir,prefix);

  auto bz = mf.bz();
  auto slist = utils::find_inverse_symmetry(bz.qsymms,bz.symm_list);
  
  auto [sk,dm] = utils::generate_dmatrix<false>(mf, bz.symm_list, slist);
  auto [sk_s,dm_s] = utils::generate_dmatrix<true>(mf, bz.symm_list, slist);
  REQUIRE( nda::sum(sk-sk_s) == 0 );
  REQUIRE( dm.shape()[0] == dm_s.size() );
  for(int i=0; i<dm.shape()[0]; ++i)
  {
    // lazy
    auto Acsr = math::sparse::to_mat(dm_s[i]);
    auto d = nda::frobenius_norm(Acsr-dm.local()(i,nda::ellipsis{}));
    REQUIRE(d < 1e-6); 
  }

}

template<MEMORY_SPACE MEM>
void test_qe_xml(std::string outdir, std::string prefix)
{
  auto& mpi = utils::make_unit_test_mpi_context();
  {
    auto qe_sys = mf::qe::read_xml(mpi,outdir,prefix);
  }

  mf::qe::qe_readonly mf(mpi,outdir,prefix);
  {
    auto wfc_g = mf.wfc_truncated_grid();
    long ngm = wfc_g->size();
    memory::array<MEM, ComplexType, 1> Orb(ngm);
    mf.get_orbital('w',0,0,0,Orb(nda::range::all));
    ComplexType ov = nda::blas::dotc(Orb,Orb);
    VALUE_EQUAL(ov, 1.0);
  }

  {
    mf::qe::qe_readonly mf_(mpi,outdir,prefix,0.0,mf.nbnd()/2);
    REQUIRE(mf_.nbnd() == mf.nbnd()/2);
  }
  
  {
    memory::array<MEM, ComplexType, 1> Orb(mf.fft_grid_size());
    mf.get_orbital('g',0,0,0,Orb);
    ComplexType ov = nda::blas::dotc(Orb,Orb); 
    VALUE_EQUAL(ov, 1.0);
    mf.get_orbital('r',0,0,0,Orb);
    ov = nda::blas::dotc(Orb,Orb);
    VALUE_EQUAL(ov, double(mf.fft_grid_size()));
  }

  { 
    auto wfc_g = mf.wfc_truncated_grid();
    long ngm = wfc_g->size();
    memory::array<MEM, ComplexType, 3> Og(1,2,mf.fft_grid_size());
    memory::array<MEM, ComplexType, 3> Ow(1,2,ngm);
    mf.get_orbital_set('g',0,{0,1},{0,2},Og);
    mf.get_orbital_set('w',0,{0,1},{0,2},Ow);

    memory::array<MEM, ComplexType, 2> ov(2,2);
    nda::blas::gemm(1.0, Ow(0,nda::ellipsis{}), 
			 nda::dagger(Ow(0,nda::ellipsis{})), 0.0, ov);
    auto ov_h = nda::to_host(ov);
    for(int i=0; i<2; ++i) 
      for(int j=0; j<2; ++j) 
        if(i==j) 
          VALUE_EQUAL(ov_h(i,j),1.0);
	else
          VALUE_EQUAL(ov_h(i,j),0.0);

    nda::blas::gemm(1.0, Og(0,nda::ellipsis{}),
                         nda::dagger(Og(0,nda::ellipsis{})), 0.0, ov);
    ov_h = ov;
    for(int i=0; i<2; ++i) 
      for(int j=0; j<2; ++j) 
        if(i==j)
          VALUE_EQUAL(ov_h(i,j),1.0);
        else
          VALUE_EQUAL(ov_h(i,j),0.0);
  }

  {
    grids::truncated_g_grid g(mf.get_sys().ecutrho,mf.fft_grid_dim(),mf.recv());
    nda::array<RealType, 1> Vc(g.size());
    hamilt::potential_g(Vc,g.g_vectors(),mf.kpts()(0,nda::range::all),mf.kpts()(0,nda::range::all));
    auto& g2fft = g.gv_to_fft();
    REQUIRE(g2fft.shape()[0] == g.size());
    //REQUIRE(mf.get_sys().ngm == g.size());
  }

}

TEST_CASE("qe_xml", "[mean_field_qe]")
{
  auto [outdir,prefix] = utils::utest_filename(mf::qe_source);
  test_qe_xml<HOST_MEMORY>(outdir,prefix);
#if defined(ENABLE_DEVICE)
  test_qe_xml<DEVICE_MEMORY>(outdir,prefix);
  test_qe_xml<UNIFIED_MEMORY>(outdir,prefix);
#endif  
}

TEST_CASE("qe_h5", "[mean_field_qe]")
{
  auto [outdir,prefix] = utils::utest_filename(mf::qe_source);
  auto& mpi = utils::make_unit_test_mpi_context();
  mf::qe::qe_readonly mf(mpi,outdir,prefix,0.0,-1,mf::h5_input_type);
//  mf.get_sys().save(outdir+"/"+prefix+".aimb.h5");
}

TEST_CASE("qe_223", "[mean_field_qe]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  {
    auto [outdir,prefix] = utils::utest_filename("qe_lih223");
    mf::qe::qe_readonly mf(mpi,outdir,prefix);
  }
  {
    auto [outdir,prefix] = utils::utest_filename("qe_lih223_sym");
    mf::qe::qe_readonly mf(mpi,outdir,prefix);
  }
  {
    auto [outdir,prefix] = utils::utest_filename("qe_lih223_inv");
    mf::qe::qe_readonly mf(mpi,outdir,prefix);
  }
}

TEST_CASE("qe_GaAs_so", "[mean_field_qe]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  {
    auto [outdir,prefix] = utils::utest_filename("qe_GaAs222_so_hf");
    mf::qe::qe_readonly mf(mpi,outdir,prefix,0.0,-1,mf::h5_input_type);

    {
      auto wfc_g = mf.wfc_truncated_grid();
      long ngm = wfc_g->size();
      memory::array<HOST_MEMORY, ComplexType, 1> Orb(2*ngm);
      mf.get_orbital('w',0,0,0,Orb(nda::range(0,ngm)),nda::range(0,ngm));
      mf.get_orbital('w',0,0,0,Orb(nda::range(ngm,2*ngm)),nda::range(ngm,2*ngm));
      mf.get_orbital('w',0,0,0,Orb(nda::range(0,2*ngm)));
      ComplexType ov = nda::blas::dotc(Orb,Orb);
      VALUE_EQUAL(ov, 1.0);
    }

  }
/*
  {
    auto [outdir,prefix] = utils::utest_filename("qe_si222_so_sym");
    mf::qe::qe_readonly mf(mpi,outdir,prefix);
  }
  {
    auto [outdir,prefix] = utils::utest_filename("qe_si222_so_inv");
    mf::qe::qe_readonly mf(mpi,outdir,prefix);
  }
*/
}



} // bdft_tests
