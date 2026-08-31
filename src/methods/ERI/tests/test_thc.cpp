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
#include <algorithm>
#include "stdio.h"

#include "configuration.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"
#include "utilities/mpi_context.h"

#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/test_common.hpp"

#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/h5.hpp"

#include "mean_field/MF.hpp"
#include "mean_field/default_MF.hpp"
#include "mean_field/qe/qe_readonly.hpp"
#include "grids/g_grids.hpp"
#include "methods/ERI/eri_utils.hpp"
#include "methods/ERI/thc.h"
#include "methods/ERI/cholesky.h"

namespace bdft_tests
{

namespace detail
{

  auto eval_V_thc = [](auto const& Xa, auto const& Xb, auto const& Xc, auto const& Xd, auto const& Vuv)
  {
    long na = Xa.extent(0);
    long nb = Xb.extent(0);
    long nc = Xc.extent(0);
    long nd = Xd.extent(0);
    long nu = Xa.extent(1);
    nda::array<ComplexType,2> Xabu(na*nb,nu);
    nda::array<ComplexType,2> Yabv(na*nb,nu);
    nda::array<ComplexType,2> V(na*nb,nc*nd);
    Xabu() = ComplexType(0.0);
    Yabv() = ComplexType(0.0);
    V() = ComplexType(0.0);

    // Pabu = conj(X(a,u)) * X(b,u)
    for( long u=0; u<nu; u++ )
      for( long a=0, ab=0; a<na; a++ )
        for( long b=0; b<nb; b++, ab++ )
          Xabu(ab,u) = std::conj(Xa(a,u)) * Xb(b,u);
    nda::blas::gemm(Xabu,Vuv,Yabv);
    if(na*nb != nc*nd)
      Xabu = nda::array<ComplexType,2>(nc*nd,nu);
    for( long u=0; u<nu; u++ )
      for( long d=0, dc=0; d<nd; d++ )
        for( long c=0; c<nc; c++, dc++ )
          Xabu(dc,u) = Xd(d,u) * std::conj(Xc(c,u));
    nda::blas::gemm(Yabv,nda::transpose(Xabu),V);
    return V;
  };

#if defined(ENABLE_DEVICE)
  auto max_diff_thc = [](auto& comm, mf::MF& mf, 
                         auto const& dX1a, auto const& dX1b, auto const& dL1, 
                         auto const& dX2a, auto const& dX2b, auto const& dL2) 
  {
    auto all = nda::range::all;
    long nkpts = mf.nkpts();
    double e_g = 0.0, mx_g=0.0;
    
    nda::array<ComplexType, 4> X1a(dX1a.global_shape());
    nda::array<ComplexType, 4> X1b(dX1b.global_shape());
    nda::array<ComplexType, 3> L1(dL1.global_shape());
      
    math::nda::gather(0,dX1a,std::addressof(X1a));
    comm.broadcast_n(X1a.data(),X1a.size(),0);

    math::nda::gather(0,dX1b,std::addressof(X1b));
    comm.broadcast_n(X1b.data(),X1b.size(),0);

    math::nda::gather(0,dL1,std::addressof(L1));
    comm.broadcast_n(L1.data(),L1.size(),0);

    nda::array<ComplexType, 4> X2a(dX2a.global_shape());
    nda::array<ComplexType, 4> X2b(dX2b.global_shape());
    nda::array<ComplexType, 3> L2(dL2.global_shape());
      
    math::nda::gather(0,dX2a,std::addressof(X2a));
    comm.broadcast_n(X2a.data(),X2a.size(),0);

    math::nda::gather(0,dX2b,std::addressof(X2b));
    comm.broadcast_n(X2b.data(),X2b.size(),0);

    math::nda::gather(0,dL2,std::addressof(L2));
    comm.broadcast_n(L2.data(),L2.size(),0);

    long n1 = X1a.extent(2);;
    long n2 = X1b.extent(2);;

    utils::check(n1 == X2a.extent(2), "Size mismatch.");
    utils::check(n2 == X2b.extent(2), "Size mismatch.");

    if(comm.root()) {
      for(long q=0; q<mf.nqpts_ibz(); q++) {
        double e = 0.0, mx=0.0;
        for(long k1=0; k1<nkpts; k1++) 
          for(long k2=0; k2<nkpts; k2++) {
            //long k1=0, k2=0;
            long k1p = mf.qk_to_k2(q,k1);
            long k2p = mf.qk_to_k2(q,k2);
            auto V1 = detail::eval_V_thc(X1a(0,k1,all,all),X1b(0,k1p,all,all),X1b(0,k2p,all,all),X1a(0,k2,all,all),L1(q,all,all));
            auto V2 = detail::eval_V_thc(X2a(0,k1,all,all),X2b(0,k1p,all,all),X2b(0,k2p,all,all),X2a(0,k2,all,all),L2(q,all,all));
            for(int ab=0; ab<n1*n2; ++ab)
              for(int dc=0; dc<n1*n2; ++dc) {
                auto dV = std::abs(V1(ab,dc)-V2(ab,dc));
                e += dV;
                e_g += dV; 
                mx = std::max(mx,dV);
                mx_g = std::max(mx_g,dV);
              }
          }
        app_log(3, "q:{}, ME:{}, Max:{}",q,e/(double(n1*n2*n2*n1*nkpts*nkpts)),mx);
      }
      e_g /= double(n1*n2*n2*n1*nkpts*nkpts*mf.nqpts_ibz());
    }
    comm.broadcast_n(&e_g,1);
    comm.broadcast_n(&mx_g,1);
    return std::make_tuple(e_g,mx_g);
  };
#endif
}

using utils::VALUE_EQUAL;
using utils::ARRAY_EQUAL;
namespace mpi3 = boost::mpi3;
using methods::make_thc_ptree;

TEST_CASE("thc_intpts_ibz", "[methods]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  auto mf = mf::default_MF(mpi, "qe_lih222_sym");

  // standard cutoff
  {
    methods::thc thc(std::addressof(mf), *mpi, make_thc_ptree(mf.ecutrho(),1,1024,1e-4,1));
    auto [ri_ibz,Xa0,Xb0] = thc.interpolating_points<HOST_MEMORY>();
  }

  // test reduced cutoff
  {
    methods::thc thc(std::addressof(mf), *mpi, make_thc_ptree(mf.ecutrho()*0.3,1,1024,1e-4,1));
    auto [ri_ibz,Xa0,Xb0] = thc.interpolating_points<HOST_MEMORY>();
  }     

#if defined(ENABLE_DEVICE)
  // standard cutoff
  { 
    methods::thc thc(std::addressof(mf),mpi,make_thc_ptree(mf.ecutrho(),1,1024,1e-4,1));
    auto [ri_ibz_u,Xau,Xbu] = thc.interpolating_points<UNIFIED_MEMORY>();
  }
  
  // test reduced cutoff
  { 
    methods::thc thc(std::addressof(mf),mpi,make_thc_ptree(mf.ecutrho()*0.3,1,1024,1e-4,1));
    auto [ri_ibz_u,Xau,Xbu] = thc.interpolating_points<UNIFIED_MEMORY>();
  }
#endif

}

TEST_CASE("thc_intpts", "[methods]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  auto mf = mf::default_MF(mpi, mf::qe_source);
  int iq=mf.nqpts()-1; 

  methods::thc thc(std::addressof(mf), *mpi, make_thc_ptree(mf.ecutrho(),1));
  int npts = int(mf.nbnd())*4;
  thc.reset_timers();
  auto [ri_h,Xa0,Xb0] = thc.interpolating_points<HOST_MEMORY>(iq,npts);
  thc.print_timers();
#if defined(ENABLE_DEVICE)
  thc.reset_timers();
  auto [ri_d,Xad,Xbd] = thc.interpolating_points<DEVICE_MEMORY>(iq,npts);
  thc.print_timers();
  thc.reset_timers();
  auto [ri_u,Xau,Xbu] = thc.interpolating_points<UNIFIED_MEMORY>(iq,npts);
  thc.print_timers();
#endif
}

TEST_CASE("thc_intpts_so", "[methods]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  auto mf = mf::default_MF(mpi, "qe_GaAs222_so", mf::h5_input_type);

  methods::thc thc(std::addressof(mf), *mpi, make_thc_ptree(mf.ecutrho()*0.3,8,1024,1e-5));
  thc.reset_timers();
  auto [ri_h,Xa0,Xb0] = thc.interpolating_points<HOST_MEMORY>();
  thc.print_timers();
#if defined(ENABLE_DEVICE)
  thc.reset_timers();
  auto [ri_d,Xad,Xbd] = thc.interpolating_points<DEVICE_MEMORY>();
  thc.print_timers();
  thc.reset_timers();
  auto [ri_u,Xau,Xbu] = thc.interpolating_points<UNIFIED_MEMORY>();
  thc.print_timers();
#endif
}

TEST_CASE("thc_rotated_basis", "[methods]")
{
  decltype(nda::range::all) all;
  auto& mpi = utils::make_unit_test_mpi_context();

  auto mf = mf::default_MF(mpi, mf::qe_source);
  long nspin= mf.nspin();
  long nkpts = mf.nkpts();
  long nbnd = mf.nbnd();
  long nocc = nbnd/2; 
  nda::range a_rng(nocc);
  nda::range b_rng(nbnd);

  nda::array<ComplexType,4> C_skai(nspin,nkpts,nocc,nbnd);
  C_skai() = ComplexType(0.0);
  for(int is=0; is<nspin; is++)
   for(int ik=0; ik<nkpts; ik++)
    for(int ia=0; ia<nocc; ia++)
      C_skai(is,ik,ia,ia) = ComplexType(1.0);

  {
    // change to block=8, and sclae ecut by 0.3 after simple test passes
    methods::thc thc(std::addressof(mf), *mpi, make_thc_ptree(mf.ecutrho()*0.3,8,1024,1e-5));
    nda::array<ComplexType,2> V_r;
    nda::array<ComplexType,4> V_f(nocc,nbnd,nocc,nbnd);

    {

      auto [ri_r,Xa_r,Xb_r] = thc.interpolating_points<HOST_MEMORY>(C_skai,0,-1);
      auto Muv_r = thc.evaluate<HOST_MEMORY>(ri_r,C_skai,Xa_r,*Xb_r);

      auto const& L = std::get<0>(Muv_r);
      nda::array<ComplexType, 4> Xa(Xa_r.global_shape());
      nda::array<ComplexType, 4> Xb(Xb_r->global_shape());
      nda::array<ComplexType, 3> Vuv(L.global_shape());

      math::nda::gather(0,Xa_r,std::addressof(Xa));
      mpi->comm.broadcast_n(Xa.data(),Xa.size(),0);

      math::nda::gather(0,*Xb_r,std::addressof(Xb));
      mpi->comm.broadcast_n(Xb.data(),Xb.size(),0);

      math::nda::gather(0,L,std::addressof(Vuv));
      mpi->comm.broadcast_n(Vuv.data(),Vuv.size(),0);

      auto [ri_f,Xa_f,Xb_f] = thc.interpolating_points<HOST_MEMORY>(0,-1);
      auto Muv_f = thc.evaluate<HOST_MEMORY>(ri_f,Xa_f,Xb_f);

      auto const& Lf = std::get<0>(Muv_f);
      nda::array<ComplexType, 4> Xa_(Xa_f.global_shape());
      nda::array<ComplexType, 3> Vuv_(Lf.global_shape());
      math::nda::gather(0,Xa_f,std::addressof(Xa_));
      mpi->comm.broadcast_n(Xa_.data(),Xa_.size(),0);

      math::nda::gather(0,Lf,std::addressof(Vuv_));
      mpi->comm.broadcast_n(Vuv_.data(),Vuv_.size(),0);

      for(long q=0; q<mf.nqpts_ibz(); q++) {
        double e = 0.0, mx=0.0;
        for(long k1=0; k1<nkpts; k1++) {
        for(long k2=0; k2<nkpts; k2++) {
          //long k1=0, k2=0;
          long k1p = mf.qk_to_k2(q,k1);
          long k2p = mf.qk_to_k2(q,k2);
          V_r = detail::eval_V_thc(Xa(0,k1,all,all),Xb(0,k1p,all,all),Xb(0,k2p,all,all),Xa(0,k2,all,all),Vuv(q,all,all));
          auto Vt_f = detail::eval_V_thc(Xa_(0,k1,all,all),Xa_(0,k1p,all,all),Xa_(0,k2p,all,all),Xa_(0,k2,all,all),Vuv_(q,all,all));
          auto Vr = nda::reshape(V_r,std::array<long,4>{nocc,nbnd,nocc,nbnd});
          auto Vtf = nda::reshape(Vt_f,std::array<long,4>{nbnd,nbnd,nbnd,nbnd});
          // rotate Vtf, really simple for now
          V_f() = ComplexType(0.0);
          for(int a=0; a<nocc; ++a)
          for(int d=0; d<nocc; ++d) 
          for(int i=0; i<nbnd; ++i)
          for(int l=0; l<nbnd; ++l) 
          for(int b=0; b<nbnd; ++b)
          for(int c=0; c<nbnd; ++c) 
            V_f(a,b,d,c) += std::conj(C_skai(0,k1,a,i)) * Vtf(i,b,l,c) * C_skai(0,k2,d,l);
   
          for(int a=0; a<nocc; ++a)
          for(int b=0; b<nbnd; ++b)
          for(int d=0; d<nocc; ++d) 
          for(int c=0; c<nbnd; ++c) { 
            e += std::abs(Vr(a,b,d,c)-V_f(a,b,d,c));
            mx = std::max(mx,std::abs(Vr(a,b,d,c)-V_f(a,b,d,c)));
          }
        }
        }
        app_log(2, "q:{}, ME:{}, Max:{}",q,e/(double(nocc*nocc*nbnd*nbnd*nkpts*nkpts)),mx);
      }
    }
  }

}

TEST_CASE("thc_intpts_pyscf", "[methods]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  auto mf = mf::default_MF(mpi, mf::pyscf_source);

  methods::thc thc(std::addressof(mf), *mpi, make_thc_ptree(mf.ecutrho(),1));
  int npts = int(mf.nbnd())*4;
  auto [ri,Xa,Xb] = thc.interpolating_points<HOST_MEMORY>(0,npts);
}

TEST_CASE("thc_test", "[methods]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  auto mf = mf::default_MF(mpi, mf::qe_source);
  methods::thc thc(std::addressof(mf), *mpi, make_thc_ptree(mf.ecutrho()*0.3,8,1024,1e-5));

  { // reduced cutoff
#if defined(ENABLE_DEVICE)
//    thc.reset_timers();
//    auto [ri_u,Xau,Xbu] = thc.interpolating_points<UNIFIED_MEMORY>();
//    auto Muv_u = thc.evaluate<UNIFIED_MEMORY>(ri_u,Xau,Xbu);
//    thc.print_timers();

    thc.reset_timers();
    auto [ri_d,Xad,Xbd] = thc.interpolating_points<DEVICE_MEMORY>();
    auto Muv_d = thc.evaluate<DEVICE_MEMORY>(ri_d,Xad,Xbd);
    thc.print_timers();
#else
    thc.reset_timers();
    auto [ri,Xa,Xb] = thc.interpolating_points<HOST_MEMORY>();
    auto Muv_h = thc.evaluate<HOST_MEMORY>(ri,Xa,Xb);
    thc.print_timers();
#endif
  }
}

TEST_CASE("thc", "[methods]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  auto mf = mf::default_MF(mpi, mf::qe_source);

  { // standard cutoff
    methods::thc thc(std::addressof(mf), *mpi, make_thc_ptree(mf.ecutrho(),1));
    int npts = int(mf.nbnd())*4;
    auto [ri,Xa,Xb] = thc.interpolating_points<HOST_MEMORY>(0,npts);
    thc.reset_timers();
    auto Muv_h = thc.evaluate<HOST_MEMORY>(ri,Xa,Xb); 
    thc.print_timers();
#if defined(ENABLE_DEVICE)
    auto const& L_h = std::get<0>(Muv_h);
    auto [ri_u,Xau,Xbu] = thc.interpolating_points<UNIFIED_MEMORY>(0,npts);
    thc.reset_timers();
    auto Muv_u = thc.evaluate<UNIFIED_MEMORY>(ri_u,Xau,Xbu);
    thc.print_timers();
    auto const& L_u = std::get<0>(Muv_u);
    auto [avE,mxE] = detail::max_diff_thc(mpi.comm,mf,Xa,Xa,L_h,Xau,Xau,L_u);
    app_log(2, "  thc - standard cutoff [UNIFIED]: avE:{}   mxE:{}",avE,mxE);
    VALUE_EQUAL(avE,0.0);  
    VALUE_EQUAL(mxE,0.0);  

    auto [ri_d,Xad,Xbd] = thc.interpolating_points<DEVICE_MEMORY>(0,npts);
    thc.reset_timers();
    auto Muv_d = thc.evaluate<DEVICE_MEMORY>(ri_d,Xad,Xbd);
    thc.print_timers();
    auto const& L_d = std::get<0>(Muv_d);
    std::tie(avE,mxE) = detail::max_diff_thc(mpi.comm,mf,Xa,Xa,L_h,Xad,Xad,L_d);
    app_log(2, "  thc - standard cutoff [DEVICE]: avE:{}   mxE:{}",avE,mxE);
    VALUE_EQUAL(avE,0.0);
    VALUE_EQUAL(mxE,0.0);
#endif
  }

  { // reduced cutoff
    methods::thc thc(std::addressof(mf), *mpi, make_thc_ptree(mf.ecutrho()*0.3,1));
    int npts = int(mf.nbnd())*4;
    auto [ri,Xa,Xb] = thc.interpolating_points<HOST_MEMORY>(0,npts);
    thc.reset_timers();
    auto Muv_h = thc.evaluate<HOST_MEMORY>(ri,Xa,Xb);
    thc.print_timers();
#if defined(ENABLE_DEVICE)
    auto const& L_h = std::get<0>(Muv_h);
    auto [ri_u,Xau,Xbu] = thc.interpolating_points<UNIFIED_MEMORY>(0,npts);
    thc.reset_timers();
    auto Muv_u = thc.evaluate<UNIFIED_MEMORY>(ri_u,Xau,Xbu);
    thc.print_timers();
    auto const& L_u = std::get<0>(Muv_u);
    auto [avE,mxE] = detail::max_diff_thc(mpi.comm,mf,Xa,Xa,L_h,Xau,Xau,L_u);
    app_log(2, "  thc - reduced cutoff [UNIFIED]: avE:{}   mxE:{}",avE,mxE);
    VALUE_EQUAL(avE,0.0);  
    VALUE_EQUAL(mxE,0.0);  

    auto [ri_d,Xad,Xbd] = thc.interpolating_points<DEVICE_MEMORY>(0,npts);
    thc.reset_timers();
    auto Muv_d = thc.evaluate<DEVICE_MEMORY>(ri_d,Xad,Xbd);
    thc.print_timers();
    auto const& L_d = std::get<0>(Muv_d);
    std::tie(avE,mxE) = detail::max_diff_thc(mpi.comm,mf,Xa,Xa,L_h,Xad,Xad,L_d);
    app_log(2, "  thc - standard cutoff [DEVICE]: avE:{}   mxE:{}",avE,mxE);
    VALUE_EQUAL(avE,0.0);
    VALUE_EQUAL(mxE,0.0);
#endif
  }
}

TEST_CASE("thc_so", "[methods]")
{
  auto all = nda::range::all;
  auto& mpi = utils::make_unit_test_mpi_context();

  auto mf = mf::default_MF(mpi, "qe_GaAs222_so", mf::h5_input_type);

  { // standard cutoff
    methods::thc thc(std::addressof(mf), *mpi, make_thc_ptree(mf.ecutrho(),1));
    int npts = int(mf.nbnd())*4;
    auto [ri,Xa,Xb] = thc.interpolating_points<HOST_MEMORY>(0,npts);
    thc.reset_timers();
    auto Muv_h = thc.evaluate<HOST_MEMORY>(ri,Xa,Xb); 
    thc.print_timers();
#if defined(ENABLE_DEVICE)
    auto const& L_h = std::get<0>(Muv_h);
    thc.reset_timers();
    auto [ri_u,Xau,Xbu] = thc.interpolating_points<UNIFIED_MEMORY>(0,npts);
    auto Muv_u = thc.evaluate<UNIFIED_MEMORY>(ri_u,Xau,Xbu);
    thc.print_timers();
    auto const& L_u = std::get<0>(Muv_u);
    auto [avE,mxE] = detail::max_diff_thc(mpi.comm,mf,Xa,Xa,L_h,Xau,Xau,L_u);
    app_log(2, "  thc_so - standard cutoff [UNIFIED]: avE:{}   mxE:{}",avE,mxE);
    VALUE_EQUAL(avE,0.0);
    VALUE_EQUAL(mxE,0.0);

    auto [ri_d,Xad,Xbd] = thc.interpolating_points<DEVICE_MEMORY>(0,npts);
    thc.reset_timers();
    auto Muv_d = thc.evaluate<DEVICE_MEMORY>(ri_d,Xad,Xbd);
    thc.print_timers();
    auto const& L_d = std::get<0>(Muv_d);
    std::tie(avE,mxE) = detail::max_diff_thc(mpi.comm,mf,Xa,Xa,L_h,Xad,Xad,L_d);
    app_log(2, "  thc_so - standard cutoff [DEVICE]: avE:{}   mxE:{}",avE,mxE);
    VALUE_EQUAL(avE,0.0);
    VALUE_EQUAL(mxE,0.0);
#endif
  }

  { // reduced cutoff
    methods::thc thc(std::addressof(mf), *mpi, make_thc_ptree(mf.ecutrho()*0.3,1));
    int npts = int(mf.nbnd())*4;
    thc.reset_timers();
    auto [ri,Xa,Xb] = thc.interpolating_points<HOST_MEMORY>(0,npts);
    auto Muv_h = thc.evaluate<HOST_MEMORY>(ri,Xa,Xb);
    thc.print_timers();
#if defined(ENABLE_DEVICE)
    auto const& L_h = std::get<0>(Muv_h);
    thc.reset_timers();
    auto [ri_u,Xau,Xbu] = thc.interpolating_points<UNIFIED_MEMORY>(0,npts);
    auto Muv_u = thc.evaluate<UNIFIED_MEMORY>(ri_u,Xau,Xbu);
    thc.print_timers();
    auto const& L_u = std::get<0>(Muv_u);
    auto [avE,mxE] = detail::max_diff_thc(mpi.comm,mf,Xa,Xa,L_h,Xau,Xau,L_u);
    app_log(2, "  thc_so - reduced cutoff [UNIFIED]: avE:{}   mxE:{}",avE,mxE);
    VALUE_EQUAL(avE,0.0);
    VALUE_EQUAL(mxE,0.0);

    thc.reset_timers();
    auto [ri_d,Xad,Xbd] = thc.interpolating_points<DEVICE_MEMORY>(0,npts);
    auto Muv_d = thc.evaluate<DEVICE_MEMORY>(ri_d,Xad,Xbd);
    thc.print_timers();
    auto const& L_d = std::get<0>(Muv_d);
    std::tie(avE,mxE) = detail::max_diff_thc(mpi.comm,mf,Xa,Xa,L_h,Xad,Xad,L_d);
    app_log(2, "  thc_so - reduced cutoff [DEVICE]: avE:{}   mxE:{}",avE,mxE);
    VALUE_EQUAL(avE,0.0);
    VALUE_EQUAL(mxE,0.0);
#endif
  }

  { // compare against cholesky
    methods::cholesky chol(std::addressof(mf), *mpi, methods::make_chol_ptree(1e-6,mf.ecutrho()));
    // make check with larger block size
    methods::thc thc(std::addressof(mf), *mpi, make_thc_ptree(mf.ecutrho(),1,1024,1e-6));

    auto nspin = mf.nspin_in_basis();
    auto npol  = mf.npol_in_basis();
    auto nkpts = mf.nkpts();
    auto nbnd  = mf.nbnd();

    // interpolating points
    auto [ri,dXa,dXb] = thc.interpolating_points<HOST_MEMORY>();

    // interpolating vectors
    auto Muv = thc.evaluate<HOST_MEMORY>(ri,dXa,dXb);

    auto const& dVuv = std::get<0>(Muv);
    nda::array<ComplexType, 4> Xa(dXa.global_shape());
    nda::array<ComplexType, 3> Vuv(dVuv.global_shape());

    math::nda::gather(0,dXa,std::addressof(Xa));
    mpi->comm.broadcast_n(Xa.data(),Xa.size(),0);

    math::nda::gather(0,dVuv,std::addressof(Vuv));
    mpi->comm.broadcast_n(Vuv.data(),Vuv.size(),0);

    nda::array<ComplexType,2> V1;
    nda::array<ComplexType,2> V2(nbnd*nbnd,nbnd*nbnd);
    nda::array<ComplexType,5> L;
  
    for(long q=0; q<mf.nqpts_ibz(); q++) 
    { 
      app_log(0,"\nCholesky q:{}",q);
      auto dL = chol.evaluate<HOST_MEMORY>(q);
      utils::check( dL.global_shape()[1] == nspin*npol, "Shape mismatch.");
      L.resize(dL.global_shape());
      math::nda::gather(0,dL,std::addressof(L));
      mpi->comm.broadcast_n(L.data(),L.size(),0);
      auto L4d = nda::reshape(L,std::array<long,4>{L.extent(0),L.extent(1),L.extent(2),nbnd*nbnd});
      L4d() = nda::conj(L4d);

      double e = 0.0, mx=0.0;
      for(long k1=0, k12=0; k1<nkpts; k1++) {
      for(long k2=0; k2<nkpts; k2++, k12++) {
        if( k12%mpi->comm.size() != mpi->comm.rank() ) continue;
        //long k1=0, k2=0;
        long k1p = mf.qk_to_k2(q,k1);
        long k2p = mf.qk_to_k2(q,k2);
        for(int pp=0; pp<npol*npol; pp++) {
          int p1 = pp/npol;
          int p2 = pp%npol;
          V1 = detail::eval_V_thc(Xa(p1,k1,all,all),Xa(p1,k1p,all,all),Xa(p2,k2p,all,all),Xa(p2,k2,all,all),Vuv(q,all,all));
        // (ab,dc) = sum_n L(n,isp,k1,a,b) * conj( L(n,isp,k2,dc) )
          nda::blas::gemm(ComplexType(1.0),nda::dagger(L4d(all,p1,k1,all)),L4d(all,p2,k2,all),
                          ComplexType(0.0),V2); 
          for(int ab=0; ab<nbnd*nbnd; ++ab) {
            for(int dc=0; dc<nbnd*nbnd; ++dc) {
              e += std::abs(V1(ab,dc)-V2(ab,dc));
              mx = std::max(mx,std::abs(V1(ab,dc)-V2(ab,dc)));
            }
          }
        }
      }
      }
      mpi->comm.all_reduce_in_place_n(&e,1,std::plus<>{});
      mpi->comm.all_reduce_in_place_n(&mx,1,boost::mpi3::max<>{});
      app_log(2, "q:{}, ME:{}, Max:{}",q,e/(double(nbnd*nbnd*nbnd*nbnd*nkpts*nkpts*npol*npol)),mx);
    }
  }
}

TEST_CASE("thc_ranges", "[methods]")
{
  auto all = nda::range::all;
  auto& mpi = utils::make_unit_test_mpi_context();

  auto mf = mf::default_MF(mpi, mf::qe_source);
  long nbnd = mf.nbnd();
  long nocc = nbnd/2;
  long nkpts = mf.nkpts();
  nda::range a_rng(nocc);
  nda::range b_rng(nbnd);

  methods::thc thc(std::addressof(mf), *mpi, make_thc_ptree(mf.ecutrho()*0.3,8,1024,1e-5));

  auto [ri_r,Xa_r,Xb_r] = thc.interpolating_points<HOST_MEMORY>(0,-1,a_rng,b_rng);
  auto Muv_r = thc.evaluate<HOST_MEMORY>(ri_r,Xa_r,Xb_r,false,a_rng,b_rng);

  { // check device implementations 
    thc.print_timers();
#if defined(ENABLE_DEVICE)
    auto const& L_h = std::get<0>(Muv_r);
    auto [ri_u,Xau,Xbu] = thc.interpolating_points<UNIFIED_MEMORY>(0,-1,a_rng,b_rng);
    auto Muv_u = thc.evaluate<UNIFIED_MEMORY>(ri_u,Xau,Xbu,false,a_rng,b_rng);
    auto const& L_u = std::get<0>(Muv_u);
    auto [avE,mxE] = detail::max_diff_thc(mpi.comm,mf,Xa_r,*Xb_r,L_h,Xau,*Xbu,L_u);
    app_log(2, "  thc_so - standard cutoff [UNIFIED]: avE:{}   mxE:{}",avE,mxE);
    VALUE_EQUAL(avE,0.0);
    VALUE_EQUAL(mxE,0.0);

    auto [ri_d,Xad,Xbd] = thc.interpolating_points<DEVICE_MEMORY>(0,-1,a_rng,b_rng);
    auto Muv_d = thc.evaluate<DEVICE_MEMORY>(ri_d,Xad,Xbd,false,a_rng,b_rng);
    auto const& L_d = std::get<0>(Muv_d);
    std::tie(avE,mxE) = detail::max_diff_thc(mpi.comm,mf,Xa_r,*Xb_r,L_h,Xad,*Xbd,L_d);
    app_log(2, "  thc_so - standard cutoff [DEVICE]: avE:{}   mxE:{}",avE,mxE);
    VALUE_EQUAL(avE,0.0);
    VALUE_EQUAL(mxE,0.0);
#endif
  }

  { // compare against full calculation  
    nda::array<ComplexType,2> V_r,V_f;


    auto const& L = std::get<0>(Muv_r);
    nda::array<ComplexType, 4> Xa(Xa_r.global_shape());
    nda::array<ComplexType, 4> Xb(Xb_r->global_shape());
    nda::array<ComplexType, 3> Vuv(L.global_shape());

    math::nda::gather(0,Xa_r,std::addressof(Xa));
    mpi->comm.broadcast_n(Xa.data(),Xa.size(),0);

    math::nda::gather(0,*Xb_r,std::addressof(Xb));
    mpi->comm.broadcast_n(Xb.data(),Xb.size(),0);

    math::nda::gather(0,L,std::addressof(Vuv));
    mpi->comm.broadcast_n(Vuv.data(),Vuv.size(),0);

    auto [ri_f,Xa_f,Xb_f] = thc.interpolating_points<HOST_MEMORY>(0,-1);
    auto Muv_f = thc.evaluate<HOST_MEMORY>(ri_f,Xa_f,Xb_f);

    auto const& Lf = std::get<0>(Muv_f);
    nda::array<ComplexType, 4> Xa_(Xa_f.global_shape());
    nda::array<ComplexType, 3> Vuv_(Lf.global_shape());
    math::nda::gather(0,Xa_f,std::addressof(Xa_));
    mpi->comm.broadcast_n(Xa_.data(),Xa_.size(),0);
      
    math::nda::gather(0,Lf,std::addressof(Vuv_));
    mpi->comm.broadcast_n(Vuv_.data(),Vuv_.size(),0);
      
    for(long q=0; q<mf.nqpts_ibz(); q++) {
      double e = 0.0, mx=0.0;
      for(long k1=0; k1<nkpts; k1++) {
      for(long k2=0; k2<nkpts; k2++) {
        //long k1=0, k2=0;
        long k1p = mf.qk_to_k2(q,k1);
        long k2p = mf.qk_to_k2(q,k2);
        V_r = detail::eval_V_thc(Xa(0,k1,all,all),Xb(0,k1p,all,all),Xb(0,k2p,all,all),Xa(0,k2,all,all),Vuv(q,all,all));
        V_f = detail::eval_V_thc(Xa_(0,k1,all,all),Xa_(0,k1p,all,all),Xa_(0,k2p,all,all),Xa_(0,k2,all,all),Vuv_(q,all,all));
        auto Vr = nda::reshape(V_r,std::array<long,4>{nocc,nbnd,nocc,nbnd});
        auto Vf = nda::reshape(V_f,std::array<long,4>{nbnd,nbnd,nbnd,nbnd});
        for(int a=0; a<nocc; ++a) 
        for(int b=0; b<nbnd; ++b) 
        for(int c=0; c<nocc; ++c) 
        for(int d=0; d<nbnd; ++d) { 
          e += std::abs(Vr(a,b,c,d)-Vf(a,b,c,d));
          mx = std::max(mx,std::abs(Vr(a,b,c,d)-Vf(a,b,c,d)));
        }
      }
      }
      app_log(2, "q:{}, ME:{}, Max:{}",q,e/(double(nocc*nocc*nbnd*nbnd*nkpts*nkpts)),mx);
    }
  }
}

TEST_CASE("thc_svd", "[methods]")
{  // no need for device test
  decltype(nda::range::all) all;
  auto& mpi = utils::make_unit_test_mpi_context();

  auto mf = mf::default_MF(mpi, mf::qe_source);

  nda::array<int,3> Np(20,mf.nqpts_ibz(),3);
  nda::array<double,2> Vp(20,1300);
  Np()=0;
  Vp()=0.0;
  long Np0;

  for(int i=10; i>=1; i-=3)
  { 
    double x(i/10.0);
    app_log(0," Reducing cutoff by x:{}",x);
    methods::thc thc(std::addressof(mf), *mpi, make_thc_ptree(x*mf.ecutrho(),8,1024,1e-4));
    auto [ri,Xa,Xb] = thc.interpolating_points<HOST_MEMORY>();
    long M = ri.extent(0); 
    Np(i,all,all) = M;
    if(i==10) Np0 = M;
    app_log(0,"  - Np: {}",M); 
    auto L = thc.evaluate<HOST_MEMORY>(ri,Xa,Xb);
    auto const& V = std::get<0>(L);
    nda::array<ComplexType, 3> X;
    if( mpi->comm.root() ) X = nda::array<ComplexType, 3>(V.global_shape());
    math::nda::gather(0,V,std::addressof(X));
    if( mpi->comm.root() ) {
      nda::matrix<ComplexType,nda::F_layout> U(M,M), Vt(M,M);
      nda::array<double,1> S(M); 
      for( auto iq : nda::range(X.extent(0)) ) { 
        utils::check(nda::lapack::gesvd(nda::transpose(X(iq,all,all)),S,U,Vt)==0, "SVD error.");
        if(iq==0)
          Vp(i,all) = S(nda::range(1300));
        for(int j=0; j<M; ++j) 
          if(std::abs(S(j)) < 1e-8) {
            Np(i,iq,0) = j;
            break;
          }
        for(int j=0; j<M; ++j) 
          if(std::abs(S(j)) < 1e-7) {
            Np(i,iq,1) = j;
            break;
          }
        for(int j=0; j<M; ++j) 
          if(std::abs(S(j)) < 1e-6) {
            Np(i,iq,2) = j;
            break;
          }
      } 
    } 
  }
  app_log(0," Np0: {}",Np0);
  for(int i=10; i>=1; i-=3)
  { 
    double x(i/10.0);
    app_log(0, "  x: {}",x);
    for( auto iq : nda::range(Np.extent(1)) ) 
      app_log(0,"    - iq: {}, Np (1e-6): {}, Np (1e-5): {}, Np (1e-4): {}",
              iq,Np(i,iq,0),Np(i,iq,1),Np(i,iq,2));
  }
  if(mpi->comm.root())
    for(int i=10; i>=1; i-=3)
      std::cout<<" Vp: " <<i <<" " <<Vp(i,all) <<std::endl <<std::endl;

}

TEST_CASE("thc_nnr_blk", "[methods]")
{ 
  // testing in serial for now!
  auto& mpi = utils::make_unit_test_mpi_context();
  
  auto mf = mf::default_MF(mpi, mf::qe_source);

  methods::thc thc(std::addressof(mf), *mpi, make_thc_ptree(mf.ecutrho()*0.3,1,
                   1024,0.1,100,1));
  int npts = int(mf.nbnd())*4;
  auto [ri,dXa,dXb] = thc.interpolating_points<HOST_MEMORY>(0,npts);
  auto [V_, chi_head, chi_bar_head, bs_corr] = thc.evaluate<HOST_MEMORY>(ri,dXa,dXb);
  REQUIRE(V_.global_shape()[1] == npts); 
  REQUIRE(V_.global_shape()[2] == npts); 

  nda::array<ComplexType, 4> Xa(dXa.global_shape());
  nda::array<ComplexType, 3> V(V_.global_shape());
  nda::array<ComplexType, 3> V2(V_.global_shape());

  for(auto nnr_blk : nda::range(2,6,2) ) {

    methods::thc thc2(std::addressof(mf), *mpi, make_thc_ptree(mf.ecutrho()*0.3,1,
                     1024,0.1,100,nnr_blk));
    auto [V2_, chi_head2, chi_bar_head2, bs_corr_] = thc2.evaluate<HOST_MEMORY>(ri,dXa,dXb);

    REQUIRE(V_.global_shape() == V2_.global_shape());

    // easy for now
    math::nda::gather(0,dXa,std::addressof(Xa));
    auto X = nda::reshape(Xa,std::array<long,2>{Xa.extent(0)*Xa.extent(1)*Xa.extent(2),npts});
    mpi->comm.broadcast_n(X.data(),X.size(),0);

    math::nda::gather(0,V_,std::addressof(V));
    mpi->comm.broadcast_n(V.data(),V.size(),0);

    math::nda::gather(0,V2_,std::addressof(V2));
    mpi->comm.broadcast_n(V2.data(),V2.size(),0);

    int a=0,b=0,c=0,d=0,q=0;
    ComplexType x(0.0),x2(0.0);
    for( long u=0; u<npts; u++ ) { 
      ComplexType y(0.0),y2(0.0);
      for( long v=0; v<npts; v++ ) {
        y+=V(q,u,v)*std::conj(X(c,v))*X(d,v);
        y2+=V2(q,u,v)*std::conj(X(c,v))*X(d,v);
      }
      x += std::conj(X(a,u))*X(b,u)*y;
      x2 += std::conj(X(a,u))*X(b,u)*y2;
    }
    VALUE_EQUAL(x,x2);  

    a=0; b=1; c=2; d=3;
    x = ComplexType(0.0);
    x2 = ComplexType(0.0);
    for( long u=0; u<npts; u++ ) {
      ComplexType y(0.0),y2(0.0);
      for( long v=0; v<npts; v++)  {
        y+=V(q,u,v)*std::conj(X(c,v))*X(d,v);
        y2+=V2(q,u,v)*std::conj(X(c,v))*X(d,v);
      } 
      x += std::conj(X(a,u))*X(b,u)*y;
      x2 += std::conj(X(a,u))*X(b,u)*y2;
    }
    VALUE_EQUAL(x,x2);
  }
}
TEST_CASE("thc_io", "[methods]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  auto mf = mf::default_MF(mpi, mf::qe_source);

  methods::thc thc(std::addressof(mf), *mpi, make_thc_ptree(mf.ecutrho()));
  int npts = int(mf.nbnd())*4;
  auto [ri,Xa,Xb] = thc.interpolating_points<HOST_MEMORY>(0,npts);

  {
    if(mpi->comm.root()) {
      h5::file fh5("dummy.h5",'w');
      h5::group top{fh5};
      auto gh5 = top.create_group("hamiltonian").create_group("thc");
      thc.write_meta_data(gh5,"bdft");
      thc.evaluate<HOST_MEMORY>(gh5,"bdft",ri,Xa,Xb);
    } else {
      h5::group L0{};
      thc.evaluate<HOST_MEMORY>(L0,"bdft",ri,Xa,Xb);
    }
    mpi->comm.barrier();
  }
  thc.print_timers();
  if(mpi->comm.root()) remove("dummy.h5");

#if defined(ENABLE_DEVICE)
  {
    thc.reset_timers();
    auto [ri_d,Xa_d,Xb_d] = thc.interpolating_points<DEVICE_MEMORY>(0,npts);
    if(mpi->comm.root()) {
      h5::file fh5("dummy.h5",'w');
      h5::group top{fh5};
      auto gh5 = top.create_group("hamiltonian").create_group("thc");
      thc.write_meta_data(gh5,"bdft");
      thc.evaluate<DEVICE_MEMORY>(gh5,"bdft",ri_d,Xa_d,Xb_d);
    } else {
      h5::group L0{};
      thc.evaluate<DEVICE_MEMORY>(L0,"bdft",ri_d,Xa_d,Xb_d);
    }
    mpi->comm.barrier();
    thc.print_timers();
    if(mpi->comm.root()) remove("dummy.h5");
  }

  { 
    thc.reset_timers();
    auto [ri_u,Xa_u,Xb_u] = thc.interpolating_points<UNIFIED_MEMORY>(0,npts);
    if(mpi->comm.root()) {
      h5::file fh5("dummy.h5",'w');
      h5::group top{fh5};
      auto gh5 = top.create_group("hamiltonian").create_group("thc");
      thc.write_meta_data(gh5,"bdft");
      thc.evaluate<UNIFIED_MEMORY>(gh5,"bdft",ri_u,Xa_u,Xb_u);
    } else {
      h5::group L0{};
      thc.evaluate<UNIFIED_MEMORY>(L0,"bdft",ri_u,Xa_u,Xb_u);
    }
    mpi->comm.barrier();
    thc.print_timers();
    if(mpi->comm.root()) remove("dummy.h5");
  }
#endif

}

#if 0//defined(ENABLE_SLATE)
TEST_CASE("thc_coul_metric", "[methods]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  auto mf = mf::default_MF(mpi, mf::qe_source);

  methods::cholesky chol(std::addressof(mf), *mpi, methods::make_chol_ptree(1e-4,mf.ecutrho()));
  methods::thc thc(std::addressof(mf), *mpi, make_thc_ptree(mf.ecutrho()));

  int npts = int(mf.nbnd())*4;
  for(int q=0; q<mf.nqpts(); q++)
  {
    auto L = chol.evaluate<HOST_MEMORY>(q);
    auto [ri,Xa,Xb] = thc.interpolating_points<HOST_MEMORY>(q,npts);
    auto Muv = thc.evaluate<HOST_MEMORY>(q,ri,L);
  }
}
#endif

/*
TEST_CASE("thc_chol_ov", "[methods]") {
  auto mpi.comm = mpi3::environment::get_mpi.comm_instance();

  auto mf = mf::default_MF(mpi.comm, mf::pyscf_source);
  int nkpts = mf.nkpts();
  int ik = 2;
  double eri_cutoff = 1e-6;

  nda::array<ComplexType, 2> U_chol(mf.nbnd()*mf.nbnd(), mf.nbnd()*mf.nbnd());
  nda::array<ComplexType, 2> U_thc(mf.nbnd()*mf.nbnd(), mf.nbnd()*mf.nbnd());

  // Cholesky ERIs as the references
  methods::cholesky chol(std::addressof(mf), mpi, eri_cutoff, mf.ecutrho());
  methods::thc thc(std::addressof(mf), mpi.comm, nda::range(-1,-1), nda::range(-1,-1),
                 eri_cutoff, 50, mf.ecutrho());

  // q-independent interpolating points
  int nIpts = int(mf.nbnd())*30;
  auto [ri,Xa,Xb] = thc.interpolating_points<HOST_MEMORY>(0, nIpts);

  for (int iq = 0; iq < mf.nkpts(); ++iq) {
    auto L = chol.evaluate<HOST_MEMORY>(iq,Xa,Xb);

    auto [dPa, dPb] = thc.interpolating_basis<HOST_MEMORY>(ri, iq);
    auto Vuv = thc.evaluate<HOST_MEMORY>(iq, ri,Xa,Xb);
    int Np = (int)Vuv.local().shape(0);

    {
      auto L_loc = L.local();
      auto Lqk = nda::make_regular(L_loc(nda::range::all, 0, ik, nda::range::all, nda::range::all) );
      nda::array<ComplexType, 3> LL(Lqk.shape());

      auto L_2D = nda::reshape(Lqk, std::array<long,2>{Lqk.shape(0), Lqk.shape(1)*Lqk.shape(2)});
      auto LL_2D = nda::reshape(LL, std::array<long,2>{LL.shape(0), LL.shape(1)*LL.shape(2)});
      for (size_t P = 0; P < Lqk.shape(0); ++P) {
        auto LP = Lqk(P, nda::ellipsis{});
        LL(P, nda::ellipsis{}) = nda::conj(nda::transpose(LP));
      }
      nda::blas::gemm(1.0, nda::transpose(L_2D), LL_2D, 0.0, U_chol);
    }

    {
      auto X_sk_a_P = nda::reshape(dPa.local(), std::array<long, 3>{mf.nspin()*mf.nkpts(), mf.nbnd(), Np});
      auto XX_sk_b_P = nda::reshape(dPb.local(), std::array<long, 3>{mf.nspin()*mf.nkpts(), mf.nbnd(), Np});
      auto V_PQ = Vuv.local();

      // U_ab_cd = X_aP_conj * XX_bP * V_PQ * XX_cQ_conj * X_dQ
      //         = Y_Pab * V_PQ * YY_Qcd
      nda::array<ComplexType, 3> Y_Pab(Np, mf.nbnd(), mf.nbnd());
      nda::array<ComplexType, 3> YY_Qcd(Np, mf.nbnd(), mf.nbnd());
      nda::array<ComplexType, 1> X(mf.nbnd());
      nda::array<ComplexType, 1> XX(mf.nbnd());
      for (size_t P = 0; P < Np; ++P) {
        auto YP = Y_Pab(P, nda::range::all, nda::range::all);
        X = nda::conj(X_sk_a_P(ik, nda::range::all, P));
        XX = XX_sk_b_P(ik, nda::range::all, P);
        YP = nda::blas::outer_product(X, XX);
      }

      for (size_t Q = 0; Q < Np; ++Q) {
        auto YYQ = YY_Qcd(Q, nda::range::all, nda::range::all);
        X = X_sk_a_P(ik, nda::range::all, Q);
        XX = nda::conj(XX_sk_b_P(ik, nda::range::all, Q));
        YYQ = nda::blas::outer_product(XX, X);
      }
      // U_ab_cd = Y_Pab * V_PQ * YY_Qcd
      auto YY_Q_cd_2D = nda::reshape(YY_Qcd, std::array<long, 2>{Np, mf.nbnd()*mf.nbnd()});
      auto Y_P_ab_2D = nda::reshape(Y_Pab, std::array<long, 2>{Np, mf.nbnd()*mf.nbnd()});
      nda::array<ComplexType, 2> T_P_cd(Np, mf.nbnd()*mf.nbnd());
      nda::blas::gemm(V_PQ, YY_Q_cd_2D, T_P_cd);
      nda::blas::gemm(nda::transpose(Y_P_ab_2D), T_P_cd, U_thc);
    }
    ARRAY_EQUAL(U_thc, U_chol, eri_cutoff*10);
  }
}

#if defined(ENABLE_SLATE)
TEST_CASE("thc_chol_ls", "[methods]") {
  auto mpi.comm = mpi3::environment::get_mpi.comm_instance();

  auto mf = mf::default_MF(mpi.comm, mf::pyscf_source);
  int nkpts = mf.nkpts();
  int ik = 2;
  double eri_cutoff = 1e-6;

  nda::array<ComplexType, 2> U_chol(mf.nbnd()*mf.nbnd(), mf.nbnd()*mf.nbnd());
  nda::array<ComplexType, 2> U_thc(mf.nbnd()*mf.nbnd(), mf.nbnd()*mf.nbnd());

  // Cholesky ERIs as the references
  methods::cholesky chol(std::addressof(mf), mpi, eri_cutoff, mf.ecutrho());
  methods::thc thc(std::addressof(mf), mpi.comm, nda::range(-1,-1), nda::range(-1,-1),
                 eri_cutoff, 50, mf.ecutrho());

  // q-independent interpolating points
  int nIpts = int(mf.nbnd())*30;
  auto [ri,Xa,Xb] = thc.interpolating_points<HOST_MEMORY>(0, nIpts);

  for (int iq = 0; iq < mf.nkpts(); ++iq) {
    auto L = chol.evaluate<HOST_MEMORY>(iq,Xa,Xb);

    auto [dPa, dPb] = thc.interpolating_basis<HOST_MEMORY>(ri, iq);
    auto Vuv = thc.evaluate<HOST_MEMORY>(iq, ri, L);
    int Np = (int)Vuv.local().shape(0);

    {
      auto L_loc = L.local();
      auto Lqk = nda::make_regular(L_loc(nda::range::all, 0, ik, nda::range::all, nda::range::all) );
      nda::array<ComplexType, 3> LL(Lqk.shape());

      auto L_2D = nda::reshape(Lqk, std::array<long,2>{Lqk.shape(0), Lqk.shape(1)*Lqk.shape(2)});
      auto LL_2D = nda::reshape(LL, std::array<long,2>{LL.shape(0), LL.shape(1)*LL.shape(2)});
      for (size_t P = 0; P < Lqk.shape(0); ++P) {
        auto LP = Lqk(P, nda::ellipsis{});
        LL(P, nda::ellipsis{}) = nda::conj(nda::transpose(LP));
      }
      nda::blas::gemm(1.0, nda::transpose(L_2D), LL_2D, 0.0, U_chol);
    }

    {
      auto X_sk_a_P = nda::reshape(dPa.local(), std::array<long, 3>{mf.nspin()*mf.nkpts(), mf.nbnd(), Np});
      auto XX_sk_b_P = nda::reshape(dPb.local(), std::array<long, 3>{mf.nspin()*mf.nkpts(), mf.nbnd(), Np});
      auto V_PQ = Vuv.local();

      // U_ab_cd = X_aP_conj * XX_bP * V_PQ * XX_cQ_conj * X_dQ
      //         = Y_Pab * V_PQ * YY_Qcd
      nda::array<ComplexType, 3> Y_Pab(Np, mf.nbnd(), mf.nbnd());
      nda::array<ComplexType, 3> YY_Qcd(Np, mf.nbnd(), mf.nbnd());
      nda::array<ComplexType, 1> X(mf.nbnd());
      nda::array<ComplexType, 1> XX(mf.nbnd());
      for (size_t P = 0; P < Np; ++P) {
        auto YP = Y_Pab(P, nda::range::all, nda::range::all);
        X = nda::conj(X_sk_a_P(ik, nda::range::all, P));
        XX = XX_sk_b_P(ik, nda::range::all, P);
        YP = nda::blas::outer_product(X, XX);
      }

      for (size_t Q = 0; Q < Np; ++Q) {
        auto YYQ = YY_Qcd(Q, nda::range::all, nda::range::all);
        X = X_sk_a_P(ik, nda::range::all, Q);
        XX = nda::conj(XX_sk_b_P(ik, nda::range::all, Q));
        YYQ = nda::blas::outer_product(XX, X);
      }
      // U_ab_cd = Y_Pab * V_PQ * YY_Qcd
      auto YY_Q_cd_2D = nda::reshape(YY_Qcd, std::array<long, 2>{Np, mf.nbnd()*mf.nbnd()});
      auto Y_P_ab_2D = nda::reshape(Y_Pab, std::array<long, 2>{Np, mf.nbnd()*mf.nbnd()});
      nda::array<ComplexType, 2> T_P_cd(Np, mf.nbnd()*mf.nbnd());
      nda::blas::gemm(V_PQ, YY_Q_cd_2D, T_P_cd);
      nda::blas::gemm(nda::transpose(Y_P_ab_2D), T_P_cd, U_thc);
    }
    ARRAY_EQUAL(U_thc, U_chol, eri_cutoff*10);
  }

}
#endif
*/

} // bdft_tests
