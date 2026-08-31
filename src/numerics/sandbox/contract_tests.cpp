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

#include <complex>

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/tensor.hpp"
#include "utilities/test_common.hpp"
#include "numerics/device_kernels/kernels.h"
#include "numerics/nda_functions.hpp"
#include "utilities/Timer.hpp"

namespace bdft_tests
{

template<typename T>
void ijk_jl_ikl(int N1=10, int N2=10, int N3=10)
{
  utils::TimerManager Timer;

  auto A = nda::array<T,3>::zeros({N1,N2,N3});
  auto B = nda::array<T,2>::zeros({N2,N3});
  auto C = nda::array<T,3>::zeros({N1,N3,N3});

  static_assert(std::is_same_v<T,double> or std::is_same_v<T,std::complex<double>>,"Datatype mismatch.");
  if constexpr (std::is_same_v<T,double>)
    std::cout<<" Testing ijk,jl->ikl - double - N1, N2, N3: " <<N1 <<" " <<N2 <<" " <<N3 <<"\n";
  else if constexpr (std::is_same_v<T,std::complex<double>>)
    std::cout<<" Testing ijk,jl->ikl - dcomplex - N1, N2, N3: " <<N1 <<" " <<N2 <<" " <<N3 <<"\n";

  //warmup
  nda::tensor::contract(A,"ijk",B,"jl",C,"ikl");
  nda::blas::gemm(T(1.0),nda::transpose(A(0,nda::ellipsis{})),B,
		  T(0.0),C(0,nda::ellipsis{}));

  // tblis
  Timer.start("t1");
  nda::tensor::contract(A,"ijk",B,"jl",C,"ikl");
  Timer.stop("t1");
  std::cout<<" tblis: " <<Timer.elapsed("t1") <<std::endl;


  // manual 
  Timer.reset("t1");
  Timer.start("t1");
  for(int i=0; i<N1; ++i) {
    nda::blas::gemm(T(1.0),nda::transpose(A(i,nda::ellipsis{})),B,
		    T(0.0),C(i,nda::ellipsis{}));
  }
  Timer.stop("t1");
  std::cout<<" loop over gemm: " <<Timer.elapsed("t1") <<std::endl;

}

template<typename T>
void ijk_jk_ik(int N1=10, int N2=10, int N3=10)
{
  decltype(nda::range::all) all;
  utils::TimerManager Timer;

  auto A = nda::array<T,3>::zeros({N1,N2,N3});
  auto B = nda::array<T,2>::zeros({N2,N3});
  auto C = nda::array<T,2>::zeros({N1,N3});

  static_assert(std::is_same_v<T,double> or std::is_same_v<T,std::complex<double>>,"Datatype mismatch.");
  if constexpr (std::is_same_v<T,double>)
    std::cout<<" Testing ijk,jl->ikl - double - N1, N2, N3: " <<N1 <<" " <<N2 <<" " <<N3 <<"\n";
  else if constexpr (std::is_same_v<T,std::complex<double>>)
    std::cout<<" Testing ijk,jl->ikl - dcomplex - N1, N2, N3: " <<N1 <<" " <<N2 <<" " <<N3 <<"\n";

  //warmup
  nda::tensor::contract(A,"ijk",B,"jk",C,"ik");
  C(0,0) = nda::blas::dot(A(0,all,0),B(all,0));

  // tblis
  Timer.start("t1");
  nda::tensor::contract(A,"ijk",B,"jk",C,"ik");
  Timer.stop("t1");
  std::cout<<" tblis: " <<Timer.elapsed("t1") <<"\n";


  // manual 
  Timer.reset("t1");
  Timer.start("t1");
  for(int i=0; i<N1; ++i) 
    for(int k=0; k<N3; ++k) 
      C(i,k) = nda::blas::dot(A(i,all,k),B(all,k));
  Timer.stop("t1");
  std::cout<<" loop over dot: " <<Timer.elapsed("t1") <<"\n" 
           <<std::endl;

}

TEST_CASE("tblis", "[nda_functions]")
{
  using dcomplex = std::complex<double>;
  ijk_jl_ikl<double>(100,100,100);
  ijk_jl_ikl<double>(1000,100,100);
  ijk_jl_ikl<double>(100,1024,1024);

  ijk_jl_ikl<dcomplex>(100,100,100);
  ijk_jl_ikl<dcomplex>(1000,100,100);
  ijk_jl_ikl<dcomplex>(10,1024,1024);

  ijk_jk_ik<double>(100,100,100);
  ijk_jk_ik<double>(1000,100,100);
  ijk_jk_ik<double>(100,1024,1024);

  ijk_jk_ik<dcomplex>(100,100,100);
  ijk_jk_ik<dcomplex>(1000,100,100);
  ijk_jk_ik<dcomplex>(10,1024,1024);
}

#if defined(ENABLE_CUDA)
template<typename T>
void test_argmax()
{
  const long N = 1000;
  const long Ns = 10;
  {
    auto a = utils::make_random<T>(N);
    for( auto& v: a ) v = std::real(v);
    nda::cuarray<T,1> a_d(a);
  
    auto [p_h,v_h] = nda::argmax(a);
    auto [p_d,v_d] = nda::argmax(a_d);
    REQUIRE(p_h[0] == p_d[0]);
    REQUIRE(v_h == v_d);
  }

  { 
    auto a = utils::make_random<T>(N,N);
    for( auto& v: a ) v = std::real(v);
    nda::cuarray<T,2> a_d(a);
    
    auto [p_h,v_h] = nda::argmax(a);
    auto [p_d,v_d] = nda::argmax(a_d);
    REQUIRE(p_h[0] == p_d[0]);
    REQUIRE(p_h[1] == p_d[1]);
    REQUIRE(v_h == v_d);
  } 

  { 
    auto a = utils::make_random<T>(N,N);
    for( auto& v: a ) v = std::real(v);
    nda::cuarray<T,2> a_d(a);
    
    auto av = a();
    auto av_d = a_d();
    auto [p_h,v_h] = nda::argmax(av);
    auto [p_d,v_d] = nda::argmax(av_d);
    REQUIRE(p_h[0] == p_d[0]);
    REQUIRE(p_h[1] == p_d[1]);
    REQUIRE(v_h == v_d);
  }

  { 
    auto a = utils::make_random<T>(Ns,Ns,Ns);
    for( auto& v: a ) v = std::real(v);
    nda::cuarray<T,3> a_d(a);
    
    auto [p_h,v_h] = nda::argmax(a);
    auto [p_d,v_d] = nda::argmax(a_d);
    REQUIRE(p_h[0] == p_d[0]);
    REQUIRE(p_h[1] == p_d[1]);
    REQUIRE(p_h[2] == p_d[2]);
    REQUIRE(v_h == v_d);
  }

}

template<typename T>
void test_argmin()
{
  const long N = 1000;
  const long Ns = 10;
  { 
    auto a = utils::make_random<T>(N);
    for( auto& v: a ) v = std::real(v);
    nda::cuarray<T,1> a_d(a);
    
    auto [p_h,v_h] = nda::argmin(a);
    auto [p_d,v_d] = nda::argmin(a_d);
    REQUIRE(p_h[0] == p_d[0]);
    REQUIRE(v_h == v_d);
  }

  { 
    auto a = utils::make_random<T>(N,N);
    for( auto& v: a ) v = std::real(v);
    nda::cuarray<T,2> a_d(a);
    
    auto [p_h,v_h] = nda::argmin(a);
    auto [p_d,v_d] = nda::argmin(a_d);
    REQUIRE(p_h[0] == p_d[0]);
    REQUIRE(p_h[1] == p_d[1]);
    REQUIRE(v_h == v_d);
  }

  {
    auto a = utils::make_random<T>(N,N);
    for( auto& v: a ) v = std::real(v);
    nda::cuarray<T,2> a_d(a);

    auto av = a();
    auto av_d = a_d();
    auto [p_h,v_h] = nda::argmin(av);
    auto [p_d,v_d] = nda::argmin(av_d);
    REQUIRE(p_h[0] == p_d[0]);
    REQUIRE(p_h[1] == p_d[1]);
    REQUIRE(v_h == v_d);
  }
  
  { 
    auto a = utils::make_random<T>(Ns,Ns,Ns);
    for( auto& v: a ) v = std::real(v);
    nda::cuarray<T,3> a_d(a);
    
    auto [p_h,v_h] = nda::argmin(a);
    auto [p_d,v_d] = nda::argmin(a_d);
    REQUIRE(p_h[0] == p_d[0]);
    REQUIRE(p_h[1] == p_d[1]);
    REQUIRE(p_h[2] == p_d[2]);
    REQUIRE(v_h == v_d);
  }

}

template<typename T>
void test_copy_select()
{
  const long N = 10240;
  const long Ns = 1024;
  nda::array<T,1> s(Ns);
  nda::array<long,1> m(Ns);
  for( auto i : itertools::range(Ns) ) { if(i%2==0) s(i)=i; else s(i) = 1; }
  for( auto i : itertools::range(Ns) ) m(i)=7*i+4;
  nda::cuarray<T,1> s_d(s);
  nda::cuarray<long,1> m_d(m);

  {
    auto A = utils::make_random<T>(N);
    T sref=0.0;
    for( auto i : itertools::range(Ns) ) sref += s(i)*A(m(i)); 

    nda::array<T,1> B(Ns,T(0));
    copy_select(false,m,s,A,T(0.0),B);
    T sh=nda::sum(B);
    REQUIRE( sref == sh );

    nda::cuarray<T,1> A_d(A);
    nda::cuarray<T,1> B_d(B);
    B_d()=0.0;
    copy_select(false,m_d,s_d,A_d,T(0.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    REQUIRE( sref == sd );
  }

  {
    auto A = utils::make_random<T>(N);
    T sref=0.0;
    for( auto i : itertools::range(Ns) ) sref += A(m(i));

    nda::array<T,1> B(Ns,T(0));
    copy_select(false,m,A,T(0.0),B);
    T sh=nda::sum(B);
    REQUIRE( sref == sh );

    nda::cuarray<T,1> A_d(A);
    nda::cuarray<T,1> B_d(B);
    B_d()=0.0;
    copy_select(false,m_d,A_d,T(0.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    REQUIRE( sref == sd );
  }

  {
    auto A = utils::make_random<T>(N,32);
    T sref=0.0;
    for( auto i : itertools::range(Ns) )
      for( auto r : itertools::range(32) ) 
        sref += s(i)*A(m(i),r);

    nda::array<T,2> B(Ns,32);B()=T(0);
    copy_select(false,0,m,s,A,T(0.0),B);
    T sh=nda::sum(B);
    REQUIRE( sref == sh );

    nda::cuarray<T,2> A_d(A);
    nda::cuarray<T,2> B_d(B);
    B_d()=0.0;
    copy_select(false,0,m_d,s_d,A_d,T(0.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    REQUIRE( sref == sd );
  }

  {
    auto A = utils::make_random<T>(32,N);
    T sref=0.0;
    for( auto r : itertools::range(32) ) 
      for( auto i : itertools::range(Ns) ) 
        sref += s(i)*A(r,m(i));

    nda::array<T,2> B(32,Ns);B()=T(0);
    copy_select(false,1,m,s,A,T(0.0),B);
    T sh=nda::sum(B);
    REQUIRE( sref == sh );

    nda::cuarray<T,2> A_d(A);
    nda::cuarray<T,2> B_d(B);
    B_d()=0.0;
    copy_select(false,1,m_d,s_d,A_d,T(0.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    REQUIRE( sref == sd );
  }

  {
    auto A = utils::make_random<T>(N,32);
    T sref=0.0;
    for( auto i : itertools::range(Ns) )
      for( auto r : itertools::range(32) )
        sref += A(m(i),r);

    nda::array<T,2> B(Ns,32);B()=T(0);
    copy_select(false,0,m,A,T(0.0),B);
    T sh=nda::sum(B);
    REQUIRE( sref == sh );

    nda::cuarray<T,2> A_d(A);
    nda::cuarray<T,2> B_d(B);
    B_d()=0.0;
    copy_select(false,0,m_d,A_d,T(0.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    REQUIRE( sref == sd );
  }

  {
    auto A = utils::make_random<T>(32,N);
    T sref=0.0;
    for( auto r : itertools::range(32) ) 
      for( auto i : itertools::range(Ns) )
        sref += A(r,m(i));

    nda::array<T,2> B(32,Ns);B()=T(0);
    copy_select(false,1,m,A,T(0.0),B);
    T sh=nda::sum(B);
    REQUIRE( sref == sh );

    nda::cuarray<T,2> A_d(A);
    nda::cuarray<T,2> B_d(B);
    B_d()=0.0;
    copy_select(false,1,m_d,A_d,T(0.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    REQUIRE( sref == sd );
  }

  // expand = true now
  {
    auto A = utils::make_random<T>(Ns);
    T sref=0.0;
    for( auto i : itertools::range(Ns) ) sref += s(i)*A(i);

    nda::array<T,1> B(N,T(0));
    copy_select(true,m,s,A,T(0.0),B);
    T sh=nda::sum(B);
    REQUIRE( sref == sh );

    nda::cuarray<T,1> A_d(A);
    nda::cuarray<T,1> B_d(B);
    B_d()=0.0;
    copy_select(true,m_d,s_d,A_d,T(0.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    REQUIRE( sref == sd );
  }

  {
    auto A = utils::make_random<T>(Ns);
    T sref=0.0;
    for( auto i : itertools::range(Ns) ) sref += A(i);

    nda::array<T,1> B(N,T(0));
    copy_select(true,m,A,T(0.0),B);
    T sh=nda::sum(B);
    REQUIRE( sref == sh );

    nda::cuarray<T,1> A_d(A);
    nda::cuarray<T,1> B_d(B);
    B_d()=0.0;
    copy_select(true,m_d,A_d,T(0.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    REQUIRE( sref == sd );
  }

  {
    auto A = utils::make_random<T>(Ns,32);
    T sref=0.0;
    for( auto i : itertools::range(Ns) )
      for( auto r : itertools::range(32) )
        sref += s(i)*A(i,r);

    nda::array<T,2> B(N,32); B()=T(0);
    copy_select(true,0,m,s,A,T(0.0),B);
    T sh=nda::sum(B);
    REQUIRE( sref == sh );

    nda::cuarray<T,2> A_d(A);
    nda::cuarray<T,2> B_d(B);
    B_d()=0.0;
    copy_select(true,0,m_d,s_d,A_d,T(0.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    REQUIRE( sref == sd );
  }

  { 
    auto A = utils::make_random<T>(32,Ns);
    T sref=0.0; 
    for( auto r : itertools::range(32) ) 
      for( auto i : itertools::range(Ns) )
        sref += s(i)*A(r,i);
    
    nda::array<T,2> B(32,N); B()=T(0);
    copy_select(true,1,m,s,A,T(0.0),B);
    T sh=nda::sum(B);
    REQUIRE( sref == sh );
    
    nda::cuarray<T,2> A_d(A);
    nda::cuarray<T,2> B_d(B);
    B_d()=0.0;
    copy_select(true,1,m_d,s_d,A_d,T(0.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    REQUIRE( sref == sd );
  }

  { 
    auto A = utils::make_random<T>(Ns,32);
    T sref=0.0; 
    for( auto i : itertools::range(Ns) ) 
      for( auto r : itertools::range(32) )
        sref += A(i,r);
    
    nda::array<T,2> B(N,32); B()=T(0);
    copy_select(true,0,m,A,T(0.0),B);
    T sh=nda::sum(B);
    REQUIRE( sref == sh );
    
    nda::cuarray<T,2> A_d(A);
    nda::cuarray<T,2> B_d(B);
    B_d()=0.0;
    copy_select(true,0,m_d,A_d,T(0.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    REQUIRE( sref == sd );
  }

  { 
    auto A = utils::make_random<T>(32,Ns);
    T sref=0.0; 
    for( auto r : itertools::range(32) ) 
      for( auto i : itertools::range(Ns) )
        sref += A(r,i);
    
    nda::array<T,2> B(32,N); B()=T(0);
    copy_select(true,1,m,A,T(0.0),B);
    T sh=nda::sum(B);
    REQUIRE( sref == sh );
    
    nda::cuarray<T,2> A_d(A);
    nda::cuarray<T,2> B_d(B);
    B_d()=0.0;
    copy_select(true,1,m_d,A_d,T(0.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    REQUIRE( sref == sd );
  }
}

TEST_CASE("argmax", "[nda_functions]")
{
  test_argmax<double>();
  test_argmax<std::complex<double>>();
}

TEST_CASE("argmin", "[nda_functions]")
{
  test_argmin<double>();
  test_argmin<std::complex<double>>();
}

TEST_CASE("copy_select", "[nda_functions]")
{
  test_copy_select<double>();
  test_copy_select<std::complex<double>>();
}
#endif

}
