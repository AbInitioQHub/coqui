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
#include "nda/tensor.hpp"
#include "utilities/test_common.hpp"
#include "numerics/device_kernels/kernels.h"
#include "numerics/nda_functions.hpp"
#include "utilities/Timer.hpp"

namespace bdft_tests
{

using utils::VALUE_EQUAL;

template<typename T>
void test_argmax()
{
  const long N = 1000;
  const long Ns = 10;
  {
    auto a = utils::make_random<T>(N);
    for( auto& v: a ) v = std::real(v);
    memory::device_array<T,1> a_d(a);
  
    auto [p_h,v_h] = nda::argmax(a);
    auto [p_d,v_d] = nda::argmax(a_d);
    REQUIRE(p_h[0] == p_d[0]);
    REQUIRE(v_h == v_d);
  }

  { 
    auto a = utils::make_random<T>(N,N);
    for( auto& v: a ) v = std::real(v);
    memory::device_array<T,2> a_d(a);
    
    auto [p_h,v_h] = nda::argmax(a);
    auto [p_d,v_d] = nda::argmax(a_d);
    REQUIRE(p_h[0] == p_d[0]);
    REQUIRE(p_h[1] == p_d[1]);
    REQUIRE(v_h == v_d);
  } 

  { 
    auto a = utils::make_random<T>(N,N);
    for( auto& v: a ) v = std::real(v);
    memory::device_array<T,2> a_d(a);
    
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
    memory::device_array<T,3> a_d(a);
    
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
    memory::device_array<T,1> a_d(a);
    
    auto [p_h,v_h] = nda::argmin(a);
    auto [p_d,v_d] = nda::argmin(a_d);
    REQUIRE(p_h[0] == p_d[0]);
    REQUIRE(v_h == v_d);
  }

  { 
    auto a = utils::make_random<T>(N,N);
    for( auto& v: a ) v = std::real(v);
    memory::device_array<T,2> a_d(a);
    
    auto [p_h,v_h] = nda::argmin(a);
    auto [p_d,v_d] = nda::argmin(a_d);
    REQUIRE(p_h[0] == p_d[0]);
    REQUIRE(p_h[1] == p_d[1]);
    REQUIRE(v_h == v_d);
  }

  {
    auto a = utils::make_random<T>(N,N);
    for( auto& v: a ) v = std::real(v);
    memory::device_array<T,2> a_d(a);

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
    memory::device_array<T,3> a_d(a);
    
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
  memory::device_array<T,1> s_d(s);
  memory::device_array<long,1> m_d(m);

  {
    auto A = utils::make_random<T>(N);
    auto B0 = utils::make_random<T>(Ns);
    T sref=0.0;
    for( auto i : itertools::range(Ns) ) 
      sref += B0(i) + s(i)*A(m(i)); 

    nda::array<T,1> B(B0); 
    copy_select(false,m,s,T(1.0),A,T(1.0),B);
    T sh=nda::sum(B);
    VALUE_EQUAL( sref, sh );

    memory::device_array<T,1> A_d(A);
    memory::device_array<T,1> B_d(B0);
    copy_select(false,m_d,s_d,T(1.0),A_d,T(1.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    VALUE_EQUAL( sref, sd );
  }

  {
    auto A = utils::make_random<T>(N);
    auto B0 = utils::make_random<T>(Ns);
    T sref=0.0;
    for( auto i : itertools::range(Ns) ) 
      sref += B0(i) + A(m(i));

    nda::array<T,1> B(B0);
    copy_select(false,m,T(1.0),A,T(1.0),B);
    T sh=nda::sum(B);
    VALUE_EQUAL( sref, sh );

    memory::device_array<T,1> A_d(A);
    memory::device_array<T,1> B_d(B0);
    copy_select(false,m_d,T(1.0),A_d,T(1.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    VALUE_EQUAL( sref, sd );
  }

  {
    auto A = utils::make_random<T>(N,32);
    auto B0 = utils::make_random<T>(Ns,32);
    T sref=0.0;
    for( auto i : itertools::range(Ns) )
      for( auto r : itertools::range(32) ) 
        sref += B0(i,r) + s(i)*A(m(i),r);

    nda::array<T,2> B(B0);
    copy_select(false,0,m,s,T(1.0),A,T(1.0),B);
    T sh=nda::sum(B);
    VALUE_EQUAL( sref, sh );

    memory::device_array<T,2> A_d(A);
    memory::device_array<T,2> B_d(B0);
    copy_select(false,0,m_d,s_d,T(1.0),A_d,T(1.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    VALUE_EQUAL( sref, sd );
  }

  {
    auto A = utils::make_random<T>(32,N);
    auto B0 = utils::make_random<T>(32,Ns);
    T sref=0.0;
    for( auto r : itertools::range(32) ) 
      for( auto i : itertools::range(Ns) ) 
        sref += B0(r,i) + s(i)*A(r,m(i));

    nda::array<T,2> B(B0);
    copy_select(false,1,m,s,T(1.0),A,T(1.0),B);
    T sh=nda::sum(B);
    VALUE_EQUAL( sref, sh );

    memory::device_array<T,2> A_d(A);
    memory::device_array<T,2> B_d(B0);
    copy_select(false,1,m_d,s_d,T(1.0),A_d,T(1.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    VALUE_EQUAL( sref, sd );
  }

  {
    auto A = utils::make_random<T>(N,32);
    auto B0 = utils::make_random<T>(Ns,32);
    T sref=0.0;
    for( auto i : itertools::range(Ns) )
      for( auto r : itertools::range(32) )
        sref += B0(i,r) + A(m(i),r);

    nda::array<T,2> B(B0);
    copy_select(false,0,m,T(1.0),A,T(1.0),B);
    T sh=nda::sum(B);
    VALUE_EQUAL( sref, sh );

    memory::device_array<T,2> A_d(A);
    memory::device_array<T,2> B_d(B0);
    copy_select(false,0,m_d,T(1.0),A_d,T(1.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    VALUE_EQUAL( sref, sd );
  }

  {
    auto A = utils::make_random<T>(32,N);
    auto B0 = utils::make_random<T>(32,Ns);
    T sref=0.0;
    for( auto r : itertools::range(32) )
      for( auto i : itertools::range(Ns) )
        sref += B0(r,i) + A(r,m(i));

    nda::array<T,2> B(B0);
    copy_select(false,1,m,T(1.0),A,T(1.0),B);
    T sh=nda::sum(B);
    VALUE_EQUAL( sref, sh );

    memory::device_array<T,2> A_d(A);
    memory::device_array<T,2> B_d(B0);
    copy_select(false,1,m_d,T(1.0),A_d,T(1.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    VALUE_EQUAL( sref, sd );
  }

  // expand = true now
  {
    auto A = utils::make_random<T>(Ns);
    auto B0 = utils::make_random<T>(N);
    nda::array<T,1> B1(B0);
    for( auto i : itertools::range(Ns) ) 
      B1(m(i)) = B0(m(i)) + s(i)*A(i);
    T sref=nda::sum(B1);

    nda::array<T,1> B(B0);
    copy_select(true,m,s,T(1.0),A,T(1.0),B);
    T sh=nda::sum(B);
    VALUE_EQUAL( sref, sh );

    memory::device_array<T,1> A_d(A);
    memory::device_array<T,1> B_d(B0);
    copy_select(true,m_d,s_d,T(1.0),A_d,T(1.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    VALUE_EQUAL( sref, sd );
  }

  {
    auto A = utils::make_random<T>(Ns);
    auto B0 = utils::make_random<T>(N);
    nda::array<T,1> B1(B0);
    for( auto i : itertools::range(Ns) ) 
      B1(m(i)) = B0(m(i)) + A(i);
    T sref=nda::sum(B1);

    nda::array<T,1> B(B0);
    copy_select(true,m,T(1.0),A,T(1.0),B);
    T sh=nda::sum(B);
    VALUE_EQUAL( sref, sh );

    memory::device_array<T,1> A_d(A);
    memory::device_array<T,1> B_d(B0);
    copy_select(true,m_d,T(1.0),A_d,T(1.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    VALUE_EQUAL( sref, sd );
  }

  {
    auto A = utils::make_random<T>(Ns,32);
    auto B0 = utils::make_random<T>(N,32);
    nda::array<T,2> B1(B0);
    for( auto i : itertools::range(Ns) )
      for( auto r : itertools::range(32) )
        B1(m(i),r) = B0(m(i),r) + s(i)*A(i,r);
    T sref=nda::sum(B1);

    nda::array<T,2> B(B0);
    copy_select(true,0,m,s,T(1.0),A,T(1.0),B);
    T sh=nda::sum(B);
    VALUE_EQUAL( sref, sh );

    memory::device_array<T,2> A_d(A);
    memory::device_array<T,2> B_d(B0);
    copy_select(true,0,m_d,s_d,T(1.0),A_d,T(1.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    VALUE_EQUAL( sref, sd );
  }

  { 
    auto A = utils::make_random<T>(32,Ns);
    auto B0 = utils::make_random<T>(32,N);
    nda::array<T,2> B1(B0);
    for( auto r : itertools::range(32) ) 
      for( auto i : itertools::range(Ns) )
        B1(r,m(i)) = B0(r,m(i)) + s(i)*A(r,i);
    T sref=nda::sum(B1);
    
    nda::array<T,2> B(B0);
    copy_select(true,1,m,s,T(1.0),A,T(1.0),B);
    T sh=nda::sum(B);
    VALUE_EQUAL( sref, sh );
    
    memory::device_array<T,2> A_d(A);
    memory::device_array<T,2> B_d(B0);
    copy_select(true,1,m_d,s_d,T(1.0),A_d,T(1.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    VALUE_EQUAL( sref, sd );
  }

  { 
    auto A = utils::make_random<T>(Ns,32);
    auto B0 = utils::make_random<T>(N,32);
    nda::array<T,2> B1(B0);
    for( auto i : itertools::range(Ns) ) 
      for( auto r : itertools::range(32) )
        B1(m(i),r) = B0(m(i),r) + A(i,r);
    T sref=nda::sum(B1);
    
    nda::array<T,2> B(B0);
    copy_select(true,0,m,T(1.0),A,T(1.0),B);
    T sh=nda::sum(B);
    VALUE_EQUAL( sref, sh );
    
    memory::device_array<T,2> A_d(A);
    memory::device_array<T,2> B_d(B0);
    copy_select(true,0,m_d,T(1.0),A_d,T(1.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    VALUE_EQUAL( sref, sd );
  }

  { 
    auto A = utils::make_random<T>(32,Ns);
    auto B0 = utils::make_random<T>(32,N);
    nda::array<T,2> B1(B0);
    for( auto r : itertools::range(32) ) 
      for( auto i : itertools::range(Ns) )
        B1(r,m(i)) = B0(r,m(i)) + A(r,i);
    T sref=nda::sum(B1);
    
    nda::array<T,2> B(B0);
    copy_select(true,1,m,T(1.0),A,T(1.0),B);
    T sh=nda::sum(B);
    VALUE_EQUAL( sref, sh );
    
    memory::device_array<T,2> A_d(A);
    memory::device_array<T,2> B_d(B0);
    copy_select(true,1,m_d,T(1.0),A_d,T(1.0),B_d);
    B=B_d;
    T sd=nda::sum(B);
    VALUE_EQUAL( sref, sd );
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

}
