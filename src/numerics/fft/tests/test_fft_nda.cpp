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
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "numerics/fft/nda.hpp"
#include "utilities/test_common.hpp"

namespace bdft_tests
{

template <int Rank> using shape_t = std::array<long, Rank>;

template<typename T, MEMORY_SPACE MEM>
void basic_plan_c()
{
  using namespace math::fft; 
  const int N = 8;

  { // 1D tests
    nda::array<T, 1> y(N), y1(N);
    memory::array<MEM, T, 1> y0(N);
    {
      nda::array<T, 1> yt(N);
      utils::fillRandomArray(yt);
      y0=yt;
    }

    {
      memory::array<MEM, T, 1> yt(N);
      auto p = create_plan(yt);
      yt=y0;
      fwdfft(p,yt);
      y1=yt; 
      invfft(p,yt);
      y=yt; 
      destroy_plan(p);
    }
    { 
      memory::array<MEM, T, 1> A(N+3);
      auto Av = A(nda::range(0,N));
      auto p = create_plan(Av);
      Av = y0; 
      fwdfft(p,Av);
      REQUIRE(y1==nda::to_host(Av));
      invfft(p,Av);
      REQUIRE(y==nda::to_host(Av));
      destroy_plan(p);
    }
    {
      memory::array<MEM, T, 1> A(N), B(N);
      auto p = create_plan(A,B);
      A = y0;
      fwdfft(p,A,B);
      REQUIRE(y1==nda::to_host(B));
      invfft(p,B,A);
      REQUIRE(y==nda::to_host(A));
      destroy_plan(p);
    }
    {
      memory::array<MEM, T, 2> A(3,N);
      auto p = create_plan_many(A);
      for( int i=0; i<3; i++ ) A(i,nda::range::all) = y0;
      fwdfft(p,A);
      for( int i=0; i<3; i++  ) REQUIRE(y1==nda::to_host(A(i,nda::range::all)));
      invfft(p,A);
      for( int i=0; i<3; i++  ) REQUIRE(y==nda::to_host(A(i,nda::range::all)));
      destroy_plan(p);
    }
    {
      memory::array<MEM, T, 2> A(3,N), B(3,N);
      auto p = create_plan_many(A,B);
      for( int i=0; i<3; i++ ) A(i,nda::range::all) = y0;
      fwdfft(p,A,B);
      for( int i=0; i<3; i++  ) REQUIRE(y1==nda::to_host(B(i,nda::range::all)));
      invfft(p,B,A);
      for( int i=0; i<3; i++  ) REQUIRE(y==nda::to_host(A(i,nda::range::all)));
      destroy_plan(p);
    }
  } // 1D tests
  { // 3D tests
    nda::array<T, 3> y(N,N,N),y1(N,N,N);
    memory::array<MEM, T, 3> y0(N,N,N);
    {
      nda::array<T, 3> yt(N,N,N);
      utils::fillRandomArray(yt, -1.0, 1.0);
      y0=yt;
    }
    { 
      memory::array<MEM, T, 3> yt(N,N,N);
      auto p = create_plan(yt);
      yt=y0;
      fwdfft(p,yt);
      y1=yt;
      invfft(p,yt);
      y=yt;
      destroy_plan(p);
    } 
    {
      memory::array<MEM, T, 3> A(N,N,N);
      auto p = create_plan(A);
      A=y0;
      fwdfft(p,A);
      REQUIRE(y1==nda::to_host(A));
      invfft(p,A);
      REQUIRE(y==nda::to_host(A));
      destroy_plan(p);
    }
    { 
      memory::array<MEM, T, 3> A(N,N,N), B(N,N,N);
      auto p = create_plan(A,B);
      A=y0;
      fwdfft(p,A,B);
      REQUIRE(y1==nda::to_host(B));
      invfft(p,B,A);
      REQUIRE(y==nda::to_host(A));
      destroy_plan(p);
    } 
    {
      memory::array<MEM, T, 1> A(N*N*N);
      auto A_v = nda::reshape(A, shape_t<3>{N,N,N});
      auto p = create_plan(A_v);
      A_v=y0;	
      fwdfft(p,A_v);
      REQUIRE(y1==nda::to_host(A_v));
      invfft(p,A_v);
      REQUIRE(y==nda::to_host(A_v));
      destroy_plan(p);
    }
    {
      memory::array<MEM, T, 1> A(N*N*N), B(N*N*N);
      auto A_v = nda::reshape(A, shape_t<3>{N,N,N});
      auto B_v = nda::reshape(B, shape_t<3>{N,N,N});
      auto p = create_plan(A_v,B_v); 
      A_v=y0;	
      fwdfft(p,A_v,B_v);
      REQUIRE(y1==nda::to_host(B_v));
      invfft(p,B_v,A_v);
      REQUIRE(y==nda::to_host(A_v));
      destroy_plan(p);
    }
    if constexpr(MEM == HOST_MEMORY)  // no non-contiguous copies for rank>2 in device
    { 
      memory::array<MEM, T, 3> A(N+1,N+2,N+3);
      auto r = nda::range(0,N);
      auto A_v = A(r,r,r);
      auto p = create_plan(A_v);
      A_v=y0;
      fwdfft(p,A_v);
      REQUIRE(y1==nda::to_host(A_v));
      invfft(p,A_v);
      REQUIRE(y==nda::to_host(A_v));
      destroy_plan(p);
    } 
    if constexpr(MEM == HOST_MEMORY)  // no non-contiguous copies for rank>2 in device
    {
      memory::array<MEM, T, 3> A(N+1,N+2,N+3), B(N+3,N+1,N+2);
      auto r = nda::range(0,N);
      auto A_v = A(r,r,r);
      auto B_v = B(r,r,r);
      auto p = create_plan(A_v,B_v);
      A_v=y0;
      fwdfft(p,A_v,B_v);
      REQUIRE(y1==nda::to_host(B_v));
      A_v=B_v;
      invfft(p,A_v,B_v);
      REQUIRE(y==nda::to_host(B_v));
      destroy_plan(p);
    }
    { 
      memory::array<MEM, T, 4> A(3,N,N,N); 
      auto p = create_plan_many(A);
      for( int i=0; i<3; i++ ) A(i,nda::ellipsis{}) = y0;
      fwdfft(p,A);
      for( int i=0; i<3; i++  ) REQUIRE(y1==nda::to_host(A(i,nda::ellipsis{})));
      invfft(p,A);
      for( int i=0; i<3; i++  ) REQUIRE(y==nda::to_host(A(i,nda::ellipsis{})));
      destroy_plan(p);
    }
    { 
      memory::array<MEM, T, 4> A(3,N,N,N),B(3,N,N,N);
      auto p = create_plan_many(A,B);
      for( int i=0; i<3; i++ ) A(i,nda::ellipsis{}) = y0;
      fwdfft(p,A,B);
      for( int i=0; i<3; i++  ) REQUIRE(y1==nda::to_host(B(i,nda::ellipsis{})));
      invfft(p,B,A);
      for( int i=0; i<3; i++  ) REQUIRE(y==nda::to_host(A(i,nda::ellipsis{})));
      destroy_plan(p);
    }    
    if constexpr(MEM == HOST_MEMORY)  // no non-contiguous copies for rank>2 in device
    { 
      memory::array<MEM, T, 4> A(3,N+1,N+2,N+3);
      auto r = nda::range(0,N);
      auto A_v = A(nda::range::all,r,r,r);       
      auto p = create_plan_many(A_v);
      for( int i=0; i<3; i++ ) A_v(i,nda::ellipsis{}) = y0;
      fwdfft(p,A_v);
      for( int i=0; i<3; i++  ) REQUIRE(y1==nda::to_host(A_v(i,nda::ellipsis{})));
      invfft(p,A_v);
      for( int i=0; i<3; i++  ) REQUIRE(y==nda::to_host(A_v(i,nda::ellipsis{})));
      destroy_plan(p);
    }
    if constexpr(MEM == HOST_MEMORY)  // no non-contiguous copies for rank>2 in device
    {
      memory::array<MEM, T, 4> A(3,N+1,N+2,N+3),B(3,N+1,N+2,N+3);
      auto r = nda::range(0,N);
      auto A_v = A(nda::range::all,r,r,r);
      auto B_v = B(nda::range::all,r,r,r);
      auto p = create_plan_many(A_v,B_v);
      for( int i=0; i<3; i++ ) A_v(i,nda::ellipsis{}) = y0;
      fwdfft(p,A_v,B_v);
      for( int i=0; i<3; i++  ) REQUIRE(y1==nda::to_host(B_v(i,nda::ellipsis{})));
      invfft(p,B_v,A_v);
      for( int i=0; i<3; i++  ) REQUIRE(y==nda::to_host(A_v(i,nda::ellipsis{})));
      destroy_plan(p);
    }
    {
      memory::array<MEM, T, 3> A(y0);
      fwdfft(A);
      REQUIRE(y1==nda::to_host(A));
      invfft(A);
      REQUIRE(y==nda::to_host(A));
    }
    { 
      memory::array<MEM, T, 3> A(y0), B(N,N,N);
      fwdfft(A,B);
      REQUIRE(y1==nda::to_host(B));
      invfft(B,A);
      REQUIRE(y==nda::to_host(A));
    }
    if constexpr(MEM == HOST_MEMORY)  // no non-contiguous copies for rank>2 in device
    {
      memory::array<MEM, T, 4> A(3,N+1,N+2,N+3);
      auto r = nda::range(0,N);
      auto A_v = A(nda::range::all,r,r,r);
      for( int i=0; i<3; i++ ) A_v(i,nda::ellipsis{}) = y0;
      fwdfft_many(A_v);
      for( int i=0; i<3; i++  ) REQUIRE(y1==nda::to_host(A_v(i,nda::ellipsis{})));
      invfft_many(A_v);
      for( int i=0; i<3; i++  ) REQUIRE(y==nda::to_host(A_v(i,nda::ellipsis{})));
    }
    if constexpr(MEM == HOST_MEMORY)  // no non-contiguous copies for rank>2 in device
    {
      memory::array<MEM, T, 4> A(3,N+1,N+2,N+3),B(3,N+2,N+3,N+1);
      auto r = nda::range(0,N);
      auto A_v = A(nda::range::all,r,r,r);
      auto B_v = B(nda::range::all,r,r,r);
      for( int i=0; i<3; i++ ) A_v(i,nda::ellipsis{}) = y0;
      fwdfft_many(A_v,B_v);
      for( int i=0; i<3; i++  ) REQUIRE(y1==nda::to_host(B_v(i,nda::ellipsis{})));
      invfft_many(B_v,A_v);
      for( int i=0; i<3; i++  ) REQUIRE(y==nda::to_host(A_v(i,nda::ellipsis{})));
    }
  } // 3D tests
}

TEST_CASE("fft_nda", "[fft]")
{
 basic_plan_c<ComplexType,HOST_MEMORY>();
#if defined(ENABLE_DEVICE)
 basic_plan_c<ComplexType,DEVICE_MEMORY>();
#endif
}

} // bdft_tests
