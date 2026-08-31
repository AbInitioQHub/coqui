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

#include "nda/nda.hpp"
#include "arch/arch.h"
#include "utilities/test_common.hpp"
#include "utilities/Timer.hpp"

#include "numerics/sparse/sparse.hpp"

namespace bdft_tests
{

template<typename Type, typename IndxType, typename IntType, MEMORY_SPACE MEM>
void test_csr_blas()
{
  using math::sparse::to_csr;
  using math::sparse::T;
  using math::sparse::H;
  decltype(nda::range::all) all;
  long m = 22;
  long n = 9;
  long k = 17;

  nda::array<Type,2> Ah = utils::make_random<Type>(m,k);
  nda::array<Type,2> Bh = utils::make_random<Type>(k,n);
  nda::array<Type,2> B2h = utils::make_random<Type>(m,n);
  nda::array<Type,1> xh = utils::make_random<Type>(k);
  nda::array<Type,1> yh = utils::make_random<Type>(m);

  nda::array<Type,2> AB(m,n);  
  nda::array<Type,2> AtB2(k,n);
  nda::array<Type,2> AhB2(k,n);
  nda::array<Type,1> Ax(m);
  nda::array<Type,1> Aty(k);

  nda::blas::gemm(Type(1.0),Ah,Bh,Type(0.0),AB); 
  nda::blas::gemm(Type(1.0),nda::transpose(Ah),B2h,Type(0.0),AtB2); 
  nda::blas::gemm(Type(1.0),nda::dagger(Ah),B2h,Type(0.0),AhB2); 
  nda::blas::gemv(Type(1.0),Ah,xh,Type(0.0),Ax); 
  nda::blas::gemv(Type(1.0),nda::transpose(Ah),yh,Type(0.0),Aty); 

  // device
  memory::array<MEM,Type,2> Ad(Ah);
  memory::array<MEM,Type,2> Bd(Bh);
  memory::array<MEM,Type,2> B2d(B2h);
  memory::array<MEM,Type,1> xd(xh);
  memory::array<MEM,Type,1> yd(yh); 

  memory::array<MEM,Type,2> ABd(m,n);
  memory::array<MEM,Type,2> AtB2d(k,n);
  memory::array<MEM,Type,2> AhB2d(k,n);
  memory::array<MEM,Type,1> Axd(m);
  memory::array<MEM,Type,1> Atyd(k);

  // test A*B with A csr
  {
    auto a = to_csr<MEM,IndxType,IntType>(Ah,0.0);
    math::sparse::csrmv(Type(1.0),a,xd,Type(0.0),Axd); 
    math::sparse::csrmv(Type(1.0),T(a),yd,Type(0.0),Atyd); 
    math::sparse::csrmm(Type(1.0),a,Bd,Type(0.0),ABd); 
    math::sparse::csrmm(Type(1.0),T(a),B2d,Type(0.0),AtB2d);  
    math::sparse::csrmm(Type(1.0),H(a),B2d,Type(0.0),AhB2d); 

    utils::ARRAY_EQUAL(Ax,nda::to_host(Axd));
    utils::ARRAY_EQUAL(Aty,nda::to_host(Atyd));
    utils::ARRAY_EQUAL(AB,nda::to_host(ABd));
    utils::ARRAY_EQUAL(AtB2,nda::to_host(AtB2d));
    utils::ARRAY_EQUAL(AhB2,nda::to_host(AhB2d));
  }

  { 
    auto Ah_r = Ah(nda::range(5,15),all);
    auto B2h_r = B2h(nda::range(5,15),all);
    nda::blas::gemv(Type(1.0),nda::transpose(Ah_r),yh(nda::range(5,15)),
                    Type(0.0),Aty); 
    nda::blas::gemm(Type(1.0),nda::transpose(Ah_r),B2h_r,Type(0.0),AtB2); 
    nda::blas::gemm(Type(1.0),nda::dagger(Ah_r),B2h_r,Type(0.0),AhB2); 

    auto a_full = to_csr<MEM,IndxType,IntType>(Ah,0.0);
    auto a = a_full(nda::range(5,15));
    auto B2d_r = B2d(nda::range(5,15),all);
    math::sparse::csrmv(Type(1.0),a,xd,Type(0.0),Axd(nda::range(5,15))); 
    math::sparse::csrmv(Type(1.0),T(a),yd(nda::range(5,15)),Type(0.0),Atyd);
    math::sparse::csrmm(Type(1.0),a,Bd,Type(0.0),ABd(nda::range(5,15),all)); 
    math::sparse::csrmm(Type(1.0),T(a),B2d_r,Type(0.0),AtB2d);
    math::sparse::csrmm(Type(1.0),H(a),B2d_r,Type(0.0),AhB2d);

    utils::ARRAY_EQUAL(Ax(nda::range(5,15)),nda::to_host(Axd(nda::range(5,15))));
    utils::ARRAY_EQUAL(Aty,nda::to_host(Atyd));
    utils::ARRAY_EQUAL(AB(nda::range(5,15),all),nda::to_host(ABd(nda::range(5,15),all)));
    utils::ARRAY_EQUAL(AtB2,nda::to_host(AtB2d));
    utils::ARRAY_EQUAL(AhB2,nda::to_host(AhB2d));
  }

  // now test A*B with B csr using B^T * T(A)  
  {    
    auto b = to_csr<MEM,IndxType,IntType>(Bh,0.0); 
    math::sparse::csrmm(Type(1.0),T(b),nda::transpose(Ad),Type(0.0),nda::transpose(ABd));
    utils::ARRAY_EQUAL(AB,nda::to_host(ABd));

    math::sparse::csrmm(Type(1.0),Ad,b,Type(0.0),ABd);
    utils::ARRAY_EQUAL(AB,nda::to_host(ABd));

    auto b_r = b(nda::range(5,15));
    math::sparse::csrmm(Type(1.0),Ad(all,nda::range(5,15)),b_r,Type(0.0),ABd);
    nda::blas::gemm(Type(1.0),Ah(all,nda::range(5,15)),Bh(nda::range(5,15),all),Type(0.0),AB); 
    utils::ARRAY_EQUAL(AB,nda::to_host(ABd));
  }

}

TEST_CASE("csr_blas", "[csr]")
{
  test_csr_blas<double, long, long, HOST_MEMORY>();
  test_csr_blas<double, int, int, HOST_MEMORY>();
  test_csr_blas<std::complex<double>, long, long, HOST_MEMORY>();
  test_csr_blas<std::complex<double>, int, int, HOST_MEMORY>();
#if defined(ENABLE_DEVICE)
  test_csr_blas<double, long, long, DEVICE_MEMORY>();
  test_csr_blas<double, int, int, DEVICE_MEMORY>();
  test_csr_blas<std::complex<double>, long, long, DEVICE_MEMORY>();
  test_csr_blas<std::complex<double>, int, int, DEVICE_MEMORY>();
  test_csr_blas<double, long, long, UNIFIED_MEMORY>();
  test_csr_blas<double, int, int, UNIFIED_MEMORY>();
  test_csr_blas<std::complex<double>, long, long, UNIFIED_MEMORY>();
  test_csr_blas<std::complex<double>, int, int, UNIFIED_MEMORY>();
#endif
}

} // namespace bdft 
