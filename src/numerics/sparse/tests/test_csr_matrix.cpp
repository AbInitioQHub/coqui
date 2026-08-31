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
#include "utilities/test_common.hpp"
#include "utilities/Timer.hpp"

#include "numerics/sparse/sparse.hpp"

namespace bdft_tests
{

TEST_CASE("csr_concepts", "[csr]")
{
  using math::sparse::CSRVector;
  using math::sparse::CSRMatrix;
  using math::sparse::csr_matrix;
  using math::sparse::csr_matrix_view;

  static_assert(CSRVector<csr_matrix<double>::reference> ,"CONCEPT TEST");
  static_assert(CSRMatrix<csr_matrix<double>> ,"CONCEPT TEST");
  static_assert(CSRMatrix<csr_matrix<double>::matrix_view> ,"CONCEPT TEST");
  static_assert(CSRMatrix<csr_matrix_view<double>> ,"CONCEPT TEST");
  static_assert(nda::mem::on_host<csr_matrix<double,HOST_MEMORY>> ,"CONCEPT TEST");
  static_assert(not nda::mem::on_host<csr_matrix<double,DEVICE_MEMORY>> ,"CONCEPT TEST");
  static_assert(not nda::mem::on_host<csr_matrix<double,UNIFIED_MEMORY>> ,"CONCEPT TEST");
  static_assert(not nda::mem::on_device<csr_matrix<double,HOST_MEMORY>> ,"CONCEPT TEST");
  static_assert(nda::mem::on_device<csr_matrix<double,DEVICE_MEMORY>> ,"CONCEPT TEST");
  static_assert(not nda::mem::on_device<csr_matrix<double,UNIFIED_MEMORY>> ,"CONCEPT TEST");
  static_assert(not nda::mem::on_unified<csr_matrix<double,HOST_MEMORY>> ,"CONCEPT TEST");
  static_assert(not nda::mem::on_unified<csr_matrix<double,DEVICE_MEMORY>> ,"CONCEPT TEST");
  static_assert(nda::mem::on_unified<csr_matrix<double,UNIFIED_MEMORY>> ,"CONCEPT TEST");
}

template<typename Type, typename IndxType, typename IntType, MEMORY_SPACE MEM>
void test_csr_matrix()
{
  using ucsr_matrix = math::sparse::ucsr_matrix<Type, MEM, IndxType, IntType>;
  using csr_matrix  = math::sparse::csr_matrix<Type, MEM, IndxType, IntType>;

  auto As = nda::array<Type,2>::zeros({4,4});
  As(0,1) = 10;
  As(0,2) = 9;
  As(2,1) = 3;
  As(3,3) = 1;
  nda::array<IntType,1> nnz      = {2, 0, 1, 1};
  nda::array<IntType,1> nnz_plus = {12, 10, 11, 11};

  auto check = [](auto && A_, auto && SpM) {
    auto vals = nda::to_host(SpM.values());  
    auto cols = nda::to_host(SpM.columns());  
    auto row_begin = nda::to_host(SpM.row_begin());
    auto row_end = nda::to_host(SpM.row_end());
    auto nr = SpM.shape(0);
    auto i0 = row_begin(0);
    for(long r=0; r<nr; r++) 
      for(long i=row_begin(r); i<row_end(r); ++i)
        utils::VALUE_EQUAL(A_(r,cols(i-i0)),vals(i-i0)); 
  };

  {
    ucsr_matrix small({4, 4}, 2);
    small[3][3] = 1;
    small[0][2] = 9;
    small[2][1] = 3;
    small[0][1] = 10;

    REQUIRE(small.nnz() == 4);
    [[maybe_unused]] auto val = small.values();
    [[maybe_unused]] auto col = small.columns();
    check(As,small);

    csr_matrix csr(small);
    REQUIRE(csr.nnz() == 4);
    check(As,csr);
  }

  {
    ucsr_matrix small({4, 4});
    small.reserve(2);
    small[3][3] = 1;
    small[0][2] = 9;
    small[2][1] = 3;
    small[0][1] = 10;

    REQUIRE(small.nnz() == 4);
    check(As,small);
  }

  {
    ucsr_matrix small({4, 4}, nnz);
    small[3][3] = 1;
    small[0][2] = 9;
    small[2][1] = 3;
    small[0][1] = 10;

    REQUIRE(small.nnz() == 4);
    check(As,small);

    csr_matrix csr(small);
    REQUIRE(csr.nnz() == 4);
    check(As,csr);
  }

  {
    ucsr_matrix small({4, 4});
    small.reserve(nnz);
    small[3][3] = 1;
    small[0][2] = 9;
    small[2][1] = 3;
    small[0][1] = 10;

    REQUIRE(small.nnz() == 4);
    check(As,small);
  }

  {
    ucsr_matrix small({4, 4}, nnz_plus);
    small[3][3] = 1;
    small[0][2] = 9;
    small[2][1] = 3;
    small[0][1] = 10;

    REQUIRE(small.nnz() == 4);
    check(As,small);

    csr_matrix csr(small);
    REQUIRE(csr.nnz() == 4);
    check(As,csr);
  }

  { 
    ucsr_matrix small({4, 4});
    small.reserve(nnz_plus);
    small[3][3] = 1;
    small[0][2] = 9;
    small[2][1] = 3;
    small[0][1] = 10;
    
    REQUIRE(small.nnz() == 4);
    check(As,small);
  }

// in gpu, csr_matrix only constructible from ucsr_matrix
  if constexpr (MEM==HOST_MEMORY) 
  {
  {
    csr_matrix small({4, 4}, 2);
    small[0][1] = 10;
    small[0][2] = 9;
    small[2][1] = 3;
    small[3][3] = 1;

    REQUIRE(small.nnz() == 4);
    check(As,small);
  }

  {
    csr_matrix small({4, 4});
    small.reserve(2);
    small[0][1] = 10;
    small[0][2] = 9;
    small[2][1] = 3;
    small[3][3] = 1;

    REQUIRE(small.nnz() == 4);
    check(As,small);
  }

  {
    csr_matrix small({4, 4}, nnz);
    small[0][1] = 10;
    small[0][2] = 9;
    small[2][1] = 3;
    small[3][3] = 1;

    REQUIRE(small.nnz() == 4);
    check(As,small);
  }

  {
    csr_matrix small({4, 4});
    small.reserve(nnz);
    small[0][1] = 10;
    small[0][2] = 9;
    small[2][1] = 3;
    small[3][3] = 1;

    REQUIRE(small.nnz() == 4);
    check(As,small);
  }

  {
    csr_matrix small({4, 4}, nnz_plus);
    small[0][1] = 10;
    small[0][2] = 9;
    small[2][1] = 3;
    small[3][3] = 1;

    REQUIRE(small.nnz() == 4);
    check(As,small);

    small.remove_empty_spaces();
    REQUIRE(small.nnz() == 4);
    REQUIRE(small.row_end()(3)-small.row_begin()(0) == 4);
    check(As,small);
  }

  {
    csr_matrix small({4, 4});
    small.reserve(nnz_plus);
    small[0][1] = 10;
    small[0][2] = 9;
    small[2][1] = 3;
    small[3][3] = 1;

    REQUIRE(small.nnz() == 4);
    check(As,small);
  }

  {
    csr_matrix small({4, 4}, nnz_plus);
    small[0][1] += 10;
    small[0][2] += 9;
    small[2][1] += 3;
    small[3][3] += 1;

    REQUIRE(small.nnz() == 4);
    check(As,small);

    auto B = As;
    B *= 2.0;
    small[0][1] *= 2.0;
    small[0][2] *= 2.0;
    small[2][1] *= 2.0;
    small[3][3] *= 2.0;
    
    REQUIRE(small.nnz() == 4);
    check(B,small);
  }

  {
    csr_matrix small({4, 4}, nnz_plus);
    small[0][1] = 10;
    small[0][2] = 9;
    small[2][1] = 3;
    small[3][3] = 1;

    auto A_ = math::sparse::to_mat(small);
    check(A_,small);
  }

  {
    auto csr_ = math::sparse::to_csr(As);
    check(As,csr_);
  }

  {
    csr_matrix small({4, 4}, nnz_plus);
    small[0][1] = 10;
    small[0][2] = 9;
    small[2][1] = 3;
    small[3][3] = 1;

    csr_matrix Ac(small);
    check(As,Ac);

    csr_matrix Am(std::move(small));
    check(As,Am);

    Am = Ac;
    check(As,Am);

    Am = std::move(Ac);
    check(As,Am);
  }

  {
    csr_matrix small({8, 4}, 2);
    small[2][1] = 10;
    small[2][2] = 9;
    small[4][1] = 3;
    small[5][3] = 1;

    auto A = small(nda::range(2,6));
    check(As,A);

  }
  }

}

TEST_CASE("csr_matrix", "[csr]")
{
  test_csr_matrix<double, int, long, HOST_MEMORY>();
  test_csr_matrix<double, int, int, HOST_MEMORY>();
  test_csr_matrix<std::complex<double>, int, long, HOST_MEMORY>();
  test_csr_matrix<std::complex<double>, int, int, HOST_MEMORY>();
#if defined(ENABLE_DEVICE)
  test_csr_matrix<double, int, long, DEVICE_MEMORY>();
  test_csr_matrix<double, int, int, DEVICE_MEMORY>();
  test_csr_matrix<std::complex<double>, int, long, DEVICE_MEMORY>();
  test_csr_matrix<std::complex<double>, int, int, DEVICE_MEMORY>();
  test_csr_matrix<double, int, long, UNIFIED_MEMORY>();
  test_csr_matrix<double, int, int, UNIFIED_MEMORY>();
  test_csr_matrix<std::complex<double>, int, long, UNIFIED_MEMORY>();
  test_csr_matrix<std::complex<double>, int, int, UNIFIED_MEMORY>();
#endif
}

} // namespace bdft 
