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



#ifndef SPARSE_CSR_MATRIX_VIEW_HPP
#define SPARSE_CSR_MATRIX_VIEW_HPP

#include <array>
#include <cassert>
#include <iostream>
#include <vector>
#include <tuple>
#include <numeric>
#include <memory>
#include <type_traits> 
#include <algorithm>
#include <utility>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"

namespace math
{
namespace sparse
{

template<typename ValType, MEMORY_SPACE MEM = HOST_MEMORY, typename IndxType = int, typename IntType = long>
class csr_matrix_view  
{
  template<typename T> 
  using larray = memory::array_view<MEM, T, 1>;
  using range  = ::nda::range;
  using row_array_t = ::nda::array<IntType,1>;
  
public:
  using value_type  = ValType;
  using element     = ValType;
  using element_ptr = ValType*; 
  using index_type  = IndxType;
  using int_type    = IntType;
  static const bool sparse        = true;
  static const int dimensionality = 2;
  static const int rank           = 2;
  static const bool sorted        = true;
  static const MEMORY_SPACE mem_type = MEM;

  // to be able to reuse ops_tags 
  using Array_t = memory::array<MEM, ValType, 2>;
  static constexpr bool is_stride_order_Fortran() noexcept
    { return false; }
  static constexpr bool is_stride_order_C() noexcept
    { return true; }

protected:
  // number of rows/columns
  long size1_;
  long size2_;
  // values
  larray<value_type> data_;
  // columns 
  larray<index_type> jdata_;
  // NOTE: row_begin/row_end are kept in host memory!
  // location of first element of each row
  row_array_t row_begin_;
  // location of last element of each row
  row_array_t row_end_;

public:

  /* Constructor */
  csr_matrix_view(std::tuple<long, long> const& dims,
                  larray<value_type> && _data_,
                  larray<index_type> && _jdata_,
                  ::nda::array_view<IntType,1> && rb_,
                  ::nda::array_view<IntType,1> && re_
                 )	   
      : size1_(std::get<0>(dims)),
        size2_(std::get<1>(dims)),
        data_(_data_),
        jdata_(_jdata_),
        row_begin_(rb_),
        row_end_(re_)
  {
    utils::check(row_begin_.size() == size1_+1, "Size mismatch");
    utils::check(row_end_.size() == size1_, "Size mismatch");
    auto r0 = row_begin_(0);
    row_begin_ -= r0;
    row_end_ -= r0;
  }

  ~csr_matrix_view() = default; 

  csr_matrix_view(csr_matrix_view const& other) = default;
  csr_matrix_view(csr_matrix_view&&) = default; 
  csr_matrix_view& operator=(csr_matrix_view const&) = default;
  csr_matrix_view& operator=(csr_matrix_view&&) = default;

  // accessor functions
  auto shape() const { return std::array<long,2>{size1_,size2_}; } 
  auto shape(long i) const { return (i==0?size1_:size2_); } 
  auto capacity()  const { return row_begin_(size1_)-row_begin_(0); } 
  auto capacity(long i)  const { return row_begin_(i+1)-row_begin_(i); } 
  auto values() const { return data_(); }
  auto columns() const { return jdata_(); }
  auto row_begin() const { return row_begin_(); }
  auto row_end() const { return row_end_(); }
  bool compact() const {
    for(long r=0; r<size1_; ++r)
      if(row_end_(r) != row_begin_(r+1)) return false;
    return true;
  }
  auto nnz() const {
    long n=0;
    for(long i=0; i<size1_; i++) n += row_end_(i) - row_begin_(i);
    return n;
  }
  auto nnz(long i) const { return row_end_(i) - row_begin_(i); }

};

} // sparse
} // math

#endif
