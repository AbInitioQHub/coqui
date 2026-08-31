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



#ifndef SPARSE_CSR_MATRIX_SHARED_HPP
#define SPARSE_CSR_MATRIX_SHARED_HPP

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
#include "utilities/tuple_iterator.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_window.hpp"

#include "nda/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "numerics/sparse/ucsr_matrix.h"
#include "numerics/sparse/csr_matrix_view.hpp"

namespace math
{
namespace sparse
{

/*
 * Incompatible with GPU!
 * Subset of capabilities of csr_matrix
 * Uses math::shm::shared_array for storage.
 * Constructors are blocking operations. 
 * Does not provide synchronization, user of class must account for proper synchronizatino mechanisms.
 */ 
template<typename ValType, typename IndxType = int, typename IntType = long, typename comm_t = mpi3::shared_communicator>
class csr_matrix_shared 
{
  using base        = csr_matrix_shared<ValType, IndxType, IntType>; 

  template<typename T>
  using larray = math::shm::shared_array<::nda::array_view<T, 1>>;
  using range  = ::nda::range;

public:
/*
  using base::reserve;
  using base::shape;
  using base::capacity;
  using base::values;
  using base::columns;
  using base::row_begin;
  using base::row_end;
  using base::nnz;
  using base::compact;
  using base::clear;
  using base::serialize;
  using base::deserialize;
  using base::size_of_serialized_in_bytes;
*/

  // to be able to reuse ops_tags 
  using Array_t = memory::array<HOST_MEMORY, ValType, 2>; 
  static constexpr bool is_stride_order_Fortran() noexcept
    { return false; } 
  static constexpr bool is_stride_order_C() noexcept
    { return true; } 

  using value_type  = ValType;
  using element     = ValType;
  using element_ptr = ValType*;
  using index_type  = IndxType;
  using int_type    = IntType;
  static const bool sparse        = true;
  static const int dimensionality = 2;
  static const int rank           = 2;
  static const bool sorted        = true;
  static const MEMORY_SPACE mem_type = HOST_MEMORY;

protected:
  // number of rows/columns
  long size1_ = 0;
  long size2_ = 0;
  // values
  larray<value_type> data_;
  // columns 
  larray<index_type> jdata_;
  // NOTE: row_begin/row_end are kept in host memory!
  // location of first element of each row
  larray<int_type> row_begin_;
  // location of last element of each row
  larray<int_type> row_end_;

public:

  template<typename integer_type = long, typename = std::enable_if_t<std::is_integral_v<integer_type>>>
  csr_matrix_shared(mpi3::shared_communicator* node_comm, 
                    std::tuple<long,long> const& dims, 
                    integer_type nnzpr = 0) 
      : size1_(std::get<0>(dims)),
        size2_(std::get<1>(dims)),
        data_(node_comm, std::array<long, 1>{size1_ * long(nnzpr)}),
        jdata_(node_comm, std::array<long, 1>{size1_ * long(nnzpr)}),
        row_begin_(node_comm, std::array<long, 1>{size1_+1}),
        row_end_(node_comm, std::array<long, 1>{size1_})
  {}
  csr_matrix_shared(mpi3::shared_communicator* node_comm,
                    std::tuple<long, long> const& dims, 
                    ::nda::MemoryArrayOfRank<1> auto && nnzpr)
      : size1_(std::get<0>(dims)),
        size2_(std::get<1>(dims)),
        data_(node_comm, std::array<long, 1>{long(std::accumulate(nnzpr.begin(), nnzpr.end(), 0))}),
        jdata_(node_comm, std::array<long, 1>{data_.size()}),
        row_begin_(node_comm, std::array<long, 1>{size1_+1}),
        row_end_(node_comm, std::array<long, 1>{size1_})
  {}

  ~csr_matrix_shared() = default;

  csr_matrix_shared(csr_matrix_shared const&) = default;
  csr_matrix_shared(csr_matrix_shared&&) = default;
  csr_matrix_shared& operator=(csr_matrix_shared const&) = default;
  csr_matrix_shared& operator=(csr_matrix_shared&&) = default;

/*
  template<typename val_t, MEMORY_SPACE mem_t, typename indx_t, typename int_t,
          typename = std::enable_if_t<not (std::is_same_v<value_type,val_t> and
                                           std::is_same_v<mem_type,mem_t> and
                                           std::is_same_v<index_type,indx_t> and
                                           std::is_same_v<int_type,int_t>) >>
  csr_matrix_shared(csr_matrix_shared<val_t,mem_t,indx_t,int_t> const& other) 
  {
    size1_ = other.shape(0);
    size2_ = other.shape(1);
    data_ = other.values();
    jdata_ = other.columns();
    row_begin_ = other.row_begin();
    row_end_ = other.row_end();
  }

  template<typename val_t, MEMORY_SPACE mem_t, typename indx_t, typename int_t>
  csr_matrix_shared(ucsr_matrix_shared<val_t, mem_t, indx_t, int_t> const& other) : base()
  {
    *this = other;
  }
  template<typename val_t, MEMORY_SPACE mem_t, typename indx_t, typename int_t>
  csr_matrix_shared(ucsr_matrix_shared<val_t, mem_t, indx_t, int_t> && other) : base()
  {
    *this = std::move(other);
  }
 

  template<typename val_t, MEMORY_SPACE mem_t, typename indx_t, typename int_t>
  csr_matrix_shared& operator=(ucsr_matrix_shared<val_t, mem_t, indx_t, int_t> const& other)
  {
    using utils::make_paired_iterator;
    auto shape_ = other.shape();
    size1_      = shape_[0];
    size2_      = shape_[1];
    data_       = other.values(); 
    jdata_      = other.columns(); 
    row_begin_  = other.row_begin(); 
    row_end_    = other.row_end();
    if (size1_ == 0 || data_.size() == 0)
      return *this;

    for (long p = 0; p < size1_; p++)
    {
      auto i1 = row_begin_(p); 
      auto i2 = row_end_(p); 
      if( i1==i2 ) continue;
      if constexpr (mem_type == HOST_MEMORY) {
        std::sort(make_paired_iterator(jdata_.data() + i1, data_.data() + i1),
                  make_paired_iterator(jdata_.data() + i2, data_.data() + i2),
                  [](auto const& a, auto const& b) { return std::get<0>(a) < std::get<0>(b); });
      } else {
        utils::check(false, "finish csr_matrix_shared& operator=(ucsr_matrix_shared) ");
      }
    }
    return *this;
  }

  template<typename val_t, MEMORY_SPACE mem_t, typename indx_t, typename int_t>
  csr_matrix_shared& operator=(ucsr_matrix_shared<val_t, mem_t, indx_t, int_t> && other)
  { 
    using utils::make_paired_iterator;
    auto shape_ = other.shape();
    size1_      = shape_[0];
    size2_      = shape_[1];
    data_       = std::move(other.values());  
    jdata_      = std::move(other.columns());
    row_begin_  = std::move(other.row_begin());
    row_end_    = std::move(other.row_end());
    if (size1_ == 0 || data_.size() == 0)
      return *this;
    
    for (long p = 0; p < size1_; p++)
    { 
      auto i1 = row_begin_(p);
      auto i2 = row_end_(p); 
      if( i1==i2 ) continue; 
      if constexpr (mem_type == HOST_MEMORY) {
        std::sort(make_paired_iterator(jdata_.data() + i1, data_.data() + i1),
                  make_paired_iterator(jdata_.data() + i2, data_.data() + i2),
                  [](auto const& a, auto const& b) { return std::get<0>(a) < std::get<0>(b); });
      } else {
        utils::check(false, "finish csr_matrix_shared& operator=(ucsr_matrix_shared) ");
      }
    }
    return *this;
  }

  template<class Pair = std::array<IndxType, 2>>
  value_type& get(Pair&& indices)
  {
    using std::get;
    assert(get<0>(indices) >= 0);
    assert(get<0>(indices) < size1_);
    if constexpr (mem_type == DEVICE_MEMORY)
      utils::check(false,"Finish csr_matrix_shared::emplace()");

    auto loc   = std::lower_bound(jdata_.data() + row_begin_(get<0>(indices)), 
                                  jdata_.data() + row_end_(get<0>(indices)), get<1>(indices));
    long disp  = std::distance(jdata_.data() + row_begin_(get<0>(indices)), loc);  
    long disp_ = std::distance(loc, jdata_.data() + row_end_(get<0>(indices)));
    
    if ( not (disp_ > 0 and *loc == get<1>(indices)) )
    { 
      utils::check(row_end_(get<0>(indices)) < row_begin_(get<0>(indices) + 1), 
			"row size exceeded the maximum");
      // new value, shift back and add in correct place
      if (disp_ > 0)
      { 
        // check if row is full
        std::move_backward(data_.data() + row_begin_(get<0>(indices)) + disp,
                           data_.data() + row_end_(get<0>(indices)), 
                           data_.data() + row_end_(get<0>(indices)) + 1);
        std::move_backward(jdata_.data() + row_begin_(get<0>(indices)) + disp,
                           jdata_.data() + row_end_(get<0>(indices)), 
                           jdata_.data() + row_end_(get<0>(indices)) + 1);
      }
      ++row_end_(get<0>(indices));
      jdata_(row_begin_(get<0>(indices)) + disp) = index_type(get<1>(indices)); 
      data_(row_begin_(get<0>(indices)) + disp) = value_type(0);
    }
    return data_(row_begin_(get<0>(indices)) + disp);
  }

public:
  template<class Pair = std::array<IndxType, 2>, class... Args>
  void emplace(Pair&& indices, Args&&... args)
  {
    get(indices) = value_type(std::forward<Args>(args)...);
  }

  // new column index must be larger than all previous column indexes in the row
  // otherwise throws
  template<class Pair = std::array<IndxType, 2>, class... Args>
  void emplace_back(Pair&& indices, Args&&... args)
  {
    using std::get;
    assert(get<0>(indices) >= 0);
    assert(get<0>(indices) < size1_);
    if constexpr (mem_type == DEVICE_MEMORY)
      utils::check(false,"Finish csr_matrix_shared::emplace()");

    // check that row is not full
    utils::check(row_end_(get<0>(indices)) < row_begin_(get<0>(indices) + 1), 
			"row size exceeded the maximum");
    // check that emplaced column belongs in the end of the row 
    utils::check(row_begin_(get<0>(indices)) == row_end_(get<0>(indices)) or
                 get<1>(indices) > jdata_(row_end_(get<0>(indices)) - 1),
		 "inconsistent column index in emplace_back");
    data_(row_end_(get<0>(indices))) = value_type(std::forward<Args>(args)...);
    jdata_(row_end_(get<0>(indices))) = index_type(get<1>(indices));
    ++row_end_(get<0>(indices));
  }
  // adaptors to emplace/emplace_back 
  template<typename integer_type = IndxType, typename value_type = ValType>
  void emplace(std::tuple<integer_type, integer_type, value_type> const& val)
  {
    using std::get;
    emplace({get<0>(val), get<1>(val)}, static_cast<ValType>(get<2>(val)));
  }
  template<typename integer_type = IndxType, typename value_type = ValType>
  void emplace_back(std::tuple<integer_type, integer_type, value_type> const& val)
  {
    using std::get;
    emplace_back({get<0>(val), get<1>(val)}, static_cast<ValType>(get<2>(val)));
  }

  template<class Pair = std::array<IndxType, 2>>
  value_type get_value(Pair&& indices)
  { 
    using std::get;
    assert(get<0>(indices) >= 0);
    assert(get<0>(indices) < size1_);
    if constexpr (mem_type == DEVICE_MEMORY)
      utils::check(false,"Finish csr_matrix_shared::emplace()");
    
    auto loc   = std::lower_bound(jdata_.data() + row_begin_(get<0>(indices)), 
                                  jdata_.data() + row_end_(get<0>(indices)), get<1>(indices));
    long disp  = std::distance(jdata_.data() + row_begin_(get<0>(indices)), loc);
    long disp_ = std::distance(loc, jdata_.data() + row_end_(get<0>(indices)));

    if (disp_ > 0 && *loc == get<1>(indices)) 
    { 
      // value exists
      return data_(row_begin_(get<0>(indices)) + disp);
    }
    return value_type(0);
  }

  template<class Pair = std::array<IndxType, 2>, class TT>
  void add(Pair&& indices, TT&& v_)
  { 
    get(indices) += static_cast<value_type>(v_);
  }

  template<class Pair = std::array<IndxType, 2>, class TT>
  void scale(Pair&& indices, TT&& v_)
  { 
    get(indices) *= static_cast<value_type>(v_);
  }

  void remove_empty_spaces()
  {
    if(size1_ == 0) return;
    // MAM: make copies of ranges in GPU and just assign
    if constexpr (mem_type == DEVICE_MEMORY)
      utils::check(false,"Finish csr_matrix_shared::emplace()");
    for (long i = 0; i < size1_ - 1; i++)
    {
      if (row_end_(i) == row_begin_(i + 1))
        continue;
      auto ni = static_cast<long>(row_end_(i + 1) - row_begin_(i + 1));
      std::move(data_.data() + row_begin_(i + 1),
                data_.data() + row_end_(i + 1),
                data_.data() + row_end_(i));
      std::move(jdata_.data() + row_begin_(i + 1),
                jdata_.data() + row_end_(i + 1),
                jdata_.data() + row_end_(i));
      row_begin_(i + 1) = row_end_(i);
      row_end_(i + 1)   = row_begin_(i + 1) + ni;
    }
    row_begin_(base::size1_) = row_end_(base::size1_ - 1);
  }

protected:
  virtual int serialization_code() const { return 22; }
  struct row_reference
  {
    static constexpr int dimensionality = 1;
    static constexpr int rank = 1;
    static constexpr bool sparse        = true;
    static constexpr bool sorted        = true;
    static constexpr MEMORY_SPACE mem_type = csr_matrix_shared::mem_type;
    csr_matrix_shared& self_;
    IndxType i_;
    using value_type   = typename csr_matrix_shared::value_type;  
    using element      = value_type;
    using element_type = value_type;
    using element_ptr  = typename csr_matrix_shared::element_ptr; 
    struct element_reference
    {
      row_reference& self_;
      IndxType j_;
      template<class TT>
      element_reference&& operator=(TT&& tt) &&
      {
        self_.self_.emplace({{self_.i_, j_}}, std::forward<TT>(tt));
        return std::move(*this);
      }
      template<class TT>
      element_reference&& operator+=(TT&& tt) &&
      {
        self_.self_.add({{self_.i_, j_}}, std::forward<TT>(tt));
        return std::move(*this);
      }
      template<class TT>
      element_reference&& operator*=(TT&& tt) &&
      {
        self_.self_.scale({{self_.i_, j_}}, std::forward<TT>(tt));
        return std::move(*this);
      }
      operator value_type() const 
      {
        return self_.self_.get({{self_.i_, j_}});
      }  
    };
    using reference = element_reference;
    template<typename integer_type = long>
    reference operator[](integer_type i) &&
    {
      return reference{*this, static_cast<IndxType>(i)};
    }

    auto values() const { return self_.data_(::nda::range(self_.row_begin_(i_),self_.row_end_(i_))); }
    auto columns() const { return self_.jdata_(::nda::range(self_.row_begin_(i_),self_.row_end_(i_))); }
    auto nnz() const { return self_.nnz(i_); } 
    auto capacity() const { return self_.capacity(i_); }
    auto shape() const { return std::array<long, 1>{{self_.size2_}}; }
    template<typename integer_type = long>
    auto size() const
    {
      return long{self_.size2_};
    }
  };

  struct const_row_reference
  {
    static constexpr int dimensionality = 1;
    static constexpr int rank           = 1;
    static constexpr bool sparse        = true;
    static constexpr bool sorted        = true;
    static constexpr MEMORY_SPACE mem_type = csr_matrix_shared::mem_type;
    const csr_matrix_shared& self_;
    const IndxType i_;
    using value_type   = typename csr_matrix_shared::value_type const;
    using element      = value_type;
    using element_type = value_type;
    using element_ptr  = typename csr_matrix_shared::element_ptr const;

    auto values() const { return self_.data_(::nda::range(self_.row_begin_(i_),self_.row_end_(i_))); }
    auto columns() const { return self_.jdata_(::nda::range(self_.row_begin_(i_),self_.row_end_(i_))); }
    auto nnz() const { return self_.nnz(i_); }
    auto capacity() const { return self_.capacity(i_); }
    auto shape() const { return std::array<long, 1>{{self_.size2_}}; }
    template<typename integer_type = long>
    auto size() const
    {
      return long{self_.size2_};
    }
  };

public:
  using reference       = row_reference;
  using const_reference = const_row_reference;
  template<class integer_type>
  reference operator[](integer_type i)
  {
    return reference{*this, static_cast<IndxType>(i)};
  }

  template<class integer_type = long>
  const_reference operator[](integer_type i) const
  {
    return const_reference{*this, static_cast<IndxType>(i)};
  }

  using matrix_view = csr_matrix_shared_view<ValType, MEM, IndxType, IntType>;
  matrix_view sub_matrix(::nda::range r)
  {
    assert(r.first() >= 0 && r.last() <= size1_);
    assert(r.first() < r.last());
    return matrix_view({r.size(), size2_}, 
              data_(::nda::range(row_begin_(r.first()),row_end_(r.last()))), 
              jdata_(::nda::range(row_begin_(r.first() ),row_end_(r.last()))), 
              row_begin_(::nda::range(r.first(),r.last()+1)),
              row_end_(r) );

  }

  auto operator()(::nda::range r)
  {
    utils::check(r.step() == 1, "csr_matrix_shared::operator(): Only contiguous ranges allowed.");
    return sub_matrix(r);
  }
*/

};

} // namespace sparse
} // namespace math

#endif
