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



#ifndef SPARSE_UCSR_MATRIX_HPP
#define SPARSE_UCSR_MATRIX_HPP

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
#include "nda/mem/fill.hpp"

namespace math
{
namespace sparse
{

template<typename ValType, MEMORY_SPACE MEM = HOST_MEMORY, typename IndxType = int, typename IntType = long>
class ucsr_matrix  
{
  template<typename T> 
  using larray = memory::array<MEM, T, 1>;
  using range  = ::nda::range;
  
public:
  using value_type  = ValType;
  using element     = ValType;
  using element_ptr = ValType*; 
  using index_type  = IndxType;
  using int_type    = IntType;
  static const bool sparse        = true;
  static const int dimensionality = 2;
  static const int rank           = 2;
  static const bool sorted        = false;
  static const MEMORY_SPACE mem_type = MEM;

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
  ::nda::array<int_type, 1> row_begin_;
  // location of last element of each row
  ::nda::array<int_type, 1> row_end_;

public:

  ucsr_matrix() :
        size1_(0),
        size2_(0),
        data_(0),
        jdata_(0),
        row_begin_(1),
        row_end_(0)
  {
    row_begin_() = 0;
  }

  /* Constructor */
  template<typename integer_type = long, typename = std::enable_if_t<std::is_integral_v<integer_type>>>
  ucsr_matrix(std::tuple<long, long> const& dims, integer_type nnzpr = 0)
      : size1_(std::get<0>(dims)),
        size2_(std::get<1>(dims)),
        data_(size1_ * long(nnzpr)),
        jdata_(size1_ * long(nnzpr)),
        row_begin_(size1_+1),
        row_end_(size1_)
  {
    if(size1_ * long(nnzpr) == 0) {
      row_begin_() = 0;
      row_end_()   = 0;
      return;
    }
    for (long i = 0; i < size1_; ++i) {
      row_begin_(i) = int_type(i*nnzpr);
      row_end_(i) = int_type(i*nnzpr);
    }
    row_begin_(size1_) = int_type(size1_ * long(nnzpr));
  }

  ucsr_matrix(std::tuple<long, long> const& dims, ::nda::MemoryArrayOfRank<1> auto const& nnzpr) 
      : size1_(std::get<0>(dims)),
        size2_(std::get<1>(dims)),
        data_(long(std::accumulate(nnzpr.begin(), nnzpr.end(), 0))),
        jdata_(data_.size()),
        row_begin_(size1_+1),
        row_end_(size1_)
  { 
    utils::check(nnzpr.size() == size1_, "Size mismatch");
    if(data_.size() == 0) {
      row_begin_() = 0;
      row_end_()   = 0;
      return;
    }
    int_type cnter=0;
    for (long i = 0; i < size1_; ++i) {
      row_begin_(i) = cnter; 
      row_end_(i) = cnter; 
      cnter += static_cast<int_type>(nnzpr(i));
    }
    row_begin_(size1_) = int_type(cnter);
  }

  virtual ~ucsr_matrix() = default; 

  ucsr_matrix(ucsr_matrix const& other) = default;
  ucsr_matrix(ucsr_matrix&&) = default; 
  ucsr_matrix& operator=(ucsr_matrix const&) = default;
  ucsr_matrix& operator=(ucsr_matrix&&) = default;

  template<typename val_t, MEMORY_SPACE mem_t, typename indx_t, typename int_t,
	  typename = std::enable_if_t<not (std::is_same_v<value_type,val_t> and
	                                   mem_type == mem_t and
                                       std::is_same_v<index_type,indx_t> and
                                       std::is_same_v<int_type,int_t>) >>
  ucsr_matrix(ucsr_matrix<val_t,mem_t,indx_t,int_t> const& other) :
        size1_(other.shape()[0]),
        size2_(other.shape()[1]),
        data_(other.values()),
        jdata_(other.columns()),
        row_begin_(other.row_begin()),
        row_end_(other.row_end())
  {}

  template<typename integer_type, typename = std::enable_if_t<std::is_integral_v<integer_type>>>
  void reserve(integer_type nnzpr)
  {
    if (size1_ == 0)
      return;
    long minN = long(row_begin_(1) - row_begin_(0));
    for (long i = 0; i != size1_; ++i)
      minN = std::min(minN, long(row_begin_(i + 1) - row_begin_(i)));
    if (static_cast<long>(nnzpr) <= minN)
      return;
    ucsr_matrix other(std::make_tuple(size1_, size2_), nnzpr); 
    if (capacity() > 0)
    {
      for (long i = 0; i < size1_; i++)
      {
        other.row_end_(i) = other.row_begin_(i) + (row_end_(i) - row_begin_(i));
        auto rng_old = range(row_begin_(i),row_end_(i));
        auto rng_new = range(other.row_begin_(i),other.row_end_(i));
        other.data_(rng_new) = data_(rng_old);
        other.jdata_(rng_new) = jdata_(rng_old);
      }
    }
    *this = std::move(other);
  }
  void reserve(::nda::MemoryArrayOfRank<1> auto && nnzpr)
  {
    if (size1_ == 0)
      return;
    bool resz = false;
    utils::check(nnzpr.size() >= size1_, "Size mismatch");
    for (long i = 0; i < size1_; i++)
      if (static_cast<int_type>(nnzpr(i)) > row_begin_(i + 1) - row_begin_(i))
      { 
        resz = true;
        break;
      }
    if (not resz)
      return;
    ucsr_matrix other(std::make_tuple(size1_, size2_), nnzpr);
    if (capacity() > 0)
    { 
      for (long i = 0; i < size1_; i++)
      { 
        other.row_end_(i) = other.row_begin_(i) + (row_end_(i) - row_begin_(i));
        auto rng_old = range(row_begin_(i),row_end_(i));
        auto rng_new = range(other.row_begin_(i),other.row_end_(i));
        other.data_(rng_new) = data_(rng_old);
        other.jdata_(rng_new) = jdata_(rng_old);
      }
    }
    *this = std::move(other);
  }

  // accessor functions
  auto shape() const { return std::array<long,2>{size1_,size2_}; } 
  auto shape(long i) const { return (i==0?size1_:size2_); } 
  auto capacity()  const { return row_begin_(size1_)-row_begin_(0); } 
  auto capacity(long i)  const { return row_begin_(i+1)-row_begin_(i); } 
  auto values() const { return data_(); }
  auto columns() const { return jdata_(); }
  auto row_begin() const { return row_begin_(); }
  auto row_end() const { return row_end_(); }
  auto nnz() const {
    long n=0;
    for(long i=0; i<size1_; i++) n += row_end_(i) - row_begin_(i);
    return n;
  }
  auto nnz(long i) const { return row_end_(i) - row_begin_(i); }
  bool compact() const {
    for(long r=0; r<size1_; ++r)
      if(row_end_(r) != row_begin_(r+1)) return false;
    return true;
  }

  // emplace functions
  template<class Pair = std::array<IndxType, 2>, class... Args>
  void emplace(Pair&& indices, Args&&... args)
  {
    using std::get;
    utils::check(get<0>(indices) >= 0, "Index mismatch");
    utils::check(get<0>(indices) < size1_, "Index mismatch");
    if (row_end_(get<0>(indices)) < row_begin_(get<0>(indices) + 1))
    {
      if constexpr (mem_type == HOST_MEMORY) {
        data_(row_end_(get<0>(indices)))  = value_type(std::forward<Args>(args)...);
        jdata_(row_end_(get<0>(indices))) = index_type(get<1>(indices));
      } else {
        ::nda::mem::fill_n<to_nda_address_space(mem_type)>(data_.data()+row_end_(get<0>(indices)),1,value_type(std::forward<Args>(args)...));
        ::nda::mem::fill_n<to_nda_address_space(mem_type)>(jdata_.data()+row_end_(get<0>(indices)),1,index_type(get<1>(indices)));
      }
      ++row_end_(get<0>(indices));
    }
    else
    {
      // MAM: implement dynamic resizing if needed!
      APP_ABORT(" Error - ucsr_matrix: row size exceeded the maximum \n\n");
    }
  }

  template<typename integer_type = IndxType, typename value_type = ValType>
  void emplace(std::tuple<integer_type, integer_type, value_type> const& val)
  {
    using std::get;
    emplace({get<0>(val), get<1>(val)}, static_cast<ValType>(get<2>(val)));
  }
  template<class Pair = std::array<IndxType, 2>, class... Args>
  void emplace_back(Pair&& indices, Args&&... args)
  {
    emplace(std::forward<Pair>(indices), std::forward<Args>(args)...);
  }
  template<typename integer_type = IndxType, typename value_type = ValType>
  void emplace_back(std::tuple<integer_type, integer_type, value_type> const& val)
  {
    using std::get;
    emplace({get<0>(val), get<1>(val)}, static_cast<ValType>(get<2>(val)));
  }
  // resets to empty state
  void clear()
  {
    if(size1_*size2_==0) return;
    row_end_() = row_begin_(::nda::range(size1_));
  }

  // number of bytes needed for serialization
  long size_of_serialized_in_bytes(bool compact) {
    // store sizes of data types and compact/ucsr/csr flag
    long sz=4*sizeof(int);
    // store shape
    sz += 2*sizeof(long);
    // row_begin/end
    sz += (2*size1_+1)*sizeof(int_type);

    if( compact ) {
      // cols and value
      sz += nnz()*(sizeof(index_type) + sizeof(value_type));
    } else {
      // cols and value
      sz += data_.size()*sizeof(value_type);                       
      sz += jdata_.size()*sizeof(index_type);
    }
    return sz;
  }

  // MAM: do I serialize as is, or compacted?
  void serialize(char* ptr, long sz, bool compact=true) 
  {
    auto data_h = memory::to_memory_space<HOST_MEMORY>(data_());
    auto jdata_h = memory::to_memory_space<HOST_MEMORY>(jdata_());
    long sz_needed = size_of_serialized_in_bytes(compact); 
    utils::check(sz >= sz_needed, 
                 "Error in ucsr_matrix::serialize: bytes needed:{}, bytes provided:{}",
                 sz_needed,sz);

    {
      int* iptr = reinterpret_cast<int*>(ptr);
      iptr[0] = serialization_code();
      iptr[1] = sizeof(value_type);
      iptr[2] = sizeof(index_type);
      iptr[3] = sizeof(int_type);
      ptr += 4*sizeof(int);
    }
    {
      long* lptr = reinterpret_cast<long*>(ptr);
      lptr[0] = size1_; 
      lptr[1] = size2_;
      ptr += 2*sizeof(long);
    }
    if(compact) 
    {
      {
        ::nda::array<int_type,1> buff(2*size1_+1);
        long cnt=0;
        for(long r=0; r<size1_; ++r) {
          buff(r) = cnt;
          cnt+=nnz(r);
          buff(size1_+1+r)=cnt; 
        }
        buff(size1_)=cnt;
        int_type* iptr = reinterpret_cast<int_type*>(ptr);
        std::copy_n(buff.data(),2*size1_+1,iptr);
        ptr += (2*size1_+1)*sizeof(int_type);
      }
      {
        value_type* iptr = reinterpret_cast<value_type*>(ptr);
        for(long r=0; r<size1_; ++r) 
          iptr = std::copy_n(data_h.data()+row_begin_(r),nnz(r),iptr);
        ptr += nnz()*sizeof(value_type);
      }
      {
        index_type* iptr = reinterpret_cast<index_type*>(ptr);
        for(long r=0; r<size1_; ++r) 
          iptr = std::copy_n(jdata_h.data()+row_begin_(r),nnz(r),iptr);
        ptr += nnz()*sizeof(index_type);
      }
    } else {
      {
        int_type* iptr = reinterpret_cast<int_type*>(ptr); 
        std::copy_n(row_begin_.data(),size1_+1,iptr);
        std::copy_n(row_end_.data(),size1_,iptr+size1_+1);
        ptr += (2*size1_+1)*sizeof(int_type); 
      }
      {
        value_type* iptr = reinterpret_cast<value_type*>(ptr);
        std::copy_n(data_h.data(),data_h.size(),iptr);
        ptr += (data_h.size())*sizeof(value_type);
      }
      {
        index_type* iptr = reinterpret_cast<index_type*>(ptr);
        std::copy_n(jdata_h.data(),jdata_h.size(),iptr);
        ptr += (jdata_h.size())*sizeof(index_type);
      }
    }
  }

  auto serialize(bool compact=true) 
  {
    long sz = size_of_serialized_in_bytes(compact); 
    ::nda::array<char,1> buff(sz);
    serialize(buff.data(),sz,compact); 
    return buff;
  }

  void deserialize(char const* ptr)
  {
    {
      int const* iptr = reinterpret_cast<int const*>(ptr);
      utils::check( iptr[0] == serialization_code(), "deserialization: Incompatible with ucsr_matrix.");
      utils::check( iptr[1] == sizeof(value_type), "deserialization: Incompatible value_type.");
      utils::check( iptr[2] == sizeof(index_type), "deserialization: Incompatible index_type.");
      utils::check( iptr[3] == sizeof(int_type), "deserialization: Incompatible int_type.");
      ptr += 4*sizeof(int);
    }
    {
      long const* lptr = reinterpret_cast<long const*>(ptr);
      size1_ = lptr[0];
      size2_ = lptr[1];
      ptr += 2*sizeof(long);
    }
    if(row_begin_.size() != size1_+1) row_begin_.resize(size1_+1); 
    if(row_end_.size() != size1_) row_end_.resize(size1_); 
    if(size1_*size2_==0)
    {
      data_.resize(0);
      jdata_.resize(0);
      return;
    }
    {
      int_type const* iptr = reinterpret_cast<int_type const*>(ptr);
      std::copy_n(iptr,size1_+1,row_begin_.data());
      std::copy_n(iptr+size1_+1,size1_,row_end_.data());
      ptr += (2*size1_+1)*sizeof(int_type);
    }
    {
      long cap = (row_begin_(size1_)-row_begin_(0));
      data_.resize(cap); 
      auto data_h = memory::to_memory_space<HOST_MEMORY>(data_());
      value_type const* iptr = reinterpret_cast<value_type const*>(ptr);
      iptr = std::copy_n(iptr,cap,data_h.data());
      data_() = data_h();
      ptr += cap*sizeof(value_type);
    }
    {
      long cap = (row_begin_(size1_)-row_begin_(0));
      jdata_.resize(cap);                                            
      auto jdata_h = memory::to_memory_space<HOST_MEMORY>(jdata_());
      index_type const* iptr = reinterpret_cast<index_type const*>(ptr);
      iptr = std::copy_n(iptr,cap,jdata_h.data());
      jdata_() = jdata_h();
      ptr += cap*sizeof(index_type);
    }

    // now check
    for(long r=0; r<size1_; r++) {
      utils::check(row_end_(r) >= row_begin_(r), "Pointer mismatch");
      utils::check(row_begin_(r+1) >= row_begin_(r), "Pointer mismatch");
      utils::check(row_begin_(r+1) >= row_end_(r), "Pointer mismatch");
    }
  }

  void deserialize(::nda::ArrayOfRank<1> auto const& buff)
  {
    using v_t = typename ::nda::get_value_t<decltype(buff)>;
    static_assert(std::is_same_v<char,std::decay_t<v_t>>,"Datatype mismatch.");
    char const* ptr = reinterpret_cast<char const*>(buff.data());
    deserialize(ptr);
  }

protected:
  virtual int serialization_code() const { return 11; }
  struct row_reference
  {
    ucsr_matrix& self_;
    IndxType i_;
    static const int dimensionality = 2;
    static const int rank           = 2;
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
    };
    using reference = element_reference;
    reference operator[](IndxType i) && { return reference{*this, i}; }
  };

public:
  using reference = row_reference;
  template<typename integer_type = long>
  reference operator[](integer_type i)
  {
    return reference{*this, static_cast<IndxType>(i)};
  }
};

} // sparse
} // math

#endif
