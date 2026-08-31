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

#include "numerics/sparse/ucsr_matrix.h"

namespace math
{
namespace sparse
{

  template<typename V, MEMORY_SPACE M, typename I1, typename I2>
  ucsr_matrix<V,M,I1,I2>::ucsr_matrix() :
        size1_(0),
        size2_(0),
        data_(0),
        jdata_(0),
        row_begin_(1),
        row_end_(0)
  {
    row_begin_() = 0;
  }

/*
  template<typename V, MEMORY_SPACE M, typename I1, typename I2>
  template<typename integer_type, typename>
  ucsr_matrix<V,M,I1,I2>::ucsr_matrix(std::tuple<long, long> const& dims, integer_type nnzpr)
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

  template<typename V, MEMORY_SPACE M, typename I1, typename I2>
  ucsr_matrix<V,M,I1,I2>::ucsr_matrix(std::tuple<long, long> const& dims, ::nda::MemoryArrayOfRank<1> auto const& nnzpr) 
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

  template<typename V, MEMORY_SPACE M, typename I1, typename I2>
  template<typename val_t, MEMORY_SPACE mem_t, typename indx_t, typename int_t, typename> 
  ucsr_matrix<V,M,I1,I2>::ucsr_matrix(ucsr_matrix<val_t,mem_t,indx_t,int_t> const& other) :
        size1_(other.shape()[0]),
        size2_(other.shape()[1]),
        data_(other.values()),
        jdata_(other.columns()),
        row_begin_(other.row_begin()),
        row_end_(other.row_end())
  {}

  template<typename V, MEMORY_SPACE M, typename I1, typename I2>
  template<typename integer_type, typename> 
  void ucsr_matrix<V,M,I1,I2>::reserve(integer_type nnzpr)
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

  template<typename V, MEMORY_SPACE M, typename I1, typename I2>
  void ucsr_matrix<V,M,I1,I2>::reserve(::nda::MemoryArrayOfRank<1> auto && nnzpr)
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

  // emplace functions
  template<typename V, MEMORY_SPACE M, typename I1, typename I2>
  template<class Pair, typename val_t>
  void ucsr_matrix<V,M,I1,I2>::emplace(Pair const& indices, val_t const& v)
  {
    static_assert(std::tuple_size_v<Pair> == 2, "Pair size mismatch");
    using std::get;
    utils::check(get<0>(indices) >= 0, "Index mismatch");
    utils::check(get<0>(indices) < size1_, "Index mismatch");
    if (row_end_(get<0>(indices)) < row_begin_(get<0>(indices) + 1))
    {
      if constexpr (mem_type == HOST_MEMORY) {
        data_(row_end_(get<0>(indices)))  = value_type(v);
        jdata_(row_end_(get<0>(indices))) = index_type(get<1>(indices));
      } else {
        ::nda::mem::fill_n<to_nda_address_space(mem_type)>(data_.data()+row_end_(get<0>(indices)),1,value_type(v));
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

  template<typename V, MEMORY_SPACE M, typename I1, typename I2>
  template<typename integer_type, typename vtype>
  void ucsr_matrix<V,M,I1,I2>::emplace(std::tuple<integer_type, integer_type, vtype> const& val)
  {
    using std::get;
    emplace({get<0>(val), get<1>(val)}, static_cast<value_type>(get<2>(val)));
  }

  template<typename V, MEMORY_SPACE M, typename I1, typename I2>
  template<class Pair, typename val_t>
  void ucsr_matrix<V,M,I1,I2>::emplace_back(Pair const& indices, val_t const& v)
  {
    static_assert(std::tuple_size_v<Pair> == 2, "Pair size mismatch");
    emplace(std::forward<Pair>(indices), std::forward<val_t>(v));
  }

  template<typename V, MEMORY_SPACE M, typename I1, typename I2>
  template<typename integer_type, typename vtype>
  void ucsr_matrix<V,M,I1,I2>::emplace_back(std::tuple<integer_type, integer_type, vtype> const& val)
  {
    using std::get;
    emplace({get<0>(val), get<1>(val)}, static_cast<value_type>(get<2>(val)));
  }

  // resets to empty state
  template<typename V, MEMORY_SPACE M, typename I1, typename I2>
  void ucsr_matrix<V,M,I1,I2>::clear()
  {
    if(size1_*size2_==0) return;
    row_end_() = row_begin_(::nda::range(size1_));
  }

  // number of bytes needed for serialization
  template<typename V, MEMORY_SPACE M, typename I1, typename I2>
  long ucsr_matrix<V,M,I1,I2>::size_of_serialized_in_bytes(bool compact) {
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
  template<typename V, MEMORY_SPACE M, typename I1, typename I2>
  void ucsr_matrix<V,M,I1,I2>::serialize(char* ptr, long sz, bool compact) 
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

  template<typename V, MEMORY_SPACE M, typename I1, typename I2>
  auto ucsr_matrix<V,M,I1,I2>::serialize(bool compact) 
  {
    long sz = size_of_serialized_in_bytes(compact); 
    ::nda::array<char,1> buff(sz);
    serialize(buff.data(),sz,compact); 
    return buff;
  }

  template<typename V, MEMORY_SPACE M, typename I1, typename I2>
  void ucsr_matrix<V,M,I1,I2>::deserialize(char const* ptr)
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

  template<typename V, MEMORY_SPACE M, typename I1, typename I2>
  void ucsr_matrix<V,M,I1,I2>::deserialize(::nda::ArrayOfRank<1> auto const& buff)
  {
    using v_t = typename ::nda::get_value_t<decltype(buff)>;
    static_assert(std::is_same_v<char,std::decay_t<v_t>>,"Datatype mismatch.");
    char const* ptr = reinterpret_cast<char const*>(buff.data());
    deserialize(ptr);
  }
*/

  // Instantiation
  #define __inst__(V,M,I1,I2) \
  template ucsr_matrix<V,M,I1,I2>::ucsr_matrix();

  __inst__(double,HOST_MEMORY,int,int)
  __inst__(std::complex<double>,HOST_MEMORY,int,int)
  __inst__(double,HOST_MEMORY,long,int)
  __inst__(std::complex<double>,HOST_MEMORY,long,int)
  __inst__(double,HOST_MEMORY,long,long)
  __inst__(std::complex<double>,HOST_MEMORY,long,long)
  __inst__(double,HOST_MEMORY,int,long)
  __inst__(std::complex<double>,HOST_MEMORY,int,long)

#if defined(ENABLE_DEVICE)

  __inst__(double,DEVICE_MEMORY,int,int)
  __inst__(std::complex<double>,DEVICE_MEMORY,int,int)
  __inst__(double,DEVICE_MEMORY,long,int)
  __inst__(std::complex<double>,DEVICE_MEMORY,long,int)
  __inst__(double,DEVICE_MEMORY,long,long)
  __inst__(std::complex<double>,DEVICE_MEMORY,long,long)
  __inst__(double,DEVICE_MEMORY,int,long)
  __inst__(std::complex<double>,DEVICE_MEMORY,int,long)

  __inst__(double,UNIFIED_MEMORY,int,int)
  __inst__(std::complex<double>,UNIFIED_MEMORY,int,int)
  __inst__(double,UNIFIED_MEMORY,long,int)
  __inst__(std::complex<double>,UNIFIED_MEMORY,long,int)
  __inst__(double,UNIFIED_MEMORY,long,long)
  __inst__(std::complex<double>,UNIFIED_MEMORY,long,long)
  __inst__(double,UNIFIED_MEMORY,int,long)
  __inst__(std::complex<double>,UNIFIED_MEMORY,int,long)

#endif

} // sparse
} // math

