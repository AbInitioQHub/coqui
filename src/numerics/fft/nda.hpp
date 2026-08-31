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


#ifndef NUMERICS_FFT_NDA_INTERFACE_HPP
#define NUMERICS_FFT_NDA_INTERFACE_HPP

#include "configuration.hpp"
#include "fft_base.h"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"
#include "nda/nda.hpp"
#include "nda/tensor.hpp"

/*
 * The FFTW plus nda wrapper
 * Note: Only c-ordering for now!
 *
 */

// Add layout check!!!

namespace math{
namespace fft{

namespace impl 
{

template<int rank, typename Int>
std::array<Int, rank> strides_to_embed_dims(::nda::StdArrayOfLong auto const& str, Int n0)
{
  std::array<Int, rank> dims;
  dims[0] = n0; // must be specified, since it is not constrained by strides of parent array
  auto it = str.crbegin();
  utils::check( *it==1, " Stride error, expect c-ordering.");
  auto itr = dims.rbegin();
  // use rank instead of rend, since str may be larger than rank
  for( int i=0; i<rank-1; ++i, ++itr, ++it) *itr = Int((*(it+1))/(*it));
  return dims;
} 
 
}

template<::nda::MemoryArray IOMat>
fftplan_t create_plan(IOMat && A, const unsigned flags = FFT_DEFAULT)
{
  using A_t = typename std::decay_t<IOMat>; 
  static_assert( A_t::layout_t::is_stride_order_C(), "c-ordering mismatch.");
  // keeping this for safety!
  static_assert( ::nda::get_rank<A_t> >= 1 and ::nda::get_rank<A_t> <= 3, "Rank mismatch");  
  auto inembed = impl::strides_to_embed_dims<::nda::get_rank<A_t>, long int>(
							A.indexmap().strides(), A.shape()[0]);
    
  if constexpr (::nda::mem::on_host<IOMat>) {
    if( flags == FFT_DEFAULT) 
      return impl::host::create_plan<::nda::get_rank<A_t>>(A.shape().data(),A.data(), inembed.data(),
    		                          A.data(), inembed.data(), FFT_MEASURE); 
    else
      return impl::host::create_plan<::nda::get_rank<A_t>>(A.shape().data(),A.data(), inembed.data(),
    		                          A.data(), inembed.data(), flags); 
  } else if constexpr (::nda::mem::on_device<IOMat> or ::nda::mem::on_unified<IOMat>) {
    if( flags == FFT_DEFAULT) 
      return impl::dev::create_plan<::nda::get_rank<A_t>>(A.shape().data(),A.data(), inembed.data(),
                                          A.data(), inembed.data(), FFT_MEASURE);
    else
      return impl::dev::create_plan<::nda::get_rank<A_t>>(A.shape().data(),A.data(), inembed.data(),
                                          A.data(), inembed.data(), flags); 
  } else {
    static_assert(::nda::always_false<IOMat>, "Unknown memory space.");
  }
}

template<::nda::MemoryArray IMat, ::nda::MemoryArray OMat>
fftplan_t create_plan(IMat && A, OMat && B, const unsigned flags = FFT_DEFAULT)
{ 
  using A_t = typename std::decay_t<IMat>; 
  using B_t = typename std::decay_t<OMat>; 
  static_assert( A_t::layout_t::is_stride_order_C(), "c-ordering mismatch.");
  static_assert( B_t::layout_t::is_stride_order_C(), "c-ordering mismatch.");
  static_assert( ::nda::get_rank<A_t> == ::nda::get_rank<B_t>, "math::fft::create_plan: Rank mismatch.");
  static_assert( ::nda::get_rank<A_t> >= 1 and ::nda::get_rank<A_t> <= 3, "Rank mismatch");  
  utils::check( A.shape() == B.shape(), "math::fft::create_plan: Shape mismatch.");
  auto inembed = impl::strides_to_embed_dims<::nda::get_rank<A_t>, long int>(
							A.indexmap().strides(), A.shape()[0]);
  auto onembed = impl::strides_to_embed_dims<::nda::get_rank<B_t>, long int>(
							B.indexmap().strides(), B.shape()[0]);

  if constexpr (::nda::mem::on_host<IMat> and ::nda::mem::on_host<OMat>) {
    if( flags == FFT_DEFAULT) 
      return impl::host::create_plan<::nda::get_rank<A_t>>(A.shape().data(),A.data(), inembed.data(),
                                            B.data(), onembed.data(), FFT_MEASURE); 
    else
      return impl::host::create_plan<::nda::get_rank<A_t>>(A.shape().data(),A.data(), inembed.data(),
                                            B.data(), onembed.data(), flags);
  } else if constexpr ( (::nda::mem::on_device<IMat> or ::nda::mem::on_unified<IMat>) and
		        (::nda::mem::on_device<OMat> or ::nda::mem::on_unified<OMat>) ) {
    if( flags == FFT_DEFAULT) 
      return impl::dev::create_plan<::nda::get_rank<A_t>>(A.shape().data(),A.data(), inembed.data(),
                                            B.data(), onembed.data(), FFT_MEASURE);
    else
      return impl::dev::create_plan<::nda::get_rank<A_t>>(A.shape().data(),A.data(), inembed.data(),
                                            B.data(), onembed.data(), flags);
  } else {
    static_assert(::nda::always_false<IMat>, "Unknown memory space.");
  }
}

template<::nda::MemoryArray IOMat>
fftplan_t create_plan_many(IOMat && A, const unsigned flags = FFT_DEFAULT)
{
  using A_t = typename std::decay_t<IOMat>;
  static_assert( A_t::layout_t::is_stride_order_C(), "c-ordering mismatch.");
  static_assert( ::nda::get_rank<A_t> > 1," Rank > 1 required.");
  static_assert( ::nda::get_rank<A_t> >= 2 and ::nda::get_rank<A_t> <= 4, "Rank mismatch");  
  auto inembed = impl::strides_to_embed_dims<::nda::get_rank<A_t>-1, long int>(
							A.indexmap().strides(), A.shape()[1]);

  utils::check( A.shape()[0] > 0 , "math::fft::create_plan_many: howmany=0. "); 
  if constexpr (::nda::mem::on_host<IOMat>) {
    if( flags == FFT_DEFAULT)
      return impl::host::create_plan_many<::nda::get_rank<A_t>-1>(A.shape().data()+1,int(A.shape()[0]),A.data(), 
			  inembed.data(),1,int(A.indexmap().strides()[0]),
                          A.data(), inembed.data(),1,int(A.indexmap().strides()[0]),
                          FFT_MEASURE);
    else
      return impl::host::create_plan_many<::nda::get_rank<A_t>-1>(A.shape().data()+1,int(A.shape()[0]),A.data(), 
  		  	  inembed.data(),1,int(A.indexmap().strides()[0]),
                          A.data(), inembed.data(),1,int(A.indexmap().strides()[0]),
                          flags); 
  } else if constexpr (::nda::mem::on_device<IOMat> or ::nda::mem::on_unified<IOMat>) {
    if( flags == FFT_DEFAULT)
      return impl::dev::create_plan_many<::nda::get_rank<A_t>-1>(A.shape().data()+1,int(A.shape()[0]),A.data(),
                          inembed.data(),1,int(A.indexmap().strides()[0]),
                          A.data(), inembed.data(),1,int(A.indexmap().strides()[0]),
                          FFT_MEASURE);
    else
      return impl::dev::create_plan_many<::nda::get_rank<A_t>-1>(A.shape().data()+1,int(A.shape()[0]),A.data(),
                          inembed.data(),1,int(A.indexmap().strides()[0]),
                          A.data(), inembed.data(),1,int(A.indexmap().strides()[0]),
                          flags);
  } else {
    static_assert(::nda::always_false<IOMat>, "Unknown memory space.");
  }
}

template<::nda::MemoryArray IMat, ::nda::MemoryArray OMat>
fftplan_t create_plan_many(IMat && A, OMat && B, const unsigned flags = FFT_DEFAULT)
{ 
  using A_t = typename std::decay_t<IMat>;
  using B_t = typename std::decay_t<OMat>; 
  static_assert( A_t::layout_t::is_stride_order_C(), "c-ordering mismatch.");
  static_assert( B_t::layout_t::is_stride_order_C(), "c-ordering mismatch.");
  static_assert( ::nda::get_rank<A_t> > 1," Rank > 1 required.");
  static_assert( ::nda::get_rank<A_t> >= 2 and ::nda::get_rank<A_t> <= 4, "Rank mismatch");  
  static_assert( ::nda::get_rank<A_t> == ::nda::get_rank<B_t>, "math::fft::create_plan: Rank mismatch.");
  utils::check( A.shape() == B.shape(), "math::fft::create_plan: Shape mismatch.");
  utils::check( A.shape()[0] > 0 , "math::fft::create_plan_many: howmany=0. "); 
  auto inembed = impl::strides_to_embed_dims<::nda::get_rank<A_t>-1, long int>(
							A.indexmap().strides(), A.shape()[1]);
  auto onembed = impl::strides_to_embed_dims<::nda::get_rank<B_t>-1, long int>(
							B.indexmap().strides(), B.shape()[1]);
  if constexpr (::nda::mem::on_host<IMat> and ::nda::mem::on_host<OMat>) {
    if( flags == FFT_DEFAULT) 
      return impl::host::create_plan_many<::nda::get_rank<A_t>-1>(A.shape().data()+1,int(A.shape()[0]),A.data(),
                          inembed.data(),1,int(A.indexmap().strides()[0]),
                          B.data(), onembed.data(),1,int(B.indexmap().strides()[0]),
                          FFT_MEASURE);
    else                                    
      return impl::host::create_plan_many<::nda::get_rank<A_t>-1>(A.shape().data()+1,int(A.shape()[0]),A.data(),
                          inembed.data(),1,int(A.indexmap().strides()[0]),
                          B.data(), onembed.data(),1,int(B.indexmap().strides()[0]),
                          flags);
  } else if constexpr ( (::nda::mem::on_device<IMat> or ::nda::mem::on_unified<IMat>) and 
  		        (::nda::mem::on_device<OMat> or ::nda::mem::on_unified<OMat>) ) {
    if( flags == FFT_DEFAULT)
      return impl::dev::create_plan_many<::nda::get_rank<A_t>-1>(A.shape().data()+1,int(A.shape()[0]),A.data(),
                          inembed.data(),1,int(A.indexmap().strides()[0]),
                          B.data(), onembed.data(),1,int(B.indexmap().strides()[0]),
                          FFT_MEASURE);
    else
      return impl::dev::create_plan_many<::nda::get_rank<A_t>-1>(A.shape().data()+1,int(A.shape()[0]),A.data(),
                          inembed.data(),1,int(A.indexmap().strides()[0]),
                          B.data(), onembed.data(),1,int(B.indexmap().strides()[0]),
                          flags);
  } else {
    static_assert(::nda::always_false<IMat>, "Unknown memory space.");
  }
}

inline void destroy_plan(fftplan_t& p)
{
  if (p.bend == FFT_BACKEND_FFTW) {
    impl::host::destroy_plan(p);
  } else if (p.bend == FFT_BACKEND_CUFFT or p.bend == FFT_BACKEND_ROCM) { 
    impl::dev::destroy_plan(p);
  } else {
    if(p.fwd != nullptr or p.inv != nullptr)
      utils::check(false, "Unknown FFT backend in destroy_plan.");
  }
}

namespace impl {

// assumes checks have already been done...
template<::nda::MemoryArray IMat>
void normalize(fftplan_t& p, IMat && A)
{
  using A_t = typename std::decay_t<IMat>; 
  using value_type = typename A_t::value_type;
  const int N = (p.howmany < 1 ?0:1);
  value_type x{1.0};
  for(int i=0; i<p.rank; ++i) x *= value_type{double(A.shape()[i+N])};
  x = value_type{1.0}/x;

  ::nda::tensor::scale(x,A);
}

}

template<::nda::MemoryArray IMat>
void fwdfft(fftplan_t& p, IMat && A)
{
  using A_t = typename std::decay_t<IMat>; 
  static_assert( A_t::layout_t::is_stride_order_C(), "c-ordering mismatch.");
  if(p.howmany < 1) {
    utils::check(::nda::get_rank<A_t> == p.rank, "fwdfft: Rank mismatch: rank(A):{}, p.rank:{}, howmany:{}",::nda::get_rank<A_t>,p.rank,p.howmany);
  } else {
    utils::check(::nda::get_rank<A_t> == p.rank+1, "fwdfft: Rank mismatch: rank(A):{}, p.rank:{}",::nda::get_rank<A_t>,p.rank);
    utils::check(p.howmany == A.shape()[0], "fwdfft: Rank mismatch: p.hownamy:{}, A.shape(0):{}",p.howmany, A.shape()[0]);
  }
  if constexpr (::nda::mem::on_host<IMat>) {
    impl::host::fwdfft(p,A.data(),A.data());
    if constexpr (impl::host::__normalize__) impl::normalize(p,A); 
  } else if constexpr (::nda::mem::on_device<IMat> or ::nda::mem::on_unified<IMat>) {
    impl::dev::fwdfft(p,A.data(),A.data());
    if constexpr (impl::dev::__normalize__) impl::normalize(p,A); 
  } else {
    static_assert(::nda::always_false<IMat>, "Unknown memory space.");
  }
}

template<::nda::MemoryArray IMat, ::nda::MemoryArray OMat>
void fwdfft(fftplan_t& p, IMat && A, OMat && B)
{
  using A_t = typename std::decay_t<IMat>; 
  using B_t = typename std::decay_t<OMat>; 
  static_assert( A_t::layout_t::is_stride_order_C(), "c-ordering mismatch.");
  static_assert( B_t::layout_t::is_stride_order_C(), "c-ordering mismatch.");
  static_assert( ::nda::get_rank<A_t> == ::nda::get_rank<B_t>, "fwdfft: Rank mismatch.");
  utils::check(A.shape() == B.shape(), "fwdfft: Wrong shapes.");
  if(p.howmany < 1) {
    utils::check(::nda::get_rank<A_t> == p.rank, "fwdfft: Rank mismatch: rank(A):{}, p.rank:{}, howmany:{}",::nda::get_rank<A_t>,p.rank,p.howmany);
  } else {
    utils::check(::nda::get_rank<A_t> == p.rank+1, "fwdfft: Rank mismatch: rank(A):{}, p.rank:{}",::nda::get_rank<A_t>,p.rank);
    utils::check(p.howmany == A.shape()[0], "fwdfft: Rank mismatch: p.hownamy:{}, A.shape(0):{}",p.howmany, A.shape()[0]);
  }
  if constexpr (::nda::mem::on_host<IMat>) {
    impl::host::fwdfft(p,A.data(),B.data());
    if constexpr (impl::host::__normalize__) impl::normalize(p,B); 
  } else if constexpr ( (::nda::mem::on_device<IMat> or ::nda::mem::on_unified<IMat>) and
			(::nda::mem::on_device<OMat> or ::nda::mem::on_unified<OMat>) ) {
    impl::dev::fwdfft(p,A.data(),B.data());
    if constexpr (impl::dev::__normalize__) impl::normalize(p,B); 
  } else {
    static_assert(::nda::always_false<IMat>, "Unknown memory space.");
  }
}

template<::nda::MemoryArray IMat>
void invfft(fftplan_t& p, IMat && A)
{
  using A_t = typename std::decay_t<IMat>;
  static_assert( A_t::layout_t::is_stride_order_C(), "c-ordering mismatch.");
  if(p.howmany < 1) {
    utils::check(::nda::get_rank<A_t> == p.rank, "invfft: Rank mismatch: rank(A):{}, p.rank:{}, howmany:{}",::nda::get_rank<A_t>,p.rank,p.howmany);
  } else {
    utils::check(::nda::get_rank<A_t> == p.rank+1, "invfft: Rank mismatch: rank(A):{}, p.rank:{}",::nda::get_rank<A_t>,p.rank);
    utils::check(p.howmany == A.shape()[0], "invfft: Rank mismatch: p.hownamy:{}, A.shape(0):{}",p.howmany, A.shape()[0]);
  }
  if constexpr (::nda::mem::on_host<IMat>) {
    impl::host::invfft(p,A.data(),A.data());
  } else if constexpr ( ::nda::mem::on_device<IMat> or ::nda::mem::on_unified<IMat> ) {
    impl::dev::invfft(p,A.data(),A.data());
  } else {
    static_assert(::nda::always_false<IMat>, "Unknown memory space.");
  }
}

template<::nda::MemoryArray IMat, ::nda::MemoryArray OMat>
void invfft(fftplan_t& p, IMat && A, OMat && B)
{
  using A_t = typename std::decay_t<IMat>;
  using B_t = typename std::decay_t<OMat>;
  static_assert( A_t::layout_t::is_stride_order_C(), "c-ordering mismatch.");
  static_assert( B_t::layout_t::is_stride_order_C(), "c-ordering mismatch.");
  static_assert( ::nda::get_rank<A_t> == ::nda::get_rank<B_t>, "invfft: Rank mismatch.");
  utils::check(A.shape() == B.shape(), "Shape mismatch.");
  if(p.howmany < 1) {
    utils::check(::nda::get_rank<A_t> == p.rank, "invfft: Rank mismatch: rank(A):{}, p.rank:{}, howmany:{}",::nda::get_rank<A_t>,p.rank,p.howmany);
  } else {
    utils::check(::nda::get_rank<A_t> == p.rank+1, "invfft: Rank mismatch: rank(A):{}, p.rank:{}",::nda::get_rank<A_t>,p.rank);
    utils::check(p.howmany == A.shape()[0], "invfft: Rank mismatch: p.hownamy:{}, A.shape(0):{}",p.howmany, A.shape()[0]);
  }
  if constexpr (::nda::mem::on_host<IMat>) {
    impl::host::invfft(p,A.data(),B.data());
  } else if constexpr ( ::nda::mem::on_device<IMat> or ::nda::mem::on_unified<IMat> ) {
    impl::dev::invfft(p,A.data(),B.data());
  } else {
    static_assert(::nda::always_false<IMat>, "Unknown memory space.");
  }
}

// plan-less interface
// move to a separate file and add a monostate pattern to store the plans, 
// using a std::array<int,1> to store all relevant information of the plans created so far.
// if a plan is not found, by searching over stored plans using the array info, create and add 
inline void fwdfft(::nda::MemoryArray auto&& A, 
                   const unsigned flags = FFT_ESTIMATE | FFT_PRESERVE_INPUT) 
{
  auto p = create_plan(A,flags);
  fwdfft(p,A);
  destroy_plan(p); 
}

inline void fwdfft(::nda::MemoryArray auto&& A, ::nda::MemoryArray auto&& B, 
		   const unsigned flags = FFT_ESTIMATE | FFT_PRESERVE_INPUT)
{
  auto p = create_plan(A,B,flags);
  fwdfft(p,A,B);
  destroy_plan(p);   
}

inline void invfft(::nda::MemoryArray auto&& A, 
                   const unsigned flags = FFT_ESTIMATE | FFT_PRESERVE_INPUT)
{ 
  auto p = create_plan(A,flags);
  invfft(p,A);
  destroy_plan(p);
}

inline void invfft(::nda::MemoryArray auto&& A, ::nda::MemoryArray auto&& B, 
		   const unsigned flags = FFT_ESTIMATE | FFT_PRESERVE_INPUT)
{ 
  auto p = create_plan(A,B,flags);
  invfft(p,A,B);
  destroy_plan(p);
}

inline void fwdfft_many(::nda::MemoryArray auto&& A, 
		        const unsigned flags = FFT_ESTIMATE | FFT_PRESERVE_INPUT)
{ 
  auto p = create_plan_many(A,flags);
  fwdfft(p,A);
  destroy_plan(p);
}

inline void fwdfft_many(::nda::MemoryArray auto&& A, ::nda::MemoryArray auto&& B, 
		        const unsigned flags = FFT_ESTIMATE | FFT_PRESERVE_INPUT)
{ 
  auto p = create_plan_many(A,B,flags);
  fwdfft(p,A,B);
  destroy_plan(p);
}

inline void invfft_many(::nda::MemoryArray auto&& A,
		        const unsigned flags = FFT_ESTIMATE | FFT_PRESERVE_INPUT)
{

  auto p = create_plan_many(A,flags);
  invfft(p,A);
  destroy_plan(p);
}

inline void invfft_many(::nda::MemoryArray auto&& A, ::nda::MemoryArray auto&& B, 
		   const unsigned flags = FFT_ESTIMATE | FFT_PRESERVE_INPUT)
{
  auto p = create_plan_many(A,B,flags);
  invfft(p,A,B);
  destroy_plan(p);
}

} // fft
} // math

namespace math
{
namespace nda
{

/**
 * @class fft
 * @brief Handler for fast Fourier transform (FFT)
 *
 * This class provides an RAII-type interface for FFT.
 * fix semantics issue with reference counting, e.g. only delete when all references destroyed
 * or use a shared_pointer to the plan with a custom deleter...
 *
 * @tparam _MANY_ - Type parameter for creating single or multiple FFT plans
 */
template<bool _MANY_>
class fft
{
public:

  static constexpr bool many = _MANY_; 

  fft(::nda::MemoryArray auto&& A,
      ::nda::MemoryArray auto&& B,
      const unsigned flags = math::fft::FFT_MEASURE | math::fft::FFT_PRESERVE_INPUT):
    plan{}
  {
    if constexpr (_MANY_) { 
      plan = math::fft::create_plan_many(A,B,flags);
    } else {
      plan = math::fft::create_plan(A,B,flags);
    } 
  }

  fft(::nda::MemoryArray auto&& A, 
      const unsigned flags = math::fft::FFT_MEASURE | math::fft::FFT_PRESERVE_INPUT):
    plan{}
  {
    if constexpr (_MANY_) { 
      plan = math::fft::create_plan_many(A,flags);
    } else {
      plan = math::fft::create_plan(A,flags);
    } 
  }

  ~fft() 
  {
    math::fft::destroy_plan(plan);
  }

  // not sure what to do here, so just delete for now!
  fft( fft const& ) = delete; 
  fft( fft && other ) : 
    plan( other.plan )
  {
    other.plan = math::fft::fftplan_t{};
  } 

  fft& operator=(fft const& ) = delete;
  fft& operator=(fft && other) 
  {
    destroy_plan(plan);
    plan = other.plan;
    other.plan = math::fft::fftplan_t{};  
    return *this;
  } 

  void forward(::nda::MemoryArray auto&& A)
  {
    math::fft::fwdfft(plan,A);
  }

  void backward(::nda::MemoryArray auto&& A)
  {
    math::fft::invfft(plan,A);
  }

  void forward(::nda::MemoryArray auto&& A, ::nda::MemoryArray auto&& B)
  {
    math::fft::fwdfft(plan,A,B);
  }

  void backward(::nda::MemoryArray auto&& A, ::nda::MemoryArray auto&& B)
  {
    math::fft::invfft(plan,A,B);
  }

private:
  
  math::fft::fftplan_t plan;

};

} // nda 
} // math

#endif
