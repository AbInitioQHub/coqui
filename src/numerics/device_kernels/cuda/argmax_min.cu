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


//////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#include <stdexcept>
#include <complex>
#include <thrust/complex.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include "utilities/type_traits.hpp"
#include "numerics/device_kernels/cuda/cuda_aux.hpp"
#include "arch/CUDA/cuda_init.h"

namespace kernels::device
{

struct op_less_real
{
  template<typename T>
  __host__ __device__
  bool operator()(T const& a, T const& b)
  {
    return thrust::get<0>(a).real() < thrust::get<0>(b).real();
  }
};

template<typename T>
std::tuple<long,T> argmax(T const* x, long N)
{
  auto d = thrust::device_pointer_cast(complex_ptr_cast(x)); 
  auto cntIt = thrust::make_counting_iterator(long(0));

  typedef thrust::tuple<decltype(d),decltype(cntIt)> IteratorTuple;
  typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
  ZipIterator it(thrust::make_tuple(d,cntIt));

  long pos;
  T res=0;
  if constexpr (is_complex_v<T>) {
    auto res_d = thrust::max_element(thrust::device, it, it+N, op_less_real());
    pos = thrust::get<1>(*res_d);
    thrust::complex<utils::remove_complex_t<T>> r_ = thrust::get<0>(*res_d);
    res = T{r_.real(),r_.imag()};
  } else {
    auto res_d = thrust::max_element(thrust::device, it, it+N);
    pos = thrust::get<1>(*res_d);
    res = thrust::get<0>(*res_d);
  }

  cuda::cuda_check(cudaGetLastError());
  cuda::cuda_check(cudaDeviceSynchronize());

  return std::tuple<long,T> {pos,res};
}

template<typename T>
std::tuple<long,T> argmin(T const* x, long N)
{
  auto d = thrust::device_pointer_cast(complex_ptr_cast(x));
  auto cntIt = thrust::make_counting_iterator(long(0));

  typedef thrust::tuple<decltype(d),decltype(cntIt)> IteratorTuple;
  typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
  ZipIterator it(thrust::make_tuple(d,cntIt));

  long pos;
  T res=0;
  if constexpr (is_complex_v<T>) {
    auto res_d = thrust::min_element(thrust::device, it, it+N, op_less_real());
    pos = thrust::get<1>(*res_d);
    thrust::complex<utils::remove_complex_t<T>> r_ = thrust::get<0>(*res_d);
    res = T{r_.real(),r_.imag()};
  } else {
    auto res_d = thrust::min_element(thrust::device, it, it+N);
    pos = thrust::get<1>(*res_d);
    res = thrust::get<0>(*res_d);
  }

  cuda::cuda_check(cudaGetLastError());
  cuda::cuda_check(cudaDeviceSynchronize());

  return std::tuple<long,T> {pos,res};
}

#define _argmax_(X) template std::tuple<long,X> argmax(X const*,long);
#define _argmin_(X) template std::tuple<long,X> argmin(X const*,long);

_argmax_(double)
_argmax_(std::complex<double>)

_argmin_(double)
_argmin_(std::complex<double>)

} // namespace kernels::device
