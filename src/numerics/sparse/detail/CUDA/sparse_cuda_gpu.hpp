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


#ifndef SPARSE_CUDA_GPU_HPP
#define SPARSE_CUDA_GPU_HPP

#include <type_traits>
#include <cassert>
#include <vector>
#include <complex>
#include <cuda_runtime.h>
#include "cusparse.h"

#include "Memory/CUDA/cuda_utilities.h"
#include "Memory/custom_pointers.hpp"
#include "Memory/buffer_managers.h"
#include "Numerics/detail/CUDA/cublas_wrapper.hpp"

#include "nda/nda.hpp"

namespace math::device
{

template<typename T, typename I1, typename I2>
void csrmv(const char transa,
           const int M,
           const int K,
           const T alpha,
           const char* matdescra,
           T const* A,
           I1 const* indx,
           I2 const pntrb,
           I2 const pntre,
           T const* x,
           T const beta,
           T* y)
{
  using qmc_cuda::cusparse_check;
  using index_type = typename std::decay_t<I2>;
  static_assert(std::is_same<typename std::decay<Q1>::type, T>::value, "Wrong dispatch.\n");
  static_assert(std::is_same<typename std::decay<Q2>::type, T>::value, "Wrong dispatch.\n");
  // somehow need to check if the matrix is compact!
  index_type pb, pe;
  arch::memcopy(std::addressof(pb), raw_pointer_cast(pntrb), sizeof(index_type), arch::memcopyD2H, "sparse_cuda_gpu_ptr::csrmv");
  arch::memcopy(std::addressof(pe), raw_pointer_cast(pntre + (M - 1)), sizeof(index_type), arch::memcopyD2H,
                "sparse_cuda_gpu_ptr::csrmv");
  int nnz = int(pe - pb);
  if(pb != 0)
    throw std::runtime_error("Error: Found csr matrix with index base != 0. \n"); 

  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecx, vecy;
  size_t bufferSize = 0;

  cusparse_check(cusparseCreateCsr(&matA, M, K, nnz, (void*)raw_pointer_cast(pntrb), 
                (void*)(raw_pointer_cast(indx)),
                (void*)(raw_pointer_cast(A)), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, qmc_cuda::cusparse_data_type<T>()),
                 "csrmv::cusparseCreateCsr");
  cusparse_check(cusparseCreateDnVec(&vecx, (transa == 'N' ? K : M), (void*)raw_pointer_cast(x),
                                     qmc_cuda::cusparse_data_type<T>()),
                 "csrmv::cusparseCreateDnMat");
  cusparse_check(cusparseCreateDnVec(&vecy, (transa == 'N' ? M : K), (void*)raw_pointer_cast(y),
                                     qmc_cuda::cusparse_data_type<T>()),
                 "csrmv::cusparseCreateDnMat");
  cusparse_check(cusparseSpMV_bufferSize(arch::global_cusparse_handle, qmc_cuda::cusparseOperation(transa), &alpha, matA,
                                         vecx, &beta, vecy, qmc_cuda::cusparse_data_type<T>(), CUSPARSE_CSRMV_ALG1,
//                                         vecx, &beta, vecy, qmc_cuda::cusparse_data_type<T>(), CUSPARSE_SPMV_CSR_ALG1,
                                         &bufferSize),
                 "csrmv::cusparseSpMV_bufferSize");

  if (bufferSize > 0)
  {
    using qmcplusplus::afqmc::DeviceBufferManager;
    using pointer_t = typename DeviceBufferManager::template allocator_t<T>::pointer;
    DeviceBufferManager buffer_manager;
    auto alloc{buffer_manager.get_generator().template get_allocator<T>()};
    auto ptr = alloc.allocate(bufferSize);
    cusparse_check(cusparseSpMV(arch::global_cusparse_handle, qmc_cuda::cusparseOperation(transa), &alpha, matA, vecx,
                                &beta, vecy, qmc_cuda::cusparse_data_type<T>(), CUSPARSE_CSRMV_ALG1,
//                                &beta, vecy, qmc_cuda::cusparse_data_type<T>(), CUSPARSE_SPMV_CSR_ALG1,
                                (void*)raw_pointer_cast(ptr)),
                   "csrmv::cusparseSpMV");
    alloc.deallocate(ptr, bufferSize);
  }
  else
  {
    void* dBuffer = NULL;
    cusparse_check(cusparseSpMV(arch::global_cusparse_handle, qmc_cuda::cusparseOperation(transa), &alpha, matA, vecx,
                                &beta, vecy, qmc_cuda::cusparse_data_type<T>(), CUSPARSE_CSRMV_ALG1, dBuffer),
//                                &beta, vecy, qmc_cuda::cusparse_data_type<T>(), CUSPARSE_SPMV_CSR_ALG1, dBuffer),
                   "csrmv::cusparseSpMV");
  }

  cusparse_check(cusparseDestroySpMat(matA), "csrmv::destroyA");
  cusparse_check(cusparseDestroyDnVec(vecx), "csrmv::destroyX");
  cusparse_check(cusparseDestroyDnVec(vecy), "csrmv::destroyY");
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());
}

namespace detail 
{

template<typename T, typename I1, typename I2>
void csrmm_impl(const char transa,
           const int M,
           const int N,
           const int K,
           const T alpha,
           const char* matdescra,
           const T* A,
           const I1* indx,
           const I2* pntrb,
           const I2* pntre,
           const T* B,
           const int ldb,
           const long strideB, 
           const T beta,
           T* C,
           const int ldc,
           const long strideC,
           const int nbatch)
{
  static_assert( std::is_same<I1,int>::value or std::is_same<I1,long>::value, "Incorrect index type.");
  static_assert( std::is_same<I2,int>::value or std::is_same<I2,long>::value, "Incorrect index type.");
  using qmc_cuda::cusparse_check;
  // somehow need to check if the matrix is compact!
  I2 pb, pe;
  arch::memcopy(std::addressof(pb), pntrb, sizeof(I2), arch::memcopyD2H, "lapack_sparse_gpu_ptr::csrmm");
  arch::memcopy(std::addressof(pe), pntre + (M - 1), sizeof(I2), arch::memcopyD2H,
                "lapack_sparse_gpu_ptr::csrmm");
  long nnz = long(pe - pb);
  if(pb != 0)
    APP_ABORT("Error: Found csr matrix with index base != 0. \n");

  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  size_t bufferSize = 0;
  size_t M_         = ((transa == 'N') ? M : K);
  size_t K_         = ((transa == 'N') ? K : M);

  cusparse_check(cusparseCreateCsr(&matA, M, K, nnz, (void*)pntrb, 
        (void*) (indx), (void*) (A), qmc_cuda::cusparse_index_type<I2>(), 
	qmc_cuda::cusparse_index_type<I1>(), CUSPARSE_INDEX_BASE_ZERO, 
        qmc_cuda::cusparse_data_type<T>()), "csrmm::cusparseCreateCsr");
  cusparse_check(cusparseCreateDnMat(&matB, K_, N, ldb, (void*)B, 
        qmc_cuda::cusparse_data_type<T>(), CUSPARSE_ORDER_ROW), "csrmm::cusparseCreateDnMat");
  cusparse_check(cusparseCreateDnMat(&matC, M_, N, ldc, (void*)C, 
        qmc_cuda::cusparse_data_type<T>(), CUSPARSE_ORDER_ROW), "csrmm::cusparseCreateDnMat");
#if defined(CUSPARSE_BATCHED_CSRMM)
  if(nbatch > 1) {
    cusparse_check(cusparseCsrSetStridedBatch(matA, nbatch, 0ul, 0ul), 
        "csrmm::cusparseCsrSetStridedBatch");
    cusparse_check(cusparseDnMatSetStridedBatch(matB, nbatch, int64_t(strideB)), 
        "csrmm::cusparseDnMatSetStridedBatch");
    cusparse_check(cusparseDnMatSetStridedBatch(matC, nbatch, int64_t(strideC)), 
        "csrmm::cusparseDnMatSetStridedBatch");
  }
#else
  using qmc_cuda::global_cuda_streams;
  if (global_cuda_streams.size() < nbatch)
  {
    int n0 = global_cuda_streams.size();
    for (int n = n0; n < nbatch; n++)
    {
      global_cuda_streams.emplace_back(cudaStream_t{});
      cudaStreamCreate(&(global_cuda_streams.back()));
    }
  }
#endif
  cusparse_check(cusparseSpMM_bufferSize(arch::global_cusparse_handle, 
                qmc_cuda::cusparseOperation(transa),
                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
                qmc_cuda::cusparse_data_type<T>(), CUSPARSE_SPMM_CSR_ALG2, &bufferSize),
                "csrmm::cusparseSpMM_bufferSize");

  if (bufferSize > 0)
  {
    using qmcplusplus::afqmc::DeviceBufferManager;
    using pointer_t = typename DeviceBufferManager::template allocator_t<char>::pointer;
    DeviceBufferManager buffer_manager;
    auto alloc{buffer_manager.get_generator().template get_allocator<char>()};
#if defined(CUSPARSE_BATCHED_CSRMM)
    auto ptr = alloc.allocate(bufferSize);
    cusparse_check(cusparseSpMM(arch::global_cusparse_handle, qmc_cuda::cusparseOperation(transa),
                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
                qmc_cuda::cusparse_data_type<T>(), CUSPARSE_SPMM_CSR_ALG2, (void*)ptr),
                "csrmm::cusparseSpMM");
    alloc.deallocate(ptr, bufferSize);
#else
    cudaStream_t s0;
    cusparse_check(cusparseGetStream(arch::global_cusparse_handle,&s0),"csrmm::cusparseGetStream");
    auto ptr = alloc.allocate(nbatch*bufferSize);
    for(int n=0; n<nbatch; ++n) {
      cusparse_check(cusparseSetStream(arch::global_cusparse_handle,global_cuda_streams[n]),
                "csrmm::cusparseSetStream");
      cusparse_check(cusparseDnMatSetValues(matB, (void*)(B+n*strideB)), 
                "csrmm::cusparseDnMatSetValues");
      cusparse_check(cusparseDnMatSetValues(matC, (void*)(C+n*strideC)), 
                "csrmm::cusparseDnMatSetValues");
      cusparse_check(cusparseSpMM(arch::global_cusparse_handle, qmc_cuda::cusparseOperation(transa),
                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
                qmc_cuda::cusparse_data_type<T>(), CUSPARSE_SPMM_CSR_ALG2, 
                (void*)(ptr+n*bufferSize)), "csrmm::cusparseSpMM");
    }
    cusparse_check(cusparseSetStream(arch::global_cusparse_handle,s0),"csrmm::cusparseSetStream");
    qmc_cuda::cuda_check(cudaGetLastError());
    qmc_cuda::cuda_check(cudaDeviceSynchronize()); // sync just in case, before releasing memory
    alloc.deallocate(ptr, nbatch*bufferSize);
#endif
  }
  else
  {
    void* dBuffer = NULL;
#if defined(CUSPARSE_BATCHED_CSRMM)
    cusparse_check(cusparseSpMM(arch::global_cusparse_handle, qmc_cuda::cusparseOperation(transa),
                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
                qmc_cuda::cusparse_data_type<T>(), CUSPARSE_SPMM_CSR_ALG2, dBuffer),
                "csrmm::cusparseSpMM");
#else
    cudaStream_t s0;
    cusparse_check(cusparseGetStream(arch::global_cusparse_handle,&s0),"csrmm::cusparseGetStream");
    for(int n=0; n<nbatch; ++n) {
      cusparse_check(cusparseSetStream(arch::global_cusparse_handle,global_cuda_streams[n]),
                "csrmm::cusparseSetStream");
      cusparse_check(cusparseDnMatSetValues(matB, (void*)(B+n*strideB)),
                "csrmm::cusparseDnMatSetValues");
      cusparse_check(cusparseDnMatSetValues(matC, (void*)(C+n*strideC)),
                "csrmm::cusparseDnMatSetValues");
      cusparse_check(cusparseSpMM(arch::global_cusparse_handle, qmc_cuda::cusparseOperation(transa),
                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
                qmc_cuda::cusparse_data_type<T>(), CUSPARSE_SPMM_CSR_ALG2, dBuffer),
                "csrmm::cusparseSpMM");
    }
    cusparse_check(cusparseSetStream(arch::global_cusparse_handle,s0),"csrmm::cusparseSetStream");
#endif
  }

  cusparse_check(cusparseDestroySpMat(matA), "csrmm::destroyA");
  cusparse_check(cusparseDestroyDnMat(matB), "csrmm::destroyB");
  cusparse_check(cusparseDestroyDnMat(matC), "csrmm::destroyC");
  qmc_cuda::cuda_check(cudaGetLastError());
  qmc_cuda::cuda_check(cudaDeviceSynchronize());

}

template<typename T, typename I1, typename I2>
void csrmm_impl_helper(const char transa, const int M, const int N, const int K, const T alpha, const char* matdescra,
           const T* A, const I1* indx, const I2* pntrb, const I2* pntre, const T* B, const int ldb, const int strideB,
           const T beta, T* C, const int ldc, const int strideC, const int nbatch)
{
  detail::csrmm_impl(transa, M, N, K, alpha, matdescra, A, indx, pntrb, pntre, B, ldb, strideB,
                     beta, C, ldc, strideC, nbatch);
}

template<typename T, typename I1, typename I2>
void csrmm_impl_helper(const char transa, const int M, const int N, const int K, const T alpha, const char* matdescra,
           const T* A, const I1* indx, const I2* pntrb, const I2* pntre, const std::complex<T>* B, const int ldb,
           const int strideB, const T beta, std::complex<T>* C, const int ldc, const int strideC, const int nbatch)
{
  detail::csrmm_impl(transa, M, 2*N, K, alpha, matdescra, A, indx, pntrb, pntre, reinterpret_cast<T const*>(B), 
                     2*ldb, 2*strideB, beta, reinterpret_cast<T*>(C), 2*ldc, 2*strideC, nbatch);
}

}

// find a cleaner way to do this!
template<typename T1, typename T2, typename T3, typename T4, typename I1, typename I2>
void csrmm(const char transa, const int M, const int N, const int K, const T1 alpha, const char* matdescra,
           device::device_pointer<T2> A, device::device_pointer<I1> indx, device::device_pointer<I2> pntrb,
           device::device_pointer<I2> pntre, device::device_pointer<T3> B, const int ldb, const int strideB, 
           const T1 beta, device::device_pointer<T4> C, const int ldc, const int strideC, const int nbatch)
{
  detail::csrmm_impl_helper(transa, M, N, K, alpha, matdescra, raw_pointer_cast(A), raw_pointer_cast(indx), 
			    raw_pointer_cast(pntrb), raw_pointer_cast(pntre), raw_pointer_cast(B), ldb, strideB, 
		     	    beta, raw_pointer_cast(C), ldc, strideC, nbatch);
}




}; // namespace ma


#endif
