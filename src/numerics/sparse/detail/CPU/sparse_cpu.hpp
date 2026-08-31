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



#ifndef SPARSE_CPU_HPP
#define SPARSE_CPU_HPP

#include <stdexcept>
#include <vector>

#include "configuration.hpp"
#if defined(NDA_USE_MKL)
//#include "numerics/sparse/detail/CPU/mkl_spblas.h"
#include "mkl_spblas.h"
#endif
#include <cassert>
#include <complex>

namespace math::sparse
{
namespace backup_impl
{
template<typename T, typename I1, typename I2>
void csrmv(const char transa,
           const int M,
           const int K,
           const T alpha,
           const char* matdescra,
           const T* A,
           const I1* indx,
           const I2* pntrb,
           const I2* pntre,
           const T* x,
           const T beta,
           T* y)
{
  if(not(matdescra[0] == 'G' && (matdescra[3] == 'C' || matdescra[3]=='F')))
    throw std::runtime_error("backup_impl::csrmm: Invalid matdescra");
  auto p0   = *pntrb;
  if (transa == 'n' || transa == 'N')
  {
    for (int nr = 0; nr < M; nr++, y++, pntrb++, pntre++)
    {
      (*y) *= beta;
      for (I2 i = *pntrb - p0; i < *pntre - p0; i++)
      {
        if (*(indx + i) >= K)
          continue;
        *y += alpha * (*(A + i)) * (*(x + (*(indx + i))));
      }
    }
  }
  else if (transa == 't' || transa == 'T')
  {
    for (int k = 0; k < K; k++)
      (*(y + k)) *= beta;
    for (int nr = 0; nr < M; nr++, pntrb++, pntre++, x++)
    {
      for (I2 i = *pntrb - p0; i < *pntre - p0; i++)
      {
        if (*(indx + i) >= K)
          continue;
        *(y + (*(indx + i))) += alpha * (*(A + i)) * (*x);
      }
    }
  }
  else if (transa == 'h' || transa == 'H' || transa == 'c' || transa == 'C')
  {
    for (int k = 0; k < K; k++)
      (*(y + k)) *= beta;
    for (int nr = 0; nr < M; nr++, pntrb++, pntre++, x++)
    {
      for (I2 i = *pntrb - p0; i < *pntre - p0; i++)
      {
        if (*indx >= K)
          continue;
        *(y + (*(indx + i))) += alpha * (*(A + i)) * (*x);
      }
    }
  }
}

template<typename T, typename I1, typename I2>
void csrmv(const char transa,
           const int M,
           const int K,
           const std::complex<T> alpha,
           const char* matdescra,
           const std::complex<T>* A,
           const I1* indx,
           const I2* pntrb,
           const I2* pntre,
           const std::complex<T>* x,
           const std::complex<T> beta,
           std::complex<T>* y)
{
  if(not(matdescra[0] == 'G' && (matdescra[3] == 'C' || matdescra[3]=='F')))
    throw std::runtime_error("backup_impl::csrmm: Invalid matdescra");
  auto p0   = *pntrb;
  if (transa == 'n' || transa == 'N')
  {
    for (int nr = 0; nr < M; nr++, y++, pntrb++, pntre++)
    {
      (*y) *= beta;
      for (I2 i = *pntrb - p0; i < *pntre - p0; i++)
      {
        if (*(indx + i) >= K)
          continue;
        *y += alpha * (*(A + i)) * (*(x + (*(indx + i))));
      }
    }
  }
  else if (transa == 't' || transa == 'T')
  {
    for (int k = 0; k < K; k++)
      (*(y + k)) *= beta;
    for (int nr = 0; nr < M; nr++, pntrb++, pntre++, x++)
    {
      for (I2 i = *pntrb - p0; i < *pntre - p0; i++)
      {
        if (*(indx + i) >= K)
          continue;
        *(y + (*(indx + i))) += alpha * (*(A + i)) * (*x);
      }
    }
  }
  else if (transa == 'h' || transa == 'H' || transa == 'c' || transa == 'C')
  {
    for (int k = 0; k < K; k++)
      (*(y + k)) *= beta;
    for (int nr = 0; nr < M; nr++, pntrb++, pntre++, x++)
    {
      for (I2 i = *pntrb - p0; i < *pntre - p0; i++)
      {
        if (*indx >= K)
          continue;
        *(y + (*(indx + i))) += alpha * std::conj(*(A + i)) * (*x);
      }
    }
  }
}

template<typename T, typename I1, typename I2>
void csrmm(const char transa,
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
           const T beta,
           T* C,
           const int ldc)
{
  if(not(matdescra[0] == 'G' && (matdescra[3] == 'C' || matdescra[3]=='F')))
    throw std::runtime_error("backup_impl::csrmm: Invalid matdescra"); 
  auto p0   = *pntrb;
  if(matdescra[3] == 'F') {
    for(int nc=0; nc<N; ++nc, B+=ldb, C+=ldc) 
    {
      csrmv(transa,M,K,alpha,matdescra,A,indx,pntrb,pntre,B,beta,C);
    }
  } else {
    if (transa == 'n' || transa == 'N')
    {
      for (int nr = 0; nr < M; nr++, pntrb++, pntre++, C += ldc)
      {
        for (int i = 0; i < N; i++)
          (*(C + i)) *= beta;
        for (I2 i = *pntrb - p0; i < *pntre - p0; i++)
        {
          if (*(indx + i) >= K)
            continue;
          // at this point *(A+i) is A_rc, c=*(indx+i), *C is C(r,0)
          // C(r,:) = A_rc * B(c,:)
          const T* Bc = B + ldb * (*(indx + i));
          T* Cr       = C;
          T Arc       = alpha * (*(A + i));
          for (int k = 0; k < N; k++, Cr++, Bc++)
            *Cr += Arc * (*Bc);
        }
      }
    }
    else if (transa == 't' || transa == 'T')
    {
      // not optimal, but simple
      for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
          (*(C + i * ldc + j)) *= beta;
      for (int nr = 0; nr < M; nr++, pntrb++, pntre++, B += ldb)
      {
        for (I2 i = *pntrb - p0; i < *pntre - p0; i++)
        {
          if (*(indx + i) >= K)
            continue;
          // at this point *(A+i) is A_rc, c=*(indx+i)
          // C(c,:) = A_rc * B(r,:)
          const T* Br = B;
          T* Cc       = C + ldc * (*(indx + i));
          T Arc       = alpha * (*(A + i));
          for (int k = 0; k < N; k++, Cc++, Br++)
            *Cc += Arc * (*Br);
        }
      }
    }
    else if (transa == 'h' || transa == 'H' || transa == 'c' || transa == 'C')
    {
      // not optimal, but simple
      for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
          (*(C + i * ldc + j)) *= beta;
      for (int nr = 0; nr < M; nr++, pntrb++, pntre++, B += ldb)
      {
        for (I2 i = *pntrb - p0; i < *pntre - p0; i++)
        {
          if (*(indx + i) >= K)
            continue;
          // at this point *(A+i) is A_rc, c=*(indx+i)
          // C(c,:) = A_rc * B(r,:)
          const T* Br = B;
          T* Cc       = C + ldc * (*(indx + i));
          T Arc       = alpha * (*(A + i));
          for (int k = 0; k < N; k++, Cc++, Br++)
            *Cc += Arc * (*Br);
        }
      }
    }  // transa
  }  // matdescra
}

template<typename T, typename I1, typename I2>
void csrmm(const char transa,
           const int M,
           const int N,
           const int K,
           const std::complex<T> alpha,
           const char* matdescra,
           const std::complex<T>* A,
           const I1* indx,
           const I2* pntrb,
           const I2* pntre,
           const std::complex<T>* B,
           const int ldb,
           const std::complex<T> beta,
           std::complex<T>* C,
           const int ldc)
{
  if(not(matdescra[0] == 'G' && (matdescra[3] == 'C' || matdescra[3]=='F')))
    throw std::runtime_error("backup_impl::csrmm: Invalid matdescra");
  auto p0   = *pntrb;
  if(matdescra[3] == 'F') {
    for(int nc=0; nc<N; ++nc, B+=ldb, C+=ldc)
    { 
      csrmv(transa,M,K,alpha,matdescra,A,indx,pntrb,pntre,B,beta,C);
    }
  } 
  else 
  {
    if (transa == 'n' || transa == 'N')
    {
      for (int nr = 0; nr < M; nr++, pntrb++, pntre++, C += ldc)
      {
        for (int i = 0; i < N; i++)
          (*(C + i)) *= beta;
        for (I2 i = *pntrb - p0; i < *pntre - p0; i++)
        {
          if (*(indx + i) >= K)
            continue;
          // at this point *(A+i) is A_rc, c=*(indx+i), *C is C(r,0)
          // C(r,:) = A_rc * B(c,:)
          const std::complex<T>* Bc = B + ldb * (*(indx + i));
          std::complex<T>* Cr       = C;
          std::complex<T> Arc       = alpha * (*(A + i));
          for (int k = 0; k < N; k++, Cr++, Bc++)
            *Cr += Arc * (*Bc);
        }
      }
    }
    else if (transa == 't' || transa == 'T')
    {
      // not optimal, but simple
      for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
          (*(C + i * ldc + j)) *= beta;
      for (int nr = 0; nr < M; nr++, pntrb++, pntre++, B += ldb)
      {
        for (I2 i = *pntrb - p0; i < *pntre - p0; i++)
        {
          if (*(indx + i) >= K)
            continue;
          // at this point *(A+i) is A_rc, c=*(indx+i)
          // C(c,:) = A_rc * B(r,:)
          const std::complex<T>* Br = B;
          std::complex<T>* Cc       = C + ldc * (*(indx + i)) ;
          std::complex<T> Arc       = alpha * (*(A + i));
          for (int k = 0; k < N; k++, Cc++, Br++)
            *Cc += Arc * (*Br);
        }
      }
    }
    else if (transa == 'h' || transa == 'H' || transa == 'c' || transa == 'C')
    {
      // not optimal, but simple
      for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
          (*(C + i * ldc + j)) *= beta;
      for (int nr = 0; nr < M; nr++, pntrb++, pntre++, B += ldb)
      {
        for (I2 i = *pntrb - p0; i < *pntre - p0; i++)
        {
          if (*(indx + i) >= K)
            continue;
          // at this point *(A+i) is A_rc, c=*(indx+i)
          // C(c,:) = A_rc * B(r,:)
          const std::complex<T>* Br = B;
          std::complex<T>* Cc       = C + ldc * (*(indx + i));
          std::complex<T> Arc       = alpha * std::conj(*(A + i));
          for (int k = 0; k < N; k++, Cc++, Br++)
            *Cc += Arc * (*Br);
        }
      }
    }
  }
}

} // namespace backup_impl

namespace cpu
{
#if defined(NDA_USE_MKL)
namespace detail
{
  // MAM: Careful with indexing, right now limited to 32 bit indexing. Can you 64-bit with some changes and linking to ILP64 library
  inline auto* mklcplx(std::complex<float> *c) { return reinterpret_cast<MKL_Complex8 *>(c); }              // NOLINT
  inline auto* mklcplx(std::complex<float> const *c) { return reinterpret_cast<const MKL_Complex8 *>(c); }  // NOLINT
  inline auto* mklcplx(std::complex<double> *c) { return reinterpret_cast<MKL_Complex16 *>(c); }              // NOLINT
  inline auto* mklcplx(std::complex<double> const *c) { return reinterpret_cast<const MKL_Complex16 *>(c); }  // NOLINT

  inline MKL_Complex8 mklcplx(std::complex<float> c) { return MKL_Complex8{c.real(),c.imag()}; }
  inline MKL_Complex16 mklcplx(std::complex<double> c) { return MKL_Complex16{c.real(),c.imag()}; }

  inline sparse_operation_t mkl_operation(const char transa) {
    if(transa == 'N' or transa == 'n')
      return SPARSE_OPERATION_NON_TRANSPOSE;
    else if(transa == 'T' or transa == 't')
      return SPARSE_OPERATION_TRANSPOSE;
    else if(transa == 'C' or transa == 'c' or transa == 'H' or transa == 'h')
      return SPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    else {
      throw std::runtime_error("Invalid mkl_operation");
      assert(false);
    }
    return SPARSE_OPERATION_NON_TRANSPOSE;
  }

  inline auto make_csr_s(int rows, int cols, float* values, int* col_indx, int* rows_start, int* rows_end)
  { 
    sparse_matrix_t csr = NULL;
    auto stat = mkl_sparse_s_create_csr (std::addressof(csr), SPARSE_INDEX_BASE_ZERO, rows, cols, rows_start, rows_end, col_indx, values);  
    if( stat !=  SPARSE_STATUS_SUCCESS )
      throw std::runtime_error("MKL error: mkl_sparse_d_create_csr");
    return csr;
  }
  
  inline auto make_csr_c(int rows, int cols, std::complex<float>* values, int* col_indx, int* rows_start, int* rows_end)
  { 
    sparse_matrix_t csr = NULL;
    auto stat = mkl_sparse_c_create_csr (std::addressof(csr), SPARSE_INDEX_BASE_ZERO, rows, cols, rows_start, rows_end, col_indx, mklcplx(values));
    if( stat !=  SPARSE_STATUS_SUCCESS )
      throw std::runtime_error("MKL error: mkl_sparse_d_create_csr");
    return csr;
  }
  
  inline auto make_csr_d(int rows, int cols, double* values, int* col_indx, int* rows_start, int* rows_end)
  {
    sparse_matrix_t csr = NULL;
    auto stat = mkl_sparse_d_create_csr (std::addressof(csr), SPARSE_INDEX_BASE_ZERO, rows, cols, rows_start, rows_end, col_indx, values);
    if( stat !=  SPARSE_STATUS_SUCCESS )
      throw std::runtime_error("MKL error: mkl_sparse_d_create_csr");
    return csr;
  }

  inline auto make_csr_z(int rows, int cols, std::complex<double>* values, int* col_indx, int* rows_start, int* rows_end)
  { 
    sparse_matrix_t csr = NULL;
    auto stat = mkl_sparse_z_create_csr (std::addressof(csr), SPARSE_INDEX_BASE_ZERO, rows, cols, rows_start, rows_end, col_indx, mklcplx(values));  
    if( stat !=  SPARSE_STATUS_SUCCESS )
      throw std::runtime_error("MKL error: mkl_sparse_d_create_csr");
    return csr;
  }

  template<typename T>
  auto make_csr(int rows, int cols, T* values, int* col_indx, int* rows_start, int* rows_end)
  {  
    if constexpr (std::is_same_v<T,float>) {
      return make_csr_s(rows,cols,values,col_indx,rows_start,rows_end);
    } else if constexpr (std::is_same_v<T,double>) {
      return make_csr_d(rows,cols,values,col_indx,rows_start,rows_end);
    } else if constexpr (std::is_same_v<T,std::complex<float>>) {
      return make_csr_c(rows,cols,values,col_indx,rows_start,rows_end);
    } else if constexpr (std::is_same_v<T,std::complex<double>>) {
      return make_csr_z(rows,cols,values,col_indx,rows_start,rows_end);
    } else {
      throw std::runtime_error("MKL error: make_csr invalid datatype.");
    }
  }

  inline void destroy_csr(sparse_matrix_t A) {
    auto stat = mkl_sparse_destroy(A);
    if( stat !=  SPARSE_STATUS_SUCCESS )
      throw std::runtime_error("MKL error: mkl_sparse_destroy");
  }

}
#endif

// assuming 32-bit mkl
template<typename T, typename I1, typename I2>   
inline static void csrmv(const char transa,
                         const int M,
                         const int K,
                         const T alpha,
                         [[maybe_unused]] const char* matdescra,
                         const T* A,
                         const I1* indx,
                         const I2* pntrb,
                         const I2* pntre,
                         const T* x,
                         const T beta,
                               T* y)
{
  if constexpr (std::is_same_v<I1,int> and std::is_same_v<I2,int>) {
#if defined(NDA_USE_MKL)
    auto csr = detail::make_csr(M,K,const_cast<T*>(A),const_cast<int*>(indx),const_cast<int*>(pntrb),const_cast<int*>(pntre));
    matrix_descr descr = {SPARSE_MATRIX_TYPE_GENERAL,SPARSE_FILL_MODE_LOWER,SPARSE_DIAG_NON_UNIT};
    if constexpr (std::is_same_v<T,float>) {
      mkl_sparse_s_mv(detail::mkl_operation(transa), alpha, csr, descr, x, beta, y);
    } else if constexpr (std::is_same_v<T,double>) {
      mkl_sparse_d_mv(detail::mkl_operation(transa), alpha, csr, descr, x, beta, y);
    } else if constexpr (std::is_same_v<T,std::complex<float>>) {
      mkl_sparse_c_mv(detail::mkl_operation(transa), detail::mklcplx(alpha), csr, descr, detail::mklcplx(x), detail::mklcplx(beta), detail::mklcplx(y));
    } else if constexpr (std::is_same_v<T,std::complex<double>>) {
      mkl_sparse_z_mv(detail::mkl_operation(transa), detail::mklcplx(alpha), csr, descr, detail::mklcplx(x), detail::mklcplx(beta), detail::mklcplx(y));
    } else {
      throw std::runtime_error("MKL error: make_csr invalid datatype.");
    }
    detail::destroy_csr(csr);
#else
    backup_impl::csrmv(transa, M, K, alpha, matdescra, A, indx, pntrb, pntre, x, beta, y);
#endif
  } else { 
    backup_impl::csrmv(transa, M, K, alpha, matdescra, A, indx, pntrb, pntre, x, beta, y);
  }
}

template<typename T, typename I1, typename I2>   
inline static void csrmm_impl(const char transa,
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
                         const T beta,
                               T* C,
                         const int ldc)
{
  if constexpr (std::is_same_v<I1,int> and std::is_same_v<I2,int>) {  
#if defined(NDA_USE_MKL)
    auto csr = detail::make_csr(M,K,const_cast<T*>(A),const_cast<int*>(indx),const_cast<int*>(pntrb),const_cast<int*>(pntre));
    matrix_descr descr = {SPARSE_MATRIX_TYPE_GENERAL,SPARSE_FILL_MODE_LOWER,SPARSE_DIAG_NON_UNIT};
    sparse_layout_t lay = (matdescra[3] == 'C'?SPARSE_LAYOUT_ROW_MAJOR:SPARSE_LAYOUT_COLUMN_MAJOR);
    if constexpr (std::is_same_v<T,float>) {
      mkl_sparse_s_mm(detail::mkl_operation(transa), alpha, csr, descr, lay, B, N, ldb, beta, C, ldc);
    } else if constexpr (std::is_same_v<T,double>) {
      mkl_sparse_d_mm(detail::mkl_operation(transa), alpha, csr, descr, lay, B, N, ldb, beta, C, ldc);
    } else if constexpr (std::is_same_v<T,std::complex<float>>) {
      mkl_sparse_c_mm(detail::mkl_operation(transa), detail::mklcplx(alpha), csr, descr, lay, detail::mklcplx(B), N, ldb, detail::mklcplx(beta), detail::mklcplx(C), ldc);
    } else if constexpr (std::is_same_v<T,std::complex<double>>) {
      mkl_sparse_z_mm(detail::mkl_operation(transa), detail::mklcplx(alpha), csr, descr, lay, detail::mklcplx(B), N, ldb, detail::mklcplx(beta), detail::mklcplx(C), ldc);
    } else {
      throw std::runtime_error("MKL error: make_csr invalid datatype.");
    }
    detail::destroy_csr(csr);
#else
    backup_impl::csrmm(transa, M, N, K, alpha, matdescra, A, indx, pntrb, pntre, B, ldb, beta, C, ldc);
#endif
  } else {
    backup_impl::csrmm(transa, M, N, K, alpha, matdescra, A, indx, pntrb, pntre, B, ldb, beta, C, ldc);
  }
}

template<typename T, typename I1, typename I2>   
inline static void csrmv(const char transa,
                         const int M,
                         const int K,
                         T alpha,
                         const char* matdescra,
                         T* A,
                         const I1* indx,
                         const I2* pntrb,
                         const I2* pntre,
                         const std::complex<T>* x,
                         const T beta,
                         std::complex<T>* y)
{
  csrmm_impl(transa, M, 2, K, alpha, matdescra, A, indx, pntrb, pntre, reinterpret_cast<T const*>(x), 2, beta,
        reinterpret_cast<T*>(y), 2);
}

template<typename T, typename I1, typename I2>   
inline static void csrmm_impl(const char transa,
                         const int M,
                         const int N,
                         const int K,
                         const T alpha,
                         const char* matdescra,
                         const T* A,
                         const I1* indx,
                         const I2* pntrb,
                         const I2* pntre,
                         const std::complex<T>* B,
                         const int ldb,
                         const T beta,
                         std::complex<T>* C,
                         const int ldc)
{
  if(matdescra[3] != 'C')
    throw std::runtime_error("Mixed precision csrmm only with C_layout arrays");
  csrmm_impl(transa, M, 2 * N, K, alpha, matdescra, A, indx, pntrb, pntre, reinterpret_cast<T const*>(B), 2 * ldb, beta,
        reinterpret_cast<T*>(C), 2 * ldc);
}

template<class T1, class T2, typename I1, typename I2>
inline static void csrmm(const char transa,
                         const int M,
                         const int N,
                         const int K,
                         const T1 alpha,
                         const char* matdescra,
                         const T1* A,
                         const I1* indx,
                         const I2* pntrb,
                         const I2* pntre,
                         const T2* B,
                         const int ldb,
                         const long strideB,
                         const T1 beta,
                         T2* C,
                         const int ldc,
                         const long strideC,
                         const int nbatch)
{
  // no natching on cpu for now, :-(
  for(int n=0; n<nbatch; ++n, B+=strideB, C+=strideC)
    csrmm_impl(transa,M,N,K,alpha,matdescra,A,indx,pntrb,pntre,B,ldb,beta,C,ldc);
}

} // namespace cpu
} // namespace math::sparse
#endif
