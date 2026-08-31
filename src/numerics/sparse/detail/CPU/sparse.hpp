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


////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

#ifndef NUMERICS_DETAIL_CPU_SPARSE_HPP
#define NUMERICS_DETAIL_CPU_SPARSE_HPP

#include "configuration.hpp"
#include "numerics/sparse/detail/CPU/sparse_cpu.hpp"

namespace math::sparse
{

template<typename T1, typename T2, typename T3, typename T4,
         typename T5, typename T6, typename T7, typename T8>
void csrmv(char transa, int M, int K, T1 alpha, const char* matdescra,
           T2 A, T3 indx, T4 pntrb, T5 pntre, T6 x, T7 beta, T8 y)
{ 
  cpu::csrmv(transa,M,K,alpha,matdescra,A,indx,pntrb,pntre,x,beta,y);
}   

template<typename T1, typename T2, typename T3, typename T4,
	 typename T5, typename T6, typename T7, typename T8>
void csrmm(char transa, int M, int N, int K, T1 alpha, const char* matdescra, 
	   T2 A, T3 indx, T4 pntrb, T5 pntre, T6 B, int ldb, int strideB, 
	   T7 beta, T8 C, int ldc, int strideC, int nbatch)
{
  auto B_ = B;
  auto C_ = C;
  for(int n=0; n<nbatch; ++n, B_+=strideB, C_+=strideC)
    cpu::csrmm_impl(transa,M,N,K,alpha,matdescra,A,indx,pntrb,pntre,B_,ldb,beta,C_,ldc);
}

} // namespace math::sparse

#endif
