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


#ifndef NUMERICS_DISTRIBUTED_ARRAY_SLATE_OPS_HPP
#define NUMERICS_DISTRIBUTED_ARRAY_SLATE_OPS_HPP

/*
 * Utilities for use of SLATE with math::nda::distributed_matrix
 */ 

#include <functional>

#include "utilities/check.hpp"
#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "numerics/distributed_array/ops.hpp"
#include "numerics/distributed_array/detail/ops_aux.hpp"
#include "numerics/distributed_array/detail/slate_aux.hpp"
#if defined(ENABLE_SLATE)
#include "slate/slate.hh"
#endif 

namespace math::nda::slate_ops
{

/***************************************************************************/
/*  				Lapack	  				   */
/***************************************************************************/

/*
template<SlateMatrix A_t, SlateMatrix B_t, SlateMatrix C_t>
void svd(A_t&& A, B_t&& U, std::vector<double> S, C_t&& VH)
{
  using dA_t = typename std::decay_t<A_t>;
  using dB_t = typename std::decay_t<B_t>;
  using dC_t = typename std::decay_t<C_t>;
#if defined(ENABLE_SLATE)
  static_assert(std::is_same_v<typename dA_t::value_type, typename dB_t::value_type>,
                               "Value mismatch");
  static_assert(std::is_same_v<typename dA_t::value_type, typename dC_t::value_type>,
                               "Value mismatch");
  auto As = detail::to_slate_view<dA_t::is_stride_order_C()>(A);
  auto Us = detail::to_slate_view<dB_t::is_stride_order_C()>(U);
  auto VHs = detail::to_slate_view<dC_t::is_stride_order_C()>(VH);

  // need to conjugate matrix in c order
  //slate::svd(As,Us,S,VHs);
  utils::check(false, "svd not yet available in slate.");
#else
  utils::check(false, "svd: requires SLATE, compile with ENABLE_SLATE.");
#endif
}

template<SlateMatrix A_t, SlateMatrix B_t>
void eig(A_t&& A, std::vector<double> L, B_t&& X)
{
  using dA_t = typename std::decay_t<A_t>;
  using dB_t = typename std::decay_t<B_t>;
#if defined(ENABLE_SLATE)
  static_assert(std::is_same_v<typename dA_t::value_type, typename dB_t::value_type>,
                               "Value mismatch");
  auto As = detail::to_slate_view<dA_t::is_stride_order_C()>(A);
  auto Xs = detail::to_slate_view<dB_t::is_stride_order_C()>(X);

  // need to conjugate matrix in c order
  //slate::eig(As,L,Xs);
  utils::check(false, "eig not yet available in slate.");
#else
  utils::check(false, "eig: requires SLATE, compile with ENABLE_SLATE.");
#endif
}
*/

template<bool hermitian = false>
long lu_solve(DistributedMatrix auto&& A, DistributedMatrix auto&& B)
{
  using dA_t = typename std::decay_t<decltype(A)>;
  using dB_t = typename std::decay_t<decltype(B)>;
  using value_type = typename dA_t::value_type;
  long info=0;
  static_assert(std::is_same_v<typename dA_t::value_type, typename dB_t::value_type>,
                               "Value mismatch");
  utils::check(A.global_shape()[0] == A.global_shape()[1], "Shape mismatch in lu_solve.");
  utils::check( *(A.communicator()) == 
		*(B.communicator()),"Communicator mismatch");

  static_assert(::nda::mem::have_compatible_addr_space<typename dA_t::Array_t,
							 typename dB_t::Array_t
							>, "Memory location mismatch.");

// MAM: check if this works on device and enable!
//  if constexpr (::nda::mem::on_host<typename dA_t::Array_t,typename dB_t::Array_t>) {
  if(A.communicator()->size()==1) {

    auto Al = A.local();
    auto Bl = B.local();
    ::nda::basic_array<int, 1, ::nda::C_layout, 'A', 
        ::nda::heap<::nda::mem::get_addr_space<typename dA_t::Array_t>>> ipiv(Al.extent(0));
    info = ::nda::lapack::getrf(Al,ipiv);
    if( info != 0 ) {
      app_warning(" serial lu_solve: getrf info != 0 , info:{}",info); 
      return info;
    }

    if constexpr(dB_t::is_stride_order_Fortran()) {
      info = ::nda::lapack::getrs(Al,Bl,ipiv);
    } else {
      if constexpr ( ::nda::mem::get_addr_space<typename dA_t::Array_t> == ::nda::mem::Host ) {
        ::nda::basic_array<value_type, 2, ::nda::F_layout, 'A', 
            ::nda::heap<::nda::mem::get_addr_space<typename dB_t::Array_t>>> Bf(Bl); 
        info = ::nda::lapack::getrs(Al,Bf,ipiv);
        Bl = Bf;
      } else {
        ::nda::basic_array<value_type, 2, ::nda::F_layout, 'A', 
            ::nda::heap<::nda::mem::get_addr_space<typename dB_t::Array_t>>> Bf(Bl.shape()); 
        // since it is not clear if tensor backend will always accept mixed layouts, 
        // I'm creating a view to the transposed array
        using layout_t = typename ::nda::F_stride_layout::template mapping<2>;
        layout_t idx_{std::array<long,2>{Bl.extent(1),Bl.extent(0)},
            std::array<long,2>{Bl.strides()[1],Bl.strides()[0]}};
        ::nda::basic_array_view<value_type, 2, ::nda::F_stride_layout, 'A', 
            ::nda::default_accessor, 
            ::nda::borrowed<::nda::mem::get_addr_space<typename dB_t::Array_t>>> 
            Bl_f(idx_,Bl.data()); 
        ::nda::tensor::add(value_type(1.0),Bl_f,"ij",value_type(0.0),Bf,"ji");
        info = ::nda::lapack::getrs(Al,Bf,ipiv);
        ::nda::tensor::add(value_type(1.0),Bf,"ij",value_type(0.0),Bl_f,"ji");
      }
    }
    if( info != 0 )
      app_warning(" serial lu_solve: getri info != 0 , info:{}",info); 
    return info;
  }

  constexpr bool _dev_ = ::nda::mem::have_device_compatible_addr_space<
							 typename dA_t::Array_t,
                             typename dB_t::Array_t>;
#if defined(ENABLE_SLATE)

  auto slate_lu = [&](auto &a, auto &b) {
   if constexpr (_dev_) {
      return slate::lu_solve(a,b, {
        // Set execution target to GPU Devices
        { slate::Option::Target, slate::Target::Devices },
        { slate::Option::Lookahead, 1 }
                                    });
   }  else {
      return slate::lu_solve(a,b
#if defined(USE_SLATE_HOSTBATCH)
        ,{ { slate::Option::Target, slate::Target::HostBatch} }
#endif
        );
   } 
  };

//  static_assert(not ::nda::mem::on_device<typename dA_t::Array_t,
///					  typename dB_t::Array_t
//					 >, "lu_solve not working with device arrays!");
  if constexpr ((not hermitian) or dA_t::is_stride_order_Fortran()) {
    static_assert(dA_t::is_stride_order_Fortran(),"Stride order mismatch/hermitian mismatch.");
    auto As = detail::to_slate_view<dA_t::is_stride_order_C()>(A);
    if constexpr(dB_t::is_stride_order_C()) {
      auto Bs = detail::to_slate_view<dB_t::is_stride_order_C()>(transpose(B));
      info = slate_lu(As,Bs); 
    } else {
      auto Bs = detail::to_slate_view<dB_t::is_stride_order_C()>(B);
      info = slate_lu(As,Bs); 
    }
  } else { 
    ::nda::tensor::scale(value_type(1.0),A.local(),::nda::tensor::op::CONJ);
    auto As = detail::to_slate_view<dA_t::is_stride_order_C()>(A);
    if constexpr(dB_t::is_stride_order_C()) {
      auto Bs = detail::to_slate_view<dB_t::is_stride_order_C()>(transpose(B));
      info = slate_lu(As,Bs);
    } else {
      auto Bs = detail::to_slate_view<dB_t::is_stride_order_C()>(B);
      info = slate_lu(As,Bs);
    }
  }

#else
  utils::check(false, "lu_solve: requires SLATE, compile with ENABLE_SLATE.");
#endif
  return info;
}

template<bool hermitian = false>
long least_squares_solve(DistributedMatrix auto&& A, DistributedMatrix auto&& B)
{
  using dA_t = typename std::decay_t<decltype(A)>;
  using dB_t = typename std::decay_t<decltype(B)>;
  using value_type = typename dA_t::value_type;
  long info=0;
  static_assert(std::is_same_v<typename dA_t::value_type, typename dB_t::value_type>,
                               "Value mismatch");
  utils::check(A.global_shape()[0] == A.global_shape()[1], "Shape mismatch in lu_solve.");
  utils::check( *(A.communicator()) == 
		*(B.communicator()),"Communicator mismatch");

  static_assert(::nda::mem::have_compatible_addr_space<typename dA_t::Array_t,
							 typename dB_t::Array_t
							>, "Memory location mismatch.");

  constexpr bool _dev_ = ::nda::mem::have_device_compatible_addr_space<
                          typename dA_t::Array_t,typename dB_t::Array_t>;

  // no gels in nda cuda backend yet!
  if constexpr (not _dev_) {

    if(A.communicator()->size()==1) {

      int rank;
      auto Al = A.local();
      auto Bl = B.local();
      long dmin = std::min(Al.extent(0),Al.extent(1));
      ::nda::basic_array<double, 1, ::nda::F_layout, 'A', 
          ::nda::heap<::nda::mem::get_addr_space<typename dA_t::Array_t>>> S(dmin);

      if constexpr (dA_t::is_stride_order_Fortran()) {

        if constexpr(dB_t::is_stride_order_C()) {
          ::nda::basic_array<value_type, 2, ::nda::F_layout, 'A', 
              ::nda::heap<::nda::mem::get_addr_space<typename dB_t::Array_t>>> B_(Bl.extent(0),Bl.extent(1));
          B_() = Bl();
          long info_ = ::nda::lapack::gelss(Al,B_,S,-1.0,rank);
          Bl() = B_();
          return info_;
        } else {
          return ::nda::lapack::gelss(Al,Bl,S,-1.0,rank);
        }

      } else {

        if constexpr (hermitian) {

          ::nda::tensor::scale(value_type(1.0),Al,::nda::tensor::op::CONJ);
          if constexpr(dB_t::is_stride_order_C()) {
            ::nda::basic_array<value_type, 2, ::nda::F_layout, 'A',
                ::nda::heap<::nda::mem::get_addr_space<typename dB_t::Array_t>>> B_(Bl.extent(0),Bl.extent(1));
            B_() = Bl();
            long info_ = ::nda::lapack::gelss(::nda::transpose(Al),B_,S,-1.0,rank);
            Bl() = B_();
            return info_;
          } else {
            return ::nda::lapack::gelss(::nda::transpose(Al),Bl,S,-1.0,rank);
          }

        } else {

          ::nda::basic_array<value_type, 2, ::nda::F_layout, 'A',
              ::nda::heap<::nda::mem::get_addr_space<typename dA_t::Array_t>>> A_(Al.extent(0),Al.extent(1));
          A_() = Al();
          if constexpr(dB_t::is_stride_order_C()) {
            ::nda::basic_array<value_type, 2, ::nda::F_layout, 'A',
                ::nda::heap<::nda::mem::get_addr_space<typename dB_t::Array_t>>> B_(Bl.extent(0),Bl.extent(1));
            B_() = Bl();
            long info_ = ::nda::lapack::gelss(A_,B_,S,-1.0,rank);
            Bl() = B_();
            return info_;
          } else {
            return ::nda::lapack::gelss(A_,Bl,S,-1.0,rank);
          }

        }

      }

    } // (A.communicator()->size()==1)

  } // constexpr (not _dev_)

#if defined(ENABLE_SLATE)

  auto slate_ls = [&](auto &a, auto &b) {
   if constexpr (_dev_) {
      slate::least_squares_solve(a,b, {
        // Set execution target to GPU Devices
        { slate::Option::Target, slate::Target::Devices },
        { slate::Option::Lookahead, 1 }
                                    });
   }  else {
      slate::least_squares_solve(a,b
#if defined(USE_SLATE_HOSTBATCH)
        ,{ { slate::Option::Target, slate::Target::HostBatch} }
#endif
        );
   } 
  };

  if constexpr ((not hermitian) or dA_t::is_stride_order_Fortran()) {
    static_assert(dA_t::is_stride_order_Fortran(),"Stride order mismatch/hermitian mismatch.");
    auto As = detail::to_slate_view<dA_t::is_stride_order_C()>(A);
    if constexpr(dB_t::is_stride_order_C()) {
      auto Bs = detail::to_slate_view<dB_t::is_stride_order_C()>(transpose(B));
      slate_ls(As,Bs); 
    } else {
      auto Bs = detail::to_slate_view<dB_t::is_stride_order_C()>(B);
      slate_ls(As,Bs); 
    }
  } else { 
    ::nda::tensor::scale(value_type(1.0),A.local(),::nda::tensor::op::CONJ);
    auto As = detail::to_slate_view<dA_t::is_stride_order_C()>(A);
    if constexpr(dB_t::is_stride_order_C()) {
      auto Bs = detail::to_slate_view<dB_t::is_stride_order_C()>(transpose(B));
      slate_ls(As,Bs);
    } else {
      auto Bs = detail::to_slate_view<dB_t::is_stride_order_C()>(B);
      slate_ls(As,Bs);
    }
  }

#else
  utils::check(false, "lu_solve: requires SLATE, compile with ENABLE_SLATE.");
#endif
  return info;
}

void inverse(DistributedMatrix auto&& A)
{
  using dA_t = typename std::decay_t<decltype(A)>;
  using local_Array_t = typename dA_t::Array_t;
  using value_type = typename dA_t::value_type;
  static_assert(local_Array_t::layout_t::is_stride_order_C() or
                local_Array_t::layout_t::is_stride_order_Fortran(),
                "Layout mismatch" );

  if (A.communicator()->size() == 1) {
    auto Aloc = A.local();
    ::nda::basic_array<int, 1, ::nda::C_layout, 'A',
                       ::nda::heap<::nda::mem::get_addr_space<local_Array_t>>> ipiv(Aloc.extent(0));
    long info = ::nda::lapack::getrf(Aloc, ipiv);
    utils::check(info == 0, "inverse: getrf info: {}.", info);
    info = ::nda::lapack::getri(Aloc, ipiv);
    utils::check(info == 0, "inverse: getri info: {}.", info);
    return;
  }

  //if ( __bypass__slate__lapack__ ) {
  if ( false ) {
    ::nda::basic_array<value_type, 2, ::nda::C_layout, 'A',
          ::nda::heap<::nda::mem::get_addr_space<local_Array_t>>> A_s; 
    if(A.communicator()->root()) {
      A_s = ::nda::array<ComplexType,2>(A.global_shape()[0],A.global_shape()[1]);
      gather(0,A,std::addressof(A_s));
      ::nda::basic_array<int, 1, ::nda::C_layout, 'A',
          ::nda::heap<::nda::mem::get_addr_space<local_Array_t>>> ipiv(A_s.extent(0));
      long info = ::nda::lapack::getrf(A_s, ipiv);
      utils::check(info == 0, "inverse: getrf info: {}.", info);
      info = ::nda::lapack::getri(A_s, ipiv);
      utils::check(info == 0, "inverse: getri info: {}.", info);
      scatter(0,std::addressof(A_s),A); 
    } else {
      gather(0,A,std::addressof(A_s));
      scatter(0,std::addressof(A_s),A); 
    }
    return;
  }

#if defined(ENABLE_SLATE)

  auto As = detail::to_slate_view<dA_t::is_stride_order_C()>(A);
  if constexpr (::nda::mem::on_host<local_Array_t>) {
    slate::Pivots pivots;
    long info = slate::getrf ( As , pivots
#if defined(USE_SLATE_HOSTBATCH)
	,{ { slate::Option::Target, slate::Target::HostBatch} }
#endif
	);
    utils::check(info == 0, "inverse: getrf info: {}.", info);
    slate::getri ( As , pivots 
#if defined(USE_SLATE_HOSTBATCH)
	,{ { slate::Option::Target, slate::Target::HostBatch} }
    utils::check(info == 0, "inverse: getri info: {}.", info);
#endif
	);
  } else {
    slate::Pivots pivots ;
    long info = slate::getrf ( As , pivots, 
        // Set execution target to GPU Devices
        {{ slate::Option::Target, slate::Target::Devices },
        { slate::Option::Lookahead, 1 }});
    utils::check(info == 0, "inverse: getrf info: {}.", info);
    info = slate::getri ( As , pivots,
        // Set execution target to GPU Devices
        {{ slate::Option::Target, slate::Target::Devices },
        { slate::Option::Lookahead, 1 }} );
    utils::check(info == 0, "inverse: getri info: {}.", info);
  }
#else
  utils::check(false, "inverse: requires SLATE, compile with ENABLE_SLATE.");
#endif
}

auto determinant(DistributedMatrix auto&&A, std::vector<std::pair<long,long>> &diag_idx) {
  using dA_t = typename std::decay_t<decltype(A)>;
  using local_Array_t = typename dA_t::Array_t;
  using value_type = typename dA_t::value_type;
  value_type det_A = value_type(1.0);

  if constexpr (::nda::mem::on_host<local_Array_t>) {
    if (A.communicator()->size() == 1) {
      auto A_loc = A.local();
      ::nda::matrix_view <value_type> Am(A_loc);
      det_A = ::nda::determinant_in_place(Am);
      return det_A;
    }
  }

#if defined(ENABLE_SLATE)
  if constexpr (::nda::mem::on_host<local_Array_t>) {
    auto As = detail::to_slate_view<dA_t::is_stride_order_C()>(A);
    slate::Pivots pivots ;
    slate::getrf ( As , pivots );

    auto A_loc = A.local();
    long rows_per_blk = A.block_size()[0];
    for (auto idx: diag_idx) {
      det_A *= A_loc(idx.first, idx.second);

      // idx.first = ib * rows_per_blk + ia
      long ib = idx.first / rows_per_blk;
      long ia = idx.first % rows_per_blk;
      if (pivots[ib][ia].tileIndex() != ib and pivots[ib][ia].elementOffset() != ia) {
        det_A *= -1;
      }
    }
    A.communicator()->all_reduce_in_place_n(&det_A, 1, std::multiplies<>{});
    // temporary workaround before the pivots configuration is understood.
    //if (det_A < 0) det_A *= -1.0;
  } else {
    utils::check(false, "determinant: requires GPU supports.");
  }
#else
  utils::check(false, "determinant: requires SLATE, compile with ENABLE_SLATE.");
#endif
  return det_A;
}

/*
void cholesky(DistributedMatrix auto&& A, char UPLO = "L")
{
  using dA_t = typename std::decay_t<decltype(A)>;
  using local_Array_t = typename dA_t::Array_t;
  using value_type = typename dA_t::value_type;
  static_assert(local_Array_t::layout_t::is_stride_order_C() or
                local_Array_t::layout_t::is_stride_order_Fortran(),
                "Layout mismatch" );
  using layout_t = std::conditional_t<
                    local_Array_t::layout_t::is_stride_order_C(),
                    ::nda::C_layout,
                    ::nda::F_layout>; 

  if (A.communicator()->size() == 1) {
    auto Aloc = A.local();
    ::nda::basic_array<int, 1, ::nda::C_layout, 'A',
                       ::nda::heap<::nda::mem::get_addr_space<local_Array_t>>> ipiv(Aloc.extent(0));
    long info = ::nda::lapack::potrf(Aloc);
    utils::check(info == 0, "cholesky: potrf info: {}.", info);
    return;
  }

#if defined(ENABLE_SLATE)

  auto As = detail::to_slate_view<dA_t::is_stride_order_C()>(A);

  // redistribute into HermitianMatrix, call slate
  if constexpr (::nda::mem::on_host<local_Array_t>) {
    long info = slate::potrf ( As ,
//#if defined(USE_SLATE_HOSTBATCH)
	,{ { slate::Option::Target, slate::Target::HostBatch} }
#endif
	);
    utils::check(info == 0, "cholesky: potrf info: {}.", info);
  } else {
    long info = slate::potrf ( As, 
        // Set execution target to GPU Devices
        {{ slate::Option::Target, slate::Target::Devices },
        { slate::Option::Lookahead, 1 }});
    utils::check(info == 0, "cholesky: potrf info: {}.", info);
  }
#else
  utils::check(false, "cholesky: requires SLATE, compile with ENABLE_SLATE.");
#endif
}
*/

/***************************************************************************/
/*  				Blas	  				   */
/***************************************************************************/

namespace detail
{

template<typename T, typename A_t, typename B_t, DistributedMatrix C_t>
auto multiply_impl(T a, A_t&& A, B_t&& B, T b, C_t&& C)
{
  using dA_t = std::decay_t<A_t>;
  using dB_t = std::decay_t<B_t>;
  using dC_t = std::decay_t<C_t>;
  constexpr int Arank = ::nda::get_rank<typename dA_t::Array_t>;
  constexpr int Brank = ::nda::get_rank<typename dB_t::Array_t>;
  static_assert(Arank==2 and Brank==2,"Rank mismatch");
  static_assert(std::is_same_v<std::decay_t<typename dA_t::value_type>, std::decay_t<typename dB_t::value_type>>,
			       "Value mismatch");
  static_assert(std::is_same_v<std::decay_t<typename dA_t::value_type>, std::decay_t<typename dC_t::value_type>>,
			       "Value mismatch");
  static_assert(::nda::mem::have_compatible_addr_space<typename dA_t::Array_t,
                                                         typename dB_t::Array_t,
                                                         typename dC_t::Array_t
                                                        >, "Memory location mismatch.");
  constexpr bool _dev_ = ::nda::mem::have_device_compatible_addr_space<
                                                         typename dA_t::Array_t,
                                                         typename dB_t::Array_t,
                                                         typename dC_t::Array_t
                                                        >;
  utils::check( *((math::detail::arg(A).communicator())) == 
		*((math::detail::arg(B).communicator())),"Communicator mismatch");
  utils::check( *((math::detail::arg(A).communicator())) == 
		*(C.communicator()),"Communicator mismatch");

  if(C.communicator()->size()==1) {
    ::nda::blas::gemm(a, math::detail::local_with_tags(A), math::detail::local_with_tags(B), b, C.local());
    return std::forward<C_t>(C);
  }

#if defined(ENABLE_SLATE)
  // C is fixed
  auto As = detail::to_slate_view<dA_t::is_stride_order_C()>(A);
  auto Bs = detail::to_slate_view<dB_t::is_stride_order_C()>(B);
  auto Cs = detail::to_slate_view<dC_t::is_stride_order_C()>(C);

  if constexpr (dA_t::is_stride_order_Fortran()) {
    static_assert(dB_t::is_stride_order_Fortran(),"Stride order mismatch.");
    static_assert(dC_t::is_stride_order_Fortran(),"Stride order mismatch.");
    if constexpr (_dev_) {
      slate::multiply(a,As,Bs,b,Cs, {
	// Set execution target to GPU Devices
        { slate::Option::Target, slate::Target::Devices }, 
	{ slate::Option::Lookahead, 1 }
				    });
    } else {
      slate::multiply(a,As,Bs,b,Cs
#if defined(USE_SLATE_HOSTBATCH)
	,{ { slate::Option::Target, slate::Target::HostBatch} }
#endif	
      );
    }
  } else {
    static_assert(dB_t::is_stride_order_C(),"Stride order mismatch.");
    static_assert(dC_t::is_stride_order_C(),"Stride order mismatch.");
    if constexpr (_dev_) {
      slate::multiply(a,Bs,As,b,Cs, {
          // Set execution target to GPU Devices
          { slate::Option::Target, slate::Target::Devices },
	  { slate::Option::Lookahead, 1 }
				    });
    } else {
      slate::multiply(a,Bs,As,b,Cs
#if defined(USE_SLATE_HOSTBATCH)
	  ,{ { slate::Option::Target, slate::Target::HostBatch} }
#endif	
	);
    }
  }

#else
  utils::check(false, "requires SLATE, compile with ENABLE_SLATE.");
#endif
  return std::forward<C_t>(C);
}

}

template<typename T, typename A_t, typename B_t, DistributedArray C_t>
auto multiply(T a_v, A_t&& A, B_t&& B, T b_v, C_t&& C)
{
  decltype(::nda::range::all) all;
  using dA_t = std::decay_t<A_t>;
  using dB_t = std::decay_t<B_t>;
  using dC_t = std::decay_t<C_t>;
  constexpr int Arank = ::nda::get_rank<typename dA_t::Array_t>;
  constexpr int Brank = ::nda::get_rank<typename dB_t::Array_t>;
  constexpr int Crank = ::nda::get_rank<typename dC_t::Array_t>;
  static_assert(Arank==Brank and Brank==Crank,"Rank mismatch");

  if constexpr (Arank==2) {
    return detail::multiply_impl(a_v,std::forward<A_t>(A),std::forward<B_t>(B),
				 b_v,std::forward<C_t>(C));
  } else {

    auto& comm = *C.communicator();
    auto&& dA = math::detail::arg(A);    
    auto&& dB = math::detail::arg(B);    

    constexpr bool Atr = math::detail::is_transpose<dA_t>;
    constexpr bool Btr = math::detail::is_transpose<dB_t>;
    constexpr bool Acg = math::detail::is_conjugate_transpose<dA_t>;
    constexpr bool Bcg = math::detail::is_conjugate_transpose<dB_t>;

    // consistency checks
    utils::check(*dA.communicator()==*dB.communicator(),"Communicator mismatch"); 
    utils::check(*dA.communicator()==*C.communicator(),"Communicator mismatch"); 

    if constexpr (dA_t::is_stride_order_C()) {
      static_assert(dB_t::is_stride_order_C() and dC_t::is_stride_order_C(), "Stride mismatch");

      long color=0, px=1;
      for(int r=0; r<Arank-2; ++r) { 
        utils::check(dA.global_shape()[r]==dB.global_shape()[r] and
                     dA.global_shape()[r]==C.global_shape()[r],"Global shape mismatch"); 
        utils::check(dA.local_shape()[r]==dB.local_shape()[r] and
                     dA.local_shape()[r]==C.local_shape()[r],"Local shape mismatch"); 
        utils::check(dA.origin()[r]==dB.origin()[r] and
                     dA.origin()[r]==C.origin()[r],"Origin mismatch"); 
        // these two should ensure consistency across tasks if created with make_distributed
        // otherwise it will need communication to check
        utils::check(dA.grid()[r]==dB.grid()[r] and
                     dA.grid()[r]==C.grid()[r],"Grid mismatch"); 
        utils::check(dA.block_size()[r]==dB.block_size()[r] and
                     dA.block_size()[r]==C.block_size()[r],"Grid mismatch"); 
        color += px*C.origin()[r]; 
        px *= C.global_shape()[r];
      }

      auto get_arr = [](auto const& a) 
	{ return std::array<long,2>{*(a.rbegin()+1),*(a.rbegin())}; };

      auto new_comm = comm.split(color,comm.rank());
      // doing by hand for now
      if constexpr (Arank==3)  {
        for( auto [ia,a] : itertools::enumerate(dA.local_range(0)) ) {
          auto A2d = dA.local()(ia,all,all);
          auto B2d = dB.local()(ia,all,all);
          auto C2d = C.local()(ia,all,all);
          auto A_ = distributed_array_view<decltype(A2d),decltype(new_comm)>(
                std::addressof(new_comm),get_arr(dA.grid()),get_arr(dA.global_shape()),
                get_arr(dA.origin()),get_arr(dA.block_size()),A2d);
          auto B_ = distributed_array_view<decltype(B2d),decltype(new_comm)>(
                std::addressof(new_comm),get_arr(dB.grid()),get_arr(dB.global_shape()),
                get_arr(dB.origin()),get_arr(dB.block_size()),B2d);
          auto C_ = distributed_array_view<decltype(C2d),decltype(new_comm)>(
                std::addressof(new_comm),get_arr(C.grid()),get_arr(C.global_shape()),
                get_arr(C.origin()),get_arr(C.block_size()),C2d);
          detail::multiply_impl(a_v,math::detail::add_tags<Atr,Acg>(A_),math::detail::add_tags<Btr,Bcg>(B_),
                             b_v,C_);
        }
      } else if constexpr (Arank==4) {
        for( auto [ia,a] : itertools::enumerate(dA.local_range(0)) ) 
          for( auto [ib,b] : itertools::enumerate(dA.local_range(1)) ) {
            auto A2d = dA.local()(ia,ib,all,all);
            auto B2d = dB.local()(ia,ib,all,all);
            auto C2d = C.local()(ia,ib,all,all);
            auto A_ = distributed_array_view<decltype(A2d),decltype(new_comm)>(
                  std::addressof(new_comm),get_arr(dA.grid()),get_arr(dA.global_shape()),
                  get_arr(dA.origin()),get_arr(dA.block_size()),A2d);
            auto B_ = distributed_array_view<decltype(B2d),decltype(new_comm)>(
                  std::addressof(new_comm),get_arr(dB.grid()),get_arr(dB.global_shape()),
                  get_arr(dB.origin()),get_arr(dB.block_size()),B2d);
            auto C_ = distributed_array_view<decltype(C2d),decltype(new_comm)>(
                  std::addressof(new_comm),get_arr(C.grid()),get_arr(C.global_shape()),
                  get_arr(C.origin()),get_arr(C.block_size()),C2d);
            detail::multiply_impl(a_v,math::detail::add_tags<Atr,Acg>(A_),math::detail::add_tags<Btr,Bcg>(B_),
                                 b_v,C_);
          }
      } else if constexpr (Arank==5) {
        for( auto [ia,a] : itertools::enumerate(dA.local_range(0)) )
          for( auto [ib,b] : itertools::enumerate(dA.local_range(1)) ) 
            for( auto [ic,c] : itertools::enumerate(dA.local_range(2)) ) {
              auto A2d = dA.local()(ia,ib,ic,all,all);
              auto B2d = dB.local()(ia,ib,ic,all,all);
              auto C2d = C.local()(ia,ib,ic,all,all);
              auto A_ = distributed_array_view<decltype(A2d),decltype(new_comm)>(
                    std::addressof(new_comm),get_arr(dA.grid()),get_arr(dA.global_shape()),
                    get_arr(dA.origin()),get_arr(dA.block_size()),A2d);
              auto B_ = distributed_array_view<decltype(B2d),decltype(new_comm)>(
                    std::addressof(new_comm),get_arr(dB.grid()),get_arr(dB.global_shape()),
                    get_arr(dB.origin()),get_arr(dB.block_size()),B2d);
              auto C_ = distributed_array_view<decltype(C2d),decltype(new_comm)>(
                    std::addressof(new_comm),get_arr(C.grid()),get_arr(C.global_shape()),
                    get_arr(C.origin()),get_arr(C.block_size()),C2d);
              detail::multiply_impl(a_v, math::detail::add_tags<Atr,Acg>(A_),
                                 math::detail::add_tags<Btr,Bcg>(B_),
                                 b_v,C_);
            }
      } else {
        static_assert(Arank==2,"Finish implementation!");
      }

    } else {
      static_assert(dA_t::is_stride_order_Fortran() and
		    dB_t::is_stride_order_Fortran() and 
		    dC_t::is_stride_order_Fortran(), "Stride mismatch");

      long color=0, px=1;
      for(int r=Arank-1; r>=2; --r) {
        utils::check(dA.global_shape()[r]==dB.global_shape()[r] and
                     dA.global_shape()[r]==C.global_shape()[r],"Global shape mismatch");
        utils::check(dA.local_shape()[r]==dB.local_shape()[r] and
                     dA.local_shape()[r]==C.local_shape()[r],"Local shape mismatch");
        utils::check(dA.origin()[r]==dB.origin()[r] and
                     dA.origin()[r]==C.origin()[r],"Origin mismatch");         
        // these two should ensure consistency across tasks if created with make_distributed
        // otherwise it will need communication to check
        utils::check(dA.grid()[r]==dB.grid()[r] and
                     dA.grid()[r]==C.grid()[r],"Grid mismatch");               
        utils::check(dA.block_size()[r]==dB.block_size()[r] and
                     dA.block_size()[r]==C.block_size()[r],"Grid mismatch");
        color += px*C.origin()[r]; 
        px *= C.global_shape()[r];
      }

      auto get_arr = [](auto const& a)
        { return std::array<long,2>{*a.begin(),*(a.begin()+1)}; };

      auto new_comm = comm.split(color,comm.rank());
      // doing by hand for now
      if constexpr (Arank==3)  {
        for( auto [ia,a] : itertools::enumerate(dA.local_range(2)) ) {
          auto A2d = dA.local()(all,all,ia);
          auto B2d = dB.local()(all,all,ia);
          auto C2d = C.local()(all,all,ia);
          auto A_ = distributed_array_view<decltype(A2d),decltype(new_comm)>(
                std::addressof(new_comm),get_arr(dA.grid()),get_arr(dA.global_shape()),
                get_arr(dA.origin()),get_arr(dA.block_size()),A2d);
          auto B_ = distributed_array_view<decltype(B2d),decltype(new_comm)>(
                std::addressof(new_comm),get_arr(dB.grid()),get_arr(dB.global_shape()),
                get_arr(dB.origin()),get_arr(dB.block_size()),B2d);
          auto C_ = distributed_array_view<decltype(C2d),decltype(new_comm)>(
                std::addressof(new_comm),get_arr(C.grid()),get_arr(C.global_shape()),
                get_arr(C.origin()),get_arr(C.block_size()),C2d);
          detail::multiply_impl(a_v,math::detail::add_tags<Atr,Acg>(A_),math::detail::add_tags<Btr,Bcg>(B_),
	  			                b_v,C_);
        }
      } else if constexpr (Arank==4) {
        for( auto [ia,a] : itertools::enumerate(dA.local_range(2)) )
          for( auto [ib,b] : itertools::enumerate(dA.local_range(3)) ) {
            auto A2d = dA.local()(all,all,ia,ib);
            auto B2d = dB.local()(all,all,ia,ib);
            auto C2d = C.local()(all,all,ia,ib);
            auto A_ = distributed_array_view<decltype(A2d),decltype(new_comm)>(
                  std::addressof(new_comm),get_arr(dA.grid()),get_arr(dA.global_shape()),
                  get_arr(dA.origin()),get_arr(dA.block_size()),A2d);
            auto B_ = distributed_array_view<decltype(B2d),decltype(new_comm)>(
                  std::addressof(new_comm),get_arr(dB.grid()),get_arr(dB.global_shape()),
                  get_arr(dB.origin()),get_arr(dB.block_size()),B2d);
            auto C_ = distributed_array_view<decltype(C2d),decltype(new_comm)>(
                  std::addressof(new_comm),get_arr(C.grid()),get_arr(C.global_shape()),
                  get_arr(C.origin()),get_arr(C.block_size()),C2d);
            detail::multiply_impl(a_v,math::detail::add_tags<Atr,Acg>(A_),math::detail::add_tags<Btr,Bcg>(B_),
                                  b_v,C_);
          }
      } else if constexpr (Arank==5) {
        for( auto [ia,a] : itertools::enumerate(dA.local_range(2)) )
          for( auto [ib,b] : itertools::enumerate(dA.local_range(3)) )
            for( auto [ic,c] : itertools::enumerate(dA.local_range(4)) ) {
              auto A2d = dA.local()(all,all,ia,ib,ic);
              auto B2d = dB.local()(all,all,ia,ib,ic);
              auto C2d = C.local()(all,all,ia,ib,ic);
              auto A_ = distributed_array_view<decltype(A2d),decltype(new_comm)>(
                    std::addressof(new_comm),get_arr(dA.grid()),get_arr(dA.global_shape()),
                    get_arr(dA.origin()),get_arr(dA.block_size()),A2d);
              auto B_ = distributed_array_view<decltype(B2d),decltype(new_comm)>(
                    std::addressof(new_comm),get_arr(dB.grid()),get_arr(dB.global_shape()),
                    get_arr(dB.origin()),get_arr(dB.block_size()),B2d);
              auto C_ = distributed_array_view<decltype(C2d),decltype(new_comm)>(
                    std::addressof(new_comm),get_arr(C.grid()),get_arr(C.global_shape()),
                    get_arr(C.origin()),get_arr(C.block_size()),C2d);
              detail::multiply_impl(a_v,math::detail::add_tags<Atr,Acg>(A_),
                                    math::detail::add_tags<Btr,Bcg>(B_),
                                    b_v,C_);
            }
      } else {
        static_assert(Arank==2,"Finish implementation!");
      }

    }

  }

  return std::forward<C_t>(C);
}

template<typename A_t, typename B_t, typename C_t>
auto multiply(A_t&& A, B_t&& B, C_t&& C)
{
  using T = typename std::decay_t<A_t>::value_type;
  return multiply(T{1.0},std::forward<A_t>(A),std::forward<B_t>(B),
		  T{0.0},std::forward<C_t>(C));
}

} // math::nda

#endif
