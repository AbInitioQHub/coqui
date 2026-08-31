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


#ifndef UTILITIES_DISTRIBUTED_CHOLESKY_HPP
#define UTILITIES_DISTRIBUTED_CHOLESKY_HPP

#include <tuple>
#include <cmath>
#include <algorithm>
#include <limits>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"

#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"
#include "utilities/mpi_context.h"
#include "utilities/concepts.hpp"

#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/tensor.hpp"
#include "nda/linalg/det_and_inverse.hpp"
#include "itertools/itertools.hpp"
#include "utilities/proc_grid_partition.hpp"
#include "utilities/functions.hpp"
#include "utilities/Timer.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

namespace utils
{

/***
 * Pivoted, blocked Cholesky factorization of a distributed matrix Mab.
 * Returns a distributed matrix L_na, where Mab = sum_n L_n_a * conj(L_n_b). 
 *
 * Assumes that nproc <= ndim of M
 ***/
auto distributed_cholesky(math::nda::DistributedArrayOfRank<2> auto const& M, 
                          double thresh = 1e-12, 
                          int block_size = 32)
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<typename std::decay_t<decltype(M)>::Array_t>();
  using CArray_2D_t = memory::array<MEM,ComplexType,2>;
  decltype(nda::range::all) all;
  std::string err_("Error in utils::distributed_cholesky: ");

  utils::TimerManager Timer;

  auto M_loc = M.local();
  auto& comm = *M.communicator();
  long ndim = M.global_shape()[0];

  utils::check(M.global_shape()[1] == ndim, "Shape mismatch.");

  auto CholMat = math::nda::make_distributed_array<CArray_2D_t>(comm,
                      {1,comm.size()},{ndim,ndim},{1,1});
  auto a_rng = CholMat.local_range(1);  
  CholMat.local() = ComplexType(0.0);

  // adjust block_size
  block_size = std::min(block_size, int(a_rng.size())); 
  comm.all_reduce_in_place_n(&block_size,1,mpi3::min<>{});

  nda::array<long, 1> global_indx(block_size,0);
  nda::array<RealType,1> lmax_res_val(block_size,0.0);
  nda::array<long, 1> lmax_res_indx(block_size,0);
  nda::array<std::pair<RealType,int>,1> gmax_res(block_size);

  // working arrays to store list of indexes in current iteration and associated columns
  // remember that we assume results from computation kernels need to be reduce
  auto R = CArray_2D_t::zeros({block_size,ndim});
  auto Rc = CArray_2D_t::zeros({ndim,block_size});

  // calculate diagonal and reduce over processor grid
  memory::array<MEM,ComplexType,1> Diag(a_rng.size(),0.0);
  {
    auto b_rng = M.local_range(1);
    for( auto [ia,a] : itertools::enumerate(M.local_range(0)) ) 
      if( a >= b_rng.first() and a < b_rng.last() ) R(0,a) = M_loc(ia,a-b_rng.first()); 
  }
  comm.all_reduce_in_place_n(R.data(),ndim,std::plus<>{});
  Diag() = R(0,a_rng);

  utils::max_element_multi(Diag,lmax_res_val,lmax_res_indx);
  utils::find_distributed_maximum(comm,lmax_res_val,gmax_res);

  nda::array<int,1> piv(block_size+1,0);
  memory::unified_array<ComplexType,2> Abb(block_size,block_size);

  // a bit nicer this way...
  auto find_max = [&]() {
    return (*std::max_element(gmax_res.begin(),gmax_res.end(), [](auto& a, auto& b) {
                return a.first < b.first;
        })).first;
  };

  long nchol (0);
  auto old_max = find_max();
  app_log(3,"nchol, max |D|: ");
  while(true) { 

    // stopping condition when thresh is set
    if( thresh > 0.0 and thresh > old_max) break;
    if( nchol >= ndim ) break;

    utils::check( std::isfinite(old_max), 
          err_ + "Cholesky algorithm failed in utils::distributed_cholesky. \n" + 
          "       Found invalid residual:{}",old_max); 

    // compute columns for indexes associated with lmax_res_indx
    global_indx() = 0;
    for( int n=0; n<block_size; ++n ) {
      if( gmax_res(n).second/block_size == comm.rank() ) 
        global_indx(n) = a_rng.first() + lmax_res_indx(gmax_res(n).second%block_size);  
    }
    comm.all_reduce_in_place_n(global_indx.data(),global_indx.size(),std::plus<>{});
    {
      R() = ComplexType(0.0);
      auto rng1 = M.local_range(0);
      auto rng2 = M.local_range(1);
      for( auto [in,n] : itertools::enumerate(global_indx) ) 
        if( n >= rng2.first() and n < rng2.last() ) R(in,rng1) = M_loc(all,n-rng2.first());
    }
    comm.all_reduce_in_place_n(R.data(),R.size(),std::plus<>{});

    // view into local range
    auto Rloc = R(all,a_rng);

    // orthonormalize cholesky vector
    if(nchol > 0) {
      Rc(nda::range(nchol),all) = ComplexType(0.0);
      for( int n=0; n<block_size; ++n ) {
        if( gmax_res(n).second/block_size == comm.rank() )  { 
          long lr = lmax_res_indx(gmax_res(n).second%block_size);
          Rc(nda::range(nchol),n) = CholMat.local()(nda::range(nchol),lr);
        }
      }
      comm.all_reduce_in_place_n(Rc.data(),nchol*block_size,std::plus<>{});

      nda::blas::gemm(ComplexType(-1.0),nda::dagger(Rc(nda::range(0,nchol),all)),
                      CholMat.local()(nda::range(0,nchol),all),ComplexType(1.0),Rloc);
    }

    // form block matrix
    Abb() = ComplexType(0.0);
    for(int n=0; n<block_size; ++n) {
      if( gmax_res(n).second/block_size == comm.rank() )  { 
        long lr = lmax_res_indx(gmax_res(n).second%block_size);
        Abb(n,all) = Rloc(all,lr);  
      }
    }
    // nccl?
    if(comm.size()>1) comm.reduce_in_place_n(Abb.data(),block_size*block_size,std::plus<>{});
    if(comm.rank() == 0) {
      // pivots are guaranteed to be in ascending order!
      // MAM: inverse_in_place requires nda::matrix
      using W_type = nda::matrix<ComplexType,nda::C_layout>;
      auto W = utils::chol<false,W_type>(Abb,piv,thresh);
      W() = nda::conj(W());  // need to conjugate before inverting, since W is C_ordered
      int nc=piv(block_size);
      for(int v=0; v<nc; ++v)
        app_log(3,"  {}  {} ",nchol+v,std::real(W(v,v)*std::conj(W(v,v))));
      nda::inverse_in_place(W);
      Abb(nda::range(0,nc),nda::range(0,nc)) = W(all,all);
    }
    if(comm.size()>1) comm.broadcast_n(piv.data(),block_size+1);
    if(comm.size()>1) comm.broadcast_n(Abb.data(),block_size*block_size);

     // number of linearly independent cholesky vectors found
    long newv = piv(block_size); 
    utils::check( newv > 0, "Failed to find cholesky vector.");
    newv = std::min(newv, ndim-nchol); 

    auto Rn = CholMat.local()(nda::range(nchol,nchol+newv), all);    // new cholesky vector 
    for(int i=0; i<newv; i++) {
      utils::check(piv(i) >= i, "Failed condition: piv(i) >= i");
      if(i != piv(i))
        Rloc(i,all) = Rloc(piv(i),all);
    }
    nda::blas::gemm(Abb(nda::range(0,newv),nda::range(0,newv)),
                    Rloc(nda::range(0,newv),all), Rn);
    
    // increase counter
    nchol+=newv;
    if(nchol == ndim) break;

    // update diagonal
    // Diag(r) -= R(p,r) * std::conj(R(p,r))
    if constexpr (MEM==HOST_MEMORY) {
      for( auto v : itertools::range(newv) )
        for( auto [ir,r] : itertools::enumerate(a_rng) )
          Diag(ir) -= std::conj(Rn(v,ir)) * Rn(v,ir);
    } else {
      nda::tensor::contract(ComplexType(-1.0), nda::conj(Rn), "vr", Rn, "vr", ComplexType(1.0), Diag,"r");
    }

    // find index and value of maximum element  
    utils::max_element_multi(Diag,lmax_res_val,lmax_res_indx);
    utils::find_distributed_maximum(comm,lmax_res_val,gmax_res); 

    auto curr_max = find_max();
    utils::check( old_max >= curr_max,
          std::string("Error: Cholesky algorithm failed in utils::distributed_cholesky. \n") +
          std::string("       Found non-decreasing residual error: last it:{}, curr it:{}"),old_max,curr_max);
    utils::check( std::isfinite(curr_max),
          std::string("Error: Cholesky algorithm failed in utils::distributed_cholesky. \n") +
          std::string("       Found invalid residual:{}"),curr_max);
  
    // stopping condition when thresh is set
    if( thresh > 0.0 and thresh > curr_max) break;
    old_max = curr_max; 
  }
  utils::check(nchol > 0, "Error: Found nchol=0 in utils::distributed_cholesky.");

  auto Rt = math::nda::make_distributed_array<CArray_2D_t>(comm,
                  {comm.size(),1},{ndim,ndim},{1,1});
  utils::check(a_rng == Rt.local_range(0), "Range mismatch.");
  Rt.local() = nda::dagger(CholMat.local());
  return Rt;

}

}

#endif
