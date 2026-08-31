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


#ifndef METHODS_EMBEDDING_CHOLESKY_HPP
#define METHODS_EMBEDDING_CHOLESKY_HPP

#include <tuple>
#include <cmath>
#include <iomanip>
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

namespace methods
{

/***
 *  Evaluation of diagonal and columns are provided by functor routines.
 * Assumes a reduction is needed over the full communicator.
 * Cholesky vectors are stored and returned in distribured array.
 * Employs a convention consistent with the calculation of V_abcd in downfold_V, which is not
 * consistent with the convention of ERI::cholesky.
 * In this routine:  (ab|cd) = sum_n conj(U_n_ba) * U_n_cd, which is consistent with downfold_V.
 * ERI::cholesky uses: (ab|cd) = sum_n L_n_ab * conj(L_n_dc), the relation is: L_n_ab = conj(U_n_ba).
 ***/
template<MEMORY_SPACE MEM, typename functor_diag_t, typename functor_col_t> 
auto embed_cholesky(utils::Communicator auto &comm, long nOrb, functor_diag_t &diag_f, functor_col_t &col_f, 
                    double thresh = 1e-10, int block_size = 32)
{
  using CArray_2D_t = memory::array<MEM,ComplexType,2>;
  decltype(nda::range::all) all;
  std::string err_("Error in embed_cholesky: ");

  utils::TimerManager Timer;

  // storage space, to be adjusted dynamically later on
  long nmax = 5*nOrb; 
  long nab = nOrb*nOrb;

  //MAM: This assumes that comm.size() >= nab, need to modify implementation for cases
  //     where nab is small. Maybe in this case generate Vabcd first and factorize 
  //     with separate routine (in serial to simplify coding).
  // store cholesky matrix
  auto CholMat = math::nda::make_distributed_array<CArray_2D_t>(comm,
                      {1,comm.size()},{nmax,nab},{1,1});
  auto ab_rng = CholMat.local_range(1);  

  // adjust block_size
  block_size = std::min(block_size, int(ab_rng.size())); 
  comm.all_reduce_in_place_n(&block_size,1,mpi3::min<>{});

  nda::array<long, 1> global_indx(block_size,0);
  nda::array<RealType,1> lmax_res_val(block_size,0.0);
  nda::array<long, 1> lmax_res_indx(block_size,0);
  nda::array<std::pair<RealType,int>,1> gmax_res(block_size);

  // working arrays to store list of indexes in current iteration and associated columns
  // remember that we assume results from computation kernels need to be reduce
  auto R = CArray_2D_t::zeros({block_size,nab});
  auto Rc = CArray_2D_t::zeros({nmax,block_size});

  // calculate diagonal and reduce over processor grid
  memory::array<MEM,ComplexType,1> Diag(ab_rng.size(),0.0);
  diag_f(R(0,all));
  comm.all_reduce_in_place_n(R.data(),nab,std::plus<>{});
  Diag() = R(0,ab_rng);

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

    utils::check( std::isfinite(old_max), 
          err_ + "Cholesky algorithm failed in embed::cholesky. \n" + 
          "       Found invalid residual:{}",old_max); 

    // compute columns for indexes associated with lmax_res_indx
    global_indx() = 0;
    for( int n=0; n<block_size; ++n ) {
      if( gmax_res(n).second/block_size == comm.rank() ) 
        global_indx(n) = ab_rng.first() + lmax_res_indx(gmax_res(n).second%block_size);  
    }
    comm.all_reduce_in_place_n(global_indx.data(),global_indx.size(),std::plus<>{});
    col_f(global_indx,R);
    comm.all_reduce_in_place_n(R.data(),R.size(),std::plus<>{});

    // view into local range
    auto Rloc = R(all,ab_rng);

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
    int newv = piv(block_size); 
    utils::check( newv > 0, "Failed to find cholesky vector.");

    // resize data structures
    if(nchol+newv > nmax) {
      int nmax_new = nmax + 2*nab;
      Rc = CArray_2D_t(nmax_new,block_size);
      auto CholMat_ = math::nda::make_distributed_array<CArray_2D_t>(comm,
                      {1,comm.size()},{nmax_new,nab},{1,1});
      utils::check(ab_rng == CholMat_.local_range(1), "Range mismatch.");
      CholMat_.local()(nda::range(nchol),all) = CholMat.local()(nda::range(nchol),all);
      CholMat = std::move(CholMat_);
      nmax = nmax_new;
    }

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

    // update diagonal
    // Diag(r) -= R(p,r) * std::conj(R(p,r))
    if constexpr (MEM==HOST_MEMORY) {
      for( auto v : itertools::range(newv) )
        for( auto [ir,r] : itertools::enumerate(ab_rng) )
          Diag(ir) -= std::conj(Rn(v,ir)) * Rn(v,ir);
    } else {
      nda::tensor::contract(ComplexType(-1.0), nda::conj(Rn), "vr", Rn, "vr", ComplexType(1.0), Diag,"r");
    }

    // find index and value of maximum element  
    utils::max_element_multi(Diag,lmax_res_val,lmax_res_indx);
    utils::find_distributed_maximum(comm,lmax_res_val,gmax_res);

    auto curr_max = find_max();
    utils::check( old_max >= curr_max,
          std::string("Error: Cholesky algorithm failed in embed::cholesky. \n") +
          std::string("       Found non-decreasing residual error: last it:{}, curr it:{}"),old_max,curr_max);
    utils::check( std::isfinite(curr_max),
          std::string("Error: Cholesky algorithm failed in embed::cholesky. \n") +
          std::string("       Found invalid residual:{}"),curr_max);

    // stopping condition when thresh is set
    if( thresh > 0.0 and thresh > curr_max) break;
    old_max = curr_max; 
  }
  utils::check(nchol > 0, "Error: Found nchol=0 in embed_cholesky.");

  auto Rt = math::nda::make_distributed_array<CArray_2D_t>(comm,
                  {1,comm.size()},{nchol,nab},{1,1});
  utils::check(ab_rng == Rt.local_range(1), "Range mismatch.");
  Rt.local() = CholMat.local()(nda::range(nchol),all);
  return Rt; 

}

/***
 * see note above in cholesky routine.
 * Writes distributed Cholesku matrix to h5 file, with a format consistenn with ERI::cholesky.
 * If conjugate and/or transpose are set to true, applies conjugate and/or transposition before writting.
 ***/  
void write_cholesky_embed(h5::group &grp, std::string name,
                          math::nda::DistributedArrayOfRank<2> auto const& dV_nab,
                          bool transpose = false, bool conjugate = false) 
{
  // write in segments if this is too large to gather in a single node!!!
  int nchol = dV_nab.global_shape()[0];
  int nOrb = int(std::round(std::sqrt(dV_nab.global_shape()[1])));
  utils::check(dV_nab.global_shape()[1] == nOrb*nOrb, "Error: Not squared matrix! Oh Oh.");
  nda::array<ComplexType,2> T;
  auto comm = dV_nab.communicator();
  if(comm->root()) T = nda::array<ComplexType,2>(dV_nab.global_shape());  
  math::nda::gather(0,dV_nab,std::addressof(T));
  if(comm->root()) {
    if(conjugate) T() = nda::conj(T);
    // {nchol, nspin, nkpts, nbnd, nbnd}, assuming single spin, single kpoint for now
    auto T5 = nda::reshape(T,std::array<long,5>{nchol,1,1,nOrb,nOrb});
    if(transpose) {
      for(int n=0; n<nchol; ++n)
        for(int a=0; a<nOrb; ++a)
          for(int b=a+1; b<nOrb; ++b)
            std::swap(T5(n,0,0,a,b),T5(n,0,0,b,a));
    }
    nda::h5_write(grp,name,T5,false);
  }
  comm->barrier();
}

}

#endif
