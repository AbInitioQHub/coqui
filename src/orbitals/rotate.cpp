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


#include "configuration.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "utilities/concepts.hpp"

#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/linalg.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/distributed_array/nda.hpp"

#include "hamiltonian/one_body_hamiltonian.hpp"

namespace orbitals
{

using boost::mpi3::communicator;
using memory::darray_t;

namespace detail 
{

// diagonalizations are done in serial, round-robin over local kpoints
template<MEMORY_SPACE MEM = HOST_MEMORY, utils::Communicator comm_t>
void orthonormalize_serial_impl(darray_t<memory::array<MEM, ComplexType, 4>, comm_t>& psi,
                    double cutoff,
                    nda::array<int,2> const& ranges)
{
  decltype(nda::range::all) all;
  long nkpts_tot = psi.global_shape()[1];;
  long nbnd      = psi.global_shape()[2];
  auto comm = psi.communicator();
  int nranges = ranges.extent(0);

  utils::check(nranges>0, "Error in orthonormalize: ranges.extent(0) == 0"); 
  utils::check(nranges == 1 or cutoff==0.0, "Error in orthonormalize: Multiple ranges currently require cutoff==0.0");
  // no distribution over bands in this implementation
  utils::check(psi.grid()[2]==1, "Error: orthonormalize_serial_impl expects no band distribution.");

  long color = psi.origin()[0]*nkpts_tot+psi.origin()[1]; 
  auto k_comm = comm->split(color,comm->rank());

  auto s_range = psi.local_range(0);
  auto k_range = psi.local_range(1);

  // orthonormalize and (possibly) truncate psi
  // MAM: since the code currently requires the same number of orbitals per kpoint, 
  //      we truncate based on the smallest set of orbitals in any kpoint
  long nremove = 0;
  // check ranges and calculate buffer size
  long sn = 0, sz_max=0; 
  for( long i=0; i<nranges; ++i ) { 
    long a = ranges(i,0), b = ranges(i,1);
    utils::check( a >= 0 and a <= b and b <= nbnd, "Error: range mismatch: nbnd:{}, range:({},{})",nbnd,a,b);
    sn += (b-a)*(b-a);
    sz_max = std::max(sz_max, (b-a));
  }
  // check for overlapping ranges
  for( long i=0; i<nranges; ++i ) {
    auto a = ranges(i,0), b = ranges(i,1);
    for( long j=i+1; j<nranges; ++j ) { 
      auto c = ranges(j,0), d = ranges(j,1);
      utils::check( b <= c or d <= a, "Error in orthogonalize: Overlapping ranges: range:({},{}), range:({},{})",a,b,c,d);   
    }
  }
  memory::array<MEM, ComplexType, 2> psi_tmp(sz_max,psi.local_shape()[3]);

  auto buff = memory::array<MEM,ComplexType,2>::zeros({2,sn});
  auto bf0 = buff(0,all);
  auto bf1 = buff(1,all);

  {
    nda::array<int,2> idx(k_comm.size(),2);
    idx() = -1;
    long cnter=0;
    auto psi_loc = psi.local();

    for( auto [is,s] : itertools::enumerate(s_range) ) {
      for( auto [ik,k] : itertools::enumerate(k_range) ) {

        //
        sn = 0;
        for( long n=0; n<nranges; ++n ) {
          long a = ranges(n,0), b = ranges(n,1);
          auto rng = nda::range(a,b); 
          if(a==b) continue;
          memory::array_view<MEM,ComplexType,2> S0({b-a,b-a}, bf0.data()+sn);
          nda::blas::gemm(psi_loc(is,ik,rng,all),nda::dagger(psi_loc(is,ik,rng,all)),S0);
          sn += (b-a)*(b-a);
        }
        if(cnter == k_comm.rank()) bf1() = ComplexType(0.0);
        k_comm.reduce_n(bf0.data(),bf0.size(),bf1.data(),std::plus<>{},cnter);
        idx(cnter,0) = is;
        idx(cnter++,1) = ik;
         
        if(cnter == k_comm.size()) {

          // diagonalize S1
          sn = 0; 
          for( long n=0; n<nranges; ++n ) {
            long a = ranges(n,0), b = ranges(n,1);
            if(a==b) continue;
            memory::array_view<MEM,ComplexType,2> S1({b-a,b-a}, bf1.data()+sn);
            auto v_d = nda::linalg::detail::_eigen_element_impl(nda::transpose(S1),'V');
            auto v = nda::to_host(v_d);
            auto it = std::find_if(v.begin(),v.end(),[&](auto&& x){return x >= cutoff;});
            auto d = std::distance(v.begin(),it);
            utils::check(d >= 0 and d < (b-a), "Error: Problems with eigenvalue decomposition of overlap matrix. is:{}, ik:{}, d:{}, v0:{}, vN:{}, cutoff:{}",
                         idx(k_comm.rank(),0),idx(k_comm.rank(),1),d,v[0],v[b-a-1],cutoff);
            nremove = std::max(nremove,d);
            for(long i=d; i<(b-a); ++i)
              nda::tensor::scale(ComplexType(1.0/std::sqrt(v(i))), S1(i,all));
            sn += (b-a)*(b-a);
          }

          // rotate psi
          auto idx_it = idx.begin();
          for( int n=0; n<cnter; ++n, idx_it+=2) {
            utils::check( *idx_it >=0 and *(idx_it+1) >=0, " Error: Logic error!");
            if(n == k_comm.rank()) bf0 = bf1;
            k_comm.broadcast_n(bf0.data(),bf0.size(),n); 

            sn = 0;     
            for( long m=0; m<nranges; ++m ) {
              long a = ranges(m,0), b = ranges(m,1);
              auto rng = nda::range(a,b);
              if(a==b) continue;
              memory::array_view<MEM,ComplexType,2> S0({b-a,b-a}, bf0.data()+sn);
              nda::blas::gemm(S0,psi_loc(*idx_it,*(idx_it+1),rng,all),psi_tmp(nda::range(b-a),all));
              psi_loc(*idx_it,*(idx_it+1),rng,all) = psi_tmp(nda::range(b-a),all);
              sn += (b-a)*(b-a);
            }
          }
          cnter=0;
          idx() = -1;

        } // if(cnter == k_comm.size())  

      }
    }

    if( cnter > 0 ) { 
      // diagonalize S1
      if(cnter > k_comm.rank()) {
        sn = 0;
        for( long n=0; n<nranges; ++n ) {
          long a = ranges(n,0), b = ranges(n,1);
          if(a==b) continue;
          memory::array_view<MEM,ComplexType,2> S1({b-a,b-a}, bf1.data()+sn);
          auto v_d = nda::linalg::detail::_eigen_element_impl(nda::transpose(S1),'V');
          auto v = nda::to_host(v_d);
          auto it = std::find_if(v.begin(),v.end(),[&](auto&& x){return x >= cutoff;});
          auto d = std::distance(v.begin(),it);
          utils::check(d >= 0 and d < b-a, "Error: Problems with eigenvalue decomposition of overlap matrix. is:{}, ik:{}, d:{}, v0:{}, vN:{}, cutoff:{}",
                         idx(k_comm.rank(),0),idx(k_comm.rank(),1),d,v[0],v[b-a-1],cutoff);
          nremove = std::max(nremove,d);
          for(long i=d; i<(b-a); ++i)
            nda::tensor::scale(ComplexType(1.0/std::sqrt(v(i))), S1(i,all));
          sn += (b-a)*(b-a);
        }
      }
      // rotate psi
      auto it = idx.begin();
      for( int n=0; n<cnter; ++n, it+=2) {
        utils::check( *it >=0 and *(it+1) >=0, " Error: Logic error!");
        if(n == k_comm.rank()) bf0 = bf1;
        k_comm.broadcast_n(bf0.data(),bf0.size(),n);

        sn = 0;
        for( long m=0; m<nranges; ++m ) {
          long a = ranges(m,0), b = ranges(m,1);
          auto rng = nda::range(a,b);
          if(a==b) continue;
          memory::array_view<MEM,ComplexType,2> S0({b-a,b-a}, bf0.data()+sn);
          nda::blas::gemm(S0,psi_loc(*it,*(it+1),rng,all),psi_tmp(nda::range(b-a),all));
          psi_loc(*it,*(it+1),rng,all) = psi_tmp(nda::range(b-a),all);
          sn += (b-a)*(b-a);
        }
      }
    } // if( cnter > 0 ) 

    nremove = comm->all_reduce_value(nremove,boost::mpi3::max<>{});
    utils::check(nranges == 1 or nremove==0, "Error in orthonormalize: Multiple ranges currently require nremove==0");
    app_log(3,"Removing {} basis functions from PGTO basis.",nremove);
  }
  
  if(nremove > 0)
  {
    utils::check(ranges(0,0)==0 and ranges(0,1)==nbnd, 
                 "Error in orthogonalization: nremove>0 in finite band range. Not yet implemented.");
    auto psit = math::nda::make_distributed_array<memory::array<MEM,ComplexType,4>>(*comm,
           psi.grid(),{psi.global_shape()[0],psi.global_shape()[1],
            nbnd-nremove,psi.global_shape()[3]},
           {psi.block_size()[0],psi.block_size()[1],nbnd-nremove,psi.block_size()[3]});
    utils::check(psi.local_range(0) == psi.local_range(0), "Range mismatch");
    utils::check(psi.local_range(1) == psi.local_range(1), "Range mismatch");
    utils::check(psi.local_range(3) == psi.local_range(3), "Range mismatch");
    // copy data 
    psit.local() = psi.local()(all,all,nda::range(nremove,nbnd),all);
    // move into original variable
    psi = std::move(psit);
  }

}

// diagonalizations are done in parallel with distributed operations
// MAM: no infrastructure to use slate::HermitianMatrix, which is necessary for eig calls. 
//      Use herk with C being a slate::HermitianMatrix 
/*
template<MEMORY_SPACE MEM = HOST_MEMORY, utils::Communicator comm_t>
void orthonormalize_distr_impl(darray_t<memory::array<MEM, ComplexType, 4>, comm_t>& psi_full,
                    double cutoff,
                    nda::array<int,2> const& ranges)
{
  utils::check(false," Error: orthonormalize with band distribution not yet implemented.");
  // turn off warnings until I finish implementation
  using larray = memory::array<MEM, ComplexType, 4>;
  decltype(nda::range::all) all;
  long nkpts_tot = psi_full.global_shape()[1];
  long nbnd      = psi_full.global_shape()[2];
  long nnr       = psi_full.global_shape()[3];
  auto pgrid     = psi_full.grid();
  auto bz        = psi_full.block_size();
  auto comm      = psi_full.communicator();

  long color = psi_full.origin()[0]*nkpts_tot+psi_full.origin()[1]; 
  auto k_comm = comm->split(color,comm->rank());

  auto s_range = psi_full.local_range(0);
  auto k_range = psi_full.local_range(1);
  long nspin = s_range.size();
  long nkpts = k_range.size();

  // orthonormalize and truncate psi
  // MAM: since the code currently requires the same number of orbitals per kpoint, 
  //      we truncate based on the smallest set of orbitals in any kpoint
  long nremove = 0;
  long bz0 = std::min( nbnd/pgrid[2],nbnd/pgrid[3] );
  auto S = math::nda::make_distributed_array<memory::array<MEM,ComplexType,4>>(k_comm,
          {1,1,pgrid[2],pgrid[3]},{nspin,nkpts,nbnd,nbnd},
          {1,1,bz0,bz0},true);
  memory::darray_view_t<larray,comm_t> psi(std::addressof(k_comm),S.grid(),
          {nspin,nkpts,nbnd,nnr},{0,0,psi_full.origin()[2],psi_full.origin()[3]},
          {1,1,bz[2],bz[3]},psi_full.local()); 
  {
    //math::nda::slate_ops::multiply(psi,math::nda::H(psi),S);

    //slate::invert()

    for( int is=0, isk=0; is<nspin; is++ ) {
      for( int ik=0; ik<nkpts; ik++, isk++ ) {
        if( isk%k_comm.size() == k_comm.rank() ) {
          auto v_d = nda::linalg::detail::_eigen_element_impl(nda::transpose(S(is,ik,all,all)),'V');
          auto v = nda::to_host(v_d);
          auto it = std::find_if(v.begin(),v.end(),[&](auto&& x){return x >= cutoff;});
          auto d = std::distance(v.begin(),it);
          utils::check(d >= 0 and d < nbnd, "Error: Problems with eigenvalue decomposition of overlap matrix. is:{}, ik:{}, d:{}, v0:{}, vN:{}, cutoff:{}",is,ik,d,v[0],v[nbnd-1],cutoff);
          nremove = std::max(nremove,d);
          for(long i=d; i<nbnd; ++i)
            nda::tensor::scale(ComplexType(1.0/std::sqrt(v(i))), S(is,ik,i,all));
        }
      }
    }
    k_comm.all_reduce_in_place_n(S.data(),S.size(),std::plus<>{});
    nremove = comm->all_reduce_value(nremove,boost::mpi3::max<>{});
    app_log(3,"Removing {} basis functions from PGTO basis.",nremove);
  }
  {
    auto psi_loc = psi.local();
    auto psit = math::nda::make_distributed_array<memory::array<MEM,ComplexType,4>>(*comm,
           psi.grid(),{psi.global_shape()[0],psi.global_shape()[1],
            nbnd-nremove,psi.global_shape()[3]},
           {nspin,nkpts,nbnd-nremove,psi.block_size()[3]});
    utils::check(psi.local_range(0) == psi.local_range(0), "Range mismatch");
    utils::check(psi.local_range(1) == psi.local_range(1), "Range mismatch");
    utils::check(psi.local_range(3) == psi.local_range(3), "Range mismatch");
    auto psit_loc = psit.local();

    for( int is=0; is<nspin; is++ )
      for( int ik=0; ik<nkpts; ik++ )
        nda::blas::gemm(S(is,ik,nda::range(nremove,nbnd),all),psi_loc(is,ik,all,all),
                        psit_loc(is,ik,all,all));
    psi = std::move(psit);
  }

  //check that matrix is indeed orthonormal
  if(check) {  
    S() = ComplexType(0.0);
    if constexpr (MEM==HOST_MEMORY) {
      auto psi_loc = psi.local();
      for( auto [is,s] : itertools::enumerate(s_range) ) 
        for( auto [ik,k] : itertools::enumerate(k_range) ) 
          nda::blas::gemm(psi_loc(is,ik,all,all),nda::dagger(psi_loc(is,ik,all,all)),
                          S(is,ik,nda::range(nbnd-nremove),nda::range(nbnd-nremove)));
    } else {
      auto psi_loc = psi.local();
      nda::tensor::contract(psi_loc,"skag",nda::conj(psi_loc),"skbg",
                            S(all,all,nda::range(nbnd-nremove),nda::range(nbnd-nremove)),"skab");
    }

    k_comm.all_reduce_in_place_n(S.data(),S.size(),std::plus<>{});
    double e = 0.0;
    for( auto [is,s] : itertools::enumerate(s_range) ) 
      for( auto [ik,k] : itertools::enumerate(k_range) ) 
        for(int i=0; i<nbnd-nremove; i++)
          for(int j=0; j<nbnd-nremove; j++) {
            if (std::abs(S(is, ik, i, j) - (i == j ? ComplexType(1.0) : ComplexType(0.0))) > 1e-8)
              app_warning("Error: Matrix not orthogonal: is:{}, ik:{}, i:{}, j:{}, S(ik,i,j):{} ", 
                      is+s_range.first(), ik+k_range.first(), i, j, S(is, ik, i, j));
          }
  }
}
*/

/*
// diagonalizations are done in serial, round-robin over local kpoints
// there is room for speedup if we get read of constant reallocations, but requiring much more memory 
template<MEMORY_SPACE MEM = HOST_MEMORY, utils::Communicator comm_t>
auto canonicalize_diagonal_serial_impl(mf::MF& mf, 
                    darray_t<memory::array<MEM, ComplexType, 4>, comm_t>& psi)
{
  using larray = memory::array<MEM, ComplexType, 4>;
  decltype(nda::range::all) all;
  long nspin_tot = psi.global_shape()[0]; 
  long nkpts_tot = psi.global_shape()[1];;
  long nbnd_tot  = psi.global_shape()[2];
  long npwx      = psi.global_shape()[3];
  auto bz        = psi.block_size();
  auto pgrid     = psi.grid();
  auto comm = psi.communicator();

  APP_ABORT("Error: canonicalize_diagonal_serial_impl disabled. Contact developers.");

  // this constrain currently comes from implementation of gen_F<>(...)
  utils::check(psi.grid()[2]==1, "Error: canonicalize_diagonal_serial_impl expects no k distribution.");
  utils::check(pgrid[3] <= nbnd_tot, "canonicalize_diagonal_serial_impl: Too many processors in distribution over G vectors.");

  long color = psi.origin()[0]*nkpts_tot+psi.origin()[1]; 
  auto k_comm = comm->split(color,comm->rank());

  auto s_range = psi.local_range(0);
  auto k_range = psi.local_range(1);
  long s0 = s_range.first();
  long k0 = k_range.first();
  long nspin = s_range.size();
  long nkpts = k_range.size();

  // return eigenvalues 
  auto eigV = nda::array<ComplexType, 3>::zeros({nspin_tot,nkpts_tot,nbnd_tot});

  memory::array<MEM, ComplexType, 2> psi_tmp(nbnd_tot,psi.local_shape()[3]);
  auto Fl = memory::array<MEM,ComplexType,4>::zeros({2,1,nbnd_tot,nbnd_tot});
  auto F0 = Fl(nda::range(0,1),nda::ellipsis{});
  auto F1 = Fl(1,0,all,all);
  // redistribute over bands, since gen_F requires full G vectors
  auto psi_k = math::nda::make_distributed_array<larray>(k_comm,{1,1,pgrid[3],1},
            {1,1,nbnd_tot,npwx},{1,1,2048,2048});
  auto psi_loc = psi.local();
  {
    nda::array<int,2> idx(k_comm.size(),2);
    idx() = -1;
    long cnter=0;

    for( auto [is,s] : itertools::enumerate(s_range) ) {
      for( auto [ik,k] : itertools::enumerate(k_range) ) {

        { // redistribute into psi_k
          memory::darray_view_t<larray,comm_t> psit(std::addressof(k_comm),{1,1,1,pgrid[3]},
            {1,1,nbnd_tot,npwx},{0,0,0,psi.origin()[3]},
            {1,1,bz[2],bz[3]},psi.local()(nda::range(is,is+1),nda::range(ik,ik+1),all,all)); 
          math::nda::redistribute(psit,psi_k);
        }
        // you could avoid all the reallocations by copying body of gen_F here!
        auto Fd = hamilt::detail::gen_F<MEM>(mf,k_comm,nda::range(k,k+1),psi_k);
        if(cnter == k_comm.rank()) F0() = ComplexType(0.0);
        math::nda::gather(cnter, Fd, std::addressof(F0)); 
        idx(cnter,0) = is;
        idx(cnter++,1) = ik;
         
        if(cnter == k_comm.size()) {

          // diagonalize F0 
          auto v_d = nda::linalg::detail::_eigen_element_impl(nda::transpose(F0(0,0,all,all)),'V');
          eigV( idx(k_comm.rank(),0)+s0, idx(k_comm.rank(),1)+k0, all ) = v_d(); 
          for( auto& v: F0 ) v = std::conj(v);
          
          // rotate psi
          auto idx_it = idx.begin();
          for( int n=0; n<cnter; ++n, idx_it+=2 ) {
            utils::check( *idx_it >=0 and *(idx_it+1) >=0, " Error: Logic error!");
            if(n == k_comm.rank()) F1 = F0(0,0,all,all);
            k_comm.broadcast_n(F1.data(),F1.size(),n); 
            nda::blas::gemm(F1,psi_loc(*idx_it,*(idx_it+1),all,all),psi_tmp);
            psi_loc(*idx_it,*(idx_it+1),all,all) = psi_tmp;
          }

          cnter=0;
          idx() = -1;

        } // if(cnter == k_comm.size())  

      }
    }

    if( cnter > 0 ) { 

      // diagonalize F0 
      if(cnter > k_comm.rank()) { 
        auto v_d = nda::linalg::detail::_eigen_element_impl(nda::transpose(F0(0,0,all,all)),'V');
        eigV( idx(k_comm.rank(),0)+s0, idx(k_comm.rank(),1)+k0, all ) = v_d(); 
        for( auto& v: F0 ) v = std::conj(v);
      }

      // rotate psi
      auto idx_it = idx.begin();
      for( int n=0; n<cnter; ++n, idx_it+=2 ) {
        utils::check( *idx_it >=0 and *(idx_it+1) >=0, " Error: Logic error!");
        if(n == k_comm.rank()) F1 = F0(0,0,all,all);
        k_comm.broadcast_n(F1.data(),F1.size(),n);
        nda::blas::gemm(F1,psi_loc(*idx_it,*(idx_it+1),all,all),psi_tmp);
        psi_loc(*idx_it,*(idx_it+1),all,all) = psi_tmp;
      }

    } // if( cnter > 0 ) 

  }
  comm->all_reduce_in_place_n(eigV.data(),eigV.size(),std::plus<>{});
  return eigV;
}
*/

} // detail


template<MEMORY_SPACE MEM = HOST_MEMORY, utils::Communicator comm_t>
void orthonormalize(memory::darray_t<memory::array<MEM, ComplexType, 4>, comm_t>& psi,
                    double cutoff)  // , bool serial)
{
  nda::array<int,2> ranges(1,2);
  ranges(0,0) = 0;
  ranges(0,1) = psi.global_shape()[2];
  if(psi.grid()[2]==1)
    detail::orthonormalize_serial_impl(psi,cutoff,ranges);
  else // distr needs slate eigenvectors, needs some further work!
    utils::check(false," Error: orthonormalize with band distribution not yet implemented.");
//    detail::orthonormalize_distr_impl(psi,cutoff,ranges);
}

template<MEMORY_SPACE MEM = HOST_MEMORY, utils::Communicator comm_t>
void orthonormalize(nda::array<int,2> const& ranges,
                    memory::darray_t<memory::array<MEM, ComplexType, 4>, comm_t>& psi)
{
  if(psi.grid()[2]==1)
    detail::orthonormalize_serial_impl(psi,0.0,ranges);
  else // distr needs slate eigenvectors, needs some further work!
    utils::check(false," Error: orthonormalize with band distribution not yet implemented.");
//    detail::orthonormalize_distr_impl(psi,0.0,ranges);
}

template<MEMORY_SPACE MEM = HOST_MEMORY, utils::Communicator comm_t>
auto canonicalize_diagonal_basis(mf::MF& mf, 
                   memory::darray_t<memory::array<MEM, ComplexType, 4>, comm_t>& psi)
 -> nda::array<ComplexType, 3>
{
  APP_ABORT("Error: canonicalize_diagonal_basis disabled. Contact developers.");
  // turn off unused warnings 
  (void) mf;
  (void) psi; 
/*
  if(psi.grid()[2]==1)
    return detail::canonicalize_diagonal_serial_impl(mf,psi);
  else // distr needs slate eigenvectors, needs some further work!
    utils::check(false,"Finish!: canonicalize_diagonal_distr_impl");
//    detail::canonicalize_diagonal_distr_impl(mf,psi);
*/
  return nda::array<ComplexType, 3>(0,0,0);
}

// Instantiations
using memory::host_array;
template void orthonormalize(darray_t<host_array<ComplexType, 4>, communicator>&,double);
template void orthonormalize(nda::array<int,2> const&,
                    darray_t<host_array<ComplexType, 4>, communicator>&);
template nda::array<ComplexType, 3> canonicalize_diagonal_basis(mf::MF& mf,darray_t<host_array<ComplexType, 4>, communicator>&);
/*
#if defined(ENABLE_DEVICE)
using memory::device_array;
using memory::unified_array;
template void orthonormalize(darray_t<device_array<ComplexType, 4>, communicator>&,double);
                    darray_t<device_array<ComplexType, 4>, communicator>&,double);
template void orthonormalize(darray_t<unified_array<ComplexType, 4>, communicator>&,double);
template void orthonormalize(nda::array<int,2> const&,
                    darray_t<device_array<ComplexType, 4>, communicator>&);
template void orthonormalize(nda::array<int,2> const&,
                    darray_t<unified_array<ComplexType, 4>, communicator>&);
template nda::array<ComplexType, 3> canonicalize_diagonal_basis(mf::MF& mf,darray_t<device_array<ComplexType, 4>, communicator>&);
template nda::array<ComplexType, 3> canonicalize_diagonal_basis(mf::MF& mf,darray_t<unified_array<ComplexType, 4>, communicator>&);
#endif
*/
} // orbitals

