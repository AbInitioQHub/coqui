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


#ifndef UTILITIES_POTENTIAL_MATRIX_ELEMENTS_HPP
#define UTILITIES_POTENTIAL_MATRIX_ELEMENTS_HPP

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "grids/g_grids.hpp"
#include "utilities/proc_grid_partition.hpp"
#include "hamiltonian/potentials.hpp"
#include "nda/nda.hpp"
#include "numerics/fft/nda.hpp"
#include "numerics/distributed_array/nda.hpp"

namespace hamilt
{

/*
 * Computes V_ab = conj(< P(a,r)| V(r,r') | P(b,r) >) = sum_G P(a,G) * v(G) * conj(P(b,G))
 *   where P(a,G) = fft( P(a,r) )
 *
 *   grid_t: grid object containing the reciprocal cell vectors, G.
 *
 * GPU???
 */ 
template<nda::MemoryArrayOfRank<4> Arr_t, 
         class Vec = std::array<double,1>>
auto matrix_elements_potential_g(Arr_t& P,
		std::string basis,
	 	grids::truncated_g_grid const& g,
		nda::ArrayOfRank<1> auto const& kp,
		nda::ArrayOfRank<1> auto const& kq,
                std::string type = "coulomb",
                Vec const& params = {})
{
  // cpu only for now
  using T = typename std::decay_t<Arr_t>::value_type;
  using Array_t = nda::array<T,2>; 

  /*   Evaluate potential */
  long ngm = g.size();
  nda::array<T,1> Vg(ngm,0.0);
  potential_g(Vg,g.g_vectors(),kp,kq,type,params);

  utils::check( P.shape(1) == g.mesh(0), "Dimension mismatch.");
  utils::check( P.shape(2) == g.mesh(1), "Dimension mismatch.");
  utils::check( P.shape(3) == g.mesh(2), "Dimension mismatch.");

  long np = P.shape(0);
  Array_t V{{np,np}};

  if( basis == "r" ) {
    // fft r->G
    math::fft::fwdfft_many(P);
  } else if(basis != "g") {
    APP_ABORT("Error in potential_matrix_elements: Invalid basis: {}",basis);
  }

  auto P2d = nda::reshape(P, std::array<long,2>{ np, g.mesh(0)*g.mesh(1)*g.mesh(2) });
  Array_t Pg{ {np, ngm} };
  for( int m=0; m<np; ++m)
    for( auto [i,n] : itertools::enumerate(g.gv_to_fft()) )
      Pg(m,i) = std::sqrt(Vg(i)) * P2d(m,n);    
  nda::blas::gemm(Pg,nda::dagger(Pg),V);

  return V;
}

template<nda::MemoryArrayOfRank<4> Arr_t,
         class Vec = std::array<double,1>>
auto matrix_elements_potential_full_g(Arr_t& P,
                std::string basis,
                nda::ArrayOfRank<2> auto const& recv,
		nda::ArrayOfRank<1> auto const& kp,
		nda::ArrayOfRank<1> auto const& kq,
                std::string type = "coulomb",
                Vec const& params = {})
{ 
  // cpu only for now
  using T = typename std::decay_t<Arr_t>::value_type;
  using Array_t = nda::array<T,2>;

  long nnr = P.shape(1)*P.shape(2)*P.shape(3);
  nda::array<long,1> mesh = {P.shape(1),P.shape(2),P.shape(3)};
  utils::check( nnr>0, "Dimension mismatch.");
  
  /*   Evaluate potential */
  nda::array<T,1> Vg(nnr,0.0);
  potential_full_g(Vg,mesh,recv,kp,kq,type,params);
  
  long np = P.shape(0);
  Array_t V{{np,np}};
  
  if( basis == "r" ) {
    // fft r->G
    math::fft::fwdfft_many(P); 
  } else if(basis != "g") {
    APP_ABORT("Error in potential_matrix_elements: Invalid basis: {}",basis);
  }
  
  auto P2d = nda::reshape(P, std::array<long,2>{ np, nnr });
  for( long m=0; m<np; ++m)
    for( long i=0; i<nnr; ++i ) 
      P2d(m,i) = std::sqrt(Vg(i)) * P2d(m,i);
  nda::blas::gemm(P2d,nda::dagger(P2d),V);

  return V;
}

template<math::nda::DistributedArrayOfRank<3> Arr_t,
class Vec = std::array<double,1>>
void matrix_elements_potential_g(Arr_t& P,
		Arr_t& V,
                std::string basis,
                grids::truncated_g_grid const& g,
                ::nda::ArrayOfRank<1> auto const& mesh,
                ::nda::ArrayOfRank<2> auto const& recv,
                ::nda::ArrayOfRank<2> auto const& Q,
                std::string type = "coulomb",
                Vec const& params = {})
{ 
  // cpu only for now
  using T = typename std::decay_t<Arr_t>::value_type;
  using math::nda::make_distributed_array;
  using math::nda::distributed_array_view;
  using math::nda::transpose;
  using math::nda::dagger;
  using local_Array_t = ::nda::basic_array<ComplexType,2,::nda::C_layout,'A',::nda::heap<::nda::mem::get_addr_space<typename std::decay_t<Arr_t>::Array_t>>>;
  decltype(::nda::range::all) all;

  auto zero = memory::unified_array<RealType,1>::zeros({3});
  long nQ = Q.extent(0);
  long np = P.global_shape()[1];
  utils::check( P.global_shape()[0] == nQ, "Dimension mismatch.");
  utils::check( V.global_shape()[0] == nQ, "Dimension mismatch.");
  utils::check( V.global_shape()[1] == np, "Dimension mismatch.");
  utils::check( V.global_shape()[2] == np, "Dimension mismatch.");
  utils::check( *(V.communicator()) == *(P.communicator()), "Communicator mismatch.");
  utils::check( V.block_size()[1] == V.block_size()[2], "Block size mismatch.");

  
  if( basis == "gt" ) {

    auto Ploc = P.local();
    long ng = g.size();
    utils::check( P.global_shape()[2] == ng, "Dimension mismatch.");

    if constexpr (::nda::mem::on_host<typename std::decay_t<Arr_t>::Array_t>) {
      memory::unified_array<T,1> Vg(P.local_shape()[2]);
      for( auto [iq,q] : itertools::enumerate(P.local_range(0)) ) { 
        potential_g(Vg,g.g_vectors()(P.local_range(2),all),zero,Q(q,all),type,params);
        for( auto& v: Vg) v = std::sqrt(v);
        for( auto iu : nda::range(P.local_shape()[1]) ) 
          for( auto ig : nda::range(P.local_shape()[2]) ) 
            Ploc(iq,iu,ig) *= Vg(ig);
      }	
    } else {
      memory::unified_array<T,2> Vg(P.local_shape()[0], P.local_shape()[2]);
      for( auto [iq,q] : itertools::enumerate(P.local_range(0)) ) 
        potential_g(Vg(iq,all),g.g_vectors()(P.local_range(2),all),zero,Q(q,all),type,params);
      for( auto& v: Vg) v = std::sqrt(v);	
#if defined(ENABLE_DEVICE)
      using nda::tensor::cutensor::cutensor_desc;
      using nda::tensor::cutensor::elementwise_binary;
      using nda::tensor::op::MUL;
      cutensor_desc<ComplexType,2> a_t(Vg);
      cutensor_desc<ComplexType,3> b_t(Ploc);
      elementwise_binary(ComplexType(1.0),a_t,Vg.data(),"qg",
                                 ComplexType(1.0),b_t,Ploc.data(),"qug",
                                 Ploc.data(),MUL);
#else
      utils::check(false,"on_device without device support.");
#endif
    }
    math::nda::slate_ops::multiply(P,dagger(P),V);

  } else {

    utils::check(false,"Finish!");

  }
}


template<math::nda::DistributedArrayOfRank<2> Arr_t,
         class Vec = std::array<double,1>>
auto matrix_elements_potential_full_g(Arr_t& P,
                std::string basis,
                ::nda::ArrayOfRank<1> auto const& mesh,
                ::nda::ArrayOfRank<2> auto const& recv,
		::nda::ArrayOfRank<1> auto const& kp,
		::nda::ArrayOfRank<1> auto const& kq,
                std::string type = "coulomb",
                Vec const& params = {})
{
  // cpu only for now
  using T = typename std::decay_t<Arr_t>::value_type;
  using math::nda::make_distributed_array;
  using math::nda::distributed_array_view;
  using math::nda::transpose;
  using math::nda::dagger;

  long nnr = mesh(0)*mesh(1)*mesh(2);
  utils::check( nnr == P.global_shape()[1], "Dimension mismatch.");
  using local_Array_t = typename std::decay_t<Arr_t>::Array_t;
  using comm_t = typename std::decay_t<decltype(*(P.communicator()))>;

  /*   Evaluate potential */
  memory::unified_array<T,1> Vg(nnr,0.0);
  potential_full_g(Vg,mesh,recv,kp,kq,type,params);

  long np = P.global_shape()[0];
  auto V = make_distributed_array<local_Array_t,comm_t>(*P.communicator(),P.grid(),{np,np},
                                            {P.block_size()[0],P.block_size()[0]},true);

  if(P.grid()[1] > 1) {
    APP_ABORT("matrix_elements_potential_full_g:: Finish!!! \n");
    // redistribute matrix, since FFTs are serial
  } else {

    auto P_local = P.local();
    long np_local = P_local.shape(0);
    if( basis == "r" ) {
      // fft r->G
      auto P4d = ::nda::reshape(P_local, std::array<long,4> {np_local,mesh(0),mesh(1),mesh(2)});
      math::fft::fwdfft_many(P4d);
    } else if(basis != "g") {
      APP_ABORT("Error in potential_matrix_elements: Invalid basis: {}",basis);
    }

    for( long m=0; m<np_local; ++m)
      for( long i=0; i<nnr; ++i )
        P_local(m,i) = std::sqrt(Vg(i)) * P_local(m,i);
    math::nda::slate_ops::multiply(P,dagger(P),V);

  }

  return V;
}

/*
 * P(iq,u,r)
 * V(iq,u,v)
 */
template<math::nda::DistributedArrayOfRank<3> Arr_t,
         class Vec = std::array<double,1>>
void matrix_elements_potential_full_g(Arr_t& P, Arr_t& V,
                std::string basis,
                ::nda::ArrayOfRank<1> auto const& mesh,
                ::nda::ArrayOfRank<2> auto const& recv,
                ::nda::ArrayOfRank<2> auto const& Q,
                std::string type = "coulomb",
                Vec const& params = {})
{
  // cpu only for now
  using T = typename std::decay_t<Arr_t>::value_type;
  using math::nda::make_distributed_array;
  using math::nda::distributed_array_view;
  using math::nda::transpose;
  using math::nda::dagger;
  using local_Array_t = ::nda::basic_array<ComplexType,2,::nda::C_layout,'A',::nda::heap<::nda::mem::get_addr_space<typename std::decay_t<Arr_t>::Array_t>>>;
  decltype(::nda::range::all) all;

  long nnr = mesh(0)*mesh(1)*mesh(2);
  long nQ = Q.extent(0);
  long np = P.global_shape()[1];
  utils::check( P.global_shape()[0] == nQ, "Dimension mismatch.");
  utils::check( P.global_shape()[2] == nnr, "Dimension mismatch.");
  utils::check( V.global_shape()[0] == nQ, "Dimension mismatch.");
  utils::check( V.global_shape()[1] == np, "Dimension mismatch.");
  utils::check( V.global_shape()[2] == np, "Dimension mismatch.");
  utils::check( *(V.communicator()) == *(P.communicator()), "Communicator mismatch.");
  utils::check( V.block_size()[1] == V.block_size()[2], "Block size mismatch.");

  /*   Evaluate potential */
  auto Pcomm = V.communicator();

  if(Pcomm->size()==1) {

    memory::unified_array<T,1> Vg(nnr,0.0);
    auto zero = ::nda::array<RealType,1>::zeros({3});
    auto P4d0 = ::nda::reshape(P.local()(0,all,all),
                        std::array<long,4> {np,mesh(0),mesh(1),mesh(2)});
    auto plan = math::fft::create_plan_many(P4d0);

    for( auto [iq,q] : itertools::enumerate(V.local_range(0)) )  {

      auto Pq = P.local()(iq,all,all);
      auto Vq = V.local()(iq,all,all);
      auto P4d = ::nda::reshape(Pq,
                std::array<long,4> {np,mesh(0),mesh(1),mesh(2)});
      if( basis == "r" ) {
        // fft r->G
        math::fft::fwdfft(plan,P4d);
      } else if(basis != "g") {
        APP_ABORT("Error in potential_matrix_elements: Invalid basis: {}",basis);
      }

      potential_full_g(Vg,mesh,recv,zero,Q(q,all),type,params);
      for( auto& v: Vg ) v = std::sqrt(v);
      // prefetch Vg to device and use tensor op
      for( long m=0; m<np; ++m)
        for( long i=0; i<nnr; ++i )
          Pq(m,i) = Vg(i) * Pq(m,i);

      ::nda::blas::gemm(ComplexType(1.0),Pq,::nda::dagger(Pq),ComplexType(0.0),Vq);

    } // (iq,q)

    math::fft::destroy_plan(plan);

  } else {

    memory::unified_array<T,1> Vg(nnr,0.0);
    auto zero = ::nda::array<RealType,1>::zeros({3});
    auto q_comm = Pcomm->split(V.origin()[0],Pcomm->rank());
    using comm_t = decltype(q_comm);

    std::array<long,2> Pgrid = {P.grid()[1],P.grid()[2]};
    std::array<long,2> Porigin = {P.origin()[1],P.origin()[2]};
    std::array<long,2> Pbsize = {P.block_size()[1],P.block_size()[2]};

    std::array<long,2> Vgrid = {V.grid()[1],V.grid()[2]};
    std::array<long,2> Vorigin = {V.origin()[1],V.origin()[2]};
    std::array<long,2> Vbsize = {V.block_size()[1],V.block_size()[2]};

    // for fft calculation
    auto psi_1 = make_distributed_array<local_Array_t,comm_t>(q_comm,{q_comm.size(),1},{np,nnr},{1,1});
    psi_1.local() = ComplexType(0.0);
    auto p_local = psi_1.local();
    long np_local = psi_1.local_shape()[0];
    // better layout for gemm
    auto psi_2 = make_distributed_array<local_Array_t,comm_t>(q_comm,Vgrid,{np,nnr},
                                              {V.block_size()[1],2048});
    psi_2.local() = ComplexType(0.0);

    auto P4d = ::nda::reshape(p_local, std::array<long,4> {np_local,mesh(0),mesh(1),mesh(2)});
    auto plan = math::fft::create_plan_many(P4d);

    for( auto [iq,q] : itertools::enumerate(V.local_range(0)) )  {

    // 2D slice of distributed array
    auto Pq2D = P.local()(iq,all,all);
    auto Vq2D = V.local()(iq,all,all);
    distributed_array_view<local_Array_t,decltype(q_comm)> Pq(std::addressof(q_comm),
                Pgrid, {np,nnr}, Porigin, Pbsize, Pq2D);
    distributed_array_view<local_Array_t,decltype(q_comm)> Vq(std::addressof(q_comm),
                Vgrid, {np,np}, Vorigin, Vbsize, Vq2D);

    redistribute(Pq,psi_1);
    if( basis == "r" ) {
      // fft r->G
      math::fft::fwdfft(plan,P4d);
    } else if(basis != "g") {
      APP_ABORT("Error in potential_matrix_elements: Invalid basis: {}",basis);
    }

    potential_full_g(Vg,mesh,recv,zero,Q(q,all),type,params);
    for( auto& v: Vg ) v = std::sqrt(v);
    // prefetch Vg to device and use tensor op
    for( long m=0; m<np_local; ++m)
      for( long i=0; i<nnr; ++i )
        p_local(m,i) = Vg(i) * p_local(m,i);

    redistribute(psi_1,psi_2);


    math::nda::slate_ops::multiply(psi_2,dagger(psi_2),Vq);

    } // (iq,q)
    math::fft::destroy_plan(plan);

  }
}

} // hamilt 

#endif

