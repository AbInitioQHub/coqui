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


#ifndef NUMERICS_DISTRIBUTED_ARRAY_H5_HPP
#define NUMERICS_DISTRIBUTED_ARRAY_H5_HPP

#include "string"
#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "utilities/stl_utils.hpp"

namespace math::nda {

void h5_write(h5::group& g, std::string name, 
              DistributedArray auto const& A)  
{
#if defined(__TIME__H5__)
  H5_Timer.start("TOTAL");
#endif
  using dArray_t = std::decay_t<decltype(A)>;
  using value_type = typename dArray_t::value_type;
  constexpr int rank = math::nda::get_rank<dArray_t>;
  using local_Array_rank_t = ::nda::basic_array<value_type, rank, ::nda::C_layout,'A',::nda::heap<::nda::mem::get_addr_space<typename dArray_t::Array_t>>>;
  auto comm = A.communicator();

#if defined(HAVE_PHDF5)
#error  // finish!!!
#else

  // gather + single write
  if (comm->root()) {
    // MAM: this might be a problem in GPU! Move to host in GPU case!
    local_Array_rank_t A_loc(A.global_shape());
#if defined(__TIME__H5__)
  H5_Timer.start("COMM");
#endif
    math::nda::gather(0, A, std::addressof(A_loc));
#if defined(__TIME__H5__)
  H5_Timer.stop("COMM");
#endif
#if defined(__TIME__H5__)
  H5_Timer.start("IO");
#endif
    ::nda::h5_write(g, name, A_loc, false);
#if defined(__TIME__H5__)
  H5_Timer.stop("IO");
#endif
  } else {
    local_Array_rank_t A_loc;
    math::nda::gather(0, A, std::addressof(A_loc));
  }
  comm->barrier();
// MAM: very slow in Rusty. I suspect it is coming from the use of strided data?
/*
  std::array<long,2*rank> sz;
  auto Aloc = A.local();
  auto origin = A.origin();
  auto shape = Aloc.shape();
  for(int i=0; i<rank; ++i) { 
    sz[2*i] = origin[i];
    sz[2*i+1] = shape[i];
  }
  // no phdf5, only root writes for now...
  // slow and inefficient!!!    
  if(comm->rank()==0) {
    {
      //write dataset with global shape
#if defined(__TIME__H5__)
  H5_Timer.start("IO");
#endif
      ::nda::h5_write(g,name,::nda::zeros<value_type>(A.global_shape()),false);
#if defined(__TIME__H5__)
  H5_Timer.stop("IO");
#endif
    }
    std::array<::nda::range,rank> rng;
    {
      for(int i=0; i<rank; ++i)
        rng[i] = ::nda::range(origin[i],origin[i]+shape[i]);
      auto tpl = std::tuple_cat(rng);
//      ::nda::h5_write(g,name,A.local(),tpl,false);
// MAM: temporary hack until contrain bug is fixed
      auto A_h = ::nda::to_host(A.local());
#if defined(__TIME__H5__)
  H5_Timer.start("IO");
#endif
      ::nda::h5_write(g,name,A_h,tpl,false);
#if defined(__TIME__H5__)
  H5_Timer.stop("IO");
#endif
    }
    std::array<long,rank> ext;
    ::nda::array<value_type,1> buff; 
    for(int  i=1; i<comm->size(); ++i) {
#if defined(__TIME__H5__)
  H5_Timer.start("COMM");
#endif
      comm->receive_n(sz.data(),2*rank,i,1111+i);
#if defined(__TIME__H5__)
  H5_Timer.stop("COMM");
#endif
      long n = 1;
      for(int i=0; i<rank; ++i) n *= sz[2*i+1];
      if(buff.size() < n) buff.resize(n);
#if defined(__TIME__H5__)
  H5_Timer.start("COMM");
#endif
      comm->receive_n(buff.data(),n,i,3333+i);
#if defined(__TIME__H5__)
  H5_Timer.stop("COMM");
#endif
      for(int i=0; i<rank; ++i) ext[i] = sz[2*i+1]; 
      auto Li = ::nda::array_view<value_type,rank>(ext,buff.data());
      for(int i=0; i<rank; ++i)
        rng[i] = ::nda::range(sz[2*i],sz[2*i]+sz[2*i+1]);
      auto tpl = std::tuple_cat(rng);
#if defined(__TIME__H5__)
  H5_Timer.start("IO");
#endif
      ::nda::h5_write(g,name,Li,tpl,false);
#if defined(__TIME__H5__)
  H5_Timer.stop("IO");
#endif
    }
  } else {
    comm->send_n(sz.data(),2*rank,0,1111+comm->rank());
    auto A_h = ::nda::to_host(Aloc); 
    comm->send_n(A_h.data(),A_h.size(),0,3333+comm->rank());
  }
  comm->barrier();
*/
#endif
#if defined(__TIME__H5__)
  H5_Timer.stop("TOTAL");
#endif
}

void h5_read(h5::group& g, std::string name, DistributedArray auto & A)
{
  using dArray_t = std::decay_t<decltype(A)>;
  constexpr int rank = math::nda::get_rank<dArray_t>;
  auto rng = utils::default_array_of_ranges<rank>();
  auto origin = A.origin();
  auto shape = A.local().shape();
  for(int i=0; i<rank; ++i) 
    rng[i] = ::nda::range(origin[i],origin[i]+shape[i]); 
  auto tpl = std::tuple_cat(rng);  
  if constexpr (::nda::mem::on_device<typename dArray_t::Array_t>) {
    auto A_h = ::nda::to_host(A.local()); 
    ::nda::h5_read(g,name,A_h,tpl);
    A.local() = A_h;
  } else { 
    ::nda::h5_read(g,name,A.local_(),tpl);
  }
}

}

#endif
