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


#ifndef COQUI_MPI_CONTEXT_H
#define COQUI_MPI_CONTEXT_H

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"
#if defined(ENABLE_NCCL)
#include "mpi3/nccl/communicator.hpp"
#endif
#include "nda/nda.hpp"

namespace mpi3 = boost::mpi3;
namespace utils {

template <typename comm_t = mpi3::communicator, 
          typename shm_comm_t = mpi3::shared_communicator
#if defined(ENABLE_NCCL)
          ,typename nccl_comm_t = mpi3::nccl::communicator
#endif
         > 
struct mpi_context_t {
  comm_t comm;
  shm_comm_t node_comm;
  comm_t internode_comm;
#if defined(ENABLE_NCCL)
  nccl_comm_t dev_comm;
#endif

  mpi_context_t() = delete;
  mpi_context_t(comm_t const& c_, shm_comm_t const& s_, comm_t const& ic_
#if defined(ENABLE_NCCL)
                ,nccl_comm_t const& d_
#endif
                ) : comm(c_),node_comm(s_),internode_comm(ic_)
#if defined(ENABLE_NCCL)
                    ,dev_comm(d_)
#endif
  {}

  mpi_context_t(comm_t && c_, shm_comm_t && s_, comm_t && ic_
#if defined(ENABLE_NCCL)
                ,nccl_comm_t && d_
#endif
                ) : comm(std::move(c_)),node_comm(std::move(s_)),internode_comm(std::move(ic_))
#if defined(ENABLE_NCCL)
                    ,dev_comm(std::move(d_))
#endif
  {}

  mpi_context_t(mpi_context_t const&) = default;
  mpi_context_t(mpi_context_t &&) = default;
  mpi_context_t& operator=(mpi_context_t const&) = default;
  mpi_context_t& operator=(mpi_context_t &&) = default;

  // some auxiliary functions for nda
  template<bool use_gpu = true>
  void broadcast(nda::Array auto&& A, int root = 0)
  {
    utils::check(A.is_contiguous(),"mpi_context_t::broadcast: Array must be contiguous.");
#if defined(ENABLE_NCCL)
    if constexpr (use_gpu and ::nda::mem::have_device_compatible_addr_space<decltype(A)>) {
      dev_comm.broadcast_n(A.data(),A.size(),root);
    } else  
#else
      comm.broadcast_n(A.data(),A.size(),root);
#endif
  } 

  template<typename Op, bool use_gpu = true>
  void all_reduce(nda::Array auto&& A, Op& op)
  {
    utils::check(A.is_contiguous(),"mpi_context_t::broadcast: Array must be contiguous.");
#if defined(ENABLE_NCCL)
    if constexpr (use_gpu and ::nda::mem::have_device_compatible_addr_space<decltype(A)>) {
      using T = ::nda::get_value_t<decltype(A)>;
      if constexpr (::nda::is_complex_v<T>) { 
        comm.all_reduce_in_place_n(reinterpret_cast<T*>(A.data()),2*A.size(),op);
      } else {
        comm.all_reduce_in_place_n(A.data(),A.size(),op);
      }
    } else  
#else
      comm.all_reduce_in_place_n(A.data(),A.size(),op);
#endif
  }

};

#if defined(ENABLE_NCCL)

template<MEMORY_SPACE MEM, typename comm_t, typename shm_comm_t, typename nccl_comm_t>
auto& get_dev_comm(mpi_context_t<comm_t,shm_comm_t,nccl_comm_t>& mpi) {
  if constexpr (MEM==HOST_MEMORY)
    return mpi.comm; 
  else
    return mpi.dev_comm; 
};

#else

template<MEMORY_SPACE MEM, typename comm_t, typename shm_comm_t>
auto& get_dev_comm(mpi_context_t<comm_t,shm_comm_t>& mpi) {
  return mpi.comm;
};

#endif

mpi_context_t<mpi3::communicator> make_mpi_context(); 
mpi_context_t<mpi3::communicator> make_mpi_context(mpi3::communicator& comm); 

}
#endif // COQUI_MPI_CONTEXT_H
