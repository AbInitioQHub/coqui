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


#ifndef UTILITIES_PROC_GRID_PARTITION_HPP
#define UTILITIES_PROC_GRID_PARTITION_HPP

#include "utilities/check.hpp"

namespace utils
{

/**
 * Find a processor grid {n x m} where max(nkpts, gcomm.size()) = n*m and n is maximized
 * @tparam communicator
 * @param gcomm
 * @param nkpts
 * @return
 */
// find processor grid {n x m}, maximize and return (n) 
template<typename communicator>
long find_proc_grid_max_rows(communicator & gcomm, long nkpts)
{
  long np = std::max(nkpts,long(gcomm.size()));
  long nk = std::min(nkpts,long(gcomm.size()));
  for(long i=1; i<nk/2+1; ++i)
    if( nk%i == 0 and np%(nk/i) == 0 ) return nk/i;
  return 1;
}

inline long find_proc_grid_max_rows(long size, long nkpts)
{
  long np = std::max(nkpts,size);
  long nk = std::min(nkpts,size);
  for(long i=1; i<nk/2+1; ++i)
    if( nk%i == 0 and np%(nk/i) == 0 ) return nk/i;
  return 1;
}

/**
 * Find maximum number of pools along dimension i with the constraint
 * that 1/pool_size <= error
 * @param np    - number of processors
 * @param dim_i - dimension i
 * @param error - error constraint
 * @return - number of pools
 */
inline long find_proc_grid_max_npools(long np, long dim_i, double error) {
  long npools = 1;
  for (long i = std::min(np, dim_i); i > 0; --i) {
    long pool_size = dim_i / i;
    if (dim_i % i == 0 and np % i == 0) {
      npools = i;
      break;
    } else {
      if ((double) 1.0 / pool_size <= error and np % i == 0) {
        npools = i;
        break;
      }
    }
  }
  return npools;
}

/**
 * For a given number of processors (np),
 * find a processor grid {n x m} that minimizes (n*nc-m*nr) and n*m=np, return n
 * (basically try to find a grid that n ~ nr and m ~ nc)
 * @param np - number of processors
 * @param nr - number of rows
 * @param nc - number of columns
 * @return
 */
inline long find_proc_grid_min_diff(long np, long nr, long nc)
{
  // now look for minimum
  long maxd = np*std::max(nr,nc), indx=-1;
  for(long i=np; i>0; i--)
    if( np%i==0 and std::abs(i*nc - (np/i)*nr) < maxd ) {
      maxd = std::abs(i*nc - (np/i)*nr);
      indx = i;
    }
  utils::check(indx>0 and np%indx==0, "Problems finding processor grid.");
  return std::max(indx,np/indx);  // to make leading dimension smaller
}

/**
 * find the minimum m such that np = n * m where n is maximized and n <= nr_max
 * @return n
 */
inline long find_min_col(long np, long nr_max, long nc_min=1) {
  if (nr_max > np) {
    return 1;
  } else {
    long m = np / nr_max;
    while (np % m != 0 or m < nc_min) {
      m += 1;
    }
    return m;
  }
}

template<typename communicator>
inline auto setup_two_layer_mpi(communicator *comm, const size_t dim0, const size_t dim1) {
  size_t dim0_rank, dim0_comm_size, dim1_rank, dim1_comm_size;
  if (comm!=nullptr) {
    dim0_comm_size = find_proc_grid_min_diff(comm->size(), dim0, dim1);
    dim1_comm_size = comm->size() / dim0_comm_size;
    dim0_rank = comm->rank() / dim1_comm_size;
    dim1_rank = comm->rank() % dim1_comm_size;
  } else {
    dim0_rank = 0;
    dim0_comm_size = 1;
    dim1_rank = 0;
    dim1_comm_size = 1;
  }
  std::array<size_t, 4> mpi_info = {dim0_rank, dim0_comm_size, dim1_rank, dim1_comm_size};
  return mpi_info;
}

}


#endif
