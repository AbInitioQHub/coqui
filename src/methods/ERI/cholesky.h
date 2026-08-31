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


#ifndef METHODS_ERI_CHOL_CHOLESKY_HPP
#define METHODS_ERI_CHOL_CHOLESKY_HPP

#include <tuple>
#include <iomanip>

#include "configuration.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/check.hpp"
#include "utilities/Timer.hpp"
#include "utilities/proc_grid_partition.hpp"
#include "utilities/functions.hpp"

#include "mpi3/communicator.hpp"
#include "utilities/mpi_context.h"
#if defined(ENABLE_NCCL)
#include "mpi3/nccl/communicator.hpp"
#endif
#include "itertools/itertools.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "numerics/fft/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "nda/linalg/det_and_inverse.hpp"

#include "grids/g_grids.hpp"
#include "potentials/potentials.hpp"

#include "mean_field/MF.hpp"

namespace methods
{

namespace mpi3 = boost::mpi3;

// to do: add symmetry of Q vectors in MF object and propagate to here!
// always operate on a cholesky object as if it is calculated on the fly,
// then write cholesky_from_file which has the same interface as this class, 
// but it reads instead of calculate

// MAM: write version that only computes the "diagonal" part of the factorization, e.g. (k,k-q|k-q,k)
// where now each (q,k) combination is decomposed separately leading to smaller aux dimensions.
// This is to be used in coulomb-only rpa and/or GW.
// In this case, don't include the phases in the pair densities since they will cancel anyway 
class cholesky 
{

  template<nda::MemoryArray local_Array_t>
  using dArray_t = memory::darray_t<local_Array_t,mpi3::communicator>; 
  //using dArray_t = math::nda::distributed_array<local_Array_t,mpi3::communicator>;
  template<int N>
  using shape_t = std::array<long,N>;

  public:

  cholesky(mf::MF *mf_, utils::mpi_context_t<mpi3::communicator> &mpi_, ptree const& pt); 

  ~cholesky() {} 
   
  cholesky(cholesky const&) = default;
  cholesky(cholesky &&) = default;
  cholesky& operator=(cholesky const&) = default;
  cholesky& operator=(cholesky &&) = default;

  /**
   * Evaluate the Cholesky three-index tensor, L^{q}_{a,b}(P) for a given q-point
   * @tparam MEM
   * @param Qi - q-point
   * @param a_range - range for orbital index a
   * @param b_range - range for orbital index b
   * @param diag_kk - If true, Cholesky factorization will only reproduce k-k sector of the 2 ERI tensor. 
   * @param block_size - block size for parallelization
   * @return
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY>
  auto evaluate(int Qi,
                nda::range a_range = nda::range(-1,-1),
                nda::range b_range = nda::range(-1,-1), 
		bool diag_kk = false,
		int block_size = -1)
	-> dArray_t<memory::array<MEM,ComplexType,5>>;

  /**
   * Evaluate the Cholesky three-index tensor, L^{q}_{a,b}(P) for a given q-point
   * Stores resulting tensor in provided h5 file.
   * @tparam MEM
   * @param gh5 - h5::h5_group object where the Cholesky tensor will be written.
   * @param format -  format for the h5 dataset. 
   * @param Qi - q-point
   * @param a_range - range for orbital index a
   * @param b_range - range for orbital index b
   * @param diag_kk - If true, Cholesky factorization will only reproduce k-k sector of the 2 ERI tensor. 
   * @param block_size - block size for parallelization
   * @return
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY>
  void evaluate(h5::group& gh5, std::string format, int Qi,
                nda::range a_range = nda::range(-1,-1),
                nda::range b_range = nda::range(-1,-1), 
		bool diag_kk = false,
		int block_size = -1); 
  
  /**
   * Evaluate the Cholesky three-index tensor, L^{q}_{a,b}(P) for all q-points.
   * Stores resulting tensor in provided h5 file.
   * @tparam MEM
   * @param gh5 - h5::h5_group object where the Cholesky tensor will be written.
   * @param format -  format for the h5 dataset.
   * @param a_range - range for orbital index a
   * @param b_range - range for orbital index b
   * @param diag_kk - If true, Cholesky factorization will only reproduce k-k sector of the 2 ERI tensor.
   * @param block_size - block size for parallelization
   * @return
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY>
  void evaluate(h5::group& gh5, std::string format, 
		    nda::range a_range = nda::range(-1,-1), 
                    nda::range b_range = nda::range(-1,-1), 
		    bool diag_kk = false,
		    int block_size = -1);

  template<nda::MemoryArray local_Array_t>
  void write(h5::group& gh5, int Qi, dArray_t<local_Array_t> const& L,
	     std::string format="default");

  // writes meta data to h5 file, includes all information in addition to actual
  // cholesky vectors. File should be self contained upon read  
  void write_meta_data(h5::group& gh5, std::string format="default");

  void reset_timers() { Timer.reset(); }

  void print_timers();

  private:

  utils::mpi_context_t<mpi3::communicator> *mpi = nullptr;

  // pointer to MF object
  mf::MF *mf = nullptr;

  utils::TimerManager Timer;

  // Number of k-points pools (must be divisible by comm.size().
  // The global communicator will be split into npools sub-communicators.
  int npools=-1;
  // cutoff for Cholesky decomposition
  double cutoff=1e-4;
  int default_block_size = 32;

  // Energy cutoff for the plane-wave basis
  double ecut = 0.0;
  // The grid for the plane-wave basis
  grids::truncated_g_grid rho_g;

  // object that evaluates potential, v[G,Q]
  pots::potential_t vG;

  //fft plans
  int howmany_fft = -1;

  template<MEMORY_SPACE MEM = HOST_MEMORY>
  void evaluate_pair_densities(int Qi, dArray_t<memory::array<MEM,ComplexType,5>>&,
				nda::range a_range, nda::range b_range); 

  template<MEMORY_SPACE MEM = HOST_MEMORY>
  void evaluate_pair_densities(int Qi, int ik, dArray_t<memory::array<MEM,ComplexType,3>>&,
				nda::range a_range, nda::range b_range); 

  template<MEMORY_SPACE MEM = HOST_MEMORY, typename comm_t>
  auto evaluate_impl(comm_t& k_intra_comm, int Qi, nda::range a_range, nda::range b_range); 

  template<MEMORY_SPACE MEM = HOST_MEMORY, typename comm_t>
  auto evaluate_blocked_impl(comm_t& k_intra_comm, int Qi, 
                             nda::range a_range, nda::range b_range, int block_size);  

  template<MEMORY_SPACE MEM = HOST_MEMORY, typename comm_t>
  auto evaluate_diagkk_impl(comm_t& k_intra_comm, int Qi, int ik, 
                            nda::range a_range, nda::range b_range, int block_size);  

};

} //methods

#endif
