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


#ifndef METHODS_ERI_THC_THC_H
#define METHODS_ERI_THC_THC_H

#include <tuple>
#include <iomanip>
#include <optional>

#include "configuration.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/check.hpp"
#include "utilities/Timer.hpp"
#include "utilities/proc_grid_partition.hpp"
#include "grids/g_grids.hpp"
#include "potentials/potentials.hpp"

#include "mpi3/communicator.hpp"
#if defined(ENABLE_NCCL)
#include "mpi3/nccl/communicator.hpp"
#endif
#include "utilities/mpi_context.h"
#include "itertools/itertools.hpp"
#include "nda/nda.hpp"
#include "numerics/fft/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/h5.hpp"
#include "numerics/shared_array/nda.hpp"

#include "mean_field/MF.hpp"

namespace methods
{

/*
 * MAM: Fix problem that requires nqpool to exactly partition Qpts, this is unneccesary 
 */ 

namespace mpi3 = boost::mpi3;

class thc
{
  template<int N>
  using shape_t = std::array<long,N>;
  template<MEMORY_SPACE MEM, int N>
  using _darray_t_ = memory::darray_t<memory::array<MEM,ComplexType,N>,mpi3::communicator>;
  using mpi_context_t = utils::mpi_context_t<mpi3::communicator>;

  public:
  /*
   * Creates a thc object with arguments in property tree.
   *  Important options:
   *  - ecut: "1.4 * ecutwfc" (falls back to "0.4 * ecutrho" when no wfc grid is available),
   *          Plane wave cutoff used for the evaluation of coulomb matrix elements.
   *  - thresh: "1e-5", Threshold in cholesky decomposition.
   *  Performance related options:
   *  - matrix_block_size: 1024, Block size used in distributed arrays.
   *  - chol_block_size: "8", Block size in cholesky decomposition.
   *  - r_blk: "1", Number of iterations used to process real space grid in real space algorithm.  
   *  - distr_tol: "0.2". Controls the processor grid. Larger values lead to more processors in k/Q grid axis.
   *  - memory_frac: "0.75". fraction of available memory in a node used to estimate memory requirements/utilization. 
   */
  thc(mf::MF *mf_,
      mpi_context_t& mpi_,
      ptree const& pt,
      bool print_metadata_ = true);
 
  ~thc();

  thc(thc const&) = default;
  thc(thc &&) = default;
  thc& operator=(thc const&) = default;
  thc& operator=(thc &&) = default;

  void print_metadata();

  template<MEMORY_SPACE MEM = HOST_MEMORY>
  auto interpolating_basis(memory::array<MEM,long,1> const& IPts, int iq=0,
              nda::range k_range = nda::range(-1,-1), 
              nda::range a_range = nda::range(-1,-1),
              nda::range b_range = nda::range(-1,-1))
       -> std::tuple<memory::darray_t<memory::array<MEM,ComplexType,2>,mpi3::communicator>,
          	     memory::darray_t<memory::array<MEM,ComplexType,2>,mpi3::communicator>>;

  template<MEMORY_SPACE MEM = HOST_MEMORY>
  auto interpolating_basis(bool left, memory::array<MEM,long,1> const& IPts, int iq=0,
              nda::range k_range = nda::range(-1,-1), 
              nda::range a_range = nda::range(-1,-1),
              nda::range b_range = nda::range(-1,-1))
       -> memory::darray_t<memory::array<MEM,ComplexType,2>,mpi3::communicator>;

  /**
   * Compute interpolating points for phi^{k,*}_a(r)phi^{k-q}_b(r) at a given q-point.
   * @param iq       - [INPUT] Index of the q-point
   * @param max      - [INPUT] Maximum number of interpolating points.
   *                           If max=-1, there will be no hard limit, and the number of
   *                           interpolating points is computed until the error is smaller
   *                           than this->thresh.
   * @param a_range  - [INPUT] Orbital range for phi^{k,*}_a
   * @param b_range  - [INPUT] Orbital range for phi^{k-q}_b
   * @return A tuple containing:
   *         - index of interpolating points, (Np)
   *         - distributed array for phi^{k}_a on interpolating points: (ns, nkpts, nbnd_a, Np)
   *         - distributed array for phi^{k-q}_b on interpolating points: (ns, nkpts, nbnd_b, Np)
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY>
  auto interpolating_points(int iq = 0, int max = -1,
              nda::range a_range = nda::range(-1,-1),
              nda::range b_range = nda::range(-1,-1))
       -> std::tuple<memory::array<MEM,long,1>,
                     _darray_t_<MEM,4>, 
                     std::optional<_darray_t_<MEM,4>>
                    >; 

  template<MEMORY_SPACE MEM = HOST_MEMORY>
  auto interpolating_points(nda::MemoryArrayOfRank<4> auto const& C_skai, 
              int iq = 0, int max = -1)
       -> std::tuple<memory::array<MEM,long,1>,
                     _darray_t_<MEM,4>,
                     std::optional<_darray_t_<MEM,4>>
                    >;

  /**
   * THC-OV:
   * Calculates interpolating vectors with density fitting in the overlap metric,
   * using precalculated interpolating points, and calculate Coulomb matrices.
   * All q-points are calculated simultaneously.
   *
   *      phi^{k*}_a(r)phi^{k-q}_b(r)
   *          = \sum_{\mu} phi^{k*}_a(r_mu)phi^{k-q}_b(r_mu) \zeta^{q}_mu(r)
   *
   * @param ri  - [INPUT] interpolating points
   * @param Xa  - [INPUT] orbital "a" on interpolating points: phi^{k*}_a(r_mu)
   * @param Xb  - [INPUT] orbital "b" on interpolating points: phi^{k-q*}_b(r_mu)
   * @param return_Sinv_Ivec - [INPUT] return inverse of overlap matrix for \zeta^{q}_mu(r)
   * @param a_range - [INPUT] range of orbital "a"
   * @param b_range - [INPUT] range of orbital "b"
   * @param pgrid3D - [INPUT] processor grid
   * @return A tuple containing:
   *         - V_{\mu,\nu}: Distributed array with Coulomb matrix in the basis of interpolating vectors.
   *                        dims: (nqpts_ibz, nIpts, nIpts), distributed along nIpts rows only.
   *         -
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY, typename Tensor_t = _darray_t_<MEM,4>>
  auto evaluate(memory::array<MEM,long,1> const& ri, 
              Tensor_t const& Xa,
              std::optional<Tensor_t> const& Xb,
              bool return_Sinv_Ivec = false, 
              nda::range a_range = nda::range(-1,-1),
              nda::range b_range = nda::range(-1,-1),
              std::array<long, 3> pgrid3D = {0,0,0})
        -> std::tuple<_darray_t_<MEM,3>,
                      memory::array<MEM, ComplexType, 2>, memory::array<MEM, ComplexType, 2>, 
                      std::optional<_darray_t_<MEM,3>>
                     >;

  /**
   * THC-OV:
   * Calculates interpolating vectors with density fitting in the overlap metric,
   * using precalculated interpolating vectors. A rotation matrix can be provided, which will
   * be applied to the left hand side factor in the pair density product. The interpolating
   * points and collation matrices must be consistent with the results of interpolating_vectors
   * using an identical rotation matrix, otherwise results are incorrect.
   *
   * Returns Vuv, where:
   *  - Vuv: Distributed array with coulomb matrix in the basis of interpolating vectors.
   *    -> dims: [nIpts, nIpts], distributed along rows only.
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY, typename Tensor_t = _darray_t_<MEM,4>>
  auto evaluate(memory::array<MEM,long,1> const& ri,
              nda::MemoryArrayOfRank<4> auto const& C_skai,
              Tensor_t const& Xa,
              Tensor_t const& Xb,
              bool return_Sinv_Ivec = false,
              std::array<long, 3> pgrid3D = {0,0,0})
        -> std::tuple<_darray_t_<MEM,3>,
                      memory::array<MEM, ComplexType, 2>, memory::array<MEM, ComplexType, 2>,
                      std::optional<_darray_t_<MEM,3>>
                     >;

  /**
   * THC-LS:
   *  Interpolating vectors are obtained by solving the overdetermined linear system, using
   *  the provided density fitting basis and precalculated interpolating points.
   *
   *  Given B(ab,n), solves (omitting spin and k-point index for simplicity):
   *
   *    B(ab,n) = sum_u conj(Pa(a,ru)) * Pb(b,ru) * I(u,n),
   *
   *  where ru are provided interpolating points.
   *
   * Returns a tuple with: Vuv, where:
   *  - Vuv: Distributed array (... with coulomb matrix) in the basis of interpolating vectors.
   *    -> dims: [nIpts, nIpts], distributed along rows only.
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY>
  auto evaluate(int iq, memory::array<MEM,long,1> const& ri, 
              memory::darray_t<memory::array<MEM,ComplexType,5>,mpi3::communicator> const& B,
              nda::range a_range = nda::range(-1,-1),
              nda::range b_range = nda::range(-1,-1))
	-> memory::darray_t<memory::array<MEM,ComplexType,2>,mpi3::communicator>;

 /**
  * THC-DF:
  * Interpolating vectors are obtained by solving the overdetermined linear system, using
  * the provided density fitting basis and precalculated interpolating points.
  *
  *  Given Psi(is,ik,a,G), solves (omitting spin and kpoint index for simplicity):
  *
  *    B(q,ab,n) = sum_u conj(Pa(k,a,ru)) * Pb(k-q,b,ru) * I(q,u,n),
  *
  *  where ru are provided interpolating points, B(q,ab,n) are 3-center DF integrals
  *  calculated from Psi.
  *
  * Returns Vuv(q,u,v), where:
  *  - Vuv: Distributed array with the coulomb matrix, Vuv = sum_n I(q,u,n)*conj(I(q,v,n))
  *    -> dims: [nqpts_ibz,nIpts, nIpts], distributed along rows only.
  */
 /*
  template<MEMORY_SPACE MEM = HOST_MEMORY>
  auto evaluate(memory::array<MEM,long,1> const& ri,
              memory::darray_t<memory::array<MEM,ComplexType,3>,mpi3::communicator>& Psi,
              nda::range a_range = nda::range(-1,-1),
              nda::range b_range = nda::range(-1,-1),
              std::array<long, 3> pgrid3D = {0,0,0})
        -> memory::darray_t<memory::array<MEM,ComplexType,3>,mpi3::communicator>;
*/

  template<MEMORY_SPACE MEM = HOST_MEMORY, typename Tensor_t = _darray_t_<MEM,4>>
  void evaluate(h5::group& gh5, std::string format,
              memory::array<MEM,long,1> const& ri, 
              Tensor_t const& Xa,
              std::optional<Tensor_t> const& Xb,
              nda::range a_range = nda::range(-1,-1),
              nda::range b_range = nda::range(-1,-1),
              std::array<long, 3> pgrid3D = {0,0,0});

  template<MEMORY_SPACE MEM = HOST_MEMORY, typename Tensor_t = _darray_t_<MEM,4>>
  void evaluate(h5::group& gh5, std::string format,
              memory::array<MEM,long,1> const& ri,
              nda::MemoryArrayOfRank<4> auto const& C_skai,
              Tensor_t const& Xa,
              Tensor_t const& Xb,
              std::array<long, 3> pgrid3D = {0,0,0});

  template<MEMORY_SPACE MEM = HOST_MEMORY, typename Tensor_t = _darray_t_<MEM,4>>
  auto evaluate_isdf_only(memory::array<MEM,long,1> const& ri,
                          Tensor_t const& Xa,
                          std::optional<Tensor_t> const& Xb,
                          nda::range a_range = nda::range(-1,-1),
                          nda::range b_range = nda::range(-1,-1),
                          std::array<long, 3> pgrid3D = {0,0,0})
  -> _darray_t_<MEM,3>;

  /**
   * Saves the interpolating points and coulomb matrix to h5 group
   * @param gh5          - [INPUT]
   * @param format       - [INPUT]
   * @param ri           - [INPUT]
   * @param V            - [INPUT]
   * @param Z_head_qu    - [INPUT]
   * @param Zbar_head_qu - [INPUT]
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY>
  void save(h5::group& gh5, std::string format, memory::array<MEM,long,1> const& ri,
        memory::darray_t<memory::array<MEM,ComplexType,3>,mpi3::communicator> const& V,
        memory::array<MEM,ComplexType,2> const& Z_head_qu,
        memory::array<MEM,ComplexType,2> const& Zbar_head_qu);

  /**
   * Save the interpolating points and interpolating vectors to h5 group.
   * @param gh5      - [INPUT]
   * @param format   - [INPUT]
   * @param ri       - [INPUT]
   * @param zeta_qur - [INPUT]
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY>
  void save(h5::group& gh5, std::string format, memory::array<MEM,long,1> const& ri,
            memory::darray_t<memory::array<MEM,ComplexType,3>,mpi3::communicator> const& zeta_qur,
            bool write_zeta_on_fft_mesh=false);

  // writes metadata to h5 file, includes all information in addition to actual
  // thc vectors. File should be self-contained upon read
  void write_meta_data(h5::group& gh5, std::string format="default");
  void print_timers();
  void reset_timers() { Timer.reset(); }

  private:

  // mpi context with global, node, internode and gpu communicators
  mpi_context_t *mpi;

  // pointer to MF object
  mf::MF *mf = nullptr;

  utils::TimerManager Timer;

  // pw cutoff for density grid  
  double ecut = 0.0;

  // truncated density grid
  grids::truncated_g_grid rho_g; 

  // maps from the wfc truncated grid to full fft grid of rho_g  
  math::shm::shared_array<nda::array_view<long,1>> swfc_to_rho;

  // object that evaluates potential, v[G,Q]
  pots::potential_t vG;  

  long default_block_size;
  long default_cholesky_block_size;
  double thresh=1e-5;
  int nnr_blk = 1;
  double distr_tol = 0.2;
  double memory_frac = 0.75;
  bool use_least_squares = false;

  //fft plans
  int howmany_fft = -1;

  template<MEMORY_SPACE MEM = HOST_MEMORY>
  auto interpolating_basis_fft_grid(bool left, memory::array<MEM,long,1> const& IPts, int iq,
              nda::range k_range, nda::range a_range)
    -> memory::darray_t<memory::array<MEM,ComplexType,2>,mpi3::communicator>;

  template<MEMORY_SPACE MEM = HOST_MEMORY>
  auto interpolating_basis_nonuniform_rgrid(bool left, memory::array<MEM,long,1> const& IPts, int iq,
              nda::range k_range, nda::range a_range)
    -> memory::darray_t<memory::array<MEM,ComplexType,2>,mpi3::communicator>;

  /**
   * Solves normal equation given a set of interpolating points and three-index tensor B
   * (needs to be careful with stability issues)
   *
   *
   * @tparam MEM
   * @tparam return_coul_matrix
   * @param iq
   * @param IPoints
   * @param a_range
   * @param b_range
   * @param B
   * @return
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY, bool return_coul_matrix>
  auto intvec_impl(int iq, nda::MemoryArrayOfRank<1> auto const& IPoints,
        nda::range a_range, nda::range b_range,
        memory::darray_t<memory::array<MEM,ComplexType,5>,mpi3::communicator> const& B);

  /**
   * Solves normal equations for interpolating vectors with density fitting in the overlap metric,
   * using precalculated interpolating points, and calculate the Coulomb matrices.
   * All q-points are calculated simultaneously.
   *
   *      \phi^{k*}_a(r)\phi^{k-q}_b(r)
   *          = \sum_{\mu} \phi^{k*}_a(r_{\mu})\phi^{k-q}_b(r_{\mu}) \zeta^{q}_{\mu}(r)
   *
   * @param ri  - [INPUT] interpolating points
   * @param Xa  - [INPUT] orbital "a" on interpolating points: phi^{k*}_a(r_mu)
   * @param Xb  - [INPUT] orbital "b" on interpolating points: phi^{k-q*}_b(r_mu)
   * @param return_Sinv_Ivec - [INPUT] return inverse of overlap matrix for \zeta^{q}_mu(r)
   * @param a_range - [INPUT] range of orbital "a"
   * @param b_range - [INPUT] range of orbital "b"
   * @param pgrid3D - [INPUT] processor grid
   * @return A tuple containing:
   *         if return_coul_matrix == False:
   *           - Z_quG: Distributed array with interpolating vectors. Depending
   *             dims: (nqpts_ibz, nIpts, nG if (mf->orb_on_fft_grid()) else nr)
   *         else:
   *           - V_{\mu,\nu}: Distributed array with Coulomb matrix in the basis of interpolating vectors.
   *             dims: (nqpts_ibz, nIpts, nIpts), distributed along nIpts rows only.
   *           - \zeta^{q}_{\mu}(G=0):
   *           - \tilde{\zeta}^{q}_{\mu}(G=0):
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY, bool return_coul_matrix, typename Tensor_t, typename Tensor2_t>
  auto intvec_impl(nda::MemoryArrayOfRank<1> auto const& IPoints, 
        Tensor_t const& Xa,
        Tensor2_t const* Xb,
        bool return_Sinv_Ivec, nda::range a_range, nda::range b_range, 
        std::array<long, 3> pgrid3D={0,0,0});
  /**
   * Calculate the following quantity for orbitals stored on a non-uniform real-space grid:
   *     T_{uv} = \sum_{i} \sum_{k} \phi^{k*}_{i}(r_u)\phi^{k}_{i}(r_v)
   * This quantity is needed when solving the normal equations for interpolating vectors.
   * @param add_phase - [INPUT]
   * @param ispin     - [INPUT]
   * @param k         - [INPUT]
   * @param orb_range - [INPUT]
   * @param IPts      - [INPUT]
   * @param Tuv       - [OUTPUT]
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY>
  void get_Tuv_nonuniform_rgrid(bool add_phase, int ispin, int k,
        nda::range orb_range, memory::array<MEM,long,1> const& IPts,
        memory::darray_t<memory::array<MEM,ComplexType,2>,mpi3::communicator>& Tuv);

  /**
   * Calculate the following quantity for orbitals stored on a FFT grid:
   *     T{uv} = \sum_{i} \sum_{k} \phi^{k*}_{i}(r_u)\phi^{k}_{i}(r_v)
   * This quantity is needed when solving the normal equations for interpolating vectors.
   * @param add_phase  - [INPUT]
   * @param ispin      - [INPUT]
   * @param k          - [INPUT]
   * @param a_range    - [INPUT]
   * @param ru         - [INPUT]
   * @param Tuv        - [OUTPUT]
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY>
  void get_Tuv_fft_grid(bool add_phase, int ispin, int k, 
        nda::range a_range, memory::array<MEM,long,1> const& ru,
        memory::darray_t<memory::array<MEM,ComplexType,2>,mpi3::communicator>& Tuv);

  /**
   * Calculate the following quantity for orbitals stored on a FFT grid:
   *     T^{k}_{ur} = \sum_{i} \phi^{k*}_{i}(r_u)\phi^{k}_{i}(r)
   * This quantity is needed when solving the normal equations for interpolating vectors.
   * @param add_phase - [INPUT]
   * @param ispin     - [INPUT]
   * @param a_range   - [INPUT]
   * @param r_range   - [INPUT]
   * @param ru        - [INPUT]
   * @param Tkur      - [OUTPUT]
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY>
  void get_Tkur_fft_grid(bool add_phase, int ispin, nda::range a_range, nda::range r_range,
        memory::array<MEM,long,1> const& ru,
        memory::darray_t<memory::array<MEM,ComplexType,3>,mpi3::communicator>& Tkur);

  /**
   * Calculate the following quantity for orbitals stored on a non-uniform grid:
   *     T^{k}_{ur} = \sum_{i} \phi^{k*}_{i}(r_u)\phi^{k}_{i}(r)
   * This quantity is needed when solving the normal equations for interpolating vectors.
   * @param add_phase - [INPUT]
   * @param ispin     - [INPUT]
   * @param a_range   - [INPUT]
   * @param r_range   - [INPUT]
   * @param ru        - [INPUT]
   * @param Tkur      - [OUTPUT]
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY>
  void get_Tkur_nonuniform_rgrid(bool add_phase, int ispin, nda::range a_range, nda::range r_range,
         memory::array<MEM,long,1> const& ru,
         memory::darray_t<memory::array<MEM,ComplexType,3>,mpi3::communicator>& Tkur);
  /**
   *
   * @param ispin
   * @param kp_to_ibz
   * @param kp_order
   * @param Xa
   * @param psi
   * @param dT_g
   * @param dT_u
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY, typename dArray_t, typename dArray2_t>
  void get_Tkug(int ispin, int ipol, nda::array<int,1> const& kp_to_ibz,
                nda::array<int,1> const& kp_order,
                nda::ArrayOfRank<3> auto const& Xa, _darray_t_<MEM,5> const& psi,
                dArray_t& dT_g, dArray2_t& dT_u);

  template<MEMORY_SPACE MEM = HOST_MEMORY, typename dArray_t>
  void get_Tkug(int ispin, int ipol, nda::array<int,1> const& kp_to_ibz,
                nda::array<int,1> const& kp_order,
                nda::ArrayOfRank<3> auto const& Xa, _darray_t_<MEM,5> const& psi,
                dArray_t& dT_u);

  template<MEMORY_SPACE MEM = HOST_MEMORY, typename Tensor_t>
  auto Xskau_to_sXbkua(int ispin, nda::ArrayOfRank<1> auto const& iu_for_sXb,
                       Tensor_t const& Xa, nda::array<int,1>& kp_order);

  //template<MEMORY_SPACE MEM = HOST_MEMORY, typename Tensor_t>
  //auto Xskau_to_sXbkua(int ispin, nda::ArrayOfRank<2> auto const& iu_for_sXb,
  //                     Tensor_t const& Xa, nda::array<int,1>& kp_order);

  /**
   * Compute
   *     - Z^{q}_u(G) or Z^{q}_u(r):
   *          \sum_{ab} \sum_{k} \phi^{k}_a(r_u}) \phi^{k-q*}_b(r_u) \phi^{k*}_a(G or r) \phi^{k-q}_b(G or r)
   *     - C^{q}_{uv}: \sum_{ab} \sum_{k} \phi^{k}_a(r_u}) \phi^{k-q*}_b(r_u) \phi^{k*}_a(r_v) \phi^{k-q}_b(r_v)
   * These are intermediate quantities used in the normal equation for interpolating vectors.
   * @param IPts    - [INPUT] Interpolating points
   * @param Xa      - [INPUT] orbital "a" on interpolating points: phi^{k*}_a(r_mu)
   * @param Xb      - [INPUT] orbital "b" on interpolating points: phi^{k-q*}_b(r_mu)
   * @param a_range - [INPUT] range of orbital "a"
   * @param b_range - [INPUT] range of orbital "b"
   * @param pgrid   - [INPUT] processor grid for ZquG and Cquv
   * @return A tuple containing:
   *         - Z^{q}_u(r) or Z^{q}_u(G): Distributed array with interpolating vectors.
   *           dims: (nqpts_ibz, nIpts, nG if (mf->orb_on_fft_grid()) else nr)
   *         - Zquv: distributed array with dimensions: (nqpts_ibz, nIpts, n_Ipts)
   */
  template<MEMORY_SPACE MEM, typename Tensor_t, typename Tensor2_t>
  auto get_ZquG_Cquv(nda::MemoryArrayOfRank<1> auto const& IPts,
                     Tensor_t const& Xa,
                     Tensor2_t const* Xb,
                     nda::range a_range, nda::range b_range,
                     std::array<long, 3> pgrid); 

  /**
   * Compute the following quantities on a real-space grid:
   *     - Z^{q}_u(r): \sum_{ab} \sum_{k} \phi^{k}_a(r_u}) \phi^{k-q*}_b(r_u) \phi^{k*}_a(r) \phi^{k-q}_b(r)
   *     - C^{q}_{uv}: \sum_{ab} \sum_{k} \phi^{k}_a(r_u}) \phi^{k-q*}_b(r_u) \phi^{k*}_a(r_v) \phi^{k-q}_b(r_v)
   * Z^{q}_{u}(r) is Fourier transformed to the plane-waves basis if mf->orb_on_fft_grid() = true.
   * @param IPts    - [INPUT] Interpolating points
   * @param a_range - [INPUT] range of orbital "a"
   * @param b_range - [INPUT] range of orbital "b"
   * @param pgrid   - [INPUT] processor grid for ZquG and Cquv
   * @param block_size - [INPUT]
   * @return A tuple containing:
   *         - Z^{q}_u(r) or Z^{q}_u(G): Distributed array with interpolating vectors.
   *           dims: (nqpts_ibz, nIpts, nG if (mf->orb_on_fft_grid()) else nr)
   *         - Zquv: distributed array with dimensions: (nqpts_ibz, nIpts, n_Ipts)
   */
  template<MEMORY_SPACE MEM>
  auto get_ZquG_Cquv_rspace(nda::MemoryArrayOfRank<1> auto const& IPts,
                     nda::range a_range, nda::range b_range,
                     std::array<long, 3> pgrid,   
                     std::array<long, 3> block_size);

  template<typename Tensor_t, typename Tensor2_t>
  auto get_ZquG_Cquv_fft_shared_memory(nda::MemoryArrayOfRank<1> auto const& IPts,
                     Tensor_t const& Xa,
                     Tensor2_t const* Xb,
                     nda::range a_range, nda::range b_range,
                     std::array<long, 3> pgrid,   
                     std::array<long, 3> block_size);

  /**
   * Compute the following quantities on the plane-wave basis using FFT:
   *     - Z^{q}_u(G): \sum_{ab} \sum_{k} \phi^{k}_a(r_u}) \phi^{k-q*}_b(r_u) \phi^{k*}_a(G) \phi^{k-q}_b(G)
   *     - C^{q}_{uv}: \sum_{ab} \sum_{k} \phi^{k}_a(r_u}) \phi^{k-q*}_b(r_u) \phi^{k*}_a(r_v) \phi^{k-q}_b(r_v)
   * @param IPts    - [INPUT] Interpolating points
   * @param a_range - [INPUT] range of orbital "a"
   * @param b_range - [INPUT] range of orbital "b"
   * @param pgrid   - [INPUT] processor grid for ZquG and Cquv
   * @param block_size - [INPUT]
   * @return A tuple containing:
   *         - Z^{q}_u(G): Distributed array with interpolating vectors.
   *           dims: (nqpts_ibz, nIpts, nG)
   *         - Zquv: distributed array with dimensions: (nqpts_ibz, nIpts, n_Ipts)
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY, typename Tensor_t, typename Tensor2_t>
  auto get_ZquG_Cquv_fft(nda::MemoryArrayOfRank<1> auto const& IPts,
                     Tensor_t const& Xa,
                     Tensor2_t const* Xb,
                     nda::range a_range, nda::range b_range,
                     std::array<long, 3> pgrid,
                     std::array<long, 3> block_size);

  template<MEMORY_SPACE MEM = HOST_MEMORY, bool Ipts_only, bool return_Ruv, typename Tensor_t>
  auto chol_metric_impl(int iq, int nmax, nda::range a_range, nda::range b_range, int block_size, 
                        Tensor_t const* C_skai);

  /**
   * [symmetry-adapted version]
   * ISDF for phi^{k,*}_a(r)phi^{k-q}_b(r) at a given q-point using the Cholesky decomposition method
   * from Matthews D. A., J. Chem. Theory Comput. 2020, 16, 1382–1385.
   * @tparam Ipts_only   - interpolating points only
   * @tparam return_Ruv  -
   * @param iq           - [INPUT]  Index of the q-point
   * @param nmax         - Maximum number of interpolating points.
   *                       If max=-1, there will be no hard limit, and the number of
   *                       interpolating points is computed until the error is smaller
   *                       than this->thresh.
   * @param a_range  - [INPUT] Orbital range for phi^{k,*}_a
   * @param b_range  - [INPUT] Orbital range for phi^{k-q}_b
   * @param block_size - [INPUT] block size for the iterative pivoted Cholesky algorithm
   * @return A tuple containing:
   *         if Ipts_only is True:
   *           - index of interpolating points: (Np)
   *           - distributed array for phi^{k}_a on interpolating points: (ns, nkpts, nbnd_a, Np)
   *           - distributed array for phi^{k-q}_b on interpolating points: (ns, nkpts, nbnd_b, Np)
   *         else:
   *           if return_Ruv is True:
   *             - index of interpolating points: (Np)
   *             -
   *             - distributed array for phi^{k}_a on interpolating points: (ns, nkpts, nbnd_a, Np)
   *             - distributed array for phi^{k-q}_b on interpolating points: (ns, nkpts, nbnd_b, Np)
   *           else:
   *             - index of interpolating points: (Np)
   *             -
   *             - distributed array for phi^{k}_a on interpolating points: (ns, nkpts, nbnd_a, Np)
   *             - distributed array for phi^{k-q}_b on interpolating points: (ns, nkpts, nbnd_b, Np)
   */
  template<MEMORY_SPACE MEM = HOST_MEMORY, bool Ipts_only, bool return_Ruv>
  auto chol_metric_impl_ibz(int iq, int nmax, nda::range a_range, nda::range b_range, int block_size);
  //auto chol_metric_impl(int iq, int nmax, nda::range a_range, nda::range b_range, int block_size);


  template<MEMORY_SPACE MEM = HOST_MEMORY>
  auto load_basis_subset_nonuniform_rgrid(nda::MemoryArrayOfRank<1> auto const& IPts,
          int iq, nda::range kp_rg, nda::range a_rg, nda::range b_rg);

  template<MEMORY_SPACE MEM = HOST_MEMORY>
  auto load_basis_subset_fft_grid(nda::MemoryArrayOfRank<1> auto const& IPts,
          int iq, nda::range kp_rg, nda::range a_rg, nda::range b_rg);

  template<typename Arr>
  auto chol(Arr& A, nda::array<int,1>& piv, double cut);

  void set_range(nda::range& a_range); 

  void set_k_range(nda::range& k_range); 

  long get_nblocks_nnr() { return nnr_blk; }
};

} // methods

#endif
