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


#ifndef HAMILTONIAN_PSEUDO_NCPP_H
#define HAMILTONIAN_PSEUDO_NCPP_H

#include <iostream>
#include <memory>
#include <string>

#include "configuration.hpp"
#include "hamiltonian/pseudo/pseudopot_type.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "mean_field/mf_source.hpp"
#include "utilities/symmetry.hpp"

namespace hamilt 
{

/**
 * @class pseudopot
 * @brief Handler for the pseudopotential of a given mean-field object
 *
 * This class is responsible for computing and storing the pseudopotential for a
 * specified physical system. The system, including `nbnd`, `nkpt`, Brillouin zone info,
 * and "pseudopotential", are defined upon construction through a mean-field handler.
 *
 * Contributions of the pseudopotential can be evaluated by calling
 *
 *     pseudopot.add_Vpp(..., hpsi, Hij);
 *
 * where
 *   1. Contributions from the local potentials are added to wavefunctions "hpsi"
 *   2. Contributions from the non-local potentials are added to the Hamiltonian
 *      in second quantization "Hij"
 *
 * In addition, this class also handle the Hartree potential for a given density matrix `nij`
 *
 *     pseudopot.add_Hartree(..., nij, ..., hpsi);
 *
 * in which the Hartree potential will be added to `hpsi`.
 *
 * @tparam MF_t - Type parameter for the mean-field object
 */
class pseudopot
{
  template<typename Arr>
  using sarray_t = typename math::shm::shared_array<Arr>;

  public:

  using mpi_t = utils::mpi_context_t<mpi3::communicator,mpi3::shared_communicator>;

  template<typename MF_t>
  pseudopot(MF_t &mf, std::string const filename = "");

  ~pseudopot() {}

  pseudopot(pseudopot const&) = default;
  pseudopot(pseudopot &&) = default;
  pseudopot& operator=(pseudopot const&) = default;
  pseudopot& operator=(pseudopot &&) = default;

  pp_type_e pp_type() const { return ptype; }

  void save(std::string fname, bool append = true);
  void save(h5::group& grp);

  std::shared_ptr<mpi_t> get_mpi_context() { return mpi; }

  /**
   * Add the contributions of norm-conserving pseudopotentials:
   *
   *   1. Contributions from the local potentials are added to wavefunctions "hpsi"
   *   2. Contributions from the non-local potentials are added to the Hamiltonian
   *      in second quantization "Hij"
   *
   * @param k_range - [input] Range of k-point indices
   * @param b_range - [input] Range of orbital indices "b"
   * @param psi     - [input] Single-particle basis
   * @param hpsi    - [input] \hat{h} * psi where h is an arbitrary local operator
   *                  [output] (\hat{h} + vloc) * psi
   * @param Hij     - [input] Matrix elements of an arbitrary non-local operator H_nl
   *                  [output] Matrix elements of H_nl + Vpp_nl,
   *                           where Vpp_nl is the non-local part of the pseudopotential
   */
  void add_Vpp(boost::mpi3::communicator& comm, nda::range k_range, nda::range b_range,
               math::nda::DistributedArrayOfRank<4> auto const& psi,
               math::nda::DistributedArrayOfRank<4> auto & hpsi,
               math::nda::DistributedArrayOfRank<4> auto & Hij);

  void add_Vpp(boost::mpi3::communicator& comm, nda::range k_range, nda::range b_range,
               nda::ArrayOfRank<3> auto const& nii,
               math::nda::DistributedArrayOfRank<4> auto const& psi,
               math::nda::DistributedArrayOfRank<4> auto & hpsi,
               math::nda::DistributedArrayOfRank<4> auto & Hij);

  void add_Vpp(boost::mpi3::communicator& comm, nda::range k_range, nda::range b_range,
               nda::ArrayOfRank<4> auto const& nij,
               math::nda::DistributedArrayOfRank<4> auto const& psi,
               math::nda::DistributedArrayOfRank<4> auto & hpsi,
               math::nda::DistributedArrayOfRank<4> auto & Hij);

  /**
   * Add the contributions of the Hartree potential to the wavefunctions "hpsi"
   *
   * @param k_range - [input] Range of k-point indices
   * @param nii     - [input] Diagonal density matrix (s, k, a)
   * @param psi     - [input] Single-particle basis (s, k, a, g), where g lives in the "wavefunction" grid
   * @param hpsi    - [input] \hat{H_loc} * psi, where H_loc is an arbitrary local operator
   *                  [output] (\hat{H_loc} + V_H) * psi,
   *                           where Vpp_loc is the local part of the pseudopotential
   */
  void add_Hartree(nda::range k_range,
                   nda::ArrayOfRank<3> auto const& nii,
                   math::nda::DistributedArrayOfRank<4> auto const& psi,
                   math::nda::DistributedArrayOfRank<4> auto & hpsi,
                   bool symmetrize=false);

  /**
   * Add the contributions of the Hartree potential to the wavefunctions "hpsi"
   *
   * @param k_range - [input] Range of k-point indices
   * @param nij     - [input] Density matrix (s, k, a, b)
   * @param psi     - [input] Single-particle basis (s, k, a, g), where g lives in the "wavefunction" grid
   * @param hpsi    - [input] \hat{H_loc} * psi, where H_loc is an arbitrary local operator
   *                  [output] (\hat{H_loc} + V_H) * psi,
   *                           where Vpp_loc is the local part of the pseudopotential
   */
  void add_Hartree(nda::range k_range,
                   nda::ArrayOfRank<4> auto const& nij,
                   math::nda::DistributedArrayOfRank<4> auto const& psi,
                   math::nda::DistributedArrayOfRank<4> auto & hpsi,
                   bool symmetrize=false);

  private:

  // mpi communicators
  std::shared_ptr<mpi_t> mpi;

  // pseudo type, default to ncpp and update in constructor
  pp_type_e ptype = pp_ncpp_t;

  // input type, needed for save
  mf::mf_input_file_type_e input_file_type = mf::xml_input_type;

  // input file, needed for save
  std::string input_file_name = "";

  // basic system info
  nda::stack_array<int,3> fft_mesh;
  long nnr = 0;
 
  // reciprocal lattice vectors
  nda::stack_array<double,3,3> recv;

  // reciprocal lattice vectors
  nda::stack_array<double,3,3> lattv;

  // spin-orbit
  bool spinorbit_loc = false;
  bool spinorbit_nl = false;

  // number of spins 
  int nspin = 1;

  // number of polarizations
  int npol = 1;

  /* kpoints and symmetry properties */
  long nkpts = 0;
  long nkpts_ibz = 0;
  nda::array<double, 2> kpts;      // in cartesian coordinates
  nda::array<double, 2> kpts_crys; // in crystal coordinates
  nda::array<int, 1> kp_to_ibz;
  nda::array<bool, 1> kp_trev; // symmetry operations
  std::vector<utils::symm_op> symm_list; // symmetry operations
  nda::array<int, 1> kp_symm;   // index of symmetry operation that connects kpts/kpts_crys to IRBZ

  // type of pseudo for each atom
  nda::array<int,1> ityp;

  // number of projectors for each pseudo typle
  nda::array<int,1> nh;

  // index of first projector for each atom 
  nda::array<int,1> ofs;

  // qq
  memory::unified_array<ComplexType,1> qq;

  // Matrix elements between projectors and basis orbitals (in mf)
  sarray_t<nda::array_view<ComplexType,4>> Pskna;

  // D matrix for local projectors
  //memory::unified_array<ComplexType,3> Dnn;
  sarray_t<nda::array_view<ComplexType,3>> Dnn;

  // mapping from wfc_g grid to rho grid. 
  // hard coding ecut in mf now, allow for a custom cutoff later on
  sarray_t<nda::array_view<long,1>> swfc_to_rho;

  // local pseudopotential
  sarray_t<nda::array_view<ComplexType,3>> svloc;

  // scf local potential
  sarray_t<nda::array_view<ComplexType,3>> svsc;

  // qgm
  sarray_t<nda::array_view<ComplexType,3>> qgm;

  template<typename MF_t>
  void read_vnl_pw2bgw(MF_t &mf, std::string outdir); 

  template<typename MF_t>
  void read_vnl_h5(MF_t &mf, h5::group& grp); 

  void add_vnl_impl(nda::range k_range, nda::range b_range, 
               nda::ArrayOfRank<3> auto const& Dion, 
               math::nda::DistributedArrayOfRank<4> auto & Hij);

  /**
   * Add the contributions of a generic pseudopotentials:
   *
   *   1. Contributions from the local potentials are added to wavefunctions "hpsi"
   *   2. Contributions from the non-local potentials are added to the Hamiltonian
   *      in second quantization "Hij"
   *
   * @tparam Arr3   - Array type of nii
   * @tparam Arr4   - Array type of nij
   * @param k_range - [input] Range of k-point indices
   * @param b_range - [input] Range of orbital indices "b"
   * @param psi     - [input] Single-particle basis
   * @param hpsi    - [input] \hat{H_loc} * psi, where H_loc is an arbitrary local operator
   *                  [output] (\hat{H_loc} + Vpp_loc) * psi,
   *                           where Vpp_loc is the local part of the pseudopotential
   * @param Hij     - [input] Matrix elements of an arbitrary non-local operator H_nl
   *                  [output] Matrix elements of H_nl + Vpp_nl,
   *                           where Vpp_nl is the non-local part of the pseudopotential
   * @param nii     - [input] Diagonal density matrix (s, k, a)
   * @param nij     - [input] Density matrix (s, k, a, b)
   */
  template< nda::ArrayOfRank<3> Arr3, nda::ArrayOfRank<4> Arr4>
  void add_vpp_impl(boost::mpi3::communicator& comm,
               nda::range k_range, nda::range b_range, 
               math::nda::DistributedArrayOfRank<4> auto const& psi,
               math::nda::DistributedArrayOfRank<4> auto & hpsi,
               math::nda::DistributedArrayOfRank<4> auto & Hij,
               const Arr3 * nii, const Arr4 * nij);

  /**
   * Add the contributions of the Hartree potential to the wavefunctions "hpsi"
   *
   * @tparam Arr3   - Array type of "nii" array
   * @tparam Arr4   - Array type of "nij" array
   * @param k_range - [input] Range of k-point indices
   * @param psi     - [input] Single-particle basis (s, k, a, g), where g lives in the "wavefunction" grid
   * @param hpsi    - [input] \hat{H_loc} * psi, where H_loc is an arbitrary local operator
   *                  [output] (\hat{H_loc} + V_H) * psi,
   *                           where Vpp_loc is the local part of the pseudopotential
   * @param nii     - [input] Diagonal density matrix (s, k, a)
   * @param nij     - [input] Density matrix (s, k, a, b). Note that either "nii" or "nij"
   *                          should be provided, not both.
   */
  template<nda::ArrayOfRank<3> Arr3, nda::ArrayOfRank<4> Arr4>
  void add_Hartree_impl(nda::range k_range,
                        math::nda::DistributedArrayOfRank<4> auto const& psi,
                        math::nda::DistributedArrayOfRank<4> auto & hpsi,
                        const Arr3 *nii, const Arr4 *nij, bool symmetrize=false);


};

// if mf.get_pseudopot() returns a valid shared pointer, return it.
// otherwise, construct a new object managed by a shared pointer, 
// store the pointer in mf and return it.
template<typename MF_t>
std::shared_ptr<pseudopot> make_pseudopot(MF_t &mf)
{
  // sync for safety for now, this routine is blocking
  auto mpi = mf.mpi();
  mpi->comm.barrier();
  if(mf.get_pseudopot()) { return mf.get_pseudopot(); }
  else { 
    //
    auto psp = std::make_shared<pseudopot>(mf);
    mf.set_pseudopot(psp);
    if( not mf.get_pseudopot() )
      APP_ABORT("Error in make_pseudopot. Logic problem.");
    return psp;
  }
}

}

#endif
