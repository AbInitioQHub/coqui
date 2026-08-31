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


#ifndef COQUI_MBSTATE_H
#define COQUI_MBSTATE_H

#include "IO/app_loggers.h"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/embedding/projector_boson_t.h"

namespace methods {

/**
 * @class MBState
 * @brief Container of a many-body Green's function solution
 *
 * This class encapsulates components of a many-body Green's function solution.
 * It serves as a state holder which can be used as input/output of a
 * many-body electronic structure method.
 *
 * Note that whatever is known before a many-body calculation does not belong to this class.
 * Those components are part of the definition of a `system`.
 */
struct MBState {

public:
  using mpi_context_t = utils::mpi_context_t<>;
  template<nda::MemoryArray Array_base_t>
  using sArray_t = math::shm::shared_array<Array_base_t>;
  template<nda::MemoryArray Array_base_t>
  using dArray_t = memory::darray_t<Array_base_t, mpi3::communicator>;
  template<int N>
  using shape_t = std::array<long,N>;

public:
  // mbpt constructor
  MBState(std::shared_ptr<mpi_context_t> mpi_in, imag_axes_ft::IAFT &ft_in,
          std::string prefix, bool restart_from_checkpoint=false);

  // embedding constructors with MLWFs projector provided either directly or via hdf5 file
  MBState(imag_axes_ft::IAFT &ft_in, std::string prefix,
          std::shared_ptr<mf::MF> &mf, std::string C_file, bool translate_home_cell=false,
          bool restart_from_checkpoint=false);
  MBState(imag_axes_ft::IAFT &ft_in, std::string prefix,
          std::shared_ptr<mf::MF> &mf, const nda::array<ComplexType, 5> &C_ksIai,
          const nda::array<long, 3> &band_window, const nda::array<RealType, 2> &kpts_crys,
          bool translate_home_cell=false, bool restart_from_checkpoint=false);

  MBState(MBState const&) = default;
  MBState(MBState &&) = default;
  MBState & operator=(const MBState &) = default;
  MBState & operator=(MBState &&) = default;

  ~MBState(){}

  // TODO Function to read MBPT data

  void set_zero_local_polarizabilities();
  bool read_local_polarizabilities(long weiss_b_iter=-1);
  void set_local_polarizabilities(std::map<std::string, nda::array<ComplexType, 5>> local_polarizabilities);
  void set_local_hf_potentials(std::map<std::string, nda::array<ComplexType, 4>> local_hf_potentials);
  void set_local_selfenergies(std::map<std::string, nda::array<ComplexType, 5>> local_selfenergies);
  bool has_local_selfenergies();

public:
  std::shared_ptr<mpi_context_t> mpi;
  imag_axes_ft::IAFT* ft = nullptr;
  // do we need to store the mean-field object?

  // checkpoint file info
  std::string coqui_prefix = "coqui";
  long mbpt_iter = -1;  // iteration number of the many-body solution
  long df_1e_iter = -1; // iteration number of the downfolded 1e Hamiltonian
  long df_2e_iter = -1; // iteration number of the downfolded 2e Hamiltonian
  long embed_iter = -1; // iteration number of the embedding solution

  /** all components of a many-body state are optional **/
  /** lattice quantities **/
  // Green's function
  std::optional<sArray_t<nda::array_view<ComplexType, 5> > > sG_tskij;
  // Density matrix
  std::optional<sArray_t<nda::array_view<ComplexType, 4> > > sDm_skij;
  // Dynamic part of the self-energy
  std::optional<sArray_t<nda::array_view<ComplexType, 5> > > sSigma_tskij;
  // Static part of the self-energy
  std::optional<sArray_t<nda::array_view<ComplexType, 4> > > sF_skij;
  // Dynamically screened interaction w/o bare term
  std::optional<dArray_t<nda::array<ComplexType, 4> > > dW_qtPQ;
  // Flag to determine screening recipe to calculate the dynamically screened interaction. 
  std::string screen_type = "";
  // Head (G=G'=0) of the inverse dielectric function in the long wavelength limit (q->0 and w->0)
  std::optional<nda::array<ComplexType, 1> > eps_inv_head;

  /** Quasiparticle SCF specific */
  // Effective QP Hamiltonian in QP-SCF
  std::optional<sArray_t<nda::array_view<ComplexType, 4> > > sHeff_skij;
  // QP coefficients
  std::optional<sArray_t<nda::array_view<ComplexType, 4> > > sMO_skia;
  // QP energies
  std::optional<sArray_t<nda::array_view<ComplexType, 3> > > sE_ska;

  
  /** local quantities in MLWF basis for quantum embedding */
  // Projector from Bloch states to MLWFs. 
  std::optional<projector_boson_t> proj_boson;
  // Dynamic part of the impurity self-energy
  std::optional<nda::array<ComplexType, 5> > Sigma_imp_wsIab;
  // Static part of the impurity self-energy
  std::optional<nda::array<ComplexType, 4> > Vhf_imp_sIab;
  // Dynamic part of the double-counting self-energy
  std::optional<nda::array<ComplexType, 5> > Sigma_dc_wsIab;
  // Static part of the double-counting self-energy
  std::optional<nda::array<ComplexType, 4> > Vhf_dc_sIab;
  // Impurity polarizability
  std::optional<sArray_t<nda::array_view<ComplexType, 5> > > sPi_imp_wabcd;
  // Double-counting polarizability
  std::optional<sArray_t<nda::array_view<ComplexType, 5> > > sPi_dc_wabcd;

  // Quasiparticle approx. to the dynamic part of the double-counting self-energy
  std::optional<nda::array<ComplexType, 4> > Vcorr_dc_sIab;

};

} // methods

#endif //COQUI_MBSTATE_H
