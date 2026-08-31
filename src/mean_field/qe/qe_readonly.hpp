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


#ifndef MEANFIELD_QE_QE_READONLY_H
#define MEANFIELD_QE_QE_READONLY_H

#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <optional>
#include <algorithm>
#include "IO/AppAbort.hpp"
#include "configuration.hpp"
#include <nda/nda.hpp>
#include <nda/tensor.hpp>
#include <nda/h5.hpp>
#include "numerics/sparse/sparse.hpp"
#include "utilities/concepts.hpp"
#include "utilities/mpi_context.h"

#include "mean_field/mf_source.hpp"
#include "mean_field/qe/qe_interface.h"
#include "numerics/fft/nda.hpp"
#include "utilities/fortran_utilities.h"
#include "utilities/qe_utilities.hpp"
#include "grids/g_grids.hpp"
#include "hamiltonian/pseudo/pseudopot.h"

namespace mf
{
namespace qe 
{

namespace detail
{

/*
 * Reads miller indexes from h5 file and determines an appropriate wfc fft grid.
 */ 
grids::truncated_g_grid wfc_grid_from_h5(qe_system const& sys);

}

 /*
 * Provides interface and read-only access to data in Quantum Espresso's hdf5 checkpoint file.
 * Files are open "on-demand" and closed either at destruction or by a call to close().
 * Details here...
 */
class qe_readonly 
{
public:

  static constexpr mf_source_e mf_src = qe_source;
  static constexpr mf_source_e get_mf_source() { return mf_src; }
  static constexpr bool has_orbital_set() { return true; }

  // accessor functions
  auto mpi() const { return sys.mpi; }
  long nbnd() const { return sys.nbnd; }
  long nbnd_aux() const { return sys.nbnd_aux; }
  int fft_grid_size() const { return fft_mesh(0)*fft_mesh(1)*fft_mesh(2); }
  int nnr() const { return fft_grid_size(); }
  decltype(auto) fft_grid_dim() const { return fft_mesh(); }
  decltype(auto) lattice() const { return sys.latt(); }
  decltype(auto) recv() const { return sys.recv(); }
  decltype(auto) kpts() { return sys.bz().kpts(); }
  decltype(auto) kpts_crystal() { return sys.bz().kpts_crys(); }
  int nkpts() const { return sys.bz().nkpts; }
  int nkpts_ibz() const { return sys.bz().nkpts_ibz; }
  decltype(auto) kp_trev() { return sys.bz().kp_trev(); }
  qe_system const& get_sys() const { return sys; }
  decltype(auto) wfc_truncated_grid() const {
    return std::addressof(wfc_g);
  }
  bool has_wfc_grid() const { return true; }
  auto const& bz() const { return sys.bz(); }

public:

//MAM: number of files is missing factor of 2 in UHF case
  template<utils::Communicator comm_t>
  qe_readonly(std::shared_ptr<utils::mpi_context_t<comm_t>> mpi,
              std::string outdir, std::string prefix,
              double ecut_ = 0.0,
              long n_ = -1,
              mf_input_file_type_e fmt = xml_input_type) : 
// if fmt==qe_h5_storage and the h5 file has all the necessary information (make static check function),
// add routine to restart from h5 directly without having to reconstruct bz_symm info
    sys( (fmt==xml_input_type) ? read_xml(std::move(mpi),outdir,prefix,n_,false) :
                                 read_h5(std::move(mpi),outdir,prefix,n_) ),
    h5files(sys.nspin*sys.bz().nkpts_ibz,std::nullopt),
    k2g_list(sys.bz().nkpts,std::nullopt),
    wfc_k2g_list(sys.bz().nkpts,std::nullopt),
    ecut( ecut_>0.0 ? ecut_ : sys.ecutrho ),
    fft_mesh( ecut_>0.0 ? nda::stack_array<int, 3>{grids::find_fft_mesh(sys.mpi->comm,ecut,sys.recv,sys.bz().symm_list)} : sys.fft_mesh),
    wfc_g(detail::wfc_grid_from_h5(sys))
  {
    // build symmetry rotations
    auto slist = utils::find_inverse_symmetry(sys.bz().qsymms,sys.bz().symm_list);
    if(slist.size() > 0) 
      std::tie(sk_to_n,dmat) = utils::generate_dmatrix<true>(*this,sys.bz().symm_list,slist);

    print_metadata();
  }

  qe_readonly(qe_system const& qes, double ecut_ = 0.0) :
    sys(qes),
    h5files(sys.nspin*sys.bz().nkpts_ibz,std::nullopt),
    k2g_list(sys.bz().nkpts,std::nullopt),
    wfc_k2g_list(sys.bz().nkpts,std::nullopt),
    ecut( ecut_>0.0 ? ecut_ : sys.ecutrho ),
    fft_mesh( ecut_>0.0 ? nda::stack_array<int, 3>{grids::find_fft_mesh(sys.mpi->comm,ecut,sys.recv,sys.bz().symm_list)} : sys.fft_mesh),
    wfc_g(detail::wfc_grid_from_h5(sys))
  {
    // build symmetry rotations
    auto slist = utils::find_inverse_symmetry(sys.bz().qsymms,sys.bz().symm_list);
    if(slist.size() > 0) 
      std::tie(sk_to_n,dmat) = utils::generate_dmatrix<true>(*this,sys.bz().symm_list,slist);

    print_metadata();
  }

  qe_readonly(qe_system && qes, double ecut_ = 0.0) :
    sys(std::move(qes)),
    h5files(sys.nspin*sys.bz().nkpts_ibz,std::nullopt),
    k2g_list(sys.bz().nkpts,std::nullopt),
    wfc_k2g_list(sys.bz().nkpts,std::nullopt),
    ecut( ecut_>0.0 ? ecut_ : sys.ecutrho ),
    fft_mesh( ecut_>0.0 ? nda::stack_array<int, 3>{grids::find_fft_mesh(sys.mpi->comm, ecut, sys.recv, sys.bz().symm_list)} : sys.fft_mesh),
    wfc_g(detail::wfc_grid_from_h5(sys))
  {
    // build symmetry rotations
    auto slist = utils::find_inverse_symmetry(sys.bz().qsymms, sys.bz().symm_list);
    if(slist.size() > 0)
      std::tie(sk_to_n,dmat) = utils::generate_dmatrix<true>(*this, sys.bz().symm_list, slist);

    print_metadata();
  }

  qe_readonly(qe_readonly const& other) : 
    sys(other.sys),
    h5files(sys.nspin*sys.bz().nkpts_ibz,std::nullopt),
    k2g_list(sys.bz().nkpts,std::nullopt),
    wfc_k2g_list(sys.bz().nkpts,std::nullopt),
    ecut(other.ecut),
    fft_mesh(other.fft_mesh),
    wfc_g(other.wfc_g),
    sk_to_n(other.sk_to_n),
    dmat( other.dmat ) {}

  qe_readonly(qe_readonly && other) : 
    sys(std::move(other.sys)),
    h5files(sys.nspin*sys.bz().nkpts_ibz,std::nullopt),
    k2g_list(sys.bz().nkpts,std::nullopt),
    wfc_k2g_list(sys.bz().nkpts,std::nullopt),
    ecut( other.ecut ),
    fft_mesh(other.fft_mesh),
    wfc_g(std::move(other.wfc_g)),
    sk_to_n(std::move(other.sk_to_n)),
    dmat( std::move(other.dmat) ) {}

  ~qe_readonly() { close(); }

  qe_readonly& operator=(qe_readonly const& other)
  {
    this->sys = other.sys;
    this->sk_to_n = other.sk_to_n;
    this->dmat = other.dmat;
    close(); // close files, open on demand
    this->ecut = other.ecut;
    this->fft_mesh = other.fft_mesh;
    this->wfc_g = other.wfc_g;
    return *this;
  }

  qe_readonly& operator=(qe_readonly && other)
  {
    this->sys = std::move(other.sys);
    this->sk_to_n = std::move(other.sk_to_n);
    this->dmat = std::move(other.dmat);
    close(); // close files, open on demand
    this->ecut = other.ecut;
    this->fft_mesh = other.fft_mesh;
    this->wfc_g = std::move(other.wfc_g);
    return *this;
  }

  void print_metadata() {
    app_log(1,"  Quantum ESPRESSO reader");
    app_log(1,"  -----------------------");
    app_log(1,"  Number of spins                = {}", sys.nspin);
    app_log(1,"  Number of polarizations        = {}", sys.npol);
    app_log(1,"  Number of bands                = {}", sys.nbnd);
    app_log(1,"  Monkhorst-Pack mesh            = ({},{},{})", sys.bz().kp_grid(0), sys.bz().kp_grid(1), sys.bz().kp_grid(2));
    app_log(1,"  K-points                       = {} total, {} in the IBZ", sys.bz().nkpts, sys.bz().nkpts_ibz);
    app_log(1,"  Number of electrons            = {}", sys.nelec);
    app_log(1,"  Electron density energy cutoff = {0:.3f} a.u. | FFT mesh = ({1},{2},{3})",
            ecut, fft_mesh(0),fft_mesh(1),fft_mesh(2));
    app_log(1,"  Wavefunction energy cutoff     = {0:.3f} a.u. | FFT mesh = ({1},{2},{3}), Number of PWs = {4}\n",
            wfc_g.ecut(), wfc_g.mesh(0),wfc_g.mesh(1),wfc_g.mesh(2), wfc_g.size());
  }

  // read orbital
  /*
   * OT: Orbital Type:
   *   -'r': real space density fft grid 
   *   -'g': fourier space density fft grid 
   *   -'w': truncated wfc g grid. 
   */ 
  template<nda::ArrayOfRank<1> A1D>
  void get_orbital(char OT, int ispin, int ik, int m, A1D&& Orb, nda::range r = {-1,-1})
  {
    utils::check(OT=='r' or OT=='g' or OT=='w',"Unknown orbital type in qe_readonly::get_orbital.");
    check_dimensions(OT,Orb,r);
    orbital_from_h5(OT,ispin,ik,m,Orb,r);
  }

  // read multiple orbitals
  template<nda::ArrayOfRank<2> A2D>
  void get_orbital_set(char OT, int ispin, int k, nda::range b_rng, A2D&& Orb, nda::range r = {-1,-1}) 
  {
    static_assert(std::decay_t<A2D>::layout_t::is_stride_order_C(), "Layout mismatch.");
    utils::check(OT=='r' or OT=='g' or OT=='w',"Unknown orbital type in qe_readonly::get_orbital.");
    utils::check(Orb.shape()[0] >= b_rng.size(), "Dimension mismatch.");
    nda::range r_=r;
    check_dimensions(OT,Orb,r_);
    orbital_set_from_h5(OT,ispin,k,b_rng,Orb,r_);
  }

  template<nda::ArrayOfRank<3> A3D>
  void get_orbital_set(char OT, int ispin, nda::range k_rng, nda::range b_rng, A3D&& Orb, nda::range r = {-1,-1}) 
  {
    static_assert(std::decay_t<A3D>::layout_t::is_stride_order_C(), "Layout mismatch.");
    utils::check(OT=='r' or OT=='g' or OT=='w',"Unknown orbital type in qe_readonly::get_orbital.");
    utils::check(Orb.shape()[0] >= k_rng.size() and 
                 Orb.shape()[1] >= b_rng.size(), "Dimension mismatch.");
    for( auto [ik,k] : itertools::enumerate(k_rng) ) {
      nda::range r_=r;
      check_dimensions(OT,Orb(ik,nda::ellipsis{}),r_);
      orbital_set_from_h5(OT,ispin,k,b_rng,Orb(ik,nda::ellipsis{}),r_);
    }
  }

  template<nda::ArrayOfRank<4> A4D>
  void get_orbital_set(char OT, int ispin, nda::range k_rng, nda::range b_rng, nda::range p_rng, A4D&& Orb, nda::range r = {-1,-1})
  {
    static_assert(std::decay_t<A4D>::layout_t::is_stride_order_C(), "Layout mismatch.");
    utils::check(OT=='r' or OT=='g' or OT=='w',"Unknown orbital type in qe_readonly::get_orbital.");
    utils::check(Orb.shape()[0] >= k_rng.size() and
                 Orb.shape()[1] >= b_rng.size() and 
                 Orb.shape()[2] >= p_rng.size(), "Dimension mismatch.");
    for( auto [ik,k] : itertools::enumerate(k_rng) ) {
      nda::range r_=r;
      check_dimensions(OT,Orb(ik,nda::ellipsis{}),r_);
      orbital_set_from_h5(OT,ispin,k,b_rng,p_rng,Orb(ik,nda::ellipsis{}),r_);
    }
  }

  decltype(auto) symmetry_rotation(long s, long k) const
  {
    long ns = sys.bz().qsymms.extent(0);
    long nk = sys.bz().nkpts;
    utils::check(s>0, "Symmetry index must be > 0, since s==0 is the identity and not stored.");
    utils::check(s>0 and s < ns, "out of bounds.");
    utils::check(k>=0 and k < nk, "out of bounds.");
    long n;
    if(sys.bz().kp_trev(k))
      n = sk_to_n(s-1,sys.bz().kp_trev_pair(k));
    else
      n = sk_to_n(s-1,k);
    utils::check(dmat.size() > n, " Error: Incorrect dmat dimensions.");
    return std::make_tuple(sys.bz().kp_trev(k),  std::addressof(dmat.at(n)));
  }

  void set_pseudopot(std::shared_ptr<hamilt::pseudopot> const& psp_) { psp = psp_; } 
  std::shared_ptr<hamilt::pseudopot> get_pseudopot() { return psp; } 

  void close()
  {
    for(auto& f: h5files)
      if( f.has_value() )
        f = std::nullopt;
    for(auto& f: k2g_list)
      if( f.has_value() )
        f = std::nullopt;
    for(auto& f: wfc_k2g_list)
      if( f.has_value() )
        f = std::nullopt;
  } 

private:

  // qe_system structure
  qe_system sys;    

  // h5 handles for k-point files
  std::vector<std::optional<h5::file>> h5files;

// this should really be long!!!
  // k2g: mapping of G-vectors between "kinetic" and "FFT" grids 
  std::vector<std::optional<nda::array<int,1>>> k2g_list;

  // wfc_k2g: mapping of G-vectors between "kinetic" and  wfc grid 
  std::vector<std::optional<nda::array<int,1>>> wfc_k2g_list;

  // shared ptr to pseudopot object. Not constructed here.
  // can be set from the outside to avoid recomputing. 
  std::shared_ptr<hamilt::pseudopot> psp;

  // plane wave cutoff for the FFT grid. 
  double ecut;

  // fft mesh compatible with ecut
  nda::stack_array<int, 3> fft_mesh;

  // truncated wfc g grid
  grids::truncated_g_grid wfc_g;

  // matrices that define symmetry relations between wavefunctions at different k-points
  nda::array<int,2> sk_to_n;
  std::vector< math::sparse::csr_matrix<ComplexType,HOST_MEMORY,int,int> > dmat;

  int to_h5_index(int ispin, int ik) {
    if(sys.nspin==2) 
      utils::check(ispin==0 or ispin==1, "ispin != 0/1 in collinear");
    else 
      utils::check(ispin==0, "ispin!=0 without nspin=2");
    return sys.bz().kp_to_ibz(ik) + ispin*sys.bz().nkpts_ibz;
  }

  void open_if_needed(char OT, int index, int ispin, int ik) {
    if( not h5files[index].has_value() ) { 
      utils::check(sys.nspin==1 or sys.nspin==2, "Error: Invalid nspin:{}",sys.nspin);
      if(sys.nspin==1)
        h5files[index] = std::make_optional<h5::file>(sys.outdir+"/"+sys.prefix+".save/wfc"+std::to_string(sys.bz().kp_to_ibz(ik)+1)+".hdf5",'r'); 
      else if(ispin==0)
        h5files[index] = std::make_optional<h5::file>(sys.outdir+"/"+sys.prefix+".save/wfcup"+std::to_string(sys.bz().kp_to_ibz(ik)+1)+".hdf5",'r'); 
      else if(ispin==1)
        h5files[index] = std::make_optional<h5::file>(sys.outdir+"/"+sys.prefix+".save/wfcdw"+std::to_string(sys.bz().kp_to_ibz(ik)+1)+".hdf5",'r'); 
      else
        APP_ABORT("Error: Invalid ispin: {}",ispin);
    }
    // MAM: assuming that k2g mapping is spin independent, which it should be...
    if( (OT=='r' or OT=='g') and not k2g_list[ik].has_value() ) {
      k2g_list[ik] = std::make_optional<nda::array<int,1>>(sys.npw[ik],0);
      nda::array<int,2> miller(sys.npw[ik],3);
      h5_read(*(h5files[index]), "/MillerIndices", miller);
      auto& k2g = *(k2g_list[ik]);
      utils::generate_k2g(miller,k2g,fft_mesh);
      if(ik >= sys.bz().nkpts_ibz) {
        nda::array<ComplexType,1> *Xft = nullptr;
        nda::stack_array<double, 3> Gs;
        Gs() = 0; // Gs = ??? , these should be stored in sys! 
        utils::transform_k2g(sys.bz().kp_trev(ik),sys.bz().symm_list[sys.bz().kp_symm(ik)],Gs,fft_mesh,
                             sys.bz().kpts(sys.bz().kp_to_ibz(ik),nda::range::all),k2g,Xft);
      }
    }
    if( (OT=='w') and not wfc_k2g_list[ik].has_value() ) {
      wfc_k2g_list[ik] = std::make_optional<nda::array<int,1>>(sys.npw[ik],0);
      nda::array<int,2> miller(sys.npw[ik],3);
      h5_read(*(h5files[index]), "/MillerIndices", miller);
      auto& k2g = *(wfc_k2g_list[ik]);
      if(ik >= sys.bz().nkpts_ibz) {
        nda::stack_array<double, 3> Gs;
        Gs() = 0; // Gs = ??? , these should be stored in sys! 
        utils::transform_miller_indices(sys.bz().kp_trev(ik),
           sys.bz().symm_list[sys.bz().kp_symm(ik)],Gs,miller);
      }
      utils::generate_k2g(miller,k2g,wfc_g.mesh());
      auto fft2g = wfc_g.fft_to_gv();
      for( auto& v : k2g ) {
        v = fft2g(v);  
        utils::check(v >= 0 and v < wfc_g.size(), "wfc_k2g: index out of bounds: {}",v); 
      }
    }
  }

  template<typename Array,
           size_t rank = size_t(::nda::get_rank<Array>)>
  void check_dimensions(char OT, Array Orb, nda::range& r)
  {
    static_assert(rank==size_t(::nda::get_rank<Array>) and
                 (rank==1 or rank==2 or rank==3), "Rank mismatch.");
    // rank=1/2, polarization is included in r-grid
    // rank=3  , polarization has its separate index
    long nx = ( (rank==1 or rank==2) ? sys.npol : long(1) );
    if(OT=='r' or OT=='g') { // fft mesh for potential based on ecutrho. size = nnr. LARGER
      if(r == nda::range{-1,-1}) r = {0,nx*nnr()};
      utils::check(r.first()>=0 and r.last()<=nx*nnr(), "Range error");
      utils::check(Orb.shape()[rank-1] == r.size(), "Wrong dimensions.");
      if(OT=='r') {
        // need to request the entire nnr grid, but not all polarizations
        utils::check( r.size()%nnr()==0 and
                      r.first()%nnr()==0 and
                      r.last() > r.first(),
                     "Error: Limited range-based orbital access in real-space option, must request the full nnr grid for some subset of the polarizations: range:({},{}), nnr:{}",
                     r.first(),r.last(),nnr());
      }
    } else if(OT=='w') { // truncated g grid for wfc. 
      if(r == nda::range{-1,-1}) r = {0,nx*wfc_g.size()}; 
      utils::check(r.first()>=0 and r.last()<=nx*wfc_g.size(), "Range error");
      utils::check(Orb.shape()[rank-1] >= r.size(), "Wrong dimensions.");
    } 
  }

  bool within_range(std::pair<int,int> r, int b)
  {
    return b >= r.first and (b - r.first) < r.second;  
  }

  // The next three routines accomodate reading to device, by staging the reading in HOST memory 
  template<nda::ArrayOfRank<1> A1D>
  void orbital_from_h5(char OT, int ispin, int ik, int m, A1D&& Orb, nda::range r)
  {
    static_assert(nda::is_complex_v<typename std::decay_t<A1D>::value_type>, "Type mismatch");
    if constexpr (nda::mem::on_host<std::decay_t<A1D>> or nda::mem::on_unified<std::decay_t<A1D>>)
    {
      orbital_from_h5_impl(OT,ispin,ik,m,std::forward<A1D>(Orb),r);
    } else {
      utils::check(Orb.shape()[0] >= r.size(), "Dimension mismatch.");
      nda::array<ComplexType,1> Ohost(r.size());
      orbital_from_h5_impl(OT,ispin,ik,m,Ohost,r);
      Orb(nda::range(0,r.size())) = Ohost;
    }
    if(OT=='r') { 
      // check_dimensions limits r to full range, so this is ok!
      utils::check(Orb.strides()[0] == 1, "Stride mismatch.");
      constexpr MEMORY_SPACE MEM = memory::get_memory_space<std::decay_t<A1D>>();
      using view = ::nda::basic_array_view<ComplexType, 4, ::nda::C_layout, 'A', ::nda::default_accessor, 
                                           ::nda::borrowed<to_nda_address_space(MEM)>>; 
      utils::check( r.size()%nnr()==0, "Oh oh: Should not happen."); 
      long nx = r.size()/nnr();
      auto Offt = view(std::array<long,4>{nx,fft_mesh(0),fft_mesh(1),fft_mesh(2)},Orb.data()); 
      math::fft::invfft_many(Offt);
    }
  }

  template<nda::ArrayOfRank<2> A2D>
  void orbital_set_from_h5(char OT, int ispin, int ik, nda::range b_rng, A2D&& Orb, nda::range r)
  {
    static_assert(nda::is_complex_v<typename std::decay_t<A2D>::value_type>, "Type mismatch");
    long nb = b_rng.size();
    if constexpr (nda::mem::on_host<std::decay_t<A2D>> or nda::mem::on_unified<std::decay_t<A2D>>)
    { 
      orbital_set_from_h5_impl(OT,ispin,ik,b_rng,std::forward<A2D>(Orb),r);
    } else {
      utils::check(Orb.shape()[0] >= nb, "Dimension mismatch.");
      utils::check(Orb.shape()[1] >= r.size(), "Dimension mismatch.");
      nda::array<ComplexType,2> Ohost(nb,r.size());
      orbital_set_from_h5_impl(OT,ispin,ik,b_rng,Ohost,r);
      Orb(nda::range(0,nb),nda::range(0,r.size())) = Ohost;
    }
    if(OT=='r') { 
      // check_dimensions limits r to full range, so this is ok!
      utils::check(Orb.indexmap().is_contiguous(), "qe_readonly::orbital_set_from_h5: Layout mismatch."); 
      constexpr MEMORY_SPACE MEM = memory::get_memory_space<std::decay_t<A2D>>();
      using view = ::nda::basic_array_view<ComplexType, 4, ::nda::C_layout, 'A', ::nda::default_accessor,
                                           ::nda::borrowed<to_nda_address_space(MEM)>>;
      utils::check( r.size()%nnr()==0, "Oh oh: Should not happen."); 
      long nx = r.size()/nnr();
      auto Offt = view(std::array<long,4>{nb*nx,fft_mesh(0),fft_mesh(1),fft_mesh(2)},Orb.data());
      math::fft::invfft_many(Offt);
    }
  }

  template<nda::ArrayOfRank<3> A3D>
  void orbital_set_from_h5(char OT, int ispin, int ik, nda::range b_rng, nda::range p_rng, 
                           A3D&& Orb, nda::range r)
  {
    static_assert(nda::is_complex_v<typename std::decay_t<A3D>::value_type>, "Type mismatch");
    long nb = b_rng.size();
    long np = p_rng.size();
    if constexpr (nda::mem::on_host<std::decay_t<A3D>> or nda::mem::on_unified<std::decay_t<A3D>>)
    {
      orbital_set_from_h5_impl(OT,ispin,ik,b_rng,p_rng,std::forward<A3D>(Orb),r);
    } else {
      utils::check(Orb.shape()[0] >= b_rng.size() and 
                   Orb.shape()[1] >= p_rng.size() and
                   Orb.shape()[2] >= r.size(), "Dimension mismatch.");
      nda::array<ComplexType,3> Ohost(nb,np,r.size());
      orbital_set_from_h5_impl(OT,ispin,ik,b_rng,p_rng,Ohost,r);
      Orb(nda::range(nb),nda::range(np),nda::range(r.size())) = Ohost;
    }
    if(OT=='r') {
      // check_dimensions limits r to full range, so this is ok!
      utils::check(Orb.indexmap().is_contiguous(), "qe_readonly::orbital_set_from_h5: Layout mismatch.");
      constexpr MEMORY_SPACE MEM = memory::get_memory_space<std::decay_t<A3D>>();
      using view = ::nda::basic_array_view<ComplexType, 4, ::nda::C_layout, 'A', ::nda::default_accessor,
                                           ::nda::borrowed<to_nda_address_space(MEM)>>;
      auto Offt = view(std::array<long,4>{nb*np,fft_mesh(0),fft_mesh(1),fft_mesh(2)},Orb.data());
      math::fft::invfft_many(Offt);
    }
  }

  // These routines read into HOST memory
  template<nda::ArrayOfRank<1> A1D>
  void orbital_from_h5_impl(char OT, int ispin, int ik, int m, A1D&& Orb, nda::range r)
  { 
    static_assert(nda::mem::on_host<std::decay_t<A1D>> or nda::mem::on_unified<std::decay_t<A1D>>, "Memory mismatch.");
    static_assert(nda::is_complex_v<typename std::decay_t<A1D>::value_type>, "Type mismatch");
    using view = nda::basic_array_view<RealType, 1, nda::C_layout, 'A', 
                                       nda::default_accessor, nda::borrowed<>>; 
    using nda::range;
    auto all = range::all;
    utils::check(ispin >= 0 and ispin < sys.nspin,"ispin:{} out of range:(0,{})",
                 ispin,sys.nspin);
    utils::check(ik >= 0 and ik < sys.bz().nkpts,"ik:{} out of range:(0,{})",
                 ik,sys.bz().nkpts);
    utils::check(m >= 0 and m < sys.nbnd,"m:{} out of range:(0,{})",
                 m,sys.nbnd);
    int index = to_h5_index(ispin,ik);
    open_if_needed(OT,index,ispin,ik);
    Orb() = ComplexType(0.0);
    utils::check(Orb.shape()[0] >= r.size(), "Dimension mismatch.");
    { 
      nda::array<ComplexType,1> Ok(sys.npol*sys.npw[ik]);
      view Ok_v{{2*sys.npol*sys.npw[ik]},reinterpret_cast<RealType*>(Ok.data())};
      h5_read(*(h5files[index]), "/evc", Ok_v, std::tuple{m,all});
      // MAM: fractional translations!!!
      //if(ik >= sys.bz().nkpts_ibz and ft>0) Ok(all) *= Xft(all);
      if(OT=='w')
        utils::check(wfc_k2g_list[ik].has_value(), "Uninitialized state.");
      else 
        utils::check(k2g_list[ik].has_value(), "Uninitialized state.");
      auto k2g = ( (OT=='w') ? *(wfc_k2g_list[ik]) :  *(k2g_list[ik]) );
      // Map wfc from k-grid to g-grid. points not in k-grid would be 0.0
      long n_ = ( (OT=='w') ? wfc_g.size() : nnr() );
      if(r.first() > 0 or r.last() < sys.npol*n_) {
        for(int p=0, p0=0, p1=0; p<sys.npol; p++, p0+=n_, p1+=sys.npw[ik] ) {
          for(int i=0; i<sys.npw[ik]; i++) {
            if( p0+k2g(i) >= r.first() and p0+k2g(i) < r.last() )
              Orb( p0+k2g(i)-r.first() ) = Ok(p1+i);
          }
        }
      } else {
        for(int p=0, p0=0, p1=0; p<sys.npol; p++, p0+=n_, p1+=sys.npw[ik] )
          for(int i=0; i<sys.npw[ik]; i++) 
            Orb( p0+k2g(i) ) = Ok(p1+i);
      }
    }
    // conjugate if trev point 
    if(sys.bz().kp_trev(ik))
      nda::tensor::scale(ComplexType(1.0),Orb,nda::tensor::op::CONJ);
  }

  template<nda::ArrayOfRank<2> A2D>
  void orbital_set_from_h5_impl(char OT, int ispin, int ik, nda::range b_rng, A2D&& Orb, nda::range r)
  {
    static_assert(nda::mem::on_host<std::decay_t<A2D>> or nda::mem::on_unified<std::decay_t<A2D>>, "Memory mismatch.");
    using view = nda::array_view<RealType, 2>;
    using nda::range;
    auto all = range::all;
    static_assert(nda::is_complex_v<typename std::decay_t<A2D>::value_type>, "Type mismatch");
    int nb = b_rng.size();
    if(nb <= 0) return;
    utils::check(ispin >= 0 and ispin < sys.nspin,"ispin:{} out of range:(0,{})",ispin,sys.nspin);
    utils::check(ik >= 0 and ik < sys.bz().nkpts,"ik:{} out of range:(0,{})",ik,sys.bz().nkpts);
    utils::check(b_rng.first() >= 0 and b_rng.last() <= sys.nbnd, "Band range:({},{}) out of bounds:(0,{})",b_rng.first(),b_rng.last(),sys.nbnd);
    utils::check(Orb.shape()[0] >= nb, "Dimension mismatch.");
    utils::check(Orb.shape()[1] >= r.size(), "Dimension mismatch.");
    int index = to_h5_index(ispin,ik);
    open_if_needed(OT,index,ispin,ik);
    Orb() = ComplexType(0.0);
    int npw = sys.npw[ik];
    { 
// use fallback allocator
      nda::array<ComplexType,2> Ok(nb,sys.npol*npw);
      view Ok_v{{nb,2*sys.npol*npw},reinterpret_cast<RealType*>(Ok.data())}; 
      h5_read(*(h5files[index]), "/evc", Ok_v,std::tuple{b_rng,all});
      // MAM: fractional translations!!!
      //if(ik >= sys.bz().nkpts_ibz and ft>0) Ok(all) *= Xft(all);
      if(OT=='w')
        utils::check(wfc_k2g_list[ik].has_value(), "Uninitialized state.");
      else 
        utils::check(k2g_list[ik].has_value(), "Uninitialized state.");
      auto k2g = ( (OT=='w') ? *(wfc_k2g_list[ik]) :  *(k2g_list[ik]) );
      long n_ = ( (OT=='w') ? wfc_g.size() : nnr() );
      // implement enumerate???
      if(r.first() > 0 or r.last() < sys.npol*n_) {
        for(int b=0; b<nb; ++b)
          for(int p=0, p0=0, p1=0; p<sys.npol; p++, p0+=n_, p1+=npw)
            for(int i=0; i<npw; i++)
              if( p0+k2g(i) >= r.first() and p0+k2g(i) < r.last() )
                Orb( b, p0+k2g(i)-r.first() ) = Ok(b, p1+i);
      } else {
        // full list, no need to be careful
        for(int b=0; b<nb; ++b)
          for(int p=0, p0=0, p1=0; p<sys.npol; p++, p0+=n_, p1+=npw)
            for(int i=0; i<npw; i++)
              Orb( b, p0+k2g(i) ) = Ok(b, p1+i);
      } 
    } 
    // conjugate if trev point 
    if(sys.bz().kp_trev(ik))
      nda::tensor::scale(ComplexType(1.0),Orb,nda::tensor::op::CONJ);
  }

  template<nda::ArrayOfRank<3> A3D>
  void orbital_set_from_h5_impl(char OT, int ispin, int ik, nda::range b_rng, nda::range p_rng,
                                A3D&& Orb, nda::range r)
  {
    static_assert(nda::mem::on_host<std::decay_t<A3D>> or nda::mem::on_unified<std::decay_t<A3D>>, "Memory mismatch.");
    using view = nda::array_view<RealType, 2>;
    using nda::range;
    auto all = range::all;
    static_assert(nda::is_complex_v<typename std::decay_t<A3D>::value_type>, "Type mismatch");
    int nb = b_rng.size();
    int np = p_rng.size();
    if(nb <= 0) return;
    utils::check(ispin >= 0 and ispin < sys.nspin,"ispin:{} out of range:(0,{})",ispin,sys.nspin);
    utils::check(ik >= 0 and ik < sys.bz().nkpts,"ik:{} out of range:(0,{})",ik,sys.bz().nkpts);
    utils::check(b_rng.first() >= 0 and b_rng.last() <= sys.nbnd, "Band range:({},{}) out of bounds:(0,{})",b_rng.first(),b_rng.last(),sys.nbnd);
    utils::check(p_rng.first() >= 0 and p_rng.last() <= sys.npol, "Polarization range:({},{}) out of bounds:(0,{})",p_rng.first(),p_rng.last(),sys.npol);
    utils::check(Orb.shape()[0] >= b_rng.size() and
                 Orb.shape()[1] >= p_rng.size() and
                 Orb.shape()[2] >= r.size(), "Dimension mismatch.");
    int index = to_h5_index(ispin,ik);
    open_if_needed(OT,index,ispin,ik);
    Orb() = ComplexType(0.0);
    int npw = sys.npw[ik];
    { 
// use fallback allocator
      // reading all polarizations for efficiency, at the expense of more memory
      nda::array<ComplexType,2> Ok(nb,sys.npol*npw);
      view Ok_v{{nb,2*sys.npol*npw},reinterpret_cast<RealType*>(Ok.data())}; 
      h5_read(*(h5files[index]), "/evc", Ok_v,std::tuple{b_rng,all});
      // MAM: fractional translations!!!
      //if(ik >= sys.bz().nkpts_ibz and ft>0) Ok(all) *= Xft(all);
      if(OT=='w')
        utils::check(wfc_k2g_list[ik].has_value(), "Uninitialized state.");
      else 
        utils::check(k2g_list[ik].has_value(), "Uninitialized state.");
      auto k2g = ( (OT=='w') ? *(wfc_k2g_list[ik]) :  *(k2g_list[ik]) );
      long n_ = ( (OT=='w') ? wfc_g.size() : nnr() );
      // implement enumerate???
      if(r.first() > 0 or r.last() < n_) {
        for(int b=0; b<nb; ++b)
          for(int p=0, p1=int(p_rng.first()*npw); p<np; p++, p1+=npw)
            for(int i=0; i<npw; i++)
              if( k2g(i) >= r.first() and k2g(i) < r.last() )
                Orb( b, p, k2g(i)-r.first() ) = Ok(b, p1+i);
      } else {
        // full list, no need to be careful
        for(int b=0; b<nb; ++b)
          for(int p=0, p1=int(p_rng.first()*npw); p<np; p++, p1+=npw)
            for(int i=0; i<npw; i++)
              Orb( b, p, k2g(i) ) = Ok(b, p1+i);
      } 
    } 
    // conjugate if trev point 
    if(sys.bz().kp_trev(ik))
      nda::tensor::scale(ComplexType(1.0),Orb,nda::tensor::op::CONJ);
}

};

} // qe
} // mf 

#endif

