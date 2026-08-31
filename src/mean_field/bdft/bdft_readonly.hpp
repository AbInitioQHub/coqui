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


#ifndef COQUI_BDFT_READONLY_HPP
#define COQUI_BDFT_READONLY_HPP

#include <map>
#include <string>
#include <memory>

#include <nda/nda.hpp>
#include <h5/h5.hpp>
#include <nda/h5.hpp>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "utilities/kpoint_utils.hpp"
#include "utilities/fortran_utilities.h"
#include "utilities/qe_utilities.hpp"
#include "utilities/concepts.hpp"
#include "utilities/mpi_context.h"

#include "numerics/distributed_array/nda.hpp"
#include "mean_field/mf_source.hpp"
#include "mean_field/bdft/bdft_system.hpp"
#include "numerics/sparse/sparse.hpp"
#include "grids/g_grids.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "hamiltonian/pseudo/pseudopot_to_h5.hpp"

namespace mf {
namespace bdft {

namespace detail {

auto make_wfc(bdft_system const& sys) -> grids::truncated_g_grid;

auto make_ksymms(bz_symm const& bz) -> nda::array<int, 1>;

auto make_swfc_maps(bdft_system const &sys,
                    nda::array<int, 1> const &ksymms,
                    nda::ArrayOfRank<1> auto const &fft_mesh,
                    grids::truncated_g_grid const &wfc_g)
-> math::shm::shared_array<nda::array_view<long, 4>> {
  using arr_t = math::shm::shared_array<nda::array_view<long, 4>>;
  decltype(nda::range::all) all;

  long ngm = wfc_g.size();
  auto g2fft = wfc_g.gv_to_fft();
  auto fft2g = wfc_g.fft_to_gv();
  long nsym = ksymms.size();
  long ntrev = (std::any_of(sys.bz().kp_trev.begin(), sys.bz().kp_trev.end(), [](auto &&a) { return a; }) ? 2 : 1);
  arr_t sw2g(*(sys.mpi), std::array<long, 4>{2, nsym, ntrev, ngm});
  auto w2g = sw2g.local();
  sw2g.set_zero();

  // wfc grid
  long NX = wfc_g.mesh(0), NY = wfc_g.mesh(1), NZ = wfc_g.mesh(2);
  // density grid
  long NX2 = NX / 2, NY2 = NY / 2, NZ2 = NZ / 2;

  math::shm::shared_array<nda::array_view<int, 2>> smill(*(sys.mpi), std::array<long, 2>{ngm, 3});
  auto mill = smill.local();
  if (sys.mpi->node_comm.root()) {
    for (auto [p, N]: itertools::enumerate(g2fft)) {
      long k = N % NZ;
      if (k > NZ2) k -= NZ;
      long N_ = N / NZ;
      long j = N_ % NY;
      if (j > NY2) j -= NY;
      long i = N_ / NY;
      if (i > NX2) i -= NX;
      mill(p, 0) = i;
      mill(p, 1) = j;
      mill(p, 2) = k;
    }
  }
  sys.mpi->node_comm.barrier();

  nda::array<int, 2> mill_l(ngm, 3);
  int rank = sys.mpi->comm.rank();
  int np = sys.mpi->comm.size();
  {
    nda::stack_array<double, 3> Gs;
    Gs() = 0; // Gs = ??? , these should be stored in sys!
    long ist = 0;
    for (auto [is, s]: itertools::enumerate(ksymms)) {
      for (auto it: nda::range(ntrev)) {
        if ((ist++) % np != rank) continue;
        mill_l = mill();
        // rotate miller indices by symm_list[s]
        if (s == 0 and it == 0) {
          // nothing to do for identity operation
          w2g(0, 0, 0, all) = nda::arange(ngm);
        } else {
          utils::transform_miller_indices(it, sys.bz().symm_list[s], Gs, mill_l);
          // 1: map to wfc_g fft grid
          utils::generate_k2g(mill_l, w2g(0, is, it, all), wfc_g.mesh());
          //    map from fft back to truncated grid
          for (auto &v: w2g(0, is, it, all)) {
            v = fft2g(v);
            utils::check(v >= 0 and v < wfc_g.size(), "w2g: index out of bounds: {}", v);
          }
        }
        // 2: map to fft_mesh grid
        utils::generate_k2g(mill_l, w2g(1, is, it, all), fft_mesh);
      }
    }
  }
  sw2g.all_reduce();
  sys.mpi->comm.barrier(); // just to be safe!
  return sw2g;
}

}

// MAM: In some cases, you could generate a new h5 file with missing orbital information
//      There is also some useless redundancy in the /Orbital dataset between system and readonly 
/**
 * Basic interface to BDFT MF object.
 * Provides access to system information, orbitals, mean-field potentials, etc.   
 */
class bdft_readonly {
public:

  static constexpr mf_source_e mf_src = bdft_source;
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
  const bdft_system& get_sys() const {return sys; }
  decltype(auto) wfc_truncated_grid() const {
    return std::addressof(wfc_g);
  }
  bool has_wfc_grid() const { return true; }
  auto const& bz() const { return sys.bz(); }

public:
  template<utils::Communicator comm_t>
  bdft_readonly(std::shared_ptr<utils::mpi_context_t<comm_t>> mpi,
                std::string outdir, std::string prefix, 
                double ecut_ = 0.0, long n_ = -1):
    sys(std::move(mpi), outdir, prefix, n_),
    h5file(std::nullopt),
    ecut(ecut_<=0.1?sys.ecutrho:ecut_), 
    fft_mesh( ecut_>0.0 ? nda::stack_array<int, 3>{grids::find_fft_mesh(sys.mpi->comm,ecut,sys.recv,sys.bz().symm_list)} : sys.fft_mesh),
    wfc_g(detail::make_wfc(sys)),
    ksymms(detail::make_ksymms(sys.bz())),
    swfc_maps(detail::make_swfc_maps(sys,ksymms,fft_mesh,wfc_g))
  {
    // build symmetry rotations: only correct with orthogonal bases, fix!
    auto slist = utils::find_inverse_symmetry(sys.bz().qsymms,sys.bz().symm_list);
    if(slist.size() > 0) 
      std::tie(sk_to_n,dmat) = utils::generate_dmatrix<true>(*this, sys.bz().symm_list, slist);

    print_metadata("CoQuí mean-field reader");
  }

  bdft_readonly(bdft_system const& bdft_sys):
    sys(bdft_sys),
    h5file(std::nullopt),
    ecut(sys.ecutrho), fft_mesh(sys.fft_mesh),
    wfc_g(detail::make_wfc(sys)),
    ksymms(detail::make_ksymms(sys.bz())),
    swfc_maps(detail::make_swfc_maps(sys,ksymms,fft_mesh,wfc_g))
  {
    // build symmetry rotations
    auto slist = utils::find_inverse_symmetry(sys.bz().qsymms,sys.bz().symm_list);
    if(slist.size() > 0) 
      std::tie(sk_to_n,dmat) = utils::generate_dmatrix<true>(*this,sys.bz().symm_list,slist);

    print_metadata("CoQuí mean-field reader");
  }

  bdft_readonly(bdft_readonly const& other):
    sys(other.sys),
    h5file(std::nullopt),
    ecut(other.ecut), fft_mesh(other.fft_mesh),
    wfc_g(other.wfc_g),
    ksymms(other.ksymms),
    swfc_maps(other.swfc_maps),
    sk_to_n(other.sk_to_n),
    dmat( other.dmat ) {}

  bdft_readonly(bdft_system&& bdft_sys):
    sys(std::move(bdft_sys) ),
    h5file(std::nullopt),
    ecut(sys.ecutrho), fft_mesh(sys.fft_mesh),
    wfc_g(detail::make_wfc(sys)),
    ksymms(detail::make_ksymms(sys.bz())),
    swfc_maps(detail::make_swfc_maps(sys,ksymms,fft_mesh,wfc_g))
  {
    // build symmetry rotations
    auto slist = utils::find_inverse_symmetry(sys.bz().qsymms,sys.bz().symm_list);
    if(slist.size() > 0) 
      std::tie(sk_to_n,dmat) = utils::generate_dmatrix<true>(*this,sys.bz().symm_list,slist);

    print_metadata("CoQuí mean-field reader");
  }

  template<class MF>
  bdft_readonly(MF& mf, std::string fn,
                math::nda::DistributedArray auto const& psi,
                int b0 = -1,
                bool update_eig_occ = false) :
    sys(mf,fn,false),
    h5file(std::nullopt),
    ecut(sys.ecutrho), fft_mesh(sys.fft_mesh),
    wfc_g(*mf.wfc_truncated_grid()),
    ksymms(detail::make_ksymms(sys.bz())),
    swfc_maps(detail::make_swfc_maps(sys,ksymms,fft_mesh,wfc_g))
  { 
    constexpr int rank = math::nda::get_rank<std::decay_t<decltype(psi)>>;
    static_assert( rank==3 or rank==4, "Rank mismatch");
    constexpr int i0 = (rank==3?0:1);
    decltype(nda::range::all) all;

    utils::check( sys.nspin == sys.nspin_in_basis, "Error: Finish implementation of nspin_in_basis");
    // can only change orbitals, 
    // everything else must remain consistent, for now
    utils::check( wfc_g.size() == psi.global_shape()[i0+2], "Shape mismatch.");
    utils::check( sys.bz().nkpts_ibz == psi.global_shape()[i0+0], "Shape mismatch.");
    if constexpr (rank==4) 
      utils::check( sys.nspin == psi.global_shape()[0], "Shape mismatch.");
        
    // set number of states to keep from mf
    if(b0 < 0 or b0 > sys.nbnd) b0 = sys.nbnd;
    int norb = psi.global_shape()[i0+1]; 
    int ngm = wfc_g.size();
    sys.nbnd = norb+b0; 
    sys.occ = nda::array<double,3>::zeros({sys.nspin,sys.bz().nkpts,norb+b0}); 
    sys.eigval = nda::array<double,3>::zeros({sys.nspin,sys.bz().nkpts,norb+b0}); 
    if(update_eig_occ) {
      utils::check(false,"Finish!!!");	
    }
	
    sys.mpi->comm.barrier();
    nda::range s_range(sys.nspin);
    if constexpr (rank==4) 
      s_range = psi.local_range(0);
    auto k_range = psi.local_range(i0+0);
    auto a_range = psi.local_range(i0+1);
    auto g_range = psi.local_range(i0+2);
    // MAM: This uses too much memory, use math::nda::gather for each kpoint  
    if(sys.mpi->comm.root()) {
      h5::file file = h5::file(sys.filename, 'w');
      h5::group grp(file);
      sys.save(grp); 
      h5::group ogrp = grp.open_group("Orbitals");
      // wfc_g miller indices 
      {
        nda::array<int,2> miller(ngm,3);
        utils::generate_miller_index(wfc_g.gv_to_fft(),miller,wfc_g.mesh());
        nda::h5_write(ogrp,"miller_wfc",miller,false);
        h5::h5_write(ogrp,"wfc_ecut",wfc_g.ecut());
        nda::array<int,1> mesh(wfc_g.mesh());
        nda::h5_write(ogrp,"wfc_fft_grid",mesh,false);
        h5::h5_write(ogrp,"wfc_ngm",int(wfc_g.size()));
      }
      nda::array<ComplexType,2> psik(norb+b0,ngm);
      for( int ik=0; ik<sys.bz().nkpts_ibz; ++ik ) {
        for( int is=0; is<sys.nspin; ++is ) { 
          psik() = 0.0;	
          if(b0 > 0) 
            mf.get_orbital_set('w',is,ik,{0,b0},psik(nda::range(0,b0),all));
          if( is >= s_range.first() and is < s_range.last() and
          ik >= k_range.first() and ik < k_range.last() ) {
            if constexpr (rank == 4) {	
              psik(a_range+b0, g_range ) = psi.local()(is-s_range.first(),ik-k_range.first(),all,all);	
            } else {
              psik(a_range+b0, g_range ) = psi.local()(ik-k_range.first(),all,all);	
            }
          }
          sys.mpi->comm.reduce_in_place_n(psik.data()+b0*ngm,ngm*norb,std::plus<>{},0);
          nda::h5_write(ogrp,"psi_s"+std::to_string(is)+"_k"+std::to_string(ik),psik,false);
          sys.mpi->comm.barrier();
        } 
      } 
      h5::group hgrp = grp.create_group("Hamiltonian");
    } else {
      nda::array<ComplexType,2> psik(norb,ngm);
      for( int ik=0; ik<sys.bz().nkpts_ibz; ++ik ) {
        for( int is=0; is<sys.nspin; ++is ) {
          psik() = 0.0;
          if( is >= s_range.first() and is < s_range.last() and
              ik >= k_range.first() and ik < k_range.last() ) {
            if constexpr (rank == 4) {
              psik( a_range, g_range ) = psi.local()(is-s_range.first(),ik-k_range.first(),all,all);   
            } else {
              psik( a_range, g_range ) = psi.local()(ik-k_range.first(),all,all); 
            }
          }
          sys.mpi->comm.reduce_in_place_n(psik.data(),psik.size(),std::plus<>{},0);
          sys.mpi->comm.barrier();
        }
      }
    }
    sys.mpi->comm.barrier();

    // build symmetry rotations
    auto slist = utils::find_inverse_symmetry(sys.bz().qsymms,sys.bz().symm_list);
    if(slist.size() > 0) 
      std::tie(sk_to_n,dmat) = utils::generate_dmatrix<true>(*this,sys.bz().symm_list,slist);
    sys.mpi->comm.barrier();

    print_metadata("CoQuí mean-field reader (from mf object)");
    sys.mpi->comm.barrier();
  }

  template<class MF>
  bdft_readonly(MF& mf, std::string fn, long n0,
    	    nda::array<std::pair<long,double>,2> const& orb_list) :
    sys(mf,fn,false),
    h5file(std::nullopt),
    ecut(sys.ecutrho), fft_mesh(sys.fft_mesh),
    wfc_g(*mf.wfc_truncated_grid()),
    ksymms(detail::make_ksymms(sys.bz())),
    swfc_maps(detail::make_swfc_maps(sys,ksymms,fft_mesh,wfc_g))
  { 
    decltype(nda::range::all) all;
        
    utils::check( sys.nspin == sys.nspin_in_basis, "Error: Finish implementation of nspin_in_basis");
    utils::check( mf.nbnd_aux() == 0, "Error in bdft_readonly constructor(mf, orb_list): mf.nbnd_aux>0: {}",mf.nbnd_aux());
    utils::check(orb_list.shape()[0] == sys.bz().nkpts_ibz, "Shape mismatch.");

    
    int ngm = wfc_g.size();
    long naux = orb_list.shape()[1]; 
    long nbnd_0 = sys.nbnd;
    sys.nbnd = n0; 
    sys.nbnd_aux = naux;
    {
      auto tocc = sys.occ;
      auto teig = sys.eigval;
      sys.occ = nda::array<double,3>::zeros({sys.nspin,sys.bz().nkpts,n0}); 
      sys.eigval = nda::array<double,3>::zeros({sys.nspin,sys.bz().nkpts,n0}); 
      sys.occ = tocc(all,all,nda::range(n0));
      sys.eigval = teig(all,all,nda::range(n0));
      if(naux>0) {
        sys.eigval_aux = nda::array<double,3>::zeros({sys.nspin,sys.bz().nkpts,naux});
        sys.aux_weight = nda::array<double,3>::zeros({sys.nspin,sys.bz().nkpts,naux});
        for( int is=0; is<sys.nspin; ++is ) {
          for( int ik=0; ik<sys.bz().nkpts; ++ik ) {
            int iks = sys.bz().kp_to_ibz(ik);
            for( long ib=0; ib<naux; ++ib ) {
              utils::check(orb_list(iks,ib).first >= 0 and orb_list(iks,ib).first < nbnd_0,
                           "Index mismatch.");
              sys.eigval_aux(is,ik,ib) = teig(is,iks,orb_list(iks,ib).first);
              sys.aux_weight(is,ik,ib) = orb_list(iks,ib).second; 
            }
          }
        }
      }
    }
	
    if(sys.mpi->comm.root()) {
      h5::file file = h5::file(sys.filename, 'w');
      h5::group grp(file);
      sys.save(grp); 
      h5::group ogrp = grp.open_group("Orbitals");
      // wfc_g miller indices 
      {
        nda::array<int,2> miller(ngm,3);
        utils::generate_miller_index(wfc_g.gv_to_fft(),miller,wfc_g.mesh());
        nda::h5_write(ogrp,"miller_wfc",miller,false);
        h5::h5_write(ogrp,"wfc_ecut",wfc_g.ecut());
        nda::array<int,1> mesh(wfc_g.mesh());
        nda::h5_write(ogrp,"wfc_fft_grid",mesh,false);
        h5::h5_write(ogrp,"wfc_ngm",int(wfc_g.size()));
      }
      auto psik = nda::array<ComplexType,2>::zeros({std::max(n0,naux),ngm});
      for( int ik=0; ik<sys.bz().nkpts_ibz; ++ik ) {
        // primary orbitals
        for( int is=0; is<sys.nspin; ++is ) {
          mf.get_orbital_set('w',is,ik,nda::range(n0),psik(nda::range(n0),all));
          nda::h5_write(ogrp,"psi_s"+std::to_string(is)+"_k"+std::to_string(ik),
                        psik(nda::range(n0),all),false);
        }
        // auxiliary orbitals
        if(naux>0) {
          for( int is=0; is<sys.nspin; ++is ) { 
            for( long ib=0; ib<naux; ++ib ) 
              mf.get_orbital('w',is,ik,orb_list(ik,ib).first,psik(ib,all));
            nda::h5_write(ogrp,"aux_psi_s"+std::to_string(is)+"_k"+std::to_string(ik),
                          psik(nda::range(naux),all),false);
          } 
        } 
      } 

      // if available, write pseudopot
      if(mf.input_file_type() == mf::xml_input_type and mf.mf_type() == mf::qe_source) {
        hamilt::pseudopot_to_h5(mf.fft_grid_dim(),grp,mf.outdir(),mf::xml_input_type);
      } else if(mf.input_file_type() == mf::h5_input_type and mf.mf_type() != mf::pyscf_source) {
        hamilt::pseudopot_to_h5(mf.fft_grid_dim(),grp,mf.filename(),mf::h5_input_type);
      }
    }
    sys.mpi->comm.barrier();


    // build symmetry rotations
    auto slist = utils::find_inverse_symmetry(sys.bz().qsymms,sys.bz().symm_list);
    if(slist.size() > 0) 
      std::tie(sk_to_n,dmat) = utils::generate_dmatrix<true>(*this,sys.bz().symm_list,slist);

    print_metadata("CoQuí mean-field reader (from mf object with orbital selection)");
    sys.mpi->comm.barrier();
  }

  bdft_readonly(bdft_readonly&& other):
      sys(std::move(other.sys) ),
      h5file(std::nullopt),
      ecut(other.ecut), fft_mesh(other.fft_mesh), 
      wfc_g(std::move(other.wfc_g)), ksymms(std::move(other.ksymms)),
      swfc_maps(std::move(other.swfc_maps)), 
      sk_to_n(std::move(other.sk_to_n)), dmat( std::move(other.dmat) ) {}

  ~bdft_readonly() { close(); }

  bdft_readonly& operator=(const bdft_readonly& other) {
    this->sys = other.sys;
    this->sk_to_n = other.sk_to_n;
    this->dmat = other.dmat;
    close();
    this->ecut = other.ecut;
    this->fft_mesh = other.fft_mesh;
    this->wfc_g = other.wfc_g;
    this->ksymms = other.ksymms;
    this->swfc_maps = other.swfc_maps;
    return *this;
  }

  bdft_readonly& operator=(bdft_readonly&& other) {
    this->sys = std::move(other.sys);
    this->sk_to_n = std::move(other.sk_to_n);
    this->dmat = std::move(other.dmat);
    close();
    this->ecut = other.ecut;
    this->fft_mesh = other.fft_mesh;
    this->ksymms = std::move(other.ksymms);
    this->swfc_maps = std::move(other.swfc_maps);
    this->wfc_g = std::move(other.wfc_g);
    return *this;
  }

  void print_metadata(std::string header = "CoQuí mean-field reader") {
    app_log(1, "  {}", header);
    app_log(1, "  ------------------------------------");
    app_log(1, "  Number of spins            =  {}", sys.nspin);
    app_log(1, "  Number of spins in basis   =  {}", sys.nspin_in_basis);
    app_log(1, "  Number of polarizations    = {}", sys.npol);
    app_log(1, "  Number of bands            = {}", sys.nbnd);
    app_log(1, "  Monkhorst-Pack mesh        = ({},{},{})", sys.bz().kp_grid(0), sys.bz().kp_grid(1), sys.bz().kp_grid(2));
    app_log(1, "  K-points                   = {} total, {} in the IBZ", sys.bz().nkpts, sys.bz().nkpts_ibz);
    app_log(1, "  Number of electrons        = {}", sys.nelec);
    app_log(1, "  Density energy cutoff      = {0:.3f} a.u. | FFT mesh = ({1},{2},{3})",
            ecut,fft_mesh(0),fft_mesh(1),fft_mesh(2));
    app_log(1, "  Wavefunction energy cutoff = {0:.3f} a.u. | FFT mesh = ({1},{2},{3}), Number of PWs = {4}\n",
            wfc_g.ecut(), wfc_g.mesh(0),wfc_g.mesh(1),wfc_g.mesh(2), wfc_g.size());
  }

  // read orbital
  /*
   * OT: Orbital Type:
   *   -'r': real space fft grid 
   *   -'g': fourier space fft grid 
   *   -'w': wavefunction grid 
   */
  template<nda::ArrayOfRank<1> A1D>
  void get_orbital(char OT, int _ispin, int ik, int m, A1D&& Orb, nda::range r = {-1,-1})
  { 
    int ispin = std::min(_ispin, sys.nspin_in_basis-1);
    utils::check(OT=='r' or OT=='g' or OT=='w',"Unknown orbital type in qe_readonly::get_orbital.");
    open_if_needed();
    check_dimensions(OT,ik,0,Orb,r);
    utils::check(m <= sys.nbnd+sys.nbnd_aux, "Orbital index out of bounds - m:{}",m);
    if(m < sys.nbnd) {
      orbital_from_h5("Orbitals/psi",OT,ispin,ik,m,Orb,r);
    } else {
      orbital_from_h5("Orbitals/aux_psi",OT,ispin,ik,m-sys.nbnd,Orb,r);
    }
  }

  // read multiple orbitals
  template<nda::ArrayOfRank<2> A2D>
  void get_orbital_set(char OT, int _ispin, int k, nda::range b_rng, A2D&& Orb, nda::range r = {-1,-1})
  { 
    static_assert(std::decay_t<A2D>::layout_t::is_stride_order_C(), "Layout mismatch.");
    int ispin = std::min(_ispin, sys.nspin_in_basis-1);
    utils::check(OT=='r' or OT=='g' or OT=='w',"Unknown orbital type in qe_readonly::get_orbital.");
    utils::check(Orb.shape()[0] >= b_rng.size(), "Dimension mismatch.");
    utils::check(b_rng.first() >= 0 and b_rng.last() <= sys.nbnd + sys.nbnd_aux, "Index out of bounds.");
    open_if_needed();
    nda::range r_=r;
    check_dimensions(OT,k,1,Orb,r_);
    int nb = b_rng.size();  // # of requested orbs
    int n2 = std::max(0, int(b_rng.last()-sys.nbnd));  // # of aux orbs
    int n2b = std::max(0,int(b_rng.first()-sys.nbnd));
    if( nb > n2 )
      orbital_set_from_h5("Orbitals/psi",OT,ispin,k,b_rng.first(),nb-n2,Orb(nda::range(nb-n2),nda::range::all),r_);
    if( n2 > 0 ) 
      orbital_set_from_h5("Orbitals/aux_psi",OT,ispin,k,n2b,n2,Orb(nda::range(nb-n2,nb),nda::range::all),r_);
  }

  template<nda::ArrayOfRank<3> A3D>
  void get_orbital_set(char OT, int _ispin, nda::range k_rng, nda::range b_rng, A3D&& Orb, nda::range r = {-1,-1})
  {
    static_assert(std::decay_t<A3D>::layout_t::is_stride_order_C(), "Layout mismatch.");
    int ispin = std::min(_ispin, sys.nspin_in_basis-1);
    utils::check(OT=='r' or OT=='g' or OT=='w',"Unknown orbital type in qe_readonly::get_orbital.");
    utils::check(Orb.shape()[0] >= k_rng.size() and 
                 Orb.shape()[1] >= b_rng.size(), "Dimension mismatch.");
    utils::check(b_rng.first() >= 0 and b_rng.last() <= sys.nbnd + sys.nbnd_aux, "Index out of bounds.");
    int nb = b_rng.size();  // # of requested orbs
    int n2 = std::max(0, int(b_rng.last()-sys.nbnd));  // # of aux orbs
    int n2b = std::max(0,int(b_rng.first()-sys.nbnd));
    for( auto [ik,k] : itertools::enumerate(k_rng) ) {
      open_if_needed();
      nda::range r_=r;
      check_dimensions(OT,k,2,Orb,r_);
      if( nb > n2 )
        orbital_set_from_h5("Orbitals/psi",OT,ispin,k,b_rng.first(),nb-n2,Orb(ik,nda::range(nb-n2),nda::range::all),r_);
      if( n2 > 0 )
        orbital_set_from_h5("Orbitals/aux_psi",OT,ispin,k,n2b,n2,Orb(ik,nda::range(nb-n2,nb),nda::range::all),r_);
    }
  }

  template<nda::ArrayOfRank<4> A4D>
  void get_orbital_set(char OT, int _ispin, nda::range k_rng, nda::range b_rng, nda::range p_rng, A4D&& Orb, nda::range r = {-1,-1})
  {
    static_assert(std::decay_t<A4D>::layout_t::is_stride_order_C(), "Layout mismatch.");
    utils::check(p_rng.size()==1 and p_rng.first()==0, "npol > 1 not yet implemented in pyscf.");
    utils::check(Orb.is_contiguous(), "Layout mismatch.");
    constexpr MEMORY_SPACE MEM = memory::get_memory_space<std::decay_t<A4D>>();
    using view = ::nda::basic_array_view<ComplexType, 3, ::nda::C_layout, 'A', ::nda::default_accessor,
                                       ::nda::borrowed<to_nda_address_space(MEM)>>;
    auto O_ = view(std::array<long,3>{Orb.extent(0),Orb.extent(1),Orb.extent(3)},Orb.data());
    get_orbital_set(OT,_ispin,k_rng,b_rng,O_,r);
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

  // close h5 handles 
  void close()
  {
    if( h5file.has_value() )
        h5file = std::nullopt;
  }

  private:

  // system info
  bdft_system sys;

  // h5 handle 
  std::optional<h5::file> h5file;

  // plane wave cutoff of the FFT grid for AOs
  double ecut = 0.0;

  // fft mesh compatible with ecut
  nda::stack_array<int, 3> fft_mesh;

  // truncated g grid for wfc.
  // Constructed from the miller indices read from h5.
  // All kpoints share the same grid, orbitals in file are consistent with the grid. 
  grids::truncated_g_grid wfc_g;

  // list of symmetries found in sys.bz().kp_symm(:)
  nda::array<int,1> ksymms;

  /* 
   * Mappings associated with the wfc_g grid.
   * swfc_maps( ig, is, it, n ) :
   *   ig: output grid type. 0: w grid, 1: g grid.
   *   is: symmetry index.  {0, nsym} where 0 is always the identity.
   *   it: time reversal if present in the list of symmetries. 0: no trev, 1: with trev.
   *   n: index in wfc_g grid (or equivalently in miller_wfc).
   * The general use is given by:
   *
   *   for( auto [in, n] : itertools::enumerate(swfc_maps( ig, is, it, all )) ) 
   *     psi( ispin, ik, a, n ) = psi( ispin, kp_to_ibz(ik), a, in )
   *
   *   where: 
   *     - ig = 0 for OT=='w' and 1 for OT=='g'/'r'
   *     - is = location of kp_symm(ik) in ksymms 
   *     - it = 0 if trev(ik)==false or 1 otherwise
   */
  math::shm::shared_array<nda::array_view<long,4>> swfc_maps;

  // shared ptr to pseudopot object. Not constructed here.
  // can be set from the outside to avoid recomputing. 
  std::shared_ptr<hamilt::pseudopot> psp;

  // matrices that define symmetry relations between wavefunctions at different k-points
  nda::array<int,2> sk_to_n;  
  std::vector< math::sparse::csr_matrix<ComplexType,HOST_MEMORY,int,int> > dmat;

  void open_if_needed() {
    if( not h5file.has_value() ) 
      h5file = std::make_optional<h5::file>(sys.filename,'r');
  }

  template<class Array>
  void check_dimensions(char OT, [[maybe_unused]] int ik, int dim, Array Orb, nda::range& r)
  {
    if(OT=='r' or OT=='g') {
      if(r == nda::range{-1,-1}) r = {0,nnr()};
      utils::check(r.first()>=0 and r.last()<=nnr(), "Range error");
      utils::check(Orb.shape()[dim] == r.size(), "Wrong dimensions.");
      if(OT=='r')
        utils::check(r.size()==nnr(),"Error: range-based orbital access not allowed in real-space option.");
    } else if(OT=='w') {
      if(r == nda::range{-1,-1}) r = {0,wfc_g.size()};
      utils::check(r.first()>=0 and r.last()<=wfc_g.size(), "Range error");
      utils::check(Orb.shape()[dim] >= r.size(), "Wrong dimensions.");
    }
  }

  bool within_range(std::pair<int,int> r, int b)
  {
    return b >= r.first and (b - r.first) < r.second;
  }

  template<nda::ArrayOfRank<1> A1D>
  void orbital_from_h5(std::string orb_prefix, char OT, int is, int ik, int m, A1D&& Orb, nda::range r)
  {
    static_assert(nda::is_complex_v<typename std::decay_t<A1D>::value_type>, "Type mismatch");
    if constexpr (nda::mem::on_host<std::decay_t<A1D>> or nda::mem::on_unified<std::decay_t<A1D>>)
    {
      orbital_from_h5_impl(orb_prefix,OT,is,ik,m,std::forward<A1D>(Orb),r);
    } else {
      utils::check(Orb.shape()[0] >= r.size(), "Dimension mismatch.");
      nda::array<ComplexType,1> Ohost(r.size());
      orbital_from_h5_impl(orb_prefix,OT,is,ik,m,Ohost,r);
      Orb(nda::range(0,r.size())) = Ohost;
    }
    if(OT=='r') {
      // check_dimensions limits r to full range, so this is ok!
      auto Offt = nda::reshape(Orb,std::array<long,3>{fft_mesh(0),fft_mesh(1),fft_mesh(2)});
      math::fft::invfft(Offt);
    }
  }

  template<nda::ArrayOfRank<2> A2D>
  void orbital_set_from_h5(std::string orb_prefix, char OT, int is, int ik, int b0, int nb, A2D&& Orb, nda::range r)
  {
    static_assert(nda::is_complex_v<typename std::decay_t<A2D>::value_type>, "Type mismatch");
    if constexpr (nda::mem::on_host<std::decay_t<A2D>> or nda::mem::on_unified<std::decay_t<A2D>>)
    {
      orbital_set_from_h5_impl(orb_prefix,OT,is,ik,b0,nb,std::forward<A2D>(Orb),r);
    } else {
      utils::check(Orb.shape()[0] >= nb, "Dimension mismatch.");
      utils::check(Orb.shape()[1] >= r.size(), "Dimension mismatch.");
      nda::array<ComplexType,2> Ohost(nb,r.size());
      orbital_set_from_h5_impl(orb_prefix,OT,is,ik,b0,nb,Ohost,r);
      Orb(nda::range(0,nb),nda::range(0,r.size())) = Ohost;
    }
    if(OT=='r') {
      // check_dimensions limits r to full range, so this is ok!
      utils::check(Orb.strides()[0] == Orb.shape()[1], "qe_readonly::orbital_set_from_h5: Layout mismatch.");
      auto Offt = nda::reshape(Orb,std::array<long,4>{nb,fft_mesh(0),fft_mesh(1),fft_mesh(2)});
      math::fft::invfft_many(Offt);
    }
  }

  template<nda::ArrayOfRank<1> A1D>
  void orbital_from_h5_impl(std::string orb_prefix, char OT, int is, int ik, int m, A1D&& Orb, nda::range r)
  {
    static_assert(nda::mem::on_host<std::decay_t<A1D>> or nda::mem::on_unified<std::decay_t<A1D>>, "Memory mismatch.");
    static_assert(nda::is_complex_v<typename std::decay_t<A1D>::value_type>, "Type mismatch");
    utils::check(Orb.shape()[0] >= r.size(), "Dimension mismatch.");
    Orb() = ComplexType(0.0);
    long ngm = wfc_g.size();
    if(OT=='w' and ik < sys.bz().nkpts_ibz) {
      // read directly, no need to shuffle
      auto Ow = Orb(nda::range(r.size()));
      h5_read(*h5file, orb_prefix+"_s"+std::to_string(is) + 
                       "_k"+std::to_string(ik), Ow, std::tuple{m,r});
      // MAM: fractional translations!!!
      //if(ik >= sys.bz().nkpts_ibz and ft>0) Orb(r) *= Xft(r);
    } else {
      nda::array<ComplexType,1> Ow(ngm);
      h5_read(*h5file, orb_prefix+"_s"+std::to_string(is)+"_k"+
                       std::to_string(sys.bz().kp_to_ibz(ik)), Ow, std::tuple{m,nda::range::all});
      // apply symmetry rotation
      auto it = std::find(ksymms.begin(), ksymms.end(), sys.bz().kp_symm(ik)); 
      utils::check( it != ksymms.end(), "Error in orbital_from_h5_impl: Missing symmetry in ksymms: {}",sys.bz().kp_symm(ik));
      int isym = std::distance(ksymms.begin(),it); 
      auto w2g = swfc_maps.local()((OT=='w'?0:1), isym, (sys.bz().kp_trev(ik)?1:0), nda::range::all);
      if(r.first() > 0 or r.last() < ngm) {
        for( auto [i,n] : itertools::enumerate(w2g) )
          if( n >= r.first() and n < r.last() )
            Orb( n-r.first() ) = Ow(i);
      } else {
        for( auto [i,n] : itertools::enumerate(w2g) )
          Orb(n) = Ow(i);
      }
    }
    // conjugate if trev point 
    if(sys.bz().kp_trev(ik))
      nda::tensor::scale(ComplexType(1.0),Orb,nda::tensor::op::CONJ);
  }

  template<nda::ArrayOfRank<2> A2D>
  void orbital_set_from_h5_impl(std::string orb_prefix, char OT, int is, int ik, int b0, int nb, A2D&& Orb, nda::range r)
  {
    static_assert(nda::mem::on_host<std::decay_t<A2D>> or nda::mem::on_unified<std::decay_t<A2D>>, "Memory mismatch.");
    static_assert(nda::is_complex_v<typename std::decay_t<A2D>::value_type>, "Type mismatch");
    utils::check(Orb.shape()[0] >= nb, "Dimension mismatch.");
    utils::check(Orb.shape()[1] >= r.size(), "Dimension mismatch.");
    Orb() = ComplexType(0.0);
    long ngm = wfc_g.size();
    if(OT=='w' and ik < sys.bz().nkpts_ibz) {
      // read directly, no need to shuffle
      utils::check(Orb.strides()[0] == Orb.shape()[1], "qe_readonly::orbital_set_from_h5: Layout mismatch.");
      auto Ow = Orb(nda::range(nb),nda::range(r.size()));      
      h5_read(*h5file, orb_prefix+"_s"+std::to_string(is)+"_k"+std::to_string(sys.bz().kp_to_ibz(ik)),
              Ow, std::tuple{nda::range(b0,b0+nb),r});
      // MAM: fractional translations!!!
      //if(ik >= sys.bz().nkpts_ibz and ft>0) Orb(r_) *= Xft(r_);
    } else {
      nda::array<ComplexType,2> Ow(nb,ngm);
      h5_read(*h5file, orb_prefix+"_s"+std::to_string(is)+"_k"+
                       std::to_string(sys.bz().kp_to_ibz(ik)), Ow, 
                       std::tuple{nda::range(b0,b0+nb),nda::range::all});
      // apply symmetry rotation 
      auto it = std::find(ksymms.begin(), ksymms.end(), sys.bz().kp_symm(ik)); 
      utils::check( it != ksymms.end(), "Error in orbital_from_h5_impl: Missing symmetry in ksymms: {}",sys.bz().kp_symm(ik));
      int isym = std::distance(ksymms.begin(),it); 
      auto w2g = swfc_maps.local()((OT=='w'?0:1), isym, (sys.bz().kp_trev(ik)?1:0), nda::range::all);
      if(r.first() > 0 or r.last() < ngm) {
        for(int b=0; b<nb; ++b)
          for( auto [i,n] : itertools::enumerate(w2g) )
            if( n >= r.first() and n < r.last() )
              Orb( b, n-r.first() ) = Ow( b, i );
      } else {
        for(int b=0; b<nb; ++b)
          for( auto [i,n] : itertools::enumerate(w2g) )
            Orb( b, n ) = Ow( b , i );
      }
    }
    // conjugate if trev point 
    if(sys.bz().kp_trev(ik))
      nda::tensor::scale(ComplexType(1.0),Orb,nda::tensor::op::CONJ);
  }


};

} // namespace bdft
} // namespace mf

#endif //COQUI_BDFT_READONLY_HPP
