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


#ifndef MEANFIELD_BZ_SYMMETRY_HPP
#define MEANFIELD_BZ_SYMMETRY_HPP

#include <map>
#include <string>
#include "IO/AppAbort.hpp"
#include "configuration.hpp"
#include "mpi3/communicator.hpp"
#include <h5/h5.hpp>
#include <nda/nda.hpp>
#include <nda/h5.hpp>
#include "utilities/check.hpp"
#include "utilities/kpoint_utils.hpp"
#include "utilities/symmetry.hpp"
#include "utilities/concepts.hpp"
#include <hdf5.h>
#include <hdf5_hl.h>

namespace mf
{

/* 
 * Class that contains symmetry information on the 1BZ, both kpt and Qpt grids 
 */
struct bz_symm 
{
  public:

  bz_symm() = default;

  /**
   * @param comm       - mpi communicator
   * @param no_q_sym   - use symmetry for momentum transferred (q-points)
   * @param latt       - lattice translational vectors
   * @param recv       - reciprocal vectors
   * @param kp_grid_   - Monkhorst-Pack mesh
   * @param kpts_      - list of kpoints in IBZ
   * @param symm_list_ - list of symmetry operations
   * @param use_trev   - use time-reversal symmetry or not
   */
  bz_symm(boost::mpi3::communicator& comm,
          bool no_q_sym,
          nda::array<double, 2> latt,
          nda::array<double, 2> recv,
          nda::array<double, 1> kp_grid_,
          nda::array<double, 2> kpts_,
          std::vector<utils::symm_op> symm_list_,
          bool use_trev):
        nkpts_ibz(kpts_.shape()[0]),
        twist{0,0,0},
        kp_grid(kp_grid_),
        symm_list(symm_list_)
  {
    using v_stack = std::vector<nda::stack_array<double, 3>>;
    decltype(nda::range::all) all;
    utils::check(symm_list.size() >= 1, "At least 1 symmetry operation is expected.");
    utils::check(kpts_.shape()[1] == 3, "Wrong dimensions in bz_symm constructor.");

// MAM: this search for the 1BZ from the given IBZ is not well defined when the original grid
//      is not complete/symmetric. Instead, do this only when nk1/nk2/nk3 is not provided.
//      When we know the grid, you can reconstruct the grid without ambiguities... 

    // generate kp list in 1-BZ
    int nsymm = symm_list.size();
    v_stack kplist;  // kpoints in lattice coordinates (kpts_crys)
    std::vector<int> kp_s; // kp_symm
    std::vector<int> kp_ibz; // kp_to_ibz
    std::vector<bool> trev; // t-reversal flag 
    kp_s.reserve(96*nkpts_ibz);
    kp_ibz.reserve(96*nkpts_ibz);
    trev.reserve(96*nkpts_ibz);
	
    nda::stack_array<double,3> kp;
    double tpi = 2.0 * 3.14159265358979; 
    double tpiinv = 1.0 / tpi; 
    int k_gamma = -1;
    for(int i=0; i<nkpts_ibz; ++i) {
      nda::blas::gemv(tpiinv,latt,kpts_(i,all),0.0,kp);
      kplist.emplace_back(kp);
      kp_s.emplace_back(0);
      kp_ibz.emplace_back(i);
      trev.emplace_back(false);
      if( kp(0)*kp(0) + kp(1)*kp(1) + kp(2)*kp(2) < 1e-12 ) k_gamma = i;
      app_debug(3,"IBZ kpoint:{}  ({},{},{})",i,kp(0),kp(1),kp(2));
    }

    // cheating a bit, since it compares to kp! Careful!
    auto comp = [&kp](nda::ArrayOfRank<1> auto&& a) {
	// doing this by hand, not sure what's a better way
	double di = std::abs(a(0)-kp(0)); 
        if( std::abs(di-1.0) < 1e-4 ) di = 0.0;
        if( std::abs(di-2.0) < 1e-4 ) di = 0.0;
        if( std::abs(di-3.0) < 1e-4 ) di = 0.0;
        if( std::abs(di-4.0) < 1e-4 ) di = 0.0;
	double dj = std::abs(a(1)-kp(1)); 
        if( std::abs(dj-1.0) < 1e-4 ) dj = 0.0;
        if( std::abs(dj-2.0) < 1e-4 ) dj = 0.0;
        if( std::abs(dj-3.0) < 1e-4 ) dj = 0.0;
        if( std::abs(dj-4.0) < 1e-4 ) dj = 0.0;
	double dk = std::abs(a(2)-kp(2)); 
        if( std::abs(dk-1.0) < 1e-4 ) dk = 0.0;
        if( std::abs(dk-2.0) < 1e-4 ) dk = 0.0;
        if( std::abs(dk-3.0) < 1e-4 ) dk = 0.0;
        if( std::abs(dk-4.0) < 1e-4 ) dk = 0.0;
	return di + dj + dk < 1e-12; 
    };

    if(nsymm > 1 or use_trev) { 

      int ngrid = std::accumulate(kp_grid.begin(),kp_grid.end(),1,std::multiplies<>{});
      utils::check(ngrid > 0, "bz_symmetry: nk_grid:{} with nsymm>1 or use_trev",ngrid); 
      // full kpoint list
//MAM: Make the grid in range [-0.5,0.5) to be consistent with QE and Wannier90
      v_stack kpoints;
      {
        kpoints.reserve(ngrid);
        int ni = int(kp_grid(0))/2;
        int nj = int(kp_grid(1))/2; 
        int nk = int(kp_grid(2))/2; 
        int mni = -ni; if(int(kp_grid(0))%2==0) ni--;
        int mnj = -nj; if(int(kp_grid(1))%2==0) nj--; 
         int mnk = -nk; if(int(kp_grid(2))%2==0) nk--;
        for(int i=mni; i<=ni; ++i)
          for(int j=mnj; j<=nj; ++j)
            for(int k=mnk; k<=nk; ++k) {
              kp[0] = 1.0*i/kp_grid(0); 
              kp[1] = 1.0*j/kp_grid(1); 
              kp[2] = 1.0*k/kp_grid(2); 
              kpoints.emplace_back(kp);
            }
      }

      // kp grid is not provided, generate it using symmetry operations.
      // will generate all kpoints consistent with symmetry
      if(use_trev) {
        for(int j=0; j<nkpts_ibz; ++j) {
          if(j == k_gamma) continue;  // skip gamma to save some effort
          kp = -1.0*kplist[j];
          if( (ngrid==0 or std::any_of(kpoints.begin(),kpoints.end(),comp)) and 
              std::none_of(kplist.begin(),kplist.end(),comp) ) {
            kplist.emplace_back(kp);
            kp_s.emplace_back(0);
            kp_ibz.emplace_back(j);
            trev.emplace_back(true);
          }
        }
      }

      // being a bit careful here, (attempting...) to choose as small number of symmetries as possible
      int newv = 0;
      std::vector<bool> done(nsymm,false);
      done[0] = true;
      do {
        v_stack v_max; 
        std::vector<int> i_max;
        std::vector<bool> tflag_max;
        int is_max=0;
        newv=0;
        for(int is=0; is<nsymm; ++is) { 
          if(done[is]) continue;
          v_stack v;
          std::vector<int> idx;
          std::vector<bool> tflag;
          // create list of all new kps generated from this operation acting kplist
          for(int j=0; j<nkpts_ibz; ++j) {
            if(j == k_gamma) continue;  // skip gamma to save some effort
            nda::blas::gemv(1.0,nda::transpose(symm_list[is].Rinv),kplist[j],0.0,kp);
            if( (ngrid==0 or std::any_of(kpoints.begin(),kpoints.end(),comp)) and 
                std::none_of(kplist.begin(),kplist.end(),comp) and 
                std::none_of(v.begin(),v.end(),comp)) {
              v.emplace_back(kp);
              idx.emplace_back(j);
              tflag.emplace_back(false); 
            }
            if(use_trev) {
              kp *= -1.0; 
              if( (ngrid==0 or std::any_of(kpoints.begin(),kpoints.end(),comp)) and 
                  std::none_of(kplist.begin(),kplist.end(),comp) and 
                  std::none_of(v.begin(),v.end(),comp)) {
                v.emplace_back(kp);
                idx.emplace_back(j);
                tflag.emplace_back(true);
              }
            }
          }
          // keep it if it this symmetry generates the most new vectors
          if(v.size() > 0 and v.size() > newv) {
            newv = v.size();
            v_max = v;
            i_max = idx;
            tflag_max = tflag;
            is_max = is;
          }
        }
        // add to the list
        if(newv > 0) {
          done[is_max] = true;
          for(int i=0; i<newv; ++i) {
            kplist.emplace_back(v_max[i]);
            kp_s.emplace_back(is_max);
            kp_ibz.emplace_back(i_max[i]);	  
            trev.emplace_back(tflag_max[i]);	  
          }
        }
      }
      while(newv > 0);

    } // nsymm > 1 or trev

    auto find_last_false = [] (auto it0, auto it1) {
      if(std::distance(it0,it1) == 0) return it0; 
      do 
      {
        --it1;
        if(not (*it1)) return it1; 
        if(std::distance(it0,it1) == 0) return it1; 
      } while(std::distance(it0,it1) > 0);
      return it1;
    }; 

    // now move all trev kpoints to the end of the list
    std::vector<bool>::iterator it_b = std::find(trev.begin() + nkpts_ibz, trev.end(), true);
    utils::check(std::distance(it_b,trev.end())==0 or 
                 *it_b, "Error: Logic error in constructor of bz_symm. \n");    
    std::vector<bool>::iterator it_e = find_last_false(it_b, trev.end()); 
    utils::check(std::distance(it_e,trev.end())==0 or 
                 std::distance(it_e,it_b)==0       or  
                 not *it_e, "Error: Logic error in constructor of bz_symm. \n");    
    while (it_b != trev.end() and std::distance(it_b,it_e) > 0) {

      utils::check( *it_b and not *it_e, "Error: Logic error in constructor of bz_symm. \n");    

      // swap
      long a = std::distance(trev.begin(),it_b);
      long b = std::distance(trev.begin(),it_e);
      kp = kplist[a]; kplist[a]=kplist[b]; kplist[b]=kp;
      std::swap(kp_s[a],kp_s[b]);
      std::swap(kp_ibz[a],kp_ibz[b]);
      std::vector<bool>::swap(trev[a],trev[b]);

      // search next
      it_b = std::find(it_b, trev.end(), true);
      it_e = find_last_false(it_b, trev.end()); 

    }; 
    it_b = std::find(trev.begin(), trev.end(), true);
    nkpts_trev_pairs = std::distance(it_b,trev.end()); 

    // npw, eigval, occ, kp_weight, in 1-BZ
    nkpts    = kplist.size();
    kpts     = nda::array<double, 2>::zeros({nkpts,3}); 
    kpts_crys= nda::array<double, 2>::zeros({nkpts,3}); 
    kp_symm  = nda::array<int, 1>::zeros({nkpts});
    kp_trev  = nda::array<bool, 1>(nkpts); 
    kp_trev_pair = nda::array<int, 1>(nkpts);
    kp_trev_pair() = -1;
    kp_to_ibz   = nda::array<int, 1>::zeros({nkpts});
    for(int ik=0; ik<nkpts; ik++) {
      // keep kpoints in crystal coordinates for a bit longer
      kpts_crys(ik,all)  = kplist[ik];
      kp_symm(ik)        = kp_s[ik];	
      kp_to_ibz(ik)      = kp_ibz[ik];	
      kp_trev(ik)        = trev[ik];
    } 
    // find trev pairs
    for(int ik=nkpts-nkpts_trev_pairs; ik<nkpts; ++ik) {

      for(int n=0; n<nkpts-nkpts_trev_pairs; ++n) {
        if( kp_symm(n) == kp_symm(ik) and
            kp_to_ibz(n) == kp_to_ibz(ik) ) { 
          utils::check(not kp_trev(n), "Error: Logic error in bz_symm (1).");
          utils::check(std::abs(kpts_crys(ik,0) + kpts_crys(n,0)) +
                       std::abs(kpts_crys(ik,1) + kpts_crys(n,1)) +
                       std::abs(kpts_crys(ik,2) + kpts_crys(n,2)) < 1e-8, 
                       "Error: Logic error in bz_symm (2).");
          kp_trev_pair(ik) = n;
          kp_trev_pair(n)  = ik;
          break;
        }
      }
      utils::check(kp_trev_pair(ik)>=0, "Error: Logic error in bz_symm.");
    }

    // check or generate kp_grid
    int ngrid = std::accumulate(kp_grid.begin(),kp_grid.end(),1,std::multiplies<>{});
    // not an automatic grid, find dimensions
    if(ngrid <= 0)
      kp_grid = utils::kp_grid_dims(kpts_crys);
    ngrid = std::accumulate(kp_grid.begin(),kp_grid.end(),1,std::multiplies<>{});
    // MAM: calculate_Qpt_maps below checks the consistency between kpts_crys and Qpts 
    //      e.g. Qpts is a gamma_centered grid containing all kpoint differences.
    utils::check(nkpts == ngrid, " Error: Failed to generate regular k-point grid. Check your input grid (expected tolerance of 1e-6). Otherwise contact developers if youa re sure your input grid is regular. "); 

    // Qpts in 1-BZ
    nqpts = nkpts;   
    Qpts     = nda::array<double, 2>::zeros({nqpts,3});
    qk_to_k2 = nda::array<int, 2>::zeros({nqpts,nkpts});
    qminus   = nda::array<int, 1>::zeros({nqpts});   
    qp_symm  = nda::array<int, 1>::zeros({nqpts});
    qp_trev  = nda::array<bool, 1>(nqpts); qp_trev()=false; 
    qp_to_ibz   = nda::array<int, 1>::zeros({nqpts});

    // QPts in 1st-BZ (in crystal coords)
    if(k_gamma >= 0)
      // gamma centered grid, use Qpts_ibz=kpts_ibz
      Qpts() = kpts_crys();
    else
      utils::generate_and_check_qgrid(kp_grid,Qpts);

    // find Qpts in IBZ
    if(no_q_sym or (nsymm==1 and not use_trev)) {
      nqpts_ibz = nqpts;
      qp_symm() = 0;
      qp_trev() = false;
      for(int i=0; i<nqpts; i++) qp_to_ibz(i)=i;
    } else {
      if(k_gamma >= 0) {
        nqpts_ibz = nkpts_ibz;
        qp_symm() = kp_symm();
        qp_trev() = kp_trev();
        qp_to_ibz() = kp_to_ibz();
      } else {
        nqpts_ibz = utils::generate_ibz(use_trev,symm_list,Qpts,qp_symm,qp_trev,qp_to_ibz);
      }
    }

    std::tie(qsymms,nq_per_s,ks_to_k,Qs,qs_to_q) =
         utils::generate_qsymm_maps(use_trev,symm_list,qp_symm,qp_trev,nkpts_ibz,kpts_crys,nqpts_ibz,Qpts);

    print_logs();

    // initialize Q grids, checks Qpts against kpts_crys
    utils::calculate_Qpt_maps(comm,recv,kpts_crys,Qpts,qk_to_k2,qminus);

    // transform vectors to cartesian coordinates
    for(int ik=0; ik<nkpts; ik++)
      nda::blas::gemv(1.0,nda::transpose(recv),kpts_crys(ik,all),0.0,kpts(ik,all));
    for(int iq=0; iq<nqpts; iq++) { 
      kplist[0] = Qpts(iq,all);
      nda::blas::gemv(1.0,nda::transpose(recv),kplist[0],0.0,Qpts(iq,all));
    }

  }

  bz_symm(std::string fn) {
    h5::file file = h5::file(fn, 'r');
    h5::group grp(file);
    h5::group sgrp = grp.open_group("System");
    initialize_from_h5(sgrp);
  }

  bz_symm(h5::group& grp) {
    initialize_from_h5(grp);
  }

  template<class MF>
  bz_symm(MF const& mf, std::string fn, bool to_h5)  
  {
    initialize_from_MF(mf);
    if(to_h5) {
      h5::file file = h5::file(fn, 'a');
      h5::group grp(file);
      save(grp);
    }
  }

  bz_symm(bz_symm const&) = default;
  bz_symm(bz_symm &&) = default;

  ~bz_symm() = default;

  bz_symm& operator=(bz_symm const&) = default;
  bz_symm& operator=(bz_symm &&) = default;

  void print_logs() {
    app_log(2,"  Brillouin zone symmetry info");
    app_log(2,"  ----------------------------");
    app_log(2,"  Q-points in the irreducible zone = {}",nqpts_ibz);
    app_log(2,"  Symmetries applied to Q-points   = {}",Qs.extent(0));
    app_log(2,"  Time-reversal k-point pairs      = {}",nkpts_trev_pairs);
    for(int i=0; i<nkpts; i++)
      app_debug(3,"  Kpt:{} ({},{},{}), is:{}, ibz:{}, trev:{}, trev_pair:{}",
                i,kpts_crys(i,0),kpts_crys(i,1),kpts_crys(i,2),kp_symm(i),
                kp_to_ibz(i),kp_trev(i),kp_trev_pair(i));
    for(int i=0; i<nqpts; i++)
      app_debug(3,"  Qpt:{} ({},{},{}), is:{}, ibz:{}, trev:{}",
                i,Qpts(i,0),Qpts(i,1),Qpts(i,2),qp_symm(i),qp_to_ibz(i),qp_trev(i));
    app_log(2,"");
  }

  void save(std::string fn) const {
    h5::file file = h5::file(fn, 'a');
    h5::group grp(file);
    h5::group sgrp = ( grp.has_subgroup("System")  ?
                       grp.open_group("System")    :
                       grp.create_group("System", true) );
    save(sgrp);
  }

  void save(h5::group& grp) const {
    // for now delete old BZ dataset if one is present
    h5::group sgrp = grp.create_group("BZ", true);
    save_to_h5(sgrp);
  }

  static bool can_init_from_h5(h5::group& grp) {
    if(not grp.has_subgroup("BZ")) return false;
    std::string BZ("BZ/");
    std::string Symm("BZ/Symmetries/");

    auto bgrp = grp.open_group("BZ");

    // MAM: add has_attribute to h5 library and replace direct use of H5Aexists
    for( auto [i,v] : itertools::enumerate(std::vector<std::string>{"number_of_kpoints","number_of_kpoints_ibz","number_of_trev_kpoint_pairs","number_of_qpoints","number_of_qpoints_ibz"}) ) {
      if(not H5Aexists(h5::hid_t(bgrp),v.c_str())) return false;
    }

    for( auto [i,v] : itertools::enumerate(std::vector<std::string>{"kp_grid","kpoints","kpoints_crystal","kp_symm","kp_to_ibz","kp_trev","kp_trev_pair","qpoints","qk_to_k2","qminus","qp_symm","qp_trev","qp_to_ibz","qsymms","nq_per_s","ks_to_k","Qs","qs_to_q"}) ) { 
      if(not bgrp.has_dataset(v)) return false;
    }

    if(not bgrp.has_subgroup("Symmetries")) return false;
    auto sbgrp = bgrp.open_group("Symmetries");
    // symmetries
    {
      if(not H5Aexists(h5::hid_t(sbgrp),"number_of_symmetries")) return false;
      int nsym;
      h5::h5_read_attribute(sbgrp, "number_of_symmetries",nsym);
      for(int i=0; i<nsym; i++) {
        if(not sbgrp.has_subgroup("s"+std::to_string(i))) return false;
        if(not sbgrp.has_dataset("s"+std::to_string(i)+"/R")) return false;
        if(not sbgrp.has_dataset("s"+std::to_string(i)+"/ft")) return false;
      }
    }
    return true;
  }

  /*
   * Creates an object of type bz_symm,
   * consistent with a single kpoint calculation at the gamma point. 
   * Useful to initialize model hamiltonians. 
   */
  static auto gamma_point_instance() 
  {
    bz_symm s;

    s.nkpts = 1;                        
    s.nkpts_ibz = 1;                    
    s.nkpts_trev_pairs = 0;             
    s.nqpts = 1;                        
    s.nqpts_ibz = 1;
    s.kpts = {{0.0,0.0,0.0}}; 
    s.kpts_crys = {{0.0,0.0,0.0}};
    s.kp_grid = {1,1,1};
    s.Qpts = s.kpts;
    s.qk_to_k2 = {{0}};
    s.qminus = {0};
    s.symm_list.clear();
    s.symm_list.reserve(1);
    nda::array<double,2> R = { {1.0,0,0}, {0,1.0,0}, {0,0,1.0} }; 
    nda::array<double,1> ft = {0,0,0};
    s.symm_list.emplace_back(utils::symm_op{R,R,ft});
    s.kp_symm = {0};
    s.kp_trev = {false};
    s.kp_trev_pair = {0};
    s.kp_to_ibz = {0};
    s.qp_symm = s.kp_symm;
    s.qp_trev = s.kp_trev;
    s.qp_to_ibz = s.kp_to_ibz;
    s.qsymms = {0};
    s.nq_per_s = {1}; 
    s.Qs = {{0}};
    s.ks_to_k = {{0}}; 
    s.qs_to_q = {{0}};

    return s;
  }

  /*
   * Creates an h5 file that can be used to initialize an object of type bz_symm,
   * consistent with a single kpoint calculation at the gamma point. 
   * Useful to initialize model hamiltonians. 
   */
  static void gamma_point_h5(std::string fname)
  {
    auto s = gamma_point_instance();
    s.save(fname);
  }

  static void gamma_point_h5(h5::group& grp)
  { 
    auto s = gamma_point_instance();
    s.save(grp);
  }

  // basic info
  int nkpts = 0;                        // total # of kpts
  int nkpts_ibz = 0;                    // # of kpts in IRBZ
  int nkpts_trev_pairs = 0;             // # of kpt pairs related by trev symmetry on the list 
                                        // kpoints with trev(n)==true are guaranteed to be at the end
  int nqpts = 0;                        // total # of qpts
  int nqpts_ibz = 0;                    // # of qpts in IRBZ

  // full list of kpoints
  // the first nkpts_ibz are the kpoints in the IRBZ
  nda::array<double, 2> kpts;      // in cartesian coordinates
  nda::array<double, 2> kpts_crys; // in crystal coordinates

  // global twist in the kp grid
  nda::stack_array<double, 3> twist;

  // kpgrid : Monkhorst-Pack mesh {n1,n2,n3}
  nda::stack_array<int, 3> kp_grid;

  // full list of momentum transfers
  // the first nqpts_ibz vectors are the ones in the IRBZ
  nda::array<double, 2> Qpts;  

  // The definition of the Q vectors in the 1BZ is given by 
  // Qpts[q] + G = kpts[a] - kpts[b] , where G belongs to the rec. cell. 
  // Based on this definition/convention, we have the following mappings:
  // for qk_to_k2[q, a] = b, defines the relation: 
  // kpts[b] = kpts[a] - (Qpts[q] + G)
  nda::array<int, 2> qk_to_k2;

  // for qminus[q] = n, defines the relation:
  // Qpts[q] = G - Qpts[n], where G belongs to the rec. cell. 
  nda::array<int, 1> qminus;  

  // symmetry operations
  // identity operation should always exist and be the first
  std::vector<utils::symm_op> symm_list;

  // index of symmetry operation that connects kpts/kpts_crys to IRBZ
  nda::array<int, 1> kp_symm;  
 
  // whether time-reversal symmetry is needed in combination with the symmetry in kp_symm
  nda::array<bool, 1> kp_trev;  

  // if the n-th kpoint belongs to a time-reversible pair in the list, 
  // kp_trev_pair(n) contains the index of the kpoint 
  // if the kpoint is not part of a pair, kp_trev_pair is set to <0.
  // MAM: in principle, this should just be kminus (similar to qminus), but I can't convince
  //      myself that this is guaranteed, so keeping it in a separate structure
  nda::array<int, 1> kp_trev_pair;

  // kpoint in IRBZ that is connected to each kp in the 1st BZ (kpts/kpts_crys) by the symmetry in kp_symm
  nda::array<int, 1> kp_to_ibz;  

  // index of symmetry operation that connects Qpts to IRBZ
  nda::array<int, 1> qp_symm;

  // whether trev symmetry is needed in combination with the symmetry in qp_symm 
  nda::array<bool, 1> qp_trev;  

  // Qpoint in IRBZ that is connected to each qp in the 1st BZ (Qpts) by the symmetry in qp_symm
  nda::array<int, 1> qp_to_ibz;

  // list of symmetries used to reconstruct Qpts in 1st_BZ from IBZ. 
  // nq_per_s, Qs, qs_to_q and ks_to_k use this array to map index to actual symmetry operations.   
  nda::array<int, 1> qsymms;

  // number of Qpts in the 1st BZ that use symmetry s (index in qsymms) to map to IBZ
  // nq_per_s(is) = n, implies that qsymms(is) appears n times in qp_symm({0,nqpts}) 
  nda::array<int, 1> nq_per_s;

  // for every symmetry, qsymms(is) = S, list of qpoints that map to the IBZ through S. 
  // Qs(is,iq) = n implies that qp_symm(n) = qsymms(is).
  nda::array<int, 2> Qs;

  // maps k_ibz*S to the corresponding point in the 1stBZ 
  // concretely, ks_to_k(is,kibz) = index of kp = kibz * S, in kpts array, 
  // where S is the symmetry indexed by qsymms(is)
  nda::array<int, 2> ks_to_k;  

  // maps q_ibz*S to the corresponding point in the 1stBZ 
  // concretely, qs_to_q(is,qibz) = index of Qpts = qibz * S, in Qpts array, 
  // where S is the symmetry indexed by qsymms(is)
  nda::array<int, 2> qs_to_q;  

  void initialize_from_h5(h5::group& grp) 
  {
    // BZ info
    h5::group bgrp = grp.open_group("BZ");
    h5::group bsgrp = bgrp.open_group("Symmetries");

    h5::h5_read_attribute(bgrp, "number_of_kpoints", nkpts);
    h5::h5_read_attribute(bgrp, "number_of_kpoints_ibz", nkpts_ibz);
    h5::h5_read_attribute(bgrp, "number_of_trev_kpoint_pairs", nkpts_trev_pairs);
    nda::h5_read(grp, "BZ/kp_grid", kp_grid);
    kpts = nda::array<double, 2>(nkpts, 3);
    kpts_crys = nda::array<double, 2>(nkpts, 3);
    kp_symm = nda::array<int, 1>(nkpts);
    kp_trev = nda::array<bool, 1>(nkpts); 
    kp_trev_pair = nda::array<int, 1>(nkpts); 
    kp_to_ibz = nda::array<int, 1>(nkpts);
    nda::h5_read(grp, "BZ/kpoints", kpts);
    nda::h5_read(grp, "BZ/kpoints_crystal", kpts_crys);
    nda::h5_read(grp, "BZ/kp_symm", kp_symm);
    nda::h5_read(grp, "BZ/kp_to_ibz", kp_to_ibz);
    nda::h5_read(grp, "BZ/kp_trev", kp_trev);
    nda::h5_read(grp, "BZ/kp_trev_pair", kp_trev_pair);

    // symmetries
    {
      bool found_E = false;
      int nsym;
      nda::stack_array<double, 3, 3> R;
      nda::matrix<double> Rinv(3,3);
      nda::stack_array<double, 3> ft;
      h5::h5_read_attribute(bsgrp, "number_of_symmetries",nsym);
      symm_list.clear();
      symm_list.reserve(nsym);
      for(int i=0; i<nsym; i++) {
        nda::h5_read(grp, "BZ/Symmetries/s"+std::to_string(i)+"/R",R);
        nda::h5_read(grp, "BZ/Symmetries/s"+std::to_string(i)+"/ft",ft);
        utils::check(std::abs(ft(0)*ft(0)+ft(1)*ft(1)+ft(2)*ft(2)) < 1e-12,
            " Error: Fractional translations not yet allowed. Use force_symmorphic=true.");
        if( std::abs(
              std::pow(R(0,0)-1.0,2.0) + std::pow(R(0,1),2.0) + std::pow(R(0,2),2.0) +
              std::pow(R(1,0),2.0) + std::pow(R(1,1)-1.0,2.0) + std::pow(R(1,2),2.0) +
              std::pow(R(2,0),2.0) + std::pow(R(2,1),2.0) + std::pow(R(2,2)-1.0,2.0)
            ) < 1e-12 ) {
          utils::check(std::abs(ft(0)*ft(0)+ft(1)*ft(1)+ft(2)*ft(2)) < 1e-12,
                       "Error: Identity operation has non-zero ft:({},{},{})",ft(0),ft(1),ft(2));
          found_E = true;
          if(symm_list.size() == 0)
            symm_list.emplace_back(utils::symm_op{R,R,ft});
          else {
            symm_list.emplace_back(symm_list[0]);
            symm_list[0].R = R;
            symm_list[0].Rinv = R;
            symm_list[0].ft = ft;
          }
        } else {
          Rinv = R;
          nda::inverse3_in_place(Rinv);
          symm_list.emplace_back(utils::symm_op{R,Rinv,ft});
        }
      }
      utils::check(found_E, "Error: Identity operation not found among symmetry list.");
      utils::check(symm_list.size() == nsym,
                   "Error parsing symmetries from qe::xml. nsym:{}, # symmetries found:{}",
                   nsym,symm_list.size());
    }

    h5::h5_read_attribute(bgrp, "number_of_qpoints", nqpts);
    h5::h5_read_attribute(bgrp, "number_of_qpoints_ibz", nqpts_ibz);
    Qpts     = nda::array<double, 2>(nqpts, 3);
    qk_to_k2 = nda::array<int, 2>(nqpts, nkpts);
    qminus   = nda::array<int, 1>(nqpts);
    qp_symm = nda::array<int, 1>(nqpts);
    qp_trev = nda::array<bool, 1>(nqpts); 
    qp_to_ibz = nda::array<int, 1>(nqpts);
    nda::h5_read(grp, "BZ/qpoints", Qpts);
    nda::h5_read(grp, "BZ/qk_to_k2", qk_to_k2);
    nda::h5_read(grp, "BZ/qminus", qminus);
    nda::h5_read(grp, "BZ/qp_symm", qp_symm);
    nda::h5_read(grp, "BZ/qp_trev", qp_trev);
    nda::h5_read(grp, "BZ/qp_to_ibz", qp_to_ibz);

    int nsym=symm_list.size();
    qsymms = nda::array<int, 1>::zeros({nsym});
    nq_per_s = nda::array<int, 1>::zeros({nsym});
    ks_to_k = nda::array<int, 2>::zeros({nsym, nkpts});
    Qs = nda::array<int, 2>::zeros({nsym, 2*nqpts_ibz});
    qs_to_q = nda::array<int, 2>::zeros({nsym, nqpts_ibz});
    nda::h5_read(grp, "BZ/qsymms", qsymms);
    nda::h5_read(grp, "BZ/nq_per_s", nq_per_s);
    nda::h5_read(grp, "BZ/ks_to_k", ks_to_k);
    nda::h5_read(grp, "BZ/Qs", Qs);
    nda::h5_read(grp, "BZ/qs_to_q", qs_to_q);

    print_logs();
  }

  template<class MF>
  void initialize_from_MF(MF const& mf)
  {
    // BZ info
    nkpts      = mf.nkpts();
    nkpts_ibz  = mf.nkpts_ibz();
    nkpts_trev_pairs = mf.nkpts_trev_pairs();
    kp_grid    = mf.kp_grid();
    kpts       = mf.kpts();
    kpts_crys  = mf.kpts_crystal();

    nqpts      = mf.nqpts();
    nqpts_ibz  = mf.nqpts_ibz();
    Qpts       = mf.Qpts();
    qk_to_k2   = mf.qk_to_k2();
    qminus   = mf.qminus();

    // symmetry
    symm_list = mf.symm_list();
    kp_symm   = mf.kp_symm();
    kp_trev   = mf.kp_trev();
    kp_trev_pair   = mf.kp_trev_pair();
    kp_to_ibz = mf.kp_to_ibz();
    qp_symm   = mf.qp_symm();
    qp_trev   = mf.qp_trev();
    qp_to_ibz = mf.qp_to_ibz();
    qsymms    = mf.qsymms();
    nq_per_s  = mf.nq_per_s();
    Qs        = mf.Qs();
    ks_to_k   = mf.ks_to_k();
    qs_to_q   = mf.qs_to_q();
  }

  private:

  void save_to_h5(h5::group& grp) const
  {
    // BZ info
    h5::h5_write_attribute(grp, "number_of_kpoints", nkpts);
    h5::h5_write_attribute(grp, "number_of_kpoints_ibz", nkpts_ibz);
    h5::h5_write_attribute(grp, "number_of_trev_kpoint_pairs", nkpts_trev_pairs);
    nda::h5_write(grp, "kp_grid", kp_grid, false);
    nda::h5_write(grp, "kpoints", kpts, false);
    nda::h5_write(grp, "kpoints_crystal", kpts_crys, false);
    nda::h5_write(grp, "kp_symm", kp_symm, false);
    nda::h5_write(grp, "kp_to_ibz", kp_to_ibz, false);
    nda::h5_write(grp, "kp_trev", kp_trev, false);
    nda::h5_write(grp, "kp_trev_pair", kp_trev_pair, false);

    // symmetries
    {
      h5::group symgrp = grp.create_group("Symmetries");
      int nsym=symm_list.size();
      h5::h5_write_attribute(symgrp, "number_of_symmetries",nsym);
      for(int i=0; i<nsym; i++) {
        h5::group g_ = symgrp.create_group("s"+std::to_string(i));
        nda::h5_write(g_, "R",symm_list[i].R, false);
        nda::h5_write(g_, "ft",symm_list[i].ft, false);
      }
    }

    h5::h5_write_attribute(grp, "number_of_qpoints", nqpts);
    h5::h5_write_attribute(grp, "number_of_qpoints_ibz", nqpts_ibz);
    nda::h5_write(grp, "qpoints", Qpts);
    nda::h5_write(grp, "qk_to_k2", qk_to_k2);
    nda::h5_write(grp, "qminus", qminus);
    nda::h5_write(grp, "qp_symm", qp_symm);
    nda::h5_write(grp, "qp_trev", qp_trev);
    nda::h5_write(grp, "qp_to_ibz", qp_to_ibz);

    nda::h5_write(grp, "qsymms", qsymms, false);
    nda::h5_write(grp, "nq_per_s", nq_per_s, false);
    nda::h5_write(grp, "ks_to_k", ks_to_k, false);
    nda::h5_write(grp, "Qs", Qs, false);
    nda::h5_write(grp, "qs_to_q", qs_to_q, false);
  }

};

} // mf 

#endif
