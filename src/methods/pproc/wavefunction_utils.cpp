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



#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/Timer.hpp"
#include "utilities/check.hpp"
#include "utilities/occupations.hpp"

#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "h5/h5.hpp"

#include "numerics/nda_functions.hpp"
#include "mean_field/MF.hpp"

namespace methods
{

namespace detail
{

// need version of these functions which take a full dense matrix and write a sparse version!

/*
 * Writes an occupation matrix in "dense" format (consistent with AFQMC code) 
 * MAM: Check that the value_type of M is integer?
 */ 
template<nda::MemoryArrayOfRank<1> Arr1D>
void add_occM_dense(h5::group& grp, std::string dname, int nbnd, int nkpts, Arr1D const& M)
{
  int nel = int(M.extent(0));
  auto psi = nda::array<ComplexType,2>::zeros({nbnd*nkpts,nel});
  for( auto [i,n] : itertools::enumerate(M) )
    psi(n,i) = ComplexType(1.0);
  nda::h5_write(grp,dname,psi);
}

/*
 * Writes an occupation matrix in "dense" format (consistent with AFQMC code) 
 * MAM: Check that the value_type of M is integer?
 */ 
template<nda::MemoryArrayOfRank<1> Arr1D>
void add_occM_sparse(h5::group& grp, std::string dname, int nbnd, int nkpts, Arr1D const& M) 
{ // PsiT = transposed psi in sparse format 
  int nel = M.extent(0);
  h5::group pgrp = grp.create_group(dname);
  auto data = nda::array<ComplexType,1>::zeros({nel});
  auto jdata = nda::array<int,1>::zeros({nel});
  data() = ComplexType(1.0);
  nda::h5_write(pgrp,"data_",data);
  for( auto [i,n] : itertools::enumerate(M) )
    jdata(i) = n;
  nda::h5_write(pgrp,"jdata_",jdata);
  auto pb = nda::arange<int>(0,nel);
  nda::h5_write(pgrp,"pointers_begin_",pb);
  auto pe = nda::arange<int>(1,nel+1);
  nda::h5_write(pgrp,"pointers_end_",pe);
  auto dims = nda::array<int,1>::zeros({3});
  dims(0)=nel;
  dims(1)=nbnd*nkpts;
  dims(2)=nel;
  nda::h5_write(pgrp,"dims",dims);
}

void psi_from_occ_vector(h5::group & grp, mf::MF &mf, ptree const& pt, int nspins,  nda::ArrayOfRank<3> auto const& occ)
{
  auto nkpts = mf.nkpts();
  long mf_nspins = mf.nspin();
  long nbnd = mf.nbnd();
  long nup_mf = long(std::round(std::accumulate(occ(0,nda::ellipsis{}).begin(),
                                           occ(0,nda::ellipsis{}).end(),double(0.0))));
  long ndn_mf = ( mf_nspins!=2 ? nup_mf :
                  long(std::round(std::accumulate(occ(1,nda::ellipsis{}).begin(),
                                             occ(1,nda::ellipsis{}).end(),double(0.0)))) );
  auto upper_c = io::get_value_with_default<double>(pt,"upper_cutoff",0.95);
  auto lower_c = io::get_value_with_default<double>(pt,"lower_cutoff",0.05);

  utils::check( (nspins == 1) or (nspins == 2), " add_wavefunction: Invalid nspins:{}. Must be 1 or 2.", nspins);
  utils::check( nspins >= mf_nspins, " add_wavefunction: nspins:{} must be larger than nspins in mean_field object.",nspins);

  auto maxdet = io::get_value_with_default<int>(pt,"maxdet",1);
  auto det_limit = io::get_value_with_default<int>(pt,"det_limit",100000);  // too much, too little???

  auto confg = utils::dets_from_occupation_vector(nbnd,maxdet,nspins,nup_mf,ndn_mf,occ,upper_c,lower_c,det_limit); 
  auto ndet = confg.size();
  utils::check(ndet > 0, "add_wavefunction: Found 0 determinants. Contact developers.");
  // checking!
  int nel_in_confg = nup_mf + (nspins==2?ndn_mf:0);
  for(auto& c: confg)
  {
    utils::check( nel_in_confg == c.extent(0), 
                  "Error in add_wavefunction: Electron counts do not match. Contact developers.");
  } 

  app_log(2, " Total number of electrons in waveunction: nup:{}, ndown:{}: ",nup_mf,ndn_mf);
  app_log(2, " Number of determinants requested:{}, found:{}",maxdet,ndet); 
  app_log(2, " Number of occupied states per kpoint: ");
  for(int id=0; id<ndet; ++id) {
    nda::array<int,1> nk(nkpts);
    if(nspins==1) {
      for(int ik=0; ik<nkpts; ++ik) 
        nk(ik) = std::count_if(confg[id].begin(),confg[id].end(), 
                  [&](auto&& n) { return (n >= ik*nbnd) and (n < (ik+1)*nbnd); } );
      app_log(2, " determinant:{} {}",id,nk);
    } else {
      for(int ik=0; ik<nkpts; ++ik) 
        nk(ik) = std::count_if(confg[id].begin(),confg[id].begin()+nup_mf, 
                  [&](auto&& n) { return (n >= ik*nbnd) and (n < (ik+1)*nbnd); } );
      app_log(2, " determinant:{} spin up: {}",id,nk);
      for(int ik=0; ik<nkpts; ++ik)  
        nk(ik) = std::count_if(confg[id].begin()+nup_mf,confg[id].end(), 
                  [&](auto&& n) { return (n >= ik*nbnd) and (n < (ik+1)*nbnd); } );
      app_log(2, " determinant:{} spin down: {}",id,nk);
    }
  }

  h5::group wgrp_ = grp.create_group("Wavefunction");
  h5::group wgrp = wgrp_.create_group("NOMSD");
  detail::add_occM_dense(wgrp,"Psi0_alpha",nbnd,nkpts,confg[0](::nda::range(nup_mf)));
  if(nspins == 2) detail::add_occM_dense(wgrp,"Psi0_beta",nbnd,nkpts,confg[0](::nda::range(nup_mf,nup_mf+ndn_mf)));
  { 
    int cnt=0;
    for( auto const& [i,v] : itertools::enumerate(confg) ) {
      detail::add_occM_sparse(wgrp,"PsiT_"+std::to_string(cnt++),nbnd,nkpts,v(::nda::range(nup_mf)));
      if(nspins==2) detail::add_occM_sparse(wgrp,"PsiT_"+std::to_string(cnt++),nbnd,nkpts,v(::nda::range(nup_mf,nup_mf+ndn_mf)));
    }
  }
  {
    auto ci = nda::array<ComplexType,1>::zeros({confg.size()});
    ci()=ComplexType(1.0/std::sqrt(double(confg.size())));
    nda::h5_write(wgrp,"ci_coeffs",ci);
  }
  {
    auto dims = nda::array<int,1>::zeros({5});
    dims(0)=nbnd*nkpts;
    dims(1)=nup_mf;
    dims(2)=ndn_mf; 
    dims(3)=nspins;       // closed shell (2: collinear, 3: noncollinear, 4: fully spin polarized)
    dims(4)=confg.size(); // number of determinants
    nda::h5_write(wgrp,"dims",dims);
  }
}

// add version of add_PsiT that takes a dense matrix and writes it in sparse form

}

/*
 group      /Wavefunction
 group      /Wavefunction/NOMSD
 dataset    /Wavefunction/NOMSD/Psi0_alpha
 dataset    /Wavefunction/NOMSD/Psi0_beta
 group      /Wavefunction/NOMSD/PsiT_0
 dataset    /Wavefunction/NOMSD/PsiT_0/data_
 dataset    /Wavefunction/NOMSD/PsiT_0/dims
 dataset    /Wavefunction/NOMSD/PsiT_0/jdata_
 dataset    /Wavefunction/NOMSD/PsiT_0/pointers_begin_
 dataset    /Wavefunction/NOMSD/PsiT_0/pointers_end_
 dataset    /Wavefunction/NOMSD/ci_coeffs
 dataset    /Wavefunction/NOMSD/dims
 }
 */
void add_wavefunction(h5::group & grp, mf::MF &mf, ptree const& pt)
{
  app_log(2, "*************************************************");
  app_log(2, "               Adding  Wavefunction              "); 
  app_log(2, "*************************************************");

  auto occ = mf.occ();
  long mf_nspins = mf.nspin();
  long npol = mf.npol();
  // defaults to UHF expansion, which AFQMC expects.
  auto nspins = io::get_value_with_default<int>(pt,"nspins",(npol==1?2:1));
  auto add_wfn = io::get_value_with_default<std::string>(pt,"add_wavefunction","default");
  
  utils::check( (nspins == 1) or (nspins == 2), " add_wavefunction: Invalid nspins:{}. Must be 1 or 2.", nspins); 
  utils::check( nspins >= mf_nspins, " add_wavefunction: nspins:{} must be larger than nspins in mean_field object.",nspins);  

  if( add_wfn == "default" ) {
    app_log(2, " Adding default wavefunction (assuming MO basis) ");

    detail::psi_from_occ_vector(grp,mf,pt,nspins,occ);        

  } else if(add_wfn == "ph_excited") { 
  
    nda::array<double,3> new_occ(occ);

//    auto nup_tot = io::get_value_with_default<int>(pt,"nup",nup_mf);
//    auto ndn_tot = io::get_value_with_default<int>(pt,"ndown",ndn_mf);

    detail::psi_from_occ_vector(grp,mf,pt,nspins,new_occ);        
  
  } else if(add_wfn!="") {
    APP_ABORT("Error in write_core_hamiltonian: Invalid add_wavefunction:{}",
              add_wfn);
  } // add_wfn

  app_log(2, "*************************************************\n");

}

}

