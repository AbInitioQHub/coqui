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


#ifndef UTILITIES_OCCUPATIONS_HPP
#define UTILITIES_OCCUPATIONS_HPP

#include <algorithm> 
#include <ranges>
#include <vector>
#include <iterator>
#include <functional>
#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"

#include "nda/nda.hpp"

namespace utils
{

namespace detail
{

// for each spin, calculates the number of fully and partially occupied states
// assumes/verifies that occupations are sorted 
auto split_occupation_spaces(int nbnd, nda::ArrayOfRank<2> auto const& occ, double upperc = 0.95, double lowerc = 0.05)
{
  decltype(nda::range::all) all;
  long nk = occ.extent(0);
  std::vector<int> Nf;
  std::vector<int> Np;
  for( auto ik : nda::range(nk) ) {
    auto occ_k = occ(ik,all);
    utils::check(std::is_sorted(occ_k.begin(),occ_k.end(),[](auto && a, auto && b){return a>b;}), 
                 "Error: Unsorted occupations in split_occupation_spaces.");
    auto it1 = std::find_if(occ_k.begin(),occ_k.end(),[&](auto && a){return a < upperc;});  
    auto it2 = std::find_if(it1,occ_k.end(),[&](auto && a){return a < lowerc;});  
    long n1 = std::distance(occ_k.begin(),it1);
    long n2 = std::distance(it1,it2);
    for(int ib=0; ib<n1; ib++)
      Nf.emplace_back(ib+ik*nbnd);
    for(int ib=n1; ib<n1+n2; ib++)
      Np.emplace_back(ib+ik*nbnd);
  }
  return std::make_tuple(Nf,Np);
} 

// standard recursive algorithm to find all rank-k combinations
void find_combinations_recursive(std::vector<int> const& set, int k, int startIndex,
                               std::vector<int>& current_combination,
                               std::vector<std::vector<int>>& all_combinations) {
  // Base case: if the current combination has k elements, add it to the results
  if (current_combination.size() == k) {
      all_combinations.push_back(current_combination);
      return;
  }

  // Base case: if we have exhausted all elements or cannot form a k-combination
  if (startIndex == set.size()) {
      return;
  }

  // Include the current element
  current_combination.push_back(set[startIndex]);
  find_combinations_recursive(set, k, startIndex + 1, current_combination, all_combinations);
  current_combination.pop_back(); // Backtrack

  // Exclude the current element
  find_combinations_recursive(set, k, startIndex + 1, current_combination, all_combinations);
}

auto find_combinations(std::vector<int> const& r, int k)
{
  std::vector<std::vector<int>> all_combinations;
  std::vector<int> current_combination;
  find_combinations_recursive(r,k,0,current_combination,all_combinations);
  return all_combinations;
}

}

// MAM: needs to be generalized to nelec !/ sum(occ)
// confg: vector of nda::array<int,1>,
// confg.size()=actual number of determinants,
// confg[n](i): ith occupied orbital in nth configuration.
// confg[n].size(): nup_mf+ndn_mf, ordered by spin. USing compact indexing: index(ik,ib) = nbnd*ik+ib
// indexes within each spin will be sorted
auto dets_from_occupation_vector(int nbnd, int maxdet, int nspin, int nup, int ndn, nda::ArrayOfRank<3> auto const& occ, double upperc = 0.95, double lowerc = 0.05, int det_limit = 100000)
{
  int nkpts = occ.extent(1);
  std::vector<nda::array<int,1>> ret;
  if(nspin==1) {
    // seniority zero wfn, only doubly occupied configurations. Restricted to closed shell occ
    utils::check(occ.extent(0) == 1, "dets_from_occupation_vector: Invalid occ dimension with nspin==1. occ:{}",occ.extent(0));
    auto occ_ = occ(0,nda::ellipsis{});
    {
      double nsum = std::accumulate(occ_.begin(), occ_.end(), double(0.0)); 
      utils::check( std::abs(nsum-nup) < 1e-4, "Problems with nup in default_occupation: nup:{}, nsum:{}",nup,nsum); 
    }
    auto [Nf,Np] = detail::split_occupation_spaces(nbnd,occ_,upperc,lowerc);
    if(Np.size() > 0) {
      app_log(2, " dets_from_occupation_vector - Found {} partially filled states. Constructing up to {} determinants.",Np.size(),maxdet);
      int nel = nup-Nf.size(); 
      auto cfg = detail::find_combinations(Np, nel);
      // to prevent this from going crazy, which it can very easily do with some smearing 
      utils::check(cfg.size() < det_limit, "dets_from_occupation_vector: Found too many configurations:{}, adjust upper_cutoff/lower_cutoff.",cfg.size());
      app_log(2, " dets_from_occupation_vector - Found {} possible determinants configurations.",cfg.size());
      // keep index to avoid having to swap vectors during sort
      std::vector<std::pair<double,int>> weights(cfg.size());
      for(int ic=0; ic<cfg.size(); ++ic) {
        weights[ic].first = 1.0;
        weights[ic].second = ic;
        for(auto& ip: cfg[ic]) {
          int ik=ip/nkpts;
          int ib=ip-ik*nkpts;
          weights[ic].first *= occ(0,ik,ib);
          // do I want to put some constrains here? e.g. minimize momentum? Do it here...
        } 
      }
      // sort weights in descending order
      std::ranges::sort( weights, [](auto const& a, auto const& b) { return std::get<0>(a) > std::get<0>(b); } );
      // pick the largest weights 
      for( int ic=0; ic<std::min(maxdet,int(cfg.size())); ++ic) {
        int n = weights[ic].second; 
        ret.emplace_back(nda::array<int,1>(nup,0));
        auto det = ret[ic](); 
        auto output_it = std::copy(Nf.begin(),Nf.end(),det.begin());
        std::copy(cfg[n].begin(),cfg[n].end(),output_it);
        std::sort(det.begin(),det.end());
      }
    } else {
      utils::check(Nf.size()==nup, "dets_from_occupation_vector: Invalid dimensions. Contact developers." );
      ret.emplace_back(nda::array<int,1>(nup));
      std::copy(Nf.begin(),Nf.end(),ret[0].begin());
    }
  } else if(nspin==2) {
    utils::check((occ.extent(0) == 1) or (occ.extent(0) == 2), "dets_from_occupation_vector: Invalid occ dimension with nspin==2. occ:{}",occ.extent(0));
    auto occ_up = occ(0,nda::ellipsis{});
    auto occ_dn = occ(occ.extent(0)-1,nda::ellipsis{});
    {
      double nsum_up = std::accumulate(occ_up.begin(), occ_up.end(), double(0.0)); 
      double nsum_dn = std::accumulate(occ_dn.begin(), occ_dn.end(), double(0.0)); 
      utils::check( std::abs(nsum_up-nup) < 1e-4, "Problems with nup in default_occupation: nup:{}, nsum:{}",nup,nsum_up);
      utils::check( std::abs(nsum_dn-ndn) < 1e-4, "Problems with ndown in default_occupation: ndown:{}, nsum:{}",ndn,nsum_dn);
    }
    auto [Nf_up,Np_up] = detail::split_occupation_spaces(nbnd,occ_up,upperc,lowerc);
    auto [Nf_dn,Np_dn] = detail::split_occupation_spaces(nbnd,occ_dn,upperc,lowerc);
    if(Np_up.size()+Np_dn.size() > 0) {
      app_log(2, " dets_from_occupation_vector - Found {} partially filled states. Constructing up to {} determinants.",
                 Np_up.size()+Np_dn.size(),maxdet);
      // fixed spin polarization, can also write a version which allows different spin polarizations. Not yet available in AFQMC!!!
      auto cfg_up = detail::find_combinations(Np_up, nup-Nf_up.size());
      auto cfg_dn = detail::find_combinations(Np_dn, ndn-Nf_dn.size());
      // to prevent this from going crazy, which it can very easily do with some smearing 
      utils::check(cfg_up.size()*cfg_dn.size() < det_limit, "dets_from_occupation_vector: Found too many configurations:{}, adjust upper_cutoff/lower_cutoff.",cfg_up.size()*cfg_dn.size());
      app_log(2, " dets_from_occupation_vector - Found {} possible determinants configurations.",cfg_up.size()*cfg_dn.size());
      // keep index to avoid having to swap vectors during sort
      std::vector<std::tuple<double,int,int>> weights(cfg_up.size()*cfg_dn.size());
      for(int icu=0, ic=0; icu<cfg_up.size(); ++icu) {
        double wup = 1.0;
        for(auto& ip: cfg_up[icu]) {
          int ik=ip/nkpts;
          int ib=ip-ik*nkpts;
          wup *= occ_up(ik,ib);
        }        
        for(int icd=0; icd<cfg_dn.size(); ++icd, ++ic) {
          auto& w = std::get<0>(weights[ic]);
          w = wup;
          std::get<1>(weights[ic]) = icu;
          std::get<2>(weights[ic]) = icd;
          for(auto& ip: cfg_dn[icd]) {
            int ik=ip/nkpts;
            int ib=ip-ik*nkpts;
            w *= occ_dn(ik,ib);
          }
        }
      }
      // sort weights in descending order
      std::ranges::sort( weights, [](auto const& a, auto const& b) { return std::get<0>(a) > std::get<0>(b); } );
      // pick the largest weights 
      for( int ic=0; ic<std::min(maxdet,int(weights.size())); ++ic) {
        int icu = std::get<1>(weights[ic]);
        int icd = std::get<2>(weights[ic]);
        ret.emplace_back(nda::array<int,1>(nup+ndn,0));
        auto det = ret[ic]();
        auto it = std::copy(Nf_up.begin(),Nf_up.end(),det.begin());
        it = std::copy(cfg_up[icu].begin(),cfg_up[icu].end(),it);
        it = std::copy(Nf_dn.begin(),Nf_dn.end(),it);
        it = std::copy(cfg_dn[icd].begin(),cfg_dn[icd].end(),it);
        std::sort(det.begin(),det.begin()+nup);
        std::sort(det.begin()+nup,det.end());
      }
    } else {
      utils::check((Nf_up.size()==nup) and (Nf_dn.size()==ndn), 
                   "dets_from_occupation_vector: Invalid dimensions. Contact developers.");
      ret.emplace_back(nda::array<int,1>(nup+ndn));
      auto it = std::copy(Nf_up.begin(),Nf_up.end(),ret[0].begin());
      std::copy(Nf_dn.begin(),Nf_dn.end(),it);
    }
  } else {
    APP_ABORT("dets_from_occupation_vector: invalid npsin:{}",nspin);
  }
  return ret;
}

} // namespace utils

#endif
