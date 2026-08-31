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


#ifndef UTILITIES_INTERPOLATION_UTILS_HPP
#define UTILITIES_INTERPOLATION_UTILS_HPP

#include <vector>
#include <array>
#include <string>
#include <algorithm>

#include "nda/nda.hpp"

namespace utils {
namespace detail {

void add_to_kpath(std::vector<nda::stack_array<double,3>> & kplist,
                  nda::MemoryArrayOfRank<1> auto && kp0, 
                  nda::MemoryArrayOfRank<1> auto && kp1, 
                  int np, bool add_first)
{
  utils::check(kp0.size() == 3 and kp1.size() == 3, "Size mismatch");
  nda::stack_array<double,3> k(kp0), dk = kp1()-kp0();
  if(add_first) kplist.emplace_back(k); 
  for(int i=1; i<=np; i++) {
    k() = kp0 + dk*double(i)/double(np); 
    kplist.emplace_back(k);
  }
}

auto WS_rgrid_impl(nda::MemoryArrayOfRank<2> auto const& lattv,
              nda::MemoryArrayOfRank<1> auto && mesh, 
              nda::MemoryArrayOfRank<1> auto && rgrid,
              double tol,
              nda::array<long,2> &rp,
              nda::array<long,1> &degen)
{
  auto all = nda::range::all;
  using arr = nda::stack_array<long,3>;
  utils::check(lattv.shape() == std::array<long,2>{3,3}, "Size mismatch.");
  utils::check(mesh.size() == 3, "Size mismatch.");
  utils::check(rgrid.size() == 3, "Size mismatch.");
  nda::stack_array<double,3> r2;
  std::vector<arr> ws; 
  std::vector<long> deg; 
  long N = mesh(0)*mesh(1)*mesh(2);
  long nd = (2*(rgrid(0)+1)+1) * (2*(rgrid(1)+1)+1) * (2*(rgrid(2)+1)+1);
  nda::array<double,1> dist(nd);

  for( auto i : nda::range(-rgrid(0)*mesh(0)-1,rgrid(0)*mesh(0)+2) )
    for( auto j : nda::range(-rgrid(1)*mesh(1)-1,rgrid(1)*mesh(1)+2) )
      for( auto k : nda::range(-rgrid(2)*mesh(2)-1,rgrid(2)*mesh(2)+2) ) {
        long ip=0;
        for( auto ii : nda::range(-rgrid(0)-1,rgrid(0)+2) )
          for( auto jj : nda::range(-rgrid(1)-1,rgrid(1)+2) )
            for( auto kk : nda::range(-rgrid(2)-1,rgrid(2)+2) ) {
              r2() = lattv(0,all) * double(i - mesh(0)*ii) + 
                     lattv(1,all) * double(j - mesh(1)*jj) + 
                     lattv(2,all) * double(k - mesh(2)*kk); 
              dist(ip++) = nda::sum(r2*r2); 
            } 
        double dmin = *std::min_element(dist.begin(),dist.end());
        if( std::abs(dist((nd+1)/2-1) - dmin) < tol) {
          // count degeneracies
          deg.emplace_back(std::count_if(dist.begin(),dist.end(),[&](auto && a) 
            {return std::abs(a-dmin) < tol;}));
          ws.emplace_back(arr{i,j,k}); 
        }
      }

  double sum_degen = 0.0;
  for( auto x : deg ) sum_degen += 1.0/double(x);
  if( std::abs(sum_degen-double(N)) > 1e-8 ) return false;

  utils::check(ws.size() == deg.size(), "WS_rgrid_impl: Size mismatch");

  rp.resize(std::array<long,2>{ws.size(),3});
  for( auto [i,v] : itertools::enumerate(ws) )
    rp(i,all) = v;
  degen.resize(std::array<long,1>{deg.size()});
  std::copy_n(deg.data(),deg.size(),degen.data()); 

  return true; 
}

}

/* 
 * Generates a discrete path of kpoints along the directions specified by pts.
 */ 
template<nda::MemoryArrayOfRank<2> Arr2D>
auto generate_kpath(nda::MemoryArrayOfRank<2> auto const& recv,
                    std::vector<Arr2D> const& pts, 
                    std::vector<std::string> const& id,
                    int np0)
{
  auto all = nda::range::all;
  int nseg = pts.size(); 
  utils::check(pts.size() > 0, "Empty array");
  for(auto& v : pts) 
    utils::check(v.extent(0) == 2 and v.extent(1) == 3, "Size mismatch.");
  utils::check(id.size() == 2*nseg, "Size mismatch.");
  nda::stack_array<double,3> dk;
  dk() = recv(0,all) * (pts[0](1,0)-pts[0](0,0)) +
         recv(1,all) * (pts[0](1,1)-pts[0](0,1)) +
         recv(2,all) * (pts[0](1,2)-pts[0](0,2));
  double vlen0 = std::sqrt(nda::sum(dk*dk)); 

  // using vector since I don't know how many yet
  std::vector<nda::stack_array<double,3>> kplist;
  std::vector<long> kpidx;

  kpidx.emplace_back(0);
  detail::add_to_kpath(kplist,pts[0](0,all),pts[0](1,all),np0,true);
  kpidx.emplace_back(kplist.size());
  for(int i=1; i<nseg; ++i) {
    auto& v = pts[i];
    dk() = recv(0,all) * (v(1,0)-v(0,0)) +
           recv(1,all) * (v(1,1)-v(0,1)) +
           recv(2,all) * (v(1,2)-v(0,2));
    double vlen = std::sqrt(nda::sum(dk*dk)); 
    int np = std::max(1,int(std::round(np0*vlen/vlen0)));
    detail::add_to_kpath(kplist,v(0,all),v(1,all),np, 
              not ((id[2*i]==id[2*i-1]) and (nda::sum(nda::abs(v(0,all)-pts[i-1](1,all))) < 1e-6) )); 
    kpidx.emplace_back(kplist.size());
  }
  nda::array<double,2> res(kplist.size(),3);
  for( auto [i,v] : itertools::enumerate(kplist) ) 
    res(i,all) = v();
  nda::array<long,1> idx(kpidx.size()); 
  std::copy(kpidx.begin(),kpidx.end(),idx.begin());
  app_log(3," Number of points in kpath: {}",res.extent(0));
  return std::make_tuple(res,idx);
}


/*
 * Constructs the list of grid points and their degeneracies in the Wigner-Seitz cell of the supercell.
 */
auto WS_rgrid(nda::MemoryArrayOfRank<2> auto const& lattv,
              nda::MemoryArrayOfRank<1> auto && mesh, 
              double tol = 1e-6) {

  // MAM: not sure how to determine the optimal grid, trying until I find it
  // FIX FIX FIX: This will rurely fail for highly anisotropic grids
 
  long N = mesh(0)*mesh(1)*mesh(2);
  nda::array<long,2> rp;
  nda::array<long,1> rw;
  for(long i=2; i<6; ++i) { 
    nda::stack_array<long,3> rg = {i,i,i};
    if( detail::WS_rgrid_impl(lattv,mesh,rg,tol,rp,rw) ) 
      return std::make_tuple(rw,rp);
  }
  // Failed to generate grid, fall back to trivial choice 
  app_log(2, "  [WARNING] Failed to generate Wigner-Seitz grid. Contact developers."); 
  app_log(2, "            Falling back to uniform R-grid, which can lead to larger interpolation errors."); 
  app_log(2, "            k-mesh:{}",mesh());
  app_log(2, "            lattv:{}",lattv());

  rp.resize(std::array<long,2>{N,3});
  rw.resize(std::array<long,1>{N});
  rw() = 1;
  long nx = mesh(0); 
  long ny = mesh(1); 
  long nz = mesh(2); 
  for (int i=0; i<N; ++i) {
    long a = i / (ny * nz);
    long b = (i / nz) % ny;
    long c = i % nz;
    if (a > nx / 2) a -= nx;
    if (b > ny / 2) b -= ny;
    if (c > nz / 2) c -= nz;
    rp(i, 0) = a;
    rp(i, 1) = b;
    rp(i, 2) = c;
  }
  return std::make_tuple(rw,rp);
}

}

#endif
