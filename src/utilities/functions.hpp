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


#ifndef UTILITIES_FUNCTIONS_HPP
#define UTILITIES_FUNCTIONS_HPP

#include "utilities/check.hpp"
#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "itertools/itertools.hpp"
#include "numerics/nda_functions.hpp"

namespace utils
{

template<nda::ArrayOfRank<1> Ain, 
	 nda::ArrayOfRank<1> Avals,
	 nda::ArrayOfRank<1> Aindx>
void max_element_multi(Ain const& a, 
		       Avals && maxs, 
		       Aindx && indx) 
{
  int N = std::min(maxs.size(),a.size());
  utils::check(N > 0, "max_element_multi - size mismatch: {} > {}",N,0);
  utils::check(maxs.size() >= N, "indx.size() - size mismatch: {} >= {}", maxs.size(),N);
  utils::check(indx.size() >= N, "indx.size() - size mismatch: {} >= {}", indx.size(),N);
  static_assert(nda::mem::on_host<Avals,Aindx>, "Memory location mismatch");
  static_assert(not nda::is_complex_v<typename std::decay_t<Avals>::value_type>,"Type mismatch.");
  if constexpr (nda::mem::on_host<Ain>) {
    maxs(nda::range(0,N)) = nda::real(a(nda::range(0,N)));
    for(int i=0; i<N; ++i) indx(i)=i;
    if(a.size()<=N) {
      if(a.size()<N) std::fill(indx.begin()+N,indx.end(),-1);
      return;
    } 
    auto it = std::min_element(maxs.begin(),maxs.end()); 
    for( auto [n,v]: itertools::enumerate(a) ) 
      if( std::real(v) > *it ) {
        *it = std::real(v);
        indx(it.indices()[0]) = n; 
        it = std::min_element(maxs.begin(),maxs.end());
      }
  } else {
    // temporary hack
    auto a_h = nda::to_host(a);
    max_element_multi(a_h,std::forward<Avals>(maxs),std::forward<Aindx>(indx)); 
  }
}

// find N maximum values from a distributed list, 
// returns tuple which includes the value and the position of the element in the 
// global list. This position (e.g. p) encodes the processor who owns the element (p/N)
// and its position in the local list (p%N)
template<nda::ArrayOfRank<1> Avals, nda::ArrayOfRank<1> Apairs, typename communicator>
void find_distributed_maximum(communicator & gcomm,
			      Avals const& l,
                              Apairs & g)
requires( std::is_same_v<typename std::decay_t<Apairs>::value_type,
			 std::pair<typename std::decay_t<Avals>::value_type,int>> )
{ 
// not aware of any good way to do this, so collect values in root and bcast result
// right now assumes that sizes are consistent among all mpi tasks 
// to avoid the extra communication 
  using T = typename std::decay_t<Avals>::value_type;
  int N = l.size();    
  int np = gcomm.size();
  utils::check(g.size()==N, "find_distributed_maximum - size mismatch: {} == {}",g.size(),N);
  nda::array<T, 1> glist((gcomm.rank()==0?N*np:0));
  gcomm.gather_n(l.data(),N,glist.data(),0);
  if(gcomm.rank()==0) {
    nda::array<T, 1> vals(N);
    nda::array<int, 1> indx(N);
    utils::max_element_multi(glist,vals,indx);
    for(int i=0; i<N; i++) {
      g(i).first  = vals(i);
      g(i).second = indx(i);
    }
  }
  gcomm.broadcast_n(g.data(),N,0);
}

// same as above but over multiple sets of data
template<typename T, typename communicator>
void find_distributed_maximum(communicator & gcomm,
                              nda::array<T,2> const& l,
                              nda::array<std::pair<T,int>,2> & g)
{
// MAM: do this right
  decltype(nda::range::all) all;
  int Np = l.shape(0); // number of sets
  int N = l.shape(1);  // number of elements per set
  int np = gcomm.size();
  utils::check(g.shape()==l.shape(), "find_distributed_maximum - size mismatch");
  // temporary implementation
  nda::array<T, 1> glist((gcomm.rank()==0?N*np:0));
  nda::array<T, 1> vals((gcomm.rank()==0?N:0));
  nda::array<int, 1> indx((gcomm.rank()==0?N:0));
  for(int p=0; p<Np; ++p) {
    gcomm.gather_n(l(p,all).data(),N,glist.data(),0);
    if(gcomm.rank()==0) {
      utils::max_element_multi(glist,vals,indx);
      for(int i=0; i<N; i++) {
        g(p,i).first  = vals(i);
        g(p,i).second = indx(i);
      }
    }
  }
  gcomm.broadcast_n(g.data(),g.size(),0);     
/*
  nda::array<T, 3> glist((gcomm.rank()==0?np:0),(gcomm.rank()==0?Np:0),(gcomm.rank()==0?N:0));
  nda::array<T, 1> llist((gcomm.rank()==0?np*N:0));
  gcomm.gather_n(l.data(),Np*N,glist.data(),0);
  if(gcomm.rank()==0) {
    for(int p=0; p<Np; ++p) {
      nda::array<T, 1> vals(N);
      nda::array<int, 1> indx(N);
      utils::max_element_multi(glist,vals,indx);
      for(int i=0; i<N; i++) {
        g(i).first  = vals(i);
        g(i).second = indx(i);
      }
    }
  }
  gcomm.broadcast_n(g.data(),N,0);
*/
}

// looks for the factor of M closese to f0
inline long find_nearest_factor(long M, long f0)
{
  for(int i=0; i<M; ++i) {
    long fi = f0+i;
    if( fi <= M and M%fi==0 ) return fi;
    fi = f0-i;
    if( fi > 0 and M%fi==0 ) return fi;
  } 
  // no factors found, prime?
  if( f0 < M/2l ) return long(1);
  else return M;
}

inline auto data_binning(nda::ArrayOfRank<1> auto&& Y, int nbins = 100) 
{
  auto X = nda::array<double,2>::zeros({nbins, 2}); 
  int N = Y.size();
  double x0 = *std::min_element(Y.begin(),Y.end());
  double x1 = *std::max_element(Y.begin(),Y.end());
  double dx = (x1-x0)/nbins;
  double dN = 1.0/double(N);
  for( int i=0; i<nbins; ++i )
    X(i,0) = x0+i*dx;  
  for( auto& v : Y ) {
    int r = std::max(std::min(nbins-1,int( std::floor((v-x0)/dx) )), 0);
    X(r,1) += dN;
  }
  return X;
}

/*
 *
 *  full_factorization = true return a potentially rectangular matrix such that L*dagger(L) ~= A
 *                     = false returns a square matrix where L*dagger(L) ~= A({piv(:)},{piv(:)}) 
 *  transpose = true returns A ~= dagger(L)*L
 *            = false return A ~= L*dagger(L) 
 *
 */
template<bool full_factorization, 
         nda::ArrayOfRank<2> return_Arr = nda::array<ComplexType,2>, 
         nda::ArrayOfRank<2> Arr = nda::array<ComplexType,2> >
auto chol(Arr& A, nda::array<int,1>& piv, double cut = 1e-8, bool transpose = false)
{
  int n = A.shape()[0];
  utils::check(piv.size() >= n+1,"Size mismatch: {},{} ",piv.size(),n+1);

  if constexpr(full_factorization) {

    int nc=0;
    // computes A ~= L*dagger(L) by default, apply dagger later if needed
    for(int i=0; i<n; ++i) {
      double v = std::real(A(i,i));
      for(int j=0; j<nc; ++j)
        v -= std::real(A(i,piv(j))*std::conj(A(i,piv(j))));
      if(std::abs(v) > cut) {
        piv(nc)=i;
        v = std::sqrt(v);
        for(int j=0; j<n; ++j) {
          // if j is in current list of pivots, A(j,i) could just be set to zero
          for(int k=0; k<nc; ++k)
            A(j,i) -= A(j,piv(k))*std::conj(A(i,piv(k)));
          A(j,i) = A(j,i) / v;
        }
        nc++;
      }
    }

    piv(n)=nc;
    if(transpose) {
      return_Arr W(nc,n);
      for(int i=0; i<n; i++) {
        for(int p=0; p<nc; p++)
          W(p,i) = std::conj(A(i,piv(p)));
      }
      return W;
    } else {
      return_Arr W(n,nc);
      for(int i=0; i<n; i++) {
        for(int p=0; p<nc; p++)
          W(i,p) = A(i,piv(p));
      }
      return W;
    }     

  } else {

    int nc=0;
    // computes A ~= L*dagger(L) by default, apply dagger later if needed
    for(int i=0; i<n; ++i) { 
      for(int j=0; j<i; ++j) A(j,i) = ComplexType(0.0);
      double v = std::real(A(i,i));
      for(int j=0; j<nc; ++j)
        v -= std::real(A(i,piv(j))*std::conj(A(i,piv(j))));
      if(std::abs(v) > cut) {
        piv(nc)=i;
        v = std::sqrt(v);
        A(i,i) = v;
        for(int j=i+1; j<n; ++j) {
          for(int k=0; k<nc; ++k)
            A(j,i) -= A(j,piv(k))*std::conj(A(i,piv(k)));
          A(j,i) = A(j,i) / v;
        }
        nc++;
      }
    }

    if(nc==0) { // choose largest
      int k=0;
      double v = std::real(A(0,0));
      for(int i=1; i<n; ++i)
        if(std::real(A(i,i)) > v) {
          k=i;
          v = std::real(A(i,i));
        }
      piv(nc)=k;
      A(k,k) = v;
      for(int j=k+1; j<n; ++j)
        A(j,k) = A(j,k) / v;
      nc++;
    }
    piv(n)=nc;
    return_Arr W(nc,nc);
    if(transpose) {
      for(int i=0; i<nc; i++) {
        for(int p=0; p<nc; p++)
          W(p,i) = std::conj(A(piv(i),piv(p)));
      }
    } else {
      for(int i=0; i<nc; i++) {
        for(int p=0; p<nc; p++)
          W(i,p) = A(piv(i),piv(p));
      }
    }
    return W;

  }
}

} // utils

#endif
