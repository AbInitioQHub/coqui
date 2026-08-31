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


#ifndef GRIDS_TRUNCATED_G_GRID_HPP
#define GRIDS_TRUNCATED_G_GRID_HPP

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "nda/nda.hpp"
#include "itertools/itertools.hpp"
#include "grids/grid_utils.hpp"

namespace grids
{

/**
 * @class truncated_g_grid
 * @brief Handler for plane-wave grids in fast Fourier Transform (FFT)
 *
 * This class is responsible for the plane-wave (G) grid in FFT.
 * It provides an interface to construct a truncated PW grid based on the
 * provided energy cutoff. This enables customized ecut when evaluating
 * different physical quantities.
 */
class truncated_g_grid {

public:

  /*
   * Constructs an empty grid.
   */ 
  truncated_g_grid():
    ecut_(0.0),
    fft_mesh(0),
    recv(0,0),
    generate_fft2gv(false),
    ngm(0),
    gvecs(ngm,3),
    g2fft(ngm),
    fft2g(0)
  {
  }

  /**
   *
   * @param ecut_ - energy cutoff
   * @param mesh  - FFT grid dimension
   * @param recv_ - reciprocal vectors
   * @param _generate_fft2gv - if true, inverse mapping from fft to gv is constructed. 
   */
  truncated_g_grid(double ecut__,
	  nda::ArrayOfRank<1> auto const& mesh, 
	  nda::ArrayOfRank<2> auto const& recv_,
          bool _generate_fft2gv = false):
    ecut_(ecut__),
    fft_mesh(mesh),
    recv(recv_),
    generate_fft2gv(_generate_fft2gv),
    ngm( get_ngm(ecut_,fft_mesh,recv) ),
    gvecs(ngm,3),
    g2fft(ngm),
    fft2g(0)
  {
    utils::check( ecut_ > 0.0 , "truncated_g_grid:: ecut <= 0.0.");

    long nnr = fft_mesh(0)*fft_mesh(1)*fft_mesh(2);
    utils::check(nnr > 0, "truncated_g_grid:: nnr <= 0.");

    // MAM: do I need to check that ecut lives within fft_mesh???
    long cnt=0;
    long ni = fft_mesh(0)/2;
    long nj = fft_mesh(1)/2;
    long nk = fft_mesh(2)/2;

    if(generate_fft2gv) {
      fft2g = memory::unified_array<long, 1>(nnr);
      fft2g()=-1;
    }

    // MAM: g-vectors not sorted, sort later if needed
    for(long i = (ni-fft_mesh(0)+1); i <= ni; ++i )
    for(long j = (nj-fft_mesh(1)+1); j <= nj; ++j ) 
    for(long k = (nk-fft_mesh(2)+1); k <= nk; ++k ) {
      double gx = double(i)*recv(0,0)+double(j)*recv(1,0)+double(k)*recv(2,0);
      double gy = double(i)*recv(0,1)+double(j)*recv(1,1)+double(k)*recv(2,1);
      double gz = double(i)*recv(0,2)+double(j)*recv(1,2)+double(k)*recv(2,2);
      if( gx*gx+gy*gy+gz*gz <= 2.0*ecut_ ) {
        long ii = (i < 0 ? i+fft_mesh(0) : i); 
        long ij = (j < 0 ? j+fft_mesh(1) : j); 
        long ik = (k < 0 ? k+fft_mesh(2) : k); 
        long N = (ii*fft_mesh(1) + ij)*fft_mesh(2) + ik;
        g2fft(cnt) = N;
        gvecs(cnt, 0) = gx;
        gvecs(cnt, 1) = gy;
        gvecs(cnt, 2) = gz;
        if(generate_fft2gv) fft2g(N) = cnt; 
        cnt++;
      } 
    }
    app_log(4,"\n Generating truncated G-space grid:");
    app_log(4,"   - size: {}\n",ngm);
  }

  /**
   *
   * @param mill - array of miller indices defining the truncated grid. 
   *               Must be within cutoff.
   * @param ecut_ - energy cutoff
   * @param mesh  - FFT grid dimension
   * @param recv_ - reciprocal vectors
   * @param _generate_fft2gv - if true, inverse mapping from fft to gv is constructed. 
   */
  truncated_g_grid(
          nda::ArrayOfRank<2> auto const& mill,
          double ecut__,
          nda::ArrayOfRank<1> auto const& mesh,
          nda::ArrayOfRank<2> auto const& recv_,
          bool _generate_fft2gv = false):
    ecut_(ecut__),
    fft_mesh(mesh),
    recv(recv_),
    generate_fft2gv(_generate_fft2gv),
    ngm( mill.extent(0) ), 
    gvecs(ngm,3),
    g2fft(ngm),
    fft2g(0)
  {
    utils::check( ngm > 0 , "truncated_g_grid:: ngm == 0");
    utils::check( ecut_ > 0.0 , "truncated_g_grid:: ecut <= 0.0.");
    
    long nnr = fft_mesh(0)*fft_mesh(1)*fft_mesh(2);
    utils::check(nnr > 0, "truncated_g_grid:: nnr <= 0.");
    
    if(generate_fft2gv) {
      fft2g = memory::unified_array<long, 1>(nnr);
      fft2g()=-1;
    }
    
    // MAM: g-vectors not sorted, sort later if needed
    for( long p=0; p<ngm; ++p ) {
      long i = mill(p,0);
      long j = mill(p,1);
      long k = mill(p,2);
      double gx = double(i)*recv(0,0)+double(j)*recv(1,0)+double(k)*recv(2,0);
      double gy = double(i)*recv(0,1)+double(j)*recv(1,1)+double(k)*recv(2,1);
      double gz = double(i)*recv(0,2)+double(j)*recv(1,2)+double(k)*recv(2,2);
      utils::check( gx*gx+gy*gy+gz*gz <= 2.0*ecut_, 
               "truncated_g_grid: Point outside ecut: {}, ecut:{}",0.5*gx*gx+gy*gy+gz*gz,ecut_);
      long ii = (i < 0 ? i+fft_mesh(0) : i); 
      long ij = (j < 0 ? j+fft_mesh(1) : j); 
      long ik = (k < 0 ? k+fft_mesh(2) : k); 
      long N = (ii*fft_mesh(1) + ij)*fft_mesh(2) + ik;
      g2fft(p) = N; 
      gvecs(p, 0) = gx; 
      gvecs(p, 1) = gy; 
      gvecs(p, 2) = gz; 
      if(generate_fft2gv) fft2g(N) = p;
    }
    app_log(2,"\n Generating truncated G-space grid:");
    app_log(2,"   - size: {}\n",ngm);

  }

  ~truncated_g_grid() = default;
  truncated_g_grid(truncated_g_grid const& ) = default;
  truncated_g_grid(truncated_g_grid && ) = default;
  truncated_g_grid& operator=(truncated_g_grid const&) = default;
  truncated_g_grid& operator=(truncated_g_grid &&) = default;

  long size() const { 
    utils::check(ngm==gvecs.shape()[0],"Size mismatch."); 
    return ngm; 
  }
  long nnr() const { return fft_mesh(0)*fft_mesh(1)*fft_mesh(2); }
  auto g_vectors(long i) const { return gvecs(i,nda::range::all); };
  memory::unified_array<RealType, 2> const& g_vectors() const { 
    return gvecs; 
  };
  memory::unified_array<long,1> const& gv_to_fft() const { return g2fft; }
  long gv_to_fft(long i) const { return g2fft(i); }
  bool has_fft_to_gv() const { return generate_fft2gv; }
  memory::unified_array<long,1> const& fft_to_gv() const { 
    utils::check(generate_fft2gv,"Error: generate_fft2gv==false in call to fft_to_gv()"); 
    return fft2g; 
  }
  long fft_to_gv(long i) const { 
    utils::check(generate_fft2gv,"Error: generate_fft2gv==false in call to fft_to_gv(i)"); 
    return fft2g(i); 
  }

  double ecut() const { return ecut_; }
  long mesh(long i) const {return fft_mesh(i);}
  auto const& mesh() const {return fft_mesh; }
  auto const& reciprocal_vectors() const {return recv;}

  // add begin/end with enumerate+zip 

private:

  // energy cutoff 
  double ecut_ = 0.0; 

  // fft grid    
  nda::stack_array<long, 3> fft_mesh;

  // reciprocal vectors; recv(0), recv(1), recv(2) = b1, b2, b3
  nda::stack_array<double, 3, 3> recv;

  // controls the generation og fft2gv
  bool generate_fft2gv = false;

  // size of the reciprocal lattice (G)
  long ngm = 0;

  // G vectors inside ecut: (ngm, 3)
  memory::unified_array<RealType, 2> gvecs;  

  // map between gvecs and FFT grid
  memory::unified_array<long, 1> g2fft;

  // map between gvecs and FFT grid
  memory::unified_array<long, 1> fft2g;

};

/* 
 * generates mapping between truncated grids 
 * Aborts if any point can't be mapped (not present in second truncated grid). 
 * If full == true, maps to the full fft grid of g_out. Otherwise maps to truncated grid.
 */
void map_truncated_grids(bool full, truncated_g_grid const& g_in, truncated_g_grid const& g_out,
                         nda::ArrayOfRank<1> auto && m)
{
  utils::check(m.extent(0) == g_in.size(),"Shape mismatch");
  utils::check(nda::frobenius_norm(g_in.reciprocal_vectors() - g_out.reciprocal_vectors()) < 1e-6,
               "map_truncated_grids: recv mismatch");
  utils::check( g_out.mesh()[0] >= g_in.mesh()[0] and
                g_out.mesh()[1] >= g_in.mesh()[1] and
                g_out.mesh()[2] >= g_in.mesh()[2], 
               "Error in map_truncated_grids: g_in.fft_mesh > g_out.fft_mesh.");
  long NX = g_in.mesh(0), NY = g_in.mesh(1), NZ = g_in.mesh(2);
  long MX = g_out.mesh(0), MY = g_out.mesh(1), MZ = g_out.mesh(2);
  long NX2 = NX/2, NY2 = NY/2, NZ2 = NZ/2;
  long MX2 = MX/2, MY2 = MY/2, MZ2 = MZ/2;

  // need to build g_out.fft_to_gv
  memory::unified_array<long, 1> vec(0);
  if(not full and not g_out.has_fft_to_gv()) {
    vec = memory::unified_array<long, 1>(MX*MY*MZ);
    vec() = -1;
    for( auto [p,M] : itertools::enumerate(g_out.gv_to_fft()) ) 
      vec(M)=p;
  }
  memory::unified_array<long, 1> const& fft_to_gv(g_out.has_fft_to_gv()?g_out.fft_to_gv():vec);
  for( auto [p,N] : itertools::enumerate(g_in.gv_to_fft()) ) {

    long k = N%NZ; if( k > NZ2 ) k -= NZ;
    utils::check(std::abs(k) <= MZ2, "map_truncated_grids: Index out of bounds");
    if( k < 0 ) k += MZ;

    long N_ = N/NZ;
    long j = N_%NY; if( j > NY2 ) j -= NY;
    utils::check(std::abs(j) <= MY2, "map_truncated_grids: Index out of bounds");
    if( j < 0 ) j += MY;

    long i = N_/NY; if( i > NX2 ) i -= NX;
    utils::check(std::abs(i) <= MX2, "map_truncated_grids: Index out of bounds");
    if( i < 0 ) i += MX;

    long M = (i*MY + j)*MZ + k;
    m(p) = (full?M:fft_to_gv(M));
    utils::check(m(p) >= 0, "map_truncated_grids: Incompatible truncated grids");

  } 
}

inline auto map_truncated_grids(bool full, 
                                truncated_g_grid const& g_in, 
                                truncated_g_grid const& g_out)
{
  nda::array<long,1> m(g_in.size());   
  map_truncated_grids(full,g_in,g_out,m);
  return m;
}

/**
 * Generates mapping between a truncated grid and the fft grid of a different mesh size
 * @param g_in - [INPUT] Input truncated g-grid
 * @param mesh - [INPUT] Output fft grid
 * @param m - [OUTPUT] Mapping from "g_in" to "mesh"
 */
void map_truncated_grid_to_fft_grid(truncated_g_grid const& g_in,
                         nda::ArrayOfRank<1> auto const& mesh,
                         nda::ArrayOfRank<1> auto && m)
{
  utils::check(m.extent(0) == g_in.size(),"Shape mismatch");
  utils::check( mesh(0) >= g_in.mesh()[0] and
                mesh(1) >= g_in.mesh()[1] and
                mesh(2) >= g_in.mesh()[2],
               "Error in map_truncated_grid_to_fft_grid: g_in.fft_mesh > fft_mesh.");
  long NX = g_in.mesh(0), NY = g_in.mesh(1), NZ = g_in.mesh(2);
  long MX = mesh(0), MY = mesh(1), MZ = mesh(2);
  long NX2 = NX/2, NY2 = NY/2, NZ2 = NZ/2;
  long MX2 = MX/2, MY2 = MY/2, MZ2 = MZ/2;

  for( auto [p,N] : itertools::enumerate(g_in.gv_to_fft()) ) {

    long k = N%NZ; if( k > NZ2 ) k -= NZ;
    utils::check(std::abs(k) <= MZ2, "map_truncated_grid_to_fft_grid: Index out of bounds");
    if( k < 0 ) k += MZ;

    long N_ = N/NZ;
    long j = N_%NY; if( j > NY2 ) j -= NY;
    utils::check(std::abs(j) <= MY2, "map_truncated_grid_to_fft_grid: Index out of bounds");
    if( j < 0 ) j += MY;

    long i = N_/NY; if( i > NX2 ) i -= NX;
    utils::check(std::abs(i) <= MX2, "map_truncated_grid_to_fft_grid: Index out of bounds");
    if( i < 0 ) i += MX;

    m(p) = (i*MY + j)*MZ + k;
    utils::check(m(p) >= 0, "map_truncated_grids: Incompatible truncated grids");

  }
}

} // grids

#endif

