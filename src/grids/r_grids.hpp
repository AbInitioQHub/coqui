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


#ifndef GRIDS_R_GRID_HPP
#define GRIDS_R_GRID_HPP

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "nda/nda.hpp"
#include "itertools/itertools.hpp"
#include "numerics/distributed_array/nda.hpp"

namespace grids
{

class r_grid {

public:

  r_grid(nda::ArrayOfRank<1> auto const& mesh, 
	 nda::ArrayOfRank<2> auto const& lattv_):
    fft_mesh(mesh),
    lattv(lattv_),
    nnr(mesh(0)*mesh(1)*mesh(2)),
    rvecs(mesh(0),mesh(1),mesh(2),3)
  {
    utils::check(nnr > 0, "r_grid:: nnr <= 0.");

    for(long i = 0; i < fft_mesh(0); ++i )
    for(long j = 0; j < fft_mesh(1); ++j ) 
    for(long k = 0; k < fft_mesh(2); ++k ) {
      double i_ = double(i)/fft_mesh(0);
      double j_ = double(j)/fft_mesh(1);
      double k_ = double(k)/fft_mesh(2);
      rvecs(i,j,k,0) = double(i_)*lattv(0,0)+double(j_)*lattv(1,0)+double(k_)*lattv(2,0); 
      rvecs(i,j,k,1) = double(i_)*lattv(0,1)+double(j_)*lattv(1,1)+double(k_)*lattv(2,1); 
      rvecs(i,j,k,2) = double(i_)*lattv(0,2)+double(j_)*lattv(1,2)+double(k_)*lattv(2,2); 
    }
    app_log(2,"\n Generating real space grid:");
    app_log(2,"   - size: {}\n",nnr);
  }

  ~r_grid() = default;
  r_grid(r_grid const& ) = default;
  r_grid(r_grid && ) = default;
  r_grid& operator=(r_grid const&) = default;
  r_grid& operator=(r_grid &&) = default;

  long size() const { 
    return nnr; 
  }
  auto r_vector(long n) const { 
    utils::check(n >= 0 and n < nnr, "r_grid::r_vector: Index out of bounds.");
    auto v1D = nda::reshape(rvecs, std::array<long, 2>{nnr, 3});
    return v1D(n,nda::range::all); 
  };
  auto r_vector(long i, long j, long k) const { return rvecs(i,j,k,nda::range::all); };
  nda::array<RealType, 4> const& r_vectors() const { 
    return rvecs; 
  };

  // add begin/end with enumerate+zip 

private:
  
  // fft grid    
  nda::stack_array<long, 3> fft_mesh;

  // lattice vectors
  nda::stack_array<double, 3, 3> lattv;

  long nnr = 0;

  // r vectors 
  nda::array<RealType, 4> rvecs;  

};

class distributed_r_grid {

  using local_Array_t = nda::array<RealType, 4>;

public:

  using dgrid_t = math::nda::distributed_array<local_Array_t,
                               boost::mpi3::communicator>;

  distributed_r_grid(nda::ArrayOfRank<1> auto const& mesh, 
	 nda::ArrayOfRank<2> auto const& lattv_,
	 boost::mpi3::communicator &comm,
	 std::array<long,3> pg):
    fft_mesh(mesh),
    lattv(lattv_),
    nnr(mesh(0)*mesh(1)*mesh(2)),
    rvecs(math::nda::make_distributed_array<local_Array_t>(comm,
				 std::array<long,4>{pg[0],pg[1],pg[2],1},
				 {mesh(0),mesh(1),mesh(2),3})),
    local_nnr(rvecs.local_shape()[0]*rvecs.local_shape()[1]*rvecs.local_shape()[2])
  {
    utils::check(nnr > 0, "distributed_r_grid:: nnr <= 0.");
    utils::check(local_nnr > 0, "distributed_r_grid:: nnr <= 0.");

    auto rv = rvecs.local();
    for(auto [i, in] : itertools::enumerate(rvecs.local_range(0)))  
    for(auto [j, jn] : itertools::enumerate(rvecs.local_range(1)))  
    for(auto [k, kn] : itertools::enumerate(rvecs.local_range(2))) { 
      double i_ = double(in)/fft_mesh(0);
      double j_ = double(jn)/fft_mesh(1);
      double k_ = double(kn)/fft_mesh(2);
      rv(i,j,k,0) = double(i_)*lattv(0,0)+double(j_)*lattv(1,0)+double(k_)*lattv(2,0); 
      rv(i,j,k,1) = double(i_)*lattv(0,1)+double(j_)*lattv(1,1)+double(k_)*lattv(2,1); 
      rv(i,j,k,2) = double(i_)*lattv(0,2)+double(j_)*lattv(1,2)+double(k_)*lattv(2,2); 
    }

    app_log(2,"\n Generating real space grid:");
    app_log(2,"   - size: {}\n",nnr);
  }

  ~distributed_r_grid() = default;
  distributed_r_grid(distributed_r_grid const& ) = default;
  distributed_r_grid(distributed_r_grid && ) = default;
  distributed_r_grid& operator=(distributed_r_grid const&) = default;
  distributed_r_grid& operator=(distributed_r_grid &&) = default;

  long global_size() const { return nnr; } 
  long local_size() const { return local_nnr; } 
  auto const& dgrid() const { return rvecs; } 

private:
  
  // fft grid    
  nda::stack_array<long, 3> fft_mesh;

  // lattice vectors
  nda::stack_array<double, 3, 3> lattv;

  long nnr = 0;
  // distributed r grid 
  dgrid_t rvecs;

  long local_nnr = 0;

};

} // grids

#endif

