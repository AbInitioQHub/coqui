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


#undef NDEBUG

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/proc_grid_partition.hpp"

#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/h5.hpp"
#include "utilities/test_common.hpp"

namespace bdft_tests
{

using namespace math::nda;
template <int Rank> using shape_t = std::array<long, Rank>;

TEST_CASE("distributed_nda", "[math]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  long nx = utils::find_proc_grid_min_diff(world.size(),100,100);
  long ny = world.size()/nx;
  long ix = world.rank()/ny;
  long iy = world.rank()%ny;
  {
    using larray = nda::array<double,3>; 
    using darray = distributed_array<larray,decltype(world)>;
   
    darray A(std::addressof(world),{1,nx,ny}, 
		   {10, 2*nx,2*ny},	// global 
		   {10, 2, 2},		// local
		   {0, 2*ix, 2*iy},     // origin
		   {1, 1, 1});	        // block size
    A.local()=1.0;

    REQUIRE( A.origin() == shape_t<3>{0, 2*ix, 2*iy} );
    REQUIRE( A.global_shape() == shape_t<3>{10, 2*nx, 2*ny} );
    REQUIRE( A.local_shape() == shape_t<3>{10, 2, 2} );
    REQUIRE( A.grid() == shape_t<3>{1, nx, ny} );
    REQUIRE( *A.communicator() == world );

    // copy/move constructors
    {
      auto B{A};
      REQUIRE( A == B );
      auto C{std::move(B)};
      REQUIRE( A == C );
    }

    // copy/move assignment
    {
      auto B{A}; 
      auto C{A}; 
      B = A;
      REQUIRE( A == B );
      C = std::move(B);
      REQUIRE( A == C );
    } 

    darray B(std::addressof(world),{1,nx,ny},
                   {20, 3*nx,3*ny},     // global 
                   {0, 3, 3},           // local
                   {0, 3*ix, 3*iy},     // origin
		   {1, 1, 1});	        // block size
    B.local() = 2.0;
    {
      auto C{B};
      C = A;
      REQUIRE( A == C );
    }
  }

  // view
  {
    using larray = nda::array<double,3>; 
    using darray = distributed_array_view<larray,decltype(world)>;

    larray L(10, 2, 2);
    L() = 0.0;
    
    darray A(std::addressof(world),{1,nx,ny}, 
                   {10, 2*nx,2*ny},     // global 
                   {0, 2*ix, 2*iy},     // origin
		   {1, 1, 1},	        // block size
		   L);
    
    REQUIRE( A.origin() == shape_t<3>{0, 2*ix, 2*iy} );
    REQUIRE( A.global_shape() == shape_t<3>{10, 2*nx, 2*ny} );
    REQUIRE( A.local_shape() == shape_t<3>{10, 2, 2} );
    REQUIRE( A.grid() == shape_t<3>{1, nx, ny} );
    REQUIRE( *A.communicator() == world );
    REQUIRE( A.local() == L );
    
    // copy/move constructors
    { 
      darray B{A}; 
      REQUIRE( A == B );
      darray C{std::move(B)};
      REQUIRE( A == C );
    }
    
    // copy/move assignment
    { 
      darray B{A};
      darray C{A};
      B = A;
      REQUIRE( A == B );
      C = std::move(B);
      REQUIRE( A == C );
    }
    
    larray L2(20, 3, 3);
    L2() = 1.0;
    darray B(std::addressof(world),{1,nx,ny}, 
                   {20, 3*nx,3*ny},     // global 
                   {0, 3*ix, 3*iy},     // origin
		   {1, 1, 1},	        // block size
		   L2);
    { 
      darray C{A};
      C.rebind(B);
      REQUIRE( C == B );
      C.rebind(A);
      REQUIRE( C == A );
    }    
  
  }
}

TEST_CASE("redistribute_nda", "[math]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  long size = world.size();

  {
    using larray = nda::array<double,2>;
    auto A = make_distributed_array<larray>(world,{size,1},{10*size+7,21*size+11});
    auto B = make_distributed_array<larray>(world,{1,size},{10*size+7,21*size+11});
    auto origin = A.origin();
    auto gshape = A.global_shape();

    auto Aloc = A.local();
    for( int i=0; i<Aloc.shape(0); ++i )  
      for( int j=0; j<Aloc.shape(1); ++j )
        Aloc(i,j) = (origin[0]+i)*gshape[1] + (origin[1]+j);  

    redistribute(A,B); 
    
    auto Bloc = B.local();
    origin = B.origin();
    for( int i=0; i<Bloc.shape(0); ++i )  
      for( int j=0; j<Bloc.shape(1); ++j )
        REQUIRE( Bloc(i,j) == double((origin[0]+i)*gshape[1] + (origin[1]+j)) );  
  }
  { 
    using larray = nda::array<double,2>;
    auto A = make_distributed_array<larray>(world,{1,size},{9*size+13,3*size+7});
    auto B = make_distributed_array<larray>(world,{size,1},{9*size+13,3*size+7});
    auto origin = A.origin();
    auto gshape = A.global_shape();
    
    auto Aloc = A.local();
    for( int i=0; i<Aloc.shape(0); ++i )  
      for( int j=0; j<Aloc.shape(1); ++j )
        Aloc(i,j) = (origin[0]+i)*gshape[1] + (origin[1]+j);
    
    redistribute(A,B);
    
    auto Bloc = B.local();
    origin = B.origin();
    for( int i=0; i<Bloc.shape(0); ++i )  
      for( int j=0; j<Bloc.shape(1); ++j )
        REQUIRE( Bloc(i,j) == double((origin[0]+i)*gshape[1] + (origin[1]+j)) );
  }
  { 
    using larray = nda::array<double,2>;
    long nx = 1;
    for(int i=2; i<size/2; ++i) if( size%i == 0 ) nx=i; 
    auto A = make_distributed_array<larray>(world,{nx,size/nx},{10*size+7,21*size+11});
    auto B = make_distributed_array<larray>(world,{size/nx,nx},{10*size+7,21*size+11});
    auto origin = A.origin();
    auto gshape = A.global_shape();
    
    auto Aloc = A.local();
    for( int i=0; i<Aloc.shape(0); ++i )  
      for( int j=0; j<Aloc.shape(1); ++j )
        Aloc(i,j) = (origin[0]+i)*gshape[1] + (origin[1]+j);
    
    redistribute(A,B);
    
    auto Bloc = B.local();
    origin = B.origin();
    for( int i=0; i<Bloc.shape(0); ++i )  
      for( int j=0; j<Bloc.shape(1); ++j )
        REQUIRE( Bloc(i,j) == double((origin[0]+i)*gshape[1] + (origin[1]+j)) );
  }
  {
    using larray = nda::array<double,3>;
    auto A = make_distributed_array<larray>(world,{size,1,1},{10*size+7,21*size+11,8*size+13});
    auto B = make_distributed_array<larray>(world,{1,size,1},{10*size+7,21*size+11,8*size+13});
    auto origin = A.origin();
    auto gshape = A.global_shape();

    auto Aloc = A.local();
    for( int i=0; i<Aloc.shape(0); ++i )  
      for( int j=0; j<Aloc.shape(1); ++j )
        for( int k=0; k<Aloc.shape(2); ++k )
          Aloc(i,j,k) = (origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k);  

    redistribute(A,B); 
    
    auto Bloc = B.local();
    origin = B.origin();
    for( int i=0; i<Bloc.shape(0); ++i )  
      for( int j=0; j<Bloc.shape(1); ++j )
        for( int k=0; k<Bloc.shape(2); ++k )
          REQUIRE( Bloc(i,j,k) == double((origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k)));
  }
  { 
    using larray = nda::array<double,3>;
    auto A = make_distributed_array<larray>(world,{1,1,size},{10*size+7,21*size+11,8*size+13});
    auto B = make_distributed_array<larray>(world,{1,size,1},{10*size+7,21*size+11,8*size+13});
    auto origin = A.origin();
    auto gshape = A.global_shape();
    
    auto Aloc = A.local();
    for( int i=0; i<Aloc.shape(0); ++i )
      for( int j=0; j<Aloc.shape(1); ++j )
        for( int k=0; k<Aloc.shape(2); ++k )
          Aloc(i,j,k) = (origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k);

    redistribute(A,B);

    auto Bloc = B.local();
    origin = B.origin();
    for( int i=0; i<Bloc.shape(0); ++i )
      for( int j=0; j<Bloc.shape(1); ++j )
        for( int k=0; k<Bloc.shape(2); ++k )
          REQUIRE( Bloc(i,j,k) == double((origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k)));
  }
  { 
    using larray = nda::array<double,3>;
    auto A = make_distributed_array<larray>(world,{1,size,1},{10*size+7,21*size+11,8*size+13});
    auto B = make_distributed_array<larray>(world,{size,1,1},{10*size+7,21*size+11,8*size+13});
    auto origin = A.origin();
    auto gshape = A.global_shape();
        
    auto Aloc = A.local();
    for( int i=0; i<Aloc.shape(0); ++i )
      for( int j=0; j<Aloc.shape(1); ++j )
        for( int k=0; k<Aloc.shape(2); ++k )
          Aloc(i,j,k) = (origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k);

    redistribute(A,B);

    auto Bloc = B.local();
    origin = B.origin();
    for( int i=0; i<Bloc.shape(0); ++i )
      for( int j=0; j<Bloc.shape(1); ++j )
        for( int k=0; k<Bloc.shape(2); ++k )
          REQUIRE( Bloc(i,j,k) == double((origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k)));
  }
  { 
    using larray = nda::array<double,3>;
    long nx = 1;
    for(int i=2; i<size/2; ++i) if( size%i == 0 ) nx=i; 
    auto A = make_distributed_array<larray>(world,{nx,1,size/nx},{10*size+7,21*size+11,8*size+13});
    auto B = make_distributed_array<larray>(world,{size/nx,nx,1},{10*size+7,21*size+11,8*size+13});
    auto origin = A.origin();
    auto gshape = A.global_shape();

    auto Aloc = A.local();
    for( int i=0; i<Aloc.shape(0); ++i )
      for( int j=0; j<Aloc.shape(1); ++j )
        for( int k=0; k<Aloc.shape(2); ++k )
          Aloc(i,j,k) = (origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k);

    redistribute(A,B);

    auto Bloc = B.local();
    origin = B.origin();
    for( int i=0; i<Bloc.shape(0); ++i )
      for( int j=0; j<Bloc.shape(1); ++j )
        for( int k=0; k<Bloc.shape(2); ++k )
          REQUIRE( Bloc(i,j,k) == double((origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k)));
  }
  {
    using larray = nda::array<double,3>;
    long nx = 1;
    for(int i=2; i<size/2; ++i) if( size%i == 0 ) nx=i;
    auto A = make_distributed_array<larray>(world,{nx,1,size/nx},{10*size+7,21*size+11,8*size+13});
    auto origin = A.origin();
    auto gshape = A.global_shape();

    auto Aloc = A.local();
    for( int i=0; i<Aloc.shape(0); ++i )
      for( int j=0; j<Aloc.shape(1); ++j )
        for( int k=0; k<Aloc.shape(2); ++k )
          Aloc(i,j,k) = (origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k);

    nda::array<double,3>* L = new nda::array<double,3>(A.global_shape());

    math::nda::gather(0,A,L);
    
    if(world.rank() == 0) {
      for( int i=0; i<L->shape(0); ++i )
        for( int j=0; j<L->shape(1); ++j )
          for( int k=0; k<L->shape(2); ++k )
            REQUIRE( (*L)(i,j,k) == double(i*gshape[1]*gshape[2] + j*gshape[2] + k)); 
    } 

    A.local() = 0.0;
    math::nda::scatter(0,L,A);

    for( int i=0; i<Aloc.shape(0); ++i )
      for( int j=0; j<Aloc.shape(1); ++j )
        for( int k=0; k<Aloc.shape(2); ++k )
          REQUIRE( Aloc(i,j,k) == double((origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k)));

    if(world.size() > 1) {
    
      math::nda::gather(1,A,L);
      if(world.rank() == 0) {
        for( int i=0; i<L->shape(0); ++i )
          for( int j=0; j<L->shape(1); ++j )
            for( int k=0; k<L->shape(2); ++k )
              REQUIRE( (*L)(i,j,k) == double(i*gshape[1]*gshape[2] + j*gshape[2] + k));                                 
      }
      
      A.local() = 0.0;
      math::nda::scatter(1,L,A);

      for( int i=0; i<Aloc.shape(0); ++i )
        for( int j=0; j<Aloc.shape(1); ++j )
          for( int k=0; k<Aloc.shape(2); ++k )
            REQUIRE( Aloc(i,j,k) == double((origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k)));       

    }

  }
  {
    using larray = nda::array<double,3>;
    long nx = 1;
    for(int i=2; i<size/2; ++i) if( size%i == 0 ) nx=i;
    auto A = make_distributed_array<larray>(world,{nx,1,size/nx},{10*size+7,21*size+11,8*size+13});
    auto origin = A.origin();
    auto gshape = A.global_shape();

    auto Aloc = A.local();
    for( int i=0; i<Aloc.shape(0); ++i )
      for( int j=0; j<Aloc.shape(1); ++j )
        for( int k=0; k<Aloc.shape(2); ++k )
          Aloc(i,j,k) = (origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k);

    nda::array<double,2>* L = new nda::array<double,2>(A.global_shape()[1],A.global_shape()[2]);

    math::nda::gather_sub_matrix(3,0,A,L);

    if(world.rank() == 0) {
      for( int j=0; j<L->shape(0); ++j )
        for( int k=0; k<L->shape(1); ++k )
          REQUIRE( (*L)(j,k) == double(3*gshape[1]*gshape[2] + j*gshape[2] + k));
    }

    math::nda::gather_sub_matrix(10*size+6,0,A,L);
    
    if(world.rank() == 0) {
      for( int j=0; j<L->shape(0); ++j )
        for( int k=0; k<L->shape(1); ++k )
          REQUIRE( (*L)(j,k) == double((10*size+6)*gshape[1]*gshape[2] + j*gshape[2] + k));
    }

    if(world.size()>1) {

      math::nda::gather_sub_matrix(3,1,A,L);
    
      if(world.rank() == 1) {
        for( int j=0; j<L->shape(0); ++j )
          for( int k=0; k<L->shape(1); ++k )
            REQUIRE( (*L)(j,k) == double(3*gshape[1]*gshape[2] + j*gshape[2] + k));
      }

      math::nda::gather_sub_matrix(10*size+6,1,A,L);
   
      if(world.rank() == 1) {
        for( int j=0; j<L->shape(0); ++j )
          for( int k=0; k<L->shape(1); ++k )
            REQUIRE( (*L)(j,k) == double((10*size+6)*gshape[1]*gshape[2] + j*gshape[2] + k));
      }

    }
  }
  {
    using larray = nda::array<double,3>;
    long nx = 1;
    for(int i=2; i<size/2; ++i) if( size%i == 0 ) nx=i;
    auto A = make_distributed_array<larray>(world,{nx,1,size/nx},{10*size+7,21*size+11,8*size+13});
    auto origin = A.origin();
    auto gshape = A.global_shape();

    auto Aloc = A.local();
    for( int i=0; i<Aloc.shape(0); ++i )
      for( int j=0; j<Aloc.shape(1); ++j )
        for( int k=0; k<Aloc.shape(2); ++k )
          Aloc(i,j,k) = (origin[0]+i)*gshape[1]*gshape[2] + (origin[1]+j)*gshape[2] + (origin[2]+k);

    {
        auto I_rng = nda::range(3,10);
        auto J_rng = nda::range(10,30);
        auto K_rng = nda::range(1,20);
        
        nda::array<double,3>* L = new nda::array<double,3>(I_rng.size(),J_rng.size(),K_rng.size());
        
        math::nda::gather_ranged(0,A,L,{I_rng, J_rng, K_rng});
        
        if(world.rank() == 0) {
          for( int i=0; i<L->shape(0); ++i )
            for( int j=0; j<L->shape(1); ++j )
              for( int k=0; k<L->shape(2); ++k )
                REQUIRE( (*L)(i,j,k) == (I_rng.first()+i)*gshape[1]*gshape[2] + (J_rng.first()+j)*gshape[2] + (K_rng.first()+k));
        }
        delete L;
    }
    {
        auto I_rng = nda::range(3*size,4*size+5);
        auto J_rng = nda::range(10*size,18*size+8);
        auto K_rng = nda::range(3*size+2,7*size+13);
        
        nda::array<double,3>* L = new nda::array<double,3>(I_rng.size(),J_rng.size(),K_rng.size());
        
        math::nda::gather_ranged(0,A,L,{I_rng, J_rng, K_rng});
        
        if(world.rank() == 0) {
          for( int i=0; i<L->shape(0); ++i )
            for( int j=0; j<L->shape(1); ++j )
              for( int k=0; k<L->shape(2); ++k )
                REQUIRE( (*L)(i,j,k) == (I_rng.first()+i)*gshape[1]*gshape[2] + (J_rng.first()+j)*gshape[2] + (K_rng.first()+k));
        }
        delete L;
    }


    if(world.size()>1){
        auto I_rng = nda::range(3,10);
        auto J_rng = nda::range(10,30);
        auto K_rng = nda::range(1,20);
        
        nda::array<double,3>* L = new nda::array<double,3>(I_rng.size(),J_rng.size(),K_rng.size());
        
        math::nda::gather_ranged(1,A,L,{I_rng, J_rng, K_rng});
        
        if(world.rank() == 1) {
          for( int i=0; i<L->shape(0); ++i )
            for( int j=0; j<L->shape(1); ++j )
              for( int k=0; k<L->shape(2); ++k ) {
                REQUIRE( (*L)(i,j,k) == (I_rng.first()+i)*gshape[1]*gshape[2] + (J_rng.first()+j)*gshape[2] + (K_rng.first()+k));
              }
        }
        delete L;
    }

    if(world.size()>1){
        auto I_rng = nda::range(3*size,4*size+5);
        auto J_rng = nda::range(10*size,18*size+8);
        auto K_rng = nda::range(3*size+2,7*size+13);
        
        nda::array<double,3>* L = new nda::array<double,3>(I_rng.size(),J_rng.size(),K_rng.size());
        
        math::nda::gather_ranged(1,A,L,{I_rng, J_rng, K_rng});
        
        if(world.rank() == 1) {
          for( int i=0; i<L->shape(0); ++i )
            for( int j=0; j<L->shape(1); ++j )
              for( int k=0; k<L->shape(2); ++k )
                REQUIRE( (*L)(i,j,k) == (I_rng.first()+i)*gshape[1]*gshape[2] + (J_rng.first()+j)*gshape[2] + (K_rng.first()+k));
        }
        delete L;
    }

  }
}

} // bdft_tests
