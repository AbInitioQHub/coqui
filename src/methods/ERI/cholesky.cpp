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


#//include <tuple>
#include <iomanip>
#include <algorithm>
#include <fstream>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "arch/arch.h"

#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/tensor.hpp"
#include "h5/h5.hpp"
#include "nda/h5.hpp"
#include "itertools/itertools.hpp"
#include "numerics/fft/nda.hpp"
#include "grids/g_grids.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/h5.hpp"
#include "utilities/proc_grid_partition.hpp"
#include "utilities/functions.hpp"
#include "numerics/nda_functions.hpp"

#include "nda/linalg/det_and_inverse.hpp"

#include "methods/ERI/cholesky.h"

namespace methods
{

// todo:
//   - no k-dependent norb yet!!!
//   - improve handling of orbital I/O and temporary storage

cholesky::cholesky(mf::MF *mf_,
           utils::mpi_context_t<mpi3::communicator> &mpi_, 
           ptree const& pt
          ) :
    mpi(std::addressof(mpi_)),
    mf(mf_),
    Timer(),
    npools(utils::find_proc_grid_max_rows(mpi->comm,mf->nkpts())),
    cutoff( io::get_value_with_default<double>(pt,"tol",1e-10) ),
    default_block_size( io::get_value_with_default<int>(pt,"chol_block_size",32) ),
    ecut( io::get_value_with_default<double>(pt,"ecut",mf->ecutrho()) ),
    rho_g(ecut,mf->fft_grid_dim(),mf->recv()),
    vG( io::check_child_exists(pt,"potential") ? io::find_child(pt,"potential") : ptree{} ),
    howmany_fft(-1)
{
  utils::check(mf != nullptr, "cholesky::Null pointer.");
  utils::check(mf->has_orbital_set(), "Error in cholesky: Invalid mf type. ");

  // maximize size of pool communicator for now, change if needed later
  utils::check( npools > 0 and npools <= mf->nkpts(), "Error: npools > nkpts");
  utils::check( mpi->comm.size()%npools == 0, "Oh-oh, bug.");

  app_log(1,"*******************************");
  app_log(1," ERI::cholesky: ");
  app_log(1,"*******************************");
  app_log(1,"  -pw cutoff (Ha): {}",ecut);
  app_log(1,"  -size of PW basis: {}",rho_g.size());
  app_log(1,"  -cholesky truncation: {}",cutoff);
  app_log(1,"  -number of k-point pools: {}",npools);
  app_log(1,"  -number of processors per pools: {}",mpi->comm.size()/npools);
  app_log(1,"  -default block size: {}",default_block_size);
  app_log(1,"\n");

  for( auto& v: {"TOTAL","IO","ALLOC","COMM","FFT","FFTPLAN",
                 "Pab","DIAG","ITER","ERI","SERIAL"} )
    Timer.add(v);

}

template<MEMORY_SPACE MEM>
auto cholesky::evaluate(int Qi, nda::range a_range, nda::range b_range, 
			bool diag_kk, int block_size)
	-> dArray_t<memory::array<MEM,ComplexType,5>>
{
  if(block_size < 0) block_size = default_block_size;
  if(a_range.first() < 0 and a_range.last() < 0) a_range = nda::range(mf->nbnd());
  if(a_range.first() < 0) a_range = nda::range(0,a_range.last());
  if(a_range.last() < 0) a_range = nda::range(0,mf->nbnd());

  if(b_range.first() < 0 and b_range.last() < 0) b_range = nda::range(mf->nbnd());
  if(b_range.first() < 0) b_range = nda::range(0,b_range.last());
  if(b_range.last() < 0) b_range = nda::range(0,mf->nbnd());
  
  utils::check( a_range.last() > a_range.first() and a_range.last() <= mf->nbnd(), 
		"cholesky::evaluate: Inconsistent a_range: ({},{})",
		a_range.first(),a_range.last());
  utils::check( b_range.last() > b_range.first() and b_range.last() <= mf->nbnd(), 
		"cholesky::evaluate: Inconsistent b_range: ({},{})",
		b_range.first(),b_range.last());
  using local_Array_t = typename memory::array<MEM,ComplexType,5>;
  decltype(nda::range::all) all;
  int sz = mpi->comm.size()/npools;
  mpi3::communicator k_intra_comm = mpi->comm.split(mpi->comm.rank()/sz,mpi->comm.rank());
  utils::check(k_intra_comm.size() == sz, "Problems with comm.split, comm.size():{}, sz:{}",k_intra_comm.size(),sz);
  if(diag_kk) {
    // not optimal, decide how to do this later!
    int nkpts = mf->nkpts();
    long k0, k1;
    std::tie(k0,k1)=itertools::chunk_range(0,nkpts,npools,mpi->comm.rank()/sz);
    std::vector<dArray_t<memory::array<MEM,ComplexType,3>>> Lk;
    Lk.reserve(k1-k0); 
    long ncmax(0),ncav(0);
    for(int k=k0; k<k1; ++k) { 
      if constexpr (MEM == HOST_MEMORY)
        Lk.emplace_back(evaluate_diagkk_impl<MEM>(k_intra_comm,Qi,k,
		 a_range,b_range,block_size));
      else
	APP_ABORT(" Error: diag_kk=true in cholesky::evaluate not yet implemented.");	
      if( k > k0 ) {
        utils::check(Lk[0].global_shape()[1] == Lk[k-k0].global_shape()[1] and
                     Lk[0].global_shape()[2] == Lk[k-k0].global_shape()[2],
		"cholesky::evaluate: Inconsistent global shape.");
        utils::check(Lk[0].local_shape()[1] == Lk[k-k0].local_shape()[1] and 
                     Lk[0].local_shape()[2] == Lk[k-k0].local_shape()[2], 
		"cholesky::evaluate: Inconsistent local shape.");
        utils::check(Lk[0].grid() == Lk[k-k0].grid(), 
		"cholesky::evaluate: Inconsistent grid shape.");
      }
      ncmax = std::max(ncmax,Lk[k-k0].global_shape()[0]);
      ncav += Lk[k-k0].global_shape()[0];
    }
    ncmax = mpi->comm.max(ncmax);
    ncav = mpi->comm.reduce_value(ncav,std::plus<>{})/long(mpi->comm.size()*nkpts);
    app_log(2,"chlesky::evaluate::diag_kk: ncmax:{}, ncav:{}",ncmax,ncav);

    auto return_value = dArray_t<local_Array_t>{ std::addressof(mpi->comm), 
		 {1,1,npools,Lk[0].grid()[1],Lk[0].grid()[2]},
                 {ncmax,1,nkpts,a_range.size(),b_range.size()},
                 {ncmax,1,k1-k0,Lk[0].local_shape()[1],Lk[0].local_shape()[2]},
                 {0,0,k0,Lk[0].local_range(1).first(),Lk[0].local_range(2).first()},
                 {1,1,1,1,1}};
    auto Lloc = return_value.local();
    Lloc()=ComplexType(0.0); 
    for(int k=0; k<k1-k0; ++k) {
      auto nc = Lk[k].global_shape()[0];
      Lloc(nda::range(0,nc),0,k,all,all) = Lk[k].local()(all,all,all);
    }
    return return_value;
  } else {
    if(block_size==0) {
      return evaluate_impl<MEM>(k_intra_comm,Qi,a_range,b_range);
    } else {
      return evaluate_blocked_impl<MEM>(k_intra_comm,Qi,a_range,b_range,block_size);
    }
  }
}

template<MEMORY_SPACE MEM>
void cholesky::evaluate(h5::group& gh5, std::string format, int Qi,
              nda::range a_range, nda::range b_range, bool diag_kk, int block_size)
{
  dArray_t<memory::array<MEM,ComplexType,5>> L = evaluate<MEM>(Qi,a_range,b_range,diag_kk,block_size);
  if constexpr (MEM==HOST_MEMORY) {
    write(gh5, Qi, L, format);
  } else {
    memory::host_array<ComplexType,5> Ah = L.local(); 
    dArray_t<memory::host_array<ComplexType,5>> Lh(L.communicator(),L.grid(),L.global_shape(),
		L.origin(),L.block_size(),std::move(Ah));
    write(gh5, Qi, Lh, format);
  }
}

template<MEMORY_SPACE MEM>
void cholesky::evaluate(h5::group& gh5, std::string format,
		nda::range a_range, nda::range b_range, bool diag_kk, int block_size)
{
  int nqpts = mf->Qpts().shape()[0];
  for(int Qi=0; Qi<nqpts; ++Qi)
    evaluate<MEM>(gh5,format,Qi,a_range,b_range,diag_kk,block_size);
  mpi->comm.barrier();
}

// MAM: instead of keeping different formats between bdft and afqmc, 
//      modify afqmc to use this more general one 
template<nda::MemoryArray local_Array_t>
void cholesky::write(h5::group& gh5, int Qi, dArray_t<local_Array_t> const& L, 
           std::string format)
{
  if(format == "default" or format == "bdft") {
    // Li [ pi, ispin*npol+ipol, ki, ai, bi], pi is the cholesky index
    math::nda::h5_write(gh5, "L"+std::to_string(Qi), L);
  } else 
    APP_ABORT("Error: Unknown format type: {}",format);
}

// writes meta data to h5 file, includes all information in addition to actual
// cholesky vectors. File should be self contained upon read  
void cholesky::write_meta_data(h5::group& gh5, std::string format)
{
#ifndef HAVE_PHDF5
  if(not mpi->comm.root()) return; 
#endif
  // MAM: orbital ranges need to be stored!!!
  //      Also not being careful with nspin_in_basis/npol_in_basis vs nspin/npol!
  if(format == "default" or format == "bdft") {
    h5::h5_write(gh5, "maximum_number_of_orbitals", mf->nbnd());
    h5::h5_write(gh5, "number_of_kpoints", mf->nkpts());
    h5::h5_write(gh5, "number_of_qpoints", mf->Qpts().shape()[0]);
    h5::h5_write(gh5, "number_of_spins", mf->nspin());
    h5::h5_write(gh5, "number_of_polarizations", mf->npol());
    h5::h5_write(gh5, "number_of_spins_in_basis", mf->nspin_in_basis());
    h5::h5_write(gh5, "number_of_polarizations_in_basis", mf->npol_in_basis());
    h5::h5_write(gh5, "volume", mf->volume());
    h5::h5_write(gh5, "grid_size", rho_g.size());
    nda::h5_write(gh5, "kpoints", mf->kpts(), false);
    nda::h5_write(gh5, "qpoints", mf->Qpts(), false);
    nda::h5_write(gh5, "qk_to_k2", mf->qk_to_k2(), false);
    nda::h5_write(gh5, "qminus", mf->qminus(), false);
    // should be number of irreducible Qpts!
//    nda::array<int,1> dummy(mf->Qpts().shape()[0]);
/*
    dummy()=0; // overwritten later on
    nda::h5_write(gh5, "number_of_cholesky_vectors", dummy);
    nda::h5_write(gh5, "row_range", dummy);
    nda::h5_write(gh5, "column_range", dummy);
*/
    // what else?
  } else 
    APP_ABORT("Error: Unknown format type: {}",format);
}

void cholesky::print_timers()
{
  app_log(2,"\n***************************************************");
  app_log(2,"                  Cholesky timers ");
  app_log(2,"***************************************************");
  app_log(2,"    Total:                 {}",Timer.elapsed("TOTAL"));
  app_log(2,"    FFT_PLAN:              {}",Timer.elapsed("FFTPLAN"));
  app_log(2,"    Allocations:           {}",Timer.elapsed("ALLOC"));
  app_log(2,"    Communication:         {}",Timer.elapsed("COMM"));
  app_log(2,"    Orb. Pairs:            {}",Timer.elapsed("Pab"));
  app_log(2,"      IO:                  {}",Timer.elapsed("IO"));
  app_log(2,"      FFT:                 {}",Timer.elapsed("FFT"));
  app_log(2,"    Residual:              {}",Timer.elapsed("DIAG"));
  app_log(2,"    Iterations:            {}",Timer.elapsed("ITER"));
  app_log(2,"      ERI:                 {}",Timer.elapsed("ERI"));
  app_log(2,"      Serial (@root):      {}",Timer.elapsed("SERIAL"));
  app_log(2,"***************************************************\n");
}

}

// definition of more complicated templates
#include "methods/ERI/cholesky.icc"

// instantiation of "public" templates
namespace methods
{

template memory::darray_t<memory::host_array<ComplexType,5>,mpi3::communicator> 
cholesky::evaluate<HOST_MEMORY>(int,nda::range,nda::range,bool,int);

template void
cholesky::evaluate(h5::group&,std::string,int,nda::range,nda::range,bool,int);

template void
cholesky::evaluate(h5::group&,std::string,nda::range,nda::range,bool,int);

template void cholesky::write(h5::group&,int,
	memory::darray_t<memory::host_array<ComplexType,5>,mpi3::communicator> const&,std::string);

/*
#if defined(ENABLE_DEVICE)

template memory::darray_t<memory::device_array<ComplexType,5>,mpi3::communicator>
cholesky::evaluate<DEVICE_MEMORY>(int,nda::range,nda::range,bool,int);

template memory::darray_t<memory::unified_array<ComplexType,5>,mpi3::communicator>
cholesky::evaluate<UNIFIED_MEMORY>(int,nda::range,nda::range,bool,int);

template void cholesky::write(h5::group&,int,
        memory::darray_t<memory::device_array<ComplexType,5>,mpi3::communicator> const&,std::string);

template void cholesky::write(h5::group&,int,
        memory::darray_t<memory::unified_array<ComplexType,5>,mpi3::communicator> const&,std::string);

#endif
*/

}
