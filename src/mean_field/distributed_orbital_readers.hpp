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


#ifndef MEANFIELD_DISTRIBUTED_ORBITAL_READERS_HPP
#define MEANFIELD_DISTRIBUTED_ORBITAL_READERS_HPP

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "utilities/check.hpp"
#include "utilities/proc_grid_partition.hpp"
#include <nda/nda.hpp>
#include <nda/h5.hpp>
#include "numerics/distributed_array/nda.hpp"
#include "numerics/fft/nda.hpp"
#include "mean_field/MF.hpp"

namespace mf
{

/**
 * Read Bloch orbitals of a system into a distributed array
 * @param mfobj      - Mean-field object
 * @param comm       - communicator
 * @param OT         - 'r': on density real-space grid; 'w': on wavefunction truncated grid, 'g' on density fft grid
 * @param pgrid_out  - the processor grid for the output distributed array
 * @param ispin      - spin index range
 * @param kp         - k-point range
 * @param orb        - orbital range
 * @param block_size - block size for the output distributed array
 * @return           - Bloch orbitals in a distributed array. 
 *                     Structure depends on the rank of the template array.
 *                       rank=2: (ispin*kp*orb*npol, grid)
 *                       rank=4: (ispin, kp, orb, npol*grid)
 *                       rank=5: (ispin, kp, orb, npol, grid)
 */
template<typename local_Array_t, 
	 typename comm_t, 
	 size_t rank = size_t(::nda::get_rank<local_Array_t>)>
auto read_distributed_orbital_set(MF& mfobj, comm_t& comm, char OT, 
		std::array<long,rank> pgrid_out = {0l},
		nda::range ispin = {-1,-1}, 
		nda::range kp = {-1,-1},   
		nda::range orb = {-1,-1},
		std::array<long,rank> block_size = {-1})
{
  decltype(nda::range::all) all;
  static_assert(rank==size_t(::nda::get_rank<local_Array_t>) and 
		(rank==2 or rank==4 or rank==5), "Rank mismatch.");
  utils::check(mfobj.has_orbital_set(), "Error in read_distributed_orbital_set: Invalid mf type. ");
  if(ispin == nda::range{-1,-1}) ispin=nda::range{0,mfobj.nspin_in_basis()};
  if(kp == nda::range{-1,-1}) kp=nda::range{0,mfobj.nkpts()};
  if(orb == nda::range{-1,-1}) orb=nda::range{0,mfobj.nbnd()};
  utils::check(ispin.first()>=0 and ispin.last()<=mfobj.nspin_in_basis(), "Range mismatch");
  utils::check(kp.first()>=0 and kp.last()<=mfobj.nkpts(), "Range mismatch");
  utils::check(orb.first()>=0 and orb.last()<=mfobj.nbnd(), "Range mismatch");
  if(OT=='G') OT='g';
  if(OT=='R') OT='r';
  if(OT=='W') OT='w';
  if(OT=='w') utils::check( mfobj.has_wfc_grid(), "Error: OT==w and has_wfc_grid==false");
  utils::check(OT=='g' or OT=='r' or OT=='w', "orbital type mismatch: {}",OT);
  char OT_in = (OT=='r')? ( (mfobj.orb_on_fft_grid())? 'g' : OT) : OT;
  if(block_size[0]<0) block_size.fill(2048);

  long nspin = ispin.size();
  long nkpts = kp.size(); 
  long nbnd = orb.size();
  long npol = mfobj.npol();
  long nnr = ( OT_in == 'w' ? mfobj.wfc_truncated_grid()->size() :  mfobj.nnr() ); 

  // no distributed FFT (by choice), so always read full PW dimension if OT=='r'  
  long np0 = std::accumulate(pgrid_out.cbegin(), pgrid_out.cend(), long(1), std::multiplies<>{});
  std::array<long,rank> pgrid = pgrid_out;
  if constexpr (rank==2) {
    if( (np0 == 0) or ((OT=='r') and (pgrid[rank-1]!=1)) ) {
      pgrid = {comm.size(),1};
    } 
  } else if constexpr (rank==4){ 
    if( (np0 == 0) or ((OT=='r') and (pgrid[rank-1]!=1)) ) {
      long sz = comm.size();
      long ps = (sz%nspin==0?nspin:1);
      long n_ = sz/ps;
      long pk = utils::find_proc_grid_max_rows(n_,nkpts);
      pgrid = {ps,pk,n_/pk,1};
    }
  } else if constexpr (rank==5){ 
    if( (np0 == 0) or ((OT=='r') and (pgrid[rank-1]!=1)) ) {
      long sz = comm.size();
      long ps = (sz%nspin==0?nspin:1); 
      long n_ = sz/ps;
      long pk = utils::find_proc_grid_max_rows(n_,nkpts);
      pgrid = {ps,pk,n_/pk,1,1};
    }
  }
  if(np0 == 0) pgrid_out = pgrid;
  utils::check(comm.size() == std::accumulate(pgrid_out.cbegin(), pgrid_out.cend(), long(1), std::multiplies<>{}), 
	       "MPI size mismatch.");

  if constexpr(rank==2) {

    long norb = ispin.size()*kp.size()*orb.size()*npol;
    auto Psi0 = math::nda::make_distributed_array<local_Array_t>(comm,pgrid,{norb,nnr},block_size);
    auto Psi0loc = Psi0.local();
    auto g_range = Psi0.local_range(1);

    if(npol==1) {
      long osz = orb.size(), ksz = kp.size(), oksz = osz*ksz;
      long i0 = Psi0.local_range(0).first();
      long i = 0, iend = Psi0.local_shape()[0];
      while( i<iend ) {
        long i_ = i;
        long is = (i+i0)/oksz;  // spin index
        long n_ = (i+i0)%oksz;
        long k  = n_/osz;       // kp index
        long a  = n_%osz;       // band index
        i++;
        while( (i<iend) and 
               (is==(i+i0)/oksz) and  
               (k==((i+i0)%oksz)/osz)  
             )
        {
          i++;
        }
        is += ispin.first();
        k += kp.first();
        a += orb.first();
        mfobj.get_orbital_set(OT_in,is,k,nda::range(a,a+i-i_),Psi0loc(nda::range(i_,i),all),g_range);
      } 
    } else {
      // read single component at a time
      //  not sure how to pack the reads without making the code very complicated
      long opsz = orb.size()*npol, opksz = opsz*kp.size();
      long i0 = Psi0.local_range(0).first();
      long i = 0, iend = Psi0.local_shape()[0];
      while( i<iend ) {
        long is = (i+i0)/opksz;  // spin index
        long n_ = (i+i0)%opksz;
        long k  = n_/opsz;       // kp index
        n_ = n_%opsz;
        long a  = n_/npol;       // spinor index
        long p  = n_%npol;       // polarization index
        is += ispin.first();
        k += kp.first();
        a += orb.first();
        mfobj.get_orbital(OT_in,is,k,a,Psi0loc(i,all),g_range+p*nnr);
        ++i;
      }
    }

    if(OT=='r' and mfobj.orb_on_fft_grid()) {
      auto fft_mesh = mfobj.fft_grid_dim();
      auto Offt = nda::reshape(Psi0loc,std::array<long,4>{Psi0loc.shape()[0],fft_mesh(0),fft_mesh(1),fft_mesh(2)});
      math::fft::invfft_many(Offt);
    } 

    if( pgrid == pgrid_out ) {
      return Psi0;
    } else {
      auto Psi = math::nda::make_distributed_array<local_Array_t>(comm,pgrid_out,{norb,nnr},block_size);
      math::nda::redistribute(Psi0,Psi);
      return Psi;
    } 

  } else if constexpr(rank==4) {

    auto Psi0 = math::nda::make_distributed_array<local_Array_t>(comm,pgrid,{nspin,nkpts,nbnd,npol*nnr},block_size);
    auto Psi0loc = Psi0.local();
    auto g_range = Psi0.local_range(3);

    for( auto [is,s] : itertools::enumerate(Psi0.local_range(0)) ) 
      mfobj.get_orbital_set(OT_in,s,Psi0.local_range(1),Psi0.local_range(2),Psi0loc(is,all,all,all),g_range);

    if(OT=='r' and mfobj.orb_on_fft_grid()) {
      auto fft_mesh = mfobj.fft_grid_dim();
      long nfft = Psi0loc.shape()[0]*Psi0loc.shape()[1]*Psi0loc.shape()[2]*npol;
      auto Offt = nda::reshape(Psi0loc,std::array<long,4>{nfft,fft_mesh(0),fft_mesh(1),fft_mesh(2)});
      math::fft::invfft_many(Offt);
    }

    if( pgrid == pgrid_out ) {
      return Psi0;
    } else {
      auto Psi = math::nda::make_distributed_array<local_Array_t>(comm,pgrid_out,{nspin,nkpts,nbnd,npol*nnr},block_size);
      math::nda::redistribute(Psi0,Psi);
      return Psi;
    }

  } else if constexpr(rank==5) {

    auto Psi0 = math::nda::make_distributed_array<local_Array_t>(comm,pgrid,{nspin,nkpts,nbnd,npol,nnr},block_size);
    auto Psi0loc = Psi0.local();
    auto g_range = Psi0.local_range(4);

    for( auto [is,s] : itertools::enumerate(Psi0.local_range(0)) ) 
      mfobj.get_orbital_set(OT_in,s,Psi0.local_range(1),Psi0.local_range(2),Psi0.local_range(3),Psi0loc(is,all,all,all,all),g_range);

    if(OT=='r' and mfobj.orb_on_fft_grid()) {
      auto fft_mesh = mfobj.fft_grid_dim();
      long nfft = Psi0loc.shape()[0]*Psi0loc.shape()[1]*Psi0loc.shape()[2]*Psi0loc.shape()[3];
      auto Offt = nda::reshape(Psi0loc,std::array<long,4>{nfft,fft_mesh(0),fft_mesh(1),fft_mesh(2)});
      math::fft::invfft_many(Offt);
    }

    if( pgrid == pgrid_out ) {
      return Psi0;
    } else {
      auto Psi = math::nda::make_distributed_array<local_Array_t>(comm,pgrid_out,{nspin,nkpts,nbnd,npol,nnr},block_size);
      math::nda::redistribute(Psi0,Psi);
      return Psi;
    }

  } else {

    utils::check(false, "Should not be here!!!");

  }

}

/**
 * Similar to read_distributed_orbital_set, but the k-point range is limited to those 
 * inside the IBZ 
 */
template<typename local_Array_t,
         typename comm_t,
         size_t rank = size_t(::nda::get_rank<local_Array_t>)>
auto read_distributed_orbital_set_ibz(MF& mfobj, comm_t& comm, char OT,
                std::array<long,rank> pgrid_out = {0l},
                nda::range ispin = {-1,-1},
                nda::range kp = {-1,-1},
                nda::range orb = {-1,-1},
                std::array<long,rank> block_size = {-1})
{
  if(block_size[0]<0) block_size.fill(2048);
  if(kp == nda::range{-1,-1}) kp = {0,mfobj.nkpts_ibz()};
  utils::check(mfobj.has_orbital_set(), "Error in read_distributed_orbital_set: Invalid mf type. ");
  utils::check(kp.first() >= 0, "Range mismatch.");
  utils::check(kp.last() <= mfobj.nkpts_ibz(), "Range mismatch.");
  return read_distributed_orbital_set<local_Array_t>(mfobj,comm,OT,pgrid_out,ispin,kp,orb,block_size);
}

} // namespace mf


#endif
