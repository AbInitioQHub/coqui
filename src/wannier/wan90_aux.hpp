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


#ifndef COQUI_WAN90_AUX_HPP
#define COQUI_WAN90_AUX_HPP

#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>

#include "configuration.hpp"

#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "nda/linalg.hpp"
#include "h5/h5.hpp"

#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/mpi_context.h"
#include "utilities/check.hpp"
#include "utilities/kpoint_utils.hpp"
#include "utilities/basis_set_utilities.hpp"
#include "utilities/integration.hpp"
#include "utilities/parser.h"
#include "utilities/stl_utils.hpp"
#include "utilities/fortran_utilities.h"
#include "numerics/nda_functions.hpp"
#include "numerics/shared_array/nda.hpp"
#include "mean_field/MF.hpp"
#include "utilities/harmonics.h"
#if defined(ENABLE_SPHERICART)
#include "sphericart.hpp"
#endif
#include "wannier/wan90_utils.hpp"
#include "wannier/scdm.hpp"

namespace wannier {

struct projection
{
  // projection function centre in crystallographic coordinates relative to the direct lattice vectors
  ::nda::stack_array<double,3> center = {0.0,0.0,0.0};
  // l and mr specify the angular part Θlmr (θ, ϕ), and r specifies the radial part Rr(r) of the projection function 
  int l = 0, mr = 0, r = 0;
  // defines the axis from which the polar angle θ in spherical polar coordinates is measured
  ::nda::stack_array<double,3> zaxis = {0.0,0.0,1.0};
  // defines the axis from with the azimuthal angle ϕ in spherical polar coordinates is measured
  ::nda::stack_array<double,3> xaxis = {1.0,0.0,0.0};
  // defines the y axis of the rotated frame. Deduced from from xaxis/zaxis. 
  ::nda::stack_array<double,3> yaxis = {0.0,1.0,0.0};
  // he value of Z a associated with the radial part of the atomic orbital. Units are in reciprocal Angstrom (in nnkp file). Converted to bohr.
  double zona = 0.0;

  projection() {}
  projection(std::istringstream &in)
  {
    if(in >>center[0] >>center[1] >>center[2] >>l >>mr >>r >>zaxis[0] >>zaxis[1] >>zaxis[2]
       >>xaxis[0] >>xaxis[1] >>xaxis[2] >>zona) {}
    else utils::check(false, "read_array: Error reading projections string");
    utils::check(not in.fail(), "projection: Failed read operation.");
    // normalize axis
    xaxis() *= (1.0/std::sqrt(::nda::sum(xaxis()*xaxis())));
    zaxis() *= (1.0/std::sqrt(::nda::sum(zaxis()*zaxis())));
    // calculate yaxis = Z x X
    yaxis = { zaxis(1)*xaxis(2)-zaxis(2)*xaxis(1), 
              -zaxis(0)*xaxis(2)+zaxis(2)*xaxis(0),
              zaxis(0)*xaxis(1)-zaxis(1)*xaxis(0)};
    yaxis() *= (1.0/std::sqrt(nda::sum(yaxis()*yaxis())));
    utils::check( std::abs(nda::sum(xaxis()*xaxis())-1.0) < 1e-6, "Invalid wannier::projection::xaxis: {}",xaxis);
    utils::check( std::abs(nda::sum(yaxis()*yaxis())-1.0) < 1e-6, "Invalid wannier::projection::yaxis: {}",yaxis);
    utils::check( std::abs(nda::sum(zaxis()*zaxis())-1.0) < 1e-6, "Invalid wannier::projection::zaxis: {}",zaxis);
    utils::check( nda::sum(zaxis()*xaxis()) < 1e-6, 
              "Invalid wannier::projection::zaxis {} and xaxis {} must be orthogonal.",zaxis,xaxis);
    utils::check( nda::sum(zaxis()*yaxis()) < 1e-6, 
              "Invalid wannier::projection::zaxis {} and yaxis {} must be orthogonal.",zaxis,yaxis);
    utils::check( nda::sum(xaxis()*yaxis()) < 1e-6, 
              "Invalid wannier::projection::xaxis {} and yaxis {} must be orthogonal.",xaxis,yaxis);
    // Angstrom to Bohr: pw2wannier.f90 does not convert it to bohr!!!
    //zona *= 0.529177249; 
    utils::check( zona > 0.0, "Invalid wannier::projection::zona:{} ",zona); 
  }
  projection(int l_, int mr_, int r_, 
             ::nda::MemoryArrayOfRank<1> auto && c_, 
             ::nda::MemoryArrayOfRank<1> auto && z_,
             ::nda::MemoryArrayOfRank<1> auto && x_,
             double za_ ) :
            center(c_),
            l(l_), mr(mr_), r(r_),
            zaxis(z_), xaxis(x_), yaxis{}, zona(za_) 
  {
    // normalize axis
    xaxis() *= (1.0/std::sqrt(nda::sum(xaxis()*xaxis())));
    zaxis() *= (1.0/std::sqrt(nda::sum(zaxis()*zaxis())));
    // calculate yaxis = Z x X
    yaxis = { zaxis(1)*xaxis(2)-zaxis(2)*xaxis(1),
              -zaxis(0)*xaxis(2)+zaxis(2)*xaxis(0),
              zaxis(0)*xaxis(1)-zaxis(1)*xaxis(0)};
    yaxis() *= (1.0/std::sqrt(nda::sum(yaxis()*yaxis())));
    utils::check( std::abs(nda::sum(xaxis()*xaxis())-1.0) < 1e-6, "Invalid wannier::projection::xaxis: {}",xaxis);
    utils::check( std::abs(nda::sum(yaxis()*yaxis())-1.0) < 1e-6, "Invalid wannier::projection::yaxis: {}",yaxis);
    utils::check( std::abs(nda::sum(zaxis()*zaxis())-1.0) < 1e-6, "Invalid wannier::projection::zaxis: {}",zaxis);
    utils::check( nda::sum(zaxis()*xaxis()) < 1e-6,
              "Invalid wannier::projection::zaxis {} and xaxis {} must be orthogonal.",zaxis,xaxis);
    utils::check( nda::sum(zaxis()*yaxis()) < 1e-6,
              "Invalid wannier::projection::zaxis {} and yaxis {} must be orthogonal.",zaxis,yaxis);
    utils::check( nda::sum(xaxis()*yaxis()) < 1e-6,
              "Invalid wannier::projection::xaxis {} and yaxis {} must be orthogonal.",xaxis,yaxis);
    // Angstrom to Bohr: pw2wannier.f90 does not convert it to bohr!!!
    //zona *= 0.529177249; 
    utils::check( zona > 0.0, "Invalid wannier::projection::zona:{} ",zona);
  }
};

namespace detail {

auto read_nnkp(utils::mpi_context_t<mpi3::communicator> &mpi,mf::MF &mf, std::string fname)
{
  auto nkpts = mf.nkpts();

  nda::array<int, 1> kp_maps(nkpts);
  nda::array<double, 2> wann_kp(nkpts, 3);
  nda::array<int, 3> nnkpts;  // (nkpts, nb, 4)
  std::vector<projection> proj;      // (nproj, nparam)
  nda::array<int,1> band_list;
  int auto_projections = 0;

  if(mpi.comm.root()) {
    utils::check(std::filesystem::exists(fname), "Problems opening nnkp file:{}",fname);
    auto file_data = utils::read_file_to_string(fname);
    auto file_data_lower = io::tolower_copy(file_data);  // used to search at this level only

    {
      ::nda::stack_array<double,3,3> arr;
      auto read = [&](std::istringstream & iss)  {
        for(int i=0; i<3; i++) 
          for(int j=0; j<3; j++) 
            if(iss >> arr(i,j)) { } 
            else 
              utils::check(false, "read_block: Error reading input block real_lattice/recip_lattice");
      }; 

      // begin real_lattice
      read_block<true>(file_data,"begin real_lattice","end real_lattice",read); 
      arr *= (1.0/0.529177249); 
      utils::check(nda::sum(nda::abs(mf.lattv()-arr)) < 1e-4, 
                   "read_nnkp: real_lattice differ. CoQui:{}, nnkp:{}",mf.lattv(),arr);

      // begin recip_lattice
      read_block<true>(file_data,"begin recip_lattice","end recip_lattice",read); 
      arr *= 0.529177249;
      utils::check(nda::sum(nda::abs(mf.recv()-arr)) < 1e-4, 
                   "read_nnkp: recip_lattice differ. CoQui:{}, nnkp:{}",mf.recv(),arr);

    }

    {
      // begin kpoints
      auto read = [&](std::istringstream & iss)  {
        int nk = 0;
        if(iss >> nk) {}
        else utils::check(false, "read_block: Error reading input block: begin kpoints");
        utils::check(nk == nkpts, 
                     "read_nnkp: Inconsistent number of k-points: CoQui:{}, nnkp:{}",nkpts,nk);
        for(int i=0; i<nk; i++) {
          if(iss >> wann_kp(i,0) >>wann_kp(i,1) >> wann_kp(i,2)) { }
          else utils::check(false, "read_block: Error reading input block: begin kpoints");
        }
      };  
      read_block<true>(file_data,"begin kpoints","end kpoints",read);
      utils::calculate_kp_map(kp_maps, wann_kp, mf.kpts_crystal());
    }

    // begin nnkpts: abusing read_matrix_with_size, but it should work correctly
    {
      auto read = [&](std::istringstream & iss)  {
        int nn = 0, ik;
        if(iss >> nn) {}
        else utils::check(false, "read_block: Error reading input block: begin nnkpts");
        nnkpts = nda::array<int, 3>(nkpts,nn,4);
        for(int i=0; i<nkpts; i++) { 
          for(int j=0; j<nn; j++) { 
            if(iss >>ik >>nnkpts(i,j,0) >>nnkpts(i,j,1) >>nnkpts(i,j,2) >>nnkpts(i,j,3)) { }
            else utils::check(false, "read_block: Error reading input block: begin nnkpts");
          }   
        }   
      };
      read_block<true>(file_data,"begin nnkpts","end nnkpts",read);
    }

    // begin exclude_bands
    if(file_data_lower.find("begin exclude_bands") != std::string::npos) {
      std::vector<bool> b(mf.nbnd(),true); 
      int nex=0;
      auto read = [&](std::istringstream & iss)  {
        int ib;
        if(iss >> nex) {}
        else utils::check(false, "read_block: Error reading input block: begin exclude_bands");
        for(int j=0; j<nex; j++) {
          if(iss >>ib) {
            utils::check(ib > 0 and ib <= b.size(), 
                       "read_nnkp: Error reading exclude_bands: ib:{}",ib);
            b[ib-1] = false;
          } else utils::check(false, "read_block: Error reading input block: begin exclude_bands");
        }
      };
      bool found = read_block<false>(file_data,"begin exclude_bands","end exclude_bands",read);
      if(nex > 0 && found) {
        band_list = nda::array<int,1>(mf.nbnd()-nex);
        // now store active bands
        int ib=0; 
        for(int in=0; in<mf.nbnd(); in++) 
          if(b[in]) band_list(ib++) = in;
        utils::check((ib+nex)==mf.nbnd(), 
              "read_nnkp: Inconsistent number of excluded bands: nbnd:{}, nexclude:{}, ncnt:{}",
              mf.nbnd(),nex,ib);
      } else {
        band_list = nda::arange<int>(mf.nbnd());
      }
    } else { 
      band_list = nda::arange<int>(mf.nbnd());
    }

    // begin projections
    if(file_data_lower.find("begin projections") != std::string::npos) {
      auto read = [&](std::istringstream & iss)  {
        int np = 0;
        if(iss >> np) {}
        else utils::check(false, "read_block: Error reading input block: begin projections");
        proj.reserve(np);
        for(int ip=0; ip<np; ++ip) proj.emplace_back(iss);
      };
      read_block<false>(file_data,"begin projections","end projections",read);
    }

    // begin auto_projections
    if(file_data_lower.find("begin auto_projections") != std::string::npos) {
      auto read = [&](std::istringstream & iss)  {
        int np = 0;
        if(iss >> auto_projections >> np) {}
        else utils::check(false, "read_block: Error reading input block: begin auto_projections");
        utils::check(np==0, 
          "read_nnkp: Second row of auto_projections should be zero (according to documentation).");
      };
      read_block<false>(file_data,"begin auto_projections","end auto_projections",read);
      if(auto_projections != 0) 
        utils::check(proj.size()==0, "read_nnkp: Found both projections and auto_projections blocks with data. Not sure what to do.");
    } 

    utils::check(auto_projections>0 or proj.size()>0,"read_nnkp: Must define either projections or auto_projections. Found neither in nnkp file.");

    long sz;
    mpi.broadcast(kp_maps);
    mpi.broadcast(wann_kp);
    sz = nnkpts.extent(1);
    mpi.comm.broadcast_value(sz);
    mpi.broadcast(nnkpts);
    sz = band_list.size();
    mpi.comm.broadcast_value(sz);
    mpi.broadcast(band_list);
    sz = proj.size();
    mpi.comm.broadcast_value(sz);
    mpi.comm.broadcast_n((char*)proj.data(),proj.size()*sizeof(projection));
    mpi.comm.broadcast_value(auto_projections);

  } else {

    long sz;
    mpi.broadcast(kp_maps);
    mpi.broadcast(wann_kp);
    mpi.comm.broadcast_value(sz);
    nnkpts = nda::array<int, 3>(nkpts,sz,4);
    mpi.broadcast(nnkpts);
    mpi.comm.broadcast_value(sz);
    band_list = nda::array<int,1>(sz);
    mpi.broadcast(band_list);
    mpi.comm.broadcast_value(sz);
    proj.resize(sz);
    mpi.comm.broadcast_n((char*)proj.data(),proj.size()*sizeof(projection));
    mpi.comm.broadcast_value(auto_projections);

  }

  return std::make_tuple(kp_maps,wann_kp,nnkpts,proj,band_list,auto_projections);

}

/*
 * Computes the decomposition of the Ylm functions in the given projectors, 
 * based on the Ylm basis in CoQui. This includes:
 *  1. Account for different ordering of basis functions,
 *  2. Mixing of angular components in wannier90 when l<0,
 *  3. Effect of rotations from definition od xaxis and zaxis in each projector. 
 */
auto rotated_ylm_coeffs(int lmax, std::vector<projection> const& proj)
{
  auto all = ::nda::range::all;
  int lmax2 = (lmax+1)*(lmax+1); 
  nda::array<double,2> coeff(proj.size(),lmax2);

  if(proj.size()==0) return coeff; 

#if defined(ENABLE_SPHERICART)
  auto ylm_calculator = sphericart::SphericalHarmonics<double>(lmax);
#else
  utils::harmonics<double> ylm_calculator{};
#endif

  auto map_lm = wann90_lm_to_Ylm();
  nda::array<double,1> c(lmax2);
  // careful, adding to c directly
  auto add_to_c = [&](int l_, int mr_, double x = 1.0) {
    utils::check(mr_ <= 2*l_+1, "rotated_ylm_coeffs::add_to_c: Invalid mr:{}",mr_);
    c( map_lm(l_*l_+mr_-1) ) = x;
  };

  // compute the expansion coefficient of wannier90's Ylm's in terms of CoQui's Ylm
  for( int i=0; i<proj.size(); ++i ) { 
    
    c() = 0.0;
    // Add contributions and map to CoQui ordering
    int l = proj[i].l, mr=proj[i].mr;
    utils::check(l>=-5 and l<4, "rotated_ylm_coeffs: Invalid l:{} in projection:{}",l,i);
    utils::check(mr > 0, "rotated_ylm_coeffs: Invalid mr:{} in projection:{}",mr,i);
    if( l>= 0) {
      add_to_c(l,mr,1.0);
    } else {
      // hard-coding this
      switch(l) {
        case -1: {
          utils::check(mr <= 2, "rotated_ylm_coeffs: Invalid l:{}, mr:{} in projection:{}",l,mr,i);
          double scl = std::sqrt(0.5);
          add_to_c(0,0,scl);
          if(mr==1) {             // s+px
            add_to_c(1,2,scl);
          } else if(mr==2) {      // s-px
            add_to_c(1,2,-1.0*scl);
          }
          break; 
        }
        case -2: {
          utils::check(mr <= 3, "rotated_ylm_coeffs: Invalid l:{}, mr:{} in projection:{}",l,mr,i);
          double s2 = std::sqrt(0.5);
          double s3 = std::sqrt(1.0/3.0);
          double s6 = std::sqrt(1.0/6.0);
          add_to_c(0,0,s3);    
          if(mr==1) {              // s-px+py
            add_to_c(1,2,-s6);     
            add_to_c(1,3,s2);      
          } else if(mr==2) {       // s-px-py
            add_to_c(1,2,-s6);     
            add_to_c(1,3,-s2);     
          } else if(mr==3) {       // s+px 
            add_to_c(1,2,2.0*s6);  
          }
          break;
        }
        case -3: {
          utils::check(mr <= 4, "rotated_ylm_coeffs: Invalid l:{}, mr:{} in projection:{}",l,mr,i);
          add_to_c(0,0,0.5);    
          if(mr==1) {              // s+px+py
            add_to_c(1,2,0.5);     
            add_to_c(1,3,0.5);      
          } else if(mr==2) {       // s+px-py-pz
            add_to_c(1,2,0.5);     
            add_to_c(1,3,-0.5);      
            add_to_c(1,1,-0.5);      
          } else if(mr==3) {       // s-px+py-pz 
            add_to_c(1,2,-0.5);     
            add_to_c(1,3,0.5);      
            add_to_c(1,1,-0.5);      
          } else if(mr==4) {       // s-px-py+pz 
            add_to_c(1,2,-0.5);     
            add_to_c(1,3,-0.5);      
            add_to_c(1,1,0.5);      
          }
          break;
        }
        case -4: {
          utils::check(mr <= 5, "rotated_ylm_coeffs: Invalid l:{}, mr:{} in projection:{}",l,mr,i);
          double s2 = std::sqrt(0.5);
          double s3 = std::sqrt(1.0/3.0);
          double s6 = std::sqrt(1.0/6.0);
          if(mr==1) {              // s-px+py
            add_to_c(0,0,s3);
            add_to_c(1,2,-s6);
            add_to_c(1,3,s2);
          } else if(mr==2) {       // s-px-py
            add_to_c(0,0,s3);
            add_to_c(1,2,-s6);
            add_to_c(1,3,-s2);
          } else if(mr==3) {       // s+px 
            add_to_c(0,0,s3);
            add_to_c(1,2,2.0*s6);
          } else if(mr==4) {       // pz+dz2 
            add_to_c(1,1,s2);
            add_to_c(2,1,s2);
          } else if(mr==5) {       // -pz+dz2
            add_to_c(1,1,-s2);
            add_to_c(2,1,s2);
          }
          break;
        }
        case -5: {
          utils::check(mr <= 6, "rotated_ylm_coeffs: Invalid l:{}, mr:{} in projection:{}",l,mr,i);
          double s2 = std::sqrt(0.5);
          double s3 = std::sqrt(1.0/3.0);
          double s6 = std::sqrt(1.0/6.0);
          double s12 = std::sqrt(1.0/12.0);
          add_to_c(0,0,s6);
          if(mr==1) {              // s-px-dz2+x2y2
            add_to_c(1,2,-s2);
            add_to_c(2,1,-s12);
            add_to_c(2,5,0.5);
          } else if(mr==2) {       // s+px-dz2+x2y2
            add_to_c(1,2,s2);
            add_to_c(2,1,-s12);
            add_to_c(2,5,0.5);
          } else if(mr==3) {       // s-py-dz2-x2y2
            add_to_c(1,2,-s2);
            add_to_c(2,1,-s12);
            add_to_c(2,5,-0.5);
          } else if(mr==4) {       // s+py-dz2-x2y2
            add_to_c(1,2,s2);
            add_to_c(2,1,-s12);
            add_to_c(2,5,-0.5);
          } else if(mr==5) {       // s-pz+dz2 
            add_to_c(1,1,-s2);
            add_to_c(2,1,s3);
          } else if(mr==6) {       // s+pz+dz2 
            add_to_c(1,1,s2);
            add_to_c(2,1,s3);
          }
          break;
        }
      }
    }

    coeff(i,all) = c();

  }

  {
    // now coeff() is the Ylm coefficient vector in CoQui's ordering without rotating axis.
    // Not apply the axis rotations. Following QE's implementation which mixes all L components. 
    //    Y'(r'=Sr) = sum_lm Ylm(r'=Sr,lm) * coeff(lm) = sum_lm Ylm(r,lm) * new_coeff(lm)  
    //    where r' is the rotated point.
    // We solve for new_coeffs using lmax2 random vectors, assuming that Ylm(r,lm) is invertible 

    nda::array<double,2> Y1(lmax2,lmax2), Yinv(lmax2,lmax2);
    nda::array<double, 2> rp(lmax2, 3);
    nda::array<double, 2> rt(lmax2, 3);

    // compute random vectors rp, inv(Ylm(r))=Yinv
    {
      bool done = false;
      int cnt=0;
      while(not done) {
        utils::check(cnt<5, "rotated_ylm_coeffs: Problems finding rotation matrix.");
        for(int j=0; j<lmax2; j++) {
          auto ri = rp(j,all);
          ri() = ::nda::rand<double>(3) - 0.5;
          ri() *= (1.0/std::sqrt(nda::sum(ri*ri)));
        }  
#if defined(ENABLE_SPHERICART)
        ylm_calculator.compute_array(rp.data(),rp.size(),Yinv.data(),Yinv.size());
#else
        ylm_calculator.spherical_harmonics(lmax,rp,Yinv);
#endif    
        ::nda::array<int, 1> ipiv(lmax2);
        int info = ::nda::lapack::getrf(Yinv, ipiv); 
        if(info != 0) {
          cnt++;
          continue;  
        }
        info = ::nda::lapack::getri(Yinv, ipiv);
        if(info != 0) {
          cnt++;
          continue;  
        }
        done = true;
      };
    }

    // for each projector, compute r'=rt, Ylm(r',lm)*coeff(lm), and new_coeff. 
    nda::array<double,1> X_ = {1.0,0.0,0.0};
    nda::array<double,1> Z_ = {0.0,0.0,1.0};
    for(int i=0; i<proj.size(); i++) 
    {
      // only needed if x/z axis are rotated
      if( nda::sum(nda::abs(X_-proj[i].xaxis)) + nda::sum(nda::abs(Z_-proj[i].zaxis)) > 1e-8 ) {
        nda::array<double,2> U(3,3);
        U(0,all) = proj[i].xaxis();
        U(1,all) = proj[i].yaxis();
        U(2,all) = proj[i].zaxis();

        nda::blas::gemm(1.0,rp,nda::transpose(U),0.0,rt);
      
#if defined(ENABLE_SPHERICART)
        ylm_calculator.compute_array(rt.data(),rt.size(),Y1.data(),Y1.size());
#else
        ylm_calculator.spherical_harmonics(lmax,rt,Y1);
#endif

        ::nda::blas::gemv(1.0,Y1,coeff(i,all),0.0,c);
        ::nda::blas::gemv(1.0,Yinv,c,0.0,coeff(i,all));
      }
      app_log(2,"  Projection #:{}",i);
      app_log(2,"    l:{}, mr:{}, r:{}, zona:{}",proj[i].l,proj[i].mr,proj[i].r,proj[i].zona);
      app_log(2,"     center:{}",proj[i].center);
      app_log(2,"     x axis:{}",proj[i].xaxis);
      app_log(2,"     z axis:{}",proj[i].zaxis);
    }

  } 

  return coeff;
}

auto compute_mmn(utils::mpi_context_t<mpi3::communicator> &mpi, mf::MF &mf,    
    std::string prefix, nda::array<int, 1> const& kp_maps,
    nda::array<double, 2> const& wann_kp, nda::array<int, 3> const& nnkpts,
    nda::array<int,1> const& band_list, bool transpose, bool write_to_file)
{
  using Array_view_4D_t = nda::array_view<ComplexType,4>;
  auto all = ::nda::range::all;
  long nband = band_list.size();
  long nkpts = nnkpts.extent(0);
  long nnb = nnkpts.extent(1); 
  auto wfc_g = mf.wfc_truncated_grid();
  auto fft2gv = wfc_g->fft_to_gv();
  nda::array<long,1> k2g(wfc_g->size());
  nda::array<ComplexType,1> *Xft = nullptr;
  auto [ib_min,ib_max] = std::minmax_element(band_list.begin(),band_list.end());
  nda::range b_rng(*ib_min,*ib_max+1);
  nda::array<int,1> shifted_band_list = band_list;  // shift by b_rng.first() 
  shifted_band_list() -= b_rng.first();

  nda::array<ComplexType,2> buff(b_rng.size(),wfc_g->size());
  nda::array<ComplexType,3> psi(2,nband,wfc_g->size());
 
  utils::check(mf.nspin() == 1 and mf.npol() == 1, "wannier::compute_mmn: Finish!!! nspin>1 or npol>1"); 
  auto Mmn = math::shm::make_shared_array<Array_view_4D_t>(mpi, {nkpts,nnb,nband,nband});
  auto Mloc = Mmn.local();

  // Mmn(k,b) = Mmn(k_ibz,b_ibz) * h, 
  //            where h = transpose( S_{k_ibz+b_ibz} ) * transpose( S_{k} ) * S_{k+b} 
  for(long ik=0; ik<nkpts; ik++) {

    if( ik%mpi.comm.size() != mpi.comm.rank() ) continue;
    auto kl = kp_maps[ik]; 
    mf.get_orbital_set('w',0,kl,b_rng,buff);
    // add phase factor if needed!
    utils::check(utils::equivalent_k(wann_kp(ik,all),mf.kpts_crystal(kl),1e-6), 
        "wannier::compute_mmn: Problems mapping kpoints: nnkp:{}, CoQui:{}",
        wann_kp(ik,all),mf.kpts_crystal(kl));
    if(nda::sum( nda::abs(wann_kp(ik,all)-mf.kpts_crystal(kl)) ) > 1e-6) { 
      k2g() = wfc_g->gv_to_fft()();
      // careful with sigh of dG, since transform_k2g performs G -> G-dG
      // k2g(i) = N -> G[N] -> G - (-dG) -> N'  
      // psi(...,N) = buff(...,fft2g(N')), which is equivalent to psi_{k+dG}(G) = psi_{k}( G + dG ) 
      // for dG = wann_kp(ik,all)-mf.kpts_crystal(kl) 
      utils::transform_k2g(false, mf.symm_list(0),
                           mf.kpts_crystal(kl)-wann_kp(ik,all),
                           wfc_g->mesh(), wann_kp(ik,all), k2g, Xft);
      for(auto [i,n] : itertools::enumerate(shifted_band_list)) 
        for(auto [j,m] : itertools::enumerate(k2g)) 
          if(fft2gv(m) >= 0)
            psi(0,i,j) = std::conj(buff(n,fft2gv(m)));  // conjugate to allow use of gemm below 
    } else {
      for(auto [i,n] : itertools::enumerate(shifted_band_list)) psi(0,i,all) = nda::conj(buff(n,all)); 
    }
    for(long m=0; m<nnb; ++m) {
      // nnkpts(ik,m,0) must be mapped to the same local kpoint as k+b, so reuse kp_maps
      auto kr = kp_maps[nnkpts(ik,m,0)-1]; 
      mf.get_orbital_set('w',0,kr,b_rng,buff);
      nda::array<double,1> k_plus_b = wann_kp(nnkpts(ik,m,0)-1,all);  
      k_plus_b(0) += double(nnkpts(ik,m,1));
      k_plus_b(1) += double(nnkpts(ik,m,2));
      k_plus_b(2) += double(nnkpts(ik,m,3));
      // add phase factor if needed!
      utils::check(utils::equivalent_k(k_plus_b,mf.kpts_crystal(kr),1e-6), 
          "wannier::compute_mmn: Problems mapping kpoints: k+b:{}, CoQui:{}",
          k_plus_b,mf.kpts_crystal(kr));
      if(nda::sum( nda::abs(k_plus_b-mf.kpts_crystal(kr)) ) > 1e-6) {
        k2g() = wfc_g->gv_to_fft()();
        utils::transform_k2g(false, mf.symm_list(0),
                             mf.kpts_crystal(kr)-k_plus_b, 
                             wfc_g->mesh(), k_plus_b, k2g, Xft);
        for(auto [i,n] : itertools::enumerate(shifted_band_list))
          for(auto [j,nn] : itertools::enumerate(k2g)) 
            if(fft2gv(nn) >= 0)
              psi(1,i,j) = buff(n,fft2gv(nn)); 
      } else {
        for(auto [i,n] : itertools::enumerate(shifted_band_list)) psi(1,i,all) = buff(n,all); 
      }
      // Overlap
      if(transpose) 
        ::nda::blas::gemm(psi(1,all,all),::nda::transpose(psi(0,all,all)),Mloc(ik,m,all,all));
      else
        ::nda::blas::gemm(psi(0,all,all),::nda::transpose(psi(1,all,all)),Mloc(ik,m,all,all));
    }
  }

  if(mpi.node_comm.root()) 
    mpi.internode_comm.all_reduce_in_place_n(Mloc.data(),Mloc.size(),std::plus<>{});
  if(mpi.comm.root() and write_to_file) 
    detail::write_mmn_file(prefix,nnkpts,Mloc,transpose);
  mpi.comm.barrier();

  return Mmn;
}

auto compute_amn_projections(utils::mpi_context_t<mpi3::communicator> &mpi, mf::MF &mf,    
    ptree &pt, nda::array<int, 1> const& kp_maps,
    nda::array<double, 2> const& wann_kp, nda::array<int,1> const& band_list,
    std::vector<projection> const& proj, bool transpose, bool write_to_file) 
{
  using Array_view_3D_t = nda::array_view<ComplexType,3>;
  // options
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto prefix = io::get_value<std::string>(pt,"prefix");
  auto resolved_prefix = outdir + "/" + prefix;
  auto r0 = io::get_value_with_default<double>(pt,"r0",std::exp(-6.0));
  auto rN = io::get_value_with_default<double>(pt,"rN",10.0);
  auto nintg_points= io::get_value_with_default<long>(pt,"nr",333);

  // local variables
  auto all = ::nda::range::all;
  long nband = band_list.size();
  long nkpts = wann_kp.extent(0);
  auto lattv = mf.lattv();
  auto wfc_g = mf.wfc_truncated_grid();
  auto fft2gv = wfc_g->fft_to_gv();
  auto gvecs = wfc_g->g_vectors();
  nda::array<long,1> k2g(wfc_g->size());
  nda::array<ComplexType,1> *Xft = nullptr;
  auto [ib_min,ib_max] = std::minmax_element(band_list.begin(),band_list.end());
  nda::range b_rng(*ib_min,*ib_max+1);
  nda::array<int,1> shifted_band_list = band_list;  // shift by b_rng.first() 
  shifted_band_list() -= b_rng.first();
  long nproj = proj.size();
  int lmax = 0;

  for( auto v : proj ) {
    if(v.l >= 0) { 
      lmax = std::max(lmax,v.l);
    } else {
      if(v.l > -4)  // sp hybrids
        lmax = std::max(lmax,1);
      else          // spd hybrids
        lmax = std::max(lmax,2);
    }
  }
  int lmax2 = (lmax+1)*(lmax+1);

  nda::array<ComplexType,2> buff(b_rng.size(),wfc_g->size());
  nda::array<ComplexType,2> psi(nband,wfc_g->size());
  nda::array<ComplexType,2> wann(nproj,wfc_g->size());
 
  utils::check(mf.nspin() == 1 and mf.npol() == 1, 
               "wannier::compute_amn: Finish!!! nspin>1 or npol>1"); 

  std::array<long,3> shape = {nkpts,nband,nproj};
  if(transpose) shape = {nkpts,nproj,nband};
  auto Amn = math::shm::make_shared_array<Array_view_3D_t>(mpi, shape);
  auto Aloc = Amn.local();

  // sphericart calculator
#if defined(ENABLE_SPHERICART)
  auto ylm_calculator = sphericart::SphericalHarmonics<double>(lmax);
#else
  utils::harmonics<double> ylm_calculator{};
#endif
  nda::array<double, 2> Ylm(wfc_g->size(),lmax2), Gxyz(wfc_g->size(),3);
  nda::array<double, 1> G2(wfc_g->size());
  nda::array<double,2> F(wfc_g->size(),lmax+1);  

  auto ylm_coeffs = rotated_ylm_coeffs(lmax,proj);

  // naive parallelization, distribute over g-vectors if needed
  for(long ik=0; ik<nkpts; ik++) {

    if( ik%mpi.comm.size() != mpi.comm.rank() ) continue;
    auto kl = kp_maps[ik]; 
    mf.get_orbital_set('w',0,kl,b_rng,buff);
    // add phase factor if needed!
    utils::check(utils::equivalent_k(wann_kp(ik,all),mf.kpts_crystal(kl),1e-6), 
        "wannier::compute_amn: Problems mapping kpoints: nnkp:{}, CoQui:{}",
        wann_kp(ik,all),mf.kpts_crystal(kl));
    if(nda::sum( nda::abs(wann_kp(ik,all)-mf.kpts_crystal(kl)) ) > 1e-6) { 
      k2g() = wfc_g->gv_to_fft()();
      // careful with sigh of dG, since transform_k2g performs G -> G-dG
      // k2g(i) = N -> G[N] -> G - (-dG) -> N'  
      // psi(...,N) = buff(...,fft2g(N')), which is equivalent to psi_{k+dG}(G) = psi_{k}( G + dG ) 
      // for dG = wann_kp(ik,all)-mf.kpts_crystal(kl) 
      utils::transform_k2g(false, mf.symm_list(0),
                           mf.kpts_crystal(kl)-wann_kp(ik,all),
                           wfc_g->mesh(), wann_kp(ik,all), k2g, Xft);
      for(auto [i,n] : itertools::enumerate(shifted_band_list)) 
        for(auto [j,m] : itertools::enumerate(k2g)) 
          if(fft2gv(m) >= 0)
            psi(i,j) = std::conj(buff(n,fft2gv(m)));  // conjugate to allow use of gemm below 
    } else {
      for(auto [i,n] : itertools::enumerate(shifted_band_list)) psi(i,all) = nda::conj(buff(n,all)); 
    }

    // wann_kp in cartesian coordinates
    nda::stack_array<double,3> kp = {0.0,0.0,0.0};
    nda::stack_array<double,3> Rn = {0.0,0.0,0.0};
    nda::blas::gemv(1.0,nda::transpose(mf.recv()),wann_kp(ik,all),0.0,kp); 

    wann() = ComplexType(0.0);
    
    // build Ylm
    for(int ig=0; ig<wfc_g->size(); ++ig) {
      Gxyz(ig,all) = gvecs(ig,all)+kp;
      G2(ig) = std::sqrt(Gxyz(ig,0)*Gxyz(ig,0) + Gxyz(ig,1)*Gxyz(ig,1) + Gxyz(ig,2)*Gxyz(ig,2)); 
    }

    // Ylm(Gxyz)
    Ylm() = 0.0;
#if defined(ENABLE_SPHERICART)
    ylm_calculator.compute_array(Gxyz.data(),Gxyz.size(),Ylm.data(),Ylm.size());
    for(int ig=0; ig<wfc_g->size(); ++ig)
      if(std::abs(G2(ig)) < 1e-4) Ylm(ig,nda::range(1,lmax2)) = 0.0;
#else
    ylm_calculator.spherical_harmonics(lmax,Gxyz,Ylm);
#endif

    nda::array<double,1> ri(nintg_points), dr(nintg_points), fr(nintg_points), work(nintg_points);

    for(long m=0; m<nproj; ++m) {

      double alfa = proj[m].zona;

      if(proj[m].r==1) {
        utils::log_grid_f<double> r_grid(r0,rN,nintg_points);
        double scl = 2.0*std::pow(alfa,1.5);
        for( long i=0; i<nintg_points; ++i ) {
          std::tie(ri(i),dr(i)) = r_grid.r_dr(i);
          fr(i) =  scl * std::exp(-alfa*ri(i)); 
        }
      } else if(proj[m].r==2) {
        utils::log_grid_f<double> r_grid(r0/(0.5*alfa),rN/(0.5*alfa),nintg_points);
        double scl = 1.0/std::sqrt(8.0)*std::pow(alfa,1.5);
        for( long i=0; i<nintg_points; ++i ) {
          std::tie(ri(i),dr(i)) = r_grid.r_dr(i);
          fr(i) = scl * (2.0 - alfa*ri(i)) * std::exp(-alfa*ri(i)*0.5);
        }
      } else if(proj[m].r==3) {
        utils::log_grid_f<double> r_grid(r0*3.0/alfa,rN*3.0/alfa,nintg_points);
        double scl = std::sqrt(4.0/27.0)*std::pow(alfa,1.5);
        for( long i=0; i<nintg_points; ++i ) {
          std::tie(ri(i),dr(i)) = r_grid.r_dr(i);
          fr(i) = scl * (1.0 - 2.0/3.0*alfa*ri(i) + 2.0*std::pow(alfa*ri(i),2.0)/27.0) * 
                  std::exp(-alfa*ri(i)/3.0);
        }
      } else {
        utils::check(false, 
                     "compute_amn_projections: Invalid type (e.g. r) in projection.",proj[m].r);
      }

      for(int ig=0; ig<wfc_g->size(); ++ig) 
        utils::sph_bessel_transform<double>(nda::range(lmax+1),G2(ig),ri,dr,fr,work,F(ig,all));

      for(int ig=0; ig<wfc_g->size(); ++ig) { 
        for(int lm=0; lm<lmax2; ++lm) {
          int l = int(std::sqrt(lm*1.0));
          wann(m,ig) += ylm_coeffs(m,lm) * Ylm(ig,lm) * F(ig,l) * std::pow(ComplexType(0.0,-1.0),l); 
        } 
      }
      
      nda::blas::gemv(1.0,nda::transpose(lattv),proj[m].center,0.0,Rn); 
      for(int ig=0; ig<wfc_g->size(); ++ig) { 
        double Gr = nda::sum(Gxyz(ig,all) * Rn); 
        wann(m,ig) *= (4 * 3.14159265358979 * std::exp(ComplexType(0.0,-Gr)) / std::sqrt(mf.volume()));
      }   

    }
    
    // Overlap
    if(transpose) {
      ::nda::blas::gemm(wann(all,all),::nda::transpose(psi(all,all)),Aloc(ik,all,all));
    } else {
      ::nda::blas::gemm(psi(all,all),::nda::transpose(wann(all,all)),Aloc(ik,all,all));
    }
  }

  if(mpi.node_comm.root()) 
    mpi.internode_comm.all_reduce_in_place_n(Aloc.data(),Aloc.size(),std::plus<>{});
  if(mpi.comm.root() and write_to_file) 
    detail::write_amn_file(resolved_prefix,Aloc,transpose);
  mpi.comm.barrier();

  return Amn;

}

/* MAM: original slow version, remove when everything is well tested
auto compute_amn_projections_2(utils::mpi_context_t<mpi3::communicator> &mpi, mf::MF &mf,    
    ptree &pt, nda::array<int, 1> const& kp_maps,
    nda::array<double, 2> const& wann_kp, nda::array<int,1> const& band_list,
    std::vector<projection> const& proj, bool transpose, bool write_to_file) 
{
  using Array_view_3D_t = nda::array_view<ComplexType,3>;
  // options
  auto prefix = io::get_value<std::string>(pt,"prefix");
  auto r0 = io::get_value_with_default<double>(pt,"r0",std::exp(-6.0));
  auto rN = io::get_value_with_default<double>(pt,"rN",10.0);
  auto nintg_points= io::get_value_with_default<long>(pt,"nr",333);

  // local variables
  auto all = ::nda::range::all;
  long nband = band_list.size();
  long nkpts = wann_kp.extent(0);
  auto lattv = mf.lattv();
  auto wfc_g = mf.wfc_truncated_grid();
  auto fft2gv = wfc_g->fft_to_gv();
  auto gvecs = wfc_g->g_vectors();
  nda::array<long,1> k2g(wfc_g->size());
  nda::array<ComplexType,1> *Xft = nullptr;
  auto [ib_min,ib_max] = std::minmax_element(band_list.begin(),band_list.end());
  nda::range b_rng(*ib_min,*ib_max+1);
  nda::array<int,1> shifted_band_list = band_list;  // shift by b_rng.first() 
  shifted_band_list() -= b_rng.first();
  long nproj = proj.size();
  int lmax = 0;

  for( auto v : proj ) {
    if(v.l >= 0) { 
      lmax = std::max(lmax,v.l);
    } else {
      if(v.l > -4)  // sp hybrids
        lmax = std::max(lmax,1);
      else          // spd hybrids
        lmax = std::max(lmax,2);
    }
  }
  int lmax2 = (lmax+1)*(lmax+1);

  nda::array<ComplexType,2> buff(b_rng.size(),wfc_g->size());
  nda::array<ComplexType,2> psi(nband,wfc_g->size());
  nda::array<ComplexType,2> wann(wfc_g->size(),nproj);
 
  utils::check(mf.nspin() == 1 and mf.npol() == 1, 
               "wannier::compute_amn: Finish!!! nspin>1 or npol>1"); 

  std::array<long,3> shape = {nkpts,nband,nproj};
  if(transpose) shape = {nkpts,nproj,nband};
  auto Amn = math::shm::make_shared_array<Array_view_3D_t>(mpi, shape);
  auto Aloc = Amn.local();

  // sphericart calculator
#if defined(ENABLE_SPHERICART)
  auto ylm_calculator = sphericart::SphericalHarmonics<double>(lmax);
#else
  utils::harmonics<double> ylm_calculator{};
#endif
  nda::array<double, 1> Ylm(lmax2), Gxyz(3);

  auto ylm_coeffs = rotated_ylm_coeffs(lmax,proj);

  // naive parallelization, distribute over g-vectors if needed
  for(long ik=0; ik<nkpts; ik++) {

    if( ik%mpi.comm.size() != mpi.comm.rank() ) continue;
    auto kl = kp_maps[ik]; 
    mf.get_orbital_set('w',0,kl,b_rng,buff);
    // add phase factor if needed!
    utils::check(utils::equivalent_k(wann_kp(ik,all),mf.kpts_crystal(kl),1e-6), 
        "wannier::compute_amn: Problems mapping kpoints: nnkp:{}, CoQui:{}",
        wann_kp(ik,all),mf.kpts_crystal(kl));
    if(nda::sum( nda::abs(wann_kp(ik,all)-mf.kpts_crystal(kl)) ) > 1e-6) { 
      k2g() = wfc_g->gv_to_fft()();
      // careful with sigh of dG, since transform_k2g performs G -> G-dG
      // k2g(i) = N -> G[N] -> G - (-dG) -> N'  
      // psi(...,N) = buff(...,fft2g(N')), which is equivalent to psi_{k+dG}(G) = psi_{k}( G + dG ) 
      // for dG = wann_kp(ik,all)-mf.kpts_crystal(kl) 
      utils::transform_k2g(false, mf.symm_list(0),
                           mf.kpts_crystal(kl)-wann_kp(ik,all),
                           wfc_g->mesh(), wann_kp(ik,all), k2g, Xft);
      for(auto [i,n] : itertools::enumerate(shifted_band_list)) 
        for(auto [j,m] : itertools::enumerate(k2g)) 
          if(fft2gv(m) >= 0)
            psi(i,j) = std::conj(buff(n,fft2gv(m)));  // conjugate to allow use of gemm below 
    } else {
      for(auto [i,n] : itertools::enumerate(shifted_band_list)) psi(i,all) = nda::conj(buff(n,all)); 
    }

    // wann_kp in cartesian coordinates
    nda::stack_array<double,3> kp = {0.0,0.0,0.0};
    nda::blas::gemv(1.0,nda::transpose(mf.recv()),wann_kp(ik,all),0.0,kp); 

    wann() = ComplexType(0.0);
    for(int ig=0; ig<wfc_g->size(); ++ig) {
      Gxyz(0) = gvecs(ig,0)+kp(0);
      Gxyz(1) = gvecs(ig,1)+kp(1);
      Gxyz(2) = gvecs(ig,2)+kp(2);
      double q = std::sqrt(Gxyz(0)*Gxyz(0) + Gxyz(1)*Gxyz(1) + Gxyz(2)*Gxyz(2)); 

      // Ylm(Gxyz)
      Ylm() = 0.0;
#if defined(ENABLE_SPHERICART)
      ylm_calculator.compute_sample(Gxyz.data(),Gxyz.size(),Ylm.data(),Ylm.size());
#else
      ylm_calculator.spherical_harmonics(lmax,Gxyz,Ylm);
#endif
      // unfortunately, this seems necessary
      if(std::abs(q) < 1e-4) Ylm(nda::range(1,lmax2)) = 0.0;

      for(long m=0; m<nproj; ++m) {

        double alfa = proj[m].zona;
        nda::array<double,1> F(lmax+1);

        if(proj[m].r==1) {

          utils::log_grid_f<double> r(r0,rN,nintg_points);
          double scl = 2.0*std::pow(alfa,1.5);
          auto radial_fun = [&] (auto && r_) {
           return scl * std::exp(-alfa*r_);
          };
          utils::sph_bessel_transform_boost<double>(nda::range(lmax+1),q,r,radial_fun,F);
          
        } else if(proj[m].r==2) {

          utils::log_grid_f<double> r(r0/(0.5*alfa),rN/(0.5*alfa),nintg_points);
          double scl = 1.0/std::sqrt(8.0)*std::pow(alfa,1.5);
          auto radial_fun = [&] (auto && r_) {
           return  scl * (2.0 - alfa*r_) * std::exp(-alfa*r_*0.5);
          };
          utils::sph_bessel_transform_boost<double>(nda::range(lmax+1),q,r,radial_fun,F);

        } else if(proj[m].r==3) {

          utils::log_grid_f<double> r(r0*3.0/alfa,rN*3.0/alfa,nintg_points);
          double scl = std::sqrt(4.0/27.0)*std::pow(alfa,1.5);
          auto radial_fun = [&] (auto && r_) {
           return scl * (1.0 - 2.0/3.0*alfa*r_ + 2.0*std::pow(alfa*r_,2.0)/27.0) * std::exp(-alfa*r_/3.0);
          };
          utils::sph_bessel_transform_boost<double>(nda::range(lmax+1),q,r,radial_fun,F);

        } else {
          utils::check(false, 
                       "compute_amn_projections: Invalid type (e.g. r) in projection.",proj[m].r);
        }

        for(int lm=0; lm<lmax2; ++lm) {
          int l = int(std::sqrt(lm*1.0));
          wann(ig,m) += ylm_coeffs(m,lm) * Ylm(lm) * F(l) * std::pow(ComplexType(0.0,-1.0),l); 
        } 
        auto Rw = proj[m].center;
        double Gr = Gxyz(0) * (Rw(0)*lattv(0,0) + Rw(1)*lattv(1,0) + Rw(2)*lattv(2,0)) +
                    Gxyz(1) * (Rw(0)*lattv(0,1) + Rw(1)*lattv(1,1) + Rw(2)*lattv(2,1)) +
                    Gxyz(2) * (Rw(0)*lattv(0,2) + Rw(1)*lattv(1,2) + Rw(2)*lattv(2,2));
       
        wann(ig,m) *= (4 * 3.14159265358979 * std::exp(ComplexType(0.0,-Gr)) / std::sqrt(mf.volume()));
        
      }
    }

    // Overlap
    if(transpose) {
      ::nda::blas::gemm(::nda::transpose(wann(all,all)),::nda::transpose(psi(all,all)),Aloc(ik,all,all));
    } else {
      ::nda::blas::gemm(psi(all,all),wann(all,all),Aloc(ik,all,all));
    }
  }

  if(mpi.node_comm.root()) 
    mpi.internode_comm.all_reduce_in_place_n(Aloc.data(),Aloc.size(),std::plus<>{});
  if(mpi.comm.root() and write_to_file) 
    detail::write_amn_file(prefix,Aloc,transpose);
  mpi.comm.barrier();

  return Amn;
}
*/

// spin???
inline auto get_eig(utils::mpi_context_t<mpi3::communicator> &mpi, mf::MF &mf, std::string prefix,
    nda::array<int, 1> const& kp_map, nda::array<int,1> const& band_list, bool write_to_file) 
{
  nda::array<double,3> eigv(mf.nspin(),kp_map.size(),band_list.size());
  for( auto [ik, k] : itertools::enumerate(kp_map) )
    for( auto [in,n] : itertools::enumerate(band_list) )
      eigv(::nda::range::all,ik,in) = mf.eigval()(::nda::range::all,k,n);
  if(mpi.comm.root() and write_to_file) {
    // output 
    std::ofstream out(prefix+".eig");
    time_t timestamp;
    time(&timestamp);
    for( auto [ik, k] : itertools::enumerate(kp_map) )
      for( auto [in,n] : itertools::enumerate(band_list) )
          out<<std::fixed <<std::setw(10) <<in+1 <<" "
                    <<std::setw(10) <<ik+1 <<" "
                    <<std::setw(18) <<std::setprecision(12) <<mf.eigval(0,k,n)/3.674932540e-2 <<"\n";
    out.close();
  }
  mpi.comm.barrier();
  
  return eigv;
}  

auto wannier90_library_run(utils::mpi_context_t<mpi3::communicator> &mpi, mf::MF &mf, ptree pt)
{
  using ::nda::range;
  auto all = nda::range::all;

  auto nkpts = mf.nkpts();
  auto nspin = mf.nspin();
  auto kpts = mf.kpts_crystal();

  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto prefix = io::get_value<std::string>(pt,"prefix");
  auto resolved_prefix = outdir + "/" + prefix;
  auto write_mmn = io::get_value_with_default<bool>(pt,"write_mmn",false);
  auto write_amn = io::get_value_with_default<bool>(pt,"write_amn",false);
  auto write_eigv = io::get_value_with_default<bool>(pt,"write_eig",false);
  auto write_nnkp = io::get_value_with_default<bool>(pt,"write_nnkp",false);
  auto write_h5 = io::get_value_with_default<bool>(pt,"write_h5",true);

  int nband = 0;
  int nwann = 0;
  ::nda::array<double,2> lattv = mf.lattv()*0.529177210544;

  // mapping between wann_kp and mf.kpts()
  auto kp_map = ::nda::arange<int>(nkpts); 

  // list of kpoints ( from *win if they exist, or from CoQui otherwise)
  ::nda::array<double,2> wann_kp(kpts);

  // list of included bands
  ::nda::array<int,1> band_list;

  // atomic positions
  auto species = mf.species();
  auto nat = mf.number_of_atoms();
  nda::array<double,2> at_cart_ang = (mf.atomic_positions()*0.529177210544);
  nda::array<int,1> at_sym_sz(nat);
  for(int i=0; i<nat; i++)  
    at_sym_sz(i) = species[mf.atomic_id(i)].length(); 
  int max_sym_sz = *std::max_element(at_sym_sz.begin(),at_sym_sz.end());
  nda::array<char,2> at_sym(nat,max_sym_sz);
  at_sym() = ' ';
  for(int i=0; i<nat; i++) 
    std::copy_n(species[mf.atomic_id(i)].data(), at_sym_sz(i), at_sym(i,all).data());

  // # neighbors (estimate right now), # projections
  int n_neigh = 24, nproj_lines = 0, auto_projections = 0, max_len=0;
  // projection strings if they are found in win file
  nda::array<char,2> proj_str;
  nda::array<int,1> str_len;
  std::string exclude_string = " ";

  // read win file 
  if(mpi.comm.root()) {
 
    // read nband, nwann, nproj_lines, proj_str, auto_projections, kpoints
    // check lattv, atom positions
    utils::check(std::filesystem::exists(resolved_prefix+".win"), "Problems opening win file:{}",resolved_prefix+".win");
    auto file_data = utils::read_file_to_string(resolved_prefix+".win");
    auto file_data_lower = io::tolower_copy(file_data);  // used to search at this level only

    app_log(2," - Parsing {} input file. ",resolved_prefix+".win");
    nwann = read_key<true,int>(file_data,"num_wann");
    nband = read_key<false,int>(file_data,"num_bands",nwann);
    if( read_key<false,bool>(file_data, "auto_projections", false) ) auto_projections = nwann;

    if(file_data_lower.find("mp_grid") != std::string::npos) {
      auto grid = read_range<false>(file_data,"mp_grid"," ");
      if(grid.size() > 0) 
        utils::check(grid[0]==mf.kp_grid()(0) and grid[1]==mf.kp_grid()(1) and grid[2]==mf.kp_grid()(2),
          "Incompatible mp_grid: {}, CoQui:{}",grid,mf.kp_grid());
    }

    // if provided, read them kpoints
    if(file_data_lower.find("begin kpoints") != std::string::npos) {
      auto read = [&](std::istringstream & iss)  {
        for(int i=0; i<nkpts; i++) {
          if(iss >> wann_kp(i,0) >>wann_kp(i,1) >> wann_kp(i,2)) { }
          else utils::check(false, "read_block: Error reading input block: begin kpoints");
        }
      };
      bool found = read_block<false>(file_data,"begin kpoints","end kpoints",read);
      // this routine will abort if the mapping is not complete/correct
      if(found) utils::calculate_kp_map(kp_map, wann_kp, mf.kpts_crystal());
    }

    // check lattv and atom_positions
    {
      ::nda::stack_array<double,3,3> arr;
      bool bohr = false;
      auto read = [&](std::istringstream & iss)  {
        bohr = false;
        while(iss.peek() == ' ') iss.get();
        if(iss.peek() == 'b' or iss.peek() == 'B') {
          std::string line; 
          std::getline(iss,line);
          auto tokens = utils::split(line," ");
          utils::check(io::tolower_copy(tokens[0])=="bohr", 
            "Error reading units in unit_cell_cart block: {}",tokens[0]);
          bohr = true;
        }
        for(int i=0; i<3; i++)
          for(int j=0; j<3; j++)
            if(iss >> arr(i,j)) { }
            else
              utils::check(false, "read_block: Error reading input block ");
      };

      // begin unit_cell_cart 
      bool found = read_block<false>(file_data,"begin unit_cell_cart","end unit_cell_cart",read);
      if(found) {
        if(not bohr) arr *= (1.0/0.529177249);
        utils::check(nda::sum(nda::abs(mf.lattv()-arr)) < 1e-4,
                     "*.win: unit_cell_cart differ. CoQui:{}, win:{}",mf.lattv(),arr);
      }
    }

    bool found_atoms_cart = false;
    {
      std::vector<std::string> ids;
      std::vector<nda::stack_array<double,3>> rp;
      bool bohr = false;
      std::string end_label("end atoms_cart");
      auto read = [&](std::istringstream & iss)  {
        bohr = false;
        while(iss.peek() == ' ') iss.get();
        if(iss.peek() == 'b' or iss.peek() == 'B') {
          std::string line;
          std::getline(iss,line);
          auto tokens = utils::split(line," ");
          utils::check(io::tolower_copy(tokens[0])=="bohr",
            "Error reading units in unit_cell_cart/atoms_cart/atoms_frac block: {}",tokens[0]);
          bohr = true;
        } else if(iss.peek() == 'a' or iss.peek() == 'A') {
          std::string line;
          std::getline(iss,line);
          auto tokens = utils::split(line," ");
          utils::check(io::tolower_copy(tokens[0])=="ang",
            "Error reading units in unit_cell_cart/atoms_cart/atoms_frac block: {}",tokens[0]);
          bohr = false;
        }
        std::string line, name;
        nda::stack_array<double,3> r_ = {0.0,0.0,0.0};
        while(std::getline(iss,line)) {  
          if(io::tolower_copy(line).find(end_label) != std::string::npos) return;
          std::istringstream iss2(line);
          if(iss2 >>name >> r_(0) >> r_(1) >>r_(2)) { 
            ids.emplace_back(name);
            rp.emplace_back(r_);
          } else
            utils::check(false, "read_block: Error reading input block in atoms_cart/atoms_frac");
        };
        utils::check(false,"EOF or bad read while reading atoms_cart/atoms_frac");
      };

      // begin atom_cart 
      bohr = false;
      found_atoms_cart = read_block<false>(file_data,"begin atoms_cart","",read);
      if(found_atoms_cart) { 
        double scl = (bohr ? 0.529177249 : 1.0);
        for(int i=0; i<nat; ++i) {
          rp[i]() *= scl;
          utils::check((nda::sum(nda::abs(at_cart_ang(i,all)-rp[i])) < 1e-4) and 
                       utils::string_equal(ids[i],species[mf.atomic_id(i)]),
                       "win: atoms_cart differ. i:{} CoQui:{}, {}, win:{}, {}",
                       i,species[mf.atomic_id(i)],at_cart_ang(i,all),ids[i],rp[i]);
        }
      }

      bohr = false;
      end_label = std::string("end atoms_frac");
      bool found = read_block<false>(file_data,"begin atoms_frac","",read);
      if(found) {
        utils::check(not found_atoms_cart, 
                     "Error reading win file - Found both atoms_cart and atoms_frac.");
        utils::check(not bohr, 
                     "Error reading win file - Found unit label in atoms_frac.");
        nda::stack_array<double,3> r_ = {0.0,0.0,0.0};
        for(int i=0; i<nat; ++i) {
          nda::blas::gemv(nda::transpose(lattv),rp[i],r_); 
          utils::check((nda::sum(nda::abs(at_cart_ang(i,all)-r_)) < 1e-4) and
                     utils::string_equal(ids[i],species[mf.atomic_id(i)]),
                     "win: atoms_cart differ. i:{} CoQui:{}, {}, win:{}, {}",
                     i,species[mf.atomic_id(i)],at_cart_ang(i,all),ids[i],r_);
        }
      }
    }

    // projections
    if(file_data_lower.find("begin projections") != std::string::npos) {
      utils::check(auto_projections==0, "wannier90_library_run: Found projections block with auto_projections>0.");
      std::vector<std::string> pstr; 
      bool bohr = false;
      auto read = [&](std::istringstream & iss)  {
        std::string line;
        if(std::getline(iss,line)) {
          if(io::tolower_copy(line).find("bohr") != std::string::npos) {
            utils::check(false,"Error: begin projections units should be Ang. Bohr not yet implemented.");
            bohr = true; 
            if(std::getline(iss,line)) {}
            else
              utils::check(false, "Errors reading projections block in win file");
          }
        } else
          utils::check(false, "Errors reading projections block in win file"); 
        while(io::tolower_copy(line).find("end projections") == std::string::npos) {
          pstr.emplace_back(line); 
          if(std::getline(iss,line)) {}
          else
            utils::check(false, "Problems reading projections block: EOF or getline failed.");
        }; 
      };
      bool found = read_block<false>(file_data,"begin projections","",read);

      if(found) {
        nproj_lines = pstr.size();
        for( auto & v : pstr ) max_len = std::max(max_len,int(v.length()));
        proj_str = nda::array<char,2>(nproj_lines,max_len); 
        str_len = nda::array<int,1>(nproj_lines); 
        for( auto [i,v] : itertools::enumerate(pstr) ) { 
          str_len(i) = v.length();
          std::copy(v.begin(),v.end(),proj_str(i,all).data());
        }  
      }
    }

    // exclude_bands
    exclude_string = read_key<false, std::string>(file_data, "exclude_bands", std::string{""});
    auto excl = read_range<false>(file_data,"exclude_bands",",");
    if(excl.size() > 0) {
      const int nbnd_ = mf.nbnd();
      int nexcl = std::count_if(excl.begin(), excl.end(), [&](auto && a) {return a>0 and a<=nbnd_;} );
      band_list = nda::arange<int>(mf.nbnd()-nexcl);
      nda::array<bool,1> b(nbnd_,true);
      for( auto n : excl ) 
        if( n > 0 and n <= nbnd_ ) b[n-1] = false; 
      int cnt=0;
      for( int i=0; i<nbnd_; ++i ) 
        if(b[i]) band_list(cnt++) = i;
      utils::check(cnt == mf.nbnd()-nexcl, "Logic error: Oh oh ");
    } else {
      band_list = nda::arange<int>(mf.nbnd());
    }
    utils::check(nband == band_list.size(), "Inconsistency between num_bands and exclude_bands: num_bands:{}, # bands according to exlude_bands:{}", nband,band_list.size());

    // print some values
    app_log(2,"num_wann: {}",nwann);
    app_log(2,"num_bands: {}",nband);
  }
  mpi.comm.broadcast_value(nband);
  mpi.comm.broadcast_value(nwann);
  mpi.comm.broadcast_value(nproj_lines);
  mpi.comm.broadcast_value(max_len);
  mpi.comm.broadcast_value(auto_projections);
  mpi.broadcast(kp_map);
  mpi.broadcast(wann_kp);
  if(not mpi.comm.root()) {
    band_list = nda::array<int,1>(nband);
    proj_str = nda::array<char,2>(nproj_lines,max_len);
    str_len = nda::array<int,1>(nproj_lines);
  }
  mpi.broadcast(band_list);
  mpi.broadcast(proj_str);
  mpi.broadcast(str_len);
 
  // eigenvalues with wann_kp ordering in eV
  auto eigv = get_eig(mpi,mf,resolved_prefix,kp_map,band_list,write_eigv);
  eigv() /= 3.674932540e-2;

  // nnkp array, returned by wann90_setup: 
  // nearest neighbors k-point pairs for construction of the overlap matrices between differnt k-points. 
  ::nda::array<int,3> nnkp; 

  // projection information, returned by wann90_setup
  // for each projector: l, mr, rad, s: 4 
  ::nda::array<int,2> proj_ints(nwann,4);
  // for each projector: site(3),zaxis(3),xaxis(3),sqa(3),zona(1) : 13
  ::nda::array<double,2> proj_doubles(nwann,13);

  // wannier90 setup, basically generates information found in nnkp file
  int err = 0;
  if(mpi.comm.root()) {
    ::nda::array<int,3> nnkp_(nkpts,n_neigh,4);
    int nn = 0;
#if defined(ENABLE_WANNIER90)
    FC_wann90_setup(resolved_prefix.data(),resolved_prefix.size(),nband,nwann,
                    nat,at_cart_ang.data(),at_sym.extent(1),at_sym.data(),at_sym_sz.data(), 
                    eigv.data(),lattv.data(),
                    mf.nkpts(),mf.kp_grid().data(),wann_kp.data(),  
                    nn, n_neigh, nnkp_.data(), 
                    auto_projections, nproj_lines, proj_str.extent(1), proj_str.data(), 
                    str_len.data(), proj_ints.data(), proj_doubles.data(),
                    write_nnkp, exclude_string.data(), exclude_string.length(), err);
#else
    utils::check(false,"Calling wann90_XXX without wannier90 support. Compile with ENABLE_WANNIER90.");
#endif
    // n_neigh too small
    if( err == 1001 ) {
      n_neigh = nn;
      nnkp.resize(std::array<long,3>{nkpts,n_neigh,4});
      // call again
#if defined(ENABLE_WANNIER90)
      FC_wann90_setup(resolved_prefix.data(),resolved_prefix.size(),nband,nwann,
                      nat,at_cart_ang.data(),at_sym.extent(1),at_sym.data(),at_sym_sz.data(),
                      eigv.data(),lattv.data(),
                      mf.nkpts(),mf.kp_grid().data(),wann_kp.data(),
                      nn, n_neigh, nnkp.data(),  
                      auto_projections, nproj_lines, proj_str.extent(1), proj_str.data(), 
                      str_len.data(), proj_ints.data(), proj_doubles.data(),
                      write_nnkp, exclude_string.data(), exclude_string.length(), err);
#else
      utils::check(false,"Calling wann90_XXX without wannier90 support. Compile with ENABLE_WANNIER90.");
#endif
      utils::check(err == 0, "Error returned by FC_wann90_setup: err:{}",err);
    } else {
      utils::check(err == 0, "Error returned by FC_wann90_setup: err:{}",err);
      n_neigh = nn;
      nnkp.resize(std::array<long,3>{nkpts,n_neigh,4});
      nnkp() = nnkp_(all,range(n_neigh),all);  
    } 
  }

  mpi.comm.broadcast_value(n_neigh);
  if(not mpi.comm.root()) 
    nnkp.resize(std::array<long,3>{nkpts,n_neigh,4});
  mpi.broadcast(nnkp);
  mpi.broadcast(proj_doubles);
  mpi.broadcast(proj_ints);
  mpi.comm.barrier();

  // generate proj vector
  std::vector<projection> proj;
  if(auto_projections == 0) { 
    proj.reserve(nwann);
    for(int i=0; i<nwann; i++) {
      // for each projector: site(3),zaxis(3),xaxis(3),sqa(3),zona(1) : 13
      // for each projector: l, mr, rad, s: 4 
      proj.emplace_back(proj_ints(i,0),proj_ints(i,1),proj_ints(i,2),proj_doubles(i,range(3)),
           proj_doubles(i,range(3,6)),proj_doubles(i,range(6,9)),proj_doubles(i,12));
    }
  }

  // We can now compute Mmn and Amn
  nda::array<ComplexType,4> Amn(nspin,nkpts,nwann,nband);

  /*
   * Compute Mmn
   */
  app_log(2, " - Computing orbital overlaps, Mmn");
  // transpose=true, since it will be passed to fortran
  auto Mmn = detail::compute_mmn(mpi,mf,resolved_prefix,kp_map,wann_kp,nnkp,band_list,true,write_mmn);
  mpi.comm.barrier();

  /*
   * Compute Amn
   */
  if(proj.size() > 0) {
    app_log(2, " - Computing initial projection matrix, Amn, with basis functions.");
    auto Amn_ = detail::compute_amn_projections(mpi,mf,pt,kp_map,wann_kp,band_list,proj,true,write_amn);
    Amn(0,nda::ellipsis{}) = Amn_.local();
  } else if(auto_projections > 0) {
    app_log(2, " - Computing initial projection matrix, Amn, with SCDM."); 
    auto Amn_ = scdm(mpi,mf,pt,auto_projections,kp_map,wann_kp,band_list,true,write_amn);
    Amn(0,nda::ellipsis{}) = Amn_.local();
  } else
    utils::check(false, "to_wannier90: No projections or auto_projections found in nnkp file.");
  mpi.comm.barrier();

  // Wannierize
  ::nda::array<double,3> wann_center(nspin,nwann,3);
  ::nda::array<double,2> wann_spreads(nspin,nwann);

  if(mpi.comm.root()) {
#if defined(ENABLE_WANNIER90)
    FC_wann90_run(resolved_prefix.data(),resolved_prefix.size(),nband,nwann,
                nat,at_cart_ang.data(),at_sym.extent(1),at_sym.data(),at_sym_sz.data(),
                eigv.data(),lattv.data(),
                mf.nkpts(),mf.kp_grid().data(),wann_kp.data(), n_neigh,
                auto_projections, nproj_lines, proj_str.extent(1), proj_str.data(), str_len.data(),
                Mmn.local().data(),Amn(0,nda::ellipsis{}).data(),
                wann_center.data(),wann_spreads.data(), 
                err);
#else
    utils::check(false,"Calling wann90_XXX without wannier90 support. Compile with ENABLE_WANNIER90.");
#endif
    utils::check(err == 0, "Error returned by FC_wann90_setup: err:{}",err);
  }

  // keep shared array Amn and return it
  mpi.broadcast(Amn);
  mpi.broadcast(wann_center);
  mpi.broadcast(wann_spreads);

  if(mpi.comm.root() and write_h5) {
    write_wan90_h5(mf,pt,band_list,eigv,Amn,wann_center);
  }
  mpi.comm.barrier();

  return std::make_tuple(Amn,wann_center,wann_spreads);
}

inline auto wannier90_library_run_from_files(utils::mpi_context_t<mpi3::communicator> &mpi, mf::MF &mf, ptree pt, 
      int n_neigh, int nwann, nda::array<int, 1> const& kp_map, nda::array<double,2> const& wann_kp, 
      nda::array<int,1> const& band_list, bool write_to_file) 
{
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto prefix = io::get_value<std::string>(pt,"prefix");
  auto resolved_prefix = outdir + "/" + prefix;
  int nband = band_list.size();  
  auto nspin = mf.nspin();
  ::nda::array<double,2> lattv = mf.lattv()*0.529177210544;
  int ierr=0;
  // keeping this in local memory for now
  ::nda::array<std::complex<double>,3> Pkam(mf.nkpts(),nwann,nband);
  ::nda::array<double,3> wann_center(nspin,nwann,3);
  ::nda::array<double,2> wann_spreads(nspin,nwann);

  if(mpi.comm.root()) {
    auto eigv = get_eig(mpi,mf,resolved_prefix,kp_map,band_list,false);
    eigv() /= 3.674932540e-2;
#if defined(ENABLE_WANNIER90)
    FC_wann90_run_from_files(resolved_prefix.c_str(),resolved_prefix.size(),nband,nwann,eigv.data(),lattv.data(),
                       mf.nkpts(),mf.kp_grid().data(),wann_kp.data(),n_neigh,
                       Pkam.data(),wann_center.data(),wann_spreads.data(),ierr);
#else
    utils::check(false,"Calling wann90_XXX without wannier90 support. Compile with ENABLE_WANNIER90.");
#endif
    utils::check(ierr == 0, "Error returned by FC_wann90_run_from_files: ierr:{}",ierr);

    if(write_to_file) {  
    }
  }

  mpi.broadcast(Pkam);
  mpi.broadcast(wann_center);
  mpi.broadcast(wann_spreads);

  return std::make_tuple(Pkam,wann_center,wann_spreads);
}

/**
 * Implementation of wannier_from_standalone_output: read Wannier90 standalone outputs and generate HDF5
 */
inline void mlwf_h5_from_wannier90_output_impl(
  utils::mpi_context_t<mpi3::communicator> &mpi, mf::MF &mf, ptree &pt)
{
  auto outdir = io::get_value_with_default<std::string>(pt, "outdir", "./");
  auto prefix = io::get_value<std::string>(pt, "prefix");
  auto resolved_prefix = outdir + "/" + prefix;
  
  if(!mpi.comm.root()) return;

  // =======================================================================
  // Check for required files and determine disentanglement
  // =======================================================================
  utils::check(std::filesystem::exists(resolved_prefix + "_u.mat"),
               "wan90_aux.hpp::mlwf_h5_from_wannier90_output_impl: U matrix file not found: {}_u.mat",
               resolved_prefix);
  utils::check(std::filesystem::exists(resolved_prefix + "_centres.xyz"),
               "wan90_aux.hpp::mlwf_h5_from_wannier90_output_impl: Centres file not found: {}_centres.xyz",
               resolved_prefix);

  // Auto-determine disentanglement based on file existence
  bool use_disentangle = std::filesystem::exists(resolved_prefix + "_u_dis.mat");

  app_log(2, "All required input files found.\n");

  // =======================================================================
  // Read win file to extract nwann, nband, and band_list
  // =======================================================================
  auto file_data = utils::read_file_to_string(resolved_prefix + ".win");
  auto file_data_lower = io::tolower_copy(file_data);

  int nwann = read_key<true, int>(file_data, "num_wann");
  int nband = read_key<false, int>(file_data, "num_bands", nwann);

  // Check consistency: if no disentanglement, nwann must equal nband
  if(!use_disentangle) {
    utils::check(nwann == nband,
                 "wan90_aux.hpp::mlwf_h5_from_wannier90_output_impl: Without disentanglement, num_wann ({}) must equal num_bands ({})",
                 nwann, nband);
  }

  // Read exclude_bands to construct band_list (same as wannier90_library_run)
  nda::array<int, 1> band_list;
  auto excl = read_range<false>(file_data, "exclude_bands", ",");
  if(excl.size() > 0) {
    const int nbnd_ = mf.nbnd();
    int nexcl = std::count_if(excl.begin(), excl.end(), [&](auto && a) { return a > 0 and a <= nbnd_; });
    band_list = nda::arange<int>(mf.nbnd() - nexcl);
    nda::array<bool, 1> b(nbnd_, true);
    for(auto n : excl)
      if(n > 0 and n <= nbnd_) b[n - 1] = false;
    int cnt = 0;
    for(int i = 0; i < nbnd_; ++i)
      if(b[i]) band_list(cnt++) = i;
    utils::check(cnt == mf.nbnd() - nexcl, "Logic error in band_list construction");
  } else {
    band_list = nda::arange<int>(mf.nbnd());
  }
  utils::check(nband == band_list.size(),
               "wan90_aux.hpp::mlwf_h5_from_wannier90_output_impl: Inconsistency between num_bands and exclude_bands: num_bands={}, # bands from exclude_bands={}",
               nband, band_list.size());

  // =======================================================================
  // Read eigenvalues
  // =======================================================================
  auto nkpts = mf.nkpts();
  auto nspin = mf.nspin();
  nda::array<double, 3> eigv;

  if(std::filesystem::exists(resolved_prefix + ".eig")) {
    app_log(2, "Reading eigenvalues from {}.eig...\n", resolved_prefix);
    // When reading from standalone Wannier90 output, the .eig file contains eigenvalues
    // for Wannier bands (1 to nwann), not absolute band indices from DFT.
    // Create a temporary band_list for the Wannier bands
    auto wan_band_list = nda::arange<int>(nband);
    eigv = read_eig_file(resolved_prefix + ".eig", nkpts, wan_band_list);
  } else {
    app_log(2, "Eigenvalue file not found, generating from DFT calculation\n");
    auto kp_map = nda::arange<int>(nkpts);
    eigv = get_eig(mpi, mf, resolved_prefix, kp_map, band_list, false);
    // Convert Hartree -> eV for HDF5 output
    eigv() /= 3.674932540e-2;
  }

  // =======================================================================
  // Read U matrix
  // =======================================================================
  app_log(2, "Reading U matrix from {}_u.mat...\n", resolved_prefix);
  auto U_matrix = read_u_matrix_file(resolved_prefix + "_u.mat", nkpts, nwann);

  // =======================================================================
  // Read U_dis matrix if present
  // =======================================================================
  nda::array<ComplexType, 3> U_dis_matrix;
  if(use_disentangle) {
    app_log(2, "Reading U_dis matrix from {}_u_dis.mat...\n", resolved_prefix);
    U_dis_matrix = read_u_dis_matrix_file(resolved_prefix + "_u_dis.mat", nkpts, nband, nwann);
  }

  // =======================================================================
  // Read Wannier centres
  // =======================================================================
  app_log(2, "Reading Wannier centres from {}_centres.xyz...", resolved_prefix);
  auto wan_centres = read_wan_centres_file(resolved_prefix + "_centres.xyz", nwann);

  // =======================================================================
  // Construct projection matrix Pkwa = U_dis * U (or just U if no disentanglement)
  // =======================================================================
  nda::array<ComplexType, 4> Pkwa(nspin, nkpts, nwann, nband);
  Pkwa() = ComplexType(0.0);

  if(use_disentangle) {
    // Pkwa(s, k, w, b) = sum_w' U_dis(k, b, w') * U(k, w', w)
    auto all = nda::range::all;
    nda::array<ComplexType, 2> tmp(nband, nwann);
    for(long ik = 0; ik < nkpts; ik++) {
      // tmp(b, w) = (U_dis * U)(b, w)
      nda::blas::gemm(ComplexType(1.0), U_dis_matrix(ik, all, all), U_matrix(ik, all, all),
                      ComplexType(0.0), tmp);
      // transpose to match Pkwa(w, b)
      Pkwa(0, ik, nda::ellipsis{}) = nda::transpose(tmp);
    }
  } else {
    // No disentanglement: Pkwa = U^T
    // U_matrix shape: (nkpts, nwann, nwann)
    // Pkwa(0, :, :, :) shape: (nkpts, nwann, nband)
    // Note: Wannier90 standalone output stores U in transposed form compared to library mode
    for(long ik = 0; ik < nkpts; ik++) {
      Pkwa(0, ik, nda::ellipsis{}) = nda::transpose(U_matrix(ik, nda::ellipsis{}));
    }
  }

  // =======================================================================
  // Prepare Wannier centres array (reshaped for write_wan90_h5)
  // =======================================================================
  nda::array<double, 3> R_wan(nspin, nwann, 3);
  R_wan(0, nda::ellipsis{}) = wan_centres;

  // =======================================================================
  // Call write_wan90_h5 to generate HDF5 output
  // =======================================================================
  write_wan90_h5(mf, pt, band_list, eigv, Pkwa, R_wan);

}

} // detail

}

#endif
