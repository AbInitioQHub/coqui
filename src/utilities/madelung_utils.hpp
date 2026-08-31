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


#ifndef COQUI_MADELUNG_UTILS_HPP
#define COQUI_MADELUNG_UTILS_HPP

#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"

namespace utils
{
/*
 * Routines to calculate Madelung constant
 */

auto estimate_ewald_params(int ndim, 
                           [[maybe_unused]] nda::ArrayOfRank<2> auto const& lattv, 
                           nda::ArrayOfRank<2> auto const& recv, 
                           nda::ArrayOfRank<1> auto const& fft_mesh, 
                           double prec=1e-8)
  -> std::tuple<double, double> {
  double alpha = 0.0, r_cut = 0.0;
  if( ndim == 2 ) { 
    double Gmax = std::min( fft_mesh(0)/2 * std::sqrt(recv(0,0)*recv(0,0) + recv(0,1)*recv(0,1)),
                            fft_mesh(1)/2 * std::sqrt(recv(1,0)*recv(1,0) + recv(1,1)*recv(1,1)) ); 
    // this is approximate right now, using the fact that exp(-x*x) > erfc(x)
    double log_prec = std::log(prec / (4*M_PI*std::pow(Gmax+1e-100, 2)) );
    //double log_prec = std::log(prec * std::pow(Gmax+1e-100, 2) / (4*M_PI) );
    alpha = std::sqrt( -std::pow(Gmax, 2) / (4*log_prec) ) + 1e-100;
    r_cut = std::sqrt( -std::log(prec) / (alpha*alpha) );
  } else if( ndim == 3 ) {
    double Gmax = fft_mesh(0)/2 * std::sqrt(recv(0,0)*recv(0,0) + recv(0,1)*recv(0,1) + recv(0,2)*recv(0,2));
    for (long d = 1; d < 3; ++d) {
      double Gmax_tmp = fft_mesh(d)/2 * std::sqrt(recv(d,0)*recv(d,0) + recv(d,1)*recv(d,1) + recv(d,2)*recv(d,2));
      Gmax = std::min(Gmax, Gmax_tmp);
    }
    double log_prec = std::log(prec / (4*M_PI*std::pow(Gmax+1e-100, 2)) );
    //double log_prec = std::log(prec * std::pow(Gmax+1e-100, 2) / (4*M_PI) );
    alpha = std::sqrt( -std::pow(Gmax, 2) / (4*log_prec) ) + 1e-100;
    r_cut = std::sqrt( -std::log(prec) / (alpha*alpha) );
  } else {
    APP_ABORT(" Error in estimate_ewald_params: Invalid ndim:{}",ndim);
  }

  return std::make_tuple(alpha, r_cut);
}


template<nda::ArrayOfRank<2> MatA, nda::ArrayOfRank<2> MatB, nda::ArrayOfRank<1> Vec_MP, nda::ArrayOfRank<1> Vec_fft>
double madelung(MatA const& lattv, MatB const& recv, Vec_MP const& mp_mesh, Vec_fft const& fft_mesh, double prec=1e-8) {

  int ndim = lattv.extent(0);
  utils::check( ndim == 2 or ndim == 3, "madelung: Only 2 or 3 dimensions allowed");
  utils::check( lattv.shape() == std::array<long,2>{ndim,ndim}, "madelung: lattv shape mismatch");
  utils::check( recv.shape() == std::array<long,2>{ndim,ndim}, "madelung: recv shape mismatch");
  utils::check( mp_mesh.size() == ndim, "madelung: mp_mesh size mismatch");
  utils::check( fft_mesh.size() == ndim, "madelung: fft_mesh size mismatch");

  // scale to the supercell
  nda::array<double, 2> lattv_sc(ndim,ndim);
  nda::array<double, 2> recv_sc(ndim,ndim);
  nda::array<int, 1> fft_mesh_sc(ndim);
  for (long d = 0; d < ndim; ++d) {
    lattv_sc(d,nda::range::all) = lattv(d,nda::range::all) * mp_mesh(d);
    recv_sc(d,nda::range::all) = recv(d,nda::range::all) / mp_mesh(d);
    fft_mesh_sc(d) = fft_mesh(d) * mp_mesh(d);
  }

  double volume = 0.0;
  if( ndim == 2 ) 
    volume = std::abs(lattv_sc(0,0) * lattv_sc(1,1) - lattv_sc(1,0) * lattv_sc(0,1));
  else if( ndim == 3 ) 
    volume = std::abs(lattv_sc(0,0) * ( lattv_sc(1,1)*lattv_sc(2,2) - lattv_sc(1,2)*lattv_sc(2,1) ) -
                           lattv_sc(0,1) * ( lattv_sc(1,0)*lattv_sc(2,2) - lattv_sc(1,2)*lattv_sc(2,0) ) +
                           lattv_sc(0,2) * ( lattv_sc(1,0)*lattv_sc(2,1) - lattv_sc(1,1)*lattv_sc(2,0) ));

  auto [alpha, rcut] = estimate_ewald_params(ndim, lattv_sc, recv_sc, fft_mesh_sc, prec);

  // constant term
  double vlr_r0 = 0.5 * (2*alpha) / std::sqrt(M_PI);
  double vsr_k0 = ( ndim == 2 ? std::sqrt(M_PI) / (volume*alpha)  : 0.5 * M_PI / (volume*alpha*alpha) );

  // long-range contribution
  double e_lr = 0.0;
  nda::array<double, 1> G(ndim);
  for (long i = -fft_mesh_sc(0)/2; i < fft_mesh_sc(0)/2; ++i) {
    for (long j = -fft_mesh_sc(1)/2; j < fft_mesh_sc(1)/2; ++j) {
      if(ndim == 2) {
        if (i == 0 and j == 0) {
          continue;
        }
        G() = i*recv_sc(0,nda::range::all) + j*recv_sc(1,nda::range::all); 
        double absG = std::sqrt(nda::blas::dot(G, G));
        e_lr += 0.5 * (2*M_PI / absG) * std::erfc(absG / (2*alpha) );
      } else if( ndim == 3) {
        for (long k = -fft_mesh_sc(2)/2; k < fft_mesh_sc(2)/2; ++k) {
          if (i == 0 and j == 0 and k == 0) {
            continue;
          }
          G() = i*recv_sc(0,nda::range::all) + j*recv_sc(1,nda::range::all) + k*recv_sc(2,nda::range::all);
          double absG2 = nda::blas::dot(G, G);
          //app_log(2, "absG2 = {}", absG2);
          e_lr += 0.5 * (4*M_PI / absG2) * std::exp(-absG2 / (4*alpha*alpha) );
        }
      }
    }
  }
  e_lr /= volume;

  // short-range contribution
  double e_sr = 0.0;
  // TODO better estimation is needed
  int nimg = 8;
  nda::array<double, 1> R(ndim);
  for (long i = -nimg; i < nimg; ++i) {
    for (long j = -nimg; j < nimg; ++j) {
      if(ndim == 2) {
        if (i == 0 and j == 0) {
          continue;
        }
        R = i*lattv_sc(0,nda::range::all) + j*lattv_sc(1,nda::range::all); 
        double absR = std::sqrt(nda::blas::dot(R, R) );
        e_sr += 0.5 * std::erfc(alpha*absR) / absR;
      } else if( ndim == 3) {
        for (long k = -nimg; k < nimg; ++k) {
          if (i == 0 and j == 0 and k == 0) {
            continue;
          }
          R = i*lattv_sc(0,nda::range::all) + j*lattv_sc(1,nda::range::all) + k*lattv_sc(2,nda::range::all);
          double absR = std::sqrt(nda::blas::dot(R, R) );
          e_sr += 0.5 * std::erfc(alpha*absR) / absR;
        }
      }
    }
  }

  app_log(4, "Madelung constant: ");
  app_log(4, "  - ndim = {}", ndim);
  app_log(4, "  - alpha = {}", alpha);
  app_log(4, "  - contributions: {}, {}, {}", -(vlr_r0+vsr_k0), e_lr, e_sr);

  return e_sr + e_lr - (vlr_r0+vsr_k0);
}

}


#endif //COQUI_MADELUNG_UTILS_HPP
