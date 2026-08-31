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


#ifndef UTILITIES_QE_UTILITIES_HPP
#define UTILITIES_QE_UTILITIES_HPP

#include <complex>
#include <iostream>
#include <fstream>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "nda/nda.hpp"

namespace utils
{

inline void read_qe_plot_file(int plot_num, std::string fname,
			      nda::ArrayOfRank<1> auto const& mesh, 
			      nda::ArrayOfRank<1> auto& vloc)
{
  utils::check(plot_num >= 0 and plot_num <= 2, "Error in read_qe_plot_file: Invalid plot_num: {}",plot_num);

  std::string title;

  std::ifstream in(fname);
  utils::check(not in.fail(),"Error opening vloc txt-file: {}",fname);

  std::getline (in,title);
  utils::check(not in.fail(),"I/O error in vloc txt-file reader.");
  int nr1x,nr2x,nr3x,nr1,nr2,nr3,nat,ntyp,ibrav,plot_num_in;
  int idum;
  double ddum;

  in >>nr1x >>nr2x >>nr3x >>nr1 >>nr2 >>nr3 >>nat >>ntyp;
  utils::check(not in.fail(),"I/O error in vloc txt-file reader.");
  utils::check(nr1 == mesh(0),
    "Error vloc: FFT mesh (x) inconsistent with MF object: MF: {}, vloc: {}",mesh(0),nr1);
  utils::check(nr2 == mesh(1),
    "Error vloc: FFT mesh (y) inconsistent with MF object: MF: {}, vloc: {}",mesh(1),nr2);
  utils::check(nr3 == mesh(2),
    "Error vloc: FFT mesh (z) inconsistent with MF object: MF: {}, vloc: {}",mesh(2),nr3);
  in >>ibrav;
  utils::check(not in.fail(),"I/O error in vloc txt-file reader.");
  for(int i=0; i<6; ++i) in >> ddum; // don't need
  if(ibrav==0)
    for(int i=0; i<9; ++i) in >> ddum;   //( at(ipol,i),ipol=1,3 ) 
  in >>ddum >>ddum >>ddum >>plot_num_in;  //gcutm, dual, ecut, plot_num
  utils::check(plot_num == plot_num_in, "Error: requested plot_num ({}) does not match the one in the file: {}",
					plot_num,plot_num_in); 
  utils::check(not in.fail(),"I/O error in vloc txt-file reader.");
  //(nt, atm (nt), zv (nt), nt=1, ntyp)
  for(int i=0; i<ntyp; i++) {
    in>>idum;
    std::getline (in,title);
    utils::check(not in.fail(),"I/O error in vloc txt-file reader.");
  }
  // (na,(tau (ipol, na), ipol = 1, 3), ityp (na), na = 1, nat) 
  for(int i=0; i<nat; i++)
    in>>idum >>ddum >>ddum >>ddum >>idum;
  utils::check(not in.fail(),"I/O error in vloc txt-file reader.");

  vloc.resize(mesh(0)*mesh(1)*mesh(2));
  auto v3d = nda::reshape(vloc,std::array<long,3>{mesh(0),mesh(1),mesh(2)});
  auto ix = nda::range(0,mesh(0));
  nda::array<double,1> tmp(nr1x);

  // array in column-major format on file
  //plot (ir) , ir = 1, nr1x * nr2x * nr3)      
  for(int k=0; k<nr3x; k++)
    for(int j=0; j<nr2x; j++) {
      for(int i=0; i<nr1x; i++) in >>tmp(i);
      utils::check(not in.fail(),"I/O error in vloc txt-file reader.");
      v3d(nda::range::all,j,k) = tmp(ix);
    }
}

}
#endif
