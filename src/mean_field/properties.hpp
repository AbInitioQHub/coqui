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


#ifndef MEANFIELD_PROPERTIES_HPP 
#define MEANFIELD_PROPERTIES_HPP 

#include "IO/AppAbort.hpp"
#include "utilities/check.hpp"

#include <complex>

#include "mean_field/MF.hpp"

#include "mpi3/communicator.hpp"

#include "nda/nda.hpp"
#include "numerics/fft/nda.hpp"
#include "numerics/distributed_array/nda.hpp"

namespace mf
{

// calculates the charge density associated with the object mf and returns
// a distributed array defined by the given processor grid and communicator.
auto distributed_charge_density(MF &mf, boost::mpi3::communicator &comm, 
			        std::array<long,3> pgrid )
{
  utils::check(mf.has_orbital_set(), "Error in distributed_charge_density: Invalid mf type. ");
  int nnr = mf.fft_grid_dim(0)*mf.fft_grid_dim(1)*mf.fft_grid_dim(2); 
  // accumulate local work on full grid  
  auto grho = nda::array<RealType, 3>::zeros({mf.fft_grid_dim(0),
					      mf.fft_grid_dim(1),
					      mf.fft_grid_dim(2)}); 
  auto grho1D = nda::reshape(grho, std::array<long,1>{nnr});
  nda::array<ComplexType, 1> Psi(nnr,ComplexType(0.0));
  utils::check(mf.nspin() == 1, "finish!!!");
// MAM: need to fix spin index interface in MF object!
  
  // very simple for now!!!
  // not optimal!
  for(long is=0,cnt=0; is<mf.nspin(); ++is) 
    for(long ik=0; ik<mf.nkpts(); ++ik) 
      for(long n=0; n<mf.nbnd(); ++n) { 
        //if(mf.occ(is,ik,n) > 1e-6) {
        auto w = mf.occ(is,ik,n);
        if(w > 1e-6) {
          // simple round-robin
          if(cnt%comm.size() == comm.rank()) {
            mf.get_orbital('r',is,ik,n,Psi);
            w /= double(nnr);
            for(auto i : itertools::range(nnr)) {
              grho1D(i) += w * std::norm(Psi(i));
            }
          }
          ++cnt;
        }
      }

  // reduce
  comm.all_reduce_in_place_n(grho.data(),grho.size(),std::plus<>{});

  // distributed container
  auto drho = math::nda::make_distributed_array<nda::array<RealType,3>>(comm, 
			pgrid, {mf.fft_grid_dim(0),mf.fft_grid_dim(1),mf.fft_grid_dim(2)});
   
  // copy to local array
  auto rho = drho.local();
  rho(nda::ellipsis{}) = grho(drho.local_range(0),drho.local_range(1),drho.local_range(2));
  return drho;
}

auto distributed_charge_density(MF &mf, grids::distributed_r_grid & g) 
{
  auto& a = g.dgrid(); 
  auto comm = a.communicator();
  return distributed_charge_density(mf,*comm,
		std::array<long,3>{a.grid()[0],a.grid()[1],a.grid()[2]});
}

} // mf

#endif
