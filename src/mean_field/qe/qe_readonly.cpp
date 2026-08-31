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


#include "qe_readonly.hpp"

namespace mf::qe {
namespace detail {

  grids::truncated_g_grid wfc_grid_from_h5(qe_system const& sys) {
    nda::stack_array<int, 3> mesh;
    mesh()=0;
    utils::check(sys.nspin==1 or sys.nspin==2, "Error: Invalid nspin:{}",sys.nspin);
    //nda::array<ComplexType,1> *Xft = nullptr;
    nda::stack_array<double, 3> Gs;
    Gs() = 0; // in case PBC shift is needed
    nda::array<int,2> miller;
    double ecut = 0.0;
    for( long ik = sys.mpi->comm.rank(); ik < sys.bz().nkpts; ik += sys.mpi->comm.size() ) {
      if(sys.nspin==1) {
        h5::file f(sys.outdir+"/"+sys.prefix+".save/wfc"+std::to_string(sys.bz().kp_to_ibz(ik)+1)+".hdf5",'r');
        h5::group g(f);
        auto l = h5::array_interface::get_dataset_info(g,"/MillerIndices");
        if(miller.extent(0) != l.lengths[0]) miller = nda::array<int,2>::zeros({l.lengths[0],3});
        h5_read(g, "/MillerIndices", miller);
      } else if(sys.nspin==2) {
        // assuming MillerIndices are spin independent!!!
        h5::file f(sys.outdir+"/"+sys.prefix+".save/wfcup"+std::to_string(sys.bz().kp_to_ibz(ik)+1)+".hdf5",'r');
        h5::group g(f);
        auto l = h5::array_interface::get_dataset_info(g,"/MillerIndices");
        if(miller.extent(0) != l.lengths[0]) miller = nda::array<int,2>::zeros({l.lengths[0],3});
        h5_read(g, "/MillerIndices", miller);
      }
      if( ik >= sys.bz().nkpts_ibz )
        utils::transform_miller_indices(sys.bz().kp_trev(ik),
                                        sys.bz().symm_list[sys.bz().kp_symm(ik)],Gs,miller);
      for( auto n : nda::range(miller.extent(0)) ) {
        if(std::abs(miller(n,0)) > mesh(0)) mesh(0) = std::abs(miller(n,0));
        if(std::abs(miller(n,1)) > mesh(1)) mesh(1) = std::abs(miller(n,1));
        if(std::abs(miller(n,2)) > mesh(2)) mesh(2) = std::abs(miller(n,2));
        double n0(miller(n,0));
        double n1(miller(n,1));
        double n2(miller(n,2));
        double gx = n0*sys.recv(0,0)+n1*sys.recv(1,0)+n2*sys.recv(2,0);
        double gy = n0*sys.recv(0,1)+n1*sys.recv(1,1)+n2*sys.recv(2,1);
        double gz = n0*sys.recv(0,2)+n1*sys.recv(1,2)+n2*sys.recv(2,2);
        double ec = gx*gx+gy*gy+gz*gz;
        ecut = std::max(ecut,ec);
      }
    }
    sys.mpi->comm.all_reduce_in_place_n(mesh.data(),3,boost::mpi3::max<>{});
    sys.mpi->comm.all_reduce_in_place_n(&ecut,1,boost::mpi3::max<>{});
    ecut *= 0.5+0.00001;
    // mesh needs to accomodate to selected symmetries.
    // fix fix fix!
    mesh() = 2*mesh()+1;

    // check that Rinv is compatible with mesh
    if(sys.mpi->comm.root())
      mesh = utils::generate_consistent_fft_mesh(mesh,sys.bz().symm_list,1e-6,"qe_readonly");
    sys.mpi->comm.broadcast_n(mesh.data(),3);

    return grids::truncated_g_grid(ecut,mesh,sys.recv,true);
  }


} // detail


} // mf::qe