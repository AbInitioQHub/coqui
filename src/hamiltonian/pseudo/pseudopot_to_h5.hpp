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


#ifndef HAMILTONIAN_PSEUDOPOT_TO_H5_HPP
#define HAMILTONIAN_PSEUDOPOT_TO_H5_HPP

#include <string>

#include "configuration.hpp"
#include "hamiltonian/pseudo/pseudopot_type.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "mean_field/mf_source.hpp"

namespace hamilt
{

namespace detail
{

template<typename T>
void h5_read_write_attribute(h5::group& r_grp, h5::group& w_grp, std::string name)
{
  T a;
  h5::h5_read_attribute(r_grp,name,a);
  h5::h5_write_attribute(w_grp,name,a);
}

template<typename T>
void h5_read_write_attribute(h5::group& r_grp, h5::group& w_grp, std::string name, T& a)
{
  h5::h5_read_attribute(r_grp,name,a);
  h5::h5_write_attribute(w_grp,name,a);
}


}

void pseudopot_to_h5(nda::ArrayOfRank<1> auto const&fft_mesh, h5::group& grp0, 
                     std::string input_file_name, mf::mf_input_file_type_e input_file_type) 
{
  using detail::h5_read_write_attribute;
  if(not grp0.has_subgroup("Hamiltonian") )
    grp0.create_group("Hamiltonian");
  h5::group grp1 = grp0.open_group("Hamiltonian");

  { 
    // if dataset or attribute exist, do nothing
    std::string attr = "";
    h5::h5_read_attribute(grp1, "pp_type", attr);
    if(attr != "") {
      app_warning("********************************************************");
      app_warning("*   Hamiltonian:pp_type attribute exists in h5 file.   ");
      app_warning("*   Existing pp_type: {}                               ",attr);
      app_warning("*   Not overwriting pseudopotential.                   ");
      app_warning("********************************************************");
      return;
    }
  }

  if( input_file_type == mf::xml_input_type )
  {
    // MAM: this interface is limited to spin independent pseudopotentials
    //      For collinear or noncollinear calculations, use h5 interface.
    std::string type("ncpp");

    if( grp1.has_subgroup(type) ) {
      app_warning("**********************************************************");
      app_warning("*   h5 file already contains pseudopotential information  ");
      app_warning("*   Existing pseudopotential dataset Hamiltonian/{}       ",type);
      app_warning("*   Not overwriting pseudopotential.                      ");
      app_warning("**********************************************************");
      return;
    }

    h5::h5_write_attribute(grp1, "pp_type", type);
    h5::group grp = grp1.create_group(type);

    std::string fname = input_file_name + "/VKB";
    utils::check(std::filesystem::exists(fname),"Error: Missing file: {}",fname);
    int sz = fname.size();
    int npwx,nkb,nat,nhm,nsp,ierr,nspin,nk,npol=1;
    FC_read_pw2bgw_vkbg_header(fname.c_str(), sz, nspin, nkb, npwx, nk, nat, nsp, nhm, ierr);
    utils::check(ierr==0, "Error reading QE::pw2bgw vkbg file (header info).");
    h5::h5_write_attribute(grp,"number_of_nspins",nspin);
    h5::h5_write_attribute(grp,"number_of_polarizations",npol);
    h5::h5_write_attribute(grp,"number_of_kpoints",nk);
    h5::h5_write_attribute(grp,"total_num_of_proj",nkb);
    h5::h5_write_attribute(grp,"number_of_atoms",nat);
    h5::h5_write_attribute(grp,"number_of_species",nsp);
    h5::h5_write_attribute(grp,"max_proj_per_atom",nhm);
    h5::h5_write_attribute(grp,"max_npw",npwx);
    h5::h5_write_attribute(grp,"lspinorbit_nl",0);
    h5::h5_write_attribute(grp,"lspinorbit_loc",0);

    {
      nda::array<int,1> ityp(nat);
      nda::array<int,1> nh(nsp);
      nda::array<int,1> ngk(nk);
      nda::array<int,3> miller(nk,npwx,3);
      nda::array<ComplexType,3> vkb_r(nk,nkb,npwx);
      int msz = int(nk*npwx*3), vsz=int(nk*nkb*npwx), Dsz = int(nspin*nsp*nhm*nhm);
      int k0=0, k1=nk;
      nda::array<double,4> Dnn_r(nspin,nsp,nhm,nhm);
      FC_read_pw2bgw_vkbg(fname.c_str(), sz, k0, k1, ityp.data(), nat,
          nh.data(), nsp, ngk.data(), nk,
          Dnn_r.data(), Dsz, miller.data(), msz, vkb_r.data(), vsz, ierr);
      utils::check(ierr==0, "Error reading QE::pw2bgw vkbg file (arrays) - ierr:{}",ierr);

      nda::array<int,1> ofs(nat);
      long iofs=0;
      for( auto [i,n] : itertools::enumerate(nh) )
        for( auto iat : nda::range(nat) )
          if(ityp(iat) == i) {
            ofs(iat) = iofs;
            iofs += n;
          }
      ityp += 1;
      nda::h5_write(grp,"atomic_id",ityp);
      nda::h5_write(grp,"projector_offset",ofs);
      nda::h5_write(grp,"proj_per_atom",nh);

      nda::h5_write(grp,"npw",ngk);
      // lspinorb_nl == 0 right now!!!
      auto Dnn0 = Dnn_r(0,nda::ellipsis{});
      nda::h5_write(grp,"dion",Dnn0);

      for(int ik=0; ik<nk; ++ik) {
        auto m = miller(ik,nda::ellipsis{});
        nda::h5_write(grp,"miller_k"+std::to_string(ik),m);
        nda::array<ComplexType,2> vkb(vkb_r(ik,nda::ellipsis{}));
        nda::h5_write(grp,"projector_k"+std::to_string(ik),vkb);
      }
    }
    {
      long nnr = fft_mesh(0)*fft_mesh(1)*fft_mesh(2);
      // since local potentials were read in r-space, writing full g-space 
      // fft_mesh
      {
        auto k2g = nda::arange(nnr);
        nda::array<int,2> mill(nnr,3);
        utils::generate_miller_index(k2g,mill,fft_mesh);
        h5::h5_write_attribute(grp,"ngm",int(nnr));
        nda::h5_write(grp,"miller_g",mill);
      }

      if(std::filesystem::exists(input_file_name+"/VSC")) {
        nda::array<ComplexType,1> v;
        utils::read_qe_plot_file(1,input_file_name+"/VSC",fft_mesh,v);
        // MAM: transform to nnr grid...
        utils::check(v.extent(0) == nnr, "Error: Dimension mismatch");
        // spin dependence???
        nda::array<ComplexType,3> vl(nspin,1,nnr);
        vl(0,0,nda::range::all) = v;
        if(nspin==2) vl(1,0,nda::range::all) = v;
        auto v4D = nda::reshape(vl,std::array<long,4>{nspin,fft_mesh(0),fft_mesh(1),fft_mesh(2)});
        math::nda::fft<true> F(v4D);
        F.forward(v4D);
        nda::h5_write(grp,"scf_local_potential",vl);
      } else {
        app_warning(" pseutopot_to_h5: Missing VSC file!!! Empty array will be written!");
        nda::array<ComplexType,3> vl(nspin,1,nnr);
        vl() = ComplexType(0.0);
        nda::h5_write(grp,"scf_local_potential",vl);
      }
      
      if(std::filesystem::exists(input_file_name+"/VLTOT")) {
        nda::array<ComplexType,1> vl;
        utils::read_qe_plot_file(2,input_file_name+"/VLTOT",fft_mesh,vl);
        // MAM: transform to nnr grid...
        utils::check(vl.extent(0) == nnr, "Error: Dimension mismatch");
        auto v3D = nda::reshape(vl,std::array<long,3>{fft_mesh(0),fft_mesh(1),fft_mesh(2)});
        math::nda::fft<false> F(v3D);
        F.forward(v3D);
        nda::h5_write(grp,"pp_local_component",vl);
      } else {
        app_warning(" pseutopot_to_h5: Missing VLTOT file!!! Empty array will be written!");
        nda::array<ComplexType,1> vl(nnr);
        vl() = ComplexType(0.0);
        nda::h5_write(grp,"pp_local_component",vl);
      }
    }
  } else if( input_file_type == mf::h5_input_type ) {

    std::string type;
    h5::file file_;
    try {
      file_ = h5::file(input_file_name, 'r');
    } catch(...) {
      APP_ABORT("Failed to open h5 file: {}, mode:r",input_file_name);
    }
    h5::group grp0_(file_);
    h5::group grp1_ = grp0_.open_group("Hamiltonian");
    h5::h5_read_attribute(grp1_, "pp_type", type);
    h5::group grp_ = grp1_.open_group(type);

    if( grp1.has_subgroup(type) ) {
      app_warning("**********************************************************");
      app_warning("*   h5 file already contains pseudopotential information  ");
      app_warning("*   Existing pseudopotential dataset Hamiltonian/{}       ",type);
      app_warning("*   Not overwriting pseudopotential.                      ");
      app_warning("**********************************************************");
      return;
    }

    h5::h5_write_attribute(grp1, "pp_type", type);
    h5::group grp = grp1.create_group(type);

    int nspin, nk, npwx, npol, nat, nsp, nkb, nhm, ngm, lspinorb_nl, lspinorb_loc;
    h5_read_write_attribute<int>(grp_,grp,"number_of_nspins",nspin);
    h5_read_write_attribute<int>(grp_,grp,"number_of_polarizations",npol);
    h5_read_write_attribute<int>(grp_,grp,"number_of_kpoints",nk);
    h5_read_write_attribute<int>(grp_,grp,"max_npw",npwx);
    h5_read_write_attribute<int>(grp_,grp,"lspinorbit_nl",lspinorb_nl);
    h5_read_write_attribute<int>(grp_,grp,"lspinorbit_loc",lspinorb_loc);
    h5_read_write_attribute<int>(grp_,grp,"number_of_atoms",nat);
    h5_read_write_attribute<int>(grp_,grp,"number_of_species",nsp);
    h5_read_write_attribute<int>(grp_,grp,"total_num_of_proj",nkb);
    h5_read_write_attribute<int>(grp_,grp,"max_proj_per_atom",nhm);
    h5_read_write_attribute<int>(grp_,grp,"ngm",ngm);

    {
      nda::array<int,2> mill_g;//(ngm,3);
      nda::h5_read(grp_,"miller_g",mill_g);
      nda::h5_write(grp,"miller_g",mill_g);
    }

    {
      nda::array<ComplexType,3> vl;//(nspin*npol*npol,ngm);
      nda::h5_read(grp_,"scf_local_potential",vl);
      nda::h5_write(grp,"scf_local_potential",vl);
      if(lspinorb_loc!=0) { // nc
        nda::h5_read(grp_,"pp_local_component_nc",vl);
        nda::h5_write(grp,"pp_local_component_nc",vl);
      } else {
        nda::array<ComplexType,1> vr;//(ngm);
        nda::h5_read(grp_,"pp_local_component",vr);
        nda::h5_write(grp,"pp_local_component",vr);
      }
    }

    {
      nda::array<int,1> T;
      nda::h5_read(grp_,"proj_per_atom",T);
      nda::h5_write(grp,"proj_per_atom",T);

      nda::h5_read(grp_,"projector_offset",T);
      nda::h5_write(grp,"projector_offset",T);

      nda::h5_read(grp_,"atomic_id",T);
      nda::h5_write(grp,"atomic_id",T);
    }

    nda::array<int,1> npw(nk);
    nda::h5_read(grp_,"npw",npw);
    nda::h5_write(grp,"npw",npw);

    if(type == "ncpp") {
      if(lspinorb_nl!=0) { // nc
        nda::array<ComplexType,4> Dnn; //(nsp,nhm,nhm);
        nda::h5_read(grp_,"dion_so",Dnn);
        nda::h5_write(grp,"dion_so",Dnn);
      } else {
        nda::array<double,3> Dnn_r; //(nsp,nhm,nhm);
        nda::h5_read(grp_,"dion",Dnn_r);
        nda::h5_write(grp,"dion",Dnn_r);
      }
    } else {
      utils::check(false,"finish");
    }

    nda::array<ComplexType,2> buff(nkb,npwx);
    for(int ik=0; ik<nk; ++ik)
    {
      nda::array<int,2> mill(npw(ik),3);
      nda::h5_read(grp_,"miller_k"+std::to_string(ik),mill);
      nda::h5_write(grp,"miller_k"+std::to_string(ik),mill);

      nda::h5_read(grp_,"projector_k"+std::to_string(ik),buff);
      nda::h5_write(grp,"projector_k"+std::to_string(ik),buff);
    }

  } else {
    APP_ABORT(" Error in pseudopot::save: Invalid input file type.");
  }

}

void pseudopot_to_h5(nda::ArrayOfRank<1> auto const&fft_mesh, std::string fname, bool append, 
                     std::string input_file_name, mf::mf_input_file_type_e input_file_type) 
{
  char mode = (append?'a':'w');
  h5::file file;
  try {
    file = h5::file(fname, mode);
  } catch(...) {
    APP_ABORT("Failed to open h5 file: {}, mode:{}",fname,mode);
  }
  h5::group grp(file);
  pseudopot_to_h5(fft_mesh,grp,input_file_name,input_file_type);
}


}

#endif
