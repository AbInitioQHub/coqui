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
#include "stdio.h"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/mpi_context.h"

#include "nda/nda.hpp"
#include "utilities/test_common.hpp"

#include "mean_field/default_MF.hpp"
#include "utilities/fortran_utilities.h"
#include "utilities/qe_utilities.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "hamiltonian/pseudo/pseudopot_to_h5.hpp"


namespace bdft_tests
{
using mpi_context_t = utils::mpi_context_t<mpi3::communicator,mpi3::shared_communicator>;

TEST_CASE("fortran_utilities", "[pseudo]")
{
  FC_test_util();
  auto [outdir,prefix] = utils::utest_filename(mf::qe_source);
  std::string fname(outdir+"VKB");
  int sz = fname.size();
  int nspin,nkb,npwx,nkpts,nat,nhm,nsp,ierr;
  FC_read_pw2bgw_vkbg_header(fname.c_str(), sz, nspin, nkb, npwx, nkpts, nat, nsp, nhm,ierr);
//  utils::check(ierr==0, "Error reading QE::pw2bgw vkbg file (header info).");
  CHECK(ierr==0);

  nda::array<int,1> ityp(nat), nh(nsp), ngk(nkpts);
  nda::array<int,3> miller(nkpts,npwx,3);
  nda::array<std::complex<double>,3> vkb(nkpts,nkb,npwx);
  nda::array<double,4> Dnn(nspin,nat,nhm,nhm);

  int msz = nkpts*npwx*3, vsz=nkpts*nkb*npwx, Dsz = nspin*nat*nhm*nhm;
  int k0=0, k1=nkpts;
  FC_read_pw2bgw_vkbg(fname.c_str(), fname.size(), k0, k1, ityp.data(), nat,
        nh.data(), nsp, ngk.data(), nkpts,
        Dnn.data(), Dsz, miller.data(), msz, vkb.data(), vsz, ierr);
//  utils::check(ierr==0, "Error reading QE::pw2bgw vkbg file (arrays).");
  CHECK(ierr==0);
}

void test_h5(mpi_context_t& mpi, mf::MF &mf)
{
  if(mpi.comm.root()) {
    if(mf.input_file_type() == mf::xml_input_type and mf.mf_type() == mf::qe_source) {
      hamilt::pseudopot_to_h5(mf.fft_grid_dim(),"__dummy__.h5",false,mf.outdir(),mf::xml_input_type);
    } else if(mf.input_file_type() == mf::h5_input_type and mf.mf_type() != mf::pyscf_source) {
      hamilt::pseudopot_to_h5(mf.fft_grid_dim(),"__dummy__.h5",false,mf.filename(),mf::h5_input_type);
    }
  }
  mpi.comm.barrier();
  // create pseudo from newly created h5
  hamilt::pseudopot Vnl(mf,"__dummy__.h5");
  mpi.comm.barrier();
  if (mpi.comm.root()) 
    remove("__dummy__.h5");
  mpi.comm.barrier();
}

TEST_CASE("pseudo_h5", "[pseudo]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  SECTION("lih223")
  { 
    // CNY: mf::xml_input_type is not implemented for vxc!
    auto qe_h5 = mf::default_MF(mpi, "qe_lih223", mf::h5_input_type);
    test_h5(*mpi, qe_h5);
  }
 
  SECTION("lih223_sym")
  { 
    auto qe_h5 = mf::default_MF(mpi, "qe_lih223_sym", mf::h5_input_type);
    test_h5(*mpi, qe_h5);
  }
  
  SECTION("lih223_xml")
  { 
    auto qe_h5 = mf::default_MF(mpi, "qe_lih223", mf::xml_input_type);
    test_h5(*mpi, qe_h5);
  }
  
  SECTION("GaAs222_so")
  { 
    auto qe_h5 = mf::default_MF(mpi, "qe_GaAs222_so", mf::h5_input_type);
    test_h5(*mpi, qe_h5);
  }

}

}
