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



#include <cmath>

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/Timer.hpp"

#include "mpi3/environment.hpp"

#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "nda/linalg.hpp"
#include "utilities/functions.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "mean_field/MF.hpp"
#include "mean_field/distributed_orbital_readers.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "hamiltonian/one_body_hamiltonian.hpp"
#include "orbitals/rotate.h"

#include "orbitals/pgto.h"


namespace orbitals
{

template<MEMORY_SPACE MEM = HOST_MEMORY>
mf::MF add_pgto(mf::MF& mf,
                std::string fn, std::string basis, std::string type,
                int b0, bool diag_F, bool ortho, double cutoff, bool ortho_by_shell)
{
  using Array_t = typename memory::array<MEM,ComplexType,4>;
  decltype(nda::range::all) all;

  auto& mpi = *mf.mpi();

  pgto orbs;
  orbs.add(mf,basis,type);
  if( b0 < 0 ) b0 = int(mf.nbnd()); 
  cutoff = std::abs(cutoff);
  long nspin = mf.nspin();
  long nspin_in_basis = mf.nspin_in_basis();
  long nkpts_ibz = mf.nkpts_ibz();

  /* print basis set details */
  orbs.print_basis_info(2);

  /* 4d processor grid: {s, k,bnd,g} */
  std::array<long,4> pgrid;
  {  // working pgrid
    long sz = mpi.comm.size();
    long pk = utils::find_proc_grid_max_npools(sz,nkpts_ibz,1.0);
    pgrid = {1,pk,1,sz/pk};
  }

  /*
   * Generate PGTO basis set 
   * Spin independent at this point e.g. (1,nkps_ibz,nbnd,ngm). 
   * Proper spin dependence added below. 
   */
  auto psi_pgto = orbs.generate_basis_set<MEM,boost::mpi3::communicator>(
              mpi.comm,mf,true,pgrid);
  utils::check(psi_pgto.global_shape()[1]==nkpts_ibz,"Shape mismatch.");

//  ortho by shell here
 if(ortho_by_shell) {
    nda::array<int,2> orb_ranges(orbs.number_of_shells(),2);
    auto shell_sz = orbs.shell_sizes();
    utils::check(orb_ranges.extent(0) == shell_sz.extent(0), "Shape mismatch");
    long n=0;
    for( long a=0; a<shell_sz.extent(0); ++a ) {
      orb_ranges(a,0) = n;
      n += shell_sz(a);
      orb_ranges(a,1) = n;
    }  
    orthonormalize<MEM>(orb_ranges,psi_pgto);
  }

  /* Merge basis with requested states from mf */
  if(b0 > 0)
  {
    // MAM: add choice to construct spin independent basis in uhf/ghf
    if(nspin_in_basis == 1) {
      long npgto = psi_pgto.global_shape()[2];
      /* states kept from mf */
      auto psi_mf = mf::read_distributed_orbital_set<Array_t>(
                  mf,mpi.comm,'w',pgrid,
                  {0,nspin},{0,nkpts_ibz},{0,b0},
                  std::array<long,4>{1,1,1,psi_pgto.block_size()[3]});
      utils::check(psi_mf.global_shape()[3]==psi_pgto.global_shape()[3],"Shape mismatch.");
      utils::check(psi_mf.local_range(3)==psi_pgto.local_range(3),"Range mismatch.");

      auto psi = math::nda::make_distributed_array<memory::array<MEM,ComplexType,4>>(mpi.comm,pgrid,
           {nspin,nkpts_ibz,npgto+b0,psi_pgto.global_shape()[3]},psi_pgto.block_size());
      auto psi_loc = psi.local();
      psi_loc(all,all,nda::range(npgto),all) = psi_pgto.local();
      psi_loc(all,all,nda::range(npgto,npgto+b0),all) = psi_mf.local();
      psi_pgto = std::move(psi);
    } else if(nspin_in_basis == 2){
      utils::check(nspin==2,  
        "Error: nspin_in_basis:{} != nspin(==2) ",nspin_in_basis);
    } else { // make spin independent basis from uhf
      utils::check(false,"finish");
    }
  }  

  // orthonormalize and truncate based on eigenvalues of overlap matrix
  // if diag_F==true and cutoff==0.0, no need to do this
  if(ortho) 
    orthonormalize<MEM>(psi_pgto,cutoff);

  if(diag_F) {

    auto eigV = canonicalize_diagonal_basis<MEM>(mf,psi_pgto);

    long npgto = psi_pgto.global_shape()[2];
    app_log(3, "Eigenvalues: ");
    for( auto is : nda::range(nspin) ) { 
      app_log(3, " spin:{}",is);
      for( auto ik : nda::range(nkpts_ibz) ) { 
        app_log(3, "  kpoint:{}",ik);
        for( auto ia : nda::range(npgto) ) 
          app_log(3, "   {}: {}",ia,eigV(is,ik,ia));
      }  
    }

    return mf::MF(mf::bdft::bdft_readonly(mf, fn, psi_pgto, 0));

  } else {

    return mf::MF(mf::bdft::bdft_readonly(mf, fn, psi_pgto, 0));

  }

}

/*
 * This routine assumes that MF object contains reliable eigenvalue information.
 */ 
mf::MF eigenstate_selection(mf::MF& mf, std::string fn,
			    std::string grid_type, long n0, long nblk)
{
  decltype(nda::range::all) all;

  app_log(2,"*****************************************************");
  app_log(2,"*               Eigenstate selection:               *");
  app_log(2,"*****************************************************");
  app_log(2,"  - file name of new BDFT system: {}",fn);
  app_log(2,"  - Number of states from low energy sector (e.g. primary basis): {}",n0);
  app_log(2,"  - Number of blocks used to generate auxiliary basis: {}",nblk);

  long nbnd = mf.nbnd();
  utils::check(n0 >= 0, "eigenstate_selection: n0 < 0");
  utils::check(nbnd >= n0+nblk, "eigenstate_selection: Too many orbitals requested.");
  long nvir = nbnd-n0;
  auto nkpts_ibz = mf.nkpts_ibz();
  auto eigval = mf.eigval();
  nda::array<std::pair<long,double>,2> orb_list(nkpts_ibz,nblk);
  if(nblk==0)
    return mf::MF(mf::bdft::bdft_readonly(mf, fn, n0, orb_list));
  for(long ik=0; ik<nkpts_ibz; ++ik) {
    auto eig = eigval(0,ik,all);
    app_log(4,"k: {}\n  - E(N=0):{}\n  - E(N=n0):{}\n  - E(N=highest):{}",
		ik, eig(0),eig((n0>=nbnd?nbnd-1:n0)),eig(nbnd-1));
  }
  if (nblk > 0) {
    if (grid_type == "linear") {
      for (long ik = 0; ik < nkpts_ibz; ++ik) {
        if (n0 < nbnd) {
          auto eig = eigval(0, ik, nda::range(n0, nbnd));
          double e0 = std::real(eig(0));
          double e1 = std::real(eig(nvir-1));
          double de = (e1-e0) / nblk;
          auto it_beg = eig.begin();
          app_log(4, " ik:{}, e0:{}, e1:{}, de:{}", ik, e0, e1, de);
          for (long i = 0; i < nblk; ++i) {
            double ei = e0 + (i+0.5)*de;
            auto it = std::lower_bound(it_beg, eig.end(), ei);
            // MAM: notice that it is possible to get it==it_beg if there are large gaps in the spectrum
            // such that 2 consecutive ei values land within a gap
            // otherwise, it seems more reasonable to choose the lower energy state
            if (it != it_beg) --it;
            utils::check(it != eig.end(),
                         "Error in eigenstate_select: Problems finding eigenstate. Probably too few eigenstates to select from or too many blocks.");
            app_log(4, " ik:{}, i:{}, ei:{}, ni:{}, eig(ni):{}", ik, i, ei, n0+std::distance(eig.begin(),it), *it);
            orb_list(ik, i) = std::pair<long,double>{n0+std::distance(eig.begin(),it), 1.0};
            it_beg = ++it;
          }
        }
      }
    } else {
      utils::check(false, "eigenstate_selection: unknown grid type: {}", grid_type);
    }
  }
  return mf::MF(mf::bdft::bdft_readonly(mf, fn, n0, orb_list));
}

/*
 * Starting from an existing mean field system, this routine creates a new one 
 * by modifying the orbital set associated with it. 
 * The routine currently supports the following modifications to the orbital set:
 * 1) pgto: Adding periodic Gaussian type orbitals. 
 * 2) select: Choose a subset of the orbitals. Existing system must have valid eigenstates
 *            in assending ordered. 
 * The orbital modifyers are mutually exclusive. use separate input blocks to combine operations.
 * Required input elements:
 *  - "mean_field": {...}, valid mean_field input block defining base system
 *  - "prefix" : "...", prefix of new mean field system
 *  - "pgto" or "select" : {...}, input block for orbital modification algorithm.
 * Optional arguments:
 *  - "outdir": "./", folder where the resulting mean field system will be located.
 * Optional arguments for pgto:
 *  - "orthonormalize": (default: false) Orthonormalize combined basis.
 *  - cutoff: (default: 1e-8): Cutoff used to truncate basis during orthogonalization.
 *  - orthonormalize_by_shell: (default: true) Orthonormalize PGTO basis by shell. (No truncation). 
 *
 * Example 
 * {
 *   "prefix" : "...",
 *   "outdir" : "...",  
 *   "pgto": {
 *     "n0" : 10,
 *     "basis": "filename.txt",
 *     "cutoff" : 1e-8,
 *     "orthonormalize" : true
 *   }, 
 *   "select": {
 *     "n0" : 10,
 *     "n_blocks": 40,
 *     "weight": "{dos, count}"
 *   }
 * }
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
void orbital_factory(mf::MF &base_mf, ptree const& pt)
{
  long ndef = std::min(base_mf.nbnd(),long(std::ceil(base_mf.nelec()))); 

  // only choice is to create a bdft mf object 
  auto prefix = io::get_value<std::string>(pt,"prefix",
					   "orbitals - missing required input: prefix");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");

  if(auto node = pt.get_child_optional("pgto")) {
    auto cutoff = io::get_value_with_default<double>(*node,"cutoff",1e-8);
    auto ortho = io::get_value_with_default<bool>(*node,"orthonormalize",false);
    auto ortho_by_shell = io::get_value_with_default<bool>(*node,"orthonormalize_by_shell",true);
    auto diag_F = io::get_value_with_default<bool>(*node,"diag_F",false);
    auto n0 = io::get_value_with_default<long>(*node,"n0",ndef);
    auto basis = io::get_value<std::string>(*node,"basis",
					   "orbitals::pgto - missing required input: basis");
    auto mf = add_pgto<MEM>(base_mf,outdir+"/"+prefix+".h5",
                            basis,"nwchem",n0,diag_F,ortho,cutoff,ortho_by_shell);
  } else if(auto node_ = pt.get_child_optional("select")) { 
    auto n0 = io::get_value_with_default<long>(*node_,"n0",ndef);
    auto nblk = io::get_value_with_default<long>(*node_,"n_blocks",0);
    auto grid_type = io::get_value_with_default<std::string>(*node_,"grid","linear");
    auto mf = eigenstate_selection(base_mf,outdir+"/"+prefix+".h5", grid_type,n0,nblk);
  } else {
    APP_ABORT("orbital_factory: No orbital modifier found.");
  }
}

// instantiate templates
using boost::mpi3::communicator; 
template mf::MF add_pgto<HOST_MEMORY>(mf::MF&,std::string,std::string,std::string,int,bool,bool,double,bool);

template void orbital_factory<HOST_MEMORY>(mf::MF&,ptree const&);
/*
#if defined(ENABLE_DEVICE)
template mf::MF
add_pgto<UNIFIED_MEMORY,communicator>(utils::mpi_context_t<communicator>&,mf::MF&,std::string,std::string,std::string,int,bool,bool,double,bool);
template mf::MF
add_pgto<DEVICE_MEMORY,communicator>(utils::mpi_context_t<communicator>&,mf::MF&,std::string,std::string,std::string,int,bool,bool,double,bool);
template void orbital_factory<DEVICE_MEMORY,communicator>(utils::mpi_context_t<communicator>&,mf::MF&,ptree const&);
template void orbital_factory<UNIFIED_MEMORY,communicator>(utils::mpi_context_t<communicator>&,mf::MF&,ptree const&);
#endif
*/
}
