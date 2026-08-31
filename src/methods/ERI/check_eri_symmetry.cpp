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



#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <algorithm>
#include "cxxopts.hpp"

#include "configuration.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"

#include "utilities/check.hpp"
#include "utilities/Timer.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/functions.hpp"

#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "numerics/distributed_array/nda.hpp"

#include "mean_field/MF.hpp"
#include "mean_field/mf_utils.hpp"

#include "methods/ERI/cholesky.h"
#include "methods/ERI/thc.h"

namespace mpi3 = boost::mpi3;

template<typename MF_obj>
void get_thc_eri(MF_obj& mf, 
                 int iq, long nbnd, 
                 ::nda::MemoryArrayOfRank<4> auto const& Pa,
                 ::nda::MemoryArrayOfRank<2> auto const& Vuv,
                 ::nda::MemoryArrayOfRank<2> auto && eri);

int main(int argc, char* argv[]) 
{
  mpi3::environment env(argc, argv);
  auto world = mpi3::environment::get_world_instance();
  std::vector<std::string> inputs;

  // MAM: add ranges and other options (e.g. efficiency related ones...)
  // parse inputs
  cxxopts::Options options(argv[0], "Beyond DFT");
  options
    .positional_help("[optional args]")
    .show_positional_help();
  options.add_options()
    ("h,help", "print help message")
    ("verbosity", "0, 1, 2, ...: higher means more", cxxopts::value<int>()->default_value("2"))
    ("debug", "0, 1, 2, ...: higher means more", cxxopts::value<int>()->default_value("0"))
    ("nqpool", "number of q pools", cxxopts::value<int>()->default_value("0"))
    ("chol_cutoff", "cutoff of cholesky decomposition", cxxopts::value<double>()->default_value("1e-5"))
    ("nIpts_c", "number of interpolating points", cxxopts::value<double>()->default_value("4.0"))
    ("outdir", "path to checkpoint files/folders", cxxopts::value<std::string>()->default_value("./"))
    ("prefix", "prefix: name of checkpoint files", cxxopts::value<std::string>()->default_value("pwscf"))
    ("mf_source", "mean-field source", cxxopts::value<std::string>()->default_value("qe"))
    ("method", "method to test: chol,thc",cxxopts::value<std::string>()->default_value("chol"))
  ;
  auto args = options.parse(argc, argv);  

  std::vector<std::string> good_tags = {"chol","thc"};

  int output_level = std::max(0,args["verbosity"].as<int>());
  int debug_level = std::max(0,args["debug"].as<int>());
  int nqpool = args["nqpool"].as<int>();
  setup_loggers(world.root(), output_level, debug_level);

  std::string welcome(
        std::string("/*******************************************************************************\n") +
                    " ********    Checking symmetry relations in ERI tensor decomposition    ********\n" +
                    " *******************************************************************************\n");
  app_log(1, welcome);

  auto MF_factory = [&]()-> mf::MF {
    if (args["mf_source"].as<std::string>() == "qe") {
      return mf::make_MF(world, mf::qe_source, args["outdir"].as<std::string>(), args["prefix"].as<std::string>());
    } else if (args["mf_source"].as<std::string>() == "pyscf") {
      return mf::make_MF(world, mf::pyscf_source, args["outdir"].as<std::string>(), args["prefix"].as<std::string>());
    } else {
      return mf::make_MF(world, mf::bdft_source, args["outdir"].as<std::string>(), args["prefix"].as<std::    string>());
    }
  };
  mf::MF mf = MF_factory();

  std::string tag = args["method"].as<std::string>();
  // check requested methods
  if(std::none_of(good_tags.begin(),good_tags.end(), [&tag](std::string const& v){ return tag==v; })) {
    app_log(1,"List of acceptable decomposition methods: ");
    for( auto const& v: good_tags )
      app_log(1,"    {}",v);
    APP_ABORT("Invalid method: {}",tag);
  }

  // cholesky cutoff list

  using dArray_t = math::nda::distributed_array<nda::array<ComplexType,2>,mpi3::communicator>;
  using math::nda::dagger;
  decltype(nda::range::all) all;
  long nspins = mf.nspin();
  long nkpts = mf.nkpts();
  long nbnd = mf.nbnd(); 
  long nqpts = mf.nqpts();
  long nkpts_ibz = mf.nkpts_ibz();
  long nqpts_ibz = mf.nqpts_ibz();
  int nIpts = long(args["nIpts_c"].as<double>()*nbnd);
  //double cutoff = args["chol_cutoff"].as<double>();

  utils::check(nspins==1," Finish uhf case.");

  // parallelization is limited to k-point distribution for simplicity
  // maximize number of kpoints per q-group
  long np = world.size(), nqgrp=(np>1?0:1); 
  if( nqpool <= 0 and np>1 ) {
    // searching by hand for now
    for(int nq=1; nq<world.size(); ++nq) 
      if(np%nq==0 and (np/nq)<=nqpts_ibz and nqpts_ibz%(np/nq)==0) {
        nqgrp=nq;
        break;
      }
  } else {
    nqgrp = (np==1?1:nqpool);
  }
  if(nqgrp==0) 
    APP_ABORT("Error: Could not find grid partition: nproc:{}, nqpts_ibz:{}, nkpts:{} ",np,nqpts_ibz,nkpts);
  app_log(1,"Number of Q pools: {}",nqgrp);
  utils::check(world.size()==1, "Symmetry check only serial for now");
  // parallelization is all broken...

  long nk_per_grp = nkpts/(np/nqgrp);
  auto q_comm = world.split(world.rank()/(np/nqgrp),world.rank());
  auto qbounds = itertools::chunk_range(0,nqpts_ibz,nqgrp,world.rank()/(np/nqgrp)); 

  nda::array<double,1> stats(2);
  nda::array<double,2> qstats(2,nqpts-nqpts_ibz);
  
  std::vector<dArray_t> deri;
  std::vector<dArray_t> deri_symm;
  auto ks_to_k = mf.ks_to_k();
  auto qp_to_ibz = mf.qp_to_ibz();
  auto Qs = mf.Qs();
  auto qp_symm = mf.qp_symm();
  auto qsymms = mf.qsymms();
  auto slist = utils::find_inverse_symmetry(mf.qsymms(),mf.symm_list());
  auto dmat = utils::generate_dmatrix_old<false>(world,mf,mf.symm_list(),slist);

  for( auto i : itertools::range(nqpts-nqpts_ibz) ) {
    deri.emplace_back( math::nda::make_distributed_array<nda::array<ComplexType,2>>(q_comm,{q_comm.size(),1},
                           {nkpts*nbnd*nbnd,nkpts*nbnd*nbnd},{nbnd,nbnd}) );
    deri_symm.emplace_back( math::nda::make_distributed_array<nda::array<ComplexType,2>>(q_comm,{q_comm.size(),1},
                           {nkpts*nbnd*nbnd,nkpts*nbnd*nbnd},{nbnd,nbnd}) );
  }

  if( tag == "chol" ) {

    nda::array<ComplexType,2> Tmat(nbnd,nbnd);
    for( auto iq : itertools::range(nqpts) ) 
    {

      double chol_cutoff = args["chol_cutoff"].as<double>();
      methods::cholesky chol(std::addressof(mf),q_comm,chol_cutoff,mf.ecutrho(),32,q_comm.size());
      auto L = chol.evaluate<HOST_MEMORY>(iq,nda::range(-1,-1),nda::range(-1,-1),false);
      utils::check(L.grid()[0]==1," Distribution error.");
      utils::check(L.grid()[1]==1," Distribution error.");
      utils::check(L.global_shape()[1]==nspins," Error: Shape mismatch");
      utils::check(L.global_shape()[2]==nkpts," Error: Shape mismatch");
      utils::check(L.global_shape()[3]==nbnd," Error: Shape mismatch");
      utils::check(L.global_shape()[4]==nbnd," Error: Shape mismatch");

      if (q_comm.root()) {
        std::cout << "At iq = " << iq <<
        ", number of Cholesky vectors in pure Cholesky decomposition: " << L.global_shape()[0] << std::endl;
      }

      if( iq < nqpts_ibz ) {

        auto Lloc = L.local();
        auto Lqs = nda::make_regular(Lloc);
        auto Lqs2d = nda::reshape(Lqs,std::array<long,2>{Lqs.shape(0),Lqs.size()/Lqs.shape(0)});

        for( int iqs=nqpts_ibz; iqs<nqpts; ++iqs) {
          if(qp_to_ibz(iqs) == iq) {
          
	    int is=std::distance(qsymms.begin(),
				 std::find(qsymms.begin(),qsymms.end(),qp_symm(iqs)));
            utils::check(is > 0, "Error: isym==0 found."); 
            utils::check(is < qsymms.size(), "Error: Problems locating symmetry."); 
            auto& dERIs = deri_symm.at(iqs-nqpts_ibz);
            auto dL = math::nda::make_distributed_array_view(q_comm,{1,q_comm.size()},
						    {L.global_shape()[0],nspins*nkpts*nbnd*nbnd},
						    {L.global_shape()[0], dERIs.block_size()[0]}, Lqs2d); 
	    for(int ik=0; ik<nkpts; ik++) {
	      int ks = ks_to_k(is,ik);
              int k2 = mf.qk_to_k2(iqs,ik);
	      for(int n=0; n<Lloc.extent(0); n++) {
                nda::blas::gemm(Lloc(n,0,ks,all,all),dmat(is-1,k2,all,all),Tmat);
                nda::blas::gemm(ComplexType(1.0),nda::dagger(dmat(is-1,ik,all,all)),Tmat,ComplexType(0.0),Lqs(n,0,ik,all,all));
	      }
	    } 
            math::nda::slate_ops::multiply(dagger(dL),dL,dERIs);

          } 
        }

      } else { 

        auto& dERIs = deri.at(iq-nqpts_ibz);
        auto Lloc = L.local();
        auto Lloc2d = nda::reshape(Lloc,std::array<long,2>{Lloc.shape(0),Lloc.size()/Lloc.shape(0)});
        auto dL = math::nda::make_distributed_array_view(q_comm,{1,q_comm.size()},
						    {L.global_shape()[0],nspins*nkpts*nbnd*nbnd},
						    {L.global_shape()[0], dERIs.block_size()[0]}, Lloc2d); 

        math::nda::slate_ops::multiply(dagger(dL),dL,dERIs);  

      } 

    }

  } else if( tag == "thc" ) {

    methods::thc thc(std::addressof(mf),q_comm,mf.ecutrho(),
                     1, 1024, 1e-10, 100);
    auto [ri,Xa,Xb] = thc.interpolating_points<HOST_MEMORY>(0,nIpts);
    auto [V, chi_head, chi_bar_head, IVec_] = thc.evaluate<HOST_MEMORY>(ri,Xa,Xb);
//    auto dPa = thc.interpolating_basis<HOST_MEMORY>(true,ri,0);
    nda::array<ComplexType,2> Xau(nspins*nkpts*nbnd,nIpts);
    auto X4d = nda::reshape(Xau,std::array<long,4>{nspins,nkpts,nbnd,nIpts});
    auto P4d = nda::reshape(Xa.local(),std::array<long,4>{nspins,nkpts,nbnd,nIpts});

    for( auto iq : itertools::range(nqpts) ) 
    {

      if( iq < nqpts_ibz ) {

        for( int iqs=nqpts_ibz; iqs<nqpts; ++iqs) {
          if(qp_to_ibz(iqs) == iq) {
            
            int is=std::distance(qsymms.begin(),
                                 std::find(qsymms.begin(),qsymms.end(),qp_symm(iqs)));
            utils::check(is < qsymms.size(), "Error: Problems locating symmetry.");
            utils::check(is > 0, "Error: isym==0 found."); 

            for(int ik=0; ik<nkpts; ik++) {
              int ks = ks_to_k(is,ik);
              nda::blas::gemm(nda::transpose(dmat(is-1,ik,all,all)),P4d(0,ks,all,all),X4d(0,ik,all,all));
	    }
            get_thc_eri(mf,iqs,nbnd,Xau,V.local()(iq,all,all),deri_symm.at(iqs-nqpts_ibz).local());

          }
        }

      } else {

        get_thc_eri(mf,iq,nbnd,Xa.local(),
                    V.local()(iq,all,all),deri.at(iq-nqpts_ibz).local());

      }

    } 

  }

  // analysis and gather statistics
  stats()=0.0;
  qstats()=0.0;
  for( auto iq : itertools::range(nqpts-nqpts_ibz) ) {
    auto eri_i = nda::reshape(deri[iq].local(), std::array<long,1>{deri[iq].local().size()});
    auto eri_j = nda::reshape(deri_symm[iq].local(), std::array<long,1>{deri_symm[iq].local().size()});
    double av1(0.0),av2(0.0);
    for( auto a : itertools::range(eri_i.size()) ) {
      double d1 = std::abs(eri_i(a)-eri_j(a));
      av1 += d1;
      stats(1) = std::max(d1,stats(1));
      qstats(1,iq) = std::max(d1,qstats(1,iq));
    }
    stats(0) += av1/double((nqpts-nqpts_ibz)*nkpts*nkpts*nbnd*nbnd*nbnd*nbnd);
    qstats(0,iq) += av1/double(nkpts*nkpts*nbnd*nbnd*nbnd*nbnd);
  }
/*
  for( auto iq : itertools::range(nqpts-nqpts_ibz) ) {
    auto e1 = nda::reshape(deri[iq].local(),std::array<long,4>{nkpts,nbnd*nbnd,nkpts,nbnd*nbnd});
    auto e2 = nda::reshape(deri_symm[iq].local(),std::array<long,4>{nkpts,nbnd*nbnd,nkpts,nbnd*nbnd});
    for(int k1=0; k1<nkpts; k1++) {
      for(int k2=0; k2<nkpts; k2++) {
        double av1(0.0);
        for(int a=0; a<nbnd*nbnd; a++) 
          for(int b=0; b<nbnd*nbnd; b++) 
            av1 += std::abs(e1(k1,a,k2,b)-e2(k1,a,k2,b));
        app_log(0," iq:{}, k1:{}, k2:{}, err:{} ",iq,k1,k2,av1/double(nbnd*nbnd*nbnd*nbnd)); 
      }
    }
  }
*/

  world.reduce_in_place_n(stats.data(),1,std::plus<>());
  world.reduce_in_place_n(stats.data()+1,1,mpi3::max<>());
  world.reduce_in_place_n(qstats(0,all).data(),qstats.extent(1),std::plus<>());
  world.reduce_in_place_n(qstats(1,all).data(),qstats.extent(1),mpi3::max<>());

  app_log(0,"  mean_abs_diff: {}  max_abs_diff: {} ",stats(0),stats(1));

  app_log(0,"Results (q,K,K'): ");
  app_log(0,"   iq  mean_abs_diff  max_abs_diff  ");
  for(int iq=0; iq<nqpts-nqpts_ibz; iq++)
    app_log(0,"  iq: {}  mean_abs_diff: {}  max_abs_diff: {}",iq,qstats(0,iq),qstats(1,iq));


  return 0;
}

template<typename MF_obj>
void get_thc_eri(MF_obj& mf, 
		 int iq, long nbnd,
	         ::nda::MemoryArrayOfRank<4> auto const& Pa,     
	         ::nda::MemoryArrayOfRank<2> auto const& Vuv,
	         ::nda::MemoryArrayOfRank<2> auto&& eri)
{
  decltype(nda::range::all) all;
  long nspins = mf.nspin();
  long nkpts = mf.nkpts();
  auto Q  = mf.Qpts()(iq,all);
  int nIpts = Pa.extent(4); 

  // distributed along rows...

  // A(k,a,b,u) = conj(Pa(k,a,u)) * Pb(qk,b,u)
  // B(u,k,a,b) = Vuv * conj(A(k,a,b,v))
  // eri(k,a,b,k_,a,b) = A(k,a,b,u) * B(u,k,a,b))
  ::nda::array<ComplexType,2> A(nspins*nkpts*nbnd*nbnd,nIpts); 
  ::nda::array<ComplexType,2> B(nIpts,nspins*nkpts*nbnd*nbnd); 

  auto A5d = ::nda::reshape(A, std::array<long,5>{nspins,nkpts,nbnd,nbnd,nIpts});
//  auto Pa = ::nda::reshape(dPa, std::array<long,4>{nspins,nkpts,nbnd,nIpts});
  for( auto k : itertools::range(nkpts)) {
    int k2 = mf.qk_to_k2(iq,k);
    for( auto a : itertools::range(nbnd))
      for( auto b : itertools::range(nbnd))
        for( auto u : itertools::range(nIpts))
          A5d(0,k,a,b,u) = std::conj(Pa(0,k,a,u)) * Pa(0,k2,b,u);
  }
  ::nda::blas::gemm(Vuv,dagger(A),B);
  ::nda::blas::gemm(A,B,eri);
}

