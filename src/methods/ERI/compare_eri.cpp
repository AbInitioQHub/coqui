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
#include <tuple>
#include <vector>
#include <string>
#include <algorithm>
#include "cxxopts.hpp"

#include "configuration.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"
#include "utilities/mpi_context.h"

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

#include "hamiltonian/one_body_hamiltonian.hpp"


#include "methods/ERI/cholesky.h"
#include "methods/ERI/thc.h"
#include "methods/ERI/eri_utils.hpp"

namespace mpi3 = boost::mpi3;

template<typename MF_obj>
void get_thc_eri(MF_obj& mf, 
		 mpi3::communicator& q_comm,
                 int iq, long nbnd, 
                 ::nda::MemoryArrayOfRank<1> auto const& ri,
                 math::nda::DistributedArray auto& Xa,
                 math::nda::DistributedArray auto& Vuv,
                 math::nda::DistributedArray auto& eri);

int main(int argc, char* argv[]) 
{
  mpi3::environment env(argc, argv);
  auto world = mpi3::environment::get_world_instance();
  auto node_comm = world.split_shared();
  auto internode_comm = world.split(world.rank()%node_comm.size(), world.rank()/node_comm.size());
  utils::mpi_context_t<mpi3::communicator,mpi3::shared_communicator> mpi(world, node_comm, internode_comm);
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
    ("chol_cutoff", "cutoff of cholesky decomposition",
                     cxxopts::value<std::vector<double>>()->default_value(""))
    ("nIpts_c", "number of interpolating points", cxxopts::value<double>()->default_value("4.0"))
    ("outdir", "path to checkpoint files/folders", cxxopts::value<std::string>()->default_value("./"))
    ("prefix", "prefix: name of checkpoint files", cxxopts::value<std::string>()->default_value("pwscf"))
    ("mf_source", "mean-field source", cxxopts::value<std::string>()->default_value("qe"))
    ("methods", "comma-separated list of methods to compare: chol,cholkk,thc_chol_ov,thc_chol_ls", 
                 cxxopts::value<std::vector<std::string>>()->default_value(""))
  ;
  auto args = options.parse(argc, argv);  

  std::vector<std::string> good_tags = {"chol","thc"};

  int output_level = std::max(0,args["verbosity"].as<int>());
  int debug_level = std::max(0,args["debug"].as<int>());
  int nqpool = args["nqpool"].as<int>();
  setup_loggers(world.root(), output_level, debug_level);

  std::string welcome(
        std::string("/***********************************************************\n") +
                    " ********    Comparing ERI decomposition methods    ********\n" +
                    " ***********************************************************\n");
  app_log(1, welcome);

  auto MF_factory = [&]()-> mf::MF {
    if (args["mf_source"].as<std::string>() == "qe") {
      return mf::make_MF(mpi, mf::qe_source, args["outdir"].as<std::string>(), args["prefix"].as<std::string>());
    } else if (args["mf_source"].as<std::string>() == "pyscf") {
      return mf::make_MF(mpi, mf::pyscf_source, args["outdir"].as<std::string>(), args["prefix"].as<std::string>());
    } else {
      return mf::make_MF(mpi, mf::bdft_source, args["outdir"].as<std::string>(), args["prefix"].as<std::string>());
    }
  };
  mf::MF mf = MF_factory();

// MAM: check that methods have been provided...

  std::vector<std::string> tags = args["methods"].as<std::vector<std::string>>();
  if( tags.size() < 2) {
    app_log(1,"Must request at least 2 decomposition methods.");
    return 0;
  }
  // check requested methods
  for( auto const& s: tags) {
    if(std::none_of(good_tags.begin(),good_tags.end(), [&](std::string const& v){ return s==v; })) {
      app_log(1,"List of acceptable decomposition methods: ");
      for( auto const& v: good_tags )
        app_log(1,"    {}",v);
      APP_ABORT("Invalid method: {}",s);
    }
  } 

  // cholesky cutoff list
  std::vector<double> chol_cutoffs = args["chol_cutoff"].as<std::vector<double>>();
  if (chol_cutoffs.size() == 1) {
    chol_cutoffs.resize(tags.size(), chol_cutoffs[0]);
  }
  if (chol_cutoffs.size() != tags.size()) {
    app_log(1, "Number of cholesky cutoffs does not match number of methods");
    return 0;
  }

  using dArray_t = math::nda::distributed_array<nda::array<ComplexType,2>,mpi3::communicator>;
  using math::nda::dagger;
  decltype(nda::range::all) all;
  long nspins = mf.nspin_in_basis();
  long nkpts = mf.nkpts();
  long nbnd = mf.nbnd(); 
  long nqpts = mf.nqpts();
  long nqpts_ibz = mf.nqpts_ibz();
  auto occ = mf.occ();
  int nIpts_c = long(args["nIpts_c"].as<double>()*nbnd);
  //double cutoff = args["chol_cutoff"].as<double>();

  // parallelization is limited to k-point distribution for simplicity
  // maximize number of kpoints per q-group
  long np = world.size(), nqgrp=(np>1?0:1); 
  if( nqpool <= 0 and np>1 ) {
    // searching by hand for now
    for(int nq=1; nq<world.size(); ++nq) 
      if(np%nq==0 and (np/nq)<=nkpts and nkpts%(np/nq)==0) {
        nqgrp=nq;
        break;
      }
  } else {
    nqgrp = (np==1?1:nqpool);
  }
  if(nqgrp==0) 
    APP_ABORT("Error: Could not find grid partition: nproc:{}, nqpts_ibz:{}, nkpts:{} ",np,nqpts_ibz,nkpts);
  app_log(1,"Number of Q pools: {}",nqgrp);
  if(world.size() > 1)
    APP_ABORT("Error: Parallelization has been disabled"); 

  long nk_per_grp = nkpts/(np/nqgrp);
  auto q_comm = world.split(world.rank()/(np/nqgrp),world.rank());
  auto qbounds = itertools::chunk_range(0,nqpts_ibz,nqgrp,world.rank()/(np/nqgrp)); 

  auto q_node_comm = world.split_shared();
  auto q_internode_comm = world.split(world.rank()%node_comm.size(), world.rank()/node_comm.size());
  utils::mpi_context_t<mpi3::communicator,mpi3::shared_communicator> q_mpi(q_comm, q_node_comm, q_internode_comm);

  if( nspins>1 and q_comm.size() > 1) 
    APP_ABORT("Error: No parallelization within a q-point with nspin>1 (yet).");

  nda::array<double,3> stats(12,tags.size(),tags.size());
  stats()=0.0;

  nda::array<double,4> qstats(2,nqpts_ibz,tags.size(),tags.size());
  qstats()=0.0;

  std::vector<std::vector<dArray_t>> deri_all;

  for (int i = 0; i < tags.size(); ++i) {

    auto s = tags[i];
    deri_all.emplace_back(std::vector<dArray_t>{});    
    auto& deri = deri_all.back();

    if( s == "chol" or s == "cholkk" ) {

      for( auto [iq,q] : itertools::enumerate(itertools::range(qbounds.first,qbounds.second)) ) {
        // setup new deri
        deri.emplace_back( math::nda::make_distributed_array<nda::array<ComplexType,2>>(q_comm,{q_comm.size(),1},
                             {nspins*nkpts*nbnd*nbnd,nspins*nkpts*nbnd*nbnd},{nbnd,nbnd}) );

        utils::check(deri.back().local_shape()[0] == nspins*nk_per_grp*nbnd*nbnd,
                   " Error: Logic error, should not happen.");
        utils::check(deri.back().origin()[0] == nspins*nk_per_grp*nbnd*nbnd*q_comm.rank(),
                   " Error: Logic error, should not happen.");

        utils::check(q_comm.size()<=nkpts, "Only kp parallelization with chol.");
        utils::check(nkpts%q_comm.size()==0, "Only kp parallelization with chol.");

        // later on read parameters 
        methods::cholesky chol(std::addressof(mf),q_comm,
              methods::make_chol_ptree(chol_cutoffs[i],mf.ecutrho(),32));
        auto L = chol.evaluate<HOST_MEMORY>(q,nda::range(-1,-1),nda::range(-1,-1),s=="cholkk");
        utils::check(L.grid()[0]==1," Distribution error.");
        utils::check(L.grid()[1]==1," Distribution error.");
        utils::check(L.global_shape()[1]==nspins," Error: Shape mismatch");
        utils::check(L.global_shape()[2]==nkpts," Error: Shape mismatch");
        utils::check(L.global_shape()[3]==nbnd," Error: Shape mismatch");
        utils::check(L.global_shape()[4]==nbnd," Error: Shape mismatch");

        if (q_comm.root()) {
          std::cout << "At iq = " << q <<
          ", number of Cholesky vectors in pure Cholesky decomposition: " << L.global_shape()[0] << std::endl;
        }

        auto& dERIs = deri.back();

        // create distributed_array_view from 5-d L_{P,is,kp,i,a}
        // take spin subset if nspin>1
        auto Lloc = L.local();
        auto Lloc2d = nda::reshape(Lloc,std::array<long,2>{Lloc.shape(0),Lloc.size()/Lloc.shape(0)});
        auto dL = math::nda::make_distributed_array_view(q_comm,{1,q_comm.size()},
						    {L.global_shape()[0],nspins*nkpts*nbnd*nbnd},
						    {L.global_shape()[0], dERIs.block_size()[0]}, Lloc2d); 
        // you need L^T conj(T), so conjugate to turn it into dagger(L) * :
        Lloc = nda::conj(Lloc);

        math::nda::slate_ops::multiply(dagger(dL),dL,dERIs);  

      }

    } else if( s == "thc" ) {
        // not efficient but easy for testing

      methods::thc thc(std::addressof(mf),q_mpi,methods::make_thc_ptree(mf.ecutrho(),
                       1, 1024, chol_cutoffs[i]));
      auto [ri,Xa,Xb] = thc.interpolating_points<HOST_MEMORY>(0,nIpts_c);
      int nIpts = ri.size();
      auto [V, chi_head, chi_bar_head, IVec_] = thc.evaluate<HOST_MEMORY>(ri,Xa,Xb);

      for( auto [iq,q] : itertools::enumerate(itertools::range(qbounds.first,qbounds.second)) ) {

        deri.emplace_back( math::nda::make_distributed_array<nda::array<ComplexType,2>>(q_comm,{q_comm.size(),1},
                             {nspins*nkpts*nbnd*nbnd,nspins*nkpts*nbnd*nbnd},{nbnd,nbnd}) );

        nda::array<ComplexType,2> Vt(nIpts,nIpts);
        Vt()=ComplexType(0.0);
        for(auto [iq_,q_] : itertools::enumerate(V.local_range(0)) ) 
          if(q_==q) { 
            auto Vq = V.local()(iq_,all,all);
            for(auto [iu,u] : itertools::enumerate(V.local_range(1)) ) 
              for(auto [iv,v] : itertools::enumerate(V.local_range(2)) ) 
                Vt(u,v) = Vq(iu,iv); 
          }
        q_comm.all_reduce_in_place_n(Vt.data(),Vt.size(),std::plus<>{});

        auto Vuv{math::nda::make_distributed_array<nda::array<ComplexType,2>>(q_comm,
                          {q_comm.size(),1},{nIpts,nIpts},{512,512},true)};
        auto Vloc = Vuv.local();
        for(auto [iu,u] : itertools::enumerate(Vuv.local_range(0)) ) 
          for(auto [iv,v] : itertools::enumerate(Vuv.local_range(1)) ) 
            Vloc(iu,iv) = Vt(u,v);

        get_thc_eri(mf,q_comm,q,nbnd,ri,Xa,Vuv,deri.back());

      }

    } else {
      APP_ABORT("Error; Unknown tag {}",s);
    }

  }  // tags

    // analysis and gather statistics
  for( auto [iq,q] : itertools::enumerate(itertools::range(qbounds.first,qbounds.second)) ) {
    for(int i=0; i<tags.size(); i++) { 
      auto eri_i = nda::reshape(deri_all[i][iq].local(), std::array<long,1>{deri_all[i][iq].local().size()});
      for(int j=i+1; j<tags.size(); j++) { 
        auto eri_j = nda::reshape(deri_all[j][iq].local(), std::array<long,1>{deri_all[j][iq].local().size()});
        double av1(0.0),av2(0.0);
        for( auto a : itertools::range(eri_i.size()) ) {
          double d1 = std::abs(eri_i(a)-eri_j(a));
          double d2 = 2.0*std::abs((eri_i(a)-eri_j(a))/(eri_i(a)+eri_j(a)));
          av1 += d1;
          av2 += d2;
          stats(6,i,j) = std::max(d1,stats(6,i,j));
          stats(7,i,j) = std::max(d2,stats(7,i,j));
          qstats(1,q,i,j) = std::max(d1,qstats(1,q,i,j));
          if(q==0) {
            stats(10,i,j) = std::max(d1,stats(10,i,j));
            stats(11,i,j) = std::max(d2,stats(11,i,j));
	  }
        }
  	stats(0,i,j) += av1/double(nqpts_ibz*nkpts*nkpts*nspins*nspins*nbnd*nbnd*nbnd*nbnd);
	stats(1,i,j) += av2/double(nqpts_ibz*nkpts*nkpts*nspins*nspins*nbnd*nbnd*nbnd*nbnd);
	qstats(0,q,i,j) += av1/double(nkpts*nkpts*nspins*nspins*nbnd*nbnd*nbnd*nbnd);
        if(q==0) {
  	  stats(4,i,j) += av1/double(nkpts*nkpts*nspins*nspins*nbnd*nbnd*nbnd*nbnd);
	  stats(5,i,j) += av2/double(nkpts*nkpts*nspins*nspins*nbnd*nbnd*nbnd*nbnd);
	}
      }
    }
  }

  world.reduce_in_place_n(stats(0,all,all).data(),6*tags.size()*tags.size(),std::plus<>());
  world.reduce_in_place_n(stats(6,all,all).data(),6*tags.size()*tags.size(),mpi3::max<>());
  world.reduce_in_place_n(qstats(0,all,all,all).data(),nqpts_ibz*tags.size()*tags.size(),std::plus<>());
  world.reduce_in_place_n(qstats(1,all,all,all).data(),nqpts_ibz*tags.size()*tags.size(),mpi3::max<>());

  app_log(0,"Results (q,K,K'): ");
  app_log(0,"   iq  decom_i   decom_j  mean_abs_diff  max_abs_diff  ");
  for(int iq=0; iq<nqpts_ibz; iq++)
    for(int i=0; i<tags.size(); i++)
      for(int j=i+1; j<tags.size(); j++)
        app_log(0,"  {}  {}  {}  {}  {}",iq,i,j,qstats(0,iq,i,j),qstats(1,iq,i,j));

  app_log(0,"Results (q,K,K'): ");

  app_log(0,"  decom_i   decom_j  mean_abs_diff  mean_abs_scaled_diff  max_abs_diff  max_abs_scaled_diff ");
  for(int i=0; i<tags.size(); i++) 
    for(int j=i+1; j<tags.size(); j++) 
      app_log(0,"   {}  {}  {}  {}  {}  {}",i,j,stats(0,i,j),stats(1,i,j),stats(6,i,j),stats(7,i,j));

  app_log(0,"Results (q,K,K) only: ");
  app_log(0,"  decom_i   decom_j  mean_abs_diff  mean_abs_scaled_diff  max_abs_diff  max_abs_scaled_diff ");
  for(int i=0; i<tags.size(); i++) 
    for(int j=i+1; j<tags.size(); j++) 
      app_log(0,"   {}  {}  {}  {}  {}  {}",i,j,stats(2,i,j),stats(3,i,j),stats(8,i,j),stats(9,i,j));
  app_log(0,"Results q=0 only: ");
  app_log(0,"  decom_i   decom_j  mean_abs_diff  mean_abs_scaled_diff  max_abs_diff  max_abs_scaled_diff ");
  for(int i=0; i<tags.size(); i++)
    for(int j=i+1; j<tags.size(); j++)
      app_log(0,"   {}  {}  {}  {}  {}  {}",i,j,stats(4,i,j),stats(5,i,j),stats(10,i,j),stats(11,i,j));

  for(int i=0; i<tags.size(); i++) {
    ComplexType eJ(0.0);
    auto J = deri_all[i][0].local();
    // Hartree
    // sum_s1,s2,k1,k2,n,m eri_all[i][0](s1_k1_n_n,s2_k2_m_m)
    for(int is1=0, iskb=0; is1<nspins; is1++)
    for(int ik1=0; ik1<nkpts; ik1++, iskb+=nbnd*nbnd)
    for(int ib1=0, is_=iskb; ib1<nbnd; ib1++, is_+=nbnd+1) {
    
      if( std::abs(occ(is1,ik1,ib1)) > 1e-4 ) {
  
        auto w1 = occ(is1,ik1,ib1);
        for(int is2=0, iskb2=0; is2<nspins; is2++)
        for(int ik2=0; ik2<nkpts; ik2++, iskb2+=nbnd*nbnd) {

          auto w_ = occ(is2,ik2,all);
          for(int ib2=0, is2_=iskb2; ib2<nbnd; ib2++, is2_+=nbnd+1) 
            eJ += J(is_,is2_)*w1*w_(ib2);
        }

      }
    }
    if(nspins==1) eJ*=ComplexType(4.0);
    std::cout<<" EJ: " <<i <<" " <<eJ/double(2.0*nkpts) <<std::endl;
  }

  for(int i=0; i<tags.size(); i++) {
    ComplexType eX(0.0);
    // EXX 
    // sum_s,q,k,n,m eri_all[i][q](s_k_n_m,s_k_n_m)
    for(int iq=0; iq<nqpts_ibz; iq++) {
      auto J = deri_all[i][iq].local();
      for(int is=0, i=0; is<nspins; is++)
      for(int ik=0; ik<nkpts; ik++) { 
        auto w = occ(is,ik,all);
        for(int in=0; in<nbnd; in++) 
          for(int im=0; im<nbnd; im++,++i) 
            eX += J(i,i)*w(in)*w(im);
      }
    }
    if(nspins==1) eX*=ComplexType(2.0);
    std::cout<<" EX: " <<i <<" " <<eX/double(2.0*nkpts) <<std::endl;
  }

  return 0;
}

template<typename MF_obj>
void get_thc_eri(MF_obj& mf,
                 mpi3::communicator& q_comm,
                 int iq, long nbnd,
                 ::nda::MemoryArrayOfRank<1> auto const& ri,
                 math::nda::DistributedArray auto& Xa,
                 math::nda::DistributedArray auto& Vuv,
                 math::nda::DistributedArray auto& eri)
{
  decltype(nda::range::all) all;
  long nspins = mf.nspin_in_basis();
  long nkpts = mf.nkpts();
  auto Q  = mf.Qpts()(iq,all);
  auto k_to_k2 = mf.qk_to_k2()(iq,all);
  int nIpts = ri.shape(0);
  app_log(0,"Number of interpolating points in cholesky-based thc decomposition: {} ",nIpts);
  long bab = eri.block_size()[0];

  // distributed along rows...
  long bu = Vuv.block_size()[0]; 
  utils::check(Vuv.block_size()[1] == bu,"Error: Vuv must have squared blocks");

  // A(k,a,b,u) = conj(Pa(k,a,u)) * Pb(qk,b,u)
  // B(u,k,a,b) = Vuv * conj(A(k,a,b,v))
  // eri(k,a,b,k_,a,b) = A(k,a,b,u) * B(u,k,a,b))
  auto A = math::nda::make_distributed_array<nda::array<ComplexType,2>>(q_comm,{q_comm.size(),1},
                                              {nspins*nkpts*nbnd*nbnd,nIpts}, {bab,bu});
  utils::check(A.local_shape()[1]==nIpts," Error: Shape mismatch");
  auto B = math::nda::make_distributed_array<nda::array<ComplexType,2>>(q_comm,{1,q_comm.size()},
                                              {nIpts,nspins*nkpts*nbnd*nbnd}, {bu, bab});
  utils::check(B.local_shape()[0]==nIpts," Error: Shape mismatch");

  {
    auto Aloc = nda::reshape(A.local(), std::array<long,5>{nspins,nkpts,nbnd,nbnd,nIpts});
    auto Pa = nda::reshape(Xa.local(), std::array<long,4>{nspins,nkpts,nbnd,nIpts});
    for( auto s : itertools::range(nspins))
      for( auto k : itertools::range(nkpts)) {
        auto k2 = k_to_k2(k);
        for( auto a : itertools::range(nbnd))
          for( auto b : itertools::range(nbnd))
            for( auto u : itertools::range(nIpts))
              Aloc(s,k,a,b,u) = std::conj(Pa(s,k,a,u)) * Pa(s,k2,b,u);
      }
  }
/*
  if(q_comm.size() <= nkpts and nkpts%q_comm.size() == 0 ) {
    long nk_per_grp = nkpts/q_comm.size();
    utils::check(A.origin()[0]==nspins*nk_per_grp*nbnd*nbnd*q_comm.rank()," Error: Shape mismatch");
    utils::check(A.local_shape()[0]==nspins*nk_per_grp*nbnd*nbnd," Error: Shape mismatch");
    utils::check(B.origin()[1]==nspins*nk_per_grp*nbnd*nbnd*q_comm.rank()," Error: Shape mismatch");
    utils::check(B.local_shape()[1]==nspins*nk_per_grp*nbnd*nbnd," Error: Shape mismatch");
    auto Pa = nda::reshape(Xa.local(), std::array<long,4>{nspins,nk_per_grp,nbnd,nIpts});
    auto Pb = nda::reshape(Xb.local(), std::array<long,4>{nspins,nk_per_grp,nbnd,nIpts});
    for( auto s : itertools::range(nspins))
      for( auto k : itertools::range(nk_per_grp))
        for( auto a : itertools::range(nbnd))
          for( auto b : itertools::range(nbnd))
            for( auto u : itertools::range(nIpts))
              Aloc(s,k,a,b,u) = std::conj(Pa(s,k,a,u)) * Pb(s,k,b,u);
  } else {
    // fall back to cases where kpoints are distributed, slow but only used for testing!
    // lazy
    nda::array<ComplexType,3> Pa_g(nkpts,nbnd,nIpts);      
    nda::array<ComplexType,3> Pb_g(nkpts,nbnd,nIpts);      
    Pa_g()=ComplexType(0.0);
    Pb_g()=ComplexType(0.0);
    auto Pa_ = dPa.local();
    auto Pb_ = dPb.local();
    for( auto [i,n] : itertools::enumerate(dPa.local_range(0)) ) { 
      // n = k * nbnd + a 
      long k = n/nbnd; 
      long a = n%nbnd;
      for( auto u : itertools::range(nIpts))
        Pa_g(k,a,u) = Pa_(i,u); 
    }
    for( auto [i,n] : itertools::enumerate(dPb.local_range(0)) ) {
      // n = k * nbnd + a 
      long k = n/nbnd;
      long a = n%nbnd;
      for( auto u : itertools::range(nIpts))
        Pb_g(k,a,u) = Pb_(i,u);
    }
    q_comm.all_reduce_in_place_n(Pa_g.data(),Pa_g.size(),std::plus<>{});
    q_comm.all_reduce_in_place_n(Pb_g.data(),Pb_g.size(),std::plus<>{});
    auto Aloc = A.local(); 
    for( auto [i,n] : itertools::enumerate(A.local_range(0)) ) { 
      // n = ( k * nbnd + a ) * nbnd + b
      long b = n%nbnd;
      long n_ = n/nbnd; 
      long k = n_/nbnd; 
      long a = n_%nbnd;
      for( auto u : itertools::range(nIpts))
        Aloc(i,u) = std::conj(Pa_g(k,a,u)) * Pb_g(k,b,u);
    }
  }
*/
  math::nda::slate_ops::multiply(Vuv,dagger(A),B);
  math::nda::slate_ops::multiply(A,B,eri);
}

