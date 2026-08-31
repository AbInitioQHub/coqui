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


#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>

#include "configuration.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/parser.h"
#include "utilities/harmonics.h"
#include "utilities/math.h"
#include "utilities/proc_grid_partition.hpp"
#include "grids/g_grids.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/fft/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "mean_field/MF.hpp"
#include "orbitals/pgto.h"

namespace orbitals
{

void pgto::add(std::string tag, nda::array<double,2> const& R, std::string fn, std::string ftype)
{
  utils::check(R.shape()[1] == 3, "Shape mismatch.");
  std::vector<orbital_shell> G0;
  if(ftype=="gamess")
    G0 = parse_gamess(fn,tag);
  else if(ftype=="nwchem")
    G0 = parse_nwchem(fn,tag);
  else if(ftype=="gamess")
    G0 = parse_molpro(fn,tag);
  else 
    utils::check(false,"Error in pgto::parse: Unknown file format: {}",ftype);
  if(G0.size() == 0) {
    app_log(2,"Failed to find basis set for atom type: {}",tag);
    return;
  } 
  int nc = R.shape()[0];
  for(int i=0; i<nc; ++i) 
    for(auto& v: G0) {
      G.emplace_back(v);
      G.back().R = R(i,nda::range::all);
    }
}


void pgto::add(mf::MF& mf, std::string fn, std::string ftype)
{
  auto nsp = mf.number_of_species();
  auto at_id = mf.atomic_id();
  auto at_pos = mf.atomic_positions();
  auto species = mf.species();
  // sort atoms by species
  std::vector<std::vector<orbital_shell>> G0;
  G0.reserve(nsp);
  for( auto [i,s] : itertools::enumerate(species) )
  {
    if(ftype=="gamess")
      G0.emplace_back(parse_gamess(fn,s));
    else if(ftype=="nwchem")
      G0.emplace_back(parse_nwchem(fn,s));
    else if(ftype=="gamess")
      G0.emplace_back(parse_molpro(fn,s));
    else
      utils::check(false,"Error in pgto::parse: Unknown file format: {}",ftype);
  }  

  for( auto [i,s] : itertools::enumerate(at_id) ) {
    for(auto& v: G0[s]) {
      G.emplace_back(v);
      G.back().R = at_pos(i,nda::range::all);
    }
  }
}

std::vector<orbital_shell> pgto::parse_gamess(std::string fn, [[maybe_unused]] std::string tag) {
  std::vector<orbital_shell> G0;
  utils::check(std::filesystem::exists(fn),"Error: File doesn't exist: {}",fn);
  utils::check(false,"Finish gamess parser!");
  return G0;
}

std::vector<orbital_shell> pgto::parse_nwchem(std::string fn, [[maybe_unused]] std::string tag) {
  utils::check(std::filesystem::exists(fn),"Error: File doesn't exist: {}",fn);

  std::string line;
  std::vector<orbital_shell> G0;
  std::string L = "SPDFGHIJL";
  std::ifstream in(fn);

  // find beggining of basis set
  while(not in.eof()) {
    std::getline(in,line);
    auto words = utils::split(line,"\t\r\n ");
    if(words.size()==0 or words[0][0]=='#') continue;
    if(words[0] == "BASIS") {
      utils::check(words.size() >= 4,"Error parsing nwchem basis set file: {}",line);
      utils::check(words[1][0]=='"',"Error parsing nwchem basis set file: {}",line);
      utils::check(*(words.rbegin()+1)=="SPHERICAL","Error: Only spherical basis sets currently supported");
      break;
    } else {
      utils::check(false,"Error parsing nwchem basis set file: {}",line);
    } 
  }     
  utils::check(not in.eof(),"Error parsing nwchem basis set file. Reached end of file.");	

  auto skip_element = [&]() {
    while(not in.eof()) {
      std::getline(in,line);
      auto words = utils::split(line,"\t\r\n ");
      if(words.size()==0) continue;
      utils::check(words[0] != "END", "Error parsing nwchem basis: Reached end of file.");
      if(words.size() >= 2 and words[0] == "#BASIS" and words[1] == "SET:") break; 
    }
    utils::check(not in.eof(),"Error parsing nwchem basis set file: {}",line);	
  };

  auto parse_contraction_ = [&](std::string const& Li, std::vector<std::string> const& b)
  {
    auto pos = L.find(Li); 
    int nc = b.size();
    if(nc==0) return;
    auto words = utils::split(b[0],"\t\r\n ");
    int nt = words.size(); 
    utils::check(pos != std::string::npos, "Error in parse_contraction: Li:{}",Li);
    orbital_shell o = {int(pos),
			nda::stack_array<double,3>::zeros({3}),
			nda::array<double,2>::zeros({nc,nt})};
    auto& p = o.p;
    for(int i=0; i<nt; ++i) p(0,i) = std::stof(words[i]);
    for(int j=1; j<nc; j++) {
      words = utils::split(b[j],"\t\r\n ");
      utils::check(words.size()==nt, "Error in parse_contraction: {}",b[j]);
      for(int i=0; i<nt; ++i) p(j,i) = std::stof(words[i]);
    }
    G0.emplace_back(std::move(o));
  };

  auto read_basis_ = [&](std::string Li) {
    std::vector<std::string> b;
    while(not in.eof()) {
      std::getline(in,line);
      auto words = utils::split(line,"\t\r\n ");
      if(words.size()==0) continue;
      if(words[0] == tag) {
	parse_contraction_(Li,b);
        b.clear();
        Li = words[1];
        continue;
      } else if(words[0] == "END" or words[0] == "#BASIS") {
        parse_contraction_(Li,b);
        break;
      } else {
        b.push_back(line); 
      }
    }
  };

  // find first element
  skip_element();     

  while(not in.eof()) {
    std::getline(in,line);
    auto words = utils::split(line,"\t\r\n "); 
    if(words.size()==0) continue;
    if(words[0] == tag) {
      utils::check(L.find(words[1]) != std::string::npos, "Unknown angular momentum:{}",words[1]);
      read_basis_(words[1]);
      break;
    } else {
      skip_element();	
    } 
  }
  return G0;
}

std::vector<orbital_shell> pgto::parse_molpro(std::string fn, [[maybe_unused]] std::string tag) {
  std::vector<orbital_shell> G0;
  utils::check(std::filesystem::exists(fn),"Error: File doesn't exist: {}",fn);
  utils::check(false,"Finish molpro parser!");
  return G0;
}

// psir(r) = sum_n exp(i*k*Ln) * sum_i c(i) * Gauss(r-R(a)-Ln,a(i))
void pgto::make_orbital(nda::ArrayOfRank<2> auto const& kpts,
                  nda::ArrayOfRank<1> auto const& mesh, nda::ArrayOfRank<2> auto const& lattv,
                  math::nda::DistributedArrayOfRank<4> auto&& psir)
{
  (void) kpts,mesh,lattv,psir;
  // MAM: if I assume that the distribution occurs over nnr, 
  //      I can distribute the evaluation of expikL, Ln, ... 
  // MAM:put expikL, Ln, etc in shared memory!!!
/*
  utils::check(mesh.size()==3, "Size mismatch.");
  utils::check(psir.global_shape()[3] == mesh(0)*mesh(1)*mesh(2),"Size mismatch");

  decltype(nda::range::all) all;
  long nspin = psir.local_shape()[0];
  long nkpts = psir.local_shape()[1];
  long nbnd = psir.local_shape()[2];
  long nr = psir.local_shape()[3];
  auto psirloc = psir.local();
  psirloc()=ComplexType(0);

  int nmax = 6, n2 = 2*nmax+1;
  nda::array<double,4> Ln(n2,n2,n2,3);
  nda::array<double,1> ri(3), rij(3);
  auto Ln_2d = nda::reshape(Ln,std::array<long,2>{n2*n2*n2,3});
  for(int i=-nmax; i<=nmax; ++i) {
    ri() = double(i)*lattv(0,all);
    for(int j=-nmax; j<=nmax; ++j) {
      rij() = ri + double(j)*lattv(1,all);
      for(int k=-nmax; k<=nmax; ++k) Ln(i+nmax,j+nmax,k+nmax,all) = rij + double(k)*lattv(2,all);
    } 
  }

  // calculate and store |Ln|, sort Ln based on |Ln|, then truncate sum over n below based on
  // smallest exponent in the list 

  // can loop over batched if memory is an issue
  nda::array<ComplexType,2> expikLn(nkpts,Ln_2d.shape()[0]);
  for( auto [ik,k] : itertools::enumerate(psir.local_range(1)) )
    for( auto n : nda::range(Ln_2d.shape()[0]) ) { 
      double kr = kpts(k,0)*Ln_2d(n,0) + kpts(k,1)*Ln_2d(n,1) + kpts(k,2)*Ln_2d(n,2);
      expikLn(ik,n) = ComplexType(std::cos(kr),std::sin(kr));
    }

  int lmax = 0;
  for(int i=0; i<G.size(); ++i) if( G[i].L > lmax ) lmax = G[i].L;

  // can you do this with real values???
  nda::array<double,2> rn(nr,3);
  for( auto [ir,r] : itertools::enumerate(psir.local_range(3)) ) {
    long ki = r%mesh(2);
    long n_ = r/mesh(2);
    long ji = n_%mesh(1);
    long ii = n_/mesh(1);
    rn(ir,nda::range(3)) = (double(ii)/double(mesh(0)))*lattv(0,all) +
	                   (double(ji)/double(mesh(1)))*lattv(1,all) +
 	                   (double(ki)/double(mesh(2)))*lattv(2,all);
  }

  auto find_rcut = [&](double cutoff, int L, int nc, double const* norm, double const* c, int ldc, 
								  double const* a, int lda) {
    double r0 = 0.1, r1=10.0, gr=0.0;
    double rL = std::pow(r1, double(L));
    for(long j=0; j<nc; ++j)
      gr += norm[j]*c[j*ldc]*std::exp(-a[j*lda]*r1);
    while(gr*rL > cutoff) {
      r0 = r1;
      r1 = r1+5.0;
      gr=0.0;
      rL = std::pow(r1, double(L));	
      for(long j=0; j<nc; ++j)
        gr += norm[j]*c[j*ldc]*std::exp(-a[j*lda]*r1);
    }
    do {
      double rc = (r0+r1)/2.0;
      rL = std::pow(rc, double(L));	
      gr=0.0;
      for(long j=0; j<nc; ++j)
        gr += norm[j]*c[j*ldc]*std::exp(-a[j*lda]*rc);
      if( gr*rL > cutoff ) r0 = rc;
      else r1 = rc;
    } while(std::abs(r1-r0) > 1e-3);
    return r1;
  };

  const double pi = 4.0 * std::atan(1.0);
  long nblk = std::min(4096l,nr);
  nda::array<ComplexType,3> fr(2*lmax+1,Ln_2d.shape()[0],nblk);
  double gr;
  nda::array<ComplexType,2> pr(nkpts,nblk);
  nda::array<double,1> Ylm(2*lmax+1);
  nda::array<double,2> drn(nr,3);
  long b0 = 0;
  long ncmax = 0;
  for(int a=0; a<G.size(); ++a) ncmax = std::max(ncmax,G[a].p.shape()[0]); 
  nda::array<double,1> NormL(ncmax);

  utils::harmonics<double> ylm_compute;

  for(int a=0; a<G.size(); ++a) {

    int L = G[a].L;
    auto& coeff = G[a].p;
    int nc = coeff.shape()[0];
    int nt = coeff.shape()[1]-1;
    for(int i=0; i<nc; ++i) 
      NormL(i) = std::pow(2, L + 1) * 
		 std::sqrt(2.0 / static_cast<double>(utils::DFactorial(2 * L + 1))) * 
		 std::pow(2.0 / pi, 0.25) *
		 std::pow(coeff(i,0), 0.5 * (L + 1.0) + 0.25);
    for(int t=0; t<nt; ++t) { 
      double rc = find_rcut(1e-9,L,nc,NormL.data(),coeff.data()+t+1,nt+1,coeff.data(),nt+1);
      for( long ir = 0; ir<nr; ir+=nblk ) { 
        long nr_ = std::min(nblk,nr-ir); 
	fr()=ComplexType(0.0);
        for( auto n : nda::range(Ln_2d.shape()[0]) ) { 
          auto dR = G[a].R+Ln_2d(n,all);
	  // drn = r - Ra - Ln
          drn(nda::range(nr_),all) = rn(nda::range(ir,ir+nr_),all); 
	  // gr(r) = sum_i c(i) * exp(-a(i)*r^2)
          double* rp = drn.data(); 
          for(long i=0; i<nr_; ++i, rp+=3) {
            rp[0] -= dR(0);
            rp[1] -= dR(1);
            rp[2] -= dR(2);
            double rr = rp[0]*rp[0] + rp[1]*rp[1] + rp[2]*rp[2]; 
            if(rr < rc) {
              gr=0.0;
              for(long j=0; j<nc; ++j)
                gr += NormL(j)*coeff(j,t+1)*std::exp(-coeff(j,0)*rr);
              ylm_compute.unnormalized_solid_harmonics_l(L,rp,3l,Ylm.data(),Ylm.size());
              //ylm_compute.solid_harmonics_l(L,rp,3l,Ylm.data(),Ylm.size());
              // fr(m,n,r) = gr(r)*Ylm(m)
              for(long m=0; m<2*L+1; ++m)
                fr(m,n,i) = ComplexType(gr*Ylm(m),0.0);
            }
          }
        }
        // psir(k,a0+m,r) = sum_n expikLn(k,n)*fr(m,n,r)
        for(int m=0; m<2*L+1; ++m) {
	  nda::blas::gemm(ComplexType(1.0), expikLn, fr(m,all,nda::range(nr_)), 
			  ComplexType(0.0), pr(all,nda::range(nr_)));
	  psirloc(0,all,b0+m,nda::range(ir,ir+nr_)) = pr(all,nda::range(nr_)); 
        } 
      } // ir 
      b0 += 2*L+1;
    } // t
  } 
  utils::check(b0==size(), " Internal error.");
  if(nspin == 2)
    psirloc(1,nda::ellipsis{}) = psirloc(0,nda::ellipsis{});
*/
}

// Builds a distributed orbital set by symmetry adapting (sum over images) the GTO basis
// set. Kpoint grid is taken from mf.kpts_ibz(). A nspin=1 is always taken. Spin dependence
// of the resulting basis set is added outside this routine.
template<MEMORY_SPACE MEM, typename comm_t>
memory::darray_t<memory::array<MEM,ComplexType,4>, comm_t>
pgto::generate_basis_set(comm_t& comm, mf::MF& mf, bool normalize, 
                         std::array<long,4> const pgrid_out,std::array<long,4> const bz)
{
  decltype(nda::range::all) all;
  using larray = memory::array<MEM,ComplexType,4>;

  utils::check(false, "generate_basis_set temporarily disabled.");
  utils::check(mf.has_wfc_grid(), "Error: generate_basis_set requires has_wfc_grid() == true.");

  long nkpts = mf.nkpts_ibz();
  long nbnd = this->size(); 
  auto wfc_g = mf.wfc_truncated_grid();
  long nnr = wfc_g->nnr(); 
  long ngm = wfc_g->size(); 
  auto fft_mesh = wfc_g->mesh();
  std::array<long,4> pgrid; 
  {  // working pgrid
    long sz = comm.size();
    long pk = utils::find_proc_grid_max_npools(sz,nkpts,1.0);
    pgrid = {1,pk,sz/pk,1};
  }
  utils::check(comm.size() == std::accumulate(pgrid.cbegin(), pgrid.cend(), long(1), std::multiplies<>{}),
               "MPI size mismatch.");
  auto psir = math::nda::make_distributed_array<larray>(comm,{1,1,1,comm.size()},
                                                        {1,nkpts,nbnd,nnr},{1});  
  // fill psir
  //make_orbital(mf.kpts()(nda::range(nkpts),all),fft_mesh,mf.lattv(),psir);

  { // remove phase factor
    auto phase = memory::unified_array<ComplexType, 1>::zeros({psir.local_shape()[3]});
    auto psiloc = psir.local();
    for( auto [is,s] : itertools::enumerate(psir.local_range(0)) ) {
      for( auto [ik,k] : itertools::enumerate(psir.local_range(1)) ) {
        nda::array<double,1> mkp = -1.0*mf.kpts()(k,all);
        utils::rspace_phase_factor(mf.lattv(),mkp,fft_mesh,psir.local_range(3),phase);
        for( auto [ia,a] : itertools::enumerate(psir.local_range(2)) ) 
          psiloc(is,ik,ia,all) *= phase();
      }
    }
  }

  {
    // redistribute into layout appropriate for fft
    auto psi2 = math::nda::make_distributed_array<larray>(comm,pgrid,{1,nkpts,nbnd,nnr},bz);
    math::nda::redistribute(psir,psi2);  
    psir = std::move(psi2);
  }
 
  long nfft = psir.local_shape()[0]*psir.local_shape()[1]*psir.local_shape()[2];
  auto psir_4d = nda::reshape(psir.local(),std::array<long,4>{nfft,fft_mesh(0),
						                    fft_mesh(1),fft_mesh(2)});
  math::fft::fwdfft_many(psir_4d);

  auto psig = math::nda::make_distributed_array<larray>(comm,pgrid,{1,nkpts,nbnd,ngm},bz);

  // MAM: need to zero out contributions outside ecutwfc
  for( auto [is,s] : itertools::enumerate(psir.local_range(0)) )
    for( auto [ik,k] : itertools::enumerate(psir.local_range(1)) )
      for( auto [ia,a] : itertools::enumerate(psir.local_range(2)) ) {
        auto gloc = psig.local()(is,ik,ia,all);
        auto rloc = psir.local()(is,ik,ia,all);
        for( auto [in,n] : itertools::enumerate(wfc_g->gv_to_fft()) )
          gloc(in) = rloc(n);
      }
  psir.reset();

  if(normalize) 
  {
    auto ploc = psig.local();
    for( auto [is,s] : itertools::enumerate(psig.local_range(0)) )
      for( auto [ik,k] : itertools::enumerate(psig.local_range(1)) )
        for( auto [ia,a] : itertools::enumerate(psig.local_range(2)) ) {
          double s_ = std::abs(nda::blas::dotc(ploc(is,ik,ia,all),ploc(is,ik,ia,all))); 
          s_ = std::sqrt(1.0/s_);
          nda::blas::scal(s,ploc(is,ik,ia,all));
        } 
  }
  comm.barrier();

  if(pgrid == pgrid_out or pgrid_out == std::array<long,4>{0}) {
    return psig;
  } else {
    auto psi_ = math::nda::make_distributed_array<larray>(comm,pgrid_out,{1,nkpts,nbnd,ngm},bz); 
    math::nda::redistribute(psig,psi_);
    return psi_;
  }
}

using boost::mpi3::communicator;
using memory::darray_t;
using memory::host_array;

template darray_t<host_array<ComplexType,4>, communicator>
pgto::generate_basis_set<HOST_MEMORY,communicator>(communicator&,mf::MF&,bool,
		std::array<long,4> const, std::array<long,4> const);
/*
#if defined(ENABLE_DEVICE)
using memory::device_array;
using memory::unified_array;
template darray_t<device_array<ComplexType,4>, communicator>
pgto::generate_basis_set<DEVICE_MEMORY,communicator>(communicator&,mf::MF&,bool,
		std::array<long,4> const, std::array<long,4> const);
template darray_t<unified_array<ComplexType,4>, communicator>
pgto::generate_basis_set<UNIFIED_MEMORY,communicator>(communicator&,mf::MF&,bool,
		std::array<long,4> const, std::array<long,4> const);
#endif
*/

}

