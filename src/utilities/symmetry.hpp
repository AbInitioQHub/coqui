#ifndef UTILITIES_SYMMETRY_HPP
#define UTILITIES_SYMMETRY_HPP

#include <algorithm>
#include <vector>
#include "configuration.hpp"
#include <nda/nda.hpp>
#include <nda/blas.hpp>
#include "utilities/concepts.hpp"
#include "utilities/Timer.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/sparse/csr_utils.hpp"
#include "numerics/device_kernels/kernels.h"
#include "utilities/details/symmetry_utils.hpp"

namespace utils
{

struct symm_op
{
  // rotation
  nda::stack_array<double, 3, 3> R;
  // inverse rotation
  nda::stack_array<double, 3, 3> Rinv;
  // fractional translation
  nda::stack_array<double, 3> ft;
};

/*
 * Given a set of vectors (Qp), the routine finds all vectors in the IBZ.
 * Qp must be in crystal coordinates.
 * The Qps in the IBZ are moved to the beginning of the array, and the number of elements
 * found is returned by the routine.
 * For each element in the resulting list, the arrays sym and idx contain the index of the 
 * symmetry that generates the Qp in the list and the Qp in the IBZ that generates it. 
 */ 
int generate_ibz(bool use_trev, std::vector<symm_op> const& slist, nda::ArrayOfRank<2> auto & Qp,
	     nda::array<int,1> &sym, nda::array<bool,1> &trev, nda::array<int,1> &idx)		
{
  decltype(nda::range::all) all;
  utils::check(Qp.extent(1)==3,"Size mismatch");
  int nq = Qp.extent(0);
  utils::check(trev.extent(0)==nq,"Size mismatch");
  utils::check(sym.extent(0)==nq,"Size mismatch");
  utils::check(idx.extent(0)==nq,"Size mismatch");
  int nsym = slist.size();
  int nibz=0;
  std::vector<bool> unique(nq,true);  // assume all are unique at first
  nda::stack_array<double,3> q;
  nda::stack_array<double,3> qm;
  auto comp = [](nda::ArrayOfRank<1> auto&& a, nda::ArrayOfRank<1> auto&& b) {
        // there has to be a better way than this!
        double di = std::abs(a(0)-b(0)); 
        if( std::abs(di-1.0) < 1e-4 ) di = 0.0;
        if( std::abs(di-2.0) < 1e-4 ) di = 0.0;
        if( std::abs(di-3.0) < 1e-4 ) di = 0.0;
        double dj = std::abs(a(1)-b(1)); 
        if( std::abs(dj-1.0) < 1e-4 ) dj = 0.0;
        if( std::abs(dj-2.0) < 1e-4 ) dj = 0.0;
        if( std::abs(dj-3.0) < 1e-4 ) dj = 0.0;
        double dk = std::abs(a(2)-b(2)); 
        if( std::abs(dk-1.0) < 1e-4 ) dk = 0.0;
        if( std::abs(dk-2.0) < 1e-4 ) dk = 0.0;
        if( std::abs(dk-3.0) < 1e-4 ) dk = 0.0;
        return std::abs(di*di + dj*dj + dk*dk) < 1e-12; 
      };
  for(int iq=0; iq<nq; ++iq) {
    if(not unique[iq]) {
      // find next unique and swap. If not found, break.
      auto it = std::find(unique.begin()+iq,unique.end(),true);
      if(it == unique.end()) break;
      int n = std::distance(unique.begin()+iq,it) + iq;
      //std::swap(unique[iq],unique[n]);
      std::vector<bool>::swap(unique[iq],unique[n]);
      q = Qp(n,all);
      Qp(n,all) = Qp(iq,all);
      Qp(iq,all) = q;
      sym(n) = sym(iq);
      idx(n) = idx(iq);
      trev(n) = trev(iq);
    }
    utils::check(unique[iq],"Error: Internal logic error. FIX!"); 
    nibz++;
    sym(iq) = 0;
    trev(iq) = false;
    idx(iq) = iq;	
    if(use_trev) {
      qm = -1.0*Qp(iq,all);
      for(int ip=iq+1; ip<nq; ++ip) {
        // in this case, overwrite if already found to be not unique, since trev is a better symmetry
        if(comp(qm,Qp(ip,all))) {
          // if ip is unique, ipm should also be unique... is this always correct?
          utils::check(unique[ip],"Error: Internal logic error in qm.!"); 
          unique[ip] = false;
          sym(ip) = 0;
          trev(ip) = true;
          idx(ip) = iq;
        }
      }
    }
    // generate all Qps from iq and mark them as non-unique
    // skip is=0, since it should be the identity
    for(int is=1; is<nsym; ++is) {
      nda::blas::gemv(1.0,nda::transpose(slist[is].Rinv),Qp(iq,all),0.0,q);   
      for(int ip=iq+1; ip<nq; ++ip) {
        if(unique[ip] and comp(q,Qp(ip,all))) {
          unique[ip] = false;
          sym(ip) = is;
          trev(ip) = false;
          idx(ip) = iq;
          if(use_trev) {
            qm = -1.0*q;
            // look for qm
            for(int ipm=iq+1; ipm<nq; ++ipm) {
              if(comp(qm,Qp(ipm,all)) and ip!=ipm) {
                // if ip is unique, ipm should also be unique... is this always correct?
                utils::check(unique[ipm],"Error: Internal logic error in qm loop.!"); 
                unique[ipm] = false;
                sym(ipm) = is;
                trev(ipm) = true;
                idx(ipm) = iq;
              }
            } // for( ipm )
          } // trev
        } // if
      } // for( ip )
    }
  }
  return nibz;
}

/*
 * Generates symmetry maps. Structures are those from MF object.
 * Input:
 *   - nk: number of kpoints to consider, can be different from nkpts 
 *   - nkpts_ibz
 *   - kp_symm
 *   - kp_to_ibz 
 *   
 * Output: Tuple object with:  
 *  - ksymms(is):          list of unique symmetries in kp_symm
 *  - k_to_s(ik):          for every ik in kp_symm, the index in ksymms containing kp_symm(ik)
 *  - ns_per_kibz(ikbz):   number of symmetries associated 
 *  - Ks(ikbz, is):        list of kpoints associated to kibz
 *  - Skibz(ikbz, is):     symmetry index of each kpoint (ordering consistent with Ks) 
 */ 
auto generate_kp_maps(long nktot, long nkpts_ibz, 
                      nda::ArrayOfRank<1> auto const& kp_symm, 
                      nda::ArrayOfRank<1> auto const& kp_to_ibz)
{
  int nsym = 0;
  auto ns_per_kibz = nda::array<int,1>::zeros({nkpts_ibz});
  nda::array<int,1> ksymms;
  auto Ks = nda::array<int,2>::zeros({nkpts_ibz,96});  // list of kpoints associated to kibz
  auto Skibz = nda::array<int,2>::zeros({nkpts_ibz,96});  // list of symmetry index for Ks
  auto k_to_s = nda::array<int,1>::zeros({nktot});         // symmetry index of each kpoint
  {
    std::vector<int> syms(48,-1);
    std::vector<int> unique;
    unique.reserve(48);
    for( auto is: kp_symm(nda::range(nktot)) ) {
      if(syms[is] < 0) {
        syms[is] = unique.size();
        unique.emplace_back(is);
      }
    }
    for( auto [ik,is]: itertools::enumerate(kp_symm(nda::range(nktot))) ) {
      utils::check(syms[is]>=0, "chol_metric_impl_ibz Oh oh (syms[is]>=0)");
      if(ik < nkpts_ibz)
        utils::check(syms[is]==0, "chol_metric_impl_ibz Oh oh (syms[is]==0 for IBZ)");
      k_to_s(ik) = syms[is];
    }
    nsym = unique.size();
    ksymms = nda::array<int,1>(nsym);
    for( auto [i,is]: itertools::enumerate(unique) ) ksymms(i) = is;
    for( auto ik: kp_to_ibz(nda::range(nktot)) ) ns_per_kibz(ik)++;
    for( auto ikbz: nda::range(nkpts_ibz) )
    {
      int nk=0;
      for( auto [j,ik] : itertools::enumerate(kp_to_ibz(nda::range(nktot))) )
        if( ik == ikbz and j>=nkpts_ibz ) {
          utils::check(syms[kp_symm(j)] > 0 and syms[kp_symm(j)] < nsym, "Error: Logic error: ik:{},j:{},kp_symm:{}, sym:{}",ik,j,kp_symm(j),syms[kp_symm(j)]);
          Ks(ikbz,nk) = j;
          Skibz(ikbz,nk++) = syms[kp_symm(j)]-1;  // skip identity
        }
      utils::check(nk == (ns_per_kibz(ikbz)-1), "nk ({}) != ns_per_kibz-1 ({}): Logic error! ik:{}",nk,ns_per_kibz(ikbz),ikbz);
    }
  }
  return std::make_tuple(ksymms,k_to_s,ns_per_kibz,Ks,Skibz);
}

/**
 * This routine checks if a rotation matrix is compatible with a fft mesh.
 * In practice, it checks if R(i,j)*N(j)/N(i) is an integer for all (i,j).
 */ 
bool is_rotation_compatible_with_fft_mesh(nda::ArrayOfRank<1> auto const& mesh,
                                          nda::ArrayOfRank<2> auto const& R,
                                          double tol = 1e-4)
{
  int ndim = mesh.size();
  utils::check( R.shape() == std::array<long,2>{ndim,ndim}, "Shape mismatch." );
  for(int i=0; i<ndim; ++i) {
    for(int j=i+1; j<ndim; ++j) {
      auto x = R(i,j)*mesh(j)/mesh(i);
      if (std::abs(x - std::round(x)) > tol ) return false;
      x = R(j,i)*mesh(i)/mesh(j);
      if (std::abs(x - std::round(x)) > tol ) return false;
    }   
  }   
  return true;
}

/*
 * Calls is_rotation_compatible_with_fft_mesh and aborts if false. 
 */
void check_rotation_compatible_with_fft_mesh(nda::ArrayOfRank<1> auto const& mesh,
                                          nda::ArrayOfRank<2> auto const& R,
                                          double tol = 1e-4)
{
  utils::check(is_rotation_compatible_with_fft_mesh(mesh,R,tol),
               std::string("Error: Found incompatible rotation/fft mesh: \n") +
               " fft mesh: ({},{},{}) \n" +
               " R: ({}  {}  {} \n" +
               "     {}  {}  {} \n" +
               "     {}  {}  {}) \n"
               ,mesh(0),mesh(1),mesh(2)
               ,R(0,0),R(0,1),R(0,2)
               ,R(1,0),R(1,1),R(1,2)
               ,R(2,0),R(2,1),R(2,2));
}

/*
 * Generates a new fft grid that is compatible with the given rotation matrix.
 * Right now sets Ni=Nj whenever R(i,j)*Nj/Ni is not an integer. 
 * This might not be optimal, reconsider if needed.
 */ 
auto generate_consistent_fft_mesh(nda::ArrayOfRank<1> auto& mesh,
                                          nda::ArrayOfRank<2> auto const& R,
                                          double tol = 1e-4)
{
  int ndim = mesh.size();
  bool redo = false;
  bool changed = false;
  utils::check( R.shape() == std::array<long,2>{ndim,ndim}, "Shape mismatch." );
  do
  {
    redo = false;
    for(int i=0; i<ndim; ++i) {
      for(int j=i+1; j<ndim; ++j) {
        auto x = R(i,j)*mesh(j)/mesh(i);
        if (std::abs(x - std::round(x)) > tol ) {
          mesh(i) = std::max(mesh(i),mesh(j));
          mesh(j) = mesh(i); 
          redo = true;
          changed = true;
        } 
        x = R(j,i)*mesh(i)/mesh(j);
        if (std::abs(x - std::round(x)) > tol ) { 
          mesh(i) = std::max(mesh(i),mesh(j));
          mesh(j) = mesh(i); 
          redo = true;
          changed = true;
        } 
      }
    }
  } while(redo);
  check_rotation_compatible_with_fft_mesh(mesh,R,tol);
  return changed;
}

/*
 * Given a list of symmetry operations and an input fft mesh, this routine generates a new mesh
 * compatible with all operations. 
 */
auto generate_consistent_fft_mesh(nda::ArrayOfRank<1> auto mesh,
                                  std::vector<symm_op> const& symm_list, 
                                  double tol = 1e-4,
                                  std::string loc = "",
                                  bool inverse = false)
{
  // MAM: everytime a change happens, it invalidates the previous checks
  //      Repeat until change stops
  int ndim = mesh.size();
  int nmax = (ndim*(ndim-1))/2;
  nda::array<int, 1> new_mesh(mesh);
  bool changed = false;
  do 
  { 
    changed = false;
    utils::check(nmax-- > 0, "Error: Failed to generate consistent fft grid."); 
    for( auto& S : symm_list )
      changed = (changed or utils::generate_consistent_fft_mesh(new_mesh,(inverse?S.Rinv:S.R),tol));
  } while( changed ); 
  // now check
  for( auto& S : symm_list )
    check_rotation_compatible_with_fft_mesh(new_mesh,(inverse?S.Rinv:S.R),tol);
  if( not (mesh == new_mesh) ) 
    app_log(2, std::string(" Found symmetry operation incompatible with minimum mesh in {}. Adjusting grid. \n") +
               "    Minimum mesh: ({},{},{}) \n" +
               "    Adjuested mesh: ({},{},{}) \n",loc,mesh(0),mesh(1),mesh(2),new_mesh(0),new_mesh(1),new_mesh(2));
  return new_mesh; 
}

/**
 * Transforms k2g (the mapping from the kinetic grid to the fft grid)
 * by a provided symmetry operation.
 *
 * Given a list of G-vectors indexes (with respect to the provided fft mesh)
 * corresponding to a k-point, k_in, and a symmetry operation, Rinv, 
 * this routine transform the indexes into those corresponding to the rotated kpoint
 * k_out = sg * k_in * Rinv, where sg = -1.0 when trev=true and 1.0 otherwise. 
 * In addition, if the provided pointer is not null, 
 * returns the multiplicative term associated with fractional translations.
 *
 * In general:
 *   k_out = sg * k_in * Rinv + Gs, Gs is a shift 
 *   G_out = sg * G_in * Rinv - Gs
 *   u ( G_out ) = u ( G_in ) * exp(-i sg * Rinv * (G_in + k_in) * T)
 *
 * Note: The rotation matrix, Rinv, is compatible with periodic boundary conditions
 * in 'G' space if and only if Rinv(i,j)*mesh(j)/mesh(i) is an integer, for all (i,j).
 * The routine checks for this condition.  
 *
 * On entry:
 *   k2g(n) = index of G_in(n) in fft grid, for n in {0,npw}.
 *   
 * On exit:
 *   k2g(n) = index of G_out(n).
 *   Xft(n) = exp(-i sg * Rinv * (G_in + k_in) * T)
 *
 */
void transform_k2g(bool trev, 
                nda::stack_array<double, 3, 3> const& Rinv, 
		nda::stack_array<double, 3> const& Gs, 
		nda::stack_array<long, 3> const& mesh, 
		nda::ArrayOfRank<1> auto const& k_in,
		nda::ArrayOfRank<1> auto &&k2g, 
		nda::ArrayOfRank<1> auto *Xft)
{
  constexpr auto MEM = memory::get_memory_space<std::decay_t<decltype(k2g())>>();
  utils::check(mesh.size() == 3,"transform_k2g: mesh.size() != 3");
  utils::check(k_in.size() == 3,"transform_k2g: k_in.size() != 3");
  utils::check(Gs.size()   == 3,"transform_k2g: Gs.size() != 3");
  if( Xft != nullptr )
    utils::check(Xft->size() == k2g.size(),"transform_k2g: Xft->size() != npw");

  // check that Rinv is compatible with mesh
  check_rotation_compatible_with_fft_mesh(mesh,Rinv,1e-6);
  
  if constexpr (MEM==HOST_MEMORY) {
    int err=0;
    auto F = utils::detail::transform_k2g<decltype(k2g())>{(trev?-1.0:1.0),
            mesh,Gs,Rinv,k2g(),&err};
    // MAM: Use OpenMP!!!
    std::ranges::for_each(nda::range(k2g.extent(0)),F);
    utils::check(err==0, "Error in transform_k2g"); 
  } else {
#if defined(ENABLE_DEVICE)
    // MAM: no Xft yet 
    utils::check(Xft == nullptr, "Finish fractional translations in GPU!");
    kernels::device::transform_k2g(trev,Rinv,Gs,mesh,k2g());
#else
    static_assert(MEM!=HOST_MEMORY,"Error: Device dispatch without device support.");
#endif
  }

} 

void transform_k2g(bool trev, symm_op const& S,
                nda::ArrayOfRank<1> auto const& Gs,
                nda::ArrayOfRank<1> auto const& mesh,
                nda::ArrayOfRank<1> auto const& k_in,
                nda::ArrayOfRank<1> auto &&k2g,
                nda::ArrayOfRank<1> auto *Xft)
{
  transform_k2g(trev,S.Rinv,Gs,mesh,k_in,k2g,Xft); 
}

/*
 * Applies symmetry rotation (in place) to miller indices.
 * Indices are assumed to be in [N/2-N+1,N/2] range
 *
 * Note: This routine does not check if the rotation is compatible with the fft mesh.
 */
void transform_miller_indices(bool trev, symm_op const& S, 
		nda::ArrayOfRank<1> auto const& Gs, 
		nda::ArrayOfRank<2> auto &&m) 
{
  using value_t = typename std::decay_t<decltype(m)>::value_type;
  long npw = m.extent(0);
  utils::check(m.extent(1) == 3,"Size mismatch.");
  utils::check(Gs.size()   == 3,"Size mismatch.");
  double sg = (trev?-1.0:1.0); 

  // ignoring T and Xft term for now, need consistency (e.g. everything in crystal coords)
  for( auto i : nda::range(npw) ) {

    double n0(m(i,0)), n1(m(i,1)), n2(m(i,2));
    // G*Rinv - Gs 
    double ni_d = double(n0)*S.Rinv(0,0) + double(n1)*S.Rinv(1,0) + double(n2)*S.Rinv(2,0) - Gs(0);
    double nj_d = double(n0)*S.Rinv(0,1) + double(n1)*S.Rinv(1,1) + double(n2)*S.Rinv(2,1) - Gs(1);
    double nk_d = double(n0)*S.Rinv(0,2) + double(n1)*S.Rinv(1,2) + double(n2)*S.Rinv(2,2) - Gs(2);

    // trev
    ni_d *= sg;
    nj_d *= sg;
    nk_d *= sg;
 
    m(i,0) = value_t(std::round(ni_d)); 
    m(i,1) = value_t(std::round(nj_d)); 
    m(i,2) = value_t(std::round(nk_d)); 

    utils::check(std::abs( ni_d - double(m(i,0)) ) < 1e-6, "transform_miller_indices: Problem with symmetry rotation - ni: {}",ni_d);
    utils::check(std::abs( nj_d - double(m(i,1)) ) < 1e-6, "transform_miller_indices: Problem with symmetry rotation - nj: {}",nj_d);
    utils::check(std::abs( nk_d - double(m(i,2)) ) < 1e-6, "transform_miller_indices: Problem with symmetry rotation - nk: {}",nk_d);

  }
} 

/**
 * Transforms r(n) (index of element in real-space fft grid)
 * by a provided symmetry operation.
 *
 * Given a list of r-vector indexes (with respect to the provided fft mesh)
 * this routine transform the indexes into those corresponding to the rotated r-vector
 * r_out = S * r_in. 
 *
 * Note: This routine checks that the rotation is compatible with the fft grid. 
 *
 * On entry:
 *   r(n) = array of indexes of r-vectors in fft grid.
 *   
 * On exit:
 *   r(n) = index of r_out(n) = S * r_in(n).
 *
 *  MAM: no fractional translations for now!!! Finish!!!
 */
void transform_r(symm_op const& S, 
		nda::ArrayOfRank<1> auto const& T_frac, 
		nda::ArrayOfRank<1> auto const& mesh, 
		nda::ArrayOfRank<1> auto &&r) 
{
  long nr = r.size();
  utils::check(mesh.size() == 3,"Size mismatch.");
  utils::check(T_frac.size() == 3,"Size mismatch.");

  // check that Rinv is compatible with mesh
  check_rotation_compatible_with_fft_mesh(mesh,S.R,1e-6);

  // ignoring T for now, need consistency (e.g. everything in crystal coords)
  long NX = mesh(0), NY = mesh(1), NZ = mesh(2);
  long NX2 = NX/2, NY2 = NY/2, NZ2 = NZ/2;
  long nnr = NX*NY*NZ;
  double N10 = double(NY)/double(NX);
  double N01 = double(NX)/double(NY);
  double N12 = double(NY)/double(NZ);
  double N21 = double(NZ)/double(NY);
  double N02 = double(NX)/double(NZ);
  double N20 = double(NZ)/double(NX);
  for(auto i : nda::range(nr) ) {

    long n = r(i);
    long n2 = n%NZ; if( n2 > NZ2 ) n2 -= NZ;
    long n_ = n/NZ;
    long n1 = n_%NY; if( n1 > NY2 ) n1 -= NY;
    long n0 = n_/NY; if( n0 > NX2 ) n0 -= NX;
    utils::check(std::abs(n0) <= NX2, "transform_r: Index out of range: n0:{}, NX2:{}",n0,NX2);
    utils::check(std::abs(n1) <= NY2, "transform_r: Index out of range: n1:{}, NY2:{}",n1,NY2);
    utils::check(std::abs(n2) <= NZ2, "transform_r: Index out of range: n2:{}, NZ2:{}",n2,NZ2);

    // R*r + T_frac 
    double ni_d = S.R(0,0)*double(n0) + S.R(0,1)*N10*double(n1) + S.R(0,2)*N20*double(n2); // + T_frac(0);
    double nj_d = S.R(1,0)*N01*double(n0) + S.R(1,1)*double(n1) + S.R(1,2)*N21*double(n2); // + T_frac(1);
    double nk_d = S.R(2,0)*N02*double(n0) + S.R(2,1)*N12*double(n1) + S.R(2,2)*double(n2); // + T_frac(2);
 
    long ni_i = long(std::round(ni_d)); 
    long nj_i = long(std::round(nj_d)); 
    long nk_i = long(std::round(nk_d)); 

    utils::check(std::abs( ni_d - double(ni_i) ) < 1e-6, "transform_r: Problem with symmetry rotation - ni: {}",ni_d);
    utils::check(std::abs( nj_d - double(nj_i) ) < 1e-6, "transform_r: Problem with symmetry rotation - nj: {}",nj_d);
    utils::check(std::abs( nk_d - double(nk_i) ) < 1e-6, "transform_r: Problem with symmetry rotation - nk: {}",nk_d);

    while(ni_i<0) ni_i += NX;
    while(nj_i<0) nj_i += NY;
    while(nk_i<0) nk_i += NZ;
    while(ni_i>=NX) ni_i -= NX;
    while(nj_i>=NY) nj_i -= NY;
    while(nk_i>=NZ) nk_i -= NZ;

    r(i) = (ni_i*NY + nj_i)*NZ + nk_i;    

    utils::check(r(i) >= 0 and r(i) < nnr, "Error in transform_r: Transformed index out of bounds: {}",r(i));

  }
} 

auto generate_qsymm_maps(bool use_trev,
                         std::vector<symm_op> const& symm_list,
                         nda::ArrayOfRank<1> auto const& qp_symm,
                         [[maybe_unused]] nda::ArrayOfRank<1> auto const& qp_trev,
                         [[maybe_unused]] int nkpts_ibz,
                         nda::ArrayOfRank<2> auto const& kpts,
                         int nqpts_ibz,
                         nda::ArrayOfRank<2> auto const& Qpts)
{    
  decltype(nda::range::all) all;
  nda::stack_array<double,3> kp;
  int nkpts = kpts.extent(0);
  int nqpts = Qpts.extent(0);

  // cheating a bit, since it compares to kp! Careful!
  auto comp = [&kp](nda::ArrayOfRank<1> auto&& a) {
      // doing this by hand, not sure what's a better way
      double di = std::abs(a(0)-kp(0)); 
      if( std::abs(di-1.0) < 1e-4 ) di = 0.0;
      if( std::abs(di-2.0) < 1e-4 ) di = 0.0;
      if( std::abs(di-3.0) < 1e-4 ) di = 0.0;
      double dj = std::abs(a(1)-kp(1)); 
      if( std::abs(dj-1.0) < 1e-4 ) dj = 0.0;
      if( std::abs(dj-2.0) < 1e-4 ) dj = 0.0;
      if( std::abs(dj-3.0) < 1e-4 ) dj = 0.0;
      double dk = std::abs(a(2)-kp(2)); 
      if( std::abs(dk-1.0) < 1e-4 ) dk = 0.0;
      if( std::abs(dk-2.0) < 1e-4 ) dk = 0.0;
      if( std::abs(dk-3.0) < 1e-4 ) dk = 0.0;
      return di + dj + dk < 1e-12;
  };

  // find used symmetries and create a map
  std::vector<int> syms(symm_list.size(),-1);
  std::vector<int> unique;
  unique.reserve(symm_list.size());
  for( auto is: qp_symm ) {
    if(syms[is] < 0) {
      syms[is] = unique.size();
      unique.emplace_back(is);
    }
  }  
  int nsymm = unique.size(); 
  auto qsymms = nda::array<int, 1>::zeros({nsymm});
  auto nq_per_s = nda::array<int, 1>::zeros({nsymm});
  auto ks_to_k = nda::array<int, 2>::zeros({nsymm, nkpts});
  auto Qs = nda::array<int, 2>::zeros({nsymm, 2*nqpts_ibz});
  auto qs_to_q = nda::array<int, 2>::zeros({nsymm, nqpts_ibz});

  if(nsymm == 1 and unique[0] == 0 and not use_trev) {
    // identity, nothing to do
    qsymms(0) = 0;
    nq_per_s(0) = nqpts;
    ks_to_k(0,all) = nda::arange<int>(0,nkpts);
    Qs(0,nda::range(nqpts_ibz)) = nda::arange<int>(0,nqpts_ibz); 
    qs_to_q = Qs; 
    return std::make_tuple(std::move(qsymms),std::move(nq_per_s),std::move(ks_to_k),std::move(Qs),std::move(qs_to_q));
  }

  for( auto [i,is]: itertools::enumerate(unique) ) qsymms(i) = is; 
  for( auto is: qp_symm ) nq_per_s(syms[is])++;
  for( auto [i,is]: itertools::enumerate(qsymms) ) 
  { 
    int ns=0;
    for( auto [j,iqs] : itertools::enumerate(qp_symm) )
      if( iqs == is )
        Qs(i,ns++) = j;
    utils::check(ns == nq_per_s(i), "ns != nq_per_s: Logic error!");
    for( int ik=0; ik<nkpts; ik++ ) {
      nda::blas::gemv(1.0,nda::transpose(symm_list[is].R),kpts(ik,all),0.0,kp);
      bool found = false;
      for(int n=0; n<nkpts; n++ )
        if(comp(kpts(n,all))) {
          utils::check(not found,"count_if ks: Logic error");
          ks_to_k(i,ik) = n;
          found=true;
        }
      utils::check(found, "Error in generate_qsymm_maps: ki*S = kj");
    }
    for( int iq=0; iq<nqpts_ibz; iq++ ) {
      nda::blas::gemv(1.0,nda::transpose(symm_list[is].R),Qpts(iq,all),0.0,kp);
      bool found = false;
      for(int n=0; n<nqpts; n++ )
        if(comp(Qpts(n,all))) {
          utils::check(not found,"count_if qs: Logic error");
          qs_to_q(i,iq) = n;
          found=true;
        }
      utils::check(found, "Error in generate_qsymm_maps: Qi*S = Qj");
    }
  }
  return std::make_tuple(std::move(qsymms),std::move(nq_per_s),std::move(ks_to_k),std::move(Qs),std::move(qs_to_q));
}

/*
 * Partitions and distributes the fft real-space among the provided communicator, 
 * keeping symmetry equivalent points in the same mpi task.
 * The routine checks that the symmetry rotations are compatible with the fft grid.
 * MAM: Algorithm is simple right now, might need optimization.
 */

auto partition_irreducible_r_grid(utils::Communicator auto& comm,
                                  nda::ArrayOfRank<1> auto const& mesh,
                                  std::vector<utils::symm_op> const& symms, 
                                  [[maybe_unused]] nda::ArrayOfRank<1> auto const& slist)
{
  int nproc = comm.size();
  long NX = mesh(0), NY = mesh(1), NZ = mesh(2); 
  long nnr = NX*NY*NZ; 
  nda::array<long,1> rg(nnr,-1);
  double N10 = double(NY)/double(NX);
  double N01 = double(NX)/double(NY);
  double N12 = double(NY)/double(NZ);
  double N21 = double(NZ)/double(NY);
  double N02 = double(NX)/double(NZ);
  double N20 = double(NZ)/double(NX);

  // check that Rinv is compatible with mesh
  for( auto& S : symms ) 
    check_rotation_compatible_with_fft_mesh(mesh,S.R,1e-6);

  auto apply_S = [&] (long n0, long n1, long n2, auto const& R) {
    long ni = long(std::round(R(0,0)*double(n0) + R(0,1)*N10*double(n1) + R(0,2)*N20*double(n2))); // + T_frac(0);
    long nj = long(std::round(R(1,0)*N01*double(n0) + R(1,1)*double(n1) + R(1,2)*N21*double(n2))); // + T_frac(1);
    long nk = long(std::round(R(2,0)*N02*double(n0) + R(2,1)*N12*double(n1) + R(2,2)*double(n2))); // + T_frac(2);
    while(ni<0) ni += NX;
    while(nj<0) nj += NY;
    while(nk<0) nk += NZ;
    while(ni>=NX) ni -= NX;
    while(nj>=NY) nj -= NY;
    while(nk>=NZ) nk -= NZ;
    return std::make_tuple(ni,nj,nk,(ni*NY + nj)*NZ + nk); 
  };

  auto find_sym_eqv = [&] (long n)
  {
    std::vector<long> v;
    v.emplace_back(n);
    rg(n) = n;
    long n2 = n%NZ; 
    long n_ = n/NZ;
    long n1 = n_%NY; 
    long n0 = n_/NY; 
    for( auto& S : symms ) {

      long ni=n0,nj=n1,nk=n2,Nr;
      bool found = false;
      // what is the largest possible order? Guessing 12?
      for( int t=0; t<12; ++t ) {
        std::tie(ni,nj,nk,Nr) = apply_S(ni,nj,nk,S.R); 
        if(Nr == n) {
          found=true;
          break;
        } 
        utils::check(Nr >= 0 and Nr < nnr,"Error: Index out of bounds: {}",Nr); 
        utils::check(rg(Nr) == -1 or rg(Nr) == n, 
                     "Error in partition_irreducible_r_grid: Problems with IR generation: n:{}, Nr:{}, rg(Nr):{}",n,Nr,rg(Nr));
        if(rg(Nr) == -1) {
          v.emplace_back(Nr);
          rg(Nr) = n;
        }
      }
      utils::check(found,"Error: Logic problem in partition_irreducible_r_grid::find_sym_eqv");

    } 
    return v;
  };

  auto verify_sym_eqv = [&] (long n)
  {
    long Ns = rg(n);
    long n2 = n%NZ;     
    long n_ = n/NZ;
    long n1 = n_%NY; 
    long n0 = n_/NY; 
    for( auto& S : symms ) {
      auto [ni,nj,nk,Nr] = apply_S(n0,n1,n2,S.R);
      utils::check(Nr >= 0 and Nr < nnr,"Error: Index out of bounds: {}",Nr);
      utils::check(rg(Nr) == Ns, 
                   "Error in partition_irreducible_r_grid: Problems with IR verification: n:{}, Nr:{}, rg(Nr):{}",n,Nr,rg(Nr));
    } 
  };

  // root generates grid, since I don't know how to do it in parallel  
  std::vector<nda::array<long,1>> r_set_v;
  nda::array<long,1> set_sizes(nproc);
  if(comm.root()) {
    std::vector<std::vector<long>> r_set;
    for( auto n : nda::range(nnr) ) 
      if( rg(n) == -1 ) r_set.emplace_back(find_sym_eqv(n));
    utils::check( std::all_of(rg.begin(),rg.end(),[](auto && a) { return a>=0; }), 
                  "Error: Found unassigned points in partition_irreducible_r_grid");
    long ns = r_set.size();
    app_log(3," Size of IR fft-rgrid: {}", ns); 
    utils::check(ns >= comm.size(), "Error: More processors than IR r-grid groups:{}.",ns);

    // now distribute over processors
    std::vector<std::vector<long>> r_p(nproc);
    long av_sz = nnr/comm.size();
    auto idx = nda::arange(0,ns);
    std::sort(idx.begin(),idx.end(),
       [&] (auto const& a, auto const&b) {return r_set[a].size() > r_set[b].size();});

    for( auto i : idx ) { 
      // if it fits, dump. Otherwise dump in smallest stack
      auto it = std::find_if( r_p.begin(), r_p.end(), 
              [&] (auto const& v) { return v.size()+r_set[i].size() <= av_sz; }); 
      if( it == r_p.end() ) 
        it = std::min_element( r_p.begin(), r_p.end(),    
              [&] (auto const& a, auto const& b) { return a.size() < b.size(); }); 
      it->reserve(it->size()+r_set[i].size());
      for( auto v : r_set[i] )
        it->emplace_back(v);        
    }
    for( auto& v : r_p )
      std::sort(v.begin(),v.end());
   
    // broadcast results
    for( auto i : nda::range(nproc) ) set_sizes(i) = r_p[i].size();
    comm.broadcast_n(set_sizes.data(),nproc,0);
    app_log(4," IR set sizes"); 
    for( long i=0; i<nproc; i++ ) {
      app_log(4," i:{}  size:{}",i,set_sizes(i)); 
      r_set_v.emplace_back(nda::array<long,1>(set_sizes(i)));
      auto it = r_set_v[i].begin();
      for( auto jn : r_p[i] ) *(it++) = jn; 
      comm.broadcast_n(r_set_v[i].data(),set_sizes(i),0);
    }
  } else {
    comm.broadcast_n(set_sizes.data(),nproc,0);
    for( long i=0; i<nproc; i++ ) {
      r_set_v.emplace_back(nda::array<long,1>(set_sizes(i)));
      comm.broadcast_n(r_set_v[i].data(),set_sizes(i),0);
    }
  }

  // verify grid in parallel
  // turn off when you are sure this is always true!
  comm.broadcast_n(rg.data(),rg.size(),0);
  for( long n=comm.rank(); n<nnr; n+=comm.size() ) 
    verify_sym_eqv(n);

  return r_set_v;
}

/*
 * MAM: You only need d(R,k) for k_IBZ and for R in the little group of k_IBZ.
 *      Use the fact that any rotation in the left coset can be written in terms of
 *      the chosen symmetry at that kpoint and some element of the little group at k_IBZ.
 *      This is the same idea used to calculate orbital orverlaps for wannier90 from IBZ data only.
 *
 * Generates rotation matrices between symmetry related kpoints in the BZ, 
 * defined as:
 *     d(R, k)(a,b) = int_r conj(psi(kS^{-1},a,S*r)) * psi(k,b,r). 
 * Notice that, for k outside the IBZ, d( R, k ) = d( R*R0, k_ibz), 
 * where k_IBZ*R0^{-1} = k. Hence, we only need to calculate 'd' explicitly for k in IBZ
 * and for all {R, R0(k)}. If k involves trev symmetry (trev(k) == true), then:
 * d( R, k ) = conj(d( R*R0, k_ibz)).
 * The routine returns:
 *   - nda::array<int,2>(s,k) -> n: location of the rotation matrix in the array 
 *                                  for a given {s,k} pair, k in the full BZ.
 *   - (if Sparse==true:)  std::vector<sparse_mat> with the list of rotation matrices
 *                         (indexed by mapping) in sparse format
 *   - (if Sparse==false:) shared_array<C,3>(n,a,b) with the rotation matrices
 *                         (indexed by mapping) in dense tensor format
*/
// MAM: Need to be able to allocate sparse matrix in shared memory!
//      If the matrix is fairly dense, then this will eat up a lot of memory!
template<bool Sparse = true, typename MF_t>
auto generate_dmatrix(MF_t &mf,
                      std::vector<utils::symm_op> const& symms, 
                      nda::ArrayOfRank<1> auto const& slist,
                      bool assume_irreducible=true,
                      bool normalize=true)
{
  constexpr auto MEM = HOST_MEMORY;
  decltype(nda::range::all) all;
  using Array_view_3D_t = memory::array_view<MEM,ComplexType,3>;
  using sp_mat = typename math::sparse::csr_matrix<ComplexType,MEM,int,int>;
  using math::shm::make_shared_array;

  // checks!
  utils::check(mf.has_wfc_grid(), "Error in generate_dmat: has_wfc_grid==false");

  auto mpi = mf.mpi();
  auto comm = mpi->comm;
  auto node_comm = mpi->node_comm;
  // vector of sparse matrices, used only if Sparse==true
  std::vector<sp_mat> dmat_s;
  // shared memory array, used only if Sparse==false. Will resize later if needed
  auto shm_dmat = make_shared_array<Array_view_3D_t>(*mpi, {0,mf.nbnd(),mf.nbnd()});
  if(slist.extent(0) == 0) {
    nda::array<int,2> sk_to_n(0,0); 
    if constexpr (Sparse) {
      return std::make_tuple(sk_to_n,dmat_s);
    } else {
      return std::make_tuple(sk_to_n,shm_dmat);
    }
  }
  
  auto const & eigv = mf.get_sys().eigval();
  bool irred = (assume_irreducible and 
                std::all_of(eigv.begin(),eigv.end(),
                            [&](auto const& a) {return a!=0;}));
  constexpr double degen_tol = 1e-4; // tolerance used to group energy eigenvalues into degenerate sets
  auto const& bz = mf.bz();
  long nkpts = bz.nkpts;
  long nkpts_ibz = bz.nkpts_ibz;
  long nkpts_trev_pairs = bz.nkpts_trev_pairs;
  auto kpts = mf.kpts_crystal();
  auto const & kp_to_ibz = bz.kp_to_ibz;
  auto const & kp_symm = bz.kp_symm;
  long nbnd = mf.nbnd();
  long ns = slist.extent(0);
  long nstot = symms.size();
  auto wfc_g = mf.wfc_truncated_grid();
  auto mesh = wfc_g->mesh();
  // index of G vectors in fft grid (G vectors inside the cutoff)
  auto gv_to_fft = wfc_g->gv_to_fft();   
  // inverse mapping to gv_to_fft
  auto fft_to_gv = wfc_g->fft_to_gv();   
  nda::array<ComplexType,1> *Xft = nullptr;
  nda::array<ComplexType,3> psi(2,nbnd,wfc_g->size());
  nda::stack_array<double,3> kp;
  nda::stack_array<double,3> Gs;
  nda::array<double,2> RR0(3,3);
  memory::unified_array<long,1> Gout(wfc_g->size());

  if(irred) {
    // a hardcoded threshold for sorted eigenvalues.
    double thresh = 1e-6;
    for (int ik=0; ik<nkpts; ++ik) {
      auto iter = std::adjacent_find(eigv(0,ik,all).begin(), eigv(0,ik,all).end(),
                                     [&thresh](double a, double b) { return a-b>thresh; });
      auto idx = std::distance(eigv(0,ik,all).begin(), iter);
      // don't use utils::check, since it will evaluate the parameters (e.g. in debug mode) which will
      // cause a seg fault when the check is fine 
      if( iter!=eigv(0,ik,all).end() )
        APP_ABORT("Error in generate_dmat: assume_irreducible with unsorted eigenvalues at ik={}: \n"
                   "Error in generate_dmat: assume_irreducible with unsorted eigenvalues at ik={}: \n"
                   "eigv(i={})={}, eigv(i+1={})={}, threshold = {}.",
                   ik, idx, eigv(0,ik,idx), idx+1, eigv(0,ik,idx+1), thresh);
    }
  }

  // cheating a bit, since it is hardwired to compare to kp! Careful!
  auto comp = [&kp](nda::ArrayOfRank<1> auto&& a) {
    // doing this by hand, not sure what's a better way
    double di = std::abs(a(0)-kp(0));
    double dj = std::abs(a(1)-kp(1));
    double dk = std::abs(a(2)-kp(2));
    di -= std::floor(di); if( std::abs(di-1.0) < 1e-4 ) di = 0.0;
    dj -= std::floor(dj); if( std::abs(dj-1.0) < 1e-4 ) dj = 0.0;
    dk -= std::floor(dk); if( std::abs(dk-1.0) < 1e-4 ) dk = 0.0;
    return di + dj + dk < 1e-12;
  };

  auto comp_R = [](nda::ArrayOfRank<2> auto&& R1, nda::ArrayOfRank<2> auto&& R2) {
      return nda::frobenius_norm(R1-R2) < 1e-10;
  };

  auto find_k2 = [&kpts,&comp] () {
    for(int ik=0; ik<kpts.extent(0); ik++) {
      if(comp(kpts(ik,nda::range::all))) 
        return ik; 
    }
    utils::check(false, "Could not find k2.");
    return -1;
  };

  // list of {symm,kp} pairs needed 
  std::vector<std::pair<int,int>> sk; 
  sk.reserve(ns*nkpts_ibz); 
  // first term is reserve for the identity
  sk.emplace_back(std::make_pair(-1,-1));  

  // returned mapping, from {s,k} to dmat index (e.g. some element in sk list) 
  nda::array<int,2> sk_to_n(ns,nkpts-nkpts_trev_pairs);
  sk_to_n() = -1; // safeguard 

  // array to keep track of existing symmetries in sk list 
  nda::array<int,1> sv(nstot);

  // all kpoints outside IBZ, excluding trev ones
  nda::range k_rng(nkpts_ibz,nkpts-nkpts_trev_pairs); 

  // construct sk and sk_to_n arrays
  // find a way to parallelize this process
  for( auto ik : nda::range(nkpts_ibz) ) {
    sv() = -1;
    // first push all from slist
    for( auto [is,s] : itertools::enumerate(slist) ) {
      // by convention, d( kp_symm(k), kp_to_ibz(k) ) = delta(i,j)
      auto s_ = s;  // clang bypass: capturing a structured binding is not yet supported in OpenMP
      if( std::any_of(k_rng.begin(), k_rng.end(), 
             [&](auto const&& a) {return ((kp_to_ibz(a)==ik) and (kp_symm(a)==s_));}) ) {
        sk_to_n(is,ik) = 0;  // default position of identity!
        sv(s) = 0; 
      } else {
        sk_to_n(is,ik) = sk.size();
        sv(s) = sk.size(); 
        sk.emplace_back(std::make_pair(int(s),int(ik)));
      }
    }
    // now add all k-points from outside IBZ that are symmetry related to ik 
    for( auto kr : k_rng ) {
      if(bz.kp_to_ibz(kr) != ik) continue;
      for( auto [is,s] : itertools::enumerate(slist) ) {
        // by convention, d( kp_symm(k)^{-1}, k ) = delta(i,j)
        if( comp_R(symms[kp_symm(kr)].Rinv,symms[s].R) ) {
          sk_to_n(is,kr) = 0;
        } else {
          // RR0
          nda::blas::gemm(1.0,symms[s].R,symms[kp_symm(kr)].R,0.0,RR0);
          // look for RR0 in symms
          int ss0 = -1;
          for(int ss=0; ss<nstot; ++ss) 
            if(comp_R(symms[ss].R,RR0)) { 
              ss0 = ss; 
              break; 
            }
          utils::check(ss0 >= 0 and ss0 < nstot, "Error in generate_dmat: Could not find compound symmetry");
          if( sv(ss0) >= 0 ) {
            sk_to_n(is,kr) = sv(ss0);
          } else {
            sk_to_n(is,kr) = sk.size();
            sv(ss0) = sk.size(); 
            sk.emplace_back(std::make_pair(ss0,int(ik)));
          }
        }
      }
    }
  }
  utils::check( std::all_of( nda::range(nkpts-nkpts_trev_pairs).begin(), nda::range(nkpts-nkpts_trev_pairs).end(),
             [&](auto const&& a) {return a>=0;}), "Error in generate_dmat: Unassigned (s,k) pair");

  // resize and setup temporary objects
  sp_mat sp_d({1,1},1);
  nda::array<ComplexType,2> Temp(0,0);
  if constexpr (Sparse) {
    long nz = nbnd;
    if(irred) {
      // Find the largest degenerate eigenvalue block over all k-points
      // required for MCH hamiltonian at low potential strength  due to accidental degeneracies
      nz = 1;
      for( long ik=0; ik<nkpts; ++ik ) {
        long ib=0;
        while(ib < nbnd) {
          long nb=1;
          while( ib+nb<nbnd and std::abs(eigv(0,ik,ib)-eigv(0,ik,ib+nb)) < degen_tol ) nb++;
          nz = std::max(nz,nb);
          ib+=nb;
        }
      }
      app_log(2, "  generate_dmatrix: largest degenerate block = {}\n", nz);
    }
    sp_d = sp_mat({nbnd,nbnd},nz);
    Temp = nda::array<ComplexType,2>(nz,nz);
  } else {
    shm_dmat = make_shared_array<Array_view_3D_t>(*mpi, {sk.size(),nbnd,nbnd});
    shm_dmat.set_zero();

  }

// MAM: This assumes an orthonormal basis, the general formula is:
//      F = d * S, where F is the overlap evaluated below (the current dmat) and S is the overlap matrix 
// for states at k*Rinv. FIX!!!

  // The mesh needs to be compatible with all the relevant symmetry rotations.
  // If this is not the case, generate a new mesh, a new mapping from the old mesh to the new,
  // and apply rotations/contractions on the new mesh.
  // This does not have any implications outside this routine, only used as an intermediate space with appropriate
  // dimensions to allow symmetry rotations and matrix element calculations.
  // MAM: Currently forcing the mesh to be compatible from the start, but this can still be done if needed

  int kold = -1;
  auto [isk0, isk1] = itertools::chunk_range(0, sk.size(), comm.size() ,comm.rank());
  long bad_norm_count = 0;
  for( long isk=isk0; isk<isk1; ++isk ) { 

    if(isk == 0) { // first term is always the identity
      if constexpr (Sparse) {
        dmat_s.emplace_back(math::sparse::identity<ComplexType,MEM>(nbnd));
      } else {
        auto dsk = shm_dmat.local()(isk,nda::ellipsis{});
        nda::diagonal(dsk) = ComplexType(1.0);
      }
      continue;
    }

    int is = sk[isk].first;
    int ik = sk[isk].second;

    // read orbital if necessary
    if(kold != ik) {
      kold = ik;
      mf.get_orbital_set('w',0,ik,{0,nbnd},psi(0,all,all)); 
    }

    // kRinv = k2 - kp, kp = k2 - kRinv
    nda::blas::gemv(1.0,nda::transpose(symms[is].Rinv),kpts(ik,all),0.0,kp); 
    int k2 = find_k2();
    kp = kpts(k2,all) - kp;

    // RR0 = Rinv * R0
    nda::blas::gemm(1.0,symms[is].Rinv,symms[kp_symm(k2)].R,0.0,RR0);

    // Gs = kp * R0
    nda::blas::gemv(1.0,nda::transpose(symms[kp_symm(k2)].R),kp,0.0,Gs); 

    // rotate indexes
    // G -> (-1)^{trev} * ( G * Rinv * R0 - kp * R0 ) 
    Gout() = gv_to_fft();  // list of indexes of G vectors in truncated wfc grid
    utils::transform_k2g(bz.kp_trev(k2),RR0,Gs,mesh,kpts(ik,all),Gout,Xft);

    // rotate psi(k2)
    psi(1,all,all) = ComplexType(0.0);
    for( auto [i,n]: itertools::enumerate(Gout)) {
      if(fft_to_gv(n) >= 0) {
        if(bz.kp_trev(k2)) {
          psi(1,all,i) = psi(0,all,fft_to_gv(n));
        } else {
          psi(1,all,i) = nda::conj(psi(0,all,fft_to_gv(n)));
        }
      }
    }
    if(irred) {
      // assumes ordered eigenvalues, can be fixed otherwise
      if constexpr (Sparse) 
        sp_d.clear();
      int ib=0;
      while(ib < nbnd) {
        int nb=1;
        while( ib+nb<nbnd and std::abs(eigv(0,ik,ib)-eigv(0,ik,ib+nb)) < degen_tol ) nb++;
        nda::range b_rng(ib,ib+nb);
        if constexpr (Sparse) {
          // check for out of bounds error in dmatrix csr
          utils::check( nb <= Temp.extent(0),
                        "Error in generate_dmat: degenerate block of {} states exceeds "
                        "buffer size {} at ik={}, ib={}.", nb, Temp.extent(0), ik, ib);
          nda::blas::gemm(ComplexType(1.0),psi(1,b_rng,all),
                                           nda::transpose(psi(0,b_rng,all)),
                          ComplexType(0.0),Temp(nda::range(nb),nda::range(nb)));
          // dump into sp_d
          for( int a=0; a<nb; ++a )
            for( int b=0; b<nb; ++b )
              if( std::abs(Temp(a,b)) > 1e-8 )
                sp_d[ib+a][ib+b] = Temp(a,b);  
        } else {
          nda::blas::gemm(ComplexType(1.0),psi(1,b_rng,all),
                                           nda::transpose(psi(0,b_rng,all)),
                          ComplexType(0.0),shm_dmat.local()(isk,b_rng,b_rng));
        }
        ib+=nb;
      };
      if constexpr (Sparse)
        dmat_s.emplace_back(math::sparse::to_compact(sp_d));
    } else {
      if constexpr (Sparse) {
        nda::blas::gemm(ComplexType(1.0),psi(1,all,all),nda::transpose(psi(0,all,all)),
                        ComplexType(0.0),Temp);
        dmat_s.emplace_back(math::sparse::to_csr<MEM,int,int>(Temp,1e-8));
      } else {
        nda::blas::gemm(ComplexType(1.0),psi(1,all,all),nda::transpose(psi(0,all,all)),
                        ComplexType(0.0),shm_dmat.local()(isk,all,all));
      }
    }
    
    // check and normalize along rows, since breaking degenerate sets can cause leakage
    if(normalize) {
      for(int r=0; r<nbnd; r++) {
        double e = 0.0;
        if constexpr (Sparse) {
          auto vals = dmat_s.back()[r].values();
          e=std::sqrt(std::abs(nda::blas::dotc(vals,vals)));
          vals() /= e;
        } else {
          e=std::sqrt(std::abs(nda::blas::dotc(shm_dmat.local()(isk,r,all),shm_dmat.local()(isk,r,all))));
          shm_dmat.local()(isk,r,all) /= e;
        }
        if(std::abs(e-1.0) > 1e-3) bad_norm_count += 1;
      }
    }
  }

  // check # of Bloch orbitals with a incomplete degenerate set
  bad_norm_count = comm.all_reduce_value(bad_norm_count, std::plus<>{});
  if (bad_norm_count > 0)
    app_log(2, "  [WARNING] {} of Bloch orbitals in a reducible representation are not normalized.\n"
               "            This is because high-lying virtual bands in a reducible representation are not \n"
               "            included fully. Make sure degenerate sets of states are fully included. Otherwise, \n"
               "            this error is typically negligible as more and more virtual bands are included.\n",
               bad_norm_count);

  if constexpr (Sparse) {
    std::vector<sp_mat> dmat;
    dmat.reserve(sk.size());
    if( comm.size() == 1 ) {
      for(int i=0; i<dmat_s.size(); i++)
        dmat.emplace_back(dmat_s[i]);
    } else {
      // loop over ranks and bcast sparse arrays in compact format
      nda::array<long,1> sz(sk.size(),0);
      nda::array<char,1> buff(0);
      for(long i=0; i<dmat_s.size(); ++i)
        sz(isk0+i) = dmat_s[i].size_of_serialized_in_bytes(true); 
      comm.all_reduce_in_place_n(sz.data(),sk.size(),std::plus<>{});
      for(int r=0; r<comm.size(); r++) {
        auto [p0, p1] = itertools::chunk_range(0, sk.size(), comm.size(), r);
        long nr = std::accumulate(sz.data()+p0, sz.data()+p1, long(0));
        buff.resize(nr);
        if(comm.rank()==r) {
          for(long i=0, cnt=0; i<dmat_s.size(); ++i) {
            utils::check(isk0==p0, "Partition mismatch");
            dmat_s[i].serialize(buff.data()+cnt,sz(i+isk0),true);
            cnt+=sz(i+isk0);
          }
          dmat_s = std::vector<sp_mat>(0);
        }
        comm.broadcast_n(buff.data(),nr,r);
        for(long p=p0, cnt=0; p<p1; ++p) {
          dmat.emplace_back(sp_mat());
          dmat.back().deserialize(buff(::nda::range(cnt,cnt+sz(p))));
          cnt+=sz(p);
        }
      }
    }
    return std::make_tuple(std::move(sk_to_n),std::move(dmat));
  } else {
    node_comm.barrier();
    shm_dmat.all_reduce();
    node_comm.barrier();
    return std::make_tuple(std::move(sk_to_n),std::move(shm_dmat));
  }
}

/*
template<bool Sparse = true, typename MF_t>
auto generate_dmatrix_old(utils::Communicator auto& comm, 
                      MF_t &mf, 
                      std::vector<utils::symm_op> const& symms, 
                      nda::ArrayOfRank<1> auto const& slist,
                      bool assume_irreducible=true,
                      bool full_bz=true)
{
  constexpr auto MEM = HOST_MEMORY;
  if(slist.extent(0) == 0) {
    if constexpr (Sparse) {
      using s_mat = typename math::sparse::csr_matrix<ComplexType,MEM,int,int>;
      std::vector<s_mat> dmat_s;
      return dmat_s;
    } else {
      return memory::array<MEM,ComplexType,4>(0,0,0,0);
    }
  }

  decltype(nda::range::all) all;
  size_t nkpts = (full_bz)? mf.nkpts() : mf.nkpts_ibz();
  auto kpts = mf.kpts_crystal();
  size_t nbnd = mf.nbnd();
  size_t ns = slist.extent(0);
  auto npw = mf.npw();
  auto mesh = mf.fft_grid_dim();
  nda::array<ComplexType,1> *Xft = nullptr;
  memory::unified_array<long,2> k2g(1,mf.max_npw()); 
  nda::array<ComplexType,3> psi(2,nbnd,mf.max_npw());
  nda::stack_array<double,3> kp;
  nda::array<long,1> inv_map(mf.nnr());

  using math::shm::make_shared_array;
  using Array_view_4D_t = memory::array_view<MEM,ComplexType,4>;
  auto node_comm = comm.split_shared();
  auto internode_comm = comm.split(comm.rank()%node_comm.size(), comm.rank()/node_comm.size());
  auto sdmat = make_shared_array<Array_view_4D_t>(comm, internode_comm, node_comm,
                                                  {ns,nkpts,nbnd,nbnd});

  // cheating a bit, since it compares to kp! Careful!
  auto comp = [&kp](nda::ArrayOfRank<1> auto&& a) {
    // doing this by hand, not sure what's a better way
    double di = std::abs(a(0)-kp(0));
    double dj = std::abs(a(1)-kp(1));
    double dk = std::abs(a(2)-kp(2));
    di -= std::floor(di); if( std::abs(di-1.0) < 1e-4 ) di = 0.0;
    dj -= std::floor(dj); if( std::abs(dj-1.0) < 1e-4 ) dj = 0.0;
    dk -= std::floor(dk); if( std::abs(dk-1.0) < 1e-4 ) dk = 0.0;
    return di + dj + dk < 1e-12;
  };

  auto find_k2 = [&kpts,&comp] () {
    for(int ik=0; ik<kpts.extent(0); ik++) {
      if(comp(kpts(ik,nda::range::all))) 
        return ik; 
    }
    utils::check(false, "Could not find k2.");
    return -1;
  };

// MAM: This assumes an orthonormal basis, the general formula is:
//      F = d * S, where F is the overlap evaluated below (the current dmat) and S is the overlap matrix 
// for states at k*Rinv. FIX!!!
  sdmat.win().fence();
  for( int is=0, isk=0; is<ns; is++ ) { 
    for( int ik=0; ik<nkpts; ik++, isk++) {
      if(comm.rank() != isk%comm.size()) continue;
      auto k2g_ = k2g(nda::range(1),nda::range(npw(ik)));
      // kRinv = k2 - kp, kp = k2 - kRinv
      nda::blas::gemv(1.0,nda::transpose(symms[slist(is)].Rinv),kpts(ik,all),0.0,kp); 
      int k2 = find_k2();
      kp = kpts(k2,all) - kp;
      mf.get_orbital_set('k',0,k2,{0,nbnd},psi(0,all,all)); 
      // get k2g for kRinv and invert it 
      mf.get_k2g(nda::range(k2,k2+1),nda::range(npw(ik)),k2g_);
      inv_map()=-1;
      for( auto [a,b]: itertools::enumerate(k2g_) )
        inv_map(b) = a;
      // get k2g for ik
      mf.get_k2g(nda::range(ik,ik+1),nda::range(npw(ik)),k2g_);
      utils::transform_k2g(false,symms[slist(is)],kp,mesh,kpts(ik,all),k2g_(0,all),Xft);
      // rotate psi(k2)
      psi(1,all,all) = ComplexType(0.0);
      for( auto [i,n]: itertools::enumerate(k2g_))
        if(inv_map(n) >= 0)
          psi(1,all,i) = nda::conj(psi(0,all,inv_map(n)));
      mf.get_orbital_set('k',0,ik,{0,nbnd},psi(0,all,all)); 
      nda::blas::gemm(ComplexType(1.0),psi(1,all,nda::range(npw(ik))),
		                       nda::transpose(psi(0,all,nda::range(npw(ik)))),
                      ComplexType(0.0),sdmat.local()(is,ik,all,all));
      for(int r=0; r<nbnd; r++) {
        double e=std::sqrt(std::abs(nda::blas::dotc(sdmat.local()(is,ik,r,all),sdmat.local()(is,ik,r,all))));
        if(std::abs(e-1.0) > 1e-3)
          app_warning("Warning: Reducible representation is not normalized. Make sure degenerate sets of states are fully included. norm:{}",e*e);
        sdmat.local()(is,ik,r,all) /= e;
      }
    } 
  }
  sdmat.win().fence();
  sdmat.all_reduce();
  // MAM: improve this! You can do this above without storing full dmat in dense format
  if constexpr (Sparse) {
    using s_mat = typename math::sparse::csr_matrix<ComplexType,MEM,int,int>;
    std::vector<s_mat> dmat_s;
    dmat_s.reserve(ns*nkpts);
    for(int is=0; is<ns; ++is)
      for(int ik=0; ik<nkpts; ++ik)
        dmat_s.emplace_back(math::sparse::to_csr<MEM,int,int>(sdmat.local()(is,ik,nda::ellipsis{}),1e-8));
    return dmat_s;
  } else {
    memory::array<MEM,ComplexType,4> dmat(ns,nkpts,nbnd,nbnd);
    dmat() = ComplexType(0.0);
    for(int is=0; is<ns; ++is)
      for(int ik=0; ik<nkpts; ++ik)
        dmat(is,ik,nda::ellipsis{}) = sdmat.local()(is,ik,nda::ellipsis{});
    return dmat;
  }
}
*/

auto find_inverse_symmetry(nda::ArrayOfRank<1> auto const& qsymms,
                           std::vector<utils::symm_op> const& symm_list,
                           bool ignore_s0 = true)
{
  if( ignore_s0 and qsymms.extent(0)==1 and qsymms(0)==0 ) return nda::array<int,1>(0);
  int is0 = (ignore_s0?1:0);
  nda::array<int,1> slist(qsymms.extent(0) - is0);
  // find Sinv 
  for( auto [is,isym] : itertools::enumerate(qsymms) ) { 
    if(ignore_s0 and is == 0) {
      utils::check(isym==0,"Error in generate_dmatrix_for_Qpts: First symmetry not the identity.");
      continue;
    }
    bool found=false;
    auto& Rinv = symm_list[isym].Rinv;
    for( auto j=0; j<symm_list.size(); j++ ) { 
      auto& R = symm_list[j].R;
      double R00 = R(0,0)-Rinv(0,0), R01 = R(0,1)-Rinv(0,1), R02 = R(0,2)-Rinv(0,2);
      double R10 = R(1,0)-Rinv(1,0), R11 = R(1,1)-Rinv(1,1), R12 = R(1,2)-Rinv(1,2);
      double R20 = R(2,0)-Rinv(2,0), R21 = R(2,1)-Rinv(2,1), R22 = R(2,2)-Rinv(2,2);
      if( R00*R00 + R01*R01 + R02*R02 +
          R10*R10 + R11*R11 + R12*R12 +
          R20*R20 + R21*R21 + R22*R22 < 1e-10 ) {
        utils::check(not found, "Problems with inverse symmetey operation.");
        slist(is-is0) = j;
	found=true;
      } 
    }
    utils::check(found, "Could not find inverse symmetey operation.");
  }
  return slist; 
}

} // namespace utils

#endif
