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


#ifndef COQUI_VSPACE_FOCK_SIGMA_HPP
#define COQUI_VSPACE_FOCK_SIGMA_HPP

#include "numerics/iter_scf/diis/vspace.h"
namespace iter_scf {

// Implementation of vector algebra for Fock matrix and self-energy union
// The class must be initialized before usage

class FockSigma {
private:
    using Array_4D = nda::array<ComplexType, 4>;
    using Array_5D = nda::array<ComplexType, 5>;

    Array_4D _Fock;
    Array_5D _Sigma;
    double _mu;
    bool inited_F = false;
    bool inited_S = false;
    bool inited_mu = false;
public:
    FockSigma() {}
    FockSigma(const FockSigma & rhs) : _Fock(rhs._Fock), _Sigma(rhs._Sigma), _mu(rhs._mu) {
        inited_F = true;
        inited_S = true;
        inited_mu = true;
    }

    FockSigma(const Array_4D& Fock_, const Array_5D& Sigma_, const double mu_) : 
       _Fock(Fock_), _Sigma(Sigma_), _mu(mu_) {
        inited_F = true;
        inited_S = true;
        inited_mu = true;
    }

    FockSigma& operator =(const FockSigma& rhs) {
      _Fock = rhs._Fock;
      _Sigma = rhs._Sigma;
      _mu = rhs._mu;
        inited_F = true;
        inited_S = true;
        inited_mu = true;
      return *this;
    }

    ComplexType dot_prod(const FockSigma& rhs) const {
      utils::check(inited_F, "FockSigma: Fock matrix is not initialized");
      utils::check(inited_S, "FockSigma: Sigma is not initialized");
      size_t Fdim = std::reduce(_Fock.shape().begin(), _Fock.shape().end(), 1, std::multiplies<size_t>());
      size_t Sdim = std::reduce(_Sigma.shape().begin(), _Sigma.shape().end(), 1, std::multiplies<size_t>());
/*
      auto vec_F= nda::reshape(_Fock, std::array<long, 1>{Fdim});
      auto vec_S= nda::reshape(_Sigma, std::array<long, 1>{Sdim});
*/
      auto matvec_F= nda::reshape(_Fock, std::array<long, 2>{Fdim, 1});
      auto matvec_S= nda::reshape(_Sigma, std::array<long, 2>{Sdim, 1});

      auto rFock = rhs.get_fock();
      auto rSigma = rhs.get_sigma();
      size_t rFdim = std::reduce(rFock.shape().begin(), rFock.shape().end(), 1, std::multiplies<size_t>());
      size_t rSdim = std::reduce(rSigma.shape().begin(), rSigma.shape().end(), 1, std::multiplies<size_t>());
/*
      auto vec_rF= nda::reshape(rFock, std::array<long, 1>{rFdim});
      auto vec_rS= nda::reshape(rSigma, std::array<long, 1>{rSdim});
*/
      auto matvec_rF= nda::reshape(rFock, std::array<long, 2>{rFdim, 1});
      auto matvec_rS= nda::reshape(rSigma, std::array<long, 2>{rSdim, 1});
      //return nda::blas::dotc(vec_F,vec_rF) + nda::blas::dotc(vec_S,vec_rS);
      nda::array<ComplexType, 2> res1(1,1);
      nda::array<ComplexType, 2> res2(1,1);
      nda::blas::gemm(nda::make_regular(nda::conj(nda::transpose(matvec_F))), matvec_rF, res1);
      nda::blas::gemm(nda::make_regular(nda::conj(nda::transpose(matvec_S))), matvec_rS, res2);
      return res1(0,0) + res2(0,0);
    }

    const Array_4D& get_fock() const {
        utils::check(inited_F, "FockSigma: Fock matrix is not initialized");
        return _Fock;
    }
    const Array_5D& get_sigma() const {
        utils::check(inited_S, "FockSigma: Sigma is not initialized");
        return _Sigma;
    }
    double get_mu() const {
        utils::check(inited_mu, "FockSigma: mu is not initialized");
        return _mu;
    }

    void set_mu(double mu) { _mu = mu; inited_mu = true;}
    void set_fock(Array_4D& F_) {
        _Fock = F_;
        inited_F = true;
    }
    void set_sigma(Array_5D& S_) {
        _Sigma = S_;
        inited_S = true;
    }
    void set_fock_sigma(Array_4D& F_, Array_5D& S_) {
        set_fock(F_);
        set_sigma(S_);
    }

    void set_zero() {
        _Fock() = 0;
        _Sigma() = 0;
        _mu = 0;
        inited_F = true;
        inited_S = true;
        inited_mu = true;
    }

    FockSigma operator*=(std::complex<double> c)  {
        utils::check(inited_F, "FockSigma: Fock matrix is not initialized");
        utils::check(inited_S, "FockSigma: Sigma is not initialized");
        _Fock *= c;
        _Sigma *= c;
        return *this;
    }

    FockSigma operator+=(FockSigma & vec)  {
        utils::check(inited_F, "FockSigma: Fock matrix is not initialized");
        utils::check(inited_S, "FockSigma: Sigma is not initialized");
        _Fock += vec.get_fock();
        _Sigma += vec.get_sigma();
        return *this;
    }

    FockSigma operator+=(FockSigma && vec)  {
        utils::check(inited_F, "FockSigma: Fock matrix is not initialized");
        utils::check(inited_S, "FockSigma: Sigma is not initialized");
        _Fock += vec.get_fock();
        _Sigma += vec.get_sigma();
        return *this;
    }

    void add(FockSigma&& a, ComplexType c) {
        utils::check(inited_F, "FockSigma: Fock matrix is not initialized");
        utils::check(inited_S, "FockSigma: Sigma is not initialized");
        _Fock += c * a.get_fock();
        _Sigma += c * a.get_sigma();
    }

    void read_from_file(std::string filename, const size_t vec_number) {
        h5::file file(filename, 'r');
        auto vec_grp = h5::group(file).open_group("vec" + std::to_string(vec_number));
        
        nda::h5_read(vec_grp, "Sigma_tskij", _Sigma);
        nda::h5_read(vec_grp, "F_skij", _Fock);
        h5::h5_read(vec_grp, "mu", _mu);
        inited_F = true;
        inited_S = true;
        inited_mu = true;
    }
    void write_to_file(std::string filename, const size_t vec_number) const {
        utils::check(inited_F, "FockSigma: Fock matrix is not initialized");
        utils::check(inited_S, "FockSigma: Sigma is not initialized");
        h5::file file(filename, 'a');
        if(!h5::group(file).has_subgroup("vec" + std::to_string(vec_number))) {
            //app_log(2, "write_to_file: creating {} in file {}", "vec" + std::to_string(vec_number), filename);
            auto vec_grp = h5::group(file).create_group("vec" + std::to_string(vec_number));
            nda::h5_write(vec_grp, "Sigma_tskij", _Sigma, false);
            nda::h5_write(vec_grp, "F_skij", _Fock, false);
            h5::h5_write(vec_grp, "mu", _mu);
        } else {
            //app_log(2, "write_to_file: opening existing {} in file {}", "vec" + std::to_string(vec_number), filename);
            auto vec_grp = h5::group(file).open_group("vec" + std::to_string(vec_number));
            nda::h5_write(vec_grp, "Sigma_tskij", _Sigma, false);
            nda::h5_write(vec_grp, "F_skij", _Fock, false);
            h5::h5_write(vec_grp, "mu", _mu);
        }
    }            
    
};


/** 
 * Evaluation of the commutator in the tau space between G and G_0^{-1} - Sigma
 *
 * @param C_t     - [OUTPUT] Commutator in tau space
 * @param FT      - [INPUT] Imaginary frequency FT axes
 * @param G_t     - [INPUT] Green's function in tau space
 * @param FS_t    - [INPUT] Fock and Sigma in tau space
 * @param mu      - [INPUT] Chemical potential
 * @param S       - [INPUT] Overlap matrix
 * @param H0      - [INPUT] Non-interacting Hamiltonian
 **/
template<typename Array_G, typename Array_ov>
void commutator_t(Array_G& C_t, const imag_axes_ft::IAFT *FT,
                  const Array_G& G_t, const FockSigma& FS_t, double mu,
                  const Array_ov& S, const Array_ov& H0) {
    decltype(nda::range::all) all;
    size_t nt = G_t.shape()[0];
    size_t ns = G_t.shape()[1];
    size_t nk = G_t.shape()[2];
    size_t nao = G_t.shape()[3];
    size_t nw = FT->nw_f();
    nda::array<ComplexType, 5> G_w(nw,ns,nk,nao,nao);
    nda::array<ComplexType, 5> Sigma_w(nw,ns,nk,nao,nao);
    // G_w is filled
    FT->tau_to_w(G_t, G_w, imag_axes_ft::fermion);
    // Sigma_t is filled
    auto Sigma_t = FS_t.get_sigma();
    auto Fock = FS_t.get_fock();
    // Sigma_w is filled
    FT->tau_to_w(Sigma_t, Sigma_w, imag_axes_ft::fermion);

    nda::array<ComplexType, 4> Dm(ns,nk,nao,nao);
    FT->tau_to_beta(G_t, Dm);

    nda::array<ComplexType, 5> C_w(nw, ns, nk, nao, nao);
    C_w () = 0;
    C_t = nda::array<ComplexType, 5>(nt,ns,nk,nao,nao); // To make sure an array of appropriate size is ready
    C_t () = 0;

    nda::array<ComplexType, 2> I1(nao, nao);
    nda::array<ComplexType, 2> I2(nao, nao);

    for(size_t iw = 0; iw < nw; iw++)
    for(size_t s = 0; s < ns; s++)
    for(size_t k = 0; k < nk; k++) {
        long wn = FT->wn_mesh()(iw);
        ComplexType omega_mu = FT->omega(wn) + mu;
        auto S_sk = S(s,k,all,all);
        auto F_sk = Fock(s,k,all,all);
        auto H0_sk = H0(s,k,all,all);
        auto G_wsk = G_w(iw,s,k,all,all);
        auto Sigma_wsk = Sigma_w(iw,s,k,all,all);

        nda::array<ComplexType, 2> G0inv_Sigma_wsk = nda::make_regular(omega_mu * S_sk - H0_sk - F_sk - Sigma_wsk);
        nda::array_view<ComplexType, 2> C_wsk = C_w(iw,s,k,all,all);
        I1() = 0;
        I2() = 0;
        nda::blas::gemm(G_wsk, G0inv_Sigma_wsk, I1);
        nda::blas::gemm(G0inv_Sigma_wsk, G_wsk, I2);
        C_wsk = nda::make_regular(I1 - I2);
    }

    FT->w_to_tau(C_w, C_t, imag_axes_ft::fermion);
}


}
#endif 
