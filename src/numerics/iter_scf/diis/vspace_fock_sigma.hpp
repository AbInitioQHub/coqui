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
    FockSigma(FockSigma&&) noexcept = default;

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
    FockSigma& operator=(FockSigma&&) noexcept = default;

    ComplexType dot_prod(const FockSigma& rhs) const {
      utils::check(inited_F, "FockSigma: Fock matrix is not initialized");
      utils::check(inited_S, "FockSigma: Sigma is not initialized");
      size_t Fdim = std::reduce(_Fock.shape().begin(), _Fock.shape().end(), 1, std::multiplies<size_t>());
      size_t Sdim = std::reduce(_Sigma.shape().begin(), _Sigma.shape().end(), 1, std::multiplies<size_t>());
/*
      auto vec_F= nda::reshape(_Fock, std::array<long, 1>{Fdim});
      auto vec_S= nda::reshape(_Sigma, std::array<long, 1>{Sdim});
*/
      const auto& rFock = rhs.get_fock();
      const auto& rSigma = rhs.get_sigma();
      size_t rFdim = std::reduce(rFock.shape().begin(), rFock.shape().end(), 1, std::multiplies<size_t>());
      size_t rSdim = std::reduce(rSigma.shape().begin(), rSigma.shape().end(), 1, std::multiplies<size_t>());
      auto vec_rF = nda::reshape(rFock, std::array<long, 1>{static_cast<long>(rFdim)});
      auto vec_rS = nda::reshape(rSigma, std::array<long, 1>{static_cast<long>(rSdim)});
      auto vec_F = nda::reshape(_Fock, std::array<long, 1>{static_cast<long>(Fdim)});
      auto vec_S = nda::reshape(_Sigma, std::array<long, 1>{static_cast<long>(Sdim)});
      return nda::blas::dotc(vec_F, vec_rF) + nda::blas::dotc(vec_S, vec_rS);
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
    void set_fock(Array_4D&& F_) {
        _Fock = std::move(F_);
        inited_F = true;
    }
    void set_sigma(Array_5D& S_) {
        _Sigma = S_;
        inited_S = true;
    }
    void set_sigma(Array_5D&& S_) {
        _Sigma = std::move(S_);
        inited_S = true;
    }
    void set_fock_sigma(Array_4D& F_, Array_5D& S_) {
        set_fock(F_);
        set_sigma(S_);
    }
    void set_fock_sigma(Array_4D&& F_, Array_5D&& S_) {
        set_fock(std::move(F_));
        set_sigma(std::move(S_));
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
        const auto& aFock = a.get_fock();
        const auto& aSigma = a.get_sigma();
        utils::check(_Fock.shape() == aFock.shape(), "FockSigma::add: incompatible Fock shapes");
        utils::check(_Sigma.shape() == aSigma.shape(), "FockSigma::add: incompatible Sigma shapes");
        for (size_t i = 0; i < _Fock.size(); ++i) _Fock.data()[i] += c * aFock.data()[i];
        for (size_t i = 0; i < _Sigma.size(); ++i) _Sigma.data()[i] += c * aSigma.data()[i];
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
    const auto& Sigma_t = FS_t.get_sigma();
    const auto& Fock = FS_t.get_fock();
    C_t = nda::array<ComplexType, 5>(nt,ns,nk,nao,nao); // To make sure an array of appropriate size is ready
    C_t () = 0;

    // Stream the frequency axis. The former implementation materialized full
    // G(w), Sigma(w), and C(w) arrays simultaneously on the global root.
    nda::array<ComplexType, 4> G_wskij(ns,nk,nao,nao);
    nda::array<ComplexType, 4> Sigma_wskij(ns,nk,nao,nao);
    nda::array<ComplexType, 4> C_wskij(ns,nk,nao,nao);

    nda::array<ComplexType, 2> I1(nao, nao);
    nda::array<ComplexType, 2> I2(nao, nao);

    app_log(2, "DIIS: Streaming commutator residual over {} frequencies", nw);
    for(size_t iw = 0; iw < nw; iw++) {
        FT->tau_to_w(G_t, G_wskij, imag_axes_ft::fermion, iw);
        FT->tau_to_w(Sigma_t, Sigma_wskij, imag_axes_ft::fermion, iw);
        for(size_t s = 0; s < ns; s++)
        for(size_t k = 0; k < nk; k++) {
            long wn = FT->wn_mesh()(iw);
            ComplexType omega_mu = FT->omega(wn) + mu;
            auto S_sk = S(s,k,all,all);
            auto F_sk = Fock(s,k,all,all);
            auto H0_sk = H0(s,k,all,all);
            auto G_wsk = G_wskij(s,k,all,all);
            auto Sigma_wsk = Sigma_wskij(s,k,all,all);

            nda::array<ComplexType, 2> G0inv_Sigma_wsk = nda::make_regular(omega_mu * S_sk - H0_sk - F_sk - Sigma_wsk);
            auto C_wsk = C_wskij(s,k,all,all);
            nda::blas::gemm(G_wsk, G0inv_Sigma_wsk, I1);
            nda::blas::gemm(G0inv_Sigma_wsk, G_wsk, I2);
            C_wsk = I1 - I2;
        }
        FT->w_to_tau_partial(C_wskij, C_t, imag_axes_ft::fermion, iw);
        if ((iw + 1) == nw || (iw + 1) % std::max<size_t>(size_t{1}, nw / 10) == 0)
            app_log(2, "DIIS: Commutator frequencies {}/{}", iw + 1, nw);
    }
}


}
#endif 
