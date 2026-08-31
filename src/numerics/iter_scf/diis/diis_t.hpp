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


#ifndef COQUI_DIIS_T_HPP
#define COQUI_DIIS_T_HPP

#include <algorithm>
#include <cctype>

#include "configuration.hpp"
#include "utilities/check.hpp"

#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "numerics/iter_scf/iter_scf_type_e.hpp"

#include "numerics/iter_scf/diis/vspace.h"
#include "numerics/iter_scf/diis/vspace_fock_sigma.hpp"
#include "numerics/iter_scf/diis/vspace_heff.hpp"

#include "numerics/iter_scf/diis/state.h"
#include "numerics/iter_scf/diis/com_diis_residual.h"
#include "numerics/iter_scf/diis/qp_com_diis_residual.h"

#include "numerics/iter_scf/diis/diis_alg.hpp"
#include "numerics/iter_scf/damp/damp_t.hpp"

// TODO clean up duplicate codes between Heff and FockSigma extrapolation

namespace iter_scf {
  /**
   * Simple class connecting an abstract DIIS with specific interface requirements
   * serving as a driver for DIIS algorithm
   */
  struct diis_t {
    using Array_4D = nda::array<ComplexType,4>;
    using Array_5D = nda::array<ComplexType,5>;

    static constexpr iter_alg_e iter_alg = DIIS;
    static constexpr iter_alg_e get_iter_alg() { return iter_alg; }

  public:
    diis_t() = default;
    diis_t(double mixing_, size_t max_subsp_size_, size_t warmup_iter_,
         std::string residual_type_ = "commutator"):
        mixing(mixing_),
        max_subsp_size(max_subsp_size_),
        warmup_iter(warmup_iter_),
        residual_type(normalize_residual_type(std::move(residual_type_))) {};

    diis_t(const diis_t& other) = default;
    diis_t(diis_t&& other) = default;
    diis_t& operator=(const diis_t& other) = default;
    diis_t& operator=(diis_t&& other) = default;

    ~diis_t(){}

    /**
     * @brief Initialize the DIIS driver for Dyson-SCF (frequency-dependent commutator DIIS).
     *
     * Sets up vector spaces and residual kernels for simultaneous extrapolation of the
     * Fock matrix F (rank-4, shape [s,k,i,j]) and the self-energy Sigma (rank-5, shape
     * [t,s,k,i,j]). Must be called once before the first call to solve(F, Sigma, ...).
     *
     * @param F           Initial Fock matrix [s,k,i,j].
     * @param Sigma       Initial self-energy [t,s,k,i,j].
     * @param mu          Chemical potential.
     * @param S           Overlap matrix [s,k,i,j].
     * @param H0          Non-interacting Hamiltonian [s,k,i,j].
     * @param FT          Pointer to the imaginary-axis Fourier transform object.
     * @param mbpt_output_ Base path for the MBPT checkpoint HDF5 file.
     */
    template<nda::MemoryArrayOfRank<4> F_t, nda::MemoryArrayOfRank<5> Sigma_t,
        nda::MemoryArrayOfRank<4> S_t, nda::MemoryArrayOfRank<4> H0_t>
    void initialize(F_t &&F, Sigma_t &&Sigma, double mu, S_t &&S, H0_t &&H0,
                    const imag_axes_ft::IAFT *FT, std::string mbpt_output_) {
        warmup_count = 0; // reset warmup counter
        mbpt_output = mbpt_output_;
        initialized_dyson = true;
        initialized_qp = false;
        // Initialize the extrapolated state using the current Fock and Sigma
        extrapolated_state.initialize(FockSigma(F, Sigma, mu));
        // initialize the vector space used to extrapolation
        x_vsp.initialize("diis_vectors.h5");
        // initialize the vector space for residuals
        res_vsp.initialize("diis_residuals.h5");
        comFS_residual.initialize(&extrapolated_state, S, H0, FT, mbpt_output);
        // providing non-owning pointers to DIIS kernel as well as the starting state
        d_alg.init(&extrapolated_state, &comFS_residual, &x_vsp, &res_vsp,
                   max_subsp_size, true, FockSigma(F, Sigma, mu));
        initialized = true;
    }

    /**
     * @brief Initialize the DIIS driver for QP-SCF (effective Hamiltonian commutator DIIS).
     *
     * Sets up vector spaces and residual kernels for extrapolation of the effective
     * Hamiltonian Heff (rank-4, shape [s,k,i,j]). Must be called once before the first
     * call to solve(H, ...).
     *
     * @param H            Initial effective Hamiltonian [s,k,i,j].
     * @param S            Overlap matrix [s,k,i,j].
     * @param mbpt_output_ Base path for the MBPT checkpoint HDF5 file.
     */
    template<nda::MemoryArrayOfRank<4> H_t, nda::MemoryArrayOfRank<4> S_t>
    void initialize(H_t &&H, S_t &&S, std::string mbpt_output_) {
      warmup_count = 0; // reset warmup counter
      mbpt_output = mbpt_output_;
      initialized_dyson = false;
      initialized_qp = true;

      extrapolated_heff_state.initialize(Heff(H));
      heff_x_vsp.initialize("diis_heff_vectors.h5");
      heff_res_vsp.initialize("diis_heff_residuals.h5");
      qp_com_residual.initialize(&extrapolated_heff_state, S, mbpt_output, residual_type);
      h_alg.init(&extrapolated_heff_state, &qp_com_residual, &heff_x_vsp, &heff_res_vsp,
             max_subsp_size, true, Heff(H));
      initialized = true;
    }

    /**
     * @brief Perform one QP-SCF DIIS step on the effective Hamiltonian.
     *
     * During warmup iterations (or when the DIIS subspace is too small), simple linear
     * damping is applied instead of DIIS extrapolation. After warmup, the commutator
     * residual is computed and the DIIS extrapolation is attempted. H is updated in-place
     * with the extrapolated value if extrapolation succeeds.
     *
     * @param H       Effective Hamiltonian [s,k,i,j]; updated in-place on extrapolation.
     * @param dataset HDF5 dataset name for checkpoint I/O (passed to damping fallback).
     * @param grp     HDF5 group for checkpoint I/O (passed to damping fallback).
     * @param iter    Current SCF iteration index (1-based).
     * @return        Max absolute change in H (convergence measure).
     */
    template<nda::MemoryArrayOfRank<4> Array_H_t>
    double solve(Array_H_t &&H, std::string dataset, h5::group &grp, long iter) {
      utils::check(initialized and initialized_qp,
            "DIIS(QP): initialize(Heff, S, output) must be called before solving.");
      warmup_count += 1;
      if (heff_x_vsp.size() == 1 || warmup_count <= warmup_iter) {
        app_log(2, "DIIS(QP): Warmup iteration {}/{}. Simple damping will be executed instead.\n",
            warmup_count, warmup_iter);
        damp_t damp(mixing);

        h_alg.grow_xvsp_only = (heff_x_vsp.size() <= 1);
        h_alg.extrap = false;
        qp_com_residual.set_iteration(iter-1);
        qp_com_residual.set_previous_heff_vec_idx(static_cast<long>(heff_x_vsp.size()) - 1);
        utils::check(h_alg.next_step(Heff(nda::make_regular(H))) == 0,
               "DIIS(QP): Unexpected extrapolation while DIIS algorithm is only growing the subspace");
        app_log(4, "DIIS(QP): DIIS vector space size: {}", heff_x_vsp.size());
        app_log(4, "DIIS(QP): DIIS residual space size: {}\n", heff_res_vsp.size());

        return damp.solve(H, dataset, grp, iter);

      } else {
        h_alg.extrap = true;
        h_alg.grow_xvsp_only = false;
        qp_com_residual.set_iteration(iter-1);
        qp_com_residual.set_previous_heff_vec_idx(static_cast<long>(heff_x_vsp.size()) - 1);
        int is_extrapolated = h_alg.next_step(Heff(nda::make_regular(H)));
        if (is_extrapolated != 0) {
          auto Hdiff = nda::make_regular(H - h_alg.get_extrapolated_state().get_heff());
          auto Hmax_iter = max_element(Hdiff.data(), Hdiff.data()+Hdiff.size(),
                    [](auto a, auto b) { return std::abs(a) < std::abs(b); });
          H = h_alg.get_extrapolated_state().get_heff();
          return std::abs(*Hmax_iter);
        } else {
          app_log(2, "DIIS(QP): Performing simple damping instead.\n");
          damp_t damp(mixing);
          return damp.solve(H, dataset, grp, iter);
        }
      }
    }
   

    /**
     * @brief Perform one Dyson-SCF DIIS step on the Fock matrix and self-energy.
     *
     * During warmup iterations (or when the DIIS subspace is too small), simple linear
     * damping is applied instead of DIIS extrapolation. After warmup, the frequency-dependent
     * commutator residual is computed (Pokhilko/Yeh/Zgid, JCP 2022) and DIIS extrapolation
     * is attempted. F and Sigma are updated in-place with the extrapolated values if
     * extrapolation succeeds.
     *
     * @param F              Fock matrix [s,k,i,j]; updated in-place on extrapolation.
     * @param dataset_F      HDF5 dataset name for F checkpoint I/O.
     * @param Sigma          Self-energy [t,s,k,i,j]; updated in-place on extrapolation.
     * @param dataset_Sigma  HDF5 dataset name for Sigma checkpoint I/O.
     * @param scf_grp        HDF5 group for checkpoint I/O.
     * @param iter           Current SCF iteration index (1-based).
     * @return               {max|ΔF|, max|ΔΣ|} — convergence measures for F and Sigma.
     */
    template<nda::MemoryArray Array_4D_t, nda::MemoryArray Array_5D_t>
    std::array<double, 2> solve(
        Array_4D_t &&F, std::string dataset_F, Array_5D_t &&Sigma, std::string dataset_Sigma,
        h5::group &scf_grp, long iter) {
        utils::check(initialized, "DIIS must be initialized before solving");
        warmup_count += 1;
        if (x_vsp.size() == 1 || warmup_count <= warmup_iter) {
            app_log(2, "DIIS: Warmup iteration {}/{}. Simple damping will be executed instead.\n",
                    warmup_count, warmup_iter);
            damp_t damp(mixing);

            // grow x_vsp only if x_vsp.size() <= 1, otherwise grow both x_vsp and res_vsp
            d_alg.grow_xvsp_only = (x_vsp.size() <= 1);
            d_alg.extrap = false;
            utils::check(d_alg.next_step(FockSigma(F, Sigma, get_mu()))==0,
                         "DIIS: Unexpected extrapolation while DIIS algorithm is only growing the subspace");
            app_log(4, "DIIS: DIIS vector space size: {}", x_vsp.size());
            app_log(4, "DIIS: DIIS residual space size: {}\n", res_vsp.size());

            // damping instead
            return damp.solve(F, dataset_F, Sigma, dataset_Sigma, scf_grp, iter);

         } else {
            // DO DIIS
            d_alg.extrap = true;
            d_alg.grow_xvsp_only = false;
            FockSigma fs(F, Sigma, get_mu());
            int is_extrapolated = d_alg.next_step(FockSigma(F, Sigma, get_mu()));
            if(is_extrapolated != 0) {
                auto Fdiff = nda::make_regular(F - d_alg.get_extrapolated_state().get_fock());
                auto Sdiff = nda::make_regular(Sigma - d_alg.get_extrapolated_state().get_sigma());
                auto Fmax_iter = max_element(Fdiff.data(), Fdiff.data()+Fdiff.size(),
                                    [](auto a, auto b) { return std::abs(a) < std::abs(b); });
                auto Smax_iter = max_element(Sdiff.data(), Sdiff.data()+Sdiff.size(),
                                  [](auto a, auto b) { return std::abs(a) < std::abs(b); });
                F     = d_alg.get_extrapolated_state().get_fock();
                Sigma = d_alg.get_extrapolated_state().get_sigma();

                return std::array<double, 2>{std::abs(*Fmax_iter), std::abs(*Smax_iter)};

            } else {
                // No DIIS extrapolation has been applied
                app_log(2, "DIIS: Performing simple damping instead.\n");
                damp_t damp(mixing);

                return damp.solve(F, dataset_F, Sigma, dataset_Sigma, scf_grp, iter);
            }
        }
    }

    /**
     * @brief Compile-time guard — single-array solve for non-rank-4 inputs.
     *
     * This overload exists solely to satisfy the `std::visit` instantiation in `iter_scf_t`,
     * which requires all variant types (`damp_t`, `diis_t`) to provide a callable
     * `solve(H, dataset, grp, iter)` for any array rank. It must never be called at runtime;
     * doing so indicates a programming error.
     *
     * @throws std::logic_error unconditionally.
     */
    template<nda::MemoryArray Array_H_t>
      requires (!nda::MemoryArrayOfRank<Array_H_t, 4>)
    double solve([[maybe_unused]] Array_H_t &&H, [[maybe_unused]] std::string dataset, 
                 [[maybe_unused]] h5::group &grp, [[maybe_unused]] long iter) {
      throw std::logic_error("diis_t::solve: single-array overload called with non-rank-4 input; "
                             "this is a compile-time guard and should never be called at runtime.");
    }

    void metadata_log() const {
      app_log(2, "\nIterative algorithm for SCF");
      app_log(2, "-----------------------------");
      if (initialized_dyson) {
        app_log(2, "  * algorithm: frequency-dependent commutator DIIS\n"
                   "               P. Pokhilko, C.-N. Yeh, D. Zgid. J. Chem. Phys., 2022, 156, 094101\n"
                   "               https://doi.org/10.1063/5.0082586");
      } else {
        app_log(2, "  * algorithm: DIIS");
      } 
      app_log(2, "  * DIIS parameters: ");
      app_log(2, "    mixing            = {}", mixing);
      app_log(2, "    max subspace size = {}", max_subsp_size);
      app_log(2, "    warmup iteration  = {}", warmup_iter);
      app_log(2, "    residual type     = {}", residual_type);
      app_log(2, "    checkpoint output = {}\n", mbpt_output);
    }

  public:
    double mixing = 0.2;
    size_t max_subsp_size = 5;
    size_t warmup_iter = 5;
    size_t warmup_count = 0;
    bool initialized = false;
    bool initialized_dyson = false;
    bool initialized_qp = false;
    std::string residual_type = "commutator";
    
  private:
    VSpace<FockSigma> x_vsp;                 // vector space of Fock-self-energy vectors
    VSpace<FockSigma> res_vsp;               // vector space of residuals-commutators
    opt_state<FockSigma> extrapolated_state; // extrapolated DIIS state
    com_diis_residual comFS_residual;        // residual kernel

    diis_alg<FockSigma> d_alg;               // DIIS kernel

    VSpace<Heff> heff_x_vsp;                 // vector space of Heff vectors
    VSpace<Heff> heff_res_vsp;               // vector space of Heff commutator residuals
    opt_state<Heff> extrapolated_heff_state; // extrapolated Heff state
    qp_com_diis_residual qp_com_residual;    // Heff-Dm commutator residual kernel
    diis_alg<Heff> h_alg;                    // DIIS kernel for Heff

    std::string mbpt_output;

    static std::string normalize_residual_type(std::string residual_type) {
      std::transform(residual_type.begin(), residual_type.end(), residual_type.begin(),
                     [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      utils::check(residual_type == "commutator" or residual_type == "vector_diff",
                   "diis_t::normalize_residual_type: unknown residual_type = {}. Valid options are \"commutator\" and \"vector_diff\"",
                   residual_type);
      return residual_type;
    }

    /**
     * @brief Read the chemical potential from the latest SCF iteration in the checkpoint file.
     *
     * @return Chemical potential μ.
     */
    double get_mu() {
        long iter_from_file;
        std::string filename = mbpt_output + ".mbpt.h5";
        h5::file file(filename, 'r');
        h5::group grp(file);
        utils::check(grp.has_subgroup("scf"), "Simulation HDF5 file does not have an scf group");
        auto scf_grp = grp.open_group("scf");
        h5::h5_read(scf_grp, "final_iter", iter_from_file);
        auto iter_grp = scf_grp.open_group("iter"+std::to_string(iter_from_file));
        double mu;
        h5::h5_read(iter_grp, "mu", mu);
        return mu;
    }


  };
} // iter_scf

#endif //COQUI_DIIS_T_HPP
