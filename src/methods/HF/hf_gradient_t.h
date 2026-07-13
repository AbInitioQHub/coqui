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


#ifndef COQUI_HF_GRADIENT_T_H
#define COQUI_HF_GRADIENT_T_H

#include "mpi3/communicator.hpp"
#include "mean_field/MF.hpp"
#include "methods/ERI/detail/concepts.hpp"

namespace methods {

  namespace solvers {

    namespace mpi3 = boost::mpi3;

    class hf_gradient_t {

    public:

    hf_gradient_t(std::shared_ptr<mf::MF> MF);

    ~hf_gradient_t() = default;

    nda::array<ComplexType, 2> evaluate(const nda::MemoryArrayOfRank<4> auto &D_skij,
                                        Cholesky_ERI auto && chol);

    nda::array<ComplexType, 2> eval_grad_2e(const nda::MemoryArrayOfRank<4> auto &D_skij,
                                            Cholesky_ERI auto && chol);

    nda::array<ComplexType, 2> eval_grad_coulomb(const nda::MemoryArrayOfRank<4> auto &D_skij,
                                                 Cholesky_ERI auto && chol);

    nda::array<ComplexType, 2> eval_grad_exchange(const nda::MemoryArrayOfRank<4> auto &D_skij,
                                                  Cholesky_ERI auto && chol);

    nda::array<ComplexType, 3> eval_dm_total(const nda::MemoryArrayOfRank<4> auto &D_skij);

    nda::array<ComplexType, 6> eval_2bdm(const nda::MemoryArrayOfRank<4> auto &D_skij);

    ComplexType eval_grad_2e(int iatom, int direction, const nda::MemoryArrayOfRank<4> auto &D_skij,
                             Cholesky_ERI auto && chol);

    ComplexType eval_grad_coulomb(int iatom, int direction, const nda::MemoryArrayOfRank<3> auto &D_total_kij,
                                  Cholesky_ERI auto && chol);

    ComplexType eval_grad_exchange(int iatom, int direction, const nda::MemoryArrayOfRank<4> auto &D_skij,
                                    Cholesky_ERI auto && chol);

    void print_chol_hf_grad_timers();

    private:

    std::shared_ptr<mf::MF> _mf = nullptr;

    utils::TimerManager _Timer;

    int _natoms = 0;
    int _nbnd = 0;
    int _nbnd_aux = 0;
    int _nspin = 0;
    int _nkpts = 0;
    int _npol = 0;

    nda::array<RealType, 1> _k_weight;

  };

 } // namespace solvers

} // namespace method

#endif
