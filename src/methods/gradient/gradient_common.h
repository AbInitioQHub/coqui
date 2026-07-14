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


#ifndef METHODS_GRADIENT_GRADIENT_COMMON_H
#define METHODS_GRADIENT_GRADIENT_COMMON_H

#include <string>

#include "nda/nda.hpp"

#include "mean_field/MF.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"

namespace methods {

nda::array<ComplexType, 2> eval_grad_1e(std::shared_ptr<mf::MF> mf,
                                        const nda::MemoryArrayOfRank<4> auto &D_skij);

nda::array<ComplexType, 2> eval_grad_pulay(std::shared_ptr<mf::MF> mf,
                                           const imag_axes_ft::IAFT &FT,
                                           const nda::MemoryArrayOfRank<4> auto &S_skij,
                                           const nda::MemoryArrayOfRank<5> auto &G_tskij,
                                           double mu);

nda::array<ComplexType, 2> eval_grad_pulay(std::shared_ptr<mf::MF> mf,
                                           const nda::MemoryArrayOfRank<4> auto &D_skij,
                                           const nda::MemoryArrayOfRank<4> auto &F_skij,
                                           const nda::MemoryArrayOfRank<4> auto &S_skij,
                                           const nda::MemoryArrayOfRank<4> auto &H0_skij,
                                           bool F_has_H0);

nda::array<ComplexType, 4> eval_DE(std::shared_ptr<mf::MF> mf,
                                   const imag_axes_ft::IAFT &FT,
                                   const nda::MemoryArrayOfRank<4> auto &S_skij,
                                   const nda::MemoryArrayOfRank<5> auto &G_tskij,
                                   double mu);

nda::array<ComplexType, 4> eval_DE(std::shared_ptr<mf::MF> mf,
                                   const nda::MemoryArrayOfRank<4> auto &D_skij,
                                   const nda::MemoryArrayOfRank<4> auto &F_skij,
                                   const nda::MemoryArrayOfRank<4> auto &S_skij,
                                   const nda::MemoryArrayOfRank<4> auto &H0_skij,
                                   bool F_has_H0);


ComplexType eval_grad_1e(int iatom, int direction, std::shared_ptr<mf::MF> mf,
                         const nda::MemoryArrayOfRank<4> auto &D_skij);


ComplexType eval_grad_pulay(int iatom, int direction, std::shared_ptr<mf::MF> mf,
                            const nda::MemoryArrayOfRank<4> auto &D_skij,
                            const nda::MemoryArrayOfRank<4> auto &F_skij,
                            const nda::MemoryArrayOfRank<4> auto &S_skij,
                            const nda::MemoryArrayOfRank<4> auto &H0_skij,
                            bool F_has_H0);

ComplexType eval_grad_pulay(int iatom, int direction, std::shared_ptr<mf::MF> mf,
                            const nda::MemoryArrayOfRank<4> auto &DE_skij);

template<typename data_type>
void print_mbpt_gradient(const nda::array<data_type, 2> &gradient,
                          std::shared_ptr<mf::MF> mf,
                          const std::string& str, bool bohr = true);

template<typename communicator_t, typename data_type>
void write_mbpt_gradient(communicator_t &comm,
                          const nda::array<data_type, 2> &gradient,
                          const std::string &output, long iter);

} // namespace methods

#endif
