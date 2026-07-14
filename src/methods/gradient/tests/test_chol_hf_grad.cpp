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


#undef NDEBUG

#include "nda/h5.hpp"

#include "mean_field/default_MF.hpp"
#include "mean_field/MF.hpp"
#include "methods/ERI/eri_utils.hpp"
#include "methods/gradient/gradient_common.h"
#include "methods/HF/hf_gradient_t.h"
#include "methods/HF/hf_t.h"
#include "methods/SCF/mb_solver_t.h"
#include "methods/SCF/scf_driver.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/tests/test_common.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "utilities/mpi_context.h"
#include "utilities/test_common.hpp"

namespace bdft_tests {

  TEST_CASE("chol_hf_grad", "[methods][chol][hf_gradient][pyscf]") {

    using namespace methods;

    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto [outdir, prefix] = utils::utest_filename("pyscf_h2o2_mol");

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_h2o2_mol"));
    imag_axes_ft::IAFT ft(10000, 25, imag_axes_ft::ir_basis);

    solvers::hf_t hf;

    chol_reader_t chol_reader(mf, outdir + "gdf_eri");
    auto eri = mb_eri_t(chol_reader, chol_reader);

    simple_dyson dyson(mf.get(), &ft);
    iter_scf::iter_scf_t iter_sol("damping");
    MBState mb_state(mpi_context, ft, "h2o2");
    auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                   solvers::mb_solver_t(&hf), &iter_sol,
                                   50, false, 1e-8, false);

    VALUE_EQUAL(e_hf, -195.945095774442, 1e-6);

    auto grad_1e = nda::array<ComplexType, 2>::zeros({mf->number_of_atoms(), 3});
    auto grad_2e = nda::array<ComplexType, 2>::zeros({mf->number_of_atoms(), 3});
    auto grad_pulay = nda::array<ComplexType, 2>::zeros({mf->number_of_atoms(), 3});
    auto grad_elec = nda::array<ComplexType, 2>::zeros({mf->number_of_atoms(), 3});
    auto grad_total = nda::array<ComplexType, 2>::zeros({mf->number_of_atoms(), 3});
    solvers::hf_gradient_t hf_grad(mf);
    grad_1e = eval_grad_1e(mf, mb_state.sDm_skij.value().local());
    grad_2e = hf_grad.eval_grad_2e(mb_state.sDm_skij.value().local(), eri.hf_eri->get());
    grad_pulay = eval_grad_pulay(mf, mb_state.sDm_skij.value().local(), mb_state.sF_skij.value().local(),
                                 dyson.sS_skij().local(), dyson.sH0_skij().local(), false);
    grad_elec = grad_1e + grad_2e + grad_pulay;
    grad_total = grad_elec + mf->nuclear_gradient();
    print_mbpt_gradient(mf->nuclear_gradient(), mf, "GRAD_NUC");
    print_mbpt_gradient(grad_elec, mf, "GRAD_ELEC");
    print_mbpt_gradient(grad_total, mf, "GRAD_TOTAL");

    auto grad_total_ref = nda::array<ComplexType, 2>::zeros({mf->number_of_atoms(), 3});
    h5::file file(outdir + "/" + prefix + ".h5", 'r');
    h5::group grp(file);
    h5::group scf_grp = grp.open_group("SCF");
    nda::h5_read(scf_grp, "grad_total", grad_total_ref);
    ARRAY_EQUAL(grad_total, grad_total_ref, 1e-5);

  }

} // coqui_tests
