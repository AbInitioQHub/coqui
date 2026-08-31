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


#ifndef UTILITIES_FORTRAN_WRAPPER_H
#define UTILITIES_FORTRAN_WRAPPER_H

#include <complex>
#include "utilities/FC.h"

#define FC_test_util FC_GLOBAL(test_util,TEST_UTIL)

// pw2bgw
#define FC_read_pw2bgw_vkbg_header FC_GLOBAL(read_pw2bgw_vkbg_header,READ_PW2BGW_VKBG_HEADER)
#define FC_read_pw2bgw_vkbg FC_GLOBAL(read_pw2bgw_vkbg,READ_PW2BGW_VKBG)

#if defined(ENABLE_WANNIER90)
// wannier90
#define FC_wann90_setup FC_GLOBAL(wann90_setup,WANN90_SETUP)
#define FC_wann90_run FC_GLOBAL(wann90_run,WANN90_RUN)
#define FC_wann90_run_from_files FC_GLOBAL(wann90_run_from_files,WANN90_RUN_FROM_FILES)
#endif

extern "C"
{

void FC_test_util();

// Read pw2bgw files
void FC_read_pw2bgw_vkbg_header(char const* fname, int const& clen, int& ns, int& nkb, int& npwx, int& nkpts, int& nat, int& nsp, int& nhm, int& err);
void FC_read_pw2bgw_vkbg(char const* fname, int const& clen, int const& k0, int const& k1, int* ityp, int const& ityp_size, int* nh, int const& nh_size, int* ngk, int const& ngk_size, double* Dnn, int const& Dnn_size, int* miller, int const& miller_size, std::complex<double>* vkb, int const& vkb_size, int& err);

#if defined(ENABLE_WANNIER90)
// Call wannier90 library-mode
void FC_wann90_setup(char const* fname, int const& clen, 
                    int const& nb, int const& nw, 
                    int const& nat, double const* at_cart, int const&, char const*, int const*, 
                    double const* eival, double const* lattice_vectors,
                    int const& nk, int const* nkabc, double const* kpt, 
                    int const& nn, int & nnkp_size, int * nnkp, 
                    int const& auto_proj, int const& nproj, int const& max_len, char const* proj_str, 
                    int const* str_len, int * proj_ints, double * proj_doubles, 
                    bool const& write_nnkp, char const* ex_str, int const& ex_clen, int const& ierr); 

void FC_wann90_run(char const* fname, int const& clen,
                    int const& nb, int const& nw,
                    int const& nat, double const* at_cart, int const&, char const*, int const*,
                    double const* eival, double const* lattice_vectors,
                    int const& nk, int const* nkabc, double const* kpt, int const& nn,
                    int const& auto_proj, int const& nproj, int const& max_len,
                    char const* proj_str, int const* str_len,
                    std::complex<double> * M,
                    std::complex<double> * Uopt, double * centers, double * spreads,
                    int const& ierr);

// Calls disentangle and wannierize, assuming existing *mmn, *amn and *win files.
void FC_wann90_run_from_files(char const* fname, int const& clen,
                    int const& nb, int const& nw, double const* eival,
                    double const* lattice_vectors,
                    int const& nk, int const* nkabc, double const* kpt,
                    int const& nn,  
                    std::complex<double> * Pkmn, double * centers, double * spreads,
                    int const& ierr);
#endif

}

#endif
