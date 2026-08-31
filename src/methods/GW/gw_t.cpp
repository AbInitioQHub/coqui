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



#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "IO/app_loggers.h"
#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"

#include "mean_field/MF.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/GW/gw_t.h"

namespace methods {
  namespace solvers {

    gw_t::gw_t(const imag_axes_ft::IAFT *ft, std::string div, std::string output):
       _ft(ft),
       _div_treatment(div),
       _output(output),
       _Timer() {}

    void gw_t::print_thc_gw_timers() {
      app_log(2, "\n  THC-GW timers");
      app_log(2, "  -------------");
      app_log(2, "    Total:                   {0:.3f} sec", _Timer.elapsed("TOTAL"));
      if (_Timer.elapsed("EVALUATE_PI_K") > 0.0) {
        app_log(2, "    Evaluate Pi (k):         {0:.3f} sec", _Timer.elapsed("EVALUATE_PI_K"));
        app_log(2, "      - Alloc:               {0:.3f} sec", _Timer.elapsed("PI_ALLOC_K"));
        app_log(2, "      - Gij -> Guv:          {0:.3f} sec", _Timer.elapsed("PI_PRIM_TO_AUX"));
        app_log(2, "      - Hadamard product:    {0:.3f} sec", _Timer.elapsed("PI_HADPROD_K"));
      } else if (_Timer.elapsed("EVALUATE_PI_R") > 0.0) {
        app_log(2, "    Evaluate Pi (R):         {0:.3f} sec", _Timer.elapsed("EVALUATE_PI_R"));
        app_log(2, "      - Alloc:               {0:.3f} sec", _Timer.elapsed("PI_ALLOC_R"));
        app_log(2, "      - Gij -> Guv:          {0:.3f} sec", _Timer.elapsed("PI_PRIM_TO_AUX"));
        app_log(2, "      - FT:                  {0:.3f} sec", _Timer.elapsed("PI_FT_R"));
        app_log(2, "      - Hadamard product:    {0:.3f} sec", _Timer.elapsed("PI_HADPROD_R"));
      }
      app_log(2, "    Evaluate W:              {0:.3f} sec", _Timer.elapsed("EVALUATE_W"));
      if (_Timer.elapsed("EVALUATE_SIGMA_K") > 0.0) {
        app_log(2, "    Evaluate Sigma (K):      {0:.3f} sec", _Timer.elapsed("EVALUATE_SIGMA_K"));
        app_log(2, "      - Alloc:               {0:.3f} sec", _Timer.elapsed("SIGMA_ALLOC_K"));
        app_log(2, "      - Gij -> Guv:          {0:.3f} sec", _Timer.elapsed("SIGMA_PRIM_TO_AUX"));
        app_log(2, "      - Hadamard product:    {0:.3f} sec", _Timer.elapsed("SIGMA_HADPROD_K"));
        app_log(2, "      - Sigma_uv -> Sigma_ij {0:.3f} sec", _Timer.elapsed("SIGMA_AUX_TO_PRIM"));
        app_log(2, "      - Multiply dmat:       {0:.3f} sec", _Timer.elapsed("SIGMA_MULTIPLY_DMAT_K"));
      } else if (_Timer.elapsed("EVALUATE_SIGMA_R") > 0.0) {
        app_log(2, "    Evaluate Sigma (R):      {0:.3f} sec", _Timer.elapsed("EVALUATE_SIGMA_R"));
        app_log(2, "      - Alloc:               {0:.3f} sec", _Timer.elapsed("SIGMA_ALLOC_R"));
        app_log(2, "      - Gij -> Guv:          {0:.3f} sec", _Timer.elapsed("SIGMA_PRIM_TO_AUX"));
        app_log(2, "      - FT:                  {0:.3f} sec", _Timer.elapsed("SIGMA_FT_R"));
        app_log(2, "      - Hadamard product:    {0:.3f} sec", _Timer.elapsed("SIGMA_HADPROD_R"));
        app_log(2, "      - Sigma_uv -> Sigma_ij {0:.3f} sec", _Timer.elapsed("SIGMA_AUX_TO_PRIM"));
      }
      app_log(2, "    Imaginary FT tau->w:     {0:.3f} sec", _Timer.elapsed("IMAG_FT_TtoW"));
      app_log(2, "    Imaginary FT w->tau:     {0:.3f} sec", _Timer.elapsed("IMAG_FT_WtoT"));
      app_log(2, "      - FT_REDISTRIBUTE:     {0:.3f} sec\n", _Timer.elapsed("FT_REDISTRIBUTE"));
    }

    void gw_t::print_thc_rpa_timers() {
      app_log(2, "\n  THC-RPA timers");
      app_log(2, "  --------------");
      app_log(2, "    Total:                 {0:.3f} sec", _Timer.elapsed("TOTAL"));
      if (_Timer.elapsed("EVALUATE_PI_K") > 0.0) {
        app_log(2, "    Evaluate Pi (k):       {0:.3f} sec", _Timer.elapsed("EVALUATE_PI_K"));
        app_log(2, "      - Alloc:             {0:.3f} sec", _Timer.elapsed("PI_ALLOC_K"));
        app_log(2, "      - Gij -> Guv:        {0:.3f} sec", _Timer.elapsed("PI_PRIM_TO_AUX"));
        app_log(2, "      - Hadamard product:  {0:.3f} sec", _Timer.elapsed("PI_HADPROD_K"));
      } else if (_Timer.elapsed("EVALUATE_PI_R") > 0.0) {
        app_log(2, "    Evaluate Pi (R):       {0:.3f} sec", _Timer.elapsed("EVALUATE_PI_R"));
        app_log(2, "      - Alloc:             {0:.3f} sec", _Timer.elapsed("PI_ALLOC_R"));
        app_log(2, "      - Gij -> Guv:        {0:.3f} sec", _Timer.elapsed("PI_PRIM_TO_AUX"));
        app_log(2, "      - FT:                {0:.3f} sec", _Timer.elapsed("PI_FT_R"));
        app_log(2, "      - Hadamard product:  {0:.3f} sec", _Timer.elapsed("PI_HADPROD_R"));
      }
      app_log(2, "    Evaluate RPA:          {0:.3f} sec", _Timer.elapsed("EVALUATE_RPA"));
      app_log(2, "        - Alloc:           {0:.3f} sec", _Timer.elapsed("RPA_ALLOC"));
      app_log(2, "    Imaginary FT tau->w:   {0:.3f} sec", _Timer.elapsed("IMAG_FT_TtoW"));
      app_log(2, "      - FT_REDISTRIBUTE:   {0:.3f} sec\n", _Timer.elapsed("FT_REDISTRIBUTE"));
    }

    void gw_t::print_chol_gw_timers() {
      app_log(2, "\n  Chol-GW timers");
      app_log(2, "  --------------");
      app_log(2, "    Total:                 {0:.3f} sec", _Timer.elapsed("TOTAL"));
      app_log(2, "    Allocations:           {0:.3f} sec", _Timer.elapsed("ALLOC"));
      app_log(2, "    Communication:         {0:.3f} sec", _Timer.elapsed("COMM"));
      app_log(2, "    Evaluate_P0:           {0:.3f} sec", _Timer.elapsed("EVALUATE_P0"));
      app_log(2, "    Dyson_P:               {0:.3f} sec", _Timer.elapsed("DYSON_P"));
      app_log(2, "    Evaluate_Sigma:        {0:.3f} sec", _Timer.elapsed("EVALUATE_SIGMA"));
      app_log(2, "    Imaginary FT:          {0:.3f} sec", _Timer.elapsed("IMAG_FT"));
      app_log(2, "    Chol-ERI reader:       {0:.3f} sec\n", _Timer.elapsed("ERI_READER"));
    }

    void gw_t::print_rpa_gw_timers() {
      app_log(2, "\n  Chol-RPA timers");
      app_log(2, "  ---------------");
      app_log(2, "    Total:                 {0:.3f} sec", _Timer.elapsed("TOTAL"));
      app_log(2, "    Allocations:           {0:.3f} sec", _Timer.elapsed("ALLOC"));
      app_log(2, "    Communication:         {0:.3f} sec", _Timer.elapsed("COMM"));
      app_log(2, "    Evaluate_P0:           {0:.3f} sec", _Timer.elapsed("EVALUATE_P0"));
      app_log(2, "    Evaluate_RPA:          {0:.3f} sec", _Timer.elapsed("EVALUATE_RPA"));
      app_log(2, "    Imaginary FT:          {0:.3f} sec", _Timer.elapsed("IMAG_FT"));
      app_log(2, "    Chol-ERI reader:       {0:.3f} sec\n", _Timer.elapsed("ERI_READER"));
    }

  } // solvers
} // methods

