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


#ifndef CUDA_KERNELS_SETTINGS_HPP
#define CUDA_KERNELS_SETTINGS_HPP

#define BOOST_NO_AUTO_PTR

static constexpr size_t MAXIMUM_GRID_DIM_YZ = 65535;
static constexpr size_t SM_SIZE_KB          = 64; // should probably poke the system
static constexpr size_t DEFAULT_BLOCK_SIZE  = 32;
static constexpr size_t DOT_BLOCK_SIZE      = 32;
static constexpr size_t REDUCE_BLOCK_SIZE   = 32;
static constexpr size_t MAX_THREADS_PER_DIM = 1024;

#endif
