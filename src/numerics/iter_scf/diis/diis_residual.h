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


#ifndef COQUI_DIIS_RESIDUAL_H
#define COQUI_DIIS_RESIDUAL_H

#include "vspace.h"
#include "state.h"

namespace iter_scf {

// Fallback implementation of the residual as a difference between vectors
// Not tested well.

template<typename Vector>
class diis_residual {
protected:
    VSpace<Vector>* x_vsp = nullptr;    // vector subspace of X vectors
    opt_state<Vector>* current_state = nullptr; // pointer to the current state (used to calculate current residual)
    bool is_initialized = false;

public:
    
    diis_residual() {} // must be initialized!

    diis_residual(VSpace<Vector>* x_space, opt_state<Vector>* current_state_) {
        init(x_space, current_state_);
    }

    virtual void init(VSpace<Vector>* x_space, opt_state<Vector>* current_state_) {
        x_vsp = x_space;
        current_state = current_state_;
        is_initialized = true;
    }

    bool is_inited() const {
        return is_initialized;
    }


    /**
     * Compute the canonical residual as the difference between successive iterations
     * using the current state ("state") and the last vector in the X vector space (x_vsp).
     *
     * This should be a reasonable default choice for the DIIS residual.
     *
     * @param res - [OUTPUT] residual vector
     */
    virtual bool get_diis_residual(Vector& res) {
        utils::check(is_initialized, "DIIS difference residual is not initialized");
        if(x_vsp->size() >= 2) {
            res = current_state->get();
            res.add(x_vsp->get_vec(x_vsp->size()-1), -1.0);
            return true;
        }
        else {
            return false;
        }
    };


};
}


#endif // COQUI_DIIS_RESIDUAL_H
