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


#ifndef COQUI_DIIS_ALG_HPP
#define COQUI_DIIS_ALG_HPP

#define DIIS_DEBUG 0

#include <complex>
#include "nda/linalg/dot.hpp"
#include <nda/linalg/eigenelements.hpp>
#include "diis_residual.h"


namespace iter_scf {
/** DIIS implementation
 *
 *  Original paper:
 *  P. Pokhilko, C.-N. Yeh, D. Zgid. J. Chem. Phys., 2022, 156, 094101
 *
 *  The code is an adaptation of the DIIS implementation in UGF2 code, 
 *  adapted for nda arrays and nda eigendecomposition / SVD / linear solver
 *
 *  The class must be initialized before usage.
 *
 *  The code closely follows the original ideas from P. Pulay 
 *  and further publications. 
 *  The implementation does not use any specific choice of the residual---
 *  it is defined outside of the class and passed as a pointer. 
 *  A the moment, only commutator residual is tested.
 *  The space of probe vectors used for extrapolation and residuals are stored 
 *  in the corresponding linear spaces. 
 *  The internal linear system (composed of residual overlaps and Lagrange multipliers) 
 *  is solved in a numerically stable way:
 *  \f[ B  1  * (c) = (0) \\
 *      1  0  * (l)   (1) \f]
 *
 *  Since the B matrix of overlaps of residuals can be very small, such system is badly conditioned. 
 *  To avoid numerical instabilities, the system is modified:
 *  \f[ B c = 1 \f]
 *  And then the coefficients are normalized such that the constraint is satisfied. 
 *  In the case of a bad condition number of B (pathological linear dependency between vectors), 
 *  The linear system is solved through pseudoinverse (however, this regime is not tested well).
 *
 * **/


template<typename Vector>
  class diis_alg  {
public:
    // External control of the DIIS driver whether to do extrapolation
    // if it is false, the subspaces will grow and B-matrix will be evaluated as usual
    bool extrap;
    // External control whether to grow x_vsp "only" but not res_vsp and the B-matrix
    // (no extrapolation as well)
    bool grow_xvsp_only;

    auto get_extrapolated_state() {
        if(extrapolated_state == nullptr) {
            APP_ABORT("DIIS state is not initialized! ABORT!");
        }
        return extrapolated_state->get();
    }
private:
    
    nda::matrix<ComplexType> m_B;       // Overlap matrix of the residuals
    nda::array<ComplexType, 1> m_C;     // Vector of extrapolation coefficients
    std::complex<double> lambda;        // Lagrange multiplier for the constraint
    size_t max_subsp_size;              // maximum size of the DIIS subspace

    // non-owning pointers
    VSpace<Vector>* res_vsp;            // vector space of residuals
    VSpace<Vector>* x_vsp;              // vector space of states
    opt_state<Vector>* extrapolated_state = nullptr; // diis extrapolated state

    diis_residual<Vector>* residual;    // Defines residual. Must be already initialized

    std::string diis_str = "DIIS: ";

    void print_B() {
        std::cout << diis_str << "error overlaps B:" << std::endl;
        std::cout << std::setprecision(10);
        for(auto i : nda::range(0, m_B.shape()[0])) {
            for(auto j : nda::range(0, m_B.shape()[1])) 
                std::cout << m_B(i,j) << " ";

            std::cout << std::endl;;
        }
    }

    void print_C() {
        std::cout << diis_str << "Extrapolation coefficients:" << std::endl;
        std::cout << std::setprecision(10);
        for(auto i : nda::range(0, m_B.shape()[0]))
                std::cout << m_C(i) << " ";

            std::cout << std::endl;;
    }

public:

    /**
     * Initialization of the DIIS algorithm
     * 1. setup pointers for non-owning data used in DIIS kernel, including
     *     a. diis extrapolated state (extrapolated_state_)
     *     b. residual definition (residual_)
     *     c. vector space of states (x_vsp_)
     *     d. vector space of residuals (res_vsp_)
     * 2. add the starting vector () to the X vector space
     *
     *  @param extrapolated_state_ - [INPUT] Pointer to the diis output state
     *  @param residual_           - [INPUT] Pointer to the residual definition
     *  @param x_vsp_              - [INPUT] Pointer to the vector space of the states
     *  @param res_vsp_            - [INPUT] Pointer to the vector space of residuals
     *  @param max_subsp_size_     - [INPUT] Maximum size of the DIIS subspace
     *  @param extrap_             - [INPUT] Whether to perform extrapolation or just grow the subspace
     *  @param x_start             - [INPUT] Initial vector to be added to the X vector space
     */
    void init(opt_state<Vector>* extrapolated_state_, diis_residual<Vector>* residual_,
              VSpace<Vector>* x_vsp_, VSpace<Vector>* res_vsp_,
              size_t max_subsp_size_, bool extrap_, const Vector& x_start) {

        utils::check(residual_->is_inited(), "diis_alg: The residual is not initialized");

        extrapolated_state = extrapolated_state_;
        residual = residual_;
        x_vsp = x_vsp_;
        res_vsp = res_vsp_;
        max_subsp_size = max_subsp_size_;
        extrap = extrap_;
        x_vsp->add_to_vspace(x_start);
#if DIIS_DEBUG
        print_B();
#endif
    };

    /**
     * Perform the next DIIS step
     *
     * return 1 if extrapolation was performed
     *        0 if no extrapolation (just growing the subspace)
     */
    int next_step(const Vector& new_vec) {
        if (x_vsp->size() == 0 || grow_xvsp_only) {
            app_log(2, diis_str + "Growing vector subspace only. No extrapolation.\n");
            x_vsp->add_to_vspace(new_vec);    // growing vector space
            app_log(2, "");
            return 0;
        }

        if (res_vsp->size() < max_subsp_size) {
            // Normal execution
            app_log(2, diis_str + "Growing vector and residual subspaces for DIIS\n");
            // Fill the extrapolated state with the current vector for residual computation
            extrapolated_state->put(new_vec);
            Vector res;
            if(! residual->get_diis_residual(res) ) {
                APP_ABORT(diis_str +  "Could not get residual!!! ABORT!");
            }
            update_overlaps(res); // the overlap with res is added in any case...

            res_vsp->add_to_vspace(res);   // growing residual space
            x_vsp->add_to_vspace(new_vec); // growing vector space
        } else {
            // The subspace is already of the maximum size
            app_log(2, diis_str + "Reached maximum subspace -> the first vector will be kicked out of the subspace.\n");
            // can do it smarter and purge the one with the smallest coef
            res_vsp->purge_vec(0); // remove the first residual
            x_vsp->purge_vec(0);   // remove the first vector
            purge_overlap(0);      // purge overlap matrix of residuals

            // Fill the extrapolated state with the current vector for residual computation
            extrapolated_state->put(new_vec);
            Vector res;
            if(! residual->get_diis_residual(res) ) {
                APP_ABORT(diis_str +  "Could not get residual!!! ABORT!");
            }
            update_overlaps(res);
            res_vsp->add_to_vspace(res);   // growing residual space
            x_vsp->add_to_vspace(new_vec); // growing vector space
        }

        if (extrap && (res_vsp->size() > 1) ) {
            app_log(2, diis_str + "Performing the DIIS extrapolation. \n");
            compute_coefs(1);
            print_B();
            print_C();
            if(m_B.shape()[0] == m_B.shape()[1] && m_B.shape()[0] == m_C.shape()[0]) {
                nda::array<ComplexType,1> vec_error(m_B.shape()[0]);
                nda::blas::gemv(m_B, m_C, vec_error);
                ComplexType exp_error = nda::sum(vec_error);
                app_log(2, diis_str + "Squared predicted error of extrapolated vector (e,e) = {}", std::real(exp_error));
            }

            // build extrapolated vector
            Vector result = x_vsp->make_linear_comb(m_C);
            app_log(2, "");
            extrapolated_state->put(result); // update extrapolated state
            return 1;
            
        } else {
            app_log(2, diis_str + "DIIS extrapolation condition is not satisfied -> Skipping the extrapolation step\n");
            print_B();
            app_log(2, "");
            return 0;
        }
        app_log(2, ""); // beautification
        return 0;
    }

private:

    /* Remove overlaps with the vector k
 *     The dimensions of the matrix m_B are shrinked by 1
 *   */
    void purge_overlap(const size_t k) {
        nda::matrix<ComplexType> Bnew(m_B.shape()[0]-1,m_B.shape()[1]-1);
        for(size_t i = 0, mi = 0; i < Bnew.shape()[0]; i++, mi++) {
            if(i == k) ++mi;
            for(size_t j = 0, mj = 0; j < Bnew.shape()[1]; j++, mj++) {
                if(j==k) ++mj;
                Bnew(i, j) = m_B(mi, mj);               
            }
        }
        m_B = Bnew;
    }

    /* Add overlaps with the incoming vector 
 *     The dimensions of the matrix m_B are extended by 1
 *   */
    void update_overlaps(Vector& u) {
#if DIIS_DEBUG
        print_B();
       if(m_B.shape()[1] > 1){
       std::cout << "Before the update" << std::endl;
        print_B();
       SelfAdjointEigenSolver<MatrixXcd> es;
       es.compute(m_B);
           std::cout << "evals: " << es.eigenvalues().transpose() << std::endl;
       std::cout << "updating..." << std::endl;
       }
#endif
        nda::matrix<ComplexType> Bnew(m_B.shape()[0]+1,m_B.shape()[1]+1);
        Bnew() = 0;

        // Assing what is known already:
        for(size_t i = 0; i < m_B.shape()[0]; i++) {
            for(size_t j = 0; j < m_B.shape()[1]; j++)
                Bnew(i,j) = m_B(i,j);
        }

        // Evaluate new overlaps and add them to B:
        // Can ship this piece as a function with the vector space 
        // for good parallelization
        for(size_t i = 0; i < m_B.shape()[1]; i++) {
            Bnew(i, m_B.shape()[1]) = res_vsp->overlap(i, u);
            Bnew(m_B.shape()[1], i) = std::conj(Bnew(i, m_B.shape()[1]));
        }
       Bnew(m_B.shape()[1],m_B.shape()[1]) = res_vsp->overlap(u, u);
       m_B = Bnew;
#if DIIS_DEBUG
       std::cout << "After the update" << std::endl;
        print_B();
#endif
    }

/*
    // Simple, numerically unstable version
    void compute_coefs_simple() {
#if DIIS_DEBUG
        print_B();
#endif

        nda::array<ComplexType, 1> Cnew(m_B.shape()[1]);
        nda::matrix<ComplexType> B_cnstr(m_B.shape()[0]+1, m_B.shape()[1]+1);

        // Overlaps of error vectors
        for(size_t i = 0; i < m_B.shape()[0]; i++) {
            for(size_t j = 0; j < m_B.shape()[1]; j++)
                B_cnstr(i,j) = m_B(i,j);
        }

        // Constrants
        for(size_t i = 0; i < m_B.shape()[0]; i++) {
            B_cnstr(i, m_B.shape()[1]) = 1;
            B_cnstr(m_B.shape()[1], i) = 1;
        }

        B_cnstr(m_B.shape()[1], m_B.shape()[1]) = 0;
        
        nda::array<ComplexType, 2> b(B_cnstr.shape()[1],1);
        b() = 0;
        b(B_cnstr.shape()[1] - 1, 0) = 1; // constraint


#if DIIS_DEBUG
        std::cout << "B_cnstr:" << std::endl;
        for(size_t i = 0; i < B_cnstr.rows(); i++) {
            for(size_t j = 0; j < B_cnstr.shape()[1]; j++)
                std::cout << B_cnstr(i,j) << "  ";

            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "b" << std::endl;
        for(size_t i = 0; i < b.extent(0); i++) std::cout << b(i,0) << "  ";
        std::cout << std::endl;
#endif
    
        //FIXME! Test
        auto x = linear_solver_getrs(B_cnstr, b);
        for(size_t i = 0; i < m_B.shape()[0]; i++) {
            Cnew[i] = x(i,0);
        }
        m_C = Cnew;
        lambda = x(B_cnstr.shape()[1] - 1,0);
        app_log(2, "lambda = {}", lambda);
     }

    // Based on nda lapack example
    nda::matrix<ComplexType> linear_solver_getrs(nda::matrix<ComplexType> A, nda::array<ComplexType, 2>b) {
        nda::matrix<ComplexType> Acopy = A;
        nda::matrix<ComplexType> bcopy = b;
        nda::array<int, 1> ipiv(A.shape()[1]);
        nda::lapack::getrf(Acopy, ipiv);
        nda::lapack::getrs(Acopy, bcopy, ipiv);
        auto X = nda::matrix<ComplexType>{bcopy};
        // TODO: need to test it...
        std::cout << "X:" << std::endl;
        for(size_t i = 0; i < X.shape()[0]; i++) {
            for(size_t j = 0; j < X.shape()[1]; j++)
                std::cout << X(i,j) << "  ";
            std::cout << std::endl;
        }

        return X;
    }
*/


    void compute_coefs(size_t option) {
        switch(option) {
            case 1:
                compute_coefs_c1(); 
                break;
            case 2:
                compute_coefs_c1(); 
                //compute_coefs_simple(); 
                //compute_coefs_c2(); 
                break;
            default:
                compute_coefs_c1();
        }
    }


    void compute_coefs_c1() {

//#pragma float_control(precise, on) // Need accurate extrapolation coeffs
        auto B = nda::make_regular(nda::real(m_B)); // only real part is needed due to constraint to real coefs

        nda::array<double, 1> bb(B.shape()[1]); 
        bb() = 1.0;

        auto [eig, evecs] = nda::linalg::eigenelements(B);
        auto evecs_tr = nda::make_regular(nda::transpose(evecs));

        nda::matrix<double> Binv(B.shape()[0], B.shape()[1]); // Inverse or pseudoinverse
        nda::matrix<double> eig_inv(B.shape()[0], B.shape()[1]);
        nda::matrix<double> I(B.shape()[0], B.shape()[1]);
        Binv() = 0;
        eig_inv() = 0;

        double eig_max = nda::max_element(eig);
        double eig_min = nda::min_element(eig);
        double cond = eig_max / eig_min;

        const double eig_thresh = 1E-12;

        app_log(2, diis_str + "Condition number of B: {}", cond);

        for (auto i : nda::range(0, eig.size())) { 
            if(eig(i)*cond > eig_thresh) {
                eig_inv(i,i) = 1.0/(eig(i)); 
            }
        }

        nda::blas::gemm(evecs, eig_inv, I);
        nda::blas::gemm(I, evecs_tr, Binv);

        nda::array<double, 1> x(B.shape()[1]); 
        nda::blas::gemv(1.0, Binv, bb, 0.0, x);
        
        std::complex<double> sum = std::accumulate(x.begin(), x.end(), 0.0);
        m_C = make_regular(nda::real(x / sum));
     }


};

} // namespace
#endif // COQUI_DIIS_ALG_HPP
