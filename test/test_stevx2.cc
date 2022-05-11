// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"
#include "grid_utils.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_support_routines.hh"
#include "scalapack_copy.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#define SLATE_HAVE_SCALAPACK

//------------------------------------------------------------------------------
// Matrix detailed in Kahan; et al. 
// Matrix Test: diag=[+x,-x,+x,-x,...+x,-x] for any real x, but Kahan chooses
//                                          a tiny x.
//              offd=[1,1,...1]
// Dimension: n. 
// Computed eigenvalues:
// evalue[k] = [ x*x + 4*cos(k/(n+1))^2 ] ^(1/2), 
// evalue[n+1-k] = -evalue[k], for k=1,[n/2],
// evalue[(n+1)/2] = 0 if n is odd.
// Note k is 1-relative in these formulations.
// The eigenvalues range from (-2,+2).
// Note: This routine verified to match documentation for n=4,8,12,24.
// Note: This code is a template, it is not intended to work in complex
//       arithmetic, it is only to be translated to either single or double.
//------------------------------------------------------------------------------
template<typename scalar_t>
void stevx2_test_matrix_kahan(
    std::vector<scalar_t>& diag, std::vector<scalar_t>& offd,
    std::vector< scalar_t>& evalue, int64_t n, scalar_t my_diag_value)
{
    int64_t i,k;
    for (k = 1; k <= (n/2); ++k) {
        scalar_t ev;
        ev = (M_PI*k+0.)/(n+1.0);           // angle in radians...
        ev = cos(ev);                       // cos(angle)...
        ev *= 4.*ev;                        // 4*cos^2(angle)...
        ev += my_diag_value*my_diag_value;  // x^2 + 4*cos^2(angle)...
        ev = sqrt(ev);                      // (x^2 + 4*cos^2(angle))^(0.5).
        // we reverse the -ev and ev here, to get in ascending sorted order.
        evalue[k-1] = -ev;
        evalue[n+1-k-1] = ev;
    }

    for (i = 0; i < n-1; ++i) {
        offd[i] = 1.0;
        k=(i&1);    // 0=even, 1=odd.

        if (k) 
            diag[i]=-my_diag_value;
        else
            diag[i]=my_diag_value;
    }

    // No offd for final diagonal entry.
    k=(i&1);
    if (k) 
        diag[i]=-my_diag_value;
    else
        diag[i]=my_diag_value;
}

//------------------------------------------------------------------------------
// @brief Tests slate version of stevx2.
//
// @param[in,out] param - array of parameters
// @param[in]     run - whether to run test
//
// Sets used flags in param indicating parameters that are used.
// If run is true, also runs test and stores output parameters.
//------------------------------------------------------------------------------
template<typename scalar_t>
void test_stevx2_work(Params& params, bool run)
{
    using real_t = blas::real_type<scalar_t>;

    // Constants
    const scalar_t zero = 0.0, one = 1.0;

    // get & mark input values
    int64_t m = params.dim.m();
    int64_t nb = params.nb();
    int64_t p = params.grid.m();
    int64_t q = params.grid.n();
    bool check = params.check() == 'y';
    bool trace = params.trace() == 'y';
    int verbose = params.verbose();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();

    // mark non-standard output values
    params.time();

    if (! run)
        return;

    // skip invalid or unimplemented options
    if (target != slate::Target::HostTask) {
        params.msg() = "skipping: stevx2 is only implemented for HostTask";
        return;
    }

    slate::Options const opts =  {
        {slate::Option::Target, target}
    };

    int64_t i,j;

    std::vector<scalar_t> diag;
    std::vector<scalar_t> offd;
    std::vector<scalar_t> kahan_eigenvalues;
    std::vector<scalar_t> found_eigenvalues;
    slate::Matrix<scalar_t> found_eigenvectors;
    std::vector<int64_t> found_mult;

    diag.resize(m);
    offd.resize(m-1);
    kahan_eigenvalues.resize(m);
    found_eigenvalues.resize(m);
    found_mult.resize(m);

    //--------------------------------------------------------------------------
    // Kahan has eigenvalues from [-2.0 to +2.0]. However, eigenvalues are 
    // dense near -2.0 and +2.0, so for large matrices, the density may cause
    // eigenvalues separated by less than machine precision, which causes us
    // multiplicity (eigenvalues are identical at machine precision). We first
    // see this in single precision at m=14734, with a multiplicity of 2. 
    //--------------------------------------------------------------------------

    scalar_t my_diag=1.e-5;
    stevx2_test_matrix_kahan(diag, offd, kahan_eigenvalues, m, my_diag);
    double min_abs_ev=__DBL_MAX__, max_abs_ev=0., kond;
    for (i = 0; i < m; ++i) {
        if (fabs(kahan_eigenvalues[i]) < min_abs_ev) 
            min_abs_ev=fabs(kahan_eigenvalues[i]);
        if (fabs(kahan_eigenvalues[i]) > max_abs_ev)
            max_abs_ev=fabs(kahan_eigenvalues[i]);
    }

    kond = max_abs_ev / min_abs_ev;

    int64_t n_eig_vals=0;
    int64_t il=0, iu=500;
    int64_t ev_idx;
    scalar_t vl=1.5, vu=2.01;

    // we find the index in kahan_eigenvalues of first >=vl.
    for (ev_idx = 0; ev_idx < m; ++ev_idx)
        if (kahan_eigenvalues[ev_idx] >= vl) break;

    // Run and time stevx2, range based on values.
    if (trace)
        slate::trace::Trace::on();
    else
        slate::trace::Trace::off();

    double time = barrier_get_wtime(MPI_COMM_WORLD);

    slate::stevx2(lapack::Job::Vec, lapack::Range::Value, diag, offd, vl, vu,
                  il, iu, found_eigenvalues, found_mult, found_eigenvectors,
                  MPI_COMM_WORLD);

    n_eig_vals = found_eigenvalues.size();  // unique eigenvalues found.

    time = barrier_get_wtime(MPI_COMM_WORLD) - time;
    if (trace)
        slate::trace::Trace::finish();
    params.time() = time;

    // Test results directly. Check eigenvalues discovered by vl, vu.

    if (check) {
        //----------------------------------------------------------------------
        // Find worst eigenvalue error. However, we must worry about
        // multiplicity. In single precision this first occurs at m=14734, with
        // vl=1.5, vu=2.01; mpcity=2. At m=75000, vl=1.5, vu=2.01, mpcity=10.
        // We must also worry about the magnitude of eigenvalues; machine 
        // epsilon for large eigenvalues is much greater than for small ones.
        //----------------------------------------------------------------------

        scalar_t worst_eigvalue_error = zero;
        int64_t worst_eigvalue_found_index = 0, worst_eigvalue_global_index = 0,
                worst_eigvalue_mpcty = 0;
        int64_t max_mpcty = 0;
        scalar_t worst_eigvector_error = zero;
        int64_t worst_eigvector_index = 0;
        i=0;
        while (i < n_eig_vals
               && ev_idx < n_eig_vals) {
            if (found_mult[i] > max_mpcty) 
                max_mpcty = found_mult[i];

            for (j = 0; j < found_mult[i]; ++j) {
                double ev_eps = nexttoward(fabs(found_eigenvalues[i]),
                     __DBL_MAX__) - fabs(found_eigenvalues[i]);
                scalar_t error;
                error = fabs(found_eigenvalues[i]-kahan_eigenvalues[ev_idx]);
                error /= ev_eps;
                if (error > worst_eigvalue_error) {
                    worst_eigvalue_found_index = i;
                    worst_eigvalue_global_index = ev_idx;
                    worst_eigvalue_error = error;
                    worst_eigvalue_mpcty = found_mult[i];
}

                ++ev_idx; // advance known eigenvalue index for a multiplicity.
            }
           
            ++i; // advance to next discovered eigenvalue.
        }

        //----------------------------------------------------------------------
        // Worth reporting for debug: worst_eigvalue_index,
        // worst_eigvalue_error, max_mpcty.
        //----------------------------------------------------------------------

        params.ref_time() = time;
        params.error() = worst_eigvalue_error;
        params.okay() = (worst_eigvalue_error < (scalar_t) 3.);

        //----------------------------------------------------------------------
        // If we have no eigenvalue errors, We need to test the eigenvectors.
        // stevx2_stepe returns fabs(||(A*pvec)/pval - pvec||_maxabs) for
        // each eigenvalue and eigenvector. We track the largest value.
        // Empirically; the error grows slowly with m. We divide by epsilon,
        // and 2*ceil(log_2(m)) epsilons seems a reasonable threshold without
        // being too liberal. Obviously this is related to the number of bits
        // of error in the result. The condition number (kond) of the Kahan
        // matrix also grows nearly linearly with m; kond is computed above.
        //----------------------------------------------------------------------

        if (params.okay()) { // only test eigenvectors if no eigenvalue errors.
            for (i = 0; i < n_eig_vals; ++i) {
                double verr;
                std::vector<scalar_t> y( m ); 
                int64_t ii=0;
                slate::stevx2_get_col_vector(found_eigenvectors, y, i);

                verr=slate::stevx2_stepe<scalar_t>(&diag[0], &offd[0],
                        diag.size(), found_eigenvalues[i], y);
                if (verr > worst_eigvector_error) {
                    worst_eigvector_error = verr; 
                    worst_eigvector_index = i;
                }
            }

            // Find one norm of the tridiagonal matrix (max abs(column sum)).
            scalar_t test, one_norm = 0;
            one_norm = fabs(diag[0])+fabs(offd[0]);
            test = fabs(offd[m-2])+fabs(diag[m-1]);
            if (test > one_norm)
                one_norm = test;
            for (i = 1; i < (m-1); ++i) {
                test = fabs(offd[i-1]) + fabs(diag[i]) + fabs(offd[i]);
                if (test > one_norm)
                    one_norm = test;
            }

            // Find unit of least precision on 1-Norm.
            scalar_t ulp = nexttoward(one_norm, __DBL_MAX__) - one_norm;
            params.error() = worst_eigvector_error;
            params.okay() = ( (worst_eigvector_error / (m * ulp) ) < 1.0);
        }
    } // end if (check)
}

// -----------------------------------------------------------------------------
void test_stevx2(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_stevx2_work<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_stevx2_work<double> (params, run);
            break;

        case testsweeper::DataType::Integer:
        case testsweeper::DataType::SingleComplex:
        case testsweeper::DataType::DoubleComplex:
            throw std::exception();
            break;
    }
}


// -----------------------------------------------------------------------------
template
void stevx2_test_matrix_kahan<float>(
    std::vector<float>& diag, std::vector<float>& offd,
    std::vector<float>& evalue, int64_t n, float my_diag_value);

template
void stevx2_test_matrix_kahan<double>(
    std::vector<double>& diag, std::vector<double>& offd,
    std::vector<double>& evalue, int64_t n, double my_diag_value);
