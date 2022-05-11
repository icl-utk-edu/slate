// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
// #include "slate/Matrix.hh"
// #include "slate/Tile_blas.hh"
// #include "slate/HermitianBandMatrix.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"

#include <string.h>
#include <omp.h>
#include <lapack/config.h>
#include "lapack/fortran.h"

namespace slate {

//------------------------------------------------------------------------------
/// @ingroup svd_specialization
//
//  Only real versions (s,d).
//  This code is not designed to be called directly by users; it is a subroutine
//  for stevx2.cc.
//
//  Specifically, this is a task-based parallel algorithm, the parameters are
//  contained in the already initialized and populated stevx2_control_t; For
//  example, from stevx2:
//
//   #pragma omp parallel
//   {
//       #pragma omp single
//       {
//           stevx2_bisection(&control, ...etc...);
//       }
//   }
//
/// @param[in] *control
///          A pointer to the global variables needed.
///
/// @param[in] control->n
///          int number of rows in the matrix.
///
/// @param[in] control->diag
///          real array of [N] diagonal elements of the matrix.
///
/// @param[in] control->offd
///          real array of [N-1] sub-diagonal elements of the matrix.
///
/// @param[in] control->range
///          lapack::Range.
///              Index if user is finding eigenvalues by index range.
///              Value if user is finding eigenvuales by value range.
///
/// @param[in] control->jobtype
///          lapack::Job.
///              NoVec  if user does not want eigenvectors computed.
///              Vec    if user desires eigenvectors computed.
///
/// @param[in] control->il
///          int enum. The lower_bound of an index range if range is Index.
///
/// @param[in] control->iu
///          int enum. The upper_bound of an index range, if range is Index.
///
/// @param[in] control->stein_arrays
///          array of [max_threads], type stevx2_Stein_Array_t, contains work
///          areas per thread for invoking _stein (inverse iteration to find
///          eigenvectors).
///
/// @param[in] control->base_idx
///          The index of the least eigenvalue to be found in the bracket,
///          used to calculate the offset into the return vectors/arrays.
///
/// @param[out] control->error
///          If non-zero, the first error we encountered in the operation.
///
/// @param[out] control->pval
///          real vector of [eigenvaues] to store the eigenvalues discovered,
///          these are returned in ascending sorted order.
///
/// @param[out] control->pvec
///          Matrix of [N x eigenvalues] to store the eigenvectors, not
///          referenced unless jobtype==Vec. Stored in the same order as
///          their corresponding eigenvalue.
///
/// @param[out] control->pmul
///          int vector of [eigenvalues], the corresponding multiplicity of
///          each eigenvalue, typically == 1.
///
/// @param[in] lower_bound
///          Real lower_bound (inclusive) for range of eigenvalues to find.
///
/// @param[in] upper_bound
///          Real upper_bound (non-inclusive) of range of eigenvalues to find.
///
/// @param[in] nlt_low
///          int number of eigenvalues less than lower_bound. Computed if < 0.
///
/// @param[in] nlt_hi
///          int number of eigevalues less than upper_bound. Computed if < 0.
///
/// @param[in] num_ev
///          int number of eigenvalues in [lower_bound, upper_bound). Computed
///          if either nlt_low or nlt_hi were computed.
//
//  A 'bracket' is a range of either real eigenvalues, or eigenvalue indices,
//  that this code is given to discover. It is provided in the arguments.  Upon
//  entry, the number of theoretical eigenvalues in this range has already been
//  determined, but the actual number may be less, due to ULP-multiplicity. (ULP
//  is the Unit of Least Precision, the magnitude of the smallest change
//  possible to a given real number). To explain: A real symmetric matrix in NxN
//  should have N distinct real eigenvalues; however, if eigenvalues are closely
//  packed either absolutely (their difference is close to zero) or relatively
//  (their ratio is close to 1.0) then in real arithmetic two such eigenvalues
//  may be within ULP of each other, and thus represented by the same real
//  number. Thus we have ULP-multiplicity, two theoretically distinct
//  eigenvalues represented by the same real number.
//
//
//  This algorithm uses Bisection by the Scaled Sturm Sequence, implemented in
//  stevx2_bisection, followed by the LAPACK routine _STEIN, which uses inverse
//  iteration to find the eigenvalue.  The initial 'bracket' parameters should
//  contain the full range for the eigenvalues we are to discover. The algorithm
//  is recursively task based, at each division the bracket is divided into two
//  brackets. If either is empty (no eigenvalues) we discard it, otherwise a new
//  task is created to further subdivide the right-hand bracket while the
//  current task continues dividing the left-hand side, until it can no longer
//  divide it, and proceeds to store the eigenvalue and compute the eigenvector
//  if needed. Thus the discovery process is complete when all tasks are
//  completed. We then proceed to orthogonalizing any eigenvectors discovered;
//  because inverse iteration does not inherently ensure orthogonal
//  eigenvectors.
//
//  The most comparable serial LAPACK routine is DLAEBZ.
//
//  Once all thread work is complete, the code will condense these arrays to
//  just the actual number of unique eigenvalues found, if any ULP-multiplicity
//  is present.
//
//  For an MPI implementation; if we have K total eigenvalues to find, assign
//  each of the P nodes (K/P) eigenpairs to find. Provide each the entire range
//  and their index; they can compute their sub-range to find.

//------------------------------------------------------------------------------
// Use LAPACK stein to find a single eigenvector.  We may use this routine
// multiple times, so instead of allocating/freeing the work spaces repeatedly,
// we have an array of pointers, per thread, to workspaces we allocate if not
// already allocated for this thread. So we don't allocate more than once per
// thread. These are freed by the main program before exit.  Returns INFO.
// 0=success. <0, |INFO| is invalid argument index. >0, if eigenvector failed
// to converge.
// These cannot be a template, they need to call different lapack routines
// based on precision.
//------------------------------------------------------------------------------
int64_t LAPACK_stevx2_stein(
        const int64_t n, const float* diag, const float* offd, const float u,
        float* v, stevx2_stein_array_t<float>* my_arrays)
{
    lapack_int m = 1; // number of eigenvectors to find.
    lapack_int ldz = n;
    lapack_int info;
    lapack_int my_n = n;
    int64_t thread = omp_get_thread_num();
    // ensure all arrays are the right size; does nothing if already correct.
    my_arrays[thread].iblock.resize(n);
    my_arrays[thread].iblock[0] = 1;
    my_arrays[thread].isplit.resize(n);
    my_arrays[thread].isplit[0] = n;
    my_arrays[thread].work.resize(5*n);
    my_arrays[thread].iwork.resize(n);
    my_arrays[thread].ifail.resize(n);

    float w = u; // our eigenvalue.

    LAPACK_sstein(&my_n, diag, offd, &m, &w, &my_arrays[thread].iblock[0],
                  &my_arrays[thread].isplit[0], v, &ldz,
                  &my_arrays[thread].work[0], &my_arrays[thread].iwork[0],
                  &my_arrays[thread].ifail[0], &info);
    return info;
}

lapack_int LAPACK_stevx2_stein(
        const int64_t n, const double* diag, const double* offd, const double u,
        double* v, stevx2_stein_array_t<double>* my_arrays)
{
    lapack_int m = 1; // number of eigenvectors to find.
    lapack_int ldz = n;
    lapack_int info;
    lapack_int my_n = n;
    int64_t thread = omp_get_thread_num();
    // ensure all arrays are the right size; does nothing if already correct.
    my_arrays[thread].iblock.resize(n);
    my_arrays[thread].iblock[0] = 1;
    my_arrays[thread].isplit.resize(n);
    my_arrays[thread].isplit[0] = n;
    my_arrays[thread].work.resize(5*n);
    my_arrays[thread].iwork.resize(n);
    my_arrays[thread].ifail.resize(n);

    blas::real_type<double> w = u; // our eigenvalue.

    LAPACK_dstein(&my_n, diag, offd, &m, &w, &my_arrays[thread].iblock[0],
                  &my_arrays[thread].isplit[0], v, &ldz,
                  &my_arrays[thread].work[0], &my_arrays[thread].iwork[0],
                  &my_arrays[thread].ifail[0], &info);
    return info;
}

//-----------------------------------------------------------------------------
// This a task that subdivides a bracket, throwing off other tasks like this
// one if necessary, until the bracket zeroes in on a single eigenvalue, which
// it then stores, and possibly finds the corresponding eigenvector.
// Parameters:
//      control:    Global variables.
//      lower_bound: of bracket to subdivide.
//      upper_bound: of bracket to subdivide.
//      nlt_low:    number of eigenvalues less than lower bound.
//                  -1 if it needs to be found.
//      nlt_hi:     number of eigenvalues less than the upper bound.
//                  -1 if it needs to be found.
//      num_ev:      number of eigenvalues within bracket. Computed if either
//                  nlt_Low or nlt_hi is computed.
//-----------------------------------------------------------------------------

//------------------------------------------------------------------------------
/// computes a range of eigenvalues/eigenvectors of a symmetric tridiagonal
/// matrix, using the Sturm sequence and Bisection; followed by Inverse
/// Iteration if eigenvectors are desired.
/// Generic implementation for any target.
/// @ingroup svd_specialization
///
// ATTENTION: only host computation supported for now
//
template <typename scalar_t>
void stevx2_bisection(
           stevx2_control_t<scalar_t>* control, scalar_t lower_bound,
           scalar_t upper_bound, int64_t nlt_low, int64_t nlt_hi, int64_t num_ev)
{
    trace::Block trace_block("slate::stevx2_bisection");

    using blas::max;

    const scalar_t* diag = control->diag;
    const scalar_t* offd = control->offd;
    int64_t n = control->n;

    scalar_t cp;
    int64_t flag=0;
    int64_t ev_less;

    if (nlt_low < 0) {
        nlt_low = lapack::sturm(n, diag, offd, lower_bound);
        flag=1;
    }

    if (nlt_hi < 0) {
        nlt_hi =  lapack::sturm(n, diag, offd, upper_bound);
        flag=1;
    }

    if (flag) {
        num_ev = (nlt_hi - nlt_low);
    }

    // If there are no eigenvalues in the supplied range, we are done.
    if (num_ev < 1) return;

    if (control->range == lapack::Range::Index) {
        if (nlt_hi  < control->il ||    // e.g if il=500, and nlt_hi=499, this
                                        // bracket is under range of interest.
            nlt_low > control->iu) {    // e.g if iu=1000, and nlt_low=1001,
                                        // bracket is above range of interest.
            return;
        }
    }

    // Bisect the bracket until we can't anymore.

    flag = 0;
    for (;;) {
        cp = (lower_bound+upper_bound)*0.5;
        if (cp == lower_bound
            || cp == upper_bound) {
            // Our bracket has been narrowed to machine epsilon for this
            // magnitude (=ulp).  We are done; the bracket is always
            // [low,high). 'high' is not included, so we have num_ev eigenvalues
            // at low, whether it == 1 or is > 1. We find the eigenvector. (We
            // can test multiplicity with a GluedWilkinson matrix).

            break; // exit for(;;).
        } else {
            // we have a new cutpoint.
            ev_less = lapack::sturm(n, diag, offd, cp);
            if (ev_less < 0) {
                // We could not compute the Sturm sequence for it.
                flag = -1; // indicate an error.
                break; // exit for (;;).
            }

            // Discard empty halves in both Range Value and Index.
            // If #EV < cutpoint is the same as the #EV < high, it means
            // no EV are in [cutpoint, hi]. We can discard that range.

            if (ev_less == nlt_hi) {
                upper_bound = cp;
                continue;
            }

            // If #EV < cutpoint is the same as #EV < low, it means no
            // EV are in [low, cutpoint]. We can discard that range.

            if (ev_less == nlt_low) {
                lower_bound = cp;
                continue;
            }

            // Note: If we were Range Value the initial bounds given by the
            // user are the ranges, so we have nothing further to do. In Range
            // Index; the initial bounds are Gerschgorin limits and not enough:
            // We must further narrow to the desired indices.

            if (control->range == lapack::Range::Index) {
                // For Range Index: Recall that il, iu are 1-relative; while
                // ev_less is zero-relative; i.e.  if [il,iu]=[1,2], evless must
                // be 0, or 1.  when ev_less<cp == il-1, or just <il, cp is a
                // good boundary and we can discard the lower half.
                //
                // To judge the upper half, the cutpoint must be < iu, so if it
                // is >= iu, cannot contain eigenvalue[iu-1].  if ev_less >= iu,
                // we can discard upper half.

                if (ev_less < control->il) {
                    // The lower half [lower_bound, cp) is not needed, it has no
                    // indices >= il.

                    lower_bound = cp;
                    nlt_low    = ev_less;
                    num_ev = (nlt_hi-nlt_low);
                    continue;
                }

                if (ev_less >= control->iu) {
                    // The upper half [cp, upper_bound) is not needed, it has no
                    // indices > iu;

                    upper_bound = cp;
                    nlt_hi     = ev_less;
                    num_ev = (nlt_hi-nlt_low);
                    continue;
                }
            }

            // Here, the cutpoint has EV on both left right. We push off the
            // right bracket.  The new lower_bound is the cp, the upper_bound is
            // unchanged, the number of eigenvalues changes.

            #pragma omp task
                stevx2_bisection(control, cp, upper_bound, ev_less, nlt_hi,
                                 (int64_t) (nlt_hi-ev_less));

            // Update the Left side I kept. The new number of EV less than
            // upper_bound is ev_less, recompute number of EV in the bracket.

            upper_bound = cp;
            nlt_hi = ev_less;
            num_ev =( ev_less - nlt_low);
            continue;
         }
    }

    // Okay, count this eigenpair done, add to the Done list.
    // NOTE: nlt_low is the global zero-relative index of
    //       this set of mpcity eigenvalues.
    //       No other brackets can change our entry, so we
    //       don't need any thread block or atomicity.

    int my_idx;
    if (control->range == lapack::Range::Index) {
        my_idx = nlt_low - (control->il-1);
    } else { // range == Value
        my_idx = nlt_low - control->base_idx;
    }

    if (control->jobtype == lapack::Job::Vec) {
        // get the eigenvector.
        int ret=LAPACK_stevx2_stein(
            control->n, diag, offd, lower_bound,
            &(control->pvec[my_idx*control->n]), control->stein_arrays);
        if (ret != 0) {
            #pragma omp critical (UpdateStack)
            {
                // Only store first error we encounter
                if (control->error == 0)
                    control->error = ret;
            }
        }
    }

    // Add eigenvalue and multiplicity.
    control->pval[my_idx]=lower_bound;
    control->pmul[my_idx]=num_ev;
} // end stevx_bisection()

//------------------------------------------------------------------------------
template
void stevx2_bisection<float>(
            stevx2_control_t<float>* control, float lower_bound,
            float upper_bound, int64_t nlt_low, int64_t nlt_hi, int64_t num_ev);

template
void stevx2_bisection<double>(
            stevx2_control_t<double>* control, double lower_bound,
            double upper_bound, int64_t nlt_low, int64_t nlt_hi, int64_t num_ev);
} // end namespace slate.
