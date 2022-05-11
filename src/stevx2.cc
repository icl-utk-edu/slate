// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"

#include <string.h>
#include <omp.h>
#include <lapack/config.h>
#include <lapack/util.hh> // Gives lapack::Job::NoVec or Vec.
#include "lapack/fortran.h"

#include <vector>
#include <ctgmath>

namespace slate {

//------------------------------------------------------------------------------
/// @ingroup svd_specialization
//
///  Only real versions (s,d).
/// Symmetric Tridiagonal Eigenvalues/pairs by range.
/// Computes a caller-selected range of eigenvalues and, optionally,
/// eigenvectors of a symmetric tridiagonal matrix A.  Eigenvalues and
/// eigenvectors can be selected by specifying either a range of values or a
/// range of indices for the desired eigenvalues.
/// This is similiar to LAPACK dstevx, with more output parameters.
//
// Upon return, the vector and matrix will have sizes indicating the number of
// unique eigenvalues/pairs found. We also return a multiplicity vector of the
// same size. A real symmetric matrix in NxN should have N distinct eigenvalues,
// but these can exist within ULP (the unit of least precision) of each other,
// making them represented by the same floating point number. In such cases,
// the multiplicity vector can hold a number greater than 1, to indicate the
// number of eigenvalues within ULP of the reported eigenvalue.
//
// Finding eigenvalues alone is much faster than finding eigenpairs; and the
// majority of the time consumed when eigenvectors are found is in
// orthogonalizing the eigenvectors; an O(N*K^2) operation.
//
// MPI strategy: Given a value range, use Sturm to convert this to an index
// range; and then divide that range into P equal parts for each node to
// process. For example given the value range [-0.5, +0.5) say Sturm on each
// returns the numbers [5000, 6000], i.e. 5000 eigenvalues are less than -0.5,
// and 6000 are less than +0.5. So we have 1000 eigenvalues in the range. Given
// 8 nodes, that is index ranges [5000,5124], [5125,5249], [5250, 5374], etc.
//
// If the range is not evenly divisible by P, add 1 extra element to the first
// R modulo(P) ranges. e.g. R=1003, P=8, 1003 mod 8 = 3, so the first 3 ranges
// get 1 extra value. So index spans of 126,126,126,125,125,125,125,125 values;
// [5000,5125] [5126,5251] [5252,5377] [5378,5502] [5503,5627] [5628,5752]
// [5753,5877] [5878,6002]. The expected range for 1003 values; [5000,6002].
//
/// @param[in] jobtype enum: lapack::Job::NoVec, lapack::Job::Vec. Whether or
/// not to compute eigenvectors.

/// @param[in] range enum: lapack::Range::Value use vl, vu for range [vl, vu).
/// lapack::Range::Index use il, iu for range [il, iu]. 1-relative indices;
/// 1..n.

/// @param[in] diag: Vector of [n] diagonal entries of A. diag.size() must be
/// the order of the matrix.

/// @param[in] offd: Vector onf [n-1] off-diagonal entries of A.

/// @param[in] vl: Lowest eigenvalue included in desired range [vl, vu).  if
/// less than Gerschgorin min; we use Gerschgorin min. Unused if the Range
/// enum is lapack::Range::Index.

/// @param[in] vu: Highest eigenvalue beyond desired range, [vl,vu).  if
/// greater than Gerschgorin max, we use Gerschgorin max+ulp. Unused if the
/// Range enum is lapack::Range::Index. If used, Note that all eigenvalues will
/// be strictly less than vu, in a Range finding.

/// @param[in] il: Low Index of range. Must be in range [1,n]. Unused if the
/// Range enum is lapack::Range::Value.

/// @param[in] iu int. High index of range. Must be in range [1,n], >=il. Unused
/// if the Range enum is lapack::Range::Value.

/// @param[out] Value: vector of desired precision. Resized to fit the number of
/// eigenvalues found. pVal.size() is the number of unique eigenvalues found.

/// @param[out] Mult: vector of type lapack_int. Multiplicity count for each
/// eigenvalue in Value[]; the number of eigenvalues within ulp of the value.

/// @param[out] RetVec: Pass an empty matrix; it will be resized to a 2D Matrix,
/// n x pVal.size(); the orthonormal set of eigenvectors.  Each column 'i'
/// represents the eigenvector for eigenvalue Value[i]. If jobtype =
/// lapack::Job::NoVec, RetVec[] is not referenced or changed.
///
/// @param[in] mpi_comm: MPI communicator to distribute matrix across. To create
/// the RetVec output matrix of eigenvectors.
///
/// @retval: zero for success. Otherwise negative to indicate failure: -i means
/// the i-th argument had an illegal value.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// stevx2_get_col_vector: Retrieves a single vector from a tiled matrix.
template <typename scalar_t>
void stevx2_get_col_vector(
        Matrix<scalar_t>& source, std::vector<scalar_t>& v, int col)
{
    int64_t mb;                         // rows per tile.
    int64_t nb = source.tileNb( 0 );    // assume fixed columns per tile.
    int64_t mt = source.mt();           // Number of row tiles.
    int64_t nt = source.nt();           // Number of column tiles.
    (void) nt;                          // Don't complain about no use.

    int64_t ii=0;                       // position within vector.
    int64_t tidx = col / nb;
    int64_t toff = col - tidx*nb;

    for (int64_t row_tile=0; row_tile < mt; ++row_tile) {
        if (source.tileIsLocal( row_tile, tidx )) {
            auto tile = source( row_tile, tidx );
            mb = source.tileMb(row_tile);   // get rows in THIS tile.
            blas::copy(mb, &tile.at( 0, toff), 1, &v[ ii ], 1);
            ii += mb;
        }
    }

    return;
}

//------------------------------------------------------------------------------
// stevx2_put_col_vector: Copy a single vector into a column of a tiled matrix.
template <typename scalar_t>
void stevx2_put_col_vector(
        std::vector<scalar_t>& v, Matrix<scalar_t>& dest, int col)
{
    int64_t mb;                     // rows per tile.
    int64_t nb = dest.tileNb( 0 );  // assume fixed columns per tile.
    int64_t mt = dest.mt();         // Number of row tiles.
    int64_t nt = dest.nt();         // Number of column tiles.
    (void) nt;                      // Don't complain about no use.

    int64_t ii=0;                     // position within vector.
    int64_t tidx = col / nb;
    int64_t toff = col - tidx*nb;
    for (int64_t row_tile=0; row_tile < mt; ++row_tile) {
        if (dest.tileIsLocal( row_tile, tidx)) {
            auto tile = dest( row_tile, tidx );
            mb = dest.tileMb(row_tile);   // Get rows in THIS tile.
            blas::copy(mb, &v[ii], 1, &tile.at( 0, toff), 1);
            ii += mb;
        }
    }

    return;
}

//------------------------------------------------------------------------------
// stevx2_put_col_vector: Copy a single vector into a column of a tiled matrix.
template <typename scalar_t>
void stevx2_put_col_vector(
        scalar_t* v, Matrix<scalar_t>& dest, int col)
{
    int64_t mb;                     // rows per tile.
    int64_t nb = dest.tileNb( 0 );  // assume fixed columns per tile.
    int64_t mt = dest.mt();         // Number of row tiles.
    int64_t nt = dest.nt();         // Number of column tiles.
    (void) nt;                      // Don't complain about no use.

    int64_t ii=0;                     // position within vector.
    int64_t tidx = col / nb;
    int64_t toff = col - tidx*nb;
    for (int64_t row_tile=0; row_tile < mt; ++row_tile) {
        if (dest.tileIsLocal( row_tile, tidx)) {
            auto tile = dest( row_tile, tidx );
            mb = dest.tileMb(row_tile);   // Get rows in THIS tile.
            blas::copy(mb, &v[ii], 1, &tile.at( 0, toff), 1);
            ii += mb;
        }
    }

    return;
}

//------------------------------------------------------------------------------
// STELG: Symmetric Tridiagonal Eigenvalue Least Greatest (Min and Max).
// Finds the least and largest signed eigenvalues (not least magnitude).
// begins with bounds by Gerschgorin disc. These may be over or under
// estimated; Gerschgorin only ensures a disk will contain each. Thus we then
// use those with bisection to find the actual minimum and maximum eigenvalues.
// Note we could find the least magnitude eigenvalue by bisection between 0 and
// each extreme value.
// By Gerschgorin Circle Theorem;
// All Eigval(A) are \in [\lamda_{min}, \lambda_{max}].
// \lambda_{min} = min (i=0; i<n) diag[i]-|offd[i]| - |offd[i-1]|,
// \lambda_{max} = max (i=0; i<n) diag[i]+|offd[i]| + |offd[i-1]|,
// with offd[-1], offd[n] = 0.
// Indexes above are 0 relative.
// Although Gerschgorin is mentioned in ?larr?.f LAPACK files, it is coded
// inline there.
//------------------------------------------------------------------------------

template <typename scalar_t>
void stevx2_stelg(
    const scalar_t* diag,  const scalar_t* offd, const lapack_int n,
    scalar_t& Min, scalar_t& Max)
{
    int i;
    scalar_t test, testdi, testdim1, min=__DBL_MAX__, max=-__DBL_MAX__;

    for (i=0; i<n; i++) {
        if (i == 0)
            testdim1=0.;
        else
            testdim1=offd[i-1];

        if (i == (n-1))
            testdi=0;
        else
            testdi=offd[i];

        test=diag[i] - fabs(testdi) - fabs(testdim1);
        if (test < min)
            min=test;

        test=diag[i] + fabs(testdi) + fabs(testdim1);
        if (test > max)
            max=test;
    }


    scalar_t cp, minLB=min, minUB=max, maxLB=min, maxUB=max;
    // Within that range, find the actual minimum.
    for (;;) {
        cp = (minLB+minUB)*0.5;
        if (cp == minLB || cp == minUB) break;
        if (lapack::sturm(n, &diag[0], &offd[0], cp) == n) {
            minLB = cp;
        }
        else {
            minUB = cp;
        }
    }

    // Within that range, find the actual maximum.
    for (;;) {
        cp = (maxLB+maxUB)*0.5;
        if (cp == maxLB || cp == maxUB) break;
        if (lapack::sturm(n, &diag[0], &offd[0], cp) == n) {
            maxUB=cp;
        }
        else {
            maxLB=cp;
        }
    }

    Min = minLB;
    Max = maxUB;
}

//------------------------------------------------------------------------------
// STMV: Symmetric Tridiagonal Matrix Vector multiply.
// Matrix multiply; A * X = Y.
// A = [diag[0], offd[0],
//     [offd[0], diag[1], offd[1]
//     [      0, offd[1], diag[2], offd[2],
//     ...
//     [ 0...0                     offd[n-2], diag[n-1] ]
// LAPACK does not do just Y=A*X for a packed symmetric tridiagonal matrix.
// This routine is necessary to determine if eigenvectors should be swapped.
// This could be done by 3 daxpy, but more code and I think more confusing.
//------------------------------------------------------------------------------
template <typename scalar_t>
void stevx2_stmv(
    const scalar_t* diag, const scalar_t* offd, const int64_t n,
    std::vector< scalar_t >& X, std::vector< scalar_t >& Y)
{
    int64_t i;
    Y[0] = diag[0]*X[0] + offd[0]*X[1];
    Y[n-1] = offd[n-2]*X[n-2] + diag[n-1]*X[n-1];

    for (i = 1; i < (n-1); ++i) {
        Y[i] = offd[i-1]*X[i-1] + diag[i]*X[i] + offd[i]*X[i+1];
    }
}

//------------------------------------------------------------------------------
// STEPE: Symmetric Tridiagonal EigenPair Error.
// This routine is necessary to determine if eigenvectors should be swapped.
// eigenpair error: If A*v = u*v, then A*v-u*v should == 0. We compute the
// L_infinity norm of (A*v-u*v). (Max absolute value).
//------------------------------------------------------------------------------
template <typename scalar_t>
scalar_t stevx2_stepe(
    const scalar_t* diag,  const scalar_t* offd, int64_t n,
    scalar_t u, std::vector< scalar_t >& v)
{
    int i;
    scalar_t norm, temp;

    std::vector< scalar_t > y;
    y.resize(n);
    stevx2_stmv(diag, offd, n, v, y);      // y = A*v

    norm = fabs(y[0]-u*v[0]);              // init norm.
    for (i = 0; i < n; ++i) {
        temp = fabs(y[i] - u*v[i]);        // This should be zero.
        if (temp > norm)
            norm=temp;
    }

    return norm;
}

//------------------------------------------------------------------------------
// This is the main routine; slate_stevx2
// Arguments are described at the top of this source.
//------------------------------------------------------------------------------
template <typename scalar_t>
void stevx2(
    const lapack::Job jobtype, const lapack::Range range,
    const std::vector< scalar_t >& diag, const std::vector< scalar_t >& offd,
    scalar_t vl, scalar_t vu, const int64_t il, const int64_t iu,
    std::vector< scalar_t >& eig_val, std::vector< int64_t >& eig_mult,
    Matrix< scalar_t >& eig_vec, MPI_Comm mpi_comm)
{
    lapack_int n;
    int i, max_threads;
    // workspaces array.
    std::vector< stevx2_stein_array_t<scalar_t> > stein_arrays;

    // Check input arguments.
    if (jobtype != lapack::Job::Vec
        && jobtype != lapack::Job::NoVec) {
            slate_error("Arg 1, illegal value of jobtype");
    }

/// @param[in] range enum: lapack::Range::Value use vl, vu for range [vl, vu).
/// lapack::Range::Index use il, iu for range [il, iu]. 1-relative indices;
/// 1..n.
    if (range != lapack::Range::Value
        && range != lapack::Range::Index) {
        slate_error("Arg 2, illegal value of range");
    }

    if (diag.size() < 2) {
        slate_error("arg 3, diag must be at least 2 elements");
    }

    if (offd.size() < (diag.size()-1)) {
        slate_error("arg 4, offd must be at least diag.size()-1");
    }

    n = diag.size();

    if (range == lapack::Range::Value
        && vu <= vl ) {
        slate_error("args 5 & 6, vu must be > vl");
    }

    if (range == lapack::Range::Index) {
        if (il < 1
            || il > n) {
             slate_error("arg 7 illegal value of il");
        } else
        {
            if (iu < il
                || iu > n) {
                slate_error("arg 8, illegal value of iu");
            }
        }
    }

    max_threads = omp_get_max_threads();

    // Ensure we have a workspace per thread.
    if (jobtype == lapack::Job::Vec) {
        stein_arrays.resize( max_threads );
    }

    scalar_t glob_min_eval, glob_max_eval;

    stevx2_control_t<scalar_t> control;
    control.n = n;
    control.diag = &diag[0];
    control.offd = &offd[0];
    control.jobtype = jobtype;
    control.range = range;
    control.il = il;
    control.iu = iu;
    control.stein_arrays = &stein_arrays[0];

    // Find actual least and greatest eigenvalues.
    stevx2_stelg<scalar_t>(control.diag, control.offd, control.n,
                           glob_min_eval, glob_max_eval);

    int64_t ev_less_than_vl=0, ev_less_than_vu=n, n_eig_vals=0;
    if (range == lapack::Range::Value) {
        // We don't call Sturm if we already know the answer.
        if (vl >= glob_min_eval)
            ev_less_than_vl = lapack::sturm(n, &diag[0], &offd[0], vl);
        else
            vl = glob_min_eval; // optimize for computing step size.

        if (vu <= glob_max_eval)
            ev_less_than_vu = lapack::sturm(n, &diag[0], &offd[0], vu);
        else
            vu = nexttoward(glob_max_eval, __DBL_MAX__);

        // Compute the number of eigenvalues in [vl, vu).
        n_eig_vals = (ev_less_than_vu - ev_less_than_vl);

         control.base_idx = ev_less_than_vl;
    }
    else {
        // lapack::Range::Index. iu, il already vetted by code above.
        n_eig_vals = iu+1-il; // Index range is inclusive.

        // We still must bisect by values to discover eigenvalues, though.
        vl = glob_min_eval;
        // nextoward is to ensure we include globMaxVal as a possibility.
        vu = nexttoward(glob_max_eval, __DBL_MAX__);
        control.base_idx = 0; // There are zero eigenvalues less than vl.
    }

    std::vector< scalar_t > pvec;

    eig_val.resize(n);
    eig_mult.resize(n);
    if (jobtype == lapack::Job::Vec) {
        pvec.resize(control.n * n_eig_vals);  // Make space.
    }

    // Finish set up of Control.
    control.pval = &eig_val[0];
    control.pmul = &eig_mult[0];
    control.pvec = &pvec[0];
    int64_t int_one=1;

    // We launch the root task: The full range to subdivide.
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
                stevx2_bisection(&control, vl, vu, -int_one, -int_one,
                                 n_eig_vals);
        }
    }

    // Now, all the eigenvalues should have unit eigenvectors in the array
    // ~control.pvec.  We don't need to sort that, but we do want to compress
    // it; in case of multiplicity.
    // NOTE: An alternative exists here, if we can figure out how to make
    // inverse iteration (STEIN) produce different eigenvectors for the same
    // eigenvalue. Currently, STEIN starts with a random vector, and for the
    // same eigenvalue always produces the same vector. However, with a custom
    // version of STEIN, Mark Gates suggests using Gram Schmidt or CGS2 to
    // orthogonalize the random vector against previous eigenvectors for the
    // same eigenvalue, before beginning iteration, so perhaps it will converge
    // to a different eigenvector.

    int vectors_found = 0;
    for (i=0; i<n_eig_vals; ++i) {
        if (eig_mult[i] > 0) {
            ++vectors_found;
        }
    }

    // Compress array in case vectorsFound < nEigVals (due to multiplicities).
    // Note pMul[] init to zeros: if still zero, that represents a vector not
    // filled in due to multiplicity.
    if (vectors_found < n_eig_vals) {
        int j=0;
        for (i=0; i<n_eig_vals; ++i) {
            if (eig_mult[i] > 0) {          // If this is NOT a multiplicity,
                eig_mult[j] = eig_mult[i];  // copy to next open slot j.
                eig_val[j] = eig_val[i];
                if (control.jobtype == lapack::Job::Vec) {
                    if (j != i) {
                        memcpy(&pvec[j*control.n], &pvec[i*control.n],
                               control.n*sizeof(scalar_t));
                    }
                }

                ++j;
            } // end if we found a non-multiplicity eigenvalue.
        }
    } // end if compression is needed.

    // resize matrices based on what we have found.
    eig_val.resize(vectors_found);
    eig_mult.resize(vectors_found);
    pvec.resize(control.n * vectors_found);

    // perform QR factorization, remember the descriptor.
    int64_t p = 1, q = 1, nb = 256;

    slate::Options const opts =  {
        {slate::Option::Target, slate::Target::Host},
        {slate::Option::MaxPanelThreads, max_threads}
    };

    slate::TriangularFactors<scalar_t> tri_factors;

    // Inefficiency Here.
    // This is likely not ideal. I need to return eig_vec[] to the caller, but
    // I cannot use eig_vec.fromScaLAPACK() because the pvec[] array will be
    // de-allocated upon return. So I need to duplicate the size of the final
    // matrix, and copy pvec[] into it. Using twice as much storage as I'd
    // like.  I can't just create it with n_eig_vals before starting bisection;
    // that would work if compression were not necessary; but it is necessary;
    // because we can get two or more theoretically distinct eigenvalues within
    // the ULP of each other, thus represented by the same floating point
    // number. A workable alternative would be to NOT call stein() during the
    // bisection (as we currently do), just find all the eigenvalues first.
    // Then remove any duplicates for multiplicity, so we just compress the
    // eigenvalues vector, and THEN we can create eig_vec in its final size,
    // and use a different omp task to parallelize finding all the eigenvectors
    // with stein, for each of those values, and copy the vectors directly into
    // eig_vec[] using the stevx2_put_col_vector() function. (Stein will only
    // return a contiguous vector, but we only need one vector per thread for
    // calling.) That is a bigger change than I can test in the time I have
    // left on this project.

    eig_vec = slate::Matrix<scalar_t>(
        control.n, vectors_found, nb, p, q, mpi_comm);
    eig_vec.insertLocalTiles();

    for (i = 0; i < vectors_found; ++i) {
        stevx2_put_col_vector(&pvec[control.n * i], eig_vec, i);
    }

    // We need to make Result same size as pVec.
    // rows=control.n, columns=vectors_found, tile-size mb=nb=256, p=q=1.
    auto ident = slate::Matrix<scalar_t>(
        control.n, vectors_found, nb, p, q, mpi_comm);
    ident.insertLocalTiles();
    scalar_t zero=0.0;
    scalar_t one= 1.0;
    set(zero, one, ident);

    slate::qr_factor(eig_vec, tri_factors);

    // Extract just the Q of the QR, in normal form. We do this with unmqr; to
    // multiply the factorized Q by the Identity matrix we just built. ungqr
    // could do this without the extra space of ident, in half the flops, but
    // SLATE does not have it yet.

    slate::qr_multiply_by_q(
        slate::Side::Left, slate::Op::NoTrans, eig_vec, tri_factors, ident);

    slate::copy(ident, eig_vec);    // This is our return matrix.

    //--------------------------------------------------------------------------
    // When eigenvalue are crowded, it is possible that after orthogonalizing
    // vectors, it can be better to swap neighboring eigenvectors. We just
    // test all the pairs; basically ||(A*V-e*V)||_max is the error.  if BOTH
    // vectors in a pair have less error by being swapped, we swap them.
    //--------------------------------------------------------------------------
    if (jobtype == lapack::Job::Vec) {
        std::vector<scalar_t> y( control.n );
        std::vector<scalar_t> test( 4 );            // Four test results.
        std::vector<scalar_t> vec1( control.n );
        std::vector<scalar_t> vec2( control.n );
        stevx2_get_col_vector(eig_vec, vec2, 0);    // Copy vec[0] to temp vec2.

        // Now on each iteration, we move the current vec2 to vec1, and
        // load the next vec2. Our pairs are 0:1, 1:2, 2:3, etc.

        for (i = 1; i < vectors_found-1; ++i) {
            std::swap(vec1, vec2);
            stevx2_get_col_vector(eig_vec, vec2, i);

            // Compute 4 tests, one for each combination.
            test[0] = stevx2_stepe(
                &control.diag[0], &control.offd[0], control.n, eig_val[i], vec1);
            test[1] = stevx2_stepe(
                &control.diag[0], &control.offd[0], control.n, eig_val[i+1], vec2);
            test[2] = stevx2_stepe(
                &control.diag[0], &control.offd[0], control.n, eig_val[i], vec2);
            test[3] = stevx2_stepe(
                &control.diag[0], &control.offd[0], control.n, eig_val[i+1], vec1);

            if ( (test[2] < test[0])        // val1 & vec2 beats val1 & vec1.
                && (test[3] < test[1]) ) {  // val2 & vec1 beats val2 & vec2.
                // Copy vec2 to vec1 original position.
                stevx2_put_col_vector(vec2, eig_vec, (i-1) );

                // Copy vec1 to vec2 original position.
                stevx2_put_col_vector(vec1, eig_vec, i);

                // We have now swapped eig_vec[i-1] and eig_vec[i]. At the top
                // of the loop we are going to swap and then load the new
                // vector into vec2, but because of the swap we really need
                // vec1 to stay as it is.  So we swap here, so the vector swap
                // at the top will reverse this with another swap.
                std::swap(vec1, vec2);
            } // end this swap.
        } // end all swapping.
    } // end if we are processing eigenvectors at all.

    return;
}

template
void stevx2_stelg<float>(
    const float* diag,  const float* offd, const lapack_int n,
    float& Min, float& Max);

template
void stevx2_stelg<double>(
    const double* diag,  const double* offd, const lapack_int n,
    double& Min, double& Max);

template
void stevx2_stmv<float>(
    const float* diag, const float* offd, const int64_t n,
    std::vector<float>& X, std::vector<float>& Y);

template
void stevx2_stmv<double>(
    const double* diag, const double* offd, const int64_t n,
    std::vector<double>& X, std::vector<double>& Y);

template
float stevx2_stepe<float>(
    const float* diag,  const float* offd, const int64_t n,
    const float u, std::vector<float>& v);

template
double stevx2_stepe<double>(
    const double* diag,  const double* offd, const int64_t,
    const double u, std::vector<double>& v);

template
void stevx2<float>(
    const lapack::Job jobtype, const lapack::Range range,
    const std::vector< float >& diag, const std::vector< float >& offd,
    const float vl, const float vu, const int64_t il, const int64_t iu,
    std::vector< float >& eig_val, std::vector< int64_t >& eig_mult,
    Matrix< float >& eig_vec, MPI_Comm mpi_comm);

template
void stevx2<double>(
    const lapack::Job jobtype, const lapack::Range range,
    const std::vector< double >& diag, const std::vector< double >& offd,
    const double vl, const double vu, const int64_t il, const int64_t iu,
    std::vector< double >& eig_val, std::vector< int64_t >& eig_mult,
    Matrix< double >& eig_vec, MPI_Comm mpi_comm);

} // namespace slate.
