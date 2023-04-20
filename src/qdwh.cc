// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "internal/internal.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Distributed parallel polar decomposition based on QDWH algorithm.
///
/// Computes a polar decomposition of an m-by-n matrix $A$.
/// The factorization has the form
/// \[
///     A = UH,
/// \]
/// where $U$ is a matrix with orthonormal columns and $H$ is a Hermition
/// matrix.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the m-by-n matrix $A$.
///     On exit, if return value = 0, the orthogonal polar factor $U$
///     from the decomposition $A = U H$.
///
/// @param[out] H
///     On exit, if return value = 0, the hermitian polar factor matrix $H$
///     from the decomposition $A = U H$.
///
/// @param[out] itqr
///     The number of the QR-based iterations.
///
/// @param[out] itpo
///     The number of the Cholesky-based iterations.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Lookahead:
///       Number of panels to overlap with matrix updates.
///       lookahead >= 0. Default 1.
///     - Option::InnerBlocking:
///       Inner blocking to use for panel. Default 16.
///     - Option::MaxPanelThreads:
///       Number of threads to use for panel. Default omp_get_max_threads()/2.
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// @ingroup qdwh
///
template <typename scalar_t>
void qdwh(
          Matrix<scalar_t>& A,
          Matrix<scalar_t>& H,
          int& itqr, int& itpo,
          Options const& opts)
{
    // todo:
    // optimizations:
    // 1. reuse Q from first geqrf
    // 2. avoid rounding m if m is not divisible by nb, because then geqrf/unmqr
    // have extra zero rows

    using real_t = blas::real_type<scalar_t>;
    using blas::real;

    // Constants
    const scalar_t zero = 0.0;
    const scalar_t one = 1.0;
    const real_t r_one = 1.0;

    // Options
    Target target = get_option( opts, Option::Target, Target::HostTask );

    int64_t mt = A.mt();
    int64_t nt = A.nt();
    int64_t nb   = A.tileMb(0);

    int64_t m  = A.m();
    int64_t n  = A.n();

    if (m < n) {
        return;
    }

    real_t eps  = std::numeric_limits<real_t>::epsilon();
    real_t tol1 = 5. * eps;
    real_t tol3 = pow(tol1, r_one/real_t(3.));

    int itconv, it;
    real_t normR;
    real_t conv = 10.0;
    scalar_t alpha, beta;
    double L2, sqd, a1, a, b, c, dd;
    real_t Li, Liconv;

    // The QR-based iterations requires QR[A; Id],
    // W = [A; Id], where size of A is mxn, size of Id is nxn
    // To avoid having a clean up tile in the middle of the W matrix,
    // we round up the number of A rows (m),
    // so that size(W) = m_roundup + n
    int64_t m_roundup = roundup( m, nb );
    int64_t m_W   = m_roundup + n;
    int64_t mt_W = mt + nt;

    int nprow, npcol;
    int myrow, mycol;
    GridOrder order;
    A.gridinfo(&order, &nprow, &npcol, &myrow, &mycol);

    // allocate m_W*n work space required for the qr-based iterations QR([A;Id])
    slate::Matrix<scalar_t> W(m_W, n, nb, nprow, npcol, A.mpiComm());
    slate::Matrix<scalar_t> Q(m_W, n, nb, nprow, npcol, A.mpiComm());
    slate::TriangularFactors<scalar_t> T;

    slate::Matrix<scalar_t> Acpy;
    if (m == n) {
        // if A is square, we can use H to save a copy of A
        // this is will not work if H is stored as Hermitian matrix
        Acpy = H;
    }
    else {
        Acpy = slate::Matrix<scalar_t>(m, n, nb, nprow, npcol, A.mpiComm());
        Acpy.insertLocalTiles(target);
    }

    W.insertLocalTiles(target);
    Q.insertLocalTiles(target);

    // To compute QR[A; Id], allocate W and Q of size m_Wxn
    // First mt tiles of W contains A matrix in [0:m-1, 0:n-1]
    // W1 have more rows than A, since we round up the number of rows of W1
    auto W1 = W.sub(0, mt-1, 0, nt-1);
    auto Q1 = Q.sub(0, mt-1, 0, nt-1);
    // W10 have A matrix
    auto W10 = W1.slice(0, m-1, 0, n-1);
    auto Q10 = Q1.slice(0, m-1, 0, n-1);
    // W11 is the last tile which has more rows in case we round up m
    auto W11 = W.sub(mt-1, mt-1, 0, nt-1);

    // Second nt tiles of W, contain the identity matrix
    auto W2 = W.sub(mt, mt_W-1, 0, nt-1);
    auto Q2 = Q.sub(mt, mt_W-1, 0, nt-1);

    // backup A in Acpy to compute H
    slate::copy(A, Acpy, opts);

    // two norm estimation (largest singular value of A)
    real_t norm_est = norm2est( A, opts);
    alpha = 1.0;

    // scale the original matrix A to form A0 which is the initial matrix
    // for the first iteration
    scale(r_one, norm_est, A, opts);

    // Estimate the condition number using QR
    // This Q factor can be used in the first QR-based iteration
    // todo: use the Q factor in the first QR-based iteration
    slate::copy(A, W10, opts);
    slate::geqrf(W10, T, opts);
    //auto R  = TrapezoidMatrix<scalar_t>(
    //    Uplo::Upper, slate::Diag::NonUnit, W10 );
    auto R1 =  W10.slice(0, n-1, 0, n-1);
    auto R  = TriangularMatrix<scalar_t>(
            Uplo::Upper, slate::Diag::NonUnit, R1 );
    normR = norm(slate::Norm::One, R, opts);
    slate::trcondest(slate::Norm::One, R, &Li, opts);
    real_t smin_est = normR*Li;
    Li = smin_est / sqrt(n);

    //slate::trcondest(slate::Norm::One, R, &Li, opts);
    // *flops += FLOPS_DGEQRF( M, N );

    // Compute the number of iterations to converge
    itconv = 0; Liconv = Li;
    while (itconv == 0 || fabs(1-Liconv) > tol1) {
        // To find the minimum number of iterations to converge.
        // itconv = number of iterations needed until |Li - 1| < tol1
        // This should have converged in less than 50 iterations
        if (itconv > 100) {
            slate_error("Failed to converge.");
        }
        itconv++;

        L2  =  double(Liconv) * double(Liconv);
        dd  = std::pow( real_t(4.0) * ( r_one - L2 ), r_one / real_t(3.0) ) *
            std::pow( L2, real_t(-2.0) / real_t(3.0) );
        sqd = sqrt( r_one + real_t(dd) );
        a1  = sqd + sqrt( real_t(8.0) - real_t(4.0) * real_t(dd) +
              real_t(8.0) * ( real_t(2.0) - L2 ) / ( L2 * sqd ) ) / real_t(2.0);
        a   = real(a1);
        b   = ( a - r_one ) * ( a - r_one ) / real_t(4.0);
        c   = a + b - r_one;
        // Update Liconv
        Liconv  = Liconv * real_t(( a + b * L2 ) / ( r_one + c * L2 ));
    }

    it = 0;
    while (conv > tol3 || it < itconv) {
        // This should have converged in less than 50 iterations
        if (it > 100) {
            slate_error("Failed to converge.");
        }
        it++;

        // Compute parameters L,a,b,c
        L2  = double(Li) * double(Li);
        dd  = std::pow( real_t(4.0) * ( r_one - L2 ), r_one / real_t(3.0) ) *
            std::pow( L2, real_t(-2.0) / real_t(3.0) );
        sqd = sqrt( r_one + real_t(dd) );
        a1  = sqd + sqrt( real_t(8.0) - real_t(4.0) * real_t(dd) +
              real_t(8.0) * ( real_t(2.0) - L2 ) / ( L2 * sqd ) ) / real_t(2.0);
        a   = real(a1);
        b   = ( a - r_one ) * ( a - r_one ) / real_t(4.0);
        c   = a + b - r_one;
        // Update Li
        Li  = Li * real_t(( a + b * L2 ) / ( r_one + c * L2 ));

        if (c > 100.) {
            // Generate the matrix W = [ W1 ] = [ sqrt(c) * A ]
            //                         [ W2 ] = [ Id          ]

            // Factorize W = QR, and generate the associated Q
            alpha = scalar_t(sqrt(c));
            set(zero, zero, W11, opts);
            add(alpha, A, zero, W10, opts);
            set(zero, one, W2, opts);

            // Call a variant of geqrf and unmqr that avoid the tiles of zeros,
            // which are the tiles below the diagonal of W2 (identity)
            geqrf_qdwh_full(W, T, opts);

            set(zero, one, Q, opts);
            unmqr_qdwh_full(slate::Side::Left, slate::Op::NoTrans, W, T, Q, opts);

            // A = ( (a-b/c)/sqrt(c) ) * Q1 * Q2' + (b/c) * A
            auto Q2T = conj_transpose(Q2);
            alpha = scalar_t( (a-b/c)/sqrt(c) );
            beta  = scalar_t( b / c );
            // Save a copy of A to check the convergence of QDWH
            if (it >= itconv ) {
                slate::copy(A, W10, opts);
            }
            gemm(alpha, Q10, Q2T, beta, A, opts);

            itqr += 1;

            //facto = 0;
            // Main flops used in this step/
            //flops_dgeqrf = FLOPS_DGEQRF( 2*m, n );
            //flops_dorgqr = FLOPS_DORGQR( 2*m, n, n );
            //flops_dgemm  = FLOPS_DGEMM( m, n, n );
            //flops += flops_dgeqrf + flops_dorgqr + flops_dgemm;
        }
        else {
            // A = (b/c) * A + (a-b/c) * ( linsolve( C, linsolve( C, A') ) )';
            // Save a copy of A to check the convergence of QDWH
            if (it >= itconv) {
                slate::copy(A, W10, opts);
            }

            // Save a copy of A
            slate::copy(A, Q10, opts);

            // Compute c * A' * A + I
            auto AT = conj_transpose(Q10);
            set(zero, one, W2, opts);
            gemm(scalar_t(c), AT, A, one, W2, opts);

            // Solve R x = AT
            auto U = slate::HermitianMatrix<scalar_t>(
                    slate::Uplo::Upper, W2 );
            posv(U, AT, opts);

            // Update A = (b/c) * A + (a-b/c) * ( linsolve( C, linsolve( C, A') ) )';
            alpha = (a-b/c); beta = (b/c);
            auto AT2 = conj_transpose(AT);
            add(alpha, AT2, beta, A, opts);

            itpo += 1;

            //facto = 1;
            // Main flops used in this step
            //flops_dgemm  = FLOPS_DGEMM( m, n, n );
            //flops_dpotrf = FLOPS_DPOTRF( m );
            //flops_dtrsm  = FLOPS_DTRSM( 'L', m, n );
            //flops += flops_dgemm + flops_dpotrf + 2. * flops_dtrsm;
        }

        // Check if it converge, compute the norm of matrix A - W1
        conv = 10.0;
        if (it >= itconv) {
            add(one, A, -one, W10, opts);
            conv = norm(slate::Norm::Fro, W10, opts);
        }
    }

    if (A.mpiRank() == 0) {
        printf("%-7s  %-7s\n", "QR", "PO");
        printf("%-7d  %-7d\n", itqr, itpo);
    }

    if (itqr + itpo > 6) {
        printf("\nConverged after %d. Check what is the issue, "
                   "because QDWH needs <= 6 iterations.\n",
                   itqr+itpo);
    }

    // A = U*H ==> H = U'*A ==> H = 0.5*(H'+H)
    auto AT = conj_transpose(A);
    if (m == n) {
        auto W10_n = W1.slice(0, n-1, 0, n-1);
        gemm(one, AT, Acpy, zero, W10_n, opts);
        slate::copy(W10_n, H, opts);
    }
    else {
        gemm(one, AT, Acpy, zero, H, opts);
    }

    // todo: try something like her2k to compute H
    //her2k(one, A, W10, rzero, H, opts);
    //auto AL = HermitianMatrix<scalar_t>(
    //        slate::Uplo::Lower, H );
    //slate::copy(AL, H, opts);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void qdwh<float>(
    Matrix<float>& A,
    Matrix<float>& H,
    int& itqr, int& itpo,
    Options const& opts);

template
void qdwh<double>(
    Matrix<double>& A,
    Matrix<double>& H,
    int& itqr, int& itpo,
    Options const& opts);

template
void qdwh< std::complex<float> >(
    Matrix< std::complex<float> >& A,
    Matrix< std::complex<float> >& H,
    int& itqr, int& itpo,
    Options const& opts);

template
void qdwh< std::complex<double> >(
    Matrix< std::complex<double> >& A,
    Matrix< std::complex<double> >& H,
    int& itqr, int& itpo,
    Options const& opts);

} // namespace slate
