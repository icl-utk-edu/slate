// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "internal/internal.hh"

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// Distributed parallel polar decomposition based on QDWH algorithm.
/// Generic implementation for any target.
/// @ingroup qdwh_specialization
///
template <Target target, typename scalar_t>
void qdwh(
          Matrix<scalar_t>& A,
          Matrix<scalar_t>& B,
          int& itqr, int& itpo,
          Options const& opts)
{
    using real_t = blas::real_type<scalar_t>;
    using blas::real;

    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );
    int64_t max_panel_threads  = std::max(omp_get_max_threads()/2, 1);

    max_panel_threads = get_option<int64_t>( opts, Option::MaxPanelThreads, max_panel_threads );

    real_t eps  = 0.5 * std::numeric_limits<real_t>::epsilon();
    real_t tol1 = 5. * eps;
    real_t tol3 = pow(tol1, 1./3.);

    bool optqr = 1;
    int itconv, it;

    real_t L2, sqd, dd, a1, a, b, c;
    real_t Li, Liconv;
    real_t conv = 100.;
    scalar_t zero = 0.0, one = 1.0;
    real_t rzero = 0.0;

    scalar_t alpha, beta;

    real_t normA;
    real_t norminvR;

    int64_t mt = A.mt();
    int64_t nt = A.nt();

    int64_t m  = A.m();
    int64_t n  = A.n();

    if (m < n) {
        return;
    }

    int64_t nb   = A.tileMb(0);

    // The QR-based iterations requires QR[A; Id],
    // W = [A; Id], where size of A is mxn, size of Id is nxn
    // To avoid having a clean up tile in the middle of the W matrix,
    // we round up the number of A rows (m),
    // so that size(W) = m_roundup + n
    int64_t m_roundup =  ((m + nb - 1) / nb ) * nb;
    int64_t m_W   = m_roundup + n;
    int64_t mt_W = mt + nt;

    int nprow, npcol;
    int myrow, mycol;
    GridOrder order;
    A.gridinfo(&order, &nprow, &npcol, &myrow, &mycol);

    // allocate m_W*n work space required for the qr-based iterations QR([A;Id])
    slate::Matrix<scalar_t> W(m_W, n, nb, nprow, npcol, MPI_COMM_WORLD);
    slate::Matrix<scalar_t> Q(m_W, n, nb, nprow, npcol, MPI_COMM_WORLD);
    slate::TriangularFactors<scalar_t> T;

    // todo: anyway to avoid Acpy allocation?
    slate::Matrix<scalar_t> Acpy;
    Acpy = slate::Matrix<scalar_t>(m, n, nb, nprow, npcol, MPI_COMM_WORLD);
    Acpy.insertLocalTiles(target);

    //if (target == Target::Devices) {
        //const int64_t batch_size_zero = 0; // use default batch size
        //const int64_t num_queues = 3 + lookahead;
        // todo: I have workspace here and in other call as geqrf_qdwh_full
        // todo: in geqrf, it should use the allocated ones, it should not
        // keep allocating
        // todo: check traces to confirm
        // or in printf in blaspp/device_malloc and here
        // todo: is it enough to allocate for one matrix??
        // todo: we don't need these here because each slate call will do it.
        // todo: check if after any call it deallocate, for example after gemm, does it throw away the gpu allocation?
        //A.allocateBatchArrays(batch_size_zero, num_queues);
        //A.reserveDeviceWorkspace();
        //W.allocateBatchArrays(batch_size_zero, num_queues);
        //W.reserveDeviceWorkspace();
        //Q.allocateBatchArrays(batch_size_zero, num_queues);
        //Q.reserveDeviceWorkspace();
        //Acpy.allocateBatchArrays(batch_size_zero, num_queues);
        //Acpy.reserveDeviceWorkspace();
    //}

    // todo: do this on GPUs
    W.insertLocalTiles(target);
    Q.insertLocalTiles(target);

    // To compute QR[A; Id]
    // First mt tiles of W contains A matrix in [0:m-1, 0:n-1]
    // W1 have more rows than A, since we round up the number of rows of W1
    // W10 have A matrix
    auto W1 = W.sub(0, mt-1, 0, nt-1);
    auto W10 = W1.slice(0, m-1, 0, n-1);

    // Second nt tiles of W, contain the identity matrix
    auto W2 = W.sub(mt, mt_W-1, 0, nt-1);

    auto Q1 = Q.sub(0, mt-1, 0, nt-1);
    auto Q10 = Q1.slice(0, m-1, 0, n-1);

    auto Q2 = Q.sub(mt, mt_W-1, 0, nt-1);


    if (m_roundup != m) {
        // W11 and Q11 is the extra padded rows [m:m_round_up, 0:n-1]
        auto W11 = W1.slice(m, m_roundup-1, 0, n-1);
        auto Q11 = Q1.slice(m, m_roundup-1, 0, n-1);
    }

    // backup A in Acpy to compute H
    copy(A, Acpy);

    // two norm estimation (largest singular value of A)
    real_t norm_est = norm2est( A, opts);
    alpha = 1.0;
    norm_est = 1.;

    // scale the original matrix A to form A0 of the iterative loop
    add(alpha/scalar_t(norm_est), Acpy, zero, A, opts);

    // Calculate Li: reciprocal of condition number estimation
    // Either 1) use LU followed by gecon
    // Or     2) QR followed by trtri
    // If used the QR, use the Q factor in the first QR-based iteration
    normA = norm(slate::Norm::One, A, opts);
    if (optqr) {
        // Estimate the condition number using QR
        // This Q factor can be used in the first QR-based iteration
        copy(A, W10);
        // todo: the QR used to estimate the condition number can be used in the first QR iteration
        // todo: put bound of 1-norm est compared to 2-norm
        slate::geqrf(W1, T, opts);
        //auto R  = TrapezoidMatrix<scalar_t>(
        //    Uplo::Upper, slate::Diag::NonUnit, W10 );
        auto R1 =  W10.slice(0, n-1, 0, n-1);
        auto R  = TriangularMatrix<scalar_t>(
            Uplo::Upper, slate::Diag::NonUnit, R1 );
        auto Rh = HermitianMatrix<scalar_t>(
            Uplo::Upper, R1 );
        // todo: cheaper to do triangular solve than calling trtri
        trtri(R, opts);
        norminvR = norm(slate::Norm::One, Rh, opts);
        Li = (1.0 / norminvR) / normA;
        Li = norm_est / 1.1 * Li;
        // *flops += FLOPS_DGEQRF( M, N )
        //       + FLOPS_DTRTRI( N );
    }
    else {
        // todo
        // Estimate the condition number using LU
    }

    // Compute the number of iterations to converge
    itconv = 0; Liconv = Li;
    while (itconv == 0 || fabs(1-Liconv) > tol1) {
        // To find the minimum number of iterations to converge.
        // itconv = number of iterations needed until |Li - 1| < tol1
        // This should have converged in less than 50 iterations
        if (itconv > 100) {
            exit(-1);
            break;
        }
        itconv++;

        L2  =  Liconv * Liconv;
        dd  = pow( 4.0 * (1.0 - L2 ) / (L2 * L2), 1.0/3.0 );
        sqd = sqrt(1.0 + dd);
        a1  = sqd + sqrt( 8.0 - 4.0 * dd + 8.0 * (2.0 - L2) / (L2 * sqd) ) / 2.0;
        a   = real(a1);
        b   = (a - 1.0) * (a - 1.0) / 4.0;
        c   = a + b - 1.0;
        // Update Liconv
        Liconv  = Liconv * real_t((a + b * L2) / (1.0 + c * L2));
    }

    it = 0;
    while (conv > tol3 || it < itconv) {
        // This should have converged in less than 50 iterations
        if (it > 100) {
            exit(-1);
            break;
        }
        it++;

        // Compute parameters L,a,b,c
        //L2  = double (Li * Li);
        L2  = Li * Li;
        dd  = pow( 4.0 * (1.0 - L2 ) / (L2 * L2), 1.0/3.0 );
        sqd = sqrt(1.0 + dd);
        a1  = sqd + sqrt( 8.0 - 4.0 * dd + 8.0 * (2.0 - L2) / (L2 * sqd) ) / 2.0;
        a   = real(a1);
        b   = (a - 1.0) * (a - 1.0) / 4.0;
        c   = a + b - 1.0;
        // Update Li
        Li  = Li * real_t ((a + b * L2) / (1.0 + c * L2));

        if (real(c) > 100.) {
            //int64_t doqr = !optqr || it > 1;

            // Generate the matrix B = [ B1 ] = [ sqrt(c) * U ]
            //                         [ B2 ] = [ Id          ]
            alpha = scalar_t(sqrt(c));
            set(zero, one, W2, opts);

            //if( doqr ) {
            //    geadd(alpha, A, zero, W10, opts);
            //    geqrf_qdwh_full(W, T1, opts);
            //}
            //else {
            //    geadd(alpha, W1, zero, W10, opts);
            //    geqrf_qdwh_full2(W, T1, opts);
            //}

            // Factorize B = QR, and generate the associated Q
            // todo: replace following add by copy and scale, but why scale only scale by a real numbers?
            add(alpha, A, zero, W10, opts);
            //copy(A, W1);
            //scale(alpha, one, W1, opts);

            //geqrf(W, T, opts); // naive impl
            geqrf_qdwh_full(W, T, opts);

            set(zero, one, Q, opts);
            //unmqr(slate::Side::Left, slate::Op::NoTrans, W, T, Q, opts); //naive impl
            unmqr_qdwh_full(slate::Side::Left, slate::Op::NoTrans, W, T, Q, opts);

            // A = ( (a-b/c)/sqrt(c) ) * Q1 * Q2' + (b/c) * A
            auto Q2T = conj_transpose(Q2);
            alpha = scalar_t( (a-b/c)/sqrt(c) );
            beta  = scalar_t( b / c );
            // Copy U into C to check the convergence of QDWH
            if (it >= itconv ) {
                copy(A, W10);
            }
            gemm(alpha, Q10, Q2T, beta, A, opts);

            // Main flops used in this step/
            //flops_dgeqrf = FLOPS_DGEQRF( 2*m, n );
            //flops_dorgqr = FLOPS_DORGQR( 2*m, n, n );
            //flops_dgemm  = FLOPS_DGEMM( m, n, n );
            //flops += flops_dgeqrf + flops_dorgqr + flops_dgemm;

            itqr += 1;
            //facto = 0;
        }
        else {
            // Copy A into H to check the convergence of QDWH
            if (it >= itconv) {
                copy(A, W10);
            }

            // Make Q1 into an identity matrix
            set(zero, one, W2, opts);

            // Compute Q1 = c * A' * A + I
            copy(A, Q10);
            auto AT = conj_transpose(Q10);
            gemm(scalar_t(c), AT, A, one, W2, opts);

            // Solve R x = AT
            auto R = slate::HermitianMatrix<scalar_t>(
                    slate::Uplo::Upper, W2 );
            posv(R, AT, opts);

            // Update A =  (a-b/c) * Q1T' + (b/c) * A
            alpha = (a-b/c); beta = (b/c);
            AT = conj_transpose(AT);
            add(scalar_t(alpha), AT, scalar_t(beta), A, opts);

            // Main flops used in this step
            //flops_dgemm  = FLOPS_DGEMM( m, n, n );
            //flops_dpotrf = FLOPS_DPOTRF( m );
            //flops_dtrsm  = FLOPS_DTRSM( 'L', m, n );
            //flops += flops_dgemm + flops_dpotrf + 2. * flops_dtrsm;

            itpo += 1;
            //facto = 1;
        }

        // Check if it converge, compute the norm of matrix U - B1
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

    // A = U*H ==> H = U'*A ==> H = 0.5*(H'+H)
    auto AT = conj_transpose(A);
    gemm(one, AT, Acpy, zero, B, opts);
    // todo: try something like her2k to compute H
    //her2k(one, A, W10, rzero, H, opts);
    //auto AL = HermitianMatrix<scalar_t>(
    //        slate::Uplo::Lower, B );
    //slate::copy(AL, H, opts);

    A.releaseWorkspace();
    W.releaseWorkspace();
    Q.releaseWorkspace();
}

} // namespace impl

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
/// Complexity (in real):
/// - for $m \ge n$, $\approx 2 m n^{2} - \frac{2}{3} n^{3}$ flops;
/// - for $m \le n$, $\approx 2 m^{2} n - \frac{2}{3} m^{3}$ flops;
/// - for $m = n$,   $\approx \frac{4}{3} n^{3}$ flops.
/// .
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
/// @ingroup geqrf_computational
///
template <typename scalar_t>
void qdwh(
          Matrix<scalar_t>& A,
          Matrix<scalar_t>& B,
          int& itqr, int& itpo,
          Options const& opts)
{
    using internal::TargetType;
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
        case Target::HostNest:
        case Target::HostBatch:
            impl::qdwh<Target::HostTask>( A, B, itqr, itpo, opts );
            break;
        case Target::Devices:
            impl::qdwh<Target::Devices>( A, B, itqr, itpo, opts );
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void qdwh<float>(
    Matrix<float>& A,
    Matrix<float>& B,
    int& itqr, int& itpo,
    Options const& opts);

template
void qdwh<double>(
    Matrix<double>& A,
    Matrix<double>& B,
    int& itqr, int& itpo,
    Options const& opts);

template
void qdwh< std::complex<float> >(
    Matrix< std::complex<float> >& A,
    Matrix< std::complex<float> >& B,
    int& itqr, int& itpo,
    Options const& opts);

template
void qdwh< std::complex<double> >(
    Matrix< std::complex<double> >& A,
    Matrix< std::complex<double> >& B,
    int& itqr, int& itpo,
    Options const& opts);

} // namespace slate
