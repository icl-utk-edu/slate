// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "internal/internal.hh"

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// @internal
/// Distributed parallel Hermitian banded matrix-matrix multiplication.
/// Generic implementation for any target.
/// Dependencies enforce the following behavior:
/// - bcast communications are serialized,
/// - hbmm operations are serialized,
/// - bcasts can get ahead of hbmms by the value of lookahead.
/// Note A, B, and C are passed by value, so we can transpose if needed
/// (for side = right) without affecting caller.
/// @ingroup hbmm_impl
///
/// ColMajor layout is assumed
///
template <Target target, typename scalar_t>
void hbmm(
    Side side,
    scalar_t alpha, HermitianBandMatrix<scalar_t> A,
                    Matrix<scalar_t> B,
    scalar_t beta,  Matrix<scalar_t> C,
    Options const& opts )
{
    // Due to the symmetry, each off diagonal tile is sent twice, once as part
    // of A and once as art of A^T. In principle, this could be avoided by
    // sending each tile only once and retaining it until it is used twice.
    // This would, however, violate the upper bound on the size of communication
    // buffers.
    // The same happens in the symm routine.
    // See also the implementation remarks in the BaseMatrix::listBcast routine.

    using blas::conj;
    using blas::max;
    using blas::min;
    using BcastList = typename Matrix<scalar_t>::BcastList;
    const scalar_t one = 1.0;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // Options
    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );

    // if on right, change to left by transposing A, B, C to get
    // op(C) = op(A)*op(B)
    if (side == Side::Right) {
        A = conjTranspose(A);
        B = conjTranspose(B);
        C = conjTranspose(C);
        alpha = conj(alpha);
        beta  = conj(beta);
    }

    // B and C are mt-by-nt, A is mt-by-mt (assuming side = left)
    assert(A.mt() == B.mt());
    assert(A.nt() == B.mt());
    assert(B.mt() == C.mt());
    assert(B.nt() == C.nt());

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector<uint8_t> bcast_vector(A.nt());
    std::vector<uint8_t>  gemm_vector(A.nt());
    uint8_t* bcast = bcast_vector.data();
    uint8_t* gemm  =  gemm_vector.data();

    int64_t kd = A.bandwidth();

    // todo: initially, assume fixed size, square tiles for simplicity
    int64_t kdt = ceildiv( kd, A.tileNb(0) );

    if (target == Target::Devices) {
        C.allocateBatchArrays();
        C.reserveDeviceWorkspace();
    }

    #pragma omp parallel
    #pragma omp master
    {
        if (A.uplo() == Uplo::Lower) {
            // ----------------------------------------
            // Left, Lower/NoTrans or Upper/ConjTrans case

            // send 1st block col of A and block row of B
            #pragma omp task depend(out:bcast[0])
            {
                int64_t i_end = min(0 + kdt + 1, A.mt());

                // broadcast A(i, 0) to ranks owning block row C(i, :)
                BcastList bcast_list_A;
                for (int64_t i = 0; i < i_end; ++i)
                    bcast_list_A.push_back({i, 0, {C.sub(i, i, 0, C.nt()-1)}});
                A.template listBcast<target>(bcast_list_A, layout);

                // broadcast B(0, j) to ranks owning block col C(:, j)
                BcastList bcast_list_B;
                for (int64_t j = 0; j < B.nt(); ++j)
                    bcast_list_B.push_back({0, j, {C.sub(0, i_end-1, j, j)}});
                B.template listBcast<target>(bcast_list_B, layout);
            }

            // send next lookahead block cols of A and block rows of B
            for (int64_t k = 1; k < lookahead+1 && k < A.nt(); ++k) {
                #pragma omp task depend(in:bcast[k-1]) \
                                 depend(out:bcast[k])
                {
                    int64_t i_begin = max(k - kdt, 0);
                    int64_t i_end   = min(k + kdt + 1, A.mt());

                    BcastList bcast_list_A;

                    // broadcast A(k, i) to ranks owning block row C(i, :)
                    for (int64_t i = i_begin; i < k && i < i_end; ++i) {
                        bcast_list_A.push_back(
                            {k, i, {C.sub(i, i, 0, C.nt()-1)}});
                    }
                    // broadcast A(i, k) to ranks owning block row C(i, :)
                    for (int64_t i = k; i < i_end; ++i) {
                        bcast_list_A.push_back(
                            {i, k, {C.sub(i, i, 0, C.nt()-1)}});
                    }
                    A.template listBcast<target>(bcast_list_A, layout);

                    // broadcast B(k, j) to ranks owning block col C(0:k, j)
                    BcastList bcast_list_B;
                    for (int64_t j = 0; j < B.nt(); ++j) {
                        bcast_list_B.push_back(
                            {k, j, {C.sub(i_begin, i_end-1, j, j)}});
                    }
                    B.template listBcast<target>(bcast_list_B, layout);
                }
            }

            // multiply alpha A(:, 0) B(0, :), which is (hbmm / gemm):
            // C(0, :)      = alpha [ A(0, 0)      B(0, :) ] + beta C(0, :)
            // C(1:mt-1, :) = alpha [ A(1:i_end-1, 0) B(0, :) ] + beta C(1:i_end-1, :)
            #pragma omp task depend(in:bcast[0]) \
                             depend(out:gemm[0])
            {
                internal::hemm<Target::HostTask>(
                    Side::Left,
                    alpha, A.sub(0, 0),
                           B.sub(0, 0, 0, B.nt()-1),
                    beta,  C.sub(0, 0, 0, C.nt()-1));

                int64_t i_end = min(0 + kdt + 1, A.mt());

                if (i_end-1 > 0) {
                    internal::gemm<target>(
                        alpha, A.sub(1, i_end-1, 0, 0),
                               B.sub(0, 0, 0, B.nt()-1),
                        beta,  C.sub(1, i_end-1, 0, C.nt()-1),
                        layout);
                }

                if (beta != one) {
                    // Scale block rows of C below the bandwidth of A:
                    // C(i_end : mt-1, :) = beta * C(i_end : mt-1, :)
                    // todo: make internal::scale routine. This is HostTask.
                    for (int64_t i = i_end; i < C.mt(); ++i) {
                        for (int64_t j = 0; j < C.nt(); ++j) {
                            if (C.tileIsLocal(i, j)) {
                                #pragma omp task slate_omp_default_none \
                                    shared( C ) \
                                    firstprivate(i, j, layout, beta)
                                {
                                    C.tileGetForWriting(i, j, LayoutConvert(layout));
                                    tile::scale( beta, C(i, j) );
                                }
                            }
                        }
                    }
                    #pragma omp taskwait
                }
            }

            for (int64_t k = 1; k < A.nt(); ++k) {
                // send next block col of A and block row of B
                if (k+lookahead < A.nt()) {
                    #pragma omp task depend(in:gemm[k-1]) \
                                     depend(in:bcast[k+lookahead-1]) \
                                     depend(out:bcast[k+lookahead])
                    {
                        int64_t i_begin = max(k + lookahead - kdt, 0);
                        int64_t i_end   = min(k + lookahead + kdt + 1, A.mt());

                        BcastList bcast_list_A;

                        // broadcast A(k+la, i) to ranks owning block row C(i, :)
                        for (int64_t i = i_begin; i < k+lookahead; ++i) {
                            bcast_list_A.push_back(
                                {k+lookahead, i, {C.sub(i, i, 0, C.nt()-1)}});
                        }
                        // broadcast A(i, k+la) to ranks owning block row C(i, :)
                        for (int64_t i = k+lookahead; i < i_end; ++i) {
                            bcast_list_A.push_back(
                                {i, k+lookahead, {C.sub(i, i, 0, C.nt()-1)}});
                        }
                        A.template listBcast<target>(bcast_list_A, layout);

                        // broadcast B(k+la, j) to ranks
                        // owning block col C(0:k+la, j)
                        BcastList bcast_list_B;
                        for (int64_t j = 0; j < B.nt(); ++j) {
                            bcast_list_B.push_back(
                                {k+lookahead, j, {C.sub(i_begin, i_end-1, j, j)}});
                        }
                        B.template listBcast<target>(bcast_list_B, layout);
                    }
                }

                int64_t i_begin = max(k - kdt, 0);
                int64_t i_end   = min(k + kdt + 1, A.mt());

                // multiply alpha A(:, k) B(k, :), which is:
                // C(0:k-1, :)    += alpha [ A(k, 0:k-1)^H  B(k, :) ]  gemm
                // C(k, :)        += alpha [ A(k, k)        B(k, :) ]  hbmm
                // C(k+1:mt-1, :) += alpha [ A(k+1:mt-1, k) B(k, :) ]  gemm
                #pragma omp task depend(in:bcast[k]) \
                                 depend(in:gemm[k-1]) \
                                 depend(out:gemm[k])
                {
                    auto Arow_k = A.sub(k, k, i_begin, k-1);
                    internal::gemm<target>(
                        alpha, conj_transpose( Arow_k ),
                               B.sub(k, k, 0, B.nt()-1),
                        one,   C.sub(i_begin, k-1, 0, C.nt()-1),
                        layout);

                    internal::hemm<Target::HostTask>(
                        Side::Left,
                        alpha, A.sub(k, k),
                               B.sub(k, k, 0, B.nt()-1),
                        one,   C.sub(k, k, 0, C.nt()-1));

                    if (i_end-1 > k) {
                        internal::gemm<target>(
                            alpha, A.sub(k+1, i_end-1, k, k),
                                   B.sub(k, k, 0, B.nt()-1),
                            one,   C.sub(k+1, i_end-1, 0, C.nt()-1),
                            layout);
                    }
                }
            }
        }
        else {
            // send 1st block col (row) of A and block row of B
            #pragma omp task depend(out:bcast[0])
            {
                int64_t i_end = min(0 + kdt + 1, A.mt());

                // broadcast A(i, 0) to ranks owning block row C(i, :)
                BcastList bcast_list_A;
                for (int64_t i = 0; i < i_end; ++i)
                    bcast_list_A.push_back({0, i, {C.sub(i, i, 0, C.nt()-1)}});
                A.template listBcast<target>(bcast_list_A, layout);

                BcastList bcast_list_B;
                // broadcast B(0, j) to ranks owning block col C(:, j)
                for (int64_t j = 0; j < B.nt(); ++j)
                    bcast_list_B.push_back({0, j, {C.sub(0, i_end-1, j, j)}});
                B.template listBcast<target>(bcast_list_B, layout);
            }

            // send next lookahead block cols of A and block rows of B
            for (int64_t k = 1; k < lookahead+1 && k < A.nt(); ++k) {
                #pragma omp task depend(in:bcast[k-1]) \
                                 depend(out:bcast[k])
                {
                    int64_t i_begin = max(k - kdt, 0);
                    int64_t i_end   = min(k + kdt + 1, A.mt());

                    // broadcast A(i, k) to ranks owning block row C(i, :)
                    BcastList bcast_list_A;
                    for (int64_t i = i_begin; i < k && i < i_end; ++i) {
                        bcast_list_A.push_back(
                            {i, k, {C.sub(i, i, 0, C.nt()-1)}});
                    }
                    // broadcast A(i, k) to ranks owning block row C(i, :)
                    for (int64_t i = k; i < i_end; ++i) {
                        bcast_list_A.push_back(
                            {k, i, {C.sub(i, i, 0, C.nt()-1)}});
                    }
                    A.template listBcast<target>(bcast_list_A, layout);

                    // broadcast B(k, j) to ranks owning block col C(0:k, j)
                    BcastList bcast_list_B;
                    for (int64_t j = 0; j < B.nt(); ++j) {
                        bcast_list_B.push_back(
                            {k, j, {C.sub(i_begin, i_end-1, j, j)}});
                    }
                    B.template listBcast<target>(bcast_list_B, layout);
                }
            }

            // multiply alpha A(:, 0) B(0, :), which is (hbmm / gemm):
            // C(0, :)      = alpha [ A(0, 0)      B(0, :) ] + beta C(0, :)
            // C(1:mt-1, :) = alpha [ A(1:i_end-1, 0) B(0, :) ] + beta C(1:i_end-1, :)
            #pragma omp task depend(in:bcast[0]) \
                             depend(out:gemm[0])
            {
                internal::hemm<Target::HostTask>(
                    Side::Left,
                    alpha, A.sub(0, 0),
                           B.sub(0, 0, 0, B.nt()-1),
                    beta,  C.sub(0, 0, 0, C.nt()-1));

                int64_t i_end = min(0 + kdt + 1, A.mt());

                if (i_end-1 > 0) {
                    auto Arow_k = A.sub(0, 0, 1, i_end-1);
                    internal::gemm<target>(
                        alpha, conjTranspose(Arow_k),
                               B.sub(0, 0, 0, B.nt()-1),
                        beta,  C.sub(1, i_end-1, 0, C.nt()-1),
                        layout);
                }

                if (beta != one) {
                    for (int64_t i = i_end; i < C.mt(); ++i) {
                        for (int64_t j = 0; j < C.nt(); ++j) {
                            if (C.tileIsLocal(i, j)) {
                                #pragma omp task slate_omp_default_none \
                                    shared( C ) \
                                    firstprivate(i, j, layout, beta)
                                {
                                    C.tileGetForWriting(i, j, LayoutConvert(layout));
                                    tile::scale( beta, C(i, j) );
                                }
                            }
                        }
                    }
                    #pragma omp taskwait
                }
            }

            for (int64_t k = 1; k < A.nt(); ++k) {
                // send next block col of A and block row of B
                if (k+lookahead < A.nt()) {
                    #pragma omp task depend(in:gemm[k-1]) \
                                     depend(in:bcast[k+lookahead-1]) \
                                     depend(out:bcast[k+lookahead])
                    {
                        int64_t i_begin = max(k + lookahead - kdt, 0);
                        int64_t i_end   = min(k + lookahead + kdt + 1, A.mt());

                        // broadcast A(i, k+la) to ranks owning block row C(i, :)
                        BcastList bcast_list_A;
                        for (int64_t i = i_begin; i < k+lookahead; ++i) {
                            bcast_list_A.push_back(
                                {i, k+lookahead, {C.sub(i, i, 0, C.nt()-1)}});
                        }
                        // broadcast A(k+la, i) to ranks owning block row C(i, :)
                        for (int64_t i = k+lookahead; i < i_end; ++i) {
                            bcast_list_A.push_back(
                                {k+lookahead, i, {C.sub(i, i, 0, C.nt()-1)}});
                        }
                        A.template listBcast<target>(bcast_list_A, layout);

                        // broadcast B(k+la, j) to ranks
                        // owning block col C(0:k+la, j)
                        BcastList bcast_list_B;
                        for (int64_t j = 0; j < B.nt(); ++j) {
                            bcast_list_B.push_back(
                                {k+lookahead, j, {C.sub(i_begin, i_end-1, j, j)}});
                        }
                        B.template listBcast<target>(bcast_list_B, layout);
                    }
                }

                int64_t i_begin = max(k - kdt, 0);
                int64_t i_end   = min(k + kdt + 1, A.mt());

                // multiply alpha A(:, k) B(k, :), which is:
                // C(0:k-1, :)    += alpha [ A(k, 0:k-1)^H  B(k, :) ]  gemm
                // C(k, :)        += alpha [ A(k, k)        B(k, :) ]  hbmm
                // C(k+1:mt-1, :) += alpha [ A(k+1:mt-1, k) B(k, :) ]  gemm
                #pragma omp task depend(in:bcast[k]) \
                                 depend(in:gemm[k-1]) \
                                 depend(out:gemm[k])
                {
                    internal::gemm<target>(
                        alpha, A.sub(i_begin, k-1, k, k),
                               B.sub(k, k, 0, B.nt()-1),
                        one,   C.sub(i_begin, k-1, 0, C.nt()-1),
                        layout);

                    internal::hemm<Target::HostTask>(
                        Side::Left,
                        alpha, A.sub(k, k),
                               B.sub(k, k, 0, B.nt()-1),
                        one,   C.sub(k, k, 0, C.nt()-1));

                    if (i_end-1 > k) {
                        auto Arow_k = A.sub(k, k, k+1, i_end-1);
                        internal::gemm<target>(
                            alpha, conj_transpose( Arow_k ),
                                   B.sub(k, k, 0, B.nt()-1),
                            one,   C.sub(k+1, i_end-1, 0, C.nt()-1),
                            layout);
                    }
                }
            }
        }
    }

    C.tileUpdateAllOrigin();
    C.releaseWorkspace();
}

} // namespace impl

//------------------------------------------------------------------------------
/// Distributed parallel Hermitian banded matrix-matrix multiplication.
/// Performs one of the matrix-matrix operations
/// \[
///     C = \alpha A B + \beta C
/// \]
/// or
/// \[
///     C = \alpha B A + \beta C
/// \]
/// where alpha and beta are scalars, A is a Hermitian banded matrix and B and
/// C are m-by-n matrices.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///         One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] side
///         Whether the Hermitian banded matrix A appears on the left or right:
///         - Side::Left:  $C = \alpha A B + \beta C$
///         - Side::Right: $C = \alpha B A + \beta C$
///
/// @param[in] alpha
///         The scalar alpha.
///
/// @param[in] A
///         - If side = left,  the m-by-m Hermitian banded matrix A;
///         - if side = right, the n-by-n Hermitian banded matrix A.
///
/// @param[in] B
///         The m-by-n matrix B.
///
/// @param[in] beta
///         The scalar beta.
///
/// @param[in,out] C
///         On entry, the m-by-n matrix C.
///         On exit, overwritten by the result
///         $\alpha A B + \beta C$ or $\alpha B A + \beta C$.
///
/// @param[in] opts
///         Additional options, as map of name = value pairs. Possible options:
///         - Option::Lookahead:
///           Number of blocks to overlap communication and computation.
///           lookahead >= 0. Default 1.
///         - Option::Target:
///           Implementation to target. Possible values:
///           - HostTask:  OpenMP tasks on CPU host [default].
///           - HostNest:  nested OpenMP parallel for loop on CPU host.
///           - HostBatch: batched BLAS on CPU host.
///           - Devices:   batched BLAS on GPU device.
///
/// @ingroup hbmm
///
template <typename scalar_t>
void hbmm(Side side,
    scalar_t alpha, HermitianBandMatrix<scalar_t>& A,
                    Matrix<scalar_t>& B,
    scalar_t beta,  Matrix<scalar_t>& C,
    Options const& opts)
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            impl::hbmm<Target::HostTask>( side, alpha, A, B, beta, C, opts );
            break;

        case Target::HostNest:
            impl::hbmm<Target::HostNest>( side, alpha, A, B, beta, C, opts );
            break;

        case Target::HostBatch:
            impl::hbmm<Target::HostBatch>( side, alpha, A, B, beta, C, opts );
            break;

        case Target::Devices:
            impl::hbmm<Target::Devices>( side, alpha, A, B, beta, C, opts );
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void hbmm<float>(
    Side side,
    float alpha, HermitianBandMatrix<float>& A,
                 Matrix<float>& B,
    float beta,  Matrix<float>& C,
    Options const& opts);

template
void hbmm<double>(
    Side side,
    double alpha, HermitianBandMatrix<double>& A,
                  Matrix<double>& B,
    double beta,  Matrix<double>& C,
    Options const& opts);

template
void hbmm< std::complex<float> >(
    Side side,
    std::complex<float> alpha, HermitianBandMatrix< std::complex<float> >& A,
                               Matrix< std::complex<float> >& B,
    std::complex<float> beta,  Matrix< std::complex<float> >& C,
    Options const& opts);

template
void hbmm< std::complex<double> >(
    Side side,
    std::complex<double> alpha, HermitianBandMatrix< std::complex<double> >& A,
                                Matrix< std::complex<double> >& B,
    std::complex<double> beta,  Matrix< std::complex<double> >& C,
    Options const& opts);

} // namespace slate
