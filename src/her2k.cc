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
/// Distributed parallel Hermitian rank 2k update.
/// Generic implementation for any target.
/// Dependencies enforce the following behavior:
/// - bcast communications are serialized,
/// - her2k operations are serialized,
/// - bcasts can get ahead of her2ks by the value of lookahead.
/// Note A, B, and C are passed by value, so we can transpose if needed
/// (for uplo = Upper) without affecting caller.
/// @ingroup her2k_impl
///
template <Target target, typename scalar_t>
void her2k(
    scalar_t alpha,                 Matrix<scalar_t> A,
                                    Matrix<scalar_t> B,
    blas::real_type<scalar_t> beta, HermitianMatrix<scalar_t> C,
    Options const& opts )
{
    using real_t = blas::real_type<scalar_t>;
    using BcastList = typename Matrix<scalar_t>::BcastList;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // Options
    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );

    // if upper, change to lower
    if (C.uplo() == Uplo::Upper)
        C = conjTranspose(C);

    // A is mt-by-nt, C is mt-by-mt
    assert(A.mt() == C.mt());
    assert(B.mt() == C.mt());
    assert(A.nt() == B.nt());

    // Use only TileReleaseStrategy::Slate for her2k.
    // Internal her2k routine called here won't release
    // any tiles. This routine will clean up tiles.
    Options const opts_local =  {
        {slate::Option::TileReleaseStrategy, TileReleaseStrategy::Slate}};

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector<uint8_t> bcast_vector(A.nt());
    std::vector<uint8_t>  gemm_vector(A.nt());
    uint8_t* bcast = bcast_vector.data();
    uint8_t* gemm  =  gemm_vector.data();
    const int default_priority = 0;
    const int default_queue = 0;

    if (target == Target::Devices) {
        C.allocateBatchArrays();
        C.reserveDeviceWorkspace();
    }

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
        // Lower/NoTrans or Upper/ConjTrans case
        // send 1st block col of A
        #pragma omp task depend(out:bcast[0])
        {
            // broadcast A(i, 0) and B(i, 0) to ranks owning
            // block row C(i, 0:i) and block col C(i:n, i)
            BcastList bcast_list_A;
            BcastList bcast_list_B;
            for (int64_t i = 0; i < A.mt(); ++i) {
                bcast_list_A.push_back({i, 0, {C.sub(i, i, 0, i),
                                               C.sub(i, C.mt()-1, i, i)
                                              }});
                bcast_list_B.push_back({i, 0, {C.sub(i, i, 0, i),
                                               C.sub(i, C.mt()-1, i, i)
                                              }});
            }
            A.template listBcast<target>(bcast_list_A, layout);
            B.template listBcast<target>(bcast_list_B, layout);
        }

        // send next lookahead block cols of A
        for (int64_t k = 1; k < lookahead+1 && k < A.nt(); ++k) {
            #pragma omp task depend(in:bcast[k-1]) \
                             depend(out:bcast[k])
            {
                // broadcast A(i, k) and B(i, k) to ranks owning
                // block row C(i, 0:i) and block col C(i:n, i)
                BcastList bcast_list_A;
                BcastList bcast_list_B;
                for (int64_t i = 0; i < A.mt(); ++i) {
                    bcast_list_A.push_back({i, k, {C.sub(i, i, 0, i),
                                                   C.sub(i, C.mt()-1, i, i)}});
                    bcast_list_B.push_back({i, k, {C.sub(i, i, 0, i),
                                                   C.sub(i, C.mt()-1, i, i)}});
                }
                A.template listBcast<target>(bcast_list_A, layout);
                B.template listBcast<target>(bcast_list_B, layout);
            }
        }

        // multiply alpha A(:, 0) B(0, :)^H + beta C
        #pragma omp task depend(in:bcast[0]) \
                         depend(out:gemm[0])
        {
            auto A_col0 = A.sub( 0, A.mt()-1, 0, 0 );
            auto B_col0 = B.sub( 0, B.mt()-1, 0, 0 );
            internal::her2k<target>(
                alpha, std::move( A_col0 ),
                       std::move( B_col0 ),
                beta,  std::move( C ),
                default_priority, default_queue, layout, opts_local );

            // Erase remote tiles on all devices including host
            A_col0.eraseRemoteWorkspace();
            B_col0.eraseRemoteWorkspace();

            // Erase local workspace on devices.
            A_col0.eraseLocalWorkspace();
            B_col0.eraseLocalWorkspace();
        }

        for (int64_t k = 1; k < A.nt(); ++k) {

            // send next block col of A and block row of B
            if (k+lookahead < A.nt()) {
                #pragma omp task depend(in:gemm[k-1]) \
                                 depend(in:bcast[k+lookahead-1]) \
                                 depend(out:bcast[k+lookahead])
                {
                    // broadcast A(k+la, i) to ranks owning
                    // block row C(i, 0:i) and block col C(i:n, i)
                    BcastList bcast_list_A;
                    BcastList bcast_list_B;
                    for (int64_t i = 0; i < A.mt(); ++i) {
                        bcast_list_A.push_back(
                            {i, k+lookahead, {C.sub(i, i, 0, i),
                                              C.sub(i, C.mt()-1, i, i)}});
                        bcast_list_B.push_back(
                            {i, k+lookahead, {C.sub(i, i, 0, i),
                                              C.sub(i, C.mt()-1, i, i)}});
                    }
                    A.template listBcast<target>(bcast_list_A, layout);
                    B.template listBcast<target>(bcast_list_B, layout);
                }
            }

            // multiply alpha A(:, k) B(k, :) + C, no beta
            #pragma omp task depend(in:bcast[k]) \
                             depend(in:gemm[k-1]) \
                             depend(out:gemm[k])
            {
                auto A_colk = A.sub( 0, A.mt()-1, k, k );
                auto B_colk = B.sub( 0, B.mt()-1, k, k );
                internal::her2k<target>(
                    alpha,         std::move( A_colk ),
                                   std::move( B_colk ),
                    real_t( 1.0 ), std::move( C ),
                    default_priority, default_queue, layout, opts_local );

                // Erase remote tiles on all devices including host
                A_colk.eraseRemoteWorkspace();
                B_colk.eraseRemoteWorkspace();

                // Erase local workspace on devices.
                A_colk.eraseLocalWorkspace();
                B_colk.eraseLocalWorkspace();
            }
        }

        #pragma omp taskwait
        C.tileUpdateAllOrigin();
    }

    C.clearWorkspace();
}

} // namespace impl

//------------------------------------------------------------------------------
/// Distributed parallel Hermitian rank 2k update.
/// Performs the Hermitian rank 2k operation
/// \[
///     C = \alpha A B^H + \alpha B A^H + \beta C,
/// \]
/// where alpha and beta are scalars, C is an n-by-n Hermitian
/// matrix, and A and B are an n-by-k matrices.
/// The matrices can be conjugate-transposed beforehand, e.g.,
///
///     auto AT = slate::conjTranspose( A );
///     auto BT = slate::conjTranspose( B );
///     slate::her2k( alpha, AT, BT, beta, C );
///
/// Complexity (in real): $\approx 2 k n^{2}$ flops.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///         One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] alpha
///         The scalar alpha.
///
/// @param[in] A
///         The n-by-k matrix A.
///
/// @param[in] B
///         The n-by-k matrix B.
///
/// @param[in] beta
///         The real scalar beta.
///
/// @param[in,out] C
///         On entry, the n-by-n Hermitian matrix C.
///         On exit, overwritten by the result
///         $C = \alpha A B^H + \alpha B A^H + \beta C$.
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
/// @ingroup her2k
///
template <typename scalar_t>
void her2k(
    scalar_t alpha,                 Matrix<scalar_t>& A,
                                    Matrix<scalar_t>& B,
    blas::real_type<scalar_t> beta, HermitianMatrix<scalar_t>& C,
    Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            impl::her2k<Target::HostTask>( alpha, A, B, beta, C, opts );
            break;

        case Target::HostNest:
            impl::her2k<Target::HostNest>( alpha, A, B, beta, C, opts );
            break;

        case Target::HostBatch:
            impl::her2k<Target::HostBatch>( alpha, A, B, beta, C, opts );
            break;

        case Target::Devices:
            impl::her2k<Target::Devices>( alpha, A, B, beta, C, opts );
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void her2k<float>(
    float alpha, Matrix<float>& A,
                 Matrix<float>& B,
    float beta,  HermitianMatrix<float>& C,
    Options const& opts);

template
void her2k<double>(
    double alpha, Matrix<double>& A,
                  Matrix<double>& B,
    double beta,  HermitianMatrix<double>& C,
    Options const& opts);

template
void her2k< std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >& A,
                               Matrix< std::complex<float> >& B,
    float beta,                HermitianMatrix< std::complex<float> >& C,
    Options const& opts);

template
void her2k< std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >& A,
                                Matrix< std::complex<double> >& B,
    double beta,                HermitianMatrix< std::complex<double> >& C,
    Options const& opts);

} // namespace slate
