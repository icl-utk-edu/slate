// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/SymmetricMatrix.hh"
#include "internal/internal.hh"

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// @internal
/// Distributed parallel symmetric rank k update.
/// Generic implementation for any target.
/// Dependencies enforce the following behavior:
/// - bcast communications are serialized,
/// - syrk operations are serialized,
/// - bcasts can get ahead of syrks by the value of lookahead.
/// Note A and C are passed by value, so we can transpose if needed
/// (for uplo = Upper) without affecting caller.
/// @ingroup syrk_impl
///
template <Target target, typename scalar_t>
void syrk(
    scalar_t alpha, Matrix<scalar_t> A,
    scalar_t beta,  SymmetricMatrix<scalar_t> C,
    Options const& opts )
{
    using BcastList = typename Matrix<scalar_t>::BcastList;

    const scalar_t one = 1.0;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // Options
    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );

    // if upper, change to lower
    if (C.uplo() == Uplo::Upper)
        C = transpose(C);

    // A is mt-by-nt, C is mt-by-mt
    assert(A.mt() == C.mt());

    // Use only TileReleaseStrategy::Slate for syrk.
    // Internal syrk routine called here won't release
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
        // Lower/NoTrans or Upper/Trans case
        // send 1st block col of A
        #pragma omp task depend(out:bcast[0])
        {
            // broadcast A(i, 0) to ranks owning
            // block row C(i, 0:i) and block col C(i:n, i)
            BcastList bcast_list_A;
            for (int64_t i = 0; i < A.mt(); ++i) {
                bcast_list_A.push_back({i, 0, {C.sub(i, i, 0, i),
                                               C.sub(i, C.mt()-1, i, i)}});
            }
            A.template listBcast<target>(bcast_list_A, layout);
        }

        // send next lookahead block cols of A
        for (int64_t k = 1; k < lookahead+1 && k < A.nt(); ++k) {
            #pragma omp task depend(in:bcast[k-1]) \
                             depend(out:bcast[k])
            {
                // broadcast A(i, k) to ranks owning
                // block row C(i, 0:i) and block col C(i:n, i)
                BcastList bcast_list_A;
                for (int64_t i = 0; i < A.mt(); ++i) {
                    bcast_list_A.push_back({i, k, {C.sub(i, i, 0, i),
                                                   C.sub(i, C.mt()-1, i, i)}});
                }
                A.template listBcast<target>(bcast_list_A, layout);
            }
        }

        // multiply alpha A(:, 0) A(0, :)^T + beta C
        #pragma omp task depend(in:bcast[0]) \
                         depend(out:gemm[0])
        {
            auto A_col0 = A.sub( 0, A.mt()-1, 0, 0 );
            internal::syrk<target>(
                alpha, std::move( A_col0 ),
                beta,  std::move( C ),
                default_priority, default_queue, layout, opts_local );

            // Erase remote tiles on all devices including host
            A_col0.eraseRemoteWorkspace();

            // Erase local workspace on devices.
            A_col0.eraseLocalWorkspace();
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
                    for (int64_t i = 0; i < A.mt(); ++i) {
                        bcast_list_A.push_back(
                            {i, k+lookahead, {C.sub(i, i, 0, i),
                                              C.sub(i, C.mt()-1, i, i)}});
                    }
                    A.template listBcast<target>(bcast_list_A, layout);
                }
            }

            // multiply alpha A(:, k) A(k, :) + C, no beta
            #pragma omp task depend(in:bcast[k]) \
                             depend(in:gemm[k-1]) \
                             depend(out:gemm[k])
            {
                auto A_colk = A.sub( 0, A.mt()-1, k, k );

                internal::syrk<target>(
                    alpha, std::move( A_colk ),
                    one,   std::move( C ),
                    default_priority, default_queue, layout, opts_local );

                // Erase remote tiles on all devices including host
                A_colk.eraseRemoteWorkspace();

                // Erase local workspace on devices.
                A_colk.eraseLocalWorkspace();
            }
        }

        #pragma omp taskwait
        C.tileUpdateAllOrigin();
    }

    C.clearWorkspace();
}

} // namespace impl

//------------------------------------------------------------------------------
/// Distributed parallel symmetric rank k update.
/// Performs the symmetric rank k operation
/// \[
///     C = \alpha A A^T + \beta C,
/// \]
/// where alpha and beta are scalars, C is an n-by-n symmetric
/// matrix, and A is an n-by-k matrix.
/// The matrices can be transposed beforehand, e.g.,
///
///     auto AT = slate::transpose( A );
///     slate::syrk( alpha, AT, beta, C );
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
/// @param[in] beta
///         The scalar beta.
///
/// @param[in,out] C
///         On entry, the n-by-n symmetric matrix C.
///         On exit, overwritten by the result
///         $C = \alpha A A^T + \beta C$.
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
/// @ingroup syrk
///
template <typename scalar_t>
void syrk(
    scalar_t alpha, Matrix<scalar_t>& A,
    scalar_t beta,  SymmetricMatrix<scalar_t>& C,
    Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            impl::syrk<Target::HostTask>( alpha, A, beta, C, opts );
            break;

        case Target::HostNest:
            impl::syrk<Target::HostNest>( alpha, A, beta, C, opts );
            break;

        case Target::HostBatch:
            impl::syrk<Target::HostBatch>( alpha, A, beta, C, opts );
            break;

        case Target::Devices:
            impl::syrk<Target::Devices>( alpha, A, beta, C, opts );
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void syrk<float>(
    float alpha, Matrix<float>& A,
    float beta,  SymmetricMatrix<float>& C,
    Options const& opts);

template
void syrk<double>(
    double alpha, Matrix<double>& A,
    double beta,  SymmetricMatrix<double>& C,
    Options const& opts);

template
void syrk< std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >& A,
    std::complex<float> beta,  SymmetricMatrix< std::complex<float> >& C,
    Options const& opts);

template
void syrk< std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >& A,
    std::complex<double> beta,  SymmetricMatrix< std::complex<double> >& C,
    Options const& opts);

} // namespace slate
