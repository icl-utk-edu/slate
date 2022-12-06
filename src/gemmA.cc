// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "internal/internal.hh"
#include "internal/internal_batch.hh"
#include "auxiliary/Debug.hh"

#include <list>
#include <tuple>

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// @internal
/// Distributed parallel general matrix-matrix multiplication.
/// Designed for situations where A is larger than B or C, so the
/// algorithm does not move A, instead moving B to the location of A
/// and reducing the C matrix.
/// Generic implementation for any target.
/// Dependencies enforce the following behavior:
/// - bcast communications are serialized,
/// - gemm operations are serialized,
/// - bcasts can get ahead of gemms by the value of lookahead.
/// ColMajor layout is assumed
///
/// @ingroup gemm_specialization
///
template <Target target, typename scalar_t>
void gemmA(
    scalar_t alpha, Matrix<scalar_t>& A,
                    Matrix<scalar_t>& B,
    scalar_t beta,  Matrix<scalar_t>& C,
    Options const& opts )
{
    using BcastList = typename Matrix<scalar_t>::BcastList;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // Options
    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector<uint8_t> bcast_vector( A.nt() );
    std::vector<uint8_t> gemmA_vector( A.nt() );
    uint8_t* bcast = bcast_vector.data();
    uint8_t* gemmA = gemmA_vector.data();

    if (target == Target::Devices) {
        A.allocateBatchArrays();
        A.reserveDeviceWorkspace();
    }

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
        // broadcast 0th block col of B
        #pragma omp task depend(out:bcast[0])
        {
            // broadcast block B(i, 0) to ranks owning block col A(:, i)
            BcastList bcast_list_B;
            for (int64_t i = 0; i < B.mt(); ++i)
                bcast_list_B.push_back(
                    {i, 0, {A.sub( 0, A.mt()-1, i, i )}} );
            B.template listBcast<target>( bcast_list_B, layout );
        }

        // broadcast lookahead block cols of B
        for (int64_t k = 1; k < lookahead+1 && k < B.nt(); ++k) {
            #pragma omp task depend(in:bcast[k-1]) \
                             depend(out:bcast[k])
            {
                // broadcast block B(i, k) to ranks owning block col A(:, i)
                BcastList bcast_list_B;
                for (int64_t i = 0; i < B.mt(); ++i)
                    bcast_list_B.push_back(
                        {i, k, {A.sub( 0, A.mt()-1, i, i )}} );
                B.template listBcast<target>( bcast_list_B, layout, k );
            }
        }

        // multiply to get C(:, 0) and reduce
        #pragma omp task depend(in:bcast[0]) \
                         depend(out:gemmA[0])
        {
            // multiply C(:, 0) = alpha A(:, :) B(:, 0) + beta C(:, 0)
            // do multiplication local to A matrix; this may leave
            // some temporary tiles of C that need to be reduced
            internal::gemmA<target>(
                alpha, std::move(A),
                       B.sub( 0, B.mt()-1, 0, 0 ),
                beta,  C.sub( 0, C.mt()-1, 0, 0 ),
                layout );

            // reduce C(:, 0)
            using ReduceList = typename Matrix<scalar_t>::ReduceList;
            ReduceList reduce_list_C;
            for (int64_t i = 0; i < C.mt(); ++i)
                // reduce C(i, 0) across i_th row of A
                reduce_list_C.push_back( {i, 0,
                                          C.sub( i, i, 0, 0 ),
                                          {A.sub( i, i, 0, A.nt()-1 )}
                                        } );
            C.template listReduce( reduce_list_C, layout );
        }
        // Clean the memory introduced by internal::gemmA on Devices
        if (target == Target::Devices) {
            #pragma omp task depend( in:gemmA[ 0 ] ) \
                              shared( C )
            {
                C.sub( 0, C.mt() - 1, 0, 0 ).releaseWorkspace();
            }
        }

        // broadcast (with lookahead) and multiply the rest of the columns
        for (int64_t k = 1; k < B.nt(); ++k) {

            // send next block col of B
            if (k+lookahead < B.nt()) {
                #pragma omp task depend(in:gemmA[k-1]) \
                                 depend(in:bcast[k+lookahead-1]) \
                                 depend(out:bcast[k+lookahead])
                {
                    // broadcast B(i, k+lookahead) to ranks owning block col A(:, i)
                    BcastList bcast_list_B;
                    for (int64_t i = 0; i < B.mt(); ++i)
                        bcast_list_B.push_back(
                            {i, k+lookahead, {A.sub( 0, A.mt()-1, i, i )}} );
                    B.template listBcast<target>(
                        bcast_list_B, layout, k+lookahead );
                }
            }

            // multiply to get C(:, k) and reduce
            #pragma omp task depend(in:bcast[k]) \
                             depend(in:gemmA[k-1]) \
                             depend(out:gemmA[k])
            {
                // multiply C(:, k) = alpha A(:, :) B(:, k) + beta C(:, k)
                // do multiplication local to A matrix; this may leave
                // some temporary tiles of C that need to be reduced
                internal::gemmA<target>(
                    alpha, std::move(A),
                           B.sub( 0, B.mt()-1, k, k ),
                    beta,  C.sub( 0, C.mt()-1, k, k ),
                    layout );

                // reduce C(:, k)
                using ReduceList = typename Matrix<scalar_t>::ReduceList;
                ReduceList reduce_list_C;
                for (int64_t i = 0; i < C.mt(); ++i)
                    // reduce C(i, 0) across i_th row of A
                    reduce_list_C.push_back( {i, k,
                                              C.sub( i, i, k, k ),
                                              {A.sub( i, i, 0, A.nt()-1 )}
                                            } );
                C.template listReduce( reduce_list_C, layout );
            }
            // Clean the memory introduced by internal::gemmA on Devices
            if (target == Target::Devices) {
                #pragma omp task depend( in:gemmA[ k ] ) \
                                  shared( C ) \
                                  firstprivate( k )
                {
                    C.sub( 0, C.mt() - 1, k, k ).releaseWorkspace();
                }
            }
        }
        #pragma omp taskwait

        C.tileUpdateAllOrigin();
    }
}

} // namespace impl

//------------------------------------------------------------------------------
/// Distributed parallel general matrix-matrix multiplication.
/// Performs the matrix-matrix operation
/// \[
///     C = \alpha A B + \beta C,
/// \]
/// where alpha and beta are scalars, and $A$, $B$, and $C$ are matrices, with
/// $A$ an m-by-k matrix, $B$ a k-by-n matrix, and $C$ an m-by-n matrix.
/// The matrices can be transposed or conjugate-transposed beforehand, e.g.,
///
///     auto AT = slate::transpose( A );
///     auto BT = slate::conjTranspose( B );
///     slate::gemm( alpha, AT, BT, beta, C );
///
/// This algorithmic variant manages computation to be local to the
/// location of the A matrix.  This can be useful if size(A) >>
/// size(B), size(C).
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///         One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] alpha
///         The scalar alpha.
///
/// @param[in] A
///         The m-by-k matrix A.
///
/// @param[in] B
///         The k-by-n matrix B.
///
/// @param[in] beta
///         The scalar beta.
///
/// @param[in,out] C
///         On entry, the m-by-n matrix C.
///         On exit, overwritten by the result $\alpha A B + \beta C$.
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
/// @ingroup gemm
///
template <typename scalar_t>
void gemmA(
    scalar_t alpha, Matrix<scalar_t>& A,
                    Matrix<scalar_t>& B,
    scalar_t beta,  Matrix<scalar_t>& C,
    Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
        case Target::HostNest:
        case Target::HostBatch:
            impl::gemmA<Target::HostTask>( alpha, A, B, beta, C, opts );
            break;

        case Target::Devices:
            impl::gemmA<Target::Devices>( alpha, A, B, beta, C, opts );
            break;

    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gemmA<float>(
    float alpha, Matrix<float>& A,
                 Matrix<float>& B,
    float beta,  Matrix<float>& C,
    Options const& opts);

template
void gemmA<double>(
    double alpha, Matrix<double>& A,
                  Matrix<double>& B,
    double beta,  Matrix<double>& C,
    Options const& opts);

template
void gemmA< std::complex<float> >(
    std::complex<float> alpha, Matrix< std::complex<float> >& A,
                               Matrix< std::complex<float> >& B,
    std::complex<float> beta,  Matrix< std::complex<float> >& C,
    Options const& opts);

template
void gemmA< std::complex<double> >(
    std::complex<double> alpha, Matrix< std::complex<double> >& A,
                                Matrix< std::complex<double> >& B,
    std::complex<double> beta,  Matrix< std::complex<double> >& C,
    Options const& opts);

} // namespace slate
