// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "work/work.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::trsm from internal::specialization::trsm
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// @internal
/// Distributed parallel triangular matrix solve.
/// Generic implementation for any target.
/// @ingroup trsm_specialization
///
template <Target target, typename scalar_t>
void trsm(slate::internal::TargetType<target>,
          Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                    Matrix<scalar_t>& B,
          Options const& opts)
{
    Options opts2 = Options( opts );

    if (target == Target::Devices) {
        // Use only TileReleaseStrategy::Slate for trsm.
        // Internal routines (trsm and gemm) called here
        // won't release any tiles. Trsm will
        // clean up tiles.
        opts2[ Option::TileReleaseStrategy ] = TileReleaseStrategy::Slate;

        int64_t lookahead = get_option<int64_t>( opts2, Option::Lookahead, 1 );

        const int64_t batch_size_zero = 0;
        // Allocate batch arrays = number of kernels without
        // lookahead + lookahead
        // number of kernels without lookahead = 2
        // (internal::gemm & internal::trsm)
        // TODO
        // whereas internal::gemm with lookahead will be executed as many as
        // lookaheads, thus
        // internal::gemm with lookahead needs batch arrays equal to the
        // number of lookaheads
        // and the batch_arrays_index starts from
        // the number of kernels without lookahead, and then incremented by 1
        // for every execution for the internal::gemm with lookahead

        // Number of device queues (num_queues):
        // 1) trsm                            (         1 )
        // 2) gemm for trailing matrix update (         1 )
        // 3) lookahead number of gemm's      ( lookahead )
        const int num_queues = 2 + lookahead;
        B.allocateBatchArrays(batch_size_zero, num_queues);
        B.reserveDeviceWorkspace();
    }


    // OpenMP needs pointer types, but vectors are exception safe
    std::vector<uint8_t> row_vector(A.nt());
    uint8_t* row = row_vector.data();

    #pragma omp parallel
    #pragma omp master
    {
        omp_set_nested(1);
        #pragma omp task
        {
            work::trsm<target, scalar_t>(side, alpha, A, B, row, opts2);
            B.tileUpdateAllOrigin();
        }
    }
    B.releaseWorkspace();
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Distributed parallel triangular matrix-matrix solve.
/// Solves one of the triangular matrix equations
/// \[
///     A X = \alpha B,
/// \]
/// or
/// \[
///     X A = \alpha B,
/// \]
/// where alpha is a scalar, B is an m-by-n matrix and A is a unit or non-unit,
/// upper or lower triangular matrix. The matrix X overwrites B.
/// The matrices can be transposed or conjugate-transposed beforehand, e.g.,
///
///     auto AT = slate::transpose( A );
///     slate::trsm( Side::Left, alpha, AT, B );
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///         One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] side
///         Whether A appears on the left or on the right of X:
///         - Side::Left:  solve $A X = \alpha B$
///         - Side::Right: solve $X A = \alpha B$
///
/// @param[in] alpha
///         The scalar alpha.
///
/// @param[in] A
///         - If side = left,  the m-by-m triangular matrix A;
///         - if side = right, the n-by-n triangular matrix A.
///
/// @param[in,out] B
///         On entry, the m-by-n matrix B.
///         On exit, overwritten by the result X.
///
/// @param[in] opts
///         Additional options, as map of name = value pairs. Possible options:
///         - Option::Lookahead:
///           Number of panels to overlap with matrix updates.
///           lookahead >= 0. Default 1.
///         - Option::Target:
///           Implementation to target. Possible values:
///           - HostTask:  OpenMP tasks on CPU host [default].
///           - HostNest:  nested OpenMP parallel for loop on CPU host.
///           - HostBatch: batched BLAS on CPU host.
///           - Devices:   batched BLAS on GPU device.
///
/// @ingroup trsm
///
template <typename scalar_t>
void trsm(blas::Side side,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                    Matrix<scalar_t>& B,
          Options const& opts)
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            internal::specialization::trsm<Target::HostTask>(
                internal::TargetType<Target::HostTask>(),
                side, alpha, A, B, opts);
            break;
        case Target::HostNest:
            internal::specialization::trsm<Target::HostNest>(
                internal::TargetType<Target::HostNest>(),
                side, alpha, A, B, opts);
            break;
        case Target::HostBatch:
            internal::specialization::trsm<Target::HostBatch>(
                internal::TargetType<Target::HostBatch>(),
                side, alpha, A, B, opts);
            break;
        case Target::Devices:
            internal::specialization::trsm<Target::Devices>(
                internal::TargetType<Target::Devices>(),
                side, alpha, A, B, opts);
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void trsm<float>(
    blas::Side side,
    float alpha, TriangularMatrix<float>& A,
                           Matrix<float>& B,
    Options const& opts);

template
void trsm<double>(
    blas::Side side,
    double alpha, TriangularMatrix<double>& A,
                            Matrix<double>& B,
    Options const& opts);

template
void trsm< std::complex<float> >(
    blas::Side side,
    std::complex<float> alpha, TriangularMatrix< std::complex<float> >& A,
                                         Matrix< std::complex<float> >& B,
    Options const& opts);

template
void trsm< std::complex<double> >(
    blas::Side side,
    std::complex<double> alpha, TriangularMatrix< std::complex<double> >& A,
                                          Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
