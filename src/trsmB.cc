// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "work/work.hh"

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// @internal
/// Distributed parallel triangular matrix solve.
/// Generic implementation for any target.
/// @ingroup trsm_specialization
///
template <Target target, typename scalar_t>
void trsmB(
    Side side,
    scalar_t alpha, TriangularMatrix<scalar_t>& A,
                              Matrix<scalar_t>& B,
    Options const& opts )
{
    // Options
    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );

    if (target == Target::Devices) {
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
        B.allocateBatchArrays( batch_size_zero, num_queues );
        B.reserveDeviceWorkspace();
    }

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector<uint8_t> row_vector( A.nt() );
    uint8_t* row = row_vector.data();

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
        #pragma omp task
        {
            work::trsm<target, scalar_t>( side, alpha, A, B, row, opts );
            B.tileUpdateAllOrigin();
        }
    }
    B.releaseWorkspace();
}

} // namespace impl

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
void trsmB(
    blas::Side side,
    scalar_t alpha, TriangularMatrix<scalar_t>& A,
                              Matrix<scalar_t>& B,
    Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            impl::trsmB<Target::HostTask>( side, alpha, A, B, opts );
            break;

        case Target::HostNest:
            impl::trsmB<Target::HostNest>( side, alpha, A, B, opts );
            break;

        case Target::HostBatch:
            impl::trsmB<Target::HostBatch>( side, alpha, A, B, opts );
            break;

        case Target::Devices:
            impl::trsmB<Target::Devices>( side, alpha, A, B, opts );
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void trsmB<float>(
    blas::Side side,
    float alpha, TriangularMatrix<float>& A,
                           Matrix<float>& B,
    Options const& opts);

template
void trsmB<double>(
    blas::Side side,
    double alpha, TriangularMatrix<double>& A,
                            Matrix<double>& B,
    Options const& opts);

template
void trsmB< std::complex<float> >(
    blas::Side side,
    std::complex<float> alpha, TriangularMatrix< std::complex<float> >& A,
                                         Matrix< std::complex<float> >& B,
    Options const& opts);

template
void trsmB< std::complex<double> >(
    blas::Side side,
    std::complex<double> alpha, TriangularMatrix< std::complex<double> >& A,
                                          Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
