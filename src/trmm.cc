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
/// Distributed parallel triangular matrix-matrix multiplication.
/// Generic implementation for any target.
/// @ingroup trmm_impl
///
template <Target target, typename scalar_t>
void trmm(
    Side side,
    scalar_t alpha, TriangularMatrix<scalar_t>& A,
                              Matrix<scalar_t>& B,
    Options const& opts )
{
    // Options
    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );

    if (target == Target::Devices) {
        const int64_t batch_size_zero = 0; // use default batch size
        const int64_t num_arrays_two = 2; // Number of kernels without lookahead
        B.allocateBatchArrays(batch_size_zero, num_arrays_two);
        B.reserveDeviceWorkspace();
    }

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector<uint8_t> bcast_vector(B.mt());
    std::vector<uint8_t>  gemm_vector(B.nt());
    uint8_t* bcast = bcast_vector.data();
    uint8_t* gemm  =  gemm_vector.data();

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
        #pragma omp task
        {
            work::trmm<target, scalar_t>(side, alpha, A, B, bcast, gemm, lookahead);
            B.tileUpdateAllOrigin();
        }
    }
    B.clearWorkspace();
}

} // namespace impl

//------------------------------------------------------------------------------
/// Distributed parallel triangular matrix-matrix multiplication.
/// Performs one of the triangular matrix-matrix operations
/// \[
///     B = \alpha A B,
/// \]
/// or
/// \[
///     B = \alpha B A,
/// \]
/// where alpha is a scalar, B is an m-by-n matrix and A is a unit or non-unit,
/// upper or lower triangular matrix.
/// The matrices can be transposed or conjugate-transposed beforehand, e.g.,
///
///     auto AT = slate::transpose( A );
///     slate::trmm( Side::Left, alpha, AT, B );
///
/// Complexity (in real): $m^{2} n$ flops.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///         One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] side
///         Whether A appears on the left or on the right of B:
///         - Side::Left:  $B = \alpha A B$
///         - Side::Right: $B = \alpha B A$
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
///         On exit, overwritten by the result $\alpha A B$ or $\alpha B A$.
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
/// @ingroup trmm
///
template <typename scalar_t>
void trmm(
    blas::Side side,
    scalar_t alpha, TriangularMatrix<scalar_t>& A,
                              Matrix<scalar_t>& B,
    Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            impl::trmm<Target::HostTask>( side, alpha, A, B, opts );
            break;

        case Target::HostNest:
            impl::trmm<Target::HostNest>( side, alpha, A, B, opts );
            break;

        case Target::HostBatch:
            impl::trmm<Target::HostBatch>( side, alpha, A, B, opts );
            break;

        case Target::Devices:
            impl::trmm<Target::Devices>( side, alpha, A, B, opts );
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void trmm<float>(
    blas::Side side,
    float alpha, TriangularMatrix<float>& A,
                           Matrix<float>& B,
    Options const& opts);

template
void trmm<double>(
    blas::Side side,
    double alpha, TriangularMatrix<double>& A,
                            Matrix<double>& B,
    Options const& opts);

template
void trmm< std::complex<float> >(
    blas::Side side,
    std::complex<float> alpha, TriangularMatrix< std::complex<float> >& A,
                                         Matrix< std::complex<float> >& B,
    Options const& opts);

template
void trmm< std::complex<double> >(
    blas::Side side,
    std::complex<double> alpha, TriangularMatrix< std::complex<double> >& A,
                                          Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
