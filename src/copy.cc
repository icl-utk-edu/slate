// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "internal/internal.hh"

#include <list>
#include <tuple>

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// @internal
/// Copy and precision conversion.
/// Generic implementation for any target.
/// @ingroup copy_impl
///
template <Target target, typename src_matrix_type, typename dst_matrix_type>
void copy(
    src_matrix_type A, dst_matrix_type B,
    Options const& opts )
{
    // Usually the output matrix (B here) provides all the batch arrays.
    // Here we are using A, because of the different types.
    if (target == Target::Devices) {
        A.allocateBatchArrays();
        B.allocateBatchArrays();
        // todo: is this needed here when the matrix is already on devices?
        B.reserveDeviceWorkspace();
    }

    #pragma omp parallel
    #pragma omp master
    {
        internal::copy<target>(std::move(A), std::move(B));
        #pragma omp taskwait
        B.tileUpdateAllOrigin();
    }

    B.releaseWorkspace();
}

} // namespace impl

//------------------------------------------------------------------------------
/// Copy and precision conversion.
/// Assuming the same distribution of source and destination.
/// Transposition is currently ignored.
/// TODO: Inspect transposition?
//------------------------------------------------------------------------------
/// @tparam src_matrix_type
///     Source matrix type: Matrix, HermitianMatrix.
///
/// @tparam dst_matrix_type
///     Destination matrix type: Matrix, HermitianMatrix.
//------------------------------------------------------------------------------
/// @param[in] A
///         The m-by-n matrix A.
///
/// @param[in,out] B
///         On entry, the m-by-n matrix B.
///         On exit, overwritten by dst_matrix_type(A).
///
/// @param[in] opts
///         Additional options, as map of name = value pairs. Possible options:
///         - Option::Target:
///           Implementation to target. Possible values:
///           - HostTask:  OpenMP tasks on CPU host [default].
///           - HostNest:  nested OpenMP parallel for loop on CPU host.
///           - HostBatch: batched BLAS on CPU host.
///           - Devices:   batched BLAS on GPU device.
///
/// @ingroup copy
///
template <typename src_matrix_type, typename dst_matrix_type>
void copy(src_matrix_type& A, dst_matrix_type& B,
          Options const& opts)
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
        default: // todo: this is to silence a warning, should err otherwise
            impl::copy<Target::HostTask>( A, B, opts );
            break;
//      case Target::HostNest:
//          copy<Target::HostNest>(A, B, opts);
//          break;
//      case Target::HostBatch:
//          copy<Target::HostBatch>(A, B, opts);
//          break;

        case Target::Devices:
            impl::copy<Target::Devices>( A, B, opts );
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void copy(
    Matrix<float>& A,
    Matrix<float>& B,
    Options const& opts);

template
void copy(
    Matrix<float>& A,
    Matrix<double>& B,
    Options const& opts);

template
void copy(
    Matrix<double>& A,
    Matrix<float>& B,
    Options const& opts);

template
void copy(
    Matrix<double>& A,
    Matrix<double>& B,
    Options const& opts);

template
void copy(
    Matrix<std::complex<float> >& A,
    Matrix<std::complex<float> >& B,
    Options const& opts);

template
void copy(
    Matrix<std::complex<float> >& A,
    Matrix<std::complex<double> >& B,
    Options const& opts);

template
void copy(
    Matrix<std::complex<double> >& A,
    Matrix<std::complex<float> >& B,
    Options const& opts);

template
void copy(
    Matrix<std::complex<double> >& A,
    Matrix<std::complex<double> >& B,
    Options const& opts);

//---------------------------------------
// template
// void copy(
//     BaseTrapezoidMatrix<float>& A,
//     BaseTrapezoidMatrix<float>& B,
//     Options const& opts);

// template
// void copy(
//     BaseTrapezoidMatrix<float>& A,
//     BaseTrapezoidMatrix<double>& B,
//     Options const& opts);

// template
// void copy(
//     BaseTrapezoidMatrix<double>& A,
//     BaseTrapezoidMatrix<float>& B,
//     Options const& opts);

// template
// void copy(
//     BaseTrapezoidMatrix<double>& A,
//     BaseTrapezoidMatrix<double>& B,
//     Options const& opts);

// template
// void copy(
//     BaseTrapezoidMatrix<std::complex<float> >& A,
//     BaseTrapezoidMatrix<std::complex<float> >& B,
//     Options const& opts);

// template
// void copy(
//     BaseTrapezoidMatrix<std::complex<float> >& A,
//     BaseTrapezoidMatrix<std::complex<double> >& B,
//     Options const& opts);

// template
// void copy(
//     BaseTrapezoidMatrix<std::complex<double> >& A,
//     BaseTrapezoidMatrix<std::complex<float> >& B,
//     Options const& opts);

// template
// void copy(
//     BaseTrapezoidMatrix<std::complex<double> >& A,
//     BaseTrapezoidMatrix<std::complex<double> >& B,
//     Options const& opts);

//---------------------------------------
template
void copy(
    HermitianMatrix<float>& A,
    HermitianMatrix<float>& B,
    Options const& opts);

template
void copy(
    HermitianMatrix<float>& A,
    HermitianMatrix<double>& B,
    Options const& opts);

template
void copy(
    HermitianMatrix<double>& A,
    HermitianMatrix<float>& B,
    Options const& opts);

template
void copy(
    HermitianMatrix<double>& A,
    HermitianMatrix<double>& B,
    Options const& opts);

template
void copy(
    HermitianMatrix<std::complex<float> >& A,
    HermitianMatrix<std::complex<float> >& B,
    Options const& opts);

template
void copy(
    HermitianMatrix<std::complex<float> >& A,
    HermitianMatrix<std::complex<double> >& B,
    Options const& opts);

template
void copy(
    HermitianMatrix<std::complex<double> >& A,
    HermitianMatrix<std::complex<float> >& B,
    Options const& opts);

template
void copy(
    HermitianMatrix<std::complex<double> >& A,
    HermitianMatrix<std::complex<double> >& B,
    Options const& opts);

//---------------------------------------
template
void copy(
    SymmetricMatrix<float>& A,
    SymmetricMatrix<float>& B,
    Options const& opts);

template
void copy(
    SymmetricMatrix<float>& A,
    SymmetricMatrix<double>& B,
    Options const& opts);

template
void copy(
    SymmetricMatrix<double>& A,
    SymmetricMatrix<float>& B,
    Options const& opts);

template
void copy(
    SymmetricMatrix<double>& A,
    SymmetricMatrix<double>& B,
    Options const& opts);

template
void copy(
    SymmetricMatrix<std::complex<float> >& A,
    SymmetricMatrix<std::complex<float> >& B,
    Options const& opts);

template
void copy(
    SymmetricMatrix<std::complex<float> >& A,
    SymmetricMatrix<std::complex<double> >& B,
    Options const& opts);

template
void copy(
    SymmetricMatrix<std::complex<double> >& A,
    SymmetricMatrix<std::complex<float> >& B,
    Options const& opts);

template
void copy(
    SymmetricMatrix<std::complex<double> >& A,
    SymmetricMatrix<std::complex<double> >& B,
    Options const& opts);

//---------------------------------------
template
void copy(
    TrapezoidMatrix<float>& A,
    TrapezoidMatrix<float>& B,
    Options const& opts);

template
void copy(
    TrapezoidMatrix<float>& A,
    TrapezoidMatrix<double>& B,
    Options const& opts);

template
void copy(
    TrapezoidMatrix<double>& A,
    TrapezoidMatrix<float>& B,
    Options const& opts);

template
void copy(
    TrapezoidMatrix<double>& A,
    TrapezoidMatrix<double>& B,
    Options const& opts);

template
void copy(
    TrapezoidMatrix<std::complex<float> >& A,
    TrapezoidMatrix<std::complex<float> >& B,
    Options const& opts);

template
void copy(
    TrapezoidMatrix<std::complex<float> >& A,
    TrapezoidMatrix<std::complex<double> >& B,
    Options const& opts);

template
void copy(
    TrapezoidMatrix<std::complex<double> >& A,
    TrapezoidMatrix<std::complex<float> >& B,
    Options const& opts);

template
void copy(
    TrapezoidMatrix<std::complex<double> >& A,
    TrapezoidMatrix<std::complex<double> >& B,
    Options const& opts);

//---------------------------------------
template
void copy(
    TriangularMatrix<float>& A,
    TriangularMatrix<float>& B,
    Options const& opts);

template
void copy(
    TriangularMatrix<float>& A,
    TriangularMatrix<double>& B,
    Options const& opts);

template
void copy(
    TriangularMatrix<double>& A,
    TriangularMatrix<float>& B,
    Options const& opts);

template
void copy(
    TriangularMatrix<double>& A,
    TriangularMatrix<double>& B,
    Options const& opts);

template
void copy(
    TriangularMatrix<std::complex<float> >& A,
    TriangularMatrix<std::complex<float> >& B,
    Options const& opts);

template
void copy(
    TriangularMatrix<std::complex<float> >& A,
    TriangularMatrix<std::complex<double> >& B,
    Options const& opts);

template
void copy(
    TriangularMatrix<std::complex<double> >& A,
    TriangularMatrix<std::complex<float> >& B,
    Options const& opts);

template
void copy(
    TriangularMatrix<std::complex<double> >& A,
    TriangularMatrix<std::complex<double> >& B,
    Options const& opts);

} // namespace slate
