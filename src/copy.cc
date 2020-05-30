//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the University of Tennessee nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------
// This research was supported by the Exascale Computing Project (17-SC-20-SC),
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//------------------------------------------------------------------------------
// For assistance with SLATE, email <slate-user@icl.utk.edu>.
// You can also join the "SLATE User" Google group by going to
// https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user,
// signing in with your Google credentials, and then clicking "Join group".
//------------------------------------------------------------------------------

#include "slate/slate.hh"
#include "internal/internal.hh"

#include <list>
#include <tuple>

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::copy from internal::specialization::copy
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// @internal
/// Copy and precision conversion.
/// Generic implementation for any target.
/// @ingroup copy_specialization
///
template <Target target, typename src_matrix_type, typename dst_matrix_type>
void copy(slate::internal::TargetType<target>,
          src_matrix_type A, dst_matrix_type B,
          int64_t lookahead)
{
    // Usually the output matrix (B here) provides all the batch arrays.
    // Here we are using A, because of the differen types.
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

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup copy_specialization
///
template <Target target, typename src_matrix_type, typename dst_matrix_type>
void copy(src_matrix_type& A, dst_matrix_type& B,
          Options const& opts)
{
    int64_t lookahead;
    try {
        lookahead = opts.at(Option::Lookahead).i_;
        assert(lookahead >= 0);
    }
    catch (std::out_of_range&) {
        lookahead = 1;
    }

    internal::specialization::copy(internal::TargetType<target>(),
                                   A, B,
                                   lookahead);
}

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
/// @ingroup copy
///
template <typename src_matrix_type, typename dst_matrix_type>
void copy(src_matrix_type& A, dst_matrix_type& B,
          Options const& opts)
{
    Target target;
    try {
        target = Target(opts.at(Option::Target).i_);
    }
    catch (std::out_of_range&) {
        target = Target::HostTask;
    }

    switch (target) {
        case Target::Host:
        case Target::HostTask:
        default: // todo: this is to silence a warning, should err otherwise
            copy<Target::HostTask>(A, B, opts);
            break;
//      case Target::HostNest:
//          copy<Target::HostNest>(A, B, opts);
//          break;
//      case Target::HostBatch:
//          copy<Target::HostBatch>(A, B, opts);
//          break;
        case Target::Devices:
            copy<Target::Devices>(A, B, opts);
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
