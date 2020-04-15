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

#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/types.hh"
#include "internal/Tile_lapack.hh"
#include "internal/internal.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Reduces a complex Hermitian positive-definite generalized eigenvalue problem
/// to the standard form of single tile.
/// Dispatches to target implementations.
/// @ingroup hegst_internal
///
template <Target target, typename scalar_t>
void hegst(int64_t itype, HermitianMatrix< scalar_t >&& A,
                          HermitianMatrix< scalar_t >&& B)
{
    hegst(internal::TargetType<target>(), itype, A, B);
}

//------------------------------------------------------------------------------
/// Reduces a complex Hermitian positive-definite generalized eigenvalue problem
/// to the standard form of single tile, host implementation.
/// @ingroup hegst_internal
///
template <typename scalar_t>
void hegst(internal::TargetType<Target::HostTask>,
           int64_t itype, HermitianMatrix<scalar_t>& A,
                          HermitianMatrix<scalar_t>& B)
{
    assert(A.mt() == 1);
    assert(A.nt() == 1);
    assert(B.mt() == 1);
    assert(B.nt() == 1);

    if (A.tileIsLocal(0, 0)) {
        #pragma omp task shared(A, B)
        {
            A.tileGetForWriting(0, 0, LayoutConvert::ColMajor);
            B.tileGetForReading(0, 0, LayoutConvert::ColMajor);
            hegst(itype, A(0, 0), B(0, 0));
        }
    }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void hegst<Target::HostTask, float>(
    int64_t itype, HermitianMatrix<float>&& A,
                   HermitianMatrix<float>&& B);

// ----------------------------------------
template
void hegst<Target::HostTask, double>(
    int64_t itype, HermitianMatrix<double>&& A,
                   HermitianMatrix<double>&& B);

// ----------------------------------------
template
void hegst<Target::HostTask, std::complex<float>>(
    int64_t itype, HermitianMatrix<std::complex<float>>&& A,
                   HermitianMatrix<std::complex<float>>&& B);

// ----------------------------------------
template
void hegst<Target::HostTask, std::complex<double>>(
    int64_t itype, HermitianMatrix<std::complex<double>>&& A,
                   HermitianMatrix<std::complex<double>>&& B);

} // namespace internal
} // namespace slate
