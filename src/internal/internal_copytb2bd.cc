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
#include "slate/TriangularBandMatrix.hh"
#include "slate/types.hh"
#include "slate/Tile_blas.hh"
#include "internal/internal.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Copy bi-diagonal TriangularBand matrix to two vectors.
/// Dispatches to target implementations.
/// @ingroup copyge2tb_internal
///
template <Target target, typename scalar_t>
void copytb2bd(TriangularBandMatrix<scalar_t>& A,
               std::vector< blas::real_type<scalar_t> >& D,
               std::vector< blas::real_type<scalar_t> >& E)
{
    copytb2bd(internal::TargetType<target>(),
               A,
               D, E);
}

//------------------------------------------------------------------------------
/// Copy bi-diagonal TriangularBand matrix to two vectors.
/// Host OpenMP task implementation.
/// @ingroup copyge2tb_internal
///
template <typename scalar_t>
void copytb2bd(internal::TargetType<Target::HostTask>,
               TriangularBandMatrix<scalar_t> A,
               std::vector< blas::real_type<scalar_t> >& D,
               std::vector< blas::real_type<scalar_t> >& E)
{
    trace::Block trace_block("slate::copytb2bd");
    using blas::real;

    // If lower, change to upper.
    if (A.uplo() == Uplo::Lower) {
        A = conjTranspose(A);
    }

    // Make sure it is a bi-diagonal matrix.
    slate_assert(A.bandwidth() == 1);

    int64_t nt = A.nt();
    int64_t n = A.n();
    D.resize(n);
    E.resize(n - 1);

    // Copy diagonal & super-diagonal.
    int64_t D_index = 0;
    int64_t E_index = 0;
    for (int64_t i = 0; i < nt; ++i) {
        // Copy 1 element from super-diagonal tile to E.
        if (i > 0) {
            auto T = A(i-1, i);
            E[E_index] = real( T(T.mb()-1, 0) );
            E_index += 1;
            A.tileTick(i-1, i);
        }

        // Copy main diagonal to D.
        auto T = A(i, i);
        slate_assert(T.mb() == T.nb()); // square diagonal tile
        auto len = T.nb();
        for (int j = 0; j < len; ++j) {
            D[D_index + j] = real( T(j, j) );
        }
        D_index += len;

        // Copy super-diagonal to E.
        for (int j = 0; j < len-1; ++j) {
            E[E_index + j] = real( T(j, j+1) );
        }
        E_index += len-1;
        A.tileTick(i, i);
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void copytb2bd<Target::HostTask, float>(
    TriangularBandMatrix<float>& A,
    std::vector<float>& D,
    std::vector<float>& E);

// ----------------------------------------
template
void copytb2bd<Target::HostTask, double>(
    TriangularBandMatrix<double>& A,
    std::vector<double>& D,
    std::vector<double>& E);

// ----------------------------------------
template
void copytb2bd< Target::HostTask, std::complex<float> >(
    TriangularBandMatrix< std::complex<float> >& A,
    std::vector<float>& D,
    std::vector<float>& E);

// ----------------------------------------
template
void copytb2bd< Target::HostTask, std::complex<double> >(
    TriangularBandMatrix< std::complex<double> >& A,
    std::vector<double>& D,
    std::vector<double>& E);

} // namespace internal
} // namespace slate
