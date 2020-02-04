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
#include "aux/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Distributed parallel reduction to band for 3-stage SVD.
///
/// Reduces an n-by-n Hermitian matrix $A$ to band form using unitary
/// transformations. The factorization has the form
///
/// op              |  side = Left  |  side = Right
/// --------------- | ------------- | --------------
/// op = NoTrans    |  $Q C  $      |  $C Q  $
/// op = ConjTrans  |  $Q^H C$      |  $C Q^H$
///
/// where $Q$ is unitary and $C$ is Hermitian band
/// with nb sub and superdiagonals.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex< float >, std::complex< double >.
//------------------------------------------------------------------------------
/// @param[in] A
///     On entry, the n-by-n Hermitian matrix $A$.
///
/// @param[in] T
///     On entry, triangular matrices of the .
///
/// @param[in,out] C
///     On entry, the m-by-n matrix $C$.
///     On exit, $C$ is overwritten by $Q C$, $Q^H C$, $C Q$, or $C Q^H$.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// @ingroup unmtr_he2hb
///
template < typename scalar_t >
void unmtr_he2hb(
    Side side, Uplo uplo, Op op,
    HermitianMatrix< scalar_t > A,
    TriangularFactors< scalar_t > T,
    Matrix< scalar_t >& C,
    const std::map< Option, Value >& opts)
{
    int64_t C_sub_start_row;
    int64_t C_sub_start_col;

    if (op == Op::NoTrans) {
        C_sub_start_row = 1;
        C_sub_start_col = 0;
    }
    else if (op == Op::ConjTrans) {
        C_sub_start_row = 0;
        C_sub_start_col = 1;
    }
    else {
        throw std::runtime_error("(op == slate::Op::Trans) cann't be used.");
    }

    auto C_sub = Matrix< scalar_t >( C, C_sub_start_row, A.nt()-1,
                                        C_sub_start_col, A.nt()-1 );
    slate::TriangularFactors< scalar_t > T_sub = {
        T[ 0 ].sub( 1, A.nt()-1, 0, A.nt()-1 ),
        T[ 1 ].sub( 1, A.nt()-1, 0, A.nt()-1 ) };

    if ( uplo == Uplo::Upper ) {
        auto Q = Matrix< scalar_t >( A, 0, A.nt()-1, 1, A.nt()-1 );
        slate::unmlq( side, op, Q, T_sub, C_sub, opts );
    }
    else { // uplo == Uplo::Lower
        auto Q = Matrix< scalar_t >( A, 1, A.nt()-1, 0, A.nt()-1 );
        slate::unmqr( side, op, Q, T_sub, C_sub, opts );
    }

    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void unmtr_he2hb< float >(
    Side side, Uplo uplo, Op op,
    HermitianMatrix< float > A,
    TriangularFactors< float > T,
    Matrix< float >& C,
    const std::map< Option, Value >& opts);

template
void unmtr_he2hb< double >(
    Side side, Uplo uplo, Op op,
    HermitianMatrix< double > A,
    TriangularFactors< double > T,
    Matrix< double >& C,
    const std::map< Option, Value >& opts);

template
void unmtr_he2hb< std::complex< float > >(
    Side side, Uplo uplo, Op op,
    HermitianMatrix< std::complex< float > > A,
    TriangularFactors< std::complex< float > > T,
    Matrix< std::complex< float > >& C,
    const std::map< Option, Value >& opts);

template
void unmtr_he2hb< std::complex< double > >(
    Side side, Uplo uplo, Op op,
    HermitianMatrix< std::complex< double > > A,
    TriangularFactors< std::complex< double > > T,
    Matrix< std::complex< double > >& C,
    const std::map< Option, Value >& opts);

} // namespace slate
