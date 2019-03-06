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
#include "aux/slate_Debug.hh"
#include "slate/slate_Matrix.hh"
#include "slate/slate_Tile_blas.hh"
#include "slate/slate_TriangularMatrix.hh"
#include "internal/slate_internal.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Distributed parallel least squares solve via QR or LQ factorization.
/// op(A) is either A, or A^H, or A^T (only if A is real).
/// op(A) is m-by-n, X is n-by-nrhs, B is m-by-nrhs, BX is max(m, n)-by-nrhs.
///
/// If m >= n, solves over-determined op(A) X = B
/// with least squares solution X that minimizes || op(A) X - B ||_2.
/// BX is m-by-nrhs.
/// On input, B is all m rows of BX.
/// On output, X is first n rows of BX.
///
/// If m < n, solves under-determined op(A) X = B
/// with minimum norm solution X that minimizes || X ||_2.
/// BX is n-by-nrhs.
/// On input, B is first m rows of BX.
/// On output, X is all n rows of BX.
///
/// Note these (m, n) differ from (Sca)LAPACK, where A is M-by-N,
/// while here op(A) is m-by-n.
///
template <typename scalar_t>
void gels(Matrix<scalar_t>& opA,
          TriangularFactors<scalar_t>& T,
          Matrix<scalar_t>& BX,
          const std::map<Option, Value>& opts)
{
    // m, n of op(A) as in docs above.
    int64_t m = opA.m();
    int64_t n = opA.n();

    scalar_t one  = 1;
    scalar_t zero = 0;

    // Get original, un-transposed matrix A.
    slate::Matrix<scalar_t> A;
    if (opA.op() == Op::NoTrans)
        A = opA;
    else if (opA.op() == Op::ConjTrans)
        A = conj_transpose(opA);
    else if (opA.op() == Op::Trans && opA.is_real)
        A = transpose(opA);
    else
        slate_error("Unsupported op(A)");

    if (A.m() >= A.n()) {
        // A itself is tall: QR factorization
        geqrf(A, T, opts);
        // todo: need to take submatrix that splits tiles.
        auto R = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, A);

        if (opA.op() == Op::NoTrans) {
            // Solve A X = (QR) X = B.
            // Least squares solution X = R^{-1} Y = R^{-1} (Q^H B).

            // Y = Q^H B
            // B is all m rows of BX.
            unmqr(Side::Left, Op::ConjTrans, A, T, BX, opts);

            // X is only first n rows of BX.
            // todo: need to take submatrix that splits tiles.
            slate_assert(BX.tileMb(opA.nt()-1) == A.tileNb(A.nt()-1));
            auto X = BX.sub(0, opA.nt()-1, 0, BX.nt()-1);

            // X = R^{-1} Y
            trsm(Side::Left, one, R, X, opts);
        }
        else {
            // Solve A^H X = (QR)^H X = B.
            // Minimum norm solution X = Q Y = Q (R^{-H} B).

            // B is only first m rows of BX.
            // todo: need to take submatrix that splits tiles.
            slate_assert(BX.tileMb(opA.mt()-1) == opA.tileMb(opA.mt()-1));
            auto B = BX.sub(0, opA.mt()-1, 0, BX.nt()-1);

            // Y = R^{-H} B
            auto RH = conj_transpose(R);
            trsm(Side::Left, one, RH, B, opts);

            // X is all n rows of BX.
            // Zero out rows m:n-1 of BX.
            // todo: istart/ioffset assumes fixed nb
            int64_t istart  = opA.m() / BX.tileMb(0); // row m's tile
            int64_t ioffset = opA.m() % BX.tileMb(0); // row m's offset in tile
            for (int64_t i = istart; i < BX.mt(); ++i) {
                for (int64_t j = 0; j < BX.nt(); ++j) {
                    if (BX.tileIsLocal(i, j)) {
                        auto T = BX(i, j);
                        lapack::laset(lapack::MatrixType::General,
                                      T.mb() - ioffset, T.nb(), zero, zero,
                                      &T.at(ioffset, 0), T.stride());
                        BX.tileModified(i, j);
                    }
                }
                ioffset = 0; // no offset for subsequent block rows
            }

            // X = Q Y
            unmqr(Side::Left, Op::NoTrans, A, T, BX, opts);
        }
    }
    else {
        // todo: LQ factorization
        slate_assert(false);
    }
    // todo: return value for errors?
    // R or L is singular => A is not full rank
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gels<float>(
    Matrix<float>& A,
    TriangularFactors<float>& T,
    Matrix<float>& B,
    const std::map<Option, Value>& opts);

template
void gels<double>(
    Matrix<double>& A,
    TriangularFactors<double>& T,
    Matrix<double>& B,
    const std::map<Option, Value>& opts);

template
void gels< std::complex<float> >(
    Matrix< std::complex<float> >& A,
    TriangularFactors< std::complex<float> >& T,
    Matrix< std::complex<float> >& B,
    const std::map<Option, Value>& opts);

template
void gels< std::complex<double> >(
    Matrix< std::complex<double> >& A,
    TriangularFactors< std::complex<double> >& T,
    Matrix< std::complex<double> >& B,
    const std::map<Option, Value>& opts);

} // namespace slate
