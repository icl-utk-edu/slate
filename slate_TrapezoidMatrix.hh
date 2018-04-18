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
// Need assistance with the SLATE software? Join the "SLATE User" Google group
// by going to https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user
// and clicking "Apply to join group". Upon acceptance, email your questions and
// comments to <slate-user@icl.utk.edu>.
//------------------------------------------------------------------------------

#ifndef SLATE_TRAPEZOID_MATRIX_HH
#define SLATE_TRAPEZOID_MATRIX_HH

#include "slate_BaseTrapezoidMatrix.hh"
#include "slate_Matrix.hh"
#include "slate_Tile.hh"
#include "slate_types.hh"

#include "lapack.hh"

#include <algorithm>
#include <functional>
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include <iostream>

#ifdef SLATE_WITH_MPI
    #include <mpi.h>
#else
    #include "slate_NoMpi.hh"
#endif

namespace slate {

///=============================================================================
///
template <typename scalar_t>
class TrapezoidMatrix: public BaseTrapezoidMatrix< scalar_t > {
public:
    ///-------------------------------------------------------------------------
    /// Default constructor
    TrapezoidMatrix():
        BaseTrapezoidMatrix< scalar_t >()
    {}

    ///-------------------------------------------------------------------------
    /// Named constructor returns new TrapezoidMatrix.
    /// Construct matrix by wrapping existing memory of an m-by-n lower
    /// or upper trapezoidal LAPACK-style matrix.
    /// @see BaseTrapezoidMatrix
    static
    TrapezoidMatrix fromLAPACK(Uplo uplo, int64_t m, int64_t n,
                               scalar_t* A, int64_t lda, int64_t nb,
                               int p, int q, MPI_Comm mpi_comm)
    {
        return TrapezoidMatrix(uplo, m, n, A, lda, nb, p, q, mpi_comm);
    }

    ///-------------------------------------------------------------------------
    /// Named constructor returns new TrapezoidMatrix.
    /// Construct matrix by wrapping existing memory of an m-by-n lower
    /// or upper trapezoidal ScaLAPACK-style matrix.
    /// @see BaseTrapezoidMatrix
    static
    TrapezoidMatrix fromScaLAPACK(Uplo uplo, int64_t m, int64_t n,
                                  scalar_t* A, int64_t lda, int64_t nb,
                                  int p, int q, MPI_Comm mpi_comm)
    {
        // note extra nb
        return TrapezoidMatrix(uplo, m, n, A, lda, nb, nb, p, q, mpi_comm);
    }

    ///-------------------------------------------------------------------------
    /// @see fromLAPACK
    /// todo: make this protected
    TrapezoidMatrix(Uplo uplo, int64_t m, int64_t n,
                    scalar_t* A, int64_t lda, int64_t nb,
                    int p, int q, MPI_Comm mpi_comm):
        BaseTrapezoidMatrix< scalar_t >(uplo, m, n, A, lda, nb, p, q, mpi_comm)
    {}

    ///-------------------------------------------------------------------------
    /// @see fromScaLAPACK
    /// This differs from LAPACK constructor by adding mb.
    /// todo: make this protected
    TrapezoidMatrix(Uplo uplo, int64_t m, int64_t n,
                    scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
                    int p, int q, MPI_Comm mpi_comm):
        BaseTrapezoidMatrix< scalar_t >(uplo, m, n, A, lda, mb, nb, p, q, mpi_comm)
    {}

    ///-------------------------------------------------------------------------
    /// Conversion from trapezoid, triangular, symmetric, or Hermitian matrix
    /// creates a shallow copy view of the original matrix.
    explicit TrapezoidMatrix(BaseTrapezoidMatrix< scalar_t >& orig):
        BaseTrapezoidMatrix< scalar_t >(orig)
    {}

    ///-------------------------------------------------------------------------
    /// Conversion from trapezoid, triangular, symmetric, or Hermitian matrix
    /// creates a shallow copy view of the original matrix, A[ i1:i2, j1:j2 ].
    TrapezoidMatrix(BaseTrapezoidMatrix< scalar_t >& orig,
                    int64_t i1, int64_t i2,
                    int64_t j1, int64_t j2):
        BaseTrapezoidMatrix< scalar_t >(orig, i1, i2, j1, j2)
    {}

    ///-------------------------------------------------------------------------
    /// Conversion from general matrix
    /// creates a shallow copy view of the original matrix.
    TrapezoidMatrix(Uplo uplo, Matrix< scalar_t >& orig):
        BaseTrapezoidMatrix< scalar_t >(uplo, orig)
    {}

    ///-------------------------------------------------------------------------
    /// Conversion from general matrix, sub-matrix constructor
    /// creates shallow copy view of original matrix, A[ i1:i2, j1:j2 ].
    TrapezoidMatrix(Uplo uplo, Matrix< scalar_t >& orig,
                    int64_t i1, int64_t i2,
                    int64_t j1, int64_t j2):
        BaseTrapezoidMatrix< scalar_t >(uplo, orig, i1, i2, j1, j2)
    {}

    ///-------------------------------------------------------------------------
    /// Sub-matrix constructor creates shallow copy view of parent matrix,
    /// A[ i1:i2, j1:j2 ]. Requires i1 == j1. The new view is still a trapezoid
    /// matrix, with the same diagonal as the parent matrix.
    TrapezoidMatrix(TrapezoidMatrix& orig,
                    int64_t i1, int64_t i2,
                    int64_t j1, int64_t j2):
        BaseTrapezoidMatrix< scalar_t >(orig, i1, i2, j1, j2)
    {}

    ///-------------------------------------------------------------------------
    /// @return diagonal sub-matrix that is a shallow copy view of the
    /// parent matrix, A[ i1:i2, i1:i2 ].
    /// This version returns a TrapezoidMatrix with the same diagonal as the
    /// parent matrix.
    /// @see Matrix TrapezoidMatrix::sub(int64_t i1, int64_t i2,
    ///                                  int64_t j1, int64_t j2)
    TrapezoidMatrix sub(int64_t i1, int64_t i2)
    {
        return TrapezoidMatrix(*this, i1, i2, i1, i2);
    }

    ///-------------------------------------------------------------------------
    /// @return off-diagonal sub-matrix that is a shallow copy view of the
    /// parent matrix, A[ i1:i2, j1:j2 ].
    /// This version returns a general Matrix, which:
    /// - if uplo = Lower, is strictly below the diagonal, or
    /// - if uplo = Upper, is strictly above the diagonal.
    /// @see TrapezoidMatrix sub(int64_t i1, int64_t i2)
    Matrix< scalar_t > sub(int64_t i1, int64_t i2, int64_t j1, int64_t j2)
    {
        return BaseTrapezoidMatrix< scalar_t >::sub(i1, i2, j1, j2);
    }

    ///-------------------------------------------------------------------------
    /// Swap contents of matrices A and B.
    // (This isn't really needed over BaseTrapezoidMatrix swap, but is here as a
    // reminder in case any members are added that aren't in BaseTrapezoidMatrix.)
    friend void swap(TrapezoidMatrix& A, TrapezoidMatrix& B)
    {
        using std::swap;
        swap(static_cast< BaseTrapezoidMatrix< scalar_t >& >(A),
             static_cast< BaseTrapezoidMatrix< scalar_t >& >(B));
    }
};

} // namespace slate

#endif // SLATE_TRAPEZOID_MATRIX_HH
