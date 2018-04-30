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

#ifndef SLATE_SYMMETRIC_MATRIX_HH
#define SLATE_SYMMETRIC_MATRIX_HH

#include "slate_BaseTrapezoidMatrix.hh"
#include "slate_Matrix.hh"
#include "slate_Tile.hh"
#include "slate_types.hh"

#include "lapack.hh"

#include <algorithm>
#include <utility>

#include "slate_mpi.hh"

namespace slate {

//==============================================================================
/// Symmetric, n-by-n, distributed, tiled matrices.
template <typename scalar_t>
class SymmetricMatrix: public BaseTrapezoidMatrix< scalar_t > {
public:
    // constructors
    SymmetricMatrix();

    static
    SymmetricMatrix fromLAPACK(Uplo uplo, int64_t n,
                               scalar_t* A, int64_t lda, int64_t nb,
                               int p, int q, MPI_Comm mpi_comm);

    static
    SymmetricMatrix fromScaLAPACK(Uplo uplo, int64_t n,
                                  scalar_t* A, int64_t lda, int64_t nb,
                                  int p, int q, MPI_Comm mpi_comm);

    // conversion
    explicit SymmetricMatrix(BaseTrapezoidMatrix< scalar_t >& orig);

    SymmetricMatrix(Uplo uplo, Matrix< scalar_t >& orig);

    // sub-matrix
    SymmetricMatrix sub(int64_t i1, int64_t i2);

    Matrix< scalar_t > sub(int64_t i1, int64_t i2, int64_t j1, int64_t j2);

protected:
    // used by fromLAPACK
    SymmetricMatrix(Uplo uplo, int64_t n,
                    scalar_t* A, int64_t lda, int64_t nb,
                    int p, int q, MPI_Comm mpi_comm);

    // used by fromScaLAPACK
    SymmetricMatrix(Uplo uplo, int64_t n,
                    scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
                    int p, int q, MPI_Comm mpi_comm);

    // used by sub
    SymmetricMatrix(SymmetricMatrix& orig,
                    int64_t i1, int64_t i2);

public:
    template <typename T>
    friend void swap(SymmetricMatrix<T>& A, SymmetricMatrix<T>& B);
};

//------------------------------------------------------------------------------
/// Default constructor creates an empty matrix.
template <typename scalar_t>
SymmetricMatrix<scalar_t>::SymmetricMatrix()
    : BaseTrapezoidMatrix< scalar_t >()
{}

//------------------------------------------------------------------------------
/// [static]
/// Named constructor returns a new Matrix from LAPACK layout.
/// Construct matrix by wrapping existing memory of an n-by-n lower
/// or upper symmetric LAPACK-style matrix.
///
/// @param[in] in_uplo
///     - Upper: upper triangle of A is stored.
///     - Lower: lower triangle of A is stored.
///
/// @param[in] n
///     Number of rows and columns of the matrix. n >= 0.
///
/// @param[in,out] A
///     The n-by-n symmetric matrix A, stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of the array A. lda >= m.
///
/// @param[in] nb
///     Block size in 2D block-cyclic distribution.
///
/// @param[in] p
///     Number of block rows in 2D block-cyclic distribution. p > 0.
///
/// @param[in] q
///     Number of block columns of 2D block-cyclic distribution. q > 0.
///
/// @param[in] mpi_comm
///     MPI communicator to distribute matrix across.
///     p*q == MPI_Comm_size(mpi_comm).
template <typename scalar_t>
SymmetricMatrix<scalar_t> SymmetricMatrix<scalar_t>::fromLAPACK(
    Uplo uplo, int64_t n,
    scalar_t* A, int64_t lda, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
{
    return SymmetricMatrix<scalar_t>(uplo, n, A, lda, nb, p, q, mpi_comm);
}

//------------------------------------------------------------------------------
/// [static]
/// Named constructor returns a new Matrix from ScaLAPACK layout.
/// Construct matrix by wrapping existing memory of an n-by-n lower
/// or upper symmetric ScaLAPACK-style matrix.
/// @see BaseTrapezoidMatrix
///
/// @param[in] in_uplo
///     - Upper: upper triangle of A is stored.
///     - Lower: lower triangle of A is stored.
///
/// @param[in] n
///     Number of rows and columns of the matrix. n >= 0.
///
/// @param[in,out] A
///     The local portion of the 2D block cyclic distribution of
///     the n-by-n matrix A, with local leading dimension lda.
///
/// @param[in] lda
///     Local leading dimension of the array A. lda >= local number of rows.
///
/// @param[in] nb
///     Block size in 2D block-cyclic distribution. nb > 0.
///
/// @param[in] p
///     Number of block rows in 2D block-cyclic distribution. p > 0.
///
/// @param[in] q
///     Number of block columns of 2D block-cyclic distribution. q > 0.
///
/// @param[in] mpi_comm
///     MPI communicator to distribute matrix across.
///     p*q == MPI_Comm_size(mpi_comm).
template <typename scalar_t>
SymmetricMatrix<scalar_t> SymmetricMatrix<scalar_t>::fromScaLAPACK(
    Uplo uplo, int64_t n,
    scalar_t* A, int64_t lda, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
{
    // note extra nb
    return SymmetricMatrix<scalar_t>(uplo, n, A, lda, nb, nb, p, q, mpi_comm);
}

//------------------------------------------------------------------------------
/// @see fromLAPACK
template <typename scalar_t>
SymmetricMatrix<scalar_t>::SymmetricMatrix(
    Uplo uplo, int64_t n,
    scalar_t* A, int64_t lda, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : BaseTrapezoidMatrix< scalar_t >(uplo, n, n, A, lda, nb, p, q, mpi_comm)
{}

//------------------------------------------------------------------------------
/// @see fromScaLAPACK
/// This differs from LAPACK constructor by adding mb.
template <typename scalar_t>
SymmetricMatrix<scalar_t>::SymmetricMatrix(
    Uplo uplo, int64_t n,
    scalar_t* A, int64_t lda, int64_t mb, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : BaseTrapezoidMatrix< scalar_t >(uplo, n, n, A, lda, mb, nb, p, q, mpi_comm)
{}

//------------------------------------------------------------------------------
/// [explicit]
/// Conversion from trapezoid, triangular, symmetric, or Hermitian matrix
/// creates a shallow copy view of the original matrix.
/// Uses only square portion, Aorig[ 0:min(mt,nt)-1, 0:min(mt,nt)-1 ].
///
/// @param[in,out] orig
///     Original matrix.
template <typename scalar_t>
SymmetricMatrix<scalar_t>::SymmetricMatrix(
    BaseTrapezoidMatrix< scalar_t >& orig)
    : BaseTrapezoidMatrix< scalar_t >(
        orig,
        0, std::min( orig.mt()-1, orig.nt()-1 ),
        0, std::min( orig.mt()-1, orig.nt()-1 ))
{}

//------------------------------------------------------------------------------
/// Conversion from general matrix
/// creates a shallow copy view of the original matrix.
/// Uses only square portion, Aorig[ 0:min(mt,nt)-1, 0:min(mt,nt)-1 ].
///
/// @param[in] in_uplo
///     - Upper: upper triangle of A is stored.
///     - Lower: lower triangle of A is stored.
///
/// @param[in,out] orig
///     Original matrix.
template <typename scalar_t>
SymmetricMatrix<scalar_t>::SymmetricMatrix(
    Uplo uplo, Matrix< scalar_t >& orig)
    : BaseTrapezoidMatrix< scalar_t >(
        uplo, orig,
        0, std::min( orig.mt()-1, orig.nt()-1 ),
        0, std::min( orig.mt()-1, orig.nt()-1 ))
{}

//------------------------------------------------------------------------------
/// Sub-matrix constructor creates shallow copy view of parent matrix,
/// A[ i1:i2, i1:i2 ]. The new view is still a symmetric matrix, with the
/// same diagonal as the parent matrix.
///
/// @param[in,out] orig
///     Original matrix.
///
/// @param[in] i1
///     Starting block row and column index. 0 <= i1 < mt.
///
/// @param[in] i2
///     Ending block row and column index (inclusive). i2 < mt.
template <typename scalar_t>
SymmetricMatrix<scalar_t>::SymmetricMatrix(
    SymmetricMatrix<scalar_t>& orig,
    int64_t i1, int64_t i2)
    : BaseTrapezoidMatrix< scalar_t >(orig, i1, i2, i1, i2)
{}

//------------------------------------------------------------------------------
/// Returns sub-matrix that is a shallow copy view of the
/// parent matrix, A[ i1:i2, i1:i2 ].
/// This version returns a SymmetricMatrix with the same diagonal as the
/// parent matrix.
/// @see Matrix TrapezoidMatrix::sub(int64_t i1, int64_t i2,
///                                  int64_t j1, int64_t j2)
///
/// @param[in] i1
///     Starting block row and column index. 0 <= i1 < mt.
///
/// @param[in] i2
///     Ending block row and column index (inclusive). i2 < mt.
template <typename scalar_t>
SymmetricMatrix<scalar_t> SymmetricMatrix<scalar_t>::sub(
    int64_t i1, int64_t i2)
{
    return SymmetricMatrix<scalar_t>(*this, i1, i2);
}

//------------------------------------------------------------------------------
/// Returns off-diagonal sub-matrix that is a shallow copy view of the
/// parent matrix, A[ i1:i2, j1:j2 ].
/// This version returns a general Matrix, which:
/// - if uplo = Lower, is strictly below the diagonal, or
/// - if uplo = Upper, is strictly above the diagonal.
/// @see TrapezoidMatrix sub(int64_t i1, int64_t i2)
///
/// @param[in] i1
///     Starting block row index. 0 <= i1 < mt.
///
/// @param[in] i2
///     Ending block row index (inclusive). i2 < mt.
///
/// @param[in] j1
///     Starting block column index. 0 <= j1 < nt.
///
/// @param[in] j2
///     Ending block column index (inclusive). j2 < nt.
template <typename scalar_t>
Matrix< scalar_t > SymmetricMatrix<scalar_t>::sub(
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
{
    return BaseTrapezoidMatrix< scalar_t >::sub(i1, i2, j1, j2);
}

//------------------------------------------------------------------------------
/// Swaps contents of matrices A and B.
//
// (This isn't really needed over BaseTrapezoidMatrix swap, but is here as a
// reminder in case any members are added that aren't in BaseTrapezoidMatrix.)
template <typename scalar_t>
void swap(SymmetricMatrix<scalar_t>& A, SymmetricMatrix<scalar_t>& B)
{
    using std::swap;
    swap(static_cast< BaseTrapezoidMatrix< scalar_t >& >(A),
         static_cast< BaseTrapezoidMatrix< scalar_t >& >(B));
}

} // namespace slate

#endif // SLATE_SYMMETRIC_MATRIX_HH
