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
#include "slate/TriangularMatrix.hh"
#include "slate/types.hh"
#include "slate/Tile_blas.hh"
#include "internal/internal.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// General matrix copy to triangularband matrix,
/// Dispatches to target implementations.
/// In the complex case,
/// @ingroup copyge2tb_internal
///
template <Target target, typename scalar_t>
void copyge2tb(Matrix<scalar_t>&& A,
        TriangularBandMatrix<scalar_t>&& B)

{
    copyge2tb(internal::TargetType<target>(),
               A,
               B);
}

//------------------------------------------------------------------------------
/// General matrix copy to triangularband matrix,
/// Host OpenMP task implementation.
/// @ingroup copyge2tb_internal
///
template <typename scalar_t>
void copyge2tb(internal::TargetType<Target::HostTask>, 
           Matrix<scalar_t>& A,
           TriangularBandMatrix<scalar_t>& B)
{
    trace::Block trace_block("slate::copyge2tb");

    // make sure it is a bi-diagobal matrix
    slate_assert(B.bandwidth() == 1);

    const int64_t nb = B.tileNb(0);
    //int64_t ku = B.upperBandwidth(); // For the bidiagonal reduction the ku=nb ===> kdt = 1.
    int64_t ku = nb; // For the bidiagonal reduction the ku=nb ===> kdt = 1.
    int kdt = slate::ceildiv( ku, nb );
    // over-estimate of # tiles


    int index = 0; // index in Ad storage
    int jj = 0; // col index
    for (int j = 0; j < B.nt(); ++j) {
        int ii = 0; // row index
        for (int i = 0; i < B.mt(); ++i) {
            if (B.tileIsLocal(i, j) &&
                ((ii == jj) ||
                 ( ii < jj && (jj - (ii + B.tileMb(i) - 1)) <= (B.bandwidth()+1) ) ) )
            {
                auto T_ptr = B.tileInsert( i, j );
                T_ptr->uplo(slate::Uplo::General);
                index += 1;

                if (i > 0 && i == j) {
                    auto T_ptr = B.tileInsert( i, j-1 );
                    T_ptr->uplo(slate::Uplo::General);
                    auto T = B(i, j-1);
                    lapack::laset(lapack::MatrixType::General, T.mb(), T.nb(),
                          0, 0, T.data(), T.stride());
                }

                auto T = B(i, j);
                lapack::laset(lapack::MatrixType::General, T.mb(), T.nb(),
                      0, 0, T.data(), T.stride());
            }
            ii += B.tileMb(i);
        }
        jj += B.tileNb(j);
    }

    for (int64_t j = 0; j < B.nt(); ++j) {
        int64_t istart = blas::max( 0, j-kdt );
        int64_t iend   = j;
        for (int64_t i = istart; i <= iend; ++i) {
            if (B.tileIsLocal(i, j)) {
            //#pragma omp task shared(A, B) priority(priority)
                #pragma omp parallel
                {
                    if (i == j) { // Copy the upper part of the diagonal tiles 
                        //tzcopy( Lower, m, n, A(j, j), lda, B(nb, j), ldb);
                        lapack::lacpy(lapack::MatrixType::Upper,
                              A(i, j).mb(), A(i, j).nb(),
                              A(i, j).data(), A(j, i).stride(),
                              B(i, j).data(), B(i, j).stride() );
                    }
                    else { // Copy the lower part of the superdiagonal tile 
                        lapack::lacpy(lapack::MatrixType::Lower,
                              A(i, j).mb(), A(i, j).nb(),
                              A(i, j).data(), A(j, i).stride(),
                              B(i, j).data(), B(i, j).stride() );
                    }
                    B.tileModified(i, j);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void copyge2tb<Target::HostTask, float>(
    Matrix<float>&& A,
    TriangularBandMatrix<float>&& B);

// ----------------------------------------
template
void copyge2tb<Target::HostTask, double>(
    Matrix<double>&& A,
    TriangularBandMatrix<double>&& B);

// ----------------------------------------
template
void copyge2tb< Target::HostTask, std::complex<float> >(
    Matrix< std::complex<float> >&& A,
    TriangularBandMatrix< std::complex<float> >&& B);

// ----------------------------------------
template
void copyge2tb< Target::HostTask, std::complex<double> >(
     Matrix< std::complex<double> >&& A,
     TriangularBandMatrix< std::complex<double> >&& B);

} // namespace internal
} // namespace slate
