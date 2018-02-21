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

#include "slate.hh"
#include "slate_Debug.hh"
#include "slate_Matrix.hh"
#include "slate_internal.hh"

namespace slate {

namespace internal_specialization {

///-----------------------------------------------------------------------------
/// \brief
/// Distributed parallel Cholesky factorization.
/// Generic implementation for any target.
template <Target target, typename scalar_t>
void potrf(slate::internal::TargetType<target>,
           HermitianMatrix<scalar_t>& A, int64_t lookahead)
{
#if 0
    using namespace blas;

    uint8_t *column = new uint8_t[ A.nt() ];

    #pragma omp parallel
    #pragma omp master
    for (int64_t k = 0; k < A.nt(); ++k) {
        // panel
        #pragma omp task depend(inout:column[k]) priority(1)
        {
            internal::potrf<Target::HostTask>(A.sub(k, k), 1);

            if (k+1 <= A.nt()-1)
                A.tileBcast(k, k, A.sub(k+1, A.nt()-1, k, k));

            if (k+1 <= A.nt()-1)
                internal::trsm<Target::HostTask>(
                    Side::Right, Diag::NonUnit,
                    1.0, TriangularMatrix(A.sub(k, k)),
                         A.sub(k+1, A.nt()-1, k, k), 1);

            for (int64_t i = k+1; i < A.nt(); ++i) {
                A.tileBcast(i, k, A.sub(i, i, k+1, i),
                                  A.sub(i, A.nt()-1, i, i));
            }
        }
        // lookahead column(s)
        for (int64_t j = k+1; j < k+1+lookahead && j < A.nt(); ++j) {
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[j]) priority(1)
            {
                internal::syrk<Target::HostTask>(
                    -1.0, A.sub(j, j, k, k),
                     1.0, A.sub(j, j, j, j), 1);

                if (j+1 <= A.nt()-1)
                    internal::gemm<Target::HostTask>(
                        -1.0, A(j+1, A.nt()-1, k, k),
                              conj_transpose(A(j, j, k, k)),
                         1.0, A(j+1, A.nt()-1, j, j), 1);
            }
        }
        // trailing submatrix
        if (k+1+lookahead < A.nt())
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[k+1+lookahead]) \
                             depend(inout:column[A.nt()-1])
            {
                internal::syrk<target>(
                    Uplo::Lower, Op::NoTrans,
                    -1.0, A.sub(k+1+lookahead, A.nt()-1, k, k),
                     1.0, A.sub(k+1+lookahead, A.nt()-1, k+1+lookahead, A.nt()-1));
            }
    }

    //Debug::checkTilesLives(A);
    //Debug::printTilesLives(A);

    A.clearWorkspace();

    //Debug::printTilesMaps(A);

    delete[] column;
#endif
}

///-----------------------------------------------------------------------------
/// \brief
/// Distributed parallel Cholesky factorization.
/// Implementation for GPU device target.
template <typename scalar_t>
void potrf(slate::internal::TargetType<Target::Devices>,
           HermitianMatrix<scalar_t>& A, int64_t lookahead)
{
#if 0
    using namespace blas;

    uint8_t *column = new uint8_t[ A.nt() ];

    for (int device = 0; device < A.num_devices_; ++device)
        A.reserve(device, A.getMaxDeviceTiles(device));

    #pragma omp parallel
    #pragma omp master
    for (int64_t k = 0; k < A.nt(); ++k) {
        // panel
        #pragma omp task depend(inout:column[k])
        {
            internal::potrf<Target::HostTask>(A.uplo(), A(k, k, k, k));

            if (k+1 <= A.nt()-1)
                A.tileBcast(k, k, A(k+1, A.nt()-1, k, k));

            if (k+1 <= A.nt()-1) {
                trsm<Target::HostTask>(
                    Side::Right, Diag::NonUnit,
                    1.0, A.sub(k, k, k, k),
                         A.sub(k+1, A.nt()-1, k, k));
            }

            for (int64_t i = k+1; i < A.nt(); ++i) {
                A.template tileBcast<Target::Devices>(
                    i, k, A.sub(i, i, k+1, i),
                          A.sub(i, A.nt()-1, i, i));
            }
        }
        // trailing submatrix
        if (k+1+lookahead < A.nt())
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[k+1+lookahead]) \
                             depend(inout:column[A.nt()-1])
            {
                syrk<Target::Devices>(
                    Uplo::Lower, Op::NoTrans,
                    -1.0, A.sub(k+1+lookahead, A.nt()-1, k, k),
                     1.0, A.sub(k+1+lookahead, A.nt()-1, k+1+lookahead, A.nt()-1));
            }

        // lookahead column(s)
        for (int64_t j = k+1; j < k+1+lookahead && j < A.nt(); ++j) {
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[j])
            {
                syrk<Target::HostTask>(
                    Uplo::Lower, Op::NoTrans,
                    -1.0, A.sub(j, j, k, k),
                     1.0, A.sub(j, j, j, j));

                if (j+1 <= A.nt()-1)
                    gemm<Target::HostTask>(
                        Op::NoTrans, Op::Trans,
                        -1.0, A.sub(j+1, A.nt()-1, k, k),
                              A.sub(j, j, k, k),
                         1.0, A.sub(j+1, A.nt()-1, j, j));
            }
        }
    }

    for (int device = 0; device < A.num_devices_; ++device)
        A.memory_->clearDeviceBlocks(device);

    Debug::checkTilesLives(A);
    Debug::printTilesLives(A);

    A.clearWorkspace();

    Debug::printTilesMaps(A);

    delete[] column;
#endif
}

} // namespace internal

///-----------------------------------------------------------------------------
/// \brief
///
/// Precision and target templated function.
template <Target target, typename scalar_t>
void potrf(HermitianMatrix<scalar_t>& A, int64_t lookahead)
{
    internal_specialization::potrf(internal::TargetType<target>(), A, lookahead);
}

//------------------------------------------------------------------------------
// Explicit instantiations for double precision and various targets.
template
void potrf< Target::HostTask, double >(
    HermitianMatrix<double>& A, int64_t lookahead);

template
void potrf< Target::HostNest, double >(
    HermitianMatrix<double>& A, int64_t lookahead);

template
void potrf< Target::HostBatch, double >(
    HermitianMatrix<double>& A, int64_t lookahead);

template
void potrf< Target::Devices, double >(
    HermitianMatrix<double>& A, int64_t lookahead);

} // namespace slate
