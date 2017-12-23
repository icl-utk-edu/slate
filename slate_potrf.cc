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

#include "slate_Debug.hh"
#include "slate_Matrix.hh"

namespace slate {

namespace internal {

//------------------------------------------------------------------------------
template <typename FloatType, Target target>
void potrf(TargetType<target>,
           blas::Uplo uplo, Matrix<FloatType> a, int64_t lookahead)
{
    using namespace blas;

    uint8_t *column;

    #pragma omp parallel
    #pragma omp master
    for (int64_t k = 0; k < a.nt_; ++k) {
        // panel
        #pragma omp task depend(inout:column[k]) priority(1)
        {
            Matrix<FloatType>::template
            potrf<Target::HostTask>(uplo, a(k, k, k, k));

            if (k+1 <= a.nt_-1)
                a.tileSend(k, k, {k+1, a.nt_-1, k, k});

            if (k+1 <= a.nt_-1)
                Matrix<FloatType>::template
                trsm<Target::HostTask>(
                    Side::Right, Uplo::Lower,
                    Op::Trans, Diag::NonUnit,
                    1.0, a(k, k, k, k),
                         a(k+1, a.nt_-1, k, k));

            for (int64_t m = k+1; m < a.nt_; ++m)
                a.tileSend(m, k, {m, m, k+1, m},
                                 {m, a.nt_-1, m, m});
        }
        // lookahead column(s)
        for (int64_t n = k+1; n < k+1+lookahead && n < a.nt_; ++n) {
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[n]) priority(1)
            {
                Matrix<FloatType>::template
                syrk<Target::HostTask>(
                    Uplo::Lower, Op::NoTrans,
                    -1.0, a(n, n, k, k),
                     1.0, a(n, n, n, n));

                if (n+1 <= a.nt_-1)
                    Matrix<FloatType>::template
                    gemm<Target::HostTask>(
                        Op::NoTrans, Op::Trans,
                        -1.0, a(n+1, a.nt_-1, k, k),
                              a(n, n, k, k),
                         1.0, a(n+1, a.nt_-1, n, n));
            }
        }
        // trailing submatrix
        if (k+1+lookahead < a.nt_)
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[k+1+lookahead]) \
                             depend(inout:column[a.nt_-1])
            {
                Matrix<FloatType>::template
                syrk<target>(
                    Uplo::Lower, Op::NoTrans,
                    -1.0, a(k+1+lookahead, a.nt_-1, k, k),
                     1.0, a(k+1+lookahead, a.nt_-1, k+1+lookahead, a.nt_-1));
            }
    }

    Debug::checkTilesLives(a);
    Debug::printTilesLives(a);

    a.clean();

    Debug::printTilesMaps(a);
}

//------------------------------------------------------------------------------
template <typename FloatType>
void potrf(TargetType<Target::Devices>,
           blas::Uplo uplo, Matrix<FloatType> a, int64_t lookahead)
{
    using namespace blas;

    uint8_t *column;

    for (int device = 0; device < a.num_devices_; ++device)
        a.memory_->addDeviceBlocks(device, a.getMaxDeviceTiles(device));

    #pragma omp parallel
    #pragma omp master
    for (int64_t k = 0; k < a.nt_; ++k) {
        // panel
        #pragma omp task depend(inout:column[k])
        {
            Matrix<FloatType>::template
            potrf<Target::HostTask>(uplo, a(k, k, k, k));

            if (k+1 <= a.nt_-1)
                a.tileSend(k, k, {k+1, a.nt_-1, k, k});

            if (k+1 <= a.nt_-1)
                Matrix<FloatType>::template
                trsm<Target::HostTask>(
                    Side::Right, Uplo::Lower,
                    Op::Trans, Diag::NonUnit,
                    1.0, a(k, k, k, k),
                         a(k+1, a.nt_-1, k, k));

            for (int64_t m = k+1; m < a.nt_; ++m)
                a.template tileSend<Target::Devices>(
                    m, k, {m, m, k+1, m},
                          {m, a.nt_-1, m, m});
        }
        // trailing submatrix
        if (k+1+lookahead < a.nt_)
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[k+1+lookahead]) \
                             depend(inout:column[a.nt_-1])
            {
                Matrix<FloatType>::template
                syrk<Target::Devices>(
                    Uplo::Lower, Op::NoTrans,
                    -1.0, a(k+1+lookahead, a.nt_-1, k, k),
                     1.0, a(k+1+lookahead, a.nt_-1, k+1+lookahead, a.nt_-1));
            }

        // lookahead column(s)
        for (int64_t n = k+1; n < k+1+lookahead && n < a.nt_; ++n) {
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[n])
            {
                Matrix<FloatType>::template
                syrk<Target::HostTask>(
                    Uplo::Lower, Op::NoTrans,
                    -1.0, a(n, n, k, k),
                     1.0, a(n, n, n, n));

                if (n+1 <= a.nt_-1)
                    Matrix<FloatType>::template
                    gemm<Target::HostTask>(
                        Op::NoTrans, Op::Trans,
                        -1.0, a(n+1, a.nt_-1, k, k),
                              a(n, n, k, k),
                         1.0, a(n+1, a.nt_-1, n, n));
            }
        }
    }

    for (int device = 0; device < a.num_devices_; ++device)
        a.memory_->clearDeviceBlocks(device);

    Debug::checkTilesLives(a);
    Debug::printTilesLives(a);

    a.clean();

    Debug::printTilesMaps(a);
}

//------------------------------------------------------------------------------
// Precision and target templated function for implementing complex logic.
//
template <typename FloatType, Target target>
void potrf(blas::Uplo uplo, Matrix<FloatType> a, int64_t lookahead)
{
    potrf(TargetType<target>(), uplo, a, lookahead);
}

} // namespace internal

//------------------------------------------------------------------------------
// Target-templated, precision-overloaded functions for the user.
//
template <Target target>
void potrf(blas::Uplo uplo, Matrix<double> a, int64_t lookahead)
{
    internal::potrf<double, target>(uplo, a, lookahead);
}

//------------------------------------------------------------------------------
template
void potrf<Target::HostTask>(
    blas::Uplo uplo, Matrix<double> a, int64_t lookahead);

template
void potrf<Target::HostNest>(
    blas::Uplo uplo, Matrix<double> a, int64_t lookahead);

template
void potrf<Target::HostBatch>(
    blas::Uplo uplo, Matrix<double> a, int64_t lookahead);

template
void potrf<Target::Devices>(
    blas::Uplo uplo, Matrix<double> a, int64_t lookahead);

} // namespace slate
