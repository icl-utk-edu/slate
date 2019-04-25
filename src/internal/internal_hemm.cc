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
#include "slate/Tile_blas.hh"
#include "internal/internal.hh"
#include "internal/internal_batch.hh"

#ifdef SLATE_WITH_MKL
    #include <mkl_cblas.h>
#else
    #include <cblas.h>
#endif

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Hermitian matrix multiply to update trailing matrix,
/// where A is a single tile.
/// If side = left,  B and C are each a single block row;
/// if side = right, B and C are each a single block col.
/// Unlike most BLAS operations, here op(B) and op(C) must be
/// both the same, either both NoTrans or both ConjTrans.
/// Dispatches to target implementations.
/// @ingroup hemm_internal
///
template <Target target, typename scalar_t>
void hemm(Side side,
          scalar_t alpha, HermitianMatrix<scalar_t>&& A,
                          Matrix<scalar_t>&& B,
          scalar_t beta,  Matrix<scalar_t>&& C,
          int priority)
{
    // check dimensions
    assert(A.mt() == 1);
    assert(A.nt() == 1);
    if (side == Side::Left) {
        assert(B.mt() == 1);
        assert(C.mt() == 1);
        assert(B.nt() == C.nt());
    }
    else {
        assert(B.nt() == 1);
        assert(C.nt() == 1);
        assert(B.mt() == C.mt());
    }
    assert(B.op() == C.op());

    hemm(internal::TargetType<target>(),
         side,
         alpha, A,
                B,
         beta,  C,
         priority);
}

//------------------------------------------------------------------------------
/// Hermitian matrix multiply to update trailing matrix.
/// Host OpenMP task implementation.
/// @ingroup hemm_internal
///
template <typename scalar_t>
void hemm(internal::TargetType<Target::HostTask>,
          Side side,
          scalar_t alpha, HermitianMatrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          int priority)
{
    // CPU uses ColMajor
    // todo: relax this assumption, by allowing Tile_blas.hh::hemm() to take layout param
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'C(i, j).layout()'
    const Layout layout = Layout::ColMajor;

    int err = 0;
    if (side == Side::Left) {
        for (int64_t j = 0; j < C.nt(); ++j) {
            if (C.tileIsLocal(0, j)) {
                #pragma omp task shared(A, B, C, err) priority(priority)
                {
                    try {
                        A.tileGetForReading(0, 0, LayoutConvert(layout));
                        B.tileGetForReading(0, j, LayoutConvert(layout));
                        C.tileGetForWriting(0, j, LayoutConvert(layout));
                        hemm(side,
                             alpha, A(0, 0),
                                    B(0, j),
                             beta,  C(0, j));
                        // todo: should tileRelease()?
                        A.tileTick(0, 0);
                        B.tileTick(0, j);
                    }
                    catch (std::exception& e) {
                        err = __LINE__;
                    }
                }
            }
        }
    }
    else {
        // side == Right
        for (int64_t i = 0; i < C.mt(); ++i) {
            if (C.tileIsLocal(i, 0)) {
                #pragma omp task shared(A, B, C, err) priority(priority)
                {
                    try {
                        A.tileGetForReading(0, 0, LayoutConvert(layout));
                        B.tileGetForReading(i, 0, LayoutConvert(layout));
                        C.tileGetForWriting(i, 0, LayoutConvert(layout));
                        hemm(side,
                             alpha, A(0, 0),
                                    B(i, 0),
                             beta,  C(i, 0));
                        // todo: should tileRelease()?
                        A.tileTick(0, 0);
                        B.tileTick(i, 0);
                    }
                    catch (std::exception& e) {
                        err = __LINE__;
                    }
                }
            }
        }
    }

    #pragma omp taskwait

    if (err)
        throw std::exception();
}

//------------------------------------------------------------------------------
/// Hermitian matrix multiply to update trailing matrix.
/// Host nested OpenMP implementation.
/// @ingroup hemm_internal
///
template <typename scalar_t>
void hemm(internal::TargetType<Target::HostNest>,
          Side side,
          scalar_t alpha, HermitianMatrix<scalar_t>& A,
                          Matrix<scalar_t>& B,
          scalar_t beta,  Matrix<scalar_t>& C,
          int priority)
{
    // CPU uses ColMajor
    // todo: relax this assumption, by allowing Tile_blas.hh::hemm() to take layout param
    // todo: optimize for the number of layout conversions,
    //       by watching 'layout' and 'C(i, j).layout()'
    const Layout layout = Layout::ColMajor;

    int err = 0;
    if (side == Side::Left) {
        #pragma omp parallel for schedule(dynamic, 1) shared(err)
        for (int64_t j = 0; j < C.nt(); ++j) {
            if (C.tileIsLocal(0, j)) {
                try {
                    A.tileGetForReading(0, 0, LayoutConvert(layout));
                    B.tileGetForReading(0, j, LayoutConvert(layout));
                    C.tileGetForWriting(0, j, LayoutConvert(layout));
                    hemm(side,
                         alpha, A(0, 0),
                                B(0, j),
                         beta,  C(0, j));
                    // todo: should tileRelease()?
                    A.tileTick(0, 0);
                    B.tileTick(0, j);
                }
                catch (std::exception& e) {
                    err = __LINE__;
                }
            }
        }
    }
    else {
        // side == Right
        #pragma omp parallel for schedule(dynamic, 1) shared(err)
        for (int64_t i = 0; i < C.mt(); ++i) {
            if (C.tileIsLocal(i, 0)) {
                #pragma omp task shared(A, B, C, err) priority(priority)
                {
                    try {
                        A.tileGetForReading(0, 0, LayoutConvert(layout));
                        B.tileGetForReading(i, 0, LayoutConvert(layout));
                        C.tileGetForWriting(i, 0, LayoutConvert(layout));
                        hemm(side,
                             alpha, A(0, 0),
                                    B(i, 0),
                             beta,  C(i, 0));
                        // todo: should tileRelease()?
                        A.tileTick(0, 0);
                        B.tileTick(i, 0);
                    }
                    catch (std::exception& e) {
                        err = __LINE__;
                    }
                }
            }
        }
    }

    #pragma omp taskwait

    if (err)
        throw std::exception();
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void hemm< Target::HostTask, float >(
    Side side,
    float alpha, HermitianMatrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    int priority);

template
void hemm<Target::HostNest, float>(
    Side side,
    float alpha, HermitianMatrix<float>&& A,
                 Matrix<float>&& B,
    float beta,  Matrix<float>&& C,
    int priority);

// ----------------------------------------
template
void hemm<Target::HostTask, double>(
    Side side,
    double alpha, HermitianMatrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    int priority);

template
void hemm<Target::HostNest, double>(
    Side side,
    double alpha, HermitianMatrix<double>&& A,
                  Matrix<double>&& B,
    double beta,  Matrix<double>&& C,
    int priority);

// ----------------------------------------
template
void hemm< Target::HostTask, std::complex<float> >(
    Side side,
    std::complex<float> alpha, HermitianMatrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    int priority);

template
void hemm< Target::HostNest, std::complex<float> >(
    Side side,
    std::complex<float> alpha, HermitianMatrix< std::complex<float> >&& A,
                               Matrix< std::complex<float> >&& B,
    std::complex<float> beta,  Matrix< std::complex<float> >&& C,
    int priority);

// ----------------------------------------
template
void hemm< Target::HostTask, std::complex<double> >(
    Side side,
    std::complex<double> alpha, HermitianMatrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    int priority);

template
void hemm< Target::HostNest, std::complex<double> >(
    Side side,
    std::complex<double> alpha, HermitianMatrix< std::complex<double> >&& A,
                                Matrix< std::complex<double> >&& B,
    std::complex<double> beta,  Matrix< std::complex<double> >&& C,
    int priority);

} // namespace internal
} // namespace slate
