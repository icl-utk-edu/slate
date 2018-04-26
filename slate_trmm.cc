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
#include "slate_TriangularMatrix.hh"
#include "slate_internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::trmm from internal::specialization::trmm
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// @internal
/// Distributed parallel triangular matrix-matrix multiplication.
/// Generic implementation for any target.
/// Note A and B are passed by value, so we can transpose if needed
/// (for side = right) without affecting caller.
/// @ingroup trmm
template <Target target, typename scalar_t>
void trmm(slate::internal::TargetType<target>,
          Side side, Diag diag,
          scalar_t alpha, TriangularMatrix<scalar_t> A,
                                    Matrix<scalar_t> B,
          int64_t lookahead)
{
    using namespace blas;
    using BcastList = typename Matrix<scalar_t>::BcastList;

    // if on right, change to left by (conj)-transposing A and B to get op(B) = op(A)*op(B)
    if (side == Side::Right) {
        if (A.op() == Op::ConjTrans || B.op() == Op::ConjTrans) {
            A = conj_transpose(A);
            B = conj_transpose(B);
            alpha = conj(alpha);
        }
        else {
            A = transpose(A);
            B = transpose(B);
        }
    }

    // B is mt-by-nt, A is mt-by-mt (assuming side = left)
    assert(A.mt() == B.mt());
    assert(A.nt() == B.mt());

    int64_t mt = B.mt();
    int64_t nt = B.nt();

    if (target == Target::Devices) {
        B.allocateBatchArrays();
        B.reserveDeviceWorkspace();
    }

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > bcast_vector(mt);
    std::vector< uint8_t >  gemm_vector(mt);
    uint8_t* bcast = bcast_vector.data();
    uint8_t* gemm  =  gemm_vector.data();

    #pragma omp parallel
    #pragma omp master
    {
        if (A.uplo_logical() == Uplo::Upper) {
            // ----------------------------------------
            // Left, Upper/NoTrans or Lower/Trans case
            // Forward sweep

            // send 1st block col of A and block row of B
            #pragma omp task depend(out:bcast[0])
            {
                // broadcast A(i, 0) to ranks owning block row B(i, :),
                // for i = 0
                A.template tileBcast<target>(0, 0, B.sub(0, 0, 0, nt-1));

                // broadcast B(0, j) to ranks owning block col B(0:0, j)
                // todo: nowhere to send?
                BcastList bcast_list_B;
                for (int64_t j = 0; j < nt; ++j)
                    bcast_list_B.push_back({0, j, {B.sub(0, 0, j, j)}});
                B.template listBcast<target>(bcast_list_B);
            }

            // send next lookahead block cols of A and block rows of B
            for (int64_t k = 1; k < lookahead+1 && k < mt; ++k) {
                #pragma omp task depend(in:bcast[k-1]) \
                                 depend(out:bcast[k])
                {
                    // broadcast A(i, k) to ranks owning block row B(i, :)
                    BcastList bcast_list_A;
                    for (int64_t i = 0; i <= k; ++i) // upper
                        bcast_list_A.push_back({i, k, {B.sub(i, i, 0, nt-1)}});
                    A.template listBcast<target>(bcast_list_A);

                    // broadcast B(k, j) to ranks owning block col B(0:k, j)
                    BcastList bcast_list_B;
                    for (int64_t j = 0; j < nt; ++j)
                        bcast_list_B.push_back({k, j, {B.sub(0, k, j, j)}});
                    B.template listBcast<target>(bcast_list_B);
                }
            }

            // multiply alpha A(:, 0) B(0, :), which is:
            // B(0, :) = alpha [ A(0, 0) B(0, :) ]  trmm
            #pragma omp task depend(in:bcast[0]) \
                             depend(out:gemm[0])
            {
                internal::trmm<Target::HostTask>(
                    Side::Left, diag,
                    alpha, A.sub(0, 0),
                           B.sub(0, 0, 0, nt-1));
            }
            for (int64_t k = 1; k < mt; ++k) {

                // send next block col of A and block row of B
                if (k+lookahead < mt) {
                    #pragma omp task depend(in:gemm[k-1]) \
                                     depend(in:bcast[k+lookahead-1]) \
                                     depend(out:bcast[k+lookahead])
                    {
                        // broadcast A(i, k+la) to ranks owning
                        // block row B(i, :)
                        BcastList bcast_list_A;
                        for (int64_t i = 0; i <= k+lookahead; ++i) {  // upper
                            bcast_list_A.push_back(
                                {i, k+lookahead, {B.sub(i, i, 0, nt-1)}});
                        }
                        A.template listBcast<target>(bcast_list_A);

                        // broadcast B(k+la, j) to ranks owning
                        // block col B(0:k+la, j)
                        BcastList bcast_list_B;
                        for (int64_t j = 0; j < nt; ++j) {
                            bcast_list_B.push_back(
                                {k+lookahead, j,
                                 {B.sub(0, k+lookahead, j, j)}});
                        }
                        B.template listBcast<target>(bcast_list_B);
                    }
                }

                // multiply alpha A(:, k) B(k, :), which is:
                // B(0:k-1, :) += alpha [ A(0:k-1, k) B(k, :) ]  gemm
                // B(k, :)      = alpha [ A(k, k)     B(k, :) ]  trmm
                #pragma omp task depend(in:bcast[k]) \
                                 depend(in:gemm[k-1]) \
                                 depend(out:gemm[k])
                {
                    internal::gemm<target>(
                        alpha,         A.sub(0, k-1, k, k),
                                       B.sub(k, k, 0, nt-1),
                        scalar_t(1.0), B.sub(0, k-1, 0, nt-1));

                    // todo: target? needs batch trmm
                    internal::trmm<Target::HostTask>(
                        Side::Left, diag,
                        alpha, A.sub(k, k),
                               B.sub(k, k, 0, nt-1));
                }
            }
        }
        else {
            // ----------------------------------------
            // Left, Lower/NoTrans or Upper/Trans case
            // Backward sweep

            // send 1st block col of A and block row of B
            #pragma omp task depend(out:bcast[mt-1])
            {
                // broadcast A(i, 0) to ranks owning block row B(i, :),
                // for i = m-1
                A.template tileBcast<target>(
                    mt-1, mt-1, B.sub(mt-1, mt-1, 0, nt-1));

                // broadcast B(m-1, j) to ranks owning block col B(m-1:m-1, j)
                // todo: nowhere to send?
                BcastList bcast_list_B;
                for (int64_t j = 0; j < nt; ++j) {
                    bcast_list_B.push_back(
                        {mt-1, j, {B.sub(mt-1, mt-1, j, j)}});
                }
                B.template listBcast<target>(bcast_list_B);
            }

            // send next lookahead block cols of A and block rows of B
            for (int64_t k = mt-2; k >= mt-1-lookahead && k >= 0; --k) {
                #pragma omp task depend(in:bcast[k+1]) \
                                 depend(out:bcast[k])
                {
                    // broadcast A(i, k) to ranks owning block row B(i, :)
                    BcastList bcast_list_A;
                    for (int64_t i = k; i < mt; ++i)  // lower
                        bcast_list_A.push_back({i, k, {B.sub(i, i, 0, nt-1)}});
                    A.template listBcast<target>(bcast_list_A);

                    // broadcast B(k, j) to ranks owning block col B(k:m-1, j)
                    BcastList bcast_list_B;
                    for (int64_t j = 0; j < nt; ++j)
                        bcast_list_B.push_back({k, j, {B.sub(k, mt-1, j, j)}});
                    B.template listBcast<target>(bcast_list_B);
                }
            }

            // multiply B = alpha A(:, mt-1) B(mt-1, :), which is:
            // B(mt-1, :) = alpha [ A(mt-1, mt-1) B(mt-1, :) ]  trmm
            #pragma omp task depend(in:bcast[mt-1]) \
                             depend(out:gemm[mt-1])
            {
                internal::trmm<Target::HostTask>(
                    Side::Left, diag,
                    alpha, A.sub(mt-1, mt-1),
                           B.sub(mt-1, mt-1, 0, nt-1));
            }

            for (int64_t k = mt-2; k >= 0; --k) {

                // send next block col of A and block row of B
                if (k-lookahead >= 0) {
                    #pragma omp task depend(in:gemm[k+1]) \
                                     depend(in:bcast[k-lookahead+1]) \
                                     depend(out:bcast[k-lookahead])
                    {
                        // broadcast A(i, k-la) to ranks 
                        // owning block row B(i, :)
                        BcastList bcast_list_A;
                        for (int64_t i = k-lookahead; i < mt; ++i) {  // lower
                            bcast_list_A.push_back(
                                {i, k-lookahead, {B.sub(i, i, 0, nt-1)}});
                        }
                        A.template listBcast<target>(bcast_list_A);

                        // broadcast B(k-la, j) to ranks
                        // owning block col B(k-la:m-1, j)
                        BcastList bcast_list_B;
                        for (int64_t j = 0; j < nt; ++j) {
                            bcast_list_B.push_back(
                                {k-lookahead, j,
                                 {B.sub(k-lookahead, mt-1, j, j)}});
                        }
                        B.template listBcast<target>(bcast_list_B);
                    }
                }

                // multiply alpha A(:, k) B(k, :), which is:
                // B(k+1:m-1, :) += alpha [ A(k+1:m-1, k) B(k, :) ]  gemm
                // B(k, :)        = alpha [ A(k, k)       B(k, :) ]  trmm
                #pragma omp task depend(in:bcast[k]) \
                                 depend(in:gemm[k+1]) \
                                 depend(out:gemm[k])
                {
                    internal::gemm<target>(
                        alpha,         A.sub(k+1, mt-1, k, k),
                                       B.sub(k, k, 0, nt-1),
                        scalar_t(1.0), B.sub(k+1, mt-1, 0, nt-1));

                    // todo: target? needs batch trmm
                    internal::trmm<Target::HostTask>(
                        Side::Left, diag,
                        alpha, A.sub(k, k),
                               B.sub(k, k, 0, nt-1));
                }
            }
        } // end Lower/NoTrans
    } // end omp master

    // todo: we need a function that updates origins that are not valid
    for (int64_t i = 0; i < B.mt(); ++i)
        for (int64_t j = 0; j < B.nt(); ++j)
            if (B.tileIsLocal(i, j))
                B.tileMoveToHost(i, j, B.tileDevice(i, j));

    B.clearWorkspace();
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup trmm
template <Target target, typename scalar_t>
void trmm(blas::Side side, blas::Diag diag,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                    Matrix<scalar_t>& B,
          const std::map<Option, Value>& opts)
{
    int64_t lookahead;
    try {
        lookahead = opts.at(Option::Lookahead).i_;
        assert(lookahead >= 0);
    }
    catch (std::out_of_range) {
        lookahead = 1;
    }

    internal::specialization::trmm(internal::TargetType<target>(),
                                   side, diag,
                                   alpha, A,
                                          B,
                                   lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel triangular matrix-matrix multiplication.
/// Performs one of the triangular matrix-matrix operations
/// \[
///     B = \alpha A B,
/// \]
/// or
/// \[
///     B = \alpha B A,
/// \]
/// where alpha is a scalar, B is an m-by-n matrix and A is a unit or non-unit,
/// upper or lower triangular matrix.
/// The matrices can be transposed or conjugate-transposed beforehand, e.g.,
///
///     auto AT = slate::transpose( A );
///     slate::trmm( Side::Left, Diag::NonUnit, alpha, AT, B );
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///         One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] side
///         Whether A appears on the left or on the right of B:
///         - Side::Left:  $B = \alpha A B$
///         - Side::Right: $B = \alpha B A$
///
/// @param[in] diag
///         Whether or not A is unit triangular:
///         - Diag::NonUnit: A is non-unit triangular;
///         - Diag::Unit:    A is unit triangular.
///                          The diagonal elements of A are not referenced
///                          and are assumed to be 1.
///
/// @param[in] alpha
///         The scalar alpha.
///
/// @param[in] A
///         - If side = left,  the m-by-m triangular matrix A;
///         - if side = right, the n-by-n triangular matrix A.
///
/// @param[in,out] B
///         On entry, the m-by-n matrix B.
///         On exit, overwritten by the result $\alpha A B$ or $\alpha B A$.
///
/// @param[in] opts
///         Additional options, as map of name = value pairs. Possible options:
///         - Option::Lookahead:
///           Number of blocks to overlap communication and computation.
///           lookahead >= 0. Default 1.
///         - Option::Target:
///           Implementation to target. Possible values:
///           - HostTask:  OpenMP tasks on CPU host [default].
///           - HostNest:  nested OpenMP parallel for loop on CPU host.
///           - HostBatch: batched BLAS on CPU host.
///           - Devices:   batched BLAS on GPU device.
///
/// @ingroup trmm
template <typename scalar_t>
void trmm(blas::Side side, blas::Diag diag,
          scalar_t alpha, TriangularMatrix<scalar_t>& A,
                                    Matrix<scalar_t>& B,
          const std::map<Option, Value>& opts)
{
    Target target;
    try {
        target = Target( opts.at(Option::Target).i_ );
    }
    catch (std::out_of_range) {
        target = Target::HostTask;
    }

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            trmm<Target::HostTask>(side, diag, alpha, A, B, opts);
            break;
        case Target::HostNest:
            trmm<Target::HostNest>(side, diag, alpha, A, B, opts);
            break;
        case Target::HostBatch:
            trmm<Target::HostBatch>(side, diag, alpha, A, B, opts);
            break;
        case Target::Devices:
            trmm<Target::Devices>(side, diag, alpha, A, B, opts);
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void trmm< float >(
    blas::Side side, blas::Diag diag,
    float alpha, TriangularMatrix<float>& A,
                           Matrix<float>& B,
    const std::map<Option, Value>& opts);

template
void trmm< double >(
    blas::Side side, blas::Diag diag,
    double alpha, TriangularMatrix<double>& A,
                            Matrix<double>& B,
    const std::map<Option, Value>& opts);

template
void trmm< std::complex<float> >(
    blas::Side side, blas::Diag diag,
    std::complex<float> alpha, TriangularMatrix<std::complex<float>>& A,
                                         Matrix<std::complex<float>>& B,
    const std::map<Option, Value>& opts);

template
void trmm< std::complex<double> >(
    blas::Side side, blas::Diag diag,
    std::complex<double> alpha, TriangularMatrix<std::complex<double>>& A,
                                          Matrix<std::complex<double>>& B,
    const std::map<Option, Value>& opts);

} // namespace slate
