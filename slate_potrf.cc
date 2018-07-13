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
#include "slate_HermitianMatrix.hh"
#include "slate_TriangularMatrix.hh"
#include "slate_internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::potrf from internal::specialization::potrf
namespace internal {
namespace specialization {

///-----------------------------------------------------------------------------
/// \brief
/// Distributed parallel Cholesky factorization.
/// Generic implementation for any target.
/// Panel and lookahead computed on host using Host OpenMP task.
template <Target target, typename scalar_t>
void potrf(slate::internal::TargetType<target>,
           HermitianMatrix<scalar_t>& A, int64_t lookahead)
{
    using real_t = blas::real_type<scalar_t>;
    using BcastList = typename Matrix<scalar_t>::BcastList;

    const int64_t A_nt = A.nt();

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > column_vector(A_nt);
    uint8_t* column = column_vector.data();

    #pragma omp parallel
    #pragma omp master
    for (int64_t k = 0; k < A_nt; ++k) {
        // panel, high priority
        #pragma omp task depend(inout:column[k]) priority(1)
        {
            // factor A(k, k)
            internal::potrf<Target::HostTask>(A.sub(k, k), 1);

            // send A(k, k) down col A(k+1:nt-1, k)
            if (k+1 <= A_nt-1)
                A.tileBcast(k, k, A.sub(k+1, A_nt-1, k, k));

            // A(k+1:nt-1, k) * A(k, k)^{-H}
            if (k+1 <= A_nt-1) {
                auto Akk = A.sub(k, k);
                auto Tkk = TriangularMatrix< scalar_t >(Diag::NonUnit, Akk);
                internal::trsm<Target::HostTask>(
                    Side::Right,
                    scalar_t(1.0), conj_transpose(Tkk),
                    A.sub(k+1, A_nt-1, k, k), 1);
            }

            BcastList bcast_list_A;
            for (int64_t i = k+1; i < A_nt; ++i) {
                // send A(i, k) across row A(i, k+1:i) and down col A(i:nt-1, i)
                bcast_list_A.push_back({i, k, {A.sub(i, i, k+1, i),
                                               A.sub(i, A_nt-1, i, i)}});
            }
            A.template listBcast(bcast_list_A);
        }
        // update lookahead column(s), high priority
        for (int64_t j = k+1; j < k+1+lookahead && j < A_nt; ++j) {
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[j]) priority(1)
            {
                // A(j, j) -= A(j, k) * A(j, k)^H
                internal::herk<Target::HostTask>(
                    real_t(-1.0), A.sub(j, j, k, k),
                    real_t( 1.0), A.sub(j, j), 1);

                // A(j+1:nt, j) -= A(j+1:nt-1, k) * A(j, k)^H
                if (j+1 <= A_nt-1) {
                    auto Ajk = A.sub(j, j, k, k);
                    internal::gemm<Target::HostTask>(
                        scalar_t(-1.0), A.sub(j+1, A_nt-1, k, k),
                                        conj_transpose(Ajk),
                        scalar_t(1.0), A.sub(j+1, A_nt-1, j, j), 1);
                }
            }
        }
        // update trailing submatrix, normal priority
        if (k+1+lookahead < A_nt) {
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[k+1+lookahead]) \
                             depend(inout:column[A_nt-1])
            {
                // A(kl+1:nt-1, kl+1:nt-1) -=
                //     A(kl+1:nt-1, k) * A(kl+1:nt-1, k)^H
                // where kl = k + lookahead
                internal::herk<target>(
                    real_t(-1.0), A.sub(k+1+lookahead, A_nt-1, k, k),
                    real_t( 1.0), A.sub(k+1+lookahead, A_nt-1));
            }
        }
    }

    // Debug::checkTilesLives(A);
    // Debug::printTilesLives(A);

    A.clearWorkspace();

    // Debug::printTilesMaps(A);
}

///-----------------------------------------------------------------------------
/// \brief
/// Distributed parallel Cholesky factorization.
/// GPU device batched cuBLAS implementation.
template <typename scalar_t>
void potrf(slate::internal::TargetType<Target::Devices>,
           HermitianMatrix<scalar_t>& A, int64_t lookahead)
{
    using real_t = blas::real_type<scalar_t>;
    using BcastList = typename Matrix<scalar_t>::BcastList;

    const int64_t A_nt = A.nt();

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > column_vector(A_nt);
    uint8_t* column = column_vector.data();

    A.allocateBatchArrays();
    A.reserveDeviceWorkspace();

    #pragma omp parallel
    #pragma omp master
    for (int64_t k = 0; k < A_nt; ++k) {
        // panel, normal priority
        #pragma omp task depend(inout:column[k])
        {
            // factor A(k, k)
            internal::potrf<Target::HostTask>(A.sub(k, k));

            // send A(k, k) down col A(k+1:nt-1, k)
            if (k+1 <= A_nt-1)
                A.tileBcast(k, k, A.sub(k+1, A_nt-1, k, k));

            // A(k+1:nt-1, k) * A(k, k)^{-H}
            if (k+1 <= A_nt-1) {
                auto Akk = A.sub(k, k);
                auto Tkk = TriangularMatrix< scalar_t >(Diag::NonUnit, Akk);
                internal::trsm<Target::HostTask>(
                    Side::Right,
                    scalar_t(1.0), conj_transpose(Tkk),
                                   A.sub(k+1, A_nt-1, k, k));
            }

            BcastList bcast_list_A;
            for (int64_t i = k+1; i < A_nt; ++i) {
                // send A(i, k) across row A(i, k+1:i) and down col A(i:nt-1, i)
                bcast_list_A.push_back({i, k, {A.sub(i, i, k+1, i),
                                               A.sub(i, A_nt-1, i, i)}});
            }
            A.template listBcast<Target::Devices>(bcast_list_A);
        }
        // update trailing submatrix, normal priority
        if (k+1+lookahead < A_nt) {
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[k+1+lookahead]) \
                             depend(inout:column[A_nt-1])
            {
                // A(kl+1:nt-1, kl+1:nt-1) -=
                //     A(kl+1:nt-1, k) * A(kl+1:nt-1, k)^H
                // where kl = k + lookahead
                internal::herk<Target::Devices>(
                    real_t(-1.0), A.sub(k+1+lookahead, A_nt-1, k, k),
                    real_t( 1.0), A.sub(k+1+lookahead, A_nt-1));
            }
        }

        // update lookahead column(s), normal priority
        for (int64_t j = k+1; j < k+1+lookahead && j < A_nt; ++j) {
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[j])
            {
                // A(j, j) -= A(j, k) * A(j, k)^H
                internal::herk<Target::HostTask>(
                    real_t(-1.0), A.sub(j, j, k, k),
                    real_t( 1.0), A.sub(j, j));

                // A(j+1:nt, j) -= A(j+1:nt-1, k) * A(j, k)^H
                if (j+1 <= A_nt-1) {
                    auto Ajk = A.sub(j, j, k, k);
                    internal::gemm<Target::HostTask>(
                        scalar_t(-1.0), A.sub(j+1, A_nt-1, k, k),
                                        conj_transpose(Ajk),
                        scalar_t( 1.0), A.sub(j+1, A_nt-1, j, j));
                }
            }
        }
    }

    // Debug::checkTilesLives(A);
    // Debug::printTilesLives(A);

    A.clearWorkspace();

    // Debug::printTilesMaps(A);
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup posv_comp
template <Target target, typename scalar_t>
void potrf(HermitianMatrix<scalar_t>& A,
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

    internal::specialization::potrf(internal::TargetType<target>(),
                                    A, lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel Cholesky factorization.
/// Performs the Cholesky factorization of a Hermitian
/// (or symmetric, in the real case) positive definite
/// matrix A. The factorization has the form
/// \[
///     A = L L^H,
/// \]
/// if A is stored lower, where L is a lower triangular matrix, or
/// \[
///     A = U^H U,
/// \]
/// if A is stored upper, where U is an upper triangular matrix.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the Hermitian positive definite matrix A.
///     On exit, if return value = 0, the factor U or L from the Cholesky
///     factorization $A = U^H U$ or $A = L L^H$.
///     If scalar_t is real, A can be a SymmetricMatrix object.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Lookahead:
///       Number of panels to overlap with matrix updates.
///       lookahead >= 0. Default 1.
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// @ingroup posv_comp
template <typename scalar_t>
void potrf(HermitianMatrix<scalar_t>& A,
           const std::map<Option, Value>& opts)
{
    Target target;
    try {
        target = Target(opts.at(Option::Target).i_);
    }
    catch (std::out_of_range) {
        target = Target::HostTask;
    }

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            potrf<Target::HostTask>(A, opts);
            break;
        case Target::HostNest:
            potrf<Target::HostNest>(A, opts);
            break;
        case Target::HostBatch:
            potrf<Target::HostBatch>(A, opts);
            break;
        case Target::Devices:
            potrf<Target::Devices>(A, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void potrf<float>(
    HermitianMatrix<float>& A,
    const std::map<Option, Value>& opts);

template
void potrf<double>(
    HermitianMatrix<double>& A,
    const std::map<Option, Value>& opts);

template
void potrf< std::complex<float> >(
    HermitianMatrix< std::complex<float> >& A,
    const std::map<Option, Value>& opts);

template
void potrf< std::complex<double> >(
    HermitianMatrix< std::complex<double> >& A,
    const std::map<Option, Value>& opts);

} // namespace slate
