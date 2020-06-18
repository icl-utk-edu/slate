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
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::trtri from internal::specialization::trtri
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// Distributed parallel inverse of a triangular matrix.
/// Generic implementation for any target.
/// Panel and lookahead computed on host using Host OpenMP task.
/// @ingroup trtri_specialization
///
template <Target target, typename scalar_t>
void trtri(slate::internal::TargetType<target>,
           TriangularMatrix<scalar_t> A, int64_t lookahead)
{
    using BcastList = typename Matrix<scalar_t>::BcastList;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // if upper, change to lower
    if (A.uplo() == Uplo::Upper) {
        A = conjTranspose(A);
    }
    const int64_t A_nt = A.nt();

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > col_vector(A_nt);
    std::vector< uint8_t > row_vector(A_nt);
    uint8_t* col = col_vector.data();
    uint8_t* row = row_vector.data();

    int tag = 0;

    if (target == Target::Devices) {
        A.allocateBatchArrays();
        A.reserveDeviceWorkspace();
    }

    #pragma omp parallel
    #pragma omp master
    {
        omp_set_nested(1);
        // trsm the first column
        if (A_nt > 1) {
            #pragma omp task depend(inout:col[0]) firstprivate(tag)
            {
                // send A(0, 0) down col A(1:nt-1, 0)
                A.tileBcast(0, 0, A.sub(1, A_nt-1, 0, 0), layout, tag);

                // A(1:nt-1, 0) * A(0, 0)^{-H}
                internal::trsm<Target::HostTask>(
                    Side::Right,
                    scalar_t(-1.0), A.sub(0, 0),
                                    A.sub(1, A_nt-1, 0, 0));
            }
            ++tag;
        }

        // send A(1, 0) down
        if (A_nt > 2) {
            #pragma omp task depend(in:col[0]) \
                             depend(out:row[1]) firstprivate(tag)
            {
                // send A(1, 0) down col A(2:nt-1, 0)
                A.tileBcast(1, 0, A.sub(2, A_nt-1, 0, 0), layout, tag);
            }
            ++tag;
        }

        // invert A(0, 0)
        #pragma omp task depend(inout:col[0])
        {
            internal::trtri<Target::HostTask>(A.sub(0, 0));
        }

        // next lookahead columns trsms
        for (int64_t k = 1; k < lookahead+1 && k+1 < A_nt; ++k) {
            #pragma omp task depend(inout:col[k]) \
                             depend(inout:row[k+1]) firstprivate(tag)
            {
                // send A(k, k) down col A(k+1:nt-1, k)
                A.tileBcast(k, k, A.sub(k+1, A_nt-1, k, k), layout, tag);

                // leading column trsm, A(k+1:nt-1, k) * A(k, k)^{-H}
                internal::trsm<Target::HostTask>(
                    Side::Right,
                    scalar_t(-1.0), A.sub(k, k),
                                    A.sub(k+1, A_nt-1, k, k));

                // send leading column to the left
                BcastList bcast_list_A;
                for (int64_t i = k+1; i < A_nt; ++i) {
                    // send A(i, k) across row A(i, 0:k-1)
                    bcast_list_A.push_back({i, k, {A.sub(i, i, 0, k-1)}});
                }
                A.template listBcast<target>(bcast_list_A, layout, tag+1);
            }
            tag += 2;
        }

        for (int64_t k = 1; k < A_nt; ++k) {
            // next leading column trsm
            if (k+1+lookahead < A_nt) {
                #pragma omp task depend(in:col[k-1]) \
                                 depend(inout:col[k+lookahead]) \
                                 depend(inout:row[k+1+lookahead]) \
                                 firstprivate(tag)
                {
                    // send A(k+la, k+la) down col A(k+1+la:nt-1, k)
                    A.tileBcast(k+lookahead, k+lookahead,
                                A.sub(k+1+lookahead, A_nt-1,
                                      k+lookahead, k+lookahead), layout, tag);

                    // leading column trsm,
                    // A(k+1+la:nt-1, k+la) * A(k+la, k+la)^{-H}
                    internal::trsm<Target::HostTask>(
                        Side::Right,
                        scalar_t(-1.0), A.sub(k+lookahead, k+lookahead),
                                        A.sub(k+1+lookahead, A_nt-1,
                                              k+lookahead, k+lookahead));

                    // send leading column to the left
                    BcastList bcast_list_A;
                    for (int64_t i = k+1+lookahead; i < A_nt; ++i) {
                        // send A(i, k+la) across row A(i, 0:k+la-1)
                        bcast_list_A.push_back(
                            {i, k+lookahead, {A.sub(i, i, 0, k+lookahead-1)}});
                    }
                    A.template listBcast<target>(bcast_list_A, layout, tag+1);
                }
                tag += 2;
            }

            // update lookahead rows
            for (int64_t i = k+1; i < k+1+lookahead && i < A_nt; ++i) {
                #pragma omp task depend(in:col[k]) \
                                 depend(in:row[k]) \
                                 depend(inout:row[i]) firstprivate(tag)
                {
                    // A(i, 0:k-1) += A(i, k) * A(k, 0:k-1)
                    internal::gemm<Target::HostTask>(
                        scalar_t(1.0), A.sub(i, i, k, k),
                                       A.sub(k, k, 0, k-1),
                        scalar_t(1.0), A.sub(i, i, 0, k-1),
                        layout);

                    if (i+1 < A_nt) {
                        // send the row down
                        BcastList bcast_list_B;
                        for (int64_t j = 0; j < k+1; ++j) {
                            // send A(i, j) down col A(i+1:nt-1, j)
                            bcast_list_B.push_back(
                                {i, j, {A.sub(i+1, A_nt-1, j, j)}});
                        }
                        A.template listBcast<target>(bcast_list_B, layout, tag);
                    }
                }
                ++tag;
            }

            // update the remaining block
            #pragma omp task depend(in:col[k]) \
                             depend(in:row[k]) \
                             depend(inout:row[k+1+lookahead]) \
                             depend(inout:row[A_nt-1]) firstprivate(tag)
            {
                if (k+1+lookahead < A_nt) {
                    // A(k+1+la:nt-1) += A(k+1+la:nt-1, k) * A(k, 0:k-1)
                    internal::gemm<target>(
                        scalar_t(1.0), A.sub(k+1+lookahead, A_nt-1, k, k),
                                       A.sub(k, k, 0, k-1),
                        scalar_t(1.0), A.sub(k+1+lookahead, A_nt-1, 0, k-1),
                        layout);
                }

                if (k+2+lookahead < A_nt) {
                    // send the top row down
                    BcastList bcast_list_B;
                    for (int64_t j = 0; j < k+1; ++j) {
                        // send A(k, j) down col A(k+1:nt-1, j)
                        bcast_list_B.push_back(
                            {k+1+lookahead, j, {A.sub(k+2+lookahead, A_nt-1, j, j)}});
                    }
                    A.template listBcast<target>(bcast_list_B, layout, tag);
                }
            }
            ++tag;

            // invert the diagonal triangle
            #pragma omp task depend(inout:row[k]) \
                             depend(in:col[k-1]) \
                             firstprivate(tag)
            {
                // send A(k, k) across row A(k, 0:k-1)
                A.tileBcast(k, k, A.sub(k, k, 0, k-1), layout, tag);

                // solve A(k, k) A(k, :) = A(k, 0:k-1)
                internal::trsm<Target::HostTask>(
                    Side::Left,
                    scalar_t(1.0), A.sub(k, k),
                                   A.sub(k, k, 0, k-1));

                // invert A(k, k)
                internal::trtri<Target::HostTask>(A.sub(k, k));
            }
            ++tag;
        }

        #pragma omp taskwait
        A.tileUpdateAllOrigin();
    }

    A.releaseWorkspace();
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup trtri_specialization
///
template <Target target, typename scalar_t>
void trtri(TriangularMatrix<scalar_t>& A,
           Options const& opts)
{
    int64_t lookahead;
    try {
        lookahead = opts.at(Option::Lookahead).i_;
        assert(lookahead >= 0);
    }
    catch (std::out_of_range&) {
        lookahead = 1;
    }

    internal::specialization::trtri(internal::TargetType<target>(),
                                    A, lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel inverse of a triangular matrix.
///
/// Computes the inverse of an upper or lower triangular matrix $A$.
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the n-by-n triangular matrix $A$.
///     On exit, if return value = 0, the (triangular) inverse of the original
///     matrix $A$.
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
/// TODO: return value
/// @retval 0 successful exit
/// @retval >0 for return value = $i$, $A(i,i)$ is exactly zero. The triangular
///         matrix is singular and its inverse can not be computed.
///
/// @ingroup trtri
///
template <typename scalar_t>
void trtri(TriangularMatrix<scalar_t>& A,
           Options const& opts)
{
    Target target;
    try {
        target = Target(opts.at(Option::Target).i_);
    }
    catch (std::out_of_range&) {
        target = Target::HostTask;
    }

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            trtri<Target::HostTask>(A, opts);
            break;
        case Target::HostNest:
            trtri<Target::HostNest>(A, opts);
            break;
        case Target::HostBatch:
            trtri<Target::HostBatch>(A, opts);
            break;
        case Target::Devices:
            trtri<Target::Devices>(A, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void trtri<float>(
    TriangularMatrix<float>& A,
    Options const& opts);

template
void trtri<double>(
    TriangularMatrix<double>& A,
    Options const& opts);

template
void trtri< std::complex<float> >(
    TriangularMatrix< std::complex<float> >& A,
    Options const& opts);

template
void trtri< std::complex<double> >(
    TriangularMatrix< std::complex<double> >& A,
    Options const& opts);

} // namespace slate
