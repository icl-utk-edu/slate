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
#include "internal/internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::geqrf from internal::specialization::geqrf
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// Distributed parallel QR factorization.
/// Generic implementation for any target.
/// Panel and lookahead computed on host using Host OpenMP task.
/// @ingroup geqrf_specialization
///
template <Target target, typename scalar_t>
void geqrf(slate::internal::TargetType<target>,
           Matrix<scalar_t>& A,
           TriangularFactors<scalar_t>& T,
           int64_t ib, int max_panel_threads, int64_t lookahead)
{
    using BcastList = typename Matrix<scalar_t>::BcastList;

    using blas::real;

    const int priority_one = 1;

    int64_t A_mt = A.mt();
    int64_t A_nt = A.nt();
    int64_t A_min_mtnt = std::min(A_mt, A_nt);

    T.clear();
    T.push_back(A.emptyLike());
    T.push_back(A.emptyLike());
    auto Tlocal  = T[0];
    auto Treduce = T[1];

    // workspace
    auto W = A.emptyLike();

    if (target == Target::Devices) {
        A.allocateBatchArrays();
        A.reserveDeviceWorkspace();
        W.allocateBatchArrays();
        // todo: this is demanding too much device workspace memory
        // only one tile-row of matrix W per MPI process is going to be used,
        // but W with size of whole A is being allocated
        // thus limiting the matrix size that can be processed
        W.reserveDeviceWorkspace();
    }

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > column_vector(A_nt);
    uint8_t* column = column_vector.data();

    #pragma omp parallel
    #pragma omp master
    {
        omp_set_nested(1);
        for (int64_t k = 0; k < A_min_mtnt; ++k) {
            const int64_t diag_len = std::min(A.tileMb(k), A.tileNb(k));

            auto A_panel = A.sub(k, A_mt-1, k, k);
            auto Tl_panel = Tlocal.sub(k, A_mt-1, k, k);
            auto Tr_panel = Treduce.sub(k, A_mt-1, k, k);

            // Find ranks in this column.
            std::set<int> ranks_set;
            A_panel.getRanks(&ranks_set);

            assert(ranks_set.size() > 0);

            // Find each rank's top-most row in this panel,
            // where the triangular tile (resulting from local geqrf panel)
            // will reside.
            std::vector< int64_t > top_rows;
            top_rows.reserve(ranks_set.size());
            for (int r: ranks_set) {
                for (int64_t i = 0; i < A_panel.mt(); ++i) {
                    if (A_panel.tileRank(i, 0) == r) {
                        top_rows.push_back(i+k);
                        break;
                    }
                }
            }
            int64_t min_row = k;

            // panel, high priority
            #pragma omp task depend(inout:column[k]) priority(priority_one)
            {
                // local panel factorization
                internal::geqrf<Target::HostTask>(
                                std::move(A_panel),
                                std::move(Tl_panel),
                                diag_len, ib, max_panel_threads, priority_one);

                // triangle-triangle reductions
                // ttqrt handles tile transfers internally
                internal::ttqrt<Target::HostTask>(
                                std::move(A_panel),
                                std::move(Tr_panel));

                // if a trailing matrix exists
                if (k < A_nt-1) {

                    // bcast V across row for trailing matrix update
                    if (k < A_mt) {
                        BcastList bcast_list_V_top;
                        BcastList bcast_list_V;
                        for (int64_t i = k; i < A_mt; ++i) {
                            // send A(i, k) across row A(i, k+1:nt-1)
                            // top_rows' V's (except the top-most one) need three lives
                            if ((std::find(top_rows.begin(), top_rows.end(), i) != top_rows.end()) && (i > min_row))
                                bcast_list_V_top.push_back({i, k, {A.sub(i, i, k+1, A_nt-1)}});
                            else
                                bcast_list_V.push_back({i, k, {A.sub(i, i, k+1, A_nt-1)}});
                        }
                        A.template listBcast(bcast_list_V_top, 0, Layout::ColMajor, 3);// TODO is column major safe?
                        A.template listBcast(bcast_list_V, 0, Layout::ColMajor, 2);
                    }

                    // bcast Tlocal across row for trailing matrix update
                    if (top_rows.size() > 0) {
                        BcastList bcast_list_T;
                        for (auto it = top_rows.begin(); it < top_rows.end(); ++it) {
                            int64_t row = *it;
                            bcast_list_T.push_back({row, k, {Tlocal.sub(row, row, k+1, A_nt-1)}});
                        }
                        Tlocal.template listBcast(bcast_list_T);
                    }

                    // bcast Treduce across row for trailing matrix update
                    if (top_rows.size() > 1) {
                        BcastList bcast_list_T;
                        for (auto it = top_rows.begin(); it < top_rows.end(); ++it) {
                            int64_t row = *it;
                            if (row > min_row) // exclude the first row of this panel that has no Treduce tile
                                bcast_list_T.push_back({row, k, {Treduce.sub(row, row, k+1, A_nt-1)}});
                        }
                        Treduce.template listBcast(bcast_list_T);
                    }
                }
            }

            // update lookahead column(s) on CPU, high priority
            for (int64_t j = k+1; j < (k+1+lookahead) && j < A_nt; ++j) {
                auto A_trail_j = A.sub(k, A_mt-1, j, j);

                #pragma omp task depend(in:column[k]) \
                                 depend(inout:column[j]) \
                                 priority(priority_one)
                {
                    // Apply local reflectors
                    internal::unmqr<Target::HostTask>(
                                    Side::Left, Op::ConjTrans,
                                    std::move(A_panel),
                                    std::move(Tl_panel),
                                    std::move(A_trail_j),
                                    W.sub(k, A_mt-1, j, j));

                    // Apply triangle-triangle reduction reflectors
                    // ttmqr handles the tile broadcasting internally
                    internal::ttmqr<Target::HostTask>(
                                    Side::Left, Op::ConjTrans,
                                    std::move(A_panel),
                                    std::move(Tr_panel),
                                    std::move(A_trail_j),
                                    j);
                }
            }

            // update trailing submatrix, normal priority
            if (k+1+lookahead < A_nt) {
                int64_t j = k+1+lookahead;
                auto A_trail_j = A.sub(k, A_mt-1, j, A_nt-1);

                #pragma omp task depend(in:column[k]) \
                                 depend(inout:column[k+1+lookahead]) \
                                 depend(inout:column[A_nt-1])
                {
                    // Apply local reflectors
                    internal::unmqr<target>(
                                    Side::Left, Op::ConjTrans,
                                    std::move(A_panel),
                                    std::move(Tl_panel),
                                    std::move(A_trail_j),
                                    W.sub(k, A_mt-1, j, A_nt-1));

                    // Apply triangle-triangle reduction reflectors
                    // ttmqr handles the tile broadcasting internally
                    internal::ttmqr<Target::HostTask>(
                                    Side::Left, Op::ConjTrans,
                                    std::move(A_panel),
                                    std::move(Tr_panel),
                                    std::move(A_trail_j),
                                    j);
                }
            }
        }
    }

    A.clearWorkspace();

}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup geqrf_specialization
///
template <Target target, typename scalar_t>
void geqrf(Matrix<scalar_t>& A,
           TriangularFactors<scalar_t>& T,
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

    int64_t ib;
    try {
        ib = opts.at(Option::InnerBlocking).i_;
        assert(ib >= 0);
    }
    catch (std::out_of_range) {
        ib = 16;
    }

    int64_t max_panel_threads;
    try {
        max_panel_threads = opts.at(Option::MaxPanelThreads).i_;
        assert(max_panel_threads >= 1);
    }
    catch (std::out_of_range) {
        max_panel_threads = std::max(omp_get_max_threads()/2, 1);
    }

    internal::specialization::geqrf(internal::TargetType<target>(),
                                    A, T,
                                    ib, max_panel_threads, lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel QR factorization.
/// @ingroup geqrf_computational
///
template <typename scalar_t>
void geqrf(Matrix<scalar_t>& A,
           TriangularFactors<scalar_t>& T,
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
            geqrf<Target::HostTask>(A, T, opts);
            break;
        case Target::HostNest:
            geqrf<Target::HostNest>(A, T, opts);
            break;
        case Target::HostBatch:
            geqrf<Target::HostBatch>(A, T, opts);
            break;
        case Target::Devices:
            geqrf<Target::Devices>(A, T, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void geqrf<float>(
    Matrix<float>& A,
    TriangularFactors<float>& T,
    const std::map<Option, Value>& opts);

template
void geqrf<double>(
    Matrix<double>& A,
    TriangularFactors<double>& T,
    const std::map<Option, Value>& opts);

template
void geqrf< std::complex<float> >(
    Matrix< std::complex<float> >& A,
    TriangularFactors< std::complex<float> >& T,
    const std::map<Option, Value>& opts);

template
void geqrf< std::complex<double> >(
    Matrix< std::complex<double> >& A,
    TriangularFactors< std::complex<double> >& T,
    const std::map<Option, Value>& opts);

} // namespace slate
