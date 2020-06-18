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
///
/// ColMajor layout is assumed
///
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

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    const int priority_one = 1;

    int64_t A_mt = A.mt();
    int64_t A_nt = A.nt();
    int64_t A_min_mtnt = std::min(A_mt, A_nt);

    T.clear();
    T.push_back(A.emptyLike());
    T.push_back(A.emptyLike(ib, 0));
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
        // For now, allocate workspace tiles 1-by-1.
        //W.reserveDeviceWorkspace();
    }

    // QR tracks dependencies by block-column.
    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > block_vector(A_nt);
    uint8_t* block = block_vector.data();

    #pragma omp parallel
    #pragma omp master
    {
        omp_set_nested(1);
        for (int64_t k = 0; k < A_min_mtnt; ++k) {
            auto  A_panel =       A.sub(k, A_mt-1, k, k);
            auto Tl_panel =  Tlocal.sub(k, A_mt-1, k, k);
            auto Tr_panel = Treduce.sub(k, A_mt-1, k, k);

            // Find ranks in this column.
            std::set<int> ranks_set;
            A_panel.getRanks(&ranks_set);
            assert(ranks_set.size() > 0);

            // Find each rank's first (top-most) row in this panel,
            // where the triangular tile resulting from local geqrf panel
            // will reside.
            std::vector< int64_t > first_indices;
            first_indices.reserve(ranks_set.size());
            for (int r: ranks_set) {
                for (int64_t i = 0; i < A_panel.mt(); ++i) {
                    if (A_panel.tileRank(i, 0) == r) {
                        first_indices.push_back(i+k);
                        break;
                    }
                }
            }
            // todo: pass first_indices into internal geqrf or ttqrt?

            // panel, high priority
            #pragma omp task depend(inout:block[k]) priority(priority_one)
            {
                // local panel factorization
                internal::geqrf<Target::HostTask>(
                                std::move(A_panel),
                                std::move(Tl_panel),
                                ib, max_panel_threads, priority_one);

                // triangle-triangle reductions
                // ttqrt handles tile transfers internally
                internal::ttqrt<Target::HostTask>(
                                std::move(A_panel),
                                std::move(Tr_panel));

                // if a trailing matrix exists
                if (k < A_nt-1) {

                    // bcast V across row for trailing matrix update
                    if (k < A_mt) {
                        BcastList bcast_list_V_first;
                        BcastList bcast_list_V;
                        for (int64_t i = k; i < A_mt; ++i) {
                            // send A(i, k) across row A(i, k+1:nt-1)
                            // Vs in first_indices (except the main diagonal one) need three lives
                            if ((std::find(first_indices.begin(), first_indices.end(), i) != first_indices.end()) && (i > k))
                                bcast_list_V_first.push_back({i, k, {A.sub(i, i, k+1, A_nt-1)}});
                            else
                                bcast_list_V.push_back({i, k, {A.sub(i, i, k+1, A_nt-1)}});
                        }
                        A.template listBcast(bcast_list_V_first, layout, 0, 3);
                        A.template listBcast(bcast_list_V, layout, 0, 2);
                    }

                    // bcast Tlocal across row for trailing matrix update
                    if (first_indices.size() > 0) {
                        BcastList bcast_list_T;
                        for (int64_t row : first_indices) {
                            bcast_list_T.push_back({row, k, {Tlocal.sub(row, row, k+1, A_nt-1)}});
                        }
                        Tlocal.template listBcast(bcast_list_T, layout);
                    }

                    // bcast Treduce across row for trailing matrix update
                    if (first_indices.size() > 1) {
                        BcastList bcast_list_T;
                        for (int64_t row : first_indices) {
                            if (row > k) // exclude the first row of this panel that has no Treduce tile
                                bcast_list_T.push_back({row, k, {Treduce.sub(row, row, k+1, A_nt-1)}});
                        }
                        Treduce.template listBcast(bcast_list_T, layout);
                    }
                }
            }

            // update lookahead column(s) on CPU, high priority
            for (int64_t j = k+1; j < (k+1+lookahead) && j < A_nt; ++j) {
                auto A_trail_j = A.sub(k, A_mt-1, j, j);

                #pragma omp task depend(in:block[k]) \
                                 depend(inout:block[j]) \
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

                #pragma omp task depend(in:block[k]) \
                                 depend(inout:block[k+1+lookahead]) \
                                 depend(inout:block[A_nt-1])
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

        #pragma omp taskwait
        A.tileUpdateAllOrigin();
    }

    A.releaseWorkspace();
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

    int64_t ib;
    try {
        ib = opts.at(Option::InnerBlocking).i_;
        assert(ib >= 0);
    }
    catch (std::out_of_range&) {
        ib = 16;
    }

    int64_t max_panel_threads;
    try {
        max_panel_threads = opts.at(Option::MaxPanelThreads).i_;
        assert(max_panel_threads >= 1);
    }
    catch (std::out_of_range&) {
        max_panel_threads = std::max(omp_get_max_threads()/2, 1);
    }

    internal::specialization::geqrf(internal::TargetType<target>(),
                                    A, T,
                                    ib, max_panel_threads, lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel QR factorization.
///
/// Computes a QR factorization of an m-by-n matrix $A$.
/// The factorization has the form
/// \[
///     A = QR,
/// \]
/// where $Q$ is a matrix with orthonormal columns and $R$ is upper triangular
/// (or upper trapezoidal if m < n).
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the m-by-n matrix $A$.
///     On exit, the elements on and above the diagonal of the array contain
///     the min(m,n)-by-n upper trapezoidal matrix $R$ (upper triangular
///     if m >= n); the elements below the diagonal represent the unitary
///     matrix $Q$ as a product of elementary reflectors.
///
/// @param[out] T
///     On exit, triangular matrices of the block reflectors.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Lookahead:
///       Number of panels to overlap with matrix updates.
///       lookahead >= 0. Default 1.
///     - Option::InnerBlocking:
///       Inner blocking to use for panel. Default 16.
///     - Option::MaxPanelThreads:
///       Number of threads to use for panel. Default omp_get_max_threads()/2.
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// @ingroup geqrf_computational
///
template <typename scalar_t>
void geqrf(Matrix<scalar_t>& A,
           TriangularFactors<scalar_t>& T,
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
    Options const& opts);

template
void geqrf<double>(
    Matrix<double>& A,
    TriangularFactors<double>& T,
    Options const& opts);

template
void geqrf< std::complex<float> >(
    Matrix< std::complex<float> >& A,
    TriangularFactors< std::complex<float> >& T,
    Options const& opts);

template
void geqrf< std::complex<double> >(
    Matrix< std::complex<double> >& A,
    TriangularFactors< std::complex<double> >& T,
    Options const& opts);

} // namespace slate
