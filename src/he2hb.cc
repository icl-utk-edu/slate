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
#include "slate/HermitianMatrix.hh"
#include "internal/internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::he2hb from internal::specialization::he2hb
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// Distributed parallel reduction to band for 3-stage Hermitian eigenvalue
/// decomposition.
/// Generic implementation for any target.
/// Panel computed on host using Host OpenMP task.
///
/// ColMajor layout is assumed
///
/// @ingroup he2hb_specialization
///
template <Target target, typename scalar_t>
void he2hb(slate::internal::TargetType<target>,
           HermitianMatrix<scalar_t>& A,
           TriangularFactors<scalar_t>& T,
           int64_t ib, int max_panel_threads)
{
    using BcastList = typename HermitianMatrix<scalar_t>::BcastList;
    using blas::real;

    assert(A.uplo() == Uplo::Lower);  // for now

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    const scalar_t zero = 0;
    const scalar_t one  = 1;
    const scalar_t neg_half = -0.5;

    int64_t nt = A.nt();

    // todo: TriangularFactors takes Matrix, not BaseMatrix or HermitianMatrix.
    // This abuses the "conversion sub-matrix" constructor to get a Matrix.
    T.clear();
    auto empty = A.emptyLike();
    auto Tlocal = Matrix<scalar_t>( empty, 0, nt-1, 0, nt-1 );
    auto Treduce = Tlocal.emptyLike(ib, 0);
    T.push_back( Tlocal );
    T.push_back( Treduce );

    // workspace
    auto W = A.emptyLike();
    auto Wtmp = A.emptyLike();
    //auto TVAVT = A.emptyLike();
    auto Asave = A.emptyLike();

    // Use W(0, 0) for TVAVT, since W(0, 0) is never used otherwise.
    W.tileInsert(0, 0);
    auto TVAVT = W(0, 0);
    TVAVT.uplo(Uplo::General);

    if (target == Target::Devices) {
        A.allocateBatchArrays();
        A.reserveDeviceWorkspace();
        W.allocateBatchArrays();
    }

    // No lookahead is possible, so no need to track dependencies --
    // just execute tasks in order. Also, priority isn't needed.

    int my_rank = A.mpiRank();

    // bool debug = true;

    #pragma omp parallel
    #pragma omp master
    {
        omp_set_nested(1);
        for (int64_t k = 0; k < nt-1; ++k) {
            //----------------------------------------
            // Q panel and update.
            auto       A_panel =       A.sub(k+1, nt-1, k, k);
            auto  Tlocal_panel =  Tlocal.sub(k+1, nt-1, k, k);
            auto Treduce_panel = Treduce.sub(k+1, nt-1, k, k);

            // Find ranks in this column.
            std::set<int> panel_ranks;
            A_panel.getRanks(&panel_ranks);
            assert(panel_ranks.size() > 0);

            // Find each rank's first (top-most) row in this panel,
            // where the triangular tile resulting from local geqrf panel
            // will reside.
            std::vector< int64_t > first_indices;
            first_indices.reserve(panel_ranks.size());
            for (int r: panel_ranks) {
                for (int64_t i = 0; i < A_panel.mt(); ++i) {
                    if (A_panel.tileRank(i, 0) == r) {
                        // todo: way to get index in parent matrix, to avoid manually adding k+1?
                        first_indices.push_back(i+k+1);
                        break;
                    }
                }
            }

            //--------------------
            // QR of panel
            // local panel factorization
            internal::geqrf<Target::HostTask>(
                            std::move(A_panel),
                            std::move(Tlocal_panel),
                            ib, max_panel_threads);

            // triangle-triangle reductions
            // ttqrt handles tile transfers internally
            internal::ttqrt<Target::HostTask>(
                            std::move(A_panel),
                            std::move(Treduce_panel));

            // Done if no trailing matrix exists.
            if (k == nt-1)
                continue;

            // Send V across row i & col i for trailing matrix update.
            BcastList bcast_list_V_first;
            BcastList bcast_list_V;
            for (int64_t i = k; i < nt; ++i) {
                // Vs need 2 lives for unmqr (I - VTV^H).
                // Vs in first_indices (except top-most one, i == k+1)
                // need 3 lives: 2 for unmqr, 1 for ttmqr.
                if (i > k+1 && std::find(first_indices.begin(), first_indices.end(), i) != first_indices.end()) {
                    bcast_list_V_first.push_back(
                        {i, k, {A.sub(i, i, k+1, i)}});
                    bcast_list_V_first.push_back(
                        {i, k, {A.sub(i+1, nt-1, i, i)}});
                }
                else {
                    bcast_list_V.push_back(
                        {i, k, {A.sub(i, i, k+1, i)}});
                    bcast_list_V.push_back(
                        {i, k, {A.sub(i+1, nt-1, i, i)}});
                }
            }
            A.template listBcast(bcast_list_V_first, layout, 0, 3);
            A.template listBcast(bcast_list_V, layout, 0, 2);

            // Send Treduce across row i & col i for trailing matrix update
            if (first_indices.size() > 1) {
                BcastList bcast_list_T;
                for (int64_t i : first_indices) {
                    // Exclude first row of this panel,
                    // which doesn't have Treduce tile.
                    if (i > k+1) {
                        bcast_list_T.push_back(
                            {i, k, {Treduce.sub(i, i, k+1, i)}});
                        bcast_list_T.push_back(
                            {i, k, {Treduce.sub(i+1, nt-1, i, i)}});
                    }
                }
                Treduce.template listBcast(bcast_list_T, layout);
            }

            //----------------------------------------
            // QR update trailing submatrix.
            auto A_trail = A.sub(k+1, nt-1);

            std::vector<int64_t> indices;
            for (int panel_rank: panel_ranks) {
                // Find local indices for panel_rank.
                indices.clear();
                for (int64_t i = 0; i < A_panel.mt(); ++i) {
                    if (A_panel.tileRank(i, 0) == panel_rank) {
                        // todo: global index
                        indices.push_back(i+k+1);
                    }
                }
                int64_t i0 = indices[0];

            // Send Tlocal across row i & col i for trailing matrix update
            // todo: I think this sets Tlocal with too many lives
            // -- needs only 1 life per rank, not # tiles.
            BcastList bcast_list_T;
            for (int64_t i : indices) {
                bcast_list_T.push_back(
                    {i0, k, {Tlocal.sub(i, i, k+1, i)}});
                bcast_list_T.push_back(
                    {i0, k, {Tlocal.sub(i+1, nt-1, i, i)}});
            }
            Tlocal.template listBcast(bcast_list_T, layout);

                if (A.tileExists(i0, k)) {
                    // Save V0 and set upper(V0) to identity, to avoid trmm's.
                    Asave.tileInsert(i0, k);
                    auto Aik = A(i0, k);
                    gecopy(std::move(Aik), Asave(i0, k));
                    Aik.uplo(Uplo::Upper);
                    Aik.set(zero, one);
                }

                //--------------------
                // Apply local reflectors.
                // Compute Wi = (sum_j Aij Vj) T, for i = k+1, ..., nt-1.
                int rank_lower = -1;
                int rank_upper = -1;
                for (int64_t i = k+1; i < nt; ++i) {
                    W.tileInsert(i, k);
                    W(i, k).set(zero);
                    for (int64_t j: indices) {
                        if (i >= j) { // lower
                            rank_lower = A.tileRank(i, j);
                            if (rank_lower == my_rank) { // A.tileIsLocal(i, j)
                                if (i == j) {
                                    hemm(Side::Left, one, A(i, j), A(j, k),
                                                     one, W(i, k));
                                }
                                else {
                                    // todo: if HeMatrix returned conjTrans tiles, could merge this with one below.
                                    gemm(one, A(i, j), A(j, k), one, W(i, k));
                                }
                            }
                        }
                        else { // upper
                            rank_upper = A.tileRank(j, i);
                            if (rank_upper == my_rank) { // A.tileIsLocal(j, i)
                                gemm(one, conjTranspose(A(j, i)), A(j, k),
                                     one, W(i, k));
                            }
                        }
                    }

                    // At most 2 ranks contribute to each Wi; if I am one,
                    // exchange partial sum with neighbor and both ranks sum Wi.
                    int neighbor = -1;
                    if (rank_lower == my_rank)
                        neighbor = rank_upper;
                    else if (rank_upper == my_rank)
                        neighbor = rank_lower;
                    if (neighbor != -1 && neighbor != my_rank) {
                        Wtmp.tileInsert(i, k);
                        if (neighbor < my_rank) {
                            W   .tileSend(i, k, neighbor);
                            Wtmp.tileRecv(i, k, neighbor, layout);
                        }
                        else {
                            Wtmp.tileRecv(i, k, neighbor, layout);
                            W   .tileSend(i, k, neighbor);
                        }
                        axpy(one, Wtmp(i, k), W(i, k));
                        Wtmp.tileErase(i, k);
                    }

                    // If I contributed to Wi, multiply by T.
                    if (rank_upper == my_rank || rank_lower == my_rank) {
                        // Wi = Wi * T
                        auto T0    = Tlocal.sub(i0, i0, k, k);
                        auto TVAVT0 = W.sub(i, i, k, k);

                        int64_t mb = T0.tileMb(0);
                        int64_t nb = T0.tileNb(0);
                        bool trapezoid = (mb < nb);

                        if (trapezoid) {
                            T0     = T0.slice(0, mb-1, 0, mb-1); // first mb-by-mb part
                            TVAVT0 = TVAVT0.slice(0, mb-1, 0, mb-1); // first mb-by-mb part
                        }

                        auto Tk0 = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, T0);
                        trmm(Side::Right, Diag::NonUnit,
                             one, std::move(Tk0(0, 0)), TVAVT0(0, 0));
                    }
                }

                if (A.tileIsLocal(i0, i0)) {
                    //--------------------
                    // This rank has diagonal tiles to update.
                    // Do 2-sided Hermitian update:
                    // A = Q^H A Q
                    //   = (I - V T^H V^H) A (I - V T V^H)
                    //   = A - V W^H - W V^H
                    // where
                    // W = A V T - 0.5 V (T^H V^H (A V T)).

                    // 1a. W = AVT from above.
                    // 1b. TVAVT = V^H (AVT) = V^H W.
                    TVAVT.set(zero);
                    for (int64_t i: indices) {
                        gemm(one, conjTranspose(A(i, k)), W(i, k),
                             one, std::move(TVAVT));
                    }
                    // 1c. TVAVT = T^H (V^H AVT)
                    auto T0    = Tlocal.sub(i0, i0, k, k);
                    auto TVAVT0  = W.sub(0, 0, 0, 0);

                    int64_t mb = T0.tileMb(0);
                    int64_t nb = T0.tileNb(0);
                    bool trapezoid = (mb < nb);

                    if (trapezoid) {
                        T0     = T0.slice(0, mb-1, 0, mb-1); // first mb-by-mb part
                        TVAVT0 = TVAVT0.slice(0, mb-1, 0, nb-1); // first mb-by-nb part
                    }

                    auto Tk0 = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, T0);
                    trmm(Side::Left, Diag::NonUnit,
                         one, conjTranspose(Tk0(0, 0)), std::move(TVAVT0(0, 0)));

                    // 1d. W = W - 0.5 V TVAVT.
                    // Technically, could do a hemm here since TVAVT is Hermitian.
                    for (int64_t i: indices) {
                        gemm(neg_half, A(i, k), std::move(TVAVT), one, W(i, k));
                    }

                    // 2. Update trailing matrix.
                    for (int64_t j: indices) {
                        for (int64_t i: indices) {
                            assert(A.tileIsLocal(i, j));
                            if (i == j) {  // diag
                                // A = A - Vik Wjk^H - Wjk Vik^H
                                her2k(-one, A(i, k), W(j, k), 1.0, A(i, j));
                            }
                            else if (i > j) {  // lower
                                // A = A - Vik Wjk^H
                                gemm(-one, A(i, k), conjTranspose(W(j, k)),
                                      one, A(i, j));
                                // A = A - Wik Vjk^H
                                gemm(-one, W(i, k), conjTranspose(A(j, k)),
                                      one, A(i, j));
                            }
                            // Skip tiles in upper triangle (i < j) that are
                            // known by symmetry.
                        }
                    }
                }
                else { // off-diag
                    //--------------------
                    // This rank has only off-diagonal tiles to update (if any).
                    // Update from left:
                    // A = Q^H A
                    //   = (I - V T^H V^H) A = A - V T^H V^H A
                    //   = A - V W^H
                    // where
                    // W = A^H V T = A V T.
                    for (int64_t j = k+1; j < nt; ++j) {
                        for (int64_t i: indices) {
                            // todo: if HermitianMatrix returned conjTrans
                            // tiles, could merge these two.
                            if (i > j) {
                                if (A.tileIsLocal(i, j)) {
                                    // Aij -= Vik Wjk^H
                                    gemm(-one, A(i, k), conjTranspose(W(j, k)),
                                          one, A(i, j));
                                }
                            }
                            else if (i < j) {
                                if (A.tileIsLocal(j, i)) {
                                    // Aji -= Wjk Vik^H
                                    gemm(-one, W(j, k), conjTranspose(A(i, k)),
                                          one, A(j, i));
                                }
                            }
                            else { // i == j
                                // Diagonal tiles dealt with above.
                                assert(! A.tileIsLocal(i, j));
                            }
                        }
                    }
                }

                if (A.tileExists(i0, k)) {
                    // Restore V0.
                    gecopy(Asave(i0, k), A(i0, k));
                    Asave.tileErase(i0, k);
                }
            }

            internal::hettmqr<Target::HostTask>(
                Op::ConjTrans, std::move(A_panel), std::move(Treduce_panel),
                A.sub(k+1, nt-1));
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
/// @ingroup he2hb_specialization
///
template <Target target, typename scalar_t>
void he2hb(HermitianMatrix<scalar_t>& A,
           TriangularFactors<scalar_t>& T,
           Options const& opts)
{
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

    internal::specialization::he2hb(internal::TargetType<target>(),
                                    A, T,
                                    ib, max_panel_threads);
}

//------------------------------------------------------------------------------
/// Distributed parallel reduction to band for 3-stage SVD.
///
/// Reduces an n-by-n Hermitian matrix $A$ to band form using unitary
/// transformations. The factorization has the form
/// \[
///     A = Q B Q^H
/// \]
/// where $Q$ is unitary and $B$ is Hermitian band
/// with nb sub and superdiagonals.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the n-by-n Hermitian matrix $A$.
///     On exit:
///     - [upper is not yet implemented]
///       If A is upper, the elements Aij for j = i, ..., i+nb,
///       represent the Hermitian band matrix B.
///       The elements above the nb-th superdiagonal, along with T, represent
///       the unitary matrix $Q$ as a product of elementary reflectors.
///     - If A is lower, the elements Aij for j = i-nb, ..., i,
///       represent the Hermitian band matrix B.
///       The elements below the nb-th subdiagonal, along with T, represent
///       the unitary matrix $Q$ as a product of elementary reflectors.
///
/// @param[out] T
///     On exit, triangular matrices of the block reflectors for Q.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
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
///     Note a lookahead is not possible with he2hb due to dependencies from
///     updating on both left and right sides.
///
/// @ingroup he2hb_computational
///
template <typename scalar_t>
void he2hb(HermitianMatrix<scalar_t>& A,
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
            he2hb<Target::HostTask>(A, T, opts);
            break;
        case Target::HostNest:
            he2hb<Target::HostNest>(A, T, opts);
            break;
        case Target::HostBatch:
            he2hb<Target::HostBatch>(A, T, opts);
            break;
        case Target::Devices:
            he2hb<Target::Devices>(A, T, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void he2hb<float>(
    HermitianMatrix<float>& A,
    TriangularFactors<float>& T,
    Options const& opts);

template
void he2hb<double>(
    HermitianMatrix<double>& A,
    TriangularFactors<double>& T,
    Options const& opts);

template
void he2hb< std::complex<float> >(
    HermitianMatrix< std::complex<float> >& A,
    TriangularFactors< std::complex<float> >& T,
    Options const& opts);

template
void he2hb< std::complex<double> >(
    HermitianMatrix< std::complex<double> >& A,
    TriangularFactors< std::complex<double> >& T,
    Options const& opts);

} // namespace slate
