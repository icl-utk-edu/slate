// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "internal/internal.hh"

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// Distributed parallel reduction to band for 3-stage SVD.
/// Generic implementation for any target.
/// Panel computed on host using Host OpenMP task.
///
/// ColMajor layout is assumed
///
/// @ingroup svd_impl
///
template <Target target, typename scalar_t>
void ge2tb(
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t>& TU,
    TriangularFactors<scalar_t>& TV,
    Options const& opts )
{
    using BcastList = typename Matrix<scalar_t>::BcastList;
    using blas::real;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // Options
    int64_t ib = get_option<int64_t>( opts, Option::InnerBlocking, 16 );
    int64_t max_panel_threads  = std::max(omp_get_max_threads()/2, 1);
    max_panel_threads = get_option<int64_t>( opts, Option::MaxPanelThreads,
                                             max_panel_threads );

    int64_t A_mt = A.mt();
    int64_t A_nt = A.nt();
    int64_t A_min_mtnt = std::min(A_mt, A_nt);

    TU.clear();
    TU.push_back(A.emptyLike());
    TU.push_back(A.emptyLike(ib, 0));
    auto TUlocal  = TU[0];
    auto TUreduce = TU[1];

    // Make TVlocal have fixed, square nb-by-nb tiles,
    // and TVreduce have fixed, rectangular ib-by-nb tiles.
    // Otherwise, edge tiles are the wrong size: mb-by-nb instead of nb-by-mb.
    int64_t nb = A.tileNb(0);
    TV.clear();
    TV.push_back(A.emptyLike(nb, nb));
    TV.push_back(A.emptyLike(ib, nb));
    auto TVlocal  = TV[0];
    auto TVreduce = TV[1];
    auto TVlocalT = A.emptyLike(nb, nb, Op::ConjTrans);

    // workspace
    auto W = A.emptyLike();

    if (target == Target::Devices) {
        A.allocateBatchArrays();
        A.reserveDeviceWorkspace();
        W.allocateBatchArrays();
        // todo: this is demanding too much device workspace memory
        // only one tile-row and one tile-col of matrix W per MPI process
        // is going to be used, but W with size of whole A is being allocated
        // thus limiting the matrix size that can be processed
        // For now, allocate workspace tiles 1-by-1.
        //W.reserveDeviceWorkspace();
    }

    // Workspace for transposed panels needs one column of tiles.
    auto AT = A.emptyLike(0, 0, Op::ConjTrans);
    // todo: we really only want to insert 1 column's worth at a time.
    AT.insertLocalTiles();

    // No lookahead is possible, so no need to track dependencies --
    // just execute tasks in order. Also, priority isn't needed.

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
        for (int64_t k = 0; k < A_min_mtnt; ++k) {
            //----------------------------------------
            // U panel and update.
            auto U_panel   =        A.sub(k, A_mt-1, k, k);
            auto TUl_panel =  TUlocal.sub(k, A_mt-1, k, k);
            auto TUr_panel = TUreduce.sub(k, A_mt-1, k, k);

            // Find ranks in this column.
            std::set<int> ranks_set;
            U_panel.getRanks(&ranks_set);
            assert(ranks_set.size() > 0);

            // Find each rank's first (top-most) row in this panel,
            // where the triangular tile resulting from local geqrf panel
            // will reside.
            std::vector< int64_t > first_indices;
            first_indices.reserve(ranks_set.size());
            for (int r: ranks_set) {
                for (int64_t i = 0; i < U_panel.mt(); ++i) {
                    if (U_panel.tileRank(i, 0) == r) {
                        first_indices.push_back(i+k);
                        break;
                    }
                }
            }

            //--------------------
            // QR of U panel
            // local panel factorization
            internal::geqrf<Target::HostTask>(
                            std::move(U_panel),
                            std::move(TUl_panel),
                            ib, max_panel_threads);

            // triangle-triangle reductions
            // ttqrt handles tile transfers internally
            internal::ttqrt<Target::HostTask>(
                            std::move(U_panel),
                            std::move(TUr_panel));

            // if a trailing matrix exists
            if (k < A_nt-1) {

                // bcast V across row for trailing matrix update
                if (k < A_mt) {
                    BcastList bcast_list_V_first;
                    BcastList bcast_list_V;
                    for (int64_t i = k; i < A_mt; ++i) {
                        // send A(i, k) across row A(i, k+1:nt-1)
                        // Vs in first_indices (except main diagonal one)
                        // need three lives.
                        if ((std::find(first_indices.begin(), first_indices.end(), i) != first_indices.end()) && (i > k)) {
                            bcast_list_V_first.push_back(
                                {i, k, {A.sub(i, i, k+1, A_nt-1)}});
                        }
                        else {
                            bcast_list_V.push_back(
                                {i, k, {A.sub(i, i, k+1, A_nt-1)}});
                        }
                    }
                    A.template listBcast(bcast_list_V_first, layout, 0, 3);
                    A.template listBcast(bcast_list_V, layout, 0, 2);
                }

                // bcast TUlocal across row for trailing matrix update
                if (first_indices.size() > 0) {
                    BcastList bcast_list_T;
                    for (int64_t row : first_indices) {
                        bcast_list_T.push_back(
                            {row, k, {TUlocal.sub(row, row, k+1, A_nt-1)}});
                    }
                    TUlocal.template listBcast(bcast_list_T, layout);
                }

                // bcast TUreduce across row for trailing matrix update
                if (first_indices.size() > 1) {
                    BcastList bcast_list_T;
                    for (int64_t row : first_indices) {
                        // Exclude first row of this panel,
                        // which doesn't have TUreduce tile.
                        if (row > k) {
                            bcast_list_T.push_back(
                                {row, k, {TUreduce.sub(row, row, k+1, A_nt-1)}});
                        }
                    }
                    TUreduce.template listBcast(bcast_list_T, layout);
                }
            }

            //--------------------
            // QR update trailing submatrix.
            if (k+1 < A_nt) {
                int64_t j = k+1;
                auto A_trail_j = A.sub(k, A_mt-1, j, A_nt-1);

                // Apply local reflectors
                internal::unmqr<target>(
                                Side::Left, Op::ConjTrans,
                                std::move(U_panel),
                                std::move(TUl_panel),
                                std::move(A_trail_j),
                                W.sub(k, A_mt-1, j, A_nt-1));

                // Apply triangle-triangle reduction reflectors
                // ttmqr handles the tile broadcasting internally
                internal::ttmqr<Target::HostTask>(
                                Side::Left, Op::ConjTrans,
                                std::move(U_panel),
                                std::move(TUr_panel),
                                std::move(A_trail_j),
                                j);
            }

            //----------------------------------------
            // V panel and update.
            if (k+1 < A_nt) {
                auto   V_panel =        A.sub(k, k, k+1, A_nt-1);
                auto TVl_panel =  TVlocal.sub(k, k, k+1, A_nt-1);
                auto TVr_panel = TVreduce.sub(k, k, k+1, A_nt-1);
                // Transposed panels.
                auto   VT_panel =       AT.sub(k+1, A_nt-1, k, k);
                auto TVlT_panel = TVlocalT.sub(k+1, A_nt-1, k, k);

                // Find ranks in this row.
                ranks_set.clear();
                V_panel.getRanks(&ranks_set);
                assert(ranks_set.size() > 0);

                // Find each rank's first (left-most) col in this panel,
                // where the triangular tile resulting from local gelqf panel
                // will reside.
                first_indices.clear();
                first_indices.reserve(ranks_set.size());
                for (int r: ranks_set) {
                    for (int64_t j = 0; j < V_panel.nt(); ++j) {
                        if (V_panel.tileRank(0, j) == r) {
                            first_indices.push_back(k+1+j);
                            break;
                        }
                    }
                }

                //--------------------
                // LQ of V panel
                //----------
                // Instead of doing LQ of panel, we do QR of the transposed
                // panel, so that the panel is computed in column-major for
                // much better cache efficiency.
                for (int64_t j = 0; j < V_panel.nt(); ++j) {
                    if (V_panel.tileIsLocal(0, j)) {
                        V_panel.tileGetForReading( 0, j, HostNum, LayoutConvert(layout) );
                        VT_panel.tileGetForWriting( j, 0, HostNum, LayoutConvert(layout) );
                        tile::deepConjTranspose( V_panel(0, j), VT_panel(j, 0) );
                    }
                }

                // local panel factorization
                internal::geqrf<Target::HostTask>(
                                std::move(VT_panel),
                                std::move(TVlT_panel),
                                ib, max_panel_threads);

                // Find first local tile, which is triangular factor
                // (T in I - V T^H V^H), and copy it to TVlocal.
                for (int64_t i = 0; i < TVlT_panel.mt(); ++i) {
                    if (TVl_panel.tileIsLocal(0, i)) {
                        TVl_panel.tileInsert(0, i);
                        TVlT_panel.tileGetForReading( i, 0, HostNum, LayoutConvert(layout) );
                        TVl_panel.tileGetForWriting( 0, i, HostNum, LayoutConvert(layout) );
                        tile::gecopy( TVlT_panel(i, 0), TVl_panel(0, i) );
                        break;
                    }
                }

                // Copy result back.
                for (int64_t j = 0; j < V_panel.nt(); ++j) {
                    if (V_panel.tileIsLocal(0, j)) {
                        VT_panel.tileGetForReading( j, 0, HostNum, LayoutConvert(layout) );
                        V_panel.tileGetForWriting( 0, j, HostNum, LayoutConvert(layout) );
                        tile::deepConjTranspose( VT_panel(j, 0), V_panel(0, j) );
                    }
                }
                // todo: VT_panel.clear();
                //----------

                // triangle-triangle reductions
                // ttlqt handles tile transfers internally
                internal::ttlqt<Target::HostTask>(
                                std::move(V_panel),
                                std::move(TVr_panel));

                // if a trailing matrix exists
                if (k < A_mt-1) {

                    // bcast V down col for trailing matrix update
                    if (k+1 < A_nt) {
                        BcastList bcast_list_V_first;
                        BcastList bcast_list_V;
                        for (int64_t j = k+1; j < A_nt; ++j) {
                            // send A(k, j) down col A(k+1:mt-1, j)
                            // Vs in first_indices (except main diagonal one)
                            // need three lives.
                            if ((std::find(first_indices.begin(), first_indices.end(), j) != first_indices.end()) && (j > k+1)) {
                                bcast_list_V_first.push_back(
                                    {k, j, {A.sub(k+1, A_mt-1, j, j)}});
                            }
                            else {
                                bcast_list_V.push_back(
                                    {k, j, {A.sub(k+1, A_mt-1, j, j)}});
                            }
                        }
                        A.template listBcast(bcast_list_V_first, layout, 0, 3);
                        A.template listBcast(bcast_list_V, layout, 0, 2);
                    }

                    // bcast TVlocal down col for trailing matrix update
                    if (first_indices.size() > 0) {
                        BcastList bcast_list_T;
                        for (int64_t col : first_indices) {
                            bcast_list_T.push_back(
                                {k, col, {TVlocal.sub(k+1, A_mt-1, col, col)}});
                        }
                        TVlocal.template listBcast(bcast_list_T, layout);
                    }

                    // bcast TVreduce down col for trailing matrix update
                    if (first_indices.size() > 1) {
                        BcastList bcast_list_T;
                        for (int64_t col : first_indices) {
                            // Exclude first col of this panel,
                            // which doesn't have TVreduce tile.
                            if (col > k+1) {
                                bcast_list_T.push_back(
                                    {k, col, {TVreduce.sub(k+1, A_mt-1, col, col)}});
                            }
                        }
                        TVreduce.template listBcast(bcast_list_T, layout);
                    }
                }

                //--------------------
                // LQ update trailing submatrix
                if (k+1 < A_mt) {
                    int64_t i = k+1;
                    auto A_trail_i = A.sub(i, A_mt-1, k+1, A_nt-1);

                    // Apply local reflectors
                    internal::unmlq<target>(
                                    Side::Right, Op::ConjTrans,
                                    std::move(V_panel),
                                    std::move(TVl_panel),
                                    std::move(A_trail_i),
                                    W.sub(i, A_mt-1, k+1, A_nt-1));

                    // Apply triangle-triangle reduction reflectors
                    // ttmlq handles the tile broadcasting internally
                    internal::ttmlq<Target::HostTask>(
                                    Side::Right, Op::ConjTrans,
                                    std::move(V_panel),
                                    std::move(TVr_panel),
                                    std::move(A_trail_i),
                                    i);
                }
            }
        }

        #pragma omp taskwait
        A.tileUpdateAllOrigin();
    }

    A.releaseWorkspace();
}

} // namespace impl

//------------------------------------------------------------------------------
/// Distributed parallel reduction to band for 3-stage SVD.
///
/// Reduces an m-by-n matrix $A$ to band form using unitary transformations.
/// The factorization has the form
/// \[
///     A = U B V^H
/// \]
/// where $U$ and $V$ are unitary.
/// If m >= n, $B$ is upper triangular band with nb superdiagonals;
/// if m <  n, $B$ is lower triangular band with nb subdiagonals.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the m-by-n matrix $A$.
///     On exit:
///     - If m >= n, the elements Aij for j = i, ..., i+nb,
///       represent the upper triangular band matrix B.
///       The elements below the main diagonal, along with TU, represent
///       the unitary matrix $U$ as a product of elementary reflectors.
///       The elements above the nb-th superdiagonal, along with TV, represent
///       the unitary matrix $V$ as a product of elementary reflectors.
///     - If m < n, the elements Aij for j = i-nb, ..., i,
///       represent the lower triangular band matrix B.
///       The elements below the nb-th subdiagonal, along with TU, represent
///       the unitary matrix $U$ as a product of elementary reflectors.
///       The elements above the main diagonal, along with TV, represent
///       the unitary matrix $V$ as a product of elementary reflectors.
///
/// @param[out] TU
///     On exit, triangular matrices of the block reflectors for U.
///
/// @param[out] TV
///     On exit, triangular matrices of the block reflectors for V.
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
///     Note a lookahead is not possible with ge2tb due to dependencies from
///     updating on both left and right sides.
///
/// @ingroup svd_computational
///
template <typename scalar_t>
void ge2tb(
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t>& TU,
    TriangularFactors<scalar_t>& TV,
    Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            impl::ge2tb<Target::HostTask>( A, TU, TV, opts );
            break;

        case Target::HostNest:
            impl::ge2tb<Target::HostNest>( A, TU, TV, opts );
            break;

        case Target::HostBatch:
            impl::ge2tb<Target::HostBatch>( A, TU, TV, opts );
            break;

        case Target::Devices:
            impl::ge2tb<Target::Devices>( A, TU, TV, opts );
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void ge2tb<float>(
    Matrix<float>& A,
    TriangularFactors<float>& TU,
    TriangularFactors<float>& TV,
    Options const& opts);

template
void ge2tb<double>(
    Matrix<double>& A,
    TriangularFactors<double>& TU,
    TriangularFactors<double>& TV,
    Options const& opts);

template
void ge2tb< std::complex<float> >(
    Matrix< std::complex<float> >& A,
    TriangularFactors< std::complex<float> >& TU,
    TriangularFactors< std::complex<float> >& TV,
    Options const& opts);

template
void ge2tb< std::complex<double> >(
    Matrix< std::complex<double> >& A,
    TriangularFactors< std::complex<double> >& TU,
    TriangularFactors< std::complex<double> >& TV,
    Options const& opts);

} // namespace slate
