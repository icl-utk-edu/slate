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
/// An auxiliary routine to find each rank's first (top-most) row
/// in panel k.
///
/// @param[in] A_panel
///     Current panel, which is a sub of the input matrix $A$.
///
/// @param[in] k
///     Index of the current panel in the input matrix $A$.
///
/// @param[out] first_indices
///     The array of computed indices.
///
/// @ingroup geqrf_impl
///
template <typename scalar_t>
void geqrf_compute_first_indices(
    Matrix<scalar_t>& A_panel, int64_t k,
    std::vector< int64_t >& first_indices )
{
    // Find ranks in this column.
    std::set<int> ranks_set;
    A_panel.getRanks(&ranks_set);
    assert(ranks_set.size() > 0);

    // Find each rank's first (top-most) row in this panel,
    // where the triangular tile resulting from local geqrf panel
    // will reside.
    first_indices.reserve(ranks_set.size());
    for (int r: ranks_set) {
        for (int64_t i = 0; i < A_panel.mt(); ++i) {
            if (A_panel.tileRank(i, 0) == r) {
                first_indices.push_back(i+k);
                break;
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Distributed parallel QR factorization.
/// Generic implementation for any target.
/// Panel and lookahead computed on host using Host OpenMP task.
///
/// ColMajor layout is assumed
///
/// @ingroup geqrf_impl
///
template <Target target, typename scalar_t>
void geqrf(
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t>& T,
    Options const& opts )
{
    using BcastList = typename Matrix<scalar_t>::BcastList;
    using lapack::device_info_int;
    using blas::real;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    const int priority_zero = 0;
    const int priority_one = 1;
    const int life_factor_one = 1;

    // Options
    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );
    int64_t ib = get_option<int64_t>( opts, Option::InnerBlocking, 16 );
    int64_t max_panel_threads  = std::max(omp_get_max_threads()/2, 1);
    max_panel_threads = get_option<int64_t>( opts, Option::MaxPanelThreads,
                                             max_panel_threads );

    bool set_hold = lookahead > 0;  // Do tileGetAndHold in the bcast

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

    // setting up dummy variables for case the when target == host
    int64_t num_devices  = A.num_devices();
    int     panel_device = -1;
    size_t  work_size    = 0;

    std::vector< scalar_t* > dwork_array( num_devices, nullptr );

    if (target == Target::Devices) {
        const int64_t batch_size_zero = 0; // use default batch size
        const int64_t num_queues = 3 + lookahead;
        A.allocateBatchArrays(batch_size_zero, num_queues);
        A.reserveDeviceWorkspace();
        W.allocateBatchArrays(batch_size_zero, num_queues);
        // todo: this is demanding too much device workspace memory
        // only one tile-row of matrix W per MPI process is going to be used,
        // but W with size of whole A is being allocated
        // thus limiting the matrix size that can be processed
        // For now, allocate workspace tiles 1-by-1.
        //W.reserveDeviceWorkspace();

        // Find largest panel size and device for copying to
        // contiguous memory within internal geqrf routine
        int64_t mlocal = 0;
        int64_t first_panel_seen = -1;
        for (int64_t j = 0; j < A.nt(); ++j) {
            for (int64_t i = j; i < A.mt(); ++i) {
                if (A.tileIsLocal(i, j)) {
                    if (first_panel_seen < 0) {
                        first_panel_seen = j;
                    }
                    if (first_panel_seen == j) {
                        if (panel_device < 0) {
                            panel_device = A.tileDevice(i, j);
                        }
                        mlocal += A.tileMb(i);
                        // Asserting 1-D distribution for device
                        assert( panel_device == A.tileDevice(i, j) );
                    }
                }
            }
            if (first_panel_seen >= 0) {
                break;
            }
        }

        if (panel_device >= 0) {

            lapack::Queue* comm_queue = A.comm_queue(panel_device);

            int64_t nb       = A.tileNb(0);
            size_t  size_tau = (size_t) std::min( mlocal, nb );
            size_t  size_A   = (size_t) blas::max( 1, mlocal ) * nb;
            size_t  hsize, dsize;

            // Find size of the workspace needed
            lapack::geqrf_work_size_bytes( mlocal, nb, dwork_array[0], mlocal,
                                           &dsize, &hsize, *comm_queue );

            // Size of dA, dtau, dwork and dinfo
            work_size = size_A + size_tau + ceildiv(dsize, sizeof(scalar_t))
                        + ceildiv(sizeof(device_info_int), sizeof(scalar_t));

            for (int64_t dev = 0; dev < num_devices; ++dev) {
                lapack::Queue* queue = A.comm_queue( dev );
                dwork_array[dev] = blas::device_malloc<scalar_t>(work_size, *queue);
            }
        }
    }

    // QR tracks dependencies by block-column.
    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > block_vector(A_nt);
    uint8_t* block = block_vector.data();

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
        for (int64_t k = 0; k < A_min_mtnt; ++k) {
            auto  A_panel =       A.sub(k, A_mt-1, k, k);
            auto Tl_panel =  Tlocal.sub(k, A_mt-1, k, k);
            auto Tr_panel = Treduce.sub(k, A_mt-1, k, k);

            std::vector< int64_t > first_indices;
            geqrf_compute_first_indices(A_panel, k, first_indices);
            // todo: pass first_indices into internal geqrf or ttqrt?

            // panel, high priority
            #pragma omp task depend(inout:block[k]) priority(priority_one)
            {
                // local panel factorization
                internal::geqrf<target>(
                                std::move(A_panel),
                                std::move(Tl_panel),
                                dwork_array, work_size,
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
                        A.template listBcast<target>(bcast_list_V_first, layout, 0, 3, set_hold);
                        A.template listBcast<target>(bcast_list_V, layout, 0, 2, set_hold);
                    }

                    // bcast Tlocal across row for trailing matrix update
                    if (first_indices.size() > 0) {
                        BcastList bcast_list_T;
                        for (int64_t row : first_indices) {
                            bcast_list_T.push_back({row, k, {Tlocal.sub(row, row, k+1, A_nt-1)}});
                        }
                        Tlocal.template listBcast<target>(bcast_list_T, layout, k, life_factor_one, set_hold);
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
                    internal::unmqr<target>(
                                    Side::Left, Op::ConjTrans,
                                    std::move(A_panel),
                                    std::move(Tl_panel),
                                    std::move(A_trail_j),
                                    W.sub(k, A_mt-1, j, j),
                                    priority_one, j-k+1);

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
                    // Apply local reflectors.
                    internal::unmqr<target>(
                                    Side::Left, Op::ConjTrans,
                                    std::move(A_panel),
                                    std::move(Tl_panel),
                                    std::move(A_trail_j),
                                    W.sub(k, A_mt-1, j, A_nt-1),
                                    priority_zero, j-k+1);

                    // Apply triangle-triangle reduction reflectors.
                    // ttmqr handles the tile broadcasting internally.
                    internal::ttmqr<Target::HostTask>(
                                    Side::Left, Op::ConjTrans,
                                    std::move(A_panel),
                                    std::move(Tr_panel),
                                    std::move(A_trail_j),
                                    j);
                }
            }
            if (target == Target::Devices) {
                // Update the status of the on-hold tiles held by the invocation of
                // the tileBcast routine, and then release them to free up memory.
                // The origin must be updated with the latest modified copy
                // for memory consistency.
                // TODO: Find better solution to handle tile release, and
                //       investigate the correctness of the task dependency
                if (k >= lookahead && k < A_nt-1) {
                    #pragma omp task depend(in:block[k]) \
                                     depend(inout:block[k+1])
                    {
                        int64_t k_la = k-lookahead;
                        for (int64_t i = k_la; i < A_mt; ++i) {
                            if (A.tileIsLocal(i, k_la)) {
                                A.tileUpdateOrigin(i, k_la);

                                std::set<int> dev_set;
                                A.sub(i, i, k_la+1, A_nt-1).getLocalDevices(&dev_set);

                                for (auto device : dev_set) {
                                    A.tileUnsetHold(i, k_la, device);
                                    A.tileRelease(i, k_la, device);
                                }
                            }
                        }

                        auto A_panel_k_la = A.sub(k_la, A_mt-1, k_la, k_la);
                        std::vector< int64_t > first_indices_k_la;
                        geqrf_compute_first_indices(A_panel_k_la, k_la, first_indices_k_la);
                        if (first_indices.size() > 0) {
                            for (int64_t row : first_indices_k_la) {
                                if (Tlocal.tileIsLocal(row, k_la)) {
                                    Tlocal.tileUpdateOrigin(row, k_la);

                                    std::set<int> dev_set;
                                    Tlocal.sub(row, row, k_la+1, A_nt-1).getLocalDevices(&dev_set);

                                    for (auto device : dev_set) {
                                        Tlocal.tileUnsetHold(row, k_la, device);
                                        Tlocal.tileRelease(row, k_la, device);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        #pragma omp taskwait
        A.tileUpdateAllOrigin();
    }

    A.releaseWorkspace();

    if (target == Target::Devices) {
        for (int64_t dev = 0; dev < num_devices; ++dev) {
            blas::Queue* queue = A.comm_queue( dev );
            blas::device_free( dwork_array[dev], *queue );
            dwork_array[dev] = nullptr;
        }
    }
}

} // namespace impl

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
/// Complexity (in real):
/// - for $m \ge n$, $\approx 2 m n^{2} - \frac{2}{3} n^{3}$ flops;
/// - for $m \le n$, $\approx 2 m^{2} n - \frac{2}{3} m^{3}$ flops;
/// - for $m = n$,   $\approx \frac{4}{3} n^{3}$ flops.
/// .
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
void geqrf(
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t>& T,
    Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            impl::geqrf<Target::HostTask>( A, T, opts );
            break;

        case Target::HostNest:
            impl::geqrf<Target::HostNest>( A, T, opts );
            break;

        case Target::HostBatch:
            impl::geqrf<Target::HostBatch>( A, T, opts );
            break;

        case Target::Devices:
            impl::geqrf<Target::Devices>( A, T, opts );
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
