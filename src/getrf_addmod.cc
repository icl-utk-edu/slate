// Copyright (c) 2022-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/addmod.hh"
#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

namespace internal {

//------------------------------------------------------------------------------
/// Distributed parallel LU factorization with additive modifications
/// Generic implementation for any target.
/// Panel and lookahead computed on host using Host OpenMP task.
/// @ingroup gesv_specialization
///
template <Target target, typename scalar_t>
void getrf_addmod(Matrix<scalar_t>& A, AddModFactors<scalar_t>& W,
                  Options const& opts)
{
    using BcastList = typename Matrix<scalar_t>::BcastList;
    using BcastListTag = typename Matrix<scalar_t>::BcastListTag;
    using real_t = blas::real_type<scalar_t>;

    Layout layout = Layout::ColMajor;

    const scalar_t one = 1.0;
    const scalar_t zero = 0.0;

    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );
    int64_t ib = get_option<int64_t>( opts, Option::InnerBlocking, 16 );
    real_t mod_tol = get_option<double>( opts, Option::AdditiveTolerance, -1e-8 );
    bool useWoodbury = get_option<int64_t>( opts, Option::UseWoodbury, 1 );


    if (mod_tol < 0) {
        // When Target::Device, we don't want norm to move tiles back from device
        // So, we set the hold here to prevent norm from removing the device copy at the end
        // Then use tileUnsetHoldAllOnDevice to remove the hold but not the device copy
        if (target == Target::Devices) {
            #pragma omp parallel
            #pragma omp master
            #pragma omp taskgroup
            {
                #pragma omp task slate_omp_default_none \
                    shared( A ) firstprivate( layout )
                {
                    A.tileGetAndHoldAllOnDevices( LayoutConvert( layout ) );
                }
            }
        }

        mod_tol *= -1 * slate::norm(slate::Norm::Fro, A, opts);

        if (target == Target::Devices) {
            A.tileUnsetHoldAllOnDevices();
        }
    }


    if (target == Target::Devices) {
        // two batch arrays plus one for each lookahead
        // batch array size will be set as needed
        A.allocateBatchArrays(0, 2 + lookahead);
        A.reserveDeviceWorkspace();
    }

    MPI_Comm comm = A.mpiComm();
    const MPI_Datatype mpi_scalar_t = mpi_type<scalar_t>::value;

    const int priority_one = 1;
    const int priority_zero = 0;
    int64_t A_nt = A.nt();
    int64_t A_mt = A.mt();
    int64_t min_mt_nt = std::min(A.mt(), A.nt());
    int life_factor_one = 1;
    bool is_shared = lookahead > 0;

    W.block_size = ib;
    W.A = A;
    W.U_factors = A.emptyLike();
    W.VT_factors = A.emptyLike();
    W.singular_values.resize(min_mt_nt);
    W.modifications.resize(min_mt_nt);
    W.modification_indices.resize(min_mt_nt);

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > column_vector(A_nt);
    std::vector< uint8_t > diag_vector(A_nt+1);
    uint8_t* column = column_vector.data();
    uint8_t* diag = diag_vector.data();
    // Running two listBcastMT's simultaneously can hang due to task ordering
    // This dependency avoids that
    uint8_t listBcastMT_token;
    SLATE_UNUSED(listBcastMT_token); // Only used by OpenMP

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
        for (int64_t k = 0; k < min_mt_nt; ++k) {

            // panel, high priority
            #pragma omp task depend(inout:column[k]) \
                             depend(out:diag[k]) \
                             priority(priority_one)
            {
                auto& local_sig_vals = W.singular_values[k];
                auto& local_mod_vals = W.modifications[k];
                auto& local_mod_inds = W.modification_indices[k];

                int64_t mb = A.tileMb(k);
                local_sig_vals.resize(mb);

                if (A.tileIsLocal(k,k)) {
                    W.U_factors.tileInsert(k, k, HostNum);
                    W.VT_factors.tileInsert(k, k, HostNum);
                }

                // factor A(k, k)
                internal::getrf_addmod<Target::HostTask>(
                                A.sub(k, k, k, k),
                                W.U_factors.sub(k, k, k, k),
                                W.VT_factors.sub(k, k, k, k),
                                std::move(local_sig_vals),
                                std::move(local_mod_vals), std::move(local_mod_inds),
                                mod_tol, ib);

                // broadcast singular values
                MPI_Request req_sig_vals;
                slate_mpi_call(
                    MPI_Ibcast(local_sig_vals.data(), mb, mpi_scalar_t,
                               A.tileRank(k, k), comm, &req_sig_vals) );

                // Update panel
                int tag_k = k;
                BcastList bcast_list;
                bcast_list.push_back({k, k, {A.sub(k+1, A_mt-1, k, k),
                                               A.sub(k, k, k+1, A_nt-1)}});
                A.template listBcast<target>(
                    bcast_list, layout, tag_k, life_factor_one, true);
                W.U_factors.template listBcast<target>(
                    bcast_list, layout, tag_k, life_factor_one, true);
                W.VT_factors.template listBcast<target>(
                    bcast_list, layout, tag_k, life_factor_one, true);

                // Allow concurrent Bcast's
                slate_mpi_call( MPI_Wait(&req_sig_vals, MPI_STATUS_IGNORE) );
            }

            #pragma omp task depend(in:diag[k]) depend(in:diag[k+1])
            {
                int64_t kk_root = A.tileRank(k,k);

                auto& local_mod_inds = W.modification_indices[k];
                auto& local_mod_vals = W.modifications[k];
                int64_t num_mods = local_mod_vals.size();

                slate_mpi_call(
                    MPI_Bcast(&num_mods, 1, MPI_INT64_T,
                              kk_root, comm) );

                local_mod_vals.resize(num_mods);
                local_mod_inds.resize(num_mods);

                if (num_mods != 0) {

                    MPI_Request requests[2];
                    slate_mpi_call(
                        MPI_Ibcast(local_mod_vals.data(), num_mods, mpi_scalar_t,
                                   kk_root, comm, &requests[0]) );
                    slate_mpi_call(
                        MPI_Ibcast(local_mod_inds.data(), num_mods, MPI_INT64_T,
                                   kk_root, comm, &requests[1]) );

                    slate_mpi_call( MPI_Waitall(2, requests, MPI_STATUSES_IGNORE) );
                }
            }

            #pragma omp task depend(inout:column[k]) \
                             depend(in:diag[k]) \
                             depend(inout:listBcastMT_token) \
                             priority(priority_one)
            {

                internal::trsm_addmod<target>(
                    Side::Right, Uplo::Upper,
                    scalar_t(1.0), A.sub(k, k, k, k),
                                   W.U_factors.sub(k, k, k, k),
                                   W.VT_factors.sub(k, k, k, k),
                                   std::move(W.singular_values[k]),
                                   A.sub(k+1, A_mt-1, k, k),
                    ib, priority_one, layout, 0);


                BcastListTag bcast_list;
                // bcast the tiles of the panel to the trailing matrix
                for (int64_t i = k+1; i < A_mt; ++i) {
                    // send A(i, k) across row A(i, k+1:nt-1)
                    const int64_t tag = i;
                    bcast_list.push_back({i, k, {A.sub(i, i, k+1, A_nt-1)}, tag});
                }
                A.template listBcastMT<target>(
                  bcast_list, layout, life_factor_one, is_shared);
            }
            // update lookahead column(s), high priority
            for (int64_t j = k+1; j < k+1+lookahead && j < A_nt; ++j) {
                #pragma omp task depend(in:diag[k]) \
                                 depend(inout:column[j]) \
                                 priority(priority_one)
                {
                    int tag_j = j;

                    // solve A(k, k) A(k, j) = A(k, j)
                    internal::trsm_addmod<target>(
                        Side::Left, Uplo::Lower,
                        scalar_t(1.0), A.sub(k, k, k, k),
                                       W.U_factors.sub(k, k, k, k),
                                       W.VT_factors.sub(k, k, k, k),
                                       std::move(W.singular_values[k]),
                                       A.sub(k, k, j, j),
                        ib, priority_one, layout, j-k+1);

                    // send A(k, j) across column A(k+1:mt-1, j)
                    A.tileBcast(k, j, A.sub(k+1, A_mt-1, j, j), layout, tag_j);
                }

                #pragma omp task depend(in:column[k]) \
                                 depend(inout:column[j]) \
                                 priority(priority_one)
                {
                    // A(k+1:mt-1, j) -= A(k+1:mt-1, k) * A(k, j)
                    internal::gemm<target>(
                        -one, A.sub(k+1, A_mt-1, k, k),
                              A.sub(k, k, j, j),
                         one, A.sub(k+1, A_mt-1, j, j),
                        layout, priority_one, j-k+1);
                }
            }
            // update trailing submatrix, normal priority
            if (k+1+lookahead < A_nt) {
                #pragma omp task depend(in:diag[k]) \
                                 depend(inout:column[k+1+lookahead]) \
                                 depend(inout:column[A_nt-1]) \
                                 depend(inout:listBcastMT_token)
                {
                    // solve A(k, k) A(k, kl+1:nt-1) = A(k, kl+1:nt-1)
                    internal::trsm_addmod<target>(
                        Side::Left, Uplo::Lower,
                        scalar_t(1.0), A.sub(k, k, k, k),
                                       W.U_factors.sub(k, k, k, k),
                                       W.VT_factors.sub(k, k, k, k),
                                       std::move(W.singular_values[k]),
                                       A.sub(k, k, k+1+lookahead, A_nt-1),
                        ib, priority_zero, layout, 1);
                    // send A(k, kl+1:A_nt-1) across A(k+1:mt-1, kl+1:nt-1)
                    BcastListTag bcast_list;
                    for (int64_t j = k+1+lookahead; j < A_nt; ++j) {
                        // send A(k, j) across column A(k+1:mt-1, j)
                        // tag must be distinct from sending left panel
                        const int64_t tag = j + A_mt;
                        bcast_list.push_back({k, j, {A.sub(k+1, A_mt-1, j, j)},
                                              tag});
                    }
                    A.template listBcastMT<target>(
                        bcast_list, layout);
                }

                #pragma omp task depend(in:column[k]) \
                                 depend(inout:column[k+1+lookahead]) \
                                 depend(inout:column[A_nt-1])
                {
                    // A(k+1:mt-1, kl+1:nt-1) -= A(k+1:mt-1, k) * A(k, kl+1:nt-1)
                    internal::gemm<target>(
                        -one, A.sub(k+1, A_mt-1, k, k),
                              A.sub(k, k, k+1+lookahead, A_nt-1),
                         one, A.sub(k+1, A_mt-1, k+1+lookahead, A_nt-1),
                        layout, priority_zero, 1);
                }
            }
            if (target == Target::Devices) {
                #pragma omp task depend(inout:diag[k])
                {
                    if (A.tileIsLocal(k, k) && k+1 < A_nt) {
                        std::set<int> dev_set;
                        A.sub(k+1, A_mt-1, k, k).getLocalDevices(&dev_set);
                        A.sub(k, k, k+1, A_nt-1).getLocalDevices(&dev_set);

                        for (auto device : dev_set) {
                            A.tileUnsetHold(k, k, device);
                            A.tileRelease(k, k, device);
                            W.U_factors.tileUnsetHold(k, k, device);
                            W.U_factors.tileRelease(k, k, device);
                            W.VT_factors.tileUnsetHold(k, k, device);
                            W.VT_factors.tileRelease(k, k, device);
                        }
                    }
                }
                if (is_shared) {
                    #pragma omp task depend(inout:column[k])
                    {
                        for (int64_t i = k+1; i < A_mt; ++i) {
                            if (A.tileIsLocal(i, k)) {
                                A.tileUpdateOrigin(i, k);

                                std::set<int> dev_set;
                                A.sub(i, i, k+1, A_nt-1).getLocalDevices(&dev_set);

                                for (auto device : dev_set) {
                                    A.tileUnsetHold(i, k, device);
                                    A.tileRelease(i, k, device);
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

    // Build and factor capacitance matrix, if needed
    int64_t inner_dim = 0;
    for (int64_t i = 0; i < A_mt; ++i) {
        inner_dim += W.modifications[i].size();
    }
    if (A.mpiRank() == 0) std::cout << "Factored A w/ " << inner_dim << " modifications" << std::endl;
    inner_dim *= useWoodbury; // discard inner_dim unless using Woodbury formula
    W.num_modifications = inner_dim;
    if (inner_dim > 0) {

        auto A_tileMb     = A.tileMbFunc();
        auto A_tileRank   = A.tileRankFunc();
        auto A_tileDevice = A.tileDeviceFunc();

        W.capacitance_matrix = Matrix<scalar_t>(inner_dim, inner_dim,
                                                A_tileMb, A_tileMb,
                                                A_tileRank, A_tileDevice, A.mpiComm());
        W.capacitance_matrix.insertLocalTiles();
        W.S_VT_Rinv = Matrix<scalar_t>(inner_dim, A.m(), A_tileMb, A_tileMb,
                                       A_tileRank, A_tileDevice, A.mpiComm());
        W.S_VT_Rinv.insertLocalTiles();
        W.Linv_U = Matrix<scalar_t>(A.n(), inner_dim, A_tileMb, A_tileMb,
                                       A_tileRank, A_tileDevice, A.mpiComm());
        W.Linv_U.insertLocalTiles();

        // Build S_VT_Rinv and Linv_U
        // First, create sparse S_VT and U matrices
        // Then, apply Rinv and Linv
        #pragma omp parallel
        #pragma omp master
        {
            internal::set<Target::HostTask>(zero, zero, std::move(W.S_VT_Rinv));
            internal::set<Target::HostTask>(zero, zero, std::move(W.Linv_U));
            #pragma omp taskwait
            int64_t tile = 0, tile_offset = 0;
            for (int64_t i = 0; i < A_mt; ++i) {
                // NB The nonzeros in a row (resp column) are all within the same tile
                auto mod_inds = W.modification_indices[i];
                auto mod_vals = W.modifications[i];
                int64_t num_mods = mod_inds.size();
                int64_t mod_offset = 0;

                while (mod_offset < num_mods) {
                    int64_t chunk = std::min(num_mods - mod_offset,
                                             W.capacitance_matrix.tileMb(tile) - tile_offset);

                    if (W.S_VT_Rinv.tileIsLocal(tile, i)) {
                        #pragma omp task firstprivate(tile, i, chunk, mod_offset) \
                                         firstprivate(mod_inds, mod_vals, num_mods)
                        {
                            W.VT_factors.tileRecv(i, i, A.tileRank(i, i), layout, 2*i);
                            W.VT_factors.tileGetForReading(i, i, LayoutConvert(layout));
                            auto tile_VT = W.VT_factors(i, i);
                            W.S_VT_Rinv.tileGetForWriting(tile, i, LayoutConvert(layout));
                            auto tile_CVT = W.S_VT_Rinv(tile, i);

                            for (int ii = 0; ii < chunk; ++ii) {
                                auto s = mod_vals[mod_offset+ii];
                                auto ind = mod_inds[mod_offset+ii];
                                auto col_offset = (ind / ib)*ib;

                                int64_t kb = std::min(A.tileNb(i)-col_offset, ib);
                                for (int jj = 0; jj < kb; ++jj) {
                                    tile_CVT.at(tile_offset+ii, jj+col_offset)
                                        = s*tile_VT(ind, jj+col_offset);
                                }
                            }
                            W.VT_factors.tileTick(i, i);
                        }
                    }
                    else if (W.VT_factors.tileIsLocal(i, i)) {
                        #pragma omp task firstprivate(tile, i)
                        {
                            W.VT_factors.tileSend(i, i, W.S_VT_Rinv.tileRank(tile, i), 2*i);
                        }
                    }

                    if (W.Linv_U.tileIsLocal(i, tile)) {
                        #pragma omp task firstprivate(tile, i, chunk, mod_offset) \
                                         firstprivate(mod_inds, mod_vals, num_mods)
                        {
                            W.U_factors.tileRecv(i, i, W.U_factors.tileRank(i, i), layout, 2*i+1);
                            W.U_factors.tileGetForReading(i, i, LayoutConvert(layout));
                            auto tile_U = W.U_factors(i, i);
                            W.Linv_U.tileGetForWriting(i, tile, LayoutConvert(layout));
                            auto tile_CU = W.Linv_U(i, tile);

                            for (int jj = 0; jj < chunk; ++jj) {
                                auto ind = mod_inds[mod_offset+jj];
                                auto row_offset = (ind / ib)*ib;

                                int64_t kb = std::min(A.tileMb(i)-row_offset, ib);
                                for (int ii = 0; ii < kb; ++ii) {
                                    tile_CU.at(ii+row_offset, tile_offset+jj)
                                        = tile_U(ii+row_offset, ind);
                                }
                            }
                            W.U_factors.tileTick(i, i);
                        }
                    }
                    else if (W.U_factors.tileIsLocal(i, i)) {
                        #pragma omp task firstprivate(tile, i)
                        {
                            W.U_factors.tileSend(i, i, W.Linv_U.tileRank(i, tile), 2*i+1);
                        }
                    }

                    tile_offset += chunk;
                    if (tile_offset >= W.capacitance_matrix.tileMb(tile)) {
                        tile += 1;
                        tile_offset = 0;
                    }

                    mod_offset += chunk;
                }
            }
        }

        trsm_addmod(Side::Right, Uplo::Upper, one, W, W.S_VT_Rinv, opts);
        trsm_addmod(Side::Left,  Uplo::Lower, one, W, W.Linv_U, opts);

        // build & factor capacitance matrix
        set(zero, one, W.capacitance_matrix, opts);
        gemm(-one, W.S_VT_Rinv, W.Linv_U,
              one, W.capacitance_matrix,
             opts);

        getrf(W.capacitance_matrix, W.capacitance_pivots, opts);
    }
    A.clearWorkspace();
}

} // namespace internal

// TODO docs
template <typename scalar_t>
void getrf_addmod(Matrix<scalar_t>& A, AddModFactors<scalar_t>& W,
                  Options const& opts)
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            internal::getrf_addmod<Target::HostTask>(A, W, opts);
            break;
        case Target::HostNest:
            internal::getrf_addmod<Target::HostNest>(A, W, opts);
            break;
        case Target::HostBatch:
            internal::getrf_addmod<Target::HostBatch>(A, W, opts);
            break;
        case Target::Devices:
            internal::getrf_addmod<Target::Devices>(A, W, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void getrf_addmod<float>(
    Matrix<float>& A, AddModFactors<float>& W,
    Options const& opts);

template
void getrf_addmod<double>(
    Matrix<double>& A, AddModFactors<double>& W,
    Options const& opts);

template
void getrf_addmod< std::complex<float> >(
    Matrix< std::complex<float> >& A, AddModFactors< std::complex<float> >& W,
    Options const& opts);

template
void getrf_addmod< std::complex<double> >(
    Matrix< std::complex<double> >& A, AddModFactors< std::complex<double> >& W,
    Options const& opts);

} // namespace slate
