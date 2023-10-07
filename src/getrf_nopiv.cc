// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// Distributed parallel LU factorization without pivoting.
/// Generic implementation for any target.
/// Panel and lookahead computed on host using Host OpenMP task.
/// @ingroup gesv_impl
///
template <Target target, typename scalar_t>
int64_t getrf_nopiv(
    Matrix<scalar_t>& A,
    Options const& opts )
{
    using BcastList = typename Matrix<scalar_t>::BcastList;
    using BcastListTag = typename Matrix<scalar_t>::BcastListTag;

    // Constants
    const scalar_t one = 1.0;
    const int life_1 = 1;
    const int priority_0 = 0;
    const int priority_1 = 1;
    const int queue_0 = 0;
    const int queue_1 = 1;
    const Layout layout = Layout::ColMajor;

    // Options
    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );
    int64_t ib = get_option<int64_t>( opts, Option::InnerBlocking, 16 );

    if (target == Target::Devices) {
        // two batch arrays plus one for each lookahead
        // batch array size will be set as needed
        A.allocateBatchArrays(0, 2 + lookahead);
        A.reserveDeviceWorkspace();
    }

    int64_t info = 0;
    int64_t A_nt = A.nt();
    int64_t A_mt = A.mt();
    int64_t min_mt_nt = std::min(A.mt(), A.nt());
    bool is_shared = lookahead > 0;

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > column_vector(A_nt);
    std::vector< uint8_t > diag_vector(A_nt);
    uint8_t* column = column_vector.data();
    uint8_t* diag = diag_vector.data();
    SLATE_UNUSED( column ); // Used only by OpenMP
    SLATE_UNUSED( diag   ); // Used only by OpenMP

    // Running two listBcastMT's simultaneously can hang due to task ordering
    // This dependency avoids that
    uint8_t listBcastMT_token;
    SLATE_UNUSED( listBcastMT_token ); // Used only by OpenMP

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
        int64_t kk = 0;  // column index (not block-column)
        for (int64_t k = 0; k < min_mt_nt; ++k) {

            // panel, high priority
            #pragma omp task depend(inout:column[k]) \
                             depend(out:diag[k]) \
                             priority(1)
            {
                // factor A(k, k)
                int64_t iinfo;
                internal::getrf_nopiv<Target::HostTask>(
                    A.sub(k, k, k, k), ib, priority_1, &iinfo );
                if (info == 0 && iinfo > 0) {
                    info = kk + iinfo;
                }

                // Update panel
                int tag_k = k;
                BcastList bcast_list_A;
                bcast_list_A.push_back({k, k, {A.sub(k+1, A_mt-1, k, k),
                                               A.sub(k, k, k+1, A_nt-1)}});
                A.template listBcast<target>(
                    bcast_list_A, layout, tag_k, life_1, true );
            }

            #pragma omp task depend(inout:column[k]) \
                             depend(in:diag[k]) \
                             depend(inout:listBcastMT_token) \
                             priority(1)
            {
                auto Akk = A.sub(k, k, k, k);
                auto Tkk = TriangularMatrix<scalar_t>(Uplo::Upper, Diag::NonUnit, Akk);

                internal::trsm<target>(
                    Side::Right,
                    one, std::move( Tkk ), A.sub(k+1, A_mt-1, k, k),
                    priority_1, layout, queue_0 );


                BcastListTag bcast_list;
                // bcast the tiles of the panel to the right hand side
                for (int64_t i = k+1; i < A_mt; ++i) {
                    // send A(i, k) across row A(i, k+1:nt-1)
                    const int64_t tag = i;
                    bcast_list.push_back({i, k, {A.sub(i, i, k+1, A_nt-1)}, tag});
                }
                A.template listBcastMT<target>(
                  bcast_list, layout, life_1, is_shared );
            }
            // update lookahead column(s), high priority
            for (int64_t j = k+1; j < k+1+lookahead && j < A_nt; ++j) {
                #pragma omp task depend(in:diag[k]) \
                                 depend(inout:column[j]) \
                                 priority(1)
                {
                    int tag_j = j;
                    int queue_jk1 = j-k+1;
                    auto Akk = A.sub(k, k, k, k);
                    auto Tkk =
                        TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, Akk);

                    // solve A(k, k) A(k, j) = A(k, j)
                    internal::trsm<target>(
                        Side::Left,
                        one, std::move( Tkk ), A.sub(k, k, j, j),
                        priority_1, layout, queue_jk1 );

                    // send A(k, j) across column A(k+1:mt-1, j)
                    A.tileBcast(k, j, A.sub(k+1, A_mt-1, j, j), layout, tag_j);
                }

                #pragma omp task depend(in:column[k]) \
                                 depend(inout:column[j]) \
                                 priority(1)
                {
                    int queue_jk1 = j-k+1;
                    // A(k+1:mt-1, j) -= A(k+1:mt-1, k) * A(k, j)
                    internal::gemm<target>(
                        -one, A.sub(k+1, A_mt-1, k, k),
                              A.sub(k, k, j, j),
                        one,  A.sub(k+1, A_mt-1, j, j),
                        layout, priority_1, queue_jk1 );
                }
            }
            // update trailing submatrix, normal priority
            if (k+1+lookahead < A_nt) {
                #pragma omp task depend(in:diag[k]) \
                                 depend(inout:column[k+1+lookahead]) \
                                 depend(inout:column[A_nt-1]) \
                                 depend(inout:listBcastMT_token)
                {
                    auto Akk = A.sub(k, k, k, k);
                    auto Tkk =
                        TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, Akk);

                    // solve A(k, k) A(k, kl+1:nt-1) = A(k, kl+1:nt-1)
                    internal::trsm<target>(
                        Side::Left,
                        one, std::move( Tkk ),
                             A.sub(k, k, k+1+lookahead, A_nt-1),
                        priority_0, layout, queue_1 );

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
                        one,  A.sub(k+1, A_mt-1, k+1+lookahead, A_nt-1),
                        layout, priority_0, queue_1 );
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
            kk += A.tileNb( k );
        }

        #pragma omp taskwait
        A.tileUpdateAllOrigin();
    }
    A.clearWorkspace();

    internal::reduce_info( &info, A.mpiComm() );
    return info;
}

} // namespace impl

//------------------------------------------------------------------------------
/// Distributed parallel LU factorization without pivoting.
///
/// Computes an LU factorization without pivoting of a general m-by-n matrix $A$
///
/// The factorization has the form
/// \[
///     A = L U
/// \]
/// where $L$ is lower triangular with unit diagonal elements
/// (lower trapezoidal if m > n), and $U$ is upper triangular
/// (upper trapezoidal if m < n).
///
/// This is the right-looking Level 3 BLAS version of the algorithm.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the matrix $A$ to be factored.
///     On exit, the factors $L$ and $U$ from the factorization $A = P L U$;
///     the unit diagonal elements of $L$ are not stored.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Lookahead:
///       Number of panels to overlap with matrix updates.
///       lookahead >= 0. Default 1.
///     - Option::InnerBlocking:
///       Inner blocking to use for panel. Default 16.
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// @return 0: successful exit
/// @return i > 0: $U(i,i)$ is exactly zero, where $i$ is a 1-based index.
///         The factorization will have NaN due to division by zero.
///
/// @ingroup gesv_computational
///
template <typename scalar_t>
int64_t getrf_nopiv(
    Matrix<scalar_t>& A,
    Options const& opts )
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            return impl::getrf_nopiv<Target::HostTask>( A, opts );

        case Target::HostNest:
            return impl::getrf_nopiv<Target::HostNest>( A, opts );

        case Target::HostBatch:
            return impl::getrf_nopiv<Target::HostBatch>( A, opts );

        case Target::Devices:
            return impl::getrf_nopiv<Target::Devices>( A, opts );
    }
    return -2;  // shouldn't happen
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
int64_t getrf_nopiv<float>(
    Matrix<float>& A,
    Options const& opts);

template
int64_t getrf_nopiv<double>(
    Matrix<double>& A,
    Options const& opts);

template
int64_t getrf_nopiv< std::complex<float> >(
    Matrix< std::complex<float> >& A,
    Options const& opts);

template
int64_t getrf_nopiv< std::complex<double> >(
    Matrix< std::complex<double> >& A,
    Options const& opts);

} // namespace slate
