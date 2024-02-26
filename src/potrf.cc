// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// Distributed parallel Cholesky factorization.
/// Generic implementation for any target.
/// @ingroup posv_impl
///
template <Target target, typename scalar_t>
int64_t potrf(
    slate::internal::TargetType<target>,
    HermitianMatrix<scalar_t> A,
    Options const& opts )
{
    using real_t = blas::real_type<scalar_t>;
    using BcastListTag = typename Matrix<scalar_t>::BcastListTag;

    // Constants
    const scalar_t one = 1.0;
    const int priority_0 = 0;
    const int queue_0 = 0;
    const int queue_1 = 1;
    const int queue_2 = 2;
    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // Options
    int64_t lookahead = get_option<Option::Lookahead>( opts, 1 );
    bool hold_local_workspace = get_option<Option::HoldLocalWorkspace>( opts, false );

    // if upper, change to lower
    if (A.uplo() == Uplo::Upper) {
        A = conj_transpose( A );
    }

    int64_t info = 0;
    int64_t A_nt = A.nt();

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > column_vector(A_nt);
    uint8_t* column = column_vector.data();
    SLATE_UNUSED( column ); // Used only by OpenMP

    // Allocate batch arrays = number of kernels without lookahead + lookahead
    // number of kernels without lookahead = 3
    // (internal::potrf, internal::gemm, and internal::trsm)
    // whereas internal::herk will be executed as many as lookaheads, thus
    // internal::herk needs batch arrays equal to the number of lookaheads
    // and the batch_arrays_index starts from
    // the number of kernels without lookahead, and then incremented by 1
    // for every execution for the internal::herk
    const int64_t batch_size_default = 0;
    int num_queues = 3 + lookahead;  // Number of kernels with lookahead
    using lapack::device_info_int;
    std::vector< device_info_int* > device_info_array( A.num_devices(), nullptr );

    if (target == Target::Devices) {
        A.allocateBatchArrays( batch_size_default, num_queues );
        A.reserveDeviceWorkspace();

        // Allocate
        for (int64_t dev = 0; dev < A.num_devices(); ++dev) {
            blas::Queue* queue = A.comm_queue(dev);
            device_info_array[dev] = blas::device_malloc<device_info_int>( 1, *queue );
        }
    }

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
        int64_t kk = 0;  // column index (not block-column)
        for (int64_t k = 0; k < A_nt; ++k) {
            // Panel, normal priority
            #pragma omp task depend(inout:column[k]) priority( priority_0 ) \
                shared( info )
            {
                // factor A(k, k)
                int64_t iinfo;
                if (target == Target::Devices) {
                    iinfo = internal::potrf<target>(
                        A.sub(k, k), priority_0, queue_2,
                        device_info_array[ A.tileDevice( k, k ) ] );
                }
                else {
                    iinfo = internal::potrf<target>(
                        A.sub(k, k), priority_0, queue_2 );
                }
                if (iinfo != 0 && info == 0)
                    info = kk + iinfo;

                // send A(k, k) down col A(k+1:nt-1, k)
                if (k+1 <= A_nt-1)
                    A.tileBcast(k, k, A.sub(k+1, A_nt-1, k, k), layout);

                // A(k+1:nt-1, k) * A(k, k)^{-H}
                if (k+1 <= A_nt-1) {
                    auto Akk = A.sub(k, k);
                    auto Tkk = TriangularMatrix< scalar_t >(Diag::NonUnit, Akk);
                    internal::trsm<target>(
                        Side::Right,
                        one, conj_transpose( Tkk ),
                        A.sub(k+1, A_nt-1, k, k),
                        priority_0, layout, queue_1 );
                }

                BcastListTag bcast_list_A;
                for (int64_t i = k+1; i < A_nt; ++i) {
                    // send A(i, k) across row A(i, k+1:i) and
                    //                down col A(i:nt-1, i) with msg tag i
                    bcast_list_A.push_back({i, k, {A.sub(i, i, k+1, i),
                                                   A.sub(i, A_nt-1, i, i)},
                                            i});
                }

                A.template listBcastMT<target>(
                  bcast_list_A, layout);
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
                        real_t( 1.0), A.sub(k+1+lookahead, A_nt-1),
                        priority_0, queue_0, layout );
                }
            }

            // update lookahead column(s), normal priority
            // the batch_arrays_index_la must be initialized to the
            // lookahead base index (i.e, number of kernels without lookahead),
            // which is equal to "2" for slate::potrf, and then the variable is
            // incremented with every lookahead column "j" ( j-k+1 = 2+j-(k+1) )
            for (int64_t j = k+1; j < k+1+lookahead && j < A_nt; ++j) {
                #pragma omp task depend(in:column[k]) \
                                 depend(inout:column[j])
                {
                    // A(j, j) -= A(j, k) * A(j, k)^H
                    int queue_jk2 = j-k+2;
                    internal::herk<target>(
                        real_t(-1.0), A.sub(j, j, k, k),
                        real_t( 1.0), A.sub(j, j),
                        priority_0, queue_jk2, layout );

                    // A(j+1:nt, j) -= A(j+1:nt-1, k) * A(j, k)^H
                    if (j+1 <= A_nt-1) {
                        auto Ajk = A.sub(j, j, k, k);
                        internal::gemm<target>(
                            -one, A.sub(j+1, A_nt-1, k, k),
                                  conj_transpose( Ajk ),
                            one,  A.sub(j+1, A_nt-1, j, j),
                            layout, priority_0, queue_jk2 );
                    }
                }
            }

            #pragma omp task depend(inout:column[k])
            {
                auto panel = A.sub( k, A_nt-1, k, k );

                // Erase remote tiles on all devices including host
                panel.releaseRemoteWorkspace();

                // Update the origin tiles before their
                // workspace copies on devices are erased.
                panel.tileUpdateAllOrigin();

                // Erase local workspace on devices.
                panel.releaseLocalWorkspace();
            }
            kk += A.tileNb( k );
        }
    }
    A.tileUpdateAllOrigin();

    if (hold_local_workspace == false) {
        A.releaseWorkspace();
    }
    if (target == Target::Devices) {
        for (int64_t dev = 0; dev < A.num_devices(); ++dev) {
            blas::Queue* queue = A.comm_queue(dev);
            blas::device_free( device_info_array[dev], *queue );
        }
    }

    internal::reduce_info( &info, A.mpiComm() );
    return info;
}

} // namespace impl

//------------------------------------------------------------------------------
/// Distributed parallel Cholesky factorization.
///
/// Performs the Cholesky factorization of a Hermitian positive definite
/// matrix $A$.
///
/// The factorization has the form
/// \[
///     A = L L^H,
/// \]
/// if $A$ is stored lower, where $L$ is a lower triangular matrix, or
/// \[
///     A = U^H U,
/// \]
/// if $A$ is stored upper, where $U$ is an upper triangular matrix.
///
/// Complexity (in real): $\approx \frac{1}{3} n^{3}$ flops.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the n-by-n Hermitian positive definite matrix $A$.
///     On exit, if return value = 0, the factor $U$ or $L$ from the Cholesky
///     factorization $A = U^H U$ or $A = L L^H$.
///     If scalar_t is real, $A$ can be a SymmetricMatrix object.
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
/// @return 0: successful exit
/// @return i > 0: the leading minor of order $i$ of $A$ is not
///         positive definite, so the factorization could not
///         be completed.
///
/// @ingroup posv_computational
///
template <typename scalar_t>
int64_t potrf(
    HermitianMatrix<scalar_t>& A,
    Options const& opts)
{
    using internal::TargetType;

    Target target = get_option<Option::Target>( opts, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostNest:
        case Target::HostBatch:
        case Target::HostTask:
            return impl::potrf( TargetType<Target::HostTask>(), A, opts );

        case Target::Devices:
            return impl::potrf( TargetType<Target::Devices>(), A, opts );
    }
    return -2;  // shouldn't happen
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
int64_t potrf<float>(
    HermitianMatrix<float>& A,
    Options const& opts);

template
int64_t potrf<double>(
    HermitianMatrix<double>& A,
    Options const& opts);

template
int64_t potrf< std::complex<float> >(
    HermitianMatrix< std::complex<float> >& A,
    Options const& opts);

template
int64_t potrf< std::complex<double> >(
    HermitianMatrix< std::complex<double> >& A,
    Options const& opts);

} // namespace slate
