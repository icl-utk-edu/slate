// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "aux/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::trtrm from internal::specialization::trtrm
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
///
/// todo: update docs: multiply not inverse.
///
/// Distributed parallel inverse of a triangular matrix.
/// Generic implementation for any target.
/// Panel and lookahead computed on host using Host OpenMP task.
/// @ingroup tr_specialization
///
template <Target target, typename scalar_t>
void trtrm(slate::internal::TargetType<target>,
           TriangularMatrix<scalar_t> A, int64_t lookahead)
{
    using real_t = blas::real_type<scalar_t>;
    using BcastList = typename Matrix<scalar_t>::BcastList;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // if upper, change to lower
    if (A.uplo() == Uplo::Upper) {
        A = conjTranspose(A);
    }
    int64_t A_nt = A.nt();

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > row_vector(A_nt);
    uint8_t* row = row_vector.data();

    if (target == Target::Devices) {
        A.allocateBatchArrays();
        A.reserveDeviceWorkspace();
    }

    #pragma omp parallel
    #pragma omp master
    {
        omp_set_nested(1);
        // diagonal block, L = L^H L
        #pragma omp task depend(inout:row[0])
        {
            // A(0, 0) = A(0, 0)^H * A(0, 0)
            internal::trtrm<Target::HostTask>(A.sub(0, 0));
        }

        for (int64_t k = 1; k < A_nt; ++k) {

            #pragma omp task depend(inout:row[0])
            {
                // send leading row up
                BcastList bcast_list_A;
                for (int64_t j = 0; j < k; ++j) {
                    // send A(k, j) up column A(j:k-1, j)
                    // and across row A(j, 0:j)
                    bcast_list_A.push_back({k, j, {A.sub(j, k-1, j, j),
                                                   A.sub(j, j, 0, j)}});
                }
                A.template listBcast(bcast_list_A, layout);
            }

            // update tailing submatrix
            #pragma omp task depend(inout:row[0])
            {
                // A(0:k-1, 0:k-1) += A(k, 0:k-1)^H * A(k, 0:k-1)
                auto H = HermitianMatrix<scalar_t>(A);
                auto H0 = H.sub(0, k-1);
                auto Ak = A.sub(k, k, 0, k-1);
                Ak = conjTranspose(Ak);

                internal::herk<target>(
                    real_t(1.0), std::move(Ak),
                    real_t(1.0), std::move(H0));
            }

            // multiply the leading row by the diagonal block
            #pragma omp task depend(inout:row[0])
            {
                // send A(k, k) across row A(k, 0:k-1)
                A.tileBcast(k, k, A.sub(k, k, 0, k-1), layout);

                // A(k, 0:k-1) = A(k, 0:k-1) * A(k, k)^H
                auto Akk = A.sub(k, k);
                Akk = conjTranspose(Akk);
                internal::trmm<Target::HostTask>(
                    Side::Left,
                    scalar_t(1.0), std::move(Akk),
                                   A.sub(k, k, 0, k-1));
            }

            // diagonal block, L = L^H L
            #pragma omp task depend(inout:row[0])
            {
                // A(k, k) = A(k, k)^H * A(k, k)
                internal::trtrm<Target::HostTask>(A.sub(k, k));
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
/// @ingroup tr_specialization
///
template <Target target, typename scalar_t>
void trtrm(TriangularMatrix<scalar_t>& A,
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

    internal::specialization::trtrm(internal::TargetType<target>(),
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
///
/// @ingroup tr_computational
///
template <typename scalar_t>
void trtrm(TriangularMatrix<scalar_t>& A,
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
            trtrm<Target::HostTask>(A, opts);
            break;
        case Target::HostNest:
            trtrm<Target::HostNest>(A, opts);
            break;
        case Target::HostBatch:
            trtrm<Target::HostBatch>(A, opts);
            break;
        case Target::Devices:
            trtrm<Target::Devices>(A, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void trtrm<float>(
    TriangularMatrix<float>& A,
    Options const& opts);

template
void trtrm<double>(
    TriangularMatrix<double>& A,
    Options const& opts);

template
void trtrm< std::complex<float> >(
    TriangularMatrix< std::complex<float> >& A,
    Options const& opts);

template
void trtrm< std::complex<double> >(
    TriangularMatrix< std::complex<double> >& A,
    Options const& opts);

} // namespace slate
