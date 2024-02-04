// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "work/work.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::trsmA_addmod from internal::specialization::trsmA_addmod
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// @internal
/// Distributed parallel triangular matrix solve.
/// Generic implementation for any target.
/// @ingroup trsm_impl
///
template <Target target, typename scalar_t>
void trsmA_addmod(slate::internal::TargetType<target>,
                  Side side, Uplo uplo,
                  scalar_t alpha, AddModFactors<scalar_t>& W,
                                         Matrix<scalar_t>& B,
                  int64_t lookahead)
{
    if (target == Target::Devices) {
        const int64_t batch_size_zero = 0;
        const int64_t num_arrays_two = 2; // Number of kernels without lookahead
        // Allocate batch arrays = number of kernels without
        // lookahead + lookahead
        // number of kernels without lookahead = 2
        // (internal::gemm & internal::trsm)
        // TODO
        // whereas internal::gemm with lookahead will be executed as many as
        // lookaheads, thus
        // internal::gemm with lookahead needs batch arrays equal to the
        // number of lookaheads
        // and the batch_arrays_index starts from
        // the number of kernels without lookahead, and then incremented by 1
        // for every execution for the internal::gemm with lookahead
        B.allocateBatchArrays(batch_size_zero, num_arrays_two);
        B.reserveDeviceWorkspace();
    }

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector<uint8_t> row_vector(W.A.nt());
    uint8_t* row = row_vector.data();

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
        #pragma omp task
        {
            work::trsmA_addmod<target, scalar_t>(side, uplo, alpha, W, B, row, lookahead);
            B.tileUpdateAllOrigin();
        }
    }
    B.releaseWorkspace();
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup trsm_impl
///
template <Target target, typename scalar_t>
void trsmA_addmod(blas::Side side, blas::Uplo uplo,
                  scalar_t alpha, AddModFactors<scalar_t>& W,
                                         Matrix<scalar_t>& B,
                  Options const& opts)
{
    int64_t lookahead = get_option<int64_t>(opts, Option::Lookahead, 1);

    internal::specialization::trsmA_addmod(internal::TargetType<target>(),
                                    side, uplo,
                                    alpha, W,
                                           B,
                                    lookahead);
}

// TODO docs
template <typename scalar_t>
void trsmA_addmod(blas::Side side, blas::Uplo uplo,
                  scalar_t alpha, AddModFactors<scalar_t>& W,
                                         Matrix<scalar_t>& B,
                  Options const& opts)
{
    Target target = Target::HostTask; //get_option<Target>(opts, Option::Target, Target::HostTask);

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            trsmA_addmod<Target::HostTask>(side, uplo, alpha, W, B, opts);
            break;
        case Target::HostNest:
            trsmA_addmod<Target::HostNest>(side, uplo, alpha, W, B, opts);
            break;
        case Target::HostBatch:
            trsmA_addmod<Target::HostBatch>(side, uplo, alpha, W, B, opts);
            break;
        case Target::Devices:
            trsmA_addmod<Target::Devices>(side, uplo, alpha, W, B, opts);
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void trsmA_addmod<float>(
    blas::Side side, blas::Uplo uplo,
    float alpha, AddModFactors<float>& W,
                          Matrix<float>& B,
    Options const& opts);

template
void trsmA_addmod<double>(
    blas::Side side, blas::Uplo uplo,
    double alpha, AddModFactors<double>& W,
                           Matrix<double>& B,
    Options const& opts);

template
void trsmA_addmod< std::complex<float> >(
    blas::Side side, blas::Uplo uplo,
    std::complex<float> alpha, AddModFactors< std::complex<float> >& W,
                                        Matrix< std::complex<float> >& B,
    Options const& opts);

template
void trsmA_addmod< std::complex<double> >(
    blas::Side side, blas::Uplo uplo,
    std::complex<double> alpha, AddModFactors< std::complex<double> >& W,
                                         Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
