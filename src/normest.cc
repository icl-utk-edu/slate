// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "internal/internal.hh"
#include "internal/internal_util.hh"
#include "slate/internal/mpi.hh"

#include <list>
#include <tuple>

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::norm from internal::specialization::norm
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// @internal
/// Distributed parallel general matrix norm.
/// Generic implementation for any target.
/// @ingroup norm_specialization
///
template <Target target, typename matrix_type>
blas::real_type<typename matrix_type::value_type>
normest(slate::internal::TargetType<target>,
     Norm in_norm, matrix_type A,
     Options const& opts)
{
    using scalar_t = typename matrix_type::value_type;
    using real_t = blas::real_type<scalar_t>;

    // Undo any transpose, which switches one <=> inf norms.
    if (A.op() == Op::ConjTrans)
        A = conj_transpose(A);
    else if (A.op() == Op::Trans)
        A = transpose(A);

    //---------
    // Two norm estimation
    // First: let's compute the x vector such that
    // x_j = sum_i abs( A_{i,j} ), x here is global_sums.data
    // todo: should we add this to norm?
    //if (in_norm == Norm::One_est) {

        using blas::min;

        int p, q;
        int myrow, mycol;
        GridOrder order;
        A.gridinfo(&order, &p, &q, &myrow, &mycol);

        int64_t n = A.n();
        int64_t m = A.m();
        int64_t nb = A.tileNb(0);

        scalar_t one = 1.;
        scalar_t zero  = 0.;
        real_t alpha;

        int64_t cnt = 0;
        int64_t maxiter = min( 100, n );

        real_t e  = 0.;
        real_t e0  = 0.;
        real_t normX = 0;
        real_t normAX = 0;
        real_t tol = 1.e-1;

        std::vector<real_t> local_sums(n);

        // todo: do we still need reserveDevice here?
        if (target == Target::Devices)
            A.reserveDeviceWorkspace();

        std::vector<scalar_t> global_sums(n);
        std::vector<scalar_t> W1(m);
        std::vector<scalar_t> W2(n);

        auto XL = slate::Matrix<scalar_t>::fromLAPACK(
            n, 1, &global_sums[0], n, nb, 1, p, q, A.mpiComm());
        XL.insertLocalTiles();
        auto AX = slate::Matrix<scalar_t>::fromLAPACK(
            m, 1, &W1[0], m, nb, 1, p, q, A.mpiComm());
        AX.insertLocalTiles();
        auto X = slate::Matrix<scalar_t>::fromLAPACK(
            n, 1, &W2[0], n, nb, 1, p, q, A.mpiComm());
        X.insertLocalTiles();


        #pragma omp parallel
        #pragma omp master
        {
            internal::norm<target>(slate::Norm::One, NormScope::Matrix, std::move(A), local_sums.data());
        }

        #pragma omp critical(slate_mpi)
        {
            trace::Block trace_block("MPI_Allreduce");
            slate_mpi_call(
                MPI_Allreduce(local_sums.data(), global_sums.data(),
                              n, mpi_type<real_t>::value,
                              MPI_SUM, A.mpiComm()));
        }

        // global_sums.data() will have the sum of each column.
        // First step is done

        // Second: compute the ||x||_Fro
        //e = lapack::lange(
        //    Norm::Fro, 1, n,
        //    global_sums.data(), 1);
        e  = slate::norm(Norm::Fro, XL, opts);
        if (e == 0.) {
            return 0.;
        }
        else {
            normX = e;
        }

        // Third: start the while-loop X = X / ||X||
        while ((cnt < maxiter) && (fabs((e - e0)) > (tol * (e)))) {
            e0 = e;

            // Scale X = X / ||X||
            alpha = 1.0/normX;
            add(scalar_t(alpha), XL, zero, X, opts);

            // Compute Ax = A * sx
            gemm(one, A, X, zero, AX, opts);

            // todo: still need to add the following
            //if nnz(Sx) == 0
            //    Sx = rand(size(Sx),class(Sx));
            //end

            // Compute x = A' * A * x = A' * Ax
            auto AT = conjTranspose(A);
            // todo: why this set is needed when using multiple mpi rank
            // todo: send to Sebastien
            set(zero, zero, XL);
            gemm(one, AT, AX, zero, XL, opts);
            //gemmC(one, AT, AX, zero, XL, opts);

            // Compute ||X||, ||AX||
            normX  = norm(Norm::Fro, XL, opts);
            normAX = norm(Norm::Fro, AX, opts);

            // Uodate e
            e = normX / normAX;
            cnt++;
        }

        A.clearWorkspace();

        return e;

    //}
    //else {
    //    slate_error("invalid norm.");
    //}
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup norm_specialization
///
template <Target target, typename matrix_type>
blas::real_type<typename matrix_type::value_type>
normest(Norm norm, matrix_type& A,
     const std::map<Option, Value>& opts)
{
    return internal::specialization::normest(internal::TargetType<target>(),
                                          norm, A, opts);
}

//------------------------------------------------------------------------------
/// Distributed parallel general matrix norm.
///
//------------------------------------------------------------------------------
/// @tparam matrix_type
///     Any SLATE matrix type: Matrix, SymmetricMatrix, HermitianMatrix,
///     TriangularMatrix, etc.
//------------------------------------------------------------------------------
/// @param[in] in_norm
///     Norm to compute:
///     - Norm::Second: second norm estimation of the matrix$
///
/// @param[in] A
///     The matrix A.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// @ingroup norm
///
template <typename matrix_type>
blas::real_type<typename matrix_type::value_type>
normest(Norm in_norm, matrix_type& A,
     const std::map<Option, Value>& opts)
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            return normest<Target::HostTask>(in_norm, A, opts);
            break;
        case Target::HostBatch:
        case Target::HostNest:
        case Target::Devices:
            break;
    }
    throw std::exception();  // todo: invalid target
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
float normest(
    Norm in_norm, Matrix<float>& A,
    Options const& opts);

template
double normest(
    Norm in_norm, Matrix<double>& A,
    Options const& opts);

template
float normest(
    Norm in_norm, Matrix< std::complex<float> >& A,
    Options const& opts);

template
double normest(
    Norm in_norm, Matrix< std::complex<double> >& A,
    Options const& opts);

} // namespace slate
