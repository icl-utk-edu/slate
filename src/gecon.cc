// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "internal/internal.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Distributed parallel estimate of the reciprocal of the condition number
/// of a triangular complex matrix A, in either the 1-norm or the infinity-norm.
/// Generic implementation for any target.
///
/// ColMajor layout is assumed
///
/// @ingroup gecon_specialization
///
/// The reciprocal of the condition number computed as:
/// \[
///     rcond = \frac{1}{\|\|A\-\| \times \-\|A^{-1}\|\|}
/// \]
/// where $A$ is upper triangular computed from the QR factorization.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] in_norm
///     rcond to compute:
///     - Norm::One: 1-norm condition number
///     - Norm::Inf: infinity-norm condition number
///
/// @param[in] A
///     On entry, the n-by-n triangular matrix $A$.
///     it is the output of the QR factorization of a general matrix.
///
/// @param[in,out] rcond
///     The reciprocal of the condition number of the matrix A,
///     computed as stated above.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
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
void gecon(
           Norm in_norm,
           Matrix<scalar_t>& A,
           blas::real_type<scalar_t> *rcond,
           Options const& opts)
{
    using BcastList = typename Matrix<scalar_t>::BcastList;
    using device_info_t = lapack::device_info_int;
    using blas::real;
    using real_t = blas::real_type<scalar_t>;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    int kase, kase1;
    if (in_norm == Norm::One) {
        kase1 = 1;
    }
    else if (in_norm == Norm::Inf) {
        kase1 = 2;
    }
    else {
        slate_error("invalid norm.");
    }

    int64_t m = A.m();
    int64_t nb = A.tileNb(0);
    int p, q;
    int myrow, mycol;
    GridOrder order;
    A.gridinfo(&order, &p, &q, &myrow, &mycol);


    // Compute matrix norm
    real_t Anorm = norm(in_norm, A);
    real_t Ainvnorm = 0.0;

    // Quick return
    *rcond = 0.;
    if (m == 0) {
        *rcond = 1.;
    }
    else if (Anorm == 0.) {
        return;
    }

    scalar_t alpha = 1.;

    std::vector<scalar_t> X(m);
    std::vector<scalar_t> V(m);
    std::vector<int64_t> isgn(m);
    std::vector<int64_t> isave(3);
    isave[0] = 0;
    isave[1] = 0;
    isave[2] = 0;

    auto L  = TriangularMatrix<scalar_t>(
        Uplo::Lower, slate::Diag::Unit, A );
    auto U  = TriangularMatrix<scalar_t>(
        Uplo::Upper, slate::Diag::NonUnit, A );
    auto B = slate::Matrix<scalar_t>::fromLAPACK(
            m, 1, &X[0], m, nb, 1, p, q, A.mpiComm());

    // initial and final value of kase is 0
    kase = 0;
    lacn2( m, X, V, isgn, &Ainvnorm, &kase, isave, opts);
    MPI_Bcast( &isave[0], 3, MPI_INT, B.tileRank(0, 0), MPI_COMM_WORLD );
    MPI_Bcast( &kase, 1, MPI_INT, B.tileRank(0, 0), MPI_COMM_WORLD );
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    while (kase != 0)
    {
        if (kase == kase1) {
            // Multiply by inv(L).
            slate::trsmA(Side::Left, alpha, L, B, opts);

            // Multiply by inv(U).
            slate::trsmA(Side::Left, alpha, U, B, opts);
        }
        else {
            // Multiply by inv(U**T).
            auto UT = conjTranspose( U );
            slate::trsmA(Side::Left, alpha, UT, B, opts);

            // Multiply by inv(L**T).
            auto LT = conjTranspose( L );
            slate::trsmA(Side::Left, alpha, LT, B, opts);
        }
        lacn2( m, X, V, isgn, &Ainvnorm, &kase, isave, opts);
        MPI_Bcast( &isave[0], 3, MPI_INT, B.tileRank(0, 0), MPI_COMM_WORLD );
        MPI_Bcast( &kase, 1, MPI_INT, B.tileRank(0, 0), MPI_COMM_WORLD );
    } // while (kase != 0)

    // Compute the estimate of the reciprocal condition number.
    if (Ainvnorm != 0.0) {
        *rcond = (1.0 / Ainvnorm) / Anorm;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gecon<float>(
    Norm in_norm,
    Matrix<float>& A,
    float *rcond,
    Options const& opts);

template
void gecon<double>(
    Norm in_norm,
    Matrix<double>& A,
    double *rcond,
    Options const& opts);

template
void gecon< std::complex<float> >(
    Norm in_norm,
    Matrix< std::complex<float> >& A,
    float *rcond,
    Options const& opts);

template
void gecon< std::complex<double> >(
    Norm in_norm,
    Matrix< std::complex<double> >& A,
    double *rcond,
    Options const& opts);

} // namespace slate
