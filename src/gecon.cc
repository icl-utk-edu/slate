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
           blas::real_type<scalar_t> *Anorm,
           blas::real_type<scalar_t> *rcond,
           Options const& opts)
{
    using blas::real;
    using real_t = blas::real_type<scalar_t>;

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
    int myrow, mycol, mpi_size;
    int izero = 0;

    // Quick return
    *rcond = 0.;
    if (m == 0) {
        *rcond = 1.;
    }
    else if (*Anorm == 0.) {
        return;
    }

    scalar_t alpha = 1.;

    // todo: needed to create X fromScaLAPACK
    GridOrder grid_order;
    A.gridinfo(&grid_order, &p, &q, &myrow, &mycol);
    slate_mpi_call(
            MPI_Comm_size(A.mpiComm(), &mpi_size));
    //myrow = A.mpiRank();
    int64_t mloc  = numberLocalRowOrCol(m, nb, myrow, izero, mpi_size);
    int64_t lldA  = blas::max(1, mloc);

    real_t Ainvnorm = 0.0;
    //std::vector<scalar_t> X_data(mloc);
    std::vector<scalar_t> X_data(m);
    std::vector<scalar_t> V_data(m);
    std::vector<int64_t> isgn_data(m);
    std::vector<int64_t> isave(3);
    isave[0] = 0;
    isave[1] = 0;
    isave[2] = 0;

    auto X = slate::Matrix<scalar_t>::fromLAPACK(
            m, 1, &X_data[0], m, nb, 1, p, q, A.mpiComm());

    // todo: create X fromScaLAPACK to avoid allocating the entire X on each rank
    // X will be of size mloc
    //auto X = slate::Matrix<scalar_t>::fromScaLAPACK(
    //        m, 1, &X_data[0], lldA, nb, 1, grid_order, p, q, A.mpiComm() );
    auto V = slate::Matrix<scalar_t>::fromLAPACK(
            m, 1, &V_data[0], m, nb, 1, p, q, A.mpiComm() );
    auto isgn = slate::Matrix<int64_t>::fromLAPACK(
            m, 1, &isgn_data[0], m, nb, 1, p, q, A.mpiComm() );

    auto L  = TriangularMatrix<scalar_t>(
        Uplo::Lower, slate::Diag::Unit, A );
    auto U  = TriangularMatrix<scalar_t>(
        Uplo::Upper, slate::Diag::NonUnit, A );

    // initial and final value of kase is 0
    kase = 0;
    lacn2( X, V, isgn, &Ainvnorm, &kase, isave, opts);

    MPI_Bcast( &isave[0], 3, MPI_INT, X.tileRank(0, 0), A.mpiComm() );
    MPI_Bcast( &kase, 1, MPI_INT, X.tileRank(0, 0), A.mpiComm() );

    while (kase != 0)
    {
        if (kase == kase1) {
            // Multiply by inv(L).
            slate::trsmB(Side::Left, alpha, L, X, opts);

            // Multiply by inv(U).
            slate::trsmB(Side::Left, alpha, U, X, opts);
        }
        else {
            // Multiply by inv(U**T).
            auto UT = conjTranspose( U );
            slate::trsmB(Side::Left, alpha, UT, X, opts);

            // Multiply by inv(L**T).
            auto LT = conjTranspose( L );
            slate::trsmB(Side::Left, alpha, LT, X, opts);
        }

        lacn2( X, V, isgn, &Ainvnorm, &kase, isave, opts);
        MPI_Bcast( &isave[0], 3, MPI_INT, X.tileRank(0, 0), A.mpiComm() );
        MPI_Bcast( &kase, 1, MPI_INT, X.tileRank(0, 0), A.mpiComm() );
    } // while (kase != 0)

    // Compute the estimate of the reciprocal condition number.
    if (Ainvnorm != 0.0) {
        *rcond = (1.0 / Ainvnorm) / *Anorm;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gecon<float>(
    Norm in_norm,
    Matrix<float>& A,
    float *Anorm,
    float *rcond,
    Options const& opts);

template
void gecon<double>(
    Norm in_norm,
    Matrix<double>& A,
    double *Anorm,
    double *rcond,
    Options const& opts);

template
void gecon< std::complex<float> >(
    Norm in_norm,
    Matrix< std::complex<float> >& A,
    float *Anorm,
    float *rcond,
    Options const& opts);

template
void gecon< std::complex<double> >(
    Norm in_norm,
    Matrix< std::complex<double> >& A,
    double *Anorm,
    double *rcond,
    Options const& opts);

} // namespace slate
