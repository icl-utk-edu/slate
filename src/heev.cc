// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/HermitianBandMatrix.hh"
#include "internal/internal.hh"

#include <cuda_profiler_api.h>
namespace slate {

//------------------------------------------------------------------------------
///
/// todo: document
///
/// @ingroup heev
///
template <typename scalar_t>
void heev(
    HermitianMatrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Matrix<scalar_t>& Z,
    Options const& opts)
{
    using real_t = blas::real_type<scalar_t>;
    using std::real;

    // Constants
    const auto mpi_real_type = mpi_type< blas::real_type<scalar_t> >::value;

    int64_t n = A.n();
    bool wantz = (Z.mt() > 0);

    // Get machine constants.
    const real_t safe_min = std::numeric_limits<real_t>::min();
    const real_t eps      = std::numeric_limits<real_t>::epsilon();
    const real_t sml_num  = safe_min / eps;
    const real_t big_num  = 1 / sml_num;
    const real_t sqrt_sml = sqrt( sml_num );
    const real_t sqrt_big = sqrt( big_num );

    // Scale matrix to allowable range, if necessary.
    real_t Anorm = norm( Norm::Max, A );
    real_t alpha = 1.0;
    if (std::isnan( Anorm ) || std::isinf( Anorm )) {
        // todo: return error value? throw?
        Lambda.assign( Lambda.size(), Anorm );
        return;
    }
    else if (Anorm > 0 && Anorm < sqrt_sml) {
        alpha = sqrt_sml;
    }
    else if (Anorm > sqrt_big) {
        alpha = sqrt_big;
    }

    if (alpha != 1.0) {
        // Scale by sqrt_sml/Anorm or sqrt_big/Anorm.
        scale( alpha, Anorm, A, opts );
    }

    // 1. Reduce to band form.
    TriangularFactors<scalar_t> T;
    he2hb(A, T, opts);

    // Copy band.
    // Currently, gathers band matrix to rank 0.
    int64_t nb = A.tileNb(0);
    HermitianBandMatrix<scalar_t> Aband(A.uplo(), n, nb, nb, 1, 1, A.mpiComm());
    Aband.insertLocalTiles();
    Aband.he2hbGather(A);

    // Currently, hb2st and sterf are run on a single node.
    Lambda.resize(n);
    std::vector<real_t> E(n - 1);
    Matrix<scalar_t> V;
    if (A.mpiRank() == 0) {
        // Matrix to store Householder vectors.
        // Could pack into a lower triangular matrix, but we store each
        // parallelogram in a 2nb-by-nb tile, with nt(nt + 1)/2 tiles.
        int64_t vm = 2*nb;
        int64_t nt = A.nt();
        int64_t vn = nt*(nt + 1)/2*nb;
        V = Matrix<scalar_t>(vm, vn, vm, nb, 1, 1, A.mpiComm());
        V.insertLocalTiles();

        // 2. Reduce band to real symmetric tri-diagonal.
        hb2st(Aband, V, opts);

        // Copy diagonal and super-diagonal to vectors.
        internal::copyhb2st( Aband, Lambda, E );
    }

    // 3. Tri-diagonal eigenvalue solver.
    if (wantz) {
        // Bcast the Lambda and E vectors (diagonal and sup/super-diagonal).
        MPI_Bcast( &Lambda[0], n,   mpi_real_type, 0, A.mpiComm() );
        MPI_Bcast( &E[0],      n-1, mpi_real_type, 0, A.mpiComm() );
        // QR iteration to get eigenvalues and eigenvectors of tridiagonal.
        steqr2( Job::Vec, Lambda, E, Z );
        // Back-transform: Z = Q1 * Q2 * Z.
        cudaProfilerStart();
        #pragma omp parallel
        #pragma omp master
        {
            omp_set_nested(1);
            #pragma omp task
            {
                unmtr_hb2st( Side::Left, Op::NoTrans, V, Z );
            }
        }
        cudaProfilerStop();
        unmtr_he2hb( Side::Left, Op::NoTrans, A, T, Z );
    }
    else {
        if (A.mpiRank() == 0) {
            // QR iteration to get eigenvalues.
            sterf<real_t>( Lambda, E, opts );
        }
        // Bcast eigenvalues.
        MPI_Bcast( &Lambda[0], n, mpi_real_type, 0, A.mpiComm() );
    }

    // If matrix was scaled, then rescale eigenvalues appropriately.
    if (alpha != 1.0) {
        // Scale by Anorm/sqrt_sml or Anorm/sqrt_big.
        // todo: deal with not all eigenvalues converging, cf. LAPACK.
        blas::scal( n, Anorm/alpha, &Lambda[0], 1 );
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void heev<float>(
    HermitianMatrix<float>& A,
    std::vector<float>& Lambda,
    Matrix<float>& Z,
    Options const& opts);

template
void heev<double>(
    HermitianMatrix<double>& A,
    std::vector<double>& Lambda,
    Matrix<double>& Z,
    Options const& opts);

template
void heev< std::complex<float> >(
    HermitianMatrix< std::complex<float> >& A,
    std::vector<float>& Lambda,
    Matrix< std::complex<float> >& Z,
    Options const& opts);

template
void heev< std::complex<double> >(
    HermitianMatrix< std::complex<double> >& A,
    std::vector<double>& Lambda,
    Matrix< std::complex<double> >& Z,
    Options const& opts);

} // namespace slate
