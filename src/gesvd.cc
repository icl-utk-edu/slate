// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularBandMatrix.hh"
#include "internal/internal.hh"

namespace slate {

//------------------------------------------------------------------------------
///
/// todo: document
///
/// @ingroup svd
///
/// Note A is passed by value, so we can transpose if needed
/// without affecting caller.
///
template <typename scalar_t>
void gesvd(
    Matrix<scalar_t> A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    Matrix<scalar_t>& U,
    Matrix<scalar_t>& VT,
    Options const& opts)
{
    using real_t = blas::real_type<scalar_t>;
    using std::swap;

    // Constants
    scalar_t zero = 0;
    scalar_t one  = 1;

    const auto mpi_real_type = mpi_type< blas::real_type<scalar_t> >::value;

    Target target = get_option( opts, Option::Target, Target::HostTask );

    int64_t m = A.m();
    int64_t n = A.n();

    bool wantu  = (U.mt() > 0);
    bool wantvt = (VT.mt() > 0);

    bool flip = m < n; // Flip for fat matrix.
    if (flip) {
        slate_not_implemented("m < n not yet supported");
        swap(m, n);
        A = conj_transpose( A );
    }

    Job jobu  = Job::NoVec;
    Job jobvt = Job::NoVec;
    if (wantu) {
        jobu = Job::Vec;
    }

    if (wantvt) {
        jobvt = Job::Vec;
    }

    // Generate a matrix with row-major grid to copy V to it
    // todo: will delete this when redistribute fixed to work on transposed matrices
    int64_t nb_V = VT.tileNb( 0 );
    slate::GridOrder grid_order;
    int nprow, npcol, myrow, mycol;
    VT.gridinfo( &grid_order, &nprow, &npcol, &myrow, &mycol );
    std::function<int64_t (int64_t j)>
        tileNb = [n, nb_V] (int64_t j) {
            return (j + 1)*nb_V > n ? n%nb_V : nb_V;
        };

    std::function<int (std::tuple<int64_t, int64_t> ij)>
        tileRank = [nprow, npcol]( std::tuple<int64_t, int64_t> ij ) {
            int64_t i = std::get<0>( ij );
            int64_t j = std::get<1>( ij );
            return int( (i%nprow)*npcol + j%npcol );
        };

    int num_devices = blas::get_device_count();
    std::function<int (std::tuple<int64_t, int64_t> ij)>
        tileDevice = [nprow, num_devices]( std::tuple<int64_t, int64_t> ij ) {
            int64_t i = std::get<0>( ij );
            return int( i/nprow )%num_devices;
        };
    slate::Matrix<scalar_t> V(
           n, n, tileNb, tileNb, tileRank, tileDevice, VT.mpiComm() );

    // Scale matrix to allowable range, if necessary.
    // todo

    // 0. If m >> n, use QR factorization to reduce to square.
    // Theoretical thresholds based on flops:
    // m >=  5/3 n for no vectors,
    // m >= 16/9 n for QR iteration with vectors,
    // m >= 10/3 n for Divide & Conquer with vectors.
    // Different in practice because stages have different flop rates.
    double threshold = 5/3.;
    bool qr_path = m > threshold*n;
    Matrix<scalar_t> Ahat, Uhat, VThat;
    TriangularFactors<scalar_t> TQ;
    if (qr_path) {
        printf("\n QR path\n");
        geqrf(A, TQ, opts);

        auto R_ = A.slice(0, n-1, 0, n-1);
        TriangularMatrix<scalar_t> R(Uplo::Upper, Diag::NonUnit, R_);

        Ahat = R_.emptyLike();
        Ahat.insertLocalTiles(target);
        set(zero, Ahat);  // todo: only lower

        TriangularMatrix<scalar_t> Ahat_tr(Uplo::Upper, Diag::NonUnit, Ahat);
        copy(R, Ahat_tr);
        if (wantu) {
            slate::set( zero, one, U );
            Uhat = U.slice( 0, U.n()-1, 0, U.n()-1 );
        }
        if (wantvt) {
            slate::set( zero, one, VT );
            VThat = VT;
        }
    }
    else {
        Ahat = A;
        if (wantu) {
            slate::set( zero, one, U );
            Uhat = U;
        }
        if (wantvt) {
            slate::set( zero, one, VT );
            VThat = VT;
        }
    }

    // 1. Reduce to band form.
    TriangularFactors<scalar_t> TU, TV;
    ge2tb(Ahat, TU, TV, opts);

    // Currently, tb2bd and bdsqr run on a single node, gathers band matrix to rank 0.
    TriangularBandMatrix<scalar_t> Aband(Uplo::Upper, Diag::NonUnit,
                                         n, A.tileNb(0), A.tileNb(0),
                                         1, 1, A.mpiComm());
    Aband.insertLocalTiles();
    auto Ahat_ = Ahat.slice( 0, Ahat.n()-1, 0, Ahat.n()-1 );
    //auto Ahat_ = Ahat.sub( 0, Ahat.nt()-1, 0, Ahat.nt()-1 );
    Aband.ge2tbGather(Ahat_);

    int64_t nb = Ahat.tileNb(0);
    int64_t mt = Ahat.mt();
    int64_t nt = Ahat.nt();

    int64_t vm = 2*nb;
    int64_t vn = nt*(nt + 1)/2*nb;

    int64_t un = mt*(mt + 1)/2*nb;
    int64_t um = 2*nb;

    Matrix<scalar_t> U2, VT2;
    VT2 = Matrix<scalar_t>(vm, vn, vm, nb, 1, 1, A.mpiComm());
    U2 = Matrix<scalar_t>(um, un, um, nb, 1, 1, A.mpiComm());

    Sigma.resize(n);
    std::vector<real_t> E(n - 1);

    if (A.mpiRank() == 0) {
        VT2.insertLocalTiles();
        U2.insertLocalTiles();

        // 2. Reduce band to bi-diagonal.
        tb2bd(Aband, U2, VT2, opts);

        // Copy diagonal and super-diagonal to vectors.
        internal::copytb2bd(Aband, Sigma, E);
    }

    // 3. Bi-diagonal SVD solver.
    if (wantu || wantvt) {
        // Bcast the Sigma and E vectors (diagonal and sup/super-diagonal).
        MPI_Bcast( &Sigma[0], n,   mpi_real_type, 0, A.mpiComm() );
        MPI_Bcast( &E[0], n-1, mpi_real_type, 0, A.mpiComm() );

        // QR iteration
        bdsqr<scalar_t>(jobu, jobvt, Sigma, E, Uhat, VThat, opts);

        int mpi_size;
        // Find the total number of processors.
        slate_mpi_call(
            MPI_Comm_size(A.mpiComm(), &mpi_size));

        // Back-transform: U = U1 * U2 * U.
        // U1 is the output of ge2tb and it is saved in A
        // U2 is the output of tb2bd
        // U initially has left singular vectors of the bidiagonal matrix
        // First, U = U2 * U
        // Second, U = U1 * U
        if (wantu) {
            // Create a 1-D matrix to redistribute U
            Matrix<scalar_t> U1d(Uhat.m(), Uhat.n(), Uhat.tileNb(0), 1, mpi_size, Uhat.mpiComm());
            U1d.insertLocalTiles(target);

            // Redistribute U into 1-D U1d
            U1d.redistribute(Uhat);
            // First, back transform the vectors for the second stage (reduction to bidiagonal)
            // and the bidiagonal SVD (bdsqr). U1d = U2 * U1d
            unmtr_hb2st( Side::Left, Op::NoTrans, U2, U1d, opts );

            // Redistribute U1d into U
            Uhat.redistribute(U1d);
            // Second, back transform the vectors from the previous step and the
            // first stage (reduction to band). U = U1 * U = Ahat * U
            unmbr_ge2tb( Side::Left, Op::NoTrans, Ahat, TU, Uhat, opts );
        }

        // Back-transform: VT = VT * VT2 * VT1.
        // VT1 is the output of ge2tb and it is saved in A
        // VT2 is the output of tb2bd
        // VT initially has right singular vectors of the bidiagonal matrix
        // V = VT'
        // First, V  = VT2 * V
        // Second, V = VT1 * VT
        if (wantvt) {
            Matrix<scalar_t> V1d(VThat.m(), VThat.n(), VThat.tileNb(0), 1, mpi_size, VThat.mpiComm());
            V1d.insertLocalTiles(target);

            auto R = transpose(VThat);
            V.insertLocalTiles();
            copy(R, V);
            // Redistribute V into 1-D V1d
            V1d.redistribute(V);
            // First, back transform the vectors for the second stage (reduction to bidiagonal)
            // and the bidiagonal SVD (bdsqr). V1d = VT2 * V1d
            unmtr_hb2st( Side::Left, Op::NoTrans, VT2, V1d, opts );

            // Redistribute V1d into V
            V.redistribute(V1d);
            auto RT = transpose(V);
            copy(RT, VThat);
            // Second, back transform the vectors from the previous step and the
            // first stage (reduction to band). VT = VT1 * VT = Ahat * VT
            unmbr_ge2tb( Side::Right, Op::NoTrans, Ahat, TV, VThat, opts );
        }
    }
    else {
        if (A.mpiRank() == 0) {
            // QR iteration
            bdsqr<scalar_t>(jobu, jobvt, Sigma, E, U, VT, opts);
        }
        // Bcast singular values.
        MPI_Bcast( &Sigma[0], n, mpi_real_type, 0, A.mpiComm() );
    }

    // If matrix was scaled, then rescale singular values appropriately.
    // todo

    // todo: bcast S.

    if (qr_path) {
        // todo: test it
        // When initial QR was used.
        // U = Q*U;
        unmqr(Side::Left, slate::Op::NoTrans, A, TQ, U, opts);
    }

    if (flip) {
        // todo: test it
        // todo: swap(U, V);
        // U = V
        // VT = U'
        //auto UT = conj_transpose(U);
        //VT = UT;
        //U  = V;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gesvd<float>(
     Matrix<float> A,
     std::vector<float>& S,
     Matrix<float>& U,
     Matrix<float>& VT,
     Options const& opts);

template
void gesvd<double>(
     Matrix<double> A,
     std::vector<double>& S,
     Matrix<double>& U,
     Matrix<double>& VT,
     Options const& opts);

template
void gesvd< std::complex<float> >(
     Matrix< std::complex<float> > A,
     std::vector<float>& S,
     Matrix< std::complex<float> >& U,
     Matrix< std::complex<float> >& VT,
     Options const& opts);

template
void gesvd< std::complex<double> >(
     Matrix< std::complex<double> > A,
     std::vector<double>& S,
     Matrix< std::complex<double> >& U,
     Matrix< std::complex<double> >& VT,
     Options const& opts);

} // namespace slate
