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
    using blas::max;

    // Constants
    scalar_t zero = 0;
    scalar_t one  = 1;

    const auto mpi_real_type = mpi_type< blas::real_type<scalar_t> >::value;

    Target target = get_option( opts, Option::Target, Target::HostTask );

    int64_t m = A.m();
    int64_t n = A.n();
    int64_t min_mn = std::min(m, n);

    int nprow, npcol, myrow, mycol;

    // todo: still need to add if part of U or part of VT are needed.
    bool wantu  = (U.mt() > 0);
    bool wantvt = (VT.mt() > 0);

    // todo: Scale matrix to allowable range, if necessary.

    // 0. If m >> n, use QR factorization to reduce matrix A to a square matrix.
    // Theoretical thresholds based on flops:
    // m >=  5/3 n for no vectors,
    // m >= 16/9 n for QR iteration with vectors,
    // m >= 10/3 n for Divide & Conquer with vectors.
    // Different in practice because stages have different flop rates.
    double threshold = 5/3.;
    bool qr_path = m > threshold*n;
    bool lq_path = n > threshold*m;
    Matrix<scalar_t> Ahat, Uhat, VThat;
    TriangularFactors<scalar_t> TQ;
    if (qr_path) {
        geqrf( A, TQ, opts );

        // Upper triangular part of A (R).
        auto R_ = A.slice(0, n-1, 0, n-1);
        TriangularMatrix<scalar_t> R( Uplo::Upper, Diag::NonUnit, R_ );

        // Copy the upper triangular part to a new matrix Ahat.
        Ahat = R_.emptyLike();
        Ahat.insertLocalTiles(target);
        set( zero, Ahat, opts );  // todo: only lower

        TriangularMatrix<scalar_t> Ahat_tr( Uplo::Upper, Diag::NonUnit, Ahat );
        slate::copy( R, Ahat_tr, opts );

        if (wantu) {
            slate::set( zero, one, U, opts );
            Uhat = U.slice( 0, n-1, 0, n-1 );
        }
        if (wantvt) {
            slate::set( zero, one, VT, opts );
            VThat = VT;
        }
    }
    else if (lq_path) {
        gelqf( A, TQ, opts );
        swap(m, n);

        // Lower triangular part of A (R).
        auto R_ = A.slice(0, n-1, 0, n-1);
        TriangularMatrix<scalar_t> R( Uplo::Lower, Diag::NonUnit, R_ );

        // Copy the upper triangular part to a new matrix Ahat.
        Ahat = R_.emptyLike();
        Ahat.insertLocalTiles(target);
        set( zero, Ahat, opts );  // todo: only upper

        TriangularMatrix<scalar_t> Ahat_tr( Uplo::Lower, Diag::NonUnit, Ahat );
        slate::copy( R, Ahat_tr, opts );

        if (wantu) {
            slate::set( zero, one, U, opts );
            Uhat = U;
        }
        if (wantvt) {
            slate::set( zero, one, VT, opts );
            VThat = VT.slice( 0, n-1, 0, n-1 );
        }
    }
    else {
        Ahat = A;
        if (wantu) {
            slate::set( zero, one, U, opts );
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
    TriangularBandMatrix<scalar_t> Aband( Uplo::Upper, Diag::NonUnit,
                                         n, A.tileNb(0), A.tileNb(0),
                                         1, 1, A.mpiComm() );
    Aband.insertLocalTiles();

    // Slice Ahat here in case if A is rectangular but does not require qr_path.
    auto Ahat_ = Ahat.slice( 0, Ahat.n()-1, 0, Ahat.n()-1 );
    Aband.ge2tbGather(Ahat_);

    // Allocate U2 and VT2 matrices for tb2bd.
    int64_t nb = Ahat.tileNb(0);
    int64_t nt = Ahat.nt();

    int64_t vm = 2*nb;
    int64_t vn = nt*(nt + 1)/2*nb;

    Matrix<scalar_t> U2, VT2;
    VT2 = Matrix<scalar_t>( vm, vn, vm, nb, 1, 1, A.mpiComm() );
    U2 = Matrix<scalar_t>( vm, vn, vm, nb, 1, 1, A.mpiComm() );

    // Allocate E for super-diagonal.
    std::vector<real_t> E(n - 1);

    // 2. Reduction to bi-diagonal
    if (A.mpiRank() == 0) {
        VT2.insertLocalTiles();
        U2.insertLocalTiles();

        // Reduce band to bi-diagonal.
        tb2bd( Aband, U2, VT2, opts );

        // Copy diagonal and super-diagonal to vectors.
        internal::copytb2bd(Aband, Sigma, E);
    }

    int izero = 0;
    int64_t ncvt = 0, nru = 0, ldvt = 1, ldu = 1;

    std::vector<scalar_t> u1d(1);
    std::vector<scalar_t> vt1d(1);
    scalar_t dummy[1];

    int mpi_size;
    slate_mpi_call(
        MPI_Comm_size(A.mpiComm(), &mpi_size));

    // 3. Bi-diagonal SVD solver.
    if (wantu || wantvt) {
        // Bcast the Sigma and E vectors (diagonal and sup/super-diagonal).
        MPI_Bcast( &Sigma[0], min_mn,   mpi_real_type, 0, A.mpiComm() );
        MPI_Bcast( &E[0], min_mn-1, mpi_real_type, 0, A.mpiComm() );

        // Build the 1-dim distributed U and VT needed for bdsqr
        slate::Matrix<scalar_t> U1d_tall, V1d;
        if (wantu) {
            int64_t m_U = Uhat.m();
            int64_t mb = Uhat.tileMb(0);
            myrow = Uhat.mpiRank();
            nru  = numberLocalRowOrCol(m_U, mb, myrow, izero, mpi_size);
            ldu = max( 1, nru );
            u1d.resize(ldu*min_mn);
            U1d_tall = slate::Matrix<scalar_t>::fromScaLAPACK(
                    m_U, min_mn, &u1d[0], ldu, nb, mpi_size, 1, U.mpiComm() );
            set( zero, one, U1d_tall, opts );
        }
        if (wantvt) {
            mycol = VThat.mpiRank();
            ncvt = numberLocalRowOrCol(n, nb, mycol, izero, mpi_size);
            ldvt = max( 1, min_mn );
            vt1d.resize(ldvt*ncvt);
            V1d = slate::Matrix<scalar_t>::fromScaLAPACK(
                    min_mn, n, &vt1d[0], ldvt, nb, 1, mpi_size, VT.mpiComm() );
            set( zero, one, V1d, opts );
        }

        // QR iteration
        //bdsqr<scalar_t>(jobu, jobvt, Sigma, E, Uhat, VThat, opts);
        // Call the SVD
        lapack::bdsqr(Uplo::Upper, min_mn, ncvt, nru, 0,
                      &Sigma[0], &E[0], &vt1d[0], ldvt, &u1d[0], ldu, dummy, 1);

        // 4. Back transformation to compute U and VT of the initial matrix.
        // Back-transform: U = U1 * U2 * U.
        // U1 is the output of ge2tb and it is saved in A
        // U2 is the output of tb2bd
        // U initially has left singular vectors of the bidiagonal matrix
        // First: back transform the vectors for the second stage (reduction to bidiagonal)
        // and the bidiagonal SVD (bdsqr). U = U2 * U ===> U1d = U2 * U1d
        // Second: back transform the vectors from the previous step and the
        // first stage (reduction to band). U = U1 * U ===> U =  Ahat * U
        if (wantu) {
            // Create a 1-D matrix to redistribute U
            Matrix<scalar_t> U1d( Uhat.m(), Uhat.n(), Uhat.tileNb(0), 1, mpi_size, Uhat.mpiComm() );
            U1d.insertLocalTiles(target);

            // Redistribute U into 1-D U1d
            //U1d.redistribute(Uhat); // needed if call slate::bdsqr
            U1d.redistribute(U1d_tall);

            // First, U = U2 * U ===> U1d = U2 * U1d
            unmtr_hb2st( Side::Left, Op::NoTrans, U2, U1d, opts );

            // Redistribute U1d into U
            Uhat.redistribute(U1d);

            // todo: why unmbr_ge2tb need U not Uhat?
            // Second, U = U1 * U ===> U = Ahat * U
            unmbr_ge2tb( Side::Left, Op::NoTrans, Ahat, TU, Uhat, opts );
            if (qr_path) {
                // When initial QR was used.
                // U = Q*U;
                unmqr( Side::Left, slate::Op::NoTrans, A, TQ, U, opts );
            }
        }

        // Back-transform: VT = VT * VT2 * VT1.
        // VT1 is the output of ge2tb and it is saved in A
        // VT2 is the output of tb2bd
        // VT initially has right singular vectors of the bidiagonal matrix
        // V = VT'
        // First: back transform the vectors for the second stage (reduction to bidiagonal)
        // and the bidiagonal SVD (bdsqr). V  = VT2 * V ===> V1d = VT2 * V1d
        // Second: back transform the vectors from the previous step and the
        // first stage (reduction to band). VT = VT1 * VT ===> VT = Ahat * VT
        if (wantvt) {
            // The following V1d allocation needed if call slate::bdsqr
            //Matrix<scalar_t> V1d(VThat.m(), VThat.n(), VThat.tileNb(0), 1, mpi_size, VThat.mpiComm());
            //V1d.insertLocalTiles(target);

            // Generate a matrix with row-major grid to copy V to it
            // todo: will delete this when redistribute fixed to work on transposed matrices
            int64_t nb_V = VThat.tileNb( 0 );
            slate::GridOrder grid_order;
            VThat.gridinfo( &grid_order, &nprow, &npcol, &myrow, &mycol );
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

            int num_devices = VT.num_devices();
            std::function<int (std::tuple<int64_t, int64_t> ij)>
                tileDevice = [nprow, num_devices]( std::tuple<int64_t, int64_t> ij ) {
                    int64_t i = std::get<0>( ij );
                    return int( i/nprow )%num_devices;
                };
            slate::Matrix<scalar_t> V(
                    n, n, tileNb, tileNb, tileRank, tileDevice, VT.mpiComm() );

            // todo: will delete this when redistribute fixed to work on transposed matrices
            VThat.redistribute(V1d);
            auto R = conj_transpose(VThat);
            V.insertLocalTiles();
            slate::copy( R, V, opts );

            // Redistribute V into 1-D V1d
            V1d.redistribute(V);

            // First: V  = VT2 * V ===> V1d = VT2 * V1d
            unmtr_hb2st( Side::Left, Op::NoTrans, VT2, V1d, opts );

            // Redistribute V1d into V
            V.redistribute(V1d);

            // Second: VT = VT1 * VT ===> VT = Ahat * VT
            auto RT = conj_transpose(V);
            //slate::copy( RT, VThat, opts );
            slate::copy( RT, VThat );
            unmbr_ge2tb( Side::Right, Op::NoTrans, Ahat, TV, VThat, opts );
            if (lq_path) {
                // VT = VT*Q;
                unmlq( Side::Right, slate::Op::NoTrans, A, TQ, VT, opts );
            }
        }
    }
    else {
        if (A.mpiRank() == 0) {
            // QR iteration
            //bdsqr<scalar_t>(jobu, jobvt, Sigma, E, U, VT, opts);
            lapack::bdsqr(Uplo::Upper, min_mn, ncvt, nru, 0,
                          &Sigma[0], &E[0], &vt1d[0], ldvt, &u1d[0], ldu, dummy, 1);
        }
        // Bcast singular values.
        MPI_Bcast( &Sigma[0], min_mn, mpi_real_type, 0, A.mpiComm() );
    }

    // todo: If matrix was scaled, then rescale singular values appropriately.
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
