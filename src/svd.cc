// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/TriangularBandMatrix.hh"
#include "internal/internal.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Distributed parallel matrix singular value decomposition.
/// Computes all singular values and, optionally, singular vectors of a
/// matrix A. The matrix A is preliminary reduced to
/// bidiagonal form using a two-stage approach:
/// First stage: reduction to upper band bidiagonal form (see ge2tb);
/// Second stage: reduction from band to bidiagonal form (see tb2bd).
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] A
///     On entry, the m-by-n matrix $A$.
///     On exit, contents are destroyed.
///
/// @param[out] Sigma
///     The vector Sigma of length min( m, n ).
///     If successful, the singular values in ascending order.
///
/// @param[out] U
///     On entry, if U is empty, does not compute the left singular vectors.
///     Otherwise, the m-by-min_mn ("economy size") or m-by-m ("all vectors")
///     matrix $U$ to store the left singular vectors.
///     On exit, the left orthonormal singular vectors of the matrix A.
///
/// @param[out] VT
///     On entry, if VT is empty, does not compute the right singular vectors.
///     Otherwise, the min_mn-by-n ("economy size") or n-by-n ("all vectors")
///     matrix $VT$ to store the right singular vectors.
///     On exit, the right orthonormal singular vectors of the matrix A.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::InnerBlocking:
///       Inner blocking to use for panel. Default 16.
///     - Option::MaxPanelThreads:
///       Number of threads to use for panel. Default omp_get_max_threads()/2.
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// @ingroup svd
///
//------------------------------------------------------------------------------
/// Note A is passed by value, so we can transpose if needed
/// without affecting caller.
///
//------------------------------------------------------------------------------
//  High-level algorithm
//  Forms A = U Sigma VT = [U0] U1 U2 U3 Sigma VT3 VT2 VT1 [V0^H]
//  At most one of U0 or V0 is used, if m >> n or m << n, respectively.
//
//  min_mn = min( m, n )
//  if m >> n, mhat = min_mn = n, else mhat = m
//  if m << n, nhat = min_mn = m, else nhat = n
//
//  A    is m-by-n
//  Ahat is mhat-by-nhat
//
//  U    is m-by-min_mn (some vectors, economy size)  or  m-by-m (all vectors)
//  VT   is min_mn-by-n (some vectors, economy size)  or  n-by-n (all vectors)
//
//  U0   is order m, Householder repr is m-by-min_mn
//  VT0  is order n, Householder repr is min_mn-by-n
//
//  U1   is order mhat, Householder repr is mhat-by-min_mn
//  VT1  is order nhat, Householder repr is min_mn-by-nhat
//
//  U2, VT2, U3, VT3 are min_mn-by-min_mn
//
//  Step 0. Reduce A to square if very tall or very wide.
//  if (m >> n)
//      Factor A = QR with U0 = Q
//      Ahat = R (min_mn-by-n == n-by-n)
//  elif (m << n)
//      Factor A = LQ with V0 = Q
//      Ahat = L (m-by-min_mn == m-by-m)
//  else
//      Ahat = A (m-by-n)
//
//  Step 1. Reduce to band.
//  ge2tb( Ahat, TU1, TV1 )         // Ahat = U1 Aband VT1
//  copy band of Ahat => Aband (min_mn-by-min_mn)
//
//  Step 2. Reduce to bidiagonal.
//  tb2bd( Aband, U2, VT2 )         // Aband = U2 Abidiag VT2
//  copy diag, superdiag of Aband => Sigma, E.
//
//  [Alternative implementation:
//  For bdqsr (QR iteration), could generate U2 in U3 and VT2 in VT3,
//  and then bdsqr updates them. In that case, skip unmbr_tb2bd.
//  For bdsdc (divide and conquer) and other bidiagonal SVD, the
//  back-transform approach here is required.]
//
//  Step 3. Bidiagonal SVD.         // Abidiag = U3 Sigma VT3
//  U3 = Identity, VT3 = Identity
//  bdsvd( Sigma, E, U3, VT3 )
//
//  Backtransform vectors
//  if (want U vectors)
//      Step b2.
//      unmbr_tb2bd( U2, U3 )       // U3 = U2 * U3
//
//      U = [ U3 ] or [ U3  0 ] for all vectors
//          [ 0  ]    [ 0   I ]
//      Uhat is U( 0:mhat-1, 0:n-1 )
//
//      Step b1.
//      unmbr_ge2tb( Ahat, Uhat )   // Uhat = U1 * Uhat
//
//      Step b0.
//      if (m >> n)
//          unmqr( A, U )           // U = U0 * U
//
//  if (want V vectors)
//      Step b2.
//      unmbr_tb2bd( VT2, VT3 )     // VT3 = VT3 * VT2
//
//      VT = [ VT3 ] or [ VT3  0 ] for all vectors
//           [ 0   ]    [ 0    I ]
//      VThat is VT( 0:n-1, 0:nhat-1 )
//
//      Step b1.
//      unmbr_ge2tb( Ahat, VThat )  // VThat = VThat * VT1
//
//      Step b0.
//      if (m << n)
//          unmlq( A, VT )          // VT = VT * VT0
//
template <typename scalar_t>
void svd(
    Matrix<scalar_t> A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    Matrix<scalar_t>& U,
    Matrix<scalar_t>& VT,
    Options const& opts)
{
    Timer t_svd;
    timers.clear();

    using real_t = blas::real_type<scalar_t>;
    using blas::max;

    // Constants
    const scalar_t zero = 0;
    const scalar_t one  = 1;
    const real_t r_zero = 0.;
    const int64_t izero = 0;
    const int64_t ione  = 1;
    const int     root  = 0;

    const real_t eps      = std::numeric_limits<real_t>::epsilon();
    const real_t safe_min = std::numeric_limits<real_t>::min();
    // ScaLAPACK uses rmin   = sqrt( safe_min / eps ) = 1.3e-138; SLATE uses this.
    // LAPACK    uses smlnum = sqrt( safe_min ) / eps = 1.4e-147.
    const real_t sml_num  = sqrt( safe_min / eps );
    const real_t big_num  = 1 / sml_num;

    const auto mpi_real_type = mpi_type< blas::real_type<scalar_t> >::value;

    // Options
    Target target = get_option( opts, Option::Target, Target::HostTask );

    int64_t m = A.m();
    int64_t n = A.n();
    int64_t min_mn = std::min( m, n );
    int64_t nb = A.tileNb( 0 );

    bool wantu  = (U.mt() > 0);
    bool wantvt = (VT.mt() > 0);
    bool allvec = (m > n && U.n() == m) || (m < n && VT.m() == n);
    //printf( "wantu %d, wantvt %d, allvec %d ", wantu, wantvt, allvec );

    // Scale A if max element outside range (sml_num, big_num).
    real_t Anorm = norm( Norm::Max, A );
    real_t scl = 1.;
    if (std::isnan( Anorm ) || std::isinf( Anorm )) {
        // todo: return error value? throw?
        Sigma.assign( Sigma.size(), Anorm );
        return;
    }
    else if (Anorm > r_zero && Anorm < sml_num) {
        // A *= (sml_num / Anorm)
        scl = sml_num;
        scale( scl, Anorm, A, opts );
    }
    else if (Anorm > big_num) {
        // A *= (big_num / Anorm)
        scl = big_num;
        scale( scl, Anorm, A, opts );
    }

    // 0. If m >> n, use QR factorization to reduce matrix A to a square matrix.
    //    If n << m, use LQ factorization to reduce matrix A to a square matrix.
    // Theoretical thresholds based on flops:
    // m >=  5/3 n for no vectors,
    // m >= 16/9 n for QR iteration with vectors,
    // m >= 10/3 n for Divide & Conquer with vectors.
    // Different in practice because stages have different flop rates.
    double threshold = 5/3.;
    bool qr_path = m > threshold*n;
    //
    // todo: if A is a little wide, we either:
    // 1. shallow transpose A, then A is a little tall, but we need to debug geqrf
    // 2. deep transpose A, then geqrf will work fine, but need to also transpose
    // VT and U, and this need debugging.
    // Also update bdsqr call to be Lower instead of Upper.
    //
    //bool lq_path = n > threshold*m;
    bool lq_path = n > m;
    Matrix<scalar_t> Ahat;
    TriangularFactors<scalar_t> TQ;
    if (qr_path) {
        Timer t_geqrf;
        geqrf( A, TQ, opts );
        timers[ "svd::geqrf" ] = t_geqrf.stop();

        // R is upper triangular part of A.
        auto R_ = A.slice( 0, n-1, 0, n-1 );
        TriangularMatrix<scalar_t> R( Uplo::Upper, Diag::NonUnit, R_ );

        // Copy R to a new matrix Ahat.
        Ahat = R_.emptyLike();
        Ahat.insertLocalTiles( target );
        set( zero, Ahat, opts );  // todo: only lower

        TriangularMatrix<scalar_t> Ahat_tr( Uplo::Upper, Diag::NonUnit, Ahat );
        slate::copy( R, Ahat_tr, opts );
    }
    else if (lq_path) {
        Timer t_gelqf;
        gelqf( A, TQ, opts );
        timers[ "svd::gelqf" ] = t_gelqf.stop();

        // L is lower triangular part of A.
        auto L_ = A.slice( 0, m-1, 0, m-1 );
        TriangularMatrix<scalar_t> L( Uplo::Lower, Diag::NonUnit, L_ );

        // Copy L to a new matrix Ahat.
        Ahat = L_.emptyLike();
        Ahat.insertLocalTiles( target );
        set( zero, Ahat, opts );  // todo: only upper

        TriangularMatrix<scalar_t> Ahat_tr( Uplo::Lower, Diag::NonUnit, Ahat );
        slate::copy( L, Ahat_tr, opts );
    }
    else {
        Ahat = A;
    }

    // 1. Reduce to band form.
    TriangularFactors<scalar_t> TU1, TV1;
    Timer t_ge2tb;
    ge2tb( Ahat, TU1, TV1, opts );
    timers[ "svd::ge2tb" ] = t_ge2tb.stop();

    // Currently, tb2bd and bdsqr run on a single node,
    // gathers band matrix to rank 0.
    TriangularBandMatrix<scalar_t> Aband( Uplo::Upper, Diag::NonUnit,
                                          min_mn, nb, nb,
                                          1, 1, A.mpiComm() );
    Aband.insertLocalTiles();

    // Slice in case Ahat is rectangular.
    auto Ahat_11 = Ahat.slice( 0, min_mn-1, 0, min_mn-1 );
    Aband.ge2tbGather( Ahat_11 );

    // Allocate U2 and VT2 matrices for tb2bd.
    // These are (2nb)-by-vn with (2nb)-by-nb tile size.
    // vn has space for tiles to cover the lower or upper triangle.
    int64_t nt = Aband.nt();
    int64_t vm = 2*nb;
    int64_t vn = nt*(nt + 1)/2*nb;
    Matrix<scalar_t> VT2( vm, vn, vm, nb, 1, 1, A.mpiComm() );
    Matrix<scalar_t>  U2( vm, vn, vm, nb, 1, 1, A.mpiComm() );

    // Allocate E for super-diagonal.
    std::vector<real_t> E( min_mn - 1 );

    // 2. Reduce band to bi-diagonal.
    if (A.mpiRank() == root) {
        VT2.insertLocalTiles();
        U2.insertLocalTiles();

        Timer t_tb2bd;
        tb2bd( Aband, U2, VT2, opts );
        timers[ "svd::tb2bd" ] = t_tb2bd.stop();

        // Copy diagonal and super-diagonal to vectors.
        internal::copytb2bd( Aband, Sigma, E );

        Aband.releaseRemoteWorkspace();
    }

    scalar_t dummy[1];

    if (wantu || wantvt) {
        int mpi_size;
        slate_mpi_call(
            MPI_Comm_size( A.mpiComm(), &mpi_size ) );

        // Bcast the Sigma and E vectors (diagonal and sup/super-diagonal).
        slate_mpi_call(
            MPI_Bcast( &Sigma[0], min_mn,   mpi_real_type, root, A.mpiComm() ) );
        slate_mpi_call(
            MPI_Bcast( &E[0],     min_mn-1, mpi_real_type, root, A.mpiComm() ) );

        // Build the 1D distributed U and VT needed for bdsqr.
        // U3_1d_col  is mlocal_U-by-min_mn  on np-by-1 col grid (np = mpi_size).
        // VT3_1d_row is min_mn-by-nlocal_VT on 1-by-np row grid.
        int64_t nlocal_VT = 0, mlocal_U = 0, ldvt = 1, ldu = 1;
        std::vector<scalar_t> U3_1d_col_data( 1 );
        std::vector<scalar_t> VT3_1d_row_data( 1 );
        Matrix<scalar_t> U3_1d_col, VT3_1d_row;
        if (wantu) {
            int myrow = A.mpiRank();
            mlocal_U = num_local_rows_cols( min_mn, nb, myrow, izero, mpi_size );
            ldu = max( 1, mlocal_U );
            U3_1d_col_data.resize( ldu*min_mn );
            U3_1d_col = Matrix<scalar_t>::fromScaLAPACK(
                    min_mn, min_mn, &U3_1d_col_data[0], ldu, nb,
                    mpi_size, 1, A.mpiComm() );
            set( zero, one, U3_1d_col, opts ); // Identity
        }
        if (wantvt) {
            int mycol = A.mpiRank();
            nlocal_VT = num_local_rows_cols( min_mn, nb, mycol, izero, mpi_size );
            ldvt = max( 1, min_mn );
            VT3_1d_row_data.resize( ldvt*nlocal_VT );
            VT3_1d_row = Matrix<scalar_t>::fromScaLAPACK(
                    min_mn, min_mn, &VT3_1d_row_data[0], ldvt, nb,
                    1, mpi_size, A.mpiComm() );
            set( zero, one, VT3_1d_row, opts ); // Identity
        }

        // 3. Bi-diagonal SVD solver.
        // Call LAPACK bdsqr directly, since SLATE bdsqr hides some
        // redistribute calls that we want to see to optimize the code
        // and reuse memory.
        //bdsqr<scalar_t>( jobu, jobvt, Sigma, E, Uhat, VThat, opts );
        Timer t_bdsvd;
        lapack::bdsqr( Uplo::Upper, min_mn, nlocal_VT, mlocal_U, 0,
                       &Sigma[0], &E[0],
                       &VT3_1d_row_data[0], ldvt,
                       &U3_1d_col_data[0], ldu,
                       dummy, 1 );
        timers[ "svd::bdsvd" ] = t_bdsvd.stop();

        // Back-transform: U = U0 * U1 * U2 * U3.
        // U0 is the output of geqrf with repr in (A, TQ), if (qr_path).
        // U1 is the output of ge2tb with repr in (Ahat, TU1).
        // U2 is the output of tb2bd.
        // U3 is the output of bdsqr.
        if (wantu) {
            // Redistribute U3 from np-by-1 col grid to 1-by-np row grid.
            Matrix<scalar_t> U3_1d_row(
                min_mn, min_mn, nb, 1, mpi_size, A.mpiComm() );
            U3_1d_row.insertLocalTiles( target );
            Timer t_red;
            redistribute( U3_1d_col, U3_1d_row, opts );
            timers[ "svd::redistribute" ] = t_red.stop();

            // 2b. Backtransform tb2bd: U3 = U2 * U3.
            Timer t_unmbr_tb2bd_U;
            unmtr_hb2st( Side::Left, Op::NoTrans, U2, U3_1d_row, opts );
            timers[ "svd::unmbr_tb2bd_U" ] = t_unmbr_tb2bd_U.stop();

            // Redistribute U3 to top-left of U.
            t_red.start();
            auto U_11 = U.slice( 0, min_mn-1, 0, min_mn-1 );
            redistribute( U3_1d_row, U_11, opts );          // U_11 = U3
            timers[ "svd::redistribute" ] += t_red.stop();

            // Rest of U is identity.
            if (m > n) {
                auto U_21 = U.slice( n, m-1, 0, n-1 );
                slate::set( zero, U_21, opts );             // U_21 = 0
                if (allvec) {
                    auto U_12 = U.slice( 0, n-1, n, m-1 );
                    slate::set( zero, U_12, opts );         // U_12 = 0
                    auto U_22 = U.slice( n, m-1, n, m-1 );
                    slate::set( zero, one, U_22, opts );    // U_22 = Identity
                }
            }
            //print( "U", U );

            Matrix<scalar_t> Uhat;
            if (qr_path) {
                Uhat = U_11;
            }
            else {
                Uhat = U;
            }

            // 1b. Backtransform ge2tb: Uhat = U1 * Uhat,
            // with U1 repr in (Ahat, TU1).
            Timer t_unmbr_ge2tb_U;
            unmbr_ge2tb( Side::Left, Op::NoTrans, Ahat, TU1, Uhat, opts );
            timers[ "svd::unmbr_ge2tb_U" ] = t_unmbr_ge2tb_U.stop();

            // 0b. Backtransform geqrf: U = U0 * U with U0 repr in (A, TQ).
            Timer t_unmqr;
            if (qr_path) {
                unmqr( Side::Left, Op::NoTrans, A, TQ, U, opts );
            }
            timers[ "svd::unmqr" ] = t_unmqr.stop();
        }

        // Back-transform: VT = VT3 * VT2 * VT1 * VT0.
        // VT0 is the output of gelqf with repr in (A, TQ), if (lq_path).
        // VT1 is the output of ge2tb with repr in (Ahat, TV1).
        // VT2 is the output of tb2bd.
        // VT3 is the output of bdsqr.
        if (wantvt) {
            // Redistribute and conjugate-transpose VT3_1d_row to
            // V3_1d_row (no-trans), both on 1-by-np row grid.
            Matrix<scalar_t> V3_1d_row(
                min_mn, min_mn, nb, 1, mpi_size, A.mpiComm() );
            V3_1d_row.insertLocalTiles( target );

            Timer t_red;
            auto V3 = conj_transpose( VT3_1d_row );
            redistribute( V3, V3_1d_row, opts );
            timers[ "svd::redistribute" ] = t_red.stop();

            // 2b. Backtransform tb2bd: V = VT2 * V.
            Timer t_unmbr_tb2bd_V;
            unmtr_hb2st( Side::Left, Op::NoTrans, VT2, V3_1d_row, opts );
            timers[ "svd::unmbr_tb2bd_V" ] = t_unmbr_tb2bd_V.stop();

            // Redistribute and conjugate-transpose V3_1d_row to top-left of VT.
            t_red.start();
            auto VT_11 = VT.slice( 0, min_mn-1, 0, min_mn-1 );
            auto VT3 = conj_transpose( V3_1d_row );
            redistribute( VT3, VT_11, opts );               // VT_11 = VT3
            timers[ "svd::redistribute" ] += t_red.stop();

            // Rest of VT is identity.
            if (n > m) {
                auto VT_12 = VT.slice( 0, m-1, m, n-1 );
                slate::set( zero, VT_12, opts );            // VT_12 = 0
                if (allvec) {
                    auto VT_21 = VT.slice( m, n-1, 0, m-1 );
                    slate::set( zero, VT_21, opts );        // VT_21 = 0
                    auto VT_22 = VT.slice( m, n-1, m, n-1 );
                    slate::set( zero, one, VT_22, opts );   // VT_22 = Identity
                }
            }
            //print( "VT", VT );

            Matrix<scalar_t> VThat;
            if (lq_path) {
                VThat = VT_11;
            }
            else {
                VThat = VT;
            }

            // 1b. Backtransform ge2tb: VThat = VT1 * VThat,
            // with VT1 in (Ahat, TV1).
            Timer t_unmbr_ge2tb_V;
            unmbr_ge2tb( Side::Right, Op::NoTrans, Ahat, TV1, VThat, opts );
            timers[ "svd::unmbr_ge2tb_V" ] = t_unmbr_ge2tb_V.stop();

            // 0b. Backtransform gelqf: VT = VT * VT0 with VT0 repr in (A, TQ).
            Timer t_unmlq;
            if (lq_path) {
                unmlq( Side::Right, Op::NoTrans, A, TQ, VT, opts );
            }
            timers[ "svd::unmlq" ] = t_unmlq.stop();
        }
    }
    else {
        // Singular values only
        Timer t_bdsvd;
        if (A.mpiRank() == root) {
            // QR iteration
            //bdsqr<scalar_t>( jobu, jobvt, Sigma, E, U, VT, opts );
            lapack::bdsqr( Uplo::Upper, min_mn, 0, 0, 0,
                           &Sigma[0], &E[0],
                           dummy, 1,
                           dummy, 1,
                           dummy, 1 );
        }

        // Bcast singular values.
        MPI_Bcast( &Sigma[0], min_mn, mpi_real_type, root, A.mpiComm() );
        timers[ "svd::bdsvd" ] = t_bdsvd.stop();
    }

    // If matrix was scaled, then rescale singular values appropriately.
    // All ranks compute redundantly.
    if (scl != 1.) {
        // Sigma *= (Anorm / scl)
        // Note order:
        // LAPACK lascl has denominator, numerator;
        // SLATE  scale has numerator, denominator.
        lapack::lascl( lapack::MatrixType::General, izero, izero,
                       scl, Anorm,
                       min_mn, ione,
                       &Sigma[0], ione );
    }

    timers[ "svd" ] = t_svd.stop();
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void svd<float>(
     Matrix<float> A,
     std::vector<float>& S,
     Matrix<float>& U,
     Matrix<float>& VT,
     Options const& opts);

template
void svd<double>(
     Matrix<double> A,
     std::vector<double>& S,
     Matrix<double>& U,
     Matrix<double>& VT,
     Options const& opts);

template
void svd< std::complex<float> >(
     Matrix< std::complex<float> > A,
     std::vector<float>& S,
     Matrix< std::complex<float> >& U,
     Matrix< std::complex<float> >& VT,
     Options const& opts);

template
void svd< std::complex<double> >(
     Matrix< std::complex<double> > A,
     std::vector<double>& S,
     Matrix< std::complex<double> >& U,
     Matrix< std::complex<double> >& VT,
     Options const& opts);

} // namespace slate
