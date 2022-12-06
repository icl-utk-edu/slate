// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "auxiliary/Debug.hh"
#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

#include "slate/Tile_blas.hh"


namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// Distributed parallel Hermitian indefinite $LTL^T$ factorization.
/// Generic implementation for any target.
/// <b>GPU version not yet implemented.</b>
/// @ingroup hesv_impl
///
template <Target target, typename scalar_t>
void hetrf(
    HermitianMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& H,
    Options const& opts)
{
    using blas::conj;
    using BcastList  = typename Matrix<scalar_t>::BcastList;
    using ReduceList = typename Matrix<scalar_t>::ReduceList;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // Options
    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );
    int64_t ib = get_option<int64_t>( opts, Option::InnerBlocking, 16 );
    int64_t max_panel_threads = std::max( omp_get_max_threads()/2, 1 );
    max_panel_threads = get_option<int64_t>( opts, Option::MaxPanelThreads,
                                             max_panel_threads );

    int64_t A_mt = A.mt();

    std::vector< uint8_t > column_vectorL(A_mt);
    std::vector< uint8_t > column_vectorT(A_mt);
    uint8_t* columnL = column_vectorL.data();
    uint8_t* columnT = column_vectorT.data();

    std::vector< uint8_t > column_vectorH1(A_mt);
    std::vector< uint8_t > column_vectorH2(A_mt);
    uint8_t* columnH1 = column_vectorH1.data();
    uint8_t* columnH2 = column_vectorH2.data();

    //std::vector< uint8_t > Ind1(1);
    //std::vector< uint8_t > Ind2(A_mt);
    //uint8_t* ind1 = Ind1.data();
    //uint8_t* ind2 = Ind2.data();

    const scalar_t zero = 0.0;
    const scalar_t one  = 1.0;
    int64_t ione  = 1;
    int64_t izero = 0;
    int priority_one = 1;
    assert(A.uplo() == Uplo::Lower); // upper not implemented, yet

    pivots.resize(A_mt);

    int rank;
    MPI_Comm_rank(A.mpiComm(), &rank);
    #pragma omp parallel
    #pragma omp master
    for (int64_t k = 0; k < A_mt; ++k) {
        //printf( "\n == k = %ld on rank-%d ==\n",k,rank ); fflush(stdout);
        int tag  = 1+k;
        int tag1 = 1+k+A_mt*1;
        int tag2 = 1+k+A_mt*2;
        int tag3 = 1+k+A_mt*3;
        int tag4 = 1+k+A_mt*4;

        // compute H(k, i) := L(k, [i-1, i, i+1]) * T([i-1, i, i+1], i), for i = 1, .., k-1
        // i.e., H(k, 0:k-1) := L(k, 1:k) * T(0:k-1, 1:k)
        // H(1, k) not needed, and thus not computed
        if (k > 1) {
            #pragma omp task depend(in:columnT[k]) \
                             depend(in:columnL[k-1]) \
                             depend(out:columnH1[k]) \
                             priority(1)
            {
                //printf( " >> compute H(%ld, %d:%ld) on rank-%d <<\n", k, 0,k-1, rank); fflush(stdout);
                // going by row, H(k, i) = L(k, :) * T(:, i) for i=0,..,k+1
                // send L(k, j) that are needed to compute H(:, k)
                for (int64_t j=0; j<k; j++) {
                    //printf( " %d: >> receiving A(%ld,:%ld) <<\n",rank,k,j  );
                    A.tileBcast(k, j, H.sub(k, k, std::max(j, ione)-1, std::min(j+2, k-1)-1), layout, tag);
                }
                for (int64_t i = 1; i < k; i++) {
                    if (H.tileIsLocal(k, i-1)) {
                        #pragma omp task
                        {
                            H.tileInsert(k, i-1);
                            scalar_t beta = zero;
                            for (int64_t j = std::max(i-1, ione); j <= std::min(i+1, k); j++) {
                                tile::gemm<scalar_t>(
                                    one,  A(k, j-1), T(j, i),
                                    beta, H(k, i-1) );
                                beta = one;
                            }
                        }
                    }
                }
                #pragma omp taskwait
                //printf( " >> compute H(%ld, %d:%ld) on rank-%d done <<\n", k, 0,k-1, rank); fflush(stdout);
            }
        }

        // > T(k, k) := A(k, k)
        if (T.tileIsLocal(k, k)) {
            // Intel icpc doesn't like std::max in omp task depend clause.
            int64_t k_1 = std::max(izero, k-1);
            #pragma omp task depend(in:columnL[k_1]) \
                             depend(out:columnT[k])
            {
                //printf( " >> copy A(%ld, %ld) into T(%ld, %ld) <<\n", k, k, k, k); fflush(stdout);
                T.tileInsert(k, k);
                lapack::lacpy(lapack::MatrixType::Lower,
                      A(k, k).mb(), A(k, k).nb(),
                      A(k, k).data(), A(k, k).stride(),
                      T(k, k).data(), T(k, k).stride() );
                T.tileModified(k, k);

                if (k == 0) {
                    //printf( " ++ expanding ++\n" ); fflush(stdout);
                    int64_t ldt = T(k, k).stride();
                    scalar_t *tkk = T(k, k).data();
                    for (int i = 0; i < T(k, k).mb(); i++) {
                        for (int j = i; j < T(k, k).nb(); j++) {
                            tkk[i + j*ldt] = conj( tkk[j + i*ldt] );
                        }
                    }
                }
            }
        }

        if (k > 1) {
            #pragma omp task depend(in:columnH1[k]) \
                             depend(inout:columnT[k]) \
                             priority(1)
            {
                //printf( " >> update T(%ld, %ld) on rank-%d <<\n", k, k, rank); fflush(stdout);
                auto Hj = H.sub(k, k, 0, k-2);
                Hj = conjTranspose(Hj);

                #if 0
                slate::internal::gemm_W<Target::HostTask>(
                    scalar_t(-1.0), A.sub(k, k,   0, k-2),
                                    Hj.sub(0, k-2, 0, 0),
                    scalar_t( 1.0), T.sub(k, k,   k, k),
                              ind1, std::move(W1));
                #else
                slate::internal::gemmA<Target::HostTask>(
                    -one, A.sub(k, k,   0, k-2),
                          Hj.sub(0, k-2, 0, 0),
                    one,  T.sub(k, k,   k, k), layout);
                #endif

                ReduceList reduce_list;
                reduce_list.push_back({k, k,
                                        T.sub(k, k, k, k),
                                        {A.sub(k, k, 0, k-2)}
                                      });
                T.template listReduce<target>(reduce_list, layout, tag);

                // T(k, k) -= L(k, k)*T(k, k-1)* L(k,k-1)'
                // using H(k, k) as workspace
                // > both L(k, k) and L(k, k-1) have been sent to (k, k)-th process
                //   for updating T(k, k)
                A.tileBcast(k, k-2, H.sub(k, k, k, k), layout, tag);
                A.tileBcast(k, k-1, T.sub(k, k, k, k), layout, tag);
                if (T.tileIsLocal(k, k)) {
                    H.tileInsert(k, k);
                    auto Lkj = A.sub(k, k, k-2, k-2);
                    Lkj = conjTranspose(Lkj);
                    tile::gemm<scalar_t>(
                        one,  T(k,   k-1),
                              Lkj(0, 0),
                        zero, H(k,   k) );
                    tile::gemm<scalar_t>(
                        -one, A(k, k-1),
                              H(k, k),
                        one,  T(k, k) );
                }
            }
        }

        if (k > 0) {
            #pragma omp task depend(in:columnL[k-1]) \
                             depend(inout:columnT[k]) \
                             priority(1)
            {
                // compute T(k, k) = L(k, k)^{-1} * T(k, k) * L(k, k)^{-T}
                //printf( " trsm for T(%ld, %ld) <<\n", k, k); fflush(stdout);
                if (k == 1) {
                    // > otherwise L(k, k) has been already sent to T(k, k) for updating A(k, k)
                    A.tileBcast(k, k-1, T.sub(k, k, k, k), layout, tag);
                }
                if (T.tileIsLocal(k, k)) {
                    auto Akk = A.sub(k, k, k-1, k-1);
                    auto Lkk = TriangularMatrix< scalar_t >(Uplo::Lower, Diag::NonUnit, Akk);

                    int64_t itype = 1;
                    lapack::hegst(
                        itype, lapack::Uplo::Lower, Lkk(0, 0).mb(),
                        T(k, k).data(),   T(k, k).stride(),
                        Lkk(0, 0).data(), Lkk(0, 0).stride());
                    Lkk.tileModified(0, 0);

                    //printf( " ++ expanding ++\n" ); fflush(stdout);
                    int64_t ldt = T(k, k).stride();
                    scalar_t *tkk = T(k, k).data();
                    for (int i = 0; i < T(k, k).mb(); i++) {
                        for (int j = i; j < T(k, k).nb(); j++) {
                            tkk[i + j*ldt] = conj( tkk[j + i*ldt] );
                        }
                    }
                    T.tileModified(k, k);
                }
                if (k+1 < A_mt) {
                    // send T(k, k) for computing H(k, k), moved from below?
                    //printf( " tileBcast(T(%ld,%ld) to H(%ld,%ld) )\n",k,k,k,k-1 );
                    T.tileBcast(k, k, H.sub(k, k, k-1, k-1), layout, tag);
                }
            }

            if (k+1 < A_mt) {
                #pragma omp task depend(in:columnT[k]) \
                                 depend(out:columnT[k+1]) \
                                 priority(1)
                {
                    // send T(k, k) that are needed to compute H(k+1:mt-1, k-1)
                    //printf( " %d: Bcast( T(%ld,%ld) )\n",rank,k,k ); fflush(stdout);
                    T.tileBcast(k, k, H.sub(k+1, A_mt-1, k-1, k-1), layout, tag2);
                }
            }
        }

        if (k+1 < A_mt) {
            if (k > 0) {
                #pragma omp task depend(in:columnT[k]) \
                                 depend(in:columnL[k-1]) \
                                 depend(inout:columnH2[k]) \
                                 priority(1)
                {
                    //printf( " >> compute H(%ld, %ld) on rank-%d <<\n", k, k, rank); fflush(stdout);
                    // compute H(k, k) = T(k, k) * L(k, k)^T
                    //T.tileBcast(k, k, H.sub(k, k, k-1, k-1), tag);
                    if (H.tileIsLocal(k, k-1)) {
                        H.tileInsert(k, k-1);
                        tile::gemm<scalar_t>(
                            one,  A(k, k-1),
                                  T(k, k),
                            zero, H(k, k-1) );
                    }
                    if (k > 1) {
                        // compute H(k, k) += T(k, k-1) * L(k, k-1)^T
                        A.tileBcast(k, k-2, H.sub(k, k, k-1, k-1), layout, tag);
                        if (H.tileIsLocal(k, k-1)) {
                            tile::gemm<scalar_t>(
                                one, A(k,   k-2),
                                     T(k-1, k),
                                one, H(k,   k-1) );
                        }
                    }
                }

                // Big left-looking Gemm: A(k+1:mt, k) -= L(k+1:mt, 1:k-2) * H(k, 2:k-2)^T
                if (k > 1) {
                    #pragma omp task depend(in:columnH1[k]) \
                                     depend(in:columnL[k-1]) \
                                     depend(inout:columnL[k]) \
                                     priority(1)
                    {
                        //printf( " >> update A1(%ld:%ld, %ld) on rank-%d <<\n", k+1,A_mt-1, k, rank); fflush(stdout);
                        if (k > 2) {
                            for (int64_t j = 0; j < k-1; j++) {
                                H.tileBcast(k, j, A.sub(k+1, A_mt-1, j, j), layout, tag1);
                            }
                            auto Hj = H.sub(k, k, 0, k-2);
                            Hj = conjTranspose(Hj);

                            #if 1
                                slate::internal::gemmA<Target::HostTask>(
                                    -one, A.sub(k+1, A_mt-1, 0, k-2),
                                          Hj.sub(0, k-2, 0, 0),
                                    one,  A.sub(k+1, A_mt-1, k, k), layout);
                            #else
                                if (A_mt - (k+1) > max_panel_threads) {
                                    slate::internal::gemmA<Target::HostTask>(
                                        scalar_t(-1.0), A.sub(k+1, A_mt-1, 0, k-2),
                                                        Hj.sub(0, k-2, 0, 0),
                                        scalar_t( 1.0), A.sub(k+1, A_mt-1, k, k));
                                }
                                else {
                                    slate::internal::gemm_W<Target::HostTask>(
                                        scalar_t(-1.0), A.sub(k+1, A_mt-1, 0, k-2),
                                                        Hj.sub(0, k-2, 0, 0),
                                        scalar_t( 1.0), A.sub(k+1, A_mt-1, k, k),
                                        ind2,           W2.sub(k+1, A_mt-1, 0, W2.nt()-1));
                                }
                            #endif

                            ReduceList reduce_list;
                            for (int i = k+1; i < A_mt; ++i) {
                                reduce_list.push_back({i, k,
                                                        A.sub(i, i, k, k),
                                                        {A.sub(i, i, 0, k-2)}
                                                      });
                            }
                            A.template listReduce<target>(reduce_list, layout, tag1);
                        }
                        else {
                            for (int64_t j = 0; j < k-1; j++) {
                                for (int64_t i = k+1; i < A_mt; i++) {
                                    A.tileBcast(i, j, A.sub(i, i, k, k), layout, tag1);
                                }
                                H.tileBcast(k, j, A.sub(k+1, A_mt-1, k, k), layout, tag1);
                            }
                            for (int64_t j = 0; j < k-1; j++) {
                                auto Hj = H.sub(k, k, j, j);
                                Hj = conjTranspose(Hj);
                                slate::internal::gemm<target>(
                                    -one, A.sub(k+1, A_mt-1, j, j),
                                          Hj.sub(0, 0, 0, 0),
                                    one,  A.sub(k+1, A_mt-1, k, k),
                                    layout, priority_one);
                            }
                        }
                    }
                }
                // Big left-looking Gemm: A(k+1:mt, k) -= L(k+1:mt, k-1) * H(k, k-1)^T
                #pragma omp task depend(in:columnH2[k]) \
                                 depend(inout:columnL[k]) \
                                 priority(1)
                {
                    //printf( " >> update A2(%ld:%ld, %ld) on rank-%d <<\n", k+1,A_mt-1, k, rank); fflush(stdout);
                    for (int64_t i2 = k+1; i2 < A_mt; i2++) {
                        A.tileBcast(i2, k-1, A.sub(i2, i2, k, k), layout, tag1);
                    }
                    H.tileBcast(k, k-1, A.sub(k+1, A_mt-1, k, k), layout, tag1);

                    auto Hj = H.sub(k, k, k-1, k-1);
                    Hj = conjTranspose(Hj);
                    slate::internal::gemm<target>(
                        -one, A.sub(k+1, A_mt-1, k-1, k-1),
                              Hj.sub(0,   0,     0, 0),
                        one,  A.sub(k+1, A_mt-1, k, k),
                        layout, priority_one);
                }
            }

            int64_t diag_len = std::min(A.tileMb(k+1), A.tileNb(k));
            pivots.at(k+1).resize(diag_len);
            #pragma omp task depend(inout:columnL[k]) priority(1)
            {
                //printf( " >> LU panel(%ld:%ld,%ld) diag_len=%ld on rank-%d <<\n", k+1, A_mt-1, k, diag_len, rank); fflush(stdout);
                internal::getrf_panel<Target::HostTask>(
                    A.sub(k+1, A_mt-1, k, k), diag_len, ib,
                    pivots.at(k+1), max_panel_threads, priority_one);

                // copy U(k, k) into T(k+1, k)
                //printf( " >> compute T(%ld,%ld) on rank-%d <<\n", k+1, k, rank); fflush(stdout);
                if (T.tileIsLocal(k+1, k)) {
                    T.tileInsert(k+1, k);
                    lapack::lacpy(
                        lapack::MatrixType::Upper,
                        A(k+1, k).mb(), A(k+1, k).nb(),
                        A(k+1, k).data(), A(k+1, k).stride(),
                        T(k+1, k).data(), T(k+1, k).stride() );
                    lapack::laset(
                        lapack::MatrixType::Lower,
                        T(k+1, k).mb()-1, T(k+1, k).nb()-1,
                        zero, zero,
                        T(k+1, k).data()+1, T(k+1, k).stride());
                    T.tileModified(k+1, k);

                    // zero out upper-triangular of L(k, k)
                    // and set diagonal to one.
                    lapack::laset(
                        lapack::MatrixType::Upper,
                        A(k+1, k).mb(), A(k+1, k).nb(),
                        zero, one,
                        A(k+1, k).data(), A(k+1, k).stride());
                    A.tileModified(k+1, k);
                }
            }

            #pragma omp task depend(inout:columnL[k]) \
                             depend(inout:columnT[k+1]) priority(1)
            {
                if (k > 0) {
                    // T(k+1,k) /= L(k,k)^T
                    //printf( " >> update T(%ld,%ld) on rank-%d <<\n", k+1, k, rank); fflush(stdout);
                    A.tileBcast(k, k-1, T.sub(k+1, k+1, k, k), layout, tag);

                    if (T.tileIsLocal(k+1, k)) {
                        auto Akk = A.sub(k, k, k-1, k-1);
                        auto Lkk = TriangularMatrix< scalar_t >(Uplo::Lower, Diag::NonUnit, Akk);

                        Lkk = conjTranspose(Lkk);
                        tile::trsm(
                            Side::Right, Diag::Unit,
                            one, Lkk(0, 0), T(k+1, k) );
                    }
                }
                // copy T(k+1, k)^T into T(k, k+1)
                //printf( " >> copy T(%ld,%ld) on rank-%d <<\n", k, k+1, rank); fflush(stdout);
                T.tileBcast(k+1, k, T.sub(k, k, k+1, k+1), layout, tag);
                if (T.tileIsLocal(k, k+1)) {
                    T.tileInsert(k, k+1);
                    int64_t ldt1 = T(k+1, k).stride();
                    int64_t ldt2 = T(k, k+1).stride();
                    scalar_t *tkk1 = T(k+1, k).data();
                    scalar_t *tkk2 = T(k, k+1).data();
                    for (int i=0; i < T(k+1, k).mb(); i++) {
                        for (int j = 0; j < i; j++) {
                            tkk2[j + i*ldt2] = 0.0;
                        }
                        for (int j = i; j < T(k+1, k).nb(); j++) {
                            tkk2[j + i*ldt2] = conj( tkk1[i + j*ldt1] );
                        }
                    }
                    T.tileModified(k, k+1);
                }
                if (k > 0 && k+1 < A_mt) {
                    // send T(i, j) that are needed to compute H(k, :)
                    T.tileBcast(k, k+1, H.sub(k+1, A_mt-1, k,   k), layout, tag);

                    //T.tileBcast(k+1, k, H.sub(k+1, A_mt-1, k-1, k-1), tag);
                    BcastList bcast_list_T;
                    // for computing H(j, 1:j-1)
                    bcast_list_T.push_back({k+1, k, {A.sub(k+1, A_mt-1, k-1, k-1)}});
                    // for computing T(j, j)
                    bcast_list_T.push_back({k+1, k, {A.sub(k+1, k+1,    k+1, k+1)}});
                    T.template listBcast(bcast_list_T, layout, tag);
                }
            }
            #pragma omp task depend(inout:columnL[k])
            {
                {
                    //printf( " MPI_Bcast(pivot(%ld): size=%ld\n",k+1,pivots.at(k+1).size() );
                    trace::Block trace_block("MPI_Bcast");
                    MPI_Bcast(pivots.at(k+1).data(),
                              sizeof(Pivot)*pivots.at(k+1).size(),
                              MPI_BYTE, A.tileRank(k+1, k), A.mpiComm());
                }
                if (k > 0) {
                    // swap previous rows in A(k+1:mt-1, 0:k-1)
                    //printf( " +++ swap previous L (%ld: Asub(%ld:%ld, 0:%ld))\n",k,k+1,A_mt-1,k-1);
                    #pragma omp task
                    {
                        internal::permuteRows<Target::HostTask>(
                            Direction::Forward, A.sub(k+1, A_mt-1, 0, k-1),
                            pivots.at(k+1), layout, 1, tag3);
                    }
                }
                // symmetric swap of A(k+1:mt-1, k+1:mt-1)
                //printf( " +++ symmetric swap A(%ld:%ld, %ld:%ld) +++\n",k+1,A_mt-1, k+1,A_mt-1 );
                #pragma omp task
                {
                    internal::permuteRowsCols<Target::HostTask>(
                        Direction::Forward, A.sub(k+1, A_mt-1),
                        pivots.at(k+1), 1, tag4);
                }
                #pragma omp taskwait
            }
        }
    }

    // Debug::checkTilesLives(A);
    // Debug::printTilesLives(A);

    // second-stage (facorization of band matrix)
    gbtrf(T, pivots2, {
        {Option::InnerBlocking, ib},
        {slate::Option::Lookahead, lookahead},
        {slate::Option::MaxPanelThreads, max_panel_threads}});

    A.clearWorkspace();

    // Debug::printTilesMaps(A);
}

} // namespace impl

//------------------------------------------------------------------------------
/// Distributed parallel Hermitian indefinite $LTL^T$ factorization.
///
/// Computes the factorization of a Hermitian matrix $A$
/// using Aasen's 2-stage algorithm.  The form of the factorization is
/// \[
///     A = L T L^H,
/// \]
/// if $A$ is stored lower, where $L$ is a product of permutation and unit
/// lower triangular matrices, or
/// \[
///     P A P^H = U^H T U,
/// \]
/// if $A$ is stored upper, where $U$ is a product of permutation and unit
/// upper triangular matrices.
/// $T$ is a Hermitian band matrix that is LU factorized with partial pivoting.
///
/// Complexity (in real): $\approx \frac{1}{3} n^{3}$ flops.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the n-by-n Hermitian matrix $A$.
///     On exit, if return value = 0, overwritten by the factor $U$ or $L$ from
///     the factorization $A = U^H T U$ or $A = L T L^H$.
///     If scalar_t is real, $A$ can be a SymmetricMatrix object.
///
/// @param[out] pivots
///     On exit, details of the interchanges applied to $A$, i.e.,
///     row and column k of $A$ were swapped with row and column pivots(k).
///
/// @param[out] T
///     On exit, details of the LU factorization of the band matrix.
///
/// @param[out] pivots2
///     On exit, details of the interchanges applied to $T$, i.e.,
///     row and column k of $T$ were swapped with row and column pivots2(k).
///
/// @param[out] H
///     Auxiliary matrix used during the factorization.
///     TODO: can this be made internal?
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
/// @ingroup hesv_computational
///
template <typename scalar_t>
void hetrf(
    HermitianMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& H,
    Options const& opts)
{
    Target target = get_option( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            impl::hetrf<Target::HostTask>( A, pivots, T, pivots2, H, opts );
            break;

        case Target::HostNest:
            impl::hetrf<Target::HostNest>( A, pivots, T, pivots2, H, opts );
            break;

        case Target::HostBatch:
            impl::hetrf<Target::HostBatch>( A, pivots, T, pivots2, H, opts );
            break;

        case Target::Devices:
            slate_not_implemented( "hetrf not yet implemented for GPU devices" );
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void hetrf<float>(
    HermitianMatrix<float>& A, Pivots& pivots,
         BandMatrix<float>& T, Pivots& pivots2,
             Matrix<float>& H,
    Options const& opts);

template
void hetrf<double>(
    HermitianMatrix<double>& A, Pivots& pivots,
         BandMatrix<double>& T, Pivots& pivots2,
             Matrix<double>& H,
    Options const& opts);

template
void hetrf< std::complex<float> >(
    HermitianMatrix< std::complex<float> >& A, Pivots& pivots,
         BandMatrix< std::complex<float> >& T, Pivots& pivots2,
             Matrix< std::complex<float> >& H,
    Options const& opts);

template
void hetrf< std::complex<double> >(
    HermitianMatrix< std::complex<double> >& A, Pivots& pivots,
         BandMatrix< std::complex<double> >& T, Pivots& pivots2,
             Matrix< std::complex<double> >& H,
    Options const& opts);

} // namespace slate
