// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "internal/internal.hh"

namespace slate {

namespace impl {

//------------------------------------------------------------------------------
/// @internal
/// Distributed parallel Hermitian matrix-matrix multiplication.
/// Generic implementation for any target.
/// Dependencies enforce the following behavior:
/// - bcast communications are serialized,
/// - hemm operations are serialized,
/// - bcasts can get ahead of hemms by the value of lookahead.
/// Note A, B, and C are passed by value, so we can transpose if needed
/// (for side = right) without affecting caller.
/// @ingroup hemm_impl
///
/// ColMajor layout is assumed
///
template <Target target, typename scalar_t>
void hemmA(
    Side side,
    scalar_t alpha, HermitianMatrix<scalar_t> A,
                    Matrix<scalar_t> B,
    scalar_t beta,  Matrix<scalar_t> C,
    Options const& opts )
{
    using blas::conj;
    //using BcastListTag = typename Matrix<scalar_t>::BcastListTag;
    using BcastList = typename Matrix<scalar_t>::BcastList;

    const scalar_t one = 1.0;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    // Options
    int64_t lookahead = get_option<int64_t>( opts, Option::Lookahead, 1 );

    // if on right, change to left by transposing A, B, C to get
    // op(C) = op(A)*op(B)
    if (side == Side::Right) {
        A = conjTranspose(A);
        B = conjTranspose(B);
        C = conjTranspose(C);
        alpha = conj(alpha);
        beta  = conj(beta);
    }

    // B and C are mt-by-nt, A is mt-by-mt (assuming side = left)
    assert(A.mt() == B.mt());
    assert(A.nt() == B.mt());
    assert(B.mt() == C.mt());
    assert(B.nt() == C.nt());

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector<uint8_t> bcast_vector(A.nt());
    std::vector<uint8_t>  gemm_vector(A.nt());
    uint8_t* bcast = bcast_vector.data();
    uint8_t* gemm  =  gemm_vector.data();


    if (target == Target::Devices) {
        C.allocateBatchArrays();
        C.reserveDeviceWorkspace();
    }

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
        if (A.uplo() == Uplo::Lower) {
            // ----------------------------------------
            // Left, Lower/NoTrans or Upper/ConjTrans case

            // send 1st block row of B
            #pragma omp task depend(out:bcast[0])
            {
                // Bcast B(0, :) to ranks owning A(:, 0)
                BcastList bcast_list_B;
                for (int64_t j = 0; j < B.nt(); ++j)
                    bcast_list_B.push_back({0, j, {A.sub(0, A.mt()-1, 0, 0)}});
                B.template listBcast<target>(bcast_list_B, layout);

                // Move C(:, 0) where A(:, 0) is located and reset the
                // original C to 0 if moved
                // NOTE: if A(i,0) does not have C(i,:), the tile is created
                // and will be filled-in when the original C(i,:) is sent.

                // Create C tiles
                for (int64_t i = 0; i < A.mt(); ++i) {
                    if (A.tileIsLocal(i, 0)) {
                        for (int64_t j = 0; j < B.nt(); ++j) {
                            if (! C.tileIsLocal(i, j)) {
                                C.tileInsert(i, j);
                                C.at(i, j).set(0);
                            }
                        }
                    }
                }

                // Move C
                for (int64_t i = 0; i < C.mt(); ++i) {
                    for (int64_t j = 0; j < C.nt(); ++j) {
                        if (A.tileIsLocal(i, 0) && ! C.tileIsLocal(i, j)) {
                            int root = C.tileRank(i, j);

                            C.tileRecv(i, j, root, layout);
                        }
                        else if (C.tileIsLocal(i, j) && ! A.tileIsLocal(i, 0)) {
                            int dest = A.tileRank(i, 0);
                            C.tileSend(i, j, dest);

                            // Since C is moved, the local tile has to
                            // be reset to 0 in order to perform the reduce
                            // operation later.
                            C.at(i, j).set(0);
                        }
                    }
                }
            }

            // send next lookahead block cols of A and block rows of B
            for (int64_t k = 1; k < lookahead+1 && k < A.nt(); ++k) {
                #pragma omp task depend(in:bcast[k-1]) \
                                 depend(out:bcast[k])
                {
                    // broadcast B(k, j) to ranks owning block col A(:, k)
                    // which is in reality A(k:end, k) and A(k, 1:k-1).
                    BcastList bcast_list_B;
                    for (int64_t j = 0; j < B.nt(); ++j) {
                        bcast_list_B.push_back(
                            {k, j, {
                                    A.sub(k, k, 0, k-1),
                                    A.sub(k, A.mt()-1, k, k)
                                    }});
                    }
                    B.template listBcast<target>(bcast_list_B, layout);

                    // Create C tiles
                    for (int64_t i = 0; i < A.mt(); ++i) {
                        if (i < k) {
                            if (A.tileIsLocal(k, i)) {
                                for (int64_t j = 0; j < B.nt(); ++j) {
                                    if (! C.tileIsLocal(i, j)
                                        && ! C.tileExists(i, j))
                                    {
                                        C.tileInsert(i, j);
                                        C.at(i, j).set(0);
                                    }
                                }
                            }
                        }
                        else {
                            if (A.tileIsLocal(i, k)) {
                                for (int64_t j = 0; j < B.nt(); ++j) {
                                    if (! C.tileIsLocal(i, j)
                                        && ! C.tileExists(i, j))
                                    {
                                        C.tileInsert(i, j);
                                        C.at(i, j).set(0);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // multiply alpha A(:, 0) B(0, :), which is (hemm / gemm):
            // C(0, :)      = alpha [ A(0, 0)      B(0, :) ] + beta C(0, :)
            // C(1:mt-1, :) = alpha [ A(1:mt-1, 0) B(0, :) ] + beta C(1:mt-1, :)
            #pragma omp task depend(in:bcast[0]) \
                             depend(out:gemm[0])
            {
                internal::hemmA<Target::HostTask>(
                    Side::Left,
                    alpha, A.sub(0, 0),
                           B.sub(0, 0, 0, B.nt()-1),
                    beta,  C.sub(0, 0, 0, C.nt()-1));

                if (A.mt()-1 > 0) {
                    internal::gemmA<target>(
                        alpha,  A.sub(1, A.mt()-1, 0, 0),
                                B.sub(0, 0, 0, B.nt()-1),
                        beta,   C.sub(1, C.mt()-1, 0, C.nt()-1),
                                layout);
                }
            }

            // Main loop
            for (int64_t k = 1; k < A.nt(); ++k) {

                // send next block col of A and block row of B
                if (k+lookahead < A.nt()) {
                    #pragma omp task depend(in:gemm[k-1]) \
                                     depend(in:bcast[k+lookahead-1]) \
                                     depend(out:bcast[k+lookahead])
                    {
                        // broadcast B(k+la, j) to ranks
                        // owning block col C(0:k+la, j)
                        BcastList bcast_list_B;
                        for (int64_t j = 0; j < B.nt(); ++j) {
                            bcast_list_B.push_back(
                                {k+lookahead, j, {
                                                  A.sub(k+lookahead, k+lookahead,
                                                        0,  k+lookahead-1),
                                                  A.sub(k+lookahead, A.mt()-1,
                                                        k+lookahead, k+lookahead)
                                                  }});
                        }
                        B.template listBcast<target>(bcast_list_B, layout);

                        // Create C tiles
                        for (int64_t i = 0; i < A.mt(); ++i) {
                            if (i < k + lookahead) {
                                if (A.tileIsLocal(k+lookahead, i)) {
                                    for (int64_t j = 0; j < B.nt(); ++j) {
                                        if (! C.tileIsLocal(i, j)
                                            && ! C.tileExists(i, j))
                                        {
                                            C.tileInsert(i, j);
                                            C.at(i, j).set(0);
                                        }
                                    }
                                }
                            }
                            else {
                                if (A.tileIsLocal(i, k+lookahead)) {
                                    for (int64_t j = 0; j < B.nt(); ++j) {
                                        if (! C.tileIsLocal(i, j)
                                            && ! C.tileExists(i, j))
                                        {
                                            C.tileInsert(i, j);
                                            C.at(i, j).set(0);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // multiply alpha A(:, k) B(k, :), which is:
                // C(0:k-1, :)    += alpha [ A(k, 0:k-1)^H  B(k, :) ]  gemm
                // C(k, :)        += alpha [ A(k, k)        B(k, :) ]  hemm
                // C(k+1:mt-1, :) += alpha [ A(k+1:mt-1, k) B(k, :) ]  gemm
                #pragma omp task depend(in:bcast[k]) \
                                 depend(in:gemm[k-1]) \
                                 depend(out:gemm[k])
                {
                    auto Arow_k = A.sub(k, k, 0, k-1);

                    internal::gemmA<target>(
                        alpha, conj_transpose( Arow_k ),
                               B.sub(k, k, 0, B.nt()-1),
                        one,   C.sub(0, k-1, 0, C.nt()-1),
                        layout);

                    internal::hemmA<Target::HostTask>(
                        Side::Left,
                        alpha, A.sub(k, k),
                               B.sub(k, k, 0, B.nt()-1),
                        one,   C.sub(k, k, 0, C.nt()-1));

                    if (A.mt()-1 > k) {
                        internal::gemmA<target>(
                            alpha, A.sub(k+1, A.mt()-1, k, k),
                                   B.sub(k, k, 0, B.nt()-1),
                            one,   C.sub(k+1, C.mt()-1, 0, C.nt()-1),
                            layout);
                    }
                }

            }

            #pragma omp task depend(in:gemm[A.nt()-1])
            {
                // Move the solution to the right place
                using ReduceList = typename Matrix<scalar_t>::ReduceList;
                ReduceList reduce_list_C;
                for (int64_t i = 0; i < C.mt(); ++i) {
                    for (int64_t j = 0; j < C.nt(); ++j) {
                        if (i == 0) {
                            reduce_list_C.push_back({i, j,
                                C.sub(i, i, j, j),
                                { A.sub(i, A.mt()-1, i, i) }
                                });
                        }
                        else {
                            reduce_list_C.push_back({i, j,
                                C.sub(i, i, j, j),
                                { A.sub(i, i, 0, i-1),
                                  A.sub(i, A.mt()-1, i, i) }
                                });
                        }
                        C.template listReduce<target>(reduce_list_C, layout);
                        reduce_list_C.clear();
                        // Release the memory
                        if (C.tileExists(i, j) && ! C.tileIsLocal(i, j))
                            C.tileErase(i, j);
                    }
                }
            }
        }
        else {
            // ----------------------------------------
            // Left, Upper/NoTrans or Lower/ConjTrans case

            // send 1st block row of B
            #pragma omp task depend(out:bcast[0])
            {
                // Bcast B(0, :) to ranks owning A(:, 0)
                BcastList bcast_list_B;
                for (int64_t j = 0; j < B.nt(); ++j)
                    bcast_list_B.push_back({0, j, {A.sub(0, 0, 0, A.nt()-1)}});
                B.template listBcast<target>(bcast_list_B, layout);

                // Move C(:, 0) where A(:, 0) is located and reset the
                // original C to 0 if moved
                // NOTE: if A(i,0) does not have C(i,:), the tile is created
                // and will be filled-in when the original C(i,:) is sent.

                // Create C tiles
                for (int64_t i = 0; i < A.nt(); ++i) {
                    if (A.tileIsLocal(0, i)) {
                        for (int64_t j = 0; j < B.nt(); ++j) {
                            if (! C.tileIsLocal(i, j)) {
                                C.tileInsert(i, j);
                                C.at(i, j).set(0);
                            }
                        }
                    }
                }

                // Move C
                for (int64_t i = 0; i < C.mt(); ++i) {
                    for (int64_t j = 0; j < C.nt(); ++j) {
                        if (A.tileIsLocal(0, i) && ! C.tileIsLocal(i, j)) {
                            int root = C.tileRank(i, j);
                            C.tileRecv(i, j, root, layout);
                        }
                        else if (C.tileIsLocal(i, j) && ! A.tileIsLocal(0, i)) {
                            int dest = A.tileRank(0, i);
                            C.tileSend(i, j, dest);

                            // Since C is moved, the local tile has to
                            // be reset to 0 in order to perform the reduce
                            // operation later.
                            C.at(i, j).set(0);
                        }
                    }
                }
            }

            // send next lookahead block cols of A and block rows of B
            for (int64_t k = 1; k < lookahead+1 && k < A.nt(); ++k) {
                #pragma omp task depend(in:bcast[k-1]) \
                                 depend(out:bcast[k])
                {
                    // broadcast B(k, j) to ranks owning block col A(:, k)
                    // which is in reality A(0:k-1, k) and A(k, k:end).
                    BcastList bcast_list_B;
                    for (int64_t j = 0; j < B.nt(); ++j) {
                        bcast_list_B.push_back(
                            {k, j, {
                                    A.sub(0, k-1, k, k),
                                    A.sub(k, k, k, A.nt()-1),
                                    }});
                    }
                    B.template listBcast<target>(bcast_list_B, layout);

                    // Create C tiles
                    for (int64_t i = 0; i < A.nt(); ++i) {
                        if (i < k) {
                            if (A.tileIsLocal(i, k)) {
                                for (int64_t j = 0; j < B.nt(); ++j) {
                                    if (! C.tileIsLocal(i, j)
                                        && ! C.tileExists(i, j))
                                    {
                                        C.tileInsert(i, j);
                                        C.at(i, j).set(0);
                                    }
                                }
                            }
                        }
                        else {
                            if (A.tileIsLocal(k, i)) {
                                for (int64_t j = 0; j < B.nt(); ++j) {
                                    if (! C.tileIsLocal(i, j)
                                        && ! C.tileExists(i, j))
                                    {
                                        C.tileInsert(i, j);
                                        C.at(i, j).set(0);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // multiply alpha A(:, 0) B(0, :), which is (hemm / gemm):
            // C(0, :)      = alpha [ A(0, 0)        B(0, :) ] + beta C(0, :)
            // C(1:mt-1, :) = alpha [ A(0, 1:mt-1)^H B(0, :) ] + beta C(1:mt-1, :)
            #pragma omp task depend(in:bcast[0]) \
                             depend(out:gemm[0])
            {
                internal::hemmA<Target::HostTask>(
                    Side::Left,
                    alpha, A.sub(0, 0),
                           B.sub(0, 0, 0, B.nt()-1),
                    beta,  C.sub(0, 0, 0, C.nt()-1));

                if (A.mt()-1 > 0) {
                    auto Arow_k = A.sub(0, 0, 1, A.nt()-1);
                    internal::gemmA<target>(
                        alpha, conjTranspose(Arow_k),
                               B.sub(0, 0, 0, B.nt()-1),
                        beta,  C.sub(1, C.mt()-1, 0, C.nt()-1),
                        layout);
                }
            }

            // Main loop
            for (int64_t k = 1; k < A.nt(); ++k) {

                // send next block col of A and block row of B
                if (k+lookahead < A.nt()) {
                    #pragma omp task depend(in:gemm[k-1]) \
                                     depend(in:bcast[k+lookahead-1]) \
                                     depend(out:bcast[k+lookahead])
                    {
                        // broadcast B(k+la, j) to ranks
                        // owning block col C(0:k+la, j)
                        BcastList bcast_list_B;
                        for (int64_t j = 0; j < B.nt(); ++j) {
                            bcast_list_B.push_back(
                                {k+lookahead, j, {
                                                  A.sub(0, k+lookahead-1,
                                                        k+lookahead, k+lookahead),
                                                  A.sub(k+lookahead, k+lookahead,
                                                        k+lookahead, A.nt()-1),
                                                  }});
                        }
                        B.template listBcast<target>(bcast_list_B, layout);

                        // Create C tiles
                        for (int64_t i = 0; i < A.nt(); ++i) {
                            if (i < k + lookahead) {
                                if (A.tileIsLocal(i, k+lookahead)) {
                                    for (int64_t j = 0; j < B.nt(); ++j) {
                                        if (! C.tileIsLocal(i, j)
                                            && ! C.tileExists(i, j))
                                        {
                                            C.tileInsert(i, j);
                                            C.at(i, j).set(0);
                                        }
                                    }
                                }
                            }
                            else {
                                if (A.tileIsLocal(k+lookahead, i)) {
                                    for (int64_t j = 0; j < B.nt(); ++j) {
                                        if (! C.tileIsLocal(i, j)
                                            && ! C.tileExists(i, j))
                                        {
                                            C.tileInsert(i, j);
                                            C.at(i, j).set(0);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // multiply alpha A(:, k) B(k, :), which is:
                // C(0:k-1, :)    += alpha [ A(k, 0:k-1)      B(k, :) ]  gemm
                // C(k, :)        += alpha [ A(k, k)          B(k, :) ]  hemm
                // C(k+1:mt-1, :) += alpha [ A(k+1:mt-1, k)^H B(k, :) ]  gemm
                #pragma omp task depend(in:bcast[k]) \
                                 depend(in:gemm[k-1]) \
                                 depend(out:gemm[k])
                {
                    internal::gemmA<target>(
                        alpha, A.sub(0, k-1, k, k),
                               B.sub(k, k, 0, B.nt()-1),
                        one,   C.sub(0, k-1, 0, C.nt()-1),
                        layout);

                    internal::hemmA<Target::HostTask>(
                        Side::Left,
                        alpha, A.sub(k, k),
                               B.sub(k, k, 0, B.nt()-1),
                        one,   C.sub(k, k, 0, C.nt()-1));

                    if (A.nt()-1 > k) {
                        auto Arow_k = A.sub(k, k, k+1, A.nt()-1);
                        internal::gemmA<target>(
                            alpha, conj_transpose( Arow_k ),
                                   B.sub(k, k, 0, B.nt()-1),
                            one,   C.sub(k+1, C.mt()-1, 0, C.nt()-1),
                            layout);
                    }
                }
            }

            #pragma omp task depend(in:gemm[A.nt()-1])
            {
                // Move the solution to the right place
                using ReduceList = typename Matrix<scalar_t>::ReduceList;
                ReduceList reduce_list_C;
                for (int64_t i = 0; i < C.mt(); ++i) {
                    for (int64_t j = 0; j < C.nt(); ++j) {
                        if (i == 0) {
                            reduce_list_C.push_back({i, j,
                                C.sub(i, i, j, j),
                                { A.sub(i, i, i, A.nt()-1) }
                                });
                        }
                        else {
                            reduce_list_C.push_back({i, j,
                                  C.sub(i, i, j, j),
                                  { A.sub(0, i-1, i, i),
                                    A.sub(i, i, i, A.nt()-1)
                                  }
                                });
                        }
                        C.template listReduce<target>(reduce_list_C, layout);
                        reduce_list_C.clear();
                        // Release the memory
                        if (C.tileExists(i, j) && ! C.tileIsLocal(i, j))
                            C.tileErase(i, j);
                    }
                }
            }
        }

        #pragma omp taskwait
        C.tileUpdateAllOrigin();
    }
    C.releaseWorkspace();
}

} // namespace impl

//------------------------------------------------------------------------------
/// Distributed parallel Hermitian matrix-matrix multiplication.
/// Performs one of the matrix-matrix operations
/// \[
///     C = \alpha A B + \beta C
/// \]
/// or
/// \[
///     C = \alpha B A + \beta C
/// \]
/// where alpha and beta are scalars, A is a Hermitian matrix and B and
/// C are m-by-n matrices.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///         One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] side
///         Whether the Hermitian matrix A appears on the left or right:
///         - Side::Left:  $C = \alpha A B + \beta C$
///         - Side::Right: $C = \alpha B A + \beta C$
///
/// @param[in] alpha
///         The scalar alpha.
///
/// @param[in] A
///         - If side = left,  the m-by-m Hermitian matrix A;
///         - if side = right, the n-by-n Hermitian matrix A.
///
/// @param[in] B
///         The m-by-n matrix B.
///
/// @param[in] beta
///         The scalar beta.
///
/// @param[in,out] C
///         On entry, the m-by-n matrix C.
///         On exit, overwritten by the result
///         $\alpha A B + \beta C$ or $\alpha B A + \beta C$.
///
/// @param[in] opts
///         Additional options, as map of name = value pairs. Possible options:
///         - Option::Lookahead:
///           Number of blocks to overlap communication and computation.
///           lookahead >= 0. Default 1.
///         - Option::Target:
///           Implementation to target. Possible values:
///           - HostTask:  OpenMP tasks on CPU host [default].
///           - HostNest:  nested OpenMP parallel for loop on CPU host.
///           - HostBatch: batched BLAS on CPU host.
///           - Devices:   batched BLAS on GPU device.
///
/// @ingroup hemm
///
template <typename scalar_t>
void hemmA(
    Side side,
    scalar_t alpha, HermitianMatrix<scalar_t>& A,
                    Matrix<scalar_t>& B,
    scalar_t beta,  Matrix<scalar_t>& C,
    Options const& opts )
{
    Target target = get_option<Target>( opts, Option::Target, Target::HostTask );

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            impl::hemmA<Target::HostTask>( side, alpha, A, B, beta, C, opts );
            break;

        case Target::HostNest:
        case Target::HostBatch:
        case Target::Devices:
            slate_not_implemented("target not yet supported");
            break;
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void hemmA<float>(
    Side side,
    float alpha, HermitianMatrix<float>& A,
                 Matrix<float>& B,
    float beta,  Matrix<float>& C,
    Options const& opts);

template
void hemmA<double>(
    Side side,
    double alpha, HermitianMatrix<double>& A,
                  Matrix<double>& B,
    double beta,  Matrix<double>& C,
    Options const& opts);

template
void hemmA< std::complex<float> >(
    Side side,
    std::complex<float> alpha, HermitianMatrix< std::complex<float> >& A,
                               Matrix< std::complex<float> >& B,
    std::complex<float> beta,  Matrix< std::complex<float> >& C,
    Options const& opts);

template
void hemmA< std::complex<double> >(
    Side side,
    std::complex<double> alpha, HermitianMatrix< std::complex<double> >& A,
                                Matrix< std::complex<double> >& B,
    std::complex<double> beta,  Matrix< std::complex<double> >& C,
    Options const& opts);

} // namespace slate
