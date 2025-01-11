// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "slate/types.hh"
#include "internal/internal.hh"

#include <vector>

namespace slate {

namespace internal {

// Helper function to build a bcast list for U and V
template<typename scalar_t>
void gerbt_setup_bcast(Side side, Matrix<scalar_t> A, int64_t i1, int64_t i2,
                       typename Matrix<scalar_t>::BcastListTag& bcast_list)
{
    int64_t n = i2-i1;
    if (side == Side::Left) {
        for (int64_t ii = 0; ii < n && ii < A.mt(); ++ii) {
            int64_t tag = ii+i1;
            bcast_list.push_back( {ii+i1, 0, {A.sub(ii, ii, 0, A.nt()-1)}, tag} );
        }
    }
    else {
        for (int64_t jj = 0; jj < n && jj < A.nt(); ++jj) {
            int64_t tag = jj+i1;
            bcast_list.push_back( {jj+i1, 0, {A.sub(0, A.mt()-1, jj, jj)}, tag} );
        }
    }
}

// Combine entries for the same source tile
template<typename scalar_t>
void gerbt_bcast_filter_duplicates(typename Matrix<scalar_t>::BcastListTag& bcast_list)
{
    for (auto outer = bcast_list.begin(); outer < bcast_list.end(); ++outer) {
        // For each entry in the bcast list, check if there are any subsequent
        // entries that have the same source tile
        for (auto inner = outer+1; inner < bcast_list.end(); ) {
            if (std::get<0>(*outer) == std::get<0>(*inner)
                && std::get<1>(*outer) == std::get<1>(*inner)) {

                // If so, add the destination tiles to the first entry
                // and remove the subsequent entry
                std::get<2>(*outer).splice(std::get<2>(*outer).begin(), std::get<2>(*inner));
                inner = bcast_list.erase(inner);
            }
            else {
                ++inner;
            }
        }
    }
}

// Helper functions to iterate over the butterfly indices
void gerbt_iterate_2d(int64_t d, int64_t inner_len, int64_t mt, int64_t nt,
                      std::function<void(int64_t, int64_t, int64_t,
                                      int64_t, int64_t, int64_t)> body)
{
    // 2-sided butterflies are applied smallest to largest
    for (int64_t k = d-1; k >= 0; --k) {
        const int64_t num_bt = 1 << k;
        // bt_len = ceil(nt * 2^-d) * 2^d * 2^-k
        // half_len = 1/2 * bt_len
        const int64_t half_len = (1 << (d-k-1))*inner_len;

        for (int64_t bi = 0; bi < num_bt; ++bi) {
            const int64_t i1 = bi*2*half_len;
            const int64_t i2 = std::min(i1+half_len, mt);
            const int64_t i3 = std::min(i2+half_len, mt);
            for (int64_t bj = 0; bj < num_bt; ++bj) {
                const int64_t j1 = bj*2*half_len;
                const int64_t j2 = std::min(j1+half_len, nt);
                const int64_t j3 = std::min(j2+half_len, nt);

                body(i1, i2, i3, j1, j2, j3);
            }
        }
    }
}
void gerbt_iterate_1d(Op trans, int64_t d, int64_t inner_len, int64_t mt,
                      std::function<void(int64_t, int64_t, int64_t)> body)
{
    for (int64_t k_iter = 0; k_iter < d; ++k_iter) {
        // Regular butterflies are applied largest to smallest
        // Transposed butterflies are applied smallest to largest
        const int64_t k = (trans == Op::NoTrans) ? k_iter : d-k_iter-1;

        const int64_t num_bt = 1 << k;
        // bt_len = ceil(mt * 2^-d) * 2^d * 2^-k
        // half_len = 1/2 * bt_len
        const int64_t half_len = (1 << (d-k-1))*inner_len;

        for (int64_t bi = 0; bi < num_bt; ++bi) {
            const int64_t i1 = bi*2*half_len;
            const int64_t i2 = std::min(i1+half_len, mt);
            const int64_t i3 = std::min(i2+half_len, mt);

            body(i1, i2, i3);
        }
    }
}

} // namespace internal

//------------------------------------------------------------------------------
/// Applies a 2-sided RBT to the given matrix.
///
/// @param[in] U_in
///     The left transform in packed storage. Should be transposed.
///
/// @param[in, out] A
///     The matrix to transform
///
/// @param[in] V
///     The right transform in packed storage. Should not be transposed.
///
/// @ingroup gesv_computational
///
template<typename scalar_t>
void gerbt(Matrix<scalar_t>& U_in,
           Matrix<scalar_t>& A,
           Matrix<scalar_t>& V)
{
    using BcastListTag = typename Matrix<scalar_t>::BcastListTag;

    slate_assert(U_in.op() == Op::Trans);
    slate_assert(V.op() == Op::NoTrans);

    Matrix<scalar_t> U = transpose(U_in);

    slate_assert(A.op() == Op::NoTrans);
    slate_assert(U.op() == Op::NoTrans);
    slate_assert(A.layout() == Layout::ColMajor);
    slate_assert(U.layout() == Layout::ColMajor);
    slate_assert(V.layout() == Layout::ColMajor);

    slate_assert(U.n() == V.n());

    const int64_t d = U.n();
    const int64_t mt = A.mt();
    const int64_t nt = A.nt();

    if (d == 0) {
        return;
    }

    int64_t inner_len = int64_t(std::ceil(nt / double(1 << d)));

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
        // Loop over the butterflies twice to reduce transfer of random factors

        // Plan which random factors are needed where
        BcastListTag bcast_list_U, bcast_list_V;
        internal::gerbt_iterate_2d(d, inner_len, mt, nt,
                [&](int64_t i1, int64_t i2, int64_t i3,
                    int64_t j1, int64_t j2, int64_t j3) {

                    auto A11 = A.sub(i1, i2-1, j1, j2-1);
                    internal::gerbt_setup_bcast( Side::Left, A11, i1, i2, bcast_list_U );
                    internal::gerbt_setup_bcast( Side::Right, A11, j1, j2, bcast_list_V );

                    if (i2 < i3) {
                        internal::gerbt_setup_bcast( Side::Left, A11, i2, i3, bcast_list_U );
                    }
                    if (j2 < j3) {
                        internal::gerbt_setup_bcast( Side::Right, A11, j2, j3, bcast_list_V );
                    }
                });

        // Bcast random factors
        internal::gerbt_bcast_filter_duplicates<scalar_t>(bcast_list_U);
        internal::gerbt_bcast_filter_duplicates<scalar_t>(bcast_list_V);

        U.template listBcastMT<>( bcast_list_U, Layout::ColMajor );
        V.template listBcastMT<>( bcast_list_V, Layout::ColMajor );

        // NB: only tasks created so far are in listBcastMT

        // Do computation
        internal::gerbt_iterate_2d(d, inner_len, mt, nt,
                [&](int64_t i1, int64_t i2, int64_t i3,
                    int64_t j1, int64_t j2, int64_t j3) {

                    auto A11 = A.sub(i1, i2-1, j1, j2-1);
                    auto A12 = A.sub(i1, i2-1, j2, j3-1);
                    auto A21 = A.sub(i2, i3-1, j1, j2-1);
                    auto A22 = A.sub(i2, i3-1, j2, j3-1);

                    auto U1 = U.sub(i1, i2-1, 0, 0);
                    auto U2 = U.sub(i2, i3-1, 0, 0);
                    auto V1 = V.sub(j1, j2-1, 0, 0);
                    auto V2 = V.sub(j2, j3-1, 0, 0);

                    internal::gerbt( A11, A12, A21, A22, U1, U2, V1, V2 );
                    // NB internal::gerbt ends with taskwait
            });

        #pragma omp taskwait
        U.releaseRemoteWorkspace();
        U.releaseLocalWorkspace();
        V.releaseRemoteWorkspace();
        V.releaseLocalWorkspace();

        A.tileUpdateAllOrigin();
    }
    A.clearWorkspace();
}

template
void gerbt(Matrix<float>&,
           Matrix<float>&,
           Matrix<float>&);

template
void gerbt(Matrix<double>&,
           Matrix<double>&,
           Matrix<double>&);

template
void gerbt(Matrix<std::complex<float>>&,
           Matrix<std::complex<float>>&,
           Matrix<std::complex<float>>&);

template
void gerbt(Matrix<std::complex<double>>&,
           Matrix<std::complex<double>>&,
           Matrix<std::complex<double>>&);


//------------------------------------------------------------------------------
/// Applies a 1-sided RBT to the given matrix on the left.
///
/// @param[in] U_in
///     The transform in packed storage
///
/// @param[in, out] A
///     The matrix to transform
///
/// @ingroup gesv_computational
///
template<typename scalar_t>
void gerbt(Matrix<scalar_t>& Uin,
           Matrix<scalar_t>& B)
{
    using BcastListTag = typename Matrix<scalar_t>::BcastListTag;

    Op trans = Uin.op();
    Matrix<scalar_t> U = (trans == Op::NoTrans) ? Uin : transpose(Uin);

    slate_assert(B.op() == Op::NoTrans);
    slate_assert(U.op() == Op::NoTrans);
    slate_assert(B.layout() == Layout::ColMajor);
    slate_assert(U.layout() == Layout::ColMajor);

    slate_assert(B.mt() == U.mt());

    const int64_t d = U.n();
    const int64_t mt = B.mt();
    const int64_t nt = B.nt();

    if (d == 0) {
        return;
    }

    int64_t inner_len = int64_t(std::ceil(mt / double(1 << d)));

    // set min number for omp nested active parallel regions
    slate::OmpSetMaxActiveLevels set_active_levels( MinOmpActiveLevels );

    #pragma omp parallel
    #pragma omp master
    {
        // Loop over the butterflies twice to reduce transfer of random factors

        // Plan which random factors are needed where
        BcastListTag bcast_list;
        internal::gerbt_iterate_1d(trans, d, inner_len, mt,
                [&](int64_t i1, int64_t i2, int64_t i3) {

                    if (i2 < i3) {
                        auto B1 = B.sub(i1, i2-1, 0, nt-1);

                        internal::gerbt_setup_bcast( Side::Left, B1, i1, i2, bcast_list );
                        internal::gerbt_setup_bcast( Side::Left, B1, i2, i3, bcast_list );
                    }
                });

        // Bcast random factors
        internal::gerbt_bcast_filter_duplicates<scalar_t>(bcast_list);
        U.template listBcastMT<>( bcast_list, Layout::ColMajor );

        // NB: only tasks created so far are in listBcastMT

        internal::gerbt_iterate_1d(trans, d, inner_len, mt,
                [&](int64_t i1, int64_t i2, int64_t i3) {
                    auto B1 = B.sub(i1, i2-1, 0, nt-1);
                    auto B2 = B.sub(i2, i3-1, 0, nt-1);

                    auto U1 = U.sub(i1, i2-1, 0, 0);
                    auto U2 = U.sub(i2, i3-1, 0, 0);

                    internal::gerbt( Side::Left, trans, B1, B2, U1, U2 );
                    // NB internal::gerbt ends with taskwait
                });

        #pragma omp taskwait
        U.releaseRemoteWorkspace();
        U.releaseLocalWorkspace();

        B.tileUpdateAllOrigin();
    }
    B.clearWorkspace();
}

template
void gerbt(Matrix<float>&,
           Matrix<float>&);

template
void gerbt(Matrix<double>&,
           Matrix<double>&);

template
void gerbt(Matrix<std::complex<float>>&,
           Matrix<std::complex<float>>&);

template
void gerbt(Matrix<std::complex<double>>&,
           Matrix<std::complex<double>>&);



} // namespace slate
