// Copyright (c) 2020-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Tile.hh"

namespace slate {

namespace internal {

template<typename scalar_t>
void gerbt(Tile<scalar_t> A11,
           Tile<scalar_t> A12,
           Tile<scalar_t> A21,
           Tile<scalar_t> A22,
           Tile<scalar_t> U1,
           Tile<scalar_t> U2,
           Tile<scalar_t> V1,
           Tile<scalar_t> V2)
{
    slate_assert(A11.mb() >= A22.mb());
    slate_assert(A11.nb() >= A22.nb());
    const int64_t mb = A22.mb();
    const int64_t nb = A22.nb();
    const int64_t mb_full = A11.mb();
    const int64_t nb_full = A11.nb();

    for (int64_t j = 0; j < nb; ++j) {
        const scalar_t v1 = V1(j, 0);
        const scalar_t v2 = V2(j, 0);

        for (int64_t i = 0; i < mb; ++i) {
            const scalar_t u1 = U1(i, 0);
            const scalar_t u2 = U2(i, 0);

            const scalar_t a11 = A11(i, j);
            const scalar_t a12 = A12(i, j);
            const scalar_t a21 = A21(i, j);
            const scalar_t a22 = A22(i, j);

            const scalar_t sum1 = a11 + a12;
            const scalar_t sum2 = a21 + a22;
            const scalar_t dif1 = a11 - a12;
            const scalar_t dif2 = a21 - a22;

            A11.at(i, j) = u1*v1*(sum1 + sum2);
            A12.at(i, j) = u1*v2*(dif1 + dif2);
            A21.at(i, j) = u2*v1*(sum1 - sum2);
            A22.at(i, j) = u2*v2*(dif1 - dif2);
        }
    }

    for (int64_t j = 0; j < nb; ++j) {
        const scalar_t v1 = V1(j, 0);
        const scalar_t v2 = V2(j, 0);

        for (int64_t i = mb; i < mb_full; ++i) {
            const scalar_t a11 = A11(i, j);
            const scalar_t a12 = A12(i, j);

            A11.at(i, j) = v1*(a11 + a12);
            A12.at(i, j) = v2*(a11 - a12);
        }
    }

    // Note loop order is switched
    for (int64_t i = 0; i < mb; ++i) {
        const scalar_t u1 = U1(i, 0);
        const scalar_t u2 = U2(i, 0);

        for (int64_t j = nb; j < nb_full; ++j) {
            const scalar_t a11 = A11(i, j);
            const scalar_t a12 = A21(i, j);

            A11.at(i, j) = u1*(a11 + a12);
            A21.at(i, j) = u2*(a11 - a12);
        }
    }
}

template<typename scalar_t>
void gerbt_left_notrans(Tile<scalar_t> B1,
                        Tile<scalar_t> B2,
                        Tile<scalar_t> U1,
                        Tile<scalar_t> U2)
{
    const int64_t mb = std::min(B1.mb(), B2.mb());
    const int64_t nb = std::min(B1.nb(), B2.nb());

    for (int64_t i = 0; i < mb; ++i) {
        const scalar_t u1 = U1(i, 0);
        const scalar_t u2 = U2(i, 0);
        for (int64_t j = 0; j < nb; ++j) {

            const scalar_t b1 = B1(i, j);
            const scalar_t b2 = B2(i, j);

            B1.at(i, j) = u1*b1 + u2*b2;
            B2.at(i, j) = u1*b1 - u2*b2;
        }
    }
}

template<typename scalar_t>
void gerbt_right_notrans(Tile<scalar_t> B1,
                         Tile<scalar_t> B2,
                         Tile<scalar_t> U1,
                         Tile<scalar_t> U2)
{
    const int64_t mb = std::min(B1.mb(), B2.mb());
    const int64_t nb = std::min(B1.nb(), B2.nb());

    for (int64_t j = 0; j < nb; ++j) {
        const scalar_t u1 = U1(j, 0);
        const scalar_t u2 = U2(j, 0);
        for (int64_t i = 0; i < mb; ++i) {

            const scalar_t b1 = B1(i, j);
            const scalar_t b2 = B2(i, j);

            B1.at(i, j) = u1*(b1 + b2);
            B2.at(i, j) = u2*(b1 - b2);
        }
    }
}

template<typename scalar_t>
void gerbt_left_trans(Tile<scalar_t> B1,
                      Tile<scalar_t> B2,
                      Tile<scalar_t> U1,
                      Tile<scalar_t> U2)
{
    const int64_t mb = std::min(B1.mb(), B2.mb());
    const int64_t nb = std::min(B1.nb(), B2.nb());

    for (int64_t i = 0; i < mb; ++i) {
        const scalar_t u1 = U1(i, 0);
        const scalar_t u2 = U2(i, 0);
        for (int64_t j = 0; j < nb; ++j) {

            const scalar_t b1 = B1(i, j);
            const scalar_t b2 = B2(i, j);

            B1.at(i, j) = u1*(b1 + b2);
            B2.at(i, j) = u2*(b1 - b2);
        }
    }
}

template<typename scalar_t>
void gerbt_right_trans(Tile<scalar_t> B1,
                        Tile<scalar_t> B2,
                        Tile<scalar_t> U1,
                        Tile<scalar_t> U2)
{
    const int64_t mb = std::min(B1.mb(), B2.mb());
    const int64_t nb = std::min(B1.nb(), B2.nb());

    for (int64_t j = 0; j < nb; ++j) {
        const scalar_t u1 = U1(j, 0);
        const scalar_t u2 = U2(j, 0);
        for (int64_t i = 0; i < mb; ++i) {

            const scalar_t b1 = B1(i, j);
            const scalar_t b2 = B2(i, j);

            B1.at(i, j) = u1*b1 + u2*b2;
            B2.at(i, j) = u1*b1 - u2*b2;
        }
    }
}

} // namespace internal

} // namespace slate

