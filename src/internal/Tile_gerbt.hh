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
    const scalar_t inv_sqrt_2 = 1.0 / std::sqrt(2.0);
    const scalar_t inv_2 = 1.0 / 2.0;

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

            A11.at(i, j) = inv_2*u1*(sum1 + sum2)*v1;
            A12.at(i, j) = inv_2*u1*(dif1 + dif2)*v2;
            A21.at(i, j) = inv_2*u2*(sum1 - sum2)*v1;
            A22.at(i, j) = inv_2*u2*(dif1 - dif2)*v2;
        }
    }

    for (int64_t j = 0; j < nb; ++j) {
        const scalar_t v1 = V1(j, 0);
        const scalar_t v2 = V2(j, 0);

        for (int64_t i = mb; i < mb_full; ++i) {
            const scalar_t u1 = U1(j, 0);

            const scalar_t a11 = A11(i, j);
            const scalar_t a12 = A12(i, j);

            A11.at(i, j) = inv_sqrt_2*u1*(a11 + a12)*v1;
            A12.at(i, j) = inv_sqrt_2*u1*(a11 - a12)*v2;
        }
    }

    for (int64_t j = nb; j < nb_full; ++j) {
        const scalar_t v1 = V1(j, 0);

        for (int64_t i = 0; i < mb; ++i) {
            const scalar_t u1 = U1(i, 0);
            const scalar_t u2 = U2(i, 0);

            const scalar_t a11 = A11(i, j);
            const scalar_t a12 = A21(i, j);

            A11.at(i, j) = inv_sqrt_2*u1*(a11 + a12)*v1;
            A21.at(i, j) = inv_sqrt_2*u2*(a11 - a12)*v1;
        }
    }

    for (int64_t j = nb; j < nb_full; ++j) {
        const scalar_t v1 = V1(j, 0);

        for (int64_t i = mb; i < mb_full; ++i) {
            const scalar_t u1 = U1(i, 0);

            const scalar_t a11 = A11(i, j);

            A11.at(i, j) = u1*a11*v1;
        }
    }
}

template<typename scalar_t>
void gerbt_left_notrans(Tile<scalar_t> B1,
                        Tile<scalar_t> B2,
                        Tile<scalar_t> U1,
                        Tile<scalar_t> U2)
{
    const scalar_t inv_sqrt_2 = 1.0 / std::sqrt(2);

    slate_assert(B1.mb() >= B2.mb());
    const int64_t mb = B2.mb();
    const int64_t mb_full = B1.mb();
    const int64_t nb = std::min(B1.nb(), B2.nb());

    for (int64_t i = 0; i < mb; ++i) {
        const scalar_t u1 = U1(i, 0);
        const scalar_t u2 = U2(i, 0);
        for (int64_t j = 0; j < nb; ++j) {

            const scalar_t b1 = B1(i, j);
            const scalar_t b2 = B2(i, j);

            B1.at(i, j) = inv_sqrt_2*(u1*b1 + u2*b2);
            B2.at(i, j) = inv_sqrt_2*(u1*b1 - u2*b2);
        }
    }
    for (int64_t i = mb; i < mb_full; ++i) {
        const scalar_t u1 = U1(i, 0);
        for (int64_t j = 0; j < nb; ++j) {

            const scalar_t b1 = B1(i, j);

            B1.at(i, j) = u1*b1;
        }
    }
}

template<typename scalar_t>
void gerbt_right_notrans(Tile<scalar_t> B1,
                         Tile<scalar_t> B2,
                         Tile<scalar_t> U1,
                         Tile<scalar_t> U2)
{
    const scalar_t inv_sqrt_2 = 1.0 / std::sqrt(2);

    slate_assert(B1.nb() >= B2.nb());
    const int64_t nb = B2.nb();
    const int64_t nb_full = B1.nb();
    const int64_t mb = std::min(B1.mb(), B2.mb());

    for (int64_t j = 0; j < nb; ++j) {
        const scalar_t u1 = U1(j, 0);
        const scalar_t u2 = U2(j, 0);
        for (int64_t i = 0; i < mb; ++i) {

            const scalar_t b1 = B1(i, j);
            const scalar_t b2 = B2(i, j);

            B1.at(i, j) = inv_sqrt_2*u1*(b1 + b2);
            B2.at(i, j) = inv_sqrt_2*u2*(b1 - b2);
        }
    }
    for (int64_t j = nb; j < nb_full; ++j) {
        const scalar_t u1 = U1(j, 0);
        for (int64_t i = 0; i < mb; ++i) {

            const scalar_t b1 = B1(i, j);

            B1.at(i, j) = u1*b1;
        }
    }
}

template<typename scalar_t>
void gerbt_left_trans(Tile<scalar_t> B1,
                      Tile<scalar_t> B2,
                      Tile<scalar_t> U1,
                      Tile<scalar_t> U2)
{
    const scalar_t inv_sqrt_2 = 1.0 / std::sqrt(2);

    slate_assert(B1.mb() >= B2.mb());
    const int64_t mb = B2.mb();
    const int64_t mb_full = B1.mb();
    const int64_t nb = std::min(B1.nb(), B2.nb());

    for (int64_t i = 0; i < mb; ++i) {
        const scalar_t u1 = U1(i, 0);
        const scalar_t u2 = U2(i, 0);
        for (int64_t j = 0; j < nb; ++j) {

            const scalar_t b1 = B1(i, j);
            const scalar_t b2 = B2(i, j);

            B1.at(i, j) = inv_sqrt_2*u1*(b1 + b2);
            B2.at(i, j) = inv_sqrt_2*u2*(b1 - b2);
        }
    }
    for (int64_t i = mb; i < mb_full; ++i) {
        const scalar_t u1 = U1(i, 0);
        for (int64_t j = 0; j < nb; ++j) {

            const scalar_t b1 = B1(i, j);

            B1.at(i, j) = u1*b1;
        }
    }
}

template<typename scalar_t>
void gerbt_right_trans(Tile<scalar_t> B1,
                        Tile<scalar_t> B2,
                        Tile<scalar_t> U1,
                        Tile<scalar_t> U2)
{
    const scalar_t inv_sqrt_2 = 1.0 / std::sqrt(2);

    slate_assert(B1.nb() >= B2.nb());
    const int64_t nb = B2.nb();
    const int64_t nb_full = B1.nb();
    const int64_t mb = std::min(B1.mb(), B2.mb());

    for (int64_t j = 0; j < nb; ++j) {
        const scalar_t u1 = U1(j, 0);
        const scalar_t u2 = U2(j, 0);
        for (int64_t i = 0; i < mb; ++i) {

            const scalar_t b1 = B1(i, j);
            const scalar_t b2 = B2(i, j);

            B1.at(i, j) = inv_sqrt_2*(u1*b1 + u2*b2);
            B2.at(i, j) = inv_sqrt_2*(u1*b1 - u2*b2);
        }
    }
    for (int64_t j = nb; j < nb_full; ++j) {
        const scalar_t u1 = U1(j, 0);
        for (int64_t i = 0; i < mb; ++i) {

            const scalar_t b1 = B1(i, j);

            B1.at(i, j) = u1*b1;
        }
    }
}

} // namespace internal

} // namespace slate

