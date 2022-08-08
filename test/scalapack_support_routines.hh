// Copyright (c) 2009-2022, University of Tennessee. All rights reserved.
// Copyright (c) 2010,      University of Denver, Colorado.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_SCALAPACK_SUPPORT_HH
#define SLATE_SCALAPACK_SUPPORT_HH

#include <complex>

#include "scalapack_wrappers.hh"

//------------------------------------------------------------------------------
// Matrix generation
#define Rnd64_A  6364136223846793005ULL
#define Rnd64_C  1ULL
#define RndF_Mul 5.4210108624275222e-20f
#define RndD_Mul 5.4210108624275222e-20

typedef unsigned long long ull;

static inline ull Rnd64_jump(ull n, ull seed)
{
    ull a_k, c_k, ran;
    int64_t i;

    a_k = Rnd64_A;
    c_k = Rnd64_C;
    ran = seed;
    for (i = 0; n; n >>= 1, ++i) {
        if (n & 1)
            ran = a_k*ran + c_k;
        c_k *= (a_k + 1);
        a_k *= a_k;
    }
    return ran;
}

template<typename scalar_t>
static inline void CORE_plrnt(int64_t m, int64_t n, scalar_t* A, int64_t lda,
                              int64_t bigM, int64_t m0, int64_t n0, ull seed)
{
    scalar_t* tmp = A;
    int64_t i, j;
    ull ran, jump;

    jump = (ull)m0 + (ull)n0*(ull)bigM;
    for (j = 0; j < n; ++j) {
        ran = Rnd64_jump(jump, (ull)seed);
        for (i = 0; i < m; ++i) {
            *tmp = 0.5f - ran*RndF_Mul;
            ran  = Rnd64_A*ran + Rnd64_C;
            tmp++;
        }
        tmp  += lda - i;
        jump += bigM;
    }
}

template<typename scalar_t>
static inline void CORE_plghe(scalar_t bump, int64_t m, int64_t n, scalar_t* A, int64_t lda,
                              int64_t gM, int64_t m0, int64_t n0, ull seed)
{
    scalar_t* tmp = A;
    int64_t i, j;
    ull ran, jump;

    jump = (ull)m0 + (ull)n0*(ull)gM;
    /* Tile diagonal */
    if (m0 == n0) {
        for (j = 0; j < n; j++) {
            ran = Rnd64_jump(jump, seed);

            for (i = j; i < m; i++) {
                *tmp = 0.5f - ran*RndF_Mul;
                ran  = Rnd64_A*ran + Rnd64_C;
                tmp++;
            }
            tmp  += (lda - i + j + 1);
            jump += gM + 1;
        }
        for (j = 0; j < n; j++) {
            A[j + j*lda] += bump;

            for (i = 0; i < j; i++)
                A[lda*j + i] = A[lda*i + j];
        }
    }

    /* Lower part */
    else if (m0 > n0) {
        for (j = 0; j < n; j++) {
            ran = Rnd64_jump(jump, seed);

            for (i = 0; i < m; i++) {
                *tmp = 0.5f - ran*RndF_Mul;
                ran  = Rnd64_A*ran + Rnd64_C;
                tmp++;
            }
            tmp  += (lda - i);
            jump += gM;
        }
    }

    /* Upper part */
    else if (m0 < n0) {
        /* Overwrite jump */
        jump = (ull)n0 + (ull)m0*(ull)gM;

        for (i = 0; i < m; i++) {
            ran = Rnd64_jump(jump, seed);

            for (j = 0; j < n; j++) {
                A[j*lda + i] = 0.5f - ran*RndF_Mul;
                ran = Rnd64_A*ran + Rnd64_C;
            }
            jump += gM;
        }
    }
}

template<typename scalar_t>
static void scalapack_pplrnt(scalar_t* A,
                             int64_t m, int64_t n,
                             int64_t mb, int64_t nb,
                             int myrow, int mycol,
                             int nprow, int npcol,
                             int64_t lldA,
                             int64_t seed)
{
    int i, j;
    int idum1, idum2, iloc, jloc, i0 = 0;
    int tempm, tempn;
    scalar_t* Ab;
    int mb_ = (int)mb;
    int nb_ = (int)nb;

    // #pragma omp parallel for
    for (i = 1; i <= m; i += mb) {
        for (j = 1; j <= n; j += nb) {
            if ((myrow == scalapack_indxg2p(&i, &mb_, &idum1, &i0, &nprow)) &&
                (mycol == scalapack_indxg2p(&j, &nb_, &idum1, &i0, &npcol))) {
                iloc = scalapack_indxg2l(&i, &mb_, &idum1, &idum2, &nprow);
                jloc = scalapack_indxg2l(&j, &nb_, &idum1, &idum2, &npcol);

                Ab =  &A[(jloc - 1)*lldA + (iloc - 1) ];
                tempm = (m - i + 1) > mb ? mb : (m - i + 1);
                tempn = (n - j + 1) > nb ? nb : (n - j + 1);
                CORE_plrnt(tempm, tempn, Ab, lldA,
                           m, (i - 1), (j - 1), seed);
            }
        }
    }
}


template<typename scalar_t>
static void scalapack_pplghe(scalar_t* A,
                             int64_t m, int64_t n,
                             int64_t mb, int64_t nb,
                             int myrow, int mycol,
                             int nprow, int npcol,
                             int64_t lldA,
                             int64_t seed)
{
    int i, j;
    int idum1, idum2, iloc, jloc, i0 = 0;
    int64_t tempm, tempn;
    scalar_t* Ab;
    scalar_t bump = (scalar_t)m;
    int mb_ = (int)mb;
    int nb_ = (int)nb;

    // #pragma omp parallel for
    for (i = 1; i <= m; i += mb) {
        for (j = 1; j <= n; j += nb) {
            if ((myrow == scalapack_indxg2p(&i, &mb_, &idum1, &i0, &nprow)) &&
                (mycol == scalapack_indxg2p(&j, &nb_, &idum1, &i0, &npcol))) {
                iloc = scalapack_indxg2l(&i, &mb_, &idum1, &idum2, &nprow);
                jloc = scalapack_indxg2l(&j, &nb_, &idum1, &idum2, &npcol);

                Ab =  &A[(jloc - 1)*lldA + (iloc - 1) ];
                tempm = (m - i + 1) > mb ? mb : (m - i + 1);
                tempn = (n - j + 1) > nb ? nb : (n - j + 1);
                CORE_plghe(bump, tempm, tempn, Ab, lldA,
                           m, (i - 1), (j - 1), seed);
            }
        }
    }
}

#endif // SLATE_SCALAPACK_SUPPORT_HH
