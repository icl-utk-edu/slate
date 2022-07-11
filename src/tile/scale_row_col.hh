// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TILE_SCALE_HH
#define SLATE_TILE_SCALE_HH

#include "slate/Tile.hh"

namespace slate {

namespace tile {

//------------------------------------------------------------------------------
/// Apply row or column scaling, or both, to a Tile.
/// @ingroup scale_tile
///
template <typename scalar_t, typename scalar_t2>
void scale_row_col(
    Equed equed,
    scalar_t2 const* R,
    scalar_t2 const* C,
    Tile<scalar_t>&& A)
{
    assert( A.layout() == Layout::ColMajor );  // todo: row major
    assert( A.uplo() == Uplo::General );  // todo: upper & lower
    assert( A.op() == Op::NoTrans );  // todo: transposition

    int64_t mb = A.mb();
    int64_t nb = A.nb();
    int64_t lda = A.stride();
    scalar_t* data = A.data();

    if (equed == Equed::Both) {
        for (int64_t j = 0; j < nb; ++j) {
            scalar_t2 cj = C[ j ];
            for (int64_t i = 0; i < mb; ++i) {
                scalar_t2 ri = R[ i ];
                data[ i + j*lda ] *= ri * cj;
            }
        }
    }
    else if (equed == Equed::Row) {
        for (int64_t j = 0; j < nb; ++j) {
            for (int64_t i = 0; i < mb; ++i) {
                scalar_t2 ri = R[ i ];
                data[ i + j*lda ] *= ri;
            }
        }
    }
    else if (equed == Equed::Col) {
        for (int64_t j = 0; j < nb; ++j) {
            scalar_t2 cj = C[ j ];
            for (int64_t i = 0; i < mb; ++i) {
                data[ i + j*lda ] *= cj;
            }
        }
    }
}

} // namespace tile

} // namespace slate

#endif // SLATE_TILE_BLAS_HH
