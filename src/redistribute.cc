// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/HermitianBandMatrix.hh"
#include "slate/types.hh"
#include "slate/Tile_blas.hh"
#include "slate/Matrix.hh"
#include "internal/internal.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Redistribute a matrix A from one distribution into matrix B with another
/// distribution.
/// @ingroup copy_internal
///
template <typename scalar_t>
void redistribute(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    Options const& opts )
{
    trace::Block trace_block("slate::redistribute");

    int64_t mt = B.mt();
    int64_t nt = B.nt();

    bool is_conj = false;
    if (A.op() != B.op()) {
        if (A.op() == Op::ConjTrans || B.op() == Op::ConjTrans)
            is_conj = true;

        auto BT = A.emptyLike();
        for (int64_t j = 0; j < nt; ++j) {
            for (int64_t i = 0; i < mt; ++i) {
                if (B.tileIsLocal(i, j)) {
                    B.tileGetForWriting( i, j, LayoutConvert::None );
                    if (! A.tileIsLocal(i, j)) {
                        auto Bij = B(i, j);
                        if (Bij.mb() == Bij.nb()) {
                            Bij.recv(A.tileRank(i, j), A.mpiComm(),  A.layout());
                            B.tileGetForWriting( i, j, LayoutConvert::None );
                            if (is_conj)
                                tile::deepConjTranspose( std::move(Bij) );
                            else
                                tile::deepTranspose( std::move(Bij) );
                        }
                        else {
                            BT.tileInsert(i, j);
                            auto BTij = BT(i, j);
                            BT.tileGetForWriting( i, j, LayoutConvert::None );
                            BTij.recv(A.tileRank(i, j), A.mpiComm(),  A.layout());
                            if (is_conj) {
                                auto AijT = conj_transpose(BTij);
                                tile::deepConjTranspose( std::move(AijT), std::move(Bij) );
                            }
                            else {
                                auto AijT = transpose(BTij);
                                tile::deepTranspose( std::move(AijT), std::move(Bij) );
                            }
                        }
                    }
                    else {
                        A.tileGetForReading(i, j, LayoutConvert::None);
                        // copy local tiles if needed.
                        auto Aij = A(i, j);
                        auto Bij = B(i, j);
                        if (Bij.mb() == Bij.nb()) {
                            if (is_conj)
                                tile::deepConjTranspose( std::move(Aij), std::move(Bij) );
                            else
                                tile::deepTranspose( std::move(Aij), std::move(Bij) );
                        }
                        else {
                            if (is_conj) {
                                auto AijT = conj_transpose(Aij);
                                tile::deepConjTranspose( std::move(AijT), std::move(Bij) );
                            }
                            else {
                                auto AijT = transpose(Aij);
                                tile::deepTranspose( std::move(AijT), std::move(Bij) );
                            }
                        }
                    }
                }
                else if (A.tileIsLocal(i, j)) {
                    A.tileGetForReading(i, j, LayoutConvert::None);
                    auto Aij = A(i, j);
                    Aij.send(B.tileRank(i, j), B.mpiComm());
                }
            }
        }
    }
    else {
        for (int64_t j = 0; j < nt; ++j) {
            for (int64_t i = 0; i < mt; ++i) {
                if (B.tileIsLocal(i, j)) {
                    B.tileGetForWriting( i, j, LayoutConvert::None );
                    if (! A.tileIsLocal(i, j)) {
                        auto Bij = B(i, j);
                        Bij.recv(A.tileRank(i, j), A.mpiComm(),  A.layout());
                    }
                    else {
                        A.tileGetForReading(i, j, LayoutConvert::None);
                        // copy local tiles if needed.
                        auto Aij = A(i, j);
                        auto Bij = B(i, j);
                        if (Aij.data() != Bij.data() ) {
                            tile::gecopy( Aij, Bij );
                            // deep conj tile after recieve if its square
                            // if rectangular, recv in a tmp tile then transpose it
                        }
                    }
                }
                else if (A.tileIsLocal(i, j)) {
                    A.tileGetForReading(i, j, LayoutConvert::None);
                    auto Aij = A(i, j);
                    Aij.send(B.tileRank(i, j), B.mpiComm());
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
// ----------------------------------------
template
void redistribute<float>(
    Matrix<float>& A,
    Matrix<float>& B,
    Options const& opts);

template
void redistribute<double>(
    Matrix<double>& A,
    Matrix<double>& B,
    Options const& opts);

template
void redistribute< std::complex<float> >(
    Matrix< std::complex<float> >& A,
    Matrix< std::complex<float> >& B,
    Options const& opts);

template
void redistribute< std::complex<double> >(
    Matrix< std::complex<double> >& A,
    Matrix< std::complex<double> >& B,
    Options const& opts);

} // namespace slate
