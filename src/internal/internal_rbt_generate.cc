// Copyright (c) 2020-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "internal/internal.hh"
#include "slate/Matrix.hh"

#include <cmath>
#include <stdlib.h>
#include <vector>


namespace slate {

namespace internal {


//------------------------------------------------------------------------------
/// Allocates and fills a random butterfly transform in packed storage.
/// The depth is computed based on the number of column in U.
///
/// @ingroup gesv_internal
///
template<typename scalar_t>
void rbt_fill(Matrix<scalar_t>& U, const int64_t seed)
{
    using real_t = blas::real_type<scalar_t>;

    const int64_t d = U.n();
    slate_assert(d == U.tileNb(0));
    slate_assert(U.nt() == 1);
    const real_t scale_20 = 20.0;

    U.insertLocalTiles( Target::Host );
    for (int64_t i = 0; i < U.mt(); i++) {
        if (U.tileIsLocal(i, 0)) {
            U.tileGetForWriting( i, 0, LayoutConvert::None );
            Tile<scalar_t> U_i = U(i, 0);

            const int64_t mb = U_i.mb();

            #pragma omp task
            {
                int64_t iseed[4] = {(seed + i) % 4096, 578, 361, 115};
                lapack::larnv( 2, iseed, mb*d, U_i.data() );

                for (int64_t k = 0; k < d; k++) {
                    for (int64_t jj = 0; jj < mb; jj++) {
                        real_t U_jk = blas::real( U_i.at(jj, k) );
                        U_i.at(jj, k) = std::exp( U_jk/scale_20 );
                    }
                }
            }
        }
    }

}

//------------------------------------------------------------------------------
/// Constructs two random bufferfly matrice in packed storage to transform the
/// given matrix.
///
/// @param[in] A
///     The matrix to be transformed
///
/// @param[in] d
///     The depth of the transform
///
/// @param[in] seed
///     A seed for controlling the random number generation
///
/// @return a tuple containing the left and right transforms
///
/// @ingroup gesv_internal
///
template<typename scalar_t>
std::pair<Matrix<scalar_t>, Matrix<scalar_t>> rbt_generate(
    const Matrix<scalar_t>& A,
    const int64_t d,
    const int64_t seed)
{

    typedef typename Matrix<scalar_t>::ij_tuple ij_tuple;

    const int64_t m = A.m();
    const int64_t n = A.n();
    const int64_t mt = A.mt();
    const int64_t nt = A.nt();
    const MPI_Comm mpi_comm = A.mpiComm();

    std::vector<int64_t> tileMb(mt);
    for (int64_t i = 0; i < mt; ++i) {
        tileMb[i] = A.tileMb(i);
    }
    std::function<int64_t(int64_t)> tileMb_lambda = [tileMb] (int64_t i) {
        return tileMb[i];
    };

    std::vector<int64_t> tileNb(nt);
    for (int64_t i = 0; i < nt; ++i) {
        tileNb[i] = A.tileNb(i);
    }
    std::function<int64_t(int64_t)> tileNb_lambda = [tileNb] (int64_t i) {
        return tileNb[i];
    };

    std::function<int64_t(int64_t)> d_lambda = [d] (int64_t) {
        return d;
    };

    std::vector<int> tileRank(nt);
    for (int64_t i = 0; i < mt; ++i) {
        tileRank[i] = A.tileRank(i, 0);
    }
    std::function<int(ij_tuple)> tileRank_lambda = [tileRank] (ij_tuple ij) {
        return tileRank[std::get<0>(ij)];
    };

    std::function<int(ij_tuple)> tileDevice_lambda = [] (ij_tuple) {
        return HostNum;
    };

    Matrix<scalar_t> U( m, d, tileMb_lambda, d_lambda, tileRank_lambda,
                        tileDevice_lambda, mpi_comm );
    Matrix<scalar_t> V( n, d, tileNb_lambda, d_lambda, tileRank_lambda,
                        tileDevice_lambda, mpi_comm );

    if (d > 0) {
        #pragma omp parallel
        #pragma omp master
        {
            rbt_fill( U, seed );
            rbt_fill( V, seed+mt );

            #pragma omp taskwait
        }
    }

    return std::make_pair( transpose(U), V );
}

template
std::pair<Matrix<float>, Matrix<float>> rbt_generate(const Matrix<float> &,
                                                     const int64_t,
                                                     const int64_t);


template
std::pair<Matrix<double>, Matrix<double>> rbt_generate(const Matrix<double> &,
                                                       const int64_t,
                                                       const int64_t);

template
std::pair<Matrix<std::complex<float>>, Matrix<std::complex<float>>> rbt_generate(
        const Matrix<std::complex<float>> &,
        const int64_t,
        const int64_t);

template
std::pair<Matrix<std::complex<double>>, Matrix<std::complex<double>>> rbt_generate(
        const Matrix<std::complex<double>> &,
        const int64_t,
        const int64_t);

} // namespace internal
} // namespace slate
