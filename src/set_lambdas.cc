// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Set matrix entries.
/// Transposition is automatically handled. 
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] value
///     A function that takes global row and column indices i and j,
///     and returns the scalar value for entry Aij.
///
/// @param[in,out] A
///     The m-by-n matrix A.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Currently no options.
///     It always uses Target = HostTask, since lambda is a CPU function.
///
/// @ingroup set
///
template <typename scalar_t>
void set(
    std::function< scalar_t (int64_t i, int64_t j) > const& value,
    Matrix<scalar_t>& A,
    Options const& opts )
{
    int64_t mt = A.mt();
    int64_t nt = A.nt();

    #pragma omp parallel
    #pragma omp master
    {
        int64_t i_global = 0;
        for (int64_t i = 0; i < mt; ++i) {
            const int64_t mb = A.tileMb( i );
            int64_t j_global = 0;
            for (int64_t j = 0; j < nt; ++j) {
                const int64_t nb = A.tileNb( j );
                if (A.tileIsLocal( i, j )) {
                    #pragma omp task slate_omp_default_none shared( A ) \
                        firstprivate( i, j, mb, nb, i_global, j_global, value )
                    {
                        A.tileGetForWriting( i, j, LayoutConvert::ColMajor );
                        auto Aij = A( i, j );
                        for (int64_t jj = 0; jj < nb; ++jj) {
                            for (int64_t ii = 0; ii < mb; ++ii) {
                                Aij.at( ii, jj )
                                    = value( i_global + ii, j_global + jj );
                            }
                        }
                    }
                }
                j_global += nb;
            }
            i_global += mb;
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void set(
    std::function< float (int64_t i, int64_t j) > const& value,
    Matrix<float>& A,
    Options const& opts);

template
void set(
    std::function< double (int64_t i, int64_t j) > const& value,
    Matrix<double>& A,
    Options const& opts);

template
void set(
    std::function< std::complex<float> (int64_t i, int64_t j) > const& value,
    Matrix< std::complex<float> >& A,
    Options const& opts);

template
void set(
    std::function< std::complex<double> (int64_t i, int64_t j) > const& value,
    Matrix< std::complex<double> >& A,
    Options const& opts);

//==============================================================================
// For BaseTrapezoidMatrix.
//==============================================================================

//------------------------------------------------------------------------------
/// Set matrix entries.
/// Transposition is automatically handled.
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] value
///     A function that takes global row and column indices i and j,
///     and returns the scalar value for entry Aij.
///
/// @param[in,out] A
///     The m-by-n matrix A.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Currently no options.
///     It always uses Target = HostTask, since lambda is a CPU function.
///
/// @ingroup set
///
template <typename scalar_t>
void set(
    std::function< scalar_t (int64_t i, int64_t j) > const& value,
    BaseTrapezoidMatrix<scalar_t>& A,
    Options const& opts )
{
    // TODO
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void set(
    std::function< float (int64_t i, int64_t j) > const& value,
    BaseTrapezoidMatrix<float>& A,
    Options const& opts);

template
void set(
    std::function< double (int64_t i, int64_t j) > const& value,
    BaseTrapezoidMatrix<double>& A,
    Options const& opts);

template
void set(
    std::function< std::complex<float> (int64_t i, int64_t j) > const& value,
    BaseTrapezoidMatrix< std::complex<float> >& A,
    Options const& opts);

template
void set(
    std::function< std::complex<double> (int64_t i, int64_t j) > const& value,
    BaseTrapezoidMatrix< std::complex<double> >& A,
    Options const& opts);

} // namespace slate
