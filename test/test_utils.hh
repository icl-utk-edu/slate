// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TEST_UTILS_HH
#define SLATE_TEST_UTILS_HH

#include "slate/slate.hh"
#include "test.hh"

///-----------------------------------------------------------------------------
/// Checks for common invalid parameter combinations
///
/// @return true if the configuration should be skipped
///
inline bool is_invalid_parameters(Params& params, bool keep_nonuniform_ref = false)
{
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    slate::Dist dev_dist = params.dev_dist();
    bool nonuniform_nb = params.nonuniform_nb() == 'y';

    if (target != slate::Target::Devices && dev_dist == slate::Dist::Col) {
        params.msg() = "skipping: dev_dist = Col applies only to target devices";
        return true;
    }

    if (dev_dist == slate::Dist::Col && origin == slate::Origin::ScaLAPACK) {
        params.msg() = "skipping: dev_dist = Col tile not supported with origin=ScaLAPACK";
        return true;
    }

    if (nonuniform_nb && origin == slate::Origin::ScaLAPACK) {
        params.msg() = "skipping: nonuniform tile not supported with origin=ScaLAPACK";
        return true;
    }

    #ifdef SLATE_HAVE_SCALAPACK
        if (!keep_nonuniform_ref && nonuniform_nb && params.ref() != 'n') {
            params.msg() = "skipping reference: nonuniform tile not supported with ScaLAPACK";
            if (params.ref() == 'o') {
                // If ref=='o', the user doesn't want to run SLATE version
                return true;
            }
            else {
                params.ref() = 'n';
            }
        }
    #else
        // Can only run ref when we have ScaLAPACK
        if (params.ref()) {
            params.msg() = "skipping reference: ScaLAPACK not available";
            params.ref() = false;
        }
    #endif


    return false;
}

///-----------------------------------------------------------------------------
/// Applies the operator thunk to each element of A and B to update B.
/// The matrices must have the same size, but can have different tile sizes and
/// distributions. However, the elements of a tile of B must all belong to the
/// same tile of A. For example, this is satisfied in the testers if B
/// has tiles of size nb and A has tiles of size nb or 2*nb.
///
template <typename matrix_type>
void matrix_iterator(
    matrix_type& A, matrix_type& B,
    std::function< void( typename matrix_type::value_type const&,
                         typename matrix_type::value_type& ) > thunk )
{
    using scalar_t = typename matrix_type::value_type;
    assert( A.m() == B.m() );
    assert( A.n() == B.n() );

    const auto ColMajor = slate::LayoutConvert::ColMajor;

    int64_t B_mt = B.mt();
    int64_t B_nt = B.nt();

    constexpr bool is_general = std::is_same_v<matrix_type, slate::Matrix<scalar_t>>;
    assert( is_general == (A.uplo() == slate::Uplo::General) );
    bool is_upper = (A.uplo() == slate::Uplo::Upper);

    int64_t A_j = 0, A_jj = 0;
    for (int64_t B_j = 0; B_j < B_nt; ++B_j) {

        int64_t A_i = 0, A_ii = 0;
        for (int64_t B_i = 0; B_i < B_mt; ++B_i) {

            if (is_general || (is_upper ? (B_i <= B_j) : (B_i >= B_j))) {
                #pragma omp task shared(A, B) \
                                 firstprivate( B_i, B_j, A_i, A_j, A_ii, A_jj )
                {
                    int tag = A_i + A_j * A.mt();
                    if (B.tileIsLocal( B_i, B_j )) {
                        A.tileRecv( A_i, A_j, A.tileRank( A_i, A_j ),
                                    slate::Layout::ColMajor, tag );

                        A.tileGetForReading( A_i, A_j, ColMajor );
                        B.tileGetForWriting( B_i, B_j, ColMajor );
                        auto TA = A( A_i, A_j );
                        auto TB = B( B_i, B_j );
                        int64_t mb = TB.mb();
                        int64_t nb = TB.nb();
                        assert( A_ii + mb <= TA.mb() );
                        assert( A_jj + nb <= TA.nb() );
                        int64_t lda = TA.stride();
                        int64_t ldb = TB.stride();
                        scalar_t const* TA_data = TA.data();
                        scalar_t*       TB_data = TB.data();
                        for (int64_t jj = 0; jj < nb; ++jj) {
                            int64_t ii_start = 0, ii_end = mb;
                            if (!is_general && B_i == B_j) { // diagonal tile
                                if (is_upper)
                                    ii_end = std::min(jj+1, mb);
                                else
                                    ii_start = jj;
                            }
                            for (int64_t ii = ii_start; ii < ii_end; ++ii) {
                                thunk( TA_data[ (A_ii+ii) + (A_jj+jj)*lda ],
                                       TB_data[ ii + jj*ldb ] );
                            }
                        }
                    }
                    else if (A.tileIsLocal( A_i, A_j )) {
                        A.tileSend( A_i, A_j, B.tileRank( B_i, B_j ), tag );
                    }
                }

            }
            A_ii += B.tileMb( B_i );
            assert( A_ii <= A.tileMb( A_i ) );
            if (A_ii == A.tileMb( A_i )) {
                ++A_i;
                A_ii = 0;
            }
        }
        A_jj += B.tileNb( B_j );
        assert( A_jj <= A.tileNb( A_j ) );
        if (A_jj == A.tileNb( A_j )) {
            ++A_j;
            A_jj = 0;
        }
    }

    A.releaseRemoteWorkspace();
}

///-----------------------------------------------------------------------------
/// subtract_matrices takes input matrices A and B, and performs B = B - A.
/// The matrices must have the same size, but can have different tile sizes and
/// distributions. However, the elements of a tile of B must all belong to the
/// same tile of A.
///
template <typename matrix_type>
void subtract_matrices( matrix_type& A, matrix_type& B )
{
    using scalar_t = typename matrix_type::value_type;

    matrix_iterator( A, B, [](const scalar_t& a, scalar_t& b) { b -= a; } );
}

///-----------------------------------------------------------------------------
/// copy_matrix copies A to B
/// The matrices must have the same size, but can have different tile sizes and
/// distributions. However, the elements of a tile of B must all belong to the
/// same tile of A.
///
template <typename matrix_type>
void copy_matrix( matrix_type& A, matrix_type& B )
{
    using scalar_t = typename matrix_type::value_type;

    matrix_iterator( A, B, [](const scalar_t& a, scalar_t& b) { b = a; } );
}


#endif // SLATE_TEST_UTILS_HH
