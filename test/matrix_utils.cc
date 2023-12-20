// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "matrix_utils.hh"

using nb_func_t = std::function< int64_t(int64_t) >;
using dist_func_t = std::function< int(std::tuple<int64_t, int64_t>) >;

//------------------------------------------------------------------------------
/// Shared logic of the allocate_test_* routines
template <typename matrix_type, typename irregular_constructor_t,
          typename regular_constructor_t, typename scalapack_constructor_t>
static TestMatrix<matrix_type> allocate_test_shared(
    bool ref_matrix, bool nonuniform_ref, int64_t m, int64_t n, Params& params,
    irregular_constructor_t construct_irregular,
    regular_constructor_t construct_regular,
    scalapack_constructor_t construct_scalapack)
{
    // Load params variables
    int p = params.grid.m();
    int q = params.grid.n();
    slate::Dist dev_dist = params.dev_dist();
    int64_t nb = params.nb();
    bool nonuniform_nb = params.nonuniform_nb() == 'y';
    slate::Origin origin = params.origin();
    slate::GridOrder grid_order = params.grid_order();

    // The object to be returned
    TestMatrix<matrix_type> matrix( m, n, nb, p, q, grid_order );

    // Functions for nonuniform tile sizes or row device distribution
    nb_func_t tileNb;
    if (nonuniform_nb) {
        tileNb = [nb](int64_t j) {
            // for non-uniform tile size
            return (j % 2 != 0 ? nb*2 : nb);
        };
    }
    else {
        // NB. we let BaseMatrix truncate the final tile length
        // TrapezoidMatrix only takes 1 function for both dimensions (of different sizes)
        tileNb = [nb](int64_t j) {
            return nb;
        };
    }

    auto tileRank = slate::func::process_2d_grid( grid_order, p, q );
    int num_devices_ = blas::get_device_count();
    auto tileDevice = slate::func::device_1d_grid( slate::GridOrder( dev_dist ),
                                                   p, num_devices_ );

    // Setup matrix to test SLATE with
    if (origin != slate::Origin::ScaLAPACK) {
        // SLATE allocates CPU or GPU tiles.
        slate::Target origin_target = origin2target( origin );
        if (nonuniform_nb || dev_dist == slate::Dist::Col) {
            matrix.A = construct_irregular( tileNb, tileRank, tileDevice );
        }
        else {
            matrix.A = construct_regular( nb, grid_order, p, q );
        }
        matrix.A.insertLocalTiles( origin_target );
    }
    else {
        assert( !nonuniform_nb );
        assert( dev_dist == slate::Dist::Row );
        // Create SLATE matrix from the ScaLAPACK layouts
        matrix.A_data.resize( matrix.lld * matrix.nloc );
        matrix.A = construct_scalapack( &matrix.A_data[0], matrix.lld, nb,
                                        grid_order, p, q );
    }

    // Setup reference matrix
    if (ref_matrix) {
        if (nonuniform_nb && nonuniform_ref) {
            matrix.Aref = construct_irregular( tileNb, tileRank, tileDevice );
            matrix.Aref.insertLocalTiles( slate::Target::Host );
        }
        else {
            matrix.Aref_data.resize( matrix.lld * matrix.nloc );
            matrix.Aref = construct_scalapack( &matrix.Aref_data[0], matrix.lld, nb,
                                           grid_order, p, q );
        }
    }

    return matrix;
}

// -----------------------------------------------------------------------------
/// Allocates a Matrix<scalar_t> and optionally a reference version for testing.
///
/// @param[in] ref_matrix
///     Whether to allocate a reference matrix
///
/// @param[in] nonuniform_ref
///     If params.nonuniform_nb(), whether to also allocate the reference matrix
///     with non-uniform tiles.
///
/// @param[in] m
///     The number of rows
///
/// @param[in] n
///     The number of columns
///
/// @param[in] params
///     The test params object which contains many of the key parameters
///
template <typename scalar_t>
TestMatrix<slate::Matrix<scalar_t>> allocate_test_Matrix(
    bool ref_matrix, bool nonuniform_ref, int64_t m, int64_t n, Params& params)
{
    auto construct_irregular = [&] (nb_func_t tileNb,
                                    dist_func_t tileRank, dist_func_t tileDevice)
    {
        return slate::Matrix<scalar_t>( m, n, tileNb, tileNb,
                                        tileRank, tileDevice, MPI_COMM_WORLD );
    };
    auto construct_regular = [&] (int64_t nb, slate::GridOrder grid_order, int p, int q )
    {
        return slate::Matrix<scalar_t>( m, n, nb, nb, grid_order, p, q, MPI_COMM_WORLD );
    };
    auto construct_scalapack = [&] (scalar_t* data, int64_t lld, int64_t nb,
                                    slate::GridOrder grid_order, int p, int q )
    {
        return slate::Matrix<scalar_t>::fromScaLAPACK(
                                            m, n, data, lld, nb, nb,
                                            grid_order, p, q, MPI_COMM_WORLD );
    };

    return allocate_test_shared<slate::Matrix<scalar_t>>(
                    ref_matrix, nonuniform_ref, m, n, params,
                     construct_irregular, construct_regular, construct_scalapack );
}
// Explicit instantiations.
template
TestMatrix<slate::Matrix<float>> allocate_test_Matrix<float>(
    bool ref_matrix, bool nonuniform_ref, int64_t m, int64_t n, Params& params);

template
TestMatrix<slate::Matrix<double>> allocate_test_Matrix<double>(
    bool ref_matrix, bool nonuniform_ref, int64_t m, int64_t n, Params& params);

template
TestMatrix<slate::Matrix<std::complex<float>>> allocate_test_Matrix<std::complex<float>>(
    bool ref_matrix, bool nonuniform_ref, int64_t m, int64_t n, Params& params);

template
TestMatrix<slate::Matrix<std::complex<double>>> allocate_test_Matrix<std::complex<double>>(
    bool ref_matrix, bool nonuniform_ref, int64_t m, int64_t n, Params& params);

//------------------------------------------------------------------------------
/// Helper routine to avoid duplicating logic between HermitianMatrix
/// and SymmetricMatrix
///
template <typename matrix_type>
TestMatrix<matrix_type> allocate_test_HeSyMatrix(
    bool ref_matrix, bool nonuniform_ref, int64_t n, Params& params)
{
    using scalar_t = typename matrix_type::value_type;

    slate::Uplo uplo = params.uplo();

    auto construct_irregular = [&] (nb_func_t tileNb,
                                    dist_func_t tileRank, dist_func_t tileDevice)
    {
        return matrix_type( uplo, n, tileNb,
                            tileRank, tileDevice, MPI_COMM_WORLD );
    };
    auto construct_regular = [&] (int64_t nb, slate::GridOrder grid_order, int p, int q )
    {
        return matrix_type( uplo, n, nb, grid_order, p, q, MPI_COMM_WORLD );
    };
    auto construct_scalapack = [&] (scalar_t* data, int64_t lld, int64_t nb,
                                    slate::GridOrder grid_order, int p, int q )
    {
        return matrix_type::fromScaLAPACK( uplo, n, data, lld, nb,
                                           grid_order, p, q, MPI_COMM_WORLD );
    };

    return allocate_test_shared<matrix_type>(
                    ref_matrix, nonuniform_ref, n, n, params,
                     construct_irregular, construct_regular, construct_scalapack );
}

//------------------------------------------------------------------------------
/// Allocates a HermitianMatrix<scalar_t> and optionally a reference
/// version for testing.
///
/// @param[in] ref_matrix
///     Whether to allocate a reference matrix
///
/// @param[in] nonuniform_ref
///     If params.nonuniform_nb(), whether to also allocate the reference matrix
///     with non-uniform tiles.
///
/// @param[in] n
///     The number of columns
///
/// @param[in] params
///     The test params object which contains many of the key parameters
///
template <typename scalar_t>
TestMatrix<slate::HermitianMatrix<scalar_t>> allocate_test_HermitianMatrix(
    bool ref_matrix, bool nonuniform_ref, int64_t n, Params& params)
{
    return allocate_test_HeSyMatrix<slate::HermitianMatrix<scalar_t>>(
                ref_matrix, nonuniform_ref, n, params );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
TestMatrix<slate::HermitianMatrix<float>> allocate_test_HermitianMatrix<float>(
    bool ref_matrix, bool nonuniform_ref, int64_t n, Params& params);

template
TestMatrix<slate::HermitianMatrix<double>> allocate_test_HermitianMatrix<double>(
    bool ref_matrix, bool nonuniform_ref, int64_t n, Params& params);

template
TestMatrix<slate::HermitianMatrix<std::complex<float>>> allocate_test_HermitianMatrix<std::complex<float>>(
    bool ref_matrix, bool nonuniform_ref, int64_t n, Params& params);

template
TestMatrix<slate::HermitianMatrix<std::complex<double>>> allocate_test_HermitianMatrix<std::complex<double>>(
    bool ref_matrix, bool nonuniform_ref, int64_t n, Params& params);

//------------------------------------------------------------------------------
/// Allocates a SymmetricMatrix<scalar_t> and optionally a reference
/// version for testing.
///
/// @param[in] ref_matrix
///     Whether to allocate a reference matrix
///
/// @param[in] nonuniform_ref
///     If params.nonuniform_nb(), whether to also allocate the reference matrix
///     with non-uniform tiles.
///
/// @param[in] n
///     The number of columns
///
/// @param[in] params
///     The test params object which contains many of the key parameters
///
template <typename scalar_t>
TestMatrix<slate::SymmetricMatrix<scalar_t>> allocate_test_SymmetricMatrix(
    bool ref_matrix, bool nonuniform_ref, int64_t n, Params& params)
{
    return allocate_test_HeSyMatrix<slate::SymmetricMatrix<scalar_t>>(
                ref_matrix, nonuniform_ref, n, params );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
TestMatrix<slate::SymmetricMatrix<float>> allocate_test_SymmetricMatrix<float>(
    bool ref_matrix, bool nonuniform_ref, int64_t n, Params& params);

template
TestMatrix<slate::SymmetricMatrix<double>> allocate_test_SymmetricMatrix<double>(
    bool ref_matrix, bool nonuniform_ref, int64_t n, Params& params);

template
TestMatrix<slate::SymmetricMatrix<std::complex<float>>> allocate_test_SymmetricMatrix<std::complex<float>>(
    bool ref_matrix, bool nonuniform_ref, int64_t n, Params& params);

template
TestMatrix<slate::SymmetricMatrix<std::complex<double>>> allocate_test_SymmetricMatrix<std::complex<double>>(
    bool ref_matrix, bool nonuniform_ref, int64_t n, Params& params);


//------------------------------------------------------------------------------
/// Allocates a TriangularMatrix<scalar_t> and optionally a reference
/// version for testing.
///
/// @param[in] ref_matrix
///     Whether to allocate a reference matrix
///
/// @param[in] nonuniform_ref
///     If params.nonuniform_nb(), whether to also allocate the reference matrix
///     with non-uniform tiles.
///
/// @param[in] n
///     The number of columns
///
/// @param[in] params
///     The test params object which contains many of the key parameters
///
template <typename scalar_t>
TestMatrix<slate::TriangularMatrix<scalar_t>> allocate_test_TriangularMatrix(
    bool ref_matrix, bool nonuniform_ref, int64_t n, Params& params)
{
    // Load params variables
    slate::Uplo uplo = params.uplo();
    slate::Diag diag = params.diag();

    auto construct_irregular = [&] (nb_func_t tileNb,
                                    dist_func_t tileRank, dist_func_t tileDevice)
    {
        return slate::TriangularMatrix<scalar_t>( uplo, diag, n, tileNb,
                                                  tileRank, tileDevice, MPI_COMM_WORLD );
    };
    auto construct_regular = [&] (int64_t nb, slate::GridOrder grid_order, int p, int q )
    {
        return slate::TriangularMatrix<scalar_t>( uplo, diag, n, nb,
                                                  grid_order, p, q, MPI_COMM_WORLD );
    };
    auto construct_scalapack = [&] (scalar_t* data, int64_t lld, int64_t nb,
                                    slate::GridOrder grid_order, int p, int q )
    {
        return slate::TriangularMatrix<scalar_t>::fromScaLAPACK(
                                            uplo, diag, n, data, lld, nb,
                                            grid_order, p, q, MPI_COMM_WORLD );
    };

    return allocate_test_shared<slate::TriangularMatrix<scalar_t>>(
                    ref_matrix, nonuniform_ref, n, n, params,
                     construct_irregular, construct_regular, construct_scalapack );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
TestMatrix<slate::TriangularMatrix<float>> allocate_test_TriangularMatrix<float>(
    bool ref_matrix, bool nonuniform_ref, int64_t n, Params& params);

template
TestMatrix<slate::TriangularMatrix<double>> allocate_test_TriangularMatrix<double>(
    bool ref_matrix, bool nonuniform_ref, int64_t n, Params& params);

template
TestMatrix<slate::TriangularMatrix<std::complex<float>>> allocate_test_TriangularMatrix<std::complex<float>>(
    bool ref_matrix, bool nonuniform_ref, int64_t n, Params& params);

template
TestMatrix<slate::TriangularMatrix<std::complex<double>>> allocate_test_TriangularMatrix<std::complex<double>>(
    bool ref_matrix, bool nonuniform_ref, int64_t n, Params& params);

//------------------------------------------------------------------------------
/// Allocates a TrapezoidMatrix<scalar_t> and a reference version for testing.
///
/// @param ref_matrix[in]
///     Whether to allocate a reference matrix
///
/// @param nonuniform_ref[in]
///     If params.nonuniform_nb(), whether to also allocate the reference matrix
///     with non-uniform tiles.
///
/// @param m[in]
///     The number of rows
///
/// @param n[in]
///     The number of columns
///
/// @param params[in]
///     The test params object which contains many of the key parameters
///
template <typename scalar_t>
TestMatrix<slate::TrapezoidMatrix<scalar_t>> allocate_test_TrapezoidMatrix(
    bool ref_matrix, bool nonuniform_ref, int64_t m, int64_t n, Params& params)
{
    // Load params variables
    slate::Uplo uplo = params.uplo();
    slate::Diag diag = params.diag();

    auto construct_irregular = [&] (nb_func_t tileNb,
                                    dist_func_t tileRank, dist_func_t tileDevice)
    {
        return slate::TrapezoidMatrix<scalar_t>( uplo, diag, m, n, tileNb,
                                                  tileRank, tileDevice, MPI_COMM_WORLD );
    };
    auto construct_regular = [&] (int64_t nb, slate::GridOrder grid_order, int p, int q )
    {
        return slate::TrapezoidMatrix<scalar_t>( uplo, diag, m, n, nb,
                                                  grid_order, p, q, MPI_COMM_WORLD );
    };
    auto construct_scalapack = [&] (scalar_t* data, int64_t lld, int64_t nb,
                                    slate::GridOrder grid_order, int p, int q )
    {
        return slate::TrapezoidMatrix<scalar_t>::fromScaLAPACK(
                                            uplo, diag, m, n, data, lld, nb,
                                            grid_order, p, q, MPI_COMM_WORLD );
    };

    return allocate_test_shared<slate::TrapezoidMatrix<scalar_t>>(
                    ref_matrix, nonuniform_ref, m, n, params,
                     construct_irregular, construct_regular, construct_scalapack );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
TestMatrix<slate::TrapezoidMatrix<float>> allocate_test_TrapezoidMatrix<float>(
    bool ref_matrix, bool nonuniform_ref, int64_t m, int64_t n, Params& params);

template
TestMatrix<slate::TrapezoidMatrix<double>> allocate_test_TrapezoidMatrix<double>(
    bool ref_matrix, bool nonuniform_ref, int64_t m, int64_t n, Params& params);

template
TestMatrix<slate::TrapezoidMatrix<std::complex<float>>> allocate_test_TrapezoidMatrix<std::complex<float>>(
    bool ref_matrix, bool nonuniform_ref, int64_t m, int64_t n, Params& params);

template
TestMatrix<slate::TrapezoidMatrix<std::complex<double>>> allocate_test_TrapezoidMatrix<std::complex<double>>(
    bool ref_matrix, bool nonuniform_ref, int64_t m, int64_t n, Params& params);
