// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_copy.hh"
#include "print_matrix.hh"

#include "matrix_utils.hh"
#include "test_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <numeric>
#include <random>
#include <unistd.h> // getpid for seed

using std::real, std::imag;
using slate::Uplo, slate::Diag, slate::LayoutConvert, slate::Norm,
      slate::NormScope;
using slate::Matrix, slate::TriangularMatrix, slate::TrapezoidMatrix,
      slate::SymmetricMatrix, slate::HermitianMatrix;
using blas::max;
using slate::ceildiv;

//------------------------------------------------------------------------------
/// @return true if either real( x ) or imag( x ) is NaN (not-a-number).
///
template <typename real_t>
bool isnan( std::complex<real_t> const& x )
{
    return std::isnan( real( x ) ) || std::isnan( imag( x ) );
}

using std::isnan;  // for real types

//------------------------------------------------------------------------------
/// @return true if either real( x ) or imag( x ) is inf.
///
template <typename real_t>
bool isinf( std::complex<real_t> const& x )
{
    return std::isinf( real( x ) ) || std::isinf( imag( x ) );
}

using std::isinf;  // for real types

//------------------------------------------------------------------------------
/// Dispatches to the appropriate ScaLAPACK routine depending on the SLATE
/// matrix type. The matrix A is passed solely to determine the matrix type;
/// it is not referenced.
template <typename matrix_type>
auto scalapack_norm_dispatch(
    slate::Norm norm, slate::Uplo uplo, slate::Diag diag,
    int64_t m, int64_t n,
    matrix_type& A,
    typename matrix_type::value_type const* A_data, blas_int* A_desc,
    blas::real_type< typename matrix_type::value_type >* work )
{
    using scalar_t = typename matrix_type::value_type;
    using real_t = blas::real_type<scalar_t>;

    real_t A_norm = 0;
    if (std::is_same< matrix_type, Matrix<scalar_t> >::value) {
        A_norm = scalapack::lange(
            norm, m, n,
            A_data, 1, 1, A_desc, work );
    }
    else if (std::is_same< matrix_type, TriangularMatrix<scalar_t> >::value
             || std::is_same< matrix_type, TrapezoidMatrix<scalar_t> >::value ) {
        A_norm = scalapack::lantr(
            norm, uplo, diag, m, n,
            A_data, 1, 1, A_desc, work );
    }
    else if (std::is_same< matrix_type, SymmetricMatrix<scalar_t> >::value) {
        A_norm = scalapack::lansy(
            norm, uplo, n,
            A_data, 1, 1, A_desc, work );
    }
    else if (std::is_same< matrix_type, HermitianMatrix<scalar_t> >::value) {
        A_norm = scalapack::lanhe(
            norm, uplo, n,
            A_data, 1, 1, A_desc, work );
    }
    else {
        slate_error( "Unknown matrix type" );
    }
    return A_norm;
}

//------------------------------------------------------------------------------
/// Simple version of std::erase_if from C++20; we currently use C++17.
/// Doesn't return count.
template <typename T, typename Alloc, typename Pred>
void erase_if( std::vector<T, Alloc>& vec, Pred pred )
{
    auto iter = std::remove_if( vec.begin(), vec.end(), pred );
    vec.erase( iter, vec.end() );
}

//------------------------------------------------------------------------------
/// Extended tests set one entry to a large value, and tests that the
/// norm finds that entry. It tests tiles in the Cartesian product of
/// block-row and block-col indices covering the 2x2 set of tiles in all
/// 4 corners ("x" tiles), and one random row i and col j ("r" tiles),
/// giving up to 25 tiles:
///
///               j
///     [ x x ... r ... x x ]
///     [ x x ... r ... x x ]
///     [     ...   ...     ]
///     [ r r ... r ... r r ] i
///     [     ...   ...     ]
///     [ x x ... r ... x x ]
///     [ x x ... r ... x x ]
///
/// For each of those tiles, it tests entries in a similar Cartesian
/// product of row and col indices, up to 25 entries. (Same diagram.)
///
/// For entries, it picks a value that is finite, infinite, or NaN.
/// For complex, use all combinations of { 0, finite, infinite, NaN }^2,
/// excluding { 0, 0 }.
///
/// Rather than doing an exhaustive test of (25 tiles * 25 entries * 16
/// values)^2 = 1e8 tests, it does params.extended number of random tests.
/// Each test picks 2 random tiles at a time, which could be the same tile,
/// to test if it will correctly find NaN and Inf values.
///
template <typename matrix_type>
void extended_tests(
    Params& params,
    slate::Norm norm, slate::Uplo uplo, slate::Diag diag,
    int64_t m, int64_t n,
    matrix_type& A,
    typename matrix_type::value_type* Aref_data,
    blas_int* Aref_desc,
    blas::real_type< typename matrix_type::value_type >* work,
    slate::Options const& opts )
{
    using scalar_t = typename matrix_type::value_type;
    using real_t = blas::real_type<scalar_t>;

    const real_t NaN = std::numeric_limits<real_t>::quiet_NaN();
    const real_t inf = std::numeric_limits<real_t>::infinity();
    const real_t eps = std::numeric_limits<real_t>::epsilon();

    int verbose = params.verbose();
    int extended = params.extended();

    // Seed all MPI processes the same for extended tests.
    // todo: some way to specify seed on command line.
    // Should this seeding go in test.cc?
    static unsigned int seed = 0;
    if (seed == 0) {
        std::random_device rd;
        seed = rd();
        MPI_Bcast( &seed, 1, MPI_INT, 0, MPI_COMM_WORLD );
        if (A.mpiRank() == 0) {
            printf( "seed %u\n", seed );
        }
    }

    // Mersenne twister random number generator.
    std::mt19937 rng;
    rng.seed( seed );

    // Allow for difference, except max norm in real should be exact.
    real_t tol;
    if (norm == Norm::Max && ! slate::is_complex<scalar_t>::value)
        tol = 0;
    else
        tol = 10*eps;

    int p = params.grid.m();
    int q = params.grid.n();
    int64_t mb  = Aref_desc[ 4 ];
    int64_t nb  = Aref_desc[ 5 ];
    int64_t lld = Aref_desc[ 8 ];

    // Test 2x2 set of tiles in all 4 corners, and 1 other random row and col.
    int64_t mt = A.mt();
    int64_t nt = A.nt();
    std::vector<int64_t> i_array = { 0, 1, mt - 2, mt - 1 };
    if (mt > 4)
        i_array.push_back( rng() % (mt - 4) + 2 );

    std::vector<int64_t> j_array = { 0, 1, nt - 2, nt - 1 };
    if (nt > 4)
        j_array.push_back( rng() % (nt - 4) + 2 );

    // Erase out-of-bounds indices.
    erase_if( i_array, [ mt ](int64_t i) { return i < 0 || i >= mt; } );
    erase_if( j_array, [ nt ](int64_t j) { return j < 0 || j >= nt; } );
    int ni = i_array.size();
    int nj = j_array.size();
    assert( ni >= 1 && nj >= 1 );

    // Build list of finite, inf, and nan values.
    // We want large real numbers, say in [ 1e6, 1e7 ].
    std::uniform_real_distribution< real_t > uniform_1e6( 1e6, 1e7 );
    real_t val  = uniform_1e6( rng );
    real_t val2 = uniform_1e6( rng );
    auto r = rng();
    if (r & 0x1)
        val  = -val;
    if (r & 0x2)
        val2 = -val2;

    std::vector<scalar_t> values;
    if constexpr (blas::is_complex<scalar_t>::value) {
        values = {
                          { 0.0, val2 }, { 0.0, inf }, { 0.0, NaN },
            { val, 0.0 }, { val, val2 }, { val, inf }, { val, NaN },
            { inf, 0.0 }, { inf, val2 }, { inf, inf }, { inf, NaN },
            { NaN, 0.0 }, { NaN, val2 }, { NaN, inf }, { NaN, NaN },
            // Extra finite values to increase chance of finite result.
            { val2, 0.0 }, { 0.0, val }, { val2, val }, { val, val2 },
            { val2, 0.0 }, { 0.0, val }, { val2, val }, { val, val2 },
        };
    }
    else {
        values = { val, val2, inf, NaN };
    }
    int nvalues = values.size();

    for (int cnt = 0; cnt < extended; ++cnt) {
        // Get 2 random tiles; can be the same.
        // If triangular, get in lower or upper part.
        // i, j are tile block-row, block-col indices.
        // Variables ending in _ are 2-element arrays for the 2 tiles.
        int64_t i_[ 2 ], j_[ 2 ];
        for (int k : {0, 1}) {
            do {
                i_[ k ] = i_array[ rng() % ni ];
                j_[ k ] = j_array[ rng() % nj ];
            } while (   (uplo == Uplo::Lower && i_[ k ] < j_[ k ])
                     || (uplo == Uplo::Upper && i_[ k ] > j_[ k ]));
        }

        // ii, jj are row, col indices within a tile.
        int64_t ii_[ 2 ], jj_[ 2 ];
        for (int k : {0, 1}) {
            // Test 2x2 set of entries in all 4 corners,
            // and 1 other random row and col.
            int64_t ib = A.tileMb( i_[ k ] );
            int64_t jb = A.tileNb( j_[ k ] );

            std::vector<int64_t> ii_array = { 0, 1, ib - 2, ib - 1 };
            if (ib > 4)
                ii_array.push_back( rng() % (ib - 4) + 2 );

            std::vector<int64_t> jj_array = { 0, 1, jb - 2, jb - 1 };
            if (jb > 4)
                jj_array.push_back( rng() % (jb - 4) + 2 );

            // Erase out-of-bounds indices.
            erase_if( ii_array, [ ib ](int64_t i) { return i < 0 || i >= ib; } );
            erase_if( jj_array, [ jb ](int64_t j) { return j < 0 || j >= jb; } );
            int nii = ii_array.size();
            int njj = jj_array.size();
            assert( nii > 0 && njj > 0 );

            // Get 2 random entries.
            // If triangular, get in lower or upper part.
            // If 2nd entry (k == 1), 2 tiles are the same (i, j),
            // there are multiple entries to choose from (nii, njj) ["same_tile"],
            // and 2 entries are the same (ii, jj), get new entry.
            bool same_tile = k == 1 && i_[0] == i_[1] && j_[0] == j_[1]
                             && nii > 1 && njj > 1;
            do {
                ii_[ k ] = ii_array[ rng() % nii ];
                jj_[ k ] = jj_array[ rng() % njj ];
            } while (   (uplo == Uplo::Lower && ii_[ k ] < jj_[ k ])
                     || (uplo == Uplo::Upper && ii_[ k ] > jj_[ k ])
                     || (same_tile && ii_[0] == ii_[1] && jj_[0] == jj_[1]));
        }

        // Get 2 random values.
        scalar_t value_[ 2 ] = {
            values[ rng() % nvalues ],
            values[ rng() % nvalues ]
        };

        // Hermitian matrices have real diagonals.
        if constexpr (std::is_same< matrix_type, HermitianMatrix<scalar_t> >::value
                      && blas::is_complex<scalar_t>::value) {
            for (int k : {0, 1}) {
                if (i_[ k ] == j_[ k ] && ii_[ k ] == jj_[ k ])
                    value_[ k ] = real( value_[ k ] );
            }
        }

        // Triangular and trapezoid matrices can have implicit unit diagonals.
        if constexpr (std::is_same< matrix_type, TriangularMatrix<scalar_t> >::value
                      || std::is_same< matrix_type, TrapezoidMatrix<scalar_t> >::value) {
            if (diag == Diag::Unit) {
                for (int k : {0, 1}) {
                    if (i_[ k ] == j_[ k ] && ii_[ k ] == jj_[ k ])
                        value_[ k ] = 1.0;
                }
            }
        }

        // Save entries and set to values.
        scalar_t save_[ 2 ] = { 0, 0 };
        for (int k : {0, 1}) {
            int64_t i  =  i_[ k ];
            int64_t j  =  j_[ k ];
            int64_t ii = ii_[ k ];
            int64_t jj = jj_[ k ];
            int64_t ilocal = int( i / p )*mb + ii;
            int64_t jlocal = int( j / q )*nb + jj;
            if (A.tileIsLocal( i, j )) {
                A.tileGetForWriting( i, j, LayoutConvert::ColMajor );
                auto T = A( i, j );
                save_[ k ] = T( ii, jj );
                T.at( ii, jj ) = value_[ k ];
                Aref_data[ ilocal + jlocal*lld ] = value_[ k ];
            }
        }

        //==================================================
        // Run SLATE test.
        //==================================================
        real_t A_norm = slate::norm( norm, A, opts );

        //==================================================
        // Check error against NaN, Inf, or ScaLAPACK result.
        //==================================================
        real_t A_norm_ref = -1;
        real_t error = -1;

        // If a value is nan xor inf, expect A_norm to be the same.
        // If values are both nan and inf, A_norm can be either nan or inf.
        // E.g., abs( complex( nan, inf ) ) = inf in gfortran 12.
        bool value_isnan = isnan( value_[0] ) || isnan( value_[1] );
        bool value_isinf = isinf( value_[0] ) || isinf( value_[1] );
        bool okay;
        if (value_isnan && value_isinf) {
            okay = isnan( A_norm ) || isinf( A_norm );
        }
        else if (value_isnan) {
            okay = isnan( A_norm );
        }
        else if (value_isinf) {
            okay = isinf( A_norm );
        }
        else {
            A_norm_ref = scalapack_norm_dispatch(
                norm, uplo, diag, m, n, A, &Aref_data[0], Aref_desc, &work[0] );

            // difference between norms
            error = std::abs( A_norm - A_norm_ref ) / A_norm_ref;
            if (norm == Norm::One) {
                error /= sqrt( m );
            }
            else if (norm == Norm::Inf) {
                error /= sqrt( n );
            }
            else if (norm == Norm::Fro) {
                error /= sqrt( m*n );
            }
            okay = error <= tol;
        }
        params.okay() = params.okay() && okay;

        if (A.mpiRank() == 0 && (verbose || ! okay)) {
            printf( "entry 0: i %5lld, j %5lld, ii %3lld, jj %3lld, value %9.2e + %9.2ei\n"
                    "entry 1: i %5lld, j %5lld, ii %3lld, jj %3lld, value %9.2e + %9.2ei  "
                    "norm %9.2e, ref %9.2e, error %9.2e, %s\n\n",
                    llong( i_[0] ), llong( j_[0] ), llong( ii_[0] ), llong( jj_[0] ),
                    real( value_[0] ), imag( value_[0] ),
                    llong( i_[1] ), llong( j_[1] ), llong( ii_[1] ), llong( jj_[1] ),
                    real( value_[1] ), imag( value_[1] ),
                    A_norm, A_norm_ref, error, (okay ? "pass" : "failed") );
        }

        // Restore entries, in reverse order, in case same entry is set twice.
        for (int k : {1, 0}) {
            int64_t i  =  i_[ k ];
            int64_t j  =  j_[ k ];
            int64_t ii = ii_[ k ];
            int64_t jj = jj_[ k ];
            int64_t ilocal = int( i / p )*mb + ii;
            int64_t jlocal = int( j / q )*nb + jj;
            if (A.tileIsLocal( i, j )) {
                A.tileGetForWriting( i, j, LayoutConvert::ColMajor );
                auto T = A( i, j );
                T.at( ii, jj ) = save_[ k ];
                Aref_data[ ilocal + jlocal*lld ] = save_[ k ];
            }
        }
    }
}

//------------------------------------------------------------------------------
template <typename matrix_type>
void test_norm_work( Params& params, bool run )
{
    using scalar_t = typename matrix_type::value_type;
    using real_t = blas::real_type<scalar_t>;

    const real_t eps = std::numeric_limits<real_t>::epsilon();

    // get & mark input values
    slate::Norm norm = params.norm();
    slate::NormScope scope = params.scope();
    slate::Op trans = params.trans();
    slate::Uplo uplo = params.uplo();
    slate::Diag diag = params.diag();

    int64_t m = params.dim.m();
    int64_t n;
    if (std::is_same< matrix_type, TriangularMatrix<scalar_t> >::value
        || std::is_same< matrix_type, SymmetricMatrix<scalar_t> >::value
        || std::is_same< matrix_type, HermitianMatrix<scalar_t> >::value) {
        n = m;  // square
    }
    else {
        n = params.dim.n();
    }

    int p = params.grid.m();
    int q = params.grid.n();
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    bool trace = params.trace() == 'y';
    bool nonuniform_nb = params.nonuniform_nb() == 'y';
    bool ref_copy = nonuniform_nb && (check || ref);
    int verbose = params.verbose();
    int extended = params.extended();
    slate::Origin origin = params.origin();
    slate::Target target = params.target();
    slate::GridOrder grid_order = params.grid_order();
    params.matrix.mark();

    mark_params_for_test_Matrix( params );

    // mark non-standard output values
    params.time();
    params.ref_time();

    if (! run)
        return;

    // Check for common invalid combinations
    if (is_invalid_parameters( params, true )) {
        return;
    }

    slate::Options const opts =  {
        {slate::Option::Target, target}
    };

    auto A_alloc = allocate_test_Matrix<scalar_t>( ref_copy, false, m, n, params );

    auto& Afull     = A_alloc.A;
    auto& Aref_full = A_alloc.Aref;

    // Cast to desired matrix type.
    matrix_type A    = matrix_cast< matrix_type >( Afull,     uplo, diag );
    matrix_type Aref = matrix_cast< matrix_type >( Aref_full, uplo, diag );

    slate::generate_matrix( params.matrix, A );

    if (ref_copy) {
        copy_matrix( A, Aref );
    }

    std::vector<real_t> values;
    if (scope == NormScope::Columns) {
        values.resize( A.n() );
    }
    else if (scope == NormScope::Rows) {
        values.resize( A.m() );
    }

    if (trans == slate::Op::Trans)
        A = transpose( A );
    else if (trans == slate::Op::ConjTrans)
        A = conj_transpose( A );

    print_matrix( "A", A, params );

    real_t A_norm = 0;
    if (! ref_only) {
        if (trace) slate::trace::Trace::on();
        else slate::trace::Trace::off();

        //==================================================
        // Run SLATE test.
        // Compute || A ||_norm.
        //==================================================
        double time = barrier_get_wtime( MPI_COMM_WORLD );

        if (scope == NormScope::Matrix) {
            A_norm = slate::norm( norm, A, opts );
        }
        else if (scope == NormScope::Columns) {
            if constexpr (std::is_same< matrix_type, Matrix<scalar_t> >::value) {
                slate::colNorms( norm, A, values.data(), opts );
            }
            else {
                slate_error( "Unsupported matrix for col norm scope" );
            }
        }
        else {
            slate_error( "Unsupported norm scope" );
        }

        time = barrier_get_wtime( MPI_COMM_WORLD ) - time;

        if (trace) slate::trace::Trace::finish();

        // compute and save timing/performance
        params.time() = time;
    }

    if (check || ref) {
        #ifdef SLATE_HAVE_SCALAPACK
            // comparison with reference routine from ScaLAPACK

            // initialize BLACS and ScaLAPACK
            blas_int ictxt, Aref_desc[9];
            A_alloc.create_ScaLAPACK_context( &ictxt );
            A_alloc.ScaLAPACK_descriptor( ictxt, Aref_desc );

            auto& Aref_data = ref_copy ? A_alloc.Aref_data : A_alloc.A_data;

            if (origin != slate::Origin::ScaLAPACK && !ref_copy) {
                Aref_data.resize( A_alloc.lld * A_alloc.nloc );

                copy( A, &Aref_data[0], Aref_desc );
            }

            // allocate work space
            int64_t lwork;
            if (std::is_same< matrix_type, SymmetricMatrix<scalar_t> >::value
                || std::is_same< matrix_type, HermitianMatrix<scalar_t> >::value) {
                lwork = 2*A_alloc.mloc + A_alloc.nloc
                      + nb*ceildiv( ceildiv( A_alloc.nloc, nb ),
                                    std::lcm( p, q ) / p );
            }
            else {
                lwork = max( A_alloc.mloc, A_alloc.nloc );
            }
            std::vector<real_t> work( lwork );

            // (Sca)LAPACK norms don't support trans; map One <=> Inf norm.
            Norm op_norm = norm;
            if (trans == slate::Op::Trans || trans == slate::Op::ConjTrans) {
                if (norm == Norm::One)
                    op_norm = Norm::Inf;
                else if (norm == Norm::Inf)
                    op_norm = Norm::One;
            }

            // difference between norms
            real_t error = 0.;
            real_t A_norm_ref = 0;

            //==================================================
            // Run ScaLAPACK reference routine.
            //==================================================
            double time = barrier_get_wtime( MPI_COMM_WORLD );
            if (scope == NormScope::Matrix) {
                A_norm_ref = scalapack_norm_dispatch(
                    op_norm, uplo, diag, m, n, A, &Aref_data[0], Aref_desc, &work[0] );

                // difference between norms
                error = std::abs( A_norm - A_norm_ref ) / A_norm_ref;
                if (op_norm == Norm::One) {
                    error /= sqrt( m );
                }
                else if (op_norm == Norm::Inf) {
                    error /= sqrt( n );
                }
                else if (op_norm == Norm::Fro) {
                    error /= sqrt( m*n );
                }

                if (verbose && A.mpiRank() == 0) {
                    printf( "norm %15.8e, ref %15.8e, ref - norm %5.2f, error %9.2e\n",
                            A_norm, A_norm_ref, A_norm_ref - A_norm, error );
                }
            }
            else if (scope == NormScope::Columns) {
                if (! std::is_same< matrix_type, Matrix<scalar_t> >::value ) {
                    slate_error( "Unsupported matrix type for col scope" );
                }
                for (int64_t j = 0; j < n; ++j) {
                    A_norm_ref = scalapack::lange(
                        norm, m, 1,
                        &Aref_data[0], 1, j+1, Aref_desc, &work[0] );
                    error += std::abs( values[ j ] - A_norm_ref ) / A_norm_ref;
                }
            }
            else {
                slate_error( "Unknown norm scope" );
            }
            time = barrier_get_wtime( MPI_COMM_WORLD ) - time;

            params.ref_time() = time;
            params.error() = error;

            // Allow for difference, except max norm in real should be exact.
            real_t tol;
            if (norm == Norm::Max && ! slate::is_complex<scalar_t>::value)
                tol = 0;
            else
                tol = 10*eps;
            params.okay() = (params.error() <= tol);

            //---------- extended tests
            if (extended && scope == NormScope::Matrix) {
                if (grid_order != slate::GridOrder::Col) {
                    params.msg() = "extended tests not implemented on row-major grid";
                }
                else {
                    extended_tests( params, op_norm, uplo, diag, m, n, A,
                                    &Aref_data[0], Aref_desc, &work[0], opts );
                }
            }

            Cblacs_gridexit( ictxt );
            //Cblacs_exit( 1 ) does not handle re-entering
        #else  // not SLATE_HAVE_SCALAPACK
            SLATE_UNUSED( A_norm );
            SLATE_UNUSED( extended );
            SLATE_UNUSED( verbose );
            if (A.mpiRank() == 0)
                printf( "ScaLAPACK not available\n" );
        #endif
    }
}

//------------------------------------------------------------------------------
template <typename scalar_t >
void test_norm_dispatch( Params& params, bool run )
{
    std::string routine = params.routine;
    if (routine == "genorm") {
        test_norm_work< slate::Matrix<scalar_t> >( params, run );
    }
    else if (routine == "tznorm") {
        test_norm_work< slate::TrapezoidMatrix<scalar_t> >( params, run );
    }
    else if (routine == "trnorm") {
        test_norm_work< slate::TriangularMatrix<scalar_t> >( params, run );
    }
    else if (routine == "synorm") {
        test_norm_work< slate::SymmetricMatrix<scalar_t> >( params, run );
    }
    else if (routine == "henorm") {
        test_norm_work< slate::HermitianMatrix<scalar_t> >( params, run );
    }
    else {
        throw slate::Exception( "unknown routine: " + routine );
    }
}

//------------------------------------------------------------------------------
void test_norm( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_norm_dispatch<float>( params, run );
            break;

        case testsweeper::DataType::Double:
            test_norm_dispatch<double>( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_norm_dispatch<std::complex<float>>( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_norm_dispatch<std::complex<double>>( params, run );
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}
