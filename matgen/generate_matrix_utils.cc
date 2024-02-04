// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "slate/generate_matrix.hh"
#include "random.hh"
#include "generate_matrix_utils.hh"

#include <exception>
#include <string>
#include <vector>
#include <limits>
#include <complex>
#include <chrono>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace slate {

const double inf = std::numeric_limits<double>::infinity();

// -----------------------------------------------------------------------------
// ANSI color codes - enabled by default
#ifndef NO_COLOR
    const char *ansi_esc     = "\x1b[";
    const char *ansi_red     = "\x1b[31m";
    const char *ansi_bold    = "\x1b[1m";
    const char *ansi_normal  = "\x1b[0m";
#else
    const char *ansi_esc     = "";
    const char *ansi_red     = "";
    const char *ansi_bold    = "";
    const char *ansi_normal  = "";
#endif

//------------------------------------------------------------------------------
/// Splits a string by any of the delimiters.
/// Adjacent delimiters will give empty tokens.
/// See https://stackoverflow.com/questions/53849
/// @ingroup util
std::vector< std::string >
    split( const std::string& str, const std::string& delims )
{
    size_t npos = std::string::npos;
    std::vector< std::string > tokens;
    size_t start = (str.size() > 0 ? 0 : npos);
    while (start != npos) {
        size_t end = str.find_first_of( delims, start );
        tokens.push_back( str.substr( start, end - start ));
        start = (end == npos ? npos : end + 1);
    }
    return tokens;
}

//------------------------------------------------------------------------------
void generate_matrix_usage()
{
    printf(
    "The --matrix, --cond, and --condD parameters specify a test matrix.\n"
    "See Test routines: generate_matrix in the HTML documentation for a\n"
    "complete description.\n"
    "\n"
    "%s--matrix%s is one of following:\n"
    "\n"
    "%sMatrix%s    |  %sDescription%s\n"
    "----------|-------------\n"
    "zeros     |  all zero\n"
    "ones      |  all one\n"
    "identity  |  ones on diagonal, rest zero\n"
    "ij        |  Aij = i + j / 10^ceil( log10( max( m, n ) ) )\n"
    "jordan    |  ones on diagonal and first superdiagonal, rest zero\n"
    "jordanT   |  ones on diagonal and first subdiagonal, rest zero\n"
    "chebspec  |  non-singular Chebyshev spectral differentiation matrix\n"
    "circul    |  circulant matrix where the first column is [1, 2, ..., n]^T\n"
    "fiedler   |  matrix entry i,j equal to |i - j|\n"
    "gfpp      |  growth factor for gesv of 1.5^n\n"
    "kms       |  Kac-Murdock-Szego Toeplitz matrix\n"
    "orthog    |  matrix entry i,j equal to sqrt(2/(n+1))sin((i+1)(j+1)pi/(n+1))\n"
    "riemann   |  matrix entry i,j equal to i+1 if j+2 divides i+2 else -1\n"
    "ris       |  matrix entry i,j equal to 0.5/(n-i-j+1.5)\n"
    "zielkeNS  |  nonsymmetric matrix of Zielke\n"
    "          |  \n"
    "rand@     |  matrix entries random uniform on (0, 1)\n"
    "rands@    |  matrix entries random uniform on (-1, 1)\n"
    "randn@    |  matrix entries random normal with mean 0, std 1\n"
    "randb@    |  matrix entries random uniform from {0, 1}\n"
    "randr@    |  matrix entries random uniform from {-1, 1}\n"
    "          |  \n"
    "diag^@    |  A = Sigma\n"
    "svd^@     |  A = U Sigma V^H\n"
    "poev^@    |  A = V Sigma V^H  (eigenvalues positive, i.e., matrix SPD)\n"
    "spd^@     |  alias for poev\n"
    "heev^@    |  A = V Lambda V^H (eigenvalues mixed signs)\n"
    "syev^@    |  alias for heev\n"
    "geev^@    |  A = V T V^H, Schur-form T                       [not yet implemented]\n"
    "geevx^@   |  A = X T X^{-1}, Schur-form T, X ill-conditioned [not yet implemented]\n"
    "\n"
    "^ and @ denote optional suffixes described below.\n"
    "\n"
    "%s^ Distribution%s  |  %sDescription%s\n"
    "----------------|-------------\n"
    "_logrand        |  log(sigma_i) random uniform on [ log(1/cond), log(1) ]; default\n"
    "_arith          |  sigma_i = 1 - frac{i - 1}{n - 1} (1 - 1/cond); arithmetic: sigma_{i+1} - sigma_i is constant\n"
    "_geo            |  sigma_i = (cond)^{ -(i-1)/(n-1) };             geometric:  sigma_{i+1} / sigma_i is constant\n"
    "_cluster0       |  Sigma = [ 1, 1/cond, ..., 1/cond ];  1  unit value,  n-1 small values\n"
    "_cluster1       |  Sigma = [ 1, ..., 1, 1/cond ];      n-1 unit values,  1  small value\n"
    "_rarith         |  _arith,    reversed order\n"
    "_rgeo           |  _geo,      reversed order\n"
    "_rcluster0      |  _cluster0,  reversed order\n"
    "_rcluster1      |  _cluster1, reversed order\n"
    "_specified      |  user specified Sigma on input\n"
    "                |  \n"
    "_rand           |  sigma_i random uniform on (0, 1)\n"
    "_rands          |  sigma_i random uniform on (-1, 1)\n"
    "_randn          |  sigma_i random normal with mean 0, std 1\n"
    "\n"
    "%s@ Scaling%s       |  %sDescription%s\n"
    "----------------|-------------\n"
    "_ufl            |  scale near underflow         = 1e-308 for double\n"
    "_ofl            |  scale near overflow          = 2e+308 for double\n"
    "_small          |  scale near sqrt( underflow ) = 1e-154 for double\n"
    "_large          |  scale near sqrt( overflow  ) = 6e+153 for double\n"
    "\n"
    "%s@ Modifier%s      |  %sDescription%s\n"
    "----------------|-------------\n"
    "_dominant       |  make matrix diagonally dominant\n"
    "_zerocolN       |  set column N to zero, 0 <= N < n\n"
    "_zerocolFRAC    |  set column N = FRAC * (n-1) to zero, 0 <= FRAC <= 1.0\n"
    "                |  For Hermitian and symmetric matrices (and currently any\n"
    "                |  trapezoid matrix), sets row and column N to zero.\n"
    "\n",
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal
    );
}

//------------------------------------------------------------------------------
/// Decode matrix type, distribution, scaling and modifier.
///
template <typename scalar_t>
void decode_matrix(
    MatgenParams& params,
    BaseMatrix<scalar_t>& A,
    TestMatrixType& type,
    TestMatrixDist& dist,
    blas::real_type<scalar_t>& cond,
    blas::real_type<scalar_t>& condD,
    blas::real_type<scalar_t>& sigma_max,
    bool& dominant,
    int64_t& zero_col )
{
    using real_t = blas::real_type<scalar_t>;

    //                                                single      double
    // underflow level (ufl) == lamch("safe min")  ==  1e-38  or  2e-308
    const real_t ufl = std::numeric_limits< real_t >::min();

    // overflow  level (ofl) ==                    ==   8e37  or  4e307
    const real_t ofl = 1 / ufl;

    // eps                   == lamch("precision") == 1.2e-7  or  2.2e-16
    const real_t eps = std::numeric_limits< real_t >::epsilon();

    // locals
    char msg[ 256 ];
    std::string kind = params.kind;

    //---------------
    cond = params.cond_request;
    bool cond_default = std::isnan( cond );
    if (cond_default) {
        cond = 1 / sqrt( eps );
    }

    condD = params.condD;
    bool condD_default = std::isnan( condD );
    if (condD_default) {
        condD = 1;
    }

    //---------------
    sigma_max = 1;
    std::vector< std::string > tokens = split( kind, "-_" );

    // ----- decode matrix type
    auto token_iter = tokens.begin();
    if (token_iter == tokens.end()) {
        throw std::runtime_error( "empty matrix kind" );
    }
    std::string base = *token_iter;
    ++token_iter;
    type = TestMatrixType::identity;
    if      (base == "zeros"   ) { type = TestMatrixType::zeros;    }
    else if (base == "ones"    ) { type = TestMatrixType::ones;     }
    else if (base == "identity") { type = TestMatrixType::identity; }
    else if (base == "ij"      ) { type = TestMatrixType::ij;       }
    else if (base == "jordan"  ) { type = TestMatrixType::jordan;   }
    else if (base == "jordanT" ) { type = TestMatrixType::jordanT;  }
    else if (base == "chebspec") { type = TestMatrixType::chebspec; }
    else if (base == "circul"  ) { type = TestMatrixType::circul;   }
    else if (base == "fiedler" ) { type = TestMatrixType::fiedler;  }
    else if (base == "gfpp"    ) { type = TestMatrixType::gfpp;     }
    else if (base == "kms"     ) { type = TestMatrixType::kms;      }
    else if (base == "orthog"  ) { type = TestMatrixType::orthog;   }
    else if (base == "riemann" ) { type = TestMatrixType::riemann;  }
    else if (base == "ris"     ) { type = TestMatrixType::ris;      }
    else if (base == "zielkeNS") { type = TestMatrixType::zielkeNS; }
    else if (base == "randb"   ) { type = TestMatrixType::randb;    }
    else if (base == "randr"   ) { type = TestMatrixType::randr;    }
    else if (base == "randn"   ) { type = TestMatrixType::randn;    }
    else if (base == "rands"   ) { type = TestMatrixType::rands;    }
    else if (base == "rand"    ) { type = TestMatrixType::rand;     }
    else if (base == "diag"    ) { type = TestMatrixType::diag;     }
    else if (base == "svd"     ) { type = TestMatrixType::svd;      }
    else if (base == "poev" ||
             base == "spd"     ) { type = TestMatrixType::poev;     }
    else if (base == "heev" ||
             base == "syev"    ) { type = TestMatrixType::heev;     }
    else if (base == "geevx"   ) { type = TestMatrixType::geevx;    }
    else if (base == "geev"    ) { type = TestMatrixType::geev;     }
    else {
        snprintf( msg, sizeof( msg ), "in '%s': unknown matrix '%s'",
                  kind.c_str(), base.c_str() );
        throw std::runtime_error( msg );
    }

    std::string token;
    dist      = TestMatrixDist::none;
    sigma_max = 1;
    dominant  = false;
    zero_col  = -1;

    while (token_iter != tokens.end()) {
        token = *token_iter;

        // ----- decode distribution
        if      (token == "randn"    ) { dist = TestMatrixDist::randn;     }
        else if (token == "rands"    ) { dist = TestMatrixDist::rands;     }
        else if (token == "rand"     ) { dist = TestMatrixDist::rand;      }
        else if (token == "logrand"  ) { dist = TestMatrixDist::logrand;   }
        else if (token == "arith"    ) { dist = TestMatrixDist::arith;     }
        else if (token == "geo"      ) { dist = TestMatrixDist::geo;       }
        else if (token == "cluster1" ) { dist = TestMatrixDist::cluster1;  }
        else if (token == "cluster0" ) { dist = TestMatrixDist::cluster0;  }
        else if (token == "rarith"   ) { dist = TestMatrixDist::rarith;    }
        else if (token == "rgeo"     ) { dist = TestMatrixDist::rgeo;      }
        else if (token == "rcluster1") { dist = TestMatrixDist::rcluster1; }
        else if (token == "rcluster0") { dist = TestMatrixDist::rcluster0; }
        else if (token == "specified") { dist = TestMatrixDist::specified; }

        // ----- decode scaling
        else if (token == "small") { sigma_max = sqrt( ufl ); }
        else if (token == "large") { sigma_max = sqrt( ofl ); }
        else if (token == "ufl"  ) { sigma_max = ufl; }
        else if (token == "ofl"  ) { sigma_max = ofl; }

        // ----- decode modifiers
        else if (token == "dominant") { dominant = true; }
        else if (token.find( "zerocol" ) == 0) {
            // zeroN for integer N, or zeroR for
            token = token.substr( 7 );  // skip "zerocol"
            size_t pos;
            zero_col = std::stoi( token, &pos, 0 );
            if (pos < token.size()) {
                double fraction = std::stod( token, &pos );
                if (pos < token.size()) {
                    snprintf( msg, sizeof( msg ),
                              "in '%s': can't parse number after 'zerocol'",
                              kind.c_str() );
                    throw std::runtime_error( msg );
                }
                if (fraction < 0.0 || fraction > 1.0) {
                    snprintf( msg, sizeof( msg ),
                              "in '%s': fraction outside [0.0, 1.0]",
                              kind.c_str() );
                    throw std::runtime_error( msg );
                }
                zero_col = int64_t( fraction * (A.n() - 1) );
            }
            if (zero_col < 0 || zero_col >= A.n()) {
                snprintf( msg, sizeof(msg),
                          "in '%s', column index %lld outside [0, n=%lld)",
                          kind.c_str(), llong( zero_col ), llong( A.n() ) );
                throw std::runtime_error( msg );
            }
        }
        else {
            snprintf( msg, sizeof( msg ), "in '%s': unknown suffix '%s'",
                      kind.c_str(), token.c_str() );
            throw std::runtime_error( msg );
        }

        ++token_iter;
    }

    // Validate distribution.
    if (dist != TestMatrixDist::none) {
        // Error if matrix type doesn't support distribution.
        if (! (type == TestMatrixType::diag
               || type == TestMatrixType::svd
               || type == TestMatrixType::poev
               || type == TestMatrixType::heev
               || type == TestMatrixType::geev
               || type == TestMatrixType::geevx)) {
            snprintf( msg, sizeof( msg ),
                      "in '%s': matrix '%s' doesn't support distribution",
                      kind.c_str(), base.c_str() );
            throw std::runtime_error( msg );
        }
    }
    else {
        dist = TestMatrixDist::logrand;  // default
    }

    // Error if matrix type doesn't support scaling.
    if (sigma_max != 1
        && ! (type == TestMatrixType::rand
              || type == TestMatrixType::rands
              || type == TestMatrixType::randn
              || type == TestMatrixType::randb
              || type == TestMatrixType::randr
              || type == TestMatrixType::svd
              || type == TestMatrixType::poev
              || type == TestMatrixType::heev
              || type == TestMatrixType::geev
              || type == TestMatrixType::geevx)) {
        snprintf( msg, sizeof( msg ),
                  "in '%s': matrix '%s' doesn't support scaling",
                  kind.c_str(), base.c_str() );
        throw std::runtime_error( msg );
    }

    // Error if matrix type doesn't support diagonally dominant.
    if (dominant
        && ! (type == TestMatrixType::rand
              || type == TestMatrixType::rands
              || type == TestMatrixType::randn
              || type == TestMatrixType::randb
              || type == TestMatrixType::randr
              || type == TestMatrixType::svd
              || type == TestMatrixType::poev
              || type == TestMatrixType::heev
              || type == TestMatrixType::geev
              || type == TestMatrixType::geevx))
    {
        snprintf( msg, sizeof( msg ),
                  "in '%s': matrix '%s' doesn't support diagonally dominant",
                  kind.c_str(), base.c_str() );
        throw std::runtime_error( msg );
    }

    // ----- check compatability of options
    if (A.m() != A.n()
        && (type == TestMatrixType::poev
            || type == TestMatrixType::heev
            || type == TestMatrixType::geev
            || type == TestMatrixType::geevx))
    {
        snprintf( msg, sizeof( msg ), "in '%s': matrix '%s' requires m == n",
                  kind.c_str(), base.c_str() );
        throw std::runtime_error( msg );
    }

    // Check if cond is known or unused.
    if (type == TestMatrixType::zeros
        || type == TestMatrixType::ones
        || zero_col >= 0) {
        cond = inf;
    }
    else if (type == TestMatrixType::identity
             || type == TestMatrixType::orthog) {
        cond = 1;
    }
    else if (type != TestMatrixType::svd
             && type != TestMatrixType::heev
             && type != TestMatrixType::poev
             && type != TestMatrixType::geev
             && type != TestMatrixType::geevx) {
        // cond unused
        cond = nan("1234");
    }
    params.cond_actual = cond;

    // Warn if user set condD and matrix type doesn't use it.
    if (! condD_default
        && type != TestMatrixType::svd
        && type != TestMatrixType::heev
        && type != TestMatrixType::poev)
    {
        fprintf( stderr, "%sWarning: matrix '%s' ignores condD %.2e.%s\n",
                 ansi_red, kind.c_str(), params.condD, ansi_normal );
    }

    // Warn if SPD requested, but distribution is not > 0.
    if (type == TestMatrixType::poev
        && (dist == TestMatrixDist::rands
            || dist == TestMatrixDist::randn)) {
        fprintf( stderr, "%sWarning: matrix '%s' using rands or randn "
                 "will not generate SPD matrix; use rand instead.%s\n",
                 ansi_red, kind.c_str(), ansi_normal );
    }
}

//------------------------------------------------------------------------------
/// Generates the actual seed from the user provided seed.
int64_t configure_seed(MPI_Comm comm, int64_t user_seed)
{
    // if the given seed is -1, generate a new seed
    if (user_seed == -1) {
        // use the highest resolution clock as the seed
        using namespace std::chrono;
        user_seed = duration_cast<high_resolution_clock::duration>(
                    high_resolution_clock::now().time_since_epoch()).count();
    }

    // ensure seeds are uniform across MPI ranks
    slate_mpi_call(
        MPI_Bcast( &user_seed, 1, MPI_INT64_T, 0, comm ) );

    return user_seed;
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void decode_matrix(
    MatgenParams& params,
    BaseMatrix<float>& A,
    TestMatrixType& type,
    TestMatrixDist& dist,
    blas::real_type<float>& cond,
    blas::real_type<float>& condD,
    blas::real_type<float>& sigma_max,
    bool& dominant,
    int64_t& zero_col );

template
void decode_matrix(
    MatgenParams& params,
    BaseMatrix<double>& A,
    TestMatrixType& type,
    TestMatrixDist& dist,
    blas::real_type<double>& cond,
    blas::real_type<double>& condD,
    blas::real_type<double>& sigma_max,
    bool& dominant,
    int64_t& zero_col );

template
void decode_matrix(
    MatgenParams& params,
    BaseMatrix<std::complex<float>>& A,
    TestMatrixType& type,
    TestMatrixDist& dist,
    blas::real_type<std::complex<float>>& cond,
    blas::real_type<std::complex<float>>& condD,
    blas::real_type<std::complex<float>>& sigma_max,
    bool& dominant,
    int64_t& zero_col );

template
void decode_matrix(
    MatgenParams& params,
    BaseMatrix<std::complex<double>>& A,
    TestMatrixType& type,
    TestMatrixDist& dist,
    blas::real_type<std::complex<double>>& cond,
    blas::real_type<std::complex<double>>& condD,
    blas::real_type<std::complex<double>>& sigma_max,
    bool& dominant,
    int64_t& zero_col );

} // namespace slate
