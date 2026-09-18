// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_GENERATE_MATRIX_HH
#define SLATE_GENERATE_MATRIX_HH

#include "blas.hh"
#include "lapack.hh"
#include "slate/slate.hh"

namespace slate {

//------------------------------------------------------------------------------
// Parameters for Matrices.
/// Parameters for matrix generation.
/// @ingroup generate_matrix
class MatgenParams {
public:
    /// Verbose output level.
    int64_t verbose;

    /// Kind of matrix to generate (e.g., "svd", "heev", "geev").
    std::string kind;

    /// Requested condition number.
    double cond_request;

    /// Actual condition number generated (output).
    double cond_actual;

    /// Condition number for diagonal scaling (for some matrix types).
    double condD;

    /// Random number seed.
    int64_t seed;
};

//------------------------------------------------------------------------------
// Overload with sigma.

//------------------------------------------------------------------------------
/// Generates an m-by-n general-storage test matrix A.
/// Handles Matrix class.
///
/// @param[in,out] params
///     Parameters for matrix generation (kind, cond, seed, etc.).
///
/// @param[in,out] A
///     The matrix to generate.
///
/// @param[out] Sigma
///     Vector of singular values or eigenvalues, if applicable.
///
/// @param[in] opts
///     Options for generation.
///
/// @ingroup generate_matrix
///
template <typename scalar_t>
void generate_matrix(
    MatgenParams& params,
    slate::Matrix< scalar_t >& A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    slate::Options const& opts = slate::Options() );

//------------------------------------------------------------------------------
/// Generates a trapezoid/triangular test matrix A.
/// Handles TrapezoidMatrix and TriangularMatrix classes.
///
/// @param[in,out] params
///     Parameters for matrix generation.
///
/// @param[in,out] A
///     The matrix to generate.
///
/// @param[out] Sigma
///     Vector of singular values or eigenvalues, if applicable.
///
/// @param[in] opts
///     Options for generation.
///
/// @ingroup generate_matrix
///
template <typename scalar_t>
void generate_matrix(
    MatgenParams& params,
    slate::BaseTrapezoidMatrix< scalar_t >& A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    slate::Options const& opts = slate::Options() );

//------------------------------------------------------------------------------
/// Generates a Hermitian/symmetric test matrix A.
/// Handles HermitianMatrix and SymmetricMatrix classes.
///
/// @param[in,out] params
///     Parameters for matrix generation.
///
/// @param[in,out] A
///     The matrix to generate.
///
/// @param[out] Sigma
///     Vector of eigenvalues, if applicable.
///
/// @param[in] opts
///     Options for generation.
///
/// @ingroup generate_matrix
///
template <typename scalar_t>
void generate_matrix(
    MatgenParams& params,
    slate::HermitianMatrix< scalar_t >& A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    slate::Options const& opts = slate::Options() );

//------------------------------------------------------------------------------
// Overload without sigma.

//------------------------------------------------------------------------------
/// Generates an m-by-n general-storage test matrix A.
/// Overload without Sigma output.
///
/// @param[in,out] params
///     Parameters for matrix generation.
///
/// @param[in,out] A
///     The matrix to generate.
///
/// @param[in] opts
///     Options for generation.
///
/// @ingroup generate_matrix
///
template <typename scalar_t>
void generate_matrix(
    MatgenParams& params,
    slate::Matrix< scalar_t >& A,
    slate::Options const& opts = slate::Options() );

//------------------------------------------------------------------------------
/// Generates a trapezoid/triangular test matrix A.
/// Overload without Sigma output.
///
/// @param[in,out] params
///     Parameters for matrix generation.
///
/// @param[in,out] A
///     The matrix to generate.
///
/// @param[in] opts
///     Options for generation.
///
/// @ingroup generate_matrix
///
template <typename scalar_t>
void generate_matrix(
    MatgenParams& params,
    slate::BaseTrapezoidMatrix< scalar_t >& A,
    slate::Options const& opts = slate::Options() );

//------------------------------------------------------------------------------
/// Generates a Hermitian/symmetric test matrix A.
/// Overload without Sigma output.
///
/// @param[in,out] params
///     Parameters for matrix generation.
///
/// @param[in,out] A
///     The matrix to generate.
///
/// @param[in] opts
///     Options for generation.
///
/// @ingroup generate_matrix
///
template <typename scalar_t>
void generate_matrix(
    MatgenParams& params,
    slate::HermitianMatrix< scalar_t >& A,
    slate::Options const& opts = slate::Options() );

//------------------------------------------------------------------------------
/// Prints usage information for matrix generation parameters.
/// @ingroup generate_matrix
void generate_matrix_usage();

} // namespace slate

#endif // SLATE_GENERATE_MATRIX_HH
