// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
/// @file
///
#ifndef SLATE_ENUMS_HH
#define SLATE_ENUMS_HH

#include <blas.hh>
#include <lapack.hh>

namespace slate {

typedef blas::Op Op;
typedef blas::Uplo Uplo;
typedef blas::Diag Diag;
typedef blas::Side Side;
typedef blas::Layout Layout;

using lapack::Equed;
using lapack::RowCol;
typedef lapack::Norm Norm;
typedef lapack::Direction Direction;

typedef lapack::Job Job;

//------------------------------------------------------------------------------
/// Location and method of computation.
/// @ingroup enum
///
enum class Target : char {
    Host      = 'H',    ///< data resides on host
    HostTask  = 'T',    ///< computation using OpenMP nested tasks on host
    HostNest  = 'N',    ///< computation using OpenMP nested parallel for loops on host
    HostBatch = 'B',    ///< computation using batch BLAS on host (Intel MKL)
    Devices   = 'D',    ///< computation using batch BLAS on devices (cuBLAS)
};

namespace internal {

/// TargetType is used to overload functions, since there is no C++
/// partial specialization of functions, only of classes.
template <Target> class TargetType {};

} // namespace internal

//------------------------------------------------------------------------------
/// Eigenvalue algorithm to used in heev routine.
/// @ingroup enum
///
enum class MethodEig : char {
    QR        = 'Q',    ///< QR iteration for finding eigenvalues
    DC        = 'D',    ///< Divide and conquer algorithm for finding eigenvalues
};

//------------------------------------------------------------------------------
/// Keys for options to pass to SLATE routines.
/// @ingroup enum
///
enum class Option : char {
    ChunkSize,          ///< chunk size, >= 1
    Lookahead,          ///< lookahead depth, >= 0
    BlockSize,          ///< block size, >= 1
    InnerBlocking,      ///< inner blocking size, >= 1
    MaxPanelThreads,    ///< max number of threads for panel, >= 1
    Tolerance,          ///< tolerance for iterative methods, default epsilon
    Target,             ///< computation method (@see Target)
    HoldLocalWorkspace, ///< do not erase local workspace tiles for enabling
                        ///< resue of the tiles by the next routine
    Depth,              ///< depth for the RBT solver
    MaxIterations,      ///< maximum iteration count
    UseFallbackSolver,  ///< whether to fallback to a robust solver if iterations do not converge
    PivotThreshold,     ///< threshold for pivoting, >= 0, <= 1

    // Printing parameters
    PrintVerbose = 50,  ///< verbose, 0: no printing,
                        ///< verbose, 1: print metadata only (dimensions, uplo, etc.)
                        ///< verbose, 2: print first & last PrintEdgeItems rows & cols
                        ///< from the four corner tiles
                        ///< verbose, 3: print 4 corner elements of every tile
                        ///< verbose, 4: print full matrix
    PrintEdgeItems,     ///< edgeitems: number of first & last rows & cols of matrix
                        ///< to print
    PrintWidth,         ///< width print format specifier
    PrintPrecision,     ///< precision print format specifier
                        ///< For correct printing, PrintWidth = PrintPrecision + 6.

    // Methods, listed alphabetically.
    MethodCholQR = 60,  ///< Select the algorithm to compute A^H * A
    MethodEig,          ///< Select the algorithm to compute eigenpairs of tridiagonal matrix
    MethodGels,         ///< Select the gels algorithm
    MethodGemm,         ///< Select the gemm algorithm
    MethodHemm,         ///< Select the hemm algorithm
    MethodLU,           ///< Select the LU (getrf) algorithm
    MethodTrsm,         ///< Select the trsm algorithm
};

//------------------------------------------------------------------------------
/// To convert matrix between column-major and row-major.
/// @ingroup enum
///
enum class LayoutConvert : char {
    ColMajor = 'C',     ///< Convert to column-major
    RowMajor = 'R',     ///< Convert to row-major
    None     = 'N',     ///< No conversion
};

//------------------------------------------------------------------------------
/// Whether computing matrix norm, column norms, or row norms.
/// @ingroup enum
///
enum class NormScope : char {
    Columns = 'C',      ///< Compute column norms
    Rows    = 'R',      ///< Compute row norms
    Matrix  = 'M',      ///< Compute matrix norm
};

//------------------------------------------------------------------------------
/// Order to map MPI processes to tile grid.
/// @ingroup enum
///
enum class GridOrder : char {
    Col      = 'C',     ///< Column major
    Row      = 'R',     ///< Row major
    Unknown  = 'U',     ///< Unknown (e.g., if using lambda functions)
};

//------------------------------------------------------------------------------
const int HostNum = -1;
const int AllDevices = -2;
const int AnyDevice  = -3;

//------------------------------------------------------------------------------
/// A tile state in the MOSI coherency protocol
enum MOSI {
    Modified = 0x100,   ///< tile data is modified, other instances should be Invalid, cannot be purged
    OnHold = 0x1000,  ///< a hold is placed on this tile instance, cannot be purged
    Shared = 0x010,   ///< tile data is up-to-date, other instances may be Shared, or Invalid, may be purged
    Invalid = 0x001,   ///< tile data is obsolete, other instances may be Modified, Shared, or Invalid, may be purged
};
typedef short MOSI_State;


} // namespace slate

#endif // SLATE_ENUMS_HH
