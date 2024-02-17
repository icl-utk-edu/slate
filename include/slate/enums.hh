// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
/// @file
///
#ifndef SLATE_ENUMS_HH
#define SLATE_ENUMS_HH

#include "slate/Exception.hh"

#include <blas.hh>
#include <lapack.hh>

#include <algorithm>

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
// Methods

//------------------------------------------------------------------------------
/// Algorithm to use for triangular solve (trsm).
/// @ingroup method
///
enum class MethodTrsm : char {
    Auto      = '*',    ///< Let SLATE decide
    A         = 'A',    ///< Matrix A is stationary, B is sent; use when B is small
    B         = 'B',    ///< Matrix B is stationary, A is sent; use when B is large
    TrsmA     = 'A',    // old alias
    TrsmB     = 'B',    // old alias
};

//-----------------------------------
inline void from_string( const char* name, MethodTrsm* method )
{
    std::string name_ = name;
    std::transform( name_.begin(), name_.end(), name_.begin(), ::tolower );

    if (name_ == "auto")
        *method = MethodTrsm::Auto;
    else if (name_ == "a" || name_ == "trsma")
        *method = MethodTrsm::A;
    else if (name_ == "b" || name_ == "trsmb")
        *method = MethodTrsm::B;
    else
        throw slate::Exception( "unknown trsm method" );
}

//-----------------------------------
inline const char* to_string( MethodTrsm method )
{
    switch (method) {
        case MethodTrsm::Auto: return "auto";
        case MethodTrsm::A:    return "A";
        case MethodTrsm::B:    return "B";
        default:
            throw slate::Exception( "unknown trsm method" );
    }
}

//------------------------------------------------------------------------------
/// Algorithm to use for general matrix multiply (gemm).
/// @ingroup method
///
enum class MethodGemm : char {
    Auto      = '*',    ///< Let SLATE decide
    A         = 'A',    ///< Matrix A is stationary, C is sent; use when C is small
    C         = 'C',    ///< Matrix C is stationary, A is sent; use when C is large
    GemmA     = 'A',    // old alias
    GemmC     = 'C',    // old alias
};

//-----------------------------------
inline void from_string( const char* name, MethodGemm* method )
{
    std::string name_ = name;
    std::transform( name_.begin(), name_.end(), name_.begin(), ::tolower );

    if (name_ == "auto")
        *method = MethodGemm::Auto;
    else if (name_ == "a" || name_ == "gemma")
        *method = MethodGemm::A;
    else if (name_ == "c" || name_ == "gemmc")
        *method = MethodGemm::C;
    else
        throw slate::Exception( "unknown gemm method" );
}

//-----------------------------------
inline const char* to_string( MethodGemm method )
{
    switch (method) {
        case MethodGemm::Auto: return "auto";
        case MethodGemm::A:    return "A";
        case MethodGemm::C:    return "C";
        default:
            throw slate::Exception( "unknown gemm method" );
    }
}

//------------------------------------------------------------------------------
/// Algorithm to use for Hermitian matrix multiply (hemm).
/// @ingroup method
///
enum class MethodHemm : char {
    Auto      = '*',    ///< Let SLATE decide
    A         = 'A',    ///< Matrix A is stationary, C is sent; use when C is small
    C         = 'C',    ///< Matrix C is stationary, A is sent; use when C is large
    HemmA     = 'A',    /// old alias
    HemmC     = 'C',    /// old alias
};

//-----------------------------------
inline void from_string( const char* name, MethodHemm* method )
{
    std::string name_ = name;
    std::transform( name_.begin(), name_.end(), name_.begin(), ::tolower );

    if (name_ == "auto")
        *method = MethodHemm::Auto;
    else if (name_ == "a" || name_ == "hemma")
        *method = MethodHemm::A;
    else if (name_ == "c" || name_ == "hemmc")
        *method = MethodHemm::C;
    else
        throw slate::Exception( "unknown hemm method" );
}

//-----------------------------------
inline const char* to_string( MethodHemm method )
{
    switch (method) {
        case MethodHemm::Auto: return "auto";
        case MethodHemm::A:    return "A";
        case MethodHemm::C:    return "C";
        default:
            throw slate::Exception( "unknown hemm method" );
    }
}

//------------------------------------------------------------------------------
/// Algorithm to use for Cholesky QR.
/// @ingroup method
///
enum class MethodCholQR : char {
    Auto      = '*',    ///< Let SLATE decide
    GemmA     = 'A',    ///< Use gemm-A algorithm to compute A^H A
    GemmC     = 'C',    ///< Use gemm-C algorithm to compute A^H A
    HerkA     = 'R',    ///< Use herk-A algorithm to compute A^H A; not yet implemented
    HerkC     = 'K',    ///< Use herk-C algorithm to compute A^H A
};

//-----------------------------------
inline void from_string( const char* name, MethodCholQR* method )
{
    std::string name_ = name;
    std::transform( name_.begin(), name_.end(), name_.begin(), ::tolower );

    if (name_ == "auto")
        *method = MethodCholQR::Auto;
    else if (name_ == "gemma")
        *method = MethodCholQR::GemmA;
    else if (name_ == "gemmc")
        *method = MethodCholQR::GemmC;
    else if (name_ == "herka")
        *method = MethodCholQR::HerkA;
    else if (name_ == "herkc")
        *method = MethodCholQR::HerkC;
    else
        throw slate::Exception( "unknown Cholesky QR method" );
}

//-----------------------------------
inline const char* to_string( MethodCholQR method )
{
    switch (method) {
        case MethodCholQR::Auto:  return "auto";
        case MethodCholQR::GemmA: return "gemm-A";
        case MethodCholQR::GemmC: return "gemm-C";
        case MethodCholQR::HerkA: return "herk-A";
        case MethodCholQR::HerkC: return "herk-C";
        default:
            throw slate::Exception( "unknown Cholesky QR method" );
    }
}

//------------------------------------------------------------------------------
/// Algorithm to use for least squares (gels).
/// @ingroup method
///
enum class MethodGels : char {
    Auto      = '*',    ///< Let SLATE decide
    QR        = 'Q',    ///< Use Householder QR factorization
    Geqrf     = 'Q',    ///< old alias
    CholQR    = 'C',    ///< Use Cholesky QR factorization; use when A is well-conditioned
    Cholqr    = 'C',    ///< old alias
};

//-----------------------------------
inline void from_string( const char* name, MethodGels* method )
{
    std::string name_ = name;
    std::transform( name_.begin(), name_.end(), name_.begin(), ::tolower );

    if (name_ == "auto")
        *method = MethodGels::Auto;
    else if (name_ == "qr" || name_ == "geqrf")
        *method = MethodGels::QR;
    else if (name_ == "cholqr")
        *method = MethodGels::CholQR;
    else
        throw slate::Exception( "unknown least squares (gels) method" );
}

//-----------------------------------
inline const char* to_string( MethodGels method )
{
    switch (method) {
        case MethodGels::Auto:   return "auto";
        case MethodGels::QR:     return "QR";
        case MethodGels::CholQR: return "CholQR";
        default:
            throw slate::Exception( "unknown least squares (gels) method" );
    }
}

//------------------------------------------------------------------------------
/// Algorithm to use for LU factorization and solve.
/// @ingroup method
///
enum class MethodLU : char {
    Auto       = '*',   ///< Let SLATE decide
    PartialPiv = 'P',   ///< Use classical partial pivoting
    CALU       = 'C',   ///< Use Communication Avoiding LU (CALU)
    NoPiv      = 'N',   ///< Use no-pivoting LU
    RBT        = 'R',   ///< Use Random Butterfly Transform (RBT)
    BEAM       = 'B',   ///< Use BEAM LU factorization
};

//-----------------------------------
inline void from_string( const char* name, MethodLU* method )
{
    std::string name_ = name;
    std::transform( name_.begin(), name_.end(), name_.begin(), ::tolower );

    if (name_ == "auto")
        *method = MethodLU::Auto;
    else if (name_ == "pplu" || name_ == "partialpiv")
        *method = MethodLU::PartialPiv;
    else if (name_ == "calu")
        *method = MethodLU::CALU;
    else if (name_ == "nopiv")
        *method = MethodLU::NoPiv;
    else if (name_ == "rbt")
        *method = MethodLU::RBT;
    else if (name_ == "beam")
        *method = MethodLU::BEAM;
    else
        throw slate::Exception( "unknown LU method" );
}

//-----------------------------------
inline const char* to_string( MethodLU method )
{
    switch (method) {
        case MethodLU::Auto:       return "auto";
        case MethodLU::PartialPiv: return "PPLU";
        case MethodLU::CALU:       return "CALU";
        case MethodLU::NoPiv:      return "NoPiv";
        case MethodLU::RBT:        return "RBT";
        case MethodLU::BEAM:       return "BEAM";
        default:
            throw slate::Exception( "unknown LU method" );
    }
}

//------------------------------------------------------------------------------
/// Algorithm to use for eigenvalues (eig).
/// @ingroup method
///
enum class MethodEig : char {
    Auto      = '*',    ///< Let SLATE decide
    QR        = 'Q',    ///< QR iteration
    DC        = 'D',    ///< Divide and conquer
    Bisection = 'B',    ///< Bisection; not yet implemented
    MRRR      = 'R',    ///< Multiple Relatively Robust Representations (MRRR); not yet implemented
};

//-----------------------------------
inline void from_string( const char* name, MethodEig* method )
{
    std::string name_ = name;
    std::transform( name_.begin(), name_.end(), name_.begin(), ::tolower );

    if (name_ == "auto")
        *method = MethodEig::Auto;
    else if (name_ == "qr")
        *method = MethodEig::QR;
    else if (name_ == "dc")
        *method = MethodEig::DC;
    else if (name_ == "bisection")
        *method = MethodEig::Bisection;
    else if (name_ == "mrrr")
        *method = MethodEig::MRRR;
    else
        throw slate::Exception( "unknown eig method" );
}

//-----------------------------------
inline const char* to_string( MethodEig method )
{
    switch (method) {
        case MethodEig::Auto:      return "auto";
        case MethodEig::QR:        return "QR";
        case MethodEig::DC:        return "DC";
        case MethodEig::Bisection: return "Bisection";
        case MethodEig::MRRR:      return "MRRR";
        default:
            throw slate::Exception( "unknown eig method" );
    }
}

//------------------------------------------------------------------------------
/// Algorithm to use for singular value decomposition (SVD).
/// @ingroup method
///
enum class MethodSVD : char {
    Auto      = '*',    ///< Let SLATE decide
    QR        = 'Q',    ///< QR iteration
    DC        = 'D',    ///< Divide and conquer; not yet implemented
    Bisection = 'B',    ///< Bisection; not yet implemented
};

//-----------------------------------
inline void from_string( const char* name, MethodSVD* method )
{
    std::string name_ = name;
    std::transform( name_.begin(), name_.end(), name_.begin(), ::tolower );

    if (name_ == "auto")
        *method = MethodSVD::Auto;
    else if (name_ == "qr")
        *method = MethodSVD::QR;
    else if (name_ == "dc")
        *method = MethodSVD::DC;
    else if (name_ == "bisection")
        *method = MethodSVD::Bisection;
    else
        throw slate::Exception( "unknown SVD method" );
}

//-----------------------------------
inline const char* to_string( MethodSVD method )
{
    switch (method) {
        case MethodSVD::Auto:      return "auto";
        case MethodSVD::QR:        return "QR";
        case MethodSVD::DC:        return "DC";
        case MethodSVD::Bisection: return "Bisection";
        default:
            throw slate::Exception( "unknown SVD method" );
    }
}

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
    MethodCholQR = 60,  ///< Select the algorithm to compute A^H A
    MethodEig,          ///< Select the algorithm to compute eigenpairs of tridiagonal matrix
    MethodGels,         ///< Select the gels algorithm
    MethodGemm,         ///< Select the gemm algorithm
    MethodHemm,         ///< Select the hemm algorithm
    MethodLU,           ///< Select the LU (getrf) algorithm
    MethodTrsm,         ///< Select the trsm algorithm
    MethodSVD,          ///< Select the algorithm to compute singular values of bidiagonal matrix
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
