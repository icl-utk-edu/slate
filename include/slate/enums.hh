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

extern const char* MethodTrsm_help;

//-----------------------------------
inline const char* to_c_string( MethodTrsm value )
{
    switch (value) {
        case MethodTrsm::Auto: return "auto";
        case MethodTrsm::A:    return "A";
        case MethodTrsm::B:    return "B";
    }
    return "?";
}

//-----------------------------------
inline std::string to_string( MethodTrsm value )
{
    return to_c_string( value );
}

//-----------------------------------
inline MethodTrsm from_string( std::string const& str, MethodTrsm dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "auto")
        return MethodTrsm::Auto;
    else if (str_ == "a" || str_ == "trsma")
        return MethodTrsm::A;
    else if (str_ == "b" || str_ == "trsmb")
        return MethodTrsm::B;
    else
        throw Exception( "unknown trsm method: " + str );
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

extern const char* MethodGemm_help;

//-----------------------------------
inline const char* to_c_string( MethodGemm value )
{
    switch (value) {
        case MethodGemm::Auto: return "auto";
        case MethodGemm::A:    return "A";
        case MethodGemm::C:    return "C";
    }
    return "?";
}

//-----------------------------------
inline std::string to_string( MethodGemm value )
{
    return to_c_string( value );
}

//-----------------------------------
inline MethodGemm from_string( std::string const& str, MethodGemm dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "auto")
        return MethodGemm::Auto;
    else if (str_ == "a" || str_ == "gemma")
        return MethodGemm::A;
    else if (str_ == "c" || str_ == "gemmc")
        return MethodGemm::C;
    else
        throw Exception( "unknown gemm method: " + str );
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

extern const char* MethodHemm_help;

//-----------------------------------
inline const char* to_c_string( MethodHemm value )
{
    switch (value) {
        case MethodHemm::Auto: return "auto";
        case MethodHemm::A:    return "A";
        case MethodHemm::C:    return "C";
    }
    return "?";
}

//-----------------------------------
inline std::string to_string( MethodHemm value )
{
    return to_c_string( value );
}

//-----------------------------------
inline MethodHemm from_string( std::string const& str, MethodHemm dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "auto")
        return MethodHemm::Auto;
    else if (str_ == "a" || str_ == "hemma")
        return MethodHemm::A;
    else if (str_ == "c" || str_ == "hemmc")
        return MethodHemm::C;
    else
        throw Exception( "unknown hemm method: " + str );
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

extern const char* MethodCholQR_help;

//-----------------------------------
inline const char* to_c_string( MethodCholQR value )
{
    switch (value) {
        case MethodCholQR::Auto:  return "auto";
        case MethodCholQR::GemmA: return "gemm-A";
        case MethodCholQR::GemmC: return "gemm-C";
        case MethodCholQR::HerkA: return "herk-A";
        case MethodCholQR::HerkC: return "herk-C";
    }
    return "?";
}

//-----------------------------------
inline std::string to_string( MethodCholQR value )
{
    return to_c_string( value );
}

//-----------------------------------
inline MethodCholQR from_string( std::string const& str, MethodCholQR dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "auto")
        return MethodCholQR::Auto;
    else if (str_ == "gemma")
        return MethodCholQR::GemmA;
    else if (str_ == "gemmc")
        return MethodCholQR::GemmC;
    else if (str_ == "herka")
        return MethodCholQR::HerkA;
    else if (str_ == "herkc")
        return MethodCholQR::HerkC;
    else
        throw Exception( "unknown Cholesky QR method: " + str );
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

extern const char* MethodGels_help;

//-----------------------------------
inline const char* to_c_string( MethodGels value )
{
    switch (value) {
        case MethodGels::Auto:   return "auto";
        case MethodGels::QR:     return "QR";
        case MethodGels::CholQR: return "CholQR";
    }
    return "?";
}

//-----------------------------------
inline std::string to_string( MethodGels value )
{
    return to_c_string( value );
}

//-----------------------------------
inline MethodGels from_string( std::string const& str, MethodGels dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "auto")
        return MethodGels::Auto;
    else if (str_ == "qr" || str_ == "geqrf")
        return MethodGels::QR;
    else if (str_ == "cholqr")
        return MethodGels::CholQR;
    else
        throw Exception( "unknown least squares (gels) method: " + str );
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

extern const char* MethodLU_help;

//-----------------------------------
inline const char* to_c_string( MethodLU value )
{
    switch (value) {
        case MethodLU::Auto:       return "auto";
        case MethodLU::PartialPiv: return "PPLU";
        case MethodLU::CALU:       return "CALU";
        case MethodLU::NoPiv:      return "NoPiv";
        case MethodLU::RBT:        return "RBT";
        case MethodLU::BEAM:       return "BEAM";
    }
    return "?";
}

//-----------------------------------
inline std::string to_string( MethodLU value )
{
    return to_c_string( value );
}

//-----------------------------------
inline MethodLU from_string( std::string const& str, MethodLU dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "auto")
        return MethodLU::Auto;
    else if (str_ == "pplu" || str_ == "partialpiv")
        return MethodLU::PartialPiv;
    else if (str_ == "calu")
        return MethodLU::CALU;
    else if (str_ == "nopiv")
        return MethodLU::NoPiv;
    else if (str_ == "rbt")
        return MethodLU::RBT;
    else if (str_ == "beam")
        return MethodLU::BEAM;
    else
        throw Exception( "unknown LU method: " + str );
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

extern const char* MethodEig_help;

//-----------------------------------
inline const char* to_c_string( MethodEig value )
{
    switch (value) {
        case MethodEig::Auto:      return "auto";
        case MethodEig::QR:        return "QR";
        case MethodEig::DC:        return "DC";
        case MethodEig::Bisection: return "Bisection";
        case MethodEig::MRRR:      return "MRRR";
    }
    return "?";
}

//-----------------------------------
inline std::string to_string( MethodEig value )
{
    return to_c_string( value );
}

//-----------------------------------
inline MethodEig from_string( std::string const& str, MethodEig dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "auto")
        return MethodEig::Auto;
    else if (str_ == "qr")
        return MethodEig::QR;
    else if (str_ == "dc")
        return MethodEig::DC;
    else if (str_ == "bisection")
        return MethodEig::Bisection;
    else if (str_ == "mrrr")
        return MethodEig::MRRR;
    else
        throw Exception( "unknown eig method: " + str );
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

extern const char* MethodSVD_help;

//-----------------------------------
inline const char* to_c_string( MethodSVD value )
{
    switch (value) {
        case MethodSVD::Auto:      return "auto";
        case MethodSVD::QR:        return "QR";
        case MethodSVD::DC:        return "DC";
        case MethodSVD::Bisection: return "Bisection";
    }
    return "?";
}

//-----------------------------------
inline std::string to_string( MethodSVD value )
{
    return to_c_string( value );
}

//-----------------------------------
inline MethodSVD from_string( std::string const& str, MethodSVD dummy )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "auto")
        return MethodSVD::Auto;
    else if (str_ == "qr")
        return MethodSVD::QR;
    else if (str_ == "dc")
        return MethodSVD::DC;
    else if (str_ == "bisection")
        return MethodSVD::Bisection;
    else
        throw Exception( "unknown SVD method: " + str );
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
