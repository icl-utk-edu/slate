// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_SCALAPACK_API_COMMON_HH
#define SLATE_SCALAPACK_API_COMMON_HH

// get BLAS_FORTRAN_NAME and blas_int
#include "blas/fortran.h"

#include "slate/slate.hh"

//==============================================================================
// Prototypes for BLACS routines.
extern "C" {

// Get my process number and the number of processes.
void Cblacs_pinfo( blas_int* mypnum, blas_int* nprocs );

// Get row and col in 2D process grid for process pnum.
void Cblacs_pcoord( blas_int context, blas_int pnum,
                    blas_int* prow, blas_int* pcol );

// Lookup BLACS information.
void Cblacs_get( blas_int context, blas_int what, blas_int* val );

// Get 2D process grid size and my row and col in the grid.
void Cblacs_gridinfo( blas_int context,
                      blas_int* nprow, blas_int* npcol,
                      blas_int* myprow, blas_int* mypcol );

} // extern "C"

//==============================================================================
namespace slate {
namespace scalapack_api {

#define logprintf(fmt, ...) \
    do { \
        fprintf( stdout, "%s:%d %s(): " fmt, \
                 __FILE__, __LINE__, __func__, __VA_ARGS__ ); \
        fflush(0); \
    } while (0)

enum dtype {
    BlockCyclic2D     = 1,
    BlockCyclic2D_INB = 2,
};

enum desc {
    DTYPE_ = 0,
    CTXT_,
    M_,
    N_,
    MB_,
    NB_,
    RSRC_,
    CSRC_,
    LLD_,
};

enum desc_inb {
    DTYPE_INB = 0,
    CTXT_INB,
    M_INB,
    N_INB,
    IMB_INB,
    INB_INB,
    MB_INB,
    NB_INB,
    RSRC_INB,
    CSRC_INB,
    LLD_INB
};

//------------------------------------------------------------------------------
inline blas_int desc_ctxt( blas_int const* descA )
{
    return descA[ CTXT_ ];
}

inline blas_int desc_m( blas_int const* descA )
{
    return descA[ M_ ];
}

inline blas_int desc_n( blas_int const* descA )
{
    return descA[ N_ ];
}

inline blas_int desc_mb( blas_int const* descA )
{
    return (descA[ DTYPE_ ] == BlockCyclic2D) ? descA[ MB_ ] : descA[ MB_INB ];
}

inline blas_int desc_nb( blas_int const* descA )
{
    return (descA[ DTYPE_ ] == BlockCyclic2D) ? descA[ NB_ ] : descA[ NB_INB ];
}

inline blas_int desc_lld( blas_int const* descA )
{
    return (descA[ DTYPE_ ] == BlockCyclic2D) ? descA[ LLD_ ] : descA[ LLD_INB ];
}

//------------------------------------------------------------------------------
/// Determine grid order for default BLACS context.
inline slate::GridOrder slate_scalapack_blacs_grid_order()
{
    // if nprocs == 1, the grid layout is irrelevant, all-OK
    // if nprocs > 1 check the grid location of process-number-1 pnum(1).
    // if pnum(1) is at grid-coord(0, 1) then grid is col-major
    // else if pnum(1) is not at grid-coord(0, 1) then grid is row-major
    blas_int mypnum, nprocs, prow, pcol, icontxt=-1, imone=-1, izero=0, pnum_1=1;
    Cblacs_pinfo( &mypnum, &nprocs );
    if (nprocs == 1) // only one process, so col-major grid-layout
        return slate::GridOrder::Col;
    Cblacs_get( imone, izero, &icontxt );
    Cblacs_pcoord( icontxt, pnum_1, &prow, &pcol );
    if (prow == 0 && pcol == 1) { // col-major grid-layout
        return slate::GridOrder::Col;
    }
    else { // row-major grid-layout
        return slate::GridOrder::Row;
    }
}

template <typename scalar_t>
slate::Matrix<scalar_t> slate_scalapack_submatrix(
    blas_int Am, blas_int An, slate::Matrix<scalar_t>& A,
    blas_int ia, blas_int ja, blas_int const* descA )
{
    // logprintf("Am %d An %d ia %d ja %d desc_mb( descA ) %d desc_nb( descA ) %d A.m() %ld A.n() %ld \n", Am, An, ia, ja, desc_mb( descA ), desc_nb( descA ), A.m(), A.n());
    if (ia == 1 && ja == 1 && Am == A.m() && An == A.n())
        return A;
    assert( (ia-1) % desc_mb( descA ) == 0 );
    assert( (ja-1) % desc_nb( descA ) == 0 );
    assert( Am % desc_mb( descA ) == 0 );
    assert( An % desc_nb( descA ) == 0 );
    int64_t i1 = (ia-1)/desc_mb( descA );
    int64_t i2 = i1 + (Am/desc_mb( descA )) - 1;
    int64_t j1 = (ja-1)/desc_nb( descA );
    int64_t j2 = j1 + (An/desc_nb( descA )) - 1;
    return A.sub( i1, i2, j1, j2 );
}

template <typename scalar_t>
slate::SymmetricMatrix<scalar_t> slate_scalapack_submatrix(
    blas_int Am, blas_int An, slate::SymmetricMatrix<scalar_t>& A,
    blas_int ia, blas_int ja, blas_int const* descA )
{
    //logprintf("Am %d An %d ia %d ja %d desc_mb( descA ) %d desc_nb( descA ) %d \n", Am, An, ia, ja, desc_mb( descA ), desc_nb( descA ));
    if (ia == 1 && ja == 1 && Am == A.m() && An == A.n())
        return A;
    assert( (ia-1) % desc_mb( descA ) == 0 );
    assert( Am % desc_mb( descA ) == 0 );
    int64_t i1 = (ia-1)/desc_mb( descA );
    int64_t i2 = i1 + (Am/desc_mb( descA )) - 1;
    return A.sub( i1, i2 );
}

template <typename scalar_t>
slate::TriangularMatrix<scalar_t> slate_scalapack_submatrix(
    blas_int Am, blas_int An, slate::TriangularMatrix<scalar_t>& A,
    blas_int ia, blas_int ja, blas_int const* descA )
{
    if (ia == 1 && ja == 1 && Am == A.m() && An == A.n())
        return A;
    assert( (ia-1) % desc_mb( descA ) == 0 );
    assert( Am % desc_mb( descA ) == 0 );
    int64_t i1 = (ia-1)/desc_mb( descA );
    int64_t i2 = i1 + (Am/desc_mb( descA )) - 1;
    return A.sub( i1, i2 );
}

template <typename scalar_t>
slate::TrapezoidMatrix<scalar_t> slate_scalapack_submatrix(
    blas_int Am, blas_int An, slate::TrapezoidMatrix<scalar_t>& A,
    blas_int ia, blas_int ja, blas_int const* descA )
{
    if (ia == 1 && ja == 1 && Am == A.m() && An == A.n())
        return A;
    assert( (ia-1) % desc_nb( descA ) == 0);
    assert( An % desc_nb( descA) == 0);
    int64_t i1 = (ia-1)/desc_nb( descA );
    int64_t i2 = i1 + (Am/desc_nb( descA )) - 1;
    return A.sub( i1, i2, i2 );
}

template <typename scalar_t>
slate::HermitianMatrix<scalar_t> slate_scalapack_submatrix(
    blas_int Am, blas_int An, slate::HermitianMatrix<scalar_t>& A,
    blas_int ia, blas_int ja, blas_int const* descA )
{
    if (ia == 1 && ja == 1 && Am == A.m() && An == A.n())
        return A;
    assert( (ia-1) % desc_nb( descA ) == 0 );
    assert( An % desc_nb( descA ) == 0 );
    int64_t i1 = (ia-1)/desc_nb( descA );
    int64_t i2 = i1 + (Am/desc_nb( descA )) - 1;
    return A.sub( i1, i2 );
}

//==============================================================================
/// Initialize target setting from environment variable.
/// Uses thread-safe Scott Meyers Singleton.
class TargetConfig
{
public:
    /// @return target (HostTask, Devices, etc.) to use.
    static slate::Target value()
    {
        return instance().target_;
    }

    /// Set target to use.
    static void value( slate::Target target )
    {
        instance().target_ = target;
    }

private:
    /// On first call, creates the singleton instance, which queries the
    /// environment variable.
    /// @return singleton instance.
    static TargetConfig& instance()
    {
        static TargetConfig instance_;
        return instance_;
    }

    /// Constructor queries the environment variable or sets to default value.
    TargetConfig()
    {
        target_ = slate::Target::HostTask;
        const char* str = std::getenv( "SLATE_SCALAPACK_TARGET" );
        if (str) {
            std::string str_ = str;
            std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );
            if (str_ == "devices")
                target_ = slate::Target::Devices;
            else if (str_ == "hosttask")
                target_ = slate::Target::HostTask;
            else if (str_ == "hostnest")
                target_ = slate::Target::HostNest;
            else if (str_ == "hostbatch")
                target_ = slate::Target::HostBatch;
            else
                slate_error( std::string( "Invalid target: " ) + str );
        }
    }

    // Prevent copy construction and copy assignment.
    TargetConfig( const TargetConfig& orig ) = delete;
    TargetConfig& operator= ( const TargetConfig& orig ) = delete;

    //----------------------------------------
    // Data
    slate::Target target_;
};

//==============================================================================
/// Initialize panel threads setting from environment variable.
/// Uses thread-safe Scott Meyers Singleton.
class PanelThreadsConfig
{
public:
    /// @return number of panel threads to use.
    static int value()
    {
        return instance().panel_threads_;
    }

    /// Set number of panel threads to use.
    static void value( int panel_threads )
    {
        instance().panel_threads_ = panel_threads;
    }

private:
    /// On first call, creates the singleton instance, which queries the
    /// environment variable.
    /// @return singleton instance.
    static PanelThreadsConfig& instance()
    {
        static PanelThreadsConfig instance_;
        return instance_;
    }

    /// Constructor queries the environment variable or sets to default value.
    PanelThreadsConfig()
    {
        panel_threads_ = blas::max( omp_get_max_threads()/2, 1 );
        const char* str = std::getenv( "SLATE_SCALAPACK_PANELTHREADS" );
        if (str) {
            panel_threads_ = blas::max( strtol( str, NULL, 0 ), 1 );
        }
    }

    // Prevent copy construction and copy assignment.
    PanelThreadsConfig( const PanelThreadsConfig& orig ) = delete;
    PanelThreadsConfig& operator= ( const PanelThreadsConfig& orig ) = delete;

    //----------------------------------------
    // Data
    int panel_threads_;
};

//==============================================================================
/// Initialize ib setting from environment variable.
/// Uses thread-safe Scott Meyers Singleton.
class IBConfig
{
public:
    /// @return inner blocking to use.
    static int64_t value()
    {
        return instance().ib_;
    }

    /// Set inner blocking to use.
    static void value( int64_t ib )
    {
        instance().ib_ = ib;
    }

private:
    /// On first call, creates the singleton instance, which queries the
    /// environment variable.
    /// @return singleton instance.
    static IBConfig& instance()
    {
        static IBConfig instance_;
        return instance_;
    }

    /// Constructor queries the environment variable or sets to default value.
    IBConfig()
    {
        ib_ = 16;
        const char* str = std::getenv( "SLATE_SCALAPACK_IB" );
        if (str) {
            ib_ = blas::max( strtol( str, NULL, 0 ), 1 );
        }
    }

    // Prevent copy construction and copy assignment.
    IBConfig( const IBConfig& orig ) = delete;
    IBConfig& operator= ( const IBConfig& orig ) = delete;

    //----------------------------------------
    // Data
    int64_t ib_;
};

//==============================================================================
/// Initialize verbose setting from environment variable.
/// Uses thread-safe Scott Meyers Singleton.
class VerboseConfig
{
public:
    /// @return verbose flag to use.
    static int value()
    {
        return instance().verbose_;
    }

    /// Set verbose flag to use.
    static void value( int verbose )
    {
        instance().verbose_ = verbose;
    }

private:
    /// On first call, creates the singleton instance, which queries the
    /// environment variable.
    /// @return singleton instance.
    static VerboseConfig& instance()
    {
        static VerboseConfig instance_;
        return instance_;
    }

    /// Constructor queries the environment variable or sets to default value.
    VerboseConfig()
    {
        verbose_ = 0;
        const char* str = std::getenv( "SLATE_SCALAPACK_VERBOSE" );
        if (str) {
            verbose_ = strtol( str, NULL, 0 );
        }
    }

    // Prevent copy construction and copy assignment.
    VerboseConfig( const VerboseConfig& orig ) = delete;
    VerboseConfig& operator= ( const VerboseConfig& orig ) = delete;

    //----------------------------------------
    // Data
    int verbose_;
};

//==============================================================================
/// Initialize lookahead setting from environment variable.
/// Uses thread-safe Scott Meyers Singleton.
class LookaheadConfig
{
public:
    /// @return lookahead to use.
    static int64_t value()
    {
        return instance().lookahead_;
    }

    /// Set lookahead to use.
    static void value( int64_t lookahead )
    {
        instance().lookahead_ = lookahead;
    }

private:
    /// On first call, creates the singleton instance, which queries the
    /// environment variable.
    /// @return singleton instance.
    static LookaheadConfig& instance()
    {
        static LookaheadConfig instance_;
        return instance_;
    }

    /// Constructor queries the environment variable or sets to default value.
    LookaheadConfig()
    {
        lookahead_ = 1;
        const char* str = std::getenv( "SLATE_SCALAPACK_LOOKAHEAD" );
        if (str) {
            lookahead_ = blas::max( strtol( str, NULL, 0 ), 1 );
        }
    }

    // Prevent copy construction and copy assignment.
    LookaheadConfig( const LookaheadConfig& orig ) = delete;
    LookaheadConfig& operator= ( const LookaheadConfig& orig ) = delete;

    //----------------------------------------
    // Data
    int64_t lookahead_;
};

//==============================================================================
// This is duplicated from blaspp/src/blas_internal.hh

//------------------------------------------------------------------------------
/// @see to_blas_int
///
inline blas_int to_blas_int_( int64_t x, const char* x_str )
{
    if constexpr (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if_msg( x > std::numeric_limits<blas_int>::max(), "%s", x_str );
    }
    return blas_int( x );
}

//----------------------------------------
/// Convert int64_t to blas_int.
/// If blas_int is 64-bit, this does nothing.
/// If blas_int is 32-bit, throws if x > INT_MAX, so conversion would overflow.
///
/// Note this is in src/blas_internal.hh, so this macro won't pollute
/// the namespace when apps #include <blas.hh>.
///
#define to_blas_int( x ) to_blas_int_( x, #x )

//==============================================================================


// -----------------------------------------------------------------------------
// TODO: this is duplicated at the testing module
#define SCALAPACK_numroc BLAS_FORTRAN_NAME( numroc, NUMROC )
extern "C"
blas_int SCALAPACK_numroc(
    blas_int* n, blas_int* nb, blas_int* iproc, blas_int* isrcproc,
    blas_int* nprocs );

inline int64_t scalapack_numroc(
    int64_t n, int64_t nb, blas_int iproc, blas_int isrcproc, blas_int nprocs )
{
    blas_int n_    = to_blas_int( n );
    blas_int nb_   = to_blas_int( nb );
    blas_int nroc_ = SCALAPACK_numroc( &n_, &nb_, &iproc, &isrcproc, &nprocs );
    return nroc_;
}

#define scalapack_indxl2g BLAS_FORTRAN_NAME( indxl2g, INDXL2G )
extern "C"
blas_int scalapack_indxl2g(
    blas_int* indxloc, blas_int* nb, blas_int* iproc, blas_int* isrcproc,
    blas_int* nprocs );

} // namespace scalapack_api
} // namespace slate

#endif // SLATE_SCALAPACK_API_COMMON_HH
