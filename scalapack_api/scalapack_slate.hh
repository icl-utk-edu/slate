// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_SCALAPACK_API_COMMON_HH
#define SLATE_SCALAPACK_API_COMMON_HH

// get BLAS_FORTRAN_NAME and blas_int
#include "blas/fortran.h"

#include "slate/slate.hh"

extern "C" void Cblacs_pinfo(int* mypnum, int* nprocs);
extern "C" void Cblacs_pcoord(int icontxt, int pnum, int* prow, int* pcol);
extern "C" void Cblacs_get(int icontxt, int what, int* val);

#include <complex>

namespace slate {
namespace scalapack_api {

#define logprintf(fmt, ...) \
    do { fprintf(stderr, "%s:%d %s(): " fmt, __FILE__, __LINE__, __func__, __VA_ARGS__); fflush(0); } while (0)

enum slate_scalapack_dtype {BLOCK_CYCLIC_2D=1, BLOCK_CYCLIC_2D_INB=2};
enum slate_scalapack_desc {DTYPE_=0, CTXT_, M_, N_, MB_, NB_, RSRC_, CSRC_, LLD_};
enum slate_scalapack_desc_inb {DTYPE_INB=0, CTXT_INB, M_INB, N_INB, IMB_INB, INB_INB, MB_INB, NB_INB, RSRC_INB, CSRC_INB, LLD_INB};

inline int desc_CTXT(int* desca)
{
    return desca[1];
}

inline int desc_M(int* desca)
{
    return desca[2];
}

inline int desc_N(int* desca)
{
    return desca[3];
}

inline int desc_MB(int* desca)
{
    return (desca[0] == BLOCK_CYCLIC_2D) ? desca[MB_] : desca[MB_INB];
}

inline int desc_NB(int* desca)
{
    return (desca[0] == BLOCK_CYCLIC_2D) ? desca[NB_] : desca[NB_INB];
}

inline int desc_LLD(int* desca)
{
    return (desca[0] == BLOCK_CYCLIC_2D) ? desca[LLD_] : desca[LLD_INB];
}

inline slate::GridOrder slate_scalapack_blacs_grid_order()
{
    // if nprocs == 1, the grid layout is irrelevant, all-OK
    // if nprocs > 1 check the grid location of process-number-1 pnum(1).
    // if pnum(1) is at grid-coord(0, 1) then grid is col-major
    // else if pnum(1) is not at grid-coord(0, 1) then grid is row-major
    int mypnum, nprocs, prow, pcol, icontxt=-1, imone=-1, izero=0, pnum_1=1;
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

template< typename scalar_t >
inline slate::Matrix<scalar_t> slate_scalapack_submatrix(int Am, int An, slate::Matrix<scalar_t>& A, int ia, int ja, int* desca)
{
    // logprintf("Am %d An %d ia %d ja %d desc_MB(desca) %d desc_NB(desca) %d A.m() %ld A.n() %ld \n", Am, An, ia, ja, desc_MB(desca), desc_NB(desca), A.m(), A.n());
    if (ia == 1 && ja == 1 && Am == A.m() && An == A.n()) return A;
    assert((ia-1) % desc_MB(desca) == 0);
    assert((ja-1) % desc_NB(desca) == 0);
    assert(Am % desc_MB(desca) == 0);
    assert(An % desc_NB(desca) == 0);
    int64_t i1 = (ia-1)/desc_MB(desca);
    int64_t i2 = i1 + (Am/desc_MB(desca)) - 1;
    int64_t j1 = (ja-1)/desc_NB(desca);
    int64_t j2 = j1 + (An/desc_NB(desca)) - 1;
    return A.sub(i1, i2, j1, j2);
}

template< typename scalar_t >
inline slate::SymmetricMatrix<scalar_t> slate_scalapack_submatrix(int Am, int An, slate::SymmetricMatrix<scalar_t>& A, int ia, int ja, int* desca)
{
    //logprintf("Am %d An %d ia %d ja %d desc_MB(desca) %d desc_NB(desca) %d \n", Am, An, ia, ja, desc_MB(desca), desc_NB(desca));
    if (ia == 1 && ja == 1 && Am == A.m() && An == A.n()) return A;
    assert((ia-1) % desc_MB(desca) == 0);
    assert(Am % desc_MB(desca) == 0);
    int64_t i1 = (ia-1)/desc_MB(desca);
    int64_t i2 = i1 + (Am/desc_MB(desca)) - 1;
    return A.sub(i1, i2);
}

template< typename scalar_t >
inline slate::TriangularMatrix<scalar_t> slate_scalapack_submatrix(int Am, int An, slate::TriangularMatrix<scalar_t>& A, int ia, int ja, int* desca)
{
    if (ia == 1 && ja == 1 && Am == A.m() && An == A.n()) return A;
    assert((ia-1) % desc_MB(desca) == 0);
    assert(Am % desc_MB(desca) == 0);
    int64_t i1 = (ia-1)/desc_MB(desca);
    int64_t i2 = i1 + (Am/desc_MB(desca)) - 1;
    return A.sub(i1, i2);
}

template< typename scalar_t >
inline slate::TrapezoidMatrix<scalar_t> slate_scalapack_submatrix(int Am, int An, slate::TrapezoidMatrix<scalar_t>& A, int ia, int ja, int* desca)
{
    if (ia == 1 && ja == 1 && Am == A.m() && An == A.n()) return A;
    assert((ia-1) % desc_NB(desca) == 0);
    assert(An % desc_NB(desca) == 0);
    int64_t i1 = (ia-1)/desc_NB(desca);
    int64_t i2 = i1 + (Am/desc_NB(desca)) - 1;
    return A.sub(i1, i2, i2);
}

template< typename scalar_t >
inline slate::HermitianMatrix<scalar_t> slate_scalapack_submatrix(int Am, int An, slate::HermitianMatrix<scalar_t>& A, int ia, int ja, int* desca)
{
    if (ia == 1 && ja == 1 && Am == A.m() && An == A.n()) return A;
    assert((ia-1) % desc_NB(desca) == 0);
    assert(An % desc_NB(desca) == 0);
    int64_t i1 = (ia-1)/desc_NB(desca);
    int64_t i2 = i1 + (Am/desc_NB(desca)) - 1;
    return A.sub(i1, i2);
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

// -----------------------------------------------------------------------------
// helper funtion to check and do type conversion
// TODO: this is duplicated at the testing module
inline int int64_to_int(int64_t n)
{
    if (sizeof(int64_t) > sizeof(blas_int))
        assert(n < std::numeric_limits<int>::max());
    int n_ = (int)n;
    return n_;
}

// -----------------------------------------------------------------------------
// TODO: this is duplicated at the testing module
#define scalapack_numroc BLAS_FORTRAN_NAME(numroc,NUMROC)
extern "C" int scalapack_numroc(int* n, int* nb, int* iproc, int* isrcproc, int* nprocs);
inline int64_t scalapack_numroc(int64_t n, int64_t nb, int iproc, int isrcproc, int nprocs)
{
    int n_ = int64_to_int(n);
    int nb_ = int64_to_int(nb);
    int nroc_ = scalapack_numroc(&n_, &nb_, &iproc, &isrcproc, &nprocs);
    int64_t nroc = (int64_t)nroc_;
    return nroc;
}

#define scalapack_indxl2g BLAS_FORTRAN_NAME(indxl2g,INDXL2G)
extern "C" int scalapack_indxl2g(int* indxloc, int* nb, int* iproc, int* isrcproc, int* nprocs);

} // namespace scalapack_api
} // namespace slate

#endif // SLATE_SCALAPACK_API_COMMON_HH
