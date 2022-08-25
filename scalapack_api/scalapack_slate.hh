// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
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

inline slate::Target slate_scalapack_set_target()
{
    // set the SLATE default computational target
    // 5th character from: hostTask hostNest hostBatch deviCes
    slate::Target target = slate::Target::HostTask;
    char* targetstr = std::getenv("SLATE_SCALAPACK_TARGET");
    if (targetstr) {
        char targetchar = (char)(toupper(targetstr[4]));
        if (targetchar == 'T') target = slate::Target::HostTask;
        else if (targetchar == 'N') target = slate::Target::HostNest;
        else if (targetchar == 'B') target = slate::Target::HostBatch;
        else if (targetchar == 'C') target = slate::Target::Devices;
    }
    return target;
}

inline int64_t slate_scalapack_set_panelthreads()
{
    int64_t max_panel_threads = 1;
    char* thrstr = std::getenv("SLATE_SCALAPACK_PANELTHREADS");
    if (thrstr) {
        max_panel_threads = (int64_t)strtol(thrstr, NULL, 0);
        if (max_panel_threads != 0) return max_panel_threads;
    }
    return std::max(omp_get_max_threads()/2, 1);
}

inline int64_t slate_scalapack_set_ib()
{
    int64_t ib = 0;
    char* ibstr = std::getenv("SLATE_SCALAPACK_IB");
    if (ibstr) {
        ib = (int64_t)strtol(ibstr, NULL, 0);
        if (ib != 0) return ib;
    }
    return 16;
}

inline int slate_scalapack_set_verbose()
{
    // set the SLATE verbose (specific to scalapack_api)
    int verbose = 0; // default
    char* verbosestr = std::getenv("SLATE_SCALAPACK_VERBOSE");
    if (verbosestr) {
        if (verbosestr[0] == '1')
            verbose = 1;
    }
    return verbose;
}

inline int64_t slate_scalapack_set_lookahead()
{
    int64_t la = 0;
    char* lastr = std::getenv("SLATE_SCALAPACK_LOOKAHEAD");
    if (lastr) {
        la = (int64_t)strtol(lastr, NULL, 0);
        if (la != 0) return la;
    }
    return 1;
}

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
