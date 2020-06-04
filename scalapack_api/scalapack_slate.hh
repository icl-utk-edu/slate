//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the University of Tennessee nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------
// This research was supported by the Exascale Computing Project (17-SC-20-SC),
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//------------------------------------------------------------------------------
// For assistance with SLATE, email <slate-user@icl.utk.edu>.
// You can also join the "SLATE User" Google group by going to
// https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user,
// signing in with your Google credentials, and then clicking "Join group".
//------------------------------------------------------------------------------

#ifndef SLATE_SCALAPACK_API_COMMON_HH
#define SLATE_SCALAPACK_API_COMMON_HH

// get BLAS_FORTRAN_NAME and blas_int
#include "blas_fortran.hh"

#include "slate/slate.hh"
#include <complex>

namespace slate {
namespace scalapack_api {

#define logprintf(fmt, ...)                                             \
    do { fprintf(stderr, "%s:%d %s(): " fmt, __FILE__, __LINE__, __func__, __VA_ARGS__); } while (0)

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

template< typename scalar_t >
inline slate::Matrix<scalar_t> slate_scalapack_submatrix(int Am, int An, slate::Matrix<scalar_t>& A, int ia, int ja, int* desca)
{
    //logprintf("Am %d An %d ia %d ja %d desc_MB(desca) %d desc_NB(desca) %d A.m() %d A.n() %d LLD_ %d %d \n", Am, An, ia, ja, desc_MB(desca), desc_NB(desca), A.m(), A.n());
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


//------------------------------------------------------------------------------
// BLAS thread management.
// Note this is duplicated in the testing module
#ifdef SLATE_WITH_MKL
#include <mkl_service.h>
inline int slate_set_num_blas_threads(const int nt)
{
    int old = mkl_get_max_threads();
    mkl_set_num_threads(nt);
    return old;
}
#else
inline int slate_set_num_blas_threads(const int nt) { return -1; }
#endif


} // namespace scalapack_api
} // namespace slate

#endif // SLATE_SCALAPACK_API_COMMON_HH
