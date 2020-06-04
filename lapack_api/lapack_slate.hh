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

#ifndef SLATE_LAPACK_API_COMMON_HH
#define SLATE_LAPACK_API_COMMON_HH

#include "slate/slate.hh"
#include <complex>

#ifdef SLATE_WITH_MKL
    extern "C" int MKL_Set_Num_Threads(int nt);
#endif

namespace slate {
namespace lapack_api {

#define logprintf(fmt, ...)                                             \
    do { fprintf(stdout, "slate_lapack_api: " fmt, __VA_ARGS__); } while (0)

//    do { fprintf(stdout, "%s:%d %s(): " fmt, __FILE__, __LINE__, __func__, __VA_ARGS__); } while (0)

inline int slate_lapack_set_num_blas_threads(const int nt)
{
    #ifdef SLATE_WITH_MKL
    return MKL_Set_Num_Threads(nt);
    #endif
    return 1;
}

inline char slate_lapack_scalar_t_to_char(int* a) { return 'i'; }
inline char slate_lapack_scalar_t_to_char(float* a) { return 's'; }
inline char slate_lapack_scalar_t_to_char(double* a) { return 'd'; }
inline char slate_lapack_scalar_t_to_char(std::complex<float>* a) { return 'c'; }
inline char slate_lapack_scalar_t_to_char(std::complex<double>* a) { return 'z'; }

inline slate::Target slate_lapack_set_target()
{
    // set the SLATE default computational target
    slate::Target target = slate::Target::HostTask;
    char* targetstr = std::getenv("SLATE_LAPACK_TARGET");
    if (targetstr) {
        char targetchar = (char)(toupper(targetstr[4]));
        if (targetchar=='T') target = slate::Target::HostTask;
        else if (targetchar=='N') target = slate::Target::HostNest;
        else if (targetchar=='B') target = slate::Target::HostBatch;
        else if (targetchar=='C') target = slate::Target::Devices;
        return target;
    }
    // todo: should the device be set to cude automatically
    int cudadevcount;
    if (cudaGetDeviceCount(&cudadevcount)==cudaSuccess && cudadevcount>0)
        target = slate::Target::Devices;
    return target;
}

inline int64_t slate_lapack_set_panelthreads()
{
    int64_t max_panel_threads = 1;
    int max_omp_threads = 1;
    char* thrstr = std::getenv("SLATE_LAPACK_PANELTHREADS");
    if (thrstr) {
        max_panel_threads = (int64_t)strtol(thrstr, NULL, 0);
        if (max_panel_threads!=0) return max_panel_threads;
    }
    max_omp_threads = omp_get_max_threads();
    return std::max(max_omp_threads/4, 1);
}

inline int64_t slate_lapack_set_ib()
{
    int64_t ib = 0;
    char* ibstr = std::getenv("SLATE_LAPACK_IB");
    if (ibstr) {
        ib = (int64_t)strtol(ibstr, NULL, 0);
        if (ib!=0) return ib;
    }
    return 16;
}

inline int slate_lapack_set_verbose()
{
    // set the SLATE verbose (specific to lapack_api)
    int verbose = 0; // default
    char* verbosestr = std::getenv("SLATE_LAPACK_VERBOSE");
    if (verbosestr) {
        if (verbosestr[0]=='1')
            verbose = 1;
    }
    return verbose;
}


inline int64_t slate_lapack_set_nb(slate::Target target)
{
    // set nb if not already done
    int64_t nb = 0;
    char* nbstr = std::getenv("SLATE_LAPACK_NB");
    if (nbstr) {
        nb = (int64_t)strtol(nbstr, NULL, 0);
        if (nb!=0) return nb;
    }
    if (nb==0 && target==slate::Target::Devices)
        return 1024;
    if (nb==0 && target==slate::Target::HostTask)
        return 512;
    return 256;
}

} // namespace lapack_api
} // namespace slate

#endif // SLATE_LAPACK_API_COMMON_HH
