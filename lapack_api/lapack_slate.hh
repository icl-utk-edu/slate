// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_LAPACK_API_COMMON_HH
#define SLATE_LAPACK_API_COMMON_HH

// get BLAS_FORTRAN_NAME and blas_int
#include "blas/fortran.h"

#include "slate/slate.hh"

#include <complex>

namespace slate {
namespace lapack_api {

#define logprintf(fmt, ...) \
    do { fprintf(stdout, "slate_lapack_api: " fmt, __VA_ARGS__); } while (0)

//    do { fprintf(stdout, "%s:%d %s(): " fmt, __FILE__, __LINE__, __func__, __VA_ARGS__); } while (0)

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
        if (targetchar == 'T') target = slate::Target::HostTask;
        else if (targetchar == 'N') target = slate::Target::HostNest;
        else if (targetchar == 'B') target = slate::Target::HostBatch;
        else if (targetchar == 'C') target = slate::Target::Devices;
        return target;
    }
    // todo: should the device be set to cude automatically

    int num_devices = blas::get_device_count();
    if (num_devices > 0)
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
        if (max_panel_threads != 0) return max_panel_threads;
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
        if (ib != 0) return ib;
    }
    return 16;
}

inline int slate_lapack_set_verbose()
{
    // set the SLATE verbose (specific to lapack_api)
    int verbose = 0; // default
    char* verbosestr = std::getenv("SLATE_LAPACK_VERBOSE");
    if (verbosestr) {
        if (verbosestr[0] == '1')
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
        if (nb != 0) return nb;
    }
    if (nb == 0 && target == slate::Target::Devices)
        return 1024;
    if (nb == 0 && target == slate::Target::HostTask)
        return 512;
    return 256;
}

} // namespace lapack_api
} // namespace slate

#endif // SLATE_LAPACK_API_COMMON_HH
