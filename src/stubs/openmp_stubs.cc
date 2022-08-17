// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
/// @file
///
#include "slate/internal/openmp.hh"

#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

int omp_get_initial_device()
{
    return -10;
}

int omp_get_max_threads()
{
    return 1;
}

int omp_get_num_devices()
{
    return 0;
}

int omp_get_num_threads(void)
{
    return 1;
}

int omp_get_thread_num(void)
{
    return 0;
}

double omp_get_wtime()
{
    struct timeval  time;
    struct timezone zone;

    gettimeofday(&time, &zone);

    double sec = time.tv_sec;
    double usec = time.tv_usec;

    return sec + usec/1000000.0;
}

void omp_destroy_lock(omp_lock_t* lock)
{
    return;
}

void omp_init_lock(omp_lock_t* lock)
{
    return;
}

void omp_set_lock(omp_lock_t* lock)
{
    return;
}

void omp_set_nested(int nested)
{
    return;
}

void omp_unset_lock(omp_lock_t* lock)
{
    return;
}

void omp_destroy_nest_lock(omp_nest_lock_t* lock)
{
    return;
}

void omp_init_nest_lock(omp_nest_lock_t* lock)
{
    return;
}

void omp_set_nest_lock(omp_nest_lock_t* lock)
{
    return;
}

void omp_unset_nest_lock(omp_nest_lock_t* lock)
{
    return;
}

#ifdef __cplusplus
}
#endif
