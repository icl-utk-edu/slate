// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
/// @file
///
#ifndef SLATE_OPENMP_HH
#define SLATE_OPENMP_HH

#ifndef slate_omp_default_none
#define slate_omp_default_none
#endif

#ifdef _OPENMP
    #include <omp.h>
#else

typedef int omp_lock_t;
typedef int omp_nest_lock_t;

#ifdef __cplusplus
extern "C" {
#endif

int omp_get_initial_device();
int omp_get_max_threads();
int omp_get_num_devices();
int omp_get_num_threads(void);
int omp_get_thread_num(void);

double omp_get_wtime();

void omp_destroy_lock(omp_lock_t* lock);
void omp_init_lock(omp_lock_t* lock);
void omp_set_lock(omp_lock_t* lock);
void omp_set_nested(int nested);
void omp_unset_lock(omp_lock_t* lock);

void omp_destroy_nest_lock(omp_nest_lock_t* lock);
void omp_init_nest_lock(omp_nest_lock_t* lock);
void omp_set_nest_lock(omp_nest_lock_t* lock);
void omp_unset_nest_lock(omp_nest_lock_t* lock);

#ifdef __cplusplus
}
#endif

#endif // not _OPENMP

// Defines a small class to wrap omp_set_max_active_levels()
#include "slate/internal/OmpSetMaxActiveLevels.hh"

#endif // SLATE_OPENMP_HH
