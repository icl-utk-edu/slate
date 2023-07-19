// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
/// @file
///
#ifndef SLATE_OPENMP_HH
#define SLATE_OPENMP_HH

// In OpenMP, using "#pragma omp default(none)" to make programmers
// specify variable usage is good practice.  However some compilers
// and libraries contain hidden variables that cause unexpected
// warnings and errors.  SLATE uses a macro that can be defined at
// compile time for debug building and testing.
//
// For example:  CXXFLAGS += -Dslate_omp_default_none='default(none)'
//
#ifndef slate_omp_default_none
#define slate_omp_default_none
#endif

// Include OpenMP headers
//
// Note: There is no _OPENMP guard because SLATE requires OpenMP and
// checks for it at compile time.
//
// todo: On Intel SYCL platforms using both "-fsycl -fiopenmp" flags
// can leave the _OPENMP macro undefined.  There are 2 sweeps over the
// code, use ifdefs via __SYCL_DEVICE_ONLY__ macro to make code legal
// for the SYCL pass.
//
#include <omp.h>

// Defines a small class to wrap omp_set_max_active_levels()
#include "slate/internal/OmpSetMaxActiveLevels.hh"

#endif // SLATE_OPENMP_HH
