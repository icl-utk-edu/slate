// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/internal/util.hh"
#include "internal/internal_util.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// [internal]
/// Computes the power function for integer arguments.
///
template <typename T>
T pow(T base, T exp)
{
    T pow = 1;
    for (T i = 0; i < exp; ++i)
        pow *= base;

    return pow;
}

//------------------------------
// explicit instantiation
template
int pow<int>(int base, int exp);

//------------------------------------------------------------------------------
/// [internal]
/// Implememts a custom MPI reduction that propagates NaNs.
///
void mpi_max_nan(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype)
{
    if (*datatype == MPI_DOUBLE) {
        double* x = (double*) invec;
        double* y = (double*) inoutvec;
        for (int i = 0; i < *len; ++i)
            y[i] = max_nan(x[i], y[i]);
    }
    else if (*datatype == MPI_FLOAT) {
        float* x = (float*) invec;
        float* y = (float*) inoutvec;
        for (int i = 0; i < *len; ++i)
            y[i] = max_nan(x[i], y[i]);
    }
}

} // namespace internal
} // namespace slate
