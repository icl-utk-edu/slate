// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_CONFIG_HH
#define SLATE_CONFIG_HH

#include <string.h>
#include <stdlib.h>

namespace slate {

//------------------------------------------------------------------------------
/// Query whether MPI is GPU-aware.
class GPU_Aware_MPI
{
public:
    /// @see bool gpu_aware_mpi()
    static bool value()
    {
        return instance().gpu_aware_mpi_;
    }

    /// @see void gpu_aware_mpi( bool )
    static void value( bool val )
    {
        instance().gpu_aware_mpi_ = val;
    }

private:
    /// @return GPU_Aware_MPI singleton.
    /// Uses thread-safe Scott Meyers' singleton to query on first call only.
    static GPU_Aware_MPI& instance()
    {
        static GPU_Aware_MPI instance_;
        return instance_;
    }

    /// Constructor checks $SLATE_GPU_AWARE_MPI.
    GPU_Aware_MPI()
    {
        const char* env = getenv( "SLATE_GPU_AWARE_MPI" );
        gpu_aware_mpi_ = env != nullptr
                         && (strcmp( env, "" ) == 0
                             || strcmp( env, "1" ) == 0);
    }

    //----------------------------------------
    // Data

    /// Cached value whether MPI is GPU-aware.
    bool gpu_aware_mpi_;
};

//------------------------------------------------------------------------------
/// @return true if MPI is GPU-aware.
/// Initially checks if environment variable $SLATE_GPU_AWARE_MPI is set
/// and either empty or 1. Can be overridden by gpu_aware_mpi( bool ).
/// In the future, could also check
/// `MPIX_GPU_query_support` (MPICH) or
/// `MPIX_Query_cuda_support` (Open MPI).
inline bool gpu_aware_mpi()
{
    return GPU_Aware_MPI::value();
}

//------------------------------------------------------------------------------
/// Set whether MPI is GPU-aware. Overrides $SLATE_GPU_AWARE_MPI.
/// @param[in] value: true if MPI is GPU-aware.
inline void gpu_aware_mpi( bool value )
{
    return GPU_Aware_MPI::value( value );
}

}  // namespace slate

#endif // SLATE_CONFIG_HH
