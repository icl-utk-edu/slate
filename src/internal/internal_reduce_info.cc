// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/types.hh"
#include "internal/internal.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// MPI reduce info, used in getrf, hetrf, etc.
///
/// @param[in,out] info
///     On input, status on each rank; 0 means no error.
///     On output, smallest non-zero info among all MPI ranks,
///     or zero if info is zero on all MPI ranks.
///
/// @param[in] mpi_comm
///     MPI communicator.
///
void reduce_info( int64_t* info, MPI_Comm mpi_comm )
{
    // Use int64_max as a sentinel to indicate no error.
    int64_t send_info = *info;
    if (send_info == 0)
        send_info = std::numeric_limits<int64_t>::max();

    slate_mpi_call(
        MPI_Allreduce( &send_info, info, 1, mpi_type<int64_t>::value, MPI_MIN,
                       mpi_comm ) );

    if (*info == std::numeric_limits<int64_t>::max())
        *info = 0;
}

} // namespace internal
} // namespace slate
