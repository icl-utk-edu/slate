// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
/// @file
///
#ifndef SLATE_INTERNAL_COMM_HH
#define SLATE_INTERNAL_COMM_HH

#include <list>
#include <set>

#include "slate/internal/mpi.hh"

namespace slate {
namespace internal {

MPI_Comm commFromSet(const std::set<int>& bcast_set,
                     MPI_Comm mpi_comm, MPI_Group mpi_group,
                     const int in_rank, int& out_rank, int tag = 0);

void cubeBcastPattern(int size, int rank, int radix,
                      std::list<int>& recv_from, std::list<int>& send_to);

void cubeReducePattern(int size, int rank, int radix,
                       std::list<int>& recv_from, std::list<int>& send_to);

} // namespace internal
} // namespace slate

#endif // SLATE_INTERNAL_COMM_HH
