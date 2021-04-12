// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Exception.hh"
#include "slate/internal/comm.hh"
#include "internal/internal_util.hh"
#include "slate/internal/Trace.hh"

#include <cassert>
#include <vector>

namespace slate {
namespace internal {

MPI_Comm commFromSet(const std::set<int>& bcast_set,
                     MPI_Comm mpi_comm, MPI_Group mpi_group,
                     const int in_rank, int& out_rank)
{
    // Convert the set of ranks to a vector.
    std::vector<int> bcast_vec(bcast_set.begin(), bcast_set.end());

    // Create the broadcast group.
    MPI_Group bcast_group;
    #pragma omp critical(slate_mpi)
    slate_mpi_call(
        MPI_Group_incl(mpi_group, bcast_vec.size(), bcast_vec.data(),
                       &bcast_group));

    // Create a broadcast communicator.
    int tag = 0;
    MPI_Comm bcast_comm;
    #pragma omp critical(slate_mpi)
    {
        trace::Block trace_block("MPI_Comm_create_group");
        slate_mpi_call(
            MPI_Comm_create_group(mpi_comm, bcast_group, tag, &bcast_comm));
    }
    assert(bcast_comm != MPI_COMM_NULL);

    // Translate the input rank.
    #pragma omp critical(slate_mpi)
    slate_mpi_call(
        MPI_Group_translate_ranks(mpi_group, 1, &in_rank,
                                  bcast_group, &out_rank));

    return bcast_comm;
}

//------------------------------------------------------------------------------
/// [internal]
/// Implements a hypercube broadcast pattern. For a given rank, finds the rank
/// to receive from and the list of ranks to forward to. Assumes rank 0 as the
/// root of the broadcast.
///
/// @param[in] size
///     Number of ranks participating in the broadcast.
///
/// @param[in] rank
///     Rank of the local process.
///
/// @param[in] radix
///     Dimension of the cube.
///
/// @param[out] recv_rank
///     List containing the the rank to receive from.
///     Empty list for rank 0.
///
/// @param[out] send_to
///     List of ranks to forward to.
///
void cubeBcastPattern(int size, int rank, int radix,
                      std::list<int>& recv_from, std::list<int>& send_to)
{
    //-------------------------------------------
    // Find the cube's and the rank's attributes:

    // Find the number of cube's dimensions.
    int num_dimensions = 1;
    int max_rank = size-1;
    while ((max_rank /= radix) > 0)
        ++num_dimensions;

    int dimension; // how deep the rank is in the cube
    int position;  // position in the last dimension
    int stride;    // stride of the last dimension

    // Find the rank's dimension, position, and stride.
    int radix_pow = pow(radix, num_dimensions-1);
    dimension = 0;
    while (rank%radix_pow != 0) {
        ++dimension;
        radix_pow /= radix;
    }
    stride = pow(radix, num_dimensions-dimension-1);
    position = rank%pow(radix, num_dimensions-dimension)/stride;

    //--------------------------------------
    // Find the origin and the destinations.

    // Unless root, receive from the predecessor.
    if (rank != 0)
        recv_from.push_back(rank-stride);

    // If not on the edge and successor exists, send to the successor.
    if (position < radix-1 && rank+stride < size)
        send_to.push_back(rank+stride);

    // Forward to all lower dimensions.
    for (int dim = dimension+1; dim < num_dimensions; ++dim) {
        stride /= radix;
        if (rank+stride < size)
            send_to.push_back(rank+stride);
    }
}

//------------------------------------------------------------------------------
///
void cubeReducePattern(int size, int rank, int radix,
                       std::list<int>& recv_from, std::list<int>& send_to)
{
    cubeBcastPattern(size, rank, radix, send_to, recv_from);
}


// A gatherv implementation that uses tags
void tagged_gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype,
                    int root, int tag, MPI_Comm comm) {
    //MPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm);

    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    if (rank != root) {
        if (sendcount != 0) {
            MPI_Send(sendbuf, sendcount, sendtype, root, tag, comm);
        }
        return;
    }

    int request_count = rank == root ? size+1 : 1;
    std::vector<MPI_Request> requests_vect (request_count);
    MPI_Request* requests = requests_vect.data();

    if (rank == root) {
        for (int i = 0; i < size; ++i) {
            if (recvcounts[i] != 0) {
                void *recvbuf_i = (void*)(displs[i] + (char*)recvbuf);
                MPI_Irecv(recvbuf_i, recvcounts[i], recvtype,
                          i, tag, comm, &requests[i+1]);
            } else {
                requests[i+1] = MPI_REQUEST_NULL;
            }
        }
    }

    if (sendcount != 0) {
        MPI_Isend(sendbuf, sendcount, sendtype, root, tag, comm, &requests[0]);
    } else {
        requests[0] = MPI_REQUEST_NULL;
    }

    MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
}

// A scatterv implementation that uses tags
void tagged_scatterv(const void *sendbuf, const int sendcounts[], const int displs[], MPI_Datatype sendtype,
                     void *recvbuf, int recvcount, MPI_Datatype recvtype,
                     int root, int tag, MPI_Comm comm) {
    //MPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm);

    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    if (rank != root) {
        if (recvcount != 0) {
            MPI_Recv(recvbuf, recvcount, recvtype, root, tag, comm, MPI_STATUS_IGNORE);
        }
        return;
    }

    int request_count = rank == root ? size+1 : 1;
    std::vector<MPI_Request> requests_vect (request_count);
    MPI_Request* requests = requests_vect.data();

    if (rank == root) {
        for (int i = 0; i < size; ++i) {
            if (sendcounts[i] != 0) {
                void *sendbuf_i = (void*)(displs[i] + (char*)sendbuf);
                MPI_Isend(sendbuf_i, sendcounts[i], sendtype,
                          i, tag, comm, &requests[i+1]);
            } else {
                requests[i+1] = MPI_REQUEST_NULL;
            }
        }
    }

    if (recvcount != 0) {
        MPI_Irecv(recvbuf, recvcount, recvtype, root, tag, comm, &requests[0]);
    } else {
        requests[0] = MPI_REQUEST_NULL;
    }

    MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
}

} // namespace internal
} // namespace slate
