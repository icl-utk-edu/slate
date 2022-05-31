// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
/// @file
///
#include "slate/internal/mpi.hh"

#include <cassert>
#include <complex>

int* MPI_STATUS_IGNORE;

#ifdef __cplusplus
extern "C" {
#endif

int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype* newtype)
{
    assert(0);
}

int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    assert(count == 1);
    assert(op == MPI_MAX);

    switch (datatype) {
        case MPI_FLOAT:
            *(float*)recvbuf = *(float*)sendbuf;
            break;
        case MPI_DOUBLE:
            *(double*)recvbuf = *(double*)sendbuf;
            break;
        case MPI_C_COMPLEX:
            *(std::complex<float>*)recvbuf = *(std::complex<float>*)sendbuf;
            break;
        case MPI_C_DOUBLE_COMPLEX:
            *(std::complex<double>*)recvbuf = *(std::complex<double>*)sendbuf;
            break;
        default:
            assert(0);
    }
    return MPI_SUCCESS;
}

int MPI_Barrier(MPI_Comm comm)
{
    return MPI_SUCCESS;
}

int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root,
              MPI_Comm comm)
{
    return MPI_SUCCESS;
}

int MPI_Comm_create_group(MPI_Comm comm, MPI_Group group, int tag,
                          MPI_Comm* newcomm)
{
    return MPI_SUCCESS;
}

int MPI_Comm_free(MPI_Comm* comm)
{
    return MPI_SUCCESS;
}

int MPI_Comm_group(MPI_Comm comm, MPI_Group* group)
{
    return MPI_SUCCESS;
}

int MPI_Comm_rank(MPI_Comm comm, int* rank)
{
    *rank = 0;
    return MPI_SUCCESS;
}

int MPI_Comm_size(MPI_Comm comm, int* size)
{
    *size = 1;
    return MPI_SUCCESS;
}

MPI_Fint MPI_Comm_f2c(MPI_Comm comm)
{
    assert(0);
    return 0;
}

int MPI_Group_free(MPI_Group* group)
{
    assert(0);
}

int MPI_Group_incl(MPI_Group group, int n, const int ranks[],
                   MPI_Group* newgroup)
{
    assert(0);
}

int MPI_Group_translate_ranks(MPI_Group group1, int n, const int ranks1[],
                              MPI_Group group2, int ranks2[])
{
    assert(0);
}

int MPI_Init(int* argc, char*** argv)
{
    return MPI_SUCCESS;
}

int MPI_Initialized(int* flag)
{
    *flag = 1;
    return MPI_SUCCESS;
}

int MPI_Init_thread(int* argc, char*** argv, int required, int* provided)
{
    *provided = MPI_THREAD_MULTIPLE;
    return MPI_SUCCESS;
}

int MPI_Irecv(void* buf, int count, MPI_Datatype datatype, int source,
              int tag, MPI_Comm comm, MPI_Request* request)
{
    assert(0);
}

int MPI_Isend(const void* buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request* request)
{
    assert(0);
}

int MPI_Recv(void* buf, int count, MPI_Datatype datatype, int source,
             int tag, MPI_Comm comm, MPI_Status* status)
{
    assert(0);
}

int MPI_Op_create(MPI_User_function* user_fn, int commute, MPI_Op* op)
{
    assert(0);
}

int MPI_Op_free(MPI_Op* op)
{
    assert(0);
}

int MPI_Reduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
               MPI_Op op, int root, MPI_Comm comm)
{
    assert(count == 1);
    assert(op == MPI_MAX);

    switch (datatype) {
        case MPI_FLOAT:
            *(float*)recvbuf = *(float*)sendbuf;
            break;
        case MPI_DOUBLE:
            *(double*)recvbuf = *(double*)sendbuf;
            break;
        case MPI_C_COMPLEX:
            *(std::complex<float>*)recvbuf = *(std::complex<float>*)sendbuf;
            break;
        case MPI_C_DOUBLE_COMPLEX:
            *(std::complex<double>*)recvbuf = *(std::complex<double>*)sendbuf;
            break;
        default:
            assert(0);
    }
    return MPI_SUCCESS;
}

int MPI_Request_free(MPI_Request* request)
{
    assert(0);
}

int MPI_Send(const void* buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm)
{
    assert(0);
}

int MPI_Sendrecv(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                 int dest, int sendtag, void* recvbuf, int recvcount,
                 MPI_Datatype recvtype, int source, int recvtag,
                 MPI_Comm comm, MPI_Status* status)
{
    assert(0);
}

int MPI_Type_commit(MPI_Datatype* datatype)
{
    assert(0);
}

int MPI_Type_free(MPI_Datatype* datatype)
{
    assert(0);
}

int MPI_Type_vector(int count, int blocklength, int stride,
                    MPI_Datatype oldtype, MPI_Datatype* newtype)
{
    assert(0);
}

int MPI_Wait(MPI_Request* request, MPI_Status* status)
{
    assert(0);
}

int MPI_Waitall(int count, MPI_Request requests[], MPI_Status statuses[])
{
    assert(0);
}

int MPI_Error_string(int errorcode, char* string, int* resultlen)
{
    assert(0);
}

int MPI_Finalize(void)
{
    return MPI_SUCCESS;
}
#ifdef __cplusplus
}
#endif
