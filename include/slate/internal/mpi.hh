// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
/// @file
///
#ifndef SLATE_MPI_HH
#define SLATE_MPI_HH

#include "slate/Exception.hh"

#ifndef SLATE_NO_MPI
    #include <mpi.h>
#else

typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Group;
typedef int MPI_Request;
typedef int MPI_Status;
typedef int MPI_Op;
typedef int MPI_Fint;

enum {
    MPI_COMM_NULL,
    MPI_COMM_WORLD,

    MPI_BYTE,
    MPI_CHAR,
    MPI_INT,
    MPI_UNSIGNED,
    MPI_LONG,
    MPI_FLOAT,
    MPI_FLOAT_INT,
    MPI_DOUBLE,
    MPI_DOUBLE_INT,
    MPI_C_COMPLEX,
    MPI_C_DOUBLE_COMPLEX,

    MPI_INT64_T,

    MPI_2INT,

    MPI_MAX,
    MPI_MAXLOC,
    MPI_SUM,

    MPI_SUCCESS,
    MPI_THREAD_MULTIPLE,
    MPI_THREAD_SERIALIZED,
};

#define MPI_MAX_ERROR_STRING 512

extern int* MPI_STATUS_IGNORE;
#define MPI_STATUSES_IGNORE NULL
#define MPI_REQUEST_NULL 0

typedef void (MPI_User_function) (void* a,
                                  void* b, int* len, MPI_Datatype* type);

#ifdef __cplusplus
extern "C" {
#endif

int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype* newtype);

int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

int MPI_Barrier(MPI_Comm comm);

int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root,
              MPI_Comm comm);

int MPI_Comm_create_group(MPI_Comm comm, MPI_Group group, int tag,
                          MPI_Comm* newcomm);

int MPI_Comm_free(MPI_Comm* comm);
int MPI_Comm_group(MPI_Comm comm, MPI_Group* group);
int MPI_Comm_rank(MPI_Comm comm, int* rank);
int MPI_Comm_size(MPI_Comm comm, int* size);
MPI_Fint MPI_Comm_f2c(MPI_Comm comm);

int MPI_Group_free(MPI_Group* group);

int MPI_Group_incl(MPI_Group group, int n, const int ranks[],
                   MPI_Group* newgroup);

int MPI_Group_translate_ranks(MPI_Group group1, int n, const int ranks1[],
                              MPI_Group group2, int ranks2[]);

int MPI_Init(int* argc, char*** argv);

int MPI_Init_thread(int* argc, char*** argv, int required, int* provided);

int MPI_Initialized(int* flag);

int MPI_Irecv(void* buf, int count, MPI_Datatype datatype, int source,
              int tag, MPI_Comm comm, MPI_Request* request);

int MPI_Isend(const void* buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request* request);

int MPI_Recv(void* buf, int count, MPI_Datatype datatype, int source,
             int tag, MPI_Comm comm, MPI_Status* status);

int MPI_Op_create(MPI_User_function *user_fn, int commute, MPI_Op *op);

int MPI_Op_free(MPI_Op *op);

int MPI_Reduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
               MPI_Op op, int root, MPI_Comm comm);

int MPI_Request_free(MPI_Request* request);

int MPI_Send(const void* buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm);

int MPI_Sendrecv(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                 int dest, int sendtag, void* recvbuf, int recvcount,
                 MPI_Datatype recvtype, int source, int recvtag,
                 MPI_Comm comm, MPI_Status *status);

int MPI_Type_commit(MPI_Datatype* datatype);

int MPI_Type_free(MPI_Datatype* datatype);

int MPI_Type_vector(int count, int blocklength, int stride,
                    MPI_Datatype oldtype, MPI_Datatype* newtype);

int MPI_Wait(MPI_Request* request, MPI_Status* status);

int MPI_Waitall(int count, MPI_Request requests[], MPI_Status statuses[]);

int MPI_Error_string(int errorcode, char* string, int* resultlen);

int MPI_Finalize(void);

#ifdef __cplusplus
}
#endif

#endif // SLATE_NO_MPI

namespace slate {

//------------------------------------------------------------------------------
/// Exception class for slate_mpi_call().
class MpiException : public Exception {
public:
    MpiException(const char* call,
                 int code,
                 const char* func,
                 const char* file,
                 int line)
        : Exception()
    {
        char string[MPI_MAX_ERROR_STRING] = "unknown error";
        int resultlen;
        MPI_Error_string(code, string, &resultlen);

        what(std::string("SLATE MPI ERROR: ")
             + call + " failed: " + string
             + " (" + std::to_string(code) + ")",
             func, file, line);
    }
};

/// Throws an MpiException if the MPI call fails.
/// Example:
///
///     try {
///         slate_mpi_call( MPI_Barrier( MPI_COMM_WORLD ) );
///     }
///     catch (MpiException& e) {
///         ...
///     }
///
#define slate_mpi_call(call) \
    do { \
        int slate_mpi_call_ = call; \
        if (slate_mpi_call_ != MPI_SUCCESS) \
            throw slate::MpiException( \
                #call, slate_mpi_call_, __func__, __FILE__, __LINE__); \
    } while(0)

} // namespace slate

#endif // SLATE_MPI_HH
