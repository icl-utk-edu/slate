//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the University of Tennessee nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------
// This research was supported by the Exascale Computing Project (17-SC-20-SC),
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//------------------------------------------------------------------------------
// For assistance with SLATE, email <slate-user@icl.utk.edu>.
// You can also join the "SLATE User" Google group by going to
// https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user,
// signing in with your Google credentials, and then clicking "Join group".
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
/// @file
///
#ifndef SLATE_MPI_HH
#define SLATE_MPI_HH

#ifndef SLATE_NO_MPI
    #include <mpi.h>
#else

typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Group;
typedef int MPI_Request;
typedef int MPI_Status;
typedef int MPI_Op;

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

#endif // SLATE_MPI_HH
