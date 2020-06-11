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
#include "slate/internal/mpi.hh"

#include <cassert>
#include <complex>

int* MPI_STATUS_IGNORE;

#ifdef __cplusplus
extern "C" {
#endif

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
