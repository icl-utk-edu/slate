
#ifndef SLATE_NO_MPI_HH
#define SLATE_NO_MPI_HH

#include <cassert>

typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Group;
typedef int MPI_Request;
typedef int MPI_Status;

enum {
    MPI_COMM_NULL,
    MPI_COMM_WORLD,
    MPI_DOUBLE,
    MPI_SUCCESS,
    MPI_THREAD_MULTIPLE
};

int *MPI_STATUS_IGNORE;

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
int MPI_Barrier(MPI_Comm comm)
{
    return MPI_SUCCESS;
}

//------------------------------------------------------------------------------
int MPI_Comm_group(MPI_Comm comm, MPI_Group *group)
{
    return MPI_SUCCESS;
}

//------------------------------------------------------------------------------
int MPI_Comm_rank(MPI_Comm comm, int *rank)
{
    *rank = 0;
    return MPI_SUCCESS;
}

//------------------------------------------------------------------------------
int MPI_Comm_size(MPI_Comm comm, int *size)
{
    *size = 1;
    return MPI_SUCCESS;
}

//------------------------------------------------------------------------------
int MPI_Init_thread(int *argc, char ***argv, int required, int *provided)
{
    *provided = MPI_THREAD_MULTIPLE;
    return MPI_SUCCESS;
}

//------------------------------------------------------------------------------
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,
              int tag, MPI_Comm comm, MPI_Request *request)
{
    assert(0);
}

//------------------------------------------------------------------------------
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request *request)
{
    assert(0);
}
//------------------------------------------------------------------------------
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source,
             int tag, MPI_Comm comm, MPI_Status *status)
{
    assert(0);
}

//------------------------------------------------------------------------------
int MPI_Request_free(MPI_Request *request)
{
    assert(0);
}

//------------------------------------------------------------------------------
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm)
{
    assert(0);
}

//------------------------------------------------------------------------------
int MPI_Wait(MPI_Request *request, MPI_Status *status)
{
    assert(0);
}

#ifdef __cplusplus
}
#endif

#endif // SLATE_NO_MPI_HH
