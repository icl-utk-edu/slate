
#include "slate_NoMpi.hh"

#include <cassert>

int *MPI_STATUS_IGNORE;

#ifdef __cplusplus
extern "C" {
#endif

int MPI_Barrier(MPI_Comm comm)
{
    return MPI_SUCCESS;
}

int MPI_Comm_group(MPI_Comm comm, MPI_Group *group)
{
    return MPI_SUCCESS;
}

int MPI_Comm_rank(MPI_Comm comm, int *rank)
{
    *rank = 0;
    return MPI_SUCCESS;
}

int MPI_Comm_size(MPI_Comm comm, int *size)
{
    *size = 1;
    return MPI_SUCCESS;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided)
{
    *provided = MPI_THREAD_MULTIPLE;
    return MPI_SUCCESS;
}

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,
              int tag, MPI_Comm comm, MPI_Request *request)
{
    assert(0);
}

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request *request)
{
    assert(0);
}

int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source,
             int tag, MPI_Comm comm, MPI_Status *status)
{
    assert(0);
}

int MPI_Request_free(MPI_Request *request)
{
    assert(0);
}

int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm)
{
    assert(0);
}

int MPI_Wait(MPI_Request *request, MPI_Status *status)
{
    assert(0);
}

#ifdef __cplusplus
}
#endif
