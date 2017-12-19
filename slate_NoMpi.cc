
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

int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root,
              MPI_Comm comm)
{
    assert(0);
}

int MPI_Comm_create_group(MPI_Comm comm, MPI_Group group, int tag,
                          MPI_Comm *newcomm)
{
    assert(0);
}

int MPI_Comm_free(MPI_Comm *comm)
{
    assert(0);
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

int MPI_Group_free(MPI_Group *group)
{
    assert(0);
}

int MPI_Group_incl(MPI_Group group, int n, const int ranks[],
                   MPI_Group *newgroup)
{
    assert(0);
}

int MPI_Group_translate_ranks(MPI_Group group1, int n, const int ranks1[],
                              MPI_Group group2, int ranks2[])
{
    assert(0);
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
