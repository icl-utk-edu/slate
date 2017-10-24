
#ifndef SLATE_NO_MPI_HH
#define SLATE_NO_MPI_HH

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

extern int *MPI_STATUS_IGNORE;

#ifdef __cplusplus
extern "C" {
#endif

int MPI_Barrier(MPI_Comm comm);

int MPI_Comm_group(MPI_Comm comm, MPI_Group *group);
int MPI_Comm_rank(MPI_Comm comm, int *rank);
int MPI_Comm_size(MPI_Comm comm, int *size);

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided);

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,
              int tag, MPI_Comm comm, MPI_Request *request);

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request *request);

int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source,
             int tag, MPI_Comm comm, MPI_Status *status);

int MPI_Request_free(MPI_Request *request);

int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm);

int MPI_Wait(MPI_Request *request, MPI_Status *status);

#ifdef __cplusplus
}
#endif

#endif // SLATE_NO_MPI_HH
