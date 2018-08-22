#include "blas.hh"

#include <mpi.h>
#include <string>
#include <cstdio>

template <typename scalar_t>
void print_matrix(
    int64_t mlocal, int64_t nlocal, scalar_t* A, int64_t lda,
    int p, int q, MPI_Comm comm, int width=8, int precision=4)
{
    using blas::real;
    using blas::imag;

    int mpi_rank;
    MPI_Comm_rank(comm, &mpi_rank);

    char buf[ 1024 ];
    std::string msg;

    width = std::max( width, precision + 3 );

    // loop over process rows & cols
    for (int prow = 0; prow < q; ++prow) {
        for (int pcol = 0; pcol < p; ++pcol) {
            int rank = prow + pcol*p;

            if (rank == mpi_rank) {
                snprintf( buf, sizeof(buf), "A%d_%d = [\n", prow, pcol );
                msg += buf;
                for (int64_t i = 0; i < mlocal; ++i) {
                    for (int64_t j = 0; j < nlocal; ++j) {
                        if (slate::is_complex<scalar_t>::value) {
                            snprintf( buf, sizeof(buf), " %*.*f + %*.*fi",
                                      width, precision, real( A[i + j*lda] ),
                                      width, precision, imag( A[i + j*lda] ) );
                        }
                        else {
                            snprintf( buf, sizeof(buf),
                                      " %*.*f",
                                      width, precision, real( A[i + j*lda] ) );
                        }
                        msg += buf;
                    }
                    msg += "\n";
                }
                msg += "]\n\n";

                if (mpi_rank != 0) {
                    // Send msg to root, which handles actual I/O.
                    int len = int( msg.size() );
                    MPI_Send( &len, 1, MPI_INT, 0, 0, comm );
                    MPI_Send( msg.c_str(), len, MPI_CHAR, 0, 0, comm );
                }
                else {
                    // Already on root, just print it.
                    printf( "%s", msg.c_str() );
                }
            }
            else if (mpi_rank == 0) {
                // Root receives msg and handles actual I/O.
                MPI_Status status;
                int len;
                MPI_Recv( &len, 1, MPI_INT, rank, 0, comm, &status );
                msg.resize( len );
                MPI_Recv( &msg[0], len, MPI_CHAR, rank, 0, comm, &status );
                printf( "%s", msg.c_str() );
            }
        }
    }
    if (mpi_rank == 0) {
        fflush( stdout );
    }
    MPI_Barrier( comm );
}
