#include "blas.hh"

#include <slate_mpi.hh>
#include <string>
#include <cstdio>

#include <slate_Matrix.hh>
#include <slate_BaseTrapezoidMatrix.hh>
#include <slate_BandMatrix.hh>

//------------------------------------------------------------------------------
/// Print value to a buffer buf of length buf_len.
/// Prints with format %w.pf, where w = width and p = precision.
/// For complex values, prints both real and imaginary parts.
/// If a real or complex value is exactly an integer, it is printed with
/// format %v.0f where v = w-p, i.e., with no digits after decimal point.
template <typename scalar_t>
void snprintf_value(
    char* buf, size_t buf_len, int width, int precision, scalar_t value)
{
    using blas::real;
    using blas::imag;

    if (value == scalar_t( int( real( value )))) {
        // exactly integer, print without digits after decimal point
        if (slate::is_complex<scalar_t>::value) {
            snprintf( buf, buf_len, " %#*.0f%*s   %*s",
                      width-precision, real(value), precision, "",
                      width, "" );
        }
        else {
            snprintf( buf, buf_len,
                      " %#*.0f%*s",
                      width-precision, real(value), precision, "" );
        }
    }
    else {
        // general case
        if (slate::is_complex<scalar_t>::value) {
            snprintf( buf, buf_len, " %*.*f + %*.*fi",
                      width, precision, real(value),
                      width, precision, imag(value) );
        }
        else {
            snprintf( buf, buf_len,
                      " %*.*f",
                      width, precision, real(value) );
        }
    }
}

//------------------------------------------------------------------------------
/// Print a ScaLAPACK distributed matrix.
/// Prints each rank's data as a contiguous block, numbered by the block row &
/// column indices. Rank 0 does the printing.
///
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
                        snprintf_value( buf, sizeof(buf), width, precision, A[i + j*lda] );
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

//------------------------------------------------------------------------------
/// Print a SLATE distributed matrix.
/// Rank 0 does the printing, and must have enough memory to fit one entire
/// block row of the matrix.
///
template <typename scalar_t>
void print_matrix(
    const char* label,
    slate::Matrix<scalar_t>& A, int width=8, int precision=4)
{
    using blas::real;
    using blas::imag;

    int mpi_rank = A.mpiRank();
    MPI_Comm comm = A.mpiComm();

    width = std::max( width, precision + 3 );

    char buf[ 1024 ];
    std::string msg = label;
    msg += " = [\n";

    for (int64_t i = 0; i < A.mt(); ++i) {
        // gather block row to rank 0
        for (int64_t j = 0; j < A.nt(); ++j) {
            int tile_rank = A.tileRank(i, j);
            if (tile_rank != 0) {
                if (A.tileIsLocal(i, j)) {
                    auto T = A(i, j);
                    T.send( 0, comm );
                }
                else if (mpi_rank == 0) {
                    A.tileInsert( i, j );
                    A(i, j).recv( tile_rank, comm );
                }
            }
        }

        if (mpi_rank == 0) {
            // print block row
            for (int64_t ti = 0; ti < A.tileMb(i); ++ti) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    auto T = A(i, j);
                    for (int64_t tj = 0; tj < A.tileNb(j); ++tj) {
                        snprintf_value( buf, sizeof(buf), width, precision, T(ti, tj) );
                        msg += buf;
                    }
                    if (j < A.nt()-1)
                        msg += "    ";
                    else
                        msg += "\n";
                }
            }
            if (i < A.mt()-1)
                msg += "\n";
            else
                msg += "]\n";
            printf( "%s", msg.c_str() );
            msg.clear();

            // cleanup data
            for (int64_t j = 0; j < A.nt(); ++j) {
                if (! A.tileIsLocal(i, j)) {
                    A.tileErase(i, j);
                }
            }
        }
    }

    MPI_Barrier( comm );
}

//------------------------------------------------------------------------------
/// Print a SLATE distributed matrix.
/// Rank 0 does the printing, and must have enough memory to fit one entire
/// block row of the matrix.
///
template <typename scalar_t>
void print_matrix(
    const char* label,
    slate::BandMatrix<scalar_t>& A, int width=8, int precision=4)
{
    using blas::real;
    using blas::imag;

    int mpi_rank = A.mpiRank();
    MPI_Comm comm = A.mpiComm();

    width = std::max( width, precision + 3 );

    char buf[ 1024 ];
    std::string msg = label;
    msg += " = [\n";

    // for tiles outside bandwidth
    char outside[ 80 ];
    if (slate::is_complex<scalar_t>::value) {
        snprintf( outside, sizeof(outside), " %*f   %*s ",
                  width, NAN, width, "" );
    }
    else {
        snprintf( outside, sizeof(outside), " %*f",
                  width, NAN );
    }

    // todo: initially, assume fixed size, square tiles for simplicity
    int64_t kl = slate::ceildiv( A.lowerBandwidth(), A.tileNb(0) );
    int64_t ku = slate::ceildiv( A.upperBandwidth(), A.tileNb(0) );

    for (int64_t i = 0; i < A.mt(); ++i) {
        // gather block row to rank 0
        for (int64_t j = 0; j < A.nt(); ++j) {
            if (-kl <= j-i && j-i <= ku) { // inside bandwidth
                int tile_rank = A.tileRank(i, j);
                if (tile_rank != 0) {
                    if (A.tileIsLocal(i, j)) {
                        auto T = A(i, j);
                        T.send( 0, comm );
                    }
                    else if (mpi_rank == 0) {
                        A.tileInsert( i, j );
                        A(i, j).recv( tile_rank, comm );
                    }
                }
            }
        }

        if (mpi_rank == 0) {
            // print block row
            for (int64_t ti = 0; ti < A.tileMb(i); ++ti) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    if (-kl <= j-i && j-i <= ku) { // inside bandwidth
                        auto T = A(i, j);
                        for (int64_t tj = 0; tj < A.tileNb(j); ++tj) {
                            snprintf_value( buf, sizeof(buf), width, precision, T(ti, tj) );
                            msg += buf;
                        }
                    }
                    else {
                        for (int64_t tj = 0; tj < A.tileNb(j); ++tj) {
                            msg += outside;
                        }
                    }
                    if (j < A.nt()-1)
                        msg += "    ";
                    else
                        msg += "\n";
                }
            }
            if (i < A.mt()-1)
                msg += "\n";
            else
                msg += "]\n";
            printf( "%s", msg.c_str() );
            msg.clear();

            // cleanup data
            for (int64_t j = 0; j < A.nt(); ++j) {
                if (! A.tileIsLocal(i, j)) {
                    A.tileErase(i, j);
                }
            }
        }
    }

    MPI_Barrier( comm );
}

//------------------------------------------------------------------------------
/// Print a SLATE distributed matrix.
/// Rank 0 does the printing, and must have enough memory to fit one entire
/// block row of the matrix.
///
/// This version handles trapezoid, triangular, symmetric, and Hermitian
/// matrices. Entries in the A.uplo triangle are printed; entries in the
/// opposite triangle are printed as "nan".
///
template <typename scalar_t>
void print_matrix(
    const char* label,
    slate::BaseTrapezoidMatrix<scalar_t>& A, int width=8, int precision=4)
{
    using blas::real;
    using blas::imag;

    int mpi_rank = A.mpiRank();
    MPI_Comm comm = A.mpiComm();

    width = std::max( width, precision + 3 );

    char buf[ 1024 ];
    std::string msg = label;
    msg += " = [\n";

    // for entries in opposite triangle from A.uplo
    char opp[ 80 ];
    if (slate::is_complex<scalar_t>::value) {
        snprintf( opp, sizeof(opp), " %*f   %*s ",
                  width, NAN, width, "" );
    }
    else {
        snprintf( opp, sizeof(opp), " %*f",
                  width, NAN );
    }

    for (int64_t i = 0; i < A.mt(); ++i) {
        // gather block row to rank 0
        for (int64_t j = 0; j < A.nt(); ++j) {
            if ((A.uplo() == slate::Uplo::Lower && i >= j) ||
                (A.uplo() == slate::Uplo::Upper && i <= j))
            {
                int tile_rank = A.tileRank(i, j);
                if (tile_rank != 0) {
                    if (A.tileIsLocal(i, j)) {
                        auto T = A(i, j);
                        T.send( 0, comm );
                    }
                    else if (mpi_rank == 0) {
                        A.tileInsert( i, j );
                        A(i, j).recv( tile_rank, comm );
                    }
                }
            }
        }

        if (mpi_rank == 0) {
            // print block row
            for (int64_t ti = 0; ti < A.tileMb(i); ++ti) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    if ((A.uplo() == slate::Uplo::Lower && i >= j) ||
                        (A.uplo() == slate::Uplo::Upper && i <= j))
                    {
                        auto T = A(i, j);
                        for (int64_t tj = 0; tj < A.tileNb(j); ++tj) {
                            if (i != j ||
                                (A.uplo() == slate::Uplo::Lower && ti >= tj) ||
                                (A.uplo() == slate::Uplo::Upper && ti <= tj))
                            {
                                snprintf_value( buf, sizeof(buf), width, precision, T(ti, tj) );
                                msg += buf;
                            }
                            else {
                                msg += opp;
                            }
                        }
                    }
                    else {
                        for (int64_t tj = 0; tj < A.tileNb(j); ++tj) {
                            msg += opp;
                        }
                    }
                    if (j < A.nt()-1)
                        msg += "    ";
                    else
                        msg += "\n";
                }
            }
            if (i < A.mt()-1)
                msg += "\n";
            else
                msg += "]\n";
            printf( "%s", msg.c_str() );
            msg.clear();

            // cleanup data
            for (int64_t j = 0; j < A.nt(); ++j) {
                if (! A.tileIsLocal(i, j)) {
                    A.tileErase(i, j);
                }
            }
        }
    }

    MPI_Barrier( comm );
}
