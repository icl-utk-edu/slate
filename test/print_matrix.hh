#ifndef SLATE_PRINT_MATRIX_HH
#define SLATE_PRINT_MATRIX_HH

#include "slate/BaseTrapezoidMatrix.hh"
#include "slate/Matrix.hh"
#include "slate/BaseTrapezoidMatrix.hh"
#include "slate/BandMatrix.hh"

#include "blas.hh"

#include <string>
#include <cstdio>

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
            snprintf(buf, buf_len, " %#*.0f%*s   %*s ",
                     width - precision, real(value), precision, "",
                     width, "");
        }
        else {
            snprintf(buf, buf_len,
                     " %#*.0f%*s",
                     width - precision, real(value), precision, "");
        }
    }
    else {
        // general case
        if (slate::is_complex<scalar_t>::value) {
            snprintf(buf, buf_len, " %*.*f + %*.*fi",
                     width, precision, real(value),
                     width, precision, imag(value));
        }
        else {
            snprintf(buf, buf_len,
                     " %*.*f",
                     width, precision, real(value));
        }
    }
}

//------------------------------------------------------------------------------
/// Print an LAPACK matrix. Should be called from only one rank.
///
template <typename scalar_t>
void print_matrix(
    const char* label,
    int64_t m, int64_t n, scalar_t* A, int64_t lda,
    int width = 10, int precision = 6)
{
    using blas::real;
    using blas::imag;

    char buf[ 1024 ];
    std::string msg;

    width = std::max(width, precision + 3);

    printf("%s = [\n", label);
    for (int64_t i = 0; i < m; ++i) {
        msg = "";
        for (int64_t j = 0; j < n; ++j) {
            snprintf_value(buf, sizeof(buf), width, precision, A[i + j*lda]);
            msg += buf;
        }
        printf( "%s\n", msg.c_str() );
    }
    printf("];\n");
}

//------------------------------------------------------------------------------
/// Print a ScaLAPACK distributed matrix.
/// Prints each rank's data as a contiguous block, numbered by the block row &
/// column indices. Rank 0 does the printing.
///
template <typename scalar_t>
void print_matrix(
    const char* label,
    int64_t mlocal, int64_t nlocal, scalar_t* A, int64_t lda,
    int p, int q, MPI_Comm comm, int width = 10, int precision = 6)
{
    using blas::real;
    using blas::imag;

    int mpi_rank;
    MPI_Comm_rank(comm, &mpi_rank);

    char buf[ 1024 ];
    std::string msg;

    width = std::max(width, precision + 3);

    // loop over process rows & cols
    for (int prow = 0; prow < q; ++prow) {
        for (int pcol = 0; pcol < p; ++pcol) {
            int rank = prow + pcol*p;

            if (rank == mpi_rank) {
                snprintf(buf, sizeof(buf), "%s%d_%d = [\n", label, prow, pcol);
                msg += buf;
                for (int64_t i = 0; i < mlocal; ++i) {
                    for (int64_t j = 0; j < nlocal; ++j) {
                        snprintf_value(buf, sizeof(buf), width, precision, A[i + j*lda]);
                        msg += buf;
                    }
                    msg += "\n";
                }
                msg += "];\n\n";

                if (mpi_rank != 0) {
                    // Send msg to root, which handles actual I/O.
                    int len = int(msg.size());
                    MPI_Send(&len, 1, MPI_INT, 0, 0, comm);
                    MPI_Send(msg.c_str(), len, MPI_CHAR, 0, 0, comm);
                }
                else {
                    // Already on root, just print it.
                    printf("%s", msg.c_str());
                }
            }
            else if (mpi_rank == 0) {
                // Root receives msg and handles actual I/O.
                MPI_Status status;
                int len;
                MPI_Recv(&len, 1, MPI_INT, rank, 0, comm, &status);
                msg.resize(len);
                MPI_Recv(&msg[0], len, MPI_CHAR, rank, 0, comm, &status);
                printf("%s", msg.c_str());
            }
        }
    }
    if (mpi_rank == 0) {
        fflush(stdout);
    }
    MPI_Barrier(comm);
}

//------------------------------------------------------------------------------
/// Sends tiles A(i, j) and receives it on rank 0.
/// If rank != 0 and tile A(i, j) is local, sends it to rank 0.
/// If rank == 0, inserts and receives tile A(i, j),
/// unless tile didn't exist on sender.
///
template <typename scalar_t>
void send_recv_tile(
    slate::BaseMatrix<scalar_t>& A, int64_t i, int64_t j,
    int mpi_rank, MPI_Comm comm)
{
    int flag_exist   = 0;
    int flag_missing = 1;
    int flag;
    int err;
    MPI_Status status;

    int tile_rank = A.tileRank(i, j);
    if (tile_rank != 0) {
        if (A.tileIsLocal(i, j)) {
            try {
                auto T = A(i, j);
                err = MPI_Send( &flag_exist, 1, MPI_INT, 0, 0, comm);
                slate_assert(err == 0);
                T.send(0, comm);
            }
            catch (std::out_of_range const& ex) {
                err = MPI_Send( &flag_missing, 1, MPI_INT, 0, 0, comm);
                slate_assert(err == 0);
            }
        }
        else if (mpi_rank == 0) {
            err = MPI_Recv(&flag, 1, MPI_INT, tile_rank, 0, comm, &status);
            slate_assert(err == 0);
            if (flag == flag_exist) {
                A.tileInsert(i, j);
                A(i, j).recv(tile_rank, comm, A.layout());
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Returns string for row ti of tile A(i, j).
/// If tile doesn't exist, returns string with NAN values.
/// For upper or lower tiles, uses opposite for values in the opposite
/// (lower or upper, respectively) triangle.
/// Works for all matrix types.
///
template <typename scalar_t>
std::string tile_row_string(
    slate::BaseMatrix<scalar_t>& A, int64_t i, int64_t j, int64_t ti,
    int width, int precision,
    const char* opposite="")
{
    char buf[ 80 ];
    std::string msg;
    try {
        auto T = A(i, j);
        for (int64_t tj = 0; tj < A.tileNb(j); ++tj) {
            slate::Uplo uplo = T.uplo();
            if ((uplo == slate::Uplo::General) ||
                (uplo == slate::Uplo::Lower && ti >= tj) ||
                (uplo == slate::Uplo::Upper && ti <= tj))
            {
                snprintf_value(buf, sizeof(buf), width, precision, T(ti, tj));
                msg += buf;
            }
            else {
                msg += opposite;
            }
        }
    }
    catch (std::out_of_range const& ex) {
        // tile missing: print NAN
        snprintf_value(buf, sizeof(buf), width, precision, NAN);
        for (int64_t tj = 0; tj < A.tileNb(j); ++tj) {
            msg += buf;
        }
    }
    return msg;
}

//------------------------------------------------------------------------------
/// Print a SLATE distributed matrix.
/// Rank 0 does the printing, and must have enough memory to fit one entire
/// block row of the matrix.
/// For block-sparse matrices, missing tiles are print as "nan".
///
template <typename scalar_t>
void print_matrix(
    const char* label,
    slate::Matrix<scalar_t>& A, int width = 10, int precision = 6)
{
    using blas::real;
    using blas::imag;

    int mpi_rank = A.mpiRank();
    MPI_Comm comm = A.mpiComm();
    MPI_Barrier(comm);

    width = std::max(width, precision + 3);

    std::string msg = "% slate::Matrix\n";
    msg += label;
    msg += " = [\n";

    for (int64_t i = 0; i < A.mt(); ++i) {
        // gather block row to rank 0
        for (int64_t j = 0; j < A.nt(); ++j) {
            send_recv_tile(A, i, j, mpi_rank, comm);
        }

        if (mpi_rank == 0) {
            // print block row
            for (int64_t ti = 0; ti < A.tileMb(i); ++ti) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    msg += tile_row_string(A, i, j, ti, width, precision);
                    if (j < A.nt() - 1)
                        msg += "    ";
                    else
                        msg += "\n";
                }
            }
            if (i < A.mt() - 1)
                msg += "\n";
            else
                msg += "];\n";
            printf("%s", msg.c_str());
            msg.clear();

            // cleanup data
            for (int64_t j = 0; j < A.nt(); ++j) {
                if (! A.tileIsLocal(i, j)) {
                    A.tileErase(i, j);
                }
            }
        }
    }

    MPI_Barrier(comm);
}

//------------------------------------------------------------------------------
/// Print a SLATE distributed band matrix.
/// Rank 0 does the printing, and must have enough memory to fit one entire
/// block row of the matrix.
/// Tiles outside the bandwidth are printed as "0", with no trailing decimals.
/// For block-sparse matrices, missing tiles are print as "nan".
///
template <typename scalar_t>
void print_matrix(
    const char* label,
    slate::BandMatrix<scalar_t>& A, int width = 10, int precision = 6)
{
    using blas::real;
    using blas::imag;

    int mpi_rank = A.mpiRank();
    MPI_Comm comm = A.mpiComm();
    MPI_Barrier(comm);

    width = std::max(width, precision + 3);

    std::string msg = "% slate::BandMatrix\n";
    msg += label;
    msg += " = [\n";

    // for tiles outside bandwidth
    char outside[ 80 ];
    if (slate::is_complex<scalar_t>::value) {
        snprintf(outside, sizeof(outside), " %*.0f   %*s ",
                 width, 0., width, "");
    }
    else {
        snprintf(outside, sizeof(outside), " %*.0f",
                 width, 0.);
    }

    // todo: initially, assume fixed size, square tiles for simplicity
    int64_t kl = slate::ceildiv(A.lowerBandwidth(), A.tileNb(0));
    int64_t ku = slate::ceildiv(A.upperBandwidth(), A.tileNb(0));

    for (int64_t i = 0; i < A.mt(); ++i) {
        // gather block row to rank 0
        for (int64_t j = 0; j < A.nt(); ++j) {
            if (-kl <= j - i && j - i <= ku) { // inside bandwidth
                send_recv_tile(A, i, j, mpi_rank, comm);
            }
        }

        if (mpi_rank == 0) {
            // print block row
            for (int64_t ti = 0; ti < A.tileMb(i); ++ti) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    if (-kl <= j - i && j - i <= ku) { // inside bandwidth
                        msg += tile_row_string(A, i, j, ti, width, precision);
                    }
                    else {
                        for (int64_t tj = 0; tj < A.tileNb(j); ++tj) {
                            msg += outside;
                        }
                    }
                    if (j < A.nt() - 1)
                        msg += "    ";
                    else
                        msg += "\n";
                }
            }
            if (i < A.mt() - 1)
                msg += "\n";
            else
                msg += "];\n";
            printf("%s", msg.c_str());
            msg.clear();

            // cleanup data
            for (int64_t j = 0; j < A.nt(); ++j) {
                if (! A.tileIsLocal(i, j)) {
                    A.tileErase(i, j);
                }
            }
        }
    }

    MPI_Barrier(comm);
}

//------------------------------------------------------------------------------
/// Print a SLATE distributed hermitian band matrix.
/// Rank 0 does the printing, and must have enough memory to fit one entire
/// block row of the matrix.
/// Tiles outside the bandwidth are printed as "0", with no trailing decimals.
/// For block-sparse matrices, missing tiles are print as "nan".
///
/// This version handles Hermitian band matrices. Entries in the A.uplo
/// triangle are printed; entries in the opposite triangle are printed as "nan".
///
/// Having said that, if the printed matrix is a lower triangular matrix,
/// then the routine will print the tiles of upper part of the matrix as "nan",
/// and the lower part tiles that are inside the bandwidth will be printed
/// as they are, whereas the non existing tiles, tiles outside the bandwidth,
/// will be printed as "0", with no trailing decimals.
/// This is to follow MATLAB convention and to make it easier for debugging.
///
template <typename scalar_t>
void print_matrix(
    const char* label,
    slate::HermitianBandMatrix<scalar_t>& A, int width = 10, int precision = 6)
{
    using blas::real;
    using blas::imag;

    int mpi_rank = A.mpiRank();
    MPI_Comm comm = A.mpiComm();
    MPI_Barrier(comm);

    width = std::max(width, precision + 3);

    std::string msg = "% slate::HermitianBandMatrix\n";
    msg += label;
    msg += " = [\n";

    // for entries in opposite triangle from A.uplo
    char opposite[ 80 ];
    if (slate::is_complex<scalar_t>::value) {
        snprintf(opposite, sizeof(opposite), " %*f   %*s ",
                 width, NAN, width, "");
    }
    else {
        snprintf(opposite, sizeof(opposite), " %*f",
                 width, NAN);
    }

    // for tiles outside bandwidth
    char outside[ 80 ];
    if (slate::is_complex<scalar_t>::value) {
        snprintf(outside, sizeof(outside), " %*.0f   %*s ",
                 width, 0., width, "");
    }
    else {
        snprintf(outside, sizeof(outside), " %*.0f",
                 width, 0.);
    }

    int64_t kdt = slate::ceildiv(A.bandwidth(), A.tileNb(0));
    for (int64_t i = 0; i < A.mt(); ++i) {
        for (int64_t j = 0; j < A.nt(); ++j) {
            if ((A.uplo() == slate::Uplo::Lower && i <= j + kdt && j <= i) ||
                (A.uplo() == slate::Uplo::Upper && i >= j - kdt && j >= i)) {
                send_recv_tile(A, i, j, mpi_rank, comm);
            }
        }

        if (mpi_rank == 0) {
            for (int64_t ti = 0; ti < A.tileMb(i); ++ti) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    if ((A.uplo() == slate::Uplo::Lower && i <= j + kdt && j <= i) ||
                        (A.uplo() == slate::Uplo::Upper && i >= j - kdt && j >= i)) {
                        msg += tile_row_string(A, i, j, ti, width, precision, opposite);
                    }
                    else {
                        for (int64_t tj = 0; tj < A.tileNb(j); ++tj) {
                            if ((A.uplo() == slate::Uplo::Lower && j <= i) ||
                                (A.uplo() == slate::Uplo::Upper && j >= i)) {
                                msg += outside;
                            }
                            else {
                                msg += opposite;
                            }
                        }
                    }
                    if (j < A.nt() - 1)
                        msg += "    ";
                    else
                        msg += "\n";
                }
            }
            if (i < A.mt() - 1)
                msg += "\n";
            else
                msg += "];\n";
            printf("%s", msg.c_str());
            msg.clear();

            // cleanup data
            for (int64_t j = 0; j < A.nt(); ++j) {
                if (! A.tileIsLocal(i, j)) {
                    A.tileErase(i, j);
                }
            }
        }
    }

    MPI_Barrier(comm);
}

//------------------------------------------------------------------------------
/// Print a SLATE distributed trapezoid matrix.
/// Rank 0 does the printing, and must have enough memory to fit one entire
/// block row of the matrix.
/// For block-sparse matrices, missing tiles are print as "nan".
///
/// This version handles trapezoid, triangular, symmetric, and Hermitian
/// matrices. Entries in the A.uplo triangle are printed; entries in the
/// opposite triangle are printed as "nan".
///
template <typename scalar_t>
void print_matrix(
    const char* label,
    slate::BaseTrapezoidMatrix<scalar_t>& A, int width = 10, int precision = 6)
{
    using blas::real;
    using blas::imag;

    int mpi_rank = A.mpiRank();
    MPI_Comm comm = A.mpiComm();
    MPI_Barrier(comm);

    width = std::max(width, precision + 3);

    std::string msg = "% slate::BaseTrapezoidMatrix\n";
    msg += label;
    msg += " = [\n";

    // for entries in opposite triangle from A.uplo
    char opposite[ 80 ];
    if (slate::is_complex<scalar_t>::value) {
        snprintf(opposite, sizeof(opposite), " %*f   %*s ",
                 width, NAN, width, "");
    }
    else {
        snprintf(opposite, sizeof(opposite), " %*f",
                 width, NAN);
    }

    for (int64_t i = 0; i < A.mt(); ++i) {
        // gather block row to rank 0
        for (int64_t j = 0; j < A.nt(); ++j) {
            if ((A.uplo() == slate::Uplo::Lower && i >= j) ||
                (A.uplo() == slate::Uplo::Upper && i <= j))
            {
                send_recv_tile(A, i, j, mpi_rank, comm);
            }
        }

        if (mpi_rank == 0) {
            // print block row
            for (int64_t ti = 0; ti < A.tileMb(i); ++ti) {
                for (int64_t j = 0; j < A.nt(); ++j) {
                    if ((A.uplo() == slate::Uplo::Lower && i >= j) ||
                        (A.uplo() == slate::Uplo::Upper && i <= j))
                    {
                        // tile in stored triangle
                        msg += tile_row_string(A, i, j, ti, width, precision, opposite);
                    }
                    else {
                        // tile in opposite triangle
                        for (int64_t tj = 0; tj < A.tileNb(j); ++tj) {
                            msg += opposite;
                        }
                    }
                    if (j < A.nt() - 1)
                        msg += "    ";
                    else
                        msg += "\n";
                }
            }
            if (i < A.mt() - 1)
                msg += "\n";
            else
                msg += "];\n";
            printf("%s", msg.c_str());
            msg.clear();

            // cleanup data
            for (int64_t j = 0; j < A.nt(); ++j) {
                if (! A.tileIsLocal(i, j)) {
                    A.tileErase(i, j);
                }
            }
        }
    }

    MPI_Barrier(comm);
}

#endif // SLATE_PRINT_MATRIX_HH
