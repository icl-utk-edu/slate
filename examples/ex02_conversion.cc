// ex02_conversion.cc
// conversion between matrix types

/// !!!   Lines between `//---------- begin label`          !!!
/// !!!             and `//---------- end label`            !!!
/// !!!   are included in the SLATE Users' Guide.           !!!

#include <slate/slate.hh>

#include "util.hh"

int mpi_size = 0;
int mpi_rank = 0;
int grid_p = 0;
int grid_q = 0;

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_conversion()
{
    print_func( mpi_rank );

    int64_t m=2000, n=1000, nb=256;

    //---------- begin convert
    // A is defined to be a general m x n matrix of type scalar_type
    // (float, std::complex<float>, double, std::complex<double>, etc.).
    slate::Matrix<scalar_type>
        A( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );

    // Lz is a trapezoid matrix view of the lower trapezoid of A,
    // assuming Unit diagonal.
    slate::TrapezoidMatrix<scalar_type>
        Lz( slate::Uplo::Lower, slate::Diag::Unit, A );

    // Triangular, symmetric, and Hermitian matrices must be square --
    // take square slice if needed.
    int64_t min_mn = std::min( m, n );
    auto A_square = A.slice( 0, min_mn-1, 0, min_mn-1 );

    // L is a triangular matrix view of the lower triangle of A,
    // assuming Unit diagonal.
    slate::TriangularMatrix<scalar_type>
        L( slate::Uplo::Lower, slate::Diag::Unit, A_square );

    // U is a triangular matrix view of the upper triangle of A.
    slate::TriangularMatrix<scalar_type>
        U( slate::Uplo::Upper, slate::Diag::NonUnit, A_square );

    // S is a symmetric matrix view of the upper triangle of A.
    slate::SymmetricMatrix<scalar_type>
        S( slate::Uplo::Upper, A_square );

    // H is a Hermitian matrix view of the upper triangle of A.
    slate::HermitianMatrix<scalar_type>
        H( slate::Uplo::Upper, A_square );
    //---------- end convert
}

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    try {
        // Parse command line to set types for s, d, c, z precisions.
        bool types[ 4 ];
        parse_args( argc, argv, types );

        int provided = 0;
        slate_mpi_call(
            MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided ) );
        assert( provided == MPI_THREAD_MULTIPLE );

        slate_mpi_call(
            MPI_Comm_size( MPI_COMM_WORLD, &mpi_size ) );

        slate_mpi_call(
            MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank ) );

        // Determine p-by-q grid for this MPI size.
        grid_size( mpi_size, &grid_p, &grid_q );
        if (mpi_rank == 0) {
            printf( "mpi_size %d, grid_p %d, grid_q %d\n",
                    mpi_size, grid_p, grid_q );
        }

        // so random_matrix is different on different ranks.
        srand( 100 * mpi_rank );

        if (types[ 0 ]) {
            test_conversion< float >();
        }

        if (types[ 1 ]) {
            test_conversion< double >();
        }

        if (types[ 2 ]) {
            test_conversion< std::complex<float> >();
        }

        if (types[ 3 ]) {
            test_conversion< std::complex<double> >();
        }

        slate_mpi_call(
            MPI_Finalize() );
    }
    catch (std::exception const& ex) {
        fprintf( stderr, "%s", ex.what() );
        return 1;
    }
    return 0;
}
