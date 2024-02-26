// ex12_generalized_hermitian_eig.cc
// Solve generalized Hermitian eigenvalues, types:
// 1. A = B Z Lambda Z^H
// 2. A B = Z Lambda Z^H
// 3. B A = Z Lambda Z^H
// where B is Hermitian positive definite.

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
void test_hermitian_eig()
{
    using real_t = blas::real_type<scalar_type>;

    print_func( mpi_rank );

    int64_t n=1000, nb=256;

    //---------- begin eig1
    slate::HermitianMatrix<scalar_type>
        A( slate::Uplo::Lower, n, nb, grid_p, grid_q, MPI_COMM_WORLD ),
        B( slate::Uplo::Lower, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<scalar_type>
        Z( n, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    std::vector<real_t> Lambda( n );
    // ...
    //---------- end eig1

    A.insertLocalTiles();
    B.insertLocalTiles();
    Z.insertLocalTiles();
    random_matrix( A );
    random_matrix_diag_dominant( B );

    //----------------------------------------
    //---------- begin eig2
    // Type 1: A = B Z Lambda Z^H, eigenvalues only
    slate::eig_vals( 1, A, B, Lambda );  // simplified API, or
    //---------- end eig2

    random_matrix( A );
    random_matrix_diag_dominant( B );

    //---------- begin eig3
    slate::eig( 1, A, B, Lambda );       // simplified API
    //---------- end eig3

    random_matrix( A );
    random_matrix_diag_dominant( B );

    //---------- begin eig4
    slate::hegv( 1, A, B, Lambda );      // traditional API
    //---------- end eig4

    random_matrix( A );
    random_matrix_diag_dominant( B );

    //----------------------------------------
    //---------- begin eig5

    // Type 2: A B = Z Lambda Z^H, eigenvalues only
    slate::eig_vals( 2, A, B, Lambda );  // simplified API
    //---------- end eig5

    random_matrix( A );
    random_matrix_diag_dominant( B );

    //----------------------------------------
    //---------- begin eig6

    // Type 3: A = B Z Lambda Z^H, eigenvalues only
    slate::eig_vals( 3, A, B, Lambda );  // simplified API
    //---------- end eig6

    random_matrix( A );
    random_matrix_diag_dominant( B );

    //----------------------------------------
    //---------- begin eig7

    // Types 1, 2, and 3, with eigenvectors
    slate::eig( 1, A, B, Lambda, Z );    // simplified API
    //---------- end eig7

    random_matrix( A );
    random_matrix_diag_dominant( B );

    //---------- begin eig8
    slate::eig( 2, A, B, Lambda, Z );    // simplified API
    //---------- end eig8

    random_matrix( A );
    random_matrix_diag_dominant( B );

    //---------- begin eig9
    slate::eig( 3, A, B, Lambda, Z );    // simplified API
    //---------- end eig9

    random_matrix( A );
    random_matrix_diag_dominant( B );

    //---------- begin eig10
    slate::hegv( 1, A, B, Lambda, Z );   // traditional API
    //---------- end eig10

    random_matrix( A );
    random_matrix_diag_dominant( B );

    //---------- begin eig11
    slate::hegv( 2, A, B, Lambda, Z );   // traditional API
    //---------- end eig11

    random_matrix( A );
    random_matrix_diag_dominant( B );

    //---------- begin eig12
    slate::hegv( 3, A, B, Lambda, Z );   // traditional API
    //---------- end eig12
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
        // Hermitian eig requires square MPI grid.
        grid_size_square( mpi_size, &grid_p, &grid_q );
        if (mpi_rank == 0) {
            printf( "mpi_size %d, grid_p %d, grid_q %d\n",
                    mpi_size, grid_p, grid_q );
        }

        // so random_matrix is different on different ranks.
        srand( 100 * mpi_rank );

        if (types[ 0 ]) {
            test_hermitian_eig< float >();
        }

        if (types[ 1 ]) {
            test_hermitian_eig< double >();
        }

        if (types[ 2 ]) {
            test_hermitian_eig< std::complex<float> >();
        }

        if (types[ 3 ]) {
            test_hermitian_eig< std::complex<double> >();
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
