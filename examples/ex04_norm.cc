// ex04_norm.cc
// matrix norms

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
void test_general_norm()
{
    using real_type = blas::real_type<scalar_type>;

    print_func( mpi_rank );

    int64_t m=2000, n=1000, nb=256;

    //---------- begin genorm1
    slate::Matrix<scalar_type> A( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    // ...

    //---------- end genorm1

    A.insertLocalTiles();
    random_matrix( A );

    //---------- begin genorm2
    real_type A_norm_one = slate::norm( slate::Norm::One, A );
    real_type A_norm_inf = slate::norm( slate::Norm::Inf, A );
    real_type A_norm_max = slate::norm( slate::Norm::Max, A );
    real_type A_norm_fro = slate::norm( slate::Norm::Fro, A );
    //---------- end genorm2
    printf( "rank %d: norms: one %12.6f, inf %12.6f, max %12.6f, fro %12.6f\n",
            mpi_rank, A_norm_one, A_norm_inf, A_norm_max, A_norm_fro );
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_symmetric_norm()
{
    using real_type = blas::real_type<scalar_type>;

    print_func( mpi_rank );

    int64_t n=1000, nb=256;

    //---------- begin synorm1

    // norm() is overloaded for all matrix types: Symmetric, Triangular, etc.
    slate::SymmetricMatrix<scalar_type>
        S( slate::Uplo::Lower, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    // ...

    //---------- end synorm1

    S.insertLocalTiles();
    random_matrix( S );

    //---------- begin synorm2
    real_type S_norm_one = slate::norm( slate::Norm::One, S );
    real_type S_norm_inf = slate::norm( slate::Norm::Inf, S );
    real_type S_norm_max = slate::norm( slate::Norm::Max, S );
    real_type S_norm_fro = slate::norm( slate::Norm::Fro, S );
    //---------- end synorm2
    printf( "rank %d: norms: one %12.6f, inf %12.6f, max %12.6f, fro %12.6f\n",
            mpi_rank, S_norm_one, S_norm_inf, S_norm_max, S_norm_fro );
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_hermitian_norm()
{
    using real_type = blas::real_type<scalar_type>;

    print_func( mpi_rank );

    int64_t n=1000, nb=256;

    slate::HermitianMatrix<scalar_type>
        H( slate::Uplo::Lower, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    H.insertLocalTiles();
    random_matrix( H );

    real_type H_norm_one = slate::norm( slate::Norm::One, H );
    real_type H_norm_inf = slate::norm( slate::Norm::Inf, H );
    real_type H_norm_max = slate::norm( slate::Norm::Max, H );
    real_type H_norm_fro = slate::norm( slate::Norm::Fro, H );
    printf( "rank %d: norms: one %12.6f, inf %12.6f, max %12.6f, fro %12.6f\n",
            mpi_rank, H_norm_one, H_norm_inf, H_norm_max, H_norm_fro );
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_triangular_norm()
{
    using real_type = blas::real_type<scalar_type>;

    print_func( mpi_rank );

    int64_t n=1000, nb=256;

    slate::TriangularMatrix<scalar_type>
        T( slate::Uplo::Lower, slate::Diag::NonUnit, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    T.insertLocalTiles();
    random_matrix( T );

    real_type T_norm_one = slate::norm( slate::Norm::One, T );
    real_type T_norm_inf = slate::norm( slate::Norm::Inf, T );
    real_type T_norm_max = slate::norm( slate::Norm::Max, T );
    real_type T_norm_fro = slate::norm( slate::Norm::Fro, T );
    printf( "rank %d: norms: one %12.6f, inf %12.6f, max %12.6f, fro %12.6f\n",
            mpi_rank, T_norm_one, T_norm_inf, T_norm_max, T_norm_fro );
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_trapezoid_norm()
{
    using real_type = blas::real_type<scalar_type>;

    print_func( mpi_rank );

    int64_t m=2000, n=1000, nb=256;

    slate::TrapezoidMatrix<scalar_type>
        Tz( slate::Uplo::Lower, slate::Diag::NonUnit, m, n, nb,
            grid_p, grid_q, MPI_COMM_WORLD );
    Tz.insertLocalTiles();
    random_matrix( Tz );

    real_type Tz_norm_one = slate::norm( slate::Norm::One, Tz );
    real_type Tz_norm_inf = slate::norm( slate::Norm::Inf, Tz );
    real_type Tz_norm_max = slate::norm( slate::Norm::Max, Tz );
    real_type Tz_norm_fro = slate::norm( slate::Norm::Fro, Tz );
    printf( "rank %d: norms: one %12.6f, inf %12.6f, max %12.6f, fro %12.6f\n",
            mpi_rank, Tz_norm_one, Tz_norm_inf, Tz_norm_max, Tz_norm_fro );
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
            test_general_norm   < float >();
            test_symmetric_norm < float >();
            test_hermitian_norm < float >();
            test_triangular_norm< float >();
            test_trapezoid_norm < float >();
        }
        if (mpi_rank == 0)
            printf( "\n" );

        if (types[ 1 ]) {
            test_general_norm   < double >();
            test_symmetric_norm < double >();
            test_hermitian_norm < double >();
            test_triangular_norm< double >();
            test_trapezoid_norm < double >();
        }
        if (mpi_rank == 0)
            printf( "\n" );

        if (types[ 2 ]) {
            test_general_norm   < std::complex<float> >();
            test_symmetric_norm < std::complex<float> >();
            test_hermitian_norm < std::complex<float> >();
            test_triangular_norm< std::complex<float> >();
            test_trapezoid_norm < std::complex<float> >();
        }
        if (mpi_rank == 0)
            printf( "\n" );

        if (types[ 3 ]) {
            test_general_norm   < std::complex<double> >();
            test_symmetric_norm < std::complex<double> >();
            test_hermitian_norm < std::complex<double> >();
            test_triangular_norm< std::complex<double> >();
            test_trapezoid_norm < std::complex<double> >();
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
