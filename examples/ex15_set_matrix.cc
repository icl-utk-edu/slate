// ex15_set_matrix.cc
// Set matrix entries

/// !!!   Lines between `//---------- begin label`          !!!
/// !!!             and `//---------- end label`            !!!
/// !!!   are included in the SLATE Users' Guide.           !!!

#include <slate/slate.hh>

#include "util.hh"

int mpi_size = 0;
int mpi_rank = 0;
int grid_p = 1;
int grid_q = 1;

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_set_rand()
{
    using real_t = blas::real_type<scalar_type>;

    print_func( mpi_rank );

    int64_t m=20, n=20, nb=8;
    slate::Matrix<scalar_type> A( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();

    using entry_type = std::function< scalar_type (int64_t, int64_t) >;

    const real_t random_max = INT32_MAX;  // 2^31 - 1

    // Lambda to set entry A_ij.
    // This is non-deterministic since tiles are set in parallel!
    // SLATE's matgen library has a deterministic parallel random matrix generator.
    entry_type entry = [random_max]( int64_t i, int64_t j )
    {
        if constexpr (blas::is_complex<scalar_type>::value) {
            return blas::make_scalar<scalar_type>( random() / random_max,
                                                   random() / random_max );
        }
        else {
            return random() / random_max;
        };
    };

    slate::set( entry, A );
    slate::print( "A", A );
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_set_rand_hermitian()
{
    using real_t = blas::real_type<scalar_type>;

    print_func( mpi_rank );

    int64_t n=20, nb=8;
    slate::HermitianMatrix<scalar_type>
        A( slate::Uplo::Lower, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();

    using entry_type = std::function< scalar_type (int64_t, int64_t) >;

    const real_t random_max = INT32_MAX;  // 2^31 - 1

    // Lambda to set entry A_ij.
    // This is non-deterministic since tiles are set in parallel!
    // SLATE's matgen library has a deterministic parallel random matrix generator.
    entry_type entry = [random_max]( int64_t i, int64_t j )
    {
        if constexpr (blas::is_complex<scalar_type>::value) {
            return blas::make_scalar<scalar_type>( random() / random_max,
                                                   random() / random_max );
        }
        else {
            return random() / random_max;
        };
    };

    slate::set( entry, A );
    slate::print( "A", A );
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_set_ij()
{
    print_func( mpi_rank );

    int64_t m=20, n=20, nb=8;
    slate::Matrix<scalar_type> A( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();

    using entry_type = std::function< scalar_type (int64_t, int64_t) >;

    // Lambda to set entry A_ij.
    entry_type entry = []( int64_t i, int64_t j )
    {
        if constexpr (blas::is_complex<scalar_type>::value) {
            // In complex, real part is i, imag part is j.
            return blas::make_scalar<scalar_type>( i + 1, j + 1 );
        }
        else {
            // In real, integer part is i, fraction part is j.
            return i + 1 + (j + 1)/1000.;
        }
    };

    slate::set( entry, A );
    slate::print( "A", A );
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_set_stencil()
{
    print_func( mpi_rank );

    int64_t n=5, n2=n*n, nb=8;
    slate::Matrix<scalar_type> A( n2, n2, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();

    using entry_type = std::function< scalar_type (int64_t, int64_t) >;

    // Lambda for 9-point Laplacian stencil in 2D.
    entry_type entry = [n]( int64_t i, int64_t j )
    {
        if (i == j)
            return -3.0;
        else if (i == j-1 || i == j+1
              || i == j-n || i == j+n)
            return 0.5;
        else if (i == j-n-1 || i == j-n+1
              || i == j+n-1 || i == j+n+1)
            return 0.25;
        else
            return 0.0;
    };

    slate::set( entry, A );
    slate::print( "A", A );
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

        unsigned t = time( nullptr );
        printf( "srandom( %u )\n", t );
        srandom( t );

        if (types[ 0 ]) {
            test_set_rand<float>();
            test_set_rand_hermitian<float>();
            test_set_ij<float>();
            test_set_stencil<float>();
        }
        if (types[ 1 ]) {
            test_set_rand<double>();
            test_set_rand_hermitian<double>();
            test_set_ij<double>();
            test_set_stencil<double>();
        }
        if (types[ 2 ]) {
            test_set_rand< std::complex<float> >();
            test_set_rand_hermitian< std::complex<float> >();
            test_set_ij< std::complex<float> >();
            test_set_stencil< std::complex<float> >();
        }
        if (types[ 3 ]) {
            test_set_rand< std::complex<double> >();
            test_set_rand_hermitian< std::complex<double> >();
            test_set_ij< std::complex<double> >();
            test_set_stencil< std::complex<double> >();
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
