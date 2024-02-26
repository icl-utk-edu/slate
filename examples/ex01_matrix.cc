// ex01_matrix.cc
// create 2000 x 1000 matrix on 2 x 2 MPI process grid

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
void test_constructor()
{
    print_func( mpi_rank );

    int64_t m=2000, n=1000, mb=128, nb=256;

    //---------- begin constructor
    // Create an empty matrix (2D block cyclic layout, p x q grid,
    // no tiles allocated, square nb x nb tiles)
    slate::Matrix<scalar_type>
        A( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );

    // Create an empty matrix (2D block cyclic layout, p x q grid,
    // no tiles allocated, rectangular mb x nb tiles)
    slate::Matrix<scalar_type>
        B( m, n, mb, nb, grid_p, grid_q, MPI_COMM_WORLD );

    // Create an empty TriangularMatrix (2D block cyclic layout, no tiles)
    slate::TriangularMatrix<scalar_type>
        T( slate::Uplo::Lower, slate::Diag::NonUnit, n, nb,
           grid_p, grid_q, MPI_COMM_WORLD );

    // Create an empty matrix based on another matrix structure.
    slate::Matrix<scalar_type> A2 = A.emptyLike();
    //---------- end constructor
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_insert_host()
{
    print_func( mpi_rank );

    int64_t m=2000, n=1000, nb=256;

    //---------- begin insert_host
    // Create two empty matrices.
    slate::Matrix<scalar_type> A( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    auto A2 = A.emptyLike();

    // Insert tiles on the CPU host.
    A.insertLocalTiles( slate::Target::Host );

    // A2.insertLocalTiles( slate::Target::Host ) is equivalent to:
    for (int64_t j = 0; j < A2.nt(); ++j)
        for (int64_t i = 0; i < A2.mt(); ++i)
            if (A2.tileIsLocal( i, j ))
                A2.tileInsert( i, j, slate::HostNum );
    //---------- end insert_host
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_insert_device()
{
    print_func( mpi_rank );

    int64_t m=2000, n=1000, nb=256;

    //---------- begin insert_device
    // Create two empty matrices.
    slate::Matrix<scalar_type> A( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    auto A2 = A.emptyLike();

    // Insert tiles on the GPU devices.
    A.insertLocalTiles( slate::Target::Devices );

    // A2.insertLocalTiles( slate::Target::Devices ) is equivalent to:
    for (int64_t j = 0; j < A2.nt(); ++j)
        for (int64_t i = 0; i < A2.mt(); ++i)
            if (A2.tileIsLocal( i, j ))
                A2.tileInsert( i, j, A2.tileDevice( i, j ) );
    //---------- end insert_device
}

//------------------------------------------------------------------------------
// This example uses ScaLAPACK data just as an example of user-defined data,
// but it is easier to use fromScaLAPACK to create a SLATE matrix; see below.
template <typename scalar_type>
void test_insert_user()
{
    print_func( mpi_rank );

    int64_t m=2000, n=1000, nb=256;

    // User-allocated data, in ScaLAPACK format (assuming column-major grid).
    int myrow = mpi_rank % grid_p;
    int mycol = mpi_rank / grid_p;
    int64_t mlocal = slate::num_local_rows_cols( m, nb, myrow, 0, grid_p );
    int64_t nlocal = slate::num_local_rows_cols( n, nb, myrow, 0, grid_p );
    int64_t lld = mlocal; // local leading dimension
    scalar_type* A_data = new scalar_type[ lld*nlocal ];

    // lambda to get tile (i, j) in ScaLAPACK data.
    auto data = [A_data, nb, lld]( int64_t i, int64_t j ) {
        int64_t ii_local = slate::global2local( i*nb, nb, grid_p );
        int64_t jj_local = slate::global2local( j*nb, nb, grid_q );
        return &A_data[ ii_local + jj_local*lld ];
    };

    //---------- begin insert_user
    // Create an empty matrix (2D block cyclic layout, no tiles).
    slate::Matrix<scalar_type> A( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );

    // Attach user allocated tiles, from pointers in data( i, j )
    // with local stride lld between columns.
    for (int64_t j = 0; j < A.nt(); ++j) {
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal( i, j ))
                A.tileInsert( i, j, data( i, j ), lld );
        }
    }
    //---------- end insert_user
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_fromScaLAPACK()
{
    print_func( mpi_rank );

    int64_t m=2000, n=1000, nb=256;

    //---------- begin fromScaLAPACK
    // User-allocated data, in ScaLAPACK format (assuming column-major grid).
    int myrow = mpi_rank % grid_p;
    int mycol = mpi_rank / grid_p;
    int64_t mlocal = slate::num_local_rows_cols( m, nb, myrow, 0, grid_p );
    int64_t nlocal = slate::num_local_rows_cols( n, nb, myrow, 0, grid_p );
    int64_t lld = mlocal; // local leading dimension
    scalar_type* A_data = new scalar_type[ lld*nlocal ];

    // Create matrix from ScaLAPACK data.
    auto A = slate::Matrix<scalar_type>::fromScaLAPACK(
        m, n,                   // global matrix dimensions
        A_data,                 // local ScaLAPACK array data
        lld,                    // local leading dimension (column stride) for data
        nb, nb,                 // block size
        slate::GridOrder::Col,  // col- or row-major MPI process grid
        grid_p, grid_q,         // MPI process grid
        MPI_COMM_WORLD          // MPI communicator
    );
    //---------- end fromScaLAPACK
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_transpose()
{
    print_func( mpi_rank );

    int64_t m=2000, n=1000, nb=256;

    //---------- begin transpose
    slate::Matrix<scalar_type> A( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );

    // Transpose
    // AT is a transposed view of A, with flag AT.op() == Op::Trans.
    // The Tile AT( i, j ) == transpose( A( j, i ) ).
    auto AT = transpose( A );

    // Conjugate transpose
    // AH is a conjugate-transposed view of A, with flag AH.op() == Op::ConjTrans.
    // The Tile AH( i, j ) == conj_transpose( A( j, i ) ).
    auto AH = conj_transpose( A );
    //---------- end transpose
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_elements()
{
    using slate::LayoutConvert;

    print_func( mpi_rank );

    int64_t m=2000, n=1000, nb=256;

    //---------- begin elements
    slate::Matrix<scalar_type> A( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles( slate::Target::Host );

    // Loop over tiles in A.
    int64_t jj_global = 0;
    for (int64_t j = 0; j < A.nt(); ++j) {
        int64_t ii_global = 0;
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal( i, j )) {
                // For local tiles, loop over entries in tile.
                // Make sure CPU tile exists for writing.
                A.tileGetForWriting( i, j, slate::HostNum, LayoutConvert::ColMajor );
                slate::Tile<scalar_type> T = A( i, j, slate::HostNum );
                for (int64_t jj = 0; jj < T.nb(); ++jj) {
                    for (int64_t ii = 0; ii < T.mb(); ++ii) {
                        // Note: currently using T.at() is inefficient
                        // in inner loops; see below.
                        T.at( ii, jj )
                            = std::abs( (ii_global + ii) - (jj_global + jj) );
                    }
                }
            }
            ii_global += A.tileMb( i );
        }
        jj_global += A.tileMb( j );
    }
    //---------- end elements

    //---------- begin elements2
    // Loop over tiles in A, more efficient implementation.
    jj_global = 0;
    for (int64_t j = 0; j < A.nt(); ++j) {
        int64_t ii_global = 0;
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal( i, j )) {
                // For local tiles, loop over entries in tile.
                // Make sure CPU tile exists for writing.
                A.tileGetForWriting( i, j, slate::HostNum, LayoutConvert::ColMajor );
                slate::Tile<scalar_type> T = A( i, j, slate::HostNum );
                scalar_type* data = T.data();
                int64_t   mb      = T.mb();
                int64_t   nb      = T.nb();
                int64_t   stride  = T.stride();
                for (int64_t jj = 0; jj < T.nb(); ++jj) {
                    for (int64_t ii = 0; ii < T.mb(); ++ii) {
                        // Currently more efficient than using T.at().
                        data[ ii + jj*stride ]
                            = std::abs( (ii_global + ii) - (jj_global + jj) );
                    }
                }
            }
            ii_global += A.tileMb( i );
        }
        jj_global += A.tileMb( j );
    }
    //---------- end elements2
}

//------------------------------------------------------------------------------
// Map double => float, and complex<double> => complex
template <typename T>
struct mixed_precision_traits;

template <>
struct mixed_precision_traits< double >
{
    using low_type = float;
};

template <typename T>
struct mixed_precision_traits< std::complex<T> >
{
    using real_low_type = typename mixed_precision_traits<T>::low_type;
    using low_type = std::complex< real_low_type >;
};

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_copy()
{
    // low_type is float or complex<float>.
    using low_type = typename mixed_precision_traits< scalar_type >::low_type;

    print_func( mpi_rank );

    int64_t m=2000, n=1000, nb=256;

    //---------- begin copy
    // scalar_type is double or complex<double>;
    // low_type    is float  or complex<float>.
    slate::Matrix<scalar_type> A_hi( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<low_type>    A_lo( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A_hi.insertLocalTiles();
    A_lo.insertLocalTiles();

    auto A_hi_2 = A_hi.emptyLike();
    A_hi_2.insertLocalTiles();

    // Copy with precision conversion from double to float.
    copy( A_hi, A_lo );

    // Copy with precision conversion from float to double.
    copy( A_lo, A_hi );

    // Copy without conversion.
    copy( A_hi, A_hi_2 );
    //---------- end copy
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_all()
{
    test_constructor   <scalar_type>();
    test_insert_host   <scalar_type>();
    test_insert_device <scalar_type>();
    test_insert_user   <scalar_type>();
    test_fromScaLAPACK <scalar_type>();
    test_elements      <scalar_type>();
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
            test_all< float >();
        }

        if (types[ 1 ]) {
            test_all< double >();
            test_copy< double >();
        }

        if (types[ 2 ]) {
            test_all< std::complex<float> >();
        }

        if (types[ 3 ]) {
            test_all< std::complex<double> >();
            test_copy< std::complex<double> >();
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
