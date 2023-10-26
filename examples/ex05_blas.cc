// ex05_blas.cc
// BLAS routines

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
void test_gemm()
{
    print_func( mpi_rank );

    double alpha = 2.0, beta = 1.0;
    int64_t m=2000, n=1000, k=500, nb=256;

    slate::Matrix<double> A( m, k, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<double> B( k, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<double> C( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    B.insertLocalTiles();
    C.insertLocalTiles();
    random_matrix( A );
    random_matrix( B );
    random_matrix( C );

    //---------- begin gemm
    // C = alpha A B + beta C, where A, B, C are all general matrices.
    slate::multiply( alpha, A, B, beta, C );  // simplified API
    slate::gemm( alpha, A, B, beta, C );      // traditional API
    //---------- end gemm

    //--------------------
    if (blas::get_device_count() > 0) {
        //---------- begin gemm_opts
        // Execute on GPU devices with lookahead of 2.
        slate::Options opts = {
            { slate::Option::Lookahead, 2 },
            { slate::Option::Target, slate::Target::Devices },
        };
        slate::multiply( alpha, A, B, beta, C, opts );
        //---------- end gemm_opts
    }
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_gemm_trans()
{
    print_func( mpi_rank );

    double alpha = 2.0, beta = 1.0;
    int64_t m=2000, n=1000, k=500, nb=256;

    // Dimensions of A, B are backwards from A, B in test_gemm().
    slate::Matrix<double> A( k, m, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<double> B( n, k, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<double> C( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    B.insertLocalTiles();
    C.insertLocalTiles();
    random_matrix( A );
    random_matrix( B );
    random_matrix( C );

    //---------- begin gemm_trans

    // Matrices can be transposed or conjugate-transposed beforehand.
    // C = alpha A^T B^H + beta C
    auto AT = transpose( A );
    auto BH = conj_transpose( B );
    slate::multiply( alpha, AT, BH, beta, C );  // simplified API
    slate::gemm( alpha, AT, BH, beta, C );      // traditional API
    //---------- end gemm_trans

    // todo: support rvalues:
    // slate::gemm( alpha, transpose( A ), conj_transpose( B ), beta, C );
    // or
    // slate::gemm( alpha, transpose( A ), conj_transpose( B ), beta, std::move( C ) );
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_symm_left()
{
    print_func( mpi_rank );

    double alpha = 2.0, beta = 1.0;
    int64_t m=2000, n=1000, nb=256;

    // A is m-by-m, B and C are m-by-n.
    slate::SymmetricMatrix<double>
        A( slate::Uplo::Lower, m, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<double> B( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<double> C( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    B.insertLocalTiles();
    C.insertLocalTiles();
    random_matrix( A );
    random_matrix( B );
    random_matrix( C );

    //---------- begin symm_left

    // C = alpha A B + beta C, where A is symmetric, on left side
    slate::multiply( alpha, A, B, beta, C );                  // simplified API
    slate::symm( slate::Side::Left, alpha, A, B, beta, C );   // traditional API
    //---------- end symm_left
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_symm_right()
{
    print_func( mpi_rank );

    double alpha = 2.0, beta = 1.0;
    int64_t m=2000, n=1000, nb=256;

    // A is m-by-m, B and C are n-by-m (reverse of left case above).
    slate::SymmetricMatrix<double>
        A( slate::Uplo::Lower, m, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<double> B( n, m, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<double> C( n, m, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    B.insertLocalTiles();
    C.insertLocalTiles();
    random_matrix( A );
    random_matrix( B );
    random_matrix( C );

    //---------- begin symm_right

    // C = alpha B A + beta C, where A is symmetric, on right side
    // Note B, A order reversed in multiply compared to symm.
    slate::multiply( alpha, B, A, beta, C );                  // simplified API
    slate::symm( slate::Side::Right, alpha, A, B, beta, C );  // traditional API
    //---------- end symm_right
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_hemm_left()
{
    print_func( mpi_rank );

    double alpha = 2.0, beta = 1.0;
    int64_t m=2000, n=1000, nb=256;

    // A is m-by-m, B and C are m-by-n.
    slate::HermitianMatrix<double>
        A( slate::Uplo::Lower, m, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<double> B( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<double> C( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    B.insertLocalTiles();
    C.insertLocalTiles();
    random_matrix( A );
    random_matrix( B );
    random_matrix( C );

    //---------- begin hemm_left

    // C = alpha A B + beta C, where A is Hermitian, on left side
    slate::multiply( alpha, A, B, beta, C );                  // simplified API
    slate::hemm( slate::Side::Left, alpha, A, B, beta, C );   // traditional API
    //---------- end hemm_left
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_hemm_right()
{
    print_func( mpi_rank );

    double alpha = 2.0, beta = 1.0;
    int64_t m=2000, n=1000, nb=256;

    // A is m-by-m, B and C are n-by-m (reverse of left case above).
    slate::HermitianMatrix<double>
        A( slate::Uplo::Lower, m, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<double> B( n, m, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<double> C( n, m, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    B.insertLocalTiles();
    C.insertLocalTiles();
    random_matrix( A );
    random_matrix( B );
    random_matrix( C );

    //---------- begin hemm_right

    // C = alpha B A + beta C, where A is Hermitian, on right side
    // Note B, A order reversed in multiply compared to hemm.
    slate::multiply( alpha, B, A, beta, C );                  // simplified API
    slate::hemm( slate::Side::Right, alpha, A, B, beta, C );  // traditional API
    //---------- end hemm_right
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_syrk_syr2k()
{
    print_func( mpi_rank );

    double alpha = 2.0, beta = 1.0;
    int64_t n=1000, k=500, nb=256;

    slate::Matrix<double> A( n, k, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<double> B( n, k, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::SymmetricMatrix<double>
        C( slate::Uplo::Lower, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    B.insertLocalTiles();
    C.insertLocalTiles();
    random_matrix( A );
    random_matrix( B );
    random_matrix( C );

    //---------- begin syrk

    // C = alpha A A^T + beta C, where C is symmetric
    slate::rank_k_update( alpha, A, beta, C );      // simplified API
    slate::syrk( alpha, A, beta, C );               // traditional API
    //---------- end syrk

    //---------- begin syr2k

    // C = alpha A B^T + alpha B A^T + beta C, where C is symmetric
    slate::rank_2k_update( alpha, A, B, beta, C );  // simplified API
    slate::syr2k( alpha, A, B, beta, C );           // traditional API
    //---------- end syr2k
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_herk_her2k()
{
    print_func( mpi_rank );

    double alpha = 2.0, beta = 1.0;
    int64_t n=1000, k=500, nb=256;

    slate::Matrix<double> A( n, k, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<double> B( n, k, nb, grid_p, grid_q, MPI_COMM_WORLD );
    slate::HermitianMatrix<double>
        C( slate::Uplo::Lower, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    B.insertLocalTiles();
    C.insertLocalTiles();
    random_matrix( A );
    random_matrix( B );
    random_matrix( C );

    //---------- begin herk

    // C = alpha A A^H + beta C, where C is Hermitian
    slate::rank_k_update( alpha, A, beta, C );      // simplified API
    slate::herk( alpha, A, beta, C );               // traditional API
    //---------- end herk

    //---------- begin her2k

    // C = alpha A B^H + conj(alpha) B A^H + beta C, where C is Hermitian
    slate::rank_2k_update( alpha, A, B, beta, C );  // simplified API
    slate::her2k( alpha, A, B, beta, C );           // traditional API
    //---------- end her2k
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_trmm_trsm_left()
{
    print_func( mpi_rank );

    double alpha = 2.0;
    int64_t m=2000, n=1000, nb=256;

    // A is m-by-m, B is m-by-n
    slate::TriangularMatrix<double>
        A( slate::Uplo::Lower, slate::Diag::NonUnit, m, nb,
           grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<double> B( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    B.insertLocalTiles();
    random_matrix( A );
    random_matrix( B );

    //---------- begin trmm_left

    //----- left
    // B = alpha A B, where A is triangular, on left side
    slate::triangular_multiply( alpha, A, B );       // simplified API
    slate::trmm( slate::Side::Left, alpha, A, B );   // traditional API

    // Solve AX = B, where A is triangular, on left side; X overwrites B.
    // That is, B = alpha A^{-1} B.
    slate::triangular_solve( alpha, A, B );          // simplified API
    slate::trsm( slate::Side::Left, alpha, A, B );   // traditional API
    //---------- end trmm_left
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_trmm_trsm_right()
{
    print_func( mpi_rank );

    double alpha = 2.0;
    int64_t m=2000, n=1000, nb=256;

    // A is m-by-m, B is n-by-m (reverse of left case above).
    slate::TriangularMatrix<double>
        A( slate::Uplo::Lower, slate::Diag::NonUnit, m, nb,
           grid_p, grid_q, MPI_COMM_WORLD );
    slate::Matrix<double> B( n, m, nb, grid_p, grid_q, MPI_COMM_WORLD );
    A.insertLocalTiles();
    B.insertLocalTiles();
    random_matrix( A );
    random_matrix( B );

    //---------- begin trmm_right

    //----- right
    // B = alpha B A, where A is triangular, on right side
    // Note B, A order reversed in multiply compared to trmm.
    slate::triangular_multiply( alpha, B, A );       // simplified API
    slate::trmm( slate::Side::Right, alpha, A, B );  // traditional API

    // Solve XA = B, where A is triangular, on right side; X overwrites B.
    // That is, B = alpha B A^{-1}.
    // Note B, A order reversed in solve compared to trsm.
    slate::triangular_solve( alpha, B, A );          // simplified API
    slate::trsm( slate::Side::Right, alpha, A, B );  // traditional API
    //---------- end trmm_right
}

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_all()
{
    test_gemm      < scalar_type >();
    test_gemm_trans< scalar_type >();
    test_symm_left < scalar_type >();
    test_symm_right< scalar_type >();
    test_hemm_left < scalar_type >();
    test_hemm_right< scalar_type >();
    test_syrk_syr2k< scalar_type >();
    test_herk_her2k< scalar_type >();
    test_trmm_trsm_left < scalar_type >();
    test_trmm_trsm_right< scalar_type >();
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
        if (mpi_rank == 0)
            printf( "\n" );

        if (types[ 1 ]) {
            test_all< double >();
        }
        if (mpi_rank == 0)
            printf( "\n" );

        if (types[ 2 ]) {
            test_all< std::complex<float> >();
        }
        if (mpi_rank == 0)
            printf( "\n" );

        if (types[ 3 ]) {
            test_all< std::complex<double> >();
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
