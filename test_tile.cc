#include "test.hh"
#include "slate_Matrix.hh"

// -----------------------------------------------------------------------------
// sets Aij = (g_mpi_rank + 1)*1000 + i + j/1000
void tile_setup_data( slate::Tile<double>& A )
{
    //int m = A.mb();
    int n = A.nb();
    int ld = A.stride();
    double* Ad = A.data();
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < ld; ++i)  // note: to ld, not just m
            Ad[ i + j*ld ] = (g_mpi_rank + 1)*1000 + i + j/1000.;
}

// -----------------------------------------------------------------------------
// sets Aij = 0
void tile_clear_data( slate::Tile<double>& A )
{
    //int m = A.mb();
    int n = A.nb();
    int ld = A.stride();
    double* Ad = A.data();
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < ld; ++i)  // note: to ld, not just m
            Ad[ i + j*ld ] = 0;
}

// -----------------------------------------------------------------------------
// verifies that:
// Aij = (expect_rank + 1)*1000 + i + j/1000 for 0 <= i < m, using A(i,j) operator, and
// Aij = (g_mpi_rank  + 1)*1000 + i + j/1000 for m <= i < stride.
// expect_rank is where the data is coming from.
// Data in the padding shouldn't be modified.
void tile_verify_data( slate::Tile<double>& A, int expect_rank )
{
    int m = A.mb();
    int n = A.nb();
    int ld = A.stride();
    double* Ad = A.data();
    for (int j = 0; j < n; ++j) {
        // for i in [0, m), use expect_rank
        for (int i = 0; i < m; ++i)
            test_assert( A(i,j) == (expect_rank + 1)*1000 + i + j/1000. );

        // for i in [m, ld), use actual rank (data shouldn't be modified)
        for (int i = m; i < ld; ++i)
            test_assert( Ad[ i + j*ld ] = (g_mpi_rank + 1)*1000 + i + j/1000. );
    }
}

// -----------------------------------------------------------------------------
// verifies that:
// Aij = (expect_rank + 1)*1000 + j + i/1000 for 0 <= i < m, using A(i,j) operator.
// expect_rank is where the data is coming from.
// Doesn't check data in padding, since A(i,j) won't allow access.
void tile_verify_data_transpose( slate::Tile<double>& A, int expect_rank )
{
    int m = A.mb();
    int n = A.nb();
    for (int j = 0; j < n; ++j) {
        // for i in [0, m), use expect_rank
        for (int i = 0; i < m; ++i)
            test_assert( A(i,j) == (expect_rank + 1)*1000 + j + i/1000. );
    }
}

// -----------------------------------------------------------------------------
void test_tile( int m, int n )
{
    Test name( __func__ );

    // ----- empty tile
    if (g_mpi_rank == 0) {
        std::cout << "empty tile\n";
    }
    slate::Tile<double> A;
    test_assert( A.mb() == 0 );
    test_assert( A.nb() == 0 );
    test_assert( A.stride() == 0 );
    test_assert( A.op() == blas::Op::NoTrans );
    test_assert( A.uplo() == blas::Uplo::General );
    test_assert( A.data() == nullptr );
    test_assert( A.valid() == false );
    test_assert( A.origin() == true );
    test_assert( A.bytes() == 0 );
    test_assert( A.size() == 0 );

    // todo: tile doesn't have (or otherwise need) host_num,
    // so currently empty tiles just set device to -1 for convenience.
    // Empty tiles are unusable anyhow.
    //test_assert( A.device() == g_host_num );
    test_assert( A.device() == -1 );

    // ----- tile with data
    if (g_mpi_rank == 0) {
        std::cout << "m-by-n tile\n";
    }
    int ld = ((m + 31)/32)*32;
    double* Bd = new double[ ld*n ];
    slate::Tile<double> B( m, n, Bd, ld, g_host_num );
    tile_setup_data( B );
    tile_verify_data( B, g_mpi_rank );
    test_assert( B.mb() == m );
    test_assert( B.nb() == n );
    test_assert( B.stride() == ld );
    test_assert( B.op() == blas::Op::NoTrans );
    test_assert( B.uplo() == blas::Uplo::General );
    test_assert( B.data() == Bd );
    test_assert( B.valid() == true );
    test_assert( B.origin() == true );
    test_assert( B.bytes() == sizeof(double) * m * n );
    test_assert( B.size() == size_t(m * n) );
    test_assert( B.device() == g_host_num );

    // ----- transpose
    if (g_mpi_rank == 0) {
        std::cout << "transpose\n";
    }
    auto BT = transpose( B );
    tile_verify_data_transpose( BT, g_mpi_rank );
    test_assert( BT.mb() == n );
    test_assert( BT.nb() == m );
    test_assert( BT.stride() == ld );
    test_assert( BT.op() == blas::Op::Trans );
    test_assert( BT.uplo() == blas::Uplo::General );
    test_assert( BT.data() == Bd );
    test_assert( BT.valid() == true );
    test_assert( BT.origin() == true );
    test_assert( BT.bytes() == sizeof(double) * m * n );
    test_assert( BT.size() == size_t(m * n) );
    test_assert( BT.device() == g_host_num );

    // ----- transpose again
    if (g_mpi_rank == 0) {
        std::cout << "transpose (2x)\n";
    }
    auto BTT = transpose( BT );
    tile_verify_data( BTT, g_mpi_rank );
    test_assert( BTT.mb() == m );
    test_assert( BTT.nb() == n );
    test_assert( BTT.stride() == ld );
    test_assert( BTT.op() == blas::Op::NoTrans );
    test_assert( BTT.uplo() == blas::Uplo::General );
    test_assert( BTT.data() == Bd );
    test_assert( BTT.valid() == true );
    test_assert( BTT.origin() == true );
    test_assert( BTT.bytes() == sizeof(double) * m * n );
    test_assert( BTT.size() == size_t(m * n) );
    test_assert( BTT.device() == g_host_num );

    // ----- conj_transpose
    if (g_mpi_rank == 0) {
        std::cout << "conj_transpose\n";
    }
    auto BC = conj_transpose( B );
    tile_verify_data_transpose( BC, g_mpi_rank );
    test_assert( BC.mb() == n );
    test_assert( BC.nb() == m );
    test_assert( BC.stride() == ld );
    test_assert( BC.op() == blas::Op::ConjTrans );
    test_assert( BC.uplo() == blas::Uplo::General );
    test_assert( BC.data() == Bd );
    test_assert( BC.valid() == true );
    test_assert( BC.origin() == true );
    test_assert( BC.bytes() == sizeof(double) * m * n );
    test_assert( BC.size() == size_t(m * n) );
    test_assert( BC.device() == g_host_num );

    // ----- conj_transpose again
    if (g_mpi_rank == 0) {
        std::cout << "conj_transpose (2x)\n";
    }
    auto BCC = conj_transpose( BC );
    tile_verify_data( BCC, g_mpi_rank );
    test_assert( BCC.mb() == m );
    test_assert( BCC.nb() == n );
    test_assert( BCC.stride() == ld );
    test_assert( BCC.op() == blas::Op::NoTrans );
    test_assert( BCC.uplo() == blas::Uplo::General );
    test_assert( BCC.data() == Bd );
    test_assert( BCC.valid() == true );
    test_assert( BCC.origin() == true );
    test_assert( BCC.bytes() == sizeof(double) * m * n );
    test_assert( BCC.size() == size_t(m * n) );
    test_assert( BCC.device() == g_host_num );

    // ----- unsupported transpose + conj_transpose
    if (g_mpi_rank == 0) {
        std::cout << "transpose + conj_transpose (illegal)\n";
    }
    test_assert_throw( conj_transpose( BT ), std::exception );
    test_assert_throw( transpose( BC ),      std::exception );

    delete[] Bd;
}

// -----------------------------------------------------------------------------
void test_tile_send( int m, int n )
{
    Test name( __func__ );
    if (g_mpi_size == 1) {
        std::cout << "requires mpi size > 1\n";
        return;
    }

    // S uses stride ld != m, ld rounded up to multiple of 32
    int ld = ((m + 31)/32)*32;
    double* Sd = new double[ ld*n ];
    slate::Tile<double> S( m, n, Sd, ld, g_host_num );

    // C uses contiguous data with stride == m
    double* Cd = new double[ m*n ];
    slate::Tile<double> C( m, n, Cd, m, g_host_num );

    // ----- S => S
    if (g_mpi_rank == 0) {
        std::cout << "send strided    => strided\n";
    }
    tile_setup_data( S );
    tile_setup_data( C );
    if (g_mpi_rank == 0) {
        tile_verify_data( S, g_mpi_rank );
        S.send( 1, g_mpi_comm );
        tile_verify_data( S, g_mpi_rank );  // not changed
    }
    else if (g_mpi_rank == 1) {
        tile_verify_data( S, g_mpi_rank );
        S.recv( 0, g_mpi_comm );
        tile_verify_data( S, 0 );  // expect changed to rank=0
    }

    // ----- C => C
    if (g_mpi_rank == 0) {
        std::cout << "send contiguous => contiguous\n";
    }
    tile_setup_data( S );
    tile_setup_data( C );
    if (g_mpi_rank == 0) {
        tile_verify_data( C, g_mpi_rank );
        C.send( 1, g_mpi_comm );
        tile_verify_data( C, g_mpi_rank );  // not changed
    }
    else if (g_mpi_rank == 1) {
        tile_verify_data( C, g_mpi_rank );
        C.recv( 0, g_mpi_comm );
        tile_verify_data( C, 0 );  // expect changed to rank=0
    }

    // ----- S => C
    if (g_mpi_rank == 0) {
        std::cout << "send strided    => contiguous\n";
    }
    tile_setup_data( S );
    tile_setup_data( C );
    if (g_mpi_rank == 0) {
        tile_verify_data( S, g_mpi_rank );
        S.send( 1, g_mpi_comm );
        tile_verify_data( S, g_mpi_rank );  // not changed
    }
    else if (g_mpi_rank == 1) {
        tile_verify_data( C, g_mpi_rank );
        C.recv( 0, g_mpi_comm );
        tile_verify_data( C, 0 );  // expect changed to rank=0
    }

    // ----- C => S
    if (g_mpi_rank == 0) {
        std::cout << "send contiguous => strided\n";
    }
    tile_setup_data( S );
    tile_setup_data( C );
    if (g_mpi_rank == 0) {
        tile_verify_data( C, g_mpi_rank );
        C.send( 1, g_mpi_comm );
        tile_verify_data( C, g_mpi_rank );  // not changed
    }
    else if (g_mpi_rank == 1) {
        tile_verify_data( S, g_mpi_rank );
        S.recv( 0, g_mpi_comm );
        tile_verify_data( S, 0 );  // expect changed to rank=0
    }

    delete[] Sd;
    delete[] Cd;
}

// -----------------------------------------------------------------------------
void test_tile_bcast( int m, int n )
{
    Test name( __func__ );
    if (g_mpi_size <= 2) {
        std::cout << "requires mpi size >= 4\n";
        return;
    }

    // S uses stride ld != m, ld rounded up to multiple of 32
    int ld = ((m + 31)/32)*32;
    double* Sd = new double[ ld*n ];
    slate::Tile<double> S( m, n, Sd, ld, g_host_num );

    // C uses contiguous data with stride == m
    double* Cd = new double[ m*n ];
    slate::Tile<double> C( m, n, Cd, m, g_host_num );

    tile_setup_data( S );
    tile_setup_data( C );
    tile_verify_data( S, g_mpi_rank );
    tile_verify_data( C, g_mpi_rank );

    // setup broadcast from rank 2 to { 0, 2, 3 }
    std::vector<int> bcast_vec = { 0, 2, 3 };
    std::set<int> bcast_set( bcast_vec.begin(), bcast_vec.end() );
    int root_rank = 2;
    int bcast_rank, bcast_root;
    MPI_Comm bcast_comm;
    MPI_Group mpi_group, bcast_group;
    if (bcast_set.count( g_mpi_rank ) == 1) {
        test_assert( MPI_SUCCESS == MPI_Comm_group( g_mpi_comm, &mpi_group ));
        test_assert( MPI_SUCCESS == MPI_Group_incl( mpi_group, bcast_vec.size(), bcast_vec.data(), &bcast_group ));
        test_assert( MPI_SUCCESS == MPI_Comm_create_group( g_mpi_comm, bcast_group, 0, &bcast_comm ));
        test_assert( MPI_SUCCESS == MPI_Comm_rank( bcast_comm, &bcast_rank ));
        test_assert( MPI_SUCCESS == MPI_Group_translate_ranks( mpi_group, 1, &root_rank, bcast_group, &bcast_root ));

        std::cout << "rank=" << g_mpi_rank << ", expect changed (both strided and contiguous)\n";

        // strided
        if (g_mpi_rank == 0) {
            std::cout << "rank=" << g_mpi_rank << ", strided\n";
        }
        S.bcast( bcast_root, bcast_comm );
        tile_verify_data( S, root_rank );  // expect changed to rank=2

        // contiguous
        if (g_mpi_rank == 0) {
            std::cout << "rank=" << g_mpi_rank << ", contiguous\n";
        }
        C.bcast( bcast_root, bcast_comm );
        tile_verify_data( C, root_rank );  // expect changed to rank=2

        // mixed stride-contiguous broadcast
        if (g_mpi_rank == 0) {
            std::cout << "rank=" << g_mpi_rank << ", mixed\n";
        }
        // reset data
        tile_setup_data( S );
        tile_setup_data( C );

        if (g_mpi_rank == root_rank) {
            S.bcast( bcast_root, bcast_comm );
            tile_verify_data( S, root_rank );  // expect changed to rank=2
        }
        else {
            C.bcast( bcast_root, bcast_comm );
            tile_verify_data( C, root_rank );  // expect changed to rank=2
        }

        test_assert( MPI_SUCCESS == MPI_Comm_free( &bcast_comm ));
        test_assert( MPI_SUCCESS == MPI_Group_free( &bcast_group ));
    }
    else {
        std::cout << "rank=" << g_mpi_rank << ", expect not changed\n";
        tile_verify_data( S, g_mpi_rank );  // not changed
        tile_verify_data( C, g_mpi_rank );  // not changed
    }

    delete[] Sd;
    delete[] Cd;
}

// -----------------------------------------------------------------------------
void test_tile_device( int m, int n )
{
    Test name( __func__ );

    // S uses stride ld != m, ld rounded up to multiple of 32
    int ld = ((m + 31)/32)*32;
    double* Sd = new double[ ld*n ];
    slate::Tile<double> S( m, n, Sd, ld, g_host_num );

    // C uses contiguous data with stride == m
    double* Cd = new double[ m*n ];
    slate::Tile<double> C( m, n, Cd, m, g_host_num );

    double *devSd, *devCd;
    int device = 0;
    cudaStream_t stream;
    test_assert( cudaSuccess == cudaSetDevice( device ));
    test_assert( cudaSuccess == cudaStreamCreate( &stream ));
    test_assert( cudaSuccess == cudaMalloc( &devSd, sizeof(double) * ld*n ));
    test_assert( cudaSuccess == cudaMalloc( &devCd, sizeof(double) *  m*n ));
    slate::Tile<double> devS( m, n, devSd, ld, device );
    slate::Tile<double> devC( m, n, devCd,  m, device );
    MPI_Barrier( g_mpi_comm );

    // ----- S => devS => C
    if (g_mpi_rank == 0) {
        std::cout << "host (strided)    => device (strided)    => host (contiguous)\n";
    }
    tile_setup_data( S );
    tile_clear_data( C );
    tile_verify_data( S, g_mpi_rank );

    S.copyDataToDevice( &devS, stream );
    devS.copyDataToHost( &C, stream );

    tile_verify_data( C, g_mpi_rank );
    MPI_Barrier( g_mpi_comm );

    // ----- S => devC => C
    if (g_mpi_rank == 0) {
        std::cout << "host (strided)    => device (contiguous) => host (contiguous)\n";
    }
    tile_setup_data( S );
    tile_clear_data( C );
    tile_verify_data( S, g_mpi_rank );

    S.copyDataToDevice( &devC, stream );
    devC.copyDataToHost( &C, stream );

    tile_verify_data( C, g_mpi_rank );
    MPI_Barrier( g_mpi_comm );

    // ----- C => devS => S
    if (g_mpi_rank == 0) {
        std::cout << "host (contiguous) => device (strided)    => host (strided)\n";
    }
    tile_setup_data( C );
    tile_clear_data( S );
    tile_verify_data( C, g_mpi_rank );

    C.copyDataToDevice( &devS, stream );
    devS.copyDataToHost( &S, stream );

    tile_verify_data( S, g_mpi_rank );
    MPI_Barrier( g_mpi_comm );

    // ----- C => devC => S
    if (g_mpi_rank == 0) {
        std::cout << "host (contiguous) => device (contiguous) => host (strided)\n";
    }
    tile_setup_data( C );
    tile_clear_data( S );
    tile_verify_data( C, g_mpi_rank );

    C.copyDataToDevice( &devC, stream );
    devC.copyDataToHost( &S, stream );

    tile_verify_data( S, g_mpi_rank );
    MPI_Barrier( g_mpi_comm );

    cudaFree( devSd );
    cudaFree( devCd );
    cudaStreamDestroy( stream );

    delete[] Sd;
    delete[] Cd;
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    g_mpi_comm = MPI_COMM_WORLD;
    MPI_Comm_rank( g_mpi_comm, &g_mpi_rank );
    MPI_Comm_size( g_mpi_comm, &g_mpi_size );

    int m = 200;
    int n = 500;

    if (argc > 1) { m = atoi( argv[1] ); }
    if (argc > 2) { n = atoi( argv[2] ); }

    std::cout << "mpi rank=" << g_mpi_rank
              << ", mpi size=" << g_mpi_size
              << ", num devices=" << g_num_devices
              << ", m=" << m
              << ", n=" << n
              << "\n" << std::flush;
    MPI_Barrier( g_mpi_comm );

    if (g_mpi_rank == 0) {
        std::cout << "\n" << std::flush;
    }
    MPI_Barrier( g_mpi_comm );

    test_tile( m, n );
    test_tile_send( m, n );
    test_tile_bcast( m, n );
    test_tile_device( m, n );

    MPI_Finalize();
    return 0;
}
