#include "test.hh"
#include "slate_Matrix.hh"

// -----------------------------------------------------------------------------
void test_empty()
{
    Test name( __func__ );

    // ----- General
    test_message( "Matrix" );
    slate::Matrix<double> A;

    if (g_mpi_rank == 0) {
        std::cout << "General    A( empty ): mt=" << A.mt()
                  << ", nt=" << A.nt()
                  << ", op=" << char(A.op())
                  << "\n";
    }
    test_assert( A.mt() == 0 );
    test_assert( A.nt() == 0 );
    test_assert( A.op() == blas::Op::NoTrans );

    // ----- Trapezoid
    test_message( "TrapezoidMatrix" );
    slate::TrapezoidMatrix<double> Z;

    if (g_mpi_rank == 0) {
        std::cout << "Trapezoid  Z( empty ): mt=" << Z.mt()
                  << ", nt=" << Z.nt()
                  << ", op=" << char(Z.op())
                  << ", uplo=" << char(Z.uplo())
                  << "\n";
    }
    test_assert( Z.mt() == 0 );
    test_assert( Z.nt() == 0 );
    test_assert( Z.op() == blas::Op::NoTrans );
    test_assert( Z.uplo() == blas::Uplo::Lower );

    // ----- Triangular
    test_message( "TriangularMatrix" );
    slate::TriangularMatrix<double> T;

    if (g_mpi_rank == 0) {
        std::cout << "Triangular T( empty ): mt=" << T.mt()
                  << ", nt=" << T.nt()
                  << ", op=" << char(T.op())
                  << ", uplo=" << char(T.uplo())
                  << "\n";
    }
    test_assert( T.mt() == 0 );
    test_assert( T.nt() == 0 );
    test_assert( T.op() == blas::Op::NoTrans );
    test_assert( T.uplo() == blas::Uplo::Lower );

    // ----- Symmetric
    test_message( "SymmetricMatrix" );
    slate::SymmetricMatrix<double> S;

    if (g_mpi_rank == 0) {
        std::cout << "Symmetric  S( empty ): mt=" << S.mt()
                  << ", nt=" << S.nt()
                  << ", op=" << char(S.op())
                  << ", uplo=" << char(S.uplo())
                  << "\n";
    }
    test_assert( S.mt() == 0 );
    test_assert( S.nt() == 0 );
    test_assert( S.op() == blas::Op::NoTrans );
    test_assert( S.uplo() == blas::Uplo::Lower );

    // ----- Hermitian
    test_message( "HermitianMatrix" );
    slate::HermitianMatrix<double> H;

    if (g_mpi_rank == 0) {
        std::cout << "Hermitian  H( empty ): mt=" << H.mt()
                  << ", nt=" << H.nt()
                  << ", op=" << char(H.op())
                  << ", uplo=" << char(H.uplo())
                  << "\n";
    }
    test_assert( H.mt() == 0 );
    test_assert( H.nt() == 0 );
    test_assert( H.op() == blas::Op::NoTrans );
    test_assert( H.uplo() == blas::Uplo::Lower );
}

// -----------------------------------------------------------------------------
// TESTS
// constructor( m, n, ... )
// swap( A, B )
// Tile operator ( i, j )
// Tile at( i, j )
// mt, nt, op, size
// transpose
// conj_transpose
// tileRank, tileDevice, tileIsLocal, tileMb, tileNb

void test_general( int m, int n, int nb, int p, int q )
{
    Test name( __func__ );

    // A is m-by-n
    int lda = int((m + 31)/32)*32;
    double* Ad = new double[ lda*n ];
    int64_t iseed[] = { 0, 1, 2, 3 };
    lapack::larnv( 1, iseed, lda*n, Ad );
    //slate::Matrix<double> A( m, n, Ad, lda, nb, p, q, g_mpi_comm );
    auto A = slate::Matrix<double>::fromLAPACK(
                m, n, Ad, lda, nb, p, q, g_mpi_comm );

    // B is n-by-n
    int ldb = int((n + 31)/32)*32;
    double* Bd = new double[ ldb*n ];
    lapack::larnv( 1, iseed, ldb*n, Bd );
    //slate::Matrix<double> B( n, n, Bd, ldb, nb, p, q, g_mpi_comm );
    auto B = slate::Matrix<double>::fromLAPACK(
                n, n, Bd, ldb, nb, p, q, g_mpi_comm );

    if (g_mpi_rank == 0) {
        std::cout << "A( m="    << m
                  << ", n="     << n
                  << ", nb="    << nb
                  << " ): mt="  << A.mt()
                  << ", nt="    << A.nt()
                  << ", op="    << char(A.op())
                  << ", size="  << A.size()
                  << "\n";
        std::cout << "B( n="    << n
                  << ", n="     << n
                  << ", nb="    << nb
                  << " ): mt="  << B.mt()
                  << ", nt="    << B.nt()
                  << ", op="    << char(B.op())
                  << ", size="  << B.size()
                  << "\n";
    }
    test_assert( A.mt() == (m + nb - 1) / nb );
    test_assert( A.nt() == (n + nb - 1) / nb );
    test_assert( A.op() == blas::Op::NoTrans );
    if (g_mpi_rank == 0) {
        test_assert( A.tileIsLocal( 0, 0 ) );
        test_assert( A( 0, 0 ).data() == Ad );
        test_assert( A.at( 0, 0 ).data() == Ad );
    }

    test_assert( B.mt() == (n + nb - 1) / nb );
    test_assert( B.nt() == (n + nb - 1) / nb );
    test_assert( B.op() == blas::Op::NoTrans );
    if (g_mpi_rank == 0) {
        test_assert( B.tileIsLocal( 0, 0 ) );
        test_assert( B( 0, 0 ).data() == Bd );
        test_assert( B.at( 0, 0 ).data() == Bd );
    }

    // ----- verify swap
    test_message( "swap( A, B )" );
    // transpose so we can tell if op was swapped
    B = transpose( B );
    test_assert( B.op() == blas::Op::Trans );

    swap( A, B );

    // verify that all data is swapped
    test_assert( B.mt() == (m + nb - 1) / nb );
    test_assert( B.nt() == (n + nb - 1) / nb );
    test_assert( B.op() == blas::Op::NoTrans );
    if (g_mpi_rank == 0) {
        test_assert( B.tileIsLocal( 0, 0 ) );
        test_assert( B( 0, 0 ).data() == Ad );
    }

    test_assert( A.mt() == (n + nb - 1) / nb );
    test_assert( A.nt() == (n + nb - 1) / nb );
    test_assert( A.op() == blas::Op::Trans );
    if (g_mpi_rank == 0) {
        test_assert( A.tileIsLocal( 0, 0 ) );
        test_assert( A( 0, 0 ).data() == Bd );
    }

    // swap again to restore
    swap( A, B );

    // verify that all data is swapped back
    test_assert( A.mt() == (m + nb - 1) / nb );
    test_assert( A.nt() == (n + nb - 1) / nb );
    test_assert( A.op() == blas::Op::NoTrans );
    if (g_mpi_rank == 0) {
        test_assert( A.tileIsLocal( 0, 0 ) );
        test_assert( A( 0, 0 ).data() == Ad );
    }

    test_assert( B.mt() == (n + nb - 1) / nb );
    test_assert( B.nt() == (n + nb - 1) / nb );
    test_assert( B.op() == blas::Op::Trans );
    if (g_mpi_rank == 0) {
        test_assert( B.tileIsLocal( 0, 0 ) );
        test_assert( B( 0, 0 ).data() == Bd );
    }

    // transpose to restore
    B = transpose( B );
    test_assert( B.op() == blas::Op::NoTrans );

    // ----- get info
    int mt = A.mt();
    int nt = A.nt();

    // ----- verify each tile and tile sizes
    test_message( "tile members" );
    for (int j = 0; j < nt; ++j) {
        int jb = (j == nt-1 ? n - j*nb : nb);

        for (int i = 0; i < mt; ++i) {
            int ib = (i == mt-1 ? m - i*nb : nb);

            if (A.tileIsLocal( i, j )) {
                slate::Tile<double> tile  = A( i, j );

                // check tile values
                test_assert( tile.mb() == ib );
                test_assert( tile.nb() == jb );
                test_assert( tile.stride() == lda );
                test_assert( tile.data() == &Ad[ i*nb + j*nb*lda ] );

                // operator (i,j)
                for (int jj = 0; jj < jb; ++jj)
                    for (int ii = 0; ii < ib; ++ii)
                        test_assert( Ad[ (i*nb + ii) + (j*nb + jj)*lda ] == tile( ii, jj ) );

                test_assert( tile.op() == blas::Op::NoTrans );
                test_assert( tile.uplo() == blas::Uplo::General );
                test_assert( tile.origin() == true );
                test_assert( tile.valid()  == true );
                test_assert( tile.device() == g_host_num );
                test_assert( tile.size() == size_t(ib * jb) );
                test_assert( tile.bytes() == sizeof(double) * ib * jb );

                // A( i, j ) and A.at( i, j ) should return identical tiles
                slate::Tile<double> tile2 = A.at( i, j );
                test_assert( tile.mb() == tile2.mb() );
                test_assert( tile.nb() == tile2.nb() );
                test_assert( tile.stride() == tile2.stride() );
                test_assert( tile.data() == tile2.data() );
                test_assert( tile.op() == tile2.op() );
                test_assert( tile.uplo() == tile2.uplo() );
                test_assert( tile.origin() == tile2.origin() );
                test_assert( tile.valid() == tile2.valid() );
                test_assert( tile.device() == tile2.device() );
                test_assert( tile.size() == tile2.size() );
                test_assert( tile.bytes() == tile2.bytes() );
            }

            test_assert( A.tileMb( i ) == ib );
            test_assert( A.tileNb( j ) == jb );
        }
    }

    // ----- verify size (# tiles)
    test_message( "size" );
    size_t size = (mt/p + (mt%p > g_mpi_rank%p)) * (nt/q + (nt%q > g_mpi_rank/p));
    test_assert( A.size() == size );

    // ----- verify tile functions
    test_message( "tileRank, tileDevice, tileIsLocal" );
    for (int j = 0; j < nt; ++j) {
        for (int i = 0; i < mt; ++i) {
            int rank = (i % p) + (j % q)*p;
            test_assert( rank == A.tileRank( i, j ));

            if (g_num_devices > 0) {
                int dev = (j / q) % g_num_devices;
                test_assert( dev == A.tileDevice( i, j ));
            }

            test_assert( (rank == g_mpi_rank) == A.tileIsLocal( i, j ));
        }
    }

    // ----- verify tile transpose
    test_message( "[conj_]transpose( tile )" );
    for (int j = 0; j < nt; ++j) {
        for (int i = 0; i < mt; ++i) {
            if (A.tileIsLocal( i, j )) {
                slate::Tile<double> D = A( i, j );

                auto DT = transpose( D );
                test_assert( DT.op() == blas::Op::Trans );
                test_assert( DT.mb() == D.nb() );
                test_assert( DT.nb() == D.mb() );

                auto DTT = transpose( DT );
                test_assert( DTT.op() == blas::Op::NoTrans );
                test_assert( DTT.mb() == D.mb() );
                test_assert( DTT.nb() == D.nb() );

                auto DC = conj_transpose( D );
                test_assert( DC.op() == blas::Op::ConjTrans );
                test_assert( DC.mb() == D.nb() );
                test_assert( DC.nb() == D.mb() );

                auto DCC = conj_transpose( DC );
                test_assert( DCC.op() == blas::Op::NoTrans );
                test_assert( DCC.mb() == D.mb() );
                test_assert( DCC.nb() == D.nb() );

                // conj_trans( trans( D )) is okay for real, not supported for complex
                auto DTC = conj_transpose( DT );
                test_assert( DTC.op() == blas::Op::NoTrans );
                test_assert( DTC.mb() == D.mb() );
                test_assert( DTC.nb() == D.nb() );

                // trans( conj_trans( D )) is okay for real, not supported for complex
                auto DCT = transpose( DC );
                test_assert( DCT.op() == blas::Op::NoTrans );
                test_assert( DCT.mb() == D.mb() );
                test_assert( DCT.nb() == D.nb() );
            }
        }
    }

    // ----- verify matrix transpose
    test_message( "transpose( A )" );
    auto AT = transpose( A );
    test_assert( AT.mt() == A.nt() );
    test_assert( AT.nt() == A.mt() );
    test_assert( AT.op() == blas::Op::Trans );

    for (int j = 0; j < nt; ++j) {
        int jb = (j == nt-1 ? n - j*nb : nb);

        for (int i = 0; i < mt; ++i) {
            int ib = (i == mt-1 ? m - i*nb : nb);

            test_assert( A.tileRank( i, j ) == AT.tileRank( j, i ) );
            test_assert( A.tileDevice( i, j ) == AT.tileDevice( j, i ) );
            test_assert( A.tileMb( i ) == AT.tileNb( i ) );
            test_assert( A.tileMb( j ) == AT.tileNb( j ) );
            test_assert( A.tileIsLocal( i, j ) == AT.tileIsLocal( j, i ) );
            if (A.tileIsLocal( i, j )) {
                auto T = AT( j, i );
                test_assert( T.data() == A(i,j).data() );
                test_assert( T.op() == blas::Op::Trans );
                test_assert( T.uplo() == blas::Uplo::General );
                test_assert( T.mb() == jb );
                test_assert( T.nb() == ib );
                test_assert( T.stride() == lda );
            }
        }
    }

    auto ATT = transpose( AT );
    test_assert( ATT.mt() == A.mt() );
    test_assert( ATT.nt() == A.nt() );
    test_assert( ATT.op() == blas::Op::NoTrans );

    // conj_trans( trans( A )) is okay for real, not supported for complex
    auto ATC = conj_transpose( AT );
    test_assert( ATC.mt() == A.mt() );
    test_assert( ATC.nt() == A.nt() );
    test_assert( ATC.op() == blas::Op::NoTrans );

    // ----- verify matrix conj_transpose
    test_message( "conj_transpose( A )" );
    auto AC = conj_transpose( A );
    test_assert( AC.mt() == A.nt() );
    test_assert( AC.nt() == A.mt() );
    test_assert( AC.op() == blas::Op::ConjTrans );

    for (int j = 0; j < nt; ++j) {
        int jb = (j == nt-1 ? n - j*nb : nb);

        for (int i = 0; i < mt; ++i) {
            int ib = (i == mt-1 ? m - i*nb : nb);

            test_assert( A.tileRank( i, j ) == AC.tileRank( j, i ) );
            test_assert( A.tileDevice( i, j ) == AC.tileDevice( j, i ) );
            //std::cout << "i=" << i << ", j=" << j
            //          << ", A.mb=" << A.tileMb(i)
            //          << ", A.nb=" << A.tileNb(j)
            //          << ", AC.mb=" << AC.tileMb(i)
            //          << ", AC.nb=" << AC.tileNb(j) << "\n";
            test_assert( A.tileMb( i ) == AC.tileNb( i ) );
            test_assert( A.tileMb( j ) == AC.tileNb( j ) );
            test_assert( A.tileIsLocal( i, j ) == AC.tileIsLocal( j, i ) );
            if (A.tileIsLocal( i, j )) {
                auto T = AC( j, i );
                test_assert( T.data() == A(i,j).data() );
                test_assert( T.op() == blas::Op::ConjTrans );
                test_assert( T.uplo() == blas::Uplo::General );
                test_assert( T.mb() == jb );
                test_assert( T.nb() == ib );
                test_assert( T.stride() == lda );
            }
        }
    }

    auto ACC = conj_transpose( AC );
    test_assert( ACC.mt() == A.mt() );
    test_assert( ACC.nt() == A.nt() );
    test_assert( ACC.op() == blas::Op::NoTrans );

    // trans( conj_trans( A )) is okay for real, not supported for complex
    auto ACT = transpose( AC );
    test_assert( ACT.mt() == A.mt() );
    test_assert( ACT.nt() == A.nt() );
    test_assert( ACT.op() == blas::Op::NoTrans );

    delete[] Ad;

    // ----- test complex: conj_trans( trans( Z )) is unsupported
    test_message( "complex transpose and conj_transpose" );
    std::complex<double> *Zd = new std::complex<double>[ lda*n ];
    auto Z = slate::Matrix< std::complex<double> >::fromLAPACK(
                m, n, Zd, lda, nb, p, q, g_mpi_comm );

    auto ZT  = transpose( Z );
    test_assert( ZT.mt() == Z.nt() );
    test_assert( ZT.nt() == Z.mt() );
    test_assert( ZT.op() == blas::Op::Trans );

    auto ZTT = transpose( ZT );
    test_assert( ZTT.mt() == Z.mt() );
    test_assert( ZTT.nt() == Z.nt() );
    test_assert( ZTT.op() == blas::Op::NoTrans );

    test_assert_throw( conj_transpose( ZT ), std::exception );

    auto ZC  = conj_transpose( Z );
    test_assert( ZC.mt() == Z.nt() );
    test_assert( ZC.nt() == Z.mt() );
    test_assert( ZC.op() == blas::Op::ConjTrans );

    auto ZCC = conj_transpose( ZC );
    test_assert( ZCC.mt() == Z.mt() );
    test_assert( ZCC.nt() == Z.nt() );
    test_assert( ZCC.op() == blas::Op::NoTrans );

    test_assert_throw( transpose( ZC ), std::exception );

    for (int j = 0; j < Z.nt(); ++j) {
        for (int i = 0; i < Z.mt(); ++i) {
            if (Z.tileIsLocal( i, j )) {
                auto D = Z( i, j );

                auto DT  = transpose( D );
                test_assert( DT.op() == blas::Op::Trans );

                auto DTT = transpose( DT );
                test_assert( DTT.op() == blas::Op::NoTrans );

                test_assert_throw( conj_transpose( DT ), std::exception );

                auto DC  = conj_transpose( D );
                test_assert( DC.op() == blas::Op::ConjTrans );

                auto DCC = conj_transpose( DC );
                test_assert( DCC.op() == blas::Op::NoTrans );

                test_assert_throw( transpose( DC ), std::exception );
            }
        }
    }

    delete[] Zd;
}

// -----------------------------------------------------------------------------
// TESTS
// sub(), sub constructor
// getRanks
void test_general_sub( int m, int n, int nb, int p, int q )
{
    Test name( __func__ );

    // A is m-by-n
    int lda = int((m + 31)/32)*32;
    double* Ad = new double[ lda*n ];
    int64_t iseed[] = { 0, 1, 2, 3 };
    lapack::larnv( 1, iseed, lda*n, Ad );
    //slate::Matrix<double> A( m, n, Ad, lda, nb, p, q, g_mpi_comm );
    auto A = slate::Matrix<double>::fromLAPACK(
                m, n, Ad, lda, nb, p, q, g_mpi_comm );

    int64_t mt = A.mt();
    int64_t nt = A.nt();

    // test 10 random sub-matrices
    for (int cnt = 0; cnt < 10; ++cnt) {
        int i1 = rand() % mt;
        int i2 = rand() % mt;
        int j1 = rand() % nt;
        int j2 = rand() % nt;

        // rectify indices, allowing some zero-size arrays (i2 < i1 or j2 < j1)
        if (i2 < i1 - 2)
            std::swap( i1, i2 );
        if (j2 < j1 - 2)
            std::swap( j1, j2 );

        test_assert( 0 <= i1 && i1 - 2 <= i2 && i2 < mt );
        test_assert( 0 <= j1 && j1 - 2 <= j2 && j2 < nt );

        auto As = A.sub( i1, i2, j1, j2 );
        slate::Matrix<double> As2( A, i1, i2, j1, j2 );
        if (g_mpi_rank == 0) {
            std::cout << "A.sub( " << std::setw(3) << i1 << ": " << std::setw(3) << i2
                      << ", "      << std::setw(3) << j1 << ": " << std::setw(3) << j2
                      << "): mt="  << std::setw(3) << As.mt()
                      << ", nt="   << std::setw(3) << As.nt()
                      << "\n";
                      //<< ", A.size=" << A.size()
                      //<< ", A.sub.size=" << As.size() << "\n";
        }

        // fix indices so empty arrays are only i2 = i1 - 1 or j2 = j1 - 1,
        // to make mt and nt right
        if (i2 < i1 - 1)
            i2 = i1 - 1;
        if (j2 < j1 - 1)
            j2 = j1 - 1;

        test_assert( 0 <= i1 && i1 - 1 <= i2 && i2 < mt );
        test_assert( 0 <= j1 && j1 - 1 <= j2 && j2 < nt );

        test_assert( 0 <= As.mt() && As.mt() <= A.mt() );
        test_assert( 0 <= As.nt() && As.nt() <= A.nt() );

        test_assert( As.mt() == i2 - i1 + 1 );
        test_assert( As.nt() == j2 - j1 + 1 );

        test_assert( As2.mt() == i2 - i1 + 1 );
        test_assert( As2.nt() == j2 - j1 + 1 );

        std::set<int> ranks, ranks2;
        As.getRanks( &ranks );

        for (int j = 0; j < As.nt(); ++j) {
            for (int i = 0; i < As.mt(); ++i) {
                test_assert( As.tileRank   ( i, j ) == A.tileRank   ( i + i1, j + j1 ));
                test_assert( As.tileDevice ( i, j ) == A.tileDevice ( i + i1, j + j1 ));
                test_assert( As.tileRank   ( i, j ) == A.tileRank   ( i + i1, j + j1 ));
                test_assert( As.tileMb     ( i )    == A.tileMb     ( i + i1 ));
                test_assert( As.tileNb     ( j )    == A.tileNb     ( j + j1 ));
                test_assert( As.tileIsLocal( i, j ) == A.tileIsLocal( i + i1, j + j1 ));

                test_assert( As2.tileRank   ( i, j ) == A.tileRank   ( i + i1, j + j1 ));
                test_assert( As2.tileDevice ( i, j ) == A.tileDevice ( i + i1, j + j1 ));
                test_assert( As2.tileRank   ( i, j ) == A.tileRank   ( i + i1, j + j1 ));
                test_assert( As2.tileMb     ( i )    == A.tileMb     ( i + i1 ));
                test_assert( As2.tileNb     ( j )    == A.tileNb     ( j + j1 ));
                test_assert( As2.tileIsLocal( i, j ) == A.tileIsLocal( i + i1, j + j1 ));

                if (As.tileIsLocal( i, j )) {
                    auto T  = A( i + i1, j + j1 );
                    auto T1 = As( i, j );
                    auto T2 = As2( i, j );
                    test_assert( T.data() == T1.data() );
                    test_assert( T.mb()   == T1.mb()   );
                    test_assert( T.nb()   == T1.nb()   );
                    test_assert( T.op()   == T1.op()   );
                    test_assert( T.uplo() == T1.uplo() );

                    test_assert( T.data() == T2.data() );
                    test_assert( T.mb()   == T2.mb()   );
                    test_assert( T.nb()   == T2.nb()   );
                    test_assert( T.op()   == T2.op()   );
                    test_assert( T.uplo() == T2.uplo() );
                }

                ranks2.insert( As.tileRank( i, j ));
            }
        }

        test_assert( ranks == ranks2 );

        //if (g_mpi_rank == 0) {
        //    std::cout << " rank=0, getRanks={ ";
        //    for (auto iter = ranks.begin(); iter != ranks.end(); ++iter)
        //        std::cout << *iter << ", ";
        //    std::cout << "}";
        //    std::cout << "; { ";
        //    for (auto iter = ranks2.begin(); iter != ranks2.end(); ++iter)
        //        std::cout << *iter << ", ";
        //    std::cout << "}\n";
        //}
    }

    delete[] Ad;
}

// -----------------------------------------------------------------------------
// TESTS
// tileBcast( i, j, A ), tileTick, tileLife, numLocalTiles
// tileBcastToSet( i, j, bcast_set )  // implicit
void test_general_send( int m, int n, int nb, int p, int q )
{
    Test name( __func__ );

    // A is m-by-n
    int lda = int((m + 31)/32)*32;
    double* Ad = new double[ lda*n ];
    int64_t iseed[] = { 0, 1, 2, 3 };
    lapack::larnv( 1, iseed, lda*n, Ad );
    //slate::Matrix<double> A( m, n, Ad, lda, nb, p, q, g_mpi_comm );
    auto A = slate::Matrix<double>::fromLAPACK(
                m, n, Ad, lda, nb, p, q, g_mpi_comm );

    // -------------------- broadcast A(0,0) to all of A
    test_message( "broadcast A(0,0) to all of A" );
    A.tileBcast( 0, 0, A );

    auto Aij = A( 0, 0 );  // should be on all ranks now
    unused( Aij );

    // ----- verify life
    test_message( "life" );
    int life = A.tileLife( 0, 0 );
    int life2 = 0;
    if (! A.tileIsLocal( 0, 0 )) {
        for (int j = 0; j < A.nt(); ++j) {
            for (int i = 0; i < A.mt(); ++i) {
                if (A.tileIsLocal( i, j )) {
                    life2 += 1;
                    A.tileTick( 0, 0 );
                }
            }
        }
        test_assert( A.numLocalTiles() == life2 );
    }
    test_assert( life == life2 );

    // ----- verify tileTick
    test_message( "tileTick" );
    // after all the tickTicks, life of A(0,0) should have been decrement to 0
    // and deleted the A(0,0) tile
    test_assert( A.tileLife( 0, 0 ) == 0 );
    if (A.tileIsLocal( 0, 0 ))
        Aij = A( 0, 0 );  // local should still work
    else
        test_assert_throw( A( 0, 0 ), std::out_of_range );

    // -------------------- broadcast A(1,1) across row A(1,2:nt)
    test_message( "broadcast A(1,1) to A(1,2:nt)" );
    auto Arow = A.sub( 1, 1, 2, A.nt()-1 );
    A.tileBcast( 1, 1, Arow );

    // ----- verify life
    test_message( "life" );
    life2 = 0;
    bool row_is_local = A.tileIsLocal( 1, 1 );
    if (! A.tileIsLocal( 1, 1 )) {
        for (int j = 2; j < A.nt(); ++j) {
            if (A.tileIsLocal( 1, j )) {
                life2 += 1;
                row_is_local = true;
            }
        }
        test_assert( Arow.numLocalTiles() == life2 );
    }
    test_assert( A.tileLife( 1, 1 ) == life2 );
    // any rank that has a tile in row 1 should now have A11;
    // other ranks should throw exception
    if (row_is_local)
        Aij = A( 1, 1 );
    else
        test_assert_throw( A( 1, 1 ), std::exception );

    // ----- verify tileTick
    test_message( "tileTick" );
    // tileTick should reduce life to 0
    for (int j = 2; j < A.nt(); ++j) {
        if (A.tileIsLocal( 1, j )) {
            A.tileTick( 1, 1 );
        }
    }
    test_assert( A.tileLife( 1, 1 ) == 0 );

    // now only rank where A11 is local should have A11
    if (A.tileIsLocal( 1, 1 ))
        Aij = A( 1, 1 );
    else
        test_assert_throw( A( 1, 1 ), std::exception );

    // -------------------- broadcast A(1,1) across col A(2:mt,1)
    test_message( "broadcast A(1,1) to A(2:mt,1)" );
    auto Acol = A.sub( 2, A.mt()-1, 1, 1 );
    A.tileBcast( 1, 1, Acol );

    // ----- verify life
    test_message( "life" );
    life2 = 0;
    bool col_is_local = A.tileIsLocal( 1, 1 );
    if (! A.tileIsLocal( 1, 1 )) {
        for (int i = 2; i < A.mt(); ++i) {
            if (A.tileIsLocal( i, 1 )) {
                life2 += 1;
                col_is_local = true;
            }
        }
        test_assert( Acol.numLocalTiles() == life2 );
    }
    test_assert( A.tileLife( 1, 1 ) == life2 );
    // any rank that has a tile in row 1 should now have A11;
    // other ranks should throw exception
    if (col_is_local)
        Aij = A( 1, 1 );
    else
        test_assert_throw( A( 1, 1 ), std::exception );

    // ----- verify tileTick
    test_message( "tileTick" );
    // tileTick should reduce life to 0
    for (int i = 2; i < A.mt(); ++i) {
        if (A.tileIsLocal( i, 1 )) {
            A.tileTick( 1, 1 );
        }
    }
    test_assert( A.tileLife( 1, 1 ) == 0 );

    // now only rank where A11 is local should have A11
    if (A.tileIsLocal( 1, 1 ))
        Aij = A( 1, 1 );
    else
        test_assert_throw( A( 1, 1 ), std::exception );
}

// -----------------------------------------------------------------------------
// TESTS
// tileBcast( i, j, A1, A2 ), tileTick, tileLife, numLocalTiles
// tileBcastToSet( i, j, bcast_set )  // implicit
void test_general_send2( int m, int n, int nb, int p, int q )
{
    Test name( __func__ );

    // A is m-by-n
    int lda = int((m + 31)/32)*32;
    double* Ad = new double[ lda*n ];
    int64_t iseed[] = { 0, 1, 2, 3 };
    lapack::larnv( 1, iseed, lda*n, Ad );
    //slate::Matrix<double> A( m, n, Ad, lda, nb, p, q, g_mpi_comm );
    auto A = slate::Matrix<double>::fromLAPACK(
                m, n, Ad, lda, nb, p, q, g_mpi_comm );

    // -------------------- broadcast A(1,1) across row A(1,2:nt) and col A(2:mt,1)
    test_message( "broadcast A(1,1) to A(1,2:nt) and A(2:mt,1)" );
    auto Arow = A.sub( 1, 1, 2, A.nt()-1 );
    auto Acol = A.sub( 2, A.mt()-1, 1, 1 );
    A.tileBcast( 1, 1, Arow, Acol );
    slate::Tile<double> Aij;

    // ----- verify life
    test_message( "life" );
    int life2 = 0;
    bool blk_is_local = A.tileIsLocal( 1, 1 );
    if (! A.tileIsLocal( 1, 1 )) {
        for (int j = 2; j < A.nt(); ++j) {
            if (A.tileIsLocal( 1, j )) {
                life2 += 1;
                blk_is_local = true;
            }
        }
        test_assert( Arow.numLocalTiles() == life2 );

        for (int i = 2; i < A.mt(); ++i) {
            if (A.tileIsLocal( i, 1 )) {
                life2 += 1;
                blk_is_local = true;
            }
        }
        test_assert( Arow.numLocalTiles() + Acol.numLocalTiles() == life2 );
    }
    test_assert( A.tileLife( 1, 1 ) == life2 );
    // any rank that has a tile in row 1 should now have A11;
    // other ranks should throw exception
    if (blk_is_local)
        Aij = A( 1, 1 );
    else
        test_assert_throw( A( 1, 1 ), std::exception );

    // ----- verify tileTick
    test_message( "tileTick" );
    // tileTick should reduce life to 0
    for (int j = 2; j < A.nt(); ++j) {
        if (A.tileIsLocal( 1, j )) {
            A.tileTick( 1, 1 );
        }
    }
    for (int i = 2; i < A.mt(); ++i) {
        if (A.tileIsLocal( i, 1 )) {
            A.tileTick( 1, 1 );
        }
    }
    test_assert( A.tileLife( 1, 1 ) == 0 );

    // now only rank where A11 is local should have A11
    if (A.tileIsLocal( 1, 1 ))
        Aij = A( 1, 1 );
    else
        test_assert_throw( A( 1, 1 ), std::exception );
}

// -----------------------------------------------------------------------------
// TESTS
// compute_stream, comm_stream, cublas_handle
void test_cuda_streams( int m, int n, int nb, int p, int q )
{
    Test name( __func__ );

    // A is m-by-n
    int lda = int((m + 31)/32)*32;
    double* Ad = new double[ lda*n ];
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            Ad[ i + j*lda ] = i + j/1000.;
    //slate::Matrix<double> A( m, n, Ad, lda, nb, p, q, g_mpi_comm );
    auto A = slate::Matrix<double>::fromLAPACK(
                m, n, Ad, lda, nb, p, q, g_mpi_comm );

    for (int device = 0; device < g_num_devices; ++device) {
        double* devAd;
        cudaError_t err;
        size_t size = sizeof(double) * lda*n;
        err = cudaMalloc( (void**) &devAd, size );
        test_assert( err == cudaSuccess );

        // ----- compute stream
        test_message( "compute stream" );
        test_assert( A.compute_stream( device ) != nullptr );

        err = cudaMemcpyAsync( devAd, Ad, size, cudaMemcpyHostToDevice, A.compute_stream( device ));
        test_assert( err == cudaSuccess );

        err = cudaStreamSynchronize( A.compute_stream( device ));
        test_assert( err == cudaSuccess );

        // ----- comm stream
        test_message( "comm stream" );
        test_assert( A.comm_stream( device ) != nullptr );

        err = cudaMemcpyAsync( devAd, Ad, size, cudaMemcpyHostToDevice, A.comm_stream( device ));
        test_assert( err == cudaSuccess );

        err = cudaStreamSynchronize( A.comm_stream( device ));
        test_assert( err == cudaSuccess );

        // comm and compute streams different
        test_assert( A.comm_stream( device ) != A.compute_stream( device ) );

        // ----- cublas handle
        test_message( "cublas handle" );
        cudaStream_t stream;
        cublasStatus_t status;
        status = cublasGetStream( A.cublas_handle( device ), &stream );
        test_assert( status == CUBLAS_STATUS_SUCCESS );

        // cublas handle uses compute stream
        test_assert( stream == A.compute_stream( device ) );

        // verify use
        double result;
        status = cublasDasum( A.cublas_handle( device ), m, devAd, 1, &result );
        test_assert( status == CUBLAS_STATUS_SUCCESS );

        err = cudaStreamSynchronize( stream );
        test_assert( err == cudaSuccess );

        // sum( 0, ..., m-1 ) = (m-1)*m/2
        test_assert( result == (m-1)*m/2 );

        // ----- cleanup
        err = cudaFree( devAd );
        test_assert( err == cudaSuccess );
        devAd = nullptr;
    }
}

// -----------------------------------------------------------------------------
// TESTS
// allocateBatchArrays
// clearBatchArrays
// {a, b, c}_array_{host, device}
void test_batch_arrays( int m, int n, int nb, int p, int q )
{
    Test name( __func__ );

    // A is m-by-n
    int lda = int((m + 31)/32)*32;
    double* Ad = new double[ lda*n ];
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            Ad[ i + j*lda ] = i + j/1000.;
    //slate::Matrix<double> A( m, n, Ad, lda, nb, p, q, g_mpi_comm );
    auto A = slate::Matrix<double>::fromLAPACK(
                m, n, Ad, lda, nb, p, q, g_mpi_comm );

    test_message( "allocateBatchArrays" );
    A.allocateBatchArrays();

    // test filling in each host array and copying to device array
    for (int device = 0; device < g_num_devices; ++device) {
        int size = A.getMaxDeviceTiles( device );
        double **array_host, **array_device;
        cudaError_t err;

        // -----
        test_message( "a_array" );
        array_host   = A.a_array_host  ( device );
        array_device = A.a_array_device( device );
        test_assert( array_host   != nullptr );
        test_assert( array_device != nullptr );
        for (int i = 0; i < size; ++i)
            array_host[i] = Ad;
        err = cudaMemcpy( array_device, array_host, size, cudaMemcpyHostToDevice );
        test_assert( err == cudaSuccess );

        // -----
        test_message( "b_array" );
        array_host   = A.b_array_host  ( device );
        array_device = A.b_array_device( device );
        test_assert( array_host   != nullptr );
        test_assert( array_device != nullptr );
        for (int i = 0; i < size; ++i)
            array_host[i] = Ad;
        err = cudaMemcpy( array_device, array_host, size, cudaMemcpyHostToDevice );
        test_assert( err == cudaSuccess );

        // -----
        test_message( "c_array" );
        array_host   = A.c_array_host  ( device );
        array_device = A.c_array_device( device );
        test_assert( array_host   != nullptr );
        test_assert( array_device != nullptr );
        for (int i = 0; i < size; ++i)
            array_host[i] = Ad;
        err = cudaMemcpy( array_device, array_host, size, cudaMemcpyHostToDevice );
        test_assert( err == cudaSuccess );
    }

    test_message( "clearBatchArrays" );
    A.clearBatchArrays();

    // now all the arrays should give errors
    for (int device = 0; device < g_num_devices; ++device) {
        test_assert_throw( A.a_array_host( device ), std::exception );
        test_assert_throw( A.b_array_host( device ), std::exception );
        test_assert_throw( A.c_array_host( device ), std::exception );

        test_assert_throw( A.a_array_device( device ), std::exception );
        test_assert_throw( A.b_array_device( device ), std::exception );
        test_assert_throw( A.c_array_device( device ), std::exception );
    }
}

// -----------------------------------------------------------------------------
// TESTS
// tileCopyToDevice
void test_copyToDevice( int m, int n, int nb, int p, int q )
{
    Test name( __func__ );

    // A is m-by-n
    int lda = int((m + 31)/32)*32;
    double* Ad = new double[ lda*n ];
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            Ad[ i + j*lda ] = i + j/1000.;
    //slate::Matrix<double> A( m, n, Ad, lda, nb, p, q, g_mpi_comm );
    auto A = slate::Matrix<double>::fromLAPACK(
                m, n, Ad, lda, nb, p, q, g_mpi_comm );

    for (int device = 0; device < g_num_devices; ++device) {
        if (A.tileIsLocal( 0, 0 )) {
            // initially, tile exists only on host
            auto Aij = A( 0, 0 );
            test_assert_throw( Aij = A( 0, 0, device ), std::exception );

            // after copy, both host and device data are valid
            test_message( "copyToDevice" );
            A.tileCopyToDevice( 0, 0, device );
            test_assert( A( 0, 0 ).valid() );
            test_assert( A( 0, 0, device ).valid() );
        }
    }
}

// -----------------------------------------------------------------------------
// TESTS
// tileMoveToDevice
// tileCopyToHost
// tileMoveToHost
void test_copyToHost( int m, int n, int nb, int p, int q )
{
    Test name( __func__ );

    // A is m-by-n
    int lda = int((m + 31)/32)*32;
    double* Ad = new double[ lda*n ];
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            Ad[ i + j*lda ] = i + j/1000.;
    //slate::Matrix<double> A( m, n, Ad, lda, nb, p, q, g_mpi_comm );
    auto A = slate::Matrix<double>::fromLAPACK(
                m, n, Ad, lda, nb, p, q, g_mpi_comm );

    // todo: this doesn't test if the tile exists only on device,
    // because that currently doesn't happen: data starts on host and is copied
    // to the device, but the host tile always exists.
    for (int device = 0; device < g_num_devices; ++device) {
        if (A.tileIsLocal( 0, 0 )) {
            // initially, tile exists only on host
            auto Aij = A( 0, 0 );
            test_assert_throw( Aij = A( 0, 0, device ), std::exception );

            // after move, both tiles exist, but only device is valid
            test_message( "tileMoveToDevice" );
            A.tileMoveToDevice( 0, 0, device );  // invalidates host data
            test_assert( ! A( 0, 0 ).valid() );
            test_assert( A( 0, 0, device ).valid() );

            // after copy, both host and device are valid
            test_message( "tileCopyToHost" );
            A.tileCopyToHost( 0, 0, device );
            test_assert( A( 0, 0 ).valid() );
            test_assert( A( 0, 0, device ).valid() );

            // after move, both tiles exist, but only host is valid
            test_message( "tileMoveToHost" );
            A.tileMoveToHost( 0, 0, device );
            test_assert( A( 0, 0 ).valid() );
            test_assert( ! A( 0, 0, device ).valid() );
        }
    }
}

// -----------------------------------------------------------------------------
// TESTS
// reserveHostWorkspace
// reserveDeviceWorkspace
// clearWorkspace
void test_workspace( int m, int n, int nb, int p, int q )
{
    Test name( __func__ );

    // A is m-by-n
    int lda = int((m + 31)/32)*32;
    double* Ad = new double[ lda*n ];
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            Ad[ i + j*lda ] = i + j/1000.;
    //slate::Matrix<double> A( m, n, Ad, lda, nb, p, q, g_mpi_comm );
    auto A = slate::Matrix<double>::fromLAPACK(
                m, n, Ad, lda, nb, p, q, g_mpi_comm );

    A.reserveHostWorkspace();
    A.reserveDeviceWorkspace();
    A.clearWorkspace();
}

// -----------------------------------------------------------------------------
// TESTS
// general Matrix gather
void test_gather( int m, int n, int nb, int p, int q )
{
    Test name( __func__ );

    // A is m-by-n
    int lda = int((m + 31)/32)*32;
    double* Ad = new double[ lda*n ];
    double* Bd = new double[ lda*n ];
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Ad[ i + j*lda ] = g_mpi_rank*1000 + i + j/1000.;
            Bd[ i + j*lda ] = nan("");
        }
    }
    //slate::Matrix<double> A( m, n, Ad, lda, nb, p, q, g_mpi_comm );
    auto A = slate::Matrix<double>::fromLAPACK(
                m, n, Ad, lda, nb, p, q, g_mpi_comm );

    // -----
    test_message( "gather into A" );
    A.gather( Ad, lda );

    test_message( "check A" );
    if (g_mpi_rank == 0) {
        // loop over tiles
        for (int j = 0; j < A.nt(); ++j) {
            int jb = A.tileNb( j );
            for (int i = 0; i < A.mt(); ++i) {
                int ib = A.tileMb( i );

                // verify tile Aij exists
                auto Aij = A( i, j );
                unused( Aij );

                // loop over tile {i, j} in Ad, verify data came from correct rank
                int tile_rank = A.tileRank( i, j );
                for (int jj = j*nb; jj < j*nb + jb; ++jj) {
                    for (int ii = i*nb; ii < i*nb + ib; ++ii) {
                        test_assert( Ad[ ii + jj*lda ] == tile_rank*1000 + ii + jj/1000. );
                    }
                }
            }
        }
    }

    // -----
    test_message( "gather into B" );
    A.gather( Bd, lda );

    test_message( "check B" );
    if (g_mpi_rank == 0) {
        // loop over tiles
        for (int j = 0; j < A.nt(); ++j) {
            int jb = A.tileNb( j );
            for (int i = 0; i < A.mt(); ++i) {
                int ib = A.tileMb( i );

                // verify tile Aij exists
                auto Aij = A( i, j );
                unused( Aij );

                // loop over tile {i, j} in Bd, verify data came from correct rank
                int tile_rank = A.tileRank( i, j );
                for (int jj = j*nb; jj < j*nb + jb; ++jj) {
                    for (int ii = i*nb; ii < i*nb + ib; ++ii) {
                        test_assert( Bd[ ii + jj*lda ] == tile_rank*1000 + ii + jj/1000. );
                    }
                }
            }
        }
    }

    delete[] Ad;
    Ad = nullptr;

    delete[] Bd;
    Bd = nullptr;
}

// -----------------------------------------------------------------------------
// TESTS
// Hermitian gather
void test_hermitian_gather( blas::Uplo uplo, int n, int nb, int p, int q )
{
    Test name( __func__ );
    if (g_mpi_rank == 0) {
        std::cout << "uplo " << char(uplo) << "\n";
    }

    // A is n-by-n
    int lda = int((n + 31)/32)*32;
    double* Ad = new double[ lda*n ];
    double* Bd = new double[ lda*n ];
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            Ad[ i + j*lda ] = g_mpi_rank*1000 + i + j/1000.;
            Bd[ i + j*lda ] = nan("");
        }
    }
    //slate::HermitianMatrix<double> A( uplo, n, Ad, lda, nb, p, q, g_mpi_comm );
    auto A = slate::HermitianMatrix<double>::fromLAPACK(
                uplo, n, Ad, lda, nb, p, q, g_mpi_comm );

    // for lower, set block upper triangle (excluding diagonal tiles) to inf;
    // for upper, set block lower triangle (excluding diagonal tiles) to inf;
    // outside local tiles, set data to nan
    for (int j = 0; j < A.nt(); ++j) {
        int jb = A.tileNb( j );
        for (int i = 0; i < A.mt(); ++i) {
            int ib = A.tileMb( i );

            if ((uplo == blas::Uplo::Lower && i < j) ||  // block upper
                (uplo == blas::Uplo::Upper && i > j)) {  // block lower
                for (int jj = j*nb; jj < j*nb + jb; ++jj) {
                    for (int ii = i*nb; ii < i*nb + ib; ++ii) {
                        Ad[ ii + jj*lda ] = INFINITY;
                    }
                }
            }
            else if (! A.tileIsLocal( i, j )) {
                for (int jj = j*nb; jj < j*nb + jb; ++jj) {
                    for (int ii = i*nb; ii < i*nb + ib; ++ii) {
                        Ad[ ii + jj*lda ] = NAN;
                    }
                }
            }
        }
    }

    //for (int rank = 0; rank < g_mpi_size; ++rank) {
    //    if (rank == g_mpi_rank) {
    //        printf( "rank %d, A=", g_mpi_rank );
    //        print( n, n, Ad, lda );
    //    }
    //    fflush( 0 );
    //    test_message( "print A" );
    //}

    // -----
    test_message( "gather into A" );
    A.gather( Ad, lda );

    if (g_mpi_rank == 0) {
        //printf( "rank %d, A=", g_mpi_rank );
        //print( n, n, Ad, lda );

        // loop over tiles
        if (uplo == blas::Uplo::Lower) {
            for (int j = 0; j < A.nt(); ++j) {
                int jb = A.tileNb( j );
                for (int i = j; i < A.mt(); ++i) {  // lower
                    int ib = A.tileMb( i );

                    // verify tile Aij exists
                    auto Aij = A( i, j );
                    unused( Aij );

                    // loop over tile {i, j} in Ad, verify data came from correct rank
                    int tile_rank = A.tileRank( i, j );
                    for (int jj = j*nb; jj < j*nb + jb; ++jj) {
                        for (int ii = i*nb; ii < i*nb + ib; ++ii) {
                            //if (Ad[ ii + jj*lda ] != tile_rank*1000 + ii + jj/1000.) {
                            //    printf( "%3d, %3d, Ad=%8.4f, expect=%8.4f\n",
                            //            ii, jj, Ad[ ii + jj*lda ],
                            //            tile_rank*1000 + ii + jj/1000. );
                            //}
                            test_assert( Ad[ ii + jj*lda ] == tile_rank*1000 + ii + jj/1000. );
                        }
                    }
                }
            }
        }
        else {
            for (int j = 0; j < A.nt(); ++j) {
                int jb = A.tileNb( j );
                for (int i = 0; i <= j && i < A.mt(); ++i) {  // upper
                    int ib = A.tileMb( i );

                    // verify tile Aij exists
                    auto Aij = A( i, j );
                    unused( Aij );

                    // loop over tile {i, j} in Ad, verify data came from correct rank
                    int tile_rank = A.tileRank( i, j );
                    for (int jj = j*nb; jj < j*nb + jb; ++jj) {
                        for (int ii = i*nb; ii < i*nb + ib; ++ii) {
                            test_assert( Ad[ ii + jj*lda ] == tile_rank*1000 + ii + jj/1000. );
                        }
                    }
                }
            }
        }
    }

    // -----
    test_message( "gather into B" );
    A.gather( Bd, lda );

    if (g_mpi_rank == 0) {
        // loop over tiles
        if (uplo == blas::Uplo::Lower) {
            for (int j = 0; j < A.nt(); ++j) {
                int jb = A.tileNb( j );
                for (int i = j; i < A.mt(); ++i) {  // lower
                    int ib = A.tileMb( i );

                    // verify tile Aij exists
                    auto Aij = A( i, j );
                    unused( Aij );

                    // loop over tile {i, j} in Bd, verify data came from correct rank
                    int tile_rank = A.tileRank( i, j );
                    for (int jj = j*nb; jj < j*nb + jb; ++jj) {
                        for (int ii = i*nb; ii < i*nb + ib; ++ii) {
                            test_assert( Bd[ ii + jj*lda ] == tile_rank*1000 + ii + jj/1000. );
                        }
                    }
                }
            }
        }
        else {
            for (int j = 0; j < A.nt(); ++j) {
                int jb = A.tileNb( j );
                for (int i = 0; i <= j && i < A.mt(); ++i) {  // upper
                    int ib = A.tileMb( i );

                    // verify tile Aij exists
                    auto Aij = A( i, j );
                    unused( Aij );

                    // loop over tile {i, j} in Bd, verify data came from correct rank
                    int tile_rank = A.tileRank( i, j );
                    for (int jj = j*nb; jj < j*nb + jb; ++jj) {
                        for (int ii = i*nb; ii < i*nb + ib; ++ii) {
                            test_assert( Bd[ ii + jj*lda ] == tile_rank*1000 + ii + jj/1000. );
                        }
                    }
                }
            }
        }
    }

    delete[] Ad;
    Ad = nullptr;

    delete[] Bd;
    Bd = nullptr;
}

// -----------------------------------------------------------------------------
// TODO
// tileInsert( i, j, dev, life )        // implicit
// tileInsert( i, j, dev, data, ld )    // implicit
// tileErase                            // implicit
//
// clear


// -----------------------------------------------------------------------------
void test_hermitian( blas::Uplo uplo, int n, int nb, int p, int q )
{
    Test name( __func__ );

    int m = n;
    int lda = int((m + 31)/32)*32;
    double* Ad = new double[ lda*n ];
    int64_t iseed[] = { 0, 1, 2, 3 };
    lapack::larnv( 1, iseed, lda*n, Ad );

    //slate::HermitianMatrix<double> A( uplo, n, Ad, lda, nb, p, q, g_mpi_comm );
    auto A = slate::HermitianMatrix<double>::fromLAPACK(
                uplo, n, Ad, lda, nb, p, q, g_mpi_comm );

    if (g_mpi_rank == 0) {
        std::cout << "A( uplo=" << char(uplo)
                  << ", n="     << n
                  << ", nb="    << nb
                  << " ): mt="  << A.mt()
                  << ", nt="    << A.nt()
                  << ", op="    << char(A.op())
                  << ", uplo="  << char(A.uplo())
                  << ", size="  << A.size()
                  << "\n";
    }
    test_assert( A.mt() == (n + nb - 1) / nb );
    test_assert( A.nt() == (n + nb - 1) / nb );
    test_assert( A.op() == blas::Op::NoTrans );
    test_assert( A.uplo() == uplo );

    int nt = A.nt();
    int mt = A.mt();

    // ----- verify each tile and tile sizes
    test_message( "tile members" );
    for (int j = 0; j < nt; ++j) {
        int jb = (j == nt-1 ? n - j*nb : nb);

        for (int i = 0; i < mt; ++i) {
            int ib = (i == mt-1 ? m - i*nb : nb);

            if (A.tileIsLocal( i, j ) &&
                ((uplo == blas::Uplo::Lower && i >= j) ||
                 (uplo == blas::Uplo::Upper && i <= j)))
            {
                slate::Tile<double> tile = A( i, j );

                test_assert( tile.mb() == ib );
                test_assert( tile.nb() == jb );
                test_assert( tile.stride() == lda );
                test_assert( tile.data() == &Ad[ i*nb + j*nb*lda ] );

                // operator (i,j)
                for (int jj = 0; jj < jb; ++jj)
                    for (int ii = 0; ii < ib; ++ii)
                        test_assert( Ad[ (i*nb + ii) + (j*nb + jj)*lda ] == tile( ii, jj ) );

                test_assert( tile.op() == blas::Op::NoTrans );

                if (i == j) {
                    test_assert( tile.uplo() == uplo );
                }
                else {
                    test_assert( tile.uplo() == blas::Uplo::General );
                }

                test_assert( tile.origin() == true );
                test_assert( tile.valid()  == true );
                test_assert( tile.device() == g_host_num );
                test_assert( tile.size() == size_t(ib * jb) );
                test_assert( tile.bytes() == sizeof(double) * ib * jb );
            }
            test_assert( A.tileMb( i ) == ib );
            test_assert( A.tileNb( j ) == jb );
        }
    }

    // ----- verify size (# tiles)
    test_message( "size" );
    size_t size = 0;
    for (int j = 0; j < nt; ++j) {
        for (int i = 0; i < mt; ++i) {
            if ((uplo == blas::Uplo::Lower && i >= j) ||
                (uplo == blas::Uplo::Upper && i <= j)) {
                if (i % p == g_mpi_rank % p && j % q == g_mpi_rank / p) {
                    ++size;
                }
            }
        }
    }
    test_assert( A.size() == size );

    // ----- verify tile functions
    test_message( "tileRank, tileDevice, tileIsLocal" );
    for (int j = 0; j < nt; ++j) {
        for (int i = 0; i < mt; ++i) {
            int rank = (i % p) + (j % q)*p;
            test_assert( rank == A.tileRank( i, j ));

            if (g_num_devices > 0) {
                int dev = (j / q) % g_num_devices;
                test_assert( dev == A.tileDevice( i, j ));
            }

            test_assert( (rank == g_mpi_rank) == A.tileIsLocal( i, j ));
        }
    }

    delete[] Ad;
}

// -----------------------------------------------------------------------------
void test_conversion( blas::Uplo uplo, int m, int n, int nb, int p, int q )
{
    Test name( __func__ );

    if (g_mpi_rank == 0) {
        std::cout << "uplo " << char(uplo) << "\n";
    }

    // A is m-by-n general
    int lda = int((m + 31)/32)*32;
    double* Ad = new double[ lda*n ];
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            Ad[ i + j*lda ] = i + j/1000.;
    //slate::Matrix<double> A( m, n, Ad, lda, nb, p, q, g_mpi_comm );
    auto A = slate::Matrix<double>::fromLAPACK(
                m, n, Ad, lda, nb, p, q, g_mpi_comm );

    // B is n-by-m general
    int ldb = int((n + 31)/32)*32;
    double* Bd = new double[ ldb*m ];
    for (int j = 0; j < m; ++j)
        for (int i = 0; i < n; ++i)
            Bd[ i + j*ldb ] = i + j/1000.;
    //slate::Matrix<double> B( n, m, Bd, ldb, nb, p, q, g_mpi_comm );
    auto B = slate::Matrix<double>::fromLAPACK(
                n, m, Bd, ldb, nb, p, q, g_mpi_comm );

    // S is n-by-n symmetric
    int lds = int((n + 31)/32)*32;
    double* Sd = new double[ lds*n ];
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            Sd[ i + j*lds ] = i + j/1000.;
    //slate::SymmetricMatrix<double> S( uplo, n, Sd, lds, nb, p, q, g_mpi_comm );
    auto S = slate::HermitianMatrix<double>::fromLAPACK(
                uplo, n, Ad, lda, nb, p, q, g_mpi_comm );

    // -----
    // general to triangular
    // can't go backwards from {tz, sy, he, tr} => ge,
    // since half the tiles may not exist!
    test_message( "ge => tz, ge => sy, ge => he, ge => tr" );
    slate::TrapezoidMatrix<double> ZA( uplo, A );
    slate::TrapezoidMatrix<double> ZB( uplo, B );

    // sub-matrix constructor
    slate::TrapezoidMatrix<double> ZB2( ZB, 1, ZB.mt()-1, 1, ZB.nt()-1 );

    // sub-matrix constructor throws exception if i1 != j1
    test_assert_throw( slate::TrapezoidMatrix<double>( ZB, 1, ZB.mt()-1, 2, ZB.nt()-1 ),
                       std::exception );

    test_assert( ZA.mt() == A.mt() );
    test_assert( ZA.nt() == A.nt() );
    test_assert( ZB.mt() == B.mt() );
    test_assert( ZB.nt() == B.nt() );

    int min_nt = std::min( A.mt(), A.nt() );

    slate::TriangularMatrix<double> TA( uplo, A );
    slate::TriangularMatrix<double> TB( uplo, B );
    test_assert( TA.mt() == min_nt );
    test_assert( TA.nt() == min_nt );
    test_assert( TB.mt() == min_nt );
    test_assert( TB.nt() == min_nt );

    slate::SymmetricMatrix<double>  SA( uplo, A );
    slate::SymmetricMatrix<double>  SB( uplo, B );
    test_assert( SA.mt() == min_nt );
    test_assert( SA.nt() == min_nt );
    test_assert( SB.mt() == min_nt );
    test_assert( SB.nt() == min_nt );

    slate::HermitianMatrix<double>  HA( uplo, A );
    slate::HermitianMatrix<double>  HB( uplo, B );
    test_assert( HA.mt() == min_nt );
    test_assert( HA.nt() == min_nt );
    test_assert( HB.mt() == min_nt );
    test_assert( HB.nt() == min_nt );

    // -----
    test_message( "sy => he, he => sy" );
    slate::HermitianMatrix<double>  H( S );

    test_message( "sy => he, he => sy (2)" );
    slate::SymmetricMatrix<double> S2( H );

    test_message( "sy => he, he => sy (3)" );
    test_assert( S.mt()   ==  H.mt()   );
    test_assert( S.nt()   ==  H.nt()   );
    test_assert( S.uplo() ==  H.uplo() );
    test_assert( S.mt()   == S2.mt()   );
    test_assert( S.nt()   == S2.nt()   );
    test_assert( S.uplo() == S2.uplo() );

    // -----
    test_message( "sy => tr, tr => sy" );
    slate::TriangularMatrix<double> T1( S  );
    slate::SymmetricMatrix<double>  S3( T1 );
    test_assert( S.mt()   == T1.mt()   );
    test_assert( S.nt()   == T1.nt()   );
    test_assert( S.uplo() == T1.uplo() );
    test_assert( S.mt()   == S3.mt()   );
    test_assert( S.nt()   == S3.nt()   );
    test_assert( S.uplo() == S3.uplo() );

    // alt. syntax, sy => tr
    auto T2 = slate::TriangularMatrix<double>( S );
    test_assert( S.mt()   == T2.mt()   );
    test_assert( S.nt()   == T2.nt()   );
    test_assert( S.uplo() == T2.uplo() );

    // -----
    test_message( "he => tr, tr => he" );
    slate::TriangularMatrix<double> T3( H  );
    slate::HermitianMatrix<double>  H2( T3 );
    test_assert( S.mt()   == T3.mt()   );
    test_assert( S.nt()   == T3.nt()   );
    test_assert( S.uplo() == T3.uplo() );
    test_assert( S.mt()   == H2.mt()   );
    test_assert( S.nt()   == H2.nt()   );
    test_assert( S.uplo() == H2.uplo() );

    // if upper, transpose so we can access as-if lower
    if (uplo == blas::Uplo::Upper) {
        S  = transpose( S  );
        S2 = transpose( S2 );
        H  = transpose( H  );
        H2 = transpose( H2 );
        T1 = transpose( T1 );
        T2 = transpose( T2 );
        T3 = transpose( T3 );
    }

    for (int j = 0; j < S.nt(); ++j) {
        for (int i = j; i < S.mt(); ++i) {  // lower
            if (S.tileIsLocal( i, j )) {
                auto S_ij  =  S( i, j );
                auto S2_ij = S2( i, j );
                auto H_ij  =  H( i, j );
                auto H2_ij = H2( i, j );
                auto T1_ij = T1( i, j );
                auto T2_ij = T2( i, j );
                auto T3_ij = T3( i, j );
                test_assert( S_ij.data() == S2_ij.data() );
                test_assert( S_ij.data() ==  H_ij.data() );
                test_assert( S_ij.data() == H2_ij.data() );
                test_assert( S_ij.data() == T1_ij.data() );
                test_assert( S_ij.data() == T2_ij.data() );
                test_assert( S_ij.data() == T3_ij.data() );
            }
        }
    }
}

// -----------------------------------------------------------------------------
// quick test of fromLAPACK and fromScaLAPACK for various types.
// see test_{general,hermitian}_scalapack() for more specific tests.
void test_constructors( int m, int n, int nb, int p, int q )
{
    Test name( __func__ );

    blas::Uplo uplo = blas::Uplo::Lower;
    int lda = m;
    int ldb = n;
    double *Adata = new double[ lda*n ];  // m-by-n
    double *Bdata = new double[ ldb*n ];  // n-by-n

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            Adata[ i + j*lda ] = i + j/1000.;

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            Bdata[ i + j*ldb ] = i + j/1000.;

    bool verbose = false;
    //verbose = true;

    // ---------- m-by-n rectangular
    //printf( "Matrix Lapack\n" );
    //slate::Matrix<double> A1( m, n, Adata, lda, nb, p, q, g_mpi_comm );
    //printf( "Matrix ScaLapack\n" );
    //slate::Matrix<double> A2( m, n, Adata, lda, nb, nb, p, q, g_mpi_comm );
    auto A_lapack = slate::Matrix<double>::fromLAPACK( m, n, Adata, lda, nb, p, q, g_mpi_comm );
    auto A_scalapack = slate::Matrix<double>::fromScaLAPACK( m, n, Adata, lda, nb, p, q, g_mpi_comm );
    if (verbose) {
        fflush(0);
        test_message( "Matrix" );
        for (int rank = 0; rank < g_mpi_size; ++rank) {
            if (rank == g_mpi_rank) {
                printf( "rank %d\n", rank );
                print( "A_lapack", A_lapack );
                print( "A_scalapack", A_scalapack );
                printf( "\n" );
                fflush(0);
            }
            MPI_Barrier( g_mpi_comm );
        }
    }

    //printf( "TrapezoidMatrix Lapack\n" );
    //slate::TrapezoidMatrix<double> Z1( uplo, m, n, Adata, lda, nb, p, q, g_mpi_comm );
    //printf( "TrapezoidMatrix ScaLapack\n" );
    //slate::TrapezoidMatrix<double> Z2( uplo, m, n, Adata, lda, nb, nb, p, q, g_mpi_comm );
    auto Z_lapack = slate::TrapezoidMatrix<double>::fromLAPACK( uplo, m, n, Adata, lda, nb, p, q, g_mpi_comm );
    auto Z_scalapack = slate::TrapezoidMatrix<double>::fromScaLAPACK( uplo, m, n, Adata, lda, nb, p, q, g_mpi_comm );
    if (verbose) {
        fflush(0);
        test_message( "TrapezoidMatrix" );
        for (int rank = 0; rank < g_mpi_size; ++rank) {
            if (rank == g_mpi_rank) {
                printf( "rank %d\n", rank );
                print( "Z_lapack", Z_lapack );
                print( "Z_scalapack", Z_scalapack );
                printf( "\n" );
                fflush(0);
            }
            MPI_Barrier( g_mpi_comm );
        }
    }

    // ---------- n-by-n square
    //printf( "TriangularMatrix Lapack\n" );
    //slate::TriangularMatrix<double> T1( uplo, n, Bdata, ldb, nb, p, q, g_mpi_comm );
    //printf( "TriangularMatrix ScaLapack\n" );
    //slate::TriangularMatrix<double> T2( uplo, n, Bdata, ldb, nb, nb, p, q, g_mpi_comm );
    auto T_lapack = slate::TriangularMatrix<double>::fromLAPACK( uplo, n, Bdata, ldb, nb, p, q, g_mpi_comm );
    auto T_scalapack = slate::TriangularMatrix<double>::fromScaLAPACK( uplo, n, Bdata, ldb, nb, p, q, g_mpi_comm );
    if (verbose) {
        fflush(0);
        test_message( "TriangularMatrix" );
        for (int rank = 0; rank < g_mpi_size; ++rank) {
            if (rank == g_mpi_rank) {
                printf( "rank %d\n", rank );
                print( "T_lapack", T_lapack );
                print( "T_scalapack", T_scalapack );
                printf( "\n" );
                fflush(0);
            }
            MPI_Barrier( g_mpi_comm );
        }
    }

    //printf( "SymmetricMatrix Lapack\n" );
    //slate::SymmetricMatrix<double> S1( uplo, n, Bdata, ldb, nb, p, q, g_mpi_comm );
    //printf( "SymmetricMatrix ScaLapack\n" );
    //slate::SymmetricMatrix<double> S2( uplo, n, Bdata, ldb, nb, nb, p, q, g_mpi_comm );
    auto S_lapack = slate::SymmetricMatrix<double>::fromLAPACK( uplo, n, Bdata, ldb, nb, p, q, g_mpi_comm );
    auto S_scalapack = slate::SymmetricMatrix<double>::fromScaLAPACK( uplo, n, Bdata, ldb, nb, p, q, g_mpi_comm );
    if (verbose) {
        fflush(0);
        test_message( "SymmetricMatrix" );
        for (int rank = 0; rank < g_mpi_size; ++rank) {
            if (rank == g_mpi_rank) {
                printf( "rank %d\n", rank );
                print( "S_lapack", S_lapack );
                print( "S_scalapack", S_scalapack );
                printf( "\n" );
                fflush(0);
            }
            MPI_Barrier( g_mpi_comm );
        }
    }

    //printf( "HermitianMatrix Lapack\n" );
    //slate::HermitianMatrix<double> H1( uplo, n, Bdata, ldb, nb, p, q, g_mpi_comm );
    //printf( "HermitianMatrix ScaLapack\n" );
    //slate::HermitianMatrix<double> H2( uplo, n, Bdata, ldb, nb, nb, p, q, g_mpi_comm );
    auto H_lapack = slate::HermitianMatrix<double>::fromLAPACK( uplo, n, Bdata, ldb, nb, p, q, g_mpi_comm );
    auto H_scalapack = slate::HermitianMatrix<double>::fromScaLAPACK( uplo, n, Bdata, ldb, nb, p, q, g_mpi_comm );
    if (verbose) {
        fflush(0);
        test_message( "HermitianMatrix" );
        for (int rank = 0; rank < g_mpi_size; ++rank) {
            if (rank == g_mpi_rank) {
                printf( "rank %d\n", rank );
                print( "H_lapack", H_lapack );
                print( "H_scalapack", H_scalapack );
                printf( "\n" );
                fflush(0);
            }
            MPI_Barrier( g_mpi_comm );
        }
    }
}

// -----------------------------------------------------------------------------
void test_general_scalapack( int m, int n, int nb, int p, int q )
{
    Test name( __func__ );

    // maximum number of tiles per rank
    int mt = slate::ceildiv( m, nb );
    int nt = slate::ceildiv( n, nb );

    // number of local tiles
    int mt_loc = mt / p;
    if (mt % p > g_mpi_rank % p)
        mt_loc += 1;

    int nt_loc = nt / q;
    if (nt % q > g_mpi_rank / p)
        nt_loc += 1;

    // local m, n, lda (rounded up to whole tiles)
    int m_loc = mt_loc*nb;
    int n_loc = nt_loc*nb;
    int lda_loc = m_loc;

    printf( "rank %d, m %d (%d), n %d (%d), mt %d (%d), nt %d (%d), nb %d, p %d, q %d (local quantities)\n",
            g_mpi_rank, m, m_loc, n, n_loc, mt, mt_loc, nt, nt_loc, nb, p, q );

    double *Ad = new double[ lda_loc * n_loc ];
    int64_t iseed[] = { 0, 1, 2, 3 };
    lapack::larnv( 1, iseed, lda_loc*n_loc, Ad );
    auto A = slate::Matrix<double>::fromScaLAPACK(
                m, n, Ad, lda_loc, nb, p, q, g_mpi_comm );

    test_assert( A.mt() == mt );
    test_assert( A.nt() == nt );

    int j_loc = 0;
    for (int j = 0; j < nt; ++j) {
        int i_loc = 0;
        for (int i = 0; i < mt; ++i) {
            if (A.tileIsLocal( i, j )) {
                //printf( "rank %d, i=%d/%d, j=%d/%d, %p %p\n",
                //        g_mpi_rank, i, i_loc, j, j_loc,
                //        (void*) A(i, j).data(),
                //        (void*) &Ad[ i_loc*nb + j_loc*nb*lda_loc ] );
                test_assert( A(i, j).data() == &Ad[ i_loc*nb + j_loc*nb*lda_loc ] );
                ++i_loc;
            }
        }
        if (i_loc > 0) {
            test_assert( i_loc == mt_loc );
            ++j_loc;
        }
    }
    test_assert( j_loc == nt_loc );
}

// -----------------------------------------------------------------------------
void test_hermitian_scalapack( blas::Uplo uplo, int n, int nb, int p, int q )
{
    Test name( __func__ );
    if (g_mpi_rank == 0) {
        std::cout << "uplo=" << char(uplo) << "\n";
    }

    // maximum number of tiles per rank
    int nt = slate::ceildiv( n, nb );
    int mt = nt;

    // number of local tiles
    int mt_loc = mt / p;
    if (mt % p > g_mpi_rank % p)
        mt_loc += 1;

    int nt_loc = nt / q;
    if (nt % q > g_mpi_rank / p)
        nt_loc += 1;

    // local m, n, lda (rounded up to whole tiles)
    int m_loc = mt_loc*nb;
    int n_loc = nt_loc*nb;
    int lda_loc = m_loc;

    printf( "rank %d, n rows %d (%d), n %d (%d), mt %d (%d), nt %d (%d), nb %d, p %d, q %d (local quantities)\n",
            g_mpi_rank, n, m_loc, n, n_loc, mt, mt_loc, nt, nt_loc, nb, p, q );

    double *Ad = new double[ lda_loc * n_loc ];
    int64_t iseed[] = { 0, 1, 2, 3 };
    lapack::larnv( 1, iseed, lda_loc*n_loc, Ad );
    auto A = slate::HermitianMatrix<double>::fromScaLAPACK(
                uplo, n, Ad, lda_loc, nb, p, q, g_mpi_comm );

    test_assert( A.mt() == mt );
    test_assert( A.nt() == nt );

    int j_loc = 0;
    for (int j = 0; j < nt; ++j) {
        int i_loc = 0;
        for (int i = 0; i < mt; ++i) {
            if (A.tileIsLocal( i, j )) {
                if ((uplo == blas::Uplo::Lower && i >= j) ||
                    (uplo == blas::Uplo::Upper && i <= j))
                {
                    test_assert( A(i, j).data() == &Ad[ i_loc*nb + j_loc*nb*lda_loc ] );
                }
                ++i_loc;
            }
        }
        if (i_loc > 0) {
            test_assert( i_loc == mt_loc );
            ++j_loc;
        }
    }
    test_assert( j_loc == nt_loc );
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    g_mpi_comm = MPI_COMM_WORLD;
    MPI_Comm_rank( g_mpi_comm, &g_mpi_rank );
    MPI_Comm_size( g_mpi_comm, &g_mpi_size );

    int m  = 200;
    int n  = 500;
    int nb = 64;
    int p  = std::min( 2, g_mpi_size );
    int q  = g_mpi_size / p;
    std::string test = "all";

    if (argc > 1) { m  = atoi( argv[1] ); }
    if (argc > 2) { n  = atoi( argv[2] ); }
    if (argc > 3) { nb = atoi( argv[3] ); }
    if (argc > 4) { p  = atoi( argv[4] ); }
    if (argc > 5) { q  = atoi( argv[5] ); }
    if (argc > 6) { test = argv[6]; }
    bool all = test == "all";

    test_assert( p * q == g_mpi_size );

    std::cout << "mpi rank=" << g_mpi_rank
              << ", mpi size=" << g_mpi_size
              << ", num devices=" << g_num_devices
              << ", m=" << m
              << ", n=" << n
              << ", nb=" << nb
              << ", p=" << p
              << ", q=" << q
              << "\n" << std::flush;
    MPI_Barrier( g_mpi_comm );

    if (g_mpi_rank == 0) {
        std::cout << "\n" << std::flush;
    }
    MPI_Barrier( g_mpi_comm );

    if (all || test == "general") {
        test_empty();
        test_general( m, n, nb, p, q );
        test_general_sub( m, n, nb, p, q );
        test_general_send( m, n, nb, p, q );
        test_general_send2( m, n, nb, p, q );
        test_cuda_streams( m, n, nb, p, q );
        test_batch_arrays( m, n, nb, p, q );
        test_copyToDevice( m, n, nb, p, q );
        test_copyToHost( m, n, nb, p, q );
        test_workspace( m, n, nb, p, q );
    }

    if (all || test == "hermitian") {
        test_hermitian( blas::Uplo::Lower, m, nb, p, q );
        test_hermitian( blas::Uplo::Upper, m, nb, p, q );
    }

    if (all || test == "conversion") {
        test_conversion( blas::Uplo::Lower, m, n, nb, p, q );
        test_conversion( blas::Uplo::Upper, m, n, nb, p, q );
    }

    if (all || test == "gather") {
        test_gather( m, n, nb, p, q );
        test_hermitian_gather( blas::Uplo::Lower, m, nb, p, q );
        test_hermitian_gather( blas::Uplo::Upper, m, nb, p, q );
    }

    if (all || test == "scalapack") {
        test_constructors( m, n, nb, p, q );
        test_general_scalapack( m, n, nb, p, q );
        test_hermitian_scalapack( blas::Uplo::Lower, m, nb, p, q );
        test_hermitian_scalapack( blas::Uplo::Upper, m, nb, p, q );
    }

    MPI_Finalize();
    return 0;
}
