#include "test.hh"
#include "slate_Matrix.hh"

// -----------------------------------------------------------------------------
void test_empty()
{
    Test name( __func__ );

    // ----- General
    slate::Matrix<double> A;

    if (mpi_rank == 0) {
        std::cout << "General    A( empty ): mt=" << A.mt()
                  << ", nt=" << A.nt()
                  << ", op=" << char(A.op())
                  << "\n";
    }
    test_assert( A.mt() == 0 );
    test_assert( A.nt() == 0 );
    test_assert( A.op() == blas::Op::NoTrans );
    MPI_Barrier( mpi_comm );

    // ----- Trapezoid
    slate::TrapezoidMatrix<double> Z;

    if (mpi_rank == 0) {
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
    MPI_Barrier( mpi_comm );

    // ----- Triangular
    slate::TriangularMatrix<double> T;

    if (mpi_rank == 0) {
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
    MPI_Barrier( mpi_comm );

    // ----- Symmetric
    slate::SymmetricMatrix<double> S;

    if (mpi_rank == 0) {
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
    MPI_Barrier( mpi_comm );

    // ----- Hermitian
    slate::HermitianMatrix<double> H;

    if (mpi_rank == 0) {
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
    MPI_Barrier( mpi_comm );
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
    slate::Matrix<double> A( m, n, Ad, lda, nb, p, q, mpi_comm );

    // B is n-by-n
    int ldb = int((n + 31)/32)*32;
    double* Bd = new double[ ldb*n ];
    slate::Matrix<double> B( n, n, Bd, ldb, nb, p, q, mpi_comm );

    if (mpi_rank == 0) {
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
    if (mpi_rank == 0) {
        test_assert( A.tileIsLocal( 0, 0 ) );
        test_assert( A( 0, 0 ).data() == Ad );
        test_assert( A.at( 0, 0 ).data() == Ad );
    }

    test_assert( B.mt() == (n + nb - 1) / nb );
    test_assert( B.nt() == (n + nb - 1) / nb );
    test_assert( B.op() == blas::Op::NoTrans );
    if (mpi_rank == 0) {
        test_assert( B.tileIsLocal( 0, 0 ) );
        test_assert( B( 0, 0 ).data() == Bd );
        test_assert( B.at( 0, 0 ).data() == Bd );
    }
    MPI_Barrier( mpi_comm );

    // ----- verify swap
    if (mpi_rank == 0) {
        std::cout << "swap( A, B )\n";
    }
    // transpose so we can tell if op was swapped
    B = transpose( B );
    test_assert( B.op() == blas::Op::Trans );

    swap( A, B );

    // verify that all data is swapped
    test_assert( B.mt() == (m + nb - 1) / nb );
    test_assert( B.nt() == (n + nb - 1) / nb );
    test_assert( B.op() == blas::Op::NoTrans );
    if (mpi_rank == 0) {
        test_assert( B.tileIsLocal( 0, 0 ) );
        test_assert( B( 0, 0 ).data() == Ad );
    }

    test_assert( A.mt() == (n + nb - 1) / nb );
    test_assert( A.nt() == (n + nb - 1) / nb );
    test_assert( A.op() == blas::Op::Trans );
    if (mpi_rank == 0) {
        test_assert( A.tileIsLocal( 0, 0 ) );
        test_assert( A( 0, 0 ).data() == Bd );
    }

    // swap again to restore
    swap( A, B );

    // verify that all data is swapped back
    test_assert( A.mt() == (m + nb - 1) / nb );
    test_assert( A.nt() == (n + nb - 1) / nb );
    test_assert( A.op() == blas::Op::NoTrans );
    if (mpi_rank == 0) {
        test_assert( A.tileIsLocal( 0, 0 ) );
        test_assert( A( 0, 0 ).data() == Ad );
    }

    test_assert( B.mt() == (n + nb - 1) / nb );
    test_assert( B.nt() == (n + nb - 1) / nb );
    test_assert( B.op() == blas::Op::Trans );
    if (mpi_rank == 0) {
        test_assert( B.tileIsLocal( 0, 0 ) );
        test_assert( B( 0, 0 ).data() == Bd );
    }

    // transpose to restore
    B = transpose( B );
    test_assert( B.op() == blas::Op::NoTrans );
    MPI_Barrier( mpi_comm );

    // ----- get info
    int mt = A.mt();
    int nt = A.nt();

    // ----- verify each tile and tile sizes
    if (mpi_rank == 0) {
        std::cout << "tile members\n";
    }
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
                test_assert( tile.device() == host_num );
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
    MPI_Barrier( mpi_comm );

    // ----- verify size (# tiles)
    if (mpi_rank == 0) {
        std::cout << "size\n";
    }
    size_t size = (mt/p + (mt%p > mpi_rank%p)) * (nt/q + (nt%q > mpi_rank/p));
    test_assert( A.size() == size );
    MPI_Barrier( mpi_comm );

    // ----- verify tile functions
    if (mpi_rank == 0) {
        std::cout << "tileRank, tileDevice, tileIsLocal\n";
    }
    for (int j = 0; j < nt; ++j) {
        for (int i = 0; i < mt; ++i) {
            int rank = (i % p) + (j % q)*p;
            test_assert( rank == A.tileRank( i, j ));

            int dev = (j / q) % num_devices;
            test_assert( dev == A.tileDevice( i, j ));

            test_assert( (rank == mpi_rank) == A.tileIsLocal( i, j ));
        }
    }
    MPI_Barrier( mpi_comm );

    // ----- verify tile transpose
    if (mpi_rank == 0) {
        std::cout << "[conj_]transpose( tile )\n";
    }
    for (int j = 0; j < nt; ++j) {
        for (int i = 0; i < mt; ++i) {
            if (A.tileIsLocal( i, j )) {
                slate::Tile<double> B = A( i, j );

                auto BT = transpose( B );
                test_assert( BT.op() == blas::Op::Trans );
                test_assert( BT.mb() == B.nb() );
                test_assert( BT.nb() == B.mb() );

                auto BTT = transpose( BT );
                test_assert( BTT.op() == blas::Op::NoTrans );
                test_assert( BTT.mb() == B.mb() );
                test_assert( BTT.nb() == B.nb() );

                auto BC = conj_transpose( B );
                test_assert( BC.op() == blas::Op::ConjTrans );
                test_assert( BC.mb() == B.nb() );
                test_assert( BC.nb() == B.mb() );

                auto BCC = conj_transpose( BC );
                test_assert( BCC.op() == blas::Op::NoTrans );
                test_assert( BCC.mb() == B.mb() );
                test_assert( BCC.nb() == B.nb() );

                // conj_trans( trans( B )) is not supported
                // trans( conj_trans( B )) is not supported
                test_assert_throw( conj_transpose( BT ), std::exception );
                test_assert_throw( transpose( BC ),      std::exception );
            }
        }
    }
    MPI_Barrier( mpi_comm );

    // ----- verify tile transpose
    if (mpi_rank == 0) {
        std::cout << "transpose( A )\n";
    }
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
    MPI_Barrier( mpi_comm );

    // ----- verify tile conj_transpose
    if (mpi_rank == 0) {
        std::cout << "conj_transpose( A )\n";
    }
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
    MPI_Barrier( mpi_comm );

    delete[] Ad;
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
    slate::Matrix<double> A( m, n, Ad, lda, nb, p, q, mpi_comm );

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
        if (mpi_rank == 0) {
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

        //if (mpi_rank == 0) {
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
    slate::Matrix<double> A( m, n, Ad, lda, nb, p, q, mpi_comm );

    // -------------------- broadcast A(0,0) to all of A
    if (mpi_rank == 0) {
        std::cout << "broadcast A(0,0) to all of A\n";
    }
    A.tileBcast( 0, 0, A );

    auto Aij = A( 0, 0 );  // should be on all ranks now
    unused( Aij );
    MPI_Barrier( mpi_comm );

    // ----- verify life
    if (mpi_rank == 0) {
        std::cout << "life\n";
    }
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
    MPI_Barrier( mpi_comm );

    // ----- verify tileTick
    if (mpi_rank == 0) {
        std::cout << "tileTick\n";
    }
    // after all the tickTicks, life of A(0,0) should have been decrement to 0
    // and deleted the A(0,0) tile
    test_assert( A.tileLife( 0, 0 ) == 0 );
    if (A.tileIsLocal( 0, 0 ))
        Aij = A( 0, 0 );  // local should still work
    else
        test_assert_throw( A( 0, 0 ), std::out_of_range );
    MPI_Barrier( mpi_comm );

    // -------------------- broadcast A(1,1) across row A(1,2:nt)
    if (mpi_rank == 0) {
        std::cout << "broadcast A(1,1) to A(1,2:nt)\n";
    }
    auto Arow = A.sub( 1, 1, 2, A.nt()-1 );
    A.tileBcast( 1, 1, Arow );

    // ----- verify life
    if (mpi_rank == 0) {
        std::cout << "life\n";
    }
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
    if (mpi_rank == 0) {
        std::cout << "tileTick\n";
    }
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
    MPI_Barrier( mpi_comm );

    // -------------------- broadcast A(1,1) across col A(2:mt,1)
    if (mpi_rank == 0) {
        std::cout << "broadcast A(1,1) to A(2:mt,1)\n";
    }
    auto Acol = A.sub( 2, A.mt()-1, 1, 1 );
    A.tileBcast( 1, 1, Acol );

    // ----- verify life
    if (mpi_rank == 0) {
        std::cout << "life\n";
    }
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
    if (mpi_rank == 0) {
        std::cout << "tileTick\n";
    }
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
    MPI_Barrier( mpi_comm );
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
    slate::Matrix<double> A( m, n, Ad, lda, nb, p, q, mpi_comm );

    // -------------------- broadcast A(1,1) across row A(1,2:nt) and col A(2:mt,1)
    if (mpi_rank == 0) {
        std::cout << "broadcast A(1,1) to A(1,2:nt) and A(2:mt,1)\n";
    }
    auto Arow = A.sub( 1, 1, 2, A.nt()-1 );
    auto Acol = A.sub( 2, A.mt()-1, 1, 1 );
    A.tileBcast( 1, 1, Arow, Acol );
    slate::Tile<double> Aij;

    // ----- verify life
    if (mpi_rank == 0) {
        std::cout << "life\n";
    }
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
    if (mpi_rank == 0) {
        std::cout << "tileTick\n";
    }
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
    MPI_Barrier( mpi_comm );
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
    slate::Matrix<double> A( m, n, Ad, lda, nb, p, q, mpi_comm );

    for (int device = 0; device < num_devices; ++device) {
        double* devAd;
        cudaError_t err;
        size_t size = sizeof(double) * lda*n;
        err = cudaMalloc( (void**) &devAd, size );
        test_assert( err == cudaSuccess );

        // ----- compute stream
        if (mpi_rank == 0) {
            std::cout << "compute stream\n";
        }
        test_assert( A.compute_stream( device ) != nullptr );

        err = cudaMemcpyAsync( devAd, Ad, size, cudaMemcpyHostToDevice, A.compute_stream( device ));
        test_assert( err == cudaSuccess );

        err = cudaStreamSynchronize( A.compute_stream( device ));
        test_assert( err == cudaSuccess );

        // ----- comm stream
        if (mpi_rank == 0) {
            std::cout << "comm stream\n";
        }
        test_assert( A.comm_stream( device ) != nullptr );

        err = cudaMemcpyAsync( devAd, Ad, size, cudaMemcpyHostToDevice, A.comm_stream( device ));
        test_assert( err == cudaSuccess );

        err = cudaStreamSynchronize( A.comm_stream( device ));
        test_assert( err == cudaSuccess );

        // comm and compute streams different
        test_assert( A.comm_stream( device ) != A.compute_stream( device ) );

        // ----- cublas handle
        if (mpi_rank == 0) {
            std::cout << "cublas handle\n";
        }
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
    slate::Matrix<double> A( m, n, Ad, lda, nb, p, q, mpi_comm );

    if (mpi_rank == 0) {
        std::cout << "allocateBatchArrays\n";
    }
    A.allocateBatchArrays();

    // test filling in each host array and copying to device array
    for (int device = 0; device < num_devices; ++device) {
        int size = A.getMaxDeviceTiles( device );
        double **array_host, **array_device;
        cudaError_t err;

        // -----
        if (mpi_rank == 0) {
            std::cout << "a_array\n";
        }
        array_host   = A.a_array_host  ( device );
        array_device = A.a_array_device( device );
        test_assert( array_host   != nullptr );
        test_assert( array_device != nullptr );
        for (int i = 0; i < size; ++i)
            array_host[i] = Ad;
        err = cudaMemcpy( array_device, array_host, size, cudaMemcpyHostToDevice );
        test_assert( err == cudaSuccess );

        // -----
        if (mpi_rank == 0) {
            std::cout << "b_array\n";
        }
        array_host   = A.b_array_host  ( device );
        array_device = A.b_array_device( device );
        test_assert( array_host   != nullptr );
        test_assert( array_device != nullptr );
        for (int i = 0; i < size; ++i)
            array_host[i] = Ad;
        err = cudaMemcpy( array_device, array_host, size, cudaMemcpyHostToDevice );
        test_assert( err == cudaSuccess );

        // -----
        if (mpi_rank == 0) {
            std::cout << "c_array\n";
        }
        array_host   = A.c_array_host  ( device );
        array_device = A.c_array_device( device );
        test_assert( array_host   != nullptr );
        test_assert( array_device != nullptr );
        for (int i = 0; i < size; ++i)
            array_host[i] = Ad;
        err = cudaMemcpy( array_device, array_host, size, cudaMemcpyHostToDevice );
        test_assert( err == cudaSuccess );
    }

    if (mpi_rank == 0) {
        std::cout << "clearBatchArrays\n";
    }
    A.clearBatchArrays();

    // now all the arrays should give errors
    for (int device = 0; device < num_devices; ++device) {
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
    slate::Matrix<double> A( m, n, Ad, lda, nb, p, q, mpi_comm );

    for (int device = 0; device < num_devices; ++device) {
        if (A.tileIsLocal( 0, 0 )) {
            // initially, tile exists only on host
            auto Aij = A( 0, 0 );
            test_assert_throw( Aij = A( 0, 0, device ), std::exception );

            // after copy, both host and device data are valid
            if (mpi_rank == 0) {
                std::cout << "copyToDevice\n";
            }
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
    slate::Matrix<double> A( m, n, Ad, lda, nb, p, q, mpi_comm );

    // todo: this doesn't test if the tile exists only on device,
    // because that currently doesn't happen: data starts on host and is copied
    // to the device, but the host tile always exists.
    for (int device = 0; device < num_devices; ++device) {
        if (A.tileIsLocal( 0, 0 )) {
            // initially, tile exists only on host
            auto Aij = A( 0, 0 );
            test_assert_throw( Aij = A( 0, 0, device ), std::exception );

            // after move, both tiles exist, but only device is valid
            if (mpi_rank == 0) {
                std::cout << "tileMoveToDevice\n";
            }
            A.tileMoveToDevice( 0, 0, device );  // invalidates host data
            test_assert( ! A( 0, 0 ).valid() );
            test_assert( A( 0, 0, device ).valid() );

            // after copy, both host and device are valid
            if (mpi_rank == 0) {
                std::cout << "tileCopyToHost\n";
            }
            A.tileCopyToHost( 0, 0, device );
            test_assert( A( 0, 0 ).valid() );
            test_assert( A( 0, 0, device ).valid() );

            // after move, both tiles exist, but only host is valid
            if (mpi_rank == 0) {
                std::cout << "tileMoveToHost\n";
            }
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
    slate::Matrix<double> A( m, n, Ad, lda, nb, p, q, mpi_comm );

    A.reserveHostWorkspace();
    A.reserveDeviceWorkspace();
    A.clearWorkspace();
}

// -----------------------------------------------------------------------------
// TODO
// tileInsert( i, j, dev, life )        // implicit
// tileInsert( i, j, dev, data, ld )    // implicit
// tileErase                            // implicit
//
// gather
// clear
// clearWorkspace
// reserveWorkspace


// -----------------------------------------------------------------------------
void test_hermitian( blas::Uplo uplo, int n, int nb, int p, int q )
{
    Test name( __func__ );

    int m = n;
    int lda = int((m + 31)/32)*32;
    double* Ad = new double[ lda*n ];

    slate::HermitianMatrix<double> A( uplo, n, Ad, lda, nb, p, q, mpi_comm );

    if (mpi_rank == 0) {
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
    MPI_Barrier( mpi_comm );

    int nt = A.nt();
    int mt = A.mt();

    // ----- verify each tile and tile sizes
    if (mpi_rank == 0) {
        std::cout << "tile members\n";
    }
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
                test_assert( tile.device() == host_num );
                test_assert( tile.size() == size_t(ib * jb) );
                test_assert( tile.bytes() == sizeof(double) * ib * jb );
            }
            test_assert( A.tileMb( i ) == ib );
            test_assert( A.tileNb( j ) == jb );
        }
    }
    MPI_Barrier( mpi_comm );

    // ----- verify size (# tiles)
    if (mpi_rank == 0) {
        std::cout << "size\n";
    }
    size_t size = 0;
    for (int j = 0; j < nt; ++j) {
        for (int i = 0; i < mt; ++i) {
            if ((uplo == blas::Uplo::Lower && i >= j) ||
                (uplo == blas::Uplo::Upper && i <= j)) {
                if (i % p == mpi_rank % p && j % q == mpi_rank / p) {
                    ++size;
                }
            }
        }
    }
    test_assert( A.size() == size );
    MPI_Barrier( mpi_comm );

    // ----- verify tile functions
    if (mpi_rank == 0) {
        std::cout << "tileRank, tileDevice, tileIsLocal\n";
    }
    for (int j = 0; j < nt; ++j) {
        for (int i = 0; i < mt; ++i) {
            int rank = (i % p) + (j % q)*p;
            test_assert( rank == A.tileRank( i, j ));

            int dev = (j / q) % num_devices;
            test_assert( dev == A.tileDevice( i, j ));

            test_assert( (rank == mpi_rank) == A.tileIsLocal( i, j ));
        }
    }
    MPI_Barrier( mpi_comm );

    delete[] Ad;
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    mpi_comm = MPI_COMM_WORLD;
    MPI_Comm_rank( mpi_comm, &mpi_rank );
    MPI_Comm_size( mpi_comm, &mpi_size );
    slate::g_mpi_rank = mpi_rank;
    slate::g_verbose = false; //(mpi_rank == 0);

    int m  = 200;
    int n  = 500;
    int nb = 64;
    int p  = 2;
    int q  = mpi_size / p;

    if (argc > 1) { m  = atoi( argv[1] ); }
    if (argc > 2) { n  = atoi( argv[2] ); }
    if (argc > 3) { nb = atoi( argv[3] ); }
    if (argc > 4) { p  = atoi( argv[4] ); }
    if (argc > 5) { q  = atoi( argv[5] ); }

    test_assert( p * q == mpi_size );

    std::cout << "mpi rank=" << mpi_rank
              << ", mpi size=" << mpi_size
              << ", num_devices=" << num_devices
              << ", m=" << m
              << ", n=" << n
              << ", nb=" << nb
              << ", p=" << p
              << ", q=" << q
              << "\n" << std::flush;
    MPI_Barrier( mpi_comm );

    if (mpi_rank == 0) {
        std::cout << "\n" << std::flush;
    }
    MPI_Barrier( mpi_comm );

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

    test_hermitian( blas::Uplo::Lower, m, nb, p, q );
    test_hermitian( blas::Uplo::Upper, m, nb, p, q );

    MPI_Finalize();
    return 0;
}
