! slate00_matrix.f90
! create 2000 x 1000 matrix on 2 x 2 MPI process grid
program slate00_matrix
    use, intrinsic :: iso_fortran_env
    use slate
    use mpi
    implicit none

    !! constants
    integer(kind=c_int64_t), parameter            :: m  = 2000
    integer(kind=c_int64_t), parameter            :: n  = 1000
    integer(kind=c_int64_t), parameter            :: nb = 256
    integer,                 parameter            :: p  = 2
    integer,                 parameter            :: q  = 2

    !! variables
    type(c_ptr)                                   :: A
    integer(kind=c_int64_t)                       :: nt, mt, i, j
    type(slate_Tile_r32)                          :: T
    integer(kind=c_int64_t)                       :: tile_nb, tile_mb, tile_lda
    type(c_ptr)                                   :: tile_data_ptr
    real(kind=c_float),                  pointer  :: tile_data(:)
    integer(kind=c_int64_t)                       :: ii, jj

    type(c_ptr)                                   :: A_scalapack
    real(kind=c_float),                  pointer  :: A_scalapack_data(:)
    integer(kind=c_int64_t)                       :: lda

    !! MPI variables
    integer(kind=c_int)                           :: mpi_ierr
    integer(kind=c_int)                           :: mpi_rank
    integer(kind=c_int)                           :: mpi_size

    !! MPI
    call MPI_Init( mpi_ierr )
    call MPI_Comm_rank( MPI_COMM_WORLD, mpi_rank, mpi_ierr )
    call MPI_Comm_size( MPI_COMM_WORLD, mpi_size, mpi_ierr )

    lda = m
    allocate( A_scalapack_data( lda * n ) )
    A_scalapack = slate_Matrix_create_fromScaLAPACK_r32( m, n, A_scalapack_data, m, nb, nb, p, q, MPI_COMM_WORLD )
    call slate_Matrix_destroy_r32( A_scalapack );
    deallocate( A_scalapack_data )

    A = slate_Matrix_create_r32( m, n, nb, p, q, MPI_COMM_WORLD )
    call slate_Matrix_insertLocalTiles_r32( A )
    nt = slate_Matrix_nt_r32( A )
    mt = slate_Matrix_mt_r32( A )
    do j = 0, nt-1
        do i = 0, mt-1
            if ( slate_Matrix_tileIsLocal_r32( A, i, j ) ) then
                T = slate_Matrix_at_r32( A, i, j )
                tile_nb = slate_Tile_nb_r32( T )
                tile_mb = slate_Tile_mb_r32( T )
                tile_lda = slate_Tile_stride_r32( T )
                tile_data_ptr = slate_Tile_data_r32( T )
                CALL c_f_pointer( tile_data_ptr, tile_data, [ tile_mb * tile_nb ] )
                do jj = 0, tile_nb-1
                    do ii = 0, tile_mb-1
                        tile_data( ii + jj*tile_lda ) = ii + jj*100
                        print *, tile_data( ii + jj*tile_lda )
                    end do
                end do
            end if
        end do
    end do

    call slate_Matrix_destroy_r32( A );

end program slate00_matrix
