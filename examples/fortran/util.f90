! utility files
module util
    use iso_c_binding
    public

contains

    subroutine print_func( mpi_rank, func )
        use, intrinsic :: iso_fortran_env
        implicit none
        integer(kind=c_int) :: mpi_rank
        character(len=*)    :: func

        if (mpi_rank == 0) then
            print '(a, i1, a, a)', 'rank ', mpi_rank, ': ', func
        end if
    end subroutine

    subroutine random_matrix_r32( m, n, A, lda )
        use, intrinsic :: iso_fortran_env
        implicit none
        integer(kind=c_int64_t)         :: m, n, lda
        real(kind=c_float), pointer     :: A(:)
        integer                         :: i, j

        do j = 0, n-1
            do i = 1, m
                A( i + j*lda ) = rand()
            end do
        end do
    end subroutine

    subroutine random_matrix_r64( m, n, A, lda )
        use, intrinsic :: iso_fortran_env
        implicit none
        integer(kind=c_int64_t)         :: m, n, lda
        real(kind=c_double), pointer    :: A(:)
        integer                         :: i, j

        do j = 0, n-1
            do i = 1, m
                A( i + j*lda ) = rand()
            end do
        end do
    end subroutine

    subroutine random_matrix_c32( m, n, A, lda )
        use, intrinsic :: iso_fortran_env
        implicit none
        integer(kind=c_int64_t)         :: m, n, lda
        complex(kind=c_float), pointer  :: A(:)
        integer                         :: i, j

        do j = 0, n-1
            do i = 1, m
                A( i + j*lda ) = cmplx( rand(), rand() )
            end do
        end do
    end subroutine

    subroutine random_matrix_c64( m, n, A, lda )
        use, intrinsic :: iso_fortran_env
        implicit none
        integer(kind=c_int64_t)         :: m, n, lda
        complex(kind=c_double), pointer :: A(:)
        integer                         :: i, j

        do j = 0, n-1
            do i = 1, m
                A( i + j*lda ) = cmplx( rand(), rand() )
            end do
        end do
    end subroutine

    subroutine random_matrix_type_r32( A )
        use, intrinsic :: iso_fortran_env
        use slate
        implicit none

        type(c_ptr), value              :: A
        type(slate_Tile_r32)            :: T
        type(c_ptr)                     :: T_data_c
        real(kind=c_float), pointer     :: T_data(:)
        integer(kind=c_int64_t)         :: mb, nb, lda
        integer(kind=c_int64_t)         :: i, j

        do j = 0, slate_Matrix_nt_r32( A )-1
            do i = 0, slate_Matrix_mt_r32( A )-1
                if (slate_Matrix_tileIsLocal_r32( A, i, j )) then
                    T = slate_Matrix_at_r32( A, i, j )
                    mb = slate_Tile_mb_r32( T )
                    nb = slate_Tile_nb_r32( T )
                    lda = slate_Tile_stride_r32( T )
                    T_data_c = slate_Tile_data_r32( T )
                    call c_f_pointer( T_data_c, T_data, [lda*nb] )

                    call random_matrix_r32( mb, nb, T_data, lda )
                end if
            end do
        end do
    end subroutine

    subroutine random_matrix_type_r64( A )
        use, intrinsic :: iso_fortran_env
        use slate
        implicit none

        type(c_ptr), value              :: A
        type(slate_Tile_r64)            :: T
        type(c_ptr)                     :: T_data_c
        real(kind=c_double), pointer    :: T_data(:)
        integer(kind=c_int64_t)         :: mb, nb, lda
        integer(kind=c_int64_t)         :: i, j

        do j = 0, slate_Matrix_nt_r64( A )-1
            do i = 0, slate_Matrix_mt_r64( A )-1
                if (slate_Matrix_tileIsLocal_r64( A, i, j )) then
                    T = slate_Matrix_at_r64( A, i, j )
                    mb = slate_Tile_mb_r64( T )
                    nb = slate_Tile_nb_r64( T )
                    lda = slate_Tile_stride_r64( T )
                    T_data_c = slate_Tile_data_r64( T )
                    call c_f_pointer( T_data_c, T_data, [lda*nb] )

                    call random_matrix_r64( mb, nb, T_data, lda )
                end if
            end do
        end do
    end subroutine

    subroutine random_matrix_type_c32( A )
        use, intrinsic :: iso_fortran_env
        use slate
        implicit none

        type(c_ptr), value              :: A
        type(slate_Tile_c32)            :: T
        type(c_ptr)                     :: T_data_c
        complex(kind=c_float), pointer  :: T_data(:)
        integer(kind=c_int64_t)         :: mb, nb, lda
        integer(kind=c_int64_t)         :: i, j

        do j = 0, slate_Matrix_nt_c32( A )-1
            do i = 0, slate_Matrix_mt_c32( A )-1
                if (slate_Matrix_tileIsLocal_c32( A, i, j )) then
                    T = slate_Matrix_at_c32( A, i, j )
                    mb = slate_Tile_mb_c32( T )
                    nb = slate_Tile_nb_c32( T )
                    lda = slate_Tile_stride_c32( T )
                    T_data_c = slate_Tile_data_c32( T )
                    call c_f_pointer( T_data_c, T_data, [lda*nb] )

                    call random_matrix_c32( mb, nb, T_data, lda )
                end if
            end do
        end do
    end subroutine

    subroutine random_matrix_type_c64( A )
        use, intrinsic :: iso_fortran_env
        use slate
        implicit none

        type(c_ptr), value              :: A
        type(slate_Tile_c64)            :: T
        type(c_ptr)                     :: T_data_c
        complex(kind=c_double), pointer :: T_data(:)
        integer(kind=c_int64_t)         :: mb, nb, lda
        integer(kind=c_int64_t)         :: i, j

        do j = 0, slate_Matrix_nt_c64( A )-1
            do i = 0, slate_Matrix_mt_c64( A )-1
                if (slate_Matrix_tileIsLocal_c64( A, i, j )) then
                    T = slate_Matrix_at_c64( A, i, j )
                    mb = slate_Tile_mb_c64( T )
                    nb = slate_Tile_nb_c64( T )
                    lda = slate_Tile_stride_c64( T )
                    T_data_c = slate_Tile_data_c64( T )
                    call c_f_pointer( T_data_c, T_data, [lda*nb] )

                    call random_matrix_c64( mb, nb, T_data, lda )
                end if
            end do
        end do
    end subroutine

    subroutine grid_size( mpi_size, p, q )
        use, intrinsic :: iso_fortran_env
        implicit none

        integer(kind=c_int) :: mpi_size, p, q
        real                :: mpi_size_real

        mpi_size_real = mpi_size

        do p = floor( sqrt( mpi_size_real ) ), 1, -1
            q = mpi_size / p

            if (p*q == mpi_size) then
               return
            end if
        end do
    end subroutine

    subroutine parse_args( types )

        logical           :: types(4)
        character(len=64) :: arg

        if (command_argument_count() == 0) then
            types( 1 ) = .true.
            types( 2 ) = .true.
            types( 3 ) = .true.
            types( 4 ) = .true.
        else
            types( 1 ) = .false.
            types( 2 ) = .false.
            types( 3 ) = .false.
            types( 4 ) = .false.
        endif

        do i = 1, command_argument_count()
            call get_command_argument( i, arg )
            if (arg == 's') then
                types( 1 ) = .true.
            else if (arg == 'd') then
                types( 2 ) = .true.
            else if (arg == 'c') then
                types( 3 ) = .true.
            else if (arg == 'z') then
                types( 4 ) = .true.
            end if
        end do
    end subroutine

end module util

