! ex05_blas.f90
! BLAS routines
program ex05_blas
    use, intrinsic :: iso_fortran_env
    use slate
    use mpi
    use util
    implicit none

    !! Variables
    logical                            :: types(4)
    integer(kind=c_int)                :: p_grid, q_grid

    integer(kind=c_int)                :: provided, ierr
    integer(kind=c_int)                :: mpi_rank, mpi_size

    !! Get requested types
    call parse_args( types );

    !! MPI
    call MPI_Init_thread( MPI_THREAD_MULTIPLE, provided, ierr )
    if ((ierr .ne. 0) .or. (provided .ne. MPI_THREAD_MULTIPLE)) then
        print *, "Error: MPI_Init_thread"
        return
    end if
    call MPI_Comm_size( MPI_COMM_WORLD, mpi_size, ierr )
    if (ierr .ne. 0) then
        print *, "Error: MPI_Comm_size"
        return
    end if
    call MPI_Comm_rank( MPI_COMM_WORLD, mpi_rank, ierr )
    if (ierr .ne. 0) then
        print *, "Error: MPI_Comm_rank"
        return
    end if

    call grid_size( mpi_size, p_grid, q_grid )

    call srand( 100 * mpi_rank )

    if (types(1)) then
        call test_gemm_r32()
        call test_gemm_trans_r32()

        if (mpi_rank == 0) then
          print *
         end if
    end if
    if (types(2)) then
        call test_gemm_r64()
        call test_gemm_trans_r64()

        if (mpi_rank == 0) then
          print *
         end if
    end if
    if (types(3)) then
        call test_gemm_c32()
        call test_gemm_trans_c32()

        if (mpi_rank == 0) then
          print *
         end if
    end if
    if (types(4)) then
        call test_gemm_c64()
        call test_gemm_trans_c64()

        if (mpi_rank == 0) then
          print *
         end if
    end if

    call MPI_Finalize( ierr )
    if (ierr .ne. 0) then
        print *, "Error: MPI_Finalize"
        return
    end if

contains

    subroutine test_gemm_r32()
        !! Constants
        integer(kind=c_int64_t), parameter :: m  = 2000
        integer(kind=c_int64_t), parameter :: n  = 1000
        integer(kind=c_int64_t), parameter :: k  = 500
        integer(kind=c_int64_t), parameter :: nb = 256

        real(kind=c_float),      parameter :: alpha = 2.0
        real(kind=c_float),      parameter :: beta  = 1.0

        !! Variables
        integer(kind=c_int64_t)            :: i
        type(c_ptr)                        :: A, B, C, opts

        !! Example
        call print_func( mpi_rank, 'test_gemm_r32' )

        A = slate_Matrix_create_r32( m, k, nb, p_grid, q_grid, MPI_COMM_WORLD )
        B = slate_Matrix_create_r32( k, n, nb, p_grid, q_grid, MPI_COMM_WORLD )
        C = slate_Matrix_create_r32( m, n, nb, p_grid, q_grid, MPI_COMM_WORLD )
        call slate_Matrix_insertLocalTiles_r32( A )
        call slate_Matrix_insertLocalTiles_r32( B )
        call slate_Matrix_insertLocalTiles_r32( C )
        call random_matrix_type_r32( A )
        call random_matrix_type_r32( B )
        call random_matrix_type_r32( C )

        ! C = alpha A B + beta C
        call slate_multiply_r32( alpha, A, B, beta, C, c_null_ptr )

        if (slate_Matrix_num_devices_r32( C ) > 0) then
            opts = slate_Options_create()
            call slate_Options_set_Target( opts, slate_Target_Devices );
            call slate_Options_set_Lookahead( opts, 2_int64 )

            call slate_multiply_r32( alpha, A, B, beta, C, opts )

            call slate_Options_destroy( opts )
        endif


        call slate_Matrix_destroy_r32( A )
        call slate_Matrix_destroy_r32( B )
        call slate_Matrix_destroy_r32( C )

    end subroutine test_gemm_r32

    subroutine test_gemm_r64()
        !! Constants
        integer(kind=c_int64_t), parameter :: m  = 2000
        integer(kind=c_int64_t), parameter :: n  = 1000
        integer(kind=c_int64_t), parameter :: k  = 500
        integer(kind=c_int64_t), parameter :: nb = 256

        real(kind=c_double),     parameter :: alpha = 2.0
        real(kind=c_double),     parameter :: beta  = 1.0

        !! Variables
        integer(kind=c_int64_t)            :: i
        type(c_ptr)                        :: A, B, C, opts

        !! Example
        call print_func( mpi_rank, 'test_gemm_r64' )

        A = slate_Matrix_create_r64( m, k, nb, p_grid, q_grid, MPI_COMM_WORLD )
        B = slate_Matrix_create_r64( k, n, nb, p_grid, q_grid, MPI_COMM_WORLD )
        C = slate_Matrix_create_r64( m, n, nb, p_grid, q_grid, MPI_COMM_WORLD )
        call slate_Matrix_insertLocalTiles_r64( A )
        call slate_Matrix_insertLocalTiles_r64( B )
        call slate_Matrix_insertLocalTiles_r64( C )
        call random_matrix_type_r64( A )
        call random_matrix_type_r64( B )
        call random_matrix_type_r64( C )

        ! C = alpha A B + beta C
        call slate_multiply_r64( alpha, A, B, beta, C, c_null_ptr )

        if (slate_Matrix_num_devices_r64( C ) > 0) then
            opts = slate_Options_create()
            call slate_Options_set_Target( opts, slate_Target_Devices );
            call slate_Options_set_Lookahead( opts, 2_int64 )

            call slate_multiply_r64( alpha, A, B, beta, C, opts )

            call slate_Options_destroy( opts )
        endif


        call slate_Matrix_destroy_r64( A )
        call slate_Matrix_destroy_r64( B )
        call slate_Matrix_destroy_r64( C )

    end subroutine test_gemm_r64

    subroutine test_gemm_c32()
        !! Constants
        integer(kind=c_int64_t), parameter :: m  = 2000
        integer(kind=c_int64_t), parameter :: n  = 1000
        integer(kind=c_int64_t), parameter :: k  = 500
        integer(kind=c_int64_t), parameter :: nb = 256

        complex(kind=c_float),   parameter :: alpha = 2.0
        complex(kind=c_float),   parameter :: beta  = 1.0

        !! Variables
        integer(kind=c_int64_t)            :: i
        type(c_ptr)                        :: A, B, C, opts

        !! Example
        call print_func( mpi_rank, 'test_gemm_c32' )

        A = slate_Matrix_create_c32( m, k, nb, p_grid, q_grid, MPI_COMM_WORLD )
        B = slate_Matrix_create_c32( k, n, nb, p_grid, q_grid, MPI_COMM_WORLD )
        C = slate_Matrix_create_c32( m, n, nb, p_grid, q_grid, MPI_COMM_WORLD )
        call slate_Matrix_insertLocalTiles_c32( A )
        call slate_Matrix_insertLocalTiles_c32( B )
        call slate_Matrix_insertLocalTiles_c32( C )
        call random_matrix_type_c32( A )
        call random_matrix_type_c32( B )
        call random_matrix_type_c32( C )

        ! C = alpha A B + beta C
        call slate_multiply_c32( alpha, A, B, beta, C, c_null_ptr )

        if (slate_Matrix_num_devices_c32( C ) > 0) then
            opts = slate_Options_create()
            call slate_Options_set_Target( opts, slate_Target_Devices );
            call slate_Options_set_Lookahead( opts, 2_int64 )

            call slate_multiply_c32( alpha, A, B, beta, C, opts )

            call slate_Options_destroy( opts )
        endif


        call slate_Matrix_destroy_c32( A )
        call slate_Matrix_destroy_c32( B )
        call slate_Matrix_destroy_c32( C )

    end subroutine test_gemm_c32

    subroutine test_gemm_c64()
        !! Constants
        integer(kind=c_int64_t), parameter :: m  = 2000
        integer(kind=c_int64_t), parameter :: n  = 1000
        integer(kind=c_int64_t), parameter :: k  = 500
        integer(kind=c_int64_t), parameter :: nb = 256

        complex(kind=c_double),  parameter :: alpha = 2.0
        complex(kind=c_double),  parameter :: beta  = 1.0

        !! Variables
        integer(kind=c_int64_t)            :: i
        type(c_ptr)                        :: A, B, C, opts

        !! Example
        call print_func( mpi_rank, 'test_gemm_c64' )

        A = slate_Matrix_create_c64( m, k, nb, p_grid, q_grid, MPI_COMM_WORLD )
        B = slate_Matrix_create_c64( k, n, nb, p_grid, q_grid, MPI_COMM_WORLD )
        C = slate_Matrix_create_c64( m, n, nb, p_grid, q_grid, MPI_COMM_WORLD )
        call slate_Matrix_insertLocalTiles_c64( A )
        call slate_Matrix_insertLocalTiles_c64( B )
        call slate_Matrix_insertLocalTiles_c64( C )
        call random_matrix_type_c64( A )
        call random_matrix_type_c64( B )
        call random_matrix_type_c64( C )

        ! C = alpha A B + beta C
        call slate_multiply_c64( alpha, A, B, beta, C, c_null_ptr )

        if (slate_Matrix_num_devices_c64( C ) > 0) then
            opts = slate_Options_create()
            call slate_Options_set_Target( opts, slate_Target_Devices );
            call slate_Options_set_Lookahead( opts, 2_int64 )

            call slate_multiply_c64( alpha, A, B, beta, C, opts )

            call slate_Options_destroy( opts )
        endif


        call slate_Matrix_destroy_c64( A )
        call slate_Matrix_destroy_c64( B )
        call slate_Matrix_destroy_c64( C )

    end subroutine test_gemm_c64

    subroutine test_gemm_trans_r32()
        !! Constants
        integer(kind=c_int64_t), parameter :: m  = 2000
        integer(kind=c_int64_t), parameter :: n  = 1000
        integer(kind=c_int64_t), parameter :: k  = 500
        integer(kind=c_int64_t), parameter :: nb = 256

        real(kind=c_float),      parameter :: alpha = 2.0
        real(kind=c_float),      parameter :: beta  = 1.0

        !! Variables
        integer(kind=c_int64_t)            :: i
        type(c_ptr)                        :: A, B, C, opts

        !! Example
        call print_func( mpi_rank, 'test_gemm_trans_r32' )

        A = slate_Matrix_create_r32( k, m, nb, p_grid, q_grid, MPI_COMM_WORLD )
        B = slate_Matrix_create_r32( n, k, nb, p_grid, q_grid, MPI_COMM_WORLD )
        C = slate_Matrix_create_r32( m, n, nb, p_grid, q_grid, MPI_COMM_WORLD )
        call slate_Matrix_insertLocalTiles_r32( A )
        call slate_Matrix_insertLocalTiles_r32( B )
        call slate_Matrix_insertLocalTiles_r32( C )
        call random_matrix_type_r32( A )
        call random_matrix_type_r32( B )
        call random_matrix_type_r32( C )

        ! Matrices can be transposed or conjugate-transposed beforehand
        ! C = alpha AT BH + beta C
        call slate_Matrix_transpose_in_place_r32( A );
        call slate_Matrix_conj_transpose_in_place_r32( B );
        call slate_multiply_r32( alpha, A, B, beta, C, c_null_ptr )

        call slate_Matrix_destroy_r32( A )
        call slate_Matrix_destroy_r32( B )
        call slate_Matrix_destroy_r32( C )

    end subroutine test_gemm_trans_r32

    subroutine test_gemm_trans_r64()
        !! Constants
        integer(kind=c_int64_t), parameter :: m  = 2000
        integer(kind=c_int64_t), parameter :: n  = 1000
        integer(kind=c_int64_t), parameter :: k  = 500
        integer(kind=c_int64_t), parameter :: nb = 256

        real(kind=c_double),     parameter :: alpha = 2.0
        real(kind=c_double),     parameter :: beta  = 1.0

        !! Variables
        integer(kind=c_int64_t)            :: i
        type(c_ptr)                        :: A, B, C, opts

        !! Example
        call print_func( mpi_rank, 'test_gemm_trans_r64' )

        A = slate_Matrix_create_r64( k, m, nb, p_grid, q_grid, MPI_COMM_WORLD )
        B = slate_Matrix_create_r64( n, k, nb, p_grid, q_grid, MPI_COMM_WORLD )
        C = slate_Matrix_create_r64( m, n, nb, p_grid, q_grid, MPI_COMM_WORLD )
        call slate_Matrix_insertLocalTiles_r64( A )
        call slate_Matrix_insertLocalTiles_r64( B )
        call slate_Matrix_insertLocalTiles_r64( C )
        call random_matrix_type_r64( A )
        call random_matrix_type_r64( B )
        call random_matrix_type_r64( C )

        ! Matrices can be transposed or conjugate-transposed beforehand
        ! C = alpha AT BH + beta C
        call slate_Matrix_transpose_in_place_r64( A );
        call slate_Matrix_conj_transpose_in_place_r64( B );
        call slate_multiply_r64( alpha, A, B, beta, C, c_null_ptr )

        call slate_Matrix_destroy_r64( A )
        call slate_Matrix_destroy_r64( B )
        call slate_Matrix_destroy_r64( C )

    end subroutine test_gemm_trans_r64

    subroutine test_gemm_trans_c32()
        !! Constants
        integer(kind=c_int64_t), parameter :: m  = 2000
        integer(kind=c_int64_t), parameter :: n  = 1000
        integer(kind=c_int64_t), parameter :: k  = 500
        integer(kind=c_int64_t), parameter :: nb = 256

        complex(kind=c_float),   parameter :: alpha = 2.0
        complex(kind=c_float),   parameter :: beta  = 1.0

        !! Variables
        integer(kind=c_int64_t)            :: i
        type(c_ptr)                        :: A, B, C, opts

        !! Example
        call print_func( mpi_rank, 'test_gemm_trans_c32' )

        A = slate_Matrix_create_c32( k, m, nb, p_grid, q_grid, MPI_COMM_WORLD )
        B = slate_Matrix_create_c32( n, k, nb, p_grid, q_grid, MPI_COMM_WORLD )
        C = slate_Matrix_create_c32( m, n, nb, p_grid, q_grid, MPI_COMM_WORLD )
        call slate_Matrix_insertLocalTiles_c32( A )
        call slate_Matrix_insertLocalTiles_c32( B )
        call slate_Matrix_insertLocalTiles_c32( C )
        call random_matrix_type_c32( A )
        call random_matrix_type_c32( B )
        call random_matrix_type_c32( C )

        ! Matrices can be transposed or conjugate-transposed beforehand
        ! C = alpha AT BH + beta C
        call slate_Matrix_transpose_in_place_c32( A );
        call slate_Matrix_conj_transpose_in_place_c32( B );
        call slate_multiply_c32( alpha, A, B, beta, C, c_null_ptr )

        call slate_Matrix_destroy_c32( A )
        call slate_Matrix_destroy_c32( B )
        call slate_Matrix_destroy_c32( C )

    end subroutine test_gemm_trans_c32

    subroutine test_gemm_trans_c64()
        !! Constants
        integer(kind=c_int64_t), parameter :: m  = 2000
        integer(kind=c_int64_t), parameter :: n  = 1000
        integer(kind=c_int64_t), parameter :: k  = 500
        integer(kind=c_int64_t), parameter :: nb = 256

        complex(kind=c_double),  parameter :: alpha = 2.0
        complex(kind=c_double),  parameter :: beta  = 1.0

        !! Variables
        integer(kind=c_int64_t)            :: i
        type(c_ptr)                        :: A, B, C, opts

        !! Example
        call print_func( mpi_rank, 'test_gemm_trans_c64' )

        A = slate_Matrix_create_c64( k, m, nb, p_grid, q_grid, MPI_COMM_WORLD )
        B = slate_Matrix_create_c64( n, k, nb, p_grid, q_grid, MPI_COMM_WORLD )
        C = slate_Matrix_create_c64( m, n, nb, p_grid, q_grid, MPI_COMM_WORLD )
        call slate_Matrix_insertLocalTiles_c64( A )
        call slate_Matrix_insertLocalTiles_c64( B )
        call slate_Matrix_insertLocalTiles_c64( C )
        call random_matrix_type_c64( A )
        call random_matrix_type_c64( B )
        call random_matrix_type_c64( C )

        ! Matrices can be transposed or conjugate-transposed beforehand
        ! C = alpha AT BH + beta C
        call slate_Matrix_transpose_in_place_c64( A );
        call slate_Matrix_conj_transpose_in_place_c64( B );
        call slate_multiply_c64( alpha, A, B, beta, C, c_null_ptr )

        call slate_Matrix_destroy_c64( A )
        call slate_Matrix_destroy_c64( B )
        call slate_Matrix_destroy_c64( C )

    end subroutine test_gemm_trans_c64

end program ex05_blas
