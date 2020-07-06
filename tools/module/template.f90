module slate
    use iso_c_binding
    use mpi
    ! todo: use mpi_f08
    implicit none

!-------------------------------------------------------------------------------

    type, bind(c) :: slate_Tile_r32
        integer(kind=c_int64_t) :: mb_
        integer(kind=c_int64_t) :: nb_
        integer(kind=c_int64_t) :: stride_
        integer(kind=c_int64_t) :: user_stride_
        integer(kind=c_int)     :: op_
        integer(kind=c_int)     :: uplo_
        type(c_ptr)             :: data_;
        type(c_ptr)             :: user_data_;
        type(c_ptr)             :: ext_data_;
        integer(kind=c_int)     :: kind_
        integer(kind=c_int)     :: layout_
        integer(kind=c_int)     :: user_layout_
        integer(kind=c_int)     :: device_
    end type slate_Tile_r32

!-------------------------------------------------------------------------------

    ! Interfaces of the C functions.
    interface
        function slate_Matrix_create_r32_c(m, n, nb, p, q, mpi_comm) &
            bind(c, name='slate_Matrix_create_r32')
                use iso_c_binding
                implicit none
                type(c_ptr)                                :: slate_Matrix_create_r32_c
                integer(kind=c_int64_t), intent(in), value :: m
                integer(kind=c_int64_t), intent(in), value :: n
                integer(kind=c_int64_t), intent(in), value :: nb
                integer(kind=c_int),     intent(in), value :: p
                integer(kind=c_int),     intent(in), value :: q
                integer(kind=c_int),     intent(in), value :: mpi_comm ! todo: type(MPI_Comm) -> use mpi_f08
        end function
    end interface

    ! Interfaces of the C functions.
    interface
        function slate_Matrix_create_fromScaLAPACK_r32_c(m, n, A_data, lda, mb, nb, p, q, mpi_comm) &
            bind(c, name='slate_Matrix_create_fromScaLAPACK_r32')
                use iso_c_binding
                implicit none
                type(c_ptr)                                   :: slate_Matrix_create_fromScaLAPACK_r32_c
                integer(kind=c_int64_t),  intent(in), value   :: m
                integer(kind=c_int64_t),  intent(in), value   :: n
                type(c_ptr),              intent(in)          :: A_data
                integer(kind=c_int64_t),  intent(in), value   :: lda
                integer(kind=c_int64_t),  intent(in), value   :: mb
                integer(kind=c_int64_t),  intent(in), value   :: nb
                integer(kind=c_int),      intent(in), value   :: p
                integer(kind=c_int),      intent(in), value   :: q
                integer(kind=c_int),      intent(in), value   :: mpi_comm ! todo: type(mpi_comm) -> use mpi_f08
        end function
    end interface

    ! Interfaces of the C functions.
    interface
        subroutine slate_Matrix_destroy_r32_c(A) &
            bind(c, name='slate_Matrix_destroy_r32')
                use iso_c_binding
                implicit none
                type(c_ptr), intent(in), value :: A
        end subroutine
    end interface

    ! Interfaces of the C functions.
    interface
        subroutine slate_Matrix_insertLocalTiles_r32_c(A) &
            bind(c, name='slate_Matrix_insertLocalTiles_r32')
                use iso_c_binding
                implicit none
                type(c_ptr), intent(in), value :: A
        end subroutine
    end interface

    ! Interfaces of the C functions.
    interface
        function slate_Matrix_nt_r32_c(A) &
            bind(c, name='slate_Matrix_nt_r32')
                use iso_c_binding
                implicit none
                integer(kind=c_int64_t)        :: slate_Matrix_nt_r32_c
                type(c_ptr), intent(in), value :: A
        end function
    end interface

    ! Interfaces of the C functions.
    interface
        function slate_Matrix_mt_r32_c(A) &
            bind(c, name='slate_Matrix_mt_r32')
                use iso_c_binding
                implicit none
                integer(kind=c_int64_t)        :: slate_Matrix_mt_r32_c
                type(c_ptr), intent(in), value :: A
        end function
    end interface

    ! Interfaces of the C functions.
    interface
        function slate_Matrix_tileIsLocal_r32_c(A, i, j) &
            bind(c, name='slate_Matrix_tileIsLocal_r32')
                use iso_c_binding
                implicit none
                logical(kind=c_bool)                       :: slate_Matrix_tileIsLocal_r32_c
                type(c_ptr),             intent(in), value :: A
                integer(kind=c_int64_t), intent(in), value :: i
                integer(kind=c_int64_t), intent(in), value :: j
        end function
    end interface

    ! Interfaces of the C functions.
    interface
        function slate_Matrix_at_r32_c(A, i, j) &
            bind(c, name='slate_Matrix_at_r32')
                use iso_c_binding
                import slate_Tile_r32
                implicit none
                type(slate_Tile_r32)                       :: slate_Matrix_at_r32_c
                type(c_ptr),             intent(in), value :: A
                integer(kind=c_int64_t), intent(in), value :: i
                integer(kind=c_int64_t), intent(in), value :: j
        end function
    end interface

    ! Interfaces of the C functions.
    interface
        function slate_Tile_nb_r32_c(T) &
            bind(c, name='slate_Tile_nb_r32')
                use iso_c_binding
                import slate_Tile_r32
                implicit none
                integer(kind=c_int64_t)                  :: slate_Tile_nb_r32_c
                type(slate_Tile_r32), intent(in), value  :: T
        end function
    end interface

    ! Interfaces of the C functions.
    interface
        function slate_Tile_mb_r32_c(T) &
            bind(c, name='slate_Tile_mb_r32')
                use iso_c_binding
                import slate_Tile_r32
                implicit none
                integer(kind=c_int64_t)                  :: slate_Tile_mb_r32_c
                type(slate_Tile_r32), intent(in), value  :: T
        end function
    end interface

    ! Interfaces of the C functions.
    interface
        function slate_Tile_stride_r32_c(T) &
            bind(c, name='slate_Tile_stride_r32')
                use iso_c_binding
                import slate_Tile_r32
                implicit none
                integer(kind=c_int64_t)                  :: slate_Tile_stride_r32_c
                type(slate_Tile_r32), intent(in), value  :: T
        end function
    end interface

    interface
        function slate_Tile_data_r32_c(T) &
            bind(c, name='slate_Tile_data_r32')
                use iso_c_binding
                import slate_Tile_r32
                implicit none
                type(c_ptr)                              :: slate_Tile_data_r32_c
                type(slate_Tile_r32), intent(in), value  :: T
        end function
    end interface

!-------------------------------------------------------------------------------
contains

    ! Wrappers of the C functions.
    function slate_Matrix_create_r32(m, n, nb, p, q, mpi_comm) result(A)
        use iso_c_binding
        implicit none
        type(c_ptr)                                 :: A
        integer(kind=c_int64_t),  intent(in), value :: m
        integer(kind=c_int64_t),  intent(in), value :: n
        integer(kind=c_int64_t),  intent(in), value :: nb
        integer(kind=c_int),      intent(in), value :: p
        integer(kind=c_int),      intent(in), value :: q
        integer(kind=c_int),      intent(in), value :: mpi_comm ! todo: type(mpi_comm) -> use mpi_f08

        A = slate_Matrix_create_r32_c(m, n, nb, p, q, mpi_comm)
    end function

    ! Wrappers of the C functions.
    function slate_Matrix_create_fromScaLAPACK_r32(m, n, A_data, lda, mb, nb, p, q, mpi_comm) result(A)
        use iso_c_binding
        implicit none
        type(c_ptr)                                   :: A
        integer(kind=c_int64_t),  intent(in), value   :: m
        integer(kind=c_int64_t),  intent(in), value   :: n
        ! type(c_ptr),              intent(in)          :: A_data
        real(kind=c_float),                  pointer  :: A_data(:)
        integer(kind=c_int64_t),  intent(in), value   :: lda
        integer(kind=c_int64_t),  intent(in), value   :: mb
        integer(kind=c_int64_t),  intent(in), value   :: nb
        integer(kind=c_int),      intent(in), value   :: p
        integer(kind=c_int),      intent(in), value   :: q
        integer(kind=c_int),      intent(in), value   :: mpi_comm ! todo: type(mpi_comm) -> use mpi_f08

        ! A = slate_Matrix_create_fromScaLAPACK_r32_c(m, n, A_data, lda, mb, nb, p, q, mpi_comm)
        A = slate_Matrix_create_fromScaLAPACK_r32_c(m, n, c_loc(A_data), lda, mb, nb, p, q, mpi_comm)
    end function

    ! Wrappers of the C functions.
    subroutine slate_Matrix_destroy_r32(A)
        use iso_c_binding
        implicit none
        type(c_ptr), intent(in), value :: A

        call slate_Matrix_destroy_r32_c(A)
    end subroutine

    ! Wrappers of the C functions.
    subroutine slate_Matrix_insertLocalTiles_r32(A)
        use iso_c_binding
        implicit none
        type(c_ptr), intent(in), value :: A

        call slate_Matrix_insertLocalTiles_r32_c(A)
    end subroutine

    ! Wrappers of the C functions.
    function slate_Matrix_nt_r32(A) result(nt)
        use iso_c_binding
        implicit none
        integer(kind=c_int64_t)         :: nt
        type(c_ptr),  intent(in), value :: A

        nt = slate_Matrix_nt_r32_c(A)
    end function

    ! Wrappers of the C functions.
    function slate_Matrix_mt_r32(A) result(mt)
        use iso_c_binding
        implicit none
        integer(kind=c_int64_t)         :: mt
        type(c_ptr),  intent(in), value :: A

        mt = slate_Matrix_mt_r32_c(A)
    end function

    ! Wrappers of the C functions.
    function slate_Matrix_tileIsLocal_r32(A, i, j) result(is_local)
        use iso_c_binding
        implicit none
        logical(kind=c_bool)                       :: is_local
        type(c_ptr),             intent(in), value :: A
        integer(kind=c_int64_t), intent(in), value :: i
        integer(kind=c_int64_t), intent(in), value :: j

        is_local = slate_Matrix_tileIsLocal_r32_c(A, i, j)
    end function

    ! Wrappers of the C functions.
    function slate_Matrix_at_r32(A, i, j) result(T)
        use iso_c_binding
        implicit none
        type(slate_Tile_r32)                       :: T
        type(c_ptr),             intent(in), value :: A
        integer(kind=c_int64_t), intent(in), value :: i
        integer(kind=c_int64_t), intent(in), value :: j

        T = slate_Matrix_at_r32_c(A, i, j)
    end function

    ! Wrappers of the C functions.
    function slate_Tile_nb_r32(T) result(nb)
        use iso_c_binding
        implicit none
        integer(kind=c_int64_t)                  :: nb
        type(slate_Tile_r32), intent(in), value  :: T

        nb = slate_Tile_nb_r32_c(T)
    end function

    ! Wrappers of the C functions.
    function slate_Tile_mb_r32(T) result(mb)
        use iso_c_binding
        implicit none
        integer(kind=c_int64_t)                  :: mb
        type(slate_Tile_r32), intent(in), value  ::  T

        mb = slate_Tile_mb_r32_c(T)
    end function

    ! Wrappers of the C functions.
    function slate_Tile_stride_r32(T) result(stride)
        use iso_c_binding
        implicit none
        integer(kind=c_int64_t)                  :: stride
        type(slate_Tile_r32), intent(in), value  :: T

        stride = slate_Tile_stride_r32_c(T)
    end function

    ! Wrappers of the C functions.
    function slate_Tile_data_r32(T) result(data)
        use iso_c_binding
        implicit none
        type(c_ptr)                              :: data
        type(slate_Tile_r32), intent(in), value  :: T

        data = slate_Tile_data_r32_c(T)
    end function

end module slate
