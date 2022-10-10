#ifndef SLATE_INTERNAL_SWAP_HH
#define SLATE_INTERNAL_SWAP_HH

#include <blas.hh>

#include "slate/Tile.hh"
#include "slate/internal/util.hh"
#include "slate/internal/device.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Swap a partial row of two local tiles:
///     A[ i1, j_offset : j_offset+n-1 ] and
///     B[ i2, j_offset : j_offset+n-1 ].
/// Either or both tiles can be transposed to swap columns.
/// @ingroup swap_tile
///
template <typename scalar_t>
void swapLocalRow(
    int64_t j_offset, int64_t n,
    Tile<scalar_t>& A, int64_t i1,
    Tile<scalar_t>& B, int64_t i2)
{
    // todo: size assertions, quick return
    if (n <= 0) return;

    blas::swap(n, &A.at(i1,j_offset), A.rowIncrement(),
                  &B.at(i2,j_offset), B.rowIncrement());
}

//--------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup swap_tile
///
template <typename scalar_t>
void swapLocalRow(
    int64_t j_offset, int64_t n,
    Tile<scalar_t>&& A, int64_t i1,
    Tile<scalar_t>&& B, int64_t i2)
{
    swapLocalRow(j_offset, n, A, i1, B, i2);
}

//------------------------------------------------------------------------------
/// Swap a partial row, A[ i, j : j+n-1 ], with another MPI process.
/// The tile can be transposed to swap a column.
/// @ingroup swap_tile
///
template <typename scalar_t>
void swapRemoteRow(
    int64_t j, int64_t n,
    Tile<scalar_t>& A, int64_t i,
    int other_rank, MPI_Comm mpi_comm, int tag = 0)
{
    // todo: size assertions, quick return
    if (n <= 0) return;

    std::vector<scalar_t> local_row(n);
    std::vector<scalar_t> other_row(n);

    // todo: Perhaps create an MPI type and let MPI pack it?
    blas::copy(n, &A.at(i, j), A.rowIncrement(), &local_row[0], 1);

    MPI_Sendrecv(
        local_row.data(), n, mpi_type<scalar_t>::value, other_rank, tag,
        other_row.data(), n, mpi_type<scalar_t>::value, other_rank, tag,
        mpi_comm, MPI_STATUS_IGNORE);

    blas::copy(n, &other_row[0], 1, &A.at(i, j), A.rowIncrement());
}

//--------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup swap_tile
///
template <typename scalar_t>
void swapRemoteRow(
    int64_t j, int64_t n,
    Tile<scalar_t>&& A, int64_t i,
    int other_rank, MPI_Comm mpi_comm, int tag = 0)
{
    swapRemoteRow(j, n, A, i, other_rank, mpi_comm, tag);
}

//------------------------------------------------------------------------------
/// Swap a partial row, A[ i, j : j+n-1 ], on a GPU device,
/// with another MPI process.
/// The tile must be row-major, and cannot be transposed.
/// @ingroup swap_tile
///
// todo: implement with a GPUDirect call
template <typename scalar_t>
void swapRemoteRowDevice(
    int64_t j, int64_t n,
    int device, Tile<scalar_t>& A, int64_t i,
    int other_rank, MPI_Comm mpi_comm, blas::Queue& queue, int tag = 0)
{
    std::vector<scalar_t> local_row(n);
    std::vector<scalar_t> other_row(n);

    // todo: this assumes row is contiguous on GPU, right? Add asserts.

    blas::device_memcpy<scalar_t>(
        local_row.data(), &A.at(i, j), n,
        blas::MemcpyKind::DeviceToHost, queue);

    queue.sync();

    MPI_Sendrecv(
        local_row.data(), n, mpi_type<scalar_t>::value, other_rank, tag,
        other_row.data(), n, mpi_type<scalar_t>::value, other_rank, tag,
        mpi_comm, MPI_STATUS_IGNORE);

    blas::device_memcpy<scalar_t>(
        &A.at(i, j), other_row.data(), n,
        blas::MemcpyKind::HostToDevice, queue);

    queue.sync();
}

//--------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup swap_tile
///
template <typename scalar_t>
void swapRemoteRowDevice(
    int64_t j, int64_t n,
    int device, Tile<scalar_t>&& A, int64_t i,
    int other_rank, MPI_Comm mpi_comm, blas::Queue& queue, int tag = 0)
{
    swapRemoteRowDevice(j, n, device, A, i, other_rank, mpi_comm, queue, tag);
}

//------------------------------------------------------------------------------
/// Swap one element, A(i, j), with another MPI process.
/// @ingroup swap_tile
///
template <typename scalar_t>
void swapRemoteElement(
    Tile<scalar_t>& A, int64_t i, int64_t j,
    int other_rank, MPI_Comm mpi_comm, int tag = 0)
{
    scalar_t local_element = A(i, j);
    scalar_t other_element;

    MPI_Sendrecv(
        &local_element, 1, mpi_type<scalar_t>::value, other_rank, tag,
        &other_element, 1, mpi_type<scalar_t>::value, other_rank, tag,
        mpi_comm, MPI_STATUS_IGNORE);

    A.at(i, j) = other_element;
}

//--------------------------------------
/// Converts rvalue refs to lvalue refs.
/// @ingroup swap_tile
///
template <typename scalar_t>
void swapRemoteElement(
    Tile<scalar_t>&& A, int64_t i, int64_t j,
    int other_rank, MPI_Comm mpi_comm, int tag = 0)
{
    swapRemoteElement(A, i, j, other_rank, mpi_comm, tag);
}

}  // namespace slate

#endif // SLATE_INTERNAL_SWAP_HH
