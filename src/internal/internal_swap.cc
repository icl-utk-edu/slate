//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the University of Tennessee nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------
// This research was supported by the Exascale Computing Project (17-SC-20-SC),
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//------------------------------------------------------------------------------
// For assistance with SLATE, email <slate-user@icl.utk.edu>.
// You can also join the "SLATE User" Google group by going to
// https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user,
// signing in with your Google credentials, and then clicking "Join group".
//------------------------------------------------------------------------------

#include "slate/Matrix.hh"
#include "slate/types.hh"
#include "slate/Tile_blas.hh"
#include "internal/internal.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Converts serial pivot vector to parallel pivot map.
///
/// @param[in] direction
///     Direction of pivoting:
///     - Direction::Forward,
///     - Direction::Backward.
///
/// @param[in] in_pivot
///     Serial (LAPACK-style) pivot vector.
///
/// @param[in,out] pivot
///     Parallel pivot for out-of-place pivoting.
///
/// @ingroup permute_internal
///
void makeParallelPivot(
    Direction direction,
    std::vector<Pivot> const& pivot,
    std::map<Pivot, Pivot>& pivot_map)
{
    int64_t begin, end, inc;
    if (direction == Direction::Forward) {
        begin = 0;
        end = pivot.size();
        inc = 1;
    }
    else {
        begin = pivot.size()-1;
        end = -1;
        inc = -1;
    }

    // Put the participating rows in the map.
    for (int64_t i = begin; i != end; i += inc) {
        if (pivot[i] != Pivot(0, i)) {
            pivot_map[ Pivot(0, i) ] = Pivot(0, i);
            pivot_map[ pivot[i]    ] = pivot[i];
        }
    }

    // Perform pivoting in the map.
    for (int64_t i = begin; i != end; i += inc)
        if (pivot[i] != Pivot(0, i))
            std::swap( pivot_map[ pivot[i] ], pivot_map[ Pivot(0, i) ] );
/*
    std::cout << std::endl;
    for (int64_t i = begin; i != end; i += inc)
        std::cout << pivot[i].tileIndex() << "\t"
                  << pivot[i].elementOffset() << std::endl;

    std::cout << std::endl;
    for (auto it : pivot_map)
        std::cout << it.first.tileIndex() << "\t"
                  << it.first.elementOffset() << "\t\t"
                  << it.second.tileIndex() << "\t"
                  << it.second.elementOffset() << std::endl;

    std::cout << "---------------------------" << std::endl;
*/
}

/*
//------------------------------------------------------------------------------
template <Target target, typename scalar_t>
void permuteRows(
    Direction direction,
    Matrix<scalar_t>&& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag)
{
    permuteRows(internal::TargetType<target>(), direction, A, pivot,
                layout, priority, tag);
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void permuteRows(
    internal::TargetType<Target::HostTask>,
    Direction direction,
    Matrix<scalar_t>& A, std::vector<Pivot>& pivot_vec,
    Layout layout, int priority, int tag)
{
    // CPU uses ColMajor
    assert(layout == Layout::ColMajor);

    std::map<Pivot, Pivot> pivot_map;
    makeParallelPivot(direction, pivot_vec, pivot_map);

    // todo: for performance optimization, merge with the loops below,
    // at least with lookahead, probably selectively
    A.tileGetAllForWriting(A.hostNum(), LayoutConvert(layout));

    {
        trace::Block trace_block("internal::permuteRows");

        for (int64_t j = 0; j < A.nt(); ++j) {
            int64_t nb = A.tileNb(j);

            std::vector<MPI_Request> requests;
            std::vector<MPI_Status> statuses;

            // Make copies of src rows.
            // Make room for dst rows.
            std::map<Pivot, std::vector<scalar_t> > src_rows;
            std::map<Pivot, std::vector<scalar_t> > dst_rows;
            for (auto const& pivot : pivot_map) {

                bool src_local = A.tileIsLocal(pivot.second.tileIndex(), j);
                if (src_local) {
                    src_rows[pivot.second].resize(nb);
                    copyRow(nb, A(pivot.second.tileIndex(), j),
                            pivot.second.elementOffset(), 0,
                            src_rows[pivot.second].data());
                }
                bool dst_local = A.tileIsLocal(pivot.first.tileIndex(), j);
                if (dst_local)
                    dst_rows[pivot.first].resize(nb);
            }

            // Local swap.
            for (auto const& pivot : pivot_map) {

                bool src_local = A.tileIsLocal(pivot.second.tileIndex(), j);
                bool dst_local = A.tileIsLocal(pivot.first.tileIndex(), j);

                if (src_local && dst_local) {
                    memcpy(dst_rows[pivot.first].data(),
                           src_rows[pivot.second].data(),
                           sizeof(scalar_t)*nb);
                }
            }

            // Launch all MPI.
            for (auto const& pivot : pivot_map) {

                bool src_local = A.tileIsLocal(pivot.second.tileIndex(), j);
                bool dst_local = A.tileIsLocal(pivot.first.tileIndex(), j);

                if (src_local && ! dst_local) {

                    requests.resize(requests.size()+1);
                    int dest = A.tileRank(pivot.first.tileIndex(), j);
                    MPI_Isend(src_rows[pivot.second].data(), nb,
                              mpi_type<scalar_t>::value, dest, tag, A.mpiComm(),
                              &requests[requests.size()-1]);
                }
                if (! src_local && dst_local) {

                    requests.resize(requests.size()+1);
                    int source = A.tileRank(pivot.second.tileIndex(), j);
                    MPI_Irecv(dst_rows[pivot.first].data(), nb,
                              mpi_type<scalar_t>::value, source, tag,
                              A.mpiComm(), &requests[requests.size()-1]);
                }
            }

            // Waitall.
            statuses.resize(requests.size());
            MPI_Waitall(requests.size(), requests.data(), statuses.data());

            for (auto const& pivot : pivot_map) {
                bool dst_local = A.tileIsLocal(pivot.first.tileIndex(), j);
                if (dst_local) {
                    copyRow(nb, dst_rows[pivot.first].data(),
                            A(pivot.first.tileIndex(), j),
                            pivot.first.elementOffset(), 0);
                }
            }
        }
    }
}
*/

//------------------------------------------------------------------------------
/// Permutes rows of a general matrix according to the pivot vector.
/// Host implementation.
/// todo: Restructure similarly to Hermitian permuteRowsCols
///       (use the auxiliary swap functions).
///
/// @ingroup permute_internal
///
template <typename scalar_t>
void permuteRows(
    internal::TargetType<Target::HostTask>,
    Direction direction,
    Matrix<scalar_t>& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag)
{
    // todo: for performance optimization, merge with the loops below,
    // at least with lookahead, probably selectively
    A.tileGetAllForWriting(A.hostNum(), LayoutConvert(layout));

    {
        trace::Block trace_block("internal::permuteRows");

        // todo: what about parallelizing this? MPI blocking?
        for (int64_t j = 0; j < A.nt(); ++j) {
            bool root = A.mpiRank() == A.tileRank(0, j);

            // Apply pivots forward (0, ..., k-1) or reverse (k-1, ..., 0)
            int64_t begin, end, inc;
            if (direction == Direction::Forward) {
                begin = 0;
                end   = pivot.size();
                inc   = 1;
            }
            else {
                begin = pivot.size() - 1;
                end   = -1;
                inc   = -1;
            }
            for (int64_t i = begin; i != end; i += inc) {
                int pivot_rank = A.tileRank(pivot[i].tileIndex(), j);

                // If I own the pivot.
                if (pivot_rank == A.mpiRank()) {
                    // If I am the root.
                    if (root) {
                        // If pivot not on the diagonal.
                        if (pivot[i].tileIndex() > 0 ||
                            pivot[i].elementOffset() > i)
                        {
                            // local swap
                            swapLocalRow(
                                0, A.tileNb(j),
                                A(0, j), i,
                                A(pivot[i].tileIndex(), j),
                                pivot[i].elementOffset());
                        }
                    }
                    // I am not the root.
                    else {
                        // MPI swap with the root
                        swapRemoteRow(
                            0, A.tileNb(j),
                            A(pivot[i].tileIndex(), j),
                            pivot[i].elementOffset(),
                            A.tileRank(0, j), A.mpiComm(),
                            tag);
                    }
                }
                // I don't own the pivot.
                else {
                    // If I am the root.
                    if (root) {
                        // MPI swap with the pivot owner
                        swapRemoteRow(
                            0,  A.tileNb(j),
                            A(0, j), i,
                            pivot_rank, A.mpiComm(),
                            tag);
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void permuteRows(
    internal::TargetType<Target::HostNest>,
    Direction direction,
    Matrix<scalar_t>& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag)
{
    // forward to HostTask
    permuteRows(internal::TargetType<Target::HostTask>(),
                direction, A, pivot, layout, priority, tag);
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void permuteRows(
    internal::TargetType<Target::HostBatch>,
    Direction direction,
    Matrix<scalar_t>& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag)
{
    // forward to HostTask
    permuteRows(internal::TargetType<Target::HostTask>(),
                direction, A, pivot, layout, priority, tag);
}

//------------------------------------------------------------------------------
/// Permutes rows according to the pivot vector.
/// Dispatches to target implementations.
///
/// @param[in] layout
///     Indicates the Layout (ColMajor/RowMajor) to operate with.
///     Local tiles of matrix on target devices will be converted to layout.
///
/// @ingroup permute_internal
///
template <Target target, typename scalar_t>
void permuteRows(
    Direction direction,
    Matrix<scalar_t>&& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag)
{
    permuteRows(internal::TargetType<target>(), direction, A, pivot,
                layout, priority, tag);
}

//------------------------------------------------------------------------------
// Precision-independent wrappers.
inline cublasStatus_t cublasSwap(
    cublasHandle_t handle,
    int n,
    float* x, int incx,
    float* y, int incy)
{
    return cublasSswap(handle, n, x, incx, y, incy);
}

inline cublasStatus_t cublasSwap(
    cublasHandle_t handle,
    int n,
    double* x, int incx,
    double* y, int incy)
{
    return cublasDswap(handle, n, x, incx, y, incy);
}

inline cublasStatus_t cublasSwap(
    cublasHandle_t handle,
    int n,
    std::complex<float> *x, int incx,
    std::complex<float> *y, int incy)
{
    return cublasCswap(handle, n, (cuComplex*) x, incx,
                                  (cuComplex*) y, incy);
}

inline cublasStatus_t cublasSwap(
    cublasHandle_t handle,
    int n,
    std::complex<double> *x, int incx,
    std::complex<double> *y, int incy)
{
    return cublasZswap(handle, n, (cuDoubleComplex*) x, incx,
                                  (cuDoubleComplex*) y, incy);
}

//------------------------------------------------------------------------------
/// Permutes rows of a general matrix according to the pivot vector.
/// GPU device implementation.
/// todo: Restructure similarly to Hermitian permute
///       (use the auxiliary swap functions).
/// todo: Just one function forwarding target.
///
/// @ingroup permute_internal
///
template <typename scalar_t>
void permuteRows(
    internal::TargetType<Target::Devices>,
    Direction direction,
    Matrix<scalar_t>& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag)
{
    // GPU uses RowMajor
    assert(layout == Layout::RowMajor);

    // todo: for performance optimization, merge with the loops below,
    // at least with lookahead, probably selectively
    A.tileGetAllForWritingOnDevices(LayoutConvert(layout));

    {
        trace::Block trace_block("internal::permuteRows");

        std::set<int> dev_set;

        for (int64_t j = 0; j < A.nt(); ++j) {
            bool root = A.mpiRank() == A.tileRank(0, j);

            // todo: relax the assumption of 1-D block cyclic distribution on devices
            int device = A.tileDevice(0, j);
            dev_set.insert(device);

            // Apply pivots forward (0, ..., k-1) or reverse (k-1, ..., 0)
            int64_t begin, end, inc;
            if (direction == Direction::Forward) {
                begin = 0;
                end   = pivot.size();
                inc   = 1;
            }
            else {
                begin = pivot.size() - 1;
                end   = -1;
                inc   = -1;
            }
            for (int64_t i = begin; i != end; i += inc) {
                int pivot_rank = A.tileRank(pivot[i].tileIndex(), j);

                // If I own the pivot.
                if (pivot_rank == A.mpiRank()) {
                    // If I am the root.
                    if (root) {
                        // If pivot not on the diagonal.
                        if (pivot[i].tileIndex() > 0 ||
                            pivot[i].elementOffset() > i)
                        {
                            // todo: assumes 1-D block cyclic
                            assert(A(0, j, device).layout() == Layout::RowMajor);
                            int64_t i1 = i;
                            int64_t i2 = pivot[i].elementOffset();
                            int64_t idx2 = pivot[i].tileIndex();
                            slate_cuda_call(cudaSetDevice(device));
                            cublasSwap(A.cublas_handle(device),
                                       A.tileNb(j),
                                       &A(0, j, device).at(i1, 0), 1,
                                       &A(idx2, j, device).at(i2, 0), 1);
                        }
                    }
                    // I am not the root.
                    else {
                        // MPI permute with the root
                        swapRemoteRowDevice(
                            0, A.tileNb(j), device,
                            A(pivot[i].tileIndex(), j, device),
                            pivot[i].elementOffset(),
                            A.tileRank(0, j), A.mpiComm(),
                            A.compute_stream(device), tag);
                    }
                }
                // I don't own the pivot.
                else {
                    // If I am the root.
                    if (root) {
                        // MPI permute with the pivot owner
                        swapRemoteRowDevice(
                            0,  A.tileNb(j), device,
                            A(0, j, device), i,
                            pivot_rank, A.mpiComm(),
                            A.compute_stream(device), tag);
                    }
                }
            }
        }

        for (int device : dev_set) {
            slate_cuda_call(cudaSetDevice(device));
            slate_cuda_call(
                cudaStreamSynchronize(A.compute_stream(device)));
        }
    }
}

//------------------------------------------------------------------------------
/// Swap a partial row of two tiles, either locally or remotely. Swaps
///     op1( A( ij_tuple_1 ) )[ offset_i1, j_offset : j_offset+n-1 ] and
///     op2( A( ij_tuple_2 ) )[ offset_i2, j_offset : j_offset+n-1 ].
///
/// @ingroup permute_internal
///
template <typename scalar_t>
void swapRow(
    int64_t j_offset, int64_t n,
    HermitianMatrix<scalar_t>& A,
    Op op1, std::tuple<int64_t, int64_t>&& ij_tuple_1, int64_t offset_i1,
    Op op2, std::tuple<int64_t, int64_t>&& ij_tuple_2, int64_t offset_i2,
    int tag)
{
    int64_t i1 = std::get<0>(ij_tuple_1);
    int64_t j1 = std::get<1>(ij_tuple_1);

    int64_t i2 = std::get<0>(ij_tuple_2);
    int64_t j2 = std::get<1>(ij_tuple_2);

    if (A.tileRank(i1, j1) == A.mpiRank()) {
        if (A.tileRank(i2, j2) == A.mpiRank()) {
            // local swap
            swapLocalRow(
                j_offset, n,
                op1 == Op::NoTrans ? A(i1, j1) : transpose(A(i1, j1)), offset_i1,
                op2 == Op::NoTrans ? A(i2, j2) : transpose(A(i2, j2)), offset_i2);
        }
        else {
            // sending tile 1
            swapRemoteRow(
                j_offset, n,
                op1 == Op::NoTrans ? A(i1, j1) : transpose(A(i1, j1)), offset_i1,
                A.tileRank(i2, j2), A.mpiComm(), tag);
        }
    }
    else if (A.tileRank(i2, j2) == A.mpiRank()) {
        // sending tile 2
        swapRemoteRow(
            j_offset, n,
            op2 == Op::NoTrans ? A(i2, j2) : transpose(A(i2, j2)), offset_i2,
            A.tileRank(i1, j1), A.mpiComm(), tag);
    }
}

//------------------------------------------------------------------------------
/// Swap a single element of two tiles, either locally or remotely. Swaps
///     A( ij_tuple_1 )[ offset_i1, offset_j1 ] and
///     A( ij_tuple_2 )[ offset_i2, offset_j2 ].
///
/// @ingroup permute_internal
///
template <typename scalar_t>
void swapElement(
    HermitianMatrix<scalar_t>& A,
    std::tuple<int64_t, int64_t>&& ij_tuple_1,
    int64_t offset_i1, int64_t offset_j1,
    std::tuple<int64_t, int64_t>&& ij_tuple_2,
    int64_t offset_i2, int64_t offset_j2,
    int tag)
{
    int64_t i1 = std::get<0>(ij_tuple_1);
    int64_t j1 = std::get<1>(ij_tuple_1);

    int64_t i2 = std::get<0>(ij_tuple_2);
    int64_t j2 = std::get<1>(ij_tuple_2);

    if (A.tileRank(i1, j1) == A.mpiRank()) {
        if (A.tileRank(i2, j2) == A.mpiRank()) {
            // local swap
            std::swap(A(i1, j1).at(offset_i1, offset_j1),
                      A(i2, j2).at(offset_i2, offset_j2));
        }
        else {
            // sending tile 1
            swapRemoteElement(A(i1, j1), offset_i1, offset_j1,
                              A.tileRank(i2, j2), A.mpiComm(), tag);
        }
    }
    else if (A.tileRank(i2, j2) == A.mpiRank()) {
        // sending tile 2
        swapRemoteElement(A(i2, j2), offset_i2, offset_j2,
                          A.tileRank(i1, j1), A.mpiComm(), tag);
    }
}

//------------------------------------------------------------------------------
/// Permutes rows and cols, symmetrically, of a Hermitian matrix according to
/// the pivot vector.
/// Host implementation.
///
/// @ingroup permute_internal
///
template <typename scalar_t>
void permuteRowsCols(
    internal::TargetType<Target::HostTask>,
    Direction direction,
    HermitianMatrix<scalar_t>& A, std::vector<Pivot>& pivot,
    int priority, int tag)
{
    using blas::conj;

    assert(A.uplo() == Uplo::Lower);

    for (int64_t i = 0; i < A.mt(); ++i) {
        for (int64_t j = 0; j <= i; ++j) {
            if (A.tileIsLocal(i, j)) {
                #pragma omp task shared(A) priority(priority)
                {
                    A.tileGetForWriting(i, j, LayoutConvert::ColMajor);
                }
            }
        }
    }
    #pragma omp taskwait

    {
        trace::Block trace_block("internal::permuteRowsCols");

        // Apply pivots forward (0, ..., k-1) or reverse (k-1, ..., 0)
        int64_t begin, end, inc;
        if (direction == Direction::Forward) {
            begin = 0;
            end   = pivot.size();
            inc   = 1;
        }
        else {
            begin = pivot.size() - 1;
            end   = -1;
            inc   = -1;
        }
        for (int64_t i1 = begin; i1 != end; i1 += inc) {
            int64_t i2 = pivot[i1].elementOffset();
            int64_t j2 = pivot[i1].tileIndex();

            // If pivot not on the diagonal.
            if (j2 > 0 || i2 > i1) {

                // in the upper band
                swapRow(0, i1, A,
                        Op::NoTrans, {0,  0}, i1,
                        Op::NoTrans, {j2, 0}, i2, tag);
                if (j2 == 0) {
                    swapRow(i1+1, i2-i1, A,
                            Op::Trans,   {0, 0}, i1,
                            Op::NoTrans, {0, 0}, i2, tag);

                    swapRow(i2, A.tileNb(0)-i2, A,
                            Op::Trans, {0, 0}, i1,
                            Op::Trans, {0, 0}, i2, tag);
                }
                else {
                    swapRow(i1+1, A.tileNb(0)-i1-1, A,
                            Op::Trans,   {0,  0}, i1,
                            Op::NoTrans, {j2, 0}, i2, tag);

                    // in the lower band
                    swapRow(0, i2, A,
                            Op::Trans,   {j2,  0}, i1,
                            Op::NoTrans, {j2, j2}, i2, tag+1);

                    swapRow(i2+1, A.tileNb(j2)-i2-1, A,
                            Op::Trans, {j2,  0}, i1,
                            Op::Trans, {j2, j2}, i2, tag+1);
                }

                // Conjugate the crossing point.
                if (A.tileRank(j2, 0) == A.mpiRank())
                    A(j2, 0).at(i2, i1) = conj(A(j2, 0).at(i2, i1));

                // Swap the corners.
                swapElement(A,
                            {0, 0}, i1, i1,
                            {j2, j2}, i2, i2, tag);

                // before the lower band
                for (int64_t j1=1; j1 < j2; ++j1) {
                    swapRow(0, A.tileNb(j1), A,
                            Op::Trans,   {j1,  0}, i1,
                            Op::NoTrans, {j2, j1}, i2, tag+1+j1);
                }

                // after the lower band
                for (int64_t j1=j2+1; j1 < A.nt(); ++j1) {
                    swapRow(0, A.tileNb(j1), A,
                            Op::Trans, {j1,  0}, i1,
                            Op::Trans, {j1, j2}, i2, tag+1+j1);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Permutes rows and columns symmetrically according to the pivot vector.
/// Dispatches to target implementations.
/// @ingroup permute_internal
///
template <Target target, typename scalar_t>
void permuteRowsCols(
    Direction direction,
    HermitianMatrix<scalar_t>&& A, std::vector<Pivot>& pivot,
    int priority, int tag)
{
    permuteRowsCols(internal::TargetType<target>(), direction, A, pivot,
                    priority, tag);
}

//------------------------------------------------------------------------------
// Explicit instantiations for (general) Matrix.
// ----------------------------------------
template
void permuteRows<Target::HostTask, float>(
    Direction direction,
    Matrix<float>&& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag);

template
void permuteRows<Target::HostNest, float>(
    Direction direction,
    Matrix<float>&& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag);

template
void permuteRows<Target::HostBatch, float>(
    Direction direction,
    Matrix<float>&& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag);

template
void permuteRows<Target::Devices, float>(
    Direction direction,
    Matrix<float>&& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag);

// ----------------------------------------
template
void permuteRows<Target::HostTask, double>(
    Direction direction,
    Matrix<double>&& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag);

template
void permuteRows<Target::HostNest, double>(
    Direction direction,
    Matrix<double>&& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag);

template
void permuteRows<Target::HostBatch, double>(
    Direction direction,
    Matrix<double>&& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag);

template
void permuteRows<Target::Devices, double>(
    Direction direction,
    Matrix<double>&& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag);

// ----------------------------------------
template
void permuteRows< Target::HostTask, std::complex<float> >(
    Direction direction,
    Matrix< std::complex<float> >&& A,
    std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag);

template
void permuteRows< Target::HostNest, std::complex<float> >(
    Direction direction,
    Matrix< std::complex<float> >&& A,
    std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag);

template
void permuteRows< Target::HostBatch, std::complex<float> >(
    Direction direction,
    Matrix< std::complex<float> >&& A,
    std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag);

template
void permuteRows< Target::Devices, std::complex<float> >(
    Direction direction,
    Matrix< std::complex<float> >&& A,
    std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag);

// ----------------------------------------
template
void permuteRows< Target::HostTask, std::complex<double> >(
    Direction direction,
    Matrix< std::complex<double> >&& A,
    std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag);

template
void permuteRows< Target::HostNest, std::complex<double> >(
    Direction direction,
    Matrix< std::complex<double> >&& A,
    std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag);

template
void permuteRows< Target::HostBatch, std::complex<double> >(
    Direction direction,
    Matrix< std::complex<double> >&& A,
    std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag);

template
void permuteRows< Target::Devices, std::complex<double> >(
    Direction direction,
    Matrix< std::complex<double> >&& A,
    std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag);

//------------------------------------------------------------------------------
// Explicit instantiations for HermitianMatrix.
// ----------------------------------------
template
void permuteRowsCols<Target::HostTask, float>(
    Direction direction,
    HermitianMatrix<float>&& A, std::vector<Pivot>& pivot,
    int priority, int tag);

// ----------------------------------------
template
void permuteRowsCols<Target::HostTask, double>(
    Direction direction,
    HermitianMatrix<double>&& A, std::vector<Pivot>& pivot,
    int priority, int tag);

// ----------------------------------------
template
void permuteRowsCols< Target::HostTask, std::complex<float> >(
    Direction direction,
    HermitianMatrix< std::complex<float> >&& A,
    std::vector<Pivot>& pivot,
    int priority, int tag);

// ----------------------------------------
template
void permuteRowsCols< Target::HostTask, std::complex<double> >(
    Direction direction,
    HermitianMatrix< std::complex<double> >&& A,
    std::vector<Pivot>& pivot,
    int priority, int tag);

} // namespace internal
} // namespace slate
