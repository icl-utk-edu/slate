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

#include "slate_Matrix.hh"
#include "slate_types.hh"
#include "slate_Tile_blas.hh"
#include "slate_internal.hh"

namespace slate {
namespace internal {

///-----------------------------------------------------------------------------
/// \brief
/// Swaps rows according to the pivot vector.
/// Dispatches to target implementations.
template <Target target, typename scalar_t>
void swap(Direction direction,
          Matrix<scalar_t>&& A, std::vector<Pivot>& pivot,
          int priority, int tag, Layout layout)
{
    swap(internal::TargetType<target>(), direction, A, pivot,
         priority, tag, layout);
}

///-----------------------------------------------------------------------------
/// \brief
/// Swaps L shapes according to the pivot vector.
/// Dispatches to target implementations.
template <Target target, typename scalar_t>
void swap(Direction direction,
          HermitianMatrix<scalar_t>&& A, std::vector<Pivot>& pivot,
          int priority, int tag)
{
    swap(internal::TargetType<target>(), direction, A, pivot,
         priority, tag);
}

///-----------------------------------------------------------------------------
/// \brief
/// Swaps rows of a general matrix according to the pivot vector,
/// host implementation.
/// todo: Restructure similarly to Hermitian swap
///       (use the auxiliary swap functions).
template <typename scalar_t>
void swap(internal::TargetType<Target::HostTask>,
          Direction direction,
          Matrix<scalar_t>& A, std::vector<Pivot>& pivot,
          int priority, int tag, Layout layout)
{
    // CPU uses ColMajor
    assert(layout == Layout::ColMajor);
    for (int64_t i = 0; i < A.mt(); ++i) {
        for (int64_t j = 0; j < A.nt(); ++j) {
            if (A.tileIsLocal(i, j)) {
                #pragma omp task shared(A) priority(priority)
                {
                    A.tileMoveToHost(i, j, A.tileDevice(i, j));
                }
            }
        }
    }
    #pragma omp taskwait

    {
        trace::Block trace_block("internal::swap");

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
                            swap(0, A.tileNb(j),
                                 A(0, j), i,
                                 A(pivot[i].tileIndex(), j),
                                 pivot[i].elementOffset());
                        }
                    }
                    // I am not the root.
                    else {
                        // MPI swap with the root
                        swap(0, A.tileNb(j),
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
                        swap(0,  A.tileNb(j),
                             A(0, j), i,
                             pivot_rank, A.mpiComm(),
                             tag);
                    }
                }
            }
        }
    }
}

///-----------------------------------------------------------------------------
template <typename scalar_t>
void swap(int64_t j_offs, int64_t n,
          HermitianMatrix<scalar_t>& A,
          Op op1, std::tuple<int64_t, int64_t>&& ij_tuple_1, int64_t offs_1,
          Op op2, std::tuple<int64_t, int64_t>&& ij_tuple_2, int64_t offs_2,
          int tag)
{
    int64_t i1 = std::get<0>(ij_tuple_1);
    int64_t j1 = std::get<1>(ij_tuple_1);

    int64_t i2 = std::get<0>(ij_tuple_2);
    int64_t j2 = std::get<1>(ij_tuple_2);

    if (A.tileRank(i1, j1) == A.mpiRank()) {
        if (A.tileRank(i2, j2) == A.mpiRank()) {
            // local swap
            swap(j_offs, n,
                 op1 == Op::NoTrans ? A(i1, j1) : transpose(A(i1, j1)), offs_1,
                 op2 == Op::NoTrans ? A(i2, j2) : transpose(A(i2, j2)), offs_2);
        }
        else {
            // sending tile 1
            swap(j_offs, n,
                 op1 == Op::NoTrans ? A(i1, j1) : transpose(A(i1, j1)), offs_1,
                 A.tileRank(i2, j2), A.mpiComm(), tag);
        }
    }
    else {
        if (A.tileRank(i2, j2) == A.mpiRank()) {
            // sending tile 2
            swap(j_offs, n,
                 op2 == Op::NoTrans ? A(i2, j2) : transpose(A(i2, j2)), offs_2,
                 A.tileRank(i1, j1), A.mpiComm(), tag);
        }
    }
}

///-----------------------------------------------------------------------------
template <typename scalar_t>
void swap(HermitianMatrix<scalar_t>& A,
          std::tuple<int64_t, int64_t>&& ij_tuple_1,
          int64_t offs_i1, int64_t offs_j1,
          std::tuple<int64_t, int64_t>&& ij_tuple_2,
          int64_t offs_i2, int64_t offs_j2,
          int tag)
{
    int64_t i1 = std::get<0>(ij_tuple_1);
    int64_t j1 = std::get<1>(ij_tuple_1);

    int64_t i2 = std::get<0>(ij_tuple_2);
    int64_t j2 = std::get<1>(ij_tuple_2);

    if (A.tileRank(i1, j1) == A.mpiRank()) {
        if (A.tileRank(i2, j2) == A.mpiRank()) {
            // local swap
            std::swap(A(i1, j1).at(offs_i1, offs_j1),
                      A(i2, j2).at(offs_i2, offs_j2));
        }
        else {
            // sending tile 1
            swap(A(i1, j1), offs_i1, offs_j1,
                 A.tileRank(i2, j2), A.mpiComm(), tag);
        }
    }
    else {
        if (A.tileRank(i2, j2) == A.mpiRank()) {
            // sending tile 2
            swap(A(i2, j2), offs_i2, offs_j2,
                 A.tileRank(i1, j1), A.mpiComm(), tag);
        }
    }
}

///-----------------------------------------------------------------------------
/// \brief
/// (Symmetric?) Swaps of rows (and columns?) of a Hermitian matrix according to
/// the pivot vector, host implementation.
// todo: is this symmetric swapping, both rows & columns?
template <typename scalar_t>
void swap(internal::TargetType<Target::HostTask>,
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
                    A.tileMoveToHost(i, j, A.tileDevice(i, j));
                }
            }
        }
    }
    #pragma omp taskwait

    {
        trace::Block trace_block("internal::swap");

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

                int64_t j1 = 0;

                // in the upper band
                swap(0, i1, A,
                     Op::NoTrans, {0, j1}, i1,
                     Op::NoTrans, {j2, j1}, i2, tag);

                swap(i1+1, A.tileNb(j1)-i1-1, A,
                     Op::Trans, {0, j1}, i1,
                     Op::NoTrans, {j2, j1}, i2, tag);

                // before the lower band
                for (++j1; j1 < j2; ++j1) {
                    swap(0, A.tileNb(j1), A,
                         Op::Trans, {j1, 0}, i1,
                         Op::NoTrans, {j2, j1}, i2, tag);
                }

                // in the lower band
                swap(0, i2, A,
                     Op::Trans, {j1, 0}, i1,
                     Op::NoTrans, {j2, j1}, i2, tag);

                swap(i2+1, A.tileNb(j2)-i2-1, A,
                    Op::Trans, {j1, 0}, i1,
                    Op::Trans, {j2, j1}, i2, tag);

                // after the lower band
                for (++j1; j1 < A.nt(); ++j1) {
                    swap(0, A.tileNb(j2), A,
                         Op::Trans, {j1, 0}, i1,
                         Op::Trans, {j1, j2}, i2, tag);
                }

                // Conjugate the crossing poing.
                if (A.tileRank(j2, 0) == A.mpiRank())
                    A(j2, 0).at(i2, i1) = conj(A(j2, 0).at(i2, i1));

                // Swap the corners.
                swap(A,
                     {0, 0}, i1, i1,
                     {j2, j2}, i2, i2, tag);
            }
        }
    }
}

//------------------------------------------------------------------------------
// Explicit instantiations for (general) Matrix.
// ----------------------------------------
template
void swap<Target::HostTask, float>(
    Direction direction,
    Matrix<float>&& A, std::vector<Pivot>& pivot,
    int priority, int tag, Layout layout);

// ----------------------------------------
template
void swap<Target::HostTask, double>(
    Direction direction,
    Matrix<double>&& A, std::vector<Pivot>& pivot,
    int priority, int tag, Layout layout);

// ----------------------------------------
template
void swap< Target::HostTask, std::complex<float> >(
    Direction direction,
    Matrix< std::complex<float> >&& A,
    std::vector<Pivot>& pivot,
    int priority, int tag, Layout layout);

// ----------------------------------------
template
void swap< Target::HostTask, std::complex<double> >(
    Direction direction,
    Matrix< std::complex<double> >&& A,
    std::vector<Pivot>& pivot,
    int priority, int tag, Layout layout);

//------------------------------------------------------------------------------
// Explicit instantiations for HermitianMatrix.
// ----------------------------------------
template
void swap<Target::HostTask, float>(
    Direction direction,
    HermitianMatrix<float>&& A, std::vector<Pivot>& pivot,
    int priority, int tag);

// ----------------------------------------
template
void swap<Target::HostTask, double>(
    Direction direction,
    HermitianMatrix<double>&& A, std::vector<Pivot>& pivot,
    int priority, int tag);

// ----------------------------------------
template
void swap< Target::HostTask, std::complex<float> >(
    Direction direction,
    HermitianMatrix< std::complex<float> >&& A,
    std::vector<Pivot>& pivot,
    int priority, int tag);

// ----------------------------------------
template
void swap< Target::HostTask, std::complex<double> >(
    Direction direction,
    HermitianMatrix< std::complex<double> >&& A,
    std::vector<Pivot>& pivot,
    int priority, int tag);

} // namespace internal
} // namespace slate
