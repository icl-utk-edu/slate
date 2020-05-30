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

#include "slate/slate.hh"
#include "aux/Debug.hh"
#include "slate/BandMatrix.hh"
#include "slate/Tile_blas.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::gbtrf from internal::specialization::gbtrf
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// Distributed parallel band LU factorization.
/// Generic implementation for any target.
/// Panel and lookahead computed on host using Host OpenMP task.
///
/// Warning: ColMajor layout is assumed
///
template <Target target, typename scalar_t>
void gbtrf(slate::internal::TargetType<target>,
           BandMatrix<scalar_t>& A, Pivots& pivots,
           int64_t ib, int max_panel_threads, int64_t lookahead)
{
    // using real_t = blas::real_type<scalar_t>;
    using BcastList = typename BandMatrix<scalar_t>::BcastList;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    const int64_t A_nt = A.nt();
    const int64_t A_mt = A.mt();
    const int64_t min_mt_nt = std::min(A.mt(), A.nt());
    pivots.resize(min_mt_nt);

    const scalar_t zero = 0.0;

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > column_vector(A_nt);
    uint8_t* column = column_vector.data();

    int64_t kl = A.lowerBandwidth();
    int64_t ku = A.upperBandwidth();

    // todo: initially, assume fixed size, square tiles for simplicity
    int64_t klt = ceildiv( kl, A.tileNb(0) );
    int64_t kut = ceildiv( ku, A.tileNb(0) );
    int64_t ku2t = ceildiv( (ku + kl), A.tileNb(0) );

    // Insert & zero potential fill above upper bandwidth
    A.upperBandwidth(kl + ku);
    //printf( "kl %lld, ku %lld, kl+ku %lld, A.lowerBW %lld, A.upperBW %lld\n",
    //        kl, ku, kl + ku, A.lowerBandwidth(), A.upperBandwidth() );
    for (int64_t i = 0; i < min_mt_nt; ++i) {
        for (int64_t j = i + 1 + kut; j < std::min(i + 1 + ku2t, A.nt()); ++j) {
            if (A.tileIsLocal(i, j)) {
                // todo: device?
                A.tileInsert(i, j);
                auto T = A(i, j);
                lapack::laset(lapack::MatrixType::General, T.mb(), T.nb(),
                              zero, zero, T.data(), T.stride());
                A.tileModified(i, j);
            }
        }
    }

    #pragma omp parallel
    #pragma omp master
    {
        omp_set_nested(1);
        for (int64_t k = 0; k < min_mt_nt; ++k) {

            const int64_t diag_len = std::min(A.tileMb(k), A.tileNb(k));
            pivots.at(k).resize(diag_len);

            // A( k:i_end-1, k ) is the panel
            // A( k, k+1:j_end-1 ) is the trsm
            // A( k+1:i_end-1, k+1:j_end-1 ) is the gemm
            // Compared to getrf, i_end replaces A_mt, j_end replaces A_nt.
            // "end" in the usual C++ sense of element after the last element.
            int64_t i_end = std::min(k + klt + 1, A_mt);
            int64_t j_end = std::min(k + ku2t + 1, A_nt);

            // panel, high priority
            int priority_one = 1;
            #pragma omp task depend(inout:column[k]) priority(priority_one)
            {
                // factor A(k:mt-1, k)
                internal::getrf<Target::HostTask>(
                    A.sub(k, i_end-1, k, k), diag_len, ib,
                    pivots.at(k), max_panel_threads, priority_one);

                BcastList bcast_list_A;
                int tag_k = k;
                for (int64_t i = k; i < i_end; ++i) {
                    // send A(i, k) across row A(i, k+1:nt-1)
                    bcast_list_A.push_back({i, k, {A.sub(i, i, k+1, j_end-1)}});
                }
                A.template listBcast(bcast_list_A, layout, tag_k);

                // Root broadcasts the pivot to all ranks.
                // todo: Panel ranks send the pivots to the right.
                {
                    trace::Block trace_block("MPI_Bcast");

                    MPI_Bcast(pivots.at(k).data(),
                              sizeof(Pivot)*pivots.at(k).size(),
                              MPI_BYTE, A.tileRank(k, k), A.mpiComm());
                }
            }
            // update lookahead column(s), high priority
            for (int64_t j = k+1; j < k+1+lookahead && j < j_end; ++j) {
                #pragma omp task depend(in:column[k]) \
                                 depend(inout:column[j]) priority(priority_one)
                {
                    // swap rows in A(k:mt-1, j)
                    int priority_one = 1;
                    int tag_j = j;
                    internal::permuteRows<Target::HostTask>(
                        Direction::Forward, A.sub(k, i_end-1, j, j), pivots.at(k),
                        layout, priority_one, tag_j);

                    auto Akk = A.sub(k, k, k, k);
                    auto Tkk =
                        TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, Akk);

                    // solve A(k, k) A(k, j) = A(k, j)
                    internal::trsm<Target::HostTask>(
                        Side::Left,
                        scalar_t(1.0), std::move(Tkk),
                                       A.sub(k, k, j, j), priority_one);

                    // send A(k, j) across column A(k+1:mt-1, j)
                    A.tileBcast(k, j, A.sub(k+1, i_end-1, j, j), layout, tag_j);

                    // A(k+1:mt-1, j) -= A(k+1:mt-1, k) * A(k, j)
                    internal::gemm<Target::HostTask>(
                        scalar_t(-1.0), A.sub(k+1, i_end-1, k, k),
                                        A.sub(k, k, j, j),
                        scalar_t(1.0),  A.sub(k+1, i_end-1, j, j),
                        layout, priority_one);
                }
            }
            // Update trailing submatrix, normal priority.
            // Depends on the whole range k+1+lookahead : A_nt-1,
            // not just to j_end-1, as the dependencies daisy chain on A_nt-1.
            if (k+1+lookahead < j_end) {
                #pragma omp task depend(in:column[k]) \
                                 depend(inout:column[k+1+lookahead]) \
                                 depend(inout:column[A_nt-1])
                {
                    // swap rows in A(k:mt-1, kl+1:nt-1)
                    int priority_zero = 0;
                    int tag_kl1 = k+1+lookahead;
                    internal::permuteRows<Target::HostTask>(
                        Direction::Forward, A.sub(k, i_end-1, k+1+lookahead, j_end-1),
                        pivots.at(k), layout, priority_zero, tag_kl1);

                    auto Akk = A.sub(k, k, k, k);
                    auto Tkk =
                        TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, Akk);

                    // solve A(k, k) A(k, kl+1:nt-1) = A(k, kl+1:nt-1)
                    internal::trsm<Target::HostTask>(
                        Side::Left,
                        scalar_t(1.0), std::move(Tkk),
                                       A.sub(k, k, k+1+lookahead, j_end-1));

                    // send A(k, kl+1:j_end-1) across A(k+1:mt-1, kl+1:nt-1)
                    BcastList bcast_list_A;
                    for (int64_t j = k+1+lookahead; j < j_end; ++j) {
                        // send A(k, j) across column A(k+1:mt-1, j)
                        bcast_list_A.push_back({k, j, {A.sub(k+1, i_end-1, j, j)}});
                    }
                    A.template listBcast(bcast_list_A, layout, tag_kl1);

                    // A(k+1:mt-1, kl+1:nt-1) -= A(k+1:mt-1, k) * A(k, kl+1:nt-1)
                    internal::gemm<Target::HostTask>(
                        scalar_t(-1.0), A.sub(k+1, i_end-1, k, k),
                                        A.sub(k, k, k+1+lookahead, j_end-1),
                        scalar_t(1.0),  A.sub(k+1, i_end-1, k+1+lookahead, j_end-1),
                        layout);
                }
            }
        }

        #pragma omp taskwait
        A.tileUpdateAllOrigin();
    }
    // Band LU does NOT pivot to the left of the panel, since it would
    // introduce fill in the lower triangle. Instead, pivoting is done
    // during the solve (gbtrs).

    A.releaseWorkspace();

}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup gbsv_specialization
///
template <Target target, typename scalar_t>
void gbtrf(BandMatrix<scalar_t>& A, Pivots& pivots,
           Options const& opts)
{
    int64_t lookahead = 1;
    if (opts.count(Option::Lookahead) > 0) {
        lookahead = opts.at(Option::Lookahead).i_;
    }

    int64_t ib = 16;
    if (opts.count(Option::InnerBlocking) > 0) {
        ib = opts.at(Option::InnerBlocking).i_;
    }

    int64_t max_panel_threads = std::max(omp_get_max_threads()/2, 1);
    if (opts.count(Option::MaxPanelThreads) > 0) {
        max_panel_threads = opts.at(Option::MaxPanelThreads).i_;
    }

    internal::specialization::gbtrf(internal::TargetType<target>(),
                                    A, pivots,
                                    ib, max_panel_threads, lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel band LU factorization.
///
/// Computes an LU factorization of a general band m-by-n matrix $A$
/// using partial pivoting with row interchanges.
///
/// The factorization has the form
/// \[
///     A = L U
/// \]
/// where $L$ is a product of permutation and unit lower triangular matrices,
/// and $U$ is upper triangular.
///
/// This is the right-looking Level 3 BLAS version of the algorithm.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in,out] A
///     On entry, the band matrix $A$ to be factored.
///     Tiles outside the bandwidth do not need to exist.
///     For tiles that are partially outside the bandwidth,
///     data outside the bandwidth should be explicitly set to zero.
///     On exit, the factors $L$ and $U$ from the factorization $A = L U$;
///     the unit diagonal elements of $L$ are not stored.
///     The upper bandwidth is increased to accomodate fill-in of $U$.
///
/// @param[out] pivots
///     The pivot indices that define the permutations.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Lookahead:
///       Number of panels to overlap with matrix updates.
///       lookahead >= 0. Default 1.
///     - Option::InnerBlocking:
///       Inner blocking to use for panel. Default 16.
///     - Option::MaxPanelThreads:
///       Number of threads to use for panel. Default omp_get_max_threads()/2.
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// TODO: return value
/// @retval 0 successful exit
/// @retval >0 for return value = $i$, $U(i,i)$ is exactly zero. The
///         factorization has been completed, but the factor $U$ is exactly
///         singular, and division by zero will occur if it is used
///         to solve a system of equations.
///
/// @ingroup gbsv_computational
///
template <typename scalar_t>
void gbtrf(BandMatrix<scalar_t>& A, Pivots& pivots,
           Options const& opts)
{
    Target target = Target::HostTask;
    if (opts.count(Option::Target) > 0) {
        target = Target(opts.at(Option::Target).i_);
    }

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            gbtrf<Target::HostTask>(A, pivots, opts);
            break;
        case Target::HostNest:
            gbtrf<Target::HostNest>(A, pivots, opts);
            break;
        case Target::HostBatch:
            gbtrf<Target::HostBatch>(A, pivots, opts);
            break;
        case Target::Devices:
            gbtrf<Target::Devices>(A, pivots, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gbtrf<float>(
    BandMatrix<float>& A, Pivots& pivots,
    Options const& opts);

template
void gbtrf<double>(
    BandMatrix<double>& A, Pivots& pivots,
    Options const& opts);

template
void gbtrf< std::complex<float> >(
    BandMatrix< std::complex<float> >& A, Pivots& pivots,
    Options const& opts);

template
void gbtrf< std::complex<double> >(
    BandMatrix< std::complex<double> >& A, Pivots& pivots,
    Options const& opts);

} // namespace slate
