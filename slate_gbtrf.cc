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

#include "slate.hh"
#include "slate_Debug.hh"
#include "slate_BandMatrix.hh"
#include "slate_Tile_blas.hh"
#include "slate_TriangularMatrix.hh"
#include "slate_internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::gbtrf from internal::specialization::gbtrf
namespace internal {
namespace specialization {

///-----------------------------------------------------------------------------
/// \brief
/// Distributed parallel LU factorization.
/// Generic implementation for any target.
/// Panel and lookahead computed on host using Host OpenMP task.
template <Target target, typename scalar_t>
void gbtrf(slate::internal::TargetType<target>,
           BandMatrix<scalar_t>& A, Pivots& pivots,
           int64_t ib, int max_panel_threads, int64_t lookahead)
{
    // using real_t = blas::real_type<scalar_t>;
    ///using BcastList = typename BandMatrix<scalar_t>::BcastList;

    const int64_t A_nt = A.nt();
    const int64_t A_mt = A.mt();
    const int64_t min_mt_nt = std::min(A.mt(), A.nt());
    pivots.resize(min_mt_nt);

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
    for (int64_t i = 0; i < min_mt_nt; ++i) {
        for (int64_t j = i + 1 + kut; j < std::min(i + 1 + ku2t, A.nt()); ++j) {
            if (A.tileIsLocal(i, j)) {
                try {
                    // todo: device?
                    A.tileInsert(i, j);
                }
                catch (const std::exception& ex) {
                    printf( "i %lld, j %lld, ex %s\n", i, j, ex.what() );
                    // todo: pass?
                }
                auto T = A(i, j);
                lapack::laset(lapack::MatrixType::General, T.mb(), T.nb(),
                              0, 0, T.data(), T.stride());
            }
        }
    }

    /*
    #pragma omp parallel
    #pragma omp master
    for (int64_t k = 0; k < min_mt_nt; ++k) {

        // TODO: isn't there an assumption that tileMb(k) == tileNb(k),
        // i.e., that the diagonal tile is square? This min() implies that they
        // might not be.
        // Perhaps if A is not square, the last diagonal tile is not square.

        // Normally, diagonal tiles are square,
        // but the last diagonal of a non-square A may be non-square.
        const int64_t diag_len = std::min(A.tileMb(k), A.tileNb(k));
        pivots.at(k).resize(diag_len);

        // panel, high priority
        #pragma omp task depend(inout:column[k]) priority(1)
        {
            // factor A(k:mt-1, k)
            int priority_one = 1;
            internal::getrf<Target::HostTask>(
                A.sub(k, A_mt-1, k, k), diag_len, ib,
                pivots.at(k), max_panel_threads, priority_one);

            BcastList bcast_list_A;
            for (int64_t i = k; i < A_mt; ++i) {
                // send A(i, k) across row A(i, k+1:nt-1)
                bcast_list_A.push_back({i, k, {A.sub(i, i, k+1, A_nt-1)}});
            }
            A.template listBcast(bcast_list_A);

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
        for (int64_t j = k+1; j < k+1+lookahead && j < A_nt; ++j) {
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[j]) priority(1)
            {
                // swap rows in A(k:mt-1, j)
                int priority_one = 1;
                int tag_j = j;
                internal::swap<Target::HostTask>(
                    A.sub(k, A_mt-1, j, j), pivots.at(k), priority_one, tag_j);

                auto Akk = A.sub(k, k, k, k);
                auto Tkk =
                    TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, Akk);

                // solve A(k, k) A(k, j) = A(k, j)
                internal::trsm<Target::HostTask>(
                    Side::Left,
                    scalar_t(1.0), std::move(Tkk),
                                   A.sub(k, k, j, j), priority_one);

                // send A(k, j) across column A(k+1:mt-1, j)
                A.tileBcast(k, j, A.sub(k+1, A_mt-1, j, j), tag_j);

                // A(k+1:mt-1, j) -= A(k+1:mt-1, k) * A(k, j)
                internal::gemm<Target::HostTask>(
                    scalar_t(-1.0), A.sub(k+1, A_mt-1, k, k),
                                    A.sub(k, k, j, j),
                    scalar_t(1.0),  A.sub(k+1, A_mt-1, j, j), priority_one);
            }
        }
        // update trailing submatrix, normal priority
        if (k+1+lookahead < A_nt) {
            #pragma omp task depend(in:column[k]) \
                             depend(inout:column[k+1+lookahead]) \
                             depend(inout:column[A_nt-1])
            {
                // swap rows in A(k:mt-1, kl+1:nt-1)
                int priority_zero = 0;
                int tag_kl1 = k+1+lookahead;
                internal::swap<Target::HostTask>(
                    A.sub(k, A_mt-1, k+1+lookahead, A_nt-1), pivots.at(k),
                          priority_zero, tag_kl1);

                auto Akk = A.sub(k, k, k, k);
                auto Tkk =
                    TriangularMatrix<scalar_t>(Uplo::Lower, Diag::Unit, Akk);

                // solve A(k, k) A(k, kl+1:nt-1) = A(k, kl+1:nt-1)
                internal::trsm<Target::HostTask>(
                    Side::Left,
                    scalar_t(1.0), std::move(Tkk),
                                   A.sub(k, k, k+1+lookahead, A_nt-1));

                // send A(k, kl+1:A_nt-1) across A(k+1:mt-1, kl+1:nt-1)
                BcastList bcast_list_A;
                for (int64_t j = k+1+lookahead; j < A_nt; ++j) {
                    // send A(k, j) across column A(k+1:mt-1, j)
                    bcast_list_A.push_back({k, j, {A.sub(k+1, A_mt-1, j, j)}});
                }
                A.template listBcast(bcast_list_A, tag_kl1);

                // A(k+1:mt-1, kl+1:nt-1) -= A(k+1:mt-1, k) * A(k, kl+1:nt-1)
                internal::gemm<Target::HostTask>(
                    scalar_t(-1.0), A.sub(k+1, A_mt-1, k, k),
                                    A.sub(k, k, k+1+lookahead, A_nt-1),
                    scalar_t(1.0),  A.sub(k+1, A_mt-1, k+1+lookahead, A_nt-1));
            }
        }
    }
    */

    // Band LU does NOT pivot to the left of the panel, since it would
    // introduce fill in the lower triangle. Instead, pivoting is done
    // during the solve.

    // Debug::checkTilesLives(A);
    // Debug::printTilesLives(A);

    //A.clearWorkspace();

    // Debug::printTilesMaps(A);
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup gesv_comp
template <Target target, typename scalar_t>
void gbtrf(BandMatrix<scalar_t>& A, Pivots& pivots,
           const std::map<Option, Value>& opts)
{
    int64_t lookahead = 1;
    if (opts.count(Option::Lookahead) > 0) {
        lookahead = opts.at(Option::Lookahead).i_;
    }

    int64_t ib = 1;
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
/// Distributed parallel LU factorization.
///
template <typename scalar_t>
void gbtrf(BandMatrix<scalar_t>& A, Pivots& pivots,
           const std::map<Option, Value>& opts)
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
    const std::map<Option, Value>& opts);

template
void gbtrf<double>(
    BandMatrix<double>& A, Pivots& pivots,
    const std::map<Option, Value>& opts);

template
void gbtrf< std::complex<float> >(
    BandMatrix< std::complex<float> >& A, Pivots& pivots,
    const std::map<Option, Value>& opts);

template
void gbtrf< std::complex<double> >(
    BandMatrix< std::complex<double> >& A, Pivots& pivots,
    const std::map<Option, Value>& opts);

} // namespace slate
