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
#include "slate/Matrix.hh"
#include "internal/Tile_tpmlqt.hh"
#include "internal/internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::unmlq from internal::specialization::unmlq
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// Distributed parallel multiply by Q from LQ factorization.
/// Generic implementation for any target.
/// @ingroup gelqf_specialization
///
template <Target target, typename scalar_t>
void unmlq(
    slate::internal::TargetType<target>,
    Side side, Op op,
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& C)
{
    // trace::Block trace_block("unmlq");
    // const int priority_one = 1;
    using BcastList = typename Matrix<scalar_t>::BcastList;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    int64_t A_mt = A.mt();
    int64_t A_nt = A.nt();
    int64_t A_min_mtnt = std::min(A_mt, A_nt);
    int64_t C_mt = C.mt();
    int64_t C_nt = C.nt();

    if (target == Target::Devices) {
        C.allocateBatchArrays();
        C.reserveDeviceWorkspace();
    }

    assert(T.size() == 2);
    auto Tlocal  = T[0];
    auto Treduce = T[1];

    // LQ tracks dependencies by block-row.
    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > block_vector(A_mt);
    uint8_t* block = block_vector.data();

    #pragma omp parallel
    #pragma omp master
    {
        omp_set_nested(1);
        if (side == Side::Right) {

            // Reserve workspace
            auto W = C.emptyLike();

            if (target == Target::Devices) {
                W.allocateBatchArrays();
                // todo: this is demanding too much device workspace memory
                // only one tile-col of matrix W per MPI process is going to be used,
                // but W with size of whole C is being allocated
                // thus limiting the matrix size that can be processed
                W.reserveDeviceWorkspace();
            }

            if (op == Op::NoTrans) {
                // todo: NoTrans and (Conj)Trans are very similar codes,
                // just swapping order of internal unmlq and ttmlq.
                // LAPACK uses one loop with variable bounds.

                //----------------------------------------
                // Right; NoTrans: multiply C Q = C Q_K ... Q_1.
                // i.e., in reverse order of how Q_k's were created.

                // for k = A_mt-1, lastk = A_mt-1 (no previous row to depend on);
                // for k < A_mt,   lastk = k + 1.
                int64_t lastk = A_min_mtnt-1;
                // OpenMP uses lastk; compiler doesn't, so warns it is unused.
                SLATE_UNUSED(lastk);
                for (int64_t k = A_min_mtnt-1; k >= 0; --k) {

                    auto A_panel = A.sub(k, k, k, A_nt-1);

                    // Find ranks in this row.
                    std::set<int> ranks_set;
                    A_panel.getRanks(&ranks_set);
                    assert(ranks_set.size() > 0);

                    // Find each rank's first (left-most) col in this panel,
                    // where the triangular tile resulting from local gelqf
                    // panel will reside.
                    std::vector< int64_t > first_indices;
                    first_indices.reserve(ranks_set.size());
                    for (int r: ranks_set) {
                        for (int64_t j = 0; j < A_panel.nt(); ++j) {
                            if (A_panel.tileRank(0, j) == r) {
                                first_indices.push_back(j+k);
                                break;
                            }
                        }
                    }

                    #pragma omp task depend(inout:block[k]) \
                                     depend(in:block[lastk])
                    {
                        // bcast V across col of C
                        BcastList bcast_list_V_top;
                        BcastList bcast_list_V;
                        for (int64_t j = k; j < A_nt; ++j) {
                            // send A(k, j) across col C(0:mt-1, j)
                            // Vs in first_indices (except the left-most one) need three lives
                            if (std::find(first_indices.begin(), first_indices.end(), j) != first_indices.end() && j > k)
                                bcast_list_V_top.push_back({k, j, {C.sub(0, C_mt-1, j, j)}});
                            else
                                bcast_list_V.push_back({k, j, {C.sub(0, C_mt-1, j, j)}});
                        }
                        A.template listBcast(bcast_list_V_top, layout, 0, 3);
                        A.template listBcast(bcast_list_V, layout, 0, 2);

                        // bcast Tlocal across col of C
                        if (first_indices.size() > 0) {
                            BcastList bcast_list_T;
                            for (int64_t j : first_indices) {
                                bcast_list_T.push_back({k, j, {C.sub(0, C_mt-1, j, j)}});
                            }
                            Tlocal.template listBcast(bcast_list_T, layout);
                        }

                        // bcast Treduce across col of C
                        if (first_indices.size() > 1) {
                            BcastList bcast_list_T;
                            for (int64_t j : first_indices) {
                                if (j > k) // exclude the first col of this panel that has no Treduce tile
                                    bcast_list_T.push_back({k, j, {C.sub(0, C_mt-1, j, j)}});
                            }
                            Treduce.template listBcast(bcast_list_T, layout);
                        }

                        // For NoTrans,
                        //     Q   = (Qk_reduce Qk_local) ... (Q1_reduce Q1_local).
                        //     i.e., ttmlq, then unmlq.
                        // For (Conj)Trans,
                        //     Q^H = (Q1_local^H Q1_reduce^H) ... (Qk_local^H Qk_reduce^H).
                        //     i.e., unmlq, then ttmlq.
                        /// if (op == Op::NoTrans) {
                        // Apply triangle-triangle reduction reflectors
                        internal::ttmlq<Target::HostTask>(
                                        side, op,
                                        std::move(A_panel),
                                        Treduce.sub(k, k, k, A_nt-1),
                                        C.sub(0, C_mt-1, k, C_nt-1));
                        /// }

                        // Apply local reflectors
                        internal::unmlq<target>(
                                        side, op,
                                        std::move(A_panel),
                                        Tlocal.sub(k, k, k, A_nt-1),
                                        C.sub(0, C_mt-1, k, C_nt-1),
                                        W.sub(0, C_mt-1, k, C_nt-1));

                        /// if (op != Op::NoTrans) {
                        /// // Apply triangle-triangle reduction reflectors
                        /// internal::ttmlq<Target::HostTask>(
                        ///                 side, op,
                        ///                 std::move(A_panel),
                        ///                 Treduce.sub(k, k, k, A_nt-1),
                        ///                 C.sub(0, C_mt-1, k, C_nt-1));
                        /// }
                    }
                    lastk = k;
                }
            }
            else {
                //----------------------------------------
                // Right; (Conj)Trans: multiply C Q^H = C Q_1^H ... Q_K^H,
                // i.e., in same order as Q_k's were created.

                // for k = 0, lastk = 0 (no previous row to depend on);
                // for k > 0, lastk = k - 1.
                int64_t lastk = 0;
                // OpenMP uses lastk; compiler doesn't, so warns it is unused.
                SLATE_UNUSED(lastk);
                for (int64_t k = 0; k < A_min_mtnt; ++k) {

                    auto A_panel = A.sub(k, k, k, A_nt-1);

                    // Find ranks in this row.
                    std::set<int> ranks_set;
                    A_panel.getRanks(&ranks_set);
                    assert(ranks_set.size() > 0);

                    // Find each rank's first (left-most) col in this panel,
                    // where the triangular tile resulting from local gelqf
                    // panel will reside.
                    std::vector< int64_t > first_indices;
                    first_indices.reserve(ranks_set.size());
                    for (int r: ranks_set) {
                        for (int64_t j = 0; j < A_panel.nt(); ++j) {
                            if (A_panel.tileRank(0, j) == r) {
                                first_indices.push_back(j+k);
                                break;
                            }
                        }
                    }

                    #pragma omp task depend(inout:block[k]) \
                                     depend(in:block[lastk])
                    {
                        // bcast V across col of C
                        BcastList bcast_list_V_top;
                        BcastList bcast_list_V;
                        for (int64_t j = k; j < A_nt; ++j) {
                            // send A(k, j) across col C(0:mt-1, j)
                            // Vs in first_indices (except the left-most one) need three lives
                            if (std::find(first_indices.begin(), first_indices.end(), j) != first_indices.end() && j > k)
                                bcast_list_V_top.push_back({k, j, {C.sub(0, C_mt-1, j, j)}});
                            else
                                bcast_list_V.push_back({k, j, {C.sub(0, C_mt-1, j, j)}});
                        }
                        A.template listBcast(bcast_list_V_top, layout, 0, 3);
                        A.template listBcast(bcast_list_V, layout, 0, 2);

                        // bcast Tlocal across col of C
                        BcastList bcast_list_T;
                        for (int64_t j : first_indices) {
                            bcast_list_T.push_back({k, j, {C.sub(0, C_mt-1, j, j)}});
                        }
                        Tlocal.template listBcast(bcast_list_T, layout);

                        // bcast Treduce across col of C
                        if (first_indices.size() > 1) {
                            BcastList bcast_list_T;
                            for (int64_t j : first_indices) {
                                if (j > k) // exclude the first col of this panel that has no Treduce tile
                                    bcast_list_T.push_back({k, j, {C.sub(0, C_mt-1, j, j)}});
                            }
                            Treduce.template listBcast(bcast_list_T, layout);
                        }

                        // Apply local reflectors
                        internal::unmlq<target>(
                                        side, op,
                                        std::move(A_panel),
                                        Tlocal.sub(k, k, k, A_nt-1),
                                        C.sub(0, C_mt-1, k, C_nt-1),
                                        W.sub(0, C_mt-1, k, C_nt-1));

                        // Apply triangle-triangle reduction reflectors
                        internal::ttmlq<Target::HostTask>(
                                        side, op,
                                        std::move(A_panel),
                                        Treduce.sub(k, k, k, A_nt-1),
                                        C.sub(0, C_mt-1, k, C_nt-1));
                    }
                    lastk = k;
                }
            }
        }
        else {
            // TODO: side == Side::Left
            slate_not_implemented("Side::Left");
        }
    }
    C.tileUpdateAllOrigin();
    C.clearWorkspace();
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup gelqf_specialization
///
template <Target target, typename scalar_t>
void unmlq(
    Side side, Op op,
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& C,
    const std::map<Option, Value>& opts)
{
    internal::specialization::unmlq(internal::TargetType<target>(),
                                    side, op, A, T, C);
}

//------------------------------------------------------------------------------
/// Distributed parallel multiply by $Q$ from LQ factorization.
///
/// Multiplies the general m-by-n matrix $C$ by $Q$ from LQ factorization,
/// according to:
///
/// op              |  side = Left  |  side = Right
/// --------------- | ------------- | --------------
/// op = NoTrans    |  $Q C  $      |  $C Q  $
/// op = ConjTrans  |  $Q^H C$      |  $C Q^H$
///
/// where $Q$ is a unitary matrix defined as the product of k
/// elementary reflectors
/// \[
///     Q = H(1) H(2) . . . H(k)
/// \]
/// as returned by gelqf. $Q$ is of order m if side = Left
/// and of order n if side = Right.
///
//------------------------------------------------------------------------------
/// @tparam scalar_t
///     One of float, double, std::complex<float>, std::complex<double>.
//------------------------------------------------------------------------------
/// @param[in] side
///     - Side::Left:  apply $Q$ or $Q^H$ from the left;
///     - Side::Right: apply $Q$ or $Q^H$ from the right.
///
/// @param[in] op
///     - Op::NoTrans    apply $Q$;
///     - Op::ConjTrans: apply $Q^H$;
///     - Op::Trans:     apply $Q^T$ (only if real).
///       In the real case, Op::Trans is equivalent to Op::ConjTrans.
///       In the complex case, Op::Trans is not allowed.
///
/// @param[in] A
///     Details of the LQ factorization of the original matrix $A$ as returned
///     by gelqf.
///
/// @param[in] T
///     Triangular matrices of the block reflectors as returned by gelqf.
///
/// @param[in,out] C
///     On entry, the m-by-n matrix $C$.
///     On exit, $C$ is overwritten by $Q C$, $Q^H C$, $C Q$, or $C Q^H$.
///
/// @param[in] opts
///     Additional options, as map of name = value pairs. Possible options:
///     - Option::Target:
///       Implementation to target. Possible values:
///       - HostTask:  OpenMP tasks on CPU host [default].
///       - HostNest:  nested OpenMP parallel for loop on CPU host.
///       - HostBatch: batched BLAS on CPU host.
///       - Devices:   batched BLAS on GPU device.
///
/// @ingroup gelqf_computational
///
template <typename scalar_t>
void unmlq(
    Side side, Op op,
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& C,
    const std::map<Option, Value>& opts)
{
    Target target;
    try {
        target = Target(opts.at(Option::Target).i_);
    }
    catch (std::out_of_range&) {
        target = Target::HostTask;
    }

    switch (target) {
        case Target::Host:
        case Target::HostTask:
            unmlq<Target::HostTask>(side, op, A, T, C, opts);
            break;
        case Target::HostNest:
            unmlq<Target::HostNest>(side, op, A, T, C, opts);
            break;
        case Target::HostBatch:
            unmlq<Target::HostBatch>(side, op, A, T, C, opts);
            break;
        case Target::Devices:
            unmlq<Target::Devices>(side, op, A, T, C, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void unmlq<float>(
    Side side, Op op,
    Matrix<float>& A,
    TriangularFactors<float>& T,
    Matrix<float>& C,
    const std::map<Option, Value>& opts);

template
void unmlq<double>(
    Side side, Op op,
    Matrix<double>& A,
    TriangularFactors<double>& T,
    Matrix<double>& C,
    const std::map<Option, Value>& opts);

template
void unmlq< std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >& A,
    TriangularFactors< std::complex<float> >& T,
    Matrix< std::complex<float> >& C,
    const std::map<Option, Value>& opts);

template
void unmlq< std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >& A,
    TriangularFactors< std::complex<double> >& T,
    Matrix< std::complex<double> >& C,
    const std::map<Option, Value>& opts);

} // namespace slate
