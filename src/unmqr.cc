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
#include "internal/Tile_tpmqrt.hh"
#include "internal/internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::unmqr from internal::specialization::unmqr
namespace internal {
namespace specialization {

//------------------------------------------------------------------------------
/// Distributed parallel multiply by Q from QR factorization.
/// Generic implementation for any target.
/// @ingroup geqrf_specialization
///
template <Target target, typename scalar_t>
void unmqr(
    slate::internal::TargetType<target>,
    Side side, Op op,
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& C)
{
    // trace::Block trace_block("unmqr");
    // const int priority_one = 1;
    using BcastList = typename Matrix<scalar_t>::BcastList;

    // Assumes column major
    const Layout layout = Layout::ColMajor;

    int64_t A_mt = A.mt();
    int64_t A_nt = A.nt();
    int64_t A_min_mtnt = std::min(A_mt, A_nt);
    int64_t C_mt = C.mt();
    int64_t C_nt = C.nt();

    if (is_complex<scalar_t>::value && op == Op::Trans) {
        throw Exception("Complex numbers uses Op::ConjTrans, not Op::Trans.");
    }

    if (target == Target::Devices) {
        C.allocateBatchArrays();
        C.reserveDeviceWorkspace();
    }

    // Reserve workspace
    auto W = C.emptyLike();

    if (target == Target::Devices) {
        W.allocateBatchArrays();
        // todo: this is demanding too much device workspace memory
        // only one tile-row of matrix W per MPI process is going to be used,
        // but W with size of whole C is being allocated
        // thus limiting the matrix size that can be processed
        //W.reserveDeviceWorkspace();
    }

    assert(T.size() == 2);
    auto Tlocal  = T[0];
    auto Treduce = T[1];

    // QR tracks dependencies by block-column.
    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > block_vector(A_nt);
    uint8_t* block = block_vector.data();

    #pragma omp parallel
    #pragma omp master
    {
        omp_set_nested(1);

        int64_t k_begin, k_end, k_step;
        if ((side == Side::Left) == (op == Op::NoTrans)) {
            // Left,  NoTrans:     multiply Q C   = Q1 ... QK C, or
            // Right, (Conj)Trans: multiply C Q^H = C QK^H ... Q1^H,
            // i.e., in reverse order of how Qk's were created.
            k_begin = A_min_mtnt-1;
            k_end   = -1;
            k_step  = -1;
        }
        else {
            // Left, (Conj)Trans: multiply Q^H C = QK^H ... Q1^H C, or
            // Right, NoTrans:    multiply C Q   = C Q1 ... QK,
            // i.e., in same order as Qk's were created.
            k_begin = 0;
            k_end   = A_min_mtnt;
            k_step  = 1;
        }

        // for k = k_begin, lastk = k_begin (no previous col to depend on);
        // otherwise,       lastk = k - k_step (previous col).
        int64_t lastk = k_begin;
        // OpenMP uses lastk; compiler doesn't, so warns it is unused.
        SLATE_UNUSED(lastk);
        for (int64_t k = k_begin; k != k_end; k += k_step) {

            auto A_panel = A.sub(k, A_mt-1, k, k);

            // Find ranks in this column.
            std::set<int> ranks_set;
            A_panel.getRanks(&ranks_set);
            assert(ranks_set.size() > 0);

            // Find each rank's first (top-most) row in this panel,
            // where the triangular tile resulting from local geqrf
            // panel will reside.
            std::vector< int64_t > first_indices;
            first_indices.reserve(ranks_set.size());
            for (int r: ranks_set) {
                for (int64_t i = 0; i < A_panel.mt(); ++i) {
                    if (A_panel.tileRank(i, 0) == r) {
                        first_indices.push_back(i+k);
                        break;
                    }
                }
            }

            #pragma omp task depend(inout:block[k]) \
                             depend(in:block[lastk])
            {
                // Indices for row or col of C.
                int64_t i0 = -1, i1 = -1, j0 = -1, j1 = -1;
                if (side == Side::Left) {
                    j0 = 0;
                    j1 = C_nt-1;
                }
                else {
                    i0 = 0;
                    i1 = C_mt-1;
                }

                // Send V(i) across row C(i, 0:nt-1) or col C(0:mt-1, i),
                // for side = left or right, respectively.
                BcastList bcast_list_V_top;
                BcastList bcast_list_V;
                for (int64_t i = k; i < A_mt; ++i) {
                    if (side == Side::Left) {
                        i0 = i;
                        i1 = i;
                    }
                    else {
                        j0 = i;
                        j1 = i;
                    }
                    // Vs in first_indices (except the left-most one)
                    // need three lives.
                    if (i > k && std::find(first_indices.begin(), first_indices.end(), i) != first_indices.end()) {
                        bcast_list_V_top.push_back(
                            {i, k, {C.sub(i0, i1, j0, j1)}});
                    }
                    else {
                        bcast_list_V.push_back(
                            {i, k, {C.sub(i0, i1, j0, j1)}});
                    }
                }
                A.template listBcast(bcast_list_V_top, layout, 0, 3);
                A.template listBcast(bcast_list_V, layout, 0, 2);

                // Send Tlocal(i) across row C(i, 0:nt-1) or col C(0:mt-1, i).
                if (first_indices.size() > 0) {
                    BcastList bcast_list_T;
                    for (int64_t i : first_indices) {
                        if (side == Side::Left) {
                            i0 = i;
                            i1 = i;
                        }
                        else {
                            j0 = i;
                            j1 = i;
                        }
                        bcast_list_T.push_back(
                            {i, k, {C.sub(i0, i1, j0, j1)}});
                    }
                    Tlocal.template listBcast(bcast_list_T, layout);
                }

                // Send Treduce(i) across row C(i, 0:nt-1) or col C(0:mt-1, i).
                if (first_indices.size() > 1) {
                    BcastList bcast_list_T;
                    for (int64_t i : first_indices) {
                        if (side == Side::Left) {
                            i0 = i;
                            i1 = i;
                        }
                        else {
                            j0 = i;
                            j1 = i;
                        }
                        // Exclude first row of this panel,
                        // which doesn't have Treduce tile.
                        if (i > k) {
                            bcast_list_T.push_back(
                                {i, k, {C.sub(i0, i1, j0, j1)}});
                        }
                    }
                    Treduce.template listBcast(bcast_list_T, layout);
                }

                Matrix<scalar_t> C_trail, W_trail;
                if (side == Side::Left) {
                    C_trail = C.sub(k, C_mt-1, 0, C_nt-1);
                    W_trail = W.sub(k, C_mt-1, 0, C_nt-1);
                }
                else {
                    C_trail = C.sub(0, C_mt-1, k, C_nt-1);
                    W_trail = W.sub(0, C_mt-1, k, C_nt-1);
                }

                // Left,  NoTrans:     Qi C   = Qi_local Qi_reduce C, or
                // Right, (Conj)Trans: C Qi^H = C Qi_reduce^H Qi_local^H,
                // do ttmqr then unmqr.
                if ((side == Side::Left) == (op == Op::NoTrans)) {
                    // Apply triangle-triangle reduction reflectors.
                    internal::ttmqr<Target::HostTask>(
                                    side, op,
                                    std::move(A_panel),
                                    Treduce.sub(k, A_mt-1, k, k),
                                    std::move(C_trail));
                }

                // Apply local reflectors.
                internal::unmqr<target>(
                                side, op,
                                std::move(A_panel),
                                Tlocal.sub(k, A_mt-1, k, k),
                                std::move(C_trail),
                                std::move(W_trail));

                // Left,  (Conj)Trans: Qi^H C = Qi_reduce^H Qi_local^H C, or
                // Right, NoTrans:     C Qi   = C Qi_local Qi_reduce,
                // do unmqr then ttmqr.
                if ((side == Side::Left) != (op == Op::NoTrans)) {
                    // Apply triangle-triangle reduction reflectors.
                    internal::ttmqr<Target::HostTask>(
                                    side, op,
                                    std::move(A_panel),
                                    Treduce.sub(k, A_mt-1, k, k),
                                    std::move(C_trail));
                }
            }

            lastk = k;
        }

        #pragma omp taskwait
        C.tileUpdateAllOrigin();
    }
    C.clearWorkspace();
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup geqrf_specialization
///
template <Target target, typename scalar_t>
void unmqr(
    Side side, Op op,
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& C,
    Options const& opts)
{
    internal::specialization::unmqr(internal::TargetType<target>(),
                                    side, op, A, T, C);
}

//------------------------------------------------------------------------------
/// Distributed parallel multiply by $Q$ from QR factorization.
///
/// Multiplies the general m-by-n matrix $C$ by $Q$ from QR factorization,
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
/// as returned by geqrf. $Q$ is of order m if side = Left
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
///     Details of the QR factorization of the original matrix $A$ as returned
///     by geqrf.
///
/// @param[in] T
///     Triangular matrices of the block reflectors as returned by geqrf.
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
/// @ingroup geqrf_computational
///
template <typename scalar_t>
void unmqr(
    Side side, Op op,
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& C,
    Options const& opts)
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
            unmqr<Target::HostTask>(side, op, A, T, C, opts);
            break;
        case Target::HostNest:
            unmqr<Target::HostNest>(side, op, A, T, C, opts);
            break;
        case Target::HostBatch:
            unmqr<Target::HostBatch>(side, op, A, T, C, opts);
            break;
        case Target::Devices:
            unmqr<Target::Devices>(side, op, A, T, C, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void unmqr<float>(
    Side side, Op op,
    Matrix<float>& A,
    TriangularFactors<float>& T,
    Matrix<float>& C,
    Options const& opts);

template
void unmqr<double>(
    Side side, Op op,
    Matrix<double>& A,
    TriangularFactors<double>& T,
    Matrix<double>& C,
    Options const& opts);

template
void unmqr< std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >& A,
    TriangularFactors< std::complex<float> >& T,
    Matrix< std::complex<float> >& C,
    Options const& opts);

template
void unmqr< std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >& A,
    TriangularFactors< std::complex<double> >& T,
    Matrix< std::complex<double> >& C,
    Options const& opts);

} // namespace slate
