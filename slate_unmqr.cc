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
#include "slate_Matrix.hh"
#include "slate_Tile_tpmqrt.hh"
#include "slate_internal.hh"
#include "slate_internal_util.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::unmqr from internal::specialization::unmqr
namespace internal {
namespace specialization {

///-----------------------------------------------------------------------------
/// \brief
/// Distributed parallel multiply by Q from QR factorization.
/// Generic implementation for any target.
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

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > column_vector(A_nt);
    uint8_t* column = column_vector.data();

    #pragma omp parallel
    #pragma omp master
    {
        omp_set_nested(1);
        if (side == Side::Left) {

            // Reserve workspace
            auto W = C.emptyLike();

            if (target == Target::Devices) {
                W.allocateBatchArrays();
                W.reserveDeviceWorkspace();
            }

            if (op == Op::NoTrans) {
                // NoTrans: multiply by Q = Q_1 ... Q_K,
                // i.e., in reverse order of how Q_k's were created.

                // for k = A_nt-1, lastk = A_nt-1 (no previous column to depend on);
                // for k < A_nt,   lastk = k + 1.
                int64_t lastk = A_min_mtnt-1;
                for (int64_t k = A_min_mtnt-1; k >= 0; --k) {

                    auto A_panel = A.sub(k, A_mt-1, k, k);

                    // Find ranks in this column.
                    std::set<int> ranks_set;
                    A_panel.getRanks(&ranks_set);

                    assert(ranks_set.size() > 0);

                    // Find each rank's top-most row in this panel,
                    // where the triangular tile resulting from local geqrf panel will reside.
                    std::vector< int64_t > top_rows;
                    top_rows.reserve(ranks_set.size());
                    for (int r: ranks_set) {
                        for (int64_t i = 0; i < A_panel.mt(); ++i) {
                            if (A_panel.tileRank(i, 0) == r) {
                                top_rows.push_back(i+k);
                                break;
                            }
                        }
                    }
                    int64_t min_row = k;

                    #pragma omp task depend(inout:column[k]) \
                                     depend(in:column[lastk])
                    {

                        // bcast V across row of C
                        BcastList bcast_list_V_top;
                        BcastList bcast_list_V;
                        for (int64_t i = k; i < A_mt; ++i) {
                            // send A(i, k) across row C(i, 0:nt-1)
                            if (std::find(top_rows.begin(), top_rows.end(), i) != top_rows.end() && i > min_row)
                                bcast_list_V_top.push_back({i, k, {C.sub(i, i, 0, C_nt-1)}});
                            else
                                bcast_list_V.push_back({i, k, {C.sub(i, i, 0, C_nt-1)}});
                        }
                        A.template listBcast(bcast_list_V_top, 0, Layout::ColMajor, 3);// TODO is column major safe?
                        A.template listBcast(bcast_list_V, 0, Layout::ColMajor, 2);

                        // bcast Tlocal across row of C
                        if (top_rows.size() > 0){
                            BcastList bcast_list_T;
                            for (auto it = top_rows.begin(); it < top_rows.end(); ++it){
                                int64_t row = *it;
                                bcast_list_T.push_back({row, k, {C.sub(row, row, 0, C_nt-1)}});
                            }
                            Tlocal.template listBcast(bcast_list_T);
                        }

                        // bcast Treduce across row of C
                        if (top_rows.size() > 1){
                            BcastList bcast_list_T;
                            for (auto it = top_rows.begin(); it < top_rows.end(); ++it){
                                int64_t row = *it;
                                if(row > min_row)//exclude the first row of this panel that has no Treduce tile
                                    bcast_list_T.push_back({row, k, {C.sub(row, row, 0, C_nt-1)}});
                            }
                            Treduce.template listBcast(bcast_list_T);
                        }

                        if (target == Target::Devices){
                            for (auto it = top_rows.begin(); it < top_rows.end(); ++it){
                                int64_t row = *it;
                                C.sub(row, row, 0, C_nt-1).moveAllToHost();
                            }
                        }

                        // Apply triangle-triangle reduction reflectors
                        internal::ttmqr<Target::HostTask>(
                                        side, op,
                                        std::move(A_panel),
                                        Treduce.sub(k, A_mt-1, k, k),
                                        C.sub(k, C_mt-1, 0, C_nt-1));

                        // Apply local reflectors
                        internal::unmqr<target>(
                                        side, op,
                                        std::move(A_panel),
                                        Tlocal.sub(k, A_mt-1, k, k),
                                        C.sub(k, C_mt-1, 0, C_nt-1),
                                        W.sub(k, C_mt-1, 0, C_nt-1));
                    }

                    lastk = k;
                }
            }
            else {
                // Trans or ConjTrans: multiply by Q^H = Q_K^H ... Q_1^H.
                // i.e., in same order as Q_k's were created.

                // for k = 0, lastk = 0 (no previous column to depend on);
                // for k > 0, lastk = k - 1.
                int64_t lastk = 0;
                for (int64_t k = 0; k < A_min_mtnt; ++k) {

                    auto A_panel = A.sub(k, A_mt-1, k, k);

                    // Find ranks in this column.
                    std::set<int> ranks_set;
                    A_panel.getRanks(&ranks_set);

                    assert(ranks_set.size() > 0);

                    // Find each rank's top-most row in this panel,
                    // where the triangular tile resulting from local geqrf panel will reside.
                    std::vector< int64_t > top_rows;
                    top_rows.reserve(ranks_set.size());
                    for (int r: ranks_set) {
                        for (int64_t i = 0; i < A_panel.mt(); ++i) {
                            if (A_panel.tileRank(i, 0) == r) {
                                top_rows.push_back(i+k);
                                break;
                            }
                        }
                    }
                    int64_t min_row = k;

                    #pragma omp task depend(inout:column[k]) \
                                     depend(in:column[lastk])
                    {

                        // bcast V across row of C
                        BcastList bcast_list_V_top;
                        BcastList bcast_list_V;
                        for (int64_t i = k; i < A_mt; ++i) {
                            // send A(i, k) across row C(i, 0:nt-1)
                            // top_rows' V's (except the top-most one) need three lives
                            if (std::find(top_rows.begin(), top_rows.end(), i) != top_rows.end() && i > min_row)
                                bcast_list_V_top.push_back({i, k, {C.sub(i, i, 0, C_nt-1)}});
                            else
                                bcast_list_V.push_back({i, k, {C.sub(i, i, 0, C_nt-1)}});
                        }
                        A.template listBcast(bcast_list_V_top, 0, Layout::ColMajor, 3);// TODO is column major safe?
                        A.template listBcast(bcast_list_V, 0, Layout::ColMajor, 2);

                        // bcast Tlocal across row of C
                        BcastList bcast_list_T;
                        for (auto it = top_rows.begin(); it < top_rows.end(); ++it){
                            int64_t row = *it;
                            bcast_list_T.push_back({row, k, {C.sub(row, row, 0, C_nt-1)}});
                        }
                        Tlocal.template listBcast(bcast_list_T);

                        // bcast Treduce across row of C
                        if (top_rows.size() > 1){
                            BcastList bcast_list_T;
                            for (auto it = top_rows.begin(); it < top_rows.end(); ++it){
                                int64_t row = *it;
                                if(row > min_row)//exclude the first row of this panel that has no Treduce tile
                                    bcast_list_T.push_back({row, k, {C.sub(row, row, 0, C_nt-1)}});
                            }
                            Treduce.template listBcast(bcast_list_T);
                        }

                        // Apply local reflectors
                        internal::unmqr<target>(
                                        side, op,
                                        std::move(A_panel),
                                        Tlocal.sub(k, A_mt-1, k, k),
                                        C.sub(k, C_mt-1, 0, C_nt-1),
                                        W.sub(k, C_mt-1, 0, C_nt-1));

                        // Apply triangle-triangle reduction reflectors
                        internal::ttmqr<Target::HostTask>(
                                        side, op,
                                        std::move(A_panel),
                                        Treduce.sub(k, A_mt-1, k, k),
                                        C.sub(k, C_mt-1, 0, C_nt-1));
                    }
                    lastk = k;
                }
            }
        }
        else {
            // TODO: side == Side::Right
        }
    }
    C.moveAllToHost();
    C.clearWorkspace();
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup gesv_comp
template <Target target, typename scalar_t>
void unmqr(
    Side side, Op op,
    Matrix<scalar_t>& A,
    TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& C,
    const std::map<Option, Value>& opts)
{
    internal::specialization::unmqr(internal::TargetType<target>(),
                                    side, op, A, T, C);
}

//------------------------------------------------------------------------------
/// Distributed parallel multiply by Q from QR factorization.
///
template <typename scalar_t>
void unmqr(
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
    catch (std::out_of_range) {
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
    const std::map<Option, Value>& opts);

template
void unmqr<double>(
    Side side, Op op,
    Matrix<double>& A,
    TriangularFactors<double>& T,
    Matrix<double>& C,
    const std::map<Option, Value>& opts);

template
void unmqr< std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >& A,
    TriangularFactors< std::complex<float> >& T,
    Matrix< std::complex<float> >& C,
    const std::map<Option, Value>& opts);

template
void unmqr< std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >& A,
    TriangularFactors< std::complex<double> >& T,
    Matrix< std::complex<double> >& C,
    const std::map<Option, Value>& opts);

} // namespace slate
