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
    Matrix<scalar_t>& T,
    Matrix<scalar_t>& C,
    int64_t ib, int64_t lookahead)
{
    const int priority_one = 1;

    int64_t A_mt = A.mt();
    int64_t A_nt = A.nt();

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > column_vector(A_nt);
    uint8_t* column = column_vector.data();

    #pragma omp parallel
    #pragma omp master
    {
        if (side == Side::Left) {
            if (op == Op::NoTrans) {
                // NoTrans: multiply by Q = Q_1 ... Q_K,
                // i.e., in reverse order of how Q_k's were created.

                // for k = A_nt-1, lastk = A_nt-1 (no previous column to depend on);
                // for k < A_nt,   lastk = k + 1.
                int64_t lastk = A_nt-1;
                for (int64_t k = A_nt-1; k >= 0; --k) {
                    const int64_t diag_len = std::min(A.tileMb(k), A.tileNb(k));

                    #pragma omp task depend(inout:column[k]) \
                                     depend(in:column[lastk])
                    {
                        // Apply triangle-triangle reduction reflectors
                        internal::ttmqr(side, op,
                                        A.sub(k, A_mt-1, k, k),
                                        T.sub(k, A_mt-1, k, k),
                                        C.sub(k, A_mt-1, 0, C.nt()-1));

                        // Apply local reflectors
                        // TODO unmqr
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
                for (int64_t k = 0; k < A_nt; ++k) {
                    const int64_t diag_len = std::min(A.tileMb(k), A.tileNb(k));

                    // Apply local reflectors
                    // TODO unmqr

                    // Apply triangle-triangle reduction reflectors
                    // TODO ttmqr

                    lastk = k;
                    assert(lastk == k);  // TMP to silence unused variable warnings
                }
            }
        }
        else {
            // TODO: side == Side::Right
        }
    }
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
    Matrix<scalar_t>& T,
    Matrix<scalar_t>& C,
    const std::map<Option, Value>& opts)
{
    int64_t ib = 16;
    if (opts.count(Option::InnerBlocking) > 0)
        ib = opts.at(Option::InnerBlocking).i_;
    assert(ib >= 0);

    int64_t lookahead = 1;
    if (opts.count(Option::Lookahead) > 0)
        lookahead = opts.at(Option::Lookahead).i_;
    assert(lookahead >= 0);

    internal::specialization::unmqr(internal::TargetType<target>(),
                                    side, op, A, T, C,
                                    ib, lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel multiply by Q from QR factorization.
///
template <typename scalar_t>
void unmqr(
    Side side, Op op,
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& T,
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
    Matrix<float>& T,
    Matrix<float>& C,
    const std::map<Option, Value>& opts);

template
void unmqr<double>(
    Side side, Op op,
    Matrix<double>& A,
    Matrix<double>& T,
    Matrix<double>& C,
    const std::map<Option, Value>& opts);

template
void unmqr< std::complex<float> >(
    Side side, Op op,
    Matrix< std::complex<float> >& A,
    Matrix< std::complex<float> >& T,
    Matrix< std::complex<float> >& C,
    const std::map<Option, Value>& opts);

template
void unmqr< std::complex<double> >(
    Side side, Op op,
    Matrix< std::complex<double> >& A,
    Matrix< std::complex<double> >& T,
    Matrix< std::complex<double> >& C,
    const std::map<Option, Value>& opts);

} // namespace slate
