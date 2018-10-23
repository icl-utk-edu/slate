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
#include "slate_internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::geqrf from internal::specialization::geqrf
namespace internal {
namespace specialization {

///-----------------------------------------------------------------------------
/// Distributed parallel QR factorization.
/// Generic implementation for any target.
/// Panel and lookahead computed on host using Host OpenMP task.
template <Target target, typename scalar_t>
void geqrf(slate::internal::TargetType<target>,
           Matrix<scalar_t>& A,
           TriangularFactors<scalar_t>& T,
           int64_t ib, int max_panel_threads, int64_t lookahead)
{
    using blas::real;

    const int priority_one = 1;

    int64_t A_mt = A.mt();
    int64_t A_nt = A.nt();

    T.clear();
    T.push_back(A.emptyLike(A));
    T.push_back(A.emptyLike(A));
    auto Tlocal  = T[0];
    auto Treduce = T[1];

    // OpenMP needs pointer types, but vectors are exception safe
    std::vector< uint8_t > column_vector(A_nt);
    uint8_t* column = column_vector.data();

    #pragma omp parallel
    #pragma omp master
    {
        for (int64_t k = 0; k < A_nt; ++k) {
            const int64_t diag_len = std::min(A.tileMb(k), A.tileNb(k));

            #pragma omp task depend(inout:column[k])
            {
                // local panel factorization
                internal::geqrf<Target::HostTask>(
                    A.sub(k, A_mt-1, k, k), Tlocal.sub(k, k, k, k),
                    diag_len, ib, max_panel_threads, priority_one);
                // TODO: bcast V & T across row for trailing matrix update

                // triangle-triangle reductions
                internal::ttqrt<Target::HostTask>(
                    A.sub(k, A_mt-1, k, k), Treduce.sub(k, A_mt-1, k, k));
                // TODO: bcast V's & T's across rows for trailing matrix update
            }

            // trailing matrix update
            for (int64_t j = k+1; j < A_nt; ++j) {
                #pragma omp task depend(in:column[k]) \
                                 depend(inout:column[j])
                {
                    // TODO unmqr
                    // TODO ttmqr
                }
            }
        }
    }
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup gesv_comp
template <Target target, typename scalar_t>
void geqrf(Matrix<scalar_t>& A,
           TriangularFactors<scalar_t>& T,
           const std::map<Option, Value>& opts)
{
    int64_t lookahead;
    try {
        lookahead = opts.at(Option::Lookahead).i_;
        assert(lookahead >= 0);
    }
    catch (std::out_of_range) {
        lookahead = 1;
    }

    int64_t ib;
    try {
        ib = opts.at(Option::InnerBlocking).i_;
        assert(ib >= 0);
    }
    catch (std::out_of_range) {
        ib = 1;
    }

    int64_t max_panel_threads;
    try {
        max_panel_threads = opts.at(Option::MaxPanelThreads).i_;
        assert(max_panel_threads >= 1);
    }
    catch (std::out_of_range) {
        max_panel_threads = std::max(omp_get_max_threads()/2, 1);
    }

    internal::specialization::geqrf(internal::TargetType<target>(),
                                    A, T,
                                    ib, max_panel_threads, lookahead);
}

//------------------------------------------------------------------------------
/// Distributed parallel QR factorization.
///
template <typename scalar_t>
void geqrf(Matrix<scalar_t>& A,
           TriangularFactors<scalar_t>& T,
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
            geqrf<Target::HostTask>(A, T, opts);
            break;
        case Target::HostNest:
            geqrf<Target::HostNest>(A, T, opts);
            break;
        case Target::HostBatch:
            geqrf<Target::HostBatch>(A, T, opts);
            break;
        case Target::Devices:
            geqrf<Target::Devices>(A, T, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void geqrf<float>(
    Matrix<float>& A,
    TriangularFactors<float>& T,
    const std::map<Option, Value>& opts);

template
void geqrf<double>(
    Matrix<double>& A,
    TriangularFactors<double>& T,
    const std::map<Option, Value>& opts);

template
void geqrf< std::complex<float> >(
    Matrix< std::complex<float> >& A,
    TriangularFactors< std::complex<float> >& T,
    const std::map<Option, Value>& opts);

template
void geqrf< std::complex<double> >(
    Matrix< std::complex<double> >& A,
    TriangularFactors< std::complex<double> >& T,
    const std::map<Option, Value>& opts);

} // namespace slate
