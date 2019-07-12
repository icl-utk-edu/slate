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
#include "slate/Tile_blas.hh"
#include "slate/TriangularMatrix.hh"
#include "internal/internal.hh"

namespace slate {

// specialization namespace differentiates, e.g.,
// internal::getrs from internal::specialization::getrs
namespace internal {
namespace specialization {

template <typename scalar_t>
using Reflectors = std::map< std::pair<int64_t, int64_t>,
                             std::vector<scalar_t> >;

//------------------------------------------------------------------------------
template <typename scalar_t>
void tb2bd_step(Matrix<scalar_t>& A, int64_t band,
                int64_t pass, int64_t step,
                Reflectors<scalar_t>& reflectors)
{
    int64_t task = step == 0 ? 0 : (step+1)%2 + 1;
    int64_t block = (step+1)/2;
    int64_t i;
    int64_t j;
    switch (task) {
        case 0:
            i =   pass;
            j = 1+pass;
            if (i < A.m() && j < A.n()) {
                internal::gebr1<Target::HostTask>(
                    A.slice(i, std::min(i+band-1, A.m()-1),
                            j, std::min(j+band-2, A.n()-1)),
                    reflectors[{i, j}],
                    reflectors[{i+1, j}]);
            }
            break;
        case 1:
            i = (block-1)*(band-1)+1+pass;
            j =  block   *(band-1)+1+pass;
            if (i < A.m() && j < A.n()) {
                internal::gebr2<Target::HostTask>(
                    reflectors[{i, j-(band-1)}],
                    A.slice(i, std::min(i+band-2, A.m()-1),
                            j, std::min(j+band-2, A.n()-1)),
                    reflectors[{i, j}]);
            }
            break;
        case 2:
            i = block*(band-1)+1+pass;
            j = block*(band-1)+1+pass;
            if (i < A.m() && j < A.n()) {
                internal::gebr3<Target::HostTask>(
                    reflectors[{i-(band-1), j}],
                    A.slice(i, std::min(i+band-2, A.m()-1),
                            j, std::min(j+band-2, A.n()-1)),
                    reflectors[{i, j}]);
            }
            break;
    }
}

//------------------------------------------------------------------------------
/// Reduced a block-bidiagonal (triangular-band) matrix to a bidiagonal form.
/// Generic implementation for any target.
/// @ingroup tb2bd_specialization
///
template <Target target, typename scalar_t>
void tb2bd(slate::internal::TargetType<target>,
           Matrix<scalar_t>& A, int64_t band, int64_t lookahead)
{
/*
    Reflectors<scalar_t> reflectors;
    for (int64_t k = 0; k < std::min(A.m(), A.n())-2; ++k) {

        int64_t i = k;
        int64_t j = k+1;
        internal::gebr1<Target::HostTask>(
            A.slice(i, std::min(i+band-1, A.m()-1),
                    j, std::min(j+band-2, A.n()-1)),
            reflectors[{i, j}],
            reflectors[{i+1, j}]);

        ++i;
        for (; i < A.m() && j < A.n();) {
            j += (band-1);
            if (j < A.n()) {
                internal::gebr2<Target::HostTask>(
                    reflectors[{i, j-(band-1)}],
                    A.slice(i, std::min(i+band-2, A.m()-1),
                            j, std::min(j+band-2, A.n()-1)),
                    reflectors[{i, j}]);

                i += (band-1);
                if (i < A.m()) {
                    internal::gebr3<Target::HostTask>(
                        reflectors[{i-(band-1), j}],
                        A.slice(i, std::min(i+band-2, A.m()-1),
                                j, std::min(j+band-2, A.n()-1)),
                        reflectors[{i, j}]);
                }
            }
        }
    }
*/
/*
    Reflectors<scalar_t> reflectors;

    int64_t diag_len = std::min(A.m(), A.n());
    for (int64_t pass = 0; pass < diag_len-2; ++pass) {

        int64_t pass_width = diag_len-1-pass;
        int64_t num_blocks;
        if (pass_width % (band-1) == 0)
            num_blocks = pass_width / (band-1);
        else
            num_blocks = pass_width / (band-1) + 1;

        int64_t block = 0;
        int64_t step = 0;
        while (block < num_blocks) {
            tb2bd_step(A, band, pass, step, reflectors);
            ++step;
            if (step%3 == 0)
                ++block;
        }
    }
*/
    Reflectors<scalar_t> reflectors;

    int64_t diag_len = std::min(A.m(), A.n());
    int64_t chunk = 4;
    for (int64_t pass = 0; pass < diag_len-2; pass += chunk) {

        int64_t width = diag_len-1-pass;
        int64_t num_blocks = width / (band-1);
        if (width % (band-1) > 0)
            ++ num_blocks;

        for (int64_t i = 0; i < num_blocks+chunk-1; ++i) {
            for (int64_t j = 0; j <= i && j < chunk; ++j) {
                int64_t inner_pass = pass+j;
                int64_t block = i-j;
                if (block == 0) {
                    tb2bd_step(A, band, inner_pass, block, reflectors);
                }
                else {
                    tb2bd_step(A, band, inner_pass, 2*block-1, reflectors);
                    tb2bd_step(A, band, inner_pass, 2*block  , reflectors);
                }
            }
        }
    }
}

} // namespace specialization
} // namespace internal

//------------------------------------------------------------------------------
/// Version with target as template parameter.
/// @ingroup tb2bd_specialization
///
template <Target target, typename scalar_t>
void tb2bd(Matrix<scalar_t>& A, int64_t band,
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

    internal::specialization::tb2bd(internal::TargetType<target>(),
                                    A, band, lookahead);
}

//------------------------------------------------------------------------------
///
template <typename scalar_t>
void tb2bd(Matrix<scalar_t>& A, int64_t band,
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
            tb2bd<Target::HostTask>(A, band, opts);
            break;
        case Target::HostNest:
            tb2bd<Target::HostNest>(A, band, opts);
            break;
        case Target::HostBatch:
            tb2bd<Target::HostBatch>(A, band, opts);
            break;
        case Target::Devices:
            tb2bd<Target::Devices>(A, band, opts);
            break;
    }
    // todo: return value for errors?
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void tb2bd<float>(
    Matrix<float>& A, int64_t band,
    const std::map<Option, Value>& opts);

template
void tb2bd<double>(
    Matrix<double>& A, int64_t band,
    const std::map<Option, Value>& opts);

template
void tb2bd< std::complex<float> >(
    Matrix< std::complex<float> >& A, int64_t band,
    const std::map<Option, Value>& opts);

template
void tb2bd< std::complex<double> >(
    Matrix< std::complex<double> >& A, int64_t band,
    const std::map<Option, Value>& opts);

} // namespace slate
