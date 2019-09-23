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

//------------------------------------------------------------------------------
/// @file
///
#ifndef SLATE_INTERNAL_UTIL_HH
#define SLATE_INTERNAL_UTIL_HH

#include "slate/internal/mpi.hh"

#include <cmath>
#include <complex>

#include <blas.hh>

namespace slate {
namespace internal {

template <typename T>
T pow(T base, T exp);

void mpi_max_nan(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype);

//------------------------------------------
inline float real(float val) { return val; }
inline double real(double val) { return val; }
inline float real(std::complex<float> val) { return val.real(); }
inline double real(std::complex<double> val) { return val.real(); }

inline float imag(float val) { return 0.0; }
inline double imag(double val) { return 0.0; }
inline float imag(std::complex<float> val) { return val.imag(); }
inline double imag(std::complex<double> val) { return val.imag(); }

//--------------------------
template <typename scalar_t>
scalar_t make(blas::real_type<scalar_t> real, blas::real_type<scalar_t> imag);

template <>
inline float make<float>(float real, float imag) { return real; }

template <>
inline double make<double>(double real, double imag) { return real; }

template <>
inline std::complex<float> make<std::complex<float>>(float real, float imag)
{
    return std::complex<float>(real, imag);
}

template <>
inline std::complex<double> make<std::complex<double>>(double real, double imag)
{
    return std::complex<double>(real, imag);
}

//------------------------------------------------------------------------------
/// Helper function to sort by second element of a pair.
/// Used to sort rank_rows by row (see ttqrt, ttmqr), and rank_cols by col.
/// @return True if a.second < b.second.
template <typename T1, typename T2>
inline bool compareSecond(
    std::pair<T1, T2> const& a,
    std::pair<T1, T2> const& b)
{
    return a.second < b.second;
}

} // namespace internal
} // namespace slate

#endif // SLATE_INTERNAL_UTIL_HH
