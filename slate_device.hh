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
// Need assistance with the SLATE software? Join the "SLATE User" Google group
// by going to https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user
// and clicking "Apply to join group". Upon acceptance, email your questions and
// comments to <slate-user@icl.utk.edu>.
//------------------------------------------------------------------------------

#ifndef SLATE_DEVICE_HH
#define SLATE_DEVICE_HH

#include "slate_cuda.hh"

#include <complex>

namespace slate {
namespace device {

// -----------------------------------------------------------------------------
// common_type_t is defined in C++14; here's a C++11 definition
#if __cplusplus >= 201402L
    using std::common_type_t;
    using std::decay_t;
#else
    template< typename... Ts >
    using common_type_t = typename std::common_type< Ts... >::type;

    template< typename... Ts >
    using decay_t = typename std::decay< Ts... >::type;
#endif

// -----------------------------------------------------------------------------
// Based on C++14 common_type implementation from
// http://www.cplusplus.com/reference/type_traits/common_type/
// Adds promotion of complex types based on the common type of the associated
// real types. This fixes various cases:
//
// std::common_type_t< double, complex<float> > is complex<float>  (wrong)
//        scalar_type< double, complex<float> > is complex<double> (right)
//
// std::common_type_t< int, complex<long> > is not defined (compile error)
//        scalar_type< int, complex<long> > is complex<long> (right)

// for zero types
template< typename... Types >
struct scalar_type_traits;

// define scalar_type<> type alias
template< typename... Types >
using scalar_type = typename scalar_type_traits< Types... >::type;

// for one type
template< typename T >
struct scalar_type_traits< T >
{
    using type = decay_t<T>;
};

// for two types
// relies on type of ?: operator being the common type of its two arguments
template< typename T1, typename T2 >
struct scalar_type_traits< T1, T2 >
{
    using type = decay_t< decltype( true ? std::declval<T1>() : std::declval<T2>() ) >;
};

// for either or both complex,
// find common type of associated real types, then add complex
template< typename T1, typename T2 >
struct scalar_type_traits< std::complex<T1>, T2 >
{
    using type = std::complex< common_type_t< T1, T2 > >;
};

template< typename T1, typename T2 >
struct scalar_type_traits< T1, std::complex<T2> >
{
    using type = std::complex< common_type_t< T1, T2 > >;
};

template< typename T1, typename T2 >
struct scalar_type_traits< std::complex<T1>, std::complex<T2> >
{
    using type = std::complex< common_type_t< T1, T2 > >;
};

// for three or more types
template< typename T1, typename T2, typename... Types >
struct scalar_type_traits< T1, T2, Types... >
{
    using type = scalar_type< scalar_type< T1, T2 >, Types... >;
};

// -----------------------------------------------------------------------------
// for any combination of types, determine associated real, scalar,
// and complex types.
//
// real_type< float >                               is float
// real_type< float, double, complex<float> >       is double
//
// scalar_type< float >                             is float
// scalar_type< float, complex<float> >             is complex<float>
// scalar_type< float, double, complex<float> >     is complex<double>
//
// complex_type< float >                            is complex<float>
// complex_type< float, double >                    is complex<double>
// complex_type< float, double, complex<float> >    is complex<double>

// for zero types
template< typename... Types >
struct real_type_traits;

// define real_type<> type alias
template< typename... Types >
using real_type = typename real_type_traits< Types... >::real_t;

// define complex_type<> type alias
template< typename... Types >
using complex_type = std::complex< real_type< Types... > >;

// for one type
template< typename T >
struct real_type_traits<T>
{
    using real_t = T;
};

// for one complex type, strip complex
template< typename T >
struct real_type_traits< std::complex<T> >
{
    using real_t = T;
};

// for two or more types
template< typename T1, typename... Types >
struct real_type_traits< T1, Types... >
{
    using real_t = scalar_type< real_type<T1>, real_type< Types... > >;
};

//------------------------------------------------------------------------------
template <typename scalar_t>
void genormMax(
    int64_t m, int64_t n,
    scalar_t** a, int64_t lda,
    real_type<scalar_t>* max,
    int64_t batch_count,
    cudaStream_t stream);

} // namespace device
} // namespace slate

#endif // SLATE_DEVICE_HH
