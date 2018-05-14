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

///-----------------------------------------------------------------------------
/// \file
///
#ifndef SLATE_TYPES_HH
#define SLATE_TYPES_HH

#include "slate_mpi.hh"

#include <blas.hh>
#include <lapack.hh>

namespace slate {

typedef blas::Op Op;
typedef blas::Uplo Uplo;
typedef blas::Diag Diag;
typedef blas::Side Side;

typedef lapack::Norm Norm;

///-----------------------------------------------------------------------------
/// \class
/// \brief
///
enum class Target {Host, HostTask, HostNest, HostBatch, Devices};

namespace internal {
template <Target> class TargetType {};
} // namespace internal

//------------------------------------------------------------------------------
/// \class
/// \brief
///
enum class Option {
    Lookahead,
    BlockSize,
    Tolerance,
    Target,
};

//------------------------------------------------------------------------------
/// Slate::internal::Value class
/// \brief
///
class Value
{
public:
    Value()
    {}

    Value(int64_t i) : i_(i)
    {}

    Value(double d) : d_(d)
    {}

    Value(Target t) : i_(int(t))
    {}

    union {
        int64_t i_;
        double d_;
    };
};

//------------------------------------------------------------------------------
/// gives mpi_type based on actual scalar_t.
//  constants are initialized in slate_types.cc
template<typename scalar_t>
class mpi_type {};

template<>
class mpi_type<float> {
public:
    static MPI_Datatype value; // = MPI_FLOAT
};

template<>
class mpi_type<double> {
public:
    static MPI_Datatype value; // = MPI_DOUBLE
};

template<>
class mpi_type< std::complex<float> > {
public:
    static MPI_Datatype value; // = MPI_C_COMPLEX
};

template<>
class mpi_type< std::complex<double> > {
public:
    static MPI_Datatype value; // = MPI_C_DOUBLE_COMPLEX
};

//------------------------------------------------------------------------------
/// True if T is std::complex<T2> for some type T2.
template <typename T>
struct is_complex:
    std::integral_constant<bool, false>
{};

// specialize for std::complex
template <typename T>
struct is_complex< std::complex<T> >:
    std::integral_constant<bool, true>
{};

//------------------------------------------------------------------------------
// define enable_if_t if not using c++14
#if __cplusplus >= 201402L
    using std::enable_if_t;
#else
    template<bool B, class T = void>
    using enable_if_t = typename std::enable_if<B,T>::type;
#endif

} // namespace slate

#endif // SLATE_TYPES_HH
