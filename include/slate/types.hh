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
#ifndef SLATE_TYPES_HH
#define SLATE_TYPES_HH

#include "slate/enums.hh"
#include "slate/internal/mpi.hh"

#include <vector>
#include <map>

#include <blas.hh>
#include <lapack.hh>

namespace slate {

//------------------------------------------------------------------------------
/// Values for options to pass to SLATE routines.
/// Value can be:
/// - int
/// - int64_t
/// - double
/// - Target enum
/// @see Option
///
class OptionValue {
public:
    OptionValue()
    {}

    OptionValue(int i) : i_(i)
    {}

    OptionValue(int64_t i) : i_(i)
    {}

    OptionValue(double d) : d_(d)
    {}

    OptionValue(Target t) : i_(int(t))
    {}

    union {
        int64_t i_;
        double d_;
    };
};

using Options = std::map<Option, OptionValue>;
using Value   = OptionValue; ///< @deprecated

//------------------------------------------------------------------------------
class Pivot {
public:
    Pivot()
    {}

    Pivot(int64_t tile_index,
          int64_t element_offset)
        : tile_index_(tile_index),
          element_offset_(element_offset)
    {}

    int64_t tileIndex() const { return tile_index_; }
    int64_t elementOffset() const { return element_offset_; }

private:
    int64_t tile_index_;     ///< tile index in the panel submatrix
    int64_t element_offset_; ///< pivot offset in the tile
};

inline bool operator< (Pivot const& lhs, Pivot const& rhs)
{
    std::pair<int64_t, int64_t> lhs_pair(lhs.tileIndex(), lhs.elementOffset());
    std::pair<int64_t, int64_t> rhs_pair(rhs.tileIndex(), rhs.elementOffset());
    return lhs_pair < rhs_pair;
}

inline bool operator!= (Pivot const& lhs, Pivot const& rhs)
{
    std::pair<int64_t, int64_t> lhs_pair(lhs.tileIndex(), lhs.elementOffset());
    std::pair<int64_t, int64_t> rhs_pair(rhs.tileIndex(), rhs.elementOffset());
    return lhs_pair != rhs_pair;
}

using Pivots = std::vector< std::vector<Pivot> >;

//------------------------------------------------------------------------------
/// Gives mpi_type based on actual scalar_t.
//  Constants are initialized in slate_types.cc
template <typename scalar_t>
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

// for type-generic maxloc operations
template <typename real_t>
struct max_loc_type { real_t x; int i; };

template<>
class mpi_type< max_loc_type<float> > {
public:
    static MPI_Datatype value; // = MPI_FLOAT_INT
};

template<>
class mpi_type< max_loc_type<double> > {
public:
    static MPI_Datatype value; // = MPI_DOUBLE_INT
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
    using enable_if_t = typename std::enable_if<B, T>::type;
#endif

} // namespace slate

#endif // SLATE_TYPES_HH
