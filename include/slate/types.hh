// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

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

    OptionValue(TileReleaseStrategy t) : i_(int(t))
    {}

    OptionValue(MethodEig m) : i_(int(m))
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

//------------------------------------------------------------------------------
/// Extracts an option.
///
/// @param[in] opt
///     Map of options and values.
///
///  @param[in] option
///     Option to get.
///
///  @param [in] defval
///     Default option value if option is not found in map.
///
template <typename T>
T get_option( Options opts, Option option, T defval )
{
    T retval;
    auto search = opts.find( option );
    if (search != opts.end())
        retval = T(search->second.i_);
    else
        retval = defval;

    return retval;
}

//----------------------------
/// Specialization for double.
template <>
inline double get_option<double>( Options opts, Option option, double defval )
{
    double retval;
    auto search = opts.find( option );
    if (search != opts.end())
        retval = search->second.d_;
    else
        retval = defval;

    return retval;
}

//------------------------------------------------------------------------------
// For %lld printf-style printing, cast to llong; guaranteed >= 64 bits.
using llong = long long;

} // namespace slate

#endif // SLATE_TYPES_HH
