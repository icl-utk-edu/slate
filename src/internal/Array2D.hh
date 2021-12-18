// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_ARRAY2D_HH
#define SLATE_ARRAY2D_HH

namespace slate {
namespace internal {

#include <vector>

//------------------------------------------------------------------------------
/// Very simple 2D array. Uses column-major.
/// Uses rows (m) as leading dimension.
///
template <typename scalar_t>
class Array2D
{
public:
    /// Allocate array of m rows by n columns.
    Array2D( int m, int n, scalar_t value=scalar_t() ):
        m_( m ),
        n_( n ),
        data_( m_ * n_, value )
    {}

    /// @return (i, j) element. i and j are 0-based.
    scalar_t operator() (int i, int j) const
    {
        return data_[ i + j*m_ ];
    }

    /// @return reference to (i, j) element. i and j are 0-based.
    scalar_t& operator() (int i, int j)
    {
        return data_[ i + j*m_ ];
    }

    /// @return number of rows.
    int m() const { return m_; }

    /// @return number of columns.
    int n() const { return n_; }

private:
    int m_, n_;
    std::vector<scalar_t> data_;
};

} // namespace internal
} // namespace slate

#endif // SLATE_ARRAY2D_HH
