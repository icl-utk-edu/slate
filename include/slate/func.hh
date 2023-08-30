// Copyright (c) 2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_FUNC_HH
#define SLATE_FUNC_HH

#include "slate/enums.hh"

#include <functional>

namespace slate {

//------------------------------------------------------------------------------
/// @namespace slate::func
/// A set of functions useful in SLATE's "lambda" constructors
///
namespace func {

using ij_tuple = std::tuple<int64_t, int64_t>;

//------------------------------------------------------------------------------
// Block size functions

//------------------------------------------------------------------------------
/// Creates a uniform blocksize
///
/// @param[in] n
///     Global matrix size.  Needed to correctly handle the last block size
///
/// @param[in] nb
///     Block size
///
/// @retval The requested blocksize function
///
inline std::function<int64_t(int64_t)> uniform_blocksize(int64_t n, int64_t nb)
{
    return [n, nb](int64_t j) { return (j + 1)*nb > n ? n%nb : nb; };
}


//------------------------------------------------------------------------------
// Process & device distribution functions

//------------------------------------------------------------------------------
/// Distributes tiles to processes (or devices) in a 2d block-cyclic fashion
///
/// @param[in] layout
///     Whether to use a column major or a row major grid
///
/// @param[in] m
///     The number of rows in each block
///
/// @param[in] n
///     The number of columns in each block
///
/// @param[in] p
///     The number of rows in the process grid
///
/// @param[in] q
///     The number of columns in the process grid
///
/// @retval The distribution function
///
inline std::function<int(ij_tuple)>
grid_2d_block_cyclic(Layout layout, int64_t m, int64_t n, int64_t p, int64_t q)
{
    if (layout == Layout::ColMajor) {
        return [m, n, p, q]( ij_tuple ij ) {
            int64_t i = std::get<0>( ij ) / m;
            int64_t j = std::get<1>( ij ) / n;
            return int((i%p) + (j%q)*p);
        };
    }
    else {
        return [m, n, p, q]( ij_tuple ij ) {
            int64_t i = std::get<0>( ij ) / m;
            int64_t j = std::get<1>( ij ) / n;
            return int((i%p)*q + (j%q));
        };
    }
}

//------------------------------------------------------------------------------
/// Distributes tiles to processes (or devices) in a 2d cyclic fashion
///
/// @param[in] layout
///     Whether to use a column major or a row major grid
///
/// @param[in] p
///     The number of rows in the process grid
///
/// @param[in] q
///     The number of columns in the process grid
///
/// @retval The distribution function
///
inline std::function<int(ij_tuple)>
grid_2d_cyclic(Layout layout, int64_t p, int64_t q)
{
    return grid_2d_block_cyclic(layout, 1, 1, p, q);
}

//------------------------------------------------------------------------------
/// Distributes tiles to processes (or devices) in a 1d cyclic fashion
///
/// @param[in] layout
///     ColMajor distributes a single column across multiple processes
///     RowMajor distributes a single row across multiple processes
///
/// @param[in] size
///     The number of processes
///
/// @retval The distribution function
///
inline std::function<int(ij_tuple)> grid_1d_cyclic(Layout layout, int size)
{
    if (layout == Layout::ColMajor) {
        return grid_2d_cyclic(layout, size, 1);
    }
    else {
        return grid_2d_cyclic(layout, 1, size);
    }
}


//------------------------------------------------------------------------------
/// Distributes tiles to processes (or devices) in a 1d block-cyclic fashion
///
/// @param[in] layout
///     ColMajor distributes a single column across multiple processes
///     RowMajor distributes a single row across multiple processes
///
/// @param[in] block_size
///     The size of each block
///
/// @param[in] size
///     The number processes
///
/// @retval The distribution function
///
inline std::function<int(ij_tuple)>
grid_1d_block_cyclic(Layout layout, int64_t block_size, int size)
{
    if (layout == Layout::ColMajor) {
        return grid_2d_block_cyclic(layout, block_size, 1, size, 1);
    }
    else {
        return grid_2d_block_cyclic(layout, 1, block_size, 1, size);
    }
}

//------------------------------------------------------------------------------
/// Transposes the given tile distribution function
///
/// @param[in] old_func
///     The original distribution function
///
/// @retval The transposed distribution function
///
inline std::function<int(ij_tuple)>
grid_transpose(std::function<int(ij_tuple)> old_func)
{
    return [old_func]( ij_tuple ij ) {
        int64_t i = std::get<0>( ij );
        int64_t j = std::get<1>( ij );
        return old_func( ij_tuple({ j, i }) );
    };
}


} // namespace func
} // namespace slate

#endif // SLATE_FUNC_HH
