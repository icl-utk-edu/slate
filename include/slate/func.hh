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
/// @return The requested blocksize function
///
/// @ingroup func
///
inline std::function<int64_t(int64_t)> uniform_blocksize(int64_t n, int64_t nb)
{
    return [n, nb](int64_t j) { return (j + 1)*nb > n ? n%nb : nb; };
}


//------------------------------------------------------------------------------
// Process & device distribution functions

//------------------------------------------------------------------------------
/// Distributes tiles to devices in a 2d block-cyclic fashion.
///
/// When the tiles are distributed across processes with process_2d_grid,
/// this results in the local tiles being distributed cyclicly across devices.
///
/// This function can also be used to distribute tiles to processes, resulting
/// in blocks of tiles belonging to the same process.  However, the device grid
/// should then be adjusted.
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
/// @param[in] p
///     The number of rows in the device grid
///
/// @param[in] q
///     The number of columns in the device grid
///
/// @return The distribution function
///
/// @ingroup func
///
inline std::function<int(ij_tuple)>
device_2d_grid(Layout layout, int64_t m, int64_t n, int64_t p, int64_t q)
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
/// Distributes tiles to devices in a 1d block-cyclic fashion.
///
/// When the tiles are distributed across processes with process_2d_grid or
/// process_1d_grid, this results in the local tiles being distributed cyclicly
/// across devices.
///
/// This function can also be used to distribute tiles to processes, resulting
/// in blocks of tiles belonging to the same process.  However, the device grid
/// should then be adjusted.
///
/// @param[in] layout
///     ColMajor distributes a single column across multiple processes
///     RowMajor distributes a single row across multiple processes.
///
/// @param[in] block_size
///     The number of rows or columns in the process grid
///
/// @param[in] size
///     The number of rows or column in the device grid
///
/// @return The distribution function
///
/// @ingroup func
///
inline std::function<int(ij_tuple)>
device_1d_grid(Layout layout, int64_t block_size, int size)
{
    if (layout == Layout::ColMajor) {
        return device_2d_grid(layout, block_size, 1, size, 1);
    }
    else {
        return device_2d_grid(layout, 1, block_size, 1, size);
    }
}

//------------------------------------------------------------------------------
/// Distributes tiles to processes in a 2d cyclic fashion, resulting in the
/// elements being distributed in a 2d block-cyclic fashion.
///
/// This function can also be used to distribute tiles to devices in a 2d
/// cyclic fashion, *regardless of process ownership*.  Thus, care must be taken
/// to prevent all of a process's tiles from being stored on a single device.
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
/// @return The distribution function
///
/// @ingroup func
///
inline std::function<int(ij_tuple)>
process_2d_grid(Layout layout, int64_t p, int64_t q)
{
    // Device and process grids aren't any different, they're just named to be
    // easier for ScaLAPACK users.
    // Setting a block size of 1 gives a tile-cyclic layout.
    return device_2d_grid(layout, 1, 1, p, q);
}

//------------------------------------------------------------------------------
/// Distributes tiles to processes in a 1d cyclic fashion, resulting in the
/// elements being distributed in a 1d block-cyclic fashion.
///
/// This function can also be used to distribute tiles to devices in a 2d
/// cyclic fashion, *regardless of process ownership*.  Thus, care must be taken
/// to prevent all of a process's tiles from being stored on a single device.
///
/// @param[in] layout
///     ColMajor distributes a single column across multiple processes.
///     RowMajor distributes a single row across multiple processes.
///
/// @param[in] size
///     The number of processes
///
/// @return The distribution function
///
/// @ingroup func
///
inline std::function<int(ij_tuple)> process_1d_grid(Layout layout, int size)
{
    if (layout == Layout::ColMajor) {
        return process_2d_grid(layout, size, 1);
    }
    else {
        return process_2d_grid(layout, 1, size);
    }
}


//------------------------------------------------------------------------------
/// Transposes the given tile distribution function for processes or devices.
///
/// @param[in] old_func
///     The original distribution function
///
/// @return The transposed distribution function
///
/// @ingroup func
///
inline std::function<int(ij_tuple)>
transpose_grid(std::function<int(ij_tuple)> old_func)
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
