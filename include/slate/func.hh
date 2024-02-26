// Copyright (c) 2023-2023, University of Tennessee. All rights reserved.
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
/// Creates a uniform blocksize.
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
/// Computes the largest block given a blocksize function.
///
/// @param[in] nt
///     Number of tiles
///
/// @param[in] size
///     Block size function
///
/// @return The maximum blocksize
///
/// @ingroup func
///
inline int64_t max_blocksize(int64_t nt, std::function<int64_t(int64_t)> size)
{
    int64_t max = 0;
    for (int64_t i = 0; i < nt; ++i) {
        int64_t size_i = size(i);
        max = max < size_i ? size_i : max;
    }
    return max;
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
/// @param[in] order
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
device_2d_grid(GridOrder order, int64_t m, int64_t n, int64_t p, int64_t q)
{
    slate_assert( order != GridOrder::Unknown );
    if (order == GridOrder::Col) {
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
/// @param[in] order
///     Col distributes a single column across multiple processes
///     Row distributes a single row across multiple processes.
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
device_1d_grid(GridOrder order, int64_t block_size, int size)
{
    slate_assert( order != GridOrder::Unknown );
    if (order == GridOrder::Col) {
        return device_2d_grid(order, block_size, 1, size, 1);
    }
    else {
        return device_2d_grid(order, 1, block_size, 1, size);
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
/// @param[in] order
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
process_2d_grid(GridOrder order, int64_t p, int64_t q)
{
    slate_assert( order != GridOrder::Unknown );
    // Device and process grids aren't any different, they're just named to be
    // easier for ScaLAPACK users.
    // Setting a block size of 1 gives a tile-cyclic order.
    return device_2d_grid(order, 1, 1, p, q);
}

//------------------------------------------------------------------------------
/// Distributes tiles to processes in a 1d cyclic fashion, resulting in the
/// elements being distributed in a 1d block-cyclic fashion.
///
/// This function can also be used to distribute tiles to devices in a 2d
/// cyclic fashion, *regardless of process ownership*.  Thus, care must be taken
/// to prevent all of a process's tiles from being stored on a single device.
///
/// @param[in] order
///     Col distributes a single column across multiple processes.
///     Row distributes a single row across multiple processes.
///
/// @param[in] size
///     The number of processes
///
/// @return The distribution function
///
/// @ingroup func
///
inline std::function<int(ij_tuple)> process_1d_grid(GridOrder order, int size)
{
    slate_assert( order != GridOrder::Unknown );
    if (order == GridOrder::Col) {
        return process_2d_grid(order, size, 1);
    }
    else {
        return process_2d_grid(order, 1, size);
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

//------------------------------------------------------------------------------
/// Checks whether the given tile map is a 2d cyclic grid (i.e., equivalent
/// to 'process_2d_grid(order, p, q)' for some 'order', 'p', and 'q').
///
/// @param[in] mt
///     The number of tile rows to consider
///
/// @param[in] nt
///     The number of tile columns to consider
///
/// @param[in] func
///     The tile distribution to inspect
///
/// @param[out] order
///     The GridOrder detected.
///
/// @param[out] p
///     The number of rows in the grid, or -1 if the GridOrder is unknown.
///
/// @param[out] q
///     The number of columns in the grid, or -1 if the GridOrder is unknown.
///
/// @return Whether the map is a 2d cyclic grid
///
/// @ingroup func
///
inline bool is_2d_cyclic_grid(int64_t mt, int64_t nt, std::function<int(ij_tuple)> func,
                              GridOrder* order, int* p, int* q)
{
    // Hard code trivial corner cases
    if (mt == 0 || nt == 0 || (mt == 1 && nt == 1)) {
        *order = GridOrder::Col;
        *p = 1;
        *q = 1;
        return true;
    }

    *order = GridOrder::Unknown;
    *p = -1;
    *q = -1;

    // Determine possible grid order by checking f(0, 1) and f(1, 0)
    GridOrder pred_order = GridOrder::Unknown;
    if (mt == 1 || nt == 1) {
        // 1d distribution
        // Ambiguous order, so just choose col major
        pred_order = GridOrder::Col;
    }
    else if (func( {1, 0} ) == 1) {
        pred_order = GridOrder::Col;
    }
    else if (func( {0, 1} ) == 1) {
        pred_order = GridOrder::Row;
    }
    else if (func( {1, 0} ) == 0 && func( {0, 1} ) == 0) {
        pred_order = GridOrder::Col;
    }
    else {
        return false;
    }

    // Scan first row and column to determine p and q
    int64_t pred_p = 0;
    int64_t pred_q = 0;
    if (pred_order == GridOrder::Col) {
        while (pred_p < mt && func( {pred_p, 0} ) == pred_p) {
            ++pred_p;
        }
        while (pred_q < nt && func( {0, pred_q} ) == pred_q*pred_p) {
            ++pred_q;
        }
    }
    else { // pred_order == GridOrder::Row
        while (pred_q < nt && func( {0, pred_q} ) == pred_q) {
            ++pred_q;
        }
        while (pred_p < mt && func( {pred_p, 0} ) == pred_p*pred_q) {
            ++pred_p;
        }
    }

    // Verify the detected grid across the entire matrix
    auto ref = process_2d_grid( pred_order, pred_p, pred_q );
    for (int64_t i = 0; i < mt; ++i) {
        for (int64_t j = 0; j < nt; ++j) {
            if (func( {i, j} ) != ref( {i, j} )) {
                return false;
            }
        }
    }
    // Grid is 2d cyclic
    *order = pred_order;
    *p = pred_p;
    *q = pred_q;
    return true;
}

} // namespace func
} // namespace slate

#endif // SLATE_FUNC_HH
