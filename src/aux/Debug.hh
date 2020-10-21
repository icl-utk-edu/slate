// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_DEBUG_HH
#define SLATE_DEBUG_HH

#include "slate/BaseMatrix.hh"
#include "slate/internal/Memory.hh"

#include <iostream>

#include "slate/internal/cuda.hh"
#include "slate/internal/mpi.hh"
#include "slate/internal/openmp.hh"

namespace slate {

// -----------------------------------------------------------------------------
/// Slate::Debug - helper class used for debugging during development.
///
class Debug {
public:
    //--------
    /// Enable debugging output
    static void on() { Debug::debug_ = true; }
    /// Disable debugging output
    static void off() { Debug::debug_ = false; }

    template <typename scalar_t>
    static void diffLapackMatrices(int64_t m, int64_t n,
                                   scalar_t const* A, int64_t lda,
                                   scalar_t const* B, int64_t ldb,
                                   int64_t mb, int64_t nb);
    //-------------
    // BaseMatrix class
    template <typename scalar_t>
    static void checkTilesLives(BaseMatrix<scalar_t> const& A);

    template <typename scalar_t>
    static bool checkTilesLayout(BaseMatrix<scalar_t> const& A);

    template <typename scalar_t>
    static void printTilesLives(BaseMatrix<scalar_t> const& A);

    template <typename scalar_t>
    static void printTilesMaps(BaseMatrix<scalar_t> const& A);

    template <typename scalar_t>
    static void printTilesMOSI(BaseMatrix<scalar_t> const& A, const char* name,
                               const char* func, const char* file, int line);

    #define PRINTTILESMOSI(A) \
            printTilesMOSI(A, #A, __func__, __FILE__, __LINE__);

    //-------------
    // Memory class
    static void printNumFreeMemBlocks(Memory const& m);
    static void checkHostMemoryLeaks(Memory const& m);
    static void checkDeviceMemoryLeaks(Memory const& m, int device);

    template <typename scalar_t>
    static void printNumFreeMemBlocks(BaseMatrix<scalar_t> const& A)
    {
        printNumFreeMemBlocks(A.storage_->memory_);
    }

private:
    static bool debug_;

};

} // namespace slate

#endif // SLATE_DEBUG_HH
