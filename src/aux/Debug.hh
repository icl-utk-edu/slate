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
