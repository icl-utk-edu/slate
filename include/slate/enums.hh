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
#ifndef SLATE_ENUMS_HH
#define SLATE_ENUMS_HH

#include <vector>

#include <blas.hh>
#include <lapack.hh>

namespace slate {

typedef blas::Op Op;
typedef blas::Uplo Uplo;
typedef blas::Diag Diag;
typedef blas::Side Side;
typedef blas::Layout Layout;

typedef lapack::Norm Norm;
typedef lapack::Direct Direction;  // todo change LAPACK++

typedef lapack::Job Job;

//------------------------------------------------------------------------------
/// Location and method of computation.
/// @ingroup enum
///
enum class Target : char {
    Host      = 'H',    ///< data resides on host
    HostTask  = 'T',    ///< computation using OpenMP nested tasks on host
    HostNest  = 'N',    ///< computation using OpenMP nested parallel for loops on host
    HostBatch = 'B',    ///< computation using batch BLAS on host (Intel MKL)
    Devices   = 'D',    ///< computation using batch BLAS on devices (cuBLAS)
};

namespace internal {
template <Target> class TargetType {};
} // namespace internal

//------------------------------------------------------------------------------
/// Keys for options to pass to SLATE routines.
/// @ingroup enum
///
enum class Option : char {
    ChunkSize,          ///< chunk size, >= 1
    Lookahead,          ///< lookahead depth, >= 0
    BlockSize,          ///< block size, >= 1
    InnerBlocking,      ///< inner blocking size, >= 1
    MaxPanelThreads,    ///< max number of threads for panel, >= 1
    Tolerance,          ///< tolerance for iterative methods, default epsilon
    Target,             ///< computation method (@see Target)
};

//------------------------------------------------------------------------------
/// To convert matrix between column-major and row-major.
/// @ingroup enum
///
enum class LayoutConvert : char {
    ColMajor = 'C',     ///< Convert to column-major
    RowMajor = 'R',     ///< Convert to row-major
    None     = 'N',     ///< No conversion
};

//------------------------------------------------------------------------------
/// Whether computing matrix norm, column norms, or row norms.
/// @ingroup enum
///
enum class NormScope : char {
    Columns = 'C',      ///< Compute column norms
    Rows    = 'R',      ///< Compute row norms
    Matrix  = 'M',      ///< Compute matrix norm
};

const int HostNum = -1;

} // namespace slate

#endif // SLATE_ENUMS_HH
