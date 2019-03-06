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

#include "slate/internal/util.hh"
#include "internal/internal_util.hh"

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// [internal]
/// Computes the power function for integer arguments.
///
template <typename T>
T pow(T base, T exp)
{
    T pow = 1;
    for (T i = 0; i < exp; ++i)
        pow *= base;

    return pow;
}

//------------------------------
// explicit instantiation
template
int pow<int>(int base, int exp);

//------------------------------------------------------------------------------
/// [internal]
/// Implememts a custom MPI reduction that propagates NaNs.
///
void mpi_max_nan(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype)
{
    if (*datatype == MPI_DOUBLE) {
        double* x = (double*) invec;
        double* y = (double*) inoutvec;
        for (int i = 0; i < *len; ++i)
            y[i] = max_nan(x[i], y[i]);
    }
    else if (*datatype == MPI_FLOAT) {
        float* x = (float*) invec;
        float* y = (float*) inoutvec;
        for (int i = 0; i < *len; ++i)
            y[i] = max_nan(x[i], y[i]);
    }
}

} // namespace internal
} // namespace slate
