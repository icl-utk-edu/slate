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

#ifndef SLATE_C_API_TYPES_H
#define SLATE_C_API_TYPES_H

#include "slate/c_api/enums.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef union slate_OptionValue {
    int64_t       chunk_size;
    int64_t       lookahead;
    int64_t       block_size;
    int64_t       inner_blocking;
    int64_t       max_panel_threads;
    double        tolerance;
    slate_Target  target;
} slate_OptionValue;                  ///< slate::OptionValue

typedef struct slate_Options {
    slate_Option      option;
    slate_OptionValue value;
} slate_Options;                      ///< slate::Options

//------------------------------------------------------------------------------
/// slate::Pivots

struct slate_Pivots_struct;
typedef struct slate_Pivots_struct* slate_Pivots;

slate_Pivots slate_Pivots_create();
void slate_Pivots_destroy(slate_Pivots pivots);

//------------------------------------------------------------------------------
/// slate::TriangularFactors< std::complex<double> >

struct slate_TriangularFactors_struct_r32;
struct slate_TriangularFactors_struct_r64;
struct slate_TriangularFactors_struct_c32;
struct slate_TriangularFactors_struct_c64;

typedef struct slate_TriangularFactors_struct_r32* slate_TriangularFactors_r32;
typedef struct slate_TriangularFactors_struct_r64* slate_TriangularFactors_r64;
typedef struct slate_TriangularFactors_struct_c32* slate_TriangularFactors_c32;
typedef struct slate_TriangularFactors_struct_c64* slate_TriangularFactors_c64;

slate_TriangularFactors_r32 slate_TriangularFactors_create_r32();
slate_TriangularFactors_r64 slate_TriangularFactors_create_r64();
slate_TriangularFactors_c32 slate_TriangularFactors_create_c32();
slate_TriangularFactors_c64 slate_TriangularFactors_create_c64();

void slate_TriangularFactors_destroy_r32(slate_TriangularFactors_r32 T);
void slate_TriangularFactors_destroy_r64(slate_TriangularFactors_r64 T);
void slate_TriangularFactors_destroy_c32(slate_TriangularFactors_c32 T);
void slate_TriangularFactors_destroy_c64(slate_TriangularFactors_c64 T);

#ifdef __cplusplus
}  // extern "C"
#endif

namespace slate {

//------------------------------------------------------------------------------

inline std::pair<Option, OptionValue> optionvalue2cpp(
    slate_Option option, slate_OptionValue option_value)
{
    switch (option) {
        case slate_Option_ChunkSize:
            return {Option::ChunkSize, option_value.chunk_size};
        case slate_Option_Lookahead:
            return {Option::Lookahead, option_value.lookahead};
        case slate_Option_BlockSize:
            return {Option::BlockSize, option_value.block_size};
        case slate_Option_InnerBlocking:
            return {Option::InnerBlocking, option_value.inner_blocking};
        case slate_Option_MaxPanelThreads:
            return {Option::MaxPanelThreads, option_value.max_panel_threads};
        case slate_Option_Tolerance:
            return {Option::Tolerance, option_value.tolerance};
        case slate_Option_Target:
            return {Option::Target, target2cpp(option_value.target)};
        default: throw Exception("unknown option value");
    }
}

inline void options2cpp(
    int num_options, slate_Options options[], Options& options_)
{
    if (options !=  nullptr) {
        for(int i = 0; i < num_options; ++i) {
            options_.insert(
                optionvalue2cpp(options[i].option, options[i].value));
        }
    }
}

} // namespace slate

#endif // SLATE_C_API_TYPES_H
