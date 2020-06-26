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

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// slate/include/slate/Tile.hh

typedef enum slate_TileKind {
    slate_TileKind_Workspace,         ///< slate::TileKind::Workspace
    slate_TileKind_SlateOwned,        ///< slate::TileKind::SlateOwned
    slate_TileKind_UserOwned,         ///< slate::TileKind::UserOwned
} slate_TileKind;                     ///< slate::TileKind

//------------------------------------------------------------------------------
// slate/include/slate/enums.hh

typedef enum slate_Target {
    slate_Target_Host        = 'H',   ///< slate::Target::Host
    slate_Target_HostTask    = 'T',   ///< slate::Target::HostTask
    slate_Target_HostNest    = 'N',   ///< slate::Target::HostNest
    slate_Target_HostBatch   = 'B',   ///< slate::Target::HostBatch
    slate_Target_Devices     = 'D',   ///< slate::Target::Devices
} slate_Target;                       ///< slate::Target

typedef enum slate_Option {
    slate_Option_ChunkSize,           ///< slate::Option::ChunkSize
    slate_Option_Lookahead,           ///< slate::Option::Lookahead
    slate_Option_BlockSize,           ///< slate::Option::BlockSize
    slate_Option_InnerBlocking,       ///< slate::Option::InnerBlocking
    slate_Option_MaxPanelThreads,     ///< slate::Option::MaxPanelThreads
    slate_Option_Tolerance,           ///< slate::Option::Tolerance
    slate_Option_Target,              ///< slate::Option::Target
} slate_Option;                       ///< slate::Option

//------------------------------------------------------------------------------
// slate/include/slate/types.hh

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
// blaspp/include/blas_util.hh

typedef enum slate_Op {
    slate_Op_NoTrans   = 'N',         ///< slate::Op::NoTrans
    slate_Op_Trans     = 'T',         ///< slate::Op::Trans
    slate_Op_ConjTrans = 'C',         ///< slate::Op::ConjTrans
} slate_Op;                           ///< slate::Op

typedef enum slate_Uplo {
    slate_Uplo_Upper    = 'U',        ///< slate::Uplo::Upper
    slate_Uplo_Lower    = 'L',        ///< slate::Uplo::Lower
    slate_Uplo_General  = 'G',        ///< slate::Uplo::General
} slate_Uplo;                         ///< slate::Uplo

typedef enum slate_Diag {
    slate_Diag_NonUnit  = 'N',        ///< slate::Diag::NonUnit
    slate_Diag_Unit     = 'U',        ///< slate::Diag::Unit
} slate_Diag;                         ///< slate::Diag

typedef enum slate_Side {
    slate_Side_Left  = 'L',           ///< slate::Side::Left
    slate_Side_Right = 'R',           ///< slate::Side::Right
} slate_Side;                         ///< slate::Side

typedef enum slate_Layout {
    slate_Layout_ColMajor = 'C',      ///< slate::Layou::ColMajor
    slate_Layout_RowMajor = 'R',      ///< slate::Layou::RowMajor
} slate_Layout;                       ///< slate::Layout

//------------------------------------------------------------------------------
// lapackpp/include/lapack_util.hh

typedef enum slate_Norm {
    slate_Norm_One = '1',             ///< slate::Norm::One
    slate_Norm_Two = '2',             ///< slate::Norm::Two
    slate_Norm_Inf = 'I',             ///< slate::Norm::Inf
    slate_Norm_Fro = 'F',             ///< slate::Norm::Fro
    slate_Norm_Max = 'M',             ///< slate::Norm::Max
} slate_Norm;                         ///< slate::Norm

typedef enum slate_Direction {
    slate_Direction_Forward  = 'F',   ///< slate::Direction::Forward
    slate_Direction_Backward = 'B',   ///< slate::Direction::Backward
} slate_Direction;                    ///< slate::Direction

typedef enum slate_Job {
    slate_Job_NoVec        = 'N',     ///< slate::Job::NoVec
    slate_Job_Vec          = 'V',     ///< slate::Job::Vec
    slate_Job_UpdateVec    = 'U',     ///< slate::Job::UpdateVec
    slate_Job_AllVec       = 'A',     ///< slate::Job::AllVec
    slate_Job_SomeVec      = 'S',     ///< slate::Job::SomeVec
    slate_Job_OverwriteVec = 'O',     ///< slate::Job::OverwriteVec
    slate_Job_CompactVec   = 'P',     ///< slate::Job::CompactVec
    slate_Job_SomeVecTol   = 'C',     ///< slate::Job::SomeVecTol
    slate_Job_VecJacobi    = 'J',     ///< slate::Job::VecJacobi
    slate_Job_Workspace    = 'W',     ///< slate::Job::Workspace
} slate_Job;                          ///< slate::Job

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // SLATE_C_API_TYPES_H
