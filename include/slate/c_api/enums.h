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

#ifndef SLATE_C_API_ENUMS_H
#define SLATE_C_API_ENUMS_H

#include "slate/slate.hh"

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

namespace slate {

//------------------------------------------------------------------------------

inline TileKind tilekind2cpp(slate_TileKind tilekind)
{
    switch (tilekind) {
        case slate_TileKind_Workspace:  return TileKind::Workspace;
        case slate_TileKind_SlateOwned: return TileKind::SlateOwned;
        case slate_TileKind_UserOwned:  return TileKind::UserOwned;
        default: throw Exception("unknown tile kind");
    }
}

//------------------------------------------------------------------------------

inline Target target2cpp(slate_Target target)
{
    switch (target) {
        case slate_Target_Host:      return Target::Host;
        case slate_Target_HostTask:  return Target::HostTask;
        case slate_Target_HostNest:  return Target::HostNest;
        case slate_Target_HostBatch: return Target::HostBatch;
        case slate_Target_Devices:   return Target::Devices;
        default: throw Exception("unknown target");
    }
}

inline Option option2cpp(slate_Option option)
{
    switch (option) {
        case slate_Option_ChunkSize:       return Option::ChunkSize;
        case slate_Option_Lookahead:       return Option::Lookahead;
        case slate_Option_BlockSize:       return Option::BlockSize;
        case slate_Option_InnerBlocking:   return Option::InnerBlocking;
        case slate_Option_MaxPanelThreads: return Option::MaxPanelThreads;
        case slate_Option_Tolerance:       return Option::Tolerance;
        case slate_Option_Target:          return Option::Target;
        default: throw Exception("unknown option");
    }
}

//------------------------------------------------------------------------------

inline Op op2cpp(slate_Op op)
{
    switch (op) {
        case slate_Op_NoTrans:    return Op::NoTrans;
        case slate_Op_Trans:      return Op::Trans;
        case slate_Op_ConjTrans:  return Op::ConjTrans;
        default: throw Exception("unknown op");
    }
}

inline Uplo uplo2cpp(slate_Uplo uplo)
{
    switch (uplo) {
        case slate_Uplo_Upper:    return Uplo::Upper;
        case slate_Uplo_Lower:    return Uplo::Lower;
        case slate_Uplo_General:  return Uplo::General;
        default: throw Exception("unknown uplo");
    }
}

inline Diag diag2cpp(slate_Diag diag)
{
    switch (diag) {
        case slate_Diag_NonUnit:  return Diag::NonUnit;
        case slate_Diag_Unit:     return Diag::Unit;
        default: throw Exception("unknown diag");
    }
}

inline Side side2cpp(slate_Side side)
{
    switch (side) {
        case slate_Side_Left:  return Side::Left;
        case slate_Side_Right: return Side::Right;
        default: throw Exception("unknown side");
    }
}

inline Layout layout2cpp(slate_Layout layout)
{
    switch (layout) {
        case slate_Layout_ColMajor: return Layout::ColMajor;
        case slate_Layout_RowMajor: return Layout::RowMajor;
        default: throw Exception("unknown layout");
    }
}

//------------------------------------------------------------------------------

inline Norm norm2cpp(slate_Norm norm)
{
    switch (norm) {
        case slate_Norm_One: return Norm::One;
        case slate_Norm_Two: return Norm::Two;
        case slate_Norm_Inf: return Norm::Inf;
        case slate_Norm_Fro: return Norm::Fro;
        case slate_Norm_Max: return Norm::Max;
        default: throw Exception("unknown norm");
    }
}

inline Direction direct2cpp(slate_Direction direction)
{
    switch (direction) {
        case slate_Direction_Forward:  return Direction::Forward;
        case slate_Direction_Backward: return Direction::Backward;
        default: throw Exception("unknown direction");
  }
}

inline Job job2cpp(slate_Job job)
{
    switch (job) {
        case slate_Job_NoVec:         return Job::NoVec;
        case slate_Job_Vec:           return Job::Vec;
        case slate_Job_UpdateVec:     return Job::UpdateVec;
        case slate_Job_AllVec:        return Job::AllVec;
        case slate_Job_SomeVec:       return Job::SomeVec;
        case slate_Job_OverwriteVec:  return Job::OverwriteVec;
        case slate_Job_CompactVec:    return Job::CompactVec;
        case slate_Job_SomeVecTol:    return Job::SomeVecTol;
        case slate_Job_VecJacobi:     return Job::VecJacobi;
        case slate_Job_Workspace:     return Job::Workspace;
        default: throw Exception("unknown job");
    }
}

} // namespace slate

#endif // SLATE_C_API_ENUMS_H
