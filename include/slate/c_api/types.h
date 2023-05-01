// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

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

typedef enum slate_TileReleaseStrategy {
    slate_TileReleaseStrategy_None     = 'N', ///< slate::TileReleaseStrategy::None
    slate_TileReleaseStrategy_Internal = 'I', ///< slate::TileReleaseStrategy::Internal
    slate_TileReleaseStrategy_Slate    = 'S', ///< slate::TileReleaseStrategy::Slate
    slate_TileReleaseStrategy_All      = 'A', ///< slate::TileReleaseStrategy::All
} slate_TileReleaseStrategy;                  ///< slate::TileReleaseStrategy

typedef enum slate_MethodEig {
    slate_MethodEig_QR = 'Q',   ///< slate::MethodEig::QR
    slate_MethodEig_DC = 'D',   ///< slate::MethodEig::DC
} slate_MethodEig;

// todo: auto sync with include/slate/enums.hh
typedef enum slate_Option {
    slate_Option_ChunkSize,           ///< slate::Option::ChunkSize
    slate_Option_Lookahead,           ///< slate::Option::Lookahead
    slate_Option_BlockSize,           ///< slate::Option::BlockSize
    slate_Option_InnerBlocking,       ///< slate::Option::InnerBlocking
    slate_Option_MaxPanelThreads,     ///< slate::Option::MaxPanelThreads
    slate_Option_Tolerance,           ///< slate::Option::Tolerance
    slate_Option_Target,              ///< slate::Option::Target
    slate_Option_TileReleaseStrategy, ///< slate::Option::TileReleaseStrategy
    slate_Option_HoldLocalWorkspace,  ///< slate::Option::HoldLocalWorkspace
    slate_Option_PrintVerbose,        ///< slate::Option::PrintVerbose
    slate_Option_PrintEdgeItems,      ///< slate::Option::PrintEdgeItems
    slate_Option_PrintWidth,          ///< slate::Option::PrintWidth
    slate_Option_PrintPrecision,      ///< slate::Option::PrintPrecision
    slate_Option_PivotThreshold,      ///< slate::Option::PivotThreshold
    slate_Option_MethodCholQR,        ///< slate::Option::MethodCholQR
    slate_Option_MethodEig,           ///< slate::Option::MethodEig
    slate_Option_MethodGels,          ///< slate::Option::MethodGels
    slate_Option_MethodGemm,          ///< slate::Option::MethodGemm
    slate_Option_MethodHemm,          ///< slate::Option::MethodHemm
    slate_Option_MethodLU,            ///< slate::Option::MethodLU
    slate_Option_MethodTrsm,          ///< slate::Option::MethodTrsm
} slate_Option;                       ///< slate::Option

//------------------------------------------------------------------------------
// slate/include/slate/types.hh

// todo: should this be just i_ and d_, with cast from enums to int64_t?
typedef union slate_OptionValue {
    int64_t       chunk_size;
    int64_t       lookahead;
    int64_t       block_size;
    int64_t       inner_blocking;
    int64_t       max_panel_threads;
    double        tolerance;
    slate_Target  target;
    slate_TileReleaseStrategy tile_release_strategy;
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
    slate_Layout_ColMajor = 'C',      ///< slate::Layout::ColMajor
    slate_Layout_RowMajor = 'R',      ///< slate::Layout::RowMajor
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
