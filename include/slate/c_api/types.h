// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
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

typedef char slate_Target; /* enum */              ///< slate::Target
const slate_Target slate_Target_Host        = 'H'; ///< slate::Target::Host
const slate_Target slate_Target_HostTask    = 'T'; ///< slate::Target::HostTask
const slate_Target slate_Target_HostNest    = 'N'; ///< slate::Target::HostNest
const slate_Target slate_Target_HostBatch   = 'B'; ///< slate::Target::HostBatch
const slate_Target slate_Target_Devices     = 'D'; ///< slate::Target::Devices
// end slate_Target

typedef char slate_TileReleaseStrategy; /* enum */                        ///< slate::TileReleaseStrategy
const slate_TileReleaseStrategy slate_TileReleaseStrategy_None     = 'N'; ///< slate::TileReleaseStrategy::None
const slate_TileReleaseStrategy slate_TileReleaseStrategy_Internal = 'I'; ///< slate::TileReleaseStrategy::Internal
const slate_TileReleaseStrategy slate_TileReleaseStrategy_Slate    = 'S'; ///< slate::TileReleaseStrategy::Slate
const slate_TileReleaseStrategy slate_TileReleaseStrategy_All      = 'A'; ///< slate::TileReleaseStrategy::All
// end slate_TileReleaseStrategy

typedef char slate_MethodEig; /* enum */        ///< slate::MethodEig
const slate_MethodEig slate_MethodEig_QR = 'Q'; ///< slate::MethodEig::QR
const slate_MethodEig slate_MethodEig_DC = 'D'; ///< slate::MethodEig::DC
// end slate_MethodEig;

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

typedef short slate_MOSI_State;

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

typedef char slate_Op; /* enum */        ///< slate::Op
const slate_Op slate_Op_NoTrans   = 'N'; ///< slate::Op::NoTrans
const slate_Op slate_Op_Trans     = 'T'; ///< slate::Op::Trans
const slate_Op slate_Op_ConjTrans = 'C'; ///< slate::Op::ConjTrans
// end slate_Op

typedef char slate_Uplo; /* enum */         ///< slate::Uplo
const slate_Uplo slate_Uplo_Upper   = 'U'; ///< slate::Uplo::Upper
const slate_Uplo slate_Uplo_Lower   = 'L'; ///< slate::Uplo::Lower
const slate_Uplo slate_Uplo_General = 'G'; ///< slate::Uplo::General
// end slate_Uplo

typedef char slate_Diag; /* enum */         ///< slate::Diag
const slate_Diag slate_Diag_NonUnit = 'N'; ///< slate::Diag::NonUnit
const slate_Diag slate_Diag_Unit    = 'U'; ///< slate::Diag::Unit
// end slate_Diag

typedef char slate_Side; /* enum */      ///< slate_Side
const slate_Side slate_Side_Left  = 'L'; ///< slate::Side::Left
const slate_Side slate_Side_Right = 'R'; ///< slate::Side::Right
// end slate_Side

typedef char slate_Layout; /* enum */
const slate_Layout slate_Layout_ColMajor = 'C'; ///< slate::Layout::ColMajor
const slate_Layout slate_Layout_RowMajor = 'R'; ///< slate::Layout::RowMajor
// end slate_Layout

//------------------------------------------------------------------------------
// lapackpp/include/lapack_util.hh

typedef char slate_Norm; /* enum */    ///< slate::Norm
const slate_Norm slate_Norm_One = '1'; ///< slate::Norm::One
const slate_Norm slate_Norm_Two = '2'; ///< slate::Norm::Two
const slate_Norm slate_Norm_Inf = 'I'; ///< slate::Norm::Inf
const slate_Norm slate_Norm_Fro = 'F'; ///< slate::Norm::Fro
const slate_Norm slate_Norm_Max = 'M'; ///< slate::Norm::Max
// end slate_Norm

typedef char slate_Direction; /* enum */
const slate_Direction slate_Direction_Forward  = 'F'; ///< slate::Direction::Forward
const slate_Direction slate_Direction_Backward = 'B'; ///< slate::Direction::Backward
// end slate_Direction

typedef char slate_Job; /* enum */
const slate_Job slate_Job_NoVec        = 'N';     ///< slate::Job::NoVec
const slate_Job slate_Job_Vec          = 'V';     ///< slate::Job::Vec
const slate_Job slate_Job_UpdateVec    = 'U';     ///< slate::Job::UpdateVec
const slate_Job slate_Job_AllVec       = 'A';     ///< slate::Job::AllVec
const slate_Job slate_Job_SomeVec      = 'S';     ///< slate::Job::SomeVec
const slate_Job slate_Job_OverwriteVec = 'O';     ///< slate::Job::OverwriteVec
const slate_Job slate_Job_CompactVec   = 'P';     ///< slate::Job::CompactVec
const slate_Job slate_Job_SomeVecTol   = 'C';     ///< slate::Job::SomeVecTol
const slate_Job slate_Job_VecJacobi    = 'J';     ///< slate::Job::VecJacobi
const slate_Job slate_Job_Workspace    = 'W';     ///< slate::Job::Workspace
// end slate_Job

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // SLATE_C_API_TYPES_H
