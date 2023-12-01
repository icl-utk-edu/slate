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

typedef char slate_TileKind; /* enum */               ///< slate::TileKind
const slate_TileKind slate_TileKind_Workspace  = 'w'; ///< slate::TileKind::Workspace
const slate_TileKind slate_TileKind_SlateOwned = 'o'; ///< slate::TileKind::SlateOwned
const slate_TileKind slate_TileKind_UserOwned  = 'u'; ///< slate::TileKind::UserOwned
// end slate_TileKind

//------------------------------------------------------------------------------
// slate/include/slate/enums.hh

typedef char slate_Target; /* enum */              ///< slate::Target
const slate_Target slate_Target_Host        = 'H'; ///< slate::Target::Host
const slate_Target slate_Target_HostTask    = 'T'; ///< slate::Target::HostTask
const slate_Target slate_Target_HostNest    = 'N'; ///< slate::Target::HostNest
const slate_Target slate_Target_HostBatch   = 'B'; ///< slate::Target::HostBatch
const slate_Target slate_Target_Devices     = 'D'; ///< slate::Target::Devices
// end slate_Target

typedef char slate_MethodEig; /* enum */        ///< slate::MethodEig
const slate_MethodEig slate_MethodEig_QR = 'Q'; ///< slate::MethodEig::QR
const slate_MethodEig slate_MethodEig_DC = 'D'; ///< slate::MethodEig::DC
// end slate_MethodEig

// todo: auto sync with include/slate/enums.hh
typedef char slate_Option; /* enum */                      ///< slate::Option
const slate_Option slate_Option_ChunkSize            =  0; ///< slate::Option::ChunkSize
const slate_Option slate_Option_Lookahead            =  1; ///< slate::Option::Lookahead
const slate_Option slate_Option_BlockSize            =  2; ///< slate::Option::BlockSize
const slate_Option slate_Option_InnerBlocking        =  3; ///< slate::Option::InnerBlocking
const slate_Option slate_Option_MaxPanelThreads      =  4; ///< slate::Option::MaxPanelThreads
const slate_Option slate_Option_Tolerance            =  5; ///< slate::Option::Tolerance
const slate_Option slate_Option_Target               =  6; ///< slate::Option::Target
const slate_Option slate_Option_HoldLocalWorkspace   =  7; ///< slate::Option::HoldLocalWorkspace
const slate_Option slate_Option_Depth                =  8; ///< slate::Option::HoldLocalWorkspace
const slate_Option slate_Option_MaxIterations        =  9; ///< slate::Option::HoldLocalWorkspace
const slate_Option slate_Option_UseFallbackSolver    = 10; ///< slate::Option::HoldLocalWorkspace
const slate_Option slate_Option_PivotThreshold       = 11; ///< slate::Option::PivotThreshold
const slate_Option slate_Option_PrintVerbose         = 50; ///< slate::Option::PrintVerbose
const slate_Option slate_Option_PrintEdgeItems       = 51; ///< slate::Option::PrintEdgeItems
const slate_Option slate_Option_PrintWidth           = 52; ///< slate::Option::PrintWidth
const slate_Option slate_Option_PrintPrecision       = 53; ///< slate::Option::PrintPrecision
const slate_Option slate_Option_MethodCholQR         = 60; ///< slate::Option::MethodCholQR
const slate_Option slate_Option_MethodEig            = 61; ///< slate::Option::MethodEig
const slate_Option slate_Option_MethodGels           = 62; ///< slate::Option::MethodGels
const slate_Option slate_Option_MethodGemm           = 63; ///< slate::Option::MethodGemm
const slate_Option slate_Option_MethodHemm           = 64; ///< slate::Option::MethodHemm
const slate_Option slate_Option_MethodLU             = 65; ///< slate::Option::MethodLU
const slate_Option slate_Option_MethodTrsm           = 66; ///< slate::Option::MethodTrsm
// end slate_Option

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
