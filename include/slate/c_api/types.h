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

typedef char slate_MethodTrsm; /* enum */           ///< slate::MethodTrsm
const slate_MethodTrsm slate_MethodTrsm_Auto = '*'; ///< slate::MethodTrsm::Auto
const slate_MethodTrsm slate_MethodTrsm_A    = 'A'; ///< slate::MethodTrsm::A
const slate_MethodTrsm slate_MethodTrsm_B    = 'B'; ///< slate::MethodTrsm::B
// end slate_MethodTrsm

typedef char slate_MethodGemm; /* enum */           ///< slate::MethodGemm
const slate_MethodGemm slate_MethodGemm_Auto = '*'; ///< slate::MethodGemm::Auto
const slate_MethodGemm slate_MethodGemm_A    = 'A'; ///< slate::MethodGemm::A
const slate_MethodGemm slate_MethodGemm_C    = 'C'; ///< slate::MethodGemm::C
// end slate_MethodGemm

typedef char slate_MethodHemm; /* enum */           ///< slate::MethodHemm
const slate_MethodHemm slate_MethodHemm_Auto = '*'; ///< slate::MethodHemm::Auto
const slate_MethodHemm slate_MethodHemm_A    = 'A'; ///< slate::MethodHemm::A
const slate_MethodHemm slate_MethodHemm_C    = 'C'; ///< slate::MethodHemm::C
// end slate_MethodHemm

typedef char slate_MethodCholQR; /* enum */              ///< slate::MethodCholQR
const slate_MethodCholQR slate_MethodCholQR_Auto  = '*'; ///< slate::MethodCholQR::Auto
const slate_MethodCholQR slate_MethodCholQR_GemmA = 'A'; ///< slate::MethodCholQR::GemmA
const slate_MethodCholQR slate_MethodCholQR_GemmC = 'C'; ///< slate::MethodCholQR::GemmC
const slate_MethodCholQR slate_MethodCholQR_HerkA = 'R'; ///< slate::MethodCholQR::HerkA
const slate_MethodCholQR slate_MethodCholQR_HerkC = 'K'; ///< slate::MethodCholQR::HerkC
// end slate_MethodCholQR

typedef char slate_MethodGels; /* enum */             ///< slate::MethodGels
const slate_MethodGels slate_MethodGels_Auto   = '*'; ///< slate::MethodGels::Auto
const slate_MethodGels slate_MethodGels_QR     = 'Q'; ///< slate::MethodGels::QR
const slate_MethodGels slate_MethodGels_CholQR = 'C'; ///< slate::MethodGels::CholQR
// end slate_MethodGels

typedef char slate_MethodLU; /* enum */       ///< slate::MethodLU
const slate_MethodLU slate_MethodLU_Auto       = '*'; ///< slate::MethodLU::Auto
const slate_MethodLU slate_MethodLU_PartialPiv = 'P'; ///< slate::MethodLU::PartialPiv
const slate_MethodLU slate_MethodLU_CALU       = 'C'; ///< slate::MethodLU::CALU
const slate_MethodLU slate_MethodLU_NoPiv      = 'N'; ///< slate::MethodLU::NoPiv
const slate_MethodLU slate_MethodLU_RBT        = 'R'; ///< slate::MethodLU::RBT
const slate_MethodLU slate_MethodLU_BEAM       = 'B'; ///< slate::MethodLU::BEAM
// end slate_MethodLU

typedef char slate_MethodEig; /* enum */               ///< slate::MethodEig
const slate_MethodEig slate_MethodEig_Auto      = '*'; ///< slate::MethodEig::Auto
const slate_MethodEig slate_MethodEig_QR        = 'Q'; ///< slate::MethodEig::QR
const slate_MethodEig slate_MethodEig_DC        = 'D'; ///< slate::MethodEig::DC
const slate_MethodEig slate_MethodEig_Bisection = 'B'; ///< slate::MethodEig::Bisection
const slate_MethodEig slate_MethodEig_MRRR      = 'M'; ///< slate::MethodEig::MRRR
// end slate_MethodEig

typedef char slate_MethodSVD; /* enum */               ///< slate::MethodSVD
const slate_MethodSVD slate_MethodSVD_Auto      = '*'; ///< slate::MethodSVD::Auto
const slate_MethodSVD slate_MethodSVD_QR        = 'Q'; ///< slate::MethodSVD::QR
const slate_MethodSVD slate_MethodSVD_DC        = 'D'; ///< slate::MethodSVD::DC
const slate_MethodSVD slate_MethodSVD_Bisection = 'B'; ///< slate::MethodSVD::Bisection
// end slate_MethodSVD

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
