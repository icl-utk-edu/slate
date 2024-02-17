// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/enums.hh"

namespace slate {

const char* GridOrder_help    = "c or col; r or row";
const char* MethodCholQR_help = "auto, gemmA, gemmC, herkA, or herkC";
const char* MethodGels_help   = "auto, QR, geqrf, or CholQR";
const char* MethodGemm_help   = "auto; A or gemmA; C or gemmC";
const char* MethodHemm_help   = "auto; A or hemmA; C or hemmC";
const char* MethodTrsm_help   = "auto; A or trsmA; B or trsmB";
const char* MethodLU_help     = "auto; PPLU or PartialPiv; CALU, NoPiv, RBT, or BEAM";
const char* MethodEig_help    = "auto, QR (QR iteration), DC (divide & conquer), bisection, or MRRR";
const char* MethodSVD_help    = "auto, QR (QR iteration), DC (divide & conquer), or bisection";
const char* NormScope_help    = "m or matrix; c, cols, or columns; r or rows";
const char* Origin_help       = "d, dev, or devices; h or host; s, scalpk, or scalapack";
const char* Target_help       = "d, dev, or devices; h or host; t or task; n or nest; b or batch";

} // namespace slate
