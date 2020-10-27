// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_MATRIX_PARAMS_HH
#define SLATE_MATRIX_PARAMS_HH

#include "testsweeper.hh"

// =============================================================================
class MatrixParams
{
public:
    MatrixParams();

    void mark();

    int64_t verbose;
    int64_t iseed[4];

    // ---- test matrix generation parameters
    testsweeper::ParamString kind;
    testsweeper::ParamScientific cond, cond_used;
    testsweeper::ParamScientific condD;
};

#endif // SLATE_MATRIX_PARAMS_HH
