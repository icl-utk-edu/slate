// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "matrix_params.hh"

using testsweeper::ParamType;

const double inf = std::numeric_limits<double>::infinity();

// -----------------------------------------------------------------------------
/// Construct MatrixParams
MatrixParams::MatrixParams():
    verbose( 0 ),
    iseed {98, 108, 97, 115},

    //          name,    w, p, type,            default,             min, max, help
    kind      ("matrix", 0,    ParamType::List, "rand",                        "test matrix kind; see 'test --help-matrix'" ),
    cond      ("cond",   0, 1, ParamType::List, testsweeper::no_data_flag, 0, inf, "matrix condition number" ),
    cond_used ("cond",   0, 1, ParamType::List, testsweeper::no_data_flag, 0, inf, "actual condition number used" ),
    condD     ("condD",  0, 1, ParamType::List, testsweeper::no_data_flag, 0, inf, "matrix D condition number" )
{
    // Make different MatrixParams generate different matrices
    // (e.g., params.matrix and params.matrixB).
    iseed[0] = rand() % 256;
}

// -----------------------------------------------------------------------------
/// Marks matrix params as used.
void MatrixParams::mark()
{
    kind();
    cond();
    condD();
}
