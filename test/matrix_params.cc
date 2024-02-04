// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "matrix_params.hh"

#include <climits>
#include <cmath>

using llong = long long;
using testsweeper::ParamType;
using testsweeper::no_data_flag;

const ParamType List   = ParamType::List;
const ParamType Value  = ParamType::Value;
const ParamType Output = ParamType::Output;

const double inf = std::numeric_limits<double>::infinity();

//------------------------------------------------------------------------------
// globals

std::map< std::string, int > matrix_labels;

//------------------------------------------------------------------------------
/// Construct MatrixParams
MatrixParams::MatrixParams():
    verbose( 0 ),

    //          name,     w, p, type,   default,      min, max,  help
    kind(       "matrix", 0,    List,   "rand",
                "test matrix kind; see 'test --help-matrix'" ),
    cond_request(
                "cond",   0, 1, List,   no_data_flag,   0, inf,
                "requested matrix condition number" ),
    cond_actual(
                "cond",   0, 1, Value,  no_data_flag,   0, inf,
                "actual condition number used" ),
    condD(      "condD",  0, 1, List,   no_data_flag,   0, inf,
                "matrix D condition number" ),
    seed(       "seed",   0,    List,   -1,            -1, INT64_MAX,
                "Randomization seed (-1 randomizes the seed for each matrix)"),
    label(      "A",      2,    Output, 0,              0, INT_MAX,
                "index labeling matrix" ),

    marked_( false )
{
}

//------------------------------------------------------------------------------
/// Marks matrix params as used.
void MatrixParams::mark()
{
    marked_ = true;

    kind();
    cond_request();
    condD();
    seed();
    label();
}

//------------------------------------------------------------------------------
/// Generates a label string for the current matrix type and sets the
/// label index. If it's a new matrix type, adds it to the global
/// `matrix_labels` dictionary of labels.
///
void MatrixParams::generate_label()
{
    char buf[ 80 ];
    std::string lbl = kind();

    // Add cond.
    lbl += ", cond ";
    if (std::isnan( cond_actual() )) {
        lbl += "unknown";
    }
    else {
        snprintf( buf, sizeof(buf), "= %.5g", cond_actual() );
        lbl += buf;
    }
    if (! std::isnan( cond_request() )
            && (cond_request() != cond_actual() || std::isnan( cond_actual() ))) {
        lbl += " (ignoring --cond)";
    }

    // Add condD.
    if (! std::isnan( condD() )) {
        snprintf( buf, sizeof(buf), ", cond(D) = %.5g", condD() );
        lbl += buf;
    }
    // todo: warn if condD is set but not used. Add condD_actual like cond_actual.

    // Get label index from matrix_labels, or add to matrix_labels.
    if (matrix_labels.count( lbl ) == 1) {  // map::contains is c++20
        label() = matrix_labels[ lbl ];
    }
    else {
        label() = matrix_labels.size() + 1;
        matrix_labels[ lbl ] = label();
    }
}
