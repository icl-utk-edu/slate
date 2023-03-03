// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "matrix_params.hh"

#include <climits>
#include <cmath>

using llong = long long;
using testsweeper::ParamType;

const double inf = std::numeric_limits<double>::infinity();

std::map< std::string, int > matrix_labels;

// -----------------------------------------------------------------------------
/// Construct MatrixParams
MatrixParams::MatrixParams():
    verbose( 0 ),

    //          name,    w, p, type,            default,                 min, max,  help
    kind      ("matrix", 0,    ParamType::List, "rand",                             "test matrix kind; see 'test --help-matrix'" ),
    cond      ("cond",   0, 1, ParamType::List, testsweeper::no_data_flag, 0,  inf,  "matrix condition number" ),
    cond_used ("cond",   0, 1, ParamType::List, testsweeper::no_data_flag, 0,  inf,  "actual condition number used" ),
    condD     ("condD",  0, 1, ParamType::List, testsweeper::no_data_flag, 0,  inf,  "matrix D condition number" ),
    seed      ("seed",   0,    ParamType::List, -1,                        -1, std::numeric_limits<int64_t>::max(), "Randomization seed (-1 randomizes the seed for each matrix)"),
    label     ("A",      2,    ParamType::Output, 0,                       0,  INT_MAX, "index labeling matrix" ),

    marked_( false )
{
}

// -----------------------------------------------------------------------------
/// Marks matrix params as used.
void MatrixParams::mark()
{
    marked_ = true;

    kind();
    cond();
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
    if (std::isnan( cond_used() )) {
        lbl += "unknown";
    }
    else {
        snprintf( buf, sizeof(buf), "= %.5g", cond_used() );
        lbl += buf;
    }
    if (! std::isnan( cond() )
            && (cond() != cond_used() || std::isnan( cond_used() ))) {
        lbl += " (ignoring --cond)";
    }

    // Add condD.
    if (! std::isnan( condD() )) {
        snprintf( buf, sizeof(buf), ", cond(D) = %.5g", condD() );
        lbl += buf;
    }
    // todo: warn if condD is set but not used. Add condD_used like cond_used.

    // Get label index from matrix_labels, or add to matrix_labels.
    if (matrix_labels.count( lbl ) == 1) {  // map::contains is c++20
        label() = matrix_labels[ lbl ];
    }
    else {
        label() = matrix_labels.size() + 1;
        matrix_labels[ lbl ] = label();
    }
}
