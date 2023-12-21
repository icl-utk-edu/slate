// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/generate_matrix.hh"

#include <string>
#include <map>
#include <cmath>

std::map< std::string, int > matrix_labels;
//------------------------------------------------------------------------------
/// Generates a label string for the current matrix type and sets the
/// label index. If it's a new matrix type, adds it to the global
/// `matrix_labels` dictionary of labels.
///
namespace slate {

void MatgenParams::generate_label()
{
    char buf[ 80 ];
    std::string lbl = kind;

    // Add cond.
    lbl += ", cond ";
    if (std::isnan( cond_actual )) {
        lbl += "unknown";
    }
    else {
       snprintf( buf, sizeof(buf), "= %.5g", cond_actual );
       lbl += buf;
    }
    if (! std::isnan( cond_request )
           && (cond_request != cond_actual || std::isnan( cond_actual ))) {
        lbl += " (ignoring --cond)";
    }

    // Add condD.
    if (! std::isnan( condD )) {
        snprintf( buf, sizeof(buf), ", cond(D) = %.5g", condD );
        lbl += buf;
    }
    // todo: warn if condD is set but not used. Add condD_actual like cond_actual.

    // Get label index from matrix_labels, or add to matrix_labels.
    if (matrix_labels.count( lbl ) == 1) {  // map::contains is c++20
        label = matrix_labels[ lbl ];
    }
    else {
        label = matrix_labels.size() + 1;
        matrix_labels[ lbl ] = label;
    }
}

} // namespace slate
