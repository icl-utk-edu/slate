// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_MATRIX_PARAMS_HH
#define SLATE_MATRIX_PARAMS_HH

#include "testsweeper.hh"

#include <map>
#include <string>

extern std::map< std::string, int > matrix_labels;

// =============================================================================
class MatrixParams
{
public:
    MatrixParams();

    void mark();

    bool marked() const
    {
        return marked_;
    }

    void generate_label();

    int64_t verbose;

    // ---- test matrix generation parameters
    testsweeper::ParamString kind;
    testsweeper::ParamScientific cond, cond_used;
    testsweeper::ParamScientific condD;
    testsweeper::ParamInt seed;
    testsweeper::ParamInt label;
    bool marked_;

    //--------------------------------------------------------------------------
    /// Copies the value of all members (kind, cond, ...) from y.
    /// This doesn't copy the iterator index, which would create an infinite
    /// loop in main tester.
    ///
    /// @param[in] y
    ///     MatrixParams to copy.
    ///
    MatrixParams& operator = ( MatrixParams const& y )
    {
        verbose     = y.verbose;
        kind()      = y.kind();
        cond()      = y.cond();
        cond_used() = y.cond_used();
        condD()     = y.condD();
        seed()      = y.seed();
        return *this;
    }
};

//------------------------------------------------------------------------------
/// @return true if a and b are bitwise the same. True if a and b are
/// both the same NaN value, unlike (a == b) which is false for NaNs.
inline bool same( double a, double b )
{
    return memcmp( &a, &b, sizeof(double) ) == 0;
}

//------------------------------------------------------------------------------
/// @return true if x and y are equal,
/// i.e., all members (kind, cond, ...) are equal.
inline bool operator == ( MatrixParams const& x, MatrixParams const& y )
{
    return x.kind() == y.kind()
           && same( x.cond(),      y.cond()      )
           && same( x.cond_used(), y.cond_used() )
           && same( x.condD(),     y.condD()     );
}

//------------------------------------------------------------------------------
/// @return true if x and y are not equal.
inline bool operator != ( MatrixParams const& x, MatrixParams const& y )
{
    return ! (x == y);
}

#endif // SLATE_MATRIX_PARAMS_HH
