// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <string>
#include <map>

#ifndef SLATE_MATGEN_PARAMS_HH
#define SLATE_MATGEN_PARAMS_HH

extern std::map< std::string, int > matrix_labels;

namespace slate {

class MatgenParams {
public:
    int64_t verbose;
    std::string kind;
    double cond_request;
    double cond_actual;
    double condD;
    int64_t seed;
    int64_t label;
    void generate_label();
    bool marked;
};

} // slate namespace

#endif // SLATE_MATGEN_PARAMS_HH

