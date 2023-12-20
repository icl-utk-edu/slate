// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <string>


#ifndef SLATE_MATGEN_PARAMS_HH
#define SLATE_MATGEN_PARAMS_HH

namespace slate {

class MatgenParams {
    public: //setting to public, class starts as private
        int64_t verbose;
        std::string kind;
        double cond_request;
        double cond_actual;
        double condD;
        int64_t seed;
        bool marked;

        //void generate_label(); 	
};


} // slate namespace

#endif

