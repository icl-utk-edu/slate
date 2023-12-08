// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_GENERATE_TYPE_JORDAN_HH
#define SLATE_GENERATE_TYPE_JORDAN_HH


#include "slate/slate.hh"
#include "../test/test.hh"

#include <exception>
#include <string>
#include <vector>
#include <limits>
#include <complex>
#include <chrono>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include "../test/matrix_params.hh"
#include "slate/generate_matrix.hh"
#include "../test/random.hh"


namespace slate {

template <typename scalar_t>
void generate_jordan( slate::Matrix<scalar_t>& A,
                      slate::Options const& opts )
{
    int64_t mt = A.mt();
    int64_t nt = A.nt();

    for (int64_t i = 0; i < nt; ++i) {
        // Set 1 element from sub-diagonal tile to 1.
        if (i > 0) {
            if (A.tileIsLocal(i, i-1)) {
                A.tileGetForWriting( i, i-1, LayoutConvert::ColMajor );
                auto T = A(i, i-1);
                T.at(0, T.nb()-1) = 1.;
            }
        }
        // Set 1 element from sub-diagonal tile to 1.
        if (A.tileIsLocal(i, i)) {
            A.tileGetForWriting( i, i, LayoutConvert::ColMajor );
            auto T = A(i, i);
            auto len = T.nb();
            for (int j = 0; j < len-1; ++j) {
                T.at(j+1, j) = 1.;
            }
        }
    }

}

} // namespace slate

#endif // SLATE_GENERATE_TYPE_JORDAN_HH
