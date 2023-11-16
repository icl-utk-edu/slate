// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_SET_LAMBDAS_HH
#define SLATE_SET_LAMBDAS_HH

#include <vector>
#include <limits>
#include <complex>
#include <chrono>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include "matrix_params.hh"
#include "generate_matrix.hh"
#include "random.hh"

namespace slate {

void set(std::function< scalar_t (int64_t i, int64_t j) > const& value,
         Matrix<scalar_t>& A,
         Options const& opts);
} // namespace slate

#endif // SLATE_SET_LAMBDAS_HH
