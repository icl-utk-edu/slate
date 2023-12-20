// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.


#ifndef SLATE_GENERATE_MATGEN_HH
#define SLATE_GENERATE_MATGEN_HH


#include "test.hh"
#include "../matgen/matgen_params.hh"

namespace slate {

template <typename matrix_type>
void generate_matrix(
    MatrixParams& params,
    matrix_type& A,
    slate::Options const& opts = slate::Options() )
{
    MatgenParams mg_params;
    mg_params.kind = params.kind();
    mg_params.cond_request = params.cond_request();
    mg_params.condD = params.condD();
    mg_params.seed = params.seed();
    mg_params.marked = params.marked();

    //mg_params.generate_label() = params.generate_label();

    generate_matrix( mg_params, A, opts);

    //label will be put here as output as well
    mg_params.cond_actual = params.cond_actual();
}

} // namespace slate

#endif // SLATE_GENERATE_MATGEN_HH

