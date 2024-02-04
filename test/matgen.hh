// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_GENERATE_MATGEN_HH
#define SLATE_GENERATE_MATGEN_HH

#include "slate/generate_matrix.hh"

namespace slate {

template <typename matrix_type>
void generate_matrix(
    MatrixParams& params,
    matrix_type& A,
    slate::Options const& opts = slate::Options() )
{
    MatgenParams mg_params;
    mg_params.kind         = params.kind();
    mg_params.cond_request = params.cond_request();
    mg_params.condD        = params.condD();
    mg_params.seed         = params.seed();

    generate_matrix( mg_params, A, opts);

    params.cond_actual() = mg_params.cond_actual;

    if (params.marked_)
        params.generate_label();
}

} // namespace slate

#endif // SLATE_GENERATE_MATGEN_HH

