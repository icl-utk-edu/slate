// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"
#include "test.hh"

#include "scalapack_wrappers.hh"
#include "scalapack_copy.hh"
#include "print_matrix.hh"

#include "matrix_utils.hh"
#include "test_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

//------------------------------------------------------------------------------
template <typename matrix_type>
void test_matgen_work(Params& params, bool run)
{
    using scalar_t = typename matrix_type::value_type;
    //using real_t = blas::real_type<scalar_t>;
    //using blas::real;
    //using blas::imag;
    //using slate::ceildiv;

    // get & mark input values
    slate::Uplo uplo;
    if (std::is_same< matrix_type, slate::Matrix<scalar_t> >::value)
        uplo = slate::Uplo::General;
    else
        uplo = params.uplo();
    slate::Op trans = params.trans();
    slate::Diag diag = slate::Diag::NonUnit;
    int64_t m = params.dim.m();
    int64_t n;
    if (std::is_same< matrix_type, slate::TriangularMatrix<scalar_t> >::value
        || std::is_same< matrix_type, slate::SymmetricMatrix<scalar_t> >::value
        || std::is_same< matrix_type, slate::HermitianMatrix<scalar_t> >::value) {
        n = m;  // square
    }
    else {
        n = params.dim.n();
    }
    bool ref_only = params.ref() == 'o';
    bool ref = params.ref() == 'y' || ref_only;
    bool check = params.check() == 'y' && ! ref_only;
    //bool trace = params.trace() == 'y';
    slate::Target target = params.target();
    params.matrix.mark();

    mark_params_for_test_Matrix( params );

    // mark non-standard output values
    params.time();

    if (! run)
        return;

    // Check for common invalid combinations
    if (is_invalid_parameters( params )) {
        return;
    }

    slate::Options const opts =  {
        {slate::Option::Target, target}
    };

    auto A_alloc = allocate_test_Matrix<scalar_t>( check || ref, false, m, n, params );

    auto& Afull     = A_alloc.A;
    auto& Aref_full = A_alloc.Aref;
    //auto& Aref_data = A_alloc.Aref_data;

    slate::generate_matrix( params.matrix, Afull );

    if (check || ref) {
        copy_matrix( Afull, Aref_full );
    }

    // Cast to desired matrix type.
    matrix_type A = matrix_cast< matrix_type >( Afull, uplo, diag );

    if (trans == slate::Op::Trans)
        A = transpose( A );
    else if (trans == slate::Op::ConjTrans)
        A = conj_transpose( A );

    print_matrix( "Afull", Afull, params );
    print_matrix( "A", A, params );
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_matgen_dispatch(Params& params, bool run )
{
    std::string routine = params.routine;
    if (routine == "matgen") {
        test_matgen_work< slate::Matrix<scalar_t> >( params, run );
    }
    else {
        throw slate::Exception("unknown routine: " + routine);
    }
}

//------------------------------------------------------------------------------
void test_matgen(Params& params, bool run)
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_matgen_dispatch<float> (params, run);
            break;

        case testsweeper::DataType::Double:
            test_matgen_dispatch<double> (params, run);
            break;

        case testsweeper::DataType::SingleComplex:
            test_matgen_dispatch<std::complex<float>> (params, run);
            break;

        case testsweeper::DataType::DoubleComplex:
            test_matgen_dispatch<std::complex<double>> (params, run);
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}
