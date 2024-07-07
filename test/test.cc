// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <complex>

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "test.hh"
#include "slate/slate.hh"
#include "slate/generate_matrix.hh"

// -----------------------------------------------------------------------------
using testsweeper::ParamType;
using testsweeper::DataType;
using testsweeper::DataType_help;

using testsweeper::ansi_bold;
using testsweeper::ansi_red;
using testsweeper::ansi_normal;

using blas::Layout, blas::Layout_help;
using blas::Side,   blas::Side_help;
using blas::Uplo,   blas::Uplo_help;
using blas::Op,     blas::Op_help;
using blas::Diag,   blas::Diag_help;

using lapack::itype_help;
using lapack::Job;
using lapack::Job_eig_help;
using lapack::Job_eig_left_help;
using lapack::Job_eig_right_help;
using lapack::Job_svd_left_help;
using lapack::Job_svd_right_help;
using lapack::Range,      lapack::Range_help;
using lapack::Norm,       lapack::Norm_help;
using lapack::MatrixType, lapack::MatrixType_help;
using lapack::Factored,   lapack::Factored_help;
using lapack::Direction,  lapack::Direction_help;
using lapack::StoreV,     lapack::StoreV_help;
using lapack::Equed,      lapack::Equed_help;

using slate::GridOrder,    slate::GridOrder_help;
using slate::MethodCholQR, slate::MethodCholQR_help;
using slate::MethodEig,    slate::MethodEig_help;
using slate::MethodGels,   slate::MethodGels_help;
using slate::MethodGemm,   slate::MethodGemm_help;
using slate::MethodHemm,   slate::MethodHemm_help;
using slate::MethodLU,     slate::MethodLU_help;
using slate::MethodTrsm,   slate::MethodTrsm_help;
using slate::NormScope,    slate::NormScope_help;
using slate::Origin,       slate::Origin_help;
using slate::Target,       slate::Target_help;

const ParamType PT_Value = ParamType::Value;
const ParamType PT_List  = ParamType::List;
const ParamType PT_Out   = ParamType::Output;

const double no_data = testsweeper::no_data_flag;
const char*  pi_rt2i = "3.141592653589793 + 1.414213562373095i";
const char*  e_rt3i  = "2.718281828459045 + 1.732050807568877i";
const double pi      = 3.141592653589793;
const double e       = 2.718281828459045;

// Macro to add indent string to parameter help.
#define indent "                     "

// -----------------------------------------------------------------------------
// each section must have a corresponding entry in section_names
enum Section {
    newline = 0,  // zero flag forces newline
    blas3,
    gesv,
    posv,
    sysv,
    hesv,
    gels,
    qr,
    heev,
    sygv,
    geev,
    svd,
    aux,
    aux_norm,
    aux_householder,
    aux_gen,
    num_sections,  // last
};

const char* section_names[] = {
    "",  // none
    "Level 3 BLAS",
    "LU",
    "Cholesky",
    "symmetric indefinite",
    "Hermitian indefinite",
    "least squares",
    "QR, LQ, QL, RQ",
    "symmetric eigenvalues",
    "generalized symmetric eigenvalues",
    "non-symmetric eigenvalues",
    "singular value decomposition (SVD)",
    "auxiliary",
    "matrix norms",
    "auxiliary - Householder",
    "auxiliary - matrix generation",
};

// { "", nullptr, Section::newline } entries force newline in help
std::vector< testsweeper::routines_t > routines = {
    // -----
    // Level 3 BLAS
    { "gemm",               test_gemm,         Section::blas3 },
    { "gemmA",              test_gemm,         Section::blas3 },
    { "gemmC",              test_gemm,         Section::blas3 },
    { "gbmm",               test_gbmm,         Section::blas3 },
    { "",                   nullptr,           Section::newline },

    { "hemm",               test_hemm,         Section::blas3 },
    { "hemmA",              test_hemm,         Section::blas3 },
    { "hemmC",              test_hemm,         Section::blas3 },
    { "hbmm",               test_hbmm,         Section::blas3 },
    { "herk",               test_herk,         Section::blas3 },
    { "her2k",              test_her2k,        Section::blas3 },
    { "",                   nullptr,           Section::newline },

    { "symm",               test_symm,         Section::blas3 },
    { "syrk",               test_syrk,         Section::blas3 },
    { "syr2k",              test_syr2k,        Section::blas3 },
    { "",                   nullptr,           Section::newline },

    { "trmm",               test_trmm,         Section::blas3 },
    { "trsm",               test_trsm,         Section::blas3 },
    { "trsmA",              test_trsm,         Section::blas3 },
    { "trsmB",              test_trsm,         Section::blas3 },
    { "tbsm",               test_tbsm,         Section::blas3 },

    // -----
    // LU
    { "gesv",               test_gesv,         Section::gesv },
    { "gesv_nopiv",         test_gesv,         Section::gesv },
    { "gesv_tntpiv",        test_gesv,         Section::gesv },
    { "gesv_mixed",         test_gesv,         Section::gesv },
    { "gesv_mixed_gmres",   test_gesv,         Section::gesv },
    { "gesv_rbt",           test_gesv,         Section::gesv },
    { "gbsv",               test_gbsv,         Section::gesv },
    { "",                   nullptr,           Section::newline },

    { "getrf",              test_gesv,          Section::gesv },
    { "getrf_nopiv",        test_gesv,          Section::gesv },
    { "getrf_tntpiv",       test_gesv,          Section::gesv },
    { "gbtrf",              test_gbsv,          Section::gesv },
    { "",                   nullptr,            Section::newline },

    { "getrs",              test_gesv,         Section::gesv },
    { "getrs_nopiv",        test_gesv,         Section::gesv },
    { "getrs_tntpiv",       test_gesv,         Section::gesv },
    { "gbtrs",              test_gbsv,         Section::gesv },
    { "",                   nullptr,           Section::newline },

    { "getri",              test_getri,        Section::gesv },
    { "getriOOP",           test_getri,        Section::gesv },
    { "",                   nullptr,           Section::newline },

    { "trtri",              test_trtri,        Section::gesv },
    { "",                   nullptr,           Section::newline },
    { "gecondest",          test_gecondest,    Section::gesv },

    // -----
    // Cholesky
    { "posv",               test_posv,         Section::posv },
    { "posv_mixed",         test_posv,         Section::posv },
    { "posv_mixed_gmres",   test_posv,         Section::posv },
    { "pbsv",               test_pbsv,         Section::posv },
    { "",                   nullptr,           Section::newline },

    { "potrf",              test_posv,         Section::posv },
    { "pbtrf",              test_pbsv,         Section::posv },
    { "",                   nullptr,           Section::newline },

    { "potrs",              test_posv,         Section::posv },
    { "pbtrs",              test_pbsv,         Section::posv },
    { "",                   nullptr,           Section::newline },

    { "potri",              test_potri,        Section::posv },
    { "",                   nullptr,           Section::newline },
    { "pocondest",          test_pocondest,    Section::posv },

    // -----
    // symmetric indefinite
    //{ "sysv",                test_sysv,         Section::sysv },
    //{ "",                    nullptr,           Section::newline },

    //{ "sytrf",               test_sytrf,        Section::sysv },
    //{ "",                    nullptr,           Section::newline },

    //{ "sytrs",               test_sytrs,        Section::sysv },
    //{ "",                    nullptr,           Section::newline },

    // -----
    // Hermitian indefinite
    { "hesv",                test_hesv,         Section::hesv },
    { "",                    nullptr,           Section::newline },

    { "hetrf",               test_hesv,         Section::hesv },
    { "",                    nullptr,           Section::newline },

    { "hetrs",               test_hesv,         Section::hesv },
    { "",                    nullptr,           Section::newline },

    // -----
    // least squares
    { "gels",                test_gels,         Section::gels },
    { "",                    nullptr,           Section::newline },

    // -----
    // QR, LQ, RQ, QL
    { "geqrf",              test_geqrf,     Section::qr },
    { "cholqr",             test_geqrf,     Section::qr },
    { "gelqf",              test_gelqf,     Section::qr },
    //{ "geqlf",              test_geqlf,     Section::qr },
    //{ "gerqf",              test_gerqf,     Section::qr },
    //{ "",                   nullptr,        Section::newline },

    //{ "ungqr",              test_ungqr,     Section::qr },
    //{ "unglq",              test_unglq,     Section::qr },
    //{ "ungql",              test_ungql,     Section::qr },
    //{ "ungrq",              test_ungrq,     Section::qr },
    //{ "",                   nullptr,        Section::newline },

    { "unmqr",              test_unmqr,     Section::qr },
    //{ "unmlq",              test_unmlq,     Section::qr },
    //{ "unmql",              test_unmql,     Section::qr },
    //{ "unmrq",              test_unmrq,     Section::qr },
    { "",                   nullptr,        Section::newline },
    { "trcondest",          test_trcondest, Section::qr },

    // -----
    // symmetric/Hermitian eigenvalues
    { "heev",               test_heev,         Section::heev },
    { "sterf",              test_sterf,        Section::heev },
    { "steqr2",             test_steqr2,       Section::heev },
    { "",                   nullptr,           Section::newline },

    { "stedc",              test_stedc,          Section::heev },
    { "stedc_deflate",      test_stedc_deflate,  Section::heev },
    { "stedc_secular",      test_stedc_secular,  Section::heev },
    { "stedc_sort",         test_stedc_sort,     Section::heev },
    { "stedc_z_vector",     test_stedc_z_vector, Section::heev },
    { "",                   nullptr,             Section::newline },

    { "he2hb",              test_he2hb,        Section::heev },
    { "unmtr_he2hb",        test_unmtr_he2hb,  Section::heev },
    { "",                   nullptr,           Section::newline },

    { "hb2st",              test_hb2st,        Section::heev },
    { "unmtr_hb2st",        test_unmtr_hb2st,  Section::heev },
    { "",                   nullptr,           Section::newline },

    // -----
    // generalized symmetric/Hermitian eigenvalues
    { "hegv",               test_hegv,         Section::sygv },
    { "hegst",              test_hegst,        Section::sygv },
    { "",                   nullptr,           Section::newline },

    // -----
    // non-symmetric eigenvalues

    // -----
    // SVD
    { "svd",                test_svd,          Section::svd },
    { "ge2tb",              test_ge2tb,        Section::svd },
    { "tb2bd",              test_tb2bd,        Section::svd },
    { "bdsqr",              test_bdsqr,        Section::svd },
    //{ "unmbr_tb2bd",        test_unmbr_tb2bd,  Section::svd },
    { "",                   nullptr,           Section::newline },

    // -----
    // matrix norms
    { "genorm",             test_norm,         Section::aux_norm },
    { "gbnorm",             test_gbnorm,       Section::aux_norm },
    { "",                   nullptr,           Section::newline },

    { "henorm",             test_norm,         Section::aux_norm },
    { "hbnorm",             test_hbnorm,       Section::aux_norm },
    { "",                   nullptr,           Section::newline },

    { "synorm",             test_norm,         Section::aux_norm },
    { "",                   nullptr,           Section::newline },

    { "trnorm",             test_norm,         Section::aux_norm },
    { "tznorm",             test_norm,         Section::aux_norm },
    { "",                   nullptr,           Section::newline },

    // -----
    // auxiliary
    { "add",                test_add,          Section::aux },
    { "tzadd",              test_add,          Section::aux },
    { "tradd",              test_add,          Section::aux },
    { "syadd",              test_add,          Section::aux },
    { "headd",              test_add,          Section::aux },
    { "",                   nullptr,           Section::newline },

    { "copy",               test_copy,         Section::aux },
    { "tzcopy",             test_copy,         Section::aux },
    { "trcopy",             test_copy,         Section::aux },
    { "sycopy",             test_copy,         Section::aux },
    { "hecopy",             test_copy,         Section::aux },
    { "",                   nullptr,           Section::newline },

    { "scale",              test_scale,        Section::aux },
    { "tzscale",            test_scale,        Section::aux },
    { "trscale",            test_scale,        Section::aux },
    { "syscale",            test_scale,        Section::aux },
    { "hescale",            test_scale,        Section::aux },
    { "",                   nullptr,           Section::newline },

    { "scale_row_col",      test_scale_row_col, Section::aux },
    { "",                   nullptr,           Section::newline },

    { "set",                test_set,          Section::aux },
    { "tzset",              test_set,          Section::aux },
    { "trset",              test_set,          Section::aux },
    { "syset",              test_set,          Section::aux },
    { "heset",              test_set,          Section::aux },
    { "",                   nullptr,           Section::newline },
};

// -----------------------------------------------------------------------------
// Params class
// List of parameters

Params::Params():
    ParamsBase(),

    // w = width
    // p = precision
    //----- test framework parameters
    //          name,         w, type, default, valid, help
    check     ( "check",      0, PT_Value, 'y', "ny", "check the results" ),
    error_exit( "error-exit", 0, PT_Value, 'n', "ny", "check error exits" ),
    ref       ( "ref",        0, PT_Value, 'n', "nyo", "run reference; sometimes check implies ref" ),
    trace     ( "trace",      0, PT_Value, 'n', "ny",  "enable/disable traces" ),
    trace_scale( "trace-scale", 0, 0, PT_Value, 1e3, 1e-3, 1e6, "horizontal scale for traces, in pixels per sec" ),

    //          name,         w, p, type, default,  min,  max, help
    tol       ( "tol",        0, 0, PT_Value,  50,    1, 1000, "tolerance (e.g., error < tol*epsilon to pass)" ),
    repeat    ( "repeat",     0,    PT_Value,   1,    1, 1000, "times to repeat each test" ),
    verbose   ( "verbose",    0,    PT_Value,   0,    0,    4,
                "verbose level:\n"
                indent "0: no printing (default)\n"
                indent "1: print metadata only (dimensions, uplo, etc.)\n"
                indent "2: print first & last edgeitems rows & cols from the four corner tiles\n"
                indent "3: print 4 corner elements of every tile\n"
                indent "4: print full matrix" ),

    print_edgeitems( "print-edgeitems", 0, PT_Value, 16,   1, 64,
                     "for verbose=2, number of first & last rows & cols "
                     "to print from the four corner tiles" ),
    print_width    ( "print-width",     0, PT_Value, 10,   7, 24,
                     "minimum number of characters to print per value" ),
    print_precision( "print-precision", 0, PT_Value, 4,    1, 17,
                     "number of digits to print after the decimal point" ),

    timer_level( "timer-level", 0,  PT_Value,   1,    1,    2,
                 "timer level of detail:\n"
                 indent "1: driver routine (e.g., gels; default)\n"
                 indent "2: computational routines (e.g., geqrf, unmqr, trsm inside gels)" ),

    extended  ( "extended",   0,    PT_Value,   0,    0, 1000, "number of extended tests" ),
    cache     ( "cache",      0,    PT_Value,  20,    1, 1024, "total cache size, in MiB" ),
    debug_rank( "debug-rank", 0,    PT_Value,  -1,    0,  1e6,
                "given MPI rank waits for debugger (gdb/lldb) to attach; "
                "use MPI size for all ranks to wait" ),

    //----- routine parameters, enums
    //          name,         w, type,    default, help
    datatype  ( "type",       4, PT_List, DataType::Double, DataType_help ),
    origin    ( "origin",     6, PT_List, Origin::Host, Origin_help ),
    target    ( "target",     6, PT_List, Target::HostTask, Target_help ),
    hold_local_workspace( "hold-local-workspace",
                              0, PT_List, 'n', "ny", "do not erase tiles in local workspace" ),

    method_cholqr( "cholQR",  6, PT_List, MethodCholQR::Auto, MethodCholQR_help ),
    method_eig   ( "eig",     3, PT_List, MethodEig::DC, MethodEig_help ),
    method_gels  ( "gels",    6, PT_List, MethodGels::QR, MethodGels_help ),
    method_gemm  ( "gemm",    4, PT_List, MethodGemm::Auto, MethodGemm_help ),
    method_hemm  ( "hemm",    4, PT_List, MethodHemm::Auto, MethodHemm_help ),
    method_lu    ( "lu",      5, PT_List, MethodLU::PartialPiv, MethodLU_help ),
    method_trsm  ( "trsm",    4, PT_List, MethodTrsm::Auto, MethodTrsm_help ),

    grid_order( "go",         3, PT_List, GridOrder::Col, "(go) MPI grid order: c=Col, r=Row" ),
    dev_order ( "do",         3, PT_List, GridOrder::Row, "(do) Device grid order: c=Col, r=Row" ),

    // BLAS & LAPACK options
    layout    ( "layout",     6, PT_List, Layout::ColMajor, Layout_help ),
    itype     ( "itype",      5, PT_List, 1, 1, 3, itype_help ),
    jobz      ( "jobz",       5, PT_List, Job::NoVec, Job_eig_help ),
    jobvl     ( "jobvl",      5, PT_List, Job::NoVec, Job_eig_left_help ),
    jobvr     ( "jobvr",      5, PT_List, Job::NoVec, Job_eig_right_help ),
    jobu      ( "jobu",       9, PT_List, Job::NoVec, Job_svd_left_help ),
    jobvt     ( "jobvt",      9, PT_List, Job::NoVec, Job_svd_right_help ),
    // range is set by vl, vu, il, iu, fraction
    range     ( "range",      9, PT_List, Range::All, Range_help ),
    norm      ( "norm",       4, PT_List, Norm::One, Norm_help ),
    scope     ( "scope",      7, PT_List, NormScope::Matrix, NormScope_help ),
    side      ( "side",       6, PT_List, Side::Left, Side_help ),
    uplo      ( "uplo",       6, PT_List, Uplo::Lower, Uplo_help ),
    trans     ( "trans",      7, PT_List, Op::NoTrans, Op_help ),
    transA    ( "transA",     7, PT_List, Op::NoTrans, Op_help ),
    transB    ( "transB",     7, PT_List, Op::NoTrans, Op_help ),
    diag      ( "diag",       7, PT_List, Diag::NonUnit, Diag_help ),
    direction ( "direction",  8, PT_List, Direction::Forward, Direction_help ),
    storev    ( "storev",     7, PT_List, StoreV::Columnwise, StoreV_help ),
    equed     ( "equed",      5, PT_List, Equed::Both, Equed_help ),

    //----- routine parameters, numeric
    //          name,         w, p, type,    default,  min,  max, help
    dim       ( "dim",        6,    PT_List,             0, 1e10, "m by n by k dimensions" ),
    kd        ( "kd",         6,    PT_List,      10,    0,  1e6, "bandwidth" ),
    kl        ( "kl",         6,    PT_List,      10,    0,  1e6, "lower bandwidth" ),
    ku        ( "ku",         6,    PT_List,      10,    0,  1e6, "upper bandwidth" ),
    nrhs      ( "nrhs",       6,    PT_List,      10,    0, 1e10, "number of right hand sides" ),
    nb        ( "nb",         4,    PT_List,     384,    0,  1e6, "block size" ),
    ib        ( "ib",         2,    PT_List,      32,    0,  1e6, "inner blocking" ),

    vl        ( "vl",         6, 3, PT_List,    -inf, -inf,  inf, "lower bound of eigen/singular values to find" ),
    vu        ( "vu",         6, 3, PT_List,     inf, -inf,  inf, "upper bound of eigen/singular values to find" ),
    // input il, iu, or fraction; output {il, iu}_out adjusted for matrix size or set by fraction
    il        ( "il",         0,    PT_List,       1,    1, 1e10, "1-based index of smallest eigen/singular value to find" ),
    iu        ( "iu",         0,    PT_List,      -1,   -1, 1e10, "1-based index of largest  eigen/singular value to find; -1 is all" ),
    il_out    ( "il",         6,    PT_Out,        1,    1, 1e10, "1-based index of smallest eigen/singular value to find (actual value used)" ),
    iu_out    ( "iu",         6,    PT_Out,       -1,   -1, 1e10, "1-based index of largest  eigen/singular value to find (actual value used)" ),
    fraction_start( "fraction-start",
                              0, 0, PT_List,       0,    0,    1, "index of smallest eigen/singular value to find, as fraction of n; sets il = 1 + fraction_start*n" ),
    fraction  ( "fraction",   0, 0, PT_List,       1,    0,    1, "fraction of eigen/singular values to find; sets iu = il - 1 + fraction*n" ),

    alpha     ( "alpha",      3, 1, PT_List, pi_rt2i, -inf,  inf, "scalar alpha" ),
    beta      ( "beta",       3, 1, PT_List,  e_rt3i, -inf,  inf, "scalar beta" ),
    incx      ( "incx",       4,    PT_List,       1, -1e3,  1e3, "stride of x vector" ),
    incy      ( "incy",       4,    PT_List,       1, -1e3,  1e3, "stride of y vector" ),

    // SLATE options
    grid      ( "grid",       3,    PT_List,   "1x1",    0,  1e6, "MPI grid p by q dimensions" ),
    lookahead ( "la",         2,    PT_List,       1,    0,  1e6, "(la) number of lookahead panels" ),
    panel_threads( "pt",      2,    PT_List, std::max( omp_get_max_threads() / 2, 1 ),
                                                         0,  1e6, "(pt) max number of threads used in panel; default omp_num_threads / 2" ),
    nonuniform_nb( "nonuniform-nb",
                              0,    PT_List, 'n', "ny", "generate matrix with nonuniform tile sizes" ),
    pivot_threshold(
                "threshold",  6, 2, PT_List, 1.0,   0.0,     1.0, "threshold for pivoting a remote row" ),
    deflate   ( "deflate",   12,    PT_List, "",
                "multiple space-separated indices or index pairs (/-separated)"
                " to deflate, e.g., --deflate '1 2/4 3/5'" ),
    itermax   ( "itermax",    7,    PT_List, 30,     -1, 1e6, "Maximum number of iterations for refinement" ),
    fallback  ( "fallback",   0,    PT_List, 'y',  "ny",      "If refinement fails, fallback to a robust solver" ),
    depth     ( "depth",      5,    PT_List,  2,      0, 1e3, "Number of butterflies to apply" ),

    //----- output parameters
    // min, max are ignored
    // error:   %8.2e allows 9.99e-99
    // time:    %9.3f allows 99999.999 s = 2.9 days
    // gflops: %12.3f allows 99999999.999 Gflop/s = 100 Pflop/s
    //          name,         w, p, type,   default, min, max, help
    value     ( "value",      9, 3, PT_Out, no_data, 0, 0, "numerical value" ),
    value2    ( "value2",     9, 3, PT_Out, no_data, 0, 0, "numerical value" ),
    value3    ( "value3",     9, 3, PT_Out, no_data, 0, 0, "numerical value" ),

    error     ( "error",      8, 2, PT_Out, no_data, 0, 0, "numerical error" ),
    error2    ( "error2",     8, 2, PT_Out, no_data, 0, 0, "numerical error" ),
    error3    ( "error3",     8, 2, PT_Out, no_data, 0, 0, "numerical error" ),
    error4    ( "error4",     8, 2, PT_Out, no_data, 0, 0, "numerical error" ),
    error5    ( "error5",     8, 2, PT_Out, no_data, 0, 0, "numerical error" ),
    ortho     ( "orth.",      8, 2, PT_Out, no_data, 0, 0, "orthogonality error" ),
    ortho_U   ( "U orth.",    8, 2, PT_Out, no_data, 0, 0, "U orthogonality error" ),
    ortho_V   ( "V orth.",    8, 2, PT_Out, no_data, 0, 0, "V orthogonality error" ),

    time      ( "time (s)",   9, 3, PT_Out, no_data, 0, 0, "time to solution" ),
    gflops    ( "gflop/s",   12, 3, PT_Out, no_data, 0, 0, "Gflop/s rate" ),
    gbytes    ( "gbyte/s",   12, 3, PT_Out, no_data, 0, 0, "Gbyte/s rate" ),
    iters     ( "iters",      5,    PT_Out, 0,       0, 0, "iterations to solution" ),

    time2     ( "time (s)",   9, 3, PT_Out, no_data, 0, 0, "extra timer" ),
    gflops2   ( "gflop/s",   12, 3, PT_Out, no_data, 0, 0, "Gflop/s rate" ),
    gbytes2   ( "gbyte/s",   12, 3, PT_Out, no_data, 0, 0, "Gbyte/s rate" ),

    time3     ( "time (s)",   9, 3, PT_Out, no_data, 0, 0, "extra timer" ),
    time4     ( "time (s)",   9, 3, PT_Out, no_data, 0, 0, "extra timer" ),
    time5     ( "time (s)",   9, 3, PT_Out, no_data, 0, 0, "extra timer" ),
    time6     ( "time (s)",   9, 3, PT_Out, no_data, 0, 0, "extra timer" ),
    time7     ( "time (s)",   9, 3, PT_Out, no_data, 0, 0, "extra timer" ),
    time8     ( "time (s)",   9, 3, PT_Out, no_data, 0, 0, "extra timer" ),
    time9     ( "time (s)",   9, 3, PT_Out, no_data, 0, 0, "extra timer" ),
    time10    ( "time (s)",   9, 3, PT_Out, no_data, 0, 0, "extra timer" ),
    time11    ( "time (s)",   9, 3, PT_Out, no_data, 0, 0, "extra timer" ),
    time12    ( "time (s)",   9, 3, PT_Out, no_data, 0, 0, "extra timer" ),
    time13    ( "time (s)",   9, 3, PT_Out, no_data, 0, 0, "extra timer" ),

    ref_time  ( "ref time (s)",  9, 3, PT_Out, no_data, 0, 0, "reference time to solution" ),
    ref_gflops( "ref gflop/s",  12, 3, PT_Out, no_data, 0, 0, "reference Gflop/s rate" ),
    ref_gbytes( "ref gbyte/s",  12, 3, PT_Out, no_data, 0, 0, "reference Gbyte/s rate" ),
    ref_iters ( "ref iters",     5,    PT_Out, 0,       0, 0, "reference iterations to solution" ),

    // default -1 means "no check"
    //          name,         w, type, default, min, max, help
    okay      ( "status",     6, PT_Out,    -1, 0, 0, "success indicator" ),
    msg       ( "",           1, PT_Out,    "",       "error message" )
{
    // set header different than command line prefix
    lookahead.name("la", "lookahead");
    panel_threads.name("pt", "panel-threads");
    grid_order.name("go", "grid-order");
    dev_order.name("do", "dev-order");

    // Change name for the methods to use less space in the stdout
    method_cholqr.name("cholQR", "method-cholQR");
    method_eig.name("eig", "method-eig");
    method_gels.name("gels", "method-gels");
    method_gemm.name("gemm", "method-gemm");
    method_hemm.name("hemm", "method-hemm");
    method_lu.name("lu", "method-lu");
    method_trsm.name("trsm", "method-trsm");

    // change names of matrix B's params
    matrixB.kind.name( "matrixB" );
    matrixB.cond_request.name( "condB" );
    matrixB.condD.name( "condD_B" );
    matrixB.seed.name( "seedB" );
    matrixB.label.name( "B" );

    // change names of matrix C's params
    matrixC.kind.name( "matrixC" );
    matrixC.cond_request.name( "condC" );
    matrixC.condD.name( "condD_C" );
    matrixC.seed.name( "seedC" );
    matrixC.label.name( "C" );

    // mark standard set of output fields as used
    okay();
    error();
    time();

    // mark framework parameters as used, so they will be accepted on the command line
    check();
    error_exit();
    ref();
    trace();
    trace_scale();
    tol();
    repeat();
    verbose();
    cache();
    debug_rank();
    print_edgeitems();
    print_width();
    print_precision();

    // change names of grid elements
    grid.names("p", "q");
    grid.width( 3 );

    // routine's parameters are marked by the test routine; see main
}

// -----------------------------------------------------------------------------
/// Prints an error in an MPI-aware fashion.
/// If some ranks have a non-empty error message, rank 0 prints one of them
/// (currently from the lowest rank), and all ranks return non-zero.
/// If all ranks have an empty error message, nothing is printed,
/// and all ranks return zero.
///
/// @param[in] msg
///     Error message on each rank. Empty string indicates no error.
///
/// @param[in] mpi_rank
///     MPI rank within comm. Rank 0 prints output.
///
/// @param[in] comm
///     MPI communicator.
///
/// @return zero if msg was empty on all ranks, otherwise non-zero.
///
int print_reduce_error(
    const std::string& msg, int mpi_rank, MPI_Comm comm)
{
    // reduction to determine first rank with an error
    typedef struct { int err, rank; } err_rank_t;
    int err = ! msg.empty();
    err_rank_t err_rank = { err, mpi_rank };
    err_rank_t err_first = { 0, 0 };
    MPI_Allreduce(&err_rank, &err_first, 1, MPI_2INT, MPI_MAXLOC, comm);

    if (err_first.err) {
        // count ranks with an error
        int root = 0;
        int cnt = 0;
        MPI_Reduce(&err, &cnt, 1, MPI_INT, MPI_SUM, root, comm);

        // first rank with error sends msg to root
        char buf[ 255 ];
        if (mpi_rank == err_first.rank) {
            snprintf(buf, sizeof(buf), "%s", msg.c_str());
            // if rank == root, nothing to send
            if (mpi_rank != root) {
                slate_mpi_call(
                    MPI_Send(buf, sizeof(buf), MPI_CHAR, root, 0, comm));
            }
        }
        else if (mpi_rank == root) {
            MPI_Status status;
            slate_mpi_call(
                MPI_Recv(buf, sizeof(buf), MPI_CHAR, err_first.rank, 0, comm,
                         &status));
        }

        // root prints msg
        if (mpi_rank == root) {
            fprintf(stderr,
                    "\n%s%sError on rank %d: %s. (%d ranks had some error.)%s\n",
                    ansi_bold, ansi_red,
                    err_first.rank, buf, cnt,
                    ansi_normal);
        }
    }

    return err_first.err;
}

// -----------------------------------------------------------------------------
int run(int argc, char** argv)
{
    using testsweeper::QuitException;

    // These may or may not be used; mark unused to silence warnings.
    blas_unused( pi_rt2i );
    blas_unused( e_rt3i  );
    blas_unused( pi      );
    blas_unused( e       );

    // check that all sections have names
    assert(sizeof(section_names) / sizeof(*section_names) == Section::num_sections);

    // MPI initializations
    int mpi_rank = 0, mpi_size = 0, provided = 0;
    slate_mpi_call(
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided));

    int status = 0;
    std::string msg;
    try {
        if (provided < MPI_THREAD_MULTIPLE)
            throw std::runtime_error("SLATE requires MPI_THREAD_MULTIPLE");

        slate_mpi_call(
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
        slate_mpi_call(
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
        bool print = (mpi_rank == 0);

        // print input so running `test [input] > out.txt` documents input
        if (print) {
            // Version, id.
            char buf[ 1024 ];
            int version = slate::version();
            snprintf( buf, sizeof(buf), "%% SLATE version %04d.%02d.%02d, id %s\n",
                      version / 10000, (version % 10000) / 100, version % 100,
                      slate::id() );
            std::string args = buf;

            // Input line.
            args += "% input:";
            for (int i = 0; i < argc; ++i) {
                args += ' ';
                args += argv[i];
            }
            args += "\n% ";

            // Date and time, MPI, OpenMP, CUDA specs.
            std::time_t now = std::time(nullptr);
            std::strftime( buf, sizeof( buf ), "%F %T", std::localtime( &now ) );
            args += buf;
            args += ", " + std::to_string( mpi_size ) + " MPI ranks";
            if (slate::gpu_aware_mpi())
                args += ", GPU-aware MPI";
            else
                args += ", CPU-only MPI";

            args += ", " + std::to_string( omp_get_max_threads() )
                  + " OpenMP threads";

            int num_devices = blas::get_device_count();
            if (num_devices > 0)
                args += ", " + std::to_string( num_devices ) +  " GPU devices";
            args += " per MPI rank\n";

            printf("%s", args.c_str());
            slate::trace::Trace::comment(args);
        }

        // Usage: test [params] routine
        if (argc < 2
            || strcmp( argv[argc-1], "-h" ) == 0
            || strcmp( argv[argc-1], "--help" ) == 0)
        {
            if (print)
                usage(argc, argv, routines, section_names);
            throw QuitException();
        }

        if (strcmp( argv[1], "--help-matrix" ) == 0) {
            slate::generate_matrix_usage();
            throw QuitException();
        }

        // find routine to test
        const char* routine = argv[ argc-1 ];
        testsweeper::test_func_ptr test_routine = find_tester( routine, routines );
        if (test_routine == nullptr) {
            if (print)
                usage(argc, argv, routines, section_names);
            throw std::runtime_error(
                std::string("routine ") + routine + " not found");
        }

        // mark fields that are used (run=false)
        Params params;
        params.routine = routine;
        test_routine(params, false);

        // Make default p x q grid as square as possible.
        // Worst case is p=1, q=mpi_size.
        int p = 1, q = 1;
        for (p = int(sqrt(mpi_size)); p > 0; --p) {
            q = int(mpi_size / p);
            if (p*q == mpi_size)
                break;
        }
        testsweeper::int3_t grid = { p, q, 1 };
        params.grid.set_default( grid );

        // parse parameters up to routine name.
        try {
            params.parse( routine, argc-2, argv+1 );
        }
        catch (const std::exception& ex) {
            if (print)
                params.help(routine);
            throw;
        }

        // After parsing parameters, call test routine again (with run=false)
        // to mark any new fields as used (e.g., timers).
        test_routine( params, false );

        slate_assert(params.grid.m() * params.grid.n() == mpi_size);

        slate::trace::Trace::pixels_per_second(params.trace_scale());

        // Wait for debugger to attach.
        // See https://www.open-mpi.org/faq/?category=debugging#serial-debuggers
        if (params.debug_rank() == mpi_rank
            || params.debug_rank() == mpi_size) {
            volatile int i = 0;
            char hostname[256];
            gethostname( hostname, sizeof(hostname) );
            printf( "MPI rank %d, pid %d on %s ready for debugger (gdb/lldb) to attach.\n"
                    "After attaching, step out to run() and set i=1, e.g.:\n"
                    "lldb -p %d\n"
                    "(lldb) break set -n __cxa_throw  # break on C++ exception\n"
                    "(lldb) thread step-out           # repeat\n"
                    "(lldb) expr i=1\n"
                    "(lldb) continue\n",
                    mpi_rank, getpid(), hostname, getpid() );
            fflush( stdout );
            while (0 == i)
                sleep(1);
        }
        slate_mpi_call( MPI_Barrier( MPI_COMM_WORLD ) );

        // run tests
        int repeat = params.repeat();
        testsweeper::DataType last_datatype = params.datatype();

        if (print)
            params.header();
        do {
            if (params.datatype() != last_datatype) {
                last_datatype = params.datatype();
                if (print)
                    printf("\n");
            }

            for (int iter = 0; iter < repeat; ++iter) {
                try {
                    test_routine(params, true);
                }
                catch (const std::exception& ex) {
                    msg = ex.what();
                }
                int err = print_reduce_error(msg, mpi_rank, MPI_COMM_WORLD);
                if (err)
                    params.okay() = false;
                if (print) {
                    params.print();
                    fflush(stdout);
                }
                status += ! params.okay();
                params.reset_output();
                msg.clear();
            }
            if (repeat > 1 && print) {
                printf("\n");
            }
        } while (params.next());

        if (print) {
            std::vector< std::string > sort_matrix_labels(
                    matrix_labels.size() + 1 );
            for (auto& name_label_pair : matrix_labels) {
                sort_matrix_labels[ name_label_pair.second ]
                    = name_label_pair.first;
            }
            printf( "\n%% Matrix kinds:\n" );
            for (size_t i = 1; i < sort_matrix_labels.size(); ++i) {
                printf( "%% %2lld: %s\n",
                        llong( i ), sort_matrix_labels[ i ].c_str() );
            }
            printf( "\n" );

            if (status) {
                printf( "%% %d tests FAILED: %s\n", status, routine );
            }
            else {
                printf( "%% All tests passed: %s\n", routine );
            }
        }

        // Exit status is only 8 bits, and reserve status = 251, ..., 255.
        status = std::min( status, 250 );
    }
    catch (const QuitException& ex) {
        // pass: no error to print
    }
    catch (const std::exception& ex) {
        msg = ex.what();
    }
    int err = print_reduce_error(msg, mpi_rank, MPI_COMM_WORLD);
    if (err)
        status = 254;

    slate_mpi_call(
        MPI_Finalize());

    if (mpi_rank == 0)
        return status;
    else
        return 0;
}

// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int status = 0;
    try {
        status = run(argc, argv);
    }
    catch (const std::exception& ex) {
        fprintf(stderr, "Error: %s\n", ex.what());
        status = -1;
    }
    catch (...) {
        fprintf(stderr, "Unknown error\n");
        status = -2;
    }
    return status;
}
