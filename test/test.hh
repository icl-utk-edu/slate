// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TEST_HH
#define SLATE_TEST_HH

#include "testsweeper.hh"
#include "blas.hh"
#include "lapack.hh"
#include "slate/slate.hh"
#include "matrix_params.hh"
#include "slate/generate_matrix.hh"
#include "matgen.hh"

#include <exception>
#include <complex>
#include <ctype.h>

//------------------------------------------------------------------------------
namespace slate {

enum class Origin : char {
    Host = 'H',
    ScaLAPACK = 'S',
    Devices = 'D',
};

extern const char* Origin_help;

//------------------------------------------------------------------------------
inline void from_string( std::string const& str, Origin* val )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "d" || str_ == "dev" || str_ == "device"
        || str_ == "devices")
        *val = Origin::Devices;
    else if (str_ == "h" || str_ == "host")
        *val = Origin::Host;
    else if (str_ == "s" || str_ == "scalapack" || str_ == "scalpk")
        *val = Origin::ScaLAPACK;
    else
        throw Exception( "unknown Origin: " + str );
}

//----------------------------------------
inline const char* to_c_string( Origin value )
{
    switch (value) {
        case Origin::Devices:   return "dev";
        case Origin::Host:      return "host";
        case Origin::ScaLAPACK: return "scalpk";
    }
    return "?";
}

//----------------------------------------
inline std::string to_string( Origin value )
{
    return to_c_string( value );
}

//----------------------------------------
/// Convert Origin to Target.
/// Host, ScaLAPACK => Target Host; Devices => Target Devices.
inline Target origin2target( Origin origin )
{
    switch (origin) {
        case Origin::Host:
        case Origin::ScaLAPACK:
            return Target::Host;

        case Origin::Devices:
            return Target::Devices;
    }
    throw Exception( "unknown origin" );
}

//------------------------------------------------------------------------------
extern const char* Target_help;

//----------------------------------------
inline void from_string( std::string const& str, Target* val )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "t" || str_ == "task")
        *val = Target::HostTask;
    else if (str_ == "n" || str_ == "nest")
        *val = Target::HostNest;
    else if (str_ == "b" || str_ == "batch")
        *val = Target::HostBatch;
    else if (str_ == "d" || str_ == "dev" || str_ == "device"
             || str_ == "devices")
        *val = Target::Devices;
    else if (str_ == "h" || str_ == "host")
        *val = Target::Host;
    else
        throw Exception( "unknown Target: " + str );
}

//----------------------------------------
inline const char* to_c_string( Target value )
{
    switch (value) {
        case Target::HostTask:  return "task";
        case Target::HostNest:  return "nest";
        case Target::HostBatch: return "batch";
        case Target::Devices:   return "dev";
        case Target::Host:      return "host";
    }
    return "?";
}

//----------------------------------------
inline std::string to_string( Target value )
{
    return to_c_string( value );
}

//------------------------------------------------------------------------------
extern const char* GridOrder_help;

//----------------------------------------
inline void from_string( std::string const& str, GridOrder* val )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "c" || str_ == "col")
        *val = GridOrder::Col;
    else if (str_ == "r" || str_ == "row")
        *val = GridOrder::Row;
    else
        throw Exception( "unknown GridOrder: " + str );
}

//----------------------------------------
inline const char* to_c_string( GridOrder value )
{
    switch (value) {
        case GridOrder::Col:     return "col";
        case GridOrder::Row:     return "row";
        case GridOrder::Unknown: return "un";
    }
    return "?";
}

//----------------------------------------
inline std::string to_string( GridOrder value )
{
    return to_c_string( value );
}

//------------------------------------------------------------------------------
extern const char* NormScope_help;

//----------------------------------------
inline void from_string( std::string const& str, NormScope* val )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );

    if (str_ == "m" || str_ == "matrix")
        *val = NormScope::Matrix;
    else if (str_ == "c" || str_ == "cols" || str_ == "columns")
        *val = NormScope::Columns;
    else if (str_ == "r" || str_ == "rows")
        *val = NormScope::Rows;
    else
        throw Exception( "unknown NormScope: " + str );
}

//----------------------------------------
inline const char* to_c_string( NormScope value )
{
    switch (value) {
        case NormScope::Matrix:  return "matrix";
        case NormScope::Columns: return "columns";
        case NormScope::Rows:    return "rows";
    }
    return "?";
}

//----------------------------------------
inline std::string to_string( NormScope value )
{
    return to_c_string( value );
}

} // namespace slate

//------------------------------------------------------------------------------
using llong = long long;

//------------------------------------------------------------------------------
class Params: public testsweeper::ParamsBase {
public:
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    Params();

    // Field members are explicitly public.
    // Order here determines output order.
    //----- test framework parameters
    testsweeper::ParamChar   check;
    testsweeper::ParamChar   error_exit;
    testsweeper::ParamChar   ref;
    testsweeper::ParamChar   trace;
    testsweeper::ParamDouble trace_scale;
    testsweeper::ParamDouble tol;
    testsweeper::ParamInt    repeat;
    testsweeper::ParamInt    verbose;
    testsweeper::ParamInt    print_edgeitems;
    testsweeper::ParamInt    print_width;
    testsweeper::ParamInt    print_precision;
    testsweeper::ParamInt    timer_level;
    testsweeper::ParamInt    extended;
    testsweeper::ParamInt    cache;
    testsweeper::ParamInt    debug_rank;
    std::string              routine;

    //----- routine parameters, enums
    testsweeper::ParamEnum< testsweeper::DataType > datatype;
    testsweeper::ParamEnum< slate::Origin >         origin;
    testsweeper::ParamEnum< slate::Target >         target;
    testsweeper::ParamChar                          hold_local_workspace;

    testsweeper::ParamEnum< slate::MethodCholQR >   method_cholqr;
    testsweeper::ParamEnum< slate::MethodEig >      method_eig;
    testsweeper::ParamEnum< slate::MethodGels >     method_gels;
    testsweeper::ParamEnum< slate::MethodGemm >     method_gemm;
    testsweeper::ParamEnum< slate::MethodHemm >     method_hemm;
    testsweeper::ParamEnum< slate::MethodLU >       method_lu;
    testsweeper::ParamEnum< slate::MethodTrsm >     method_trsm;

    testsweeper::ParamEnum< slate::GridOrder >      grid_order;
    testsweeper::ParamEnum< slate::GridOrder >      dev_order;

    // test matrix parameters
    MatrixParams matrix;
    MatrixParams matrixB;
    MatrixParams matrixC;

    // BLAS & LAPACK options
    // The order here matches the order in most LAPACK functions, e.g.,
    // hegv ( itype, jobz, uplo, n, ... )
    // syevx( jobz, range, uplo, n, ..., vl, vu, il, iu, ... )
    // larfb( side, trans, direction, storev, m, n, k, ... )
    // lanhe( norm, uplo, n, ... )
    // pbsv ( uplo, n, kd, nrhs, ... )
    // gbsv ( n, kl, ku, nrhs, ... )
    // trsm ( side, uplo, transa, diag, m, n, alpha, ... )
    // gesvx( fact, trans, n, nrhs, ..., equed, ... )
    // ijob, itype are classified as enums due to their limited values.
    testsweeper::ParamEnum< blas::Layout >          layout;
    testsweeper::ParamInt                           itype;  // hegv
    testsweeper::ParamEnum< lapack::Job >           jobz;   // heev
    testsweeper::ParamEnum< lapack::Job >           jobvl;  // geev
    testsweeper::ParamEnum< lapack::Job >           jobvr;  // geev
    testsweeper::ParamEnum< lapack::Job >           jobu;   // svd
    testsweeper::ParamEnum< lapack::Job >           jobvt;  // svd
    testsweeper::ParamEnum< lapack::Range >         range;  // heevx
    testsweeper::ParamEnum< lapack::Norm >          norm;
    testsweeper::ParamEnum< slate::NormScope >      scope;
    testsweeper::ParamEnum< blas::Side >            side;
    testsweeper::ParamEnum< blas::Uplo >            uplo;
    testsweeper::ParamEnum< blas::Op >              trans;
    testsweeper::ParamEnum< blas::Op >              transA;
    testsweeper::ParamEnum< blas::Op >              transB;
    testsweeper::ParamEnum< blas::Diag >            diag;
    testsweeper::ParamEnum< lapack::Direction >     direction;  // larfb
    testsweeper::ParamEnum< lapack::StoreV >        storev;     // larfb
    testsweeper::ParamEnum< lapack::Equed >         equed;      // gesvx

    //----- routine parameters, numeric
    testsweeper::ParamInt3    dim;  // m, n, k
    testsweeper::ParamInt     kd;
    testsweeper::ParamInt     kl;
    testsweeper::ParamInt     ku;
    testsweeper::ParamInt     nrhs;
    testsweeper::ParamInt     nb;
    testsweeper::ParamInt     ib;
    testsweeper::ParamDouble  vl;
    testsweeper::ParamDouble  vu;
    testsweeper::ParamInt     il;
    testsweeper::ParamInt     iu;
    testsweeper::ParamInt     il_out;
    testsweeper::ParamInt     iu_out;
    testsweeper::ParamDouble  fraction_start;
    testsweeper::ParamDouble  fraction;
    testsweeper::ParamComplex alpha;
    testsweeper::ParamComplex beta;
    testsweeper::ParamInt     incx;
    testsweeper::ParamInt     incy;

    // SLATE options
    testsweeper::ParamInt3    grid;  // p x q
    testsweeper::ParamInt     lookahead;
    testsweeper::ParamInt     panel_threads;
    testsweeper::ParamChar    nonuniform_nb;
    testsweeper::ParamDouble  pivot_threshold;
    testsweeper::ParamString  deflate;
    testsweeper::ParamInt     itermax;
    testsweeper::ParamChar    fallback;
    testsweeper::ParamInt     depth;

    //----- output parameters
    testsweeper::ParamScientific value;
    testsweeper::ParamScientific value2;
    testsweeper::ParamScientific value3;
    testsweeper::ParamScientific error;
    testsweeper::ParamScientific error2;
    testsweeper::ParamScientific error3;
    testsweeper::ParamScientific error4;
    testsweeper::ParamScientific error5;
    testsweeper::ParamScientific ortho;
    testsweeper::ParamScientific ortho_U;
    testsweeper::ParamScientific ortho_V;

    testsweeper::ParamDouble     time;
    testsweeper::ParamDouble     gflops;
    testsweeper::ParamDouble     gbytes;
    testsweeper::ParamInt        iters;

    testsweeper::ParamDouble     time2;
    testsweeper::ParamDouble     gflops2;
    testsweeper::ParamDouble     gbytes2;

    testsweeper::ParamDouble     time3;
    testsweeper::ParamDouble     time4;
    testsweeper::ParamDouble     time5;
    testsweeper::ParamDouble     time6;
    testsweeper::ParamDouble     time7;
    testsweeper::ParamDouble     time8;
    testsweeper::ParamDouble     time9;
    testsweeper::ParamDouble     time10;
    testsweeper::ParamDouble     time11;
    testsweeper::ParamDouble     time12;
    testsweeper::ParamDouble     time13;

    testsweeper::ParamDouble     ref_time;
    testsweeper::ParamDouble     ref_gflops;
    testsweeper::ParamDouble     ref_gbytes;
    testsweeper::ParamInt        ref_iters;

    testsweeper::ParamOkay       okay;
    testsweeper::ParamString     msg;
};

//------------------------------------------------------------------------------
template< typename T >
inline T roundup(T x, T y)
{
    return T((x + y - 1) / y)*y;
}

//------------------------------------------------------------------------------
// Level 3 BLAS
void test_gbmm   (Params& params, bool run);
void test_gemm   (Params& params, bool run);
void test_symm   (Params& params, bool run);
void test_syr2k  (Params& params, bool run);
void test_syrk   (Params& params, bool run);
void test_tbsm   (Params& params, bool run);
void test_trsm   (Params& params, bool run);
void test_trmm   (Params& params, bool run);
void test_hemm   (Params& params, bool run);
void test_hbmm   (Params& params, bool run);
void test_her2k  (Params& params, bool run);
void test_herk   (Params& params, bool run);

// LU, general
void test_gesv       (Params& params, bool run);
void test_gecondest  (Params& params, bool run);
void test_getri      (Params& params, bool run);
void test_trtri      (Params& params, bool run);

// LU, band
void test_gbsv   (Params& params, bool run);

// Cholesky
void test_posv      (Params& params, bool run);
void test_pocondest (Params& params, bool run);
void test_potri     (Params& params, bool run);

// Cholesky, band
void test_pbsv   (Params& params, bool run);

// symmetric indefinite
void test_sysv   (Params& params, bool run);
void test_sytrf  (Params& params, bool run);
void test_sytrs  (Params& params, bool run);

// Hermitian indefinite
void test_hesv   (Params& params, bool run);
void test_hetrf  (Params& params, bool run);
void test_hetrs  (Params& params, bool run);

// QR, LQ, RQ, QL
void test_gels      (Params& params, bool run);
void test_geqrf     (Params& params, bool run);
void test_gelqf     (Params& params, bool run);
void test_unmqr     (Params& params, bool run);
void test_trcondest (Params& params, bool run);

// symmetric/Hermitian eigenvalues
void test_heev   (Params& params, bool run);
void test_sterf  (Params& params, bool run);
void test_steqr2 (Params& params, bool run);
void test_stedc  (Params& params, bool run);

void test_stedc_deflate  (Params& params, bool run);
void test_stedc_secular  (Params& params, bool run);
void test_stedc_sort     (Params& params, bool run);
void test_stedc_z_vector (Params& params, bool run);

void test_he2hb       (Params& params, bool run);
void test_hb2st       (Params& params, bool run);
void test_unmtr_he2hb (Params& params, bool run);
void test_unmtr_hb2st (Params& params, bool run);

// generalized symmetric/Hermitian eigenvalues
void test_hegv   (Params& params, bool run);
void test_hegst  (Params& params, bool run);

// SVD
void test_svd    (Params& params, bool run);
void test_ge2tb  (Params& params, bool run);
void test_tb2bd  (Params& params, bool run);
void test_bdsqr  (Params& params, bool run);
void test_unmbr_ge2tb(Params& params, bool run);
void test_unmbr_tb2bd(Params& params, bool run);

// matrix norms
void test_gbnorm (Params& params, bool run);
void test_hbnorm (Params& params, bool run);
void test_norm   (Params& params, bool run);

// auxiliary matrix routines
void test_add    (Params& params, bool run);
void test_copy   (Params& params, bool run);
void test_scale  (Params& params, bool run);
void test_scale_row_col(Params& params, bool run);
void test_set    (Params& params, bool run);

//------------------------------------------------------------------------------
inline double barrier_get_wtime(MPI_Comm comm)
{
    slate::trace::Block trace_block("MPI_Barrier");
    MPI_Barrier(comm);
    return testsweeper::get_wtime();
}

//------------------------------------------------------------------------------
/// @return true if str ends with ending.
/// std::string ends_with added in C++20. For now, do simple implementation.
///
inline bool ends_with( std::string const& str, std::string const& ending )
{
    return str.size() >= ending.size()
           && str.compare( str.size() - ending.size(), std::string::npos,
                           ending ) == 0;
}

#endif // SLATE_TEST_HH
