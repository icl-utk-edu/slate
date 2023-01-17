// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_TEST_HH
#define SLATE_TEST_HH

#include <exception>
#include <complex>
#include <ctype.h>

#include "testsweeper.hh"
#include "blas.hh"
#include "lapack.hh"
#include "slate/slate.hh"

#include "matrix_params.hh"
#include "matrix_generator.hh"

// -----------------------------------------------------------------------------
namespace slate {

enum class Origin {
    Host,
    ScaLAPACK,
    Devices,
};

enum class Dist {
    Row,
    Col,
};

} // namespace slate

// -----------------------------------------------------------------------------
using llong = long long;

// -----------------------------------------------------------------------------
class Params: public testsweeper::ParamsBase {
public:
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double pi  = 3.141592653589793;
    const double e   = 2.718281828459045;

    Params();

    // ----- test matrix parameters
    MatrixParams matrix;
    MatrixParams matrixB;
    MatrixParams matrixC;

    // Field members are explicitly public.
    // Order here determines output order.
    // ----- test framework parameters
    testsweeper::ParamChar   check;
    testsweeper::ParamChar   error_exit;
    testsweeper::ParamChar   ref;
    testsweeper::ParamChar   hold_local_workspace;
    testsweeper::ParamChar   trace;
    testsweeper::ParamDouble trace_scale;
    testsweeper::ParamDouble tol;
    testsweeper::ParamInt    repeat;
    testsweeper::ParamInt    verbose;
    testsweeper::ParamInt    print_edgeitems;
    testsweeper::ParamInt    print_width;
    testsweeper::ParamInt    print_precision;
    testsweeper::ParamInt    extended;
    testsweeper::ParamInt    cache;

    // ----- routine parameters
    // LAPACK options
    // The order here matches the order in most LAPACK functions, e.g.,
    // syevx( jobz, range, uplo, n, ..., vl, vu, il, iu, ... )
    // larfb( side, trans, direction, storev, m, n, k, ... )
    // lanhe( norm, uplo, n, ... )
    // pbsv ( uplo, n, kd, nrhs, ... )
    // gbsv ( n, kl, ku, nrhs, ... )
    // trsm ( side, uplo, transa, diag, m, n, alpha, ... )
    testsweeper::ParamEnum< testsweeper::DataType > datatype;
    testsweeper::ParamEnum< slate::Origin >         origin;
    testsweeper::ParamEnum< slate::Target >         target;

    testsweeper::ParamEnum< slate::Method >         method_cholQR;
    testsweeper::ParamEnum< slate::Method >         method_gels;
    testsweeper::ParamEnum< slate::Method >         method_gemm;
    testsweeper::ParamEnum< slate::Method >         method_hemm;
    testsweeper::ParamEnum< slate::Method >         method_lu;
    testsweeper::ParamEnum< slate::Method >         method_trsm;

    testsweeper::ParamEnum< slate::GridOrder >      grid_order;
    testsweeper::ParamEnum< slate::TileReleaseStrategy > tile_release_strategy;
    testsweeper::ParamEnum< slate::Dist >           dev_dist;
    testsweeper::ParamEnum< slate::Layout >         layout;
    testsweeper::ParamEnum< lapack::Job >           jobz;   // heev
    testsweeper::ParamEnum< lapack::Job >           jobvl;  // geev
    testsweeper::ParamEnum< lapack::Job >           jobvr;  // geev
    testsweeper::ParamEnum< lapack::Job >           jobu;   // gesvd, gesdd
    testsweeper::ParamEnum< lapack::Job >           jobvt;  // gesvd
    testsweeper::ParamEnum< lapack::Range >         range;
    testsweeper::ParamEnum< slate::Norm >           norm;
    testsweeper::ParamEnum< slate::NormScope >      scope;
    testsweeper::ParamEnum< slate::Side >           side;
    testsweeper::ParamEnum< slate::Uplo >           uplo;
    testsweeper::ParamEnum< slate::Op >             trans;
    testsweeper::ParamEnum< slate::Op >             transA;
    testsweeper::ParamEnum< slate::Op >             transB;
    testsweeper::ParamEnum< slate::Diag >           diag;
    testsweeper::ParamEnum< slate::Direction >      direction;
    testsweeper::ParamEnum< slate::Equed >          equed;
    testsweeper::ParamEnum< lapack::StoreV >        storev;
    testsweeper::ParamEnum< lapack::MatrixType >    matrixtype;

    testsweeper::ParamInt3   dim;  // m, n, k
    testsweeper::ParamInt    kd;
    testsweeper::ParamInt    kl;
    testsweeper::ParamInt    ku;
    testsweeper::ParamInt    nrhs;
    testsweeper::ParamDouble vl;
    testsweeper::ParamDouble vu;
    testsweeper::ParamInt    il;
    testsweeper::ParamInt    iu;
    testsweeper::ParamComplex alpha;
    testsweeper::ParamComplex beta;
    testsweeper::ParamInt    incx;
    testsweeper::ParamInt    incy;
    testsweeper::ParamInt    itype;

    // SLATE options
    testsweeper::ParamInt    nb;
    testsweeper::ParamInt    ib;
    testsweeper::ParamInt3   grid;  // p x q
    testsweeper::ParamInt    lookahead;
    testsweeper::ParamInt    panel_threads;
    testsweeper::ParamInt    align;
    testsweeper::ParamChar   nonuniform_nb;
    testsweeper::ParamInt    debug;
    testsweeper::ParamDouble pivot_threshold;

    // ----- output parameters
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
    testsweeper::ParamScientific error_sigma;

    testsweeper::ParamDouble     time;
    testsweeper::ParamDouble     gflops;
    testsweeper::ParamDouble     time2;
    testsweeper::ParamDouble     gflops2;
    testsweeper::ParamInt        iters;

    testsweeper::ParamDouble     ref_time;
    testsweeper::ParamDouble     ref_gflops;
    testsweeper::ParamInt        ref_iters;

    testsweeper::ParamOkay       okay;
    testsweeper::ParamString     msg;

    std::string              routine;
};


// -----------------------------------------------------------------------------
template< typename T >
inline T roundup(T x, T y)
{
    return T((x + y - 1) / y)*y;
}

// -----------------------------------------------------------------------------
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
void test_posv   (Params& params, bool run);
void test_potri  (Params& params, bool run);

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
void test_he2hb  (Params& params, bool run);
void test_hb2st  (Params& params, bool run);
void test_sterf  (Params& params, bool run);
void test_steqr2 (Params& params, bool run);
void test_unmtr_he2hb(Params& params, bool run);
void test_unmtr_hb2st(Params& params, bool run);

// generalized symmetric/Hermitian eigenvalues
void test_hegv   (Params& params, bool run);
void test_hegst  (Params& params, bool run);

// SVD
void test_gesvd  (Params& params, bool run);
void test_ge2tb  (Params& params, bool run);
void test_tb2bd  (Params& params, bool run);
void test_bdsqr  (Params& params, bool run);
void test_unmbr_ge2tb(Params& params, bool run);
void test_unmbr_tb2bd(Params& params, bool run);

// matrix norms
void test_gbnorm (Params& params, bool run);
void test_genorm (Params& params, bool run);
void test_henorm (Params& params, bool run);
void test_hbnorm (Params& params, bool run);
void test_synorm (Params& params, bool run);
void test_trnorm (Params& params, bool run);

// auxiliary matrix routines
void test_add    (Params& params, bool run);
void test_copy   (Params& params, bool run);
void test_scale  (Params& params, bool run);
void test_scale_row_col(Params& params, bool run);
void test_set    (Params& params, bool run);

// -----------------------------------------------------------------------------
inline slate::Dist str2dist(const char* dist)
{
    std::string distribution_ = dist;
    std::transform(
        distribution_.begin(),
        distribution_.end(),
        distribution_.begin(), ::tolower);
    if (distribution_ == "row" || distribution_ == "r")
        return slate::Dist::Row;
    else if (distribution_ == "col" || distribution_ == "c"
                                    || distribution_ == "column")
        return slate::Dist::Col;
    else
        throw slate::Exception("unknown distribution");
}

inline const char* dist2str(slate::Dist dist)
{
    switch (dist) {
        case slate::Dist::Row: return "row";
        case slate::Dist::Col: return "col";
    }
    return "?";
}

// -----------------------------------------------------------------------------
inline slate::Origin str2origin(const char* origin)
{
    std::string origin_ = origin;
    std::transform(origin_.begin(), origin_.end(), origin_.begin(), ::tolower);
    if (origin_ == "d" || origin_ == "dev" || origin_ == "device"
        || origin_ == "devices")
        return slate::Origin::Devices;
    else if (origin_ == "h" || origin_ == "host")
        return slate::Origin::Host;
    else if (origin_ == "s" || origin_ == "scalapack" || origin_ == "scalpk")
        return slate::Origin::ScaLAPACK;
    else
        throw slate::Exception("unknown origin");
}

inline const char* origin2str(slate::Origin origin)
{
    switch (origin) {
        case slate::Origin::Devices:   return "dev";
        case slate::Origin::Host:      return "host";
        case slate::Origin::ScaLAPACK: return "scalpk";
    }
    return "?";
}

inline slate::Target origin2target(slate::Origin origin)
{
    switch (origin) {
        case slate::Origin::Host:
        case slate::Origin::ScaLAPACK:
            return slate::Target::Host;

        case slate::Origin::Devices:
            return slate::Target::Devices;

        default:
            throw slate::Exception("unknown origin");
    }
}

// -----------------------------------------------------------------------------
inline slate::Target str2target(const char* target)
{
    std::string target_ = target;
    std::transform(target_.begin(), target_.end(), target_.begin(), ::tolower);
    if (target_ == "t" || target_ == "task")
        return slate::Target::HostTask;
    else if (target_ == "n" || target_ == "nest")
        return slate::Target::HostNest;
    else if (target_ == "b" || target_ == "batch")
        return slate::Target::HostBatch;
    else if (target_ == "d" || target_ == "dev" || target_ == "device" ||
             target_ == "devices")
        return slate::Target::Devices;
    else if (target_ == "h" || target_ == "host")
        return slate::Target::Host;
    else
        throw slate::Exception("unknown target");
}

inline const char* target2str(slate::Target target)
{
    switch (target) {
        case slate::Target::HostTask:  return "task";
        case slate::Target::HostNest:  return "nest";
        case slate::Target::HostBatch: return "batch";
        case slate::Target::Devices:   return "dev";
        case slate::Target::Host:      return "host";
    }
    return "?";
}

// -----------------------------------------------------------------------------
inline slate::GridOrder str2grid_order( const char* grid_order )
{
    std::string grid_order_ = grid_order;
    std::transform( grid_order_.begin(), grid_order_.end(),
                    grid_order_.begin(), ::tolower );
    if (grid_order_ == "c" || grid_order_ == "col")
        return slate::GridOrder::Col;
    else if (grid_order_ == "r" || grid_order_ == "row")
        return slate::GridOrder::Row;
    else
        throw slate::Exception("unknown grid_order");
}

inline const char* grid_order2str( slate::GridOrder grid_order )
{
    switch (grid_order) {
        case slate::GridOrder::Col:     return "col";
        case slate::GridOrder::Row:     return "row";
        case slate::GridOrder::Unknown: return "un";
    }
    return "?";
}

// -----------------------------------------------------------------------------
inline slate::TileReleaseStrategy str2tile_release_strategy(const char* tile_release_strategy)
{
    std::string tile_release_strategy_ = tile_release_strategy;
    std::transform(tile_release_strategy_.begin(), tile_release_strategy_.end(), tile_release_strategy_.begin(), ::tolower);
    if (tile_release_strategy_ == "n" || tile_release_strategy_ == "none")
        return slate::TileReleaseStrategy::None;
    else if (tile_release_strategy_ == "i" || tile_release_strategy_ == "internal")
        return slate::TileReleaseStrategy::Internal;
    else if (tile_release_strategy_ == "s" || tile_release_strategy_ == "src")
        return slate::TileReleaseStrategy::Slate;
    else if (tile_release_strategy_ == "a" || tile_release_strategy_ == "all")
        return slate::TileReleaseStrategy::All;
    else
        throw slate::Exception("unknown tile_release_strategy");
}

inline const char* tile_release_strategy2str(slate::TileReleaseStrategy tile_release_strategy)
{
    switch (tile_release_strategy) {
        case slate::TileReleaseStrategy::None:     return "none";
        case slate::TileReleaseStrategy::Internal: return "int";
        case slate::TileReleaseStrategy::Slate:    return "src";
        case slate::TileReleaseStrategy::All:      return "all";
    }
    return "?";
}

// -----------------------------------------------------------------------------
inline slate::NormScope str2scope(const char* scope)
{
    std::string scope_ = scope;
    std::transform(scope_.begin(), scope_.end(), scope_.begin(), ::tolower);
    if (scope_ == "m" || scope_ == "matrix")
        return slate::NormScope::Matrix;
    else if (scope_ == "c" || scope_ == "cols" || scope_ == "columns")
        return slate::NormScope::Columns;
    else if (scope_ == "r" || scope_ == "rows")
        return slate::NormScope::Rows;
    else
        throw slate::Exception("unknown scope");
}

inline const char* scope2str(slate::NormScope scope)
{
    switch (scope) {
        case slate::NormScope::Matrix:  return "matrix";
        case slate::NormScope::Columns: return "columns";
        case slate::NormScope::Rows:    return "rows";
    }
    return "?";
}

// -----------------------------------------------------------------------------
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
