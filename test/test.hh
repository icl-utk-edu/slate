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

    // Field members are explicitly public.
    // Order here determines output order.
    // ----- test framework parameters
    testsweeper::ParamChar   check;
    testsweeper::ParamChar   error_exit;
    testsweeper::ParamChar   ref;
    testsweeper::ParamChar   trace;
    testsweeper::ParamDouble trace_scale;
    testsweeper::ParamDouble tol;
    testsweeper::ParamInt    repeat;
    testsweeper::ParamInt    verbose;
    testsweeper::ParamInt    extended;
    testsweeper::ParamInt    cache;

    // ----- routine parameters
    // LAPACK options
    // The order here matches the order in most LAPACK functions, e.g.,
    // syevx( jobz, range, uplo, n, ..., vl, vu, il, iu, ... )
    // larfb( side, trans, direct, storev, m, n, k, ... )
    // lanhe( norm, uplo, n, ... )
    // pbsv ( uplo, n, kd, nrhs, ... )
    // gbsv ( n, kl, ku, nrhs, ... )
    // trsm ( side, uplo, transa, diag, m, n, alpha, ... )
    testsweeper::ParamEnum< testsweeper::DataType > datatype;
    testsweeper::ParamEnum< slate::Origin >         origin;
    testsweeper::ParamEnum< slate::Target >         target;
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
    testsweeper::ParamEnum< lapack::Direct >        direct;
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
    testsweeper::ParamDouble alpha;
    testsweeper::ParamDouble beta;
    testsweeper::ParamInt    incx;
    testsweeper::ParamInt    incy;
    testsweeper::ParamInt    itype;

    // SLATE options
    testsweeper::ParamInt    nb;
    testsweeper::ParamInt    ib;
    testsweeper::ParamInt    p;
    testsweeper::ParamInt    q;
    testsweeper::ParamInt    lookahead;
    testsweeper::ParamInt    panel_threads;
    testsweeper::ParamInt    align;
    testsweeper::ParamEnum< std::string > gemm_variant;
    testsweeper::ParamChar   nonuniform_nb;

    // ----- output parameters
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
    testsweeper::ParamInt        iters;

    testsweeper::ParamDouble     ref_time;
    testsweeper::ParamDouble     ref_gflops;
    testsweeper::ParamInt        ref_iters;

    testsweeper::ParamOkay       okay;

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
void test_getri      (Params& params, bool run);

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
void test_gels   (Params& params, bool run);
void test_geqrf  (Params& params, bool run);
void test_gelqf  (Params& params, bool run);

// symmetric/Hermitian eigenvalues
void test_heev        (Params& params, bool run);
void test_he2hb       (Params& params, bool run);
void test_unmtr_he2hb (Params& params, bool run);
void test_hb2st       (Params& params, bool run);
void test_sterf       (Params& params, bool run);
void test_steqr2      (Params& params, bool run);

// generalized symmetric/Hermitian eigenvalues
void test_hegv   (Params& params, bool run);
void test_hegst  (Params& params, bool run);

// SVD
void test_gesvd  (Params& params, bool run);
void test_ge2tb  (Params& params, bool run);
void test_tb2bd  (Params& params, bool run);
void test_bdsqr  (Params& params, bool run);

// matrix norms
void test_gbnorm (Params& params, bool run);
void test_genorm (Params& params, bool run);
void test_henorm (Params& params, bool run);
void test_hbnorm (Params& params, bool run);
void test_synorm (Params& params, bool run);
void test_trnorm (Params& params, bool run);

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
        case slate::Dist::Col: return "column";
    }
    return "?";
}

// -----------------------------------------------------------------------------
inline slate::Origin str2origin(const char* origin)
{
    std::string origin_ = origin;
    std::transform(origin_.begin(), origin_.end(), origin_.begin(), ::tolower);
    if (origin_ == "d" || origin_ == "dev" || origin_ == "device" ||
        origin_ == "devices")
        return slate::Origin::Devices;
    else if (origin_ == "h" || origin_ == "host")
        return slate::Origin::Host;
    else if (origin_ == "s" || origin_ == "scalapack")
        return slate::Origin::ScaLAPACK;
    else
        throw slate::Exception("unknown origin");
}

inline const char* origin2str(slate::Origin origin)
{
    switch (origin) {
        case slate::Origin::Devices:   return "devices";
        case slate::Origin::Host:      return "host";
        case slate::Origin::ScaLAPACK: return "scalapack";
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
        case slate::Target::Devices:   return "devices";
        case slate::Target::Host:      return "host";
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
inline std::string str2gemmVariant(const char *gemmVariant)
{
    std::string gemmVariant_ = gemmVariant;
    std::transform(gemmVariant_.begin(), gemmVariant_.end(), gemmVariant_.begin(), ::tolower);
    if (gemmVariant_ == "gemmc")
        return "gemmC";
    else if (gemmVariant_ == "gemma")
        return "gemmA";
    // todo: gemmB
    // else if (gemmVariant_ == "gemmb")
    //     return "gemmB";
    else
        throw slate::Exception("unknown gemm-variant");
}

inline const char* gemmVariant2str(std::string gemmVariant)
{
    return gemmVariant.c_str();
}


#endif // SLATE_TEST_HH
