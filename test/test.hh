#ifndef TEST_HH
#define TEST_HH

#include <exception>
#include <complex>
#include <ctype.h>

#include <assert.h>

#include "libtest.hh"
#include "blas.hh"
#include "lapack.hh"
#include "slate/slate.hh"

// -----------------------------------------------------------------------------
class Params: public libtest::ParamsBase {
public:
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double pi  = 3.141592653589793;
    const double e   = 2.718281828459045;

    Params();

    // Field members are explicitly public.
    // Order here determines output order.
    // ----- test framework parameters
    libtest::ParamChar   check;
    libtest::ParamChar   error_exit;
    libtest::ParamChar   ref;
    libtest::ParamChar   trace;
    libtest::ParamDouble tol;
    libtest::ParamInt    repeat;
    libtest::ParamInt    verbose;
    libtest::ParamInt    extended;
    libtest::ParamInt    cache;
    libtest::ParamInt    matrix;  // todo: string + generator
    libtest::ParamChar   target;  // todo: enum

    // ----- routine parameters
    // LAPACK options
    // The order here matches the order in most LAPACK functions, e.g.,
    // syevx( jobz, range, uplo, n, ..., vl, vu, il, iu, ... )
    // larfb( side, trans, direct, storev, m, n, k, ... )
    // lanhe( norm, uplo, n, ... )
    // pbsv ( uplo, n, kd, nrhs, ... )
    // gbsv ( n, kl, ku, nrhs, ... )
    // trsm ( side, uplo, transa, diag, m, n, alpha, ... )
    libtest::ParamEnum< libtest::DataType > datatype;
    libtest::ParamEnum< slate::Layout >     layout;
    libtest::ParamEnum< lapack::Job >       jobz;   // heev
    libtest::ParamEnum< lapack::Job >       jobvl;  // geev
    libtest::ParamEnum< lapack::Job >       jobvr;  // geev
    libtest::ParamEnum< lapack::Job >       jobu;   // gesvd, gesdd
    libtest::ParamEnum< lapack::Job >       jobvt;  // gesvd
    libtest::ParamEnum< lapack::Range >     range;
    libtest::ParamEnum< slate::Norm >       norm;
    libtest::ParamEnum< slate::Side >       side;
    libtest::ParamEnum< slate::Uplo >       uplo;
    libtest::ParamEnum< slate::Op >         trans;
    libtest::ParamEnum< slate::Op >         transA;
    libtest::ParamEnum< slate::Op >         transB;
    libtest::ParamEnum< slate::Diag >       diag;
    libtest::ParamEnum< lapack::Direct >    direct;
    libtest::ParamEnum< lapack::StoreV >    storev;
    libtest::ParamEnum< lapack::MatrixType > matrixtype;

    libtest::ParamInt3   dim;  // m, n, k
    libtest::ParamInt    kd;
    libtest::ParamInt    kl;
    libtest::ParamInt    ku;
    libtest::ParamInt    nrhs;
    libtest::ParamDouble vl;
    libtest::ParamDouble vu;
    libtest::ParamInt    il;
    libtest::ParamInt    iu;
    libtest::ParamDouble alpha;
    libtest::ParamDouble beta;
    libtest::ParamInt    incx;
    libtest::ParamInt    incy;

    // SLATE options
    libtest::ParamInt    nb;
    libtest::ParamInt    ib;
    libtest::ParamInt    p;
    libtest::ParamInt    q;
    libtest::ParamInt    lookahead;
    libtest::ParamInt    panel_threads;
    libtest::ParamInt    align;

    // ----- output parameters
    libtest::ParamScientific error;
    libtest::ParamScientific error2;
    libtest::ParamScientific error3;
    libtest::ParamScientific error4;
    libtest::ParamScientific error5;
    libtest::ParamScientific ortho;
    libtest::ParamScientific ortho_U;
    libtest::ParamScientific ortho_V;
    libtest::ParamScientific error_sigma;

    libtest::ParamDouble     time;
    libtest::ParamDouble     gflops;
    libtest::ParamInt        iters;

    libtest::ParamDouble     ref_time;
    libtest::ParamDouble     ref_gflops;
    libtest::ParamInt        ref_iters;

    libtest::ParamOkay       okay;

    std::string              routine;
};


// -----------------------------------------------------------------------------
template< typename T >
inline T roundup(T x, T y)
{
    return T((x + y - 1) / y)*y;
}

// -----------------------------------------------------------------------------
#define assert_throw( expr, exception_type ) \
    try { \
        expr; \
        fprintf( stderr, "Error: didn't throw expected exception at %s:%d\n", \
                 __FILE__, __LINE__ ); \
        throw std::exception(); \
    } \
    catch (exception_type& err) { \
        if (verbose >= 3) { \
            printf( "Caught expected exception: %s\n", err.what() ); \
        } \
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
void test_her2k  (Params& params, bool run);
void test_herk   (Params& params, bool run);

// LU, general
void test_gesv   (Params& params, bool run);

// LU, band
void test_gbsv   (Params& params, bool run);

// Cholesky
void test_posv   (Params& params, bool run);

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

// matrix norms
void test_gbnorm (Params& params, bool run);
void test_genorm (Params& params, bool run);
void test_henorm (Params& params, bool run);
void test_synorm (Params& params, bool run);
void test_trnorm (Params& params, bool run);

// -----------------------------------------------------------------------------
inline slate::Target char2target(char targetchar)
{
    if (targetchar == 't')
        return slate::Target::HostTask;
    else if (targetchar == 'n')
        return slate::Target::HostNest;
    else if (targetchar == 'b')
        return slate::Target::HostBatch;
    else if (targetchar == 'd')
        return slate::Target::Devices;
    return slate::Target::HostTask;
}

#endif  //  #ifndef TEST_HH
