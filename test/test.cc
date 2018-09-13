#include <complex>

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <unistd.h>


#include "test.hh"
#include "slate_mpi.hh"


// -----------------------------------------------------------------------------
using libtest::ParamType;
using libtest::DataType;
using libtest::char2datatype;
using libtest::datatype2char;
using libtest::datatype2str;
using libtest::ansi_bold;
using libtest::ansi_red;
using libtest::ansi_normal;

// -----------------------------------------------------------------------------
// each section must have a corresponding entry in section_names
enum Section {
    newline = 0,  // zero flag forces newline
    blas_section,
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
std::vector< libtest::routines_t > routines = {
    // -----
    // Level 3 BLAS
    { "gemm",               test_gemm,         Section::blas_section },
    { "",                   nullptr,           Section::newline },

    { "hemm",               test_hemm,         Section::blas_section },
    { "herk",               test_herk,         Section::blas_section },
    { "her2k",              test_her2k,        Section::blas_section },
    { "",                   nullptr,           Section::newline },

    { "symm",               test_symm,         Section::blas_section },
    { "syrk",               test_syrk,         Section::blas_section },
    { "syr2k",              test_syr2k,        Section::blas_section },
    { "",                   nullptr,           Section::newline },

    { "trmm",               test_trmm,         Section::blas_section },
    { "trsm",               test_trsm,         Section::blas_section },

    // -----
    // LU
    { "getrf",              test_getrf,        Section::gesv },
    { "gbtrf",              test_gbtrf,        Section::gesv },
    { "getrs",              test_getrs,        Section::gesv },
    { "gesv",               test_gesv,         Section::gesv },
    { "",                   nullptr,           Section::newline },

    // -----
    // Cholesky
    //{ "posv",               test_posv,         Section::posv },
    //{ "",                   nullptr,           Section::newline },

    { "potrf",              test_potrf,        Section::posv },
    { "",                   nullptr,           Section::newline },

    //{ "potrs",              test_potrs,        Section::posv },
    //{ "",                   nullptr,           Section::newline },

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
    //{ "hesv",                test_hesv,         Section::hesv },
    //{ "",                    nullptr,           Section::newline },

    //{ "hetrf",               test_hetrf,        Section::hesv },
    //{ "",                    nullptr,           Section::newline },

    //{ "hetrs",               test_hetrs,        Section::hesv },
    //{ "",                    nullptr,           Section::newline },

    // -----
    // matrix norms
    { "genorm",             test_genorm,       Section::aux_norm },
    { "henorm",             test_henorm,       Section::aux_norm },
    { "synorm",             test_synorm,       Section::aux_norm },
    { "trnorm",             test_trnorm,       Section::aux_norm },
    { "",                   nullptr,           Section::newline },
};

// -----------------------------------------------------------------------------
// Params class
// List of parameters

Params::Params():
    ParamsBase(),

    // w = width
    // p = precision
    // def = default
    // ----- test framework parameters
    //         name,       w,    type,             def, valid, help
    check     ( "check",   0,    ParamType::Value, 'y', "ny",  "check the results" ),
    error_exit( "error-exit", 0, ParamType::Value, 'n', "ny",  "check error exits" ),
    ref       ( "ref",     0,    ParamType::Value, 'y', "ny",  "run reference; sometimes check implies ref" ),
    trace     ( "trace",   0,    ParamType::Value, 'n', "ny",  "enable/disable traces" ),

    //          name,      w, p, type,             def, min,  max, help
    tol       ( "tol",     0, 0, ParamType::Value,  50,   1, 1000, "tolerance (e.g., error < tol*epsilon to pass)" ),
    repeat    ( "repeat",  0,    ParamType::Value,   1,   1, 1000, "number of times to repeat each test" ),
    verbose   ( "verbose", 0,    ParamType::Value,   0,   0,   10, "verbose level" ),
    extended  ( "extended",0,    ParamType::Value,   0,   0,   10, "extended tests" ),
    cache     ( "cache",   0,    ParamType::Value,  20,   1, 1024, "total cache size, in MiB" ),

    // ----- routine parameters
    //          name,      w,    type,            def,                    char2enum,         enum2char,         enum2str,         help
    target    ( "target",  6,    ParamType::List, 't', "tnbd", "target: t=HostTask n=HostNest b=HostBatch d=Devices" ),
    datatype  ( "type",    4,    ParamType::List, DataType::Double,       char2datatype,     datatype2char,     datatype2str,     "s=single (float), d=double, c=complex-single, z=complex-double" ),
    layout    ( "layout",  6,    ParamType::List, blas::Layout::ColMajor, blas::char2layout, blas::layout2char, blas::layout2str, "layout: r=row major, c=column major" ),
    side      ( "side",    6,    ParamType::List, blas::Side::Left,       blas::char2side,   blas::side2char,   blas::side2str,   "side: l=left, r=right" ),
    uplo      ( "uplo",    6,    ParamType::List, blas::Uplo::Lower,      blas::char2uplo,   blas::uplo2char,   blas::uplo2str,   "triangle: l=lower, u=upper" ),
    trans     ( "trans",   7,    ParamType::List, blas::Op::NoTrans,      blas::char2op,     blas::op2char,     blas::op2str,     "transpose: n=no-trans, t=trans, c=conj-trans" ),
    transA    ( "transA",  7,    ParamType::List, blas::Op::NoTrans,      blas::char2op,     blas::op2char,     blas::op2str,     "transpose of A: n=no-trans, t=trans, c=conj-trans" ),
    transB    ( "transB",  7,    ParamType::List, blas::Op::NoTrans,      blas::char2op,     blas::op2char,     blas::op2str,     "transpose of B: n=no-trans, t=trans, c=conj-trans" ),
    diag      ( "diag",    7,    ParamType::List, blas::Diag::NonUnit,    blas::char2diag,   blas::diag2char,   blas::diag2str,   "diagonal: n=non-unit, u=unit" ),
    norm      ( "norm",    7,    ParamType::List, lapack::Norm::One,      lapack::char2norm, lapack::norm2char, lapack::norm2str, "norm: o=one, 2=two, i=inf, f=fro, m=max" ),
    direct    ( "direct",  8,    ParamType::List, lapack::Direct::Forward, lapack::char2direct, lapack::direct2char, lapack::direct2str, "direction: f=forward, b=backward" ),
    storev    ( "storev", 10,    ParamType::List, lapack::StoreV::Columnwise, lapack::char2storev, lapack::storev2char, lapack::storev2str, "store vectors: c=columnwise, r=rowwise" ),
    jobz      ( "jobz",    5,    ParamType::List, lapack::Job::NoVec, lapack::char2job, lapack::job2char, lapack::job2str, "eigenvectors: n=no vectors, v=vectors" ),
    jobvl     ( "jobvl",   5,    ParamType::List, lapack::Job::NoVec, lapack::char2job, lapack::job2char, lapack::job2str, "left eigenvectors: n=no vectors, v=vectors" ),
    jobvr     ( "jobvr",   5,    ParamType::List, lapack::Job::NoVec, lapack::char2job, lapack::job2char, lapack::job2str, "right eigenvectors: n=no vectors, v=vectors" ),
    jobu      ( "jobu",    9,    ParamType::List, lapack::Job::NoVec, lapack::char2job, lapack::job2char, lapack::job2str, "left singular vectors (U): n=no vectors, s=some vectors, o=overwrite, a=all vectors" ),
    jobvt     ( "jobvt",   9,    ParamType::List, lapack::Job::NoVec, lapack::char2job, lapack::job2char, lapack::job2str, "right singular vectors (V^T): n=no vectors, s=some vectors, o=overwrite, a=all vectors" ),
    range     ( "range",   9,    ParamType::List, lapack::Range::All, lapack::char2range, lapack::range2char, lapack::range2str, "find: a=all eigen/singular values, v=values in (vl, vu], i=il-th through iu-th values" ),

    matrixtype( "matrixtype", 10, ParamType::List, lapack::MatrixType::General,
                lapack::char2matrixtype, lapack::matrixtype2char, lapack::matrixtype2str,
                "matrix type: g=general, l=lower, u=upper, h=Hessenberg, z=band-general, b=band-lower, q=band-upper" ),

    //          name,      w, p, type,            def,   min,     max, help
    dim       ( "dim",     6,    ParamType::List,          0, 1000000, "m x n x k dimensions" ),
    nb        ( "nb",      5,    ParamType::List, 50,      0, 1000000, "nb" ),
    nt        ( "nt",      5,    ParamType::List, 3,       0, 1000000, "nt" ),
    p         ( "p",       4,    ParamType::List, 1,       0, 1000000, "p" ),
    q         ( "q",       4,    ParamType::List, 1,       0, 1000000, "q" ),
    lookahead ( "lookahead", 5,  ParamType::List, 1,       0, 1000000, "number of lookahead panels" ),
    panel_threads( "panel-threads",
                           7,    ParamType::List, 1,       0, 1000000, "max number of threads used in panel" ),

    kd        ( "kd",      6,    ParamType::List, 100,     0, 1000000, "bandwidth" ),
    kl        ( "kl",      6,    ParamType::List, 100,     0, 1000000, "lower bandwidth" ),
    ku        ( "ku",      6,    ParamType::List, 100,     0, 1000000, "upper bandwidth" ),
    nrhs      ( "nrhs",    6,    ParamType::List,  10,     0, 1000000, "number of right hand sides" ),
    vl        ( "vl",      6, 3, ParamType::List,  10,     0, 1000000, "lower bound of eigen/singular values to find; default 10.0" ),
    vu        ( "vu",      6, 3, ParamType::List, 100,     0, 1000000, "upper bound of eigen/singular values to find; default 100.0" ),
    il        ( "il",      6,    ParamType::List,  10,     0, 1000000, "1-based index of smallest eigen/singular value to find; default 10" ),
    iu        ( "iu",      6,    ParamType::List, 100,     0, 1000000, "1-based index of largest  eigen/singular value to find; default 100" ),
    alpha     ( "alpha",   8, 3, ParamType::List,  pi,  -inf,     inf, "scalar alpha" ),
    beta      ( "beta",    8, 3, ParamType::List,   e,  -inf,     inf, "scalar beta" ),
    incx      ( "incx",    6,    ParamType::List,   1, -1000,    1000, "stride of x vector" ),
    incy      ( "incy",    6,    ParamType::List,   1, -1000,    1000, "stride of y vector" ),
    align     ( "align",   6,    ParamType::List,  32,     1,    1024, "column alignment (sets lda, ldb, etc. to multiple of align)" ),

    // ----- output parameters
    // min, max are ignored
    //           name,                    w, p, type,              default,               min, max, help
    error      ( "error",                 9, 2, ParamType::Output, libtest::no_data_flag,   0,   0, "numerical error" ),
    error2     ( "error2",                9, 2, ParamType::Output, libtest::no_data_flag,   0,   0, "numerical error" ),
    error3     ( "error3",                9, 2, ParamType::Output, libtest::no_data_flag,   0,   0, "numerical error" ),
    error4     ( "error4",                9, 2, ParamType::Output, libtest::no_data_flag,   0,   0, "numerical error" ),
    error5     ( "error5",                9, 2, ParamType::Output, libtest::no_data_flag,   0,   0, "numerical error" ),
    ortho      ( "orth. error",           9, 2, ParamType::Output, libtest::no_data_flag,   0,   0, "orthogonality error" ),
    ortho_U    ( "U orth.",               9, 2, ParamType::Output, libtest::no_data_flag,   0,   0, "U orthogonality error" ),
    ortho_V    ( "V orth.",               9, 2, ParamType::Output, libtest::no_data_flag,   0,   0, "V orthogonality error" ),
    error_sigma( "Sigma error",           9, 2, ParamType::Output, libtest::no_data_flag,   0,   0, "Sigma error" ),

    time      ( "SLATE\ntime (s)",       10, 4, ParamType::Output, libtest::no_data_flag,   0,   0, "time to solution" ),
    gflops    ( "SLATE\nGflop/s",        10, 3, ParamType::Output, libtest::no_data_flag,   0,   0, "Gflop/s rate" ),
    iters     ( "iters",                  6,    ParamType::Output,                     0,   0,   0, "iterations to solution" ),

    ref_time  ( "Ref.\ntime (s)",        10, 4, ParamType::Output, libtest::no_data_flag,   0,   0, "reference time to solution" ),
    ref_gflops( "Ref.\nGflop/s",         10, 3, ParamType::Output, libtest::no_data_flag,   0,   0, "reference Gflop/s rate" ),
    ref_iters ( "Ref.\niters",            6,    ParamType::Output,                     0,   0,   0, "reference iterations to solution" ),

    // default -1 means "no check"
    //          name,     w, type,              def, min, max, help
    okay      ( "status", 6, ParamType::Output,  -1,   0,   0, "success indicator" )
{
    // set header different than command line prefix
    lookahead.name( "look\nahead", "lookahead" );
    panel_threads.name( "panel\nthreads", "panel-threads" );

    // mark standard set of output fields as used
    okay  .value();
    error .value();
    time  .value();

    // mark framework parameters as used, so they will be accepted on the command line
    check  .value();
    error_exit.value();
    ref    .value();
    trace  .value();
    tol    .value();
    repeat .value();
    verbose.value();
    cache  .value();
    target .value();

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
    const std::string& msg, int mpi_rank, MPI_Comm comm )
{
    // reduction to determine first rank with an error
    typedef struct { int err, rank; } err_rank_t;
    int err = ! msg.empty();
    err_rank_t err_rank = { err, mpi_rank };
    err_rank_t err_first = { 0, 0 };
    MPI_Allreduce( &err_rank, &err_first, 1, MPI_2INT, MPI_MAXLOC, comm );

    if (err_first.err) {
        // count ranks with an error
        int root = 0;
        int cnt = 0;
        MPI_Reduce( &err, &cnt, 1, MPI_INT, MPI_SUM, root, comm );

        // first rank with error sends msg to root
        char buf[ 255 ];
        if (mpi_rank == err_first.rank) {
            snprintf( buf, sizeof(buf), "%s", msg.c_str() );
            // if rank == root, nothing to send
            if (mpi_rank != root) {
                slate_mpi_call(
                    MPI_Send( buf, sizeof(buf), MPI_CHAR, root, 0, comm ));
            }
        }
        else if (mpi_rank == root) {
            MPI_Status status;
            slate_mpi_call(
                MPI_Recv( buf, sizeof(buf), MPI_CHAR, err_first.rank, 0, comm,
                          &status ));
        }

        // root prints msg
        if (mpi_rank == root) {
            fprintf( stderr,
                     "\n%s%sError on rank %d: %s. (%d ranks had some error.)%s\n",
                     ansi_bold, ansi_red,
                     err_first.rank, buf, cnt,
                     ansi_normal );
        }
    }

    return err_first.err;
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    using libtest::QuitException;

    // check that all sections have names
    assert( sizeof(section_names)/sizeof(*section_names) == Section::num_sections );

    // MPI initializations
    int mpi_rank = 0, mpi_size = 0, provided = 0;
    int err = MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    if (err != MPI_SUCCESS) {
        fprintf( stderr, "Error: MPI could not be initialized (err = %d)\n", err );
        return -1;
    }

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
            printf( "input: %s", argv[0] );
            for (int i = 1; i < argc; ++i) {
                printf( " %s", argv[i] );
            }
            printf( "\n" );
        }

        // Usage: test routine [params]
        if (argc < 2 ||
            strcmp( argv[1], "-h" ) == 0 ||
            strcmp( argv[1], "--help" ) == 0)
        {
            if (print)
                usage( argc, argv, routines, section_names );
            throw QuitException();
        }

        // find routine to test
        const char* routine = argv[1];
        libtest::test_func_ptr test_routine = find_tester( routine, routines );
        if (test_routine == nullptr) {
            if (print)
                usage( argc, argv, routines, section_names );
            throw std::runtime_error(
                std::string("routine ") + routine + " not found" );
        }

        // mark fields that are used (run=false)
        Params params;
        test_routine( params, false );

        // Make default p x q grid as square as possible.
        // Worst case is p=1, q=mpi_size.
        int p = 1, q = 1;
        for (p = int( sqrt( mpi_size )); p > 0; --p) {
            q = int( mpi_size / p );
            if (p*q == mpi_size)
                break;
        }
        params.p.value() = p;
        params.q.value() = q;

        // parse parameters after routine name
        try {
            params.parse( routine, argc-2, argv+2 );
        }
        catch (const std::exception& ex) {
            if (print)
                params.help( routine );
            throw;
        }

        // run tests
        int repeat = params.repeat.value();
        libtest::DataType last = params.datatype.value();
        if (print)
            params.header();
        do {
            if (params.datatype.value() != last) {
                last = params.datatype.value();
                if (print)
                    printf("\n");
            }
            for (int iter = 0; iter < repeat; ++iter) {
                try {
                    test_routine( params, true );
                }
                catch (const std::exception& ex) {
                    msg = ex.what();
                }
                err = print_reduce_error( msg, mpi_rank, MPI_COMM_WORLD );
                if (err)
                    params.okay.value() = false;
                if (print)
                    params.print();
                status += ! params.okay.value();
                params.reset_output();
                msg.clear();
            }
            if (repeat > 1 && print) {
                printf( "\n" );
            }
        } while( params.next() );

        if (print) {
            if (status) {
                printf( "%d tests FAILED.\n", status );
            }
            else {
                printf( "All tests passed.\n" );
            }
        }
    }
    catch (const QuitException& ex) {
        // pass: no error to print
    }
    catch (const std::exception& ex) {
        msg = ex.what();
    }
    err = print_reduce_error( msg, mpi_rank, MPI_COMM_WORLD );
    if (err)
        status = -1;

    MPI_Finalize();

    if (mpi_rank == 0)
        return status;
    else
        return 0;
}
