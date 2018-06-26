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
   "BLAS",
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
   "auxiliary - norms",
   "auxiliary - Householder",
   "auxiliary - matrix generation",
};

// { "", nullptr, Section::newline } entries force newline in help
std::vector< libtest::routines_t > routines = {
    // BLAS
    { "gemm",               test_gemm,         Section::blas_section },
    { "symm",               test_symm,         Section::blas_section },
    { "syr2k",              test_syr2k,        Section::blas_section },
    { "syrk",               test_syrk,         Section::blas_section },
    { "trsm",               test_trsm,         Section::blas_section },
    { "trmm",               test_trmm,         Section::blas_section },
    { "hemm",               test_hemm,         Section::blas_section },
    { "her2k",              test_her2k,        Section::blas_section },
    { "herk",               test_herk,         Section::blas_section },

    // -----
  //   // LU
  //   { "gesv",               test_gesv,      Section::gesv },
  //   { "gbsv",               test_gbsv,      Section::gesv },
  //   { "gtsv",               test_gtsv,      Section::gesv },
  //   { "",                   nullptr,        Section::newline },

  // //{ "gesvx",              test_gesvx,     Section::gesv },
  // //{ "gbsvx",              test_gbsvx,     Section::gesv },
  // //{ "gtsvx",              test_gtsvx,     Section::gesv },
  //   { "",                   nullptr,        Section::newline },

  //   { "getrf",              test_getrf,     Section::gesv },
  //   { "gbtrf",              test_gbtrf,     Section::gesv },
  //   { "gttrf",              test_gttrf,     Section::gesv },
  //   { "",                   nullptr,        Section::newline },

  //   { "getrs",              test_getrs,     Section::gesv },
  //   { "gbtrs",              test_gbtrs,     Section::gesv },
  //   { "gttrs",              test_gttrs,     Section::gesv },
  //   { "",                   nullptr,        Section::newline },

  //   { "getri",              test_getri,     Section::gesv },    // lawn 41 test
  //   { "",                   nullptr,        Section::newline },

  //   { "gecon",              test_gecon,     Section::gesv },
  //   { "gbcon",              test_gbcon,     Section::gesv },
  //   { "gtcon",              test_gtcon,     Section::gesv },
  //   { "",                   nullptr,        Section::newline },

  //   { "gerfs",              test_gerfs,     Section::gesv },
  //   { "gbrfs",              test_gbrfs,     Section::gesv },
  //   { "gtrfs",              test_gtrfs,     Section::gesv },
  //   { "",                   nullptr,        Section::newline },

  //   { "geequ",              test_geequ,     Section::gesv },
  //   { "gbequ",              test_gbequ,     Section::gesv },
  //   { "",                   nullptr,        Section::newline },

  //   // -----
  //   // Cholesky
  //   { "posv",               test_posv,      Section::posv },
  //   { "ppsv",               test_ppsv,      Section::posv },
  //   { "pbsv",               test_pbsv,      Section::posv },
  //   { "ptsv",               test_ptsv,      Section::posv },
  //   { "",                   nullptr,        Section::newline },


    { "potrf",              test_potrf,        Section::posv },

  //   { "pptrf",              test_pptrf,     Section::posv },
  //   { "pbtrf",              test_pbtrf,     Section::posv },
  //   { "pttrf",              test_pttrf,     Section::posv },
    { "",                   nullptr,        Section::newline },

  //   { "potrs",              test_potrs,     Section::posv },
  //   { "pptrs",              test_pptrs,     Section::posv },
  //   { "pbtrs",              test_pbtrs,     Section::posv },
  //   { "pttrs",              test_pttrs,     Section::posv },
  //   { "",                   nullptr,        Section::newline },

  //   { "potri",              test_potri,     Section::posv },    // lawn 41 test
  //   { "pptri",              test_pptri,     Section::posv },
  //   { "",                   nullptr,        Section::newline },

  //   { "pocon",              test_pocon,     Section::posv },
  //   { "ppcon",              test_ppcon,     Section::posv },
  //   { "pbcon",              test_pbcon,     Section::posv },
  //   { "ptcon",              test_ptcon,     Section::posv },
  //   { "",                   nullptr,        Section::newline },

  //   { "porfs",              test_porfs,     Section::posv },
  //   { "pprfs",              test_pprfs,     Section::posv },
  //   { "pbrfs",              test_pbrfs,     Section::posv },
  //   { "ptrfs",              test_ptrfs,     Section::posv },
  //   { "",                   nullptr,        Section::newline },

  //   { "poequ",              test_poequ,     Section::posv },
  //   { "ppequ",              test_ppequ,     Section::posv },
  //   { "pbequ",              test_pbequ,     Section::posv },
  //   { "",                   nullptr,        Section::newline },

  //   // -----
  //   // symmetric indefinite
  //   { "sysv",               test_sysv,      Section::sysv }, // tested via LAPACKE
  //   { "spsv",               test_spsv,      Section::sysv }, // tested via LAPACKE
  //   { "",                   nullptr,        Section::newline },

  //   { "sytrf",              test_sytrf,     Section::sysv }, // tested via LAPACKE
  //   { "sptrf",              test_sptrf,     Section::sysv }, // tested via LAPACKE
  //   { "",                   nullptr,        Section::newline },

  //   { "sytrs",              test_sytrs,     Section::sysv }, // tested via LAPACKE
  //   { "sptrs",              test_sptrs,     Section::sysv }, // tested via LAPACKE
  //   { "",                   nullptr,        Section::newline },

  //   { "sytri",              test_sytri,     Section::sysv }, // tested via LAPACKE
  //   { "sptri",              test_sptri,     Section::sysv }, // tested via LAPACKE
  //   { "",                   nullptr,        Section::newline },

  //   { "sycon",              test_sycon,     Section::sysv }, // tested via LAPACKE
  //   { "spcon",              test_spcon,     Section::sysv }, // tested via LAPACKE
  //   { "",                   nullptr,        Section::newline },

  //   { "syrfs",              test_syrfs,     Section::sysv }, // tested via LAPACKE
  //   { "sprfs",              test_sprfs,     Section::sysv }, // tested via LAPACKE
  //   { "",                   nullptr,        Section::newline },

  //   // -----
  // //{ "sysv_rook",          test_sysv_rook,          Section::sysv2 },  // requires LAPACK>=3.5
  // //{ "sysv_aasen",         test_sysv_aasen,         Section::sysv2 },
  // //{ "sysv_aasen_2stage",  test_sysv_aasen_2stage,  Section::sysv2 },
  //   { "",                   nullptr,                 Section::newline },

  // //{ "sytrf_rook",         test_sytrf_rook,         Section::sysv2 },
  // //{ "sytrf_aasen",        test_sytrf_aasen,        Section::sysv2 },
  // //{ "sytrf_aasen_2stage", test_sytrf_aasen_2stage, Section::sysv2 },
  //   { "",                   nullptr,                 Section::newline },

  // //{ "sytrs_rook",         test_sytrs_rook,         Section::sysv2 },
  // //{ "sytrs_aasen",        test_sytrs_aasen,        Section::sysv2 },
  // //{ "sytrs_aasen_2stage", test_sytrs_aasen_2stage, Section::sysv2 },
  //   { "",                   nullptr,                 Section::newline },

  // //{ "sytri_rook",         test_sytri_rook,         Section::sysv2 },
  // //{ "sytri_aasen",        test_sytri_aasen,        Section::sysv2 },
  // //{ "sytri_aasen_2stage", test_sytri_aasen_2stage, Section::sysv2 },
  //   { "",                   nullptr,                 Section::newline },

  //   // -----
  //   // hermetian
  //   { "hesv",               test_hesv,      Section::hesv }, // tested via LAPACKE
  //   { "hpsv",               test_hpsv,      Section::hesv }, // tested via LAPACKE
  //   { "",                   nullptr,        Section::newline },

  //   { "hetrf",              test_hetrf,     Section::hesv }, // tested via LAPACKE
  //   { "hptrf",              test_hptrf,     Section::hesv }, // tested via LAPACKE
  //   { "",                   nullptr,        Section::newline },

  //   { "hetrs",              test_hetrs,     Section::hesv }, // tested via LAPACKE
  //   { "hptrs",              test_hptrs,     Section::hesv }, // tested via LAPACKE
  //   { "",                   nullptr,        Section::newline },

  //   { "hetri",              test_hetri,     Section::hesv }, // tested via LAPACKE
  //   { "hptri",              test_hptri,     Section::hesv }, // tested via LAPACKE
  //   { "",                   nullptr,        Section::newline },

  //   { "hecon",              test_hecon,     Section::hesv }, // tested via LAPACKE
  //   { "hpcon",              test_hpcon,     Section::hesv }, // tested via LAPACKE, error < 3*eps
  //   { "",                   nullptr,        Section::newline },

  //   { "herfs",              test_herfs,     Section::hesv }, // tested via LAPACKE
  //   { "hprfs",              test_hprfs,     Section::hesv }, // tested via LAPACKE, error < 3*eps
  //   { "",                   nullptr,        Section::newline },

  //   // -----
  //   // least squares
  //   { "gels",               test_gels,      Section::gels }, // tested via LAPACKE using gcc/MKL
  //   { "gelsy",              test_gelsy,     Section::gels }, // tested via LAPACKE using gcc/MKL FIXME jpvt[i]=i rcond=0
  // //{ "gelsd",              test_gelsd,     Section::gels },
  //   { "gelss",              test_gelss,     Section::gels }, // tested via LAPACKE using gcc/MKL FIXME rcond=n
  //   { "getsls",             test_getsls,    Section::gels }, // tested via LAPACKE using gcc/MKL
  //   { "",                   nullptr,        Section::newline },

  // //{ "gglse",              test_gglse,     Section::gels },
  // //{ "ggglm",              test_ggglm,     Section::gels },
  //   { "",                   nullptr,        Section::newline },

  //   // -----
  //   // QR, LQ, RQ, QL
  //   { "geqrf",              test_geqrf,     Section::qr }, // tested numerically
  //   { "gelqf",              test_gelqf,     Section::qr }, // tested numerically
  //   { "geqlf",              test_geqlf,     Section::qr }, // tested numerically
  //   { "gerqf",              test_gerqf,     Section::qr }, // tested numerically; R, Q are full sizeof(A), could be smaller
  //   { "",                   nullptr,        Section::newline },

  //   { "ggqrf",              test_ggqrf,     Section::qr }, // tested via LAPACKE using gcc/MKL, TODO for now use p=param.k
  // //{ "gglqf",              test_gglqf,     Section::qr },
  // //{ "ggqlf",              test_ggqlf,     Section::qr },
  //   { "ggrqf",              test_ggrqf,     Section::qr }, // tested via LAPACKE using gcc/MKL, TODO for now use p=param.k
  //   { "",                   nullptr,        Section::newline },

  //   { "ungqr",              test_ungqr,     Section::qr }, // tested numerically based on lapack; R, Q full sizes
  //   { "unglq",              test_unglq,     Section::qr }, // tested numerically based on lapack; R, Q full; m<=n, k<=m
  //   { "ungql",              test_ungql,     Section::qr }, // tested numerically based on lapack; R, Q full sizes
  //   { "ungrq",              test_ungrq,     Section::qr }, // tested numerically based on lapack; R, Q full sizes
  //   { "",                   nullptr,        Section::newline },

  // //{ "unmqr",              test_unmqr,     Section::qr },
  // //{ "unmlq",              test_unmlq,     Section::qr },
  // //{ "unmql",              test_unmql,     Section::qr },
  // //{ "unmrq",              test_unmrq,     Section::qr },
  //   { "",                   nullptr,        Section::newline },

  //   // -----
  //   // symmetric/Hermitian eigenvalues
  //   { "heev",               test_heev,      Section::heev }, // tested via LAPACKE
  //   { "hpev",               test_hpev,      Section::heev }, // tested via LAPACKE
  //   { "hbev",               test_hbev,      Section::heev }, // tested via LAPACKE
  //   { "",                   nullptr,        Section::newline },

  // //{ "heevx",              test_heevx,     Section::heev },
  // //{ "hpevx",              test_hpevx,     Section::heev },
  // //{ "hbevx",              test_hbevx,     Section::heev },
  //   { "",                   nullptr,        Section::newline },

  //   { "heevd",              test_heevd,     Section::heev }, // tested via LAPACKE using gcc/MKL
  //   { "hpevd",              test_hpevd,     Section::heev }, // tested via LAPACKE using gcc/MKL
  //   { "hbevd",              test_hbevd,     Section::heev }, // tested via LAPACKE using gcc/MKL
  //   { "",                   nullptr,        Section::newline },

  // //{ "heevr",              test_heevr,     Section::heev },
  // //{ "hpevr",              test_hpevr,     Section::heev },
  // //{ "hbevr",              test_hbevr,     Section::heev },
  //   { "",                   nullptr,        Section::newline },

  //   { "hetrd",              test_hetrd,     Section::heev }, // tested via LAPACKE using gcc/MKL
  //   { "hptrd",              test_hptrd,     Section::heev }, // tested via LAPACKE using gcc/MKL
  // //{ "hbtrd",              test_hbtrd,     Section::heev },
  //   { "",                   nullptr,        Section::newline },

  //   { "ungtr",              test_ungtr,     Section::heev }, // tested via LAPACKE using gcc/MKL
  //   { "upgtr",              test_upgtr,     Section::heev }, // tested via LAPACKE using gcc/MKL
  // //{ "obgtr",              test_obgtr,     Section::heev }, // TODO does this exist
  //   { "",                   nullptr,        Section::newline },

  //   { "unmtr",              test_unmtr,     Section::heev }, // tested via LAPACKE using gcc/MKL
  // //{ "upmtr",              test_upmtr,     Section::heev },
  // //{ "obmtr",              test_obmtr,     Section::heev }, // does this exist
  //   { "",                   nullptr,        Section::newline },

  //   // -----
  //   // generalized symmetric eigenvalues
  // //{ "sygv",               test_sygv,      Section::sygv },
  // //{ "spgv",               test_spgv,      Section::sygv },
  // //{ "sbgv",               test_sbgv,      Section::sygv },
  //   { "",                   nullptr,        Section::newline },

  // //{ "sygvx",              test_sygvx,     Section::sygv },
  // //{ "spgvx",              test_spgvx,     Section::sygv },
  // //{ "sbgvx",              test_sbgvx,     Section::sygv },
  //   { "",                   nullptr,        Section::newline },

  // //{ "sygvd",              test_sygvd,     Section::sygv },
  // //{ "spgvd",              test_spgvd,     Section::sygv },
  // //{ "sbgvd",              test_sbgvd,     Section::sygv },
  //   { "",                   nullptr,        Section::newline },

  // //{ "sygvr",              test_sygvr,     Section::sygv },
  // //{ "spgvr",              test_spgvr,     Section::sygv },
  // //{ "sbgvr",              test_sbgvr,     Section::sygv },
  //   { "",                   nullptr,        Section::newline },

  // //{ "sygst",              test_sygst,     Section::sygv },
  // //{ "spgst",              test_spgst,     Section::sygv },
  // //{ "sbgst",              test_sbgst,     Section::sygv },
  //   { "",                   nullptr,        Section::newline },

  //   // -----
  //   // non-symmetric eigenvalues
  //   { "geev",               test_geev,      Section::geev },
  // //{ "ggev",               test_ggev,      Section::geev },
  //   { "",                   nullptr,        Section::newline },

  // //{ "geevx",              test_geevx,     Section::geev },
  // //{ "ggevx",              test_ggevx,     Section::geev },
  //   { "",                   nullptr,        Section::newline },

  // //{ "gees",               test_gees,      Section::geev },
  // //{ "gges",               test_gges,      Section::geev },
  //   { "",                   nullptr,        Section::newline },

  // //{ "geesx",              test_geesx,     Section::geev },
  // //{ "ggesx",              test_ggesx,     Section::geev },
  //   { "",                   nullptr,        Section::newline },

  //   { "gehrd",              test_gehrd,     Section::geev },
  // //{ "orghr",              test_orghr,     Section::geev },
  // //{ "ormhr",              test_ormhr,     Section::geev },
  // //{ "hsein",              test_hsein,     Section::geev },
  // //{ "trevc",              test_trevc,     Section::geev },
  //   { "",                   nullptr,        Section::newline },

  //   // -----
  //   // driver: singular value decomposition
  //   { "gesvd",              test_gesvd,         Section::svd },
  // //{ "gesvd_2stage",       test_gesvd_2stage,  Section::svd },
  //   { "",                   nullptr,            Section::newline },

  //   { "gesdd",              test_gesdd,         Section::svd },
  // //{ "gesdd_2stage",       test_gesdd_2stage,  Section::svd },
  //   { "",                   nullptr,            Section::newline },

  // //{ "gesvdx",             test_gesvdx,        Section::svd },
  // //{ "gesvdx_2stage",      test_gesvdx_2stage, Section::svd },
  //   { "",                   nullptr,            Section::newline },

  // //{ "gejsv",              test_gejsv,     Section::svd },
  // //{ "gesvj",              test_gesvj,     Section::svd },
  //   { "",                   nullptr,        Section::newline },

  //   // -----
  //   // auxiliary
  //   { "lacpy",              test_lacpy,     Section::aux },
  //   { "laset",              test_laset,     Section::aux },
  //   { "laswp",              test_laswp,     Section::aux },
  //   { "",                   nullptr,        Section::newline },

  //   // auxiliary: Householder
  //   { "larfg",              test_larfg,     Section::aux_householder },
  //   { "larf",               test_larf,      Section::aux_householder },
  //   { "larfx",              test_larfx,     Section::aux_householder },
  //   { "larfb",              test_larfb,     Section::aux_householder },
  //   { "larft",              test_larft,     Section::aux_householder },
  //   { "",                   nullptr,        Section::newline },

    // auxiliary: norms
    { "genorm",             test_genorm,       Section::aux_norm },
    { "synorm",             test_synorm,       Section::aux_norm },
    { "trnorm",             test_trnorm,       Section::aux_norm },
    { "",                   nullptr,           Section::newline },

  //   { "",                   nullptr,        Section::aux_norm },
  //   { "lanhp",              test_lanhp,     Section::aux_norm },
  //   { "lansp",              test_lansp,     Section::aux_norm },
  // //{ "lantp",              test_lantp,     Section::aux_norm },
  //   { "",                   nullptr,        Section::newline },

  // //{ "langb",              test_langb,     Section::aux_norm },
  //   { "lanhb",              test_lanhb,     Section::aux_norm },
  //   { "lansb",              test_lansb,     Section::aux_norm },
  // //{ "lantb",              test_lantb,     Section::aux_norm },
  //   { "",                   nullptr,        Section::newline },

  // //{ "langt",              test_langt,     Section::aux_norm },
  // //{ "lanht",              test_lanht,     Section::aux_norm },
  // //{ "lanst",              test_lanst,     Section::aux_norm },
  //   { "",                   nullptr,        Section::newline },

  //   // auxiliary: matrix generation
  // //{ "lagge",              test_lagge,     Section::aux_gen },
  // //{ "lagsy",              test_lagsy,     Section::aux_gen },
  // //{ "laghe",              test_laghe,     Section::aux_gen },
  // //{ "lagtr",              test_lagtr,     Section::aux_gen },
  //   { "",                   nullptr,        Section::newline },
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
    lookahead ( "lookahead", 9,  ParamType::List, 1,       0, 1000000, "lookahead" ),

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

    time      ( "SLATE\ntime (s)",       15, 9, ParamType::Output, libtest::no_data_flag,   0,   0, "time to solution" ),
    gflops    ( "SLATE\nGflop/s",        10, 3, ParamType::Output, libtest::no_data_flag,   0,   0, "Gflop/s rate" ),
    iters     ( "iters",        6,    ParamType::Output,                     0,   0,   0, "iterations to solution" ),

    ref_time  ( "Ref.\ntime (s)",        15, 9, ParamType::Output, libtest::no_data_flag,   0,   0, "reference time to solution" ),
    ref_gflops( "Ref.\nGflop/s",         10, 3, ParamType::Output, libtest::no_data_flag,   0,   0, "reference Gflop/s rate" ),
    ref_iters ( "Ref.\niters",            6,    ParamType::Output,                     0,   0,   0, "reference iterations to solution" ),

    // default -1 means "no check"
    //          name,     w, type,              def, min, max, help
    okay      ( "status", 6, ParamType::Output,  -1,   0,   0, "success indicator" )
{
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
    target  .value();

    // routine's parameters are marked by the test routine; see main
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    // check that all sections have names
    assert( sizeof(section_names)/sizeof(*section_names) == Section::num_sections );

    // MPI initializations
    int mpi_rank, mpi_size, provided;
    if (! ( ( MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided) == MPI_SUCCESS ) &&
            ( provided >= MPI_THREAD_MULTIPLE ) &&
            ( MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank) == MPI_SUCCESS ) &&
            ( MPI_Comm_size(MPI_COMM_WORLD, &mpi_size) == MPI_SUCCESS ) ) ) {
        fprintf( stderr, "Error: MPI could not be initialized (requires MPI_THREAD_MULTIPLE)\n" );
        return -1;
    }

    // Usage: test routine [params]
    // find routine to test
    if (argc < 2 ||
        strcmp( argv[1], "-h" ) == 0 ||
        strcmp( argv[1], "--help" ) == 0)
    {
        if ( mpi_rank == 0 ) usage( argc, argv, routines, section_names );
        return 0;
    }

    const char* routine = argv[1];
    libtest::test_func_ptr test_routine = find_tester( routine, routines );
    if (test_routine == nullptr) {
        if ( mpi_rank == 0 ) {
            fprintf( stderr, "Error: routine %s not found\n", routine );
            usage( argc, argv, routines, section_names );
        }
        return -1;
    }

    // mark fields that are used (run=false)
    Params params;
    test_routine( params, false );

    // parse parameters after routine name
    params.parse( routine, argc-2, argv+2 );

    // print input so running `test [input] > out.txt` documents input
    if  ( mpi_rank == 0 ) {
        printf( "input: %s", argv[0] );
        for (int i = 1; i < argc; ++i) {
            printf( " %s", argv[i] );
        }
        printf( "\n" );
    }

    // run tests
    int status = 0;
    int repeat = params.repeat.value();
    libtest::DataType last = params.datatype.value();
    if ( mpi_rank == 0 ) params.header();
    do {
        if (params.datatype.value() != last) {
            last = params.datatype.value();
            printf( "\n" );
        }
        for (int iter = 0; iter < repeat; ++iter) {
            try {
                test_routine( params, true );
            }
            catch (slate::Exception& err) {
                params.okay.value() = false;
                printf( "SLATE error: %s\n", err.what() );
            }
            catch (blas::Error& err) {
                params.okay.value() = false;
                printf( "BLAS error: %s\n", err.what() );
            }
            catch (lapack::Error& err) {
                params.okay.value() = false;
                printf( "LAPACK error: %s\n", err.what() );
            }
            catch (std::exception& e) {
                // happens for assert_throw failures
                params.okay.value() = false;
                printf( "Caught std::exception\n" );
            }
            catch (...) {
                // happens for assert_throw failures
                params.okay.value() = false;
                printf( "Caught unknown error when calling test routine\n" );
            }
            if ( mpi_rank == 0 ) params.print();
            status += ! params.okay.value();
            params.reset_output();
        }
        if (repeat > 1) {
            printf( "\n" );
        }
    } while( params.next() );

    MPI_Finalize();

    if (mpi_rank==0) {
        if (status) {
            printf( "%d tests FAILED.\n", status );
        } else {
            printf( "All tests passed.\n" );
        }
    }

    if (mpi_rank==0)
        return(status);
    else
        return(0);
}
