// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "scalapack_slate.hh"

namespace slate {
namespace scalapack_api {

//------------------------------------------------------------------------------
/// SLATE ScaLAPACK wrapper sets up SLATE matrices from ScaLAPACK descriptors
/// and calls SLATE.
template <typename scalar_t>
void slate_pgels(
    const char* trans_str, blas_int m, blas_int n, blas_int nrhs,
    scalar_t* A_data, blas_int ia, blas_int ja, blas_int const* descA,
    scalar_t* B_data, blas_int ib, blas_int jb, blas_int const* descB,
    scalar_t* work, blas_int lwork,
    blas_int* info )
{
    using real_t = blas::real_type<scalar_t>;

    // Respond to workspace query with a minimal value (1); workspace
    // is allocated within the SLATE routine.
    if (lwork == -1) {
        work[0] = (real_t)1.0;
        *info = 0;
        return;
    }

    Op trans{};
    from_string( std::string( 1, trans_str[0] ), &trans );

    slate::Target target = TargetConfig::value();
    int verbose = VerboseConfig::value();
    int64_t panel_threads = PanelThreadsConfig::value();
    int64_t inner_blocking = IBConfig::value();
    int64_t lookahead = LookaheadConfig::value();
    slate::GridOrder grid_order = slate_scalapack_blacs_grid_order();

    // A is m-by-n, BX is max(m, n)-by-nrhs.
    // If op == NoTrans, op(A) is m-by-n, B is m-by-nrhs
    // otherwise,        op(A) is n-by-m, B is n-by-nrhs.
    int64_t Am = (trans == slate::Op::NoTrans ? m : n);
    int64_t An = (trans == slate::Op::NoTrans ? n : m);
    int64_t Bm = (trans == slate::Op::NoTrans ? m : n);
    int64_t Bn = nrhs;

    // create SLATE matrices from the ScaLAPACK layouts
    blas_int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo( desc_ctxt( descA ), &nprow, &npcol, &myprow, &mypcol );
    auto A = slate::Matrix<scalar_t>::fromScaLAPACK(
        desc_m( descA ), desc_n( descA ), A_data, desc_lld( descA ),
        desc_mb( descA ), desc_nb( descA ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    A = slate_scalapack_submatrix( Am, An, A, ia, ja, descA );

    Cblacs_gridinfo( desc_ctxt( descB ), &nprow, &npcol, &myprow, &mypcol );
    auto B = slate::Matrix<scalar_t>::fromScaLAPACK(
        desc_m( descB ), desc_n( descB ), B_data, desc_lld( descB ),
        desc_mb( descB ), desc_nb( descB ),
        grid_order, nprow, npcol, MPI_COMM_WORLD );
    B = slate_scalapack_submatrix( Bm, Bn, B, ib, jb, descB );

    // Apply transpose
    auto opA = A;
    if (trans == slate::Op::Trans)
        opA = transpose( A );
    else if (trans == slate::Op::ConjTrans)
        opA = conj_transpose( A );

    if (verbose && myprow == 0 && mypcol == 0)
        logprintf("%s\n", "gels");

    slate::gels( opA, B, {
        {slate::Option::Lookahead, lookahead},
        {slate::Option::Target, target},
        {slate::Option::MaxPanelThreads, panel_threads},
        {slate::Option::InnerBlocking, inner_blocking}
    });

    // todo: extract the real info
    *info = 0;
}

//------------------------------------------------------------------------------
// Fortran interfaces
// Each Fortran interface calls the type generic slate wrapper.

extern "C" {

#define SCALAPACK_psgels BLAS_FORTRAN_NAME( psgels, PSGELS )
void SCALAPACK_psgels(
    const char* trans, blas_int const* m, blas_int const* n, blas_int* nrhs,
    float* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    float* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    float* work, blas_int const* lwork,
    blas_int* info )
{
    slate_pgels(
        trans, *m, *n, *nrhs,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB,
        work, *lwork, info );
}

#define SCALAPACK_pdgels BLAS_FORTRAN_NAME( pdgels, PDGELS )
void SCALAPACK_pdgels(
    const char* trans, blas_int const* m, blas_int const* n, blas_int* nrhs,
    double* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    double* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    double* work, blas_int const* lwork,
    blas_int* info )
{
    slate_pgels(
        trans, *m, *n, *nrhs,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB,
        work, *lwork, info );
}

#define SCALAPACK_pcgels BLAS_FORTRAN_NAME( pcgels, PCGELS )
void SCALAPACK_pcgels(
    const char* trans, blas_int const* m, blas_int const* n, blas_int* nrhs,
    std::complex<float>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    std::complex<float>* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    std::complex<float>* work, blas_int const* lwork,
    blas_int* info )
{
    slate_pgels(
        trans, *m, *n, *nrhs,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB,
        work, *lwork, info );
}

#define SCALAPACK_pzgels BLAS_FORTRAN_NAME( pzgels, PZGELS )
void SCALAPACK_pzgels(
    const char* trans, blas_int const* m, blas_int const* n, blas_int* nrhs,
    std::complex<double>* A_data, blas_int const* ia, blas_int const* ja, blas_int const* descA,
    std::complex<double>* B_data, blas_int const* ib, blas_int const* jb, blas_int const* descB,
    std::complex<double>* work, blas_int const* lwork,
    blas_int* info )
{
    slate_pgels(
        trans, *m, *n, *nrhs,
        A_data, *ia, *ja, descA,
        B_data, *ib, *jb, descB,
        work, *lwork, info );
}

} // extern "C"

} // namespace scalapack_api
} // namespace slate
