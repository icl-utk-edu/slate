// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Tile.hh"
#include "slate/Tile_blas.hh"
#include "slate/internal/util.hh"
#include "slate/print.hh"

#include "unit_test.hh"

using slate::roundup;

namespace test {

//------------------------------------------------------------------------------
// global variables
int mpi_rank;
int mpi_size;
int num_devices;

//------------------------------------------------------------------------------
/// Sets Aij = (mpi_rank + 1)*1000 + i + j/1000, for all i, j.
template <typename scalar_t>
void setup_data(slate::Tile<scalar_t>& A)
{
    //int m = A.mb();
    int n = A.nb();
    int lda = A.stride();
    scalar_t* Ad = A.data();
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < lda; ++i) {  // note: to lda, not just m
            Ad[ i + j*lda ] = (mpi_rank + 1)*1000 + i + j/1000.;
        }
    }
}

//------------------------------------------------------------------------------
/// Sets Aij = 0, for all i, j.
template <typename scalar_t>
void clear_data(slate::Tile<scalar_t>& A)
{
    int m = A.mb();
    int n = A.nb();
    int lda = A.stride();
    scalar_t* Ad = A.data();
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            Ad[ i + j*lda ] = 0;
        }
    }
}

//------------------------------------------------------------------------------
/// Verifies that:
/// Aij = (expect_rank + 1)*1000 + i + j/1000 for 0 <= i < m,
/// using A(i, j) operator, and
/// Aij = (mpi_rank  + 1)*1000 + i + j/1000 for m <= i < stride.
/// expect_rank is where the data is coming from.
template <typename scalar_t>
void verify_data_(slate::Tile<scalar_t>& A, int expect_rank,
                  const char* file, int line)
{
    try {
        int m = A.mb();
        int n = A.nb();
        int lda = A.stride();
        scalar_t* Ad = A.data();
        for (int j = 0; j < n; ++j) {
            // for i in [0, m), use expect_rank
            for (int i = 0; i < m; ++i) {
                test_assert(
                    A(i, j)    == (expect_rank + 1)*1000 + i + j/1000.);
                test_assert(
                    A.at(i, j) == (expect_rank + 1)*1000 + i + j/1000.);
            }

            // for i in [m, lda), use actual rank
            // (data in padding shouldn't be modified)
            for (int i = m; i < lda; ++i) {
                test_assert(
                    Ad[i + j*lda] == (mpi_rank + 1)*1000 + i + j/1000.);
            }
        }
    }
    catch (AssertError& e) {
        throw AssertError(e.what(), file, line);
    }
}

#define verify_data(A, expect_rank) \
        verify_data_(A, expect_rank, __FILE__, __LINE__)

//------------------------------------------------------------------------------
/// Verifies that:
/// Aij = (expect_rank + 1)*1000 + j + i/1000 for 0 <= i < m,
/// using A(i, j) operator.
/// Doesn't check data in padding (m <= i < stride),
/// since A(i, j) operator won't allow access.
/// expect_rank is where the data is coming from.
template <typename scalar_t>
void verify_data_transpose_(slate::Tile<scalar_t>& A, int expect_rank,
                            const char* file, int line)
{
    try {
        int m = A.mb();
        int n = A.nb();
        for (int j = 0; j < n; ++j) {
            // for i in [0, m), use expect_rank
            for (int i = 0; i < m; ++i) {
                test_assert(
                    A(i, j) == (expect_rank + 1)*1000 + j + i/1000.);
            }
        }
    }
    catch (AssertError& e) {
        throw AssertError(e.what(), file, line);
    }
}

#define verify_data_transpose(A, expect_rank) \
        verify_data_transpose_(A, expect_rank, __FILE__, __LINE__)

//------------------------------------------------------------------------------
/// Verifies that:
/// Aij = (expect_rank + 1)*1000 + j + i/1000 for 0 <= i < m,
/// using A(i, j) operator.
/// Doesn't check data in padding (m <= i < stride),
/// since A(i, j) operator won't allow access.
/// expect_rank is where the data is coming from.
//
// todo: setup data with imaginary bits.
template <typename scalar_t>
void verify_data_conjTranspose_(slate::Tile<scalar_t>& A, int expect_rank,
                                 const char* file, int line)
{
    try {
        using blas::conj;
        int m = A.mb();
        int n = A.nb();
        for (int j = 0; j < n; ++j) {
            // for i in [0, m), use expect_rank
            for (int i = 0; i < m; ++i) {
                test_assert(
                    A(i, j) == conj((expect_rank + 1)*1000 + j + i/1000.));
            }
        }
    }
    catch (AssertError& e) {
        throw AssertError(e.what(), file, line);
    }
}

#define verify_data_conjTranspose(A, expect_rank) \
        verify_data_conjTranspose_(A, expect_rank, __FILE__, __LINE__)

//------------------------------------------------------------------------------
/// Tests Tile() default constructor and simple data accessors.
void test_Tile_default()
{
    slate::Tile<double> A;

    test_assert(A.mb() == 0);
    test_assert(A.nb() == 0);
    test_assert(A.op() == slate::Op::NoTrans);
    test_assert(A.uplo() == blas::Uplo::General);
    test_assert(A.uploLogical() == blas::Uplo::General);
    test_assert(A.data() == nullptr);
    test_assert(A.origin() == true);      // note: TileKind::UserOwned
    test_assert(A.workspace() == false);  // note
    test_assert(A.allocated() == false);  // note
    test_assert(A.device() == -1);        // what should this be?
    test_assert(A.bytes() == 0);
    test_assert(A.size() == 0);
}

//------------------------------------------------------------------------------
/// Tests Tile(m, n, data, ...) constructor and simple data accessors.
template <typename scalar_t>
void test_Tile_data()
{
    const int m = 20;
    const int n = 30;
    const int lda = roundup(m, 32);
    scalar_t data[ lda * n ];

    // with device = -1, kind = UserOwned
    slate::Tile<scalar_t> A(m, n, data, lda, -1, slate::TileKind::UserOwned);

    test_assert(A.mb() == m);
    test_assert(A.nb() == n);
    test_assert(A.stride() == lda);
    test_assert(A.op() == slate::Op::NoTrans);
    test_assert(A.uplo() == blas::Uplo::General);
    test_assert(A.uploLogical() == blas::Uplo::General);
    test_assert(A.data() == data);
    test_assert(A.origin() == true);      // note differences
    test_assert(A.workspace() == false);  // note
    test_assert(A.allocated() == false);  // note
    test_assert(A.device() == -1);        // note
    test_assert(A.bytes() == sizeof(scalar_t) * m * n);
    test_assert(A.size() == size_t(m * n));

    // with device = 1, kind = UserOwned
    slate::Tile<scalar_t> B(m, n, data, lda, 1, slate::TileKind::UserOwned);

    test_assert(B.mb() == m);
    test_assert(B.nb() == n);
    test_assert(B.stride() == lda);
    test_assert(B.op() == slate::Op::NoTrans);
    test_assert(B.uplo() == blas::Uplo::General);
    test_assert(B.uploLogical() == blas::Uplo::General);
    test_assert(B.data() == data);
    test_assert(B.origin() == true);      // note
    test_assert(B.workspace() == false);  // note
    test_assert(B.allocated() == false);  // note
    test_assert(B.device() == 1);         // note
    test_assert(B.bytes() == sizeof(scalar_t) * m * n);
    test_assert(B.size() == size_t(m * n));

    // with device = 2, kind = Workspace
    slate::Tile<scalar_t> C(m, n, data, lda, 2, slate::TileKind::Workspace);

    test_assert(C.mb() == m);
    test_assert(C.nb() == n);
    test_assert(C.stride() == lda);
    test_assert(C.op() == slate::Op::NoTrans);
    test_assert(C.uplo() == blas::Uplo::General);
    test_assert(C.uploLogical() == blas::Uplo::General);
    test_assert(C.data() == data);
    test_assert(C.origin() == false);     // note
    test_assert(C.workspace() == true);   // note
    test_assert(C.allocated() == true);   // note
    test_assert(C.device() == 2);         // note
    test_assert(C.bytes() == sizeof(scalar_t) * m * n);
    test_assert(C.size() == size_t(m * n));

    // with device = 2, kind = SlateOwned
    slate::Tile<scalar_t> D(m, n, data, lda, 2, slate::TileKind::SlateOwned);

    test_assert(D.mb() == m);
    test_assert(D.nb() == n);
    test_assert(D.stride() == lda);
    test_assert(D.op() == slate::Op::NoTrans);
    test_assert(D.uplo() == blas::Uplo::General);
    test_assert(D.uploLogical() == blas::Uplo::General);
    test_assert(D.data() == data);
    test_assert(D.origin() == true);      // note
    test_assert(D.workspace() == false);  // note
    test_assert(D.allocated() == true);   // note
    test_assert(D.device() == 2);         // note
    test_assert(D.bytes() == sizeof(scalar_t) * m * n);
    test_assert(D.size() == size_t(m * n));
}

void test_Tile_data_double()
{
    test_Tile_data< double >();

    slate::Tile<double> A;
    test_assert(A.is_real);
    test_assert(! A.is_complex);
}

void test_Tile_data_complex()
{
    test_Tile_data< std::complex<double> >();

    slate::Tile< std::complex<double> > A;
    test_assert(! A.is_real);
    test_assert(A.is_complex);
}

//------------------------------------------------------------------------------
/// Tests transpose(Tile).
template <typename scalar_t>
void test_transpose()
{
    const int m = 20;
    const int n = 30;
    const int lda = roundup(m, 32);
    scalar_t data[ lda * n ];
    slate::Tile<scalar_t> A(m, n, data, lda, -1, slate::TileKind::UserOwned);
    setup_data(A);

    //----- transpose
    auto AT = transpose(A);

    test_assert(AT.mb() == n);  // trans
    test_assert(AT.nb() == m);  // trans
    test_assert(AT.stride() == lda);
    test_assert(AT.op() == blas::Op::Trans);  // trans
    test_assert(AT.uplo() == blas::Uplo::General);
    test_assert(AT.uploLogical() == blas::Uplo::General);
    test_assert(AT.data() == data);
    test_assert(AT.origin() == true);
    test_assert(AT.device() == -1);
    test_assert(AT.bytes() == sizeof(scalar_t) * m * n);
    test_assert(AT.size() == size_t(m * n));

    verify_data_transpose(AT, mpi_rank);

    //----- transpose again
    auto ATT = transpose(AT);

    test_assert(ATT.mb() == m);  // restored
    test_assert(ATT.nb() == n);  // restored
    test_assert(ATT.stride() == lda);
    test_assert(ATT.op() == blas::Op::NoTrans);  // restored
    test_assert(ATT.uplo() == blas::Uplo::General);
    test_assert(ATT.uploLogical() == blas::Uplo::General);
    test_assert(ATT.data() == data);
    test_assert(ATT.origin() == true);
    test_assert(AT.device() == -1);
    test_assert(ATT.bytes() == sizeof(scalar_t) * m * n);
    test_assert(ATT.size() == size_t(m * n));

    verify_data(ATT, mpi_rank);
}

void test_transpose_double()
{
    test_transpose< double >();
}

void test_transpose_complex()
{
    test_transpose< std::complex<double> >();
}

//------------------------------------------------------------------------------
/// Tests conjTranspose(Tile).
template <typename scalar_t>
void test_conjTranspose()
{
    const int m = 20;
    const int n = 30;
    const int lda = roundup(m, 32);
    scalar_t data[ lda * n ];
    slate::Tile<scalar_t> A(m, n, data, lda, -1, slate::TileKind::UserOwned);
    setup_data(A);

    //----- conjTranspose
    auto AC = conjTranspose(A);

    test_assert(AC.mb() == n);  // trans
    test_assert(AC.nb() == m);  // trans
    test_assert(AC.stride() == lda);
    test_assert(AC.op() == blas::Op::ConjTrans);  // conj-trans
    test_assert(AC.uplo() == blas::Uplo::General);
    test_assert(AC.uploLogical() == blas::Uplo::General);
    test_assert(AC.data() == data);
    test_assert(AC.origin() == true);
    test_assert(AC.device() == -1);
    test_assert(AC.bytes() == sizeof(scalar_t) * m * n);
    test_assert(AC.size() == size_t(m * n));

    verify_data_conjTranspose(AC, mpi_rank);

    //----- conjTranspose again
    auto ACC = conjTranspose(AC);

    test_assert(ACC.mb() == m);  // restored
    test_assert(ACC.nb() == n);  // restored
    test_assert(ACC.stride() == lda);
    test_assert(ACC.op() == blas::Op::NoTrans);  // restored
    test_assert(ACC.uplo() == blas::Uplo::General);
    test_assert(ACC.uploLogical() == blas::Uplo::General);
    test_assert(ACC.data() == data);
    test_assert(ACC.origin() == true);
    test_assert(ACC.device() == -1);
    test_assert(ACC.bytes() == sizeof(scalar_t) * m * n);
    test_assert(ACC.size() == size_t(m * n));

    verify_data(ACC, mpi_rank);

    auto AT = transpose(A);
    if (AT.is_real) {
        //----- transpose + conjTranspose for real
        auto ATC = conjTranspose(AT);

        test_assert(ATC.mb() == m);  // restored
        test_assert(ATC.nb() == n);  // restored
        test_assert(ATC.stride() == lda);
        test_assert(ATC.op() == blas::Op::NoTrans);  // restored
        test_assert(ATC.uplo() == blas::Uplo::General);
        test_assert(ATC.uploLogical() == blas::Uplo::General);
        test_assert(ATC.data() == data);
        test_assert(ATC.origin() == true);
        test_assert(ATC.device() == -1);
        test_assert(ATC.bytes() == sizeof(scalar_t) * m * n);
        test_assert(ATC.size() == size_t(m * n));

        verify_data(ATC, mpi_rank);

        //----- conjTranspose + transpose for real
        auto ACT = transpose(AC);

        test_assert(ACT.mb() == m);  // restored
        test_assert(ACT.nb() == n);  // restored
        test_assert(ACT.stride() == lda);
        test_assert(ACT.op() == blas::Op::NoTrans);  // restored
        test_assert(ACT.uplo() == blas::Uplo::General);
        test_assert(ACT.uploLogical() == blas::Uplo::General);
        test_assert(ACT.data() == data);
        test_assert(ACT.origin() == true);
        test_assert(ACT.device() == -1);
        test_assert(ACT.bytes() == sizeof(scalar_t) * m * n);
        test_assert(ACT.size() == size_t(m * n));

        verify_data(ATC, mpi_rank);
    }
    else {
        //----- transpose + conjTranspose is unsupported for complex
        test_assert_throw_std(conjTranspose(AT) /* std::exception */);
        test_assert_throw_std(transpose(AC)      /* std::exception */);
    }
}

void test_conjTranspose_double()
{
    test_conjTranspose< double >();
}

void test_conjTranspose_complex()
{
    test_conjTranspose< std::complex<double> >();
}

//------------------------------------------------------------------------------
/// Tests setting uplo, getting uplo and uploLogical with transposes.
template <typename scalar_t>
void test_lower()
{
    const int m = 20;
    const int n = 30;
    const int lda = roundup(m, 32);
    scalar_t data[ lda * n ];
    slate::Tile<scalar_t> A(m, n, data, lda, -1, slate::TileKind::UserOwned);
    setup_data(A);

    A.uplo(slate::Uplo::Lower);
    test_assert(A.uplo()         == blas::Uplo::Lower);
    test_assert(A.uploLogical()  == blas::Uplo::Lower);
    test_assert(A.uploPhysical() == blas::Uplo::Lower);

    auto AT = transpose(A);
    test_assert(AT.uplo()         == blas::Uplo::Upper);
    test_assert(AT.uploLogical()  == blas::Uplo::Upper);
    test_assert(AT.uploPhysical() == blas::Uplo::Lower);

    auto ATT = transpose(AT);
    test_assert(ATT.uplo()         == blas::Uplo::Lower);
    test_assert(ATT.uploLogical()  == blas::Uplo::Lower);
    test_assert(ATT.uploPhysical() == blas::Uplo::Lower);

    auto AC = conjTranspose(A);
    test_assert(AC.uplo()         == blas::Uplo::Upper);
    test_assert(AC.uploLogical()  == blas::Uplo::Upper);
    test_assert(AC.uploPhysical() == blas::Uplo::Lower);

    auto ACC = conjTranspose(AC);
    test_assert(ACC.uplo()         == blas::Uplo::Lower);
    test_assert(ACC.uploLogical()  == blas::Uplo::Lower);
    test_assert(ACC.uploPhysical() == blas::Uplo::Lower);
}

void test_lower_double()
{
    test_lower< double >();
}

void test_lower_complex()
{
    test_lower< std::complex<double> >();
}

//------------------------------------------------------------------------------
/// Tests setting uplo, getting uplo and uploLogical with transposes.
template <typename scalar_t>
void test_upper()
{
    const int m = 20;
    const int n = 30;
    const int lda = roundup(m, 32);
    scalar_t data[ lda * n ];
    slate::Tile<scalar_t> A(m, n, data, lda, -1, slate::TileKind::UserOwned);
    setup_data(A);

    A.uplo(slate::Uplo::Upper);
    test_assert(A.uplo()         == blas::Uplo::Upper);
    test_assert(A.uploLogical()  == blas::Uplo::Upper);
    test_assert(A.uploPhysical() == blas::Uplo::Upper);

    auto AT = transpose(A);
    test_assert(AT.uplo()         == blas::Uplo::Lower);
    test_assert(AT.uploLogical()  == blas::Uplo::Lower);
    test_assert(AT.uploPhysical() == blas::Uplo::Upper);

    auto ATT = transpose(AT);
    test_assert(ATT.uplo()         == blas::Uplo::Upper);
    test_assert(ATT.uploLogical()  == blas::Uplo::Upper);
    test_assert(ATT.uploPhysical() == blas::Uplo::Upper);

    auto AC = conjTranspose(A);
    test_assert(AC.uplo()         == blas::Uplo::Lower);
    test_assert(AC.uploLogical()  == blas::Uplo::Lower);
    test_assert(AC.uploPhysical() == blas::Uplo::Upper);

    auto ACC = conjTranspose(AC);
    test_assert(ACC.uplo()         == blas::Uplo::Upper);
    test_assert(ACC.uploLogical()  == blas::Uplo::Upper);
    test_assert(ACC.uploPhysical() == blas::Uplo::Upper);
}

void test_upper_double()
{
    test_upper< double >();
}

void test_upper_complex()
{
    test_upper< std::complex<double> >();
}

//------------------------------------------------------------------------------
/// Tests send() and recv() between MPI ranks.
/// src/dst lda is rounded up to multiple of align_src/dst, respectively.
void test_send_recv(int align_src, int align_dst)
{
    if (mpi_size == 1) {
        test_skip("requires MPI comm size > 1");
    }

    const int m = 20;
    const int n = 30;
    // even is src, odd is dst
    int lda = roundup(m, (mpi_rank % 2 == 0 ? align_src : align_dst));
    double* data = new double[ lda * n ];
    assert(data != nullptr);
    slate::Tile<double> A(m, n, data, lda, -1, slate::TileKind::UserOwned);
    setup_data(A);

    int r = int(mpi_rank / 2) * 2;
    if (r+1 < mpi_size) {
        // send from r to r+1
        if (r == mpi_rank) {
            A.send(r+1, MPI_COMM_WORLD);
        }
        else {
            A.recv(r, MPI_COMM_WORLD, A.layout());
        }
        verify_data(A, r);
    }
    else {
        verify_data(A, mpi_rank);
    }

    delete[] data;
}

// contiguous => contiguous
void test_send_recv_cc()
{
    test_send_recv(1, 1);
}

// contiguous => strided
void test_send_recv_cs()
{
    test_send_recv(1, 32);
}

// strided => contiguous
void test_send_recv_sc()
{
    test_send_recv(32, 1);
}

// strided => strided
void test_send_recv_ss()
{
    test_send_recv(32, 32);
}

//------------------------------------------------------------------------------
/// Tests bcast() between MPI ranks.
/// src/dst lda is rounded up to multiple of align_src/dst, respectively.
void test_bcast(int align_src, int align_dst)
{
    const int m = 20;
    const int n = 30;
    // rank 0 is dst (root)
    int lda = roundup(m, (mpi_rank == 0 ? align_dst : align_src));
    double* data = new double[ lda * n ];
    assert(data != nullptr);
    slate::Tile<double> A(m, n, data, lda, -1, slate::TileKind::UserOwned);
    setup_data(A);

    // with root = 0
    A.bcast(0, MPI_COMM_WORLD);
    verify_data(A, 0);

    if (mpi_size > 1) {
        // with root = 1
        setup_data(A);
        A.bcast(1, MPI_COMM_WORLD);
        verify_data(A, 1);
    }

    delete[] data;
}

// contiguous => contiguous
void test_bcast_cc()
{
    test_bcast(1, 1);
}

// contiguous => strided
void test_bcast_cs()
{
    test_bcast(1, 32);
}

// strided => contiguous
void test_bcast_sc()
{
    test_bcast(32, 1);
}

// strided => strided
void test_bcast_ss()
{
    test_bcast(32, 32);
}

//------------------------------------------------------------------------------
/// Tests copyData().
/// host/device lda is rounded up to multiple of align_host/dev, respectively.
void test_copyData(int align_host, int align_dev)
{
    if (num_devices == 0) {
        test_skip("requires num_devices > 0");
    }

    const int m = 20;
    const int n = 30;
    int lda = roundup(m, align_host);
    int ldda = roundup(m, align_dev);
    double* dataA = new double[ lda * n ];
    double* dataB = new double[ lda * n ];
    slate::Tile<double> A(m, n, dataA, lda, -1, slate::TileKind::UserOwned);
    slate::Tile<double> B(m, n, dataB, lda, -1, slate::TileKind::UserOwned);
    setup_data(A);
    // set B, including padding, then clear B, excluding padding,
    // so the padding remains setup for verify_data.
    setup_data(B);
    clear_data(B);

    int device_idx;
    blas::get_device(&device_idx);
    const int batch_arrays_index = 0;
    blas::Queue queue(device_idx, batch_arrays_index);

    double* Adata_dev;
    double* Bdata_dev;
    Adata_dev = blas::device_malloc<double>(ldda * n, queue);
    test_assert(Adata_dev != nullptr);
    Bdata_dev = blas::device_malloc<double>(ldda * n, queue);
    test_assert(Bdata_dev != nullptr);

    slate::Tile<double> dA(m, n, Adata_dev, ldda, 0, slate::TileKind::UserOwned);
    slate::Tile<double> dB(m, n, Bdata_dev, ldda, 0, slate::TileKind::UserOwned);

    // copy H2D->D2D->D2H, then verify
    A.copyData(&dA, queue);
    dA.copyData(&dB, queue);
    dB.copyData(&B, queue);
    verify_data(B, mpi_rank);

    // copy host to host, then verify
    clear_data(B);
    A.copyData(&B);
    verify_data(B, mpi_rank);

    blas::device_free(Adata_dev, queue);
    blas::device_free(Bdata_dev, queue);

    delete[] dataA;
    delete[] dataB;
}

// contiguous => contiguous
void test_copyData_cc()
{
    test_copyData(1, 1);
}

// contiguous => strided
void test_copyData_cs()
{
    test_copyData(1, 32);
}

// strided => contiguous
void test_copyData_sc()
{
    test_copyData(32, 1);
}

// strided => strided
void test_copyData_ss()
{
    test_copyData(32, 32);
}

//------------------------------------------------------------------------------
/// Tests slate::print( "label", Tile )
template <typename scalar_t>
void test_print()
{
    const int m = 4;
    const int n = 5;
    const int lda = roundup( m, 32 );
    scalar_t data[ lda * n ];

    int device = 0;
    blas::Queue queue( device, 0 );

    // with device = -1, kind = UserOwned
    slate::Tile<scalar_t> A( m, n, data, lda, -1, slate::TileKind::UserOwned );
    setup_data( A );

    //==================================================
    // Run SLATE test
    printf( "\n" );
    slate::print( "A", A, queue );

    if (num_devices > 0) {
        scalar_t* Adata_dev = blas::device_malloc<scalar_t>( lda * n, queue );
        test_assert( Adata_dev != nullptr );

        slate::Tile<scalar_t> dA( m, n, Adata_dev, lda, 0,
                                  slate::TileKind::UserOwned );
        A.copyData( &dA, queue );

        //==================================================
        // Run SLATE test
        slate::print( "dA", dA, queue );

        blas::device_free( Adata_dev, queue);
    }
}

void test_print_double()
{
    test_print< double >();
}

void test_print_complex()
{
    test_print< std::complex<double> >();
}

//------------------------------------------------------------------------------
/// Runs all tests. Called by unit test main().
void run_tests()
{
    if (mpi_rank == 0) {
        run_test(
            test_Tile_default,
            "Tile(); also mb, nb, uplo, etc.");
        run_test(
            test_Tile_data_double,
            "Tile(m, n, data, ...) double");
        run_test(
            test_Tile_data_complex,
            "Tile(m, n, data, ...) complex");
        run_test(
            test_transpose_double,
            "transpose, double");
        run_test(
            test_transpose_complex,
            "transpose, complex");
        run_test(
            test_conjTranspose_double,
            "conjTranspose, double");
        run_test(
            test_conjTranspose_complex,
            "conjTranspose, complex");
        run_test(
            test_lower_double,
            "uplo(lower)");
        run_test(
            test_lower_complex,
            "uplo(lower)");
        run_test(
            test_upper_double,
            "uplo(upper)");
        run_test(
            test_upper_complex,
            "uplo(upper)");
        run_test(
            test_copyData_cc,
            "copyData: (H2D, D2D, D2H, H2H) contiguous => contiguous");
        run_test(
            test_copyData_cs,
            "copyData: (H2D, D2D, D2H, H2H) contiguous => strided");
        run_test(
            test_copyData_sc,
            "copyData: (H2D, D2D, D2H, H2H) strided => contiguous");
        run_test(
            test_copyData_ss,
            "copyData: (H2D, D2D, D2H, H2H) strided => strided");
        run_test(
            test_print_double,
            "print, double");
        run_test(
            test_print_complex,
            "print, complex");
    }
    run_test(
        test_send_recv_cc,
        "send and recv, contiguous => contiguous", MPI_COMM_WORLD);
    run_test(
        test_send_recv_cs,
        "send and recv, contiguous => strided",    MPI_COMM_WORLD);
    run_test(
        test_send_recv_sc,
        "send and recv, strided => contiguous",    MPI_COMM_WORLD);
    run_test(
        test_send_recv_ss,
        "send and recv, strided => strided",       MPI_COMM_WORLD);
    run_test(
        test_bcast_cc,
        "bcast, contiguous => contiguous",         MPI_COMM_WORLD);
    run_test(
        test_bcast_cs,
        "bcast, contiguous => strided",            MPI_COMM_WORLD);
    run_test(
        test_bcast_sc,
        "bcast, strided => contiguous",            MPI_COMM_WORLD);
    run_test(
        test_bcast_ss,
        "bcast, strided => strided",               MPI_COMM_WORLD);
}

}  // namespace test

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    using namespace test;  // for globals mpi_rank, etc.

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    num_devices = blas::get_device_count();

    int err = unit_test_main(MPI_COMM_WORLD);  // which calls run_tests()

    MPI_Finalize();
    return err;
}
