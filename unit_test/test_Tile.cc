//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the University of Tennessee nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------
// This research was supported by the Exascale Computing Project (17-SC-20-SC),
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//------------------------------------------------------------------------------
// For assistance with SLATE, email <slate-user@icl.utk.edu>.
// You can also join the "SLATE User" Google group by going to
// https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user,
// signing in with your Google credentials, and then clicking "Join group".
//------------------------------------------------------------------------------

#include "slate/Tile.hh"
#include "slate/Tile_blas.hh"
#include "slate/internal/util.hh"

#include "unit_test.hh"

using slate::roundup;

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

    cudaStream_t stream;
    test_assert(cudaStreamCreate(&stream) == cudaSuccess);

    double* Adata_dev;
    double* Bdata_dev;
    test_assert(cudaMalloc((void**) &Adata_dev, sizeof(double)*ldda*n) == cudaSuccess);
    test_assert(Adata_dev != nullptr);
    test_assert(cudaMalloc((void**) &Bdata_dev, sizeof(double)*ldda*n) == cudaSuccess);
    test_assert(Bdata_dev != nullptr);

    slate::Tile<double> dA(m, n, Adata_dev, ldda, 0, slate::TileKind::UserOwned);
    slate::Tile<double> dB(m, n, Bdata_dev, ldda, 0, slate::TileKind::UserOwned);

    // copy H2D->D2D->D2H, then verify
    A.copyData(&dA, stream);
    dA.copyData(&dB, stream);
    dB.copyData(&B, stream);
    verify_data(B, mpi_rank);

    // copy host to host, then verify
    clear_data(B);
    A.copyData(&B);
    verify_data(B, mpi_rank);

    test_assert(cudaFree(Adata_dev) == cudaSuccess);
    test_assert(cudaFree(Bdata_dev) == cudaSuccess);
    test_assert(cudaStreamDestroy(stream) == cudaSuccess);

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

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    cudaGetDeviceCount(&num_devices);

    int err = unit_test_main(MPI_COMM_WORLD);  // which calls run_tests()

    MPI_Finalize();
    return err;
}
