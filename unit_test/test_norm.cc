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
// Need assistance with the SLATE software? Join the "SLATE User" Google group
// by going to https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user
// and clicking "Apply to join group". Upon acceptance, email your questions and
// comments to <slate-user@icl.utk.edu>.
//------------------------------------------------------------------------------

#include "slate_Tile.hh"
#include "slate_Tile_blas.hh"
#include "slate_device.hh"

#include "unit_test.hh"

using slate::Norm;
using slate::Uplo;
using slate::Diag;

//------------------------------------------------------------------------------
// global variables
int mpi_rank;
int mpi_size;
int verbose;

lapack::Norm norms[] = {
    lapack::Norm::Max,
    lapack::Norm::One,
    lapack::Norm::Inf,
    lapack::Norm::Fro
};

lapack::Uplo uplos[] = {
    lapack::Uplo::Lower,
    lapack::Uplo::Upper
};

lapack::Diag diags[] = {
    lapack::Diag::NonUnit,
    lapack::Diag::Unit
};

//------------------------------------------------------------------------------
template <typename T>
inline constexpr T roundup(T x, T y)
{
    return T((x + y - 1) / y) * y;
}

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
void test_genorm(Norm norm)
{
    const int m = 20;
    const int n = 30;
    int lda = m;
    double* dataA = new double[ lda * n ];
    slate::Tile<double> A(m, n, dataA, lda, -1);
    setup_data(A);

    std::vector<double> values;
    if (norm == lapack::Norm::Max)
        values.resize( 1 );
    else if (norm == lapack::Norm::One)
        values.resize( n );
    else if (norm == lapack::Norm::Inf)
        values.resize( m );
    else if (norm == lapack::Norm::Fro)
        values.resize( 2 );

    slate::genorm( norm, A, &values[0] );

    double result;
    if (norm == lapack::Norm::Fro) {
        assert( values.size() == 2 );
        result = values[0] * sqrt( values[1] );
    }
    else {
        result = 0;
        for (size_t i = 0; i < values.size(); ++i)
            result = std::max( result, values[i] );  // todo: max_nan
    }

    double result_ref = lapack::lange(
        norm, A.mb(), A.nb(), A.data(), A.stride() );

    double error = std::abs( result - result_ref ) / result_ref;
    if (norm == Norm::One)
        error /= sqrt( m );
    else if (norm == Norm::Inf)
        error /= sqrt( n );
    else if (norm == Norm::Fro)
        error /= sqrt( m*n );

    if (verbose) {
        printf( "\nnorm %c, result %.2e, ref %.2e, error %.2e ",
                norm2char(norm),
                result, result_ref, error );
    }

    delete[] dataA;

    double eps = std::numeric_limits<double>::epsilon();
    test_assert( error < 3*eps );
}

//-----
void test_genorm_max()
    { test_genorm(Norm::Max); }

void test_genorm_one()
    { test_genorm(Norm::One); }

void test_genorm_inf()
    { test_genorm(Norm::Inf); }

void test_genorm_fro()
    { test_genorm(Norm::Fro); }

//------------------------------------------------------------------------------
void test_trnorm(Norm norm, Uplo uplo, Diag diag)
{
    // upper: m <= n, lower: m >= n
    int m = 20;
    int n = 30;
    if (uplo == Uplo::Lower)
        std::swap(m, n);

    int lda = m;
    double* Adata = new double[ lda * n ];
    slate::Tile<double> A(m, n, Adata, lda, -1);
    setup_data(A);
    A.uplo( uplo );

    std::vector<double> values;
    if (norm == lapack::Norm::Max)
        values.resize( 1 );
    else if (norm == lapack::Norm::One)
        values.resize( n );
    else if (norm == lapack::Norm::Inf)
        values.resize( m );
    else if (norm == lapack::Norm::Fro)
        values.resize( 2 );

    slate::trnorm( norm, diag, A, &values[0] );

    double result;
    if (norm == lapack::Norm::Fro) {
        assert( values.size() == 2 );
        result = values[0] * sqrt( values[1] );
    }
    else {
        result = 0;
        for (size_t i = 0; i < values.size(); ++i)
            result = std::max( result, values[i] );  // todo: max_nan
    }

    double result_ref = lapack::lantr(
        norm, uplo, diag, A.mb(), A.nb(), A.data(), A.stride() );

    double error = std::abs( result - result_ref ) / result_ref;
    if (norm == Norm::One)
        error /= sqrt( m );
    else if (norm == Norm::Inf)
        error /= sqrt( n );
    else if (norm == Norm::Fro)
        error /= sqrt( m*n );

    if (verbose) {
        printf( "\nnorm %c, uplo %c, diag %c, result %10.4f, ref %10.4f, error %.2e ",
                norm2char(norm), uplo2char(uplo), diag2char(diag),
                result, result_ref, error );
    }

    delete[] Adata;

    double eps = std::numeric_limits<double>::epsilon();
    test_assert( error < 3*eps );
}

//-----
void test_trnorm_max_lower_unit()
    { test_trnorm(Norm::Max, Uplo::Lower, Diag::Unit); }

void test_trnorm_max_lower_nonunit()
    { test_trnorm(Norm::Max, Uplo::Lower, Diag::NonUnit); }

void test_trnorm_max_upper_unit()
    { test_trnorm(Norm::Max, Uplo::Upper, Diag::Unit); }

void test_trnorm_max_upper_nonunit()
    { test_trnorm(Norm::Max, Uplo::Upper, Diag::NonUnit); }

//-----
void test_trnorm_one_lower_unit()
    { test_trnorm(Norm::One, Uplo::Lower, Diag::Unit); }

void test_trnorm_one_lower_nonunit()
    { test_trnorm(Norm::One, Uplo::Lower, Diag::NonUnit); }

void test_trnorm_one_upper_unit()
    { test_trnorm(Norm::One, Uplo::Upper, Diag::Unit); }

void test_trnorm_one_upper_nonunit()
    { test_trnorm(Norm::One, Uplo::Upper, Diag::NonUnit); }

//-----
void test_trnorm_inf_lower_unit()
    { test_trnorm(Norm::Inf, Uplo::Lower, Diag::Unit); }

void test_trnorm_inf_lower_nonunit()
    { test_trnorm(Norm::Inf, Uplo::Lower, Diag::NonUnit); }

void test_trnorm_inf_upper_unit()
    { test_trnorm(Norm::Inf, Uplo::Upper, Diag::Unit); }

void test_trnorm_inf_upper_nonunit()
    { test_trnorm(Norm::Inf, Uplo::Upper, Diag::NonUnit); }

//-----
void test_trnorm_fro_lower_unit()
    { test_trnorm(Norm::Fro, Uplo::Lower, Diag::Unit); }

void test_trnorm_fro_lower_nonunit()
    { test_trnorm(Norm::Fro, Uplo::Lower, Diag::NonUnit); }

void test_trnorm_fro_upper_unit()
    { test_trnorm(Norm::Fro, Uplo::Upper, Diag::Unit); }

void test_trnorm_fro_upper_nonunit()
    { test_trnorm(Norm::Fro, Uplo::Upper, Diag::NonUnit); }

//------------------------------------------------------------------------------
void test_genorm_dev(Norm norm)
{
    const int m = 20;
    const int n = 30;
    int lda = m;
    double* Adata = new double[ lda * n ];
    slate::Tile<double> A(m, n, Adata, lda, -1);
    setup_data(A);

    cudaStream_t stream;
    test_assert(cudaStreamCreate(&stream) == cudaSuccess);

    double* dAdata;
    test_assert(cudaMalloc(&dAdata, sizeof(double) * lda * n) == cudaSuccess);
    test_assert(dAdata != nullptr);
    slate::Tile<double> dA(m, n, dAdata, lda, 0);
    A.copyDataToDevice(&dA, stream);

    const int batch_count = 1;
    double* Aarray[batch_count];
    double** dAarray;
    test_assert(cudaMalloc(&dAarray, sizeof(double*) * batch_count) == cudaSuccess);
    test_assert(dAarray != nullptr);
    Aarray[0] = dA.data();
    test_assert(cudaMemcpy(dAarray, Aarray, sizeof(double*) * batch_count,
                           cudaMemcpyHostToDevice ) == cudaSuccess);

    std::vector<double> values;
    size_t ldv = 1;
    if (norm == lapack::Norm::Max)
        ldv = 1;
    else if (norm == lapack::Norm::One)
        ldv = n;
    else if (norm == lapack::Norm::Inf)
        ldv = m;
    else if (norm == lapack::Norm::Fro)
        ldv = 2;
    values.resize( ldv * batch_count );

    double* dvalues;
    test_assert(cudaMalloc(&dvalues, sizeof(double) * ldv * batch_count) == cudaSuccess);
    test_assert(dvalues != nullptr);

    slate::device::genorm( norm, m, n, dAarray, lda,
                           dvalues, ldv, batch_count, stream );
    cudaStreamSynchronize( stream );
    test_assert(cudaMemcpy( &values[0], dvalues, sizeof(double) * values.size(),
                            cudaMemcpyDeviceToHost ) == cudaSuccess );

    double result;
    if (norm == lapack::Norm::Fro) {
        assert( values.size() == 2 );
        result = values[0] * sqrt( values[1] );
    }
    else {
        result = 0;
        for (size_t i = 0; i < values.size(); ++i)
            result = std::max( result, values[i] );  // todo: max_nan
    }

    double result_ref = lapack::lange(
        norm, A.mb(), A.nb(), A.data(), A.stride() );

    double error = std::abs( result - result_ref ) / result_ref;
    if (norm == Norm::One)
        error /= sqrt( m );
    else if (norm == Norm::Inf)
        error /= sqrt( n );
    else if (norm == Norm::Fro)
        error /= sqrt( m*n );

    if (verbose) {
        printf( "\nnorm %c, result %10.4f, ref %10.4f, error %.2e ",
                norm2char(norm),
                result, result_ref, error );
    }

    cudaFree( dAdata );
    cudaFree( dAarray );
    cudaFree( dvalues );
    delete[] Adata;

    double eps = std::numeric_limits<double>::epsilon();
    test_assert( error < 3*eps );
}

//-----
void test_genorm_dev_max()
    { test_genorm_dev(Norm::Max); }

void test_genorm_dev_one()
    { test_genorm_dev(Norm::One); }

void test_genorm_dev_inf()
    { test_genorm_dev(Norm::Inf); }

void test_genorm_dev_fro()
    { test_genorm_dev(Norm::Fro); }

//------------------------------------------------------------------------------
void test_trnorm_dev(Norm norm, Uplo uplo, Diag diag)
{
    // upper: m <= n, lower: m >= n
    int m = 20;
    int n = 30;
    if (uplo == Uplo::Lower)
        std::swap(m, n);

    int lda = m;
    double* Adata = new double[ lda * n ];
    slate::Tile<double> A(m, n, Adata, lda, -1);
    setup_data(A);
    A.uplo( uplo );

    cudaStream_t stream;
    test_assert(cudaStreamCreate(&stream) == cudaSuccess);

    double* dAdata;
    test_assert(cudaMalloc(&dAdata, sizeof(double) * lda * n) == cudaSuccess);
    test_assert(dAdata != nullptr);
    slate::Tile<double> dA(m, n, dAdata, lda, 0);
    A.copyDataToDevice(&dA, stream);
    dA.uplo( uplo );

    const int batch_count = 1;
    double* Aarray[batch_count];
    double** dAarray;
    test_assert(cudaMalloc(&dAarray, sizeof(double*) * batch_count) == cudaSuccess);
    test_assert(dAarray != nullptr);
    Aarray[0] = dA.data();
    test_assert(cudaMemcpy(dAarray, Aarray, sizeof(double*) * batch_count,
                           cudaMemcpyHostToDevice ) == cudaSuccess);

    std::vector<double> values;
    size_t ldv = 1;
    if (norm == lapack::Norm::Max)
        ldv = 1;
    else if (norm == lapack::Norm::One)
        ldv = n;
    else if (norm == lapack::Norm::Inf)
        ldv = m;
    else if (norm == lapack::Norm::Fro)
        ldv = 2;
    values.resize( ldv * batch_count );

    double* dvalues;
    test_assert(cudaMalloc(&dvalues, sizeof(double) * ldv * batch_count) == cudaSuccess);
    test_assert(dvalues != nullptr);

    slate::device::trnorm( norm, uplo, diag, m, n, dAarray, lda,
                           dvalues, ldv, batch_count, stream );
    cudaStreamSynchronize( stream );
    test_assert(cudaMemcpy( &values[0], dvalues, sizeof(double) * values.size(),
                            cudaMemcpyDeviceToHost ) == cudaSuccess );

    double result;
    if (norm == lapack::Norm::Fro) {
        assert( values.size() == 2 );
        result = values[0] * sqrt( values[1] );
    }
    else {
        result = 0;
        for (size_t i = 0; i < values.size(); ++i)
            result = std::max( result, values[i] );  // todo: max_nan
    }

    double result_ref = lapack::lantr(
        norm, uplo, diag, A.mb(), A.nb(), A.data(), A.stride() );

    double error = std::abs( result - result_ref ) / result_ref;
    if (norm == Norm::One)
        error /= sqrt( m );
    else if (norm == Norm::Inf)
        error /= sqrt( n );
    else if (norm == Norm::Fro)
        error /= sqrt( m*n );

    if (verbose) {
        printf( "\nnorm %c, uplo %c, diag %c, result %10.4f, ref %10.4f, error %.2e, ",
                norm2char(norm), uplo2char(uplo), diag2char(diag),
                result, result_ref, error );
    }

    cudaFree( dAdata );
    cudaFree( dAarray );
    cudaFree( dvalues );
    delete[] Adata;

    double eps = std::numeric_limits<double>::epsilon();
    test_assert( error < 3*eps );
}

//-----
void test_trnorm_dev_max_lower_unit()
    { test_trnorm_dev(Norm::Max, Uplo::Lower, Diag::Unit); }

void test_trnorm_dev_max_lower_nonunit()
    { test_trnorm_dev(Norm::Max, Uplo::Lower, Diag::NonUnit); }

void test_trnorm_dev_max_upper_unit()
    { test_trnorm_dev(Norm::Max, Uplo::Upper, Diag::Unit); }

void test_trnorm_dev_max_upper_nonunit()
    { test_trnorm_dev(Norm::Max, Uplo::Upper, Diag::NonUnit); }

//-----
void test_trnorm_dev_one_lower_unit()
    { test_trnorm_dev(Norm::One, Uplo::Lower, Diag::Unit); }

void test_trnorm_dev_one_lower_nonunit()
    { test_trnorm_dev(Norm::One, Uplo::Lower, Diag::NonUnit); }

void test_trnorm_dev_one_upper_unit()
    { test_trnorm_dev(Norm::One, Uplo::Upper, Diag::Unit); }

void test_trnorm_dev_one_upper_nonunit()
    { test_trnorm_dev(Norm::One, Uplo::Upper, Diag::NonUnit); }

//-----
void test_trnorm_dev_inf_lower_unit()
    { test_trnorm_dev(Norm::Inf, Uplo::Lower, Diag::Unit); }

void test_trnorm_dev_inf_lower_nonunit()
    { test_trnorm_dev(Norm::Inf, Uplo::Lower, Diag::NonUnit); }

void test_trnorm_dev_inf_upper_unit()
    { test_trnorm_dev(Norm::Inf, Uplo::Upper, Diag::Unit); }

void test_trnorm_dev_inf_upper_nonunit()
    { test_trnorm_dev(Norm::Inf, Uplo::Upper, Diag::NonUnit); }

//-----
void test_trnorm_dev_fro_lower_unit()
    { test_trnorm_dev(Norm::Fro, Uplo::Lower, Diag::Unit); }

void test_trnorm_dev_fro_lower_nonunit()
    { test_trnorm_dev(Norm::Fro, Uplo::Lower, Diag::NonUnit); }

void test_trnorm_dev_fro_upper_unit()
    { test_trnorm_dev(Norm::Fro, Uplo::Upper, Diag::Unit); }

void test_trnorm_dev_fro_upper_nonunit()
    { test_trnorm_dev(Norm::Fro, Uplo::Upper, Diag::NonUnit); }

//------------------------------------------------------------------------------
/// Runs all tests. Called by unit test main().
void run_tests()
{
    if (mpi_rank == 0) {
        //-------------------- genorm
        run_test(
            test_genorm_max, "genorm( max )");
        run_test(
            test_genorm_one, "genorm( one )");
        run_test(
            test_genorm_inf, "genorm( inf )");
        run_test(
            test_genorm_fro, "genorm( fro )");

        //-------------------- genorm_dev
        run_test(
            test_genorm_dev_max, "genorm_dev( max )");
        run_test(
            test_genorm_dev_one, "genorm_dev( one )");
        run_test(
            test_genorm_dev_inf, "genorm_dev( inf )");
        run_test(
            test_genorm_dev_fro, "genorm_dev( fro )");

        //-------------------- trnorm
        run_test(
            test_trnorm_max_lower_unit,    "trnorm( max, lower, unit    )");
        run_test(
            test_trnorm_max_lower_nonunit, "trnorm( max, lower, nonunit )");
        run_test(
            test_trnorm_max_upper_unit,    "trnorm( max, upper, unit    )");
        run_test(
            test_trnorm_max_upper_nonunit, "trnorm( max, upper, nonunit )");

        //-----
        run_test(
            test_trnorm_one_lower_unit,    "trnorm( one, lower, unit    )");
        run_test(
            test_trnorm_one_lower_nonunit, "trnorm( one, lower, nonunit )");
        run_test(
            test_trnorm_one_upper_unit,    "trnorm( one, upper, unit    )");
        run_test(
            test_trnorm_one_upper_nonunit, "trnorm( one, upper, nonunit )");

        //-----
        run_test(
            test_trnorm_inf_lower_unit,    "trnorm( inf, lower, unit    )");
        run_test(
            test_trnorm_inf_lower_nonunit, "trnorm( inf, lower, nonunit )");
        run_test(
            test_trnorm_inf_upper_unit,    "trnorm( inf, upper, unit    )");
        run_test(
            test_trnorm_inf_upper_nonunit, "trnorm( inf, upper, nonunit )");

        //-----
        run_test(
            test_trnorm_fro_lower_unit,    "trnorm( fro, lower, unit    )");
        run_test(
            test_trnorm_fro_lower_nonunit, "trnorm( fro, lower, nonunit )");
        run_test(
            test_trnorm_fro_upper_unit,    "trnorm( fro, upper, unit    )");
        run_test(
            test_trnorm_fro_upper_nonunit, "trnorm( fro, upper, nonunit )");

        //-------------------- trnorm_dev
        run_test(
            test_trnorm_dev_max_lower_unit,    "trnorm_dev( max, lower, unit    )");
        run_test(
            test_trnorm_dev_max_lower_nonunit, "trnorm_dev( max, lower, nonunit )");
        run_test(
            test_trnorm_dev_max_upper_unit,    "trnorm_dev( max, upper, unit    )");
        run_test(
            test_trnorm_dev_max_upper_nonunit, "trnorm_dev( max, upper, nonunit )");

        //-----
        run_test(
            test_trnorm_dev_one_lower_unit,    "trnorm_dev( one, lower, unit    )");
        run_test(
            test_trnorm_dev_one_lower_nonunit, "trnorm_dev( one, lower, nonunit )");
        run_test(
            test_trnorm_dev_one_upper_unit,    "trnorm_dev( one, upper, unit    )");
        run_test(
            test_trnorm_dev_one_upper_nonunit, "trnorm_dev( one, upper, nonunit )");

        //-----
        run_test(
            test_trnorm_dev_inf_lower_unit,    "trnorm_dev( inf, lower, unit    )");
        run_test(
            test_trnorm_dev_inf_lower_nonunit, "trnorm_dev( inf, lower, nonunit )");
        run_test(
            test_trnorm_dev_inf_upper_unit,    "trnorm_dev( inf, upper, unit    )");
        run_test(
            test_trnorm_dev_inf_upper_nonunit, "trnorm_dev( inf, upper, nonunit )");

        //-----
        run_test(
            test_trnorm_dev_fro_lower_unit,    "trnorm_dev( fro, lower, unit    )");
        run_test(
            test_trnorm_dev_fro_lower_nonunit, "trnorm_dev( fro, lower, nonunit )");
        run_test(
            test_trnorm_dev_fro_upper_unit,    "trnorm_dev( fro, upper, unit    )");
        run_test(
            test_trnorm_dev_fro_upper_nonunit, "trnorm_dev( fro, upper, nonunit )");
    }
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    verbose = 0;
    for (int i = 1; i < argc; ++i)
        if (argv[i] == std::string("-v"))
            verbose += 1;

    int err = unit_test_main(MPI_COMM_WORLD);  // which calls run_tests()

    MPI_Finalize();
    return err;
}
