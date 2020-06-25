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

#include "slate/c_api/wrappers.h"


// gbmm
void slate_Band_multiply_r32(
    float alpha, slate_BandMatrix_r32 A,
                               slate_Matrix_r32 B,
    float beta,      slate_Matrix_r32 C,
    int num_opts, slate_Options opts[])
{
  using scalar_t   = float;
  using matrix_A_t = slate::BandMatrix<scalar_t>;
  using matrix_B_t = slate::    Matrix<scalar_t>;
  using matrix_C_t = slate::    Matrix<scalar_t>;

  auto* A_ = reinterpret_cast<matrix_A_t*>(A);
  auto* B_ = reinterpret_cast<matrix_B_t*>(B);
  auto* C_ = reinterpret_cast<matrix_C_t*>(C);

  slate::Options opts_;
  slate::options2cpp(num_opts, opts, opts_);

  slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// gbmm
void slate_Band_multiply_r64(
    double alpha, slate_BandMatrix_r64 A,
                               slate_Matrix_r64 B,
    double beta,      slate_Matrix_r64 C,
    int num_opts, slate_Options opts[])
{
  using scalar_t   = double;
  using matrix_A_t = slate::BandMatrix<scalar_t>;
  using matrix_B_t = slate::    Matrix<scalar_t>;
  using matrix_C_t = slate::    Matrix<scalar_t>;

  auto* A_ = reinterpret_cast<matrix_A_t*>(A);
  auto* B_ = reinterpret_cast<matrix_B_t*>(B);
  auto* C_ = reinterpret_cast<matrix_C_t*>(C);

  slate::Options opts_;
  slate::options2cpp(num_opts, opts, opts_);

  slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// gbmm
void slate_Band_multiply_c32(
    float _Complex alpha, slate_BandMatrix_c32 A,
                               slate_Matrix_c32 B,
    float _Complex beta,      slate_Matrix_c32 C,
    int num_opts, slate_Options opts[])
{
  using scalar_t   = std::complex<float>;
  using matrix_A_t = slate::BandMatrix<scalar_t>;
  using matrix_B_t = slate::    Matrix<scalar_t>;
  using matrix_C_t = slate::    Matrix<scalar_t>;

  auto* A_ = reinterpret_cast<matrix_A_t*>(A);
  auto* B_ = reinterpret_cast<matrix_B_t*>(B);
  auto* C_ = reinterpret_cast<matrix_C_t*>(C);

  slate::Options opts_;
  slate::options2cpp(num_opts, opts, opts_);

  slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// gemm
void slate_multiply_r32(
    float alpha, slate_Matrix_r32 A,
                           slate_Matrix_r32 B,
    float beta,  slate_Matrix_r32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::Matrix<scalar_t>;
    using matrix_B_t = slate::Matrix<scalar_t>;
    using matrix_C_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// gemm
void slate_multiply_r64(
    double alpha, slate_Matrix_r64 A,
                           slate_Matrix_r64 B,
    double beta,  slate_Matrix_r64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::Matrix<scalar_t>;
    using matrix_B_t = slate::Matrix<scalar_t>;
    using matrix_C_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// gemm
void slate_multiply_c32(
    float _Complex alpha, slate_Matrix_c32 A,
                           slate_Matrix_c32 B,
    float _Complex beta,  slate_Matrix_c32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::Matrix<scalar_t>;
    using matrix_B_t = slate::Matrix<scalar_t>;
    using matrix_C_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// Left hbmm
void slate_HermitianBand_left_multiply_r32(
    float alpha, slate_HermitianBandMatrix_r32 A,
                                        slate_Matrix_r32 B,
    float beta,               slate_Matrix_r32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::HermitianBandMatrix<scalar_t>;
    using matrix_B_t = slate::             Matrix<scalar_t>;
    using matrix_C_t = slate::             Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// Left hbmm
void slate_HermitianBand_left_multiply_r64(
    double alpha, slate_HermitianBandMatrix_r64 A,
                                        slate_Matrix_r64 B,
    double beta,               slate_Matrix_r64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::HermitianBandMatrix<scalar_t>;
    using matrix_B_t = slate::             Matrix<scalar_t>;
    using matrix_C_t = slate::             Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// Left hbmm
void slate_HermitianBand_left_multiply_c32(
    float _Complex alpha, slate_HermitianBandMatrix_c32 A,
                                        slate_Matrix_c32 B,
    float _Complex beta,               slate_Matrix_c32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::HermitianBandMatrix<scalar_t>;
    using matrix_B_t = slate::             Matrix<scalar_t>;
    using matrix_C_t = slate::             Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// Right hbmm
void slate_HermitianBand_right_multiply_r32(
    float alpha,              slate_Matrix_r32 A,
                           slate_HermitianBandMatrix_r32 B,
    float beta,               slate_Matrix_r32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::             Matrix<scalar_t>;
    using matrix_B_t = slate::HermitianBandMatrix<scalar_t>;
    using matrix_C_t = slate::             Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// Right hbmm
void slate_HermitianBand_right_multiply_r64(
    double alpha,              slate_Matrix_r64 A,
                           slate_HermitianBandMatrix_r64 B,
    double beta,               slate_Matrix_r64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::             Matrix<scalar_t>;
    using matrix_B_t = slate::HermitianBandMatrix<scalar_t>;
    using matrix_C_t = slate::             Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// Right hbmm
void slate_HermitianBand_right_multiply_c32(
    float _Complex alpha,              slate_Matrix_c32 A,
                           slate_HermitianBandMatrix_c32 B,
    float _Complex beta,               slate_Matrix_c32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::             Matrix<scalar_t>;
    using matrix_B_t = slate::HermitianBandMatrix<scalar_t>;
    using matrix_C_t = slate::             Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// Left hemm
void slate_Hermitian_left_multiply_r32(
    float alpha, slate_HermitianMatrix_r32 A,
                                    slate_Matrix_r32 B,
    float beta,           slate_Matrix_r32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;
    using matrix_C_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// Left hemm
void slate_Hermitian_left_multiply_r64(
    double alpha, slate_HermitianMatrix_r64 A,
                                    slate_Matrix_r64 B,
    double beta,           slate_Matrix_r64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;
    using matrix_C_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// Left hemm
void slate_Hermitian_left_multiply_c32(
    float _Complex alpha, slate_HermitianMatrix_c32 A,
                                    slate_Matrix_c32 B,
    float _Complex beta,           slate_Matrix_c32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;
    using matrix_C_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// Right hemm
void slate_Hermitian_right_multiply_r32(
    float alpha,          slate_Matrix_r32 A,
                           slate_HermitianMatrix_r32 B,
    float beta,           slate_Matrix_r32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::         Matrix<scalar_t>;
    using matrix_B_t = slate::HermitianMatrix<scalar_t>;
    using matrix_C_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// Right hemm
void slate_Hermitian_right_multiply_r64(
    double alpha,          slate_Matrix_r64 A,
                           slate_HermitianMatrix_r64 B,
    double beta,           slate_Matrix_r64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::         Matrix<scalar_t>;
    using matrix_B_t = slate::HermitianMatrix<scalar_t>;
    using matrix_C_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// Right hemm
void slate_Hermitian_right_multiply_c32(
    float _Complex alpha,          slate_Matrix_c32 A,
                           slate_HermitianMatrix_c32 B,
    float _Complex beta,           slate_Matrix_c32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::         Matrix<scalar_t>;
    using matrix_B_t = slate::HermitianMatrix<scalar_t>;
    using matrix_C_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// Left symm
void slate_Symmetric_left_multiply_r32(
    float alpha, slate_SymmetricMatrix_r32 A,
                                    slate_Matrix_r32 B,
    float beta,           slate_Matrix_r32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::SymmetricMatrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;
    using matrix_C_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// Left symm
void slate_Symmetric_left_multiply_r64(
    double alpha, slate_SymmetricMatrix_r64 A,
                                    slate_Matrix_r64 B,
    double beta,           slate_Matrix_r64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::SymmetricMatrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;
    using matrix_C_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// Left symm
void slate_Symmetric_left_multiply_c32(
    float _Complex alpha, slate_SymmetricMatrix_c32 A,
                                    slate_Matrix_c32 B,
    float _Complex beta,           slate_Matrix_c32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::SymmetricMatrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;
    using matrix_C_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// Right symm
void slate_Symmetric_right_multiply_r32(
    float alpha,          slate_Matrix_r32 A,
                           slate_SymmetricMatrix_r32 B,
    float beta,           slate_Matrix_r32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::         Matrix<scalar_t>;
    using matrix_B_t = slate::SymmetricMatrix<scalar_t>;
    using matrix_C_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// Right symm
void slate_Symmetric_right_multiply_r64(
    double alpha,          slate_Matrix_r64 A,
                           slate_SymmetricMatrix_r64 B,
    double beta,           slate_Matrix_r64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::         Matrix<scalar_t>;
    using matrix_B_t = slate::SymmetricMatrix<scalar_t>;
    using matrix_C_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// Right symm
void slate_Symmetric_right_multiply_c32(
    float _Complex alpha,          slate_Matrix_c32 A,
                           slate_SymmetricMatrix_c32 B,
    float _Complex beta,           slate_Matrix_c32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::         Matrix<scalar_t>;
    using matrix_B_t = slate::SymmetricMatrix<scalar_t>;
    using matrix_C_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::multiply<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// Left trmm
void slate_triangular_left_multiply_r32(
    float alpha, slate_TriangularMatrix_r32 A,
                                     slate_Matrix_r32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::TriangularMatrix<scalar_t>;
    using matrix_B_t = slate::          Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_multiply<scalar_t>(alpha, *A_, *B_, opts_);
}


// Left trmm
void slate_triangular_left_multiply_r64(
    double alpha, slate_TriangularMatrix_r64 A,
                                     slate_Matrix_r64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::TriangularMatrix<scalar_t>;
    using matrix_B_t = slate::          Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_multiply<scalar_t>(alpha, *A_, *B_, opts_);
}


// Left trmm
void slate_triangular_left_multiply_c32(
    float _Complex alpha, slate_TriangularMatrix_c32 A,
                                     slate_Matrix_c32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::TriangularMatrix<scalar_t>;
    using matrix_B_t = slate::          Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_multiply<scalar_t>(alpha, *A_, *B_, opts_);
}


// Right trmm
void slate_triangular_right_multiply_r32(
    float alpha,           slate_Matrix_r32 A,
                           slate_TriangularMatrix_r32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::          Matrix<scalar_t>;
    using matrix_B_t = slate::TriangularMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_multiply<scalar_t>(alpha, *A_, *B_, opts_);
}


// Right trmm
void slate_triangular_right_multiply_r64(
    double alpha,           slate_Matrix_r64 A,
                           slate_TriangularMatrix_r64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::          Matrix<scalar_t>;
    using matrix_B_t = slate::TriangularMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_multiply<scalar_t>(alpha, *A_, *B_, opts_);
}


// Right trmm
void slate_triangular_right_multiply_c32(
    float _Complex alpha,           slate_Matrix_c32 A,
                           slate_TriangularMatrix_c32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::          Matrix<scalar_t>;
    using matrix_B_t = slate::TriangularMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_multiply<scalar_t>(alpha, *A_, *B_, opts_);
}


// Left tbsm
void slate_Band_triangular_left_solve_r32(
    float alpha, slate_TriangularBandMatrix_r32 A,
                                         slate_Matrix_r32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::TriangularBandMatrix<scalar_t>;
    using matrix_B_t = slate::              Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_solve<scalar_t>(alpha, *A_, *B_, opts_);
}


// Left tbsm
void slate_Band_triangular_left_solve_r64(
    double alpha, slate_TriangularBandMatrix_r64 A,
                                         slate_Matrix_r64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::TriangularBandMatrix<scalar_t>;
    using matrix_B_t = slate::              Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_solve<scalar_t>(alpha, *A_, *B_, opts_);
}


// Left tbsm
void slate_Band_triangular_left_solve_c32(
    float _Complex alpha, slate_TriangularBandMatrix_c32 A,
                                         slate_Matrix_c32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::TriangularBandMatrix<scalar_t>;
    using matrix_B_t = slate::              Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_solve<scalar_t>(alpha, *A_, *B_, opts_);
}


// Right tbsm
void slate_Band_triangular_right_solve_r32(
    float alpha,               slate_Matrix_r32 A,
                           slate_TriangularBandMatrix_r32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::              Matrix<scalar_t>;
    using matrix_B_t = slate::TriangularBandMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_solve<scalar_t>(alpha, *A_, *B_, opts_);
}


// Right tbsm
void slate_Band_triangular_right_solve_r64(
    double alpha,               slate_Matrix_r64 A,
                           slate_TriangularBandMatrix_r64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::              Matrix<scalar_t>;
    using matrix_B_t = slate::TriangularBandMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_solve<scalar_t>(alpha, *A_, *B_, opts_);
}


// Right tbsm
void slate_Band_triangular_right_solve_c32(
    float _Complex alpha,               slate_Matrix_c32 A,
                           slate_TriangularBandMatrix_c32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::              Matrix<scalar_t>;
    using matrix_B_t = slate::TriangularBandMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_solve<scalar_t>(alpha, *A_, *B_, opts_);
}


// Left trsm
void slate_triangular_left_solve_r32(
    float alpha, slate_TriangularMatrix_r32 A,
                                     slate_Matrix_r32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::TriangularMatrix<scalar_t>;
    using matrix_B_t = slate::              Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_solve<scalar_t>(alpha, *A_, *B_, opts_);
}


// Left trsm
void slate_triangular_left_solve_r64(
    double alpha, slate_TriangularMatrix_r64 A,
                                     slate_Matrix_r64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::TriangularMatrix<scalar_t>;
    using matrix_B_t = slate::              Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_solve<scalar_t>(alpha, *A_, *B_, opts_);
}


// Left trsm
void slate_triangular_left_solve_c32(
    float _Complex alpha, slate_TriangularMatrix_c32 A,
                                     slate_Matrix_c32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::TriangularMatrix<scalar_t>;
    using matrix_B_t = slate::              Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_solve<scalar_t>(alpha, *A_, *B_, opts_);
}


// Right trsm
void slate_triangular_right_solve_r32(
    float alpha,           slate_Matrix_r32 A,
                           slate_TriangularMatrix_r32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::          Matrix<scalar_t>;
    using matrix_B_t = slate::TriangularMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_solve<scalar_t>(alpha, *A_, *B_, opts_);
}


// Right trsm
void slate_triangular_right_solve_r64(
    double alpha,           slate_Matrix_r64 A,
                           slate_TriangularMatrix_r64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::          Matrix<scalar_t>;
    using matrix_B_t = slate::TriangularMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_solve<scalar_t>(alpha, *A_, *B_, opts_);
}


// Right trsm
void slate_triangular_right_solve_c32(
    float _Complex alpha,           slate_Matrix_c32 A,
                           slate_TriangularMatrix_c32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::          Matrix<scalar_t>;
    using matrix_B_t = slate::TriangularMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_solve<scalar_t>(alpha, *A_, *B_, opts_);
}


// herk
void slate_Hermitian_rank_k_update_r32(
    float alpha,          slate_Matrix_r32 A,
    float beta,  slate_HermitianMatrix_r32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::         Matrix<scalar_t>;
    using matrix_C_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::rank_k_update<scalar_t>(alpha, *A_, beta, *C_, opts_);
}


// herk
void slate_Hermitian_rank_k_update_r64(
    double alpha,          slate_Matrix_r64 A,
    double beta,  slate_HermitianMatrix_r64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::         Matrix<scalar_t>;
    using matrix_C_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::rank_k_update<scalar_t>(alpha, *A_, beta, *C_, opts_);
}


// herk
void slate_Hermitian_rank_k_update_c32(
    float alpha,          slate_Matrix_c32 A,
    float beta,  slate_HermitianMatrix_c32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::         Matrix<scalar_t>;
    using matrix_C_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::rank_k_update<scalar_t>(alpha, *A_, beta, *C_, opts_);
}


// syrk
void slate_Symmetric_rank_k_update_r32(
    float alpha,           slate_Matrix_r32 A,
    float beta,   slate_SymmetricMatrix_r32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::         Matrix<scalar_t>;
    using matrix_C_t = slate::SymmetricMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::rank_k_update<scalar_t>(alpha, *A_, beta, *C_, opts_);
}


// syrk
void slate_Symmetric_rank_k_update_r64(
    double alpha,           slate_Matrix_r64 A,
    double beta,   slate_SymmetricMatrix_r64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::         Matrix<scalar_t>;
    using matrix_C_t = slate::SymmetricMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::rank_k_update<scalar_t>(alpha, *A_, beta, *C_, opts_);
}


// syrk
void slate_Symmetric_rank_k_update_c32(
    float _Complex alpha,           slate_Matrix_c32 A,
    float _Complex beta,   slate_SymmetricMatrix_c32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::         Matrix<scalar_t>;
    using matrix_C_t = slate::SymmetricMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::rank_k_update<scalar_t>(alpha, *A_, beta, *C_, opts_);
}


// herk
void slate_Hermitian_rank_2k_update_r32(
    float alpha,  slate_Matrix_r32 A,
                            slate_Matrix_r32 B,
    float beta,   slate_HermitianMatrix_r32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::         Matrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;
    using matrix_C_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::rank_2k_update<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// herk
void slate_Hermitian_rank_2k_update_r64(
    double alpha,  slate_Matrix_r64 A,
                            slate_Matrix_r64 B,
    double beta,   slate_HermitianMatrix_r64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::         Matrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;
    using matrix_C_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::rank_2k_update<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// herk
void slate_Hermitian_rank_2k_update_c32(
    float _Complex alpha,  slate_Matrix_c32 A,
                            slate_Matrix_c32 B,
    float beta,   slate_HermitianMatrix_c32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::         Matrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;
    using matrix_C_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::rank_2k_update<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// syrk
void slate_Symmetric_rank_2k_update_r32(
    float alpha,            slate_Matrix_r32 A,
                                      slate_Matrix_r32 B,
    float beta,    slate_SymmetricMatrix_r32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::         Matrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;
    using matrix_C_t = slate::SymmetricMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::rank_2k_update<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// syrk
void slate_Symmetric_rank_2k_update_r64(
    double alpha,            slate_Matrix_r64 A,
                                      slate_Matrix_r64 B,
    double beta,    slate_SymmetricMatrix_r64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::         Matrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;
    using matrix_C_t = slate::SymmetricMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::rank_2k_update<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// syrk
void slate_Symmetric_rank_2k_update_c32(
    float _Complex alpha,            slate_Matrix_c32 A,
                                      slate_Matrix_c32 B,
    float _Complex beta,    slate_SymmetricMatrix_c32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::         Matrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;
    using matrix_C_t = slate::SymmetricMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::rank_2k_update<scalar_t>(alpha, *A_, *B_, beta, *C_, opts_);
}


// gbsv
void slate_Band_lu_solve_r32(
    slate_BandMatrix_r32 A,
        slate_Matrix_r32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::BandMatrix<scalar_t>;
    using matrix_B_t = slate::    Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve<scalar_t>(*A_, *B_, opts_);
}


// gbsv
void slate_Band_lu_solve_r64(
    slate_BandMatrix_r64 A,
        slate_Matrix_r64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::BandMatrix<scalar_t>;
    using matrix_B_t = slate::    Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve<scalar_t>(*A_, *B_, opts_);
}


// gbsv
void slate_Band_lu_solve_c32(
    slate_BandMatrix_c32 A,
        slate_Matrix_c32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::BandMatrix<scalar_t>;
    using matrix_B_t = slate::    Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve<scalar_t>(*A_, *B_, opts_);
}


// gesv
void slate_lu_solve_r32(
    slate_Matrix_r32 A,
    slate_Matrix_r32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::Matrix<scalar_t>;
    using matrix_B_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve<scalar_t>(*A_, *B_, opts_);
}


// gesv
void slate_lu_solve_r64(
    slate_Matrix_r64 A,
    slate_Matrix_r64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::Matrix<scalar_t>;
    using matrix_B_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve<scalar_t>(*A_, *B_, opts_);
}


// gesv
void slate_lu_solve_c32(
    slate_Matrix_c32 A,
    slate_Matrix_c32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::Matrix<scalar_t>;
    using matrix_B_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve<scalar_t>(*A_, *B_, opts_);
}


// gesv_nopiv
void slate_lu_solve_nopiv_r32(
    slate_Matrix_r32 A,
    slate_Matrix_r32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::Matrix<scalar_t>;
    using matrix_B_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve_nopiv<scalar_t>(*A_, *B_, opts_);
}


// gesv_nopiv
void slate_lu_solve_nopiv_r64(
    slate_Matrix_r64 A,
    slate_Matrix_r64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::Matrix<scalar_t>;
    using matrix_B_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve_nopiv<scalar_t>(*A_, *B_, opts_);
}


// gesv_nopiv
void slate_lu_solve_nopiv_c32(
    slate_Matrix_c32 A,
    slate_Matrix_c32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::Matrix<scalar_t>;
    using matrix_B_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve_nopiv<scalar_t>(*A_, *B_, opts_);
}


// gbtrf
void slate_Band_lu_factor_r32(
    slate_BandMatrix_r32 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::BandMatrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* pivots_= reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_factor<scalar_t>(*A_, *pivots_, opts_);
}


// gbtrf
void slate_Band_lu_factor_r64(
    slate_BandMatrix_r64 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::BandMatrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* pivots_= reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_factor<scalar_t>(*A_, *pivots_, opts_);
}


// gbtrf
void slate_Band_lu_factor_c32(
    slate_BandMatrix_c32 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::BandMatrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* pivots_= reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_factor<scalar_t>(*A_, *pivots_, opts_);
}


// getrf
void slate_lu_factor_r32(
    slate_Matrix_r32 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::Matrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* pivots_= reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_factor<scalar_t>(*A_, *pivots_, opts_);
}


// getrf
void slate_lu_factor_r64(
    slate_Matrix_r64 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::Matrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* pivots_= reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_factor<scalar_t>(*A_, *pivots_, opts_);
}


// getrf
void slate_lu_factor_c32(
    slate_Matrix_c32 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::Matrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* pivots_= reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_factor<scalar_t>(*A_, *pivots_, opts_);
}


// getrf_nopiv
void slate_lu_factor_nopiv_r32(
    slate_Matrix_r32 A,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_factor_nopiv<scalar_t>(*A_, opts_);
}


// getrf_nopiv
void slate_lu_factor_nopiv_r64(
    slate_Matrix_r64 A,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_factor_nopiv<scalar_t>(*A_, opts_);
}


// getrf_nopiv
void slate_lu_factor_nopiv_c32(
    slate_Matrix_c32 A,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_factor_nopiv<scalar_t>(*A_, opts_);
}


// gbtrs
void slate_Band_lu_solve_using_factor_r32(
    slate_BandMatrix_r32 A, slate_Pivots pivots,
        slate_Matrix_r32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::BandMatrix<scalar_t>;
    using matrix_B_t = slate::    Matrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* B_     = reinterpret_cast<matrix_B_t*>(B);
    auto* pivots_= reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve_using_factor<scalar_t>(*A_, *pivots_, *B_, opts_);
}


// gbtrs
void slate_Band_lu_solve_using_factor_r64(
    slate_BandMatrix_r64 A, slate_Pivots pivots,
        slate_Matrix_r64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::BandMatrix<scalar_t>;
    using matrix_B_t = slate::    Matrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* B_     = reinterpret_cast<matrix_B_t*>(B);
    auto* pivots_= reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve_using_factor<scalar_t>(*A_, *pivots_, *B_, opts_);
}


// gbtrs
void slate_Band_lu_solve_using_factor_c32(
    slate_BandMatrix_c32 A, slate_Pivots pivots,
        slate_Matrix_c32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::BandMatrix<scalar_t>;
    using matrix_B_t = slate::    Matrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* B_     = reinterpret_cast<matrix_B_t*>(B);
    auto* pivots_= reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve_using_factor<scalar_t>(*A_, *pivots_, *B_, opts_);
}


// getrs
void slate_lu_solve_using_factor_r32(
    slate_Matrix_r32 A, slate_Pivots pivots,
    slate_Matrix_r32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::Matrix<scalar_t>;
    using matrix_B_t = slate::Matrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* B_     = reinterpret_cast<matrix_B_t*>(B);
    auto* pivots_= reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve_using_factor<scalar_t>(*A_, *pivots_, *B_, opts_);
}


// getrs
void slate_lu_solve_using_factor_r64(
    slate_Matrix_r64 A, slate_Pivots pivots,
    slate_Matrix_r64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::Matrix<scalar_t>;
    using matrix_B_t = slate::Matrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* B_     = reinterpret_cast<matrix_B_t*>(B);
    auto* pivots_= reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve_using_factor<scalar_t>(*A_, *pivots_, *B_, opts_);
}


// getrs
void slate_lu_solve_using_factor_c32(
    slate_Matrix_c32 A, slate_Pivots pivots,
    slate_Matrix_c32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::Matrix<scalar_t>;
    using matrix_B_t = slate::Matrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* B_     = reinterpret_cast<matrix_B_t*>(B);
    auto* pivots_= reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve_using_factor<scalar_t>(*A_, *pivots_, *B_, opts_);
}


// getrs_nopiv
void slate_lu_solve_using_factor_nopiv_r32(
    slate_Matrix_r32 A,
    slate_Matrix_r32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::Matrix<scalar_t>;
    using matrix_B_t = slate::Matrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* B_     = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve_using_factor_nopiv<scalar_t>(*A_, *B_, opts_);
}


// getrs_nopiv
void slate_lu_solve_using_factor_nopiv_r64(
    slate_Matrix_r64 A,
    slate_Matrix_r64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::Matrix<scalar_t>;
    using matrix_B_t = slate::Matrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* B_     = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve_using_factor_nopiv<scalar_t>(*A_, *B_, opts_);
}


// getrs_nopiv
void slate_lu_solve_using_factor_nopiv_c32(
    slate_Matrix_c32 A,
    slate_Matrix_c32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::Matrix<scalar_t>;
    using matrix_B_t = slate::Matrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* B_     = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve_using_factor_nopiv<scalar_t>(*A_, *B_, opts_);
}


// In-place getri
void slate_lu_inverse_using_factor_r32(
    slate_Matrix_r32 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::Matrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* pivots_= reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_inverse_using_factor<scalar_t>(*A_, *pivots_, opts_);
}


// In-place getri
void slate_lu_inverse_using_factor_r64(
    slate_Matrix_r64 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::Matrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* pivots_= reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_inverse_using_factor<scalar_t>(*A_, *pivots_, opts_);
}


// In-place getri
void slate_lu_inverse_using_factor_c32(
    slate_Matrix_c32 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::Matrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* pivots_= reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_inverse_using_factor<scalar_t>(*A_, *pivots_, opts_);
}


// Out-of-place getri
void slate_lu_inverse_using_factor_out_of_place_r32(
    slate_Matrix_r32 A, slate_Pivots pivots,
    slate_Matrix_r32 A_inverse,
    int num_opts, slate_Options opts[])
{
    using scalar_t           = float;
    using matrix_A_t         = slate::Matrix<scalar_t>;
    using matrix_A_inverse_t = slate::Matrix<scalar_t>;

    auto* A_         = reinterpret_cast<matrix_A_t*>(A);
    auto* A_inverse_ = reinterpret_cast<matrix_A_inverse_t*>(A_inverse);
    auto* pivots_    = reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_inverse_using_factor_out_of_place<scalar_t>(
        *A_, *pivots_, *A_inverse_, opts_);
}


// Out-of-place getri
void slate_lu_inverse_using_factor_out_of_place_r64(
    slate_Matrix_r64 A, slate_Pivots pivots,
    slate_Matrix_r64 A_inverse,
    int num_opts, slate_Options opts[])
{
    using scalar_t           = double;
    using matrix_A_t         = slate::Matrix<scalar_t>;
    using matrix_A_inverse_t = slate::Matrix<scalar_t>;

    auto* A_         = reinterpret_cast<matrix_A_t*>(A);
    auto* A_inverse_ = reinterpret_cast<matrix_A_inverse_t*>(A_inverse);
    auto* pivots_    = reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_inverse_using_factor_out_of_place<scalar_t>(
        *A_, *pivots_, *A_inverse_, opts_);
}


// Out-of-place getri
void slate_lu_inverse_using_factor_out_of_place_c32(
    slate_Matrix_c32 A, slate_Pivots pivots,
    slate_Matrix_c32 A_inverse,
    int num_opts, slate_Options opts[])
{
    using scalar_t           = std::complex<float>;
    using matrix_A_t         = slate::Matrix<scalar_t>;
    using matrix_A_inverse_t = slate::Matrix<scalar_t>;

    auto* A_         = reinterpret_cast<matrix_A_t*>(A);
    auto* A_inverse_ = reinterpret_cast<matrix_A_inverse_t*>(A_inverse);
    auto* pivots_    = reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_inverse_using_factor_out_of_place<scalar_t>(
        *A_, *pivots_, *A_inverse_, opts_);
}


// pbsv
void slate_Band_chol_solve_r32(
    slate_HermitianBandMatrix_r32 A,
                 slate_Matrix_r32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::HermitianBandMatrix<scalar_t>;
    using matrix_B_t = slate::             Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_solve<scalar_t>(*A_, *B_, opts_);
}


// pbsv
void slate_Band_chol_solve_r64(
    slate_HermitianBandMatrix_r64 A,
                 slate_Matrix_r64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::HermitianBandMatrix<scalar_t>;
    using matrix_B_t = slate::             Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_solve<scalar_t>(*A_, *B_, opts_);
}


// pbsv
void slate_Band_chol_solve_c32(
    slate_HermitianBandMatrix_c32 A,
                 slate_Matrix_c32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::HermitianBandMatrix<scalar_t>;
    using matrix_B_t = slate::             Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_solve<scalar_t>(*A_, *B_, opts_);
}


// posv
void slate_chol_solve_r32(
    slate_HermitianMatrix_r32 A,
             slate_Matrix_r32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_solve<scalar_t>(*A_, *B_, opts_);
}


// posv
void slate_chol_solve_r64(
    slate_HermitianMatrix_r64 A,
             slate_Matrix_r64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_solve<scalar_t>(*A_, *B_, opts_);
}


// posv
void slate_chol_solve_c32(
    slate_HermitianMatrix_c32 A,
             slate_Matrix_c32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_solve<scalar_t>(*A_, *B_, opts_);
}


// pbtrf
void slate_Band_chol_factor_r32(
    slate_HermitianBandMatrix_r32 A,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::HermitianBandMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_factor<scalar_t>(*A_, opts_);
}


// pbtrf
void slate_Band_chol_factor_r64(
    slate_HermitianBandMatrix_r64 A,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::HermitianBandMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_factor<scalar_t>(*A_, opts_);
}


// pbtrf
void slate_Band_chol_factor_c32(
    slate_HermitianBandMatrix_c32 A,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::HermitianBandMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_factor<scalar_t>(*A_, opts_);
}


// potrf
void slate_chol_factor_r32(
    slate_HermitianMatrix_r32 A,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_factor<scalar_t>(*A_, opts_);
}


// potrf
void slate_chol_factor_r64(
    slate_HermitianMatrix_r64 A,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_factor<scalar_t>(*A_, opts_);
}


// potrf
void slate_chol_factor_c32(
    slate_HermitianMatrix_c32 A,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_factor<scalar_t>(*A_, opts_);
}


// pbtrs
void slate_Band_chol_solve_using_factor_r32(
    slate_HermitianBandMatrix_r32 A,
                 slate_Matrix_r32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::HermitianBandMatrix<scalar_t>;
    using matrix_B_t = slate::             Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_solve_using_factor<scalar_t>(*A_, *B_, opts_);
}


// pbtrs
void slate_Band_chol_solve_using_factor_r64(
    slate_HermitianBandMatrix_r64 A,
                 slate_Matrix_r64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::HermitianBandMatrix<scalar_t>;
    using matrix_B_t = slate::             Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_solve_using_factor<scalar_t>(*A_, *B_, opts_);
}


// pbtrs
void slate_Band_chol_solve_using_factor_c32(
    slate_HermitianBandMatrix_c32 A,
                 slate_Matrix_c32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::HermitianBandMatrix<scalar_t>;
    using matrix_B_t = slate::             Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_solve_using_factor<scalar_t>(*A_, *B_, opts_);
}


// potrs
void slate_chol_solve_using_factor_r32(
    slate_HermitianMatrix_r32 A,
             slate_Matrix_r32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_solve_using_factor<scalar_t>(*A_, *B_, opts_);
}


// potrs
void slate_chol_solve_using_factor_r64(
    slate_HermitianMatrix_r64 A,
             slate_Matrix_r64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_solve_using_factor<scalar_t>(*A_, *B_, opts_);
}


// potrs
void slate_chol_solve_using_factor_c32(
    slate_HermitianMatrix_c32 A,
             slate_Matrix_c32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_solve_using_factor<scalar_t>(*A_, *B_, opts_);
}


// potri
void slate_chol_inverse_using_factor_r32(
    slate_HermitianMatrix_r32 A,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_inverse_using_factor<scalar_t>(*A_, opts_);
}


// potri
void slate_chol_inverse_using_factor_r64(
    slate_HermitianMatrix_r64 A,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_inverse_using_factor<scalar_t>(*A_, opts_);
}


// potri
void slate_chol_inverse_using_factor_c32(
    slate_HermitianMatrix_c32 A,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_inverse_using_factor<scalar_t>(*A_, opts_);
}


// hesv
void slate_indefinite_solve_r32(
    slate_HermitianMatrix_r32 A,
             slate_Matrix_r32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::indefinite_solve<scalar_t>(*A_, *B_, opts_);
}


// hesv
void slate_indefinite_solve_r64(
    slate_HermitianMatrix_r64 A,
             slate_Matrix_r64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::indefinite_solve<scalar_t>(*A_, *B_, opts_);
}


// hesv
void slate_indefinite_solve_c32(
    slate_HermitianMatrix_c32 A,
             slate_Matrix_c32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::indefinite_solve<scalar_t>(*A_, *B_, opts_);
}


// hetrf
void slate_indefinite_factor_r32(
    slate_HermitianMatrix_r32 A, slate_Pivots pivots,
         slate_BandMatrix_r32 T, slate_Pivots pivots2,
             slate_Matrix_r32 H,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_T_t = slate::     BandMatrix<scalar_t>;
    using matrix_H_t = slate::         Matrix<scalar_t>;

    auto* A_       = reinterpret_cast<matrix_A_t*>(A);
    auto* T_       = reinterpret_cast<matrix_T_t*>(T);
    auto* H_       = reinterpret_cast<matrix_H_t*>(H);
    auto* pivots_  = reinterpret_cast<slate::Pivots*>(pivots);
    auto* pivots2_ = reinterpret_cast<slate::Pivots*>(pivots2);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::indefinite_factor<scalar_t>(
        *A_, *pivots_, *T_, *pivots2_, *H_, opts_);
}


// hetrf
void slate_indefinite_factor_r64(
    slate_HermitianMatrix_r64 A, slate_Pivots pivots,
         slate_BandMatrix_r64 T, slate_Pivots pivots2,
             slate_Matrix_r64 H,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_T_t = slate::     BandMatrix<scalar_t>;
    using matrix_H_t = slate::         Matrix<scalar_t>;

    auto* A_       = reinterpret_cast<matrix_A_t*>(A);
    auto* T_       = reinterpret_cast<matrix_T_t*>(T);
    auto* H_       = reinterpret_cast<matrix_H_t*>(H);
    auto* pivots_  = reinterpret_cast<slate::Pivots*>(pivots);
    auto* pivots2_ = reinterpret_cast<slate::Pivots*>(pivots2);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::indefinite_factor<scalar_t>(
        *A_, *pivots_, *T_, *pivots2_, *H_, opts_);
}


// hetrf
void slate_indefinite_factor_c32(
    slate_HermitianMatrix_c32 A, slate_Pivots pivots,
         slate_BandMatrix_c32 T, slate_Pivots pivots2,
             slate_Matrix_c32 H,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_T_t = slate::     BandMatrix<scalar_t>;
    using matrix_H_t = slate::         Matrix<scalar_t>;

    auto* A_       = reinterpret_cast<matrix_A_t*>(A);
    auto* T_       = reinterpret_cast<matrix_T_t*>(T);
    auto* H_       = reinterpret_cast<matrix_H_t*>(H);
    auto* pivots_  = reinterpret_cast<slate::Pivots*>(pivots);
    auto* pivots2_ = reinterpret_cast<slate::Pivots*>(pivots2);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::indefinite_factor<scalar_t>(
        *A_, *pivots_, *T_, *pivots2_, *H_, opts_);
}


// hetrs
void slate_indefinite_solve_using_factor_r32(
    slate_HermitianMatrix_r32 A, slate_Pivots pivots,
         slate_BandMatrix_r32 T, slate_Pivots pivots2,
             slate_Matrix_r32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_T_t = slate::     BandMatrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;

    auto* A_       = reinterpret_cast<matrix_A_t*>(A);
    auto* T_       = reinterpret_cast<matrix_T_t*>(T);
    auto* B_       = reinterpret_cast<matrix_B_t*>(B);
    auto* pivots_  = reinterpret_cast<slate::Pivots*>(pivots);
    auto* pivots2_ = reinterpret_cast<slate::Pivots*>(pivots2);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::indefinite_solve_using_factor<scalar_t>(
        *A_, *pivots_, *T_, *pivots2_, *B_, opts_);
}


// hetrs
void slate_indefinite_solve_using_factor_r64(
    slate_HermitianMatrix_r64 A, slate_Pivots pivots,
         slate_BandMatrix_r64 T, slate_Pivots pivots2,
             slate_Matrix_r64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_T_t = slate::     BandMatrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;

    auto* A_       = reinterpret_cast<matrix_A_t*>(A);
    auto* T_       = reinterpret_cast<matrix_T_t*>(T);
    auto* B_       = reinterpret_cast<matrix_B_t*>(B);
    auto* pivots_  = reinterpret_cast<slate::Pivots*>(pivots);
    auto* pivots2_ = reinterpret_cast<slate::Pivots*>(pivots2);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::indefinite_solve_using_factor<scalar_t>(
        *A_, *pivots_, *T_, *pivots2_, *B_, opts_);
}


// hetrs
void slate_indefinite_solve_using_factor_c32(
    slate_HermitianMatrix_c32 A, slate_Pivots pivots,
         slate_BandMatrix_c32 T, slate_Pivots pivots2,
             slate_Matrix_c32 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_T_t = slate::     BandMatrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;

    auto* A_       = reinterpret_cast<matrix_A_t*>(A);
    auto* T_       = reinterpret_cast<matrix_T_t*>(T);
    auto* B_       = reinterpret_cast<matrix_B_t*>(B);
    auto* pivots_  = reinterpret_cast<slate::Pivots*>(pivots);
    auto* pivots2_ = reinterpret_cast<slate::Pivots*>(pivots2);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::indefinite_solve_using_factor<scalar_t>(
        *A_, *pivots_, *T_, *pivots2_, *B_, opts_);
}


// gels
void slate_least_squares_solve_r32(
    slate_Matrix_r32 A,
    slate_Matrix_r32 BX,
    int num_opts, slate_Options opts[])
{
    using scalar_t    = float;
    using matrix_A_t  = slate::Matrix<scalar_t>;
    using matrix_BX_t = slate::Matrix<scalar_t>;

    auto* A_  = reinterpret_cast<matrix_A_t*>(A);
    auto* BX_ = reinterpret_cast<matrix_BX_t*>(BX);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::least_squares_solve<scalar_t>(*A_, *BX_, opts_);
}


// gels
void slate_least_squares_solve_r64(
    slate_Matrix_r64 A,
    slate_Matrix_r64 BX,
    int num_opts, slate_Options opts[])
{
    using scalar_t    = double;
    using matrix_A_t  = slate::Matrix<scalar_t>;
    using matrix_BX_t = slate::Matrix<scalar_t>;

    auto* A_  = reinterpret_cast<matrix_A_t*>(A);
    auto* BX_ = reinterpret_cast<matrix_BX_t*>(BX);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::least_squares_solve<scalar_t>(*A_, *BX_, opts_);
}


// gels
void slate_least_squares_solve_c32(
    slate_Matrix_c32 A,
    slate_Matrix_c32 BX,
    int num_opts, slate_Options opts[])
{
    using scalar_t    = std::complex<float>;
    using matrix_A_t  = slate::Matrix<scalar_t>;
    using matrix_BX_t = slate::Matrix<scalar_t>;

    auto* A_  = reinterpret_cast<matrix_A_t*>(A);
    auto* BX_ = reinterpret_cast<matrix_BX_t*>(BX);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::least_squares_solve<scalar_t>(*A_, *BX_, opts_);
}


// geqrf
void slate_qr_factor_r32(
    slate_Matrix_r32 A, slate_TriangularFactors_r32 T,
    int num_opts, slate_Options opts[])
{
    using scalar_t               = float;
    using matrix_A_t             = slate::Matrix<scalar_t>;
    using triangular_factors_T_t = slate::TriangularFactors<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* T_ = reinterpret_cast<triangular_factors_T_t*>(T);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::qr_factor<scalar_t>(*A_, *T_, opts_);
}


// geqrf
void slate_qr_factor_r64(
    slate_Matrix_r64 A, slate_TriangularFactors_r64 T,
    int num_opts, slate_Options opts[])
{
    using scalar_t               = double;
    using matrix_A_t             = slate::Matrix<scalar_t>;
    using triangular_factors_T_t = slate::TriangularFactors<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* T_ = reinterpret_cast<triangular_factors_T_t*>(T);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::qr_factor<scalar_t>(*A_, *T_, opts_);
}


// geqrf
void slate_qr_factor_c32(
    slate_Matrix_c32 A, slate_TriangularFactors_c32 T,
    int num_opts, slate_Options opts[])
{
    using scalar_t               = std::complex<float>;
    using matrix_A_t             = slate::Matrix<scalar_t>;
    using triangular_factors_T_t = slate::TriangularFactors<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* T_ = reinterpret_cast<triangular_factors_T_t*>(T);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::qr_factor<scalar_t>(*A_, *T_, opts_);
}


// unmqr
void slate_qr_multiply_by_q_r32(
    slate_Side side, slate_Op op,
    slate_Matrix_r32 A, slate_TriangularFactors_r32 T,
    slate_Matrix_r32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t               = float;
    using matrix_A_t             = slate::Matrix<scalar_t>;
    using triangular_factors_T_t = slate::TriangularFactors<scalar_t>;
    using matrix_C_t             = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* T_ = reinterpret_cast<triangular_factors_T_t*>(T);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::qr_multiply_by_q<scalar_t>(
        slate::side2cpp(side), slate::op2cpp(op), *A_, *T_, *C_, opts_);
}


// unmqr
void slate_qr_multiply_by_q_r64(
    slate_Side side, slate_Op op,
    slate_Matrix_r64 A, slate_TriangularFactors_r64 T,
    slate_Matrix_r64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t               = double;
    using matrix_A_t             = slate::Matrix<scalar_t>;
    using triangular_factors_T_t = slate::TriangularFactors<scalar_t>;
    using matrix_C_t             = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* T_ = reinterpret_cast<triangular_factors_T_t*>(T);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::qr_multiply_by_q<scalar_t>(
        slate::side2cpp(side), slate::op2cpp(op), *A_, *T_, *C_, opts_);
}


// unmqr
void slate_qr_multiply_by_q_c32(
    slate_Side side, slate_Op op,
    slate_Matrix_c32 A, slate_TriangularFactors_c32 T,
    slate_Matrix_c32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t               = std::complex<float>;
    using matrix_A_t             = slate::Matrix<scalar_t>;
    using triangular_factors_T_t = slate::TriangularFactors<scalar_t>;
    using matrix_C_t             = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* T_ = reinterpret_cast<triangular_factors_T_t*>(T);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::qr_multiply_by_q<scalar_t>(
        slate::side2cpp(side), slate::op2cpp(op), *A_, *T_, *C_, opts_);
}


// gelqf
void slate_lq_factor_r32(
    slate_Matrix_r32 A, slate_TriangularFactors_r32 T,
    int num_opts, slate_Options opts[])
{
    using scalar_t               = float;
    using matrix_A_t             = slate::Matrix<scalar_t>;
    using triangular_factors_T_t = slate::TriangularFactors<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* T_ = reinterpret_cast<triangular_factors_T_t*>(T);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lq_factor<scalar_t>(*A_, *T_, opts_);
}


// gelqf
void slate_lq_factor_r64(
    slate_Matrix_r64 A, slate_TriangularFactors_r64 T,
    int num_opts, slate_Options opts[])
{
    using scalar_t               = double;
    using matrix_A_t             = slate::Matrix<scalar_t>;
    using triangular_factors_T_t = slate::TriangularFactors<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* T_ = reinterpret_cast<triangular_factors_T_t*>(T);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lq_factor<scalar_t>(*A_, *T_, opts_);
}


// gelqf
void slate_lq_factor_c32(
    slate_Matrix_c32 A, slate_TriangularFactors_c32 T,
    int num_opts, slate_Options opts[])
{
    using scalar_t               = std::complex<float>;
    using matrix_A_t             = slate::Matrix<scalar_t>;
    using triangular_factors_T_t = slate::TriangularFactors<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* T_ = reinterpret_cast<triangular_factors_T_t*>(T);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lq_factor<scalar_t>(*A_, *T_, opts_);
}


// unmlq
void slate_lq_multiply_by_q_r32(
    slate_Side side, slate_Op op,
    slate_Matrix_r32 A, slate_TriangularFactors_r32 T,
    slate_Matrix_r32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t               = float;
    using matrix_A_t             = slate::Matrix<scalar_t>;
    using triangular_factors_T_t = slate::TriangularFactors<scalar_t>;
    using matrix_C_t             = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* T_ = reinterpret_cast<triangular_factors_T_t*>(T);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lq_multiply_by_q<scalar_t>(
        slate::side2cpp(side), slate::op2cpp(op), *A_, *T_, *C_, opts_);
}


// unmlq
void slate_lq_multiply_by_q_r64(
    slate_Side side, slate_Op op,
    slate_Matrix_r64 A, slate_TriangularFactors_r64 T,
    slate_Matrix_r64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t               = double;
    using matrix_A_t             = slate::Matrix<scalar_t>;
    using triangular_factors_T_t = slate::TriangularFactors<scalar_t>;
    using matrix_C_t             = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* T_ = reinterpret_cast<triangular_factors_T_t*>(T);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lq_multiply_by_q<scalar_t>(
        slate::side2cpp(side), slate::op2cpp(op), *A_, *T_, *C_, opts_);
}


// unmlq
void slate_lq_multiply_by_q_c32(
    slate_Side side, slate_Op op,
    slate_Matrix_c32 A, slate_TriangularFactors_c32 T,
    slate_Matrix_c32 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t               = std::complex<float>;
    using matrix_A_t             = slate::Matrix<scalar_t>;
    using triangular_factors_T_t = slate::TriangularFactors<scalar_t>;
    using matrix_C_t             = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* T_ = reinterpret_cast<triangular_factors_T_t*>(T);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lq_multiply_by_q<scalar_t>(
        slate::side2cpp(side), slate::op2cpp(op), *A_, *T_, *C_, opts_);
}


// gesvd
void slate_svd_vals_r32(
    slate_Matrix_r32 A,
    float* Sigma,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    int64_t minmn = std::min(A_->m(), A_->n());
    std::vector< blas::real_type<scalar_t> > Sigma_(minmn);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::svd_vals<scalar_t>(*A_, Sigma_, opts_);

    Sigma = &Sigma_[0];
}


// gesvd
void slate_svd_vals_r64(
    slate_Matrix_r64 A,
    double* Sigma,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    int64_t minmn = std::min(A_->m(), A_->n());
    std::vector< blas::real_type<scalar_t> > Sigma_(minmn);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::svd_vals<scalar_t>(*A_, Sigma_, opts_);

    Sigma = &Sigma_[0];
}


// gesvd
void slate_svd_vals_c32(
    slate_Matrix_c32 A,
    float* Sigma,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    int64_t minmn = std::min(A_->m(), A_->n());
    std::vector< blas::real_type<scalar_t> > Sigma_(minmn);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::svd_vals<scalar_t>(*A_, Sigma_, opts_);

    Sigma = &Sigma_[0];
}


// heev
void slate_hermitian_eig_vals_r32(
    slate_HermitianMatrix_r32 A,
    float* Lambda,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    std::vector< blas::real_type<scalar_t> > Lambda_(A_->n());

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::eig_vals<scalar_t>(*A_, Lambda_, opts_);

    Lambda = &Lambda_[0];
}


// heev
void slate_hermitian_eig_vals_r64(
    slate_HermitianMatrix_r64 A,
    double* Lambda,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    std::vector< blas::real_type<scalar_t> > Lambda_(A_->n());

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::eig_vals<scalar_t>(*A_, Lambda_, opts_);

    Lambda = &Lambda_[0];
}


// heev
void slate_hermitian_eig_vals_c32(
    slate_HermitianMatrix_c32 A,
    float* Lambda,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    std::vector< blas::real_type<scalar_t> > Lambda_(A_->n());

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::eig_vals<scalar_t>(*A_, Lambda_, opts_);

    Lambda = &Lambda_[0];
}


// hegv
void slate_generalized_hermitian_eig_vals_r32(
    int64_t itype,
    slate_HermitianMatrix_r32 A,
    slate_HermitianMatrix_r32 B,
    float* Lambda,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = float;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_B_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    std::vector< blas::real_type<scalar_t> > Lambda_(A_->n());

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::eig_vals<scalar_t>(itype, *A_, *B_, Lambda_, opts_);

    Lambda = &Lambda_[0];
}


// hegv
void slate_generalized_hermitian_eig_vals_r64(
    int64_t itype,
    slate_HermitianMatrix_r64 A,
    slate_HermitianMatrix_r64 B,
    double* Lambda,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = double;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_B_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    std::vector< blas::real_type<scalar_t> > Lambda_(A_->n());

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::eig_vals<scalar_t>(itype, *A_, *B_, Lambda_, opts_);

    Lambda = &Lambda_[0];
}


// hegv
void slate_generalized_hermitian_eig_vals_c32(
    int64_t itype,
    slate_HermitianMatrix_c32 A,
    slate_HermitianMatrix_c32 B,
    float* Lambda,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<float>;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_B_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    std::vector< blas::real_type<scalar_t> > Lambda_(A_->n());

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::eig_vals<scalar_t>(itype, *A_, *B_, Lambda_, opts_);

    Lambda = &Lambda_[0];
}


