// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/c_api/wrappers.h"
#include "slate/c_api/util.hh"

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
double slate_norm_c64(
    slate_Norm norm, slate_Matrix_c64 A,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    return slate::norm(slate::norm2cpp(norm), *A_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
double slate_band_norm_c64(
    slate_Norm norm, slate_BandMatrix_c64 A,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::BandMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    return slate::norm(slate::norm2cpp(norm), *A_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
double slate_hermitian_norm_c64(
    slate_Norm norm, slate_HermitianMatrix_c64 A,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    return slate::norm(slate::norm2cpp(norm), *A_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
double slate_hermitian_band_norm_c64(
    slate_Norm norm, slate_HermitianBandMatrix_c64 A,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::HermitianBandMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    return slate::norm(slate::norm2cpp(norm), *A_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
double slate_symmetric_norm_c64(
    slate_Norm norm, slate_SymmetricMatrix_c64 A,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::SymmetricMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    return slate::norm(slate::norm2cpp(norm), *A_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
double slate_trapezoid_norm_c64(
    slate_Norm norm, slate_TrapezoidMatrix_c64 A,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::TrapezoidMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    return slate::norm(slate::norm2cpp(norm), *A_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_band_multiply_c64(
    double _Complex alpha, slate_BandMatrix_c64 A,
                               slate_Matrix_c64 B,
    double _Complex beta,      slate_Matrix_c64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
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
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_multiply_c64(
    double _Complex alpha, slate_Matrix_c64 A,
                           slate_Matrix_c64 B,
    double _Complex beta,  slate_Matrix_c64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
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
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_hermitian_band_left_multiply_c64(
    double _Complex alpha, slate_HermitianBandMatrix_c64 A,
                                        slate_Matrix_c64 B,
    double _Complex beta,               slate_Matrix_c64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
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
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_hermitian_band_right_multiply_c64(
    double _Complex alpha,              slate_Matrix_c64 A,
                           slate_HermitianBandMatrix_c64 B,
    double _Complex beta,               slate_Matrix_c64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
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
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_hermitian_left_multiply_c64(
    double _Complex alpha, slate_HermitianMatrix_c64 A,
                                    slate_Matrix_c64 B,
    double _Complex beta,           slate_Matrix_c64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
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
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_hermitian_right_multiply_c64(
    double _Complex alpha,          slate_Matrix_c64 A,
                           slate_HermitianMatrix_c64 B,
    double _Complex beta,           slate_Matrix_c64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
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
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_symmetric_left_multiply_c64(
    double _Complex alpha, slate_SymmetricMatrix_c64 A,
                                    slate_Matrix_c64 B,
    double _Complex beta,           slate_Matrix_c64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
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
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_symmetric_right_multiply_c64(
    double _Complex alpha,          slate_Matrix_c64 A,
                           slate_SymmetricMatrix_c64 B,
    double _Complex beta,           slate_Matrix_c64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
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
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_triangular_left_multiply_c64(
    double _Complex alpha, slate_TriangularMatrix_c64 A,
                                     slate_Matrix_c64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::TriangularMatrix<scalar_t>;
    using matrix_B_t = slate::          Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_multiply<scalar_t>(alpha, *A_, *B_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_triangular_right_multiply_c64(
    double _Complex alpha,           slate_Matrix_c64 A,
                           slate_TriangularMatrix_c64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::          Matrix<scalar_t>;
    using matrix_B_t = slate::TriangularMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_multiply<scalar_t>(alpha, *A_, *B_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_triangular_band_left_solve_c64(
    double _Complex alpha, slate_TriangularBandMatrix_c64 A,
                                         slate_Matrix_c64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::TriangularBandMatrix<scalar_t>;
    using matrix_B_t = slate::              Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_solve<scalar_t>(alpha, *A_, *B_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_triangular_band_right_solve_c64(
    double _Complex alpha,               slate_Matrix_c64 A,
                           slate_TriangularBandMatrix_c64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::              Matrix<scalar_t>;
    using matrix_B_t = slate::TriangularBandMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_solve<scalar_t>(alpha, *A_, *B_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_triangular_left_solve_c64(
    double _Complex alpha, slate_TriangularMatrix_c64 A,
                                     slate_Matrix_c64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::TriangularMatrix<scalar_t>;
    using matrix_B_t = slate::              Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_solve<scalar_t>(alpha, *A_, *B_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_triangular_right_solve_c64(
    double _Complex alpha,           slate_Matrix_c64 A,
                           slate_TriangularMatrix_c64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::          Matrix<scalar_t>;
    using matrix_B_t = slate::TriangularMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::triangular_solve<scalar_t>(alpha, *A_, *B_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_hermitian_rank_k_update_c64(
    double alpha,          slate_Matrix_c64 A,
    double beta,  slate_HermitianMatrix_c64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::         Matrix<scalar_t>;
    using matrix_C_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::rank_k_update<scalar_t>(alpha, *A_, beta, *C_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_symmetric_rank_k_update_c64(
    double _Complex alpha,           slate_Matrix_c64 A,
    double _Complex beta,   slate_SymmetricMatrix_c64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::         Matrix<scalar_t>;
    using matrix_C_t = slate::SymmetricMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* C_ = reinterpret_cast<matrix_C_t*>(C);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::rank_k_update<scalar_t>(alpha, *A_, beta, *C_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_hermitian_rank_2k_update_c64(
    double _Complex alpha,  slate_Matrix_c64 A,
                            slate_Matrix_c64 B,
    double beta,   slate_HermitianMatrix_c64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
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
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_symmetric_rank_2k_update_c64(
    double _Complex alpha,            slate_Matrix_c64 A,
                                      slate_Matrix_c64 B,
    double _Complex beta,    slate_SymmetricMatrix_c64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
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
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_band_lu_solve_c64(
    slate_BandMatrix_c64 A,
        slate_Matrix_c64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::BandMatrix<scalar_t>;
    using matrix_B_t = slate::    Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve<scalar_t>(*A_, *B_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_lu_solve_c64(
    slate_Matrix_c64 A,
    slate_Matrix_c64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::Matrix<scalar_t>;
    using matrix_B_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve<scalar_t>(*A_, *B_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_lu_solve_nopiv_c64(
    slate_Matrix_c64 A,
    slate_Matrix_c64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::Matrix<scalar_t>;
    using matrix_B_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve_nopiv<scalar_t>(*A_, *B_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_band_lu_factor_c64(
    slate_BandMatrix_c64 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::BandMatrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* pivots_= reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_factor<scalar_t>(*A_, *pivots_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_lu_factor_c64(
    slate_Matrix_c64 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::Matrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* pivots_= reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_factor<scalar_t>(*A_, *pivots_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_lu_factor_nopiv_c64(
    slate_Matrix_c64 A,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_factor_nopiv<scalar_t>(*A_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_band_lu_solve_using_factor_c64(
    slate_BandMatrix_c64 A, slate_Pivots pivots,
        slate_Matrix_c64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::BandMatrix<scalar_t>;
    using matrix_B_t = slate::    Matrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* B_     = reinterpret_cast<matrix_B_t*>(B);
    auto* pivots_= reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve_using_factor<scalar_t>(*A_, *pivots_, *B_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_lu_solve_using_factor_c64(
    slate_Matrix_c64 A, slate_Pivots pivots,
    slate_Matrix_c64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::Matrix<scalar_t>;
    using matrix_B_t = slate::Matrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* B_     = reinterpret_cast<matrix_B_t*>(B);
    auto* pivots_= reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve_using_factor<scalar_t>(*A_, *pivots_, *B_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_lu_solve_using_factor_nopiv_c64(
    slate_Matrix_c64 A,
    slate_Matrix_c64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::Matrix<scalar_t>;
    using matrix_B_t = slate::Matrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* B_     = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_solve_using_factor_nopiv<scalar_t>(*A_, *B_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_lu_inverse_using_factor_c64(
    slate_Matrix_c64 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::Matrix<scalar_t>;

    auto* A_     = reinterpret_cast<matrix_A_t*>(A);
    auto* pivots_= reinterpret_cast<slate::Pivots*>(pivots);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lu_inverse_using_factor<scalar_t>(*A_, *pivots_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_lu_inverse_using_factor_out_of_place_c64(
    slate_Matrix_c64 A, slate_Pivots pivots,
    slate_Matrix_c64 A_inverse,
    int num_opts, slate_Options opts[])
{
    using scalar_t           = std::complex<double>;
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
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_band_chol_solve_c64(
    slate_HermitianBandMatrix_c64 A,
                 slate_Matrix_c64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::HermitianBandMatrix<scalar_t>;
    using matrix_B_t = slate::             Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_solve<scalar_t>(*A_, *B_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_chol_solve_c64(
    slate_HermitianMatrix_c64 A,
             slate_Matrix_c64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_solve<scalar_t>(*A_, *B_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_band_chol_factor_c64(
    slate_HermitianBandMatrix_c64 A,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::HermitianBandMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_factor<scalar_t>(*A_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_chol_factor_c64(
    slate_HermitianMatrix_c64 A,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_factor<scalar_t>(*A_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_band_chol_solve_using_factor_c64(
    slate_HermitianBandMatrix_c64 A,
                 slate_Matrix_c64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::HermitianBandMatrix<scalar_t>;
    using matrix_B_t = slate::             Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_solve_using_factor<scalar_t>(*A_, *B_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_chol_solve_using_factor_c64(
    slate_HermitianMatrix_c64 A,
             slate_Matrix_c64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_solve_using_factor<scalar_t>(*A_, *B_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_chol_inverse_using_factor_c64(
    slate_HermitianMatrix_c64 A,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::chol_inverse_using_factor<scalar_t>(*A_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_indefinite_solve_c64(
    slate_HermitianMatrix_c64 A,
             slate_Matrix_c64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_B_t = slate::         Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::indefinite_solve<scalar_t>(*A_, *B_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_indefinite_factor_c64(
    slate_HermitianMatrix_c64 A, slate_Pivots pivots,
         slate_BandMatrix_c64 T, slate_Pivots pivots2,
             slate_Matrix_c64 H,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
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
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_indefinite_solve_using_factor_c64(
    slate_HermitianMatrix_c64 A, slate_Pivots pivots,
         slate_BandMatrix_c64 T, slate_Pivots pivots2,
             slate_Matrix_c64 B,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
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
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_least_squares_solve_c64(
    slate_Matrix_c64 A,
    slate_Matrix_c64 BX,
    int num_opts, slate_Options opts[])
{
    using scalar_t    = std::complex<double>;
    using matrix_A_t  = slate::Matrix<scalar_t>;
    using matrix_BX_t = slate::Matrix<scalar_t>;

    auto* A_  = reinterpret_cast<matrix_A_t*>(A);
    auto* BX_ = reinterpret_cast<matrix_BX_t*>(BX);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::least_squares_solve<scalar_t>(*A_, *BX_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_qr_factor_c64(
    slate_Matrix_c64 A, slate_TriangularFactors_c64 T,
    int num_opts, slate_Options opts[])
{
    using scalar_t               = std::complex<double>;
    using matrix_A_t             = slate::Matrix<scalar_t>;
    using triangular_factors_T_t = slate::TriangularFactors<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* T_ = reinterpret_cast<triangular_factors_T_t*>(T);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::qr_factor<scalar_t>(*A_, *T_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_qr_multiply_by_q_c64(
    slate_Side side, slate_Op op,
    slate_Matrix_c64 A, slate_TriangularFactors_c64 T,
    slate_Matrix_c64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t               = std::complex<double>;
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
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_lq_factor_c64(
    slate_Matrix_c64 A, slate_TriangularFactors_c64 T,
    int num_opts, slate_Options opts[])
{
    using scalar_t               = std::complex<double>;
    using matrix_A_t             = slate::Matrix<scalar_t>;
    using triangular_factors_T_t = slate::TriangularFactors<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* T_ = reinterpret_cast<triangular_factors_T_t*>(T);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::lq_factor<scalar_t>(*A_, *T_, opts_);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_lq_multiply_by_q_c64(
    slate_Side side, slate_Op op,
    slate_Matrix_c64 A, slate_TriangularFactors_c64 T,
    slate_Matrix_c64 C,
    int num_opts, slate_Options opts[])
{
    using scalar_t               = std::complex<double>;
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
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_svd_vals_c64(
    slate_Matrix_c64 A,
    double* Sigma,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    int64_t min_mn = std::min( A_->m(), A_->n() );
    std::vector< blas::real_type<scalar_t> > Sigma_( min_mn );

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::svd_vals<scalar_t>(*A_, Sigma_, opts_);

    std::copy(Sigma_.begin(), Sigma_.end(), Sigma);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function
void slate_hermitian_eig_vals_c64(
    slate_HermitianMatrix_c64 A,
    double* Lambda,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    std::vector< blas::real_type<scalar_t> > Lambda_(A_->n());

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::eig_vals<scalar_t>(*A_, Lambda_, opts_);

    std::copy(Lambda_.begin(), Lambda_.end(), Lambda);
}
// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers.py script;
// do not modify!
// @begin function

void slate_generalized_hermitian_eig_vals_c64(
    int64_t itype,
    slate_HermitianMatrix_c64 A,
    slate_HermitianMatrix_c64 B,
    double* Lambda,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::HermitianMatrix<scalar_t>;
    using matrix_B_t = slate::HermitianMatrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);
    auto* B_ = reinterpret_cast<matrix_B_t*>(B);

    std::vector< blas::real_type<scalar_t> > Lambda_(A_->n());

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::eig_vals<scalar_t>(itype, *A_, *B_, Lambda_, opts_);

    std::copy(Lambda_.begin(), Lambda_.end(), Lambda);
}
// @end function
//--------------------
