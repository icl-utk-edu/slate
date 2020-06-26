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
#include "slate/c_api/util.hh"

/// slate::Pivots
slate_Pivots slate_Pivots_create()
{
    auto* pivots = new slate::Pivots();
    return reinterpret_cast<slate_Pivots>(pivots);
}
void slate_Pivots_destroy(slate_Pivots pivots)
{
    auto* pivots_ = reinterpret_cast<slate::Pivots*>(pivots);
    delete pivots_;
}

//--------------------
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin matrix code block

//--------------------
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

slate_Matrix_c64 slate_Matrix_create_c64(
    int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    using scalar_t = std::complex<double>;
    using matrix_t = slate::Matrix<scalar_t>;

    auto* A = new matrix_t(m, n, nb, p, q, mpi_comm);

    return reinterpret_cast<slate_Matrix_c64>(A);
}

// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

void slate_Matrix_destroy_c64(slate_Matrix_c64 A)
{
    using scalar_t = std::complex<double>;
    using matrix_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_t*>(A);

    delete A_;
}

// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::Matrix<std::complex<double>>::insertLocalTiles()
void slate_Matrix_insertLocalTiles_c64(slate_Matrix_c64 A)
{
    using scalar_t = std::complex<double>;
    using matrix_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_t*>(A);

    A_->insertLocalTiles();
}

// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::Matrix<std::complex<double>>::mt()
int64_t slate_Matrix_mt_c64(slate_Matrix_c64 A)
{
    using scalar_t = std::complex<double>;
    using matrix_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_t*>(A);

    return(A_->mt());
}

// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::Matrix<std::complex<double>>::nt()
int64_t slate_Matrix_nt_c64(slate_Matrix_c64 A)
{
    using scalar_t = std::complex<double>;
    using matrix_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_t*>(A);

    return(A_->nt());
}

// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::Matrix<std::complex<double>>::m()
int64_t slate_Matrix_m_c64(slate_Matrix_c64 A)
{
    using scalar_t = std::complex<double>;
    using matrix_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_t*>(A);

    return(A_->m());
}

// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::Matrix<std::complex<double>>::n()
int64_t slate_Matrix_n_c64(slate_Matrix_c64 A)
{
    using scalar_t = std::complex<double>;
    using matrix_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_t*>(A);

    return(A_->n());
}

// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::Matrix<std::complex<double>>::tileIsLocal()
bool slate_Matrix_tileIsLocal_c64(slate_Matrix_c64 A, int64_t i, int64_t j)
{
    using scalar_t = std::complex<double>;
    using matrix_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_t*>(A);

    return(A_->tileIsLocal(i, j));
}

// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::Matrix<std::complex<double>>::at(i, j)
slate_Tile_c64 slate_Matrix_at_c64(slate_Matrix_c64 A, int64_t i, int64_t j)
{
    using scalar_t = std::complex<double>;
    using matrix_t = slate::Matrix<scalar_t>;
    using tile_t   = slate::Tile<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_t*>(A);

    tile_t T = A_->at(i, j);

    return(*reinterpret_cast<slate_Tile_c64*>(&T));
}

// @end function
//--------------------

// @end matrix code block
//--------------------

//--------------------
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::TriangularFactors<std::complex<double>>
slate_TriangularFactors_c64 slate_TriangularFactors_create_c64()
{
    using scalar_t             = std::complex<double>;
    using triangular_factors_t = slate::TriangularFactors<scalar_t>;

    auto* T = new triangular_factors_t();

    return reinterpret_cast<slate_TriangularFactors_c64>(T);
}

// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::TriangularFactors<std::complex<double>>
void slate_TriangularFactors_destroy_c64(slate_TriangularFactors_c64 T)
{
    using scalar_t             = std::complex<double>;
    using triangular_factors_t = slate::TriangularFactors<scalar_t>;

    auto* T_ = reinterpret_cast<triangular_factors_t*>(T);

    delete T_;
}

// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::Tile<std::complex<double>>::mb()
int64_t slate_Tile_mb_c64(slate_Tile_c64 T)
{
    using scalar_t = std::complex<double>;
    using tile_t   = slate::Tile<scalar_t>;

    assert(sizeof(slate_Tile_c64) == sizeof(tile_t));
    auto T_ = *reinterpret_cast<tile_t*>(&T);

    return(T_.mb());
}

// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::Tile<std::complex<double>>::nb()
int64_t slate_Tile_nb_c64(slate_Tile_c64 T)
{
    using scalar_t = std::complex<double>;
    using tile_t   = slate::Tile<scalar_t>;

    assert(sizeof(slate_Tile_c64) == sizeof(tile_t));
    auto T_ = *reinterpret_cast<tile_t*>(&T);

    return(T_.nb());
}

// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::Tile<std::complex<double>>::stride()
int64_t slate_Tile_stride_c64(slate_Tile_c64 T)
{
    using scalar_t = std::complex<double>;
    using tile_t   = slate::Tile<scalar_t>;

    assert(sizeof(slate_Tile_c64) == sizeof(tile_t));
    auto T_ = *reinterpret_cast<tile_t*>(&T);

    return(T_.stride());
}

// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::Tile<std::complex<double>>::data()
double _Complex* slate_Tile_data_c64(slate_Tile_c64 T)
{
    using scalar_t = std::complex<double>;
    using tile_t   = slate::Tile<scalar_t>;

    assert(sizeof(slate_Tile_c64) == sizeof(tile_t));
    auto T_ = *reinterpret_cast<tile_t*>(&T);

    return((double _Complex*)T_.data());
}

// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::multiply<std::complex<double>>
void slate_Band_multiply_c64(
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::multiply<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::multiply<std::complex<double>>
void slate_HermitianBand_left_multiply_c64(
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::multiply<std::complex<double>>
void slate_HermitianBand_right_multiply_c64(
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::multiply<std::complex<double>>
void slate_Hermitian_left_multiply_c64(
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::multiply<std::complex<double>>
void slate_Hermitian_right_multiply_c64(
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::multiply<std::complex<double>>
void slate_Symmetric_left_multiply_c64(
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::multiply<std::complex<double>>
void slate_Symmetric_right_multiply_c64(
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::triangular_multiply<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::triangular_multiply<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::triangular_solve<std::complex<double>>
void slate_Band_triangular_left_solve_c64(
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::triangular_solve<std::complex<double>>
void slate_Band_triangular_right_solve_c64(
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::triangular_solve<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::triangular_solve<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::rank_k_update<std::complex<double>>
void slate_Hermitian_rank_k_update_c64(
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::rank_k_update<std::complex<double>>
void slate_Symmetric_rank_k_update_c64(
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::rank_2k_update<std::complex<double>>
void slate_Hermitian_rank_2k_update_c64(
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::rank_2k_update<std::complex<double>>
void slate_Symmetric_rank_2k_update_c64(
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::lu_solve<std::complex<double>>
void slate_Band_lu_solve_c64(
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::lu_solve<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::lu_solve_nopiv<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::lu_factor<std::complex<double>>
void slate_Band_lu_factor_c64(
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::lu_factor<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::lu_factor_nopiv<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::lu_solve_using_factor<std::complex<double>>
void slate_Band_lu_solve_using_factor_c64(
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::lu_solve_using_factor<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::lu_solve_using_factor_nopiv<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::lu_inverse_using_factor<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::lu_inverse_using_factor_out_of_place<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::chol_solve<std::complex<double>>
void slate_Band_chol_solve_c64(
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::chol_solve<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::chol_factor<std::complex<double>>
void slate_Band_chol_factor_c64(
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::chol_factor<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::chol_solve_using_factor<std::complex<double>>
void slate_Band_chol_solve_using_factor_c64(
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::chol_solve_using_factor<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::chol_inverse_using_factor<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::indefinite_solve<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::indefinite_factor<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::indefinite_solve_using_factor<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::least_squares_solve<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::qr_factor<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::qr_multiply_by_q<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::lq_factor<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::lq_multiply_by_q<std::complex<double>>
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
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::svd_vals<std::complex<double>>
void slate_svd_vals_c64(
    slate_Matrix_c64 A,
    double* Sigma,
    int num_opts, slate_Options opts[])
{
    using scalar_t   = std::complex<double>;
    using matrix_A_t = slate::Matrix<scalar_t>;

    auto* A_ = reinterpret_cast<matrix_A_t*>(A);

    int64_t minmn = std::min(A_->m(), A_->n());
    std::vector< blas::real_type<scalar_t> > Sigma_(minmn);

    slate::Options opts_;
    slate::options2cpp(num_opts, opts, opts_);

    slate::svd_vals<scalar_t>(*A_, Sigma_, opts_);

    Sigma = &Sigma_[0];
}

// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::eig_vals<std::complex<double>>
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

    Lambda = &Lambda_[0];
}

// @end function
//--------------------

//--------------------
// begin/end markup used by generate_wrappers_precisions_cc.py script;
// do not modify!
// @begin function

/// slate::eig_vals<std::complex<double>>
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

    Lambda = &Lambda_[0];
}

// @end function
//--------------------
