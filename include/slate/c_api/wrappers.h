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

#ifndef SLATE_C_API_WRAPPERS_H
#define SLATE_C_API_WRAPPERS_H

#include "slate/c_api/types.h"
#include "slate/c_api/Matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Level 3 BLAS and LAPACK auxiliary

//-----------------------------------------
// multiply()

// gbmm
void slate_Band_multiply_c64(
    double _Complex alpha, slate_BandMatrix_c64 A,
                               slate_Matrix_c64 B,
    double _Complex beta,      slate_Matrix_c64 C,
    int num_opts, slate_Options opts[]);

// gemm
void slate_multiply_c64(
    double _Complex alpha, slate_Matrix_c64 A,
                           slate_Matrix_c64 B,
    double _Complex beta,  slate_Matrix_c64 C,
    int num_opts, slate_Options opts[]);

// Left hbmm
void slate_HermitianBand_left_multiply_c64(
    double _Complex alpha, slate_HermitianBandMatrix_c64 A,
                                        slate_Matrix_c64 B,
    double _Complex beta,               slate_Matrix_c64 C,
    int num_opts, slate_Options opts[]);

// Right hbmm
void slate_HermitianBand_right_multiply_c64(
    double _Complex alpha,              slate_Matrix_c64 A,
                           slate_HermitianBandMatrix_c64 B,
    double _Complex beta,               slate_Matrix_c64 C,
    int num_opts, slate_Options opts[]);

// Left hemm
void slate_Hermitian_left_multiply_c64(
    double _Complex alpha, slate_HermitianMatrix_c64 A,
                                    slate_Matrix_c64 B,
    double _Complex beta,           slate_Matrix_c64 C,
    int num_opts, slate_Options opts[]);

// Right hemm
void slate_Hermitian_right_multiply_c64(
    double _Complex alpha,          slate_Matrix_c64 A,
                           slate_HermitianMatrix_c64 B,
    double _Complex beta,           slate_Matrix_c64 C,
    int num_opts, slate_Options opts[]);

// Left symm
void slate_Symmetric_left_multiply_c64(
    double _Complex alpha, slate_SymmetricMatrix_c64 A,
                                    slate_Matrix_c64 B,
    double _Complex beta,           slate_Matrix_c64 C,
    int num_opts, slate_Options opts[]);

// Right symm
void slate_Symmetric_right_multiply_c64(
    double _Complex alpha,          slate_Matrix_c64 A,
                           slate_SymmetricMatrix_c64 B,
    double _Complex beta,           slate_Matrix_c64 C,
    int num_opts, slate_Options opts[]);

//-----------------------------------------
// triangular_multiply()

// Left trmm
void slate_triangular_left_multiply_c64(
    double _Complex alpha, slate_TriangularMatrix_c64 A,
                                     slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

// Right trmm
void slate_triangular_right_multiply_c64(
    double _Complex alpha,           slate_Matrix_c64 A,
                           slate_TriangularMatrix_c64 B,
    int num_opts, slate_Options opts[]);

//-----------------------------------------
// triangular_solve()

// Left tbsm
void slate_Band_triangular_left_solve_c64(
    double _Complex alpha, slate_TriangularBandMatrix_c64 A,
                                         slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

// Right tbsm
void slate_Band_triangular_right_solve_c64(
    double _Complex alpha,               slate_Matrix_c64 A,
                           slate_TriangularBandMatrix_c64 B,
    int num_opts, slate_Options opts[]);

// Left trsm
void slate_triangular_left_solve_c64(
    double _Complex alpha, slate_TriangularMatrix_c64 A,
                                     slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

// Right trsm
void slate_triangular_right_solve_c64(
    double _Complex alpha,           slate_Matrix_c64 A,
                           slate_TriangularMatrix_c64 B,
    int num_opts, slate_Options opts[]);

//-----------------------------------------
// rank_k_update()

// herk
void slate_Hermitian_rank_k_update_c64(
    double alpha,          slate_Matrix_c64 A,
    double beta,  slate_HermitianMatrix_c64 C,
    int num_opts, slate_Options opts[]);

// syrk
void slate_Symmetric_rank_k_update_c64(
    double _Complex alpha,           slate_Matrix_c64 A,
    double _Complex beta,   slate_SymmetricMatrix_c64 C,
    int num_opts, slate_Options opts[]);

//-----------------------------------------
// rank_2k_update()

// herk
void slate_Hermitian_rank_2k_update_c64(
    double _Complex alpha,  slate_Matrix_c64 A,
                            slate_Matrix_c64 B,
    double beta,   slate_HermitianMatrix_c64 C,
    int num_opts, slate_Options opts[]);

// syrk
void slate_Symmetric_rank_2k_update_c64(
    double _Complex alpha,            slate_Matrix_c64 A,
                                      slate_Matrix_c64 B,
    double _Complex beta,    slate_SymmetricMatrix_c64 C,
    int num_opts, slate_Options opts[]);

//------------------------------------------------------------------------------
// Linear systems

//-----------------------------------------
// LU

//-----------------------------------------
// lu_solve()

// gbsv
void slate_Band_lu_solve_c64(
    slate_BandMatrix_c64 A,
        slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

// gesv
void slate_lu_solve_c64(
    slate_Matrix_c64 A,
    slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

// gesv_nopiv
void slate_lu_solve_nopiv_c64(
    slate_Matrix_c64 A,
    slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

//-----------------------------------------
// lu_factor()

// gbtrf
void slate_Band_lu_factor_c64(
    slate_BandMatrix_c64 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[]);

// getrf
void slate_lu_factor_c64(
    slate_Matrix_c64 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[]);

// getrf_nopiv
void slate_lu_factor_nopiv_c64(
    slate_Matrix_c64 A,
    int num_opts, slate_Options opts[]);

//-----------------------------------------
// lu_solve_using_factor()

// gbtrs
void slate_Band_lu_solve_using_factor_c64(
    slate_BandMatrix_c64 A, slate_Pivots pivots,
        slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

// getrs
void slate_lu_solve_using_factor_c64(
    slate_Matrix_c64 A, slate_Pivots pivots,
    slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

// getrs_nopiv
void slate_lu_solve_using_factor_nopiv_c64(
    slate_Matrix_c64 A,
    slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

//-----------------------------------------
// lu_inverse_using_factor()

// In-place getri
void slate_lu_inverse_using_factor_c64(
    slate_Matrix_c64 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[]);

//-----------------------------------------
// lu_inverse_using_factor_out_of_place()

// Out-of-place getri
void slate_lu_inverse_using_factor_out_of_place_c64(
    slate_Matrix_c64 A, slate_Pivots pivots,
    slate_Matrix_c64 A_inverse,
    int num_opts, slate_Options opts[]);

//-----------------------------------------
// Cholesky

//-----------------------------------------
// chol_solve()

// pbsv
void slate_Band_chol_solve_c64(
    slate_HermitianBandMatrix_c64 A,
                 slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

// posv
void slate_chol_solve_c64(
    slate_HermitianMatrix_c64 A,
             slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

//-----------------------------------------
// chol_factor()

// pbtrf
void slate_Band_chol_factor_c64(
    slate_HermitianBandMatrix_c64 A,
    int num_opts, slate_Options opts[]);

// potrf
void slate_chol_factor_c64(
    slate_HermitianMatrix_c64 A,
    int num_opts, slate_Options opts[]);

//-----------------------------------------
// chol_solve_using_factor()

// pbtrs
void slate_Band_chol_solve_using_factor_c64(
    slate_HermitianBandMatrix_c64 A,
                 slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

// potrs
void slate_chol_solve_using_factor_c64(
    slate_HermitianMatrix_c64 A,
             slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

//-----------------------------------------
// chol_inverse_using_factor()

// potri
void slate_chol_inverse_using_factor_c64(
    slate_HermitianMatrix_c64 A,
    int num_opts, slate_Options opts[]);

//-----------------------------------------
// Symmetric indefinite -- block Aasen's

//-----------------------------------------
// indefinite_solve()

// hesv
void slate_indefinite_solve_c64(
    slate_HermitianMatrix_c64 A,
             slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

//-----------------------------------------
// indefinite_factor()

// hetrf
void slate_indefinite_factor_c64(
    slate_HermitianMatrix_c64 A, slate_Pivots pivots,
         slate_BandMatrix_c64 T, slate_Pivots pivots2,
             slate_Matrix_c64 H,
    int num_opts, slate_Options opts[]);

//-----------------------------------------
// indefinite_solve_using_factor()

// hetrs
void slate_indefinite_solve_using_factor_c64(
    slate_HermitianMatrix_c64 A, slate_Pivots pivots,
         slate_BandMatrix_c64 T, slate_Pivots pivots2,
             slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

//------------------------------------------------------------------------------
// QR

//-----------------------------------------
// Least squares

//-----------------------------------------
// least_squares_solve()

// gels
void slate_least_squares_solve_c64(
    slate_Matrix_c64 A,
    slate_Matrix_c64 BX,
    int num_opts, slate_Options opts[]);

//-----------------------------------------
// QR

//-----------------------------------------
// qr_factor()

// geqrf
void slate_qr_factor_c64(
    slate_Matrix_c64 A, slate_TriangularFactors_c64 T,
    int num_opts, slate_Options opts[]);

//-----------------------------------------
// qr_multiply_by_q()

// unmqr
void slate_qr_multiply_by_q_c64(
    slate_Side side, slate_Op op,
    slate_Matrix_c64 A, slate_TriangularFactors_c64 T,
    slate_Matrix_c64 C,
    int num_opts, slate_Options opts[]);

//-----------------------------------------
// LQ

//-----------------------------------------
// lq_factor()

// gelqf
void slate_lq_factor_c64(
    slate_Matrix_c64 A, slate_TriangularFactors_c64 T,
    int num_opts, slate_Options opts[]);

//-----------------------------------------
// lq_multiply_by_q()

// unmlq
void slate_lq_multiply_by_q_c64(
    slate_Side side, slate_Op op,
    slate_Matrix_c64 A, slate_TriangularFactors_c64 T,
    slate_Matrix_c64 C,
    int num_opts, slate_Options opts[]);

//------------------------------------------------------------------------------
// SVD

//-----------------------------------------
// svd_vals()

// gesvd
void slate_svd_vals_c64(
    slate_Matrix_c64 A,
    double* Sigma,
    int num_opts, slate_Options opts[]);

//------------------------------------------------------------------------------
// Eigenvalue decomposition

//-----------------------------------------
// eig_vals()

//-----------------------------------------
// Symmetric/hermitian

// heev
void slate_hermitian_eig_vals_c64(
    slate_HermitianMatrix_c64 A,
    double* Lambda,
    int num_opts, slate_Options opts[]);

//-----------------------------------------
// Generalized symmetric/hermitian

// hegv
void slate_generalized_hermitian_eig_vals_c64(
    int64_t itype,
    slate_HermitianMatrix_c64 A,
    slate_HermitianMatrix_c64 B,
    double* Lambda,
    int num_opts, slate_Options opts[]);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // SLATE_C_API_WRAPPERS_H
