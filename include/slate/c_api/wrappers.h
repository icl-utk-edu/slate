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

//------------------------------------------------------------------------------
// Auto-generated file by tools/c_api/generate_wrappers.py
#ifndef SLATE_C_API_WRAPPERS_H
#define SLATE_C_API_WRAPPERS_H

#include "slate/c_api/matrix.h"
#include "slate/c_api/types.h"

#include <complex.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

float slate_norm_r32(
    slate_Norm norm, slate_Matrix_r32 A,
    int num_opts, slate_Options opts[]);
double slate_norm_r64(
    slate_Norm norm, slate_Matrix_r64 A,
    int num_opts, slate_Options opts[]);
float slate_norm_c32(
    slate_Norm norm, slate_Matrix_c32 A,
    int num_opts, slate_Options opts[]);
double slate_norm_c64(
    slate_Norm norm, slate_Matrix_c64 A,
    int num_opts, slate_Options opts[]);

float slate_band_norm_r32(
    slate_Norm norm, slate_BandMatrix_r32 A,
    int num_opts, slate_Options opts[]);
double slate_band_norm_r64(
    slate_Norm norm, slate_BandMatrix_r64 A,
    int num_opts, slate_Options opts[]);
float slate_band_norm_c32(
    slate_Norm norm, slate_BandMatrix_c32 A,
    int num_opts, slate_Options opts[]);
double slate_band_norm_c64(
    slate_Norm norm, slate_BandMatrix_c64 A,
    int num_opts, slate_Options opts[]);

float slate_hermitian_norm_r32(
    slate_Norm norm, slate_HermitianMatrix_r32 A,
    int num_opts, slate_Options opts[]);
double slate_hermitian_norm_r64(
    slate_Norm norm, slate_HermitianMatrix_r64 A,
    int num_opts, slate_Options opts[]);
float slate_hermitian_norm_c32(
    slate_Norm norm, slate_HermitianMatrix_c32 A,
    int num_opts, slate_Options opts[]);
double slate_hermitian_norm_c64(
    slate_Norm norm, slate_HermitianMatrix_c64 A,
    int num_opts, slate_Options opts[]);

float slate_hermitian_band_norm_r32(
    slate_Norm norm, slate_HermitianBandMatrix_r32 A,
    int num_opts, slate_Options opts[]);
double slate_hermitian_band_norm_r64(
    slate_Norm norm, slate_HermitianBandMatrix_r64 A,
    int num_opts, slate_Options opts[]);
float slate_hermitian_band_norm_c32(
    slate_Norm norm, slate_HermitianBandMatrix_c32 A,
    int num_opts, slate_Options opts[]);
double slate_hermitian_band_norm_c64(
    slate_Norm norm, slate_HermitianBandMatrix_c64 A,
    int num_opts, slate_Options opts[]);

float slate_symmetric_norm_r32(
    slate_Norm norm, slate_SymmetricMatrix_r32 A,
    int num_opts, slate_Options opts[]);
double slate_symmetric_norm_r64(
    slate_Norm norm, slate_SymmetricMatrix_r64 A,
    int num_opts, slate_Options opts[]);
float slate_symmetric_norm_c32(
    slate_Norm norm, slate_SymmetricMatrix_c32 A,
    int num_opts, slate_Options opts[]);
double slate_symmetric_norm_c64(
    slate_Norm norm, slate_SymmetricMatrix_c64 A,
    int num_opts, slate_Options opts[]);

float slate_trapezoid_norm_r32(
    slate_Norm norm, slate_TrapezoidMatrix_r32 A,
    int num_opts, slate_Options opts[]);
double slate_trapezoid_norm_r64(
    slate_Norm norm, slate_TrapezoidMatrix_r64 A,
    int num_opts, slate_Options opts[]);
float slate_trapezoid_norm_c32(
    slate_Norm norm, slate_TrapezoidMatrix_c32 A,
    int num_opts, slate_Options opts[]);
double slate_trapezoid_norm_c64(
    slate_Norm norm, slate_TrapezoidMatrix_c64 A,
    int num_opts, slate_Options opts[]);

void slate_band_multiply_r32(
    float alpha, slate_BandMatrix_r32 A,
                               slate_Matrix_r32 B,
    float beta,      slate_Matrix_r32 C,
    int num_opts, slate_Options opts[]);
void slate_band_multiply_r64(
    double alpha, slate_BandMatrix_r64 A,
                               slate_Matrix_r64 B,
    double beta,      slate_Matrix_r64 C,
    int num_opts, slate_Options opts[]);
void slate_band_multiply_c32(
    float _Complex alpha, slate_BandMatrix_c32 A,
                               slate_Matrix_c32 B,
    float _Complex beta,      slate_Matrix_c32 C,
    int num_opts, slate_Options opts[]);
void slate_band_multiply_c64(
    double _Complex alpha, slate_BandMatrix_c64 A,
                               slate_Matrix_c64 B,
    double _Complex beta,      slate_Matrix_c64 C,
    int num_opts, slate_Options opts[]);

void slate_multiply_r32(
    float alpha, slate_Matrix_r32 A,
                           slate_Matrix_r32 B,
    float beta,  slate_Matrix_r32 C,
    int num_opts, slate_Options opts[]);
void slate_multiply_r64(
    double alpha, slate_Matrix_r64 A,
                           slate_Matrix_r64 B,
    double beta,  slate_Matrix_r64 C,
    int num_opts, slate_Options opts[]);
void slate_multiply_c32(
    float _Complex alpha, slate_Matrix_c32 A,
                           slate_Matrix_c32 B,
    float _Complex beta,  slate_Matrix_c32 C,
    int num_opts, slate_Options opts[]);
void slate_multiply_c64(
    double _Complex alpha, slate_Matrix_c64 A,
                           slate_Matrix_c64 B,
    double _Complex beta,  slate_Matrix_c64 C,
    int num_opts, slate_Options opts[]);

void slate_hermitian_band_left_multiply_r32(
    float alpha, slate_HermitianBandMatrix_r32 A,
                                        slate_Matrix_r32 B,
    float beta,               slate_Matrix_r32 C,
    int num_opts, slate_Options opts[]);
void slate_hermitian_band_left_multiply_r64(
    double alpha, slate_HermitianBandMatrix_r64 A,
                                        slate_Matrix_r64 B,
    double beta,               slate_Matrix_r64 C,
    int num_opts, slate_Options opts[]);
void slate_hermitian_band_left_multiply_c32(
    float _Complex alpha, slate_HermitianBandMatrix_c32 A,
                                        slate_Matrix_c32 B,
    float _Complex beta,               slate_Matrix_c32 C,
    int num_opts, slate_Options opts[]);
void slate_hermitian_band_left_multiply_c64(
    double _Complex alpha, slate_HermitianBandMatrix_c64 A,
                                        slate_Matrix_c64 B,
    double _Complex beta,               slate_Matrix_c64 C,
    int num_opts, slate_Options opts[]);

void slate_hermitian_band_right_multiply_r32(
    float alpha,              slate_Matrix_r32 A,
                           slate_HermitianBandMatrix_r32 B,
    float beta,               slate_Matrix_r32 C,
    int num_opts, slate_Options opts[]);
void slate_hermitian_band_right_multiply_r64(
    double alpha,              slate_Matrix_r64 A,
                           slate_HermitianBandMatrix_r64 B,
    double beta,               slate_Matrix_r64 C,
    int num_opts, slate_Options opts[]);
void slate_hermitian_band_right_multiply_c32(
    float _Complex alpha,              slate_Matrix_c32 A,
                           slate_HermitianBandMatrix_c32 B,
    float _Complex beta,               slate_Matrix_c32 C,
    int num_opts, slate_Options opts[]);
void slate_hermitian_band_right_multiply_c64(
    double _Complex alpha,              slate_Matrix_c64 A,
                           slate_HermitianBandMatrix_c64 B,
    double _Complex beta,               slate_Matrix_c64 C,
    int num_opts, slate_Options opts[]);

void slate_hermitian_left_multiply_r32(
    float alpha, slate_HermitianMatrix_r32 A,
                                    slate_Matrix_r32 B,
    float beta,           slate_Matrix_r32 C,
    int num_opts, slate_Options opts[]);
void slate_hermitian_left_multiply_r64(
    double alpha, slate_HermitianMatrix_r64 A,
                                    slate_Matrix_r64 B,
    double beta,           slate_Matrix_r64 C,
    int num_opts, slate_Options opts[]);
void slate_hermitian_left_multiply_c32(
    float _Complex alpha, slate_HermitianMatrix_c32 A,
                                    slate_Matrix_c32 B,
    float _Complex beta,           slate_Matrix_c32 C,
    int num_opts, slate_Options opts[]);
void slate_hermitian_left_multiply_c64(
    double _Complex alpha, slate_HermitianMatrix_c64 A,
                                    slate_Matrix_c64 B,
    double _Complex beta,           slate_Matrix_c64 C,
    int num_opts, slate_Options opts[]);

void slate_hermitian_right_multiply_r32(
    float alpha,          slate_Matrix_r32 A,
                           slate_HermitianMatrix_r32 B,
    float beta,           slate_Matrix_r32 C,
    int num_opts, slate_Options opts[]);
void slate_hermitian_right_multiply_r64(
    double alpha,          slate_Matrix_r64 A,
                           slate_HermitianMatrix_r64 B,
    double beta,           slate_Matrix_r64 C,
    int num_opts, slate_Options opts[]);
void slate_hermitian_right_multiply_c32(
    float _Complex alpha,          slate_Matrix_c32 A,
                           slate_HermitianMatrix_c32 B,
    float _Complex beta,           slate_Matrix_c32 C,
    int num_opts, slate_Options opts[]);
void slate_hermitian_right_multiply_c64(
    double _Complex alpha,          slate_Matrix_c64 A,
                           slate_HermitianMatrix_c64 B,
    double _Complex beta,           slate_Matrix_c64 C,
    int num_opts, slate_Options opts[]);

void slate_symmetric_left_multiply_r32(
    float alpha, slate_SymmetricMatrix_r32 A,
                                    slate_Matrix_r32 B,
    float beta,           slate_Matrix_r32 C,
    int num_opts, slate_Options opts[]);
void slate_symmetric_left_multiply_r64(
    double alpha, slate_SymmetricMatrix_r64 A,
                                    slate_Matrix_r64 B,
    double beta,           slate_Matrix_r64 C,
    int num_opts, slate_Options opts[]);
void slate_symmetric_left_multiply_c32(
    float _Complex alpha, slate_SymmetricMatrix_c32 A,
                                    slate_Matrix_c32 B,
    float _Complex beta,           slate_Matrix_c32 C,
    int num_opts, slate_Options opts[]);
void slate_symmetric_left_multiply_c64(
    double _Complex alpha, slate_SymmetricMatrix_c64 A,
                                    slate_Matrix_c64 B,
    double _Complex beta,           slate_Matrix_c64 C,
    int num_opts, slate_Options opts[]);

void slate_symmetric_right_multiply_r32(
    float alpha,          slate_Matrix_r32 A,
                           slate_SymmetricMatrix_r32 B,
    float beta,           slate_Matrix_r32 C,
    int num_opts, slate_Options opts[]);
void slate_symmetric_right_multiply_r64(
    double alpha,          slate_Matrix_r64 A,
                           slate_SymmetricMatrix_r64 B,
    double beta,           slate_Matrix_r64 C,
    int num_opts, slate_Options opts[]);
void slate_symmetric_right_multiply_c32(
    float _Complex alpha,          slate_Matrix_c32 A,
                           slate_SymmetricMatrix_c32 B,
    float _Complex beta,           slate_Matrix_c32 C,
    int num_opts, slate_Options opts[]);
void slate_symmetric_right_multiply_c64(
    double _Complex alpha,          slate_Matrix_c64 A,
                           slate_SymmetricMatrix_c64 B,
    double _Complex beta,           slate_Matrix_c64 C,
    int num_opts, slate_Options opts[]);

void slate_triangular_left_multiply_r32(
    float alpha, slate_TriangularMatrix_r32 A,
                                     slate_Matrix_r32 B,
    int num_opts, slate_Options opts[]);
void slate_triangular_left_multiply_r64(
    double alpha, slate_TriangularMatrix_r64 A,
                                     slate_Matrix_r64 B,
    int num_opts, slate_Options opts[]);
void slate_triangular_left_multiply_c32(
    float _Complex alpha, slate_TriangularMatrix_c32 A,
                                     slate_Matrix_c32 B,
    int num_opts, slate_Options opts[]);
void slate_triangular_left_multiply_c64(
    double _Complex alpha, slate_TriangularMatrix_c64 A,
                                     slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

void slate_triangular_right_multiply_r32(
    float alpha,           slate_Matrix_r32 A,
                           slate_TriangularMatrix_r32 B,
    int num_opts, slate_Options opts[]);
void slate_triangular_right_multiply_r64(
    double alpha,           slate_Matrix_r64 A,
                           slate_TriangularMatrix_r64 B,
    int num_opts, slate_Options opts[]);
void slate_triangular_right_multiply_c32(
    float _Complex alpha,           slate_Matrix_c32 A,
                           slate_TriangularMatrix_c32 B,
    int num_opts, slate_Options opts[]);
void slate_triangular_right_multiply_c64(
    double _Complex alpha,           slate_Matrix_c64 A,
                           slate_TriangularMatrix_c64 B,
    int num_opts, slate_Options opts[]);

void slate_triangular_band_left_solve_r32(
    float alpha, slate_TriangularBandMatrix_r32 A,
                                         slate_Matrix_r32 B,
    int num_opts, slate_Options opts[]);
void slate_triangular_band_left_solve_r64(
    double alpha, slate_TriangularBandMatrix_r64 A,
                                         slate_Matrix_r64 B,
    int num_opts, slate_Options opts[]);
void slate_triangular_band_left_solve_c32(
    float _Complex alpha, slate_TriangularBandMatrix_c32 A,
                                         slate_Matrix_c32 B,
    int num_opts, slate_Options opts[]);
void slate_triangular_band_left_solve_c64(
    double _Complex alpha, slate_TriangularBandMatrix_c64 A,
                                         slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

void slate_triangular_band_right_solve_r32(
    float alpha,               slate_Matrix_r32 A,
                           slate_TriangularBandMatrix_r32 B,
    int num_opts, slate_Options opts[]);
void slate_triangular_band_right_solve_r64(
    double alpha,               slate_Matrix_r64 A,
                           slate_TriangularBandMatrix_r64 B,
    int num_opts, slate_Options opts[]);
void slate_triangular_band_right_solve_c32(
    float _Complex alpha,               slate_Matrix_c32 A,
                           slate_TriangularBandMatrix_c32 B,
    int num_opts, slate_Options opts[]);
void slate_triangular_band_right_solve_c64(
    double _Complex alpha,               slate_Matrix_c64 A,
                           slate_TriangularBandMatrix_c64 B,
    int num_opts, slate_Options opts[]);

void slate_triangular_left_solve_r32(
    float alpha, slate_TriangularMatrix_r32 A,
                                     slate_Matrix_r32 B,
    int num_opts, slate_Options opts[]);
void slate_triangular_left_solve_r64(
    double alpha, slate_TriangularMatrix_r64 A,
                                     slate_Matrix_r64 B,
    int num_opts, slate_Options opts[]);
void slate_triangular_left_solve_c32(
    float _Complex alpha, slate_TriangularMatrix_c32 A,
                                     slate_Matrix_c32 B,
    int num_opts, slate_Options opts[]);
void slate_triangular_left_solve_c64(
    double _Complex alpha, slate_TriangularMatrix_c64 A,
                                     slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

void slate_triangular_right_solve_r32(
    float alpha,           slate_Matrix_r32 A,
                           slate_TriangularMatrix_r32 B,
    int num_opts, slate_Options opts[]);
void slate_triangular_right_solve_r64(
    double alpha,           slate_Matrix_r64 A,
                           slate_TriangularMatrix_r64 B,
    int num_opts, slate_Options opts[]);
void slate_triangular_right_solve_c32(
    float _Complex alpha,           slate_Matrix_c32 A,
                           slate_TriangularMatrix_c32 B,
    int num_opts, slate_Options opts[]);
void slate_triangular_right_solve_c64(
    double _Complex alpha,           slate_Matrix_c64 A,
                           slate_TriangularMatrix_c64 B,
    int num_opts, slate_Options opts[]);

void slate_hermitian_rank_k_update_r32(
    float alpha,          slate_Matrix_r32 A,
    float beta,  slate_HermitianMatrix_r32 C,
    int num_opts, slate_Options opts[]);
void slate_hermitian_rank_k_update_r64(
    double alpha,          slate_Matrix_r64 A,
    double beta,  slate_HermitianMatrix_r64 C,
    int num_opts, slate_Options opts[]);
void slate_hermitian_rank_k_update_c32(
    float alpha,          slate_Matrix_c32 A,
    float beta,  slate_HermitianMatrix_c32 C,
    int num_opts, slate_Options opts[]);
void slate_hermitian_rank_k_update_c64(
    double alpha,          slate_Matrix_c64 A,
    double beta,  slate_HermitianMatrix_c64 C,
    int num_opts, slate_Options opts[]);

void slate_symmetric_rank_k_update_r32(
    float alpha,           slate_Matrix_r32 A,
    float beta,   slate_SymmetricMatrix_r32 C,
    int num_opts, slate_Options opts[]);
void slate_symmetric_rank_k_update_r64(
    double alpha,           slate_Matrix_r64 A,
    double beta,   slate_SymmetricMatrix_r64 C,
    int num_opts, slate_Options opts[]);
void slate_symmetric_rank_k_update_c32(
    float _Complex alpha,           slate_Matrix_c32 A,
    float _Complex beta,   slate_SymmetricMatrix_c32 C,
    int num_opts, slate_Options opts[]);
void slate_symmetric_rank_k_update_c64(
    double _Complex alpha,           slate_Matrix_c64 A,
    double _Complex beta,   slate_SymmetricMatrix_c64 C,
    int num_opts, slate_Options opts[]);

void slate_hermitian_rank_2k_update_r32(
    float alpha,  slate_Matrix_r32 A,
                            slate_Matrix_r32 B,
    float beta,   slate_HermitianMatrix_r32 C,
    int num_opts, slate_Options opts[]);
void slate_hermitian_rank_2k_update_r64(
    double alpha,  slate_Matrix_r64 A,
                            slate_Matrix_r64 B,
    double beta,   slate_HermitianMatrix_r64 C,
    int num_opts, slate_Options opts[]);
void slate_hermitian_rank_2k_update_c32(
    float _Complex alpha,  slate_Matrix_c32 A,
                            slate_Matrix_c32 B,
    float beta,   slate_HermitianMatrix_c32 C,
    int num_opts, slate_Options opts[]);
void slate_hermitian_rank_2k_update_c64(
    double _Complex alpha,  slate_Matrix_c64 A,
                            slate_Matrix_c64 B,
    double beta,   slate_HermitianMatrix_c64 C,
    int num_opts, slate_Options opts[]);

void slate_symmetric_rank_2k_update_r32(
    float alpha,            slate_Matrix_r32 A,
                                      slate_Matrix_r32 B,
    float beta,    slate_SymmetricMatrix_r32 C,
    int num_opts, slate_Options opts[]);
void slate_symmetric_rank_2k_update_r64(
    double alpha,            slate_Matrix_r64 A,
                                      slate_Matrix_r64 B,
    double beta,    slate_SymmetricMatrix_r64 C,
    int num_opts, slate_Options opts[]);
void slate_symmetric_rank_2k_update_c32(
    float _Complex alpha,            slate_Matrix_c32 A,
                                      slate_Matrix_c32 B,
    float _Complex beta,    slate_SymmetricMatrix_c32 C,
    int num_opts, slate_Options opts[]);
void slate_symmetric_rank_2k_update_c64(
    double _Complex alpha,            slate_Matrix_c64 A,
                                      slate_Matrix_c64 B,
    double _Complex beta,    slate_SymmetricMatrix_c64 C,
    int num_opts, slate_Options opts[]);

void slate_band_lu_solve_r32(
    slate_BandMatrix_r32 A,
        slate_Matrix_r32 B,
    int num_opts, slate_Options opts[]);
void slate_band_lu_solve_r64(
    slate_BandMatrix_r64 A,
        slate_Matrix_r64 B,
    int num_opts, slate_Options opts[]);
void slate_band_lu_solve_c32(
    slate_BandMatrix_c32 A,
        slate_Matrix_c32 B,
    int num_opts, slate_Options opts[]);
void slate_band_lu_solve_c64(
    slate_BandMatrix_c64 A,
        slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

void slate_lu_solve_r32(
    slate_Matrix_r32 A,
    slate_Matrix_r32 B,
    int num_opts, slate_Options opts[]);
void slate_lu_solve_r64(
    slate_Matrix_r64 A,
    slate_Matrix_r64 B,
    int num_opts, slate_Options opts[]);
void slate_lu_solve_c32(
    slate_Matrix_c32 A,
    slate_Matrix_c32 B,
    int num_opts, slate_Options opts[]);
void slate_lu_solve_c64(
    slate_Matrix_c64 A,
    slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

void slate_lu_solve_nopiv_r32(
    slate_Matrix_r32 A,
    slate_Matrix_r32 B,
    int num_opts, slate_Options opts[]);
void slate_lu_solve_nopiv_r64(
    slate_Matrix_r64 A,
    slate_Matrix_r64 B,
    int num_opts, slate_Options opts[]);
void slate_lu_solve_nopiv_c32(
    slate_Matrix_c32 A,
    slate_Matrix_c32 B,
    int num_opts, slate_Options opts[]);
void slate_lu_solve_nopiv_c64(
    slate_Matrix_c64 A,
    slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

void slate_band_lu_factor_r32(
    slate_BandMatrix_r32 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[]);
void slate_band_lu_factor_r64(
    slate_BandMatrix_r64 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[]);
void slate_band_lu_factor_c32(
    slate_BandMatrix_c32 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[]);
void slate_band_lu_factor_c64(
    slate_BandMatrix_c64 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[]);

void slate_lu_factor_r32(
    slate_Matrix_r32 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[]);
void slate_lu_factor_r64(
    slate_Matrix_r64 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[]);
void slate_lu_factor_c32(
    slate_Matrix_c32 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[]);
void slate_lu_factor_c64(
    slate_Matrix_c64 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[]);

void slate_lu_factor_nopiv_r32(
    slate_Matrix_r32 A,
    int num_opts, slate_Options opts[]);
void slate_lu_factor_nopiv_r64(
    slate_Matrix_r64 A,
    int num_opts, slate_Options opts[]);
void slate_lu_factor_nopiv_c32(
    slate_Matrix_c32 A,
    int num_opts, slate_Options opts[]);
void slate_lu_factor_nopiv_c64(
    slate_Matrix_c64 A,
    int num_opts, slate_Options opts[]);

void slate_band_lu_solve_using_factor_r32(
    slate_BandMatrix_r32 A, slate_Pivots pivots,
        slate_Matrix_r32 B,
    int num_opts, slate_Options opts[]);
void slate_band_lu_solve_using_factor_r64(
    slate_BandMatrix_r64 A, slate_Pivots pivots,
        slate_Matrix_r64 B,
    int num_opts, slate_Options opts[]);
void slate_band_lu_solve_using_factor_c32(
    slate_BandMatrix_c32 A, slate_Pivots pivots,
        slate_Matrix_c32 B,
    int num_opts, slate_Options opts[]);
void slate_band_lu_solve_using_factor_c64(
    slate_BandMatrix_c64 A, slate_Pivots pivots,
        slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

void slate_lu_solve_using_factor_r32(
    slate_Matrix_r32 A, slate_Pivots pivots,
    slate_Matrix_r32 B,
    int num_opts, slate_Options opts[]);
void slate_lu_solve_using_factor_r64(
    slate_Matrix_r64 A, slate_Pivots pivots,
    slate_Matrix_r64 B,
    int num_opts, slate_Options opts[]);
void slate_lu_solve_using_factor_c32(
    slate_Matrix_c32 A, slate_Pivots pivots,
    slate_Matrix_c32 B,
    int num_opts, slate_Options opts[]);
void slate_lu_solve_using_factor_c64(
    slate_Matrix_c64 A, slate_Pivots pivots,
    slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

void slate_lu_solve_using_factor_nopiv_r32(
    slate_Matrix_r32 A,
    slate_Matrix_r32 B,
    int num_opts, slate_Options opts[]);
void slate_lu_solve_using_factor_nopiv_r64(
    slate_Matrix_r64 A,
    slate_Matrix_r64 B,
    int num_opts, slate_Options opts[]);
void slate_lu_solve_using_factor_nopiv_c32(
    slate_Matrix_c32 A,
    slate_Matrix_c32 B,
    int num_opts, slate_Options opts[]);
void slate_lu_solve_using_factor_nopiv_c64(
    slate_Matrix_c64 A,
    slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

void slate_lu_inverse_using_factor_r32(
    slate_Matrix_r32 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[]);
void slate_lu_inverse_using_factor_r64(
    slate_Matrix_r64 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[]);
void slate_lu_inverse_using_factor_c32(
    slate_Matrix_c32 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[]);
void slate_lu_inverse_using_factor_c64(
    slate_Matrix_c64 A, slate_Pivots pivots,
    int num_opts, slate_Options opts[]);

void slate_lu_inverse_using_factor_out_of_place_r32(
    slate_Matrix_r32 A, slate_Pivots pivots,
    slate_Matrix_r32 A_inverse,
    int num_opts, slate_Options opts[]);
void slate_lu_inverse_using_factor_out_of_place_r64(
    slate_Matrix_r64 A, slate_Pivots pivots,
    slate_Matrix_r64 A_inverse,
    int num_opts, slate_Options opts[]);
void slate_lu_inverse_using_factor_out_of_place_c32(
    slate_Matrix_c32 A, slate_Pivots pivots,
    slate_Matrix_c32 A_inverse,
    int num_opts, slate_Options opts[]);
void slate_lu_inverse_using_factor_out_of_place_c64(
    slate_Matrix_c64 A, slate_Pivots pivots,
    slate_Matrix_c64 A_inverse,
    int num_opts, slate_Options opts[]);

void slate_band_chol_solve_r32(
    slate_HermitianBandMatrix_r32 A,
                 slate_Matrix_r32 B,
    int num_opts, slate_Options opts[]);
void slate_band_chol_solve_r64(
    slate_HermitianBandMatrix_r64 A,
                 slate_Matrix_r64 B,
    int num_opts, slate_Options opts[]);
void slate_band_chol_solve_c32(
    slate_HermitianBandMatrix_c32 A,
                 slate_Matrix_c32 B,
    int num_opts, slate_Options opts[]);
void slate_band_chol_solve_c64(
    slate_HermitianBandMatrix_c64 A,
                 slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

void slate_chol_solve_r32(
    slate_HermitianMatrix_r32 A,
             slate_Matrix_r32 B,
    int num_opts, slate_Options opts[]);
void slate_chol_solve_r64(
    slate_HermitianMatrix_r64 A,
             slate_Matrix_r64 B,
    int num_opts, slate_Options opts[]);
void slate_chol_solve_c32(
    slate_HermitianMatrix_c32 A,
             slate_Matrix_c32 B,
    int num_opts, slate_Options opts[]);
void slate_chol_solve_c64(
    slate_HermitianMatrix_c64 A,
             slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

void slate_band_chol_factor_r32(
    slate_HermitianBandMatrix_r32 A,
    int num_opts, slate_Options opts[]);
void slate_band_chol_factor_r64(
    slate_HermitianBandMatrix_r64 A,
    int num_opts, slate_Options opts[]);
void slate_band_chol_factor_c32(
    slate_HermitianBandMatrix_c32 A,
    int num_opts, slate_Options opts[]);
void slate_band_chol_factor_c64(
    slate_HermitianBandMatrix_c64 A,
    int num_opts, slate_Options opts[]);

void slate_chol_factor_r32(
    slate_HermitianMatrix_r32 A,
    int num_opts, slate_Options opts[]);
void slate_chol_factor_r64(
    slate_HermitianMatrix_r64 A,
    int num_opts, slate_Options opts[]);
void slate_chol_factor_c32(
    slate_HermitianMatrix_c32 A,
    int num_opts, slate_Options opts[]);
void slate_chol_factor_c64(
    slate_HermitianMatrix_c64 A,
    int num_opts, slate_Options opts[]);

void slate_band_chol_solve_using_factor_r32(
    slate_HermitianBandMatrix_r32 A,
                 slate_Matrix_r32 B,
    int num_opts, slate_Options opts[]);
void slate_band_chol_solve_using_factor_r64(
    slate_HermitianBandMatrix_r64 A,
                 slate_Matrix_r64 B,
    int num_opts, slate_Options opts[]);
void slate_band_chol_solve_using_factor_c32(
    slate_HermitianBandMatrix_c32 A,
                 slate_Matrix_c32 B,
    int num_opts, slate_Options opts[]);
void slate_band_chol_solve_using_factor_c64(
    slate_HermitianBandMatrix_c64 A,
                 slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

void slate_chol_solve_using_factor_r32(
    slate_HermitianMatrix_r32 A,
             slate_Matrix_r32 B,
    int num_opts, slate_Options opts[]);
void slate_chol_solve_using_factor_r64(
    slate_HermitianMatrix_r64 A,
             slate_Matrix_r64 B,
    int num_opts, slate_Options opts[]);
void slate_chol_solve_using_factor_c32(
    slate_HermitianMatrix_c32 A,
             slate_Matrix_c32 B,
    int num_opts, slate_Options opts[]);
void slate_chol_solve_using_factor_c64(
    slate_HermitianMatrix_c64 A,
             slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

void slate_chol_inverse_using_factor_r32(
    slate_HermitianMatrix_r32 A,
    int num_opts, slate_Options opts[]);
void slate_chol_inverse_using_factor_r64(
    slate_HermitianMatrix_r64 A,
    int num_opts, slate_Options opts[]);
void slate_chol_inverse_using_factor_c32(
    slate_HermitianMatrix_c32 A,
    int num_opts, slate_Options opts[]);
void slate_chol_inverse_using_factor_c64(
    slate_HermitianMatrix_c64 A,
    int num_opts, slate_Options opts[]);

void slate_indefinite_solve_r32(
    slate_HermitianMatrix_r32 A,
             slate_Matrix_r32 B,
    int num_opts, slate_Options opts[]);
void slate_indefinite_solve_r64(
    slate_HermitianMatrix_r64 A,
             slate_Matrix_r64 B,
    int num_opts, slate_Options opts[]);
void slate_indefinite_solve_c32(
    slate_HermitianMatrix_c32 A,
             slate_Matrix_c32 B,
    int num_opts, slate_Options opts[]);
void slate_indefinite_solve_c64(
    slate_HermitianMatrix_c64 A,
             slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

void slate_indefinite_factor_r32(
    slate_HermitianMatrix_r32 A, slate_Pivots pivots,
         slate_BandMatrix_r32 T, slate_Pivots pivots2,
             slate_Matrix_r32 H,
    int num_opts, slate_Options opts[]);
void slate_indefinite_factor_r64(
    slate_HermitianMatrix_r64 A, slate_Pivots pivots,
         slate_BandMatrix_r64 T, slate_Pivots pivots2,
             slate_Matrix_r64 H,
    int num_opts, slate_Options opts[]);
void slate_indefinite_factor_c32(
    slate_HermitianMatrix_c32 A, slate_Pivots pivots,
         slate_BandMatrix_c32 T, slate_Pivots pivots2,
             slate_Matrix_c32 H,
    int num_opts, slate_Options opts[]);
void slate_indefinite_factor_c64(
    slate_HermitianMatrix_c64 A, slate_Pivots pivots,
         slate_BandMatrix_c64 T, slate_Pivots pivots2,
             slate_Matrix_c64 H,
    int num_opts, slate_Options opts[]);

void slate_indefinite_solve_using_factor_r32(
    slate_HermitianMatrix_r32 A, slate_Pivots pivots,
         slate_BandMatrix_r32 T, slate_Pivots pivots2,
             slate_Matrix_r32 B,
    int num_opts, slate_Options opts[]);
void slate_indefinite_solve_using_factor_r64(
    slate_HermitianMatrix_r64 A, slate_Pivots pivots,
         slate_BandMatrix_r64 T, slate_Pivots pivots2,
             slate_Matrix_r64 B,
    int num_opts, slate_Options opts[]);
void slate_indefinite_solve_using_factor_c32(
    slate_HermitianMatrix_c32 A, slate_Pivots pivots,
         slate_BandMatrix_c32 T, slate_Pivots pivots2,
             slate_Matrix_c32 B,
    int num_opts, slate_Options opts[]);
void slate_indefinite_solve_using_factor_c64(
    slate_HermitianMatrix_c64 A, slate_Pivots pivots,
         slate_BandMatrix_c64 T, slate_Pivots pivots2,
             slate_Matrix_c64 B,
    int num_opts, slate_Options opts[]);

void slate_least_squares_solve_r32(
    slate_Matrix_r32 A,
    slate_Matrix_r32 BX,
    int num_opts, slate_Options opts[]);
void slate_least_squares_solve_r64(
    slate_Matrix_r64 A,
    slate_Matrix_r64 BX,
    int num_opts, slate_Options opts[]);
void slate_least_squares_solve_c32(
    slate_Matrix_c32 A,
    slate_Matrix_c32 BX,
    int num_opts, slate_Options opts[]);
void slate_least_squares_solve_c64(
    slate_Matrix_c64 A,
    slate_Matrix_c64 BX,
    int num_opts, slate_Options opts[]);

void slate_qr_factor_r32(
    slate_Matrix_r32 A, slate_TriangularFactors_r32 T,
    int num_opts, slate_Options opts[]);
void slate_qr_factor_r64(
    slate_Matrix_r64 A, slate_TriangularFactors_r64 T,
    int num_opts, slate_Options opts[]);
void slate_qr_factor_c32(
    slate_Matrix_c32 A, slate_TriangularFactors_c32 T,
    int num_opts, slate_Options opts[]);
void slate_qr_factor_c64(
    slate_Matrix_c64 A, slate_TriangularFactors_c64 T,
    int num_opts, slate_Options opts[]);

void slate_qr_multiply_by_q_r32(
    slate_Side side, slate_Op op,
    slate_Matrix_r32 A, slate_TriangularFactors_r32 T,
    slate_Matrix_r32 C,
    int num_opts, slate_Options opts[]);
void slate_qr_multiply_by_q_r64(
    slate_Side side, slate_Op op,
    slate_Matrix_r64 A, slate_TriangularFactors_r64 T,
    slate_Matrix_r64 C,
    int num_opts, slate_Options opts[]);
void slate_qr_multiply_by_q_c32(
    slate_Side side, slate_Op op,
    slate_Matrix_c32 A, slate_TriangularFactors_c32 T,
    slate_Matrix_c32 C,
    int num_opts, slate_Options opts[]);
void slate_qr_multiply_by_q_c64(
    slate_Side side, slate_Op op,
    slate_Matrix_c64 A, slate_TriangularFactors_c64 T,
    slate_Matrix_c64 C,
    int num_opts, slate_Options opts[]);

void slate_lq_factor_r32(
    slate_Matrix_r32 A, slate_TriangularFactors_r32 T,
    int num_opts, slate_Options opts[]);
void slate_lq_factor_r64(
    slate_Matrix_r64 A, slate_TriangularFactors_r64 T,
    int num_opts, slate_Options opts[]);
void slate_lq_factor_c32(
    slate_Matrix_c32 A, slate_TriangularFactors_c32 T,
    int num_opts, slate_Options opts[]);
void slate_lq_factor_c64(
    slate_Matrix_c64 A, slate_TriangularFactors_c64 T,
    int num_opts, slate_Options opts[]);

void slate_lq_multiply_by_q_r32(
    slate_Side side, slate_Op op,
    slate_Matrix_r32 A, slate_TriangularFactors_r32 T,
    slate_Matrix_r32 C,
    int num_opts, slate_Options opts[]);
void slate_lq_multiply_by_q_r64(
    slate_Side side, slate_Op op,
    slate_Matrix_r64 A, slate_TriangularFactors_r64 T,
    slate_Matrix_r64 C,
    int num_opts, slate_Options opts[]);
void slate_lq_multiply_by_q_c32(
    slate_Side side, slate_Op op,
    slate_Matrix_c32 A, slate_TriangularFactors_c32 T,
    slate_Matrix_c32 C,
    int num_opts, slate_Options opts[]);
void slate_lq_multiply_by_q_c64(
    slate_Side side, slate_Op op,
    slate_Matrix_c64 A, slate_TriangularFactors_c64 T,
    slate_Matrix_c64 C,
    int num_opts, slate_Options opts[]);

void slate_svd_vals_r32(
    slate_Matrix_r32 A,
    float* Sigma,
    int num_opts, slate_Options opts[]);
void slate_svd_vals_r64(
    slate_Matrix_r64 A,
    double* Sigma,
    int num_opts, slate_Options opts[]);
void slate_svd_vals_c32(
    slate_Matrix_c32 A,
    float* Sigma,
    int num_opts, slate_Options opts[]);
void slate_svd_vals_c64(
    slate_Matrix_c64 A,
    double* Sigma,
    int num_opts, slate_Options opts[]);

void slate_hermitian_eig_vals_r32(
    slate_HermitianMatrix_r32 A,
    float* Lambda,
    int num_opts, slate_Options opts[]);
void slate_hermitian_eig_vals_r64(
    slate_HermitianMatrix_r64 A,
    double* Lambda,
    int num_opts, slate_Options opts[]);
void slate_hermitian_eig_vals_c32(
    slate_HermitianMatrix_c32 A,
    float* Lambda,
    int num_opts, slate_Options opts[]);
void slate_hermitian_eig_vals_c64(
    slate_HermitianMatrix_c64 A,
    double* Lambda,
    int num_opts, slate_Options opts[]);

void slate_generaized_hermitian_eig_vals_r32(
    int64_t itype,
    slate_HermitianMatrix_r32 A,
    slate_HermitianMatrix_r32 B,
    float* Lambda,
    int num_opts, slate_Options opts[]);
void slate_generaized_hermitian_eig_vals_r64(
    int64_t itype,
    slate_HermitianMatrix_r64 A,
    slate_HermitianMatrix_r64 B,
    double* Lambda,
    int num_opts, slate_Options opts[]);
void slate_generaized_hermitian_eig_vals_c32(
    int64_t itype,
    slate_HermitianMatrix_c32 A,
    slate_HermitianMatrix_c32 B,
    float* Lambda,
    int num_opts, slate_Options opts[]);
void slate_generaized_hermitian_eig_vals_c64(
    int64_t itype,
    slate_HermitianMatrix_c64 A,
    slate_HermitianMatrix_c64 B,
    double* Lambda,
    int num_opts, slate_Options opts[]);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // SLATE_C_API_WRAPPERS_H