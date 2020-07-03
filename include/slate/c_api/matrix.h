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
// Auto-generated file by tools/c_api/generate_matrix.py

#ifndef SLATE_C_API_MATRIX_H
#define SLATE_C_API_MATRIX_H

#include "slate/internal/mpi.hh"
#include "slate/c_api/types.h"

#include <complex.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// instantiate Tile for scalar_t = <float>
typedef struct
{
    int64_t mb_;
    int64_t nb_;
    int64_t stride_;
    int64_t user_stride_; // Temporarily store user-provided-memory's stride
    slate_Op op_;
    slate_Uplo uplo_;
    float* data_;
    float* user_data_; // Temporarily point to user-provided memory buffer.
    float* ext_data_; // Points to auxiliary buffer.
    slate_TileKind kind_;
    /// layout_: The physical ordering of elements in the data buffer:
    ///          - ColMajor: elements of a column are 1-strided
    ///          - RowMajor: elements of a row are 1-strided
    slate_Layout layout_;
    slate_Layout user_layout_; // Temporarily store user-provided-memory's layout
    int device_;
} slate_Tile_r32;

/// slate::Tile<float>::mb()
int64_t slate_Tile_mb_r32(slate_Tile_r32 T);

/// slate::Tile<float>::nb()
int64_t slate_Tile_nb_r32(slate_Tile_r32 T);

/// slate::Tile<float>::stride()
int64_t slate_Tile_stride_r32(slate_Tile_r32 T);

/// slate::Tile<float>::data()
float* slate_Tile_data_r32(slate_Tile_r32 T);

//------------------------------------------------------------------------------
// instantiate Tile for scalar_t = <double>
typedef struct
{
    int64_t mb_;
    int64_t nb_;
    int64_t stride_;
    int64_t user_stride_; // Temporarily store user-provided-memory's stride
    slate_Op op_;
    slate_Uplo uplo_;
    double* data_;
    double* user_data_; // Temporarily point to user-provided memory buffer.
    double* ext_data_; // Points to auxiliary buffer.
    slate_TileKind kind_;
    /// layout_: The physical ordering of elements in the data buffer:
    ///          - ColMajor: elements of a column are 1-strided
    ///          - RowMajor: elements of a row are 1-strided
    slate_Layout layout_;
    slate_Layout user_layout_; // Temporarily store user-provided-memory's layout
    int device_;
} slate_Tile_r64;

/// slate::Tile<double>::mb()
int64_t slate_Tile_mb_r64(slate_Tile_r64 T);

/// slate::Tile<double>::nb()
int64_t slate_Tile_nb_r64(slate_Tile_r64 T);

/// slate::Tile<double>::stride()
int64_t slate_Tile_stride_r64(slate_Tile_r64 T);

/// slate::Tile<double>::data()
double* slate_Tile_data_r64(slate_Tile_r64 T);

//------------------------------------------------------------------------------
// instantiate Tile for scalar_t = <float _Complex>
typedef struct
{
    int64_t mb_;
    int64_t nb_;
    int64_t stride_;
    int64_t user_stride_; // Temporarily store user-provided-memory's stride
    slate_Op op_;
    slate_Uplo uplo_;
    float _Complex* data_;
    float _Complex* user_data_; // Temporarily point to user-provided memory buffer.
    float _Complex* ext_data_; // Points to auxiliary buffer.
    slate_TileKind kind_;
    /// layout_: The physical ordering of elements in the data buffer:
    ///          - ColMajor: elements of a column are 1-strided
    ///          - RowMajor: elements of a row are 1-strided
    slate_Layout layout_;
    slate_Layout user_layout_; // Temporarily store user-provided-memory's layout
    int device_;
} slate_Tile_c32;

/// slate::Tile<std::complex<float>>::mb()
int64_t slate_Tile_mb_c32(slate_Tile_c32 T);

/// slate::Tile<std::complex<float>>::nb()
int64_t slate_Tile_nb_c32(slate_Tile_c32 T);

/// slate::Tile<std::complex<float>>::stride()
int64_t slate_Tile_stride_c32(slate_Tile_c32 T);

/// slate::Tile<std::complex<float>>::data()
float _Complex* slate_Tile_data_c32(slate_Tile_c32 T);

//------------------------------------------------------------------------------
// instantiate Tile for scalar_t = <double _Complex>
typedef struct
{
    int64_t mb_;
    int64_t nb_;
    int64_t stride_;
    int64_t user_stride_; // Temporarily store user-provided-memory's stride
    slate_Op op_;
    slate_Uplo uplo_;
    double _Complex* data_;
    double _Complex* user_data_; // Temporarily point to user-provided memory buffer.
    double _Complex* ext_data_; // Points to auxiliary buffer.
    slate_TileKind kind_;
    /// layout_: The physical ordering of elements in the data buffer:
    ///          - ColMajor: elements of a column are 1-strided
    ///          - RowMajor: elements of a row are 1-strided
    slate_Layout layout_;
    slate_Layout user_layout_; // Temporarily store user-provided-memory's layout
    int device_;
} slate_Tile_c64;

/// slate::Tile<std::complex<double>>::mb()
int64_t slate_Tile_mb_c64(slate_Tile_c64 T);

/// slate::Tile<std::complex<double>>::nb()
int64_t slate_Tile_nb_c64(slate_Tile_c64 T);

/// slate::Tile<std::complex<double>>::stride()
int64_t slate_Tile_stride_c64(slate_Tile_c64 T);

/// slate::Tile<std::complex<double>>::data()
double _Complex* slate_Tile_data_c64(slate_Tile_c64 T);

//------------------------------------------------------------------------------
/// slate::Matrix<float>
struct slate_Matrix_struct_r32;
typedef struct slate_Matrix_struct_r32* slate_Matrix_r32;

slate_Matrix_r32 slate_Matrix_create_r32(int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_Matrix_r32 slate_Matrix_create_fromScaLAPACK_r32(int64_t m, int64_t n, float* A, int64_t lda, int64_t mb, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_Matrix_r32 slate_Matrix_create_slice_r32(slate_Matrix_r32 A, int64_t i1, int64_t i2, int64_t j1, int64_t j2);

void slate_Matrix_destroy_r32(slate_Matrix_r32 A);

void slate_Matrix_insertLocalTiles_r32(slate_Matrix_r32 A);

int64_t slate_Matrix_mt_r32(slate_Matrix_r32 A);

int64_t slate_Matrix_nt_r32(slate_Matrix_r32 A);

int64_t slate_Matrix_m_r32(slate_Matrix_r32 A);

int64_t slate_Matrix_n_r32(slate_Matrix_r32 A);

bool slate_Matrix_tileIsLocal_r32(slate_Matrix_r32 A, int64_t i, int64_t j);

slate_Tile_r32 slate_Matrix_at_r32(slate_Matrix_r32 A, int64_t i, int64_t j);

void slate_Matrix_transpose_in_place_r32(slate_Matrix_r32 A);

void slate_Matrix_conjTranspose_in_place_r32(slate_Matrix_r32 A);

/// slate::Matrix<double>
struct slate_Matrix_struct_r64;
typedef struct slate_Matrix_struct_r64* slate_Matrix_r64;

slate_Matrix_r64 slate_Matrix_create_r64(int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_Matrix_r64 slate_Matrix_create_fromScaLAPACK_r64(int64_t m, int64_t n, double* A, int64_t lda, int64_t mb, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_Matrix_r64 slate_Matrix_create_slice_r64(slate_Matrix_r64 A, int64_t i1, int64_t i2, int64_t j1, int64_t j2);

void slate_Matrix_destroy_r64(slate_Matrix_r64 A);

void slate_Matrix_insertLocalTiles_r64(slate_Matrix_r64 A);

int64_t slate_Matrix_mt_r64(slate_Matrix_r64 A);

int64_t slate_Matrix_nt_r64(slate_Matrix_r64 A);

int64_t slate_Matrix_m_r64(slate_Matrix_r64 A);

int64_t slate_Matrix_n_r64(slate_Matrix_r64 A);

bool slate_Matrix_tileIsLocal_r64(slate_Matrix_r64 A, int64_t i, int64_t j);

slate_Tile_r64 slate_Matrix_at_r64(slate_Matrix_r64 A, int64_t i, int64_t j);

void slate_Matrix_transpose_in_place_r64(slate_Matrix_r64 A);

void slate_Matrix_conjTranspose_in_place_r64(slate_Matrix_r64 A);

/// slate::Matrix<std::complex<float>>
struct slate_Matrix_struct_c32;
typedef struct slate_Matrix_struct_c32* slate_Matrix_c32;

slate_Matrix_c32 slate_Matrix_create_c32(int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_Matrix_c32 slate_Matrix_create_fromScaLAPACK_c32(int64_t m, int64_t n, float _Complex* A, int64_t lda, int64_t mb, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_Matrix_c32 slate_Matrix_create_slice_c32(slate_Matrix_c32 A, int64_t i1, int64_t i2, int64_t j1, int64_t j2);

void slate_Matrix_destroy_c32(slate_Matrix_c32 A);

void slate_Matrix_insertLocalTiles_c32(slate_Matrix_c32 A);

int64_t slate_Matrix_mt_c32(slate_Matrix_c32 A);

int64_t slate_Matrix_nt_c32(slate_Matrix_c32 A);

int64_t slate_Matrix_m_c32(slate_Matrix_c32 A);

int64_t slate_Matrix_n_c32(slate_Matrix_c32 A);

bool slate_Matrix_tileIsLocal_c32(slate_Matrix_c32 A, int64_t i, int64_t j);

slate_Tile_c32 slate_Matrix_at_c32(slate_Matrix_c32 A, int64_t i, int64_t j);

void slate_Matrix_transpose_in_place_c32(slate_Matrix_c32 A);

void slate_Matrix_conjTranspose_in_place_c32(slate_Matrix_c32 A);

/// slate::Matrix<std::complex<double>>
struct slate_Matrix_struct_c64;
typedef struct slate_Matrix_struct_c64* slate_Matrix_c64;

slate_Matrix_c64 slate_Matrix_create_c64(int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_Matrix_c64 slate_Matrix_create_fromScaLAPACK_c64(int64_t m, int64_t n, double _Complex* A, int64_t lda, int64_t mb, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_Matrix_c64 slate_Matrix_create_slice_c64(slate_Matrix_c64 A, int64_t i1, int64_t i2, int64_t j1, int64_t j2);

void slate_Matrix_destroy_c64(slate_Matrix_c64 A);

void slate_Matrix_insertLocalTiles_c64(slate_Matrix_c64 A);

int64_t slate_Matrix_mt_c64(slate_Matrix_c64 A);

int64_t slate_Matrix_nt_c64(slate_Matrix_c64 A);

int64_t slate_Matrix_m_c64(slate_Matrix_c64 A);

int64_t slate_Matrix_n_c64(slate_Matrix_c64 A);

bool slate_Matrix_tileIsLocal_c64(slate_Matrix_c64 A, int64_t i, int64_t j);

slate_Tile_c64 slate_Matrix_at_c64(slate_Matrix_c64 A, int64_t i, int64_t j);

void slate_Matrix_transpose_in_place_c64(slate_Matrix_c64 A);

void slate_Matrix_conjTranspose_in_place_c64(slate_Matrix_c64 A);

//------------------------------------------------------------------------------
/// slate::BandMatrix<float>
struct slate_BandMatrix_struct_r32;
typedef struct slate_BandMatrix_struct_r32* slate_BandMatrix_r32;

slate_BandMatrix_r32 slate_BandMatrix_create_r32(int64_t m, int64_t n, int64_t kl, int64_t ku, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_BandMatrix_destroy_r32(slate_BandMatrix_r32 A);

int64_t slate_BandMatrix_mt_r32(slate_BandMatrix_r32 A);

int64_t slate_BandMatrix_nt_r32(slate_BandMatrix_r32 A);

int64_t slate_BandMatrix_m_r32(slate_BandMatrix_r32 A);

int64_t slate_BandMatrix_n_r32(slate_BandMatrix_r32 A);

bool slate_BandMatrix_tileIsLocal_r32(slate_BandMatrix_r32 A, int64_t i, int64_t j);

slate_Tile_r32 slate_BandMatrix_at_r32(slate_BandMatrix_r32 A, int64_t i, int64_t j);

void slate_BandMatrix_transpose_in_place_r32(slate_BandMatrix_r32 A);

void slate_BandMatrix_conjTranspose_in_place_r32(slate_BandMatrix_r32 A);

/// slate::BandMatrix<double>
struct slate_BandMatrix_struct_r64;
typedef struct slate_BandMatrix_struct_r64* slate_BandMatrix_r64;

slate_BandMatrix_r64 slate_BandMatrix_create_r64(int64_t m, int64_t n, int64_t kl, int64_t ku, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_BandMatrix_destroy_r64(slate_BandMatrix_r64 A);

int64_t slate_BandMatrix_mt_r64(slate_BandMatrix_r64 A);

int64_t slate_BandMatrix_nt_r64(slate_BandMatrix_r64 A);

int64_t slate_BandMatrix_m_r64(slate_BandMatrix_r64 A);

int64_t slate_BandMatrix_n_r64(slate_BandMatrix_r64 A);

bool slate_BandMatrix_tileIsLocal_r64(slate_BandMatrix_r64 A, int64_t i, int64_t j);

slate_Tile_r64 slate_BandMatrix_at_r64(slate_BandMatrix_r64 A, int64_t i, int64_t j);

void slate_BandMatrix_transpose_in_place_r64(slate_BandMatrix_r64 A);

void slate_BandMatrix_conjTranspose_in_place_r64(slate_BandMatrix_r64 A);

/// slate::BandMatrix<std::complex<float>>
struct slate_BandMatrix_struct_c32;
typedef struct slate_BandMatrix_struct_c32* slate_BandMatrix_c32;

slate_BandMatrix_c32 slate_BandMatrix_create_c32(int64_t m, int64_t n, int64_t kl, int64_t ku, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_BandMatrix_destroy_c32(slate_BandMatrix_c32 A);

int64_t slate_BandMatrix_mt_c32(slate_BandMatrix_c32 A);

int64_t slate_BandMatrix_nt_c32(slate_BandMatrix_c32 A);

int64_t slate_BandMatrix_m_c32(slate_BandMatrix_c32 A);

int64_t slate_BandMatrix_n_c32(slate_BandMatrix_c32 A);

bool slate_BandMatrix_tileIsLocal_c32(slate_BandMatrix_c32 A, int64_t i, int64_t j);

slate_Tile_c32 slate_BandMatrix_at_c32(slate_BandMatrix_c32 A, int64_t i, int64_t j);

void slate_BandMatrix_transpose_in_place_c32(slate_BandMatrix_c32 A);

void slate_BandMatrix_conjTranspose_in_place_c32(slate_BandMatrix_c32 A);

/// slate::BandMatrix<std::complex<double>>
struct slate_BandMatrix_struct_c64;
typedef struct slate_BandMatrix_struct_c64* slate_BandMatrix_c64;

slate_BandMatrix_c64 slate_BandMatrix_create_c64(int64_t m, int64_t n, int64_t kl, int64_t ku, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_BandMatrix_destroy_c64(slate_BandMatrix_c64 A);

int64_t slate_BandMatrix_mt_c64(slate_BandMatrix_c64 A);

int64_t slate_BandMatrix_nt_c64(slate_BandMatrix_c64 A);

int64_t slate_BandMatrix_m_c64(slate_BandMatrix_c64 A);

int64_t slate_BandMatrix_n_c64(slate_BandMatrix_c64 A);

bool slate_BandMatrix_tileIsLocal_c64(slate_BandMatrix_c64 A, int64_t i, int64_t j);

slate_Tile_c64 slate_BandMatrix_at_c64(slate_BandMatrix_c64 A, int64_t i, int64_t j);

void slate_BandMatrix_transpose_in_place_c64(slate_BandMatrix_c64 A);

void slate_BandMatrix_conjTranspose_in_place_c64(slate_BandMatrix_c64 A);

//------------------------------------------------------------------------------
/// slate::HermitianMatrix<float>
struct slate_HermitianMatrix_struct_r32;
typedef struct slate_HermitianMatrix_struct_r32* slate_HermitianMatrix_r32;

slate_HermitianMatrix_r32 slate_HermitianMatrix_create_r32(slate_Uplo uplo, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_HermitianMatrix_r32 slate_HermitianMatrix_create_fromScaLAPACK_r32(slate_Uplo uplo, int64_t n, float* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_HermitianMatrix_destroy_r32(slate_HermitianMatrix_r32 A);

void slate_HermitianMatrix_insertLocalTiles_r32(slate_HermitianMatrix_r32 A);

int64_t slate_HermitianMatrix_mt_r32(slate_HermitianMatrix_r32 A);

int64_t slate_HermitianMatrix_nt_r32(slate_HermitianMatrix_r32 A);

int64_t slate_HermitianMatrix_m_r32(slate_HermitianMatrix_r32 A);

int64_t slate_HermitianMatrix_n_r32(slate_HermitianMatrix_r32 A);

bool slate_HermitianMatrix_tileIsLocal_r32(slate_HermitianMatrix_r32 A, int64_t i, int64_t j);

slate_Tile_r32 slate_HermitianMatrix_at_r32(slate_HermitianMatrix_r32 A, int64_t i, int64_t j);

void slate_HermitianMatrix_transpose_in_place_r32(slate_HermitianMatrix_r32 A);

void slate_HermitianMatrix_conjTranspose_in_place_r32(slate_HermitianMatrix_r32 A);

/// slate::HermitianMatrix<double>
struct slate_HermitianMatrix_struct_r64;
typedef struct slate_HermitianMatrix_struct_r64* slate_HermitianMatrix_r64;

slate_HermitianMatrix_r64 slate_HermitianMatrix_create_r64(slate_Uplo uplo, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_HermitianMatrix_r64 slate_HermitianMatrix_create_fromScaLAPACK_r64(slate_Uplo uplo, int64_t n, double* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_HermitianMatrix_destroy_r64(slate_HermitianMatrix_r64 A);

void slate_HermitianMatrix_insertLocalTiles_r64(slate_HermitianMatrix_r64 A);

int64_t slate_HermitianMatrix_mt_r64(slate_HermitianMatrix_r64 A);

int64_t slate_HermitianMatrix_nt_r64(slate_HermitianMatrix_r64 A);

int64_t slate_HermitianMatrix_m_r64(slate_HermitianMatrix_r64 A);

int64_t slate_HermitianMatrix_n_r64(slate_HermitianMatrix_r64 A);

bool slate_HermitianMatrix_tileIsLocal_r64(slate_HermitianMatrix_r64 A, int64_t i, int64_t j);

slate_Tile_r64 slate_HermitianMatrix_at_r64(slate_HermitianMatrix_r64 A, int64_t i, int64_t j);

void slate_HermitianMatrix_transpose_in_place_r64(slate_HermitianMatrix_r64 A);

void slate_HermitianMatrix_conjTranspose_in_place_r64(slate_HermitianMatrix_r64 A);

/// slate::HermitianMatrix<std::complex<float>>
struct slate_HermitianMatrix_struct_c32;
typedef struct slate_HermitianMatrix_struct_c32* slate_HermitianMatrix_c32;

slate_HermitianMatrix_c32 slate_HermitianMatrix_create_c32(slate_Uplo uplo, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_HermitianMatrix_c32 slate_HermitianMatrix_create_fromScaLAPACK_c32(slate_Uplo uplo, int64_t n, float _Complex* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_HermitianMatrix_destroy_c32(slate_HermitianMatrix_c32 A);

void slate_HermitianMatrix_insertLocalTiles_c32(slate_HermitianMatrix_c32 A);

int64_t slate_HermitianMatrix_mt_c32(slate_HermitianMatrix_c32 A);

int64_t slate_HermitianMatrix_nt_c32(slate_HermitianMatrix_c32 A);

int64_t slate_HermitianMatrix_m_c32(slate_HermitianMatrix_c32 A);

int64_t slate_HermitianMatrix_n_c32(slate_HermitianMatrix_c32 A);

bool slate_HermitianMatrix_tileIsLocal_c32(slate_HermitianMatrix_c32 A, int64_t i, int64_t j);

slate_Tile_c32 slate_HermitianMatrix_at_c32(slate_HermitianMatrix_c32 A, int64_t i, int64_t j);

void slate_HermitianMatrix_transpose_in_place_c32(slate_HermitianMatrix_c32 A);

void slate_HermitianMatrix_conjTranspose_in_place_c32(slate_HermitianMatrix_c32 A);

/// slate::HermitianMatrix<std::complex<double>>
struct slate_HermitianMatrix_struct_c64;
typedef struct slate_HermitianMatrix_struct_c64* slate_HermitianMatrix_c64;

slate_HermitianMatrix_c64 slate_HermitianMatrix_create_c64(slate_Uplo uplo, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_HermitianMatrix_c64 slate_HermitianMatrix_create_fromScaLAPACK_c64(slate_Uplo uplo, int64_t n, double _Complex* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_HermitianMatrix_destroy_c64(slate_HermitianMatrix_c64 A);

void slate_HermitianMatrix_insertLocalTiles_c64(slate_HermitianMatrix_c64 A);

int64_t slate_HermitianMatrix_mt_c64(slate_HermitianMatrix_c64 A);

int64_t slate_HermitianMatrix_nt_c64(slate_HermitianMatrix_c64 A);

int64_t slate_HermitianMatrix_m_c64(slate_HermitianMatrix_c64 A);

int64_t slate_HermitianMatrix_n_c64(slate_HermitianMatrix_c64 A);

bool slate_HermitianMatrix_tileIsLocal_c64(slate_HermitianMatrix_c64 A, int64_t i, int64_t j);

slate_Tile_c64 slate_HermitianMatrix_at_c64(slate_HermitianMatrix_c64 A, int64_t i, int64_t j);

void slate_HermitianMatrix_transpose_in_place_c64(slate_HermitianMatrix_c64 A);

void slate_HermitianMatrix_conjTranspose_in_place_c64(slate_HermitianMatrix_c64 A);

//------------------------------------------------------------------------------
/// slate::HermitianBandMatrix<float>
struct slate_HermitianBandMatrix_struct_r32;
typedef struct slate_HermitianBandMatrix_struct_r32* slate_HermitianBandMatrix_r32;

slate_HermitianBandMatrix_r32 slate_HermitianBandMatrix_create_r32(slate_Uplo uplo, int64_t n, int64_t kd, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_HermitianBandMatrix_destroy_r32(slate_HermitianBandMatrix_r32 A);

void slate_HermitianBandMatrix_insertLocalTiles_r32(slate_HermitianBandMatrix_r32 A);

int64_t slate_HermitianBandMatrix_mt_r32(slate_HermitianBandMatrix_r32 A);

int64_t slate_HermitianBandMatrix_nt_r32(slate_HermitianBandMatrix_r32 A);

int64_t slate_HermitianBandMatrix_m_r32(slate_HermitianBandMatrix_r32 A);

int64_t slate_HermitianBandMatrix_n_r32(slate_HermitianBandMatrix_r32 A);

bool slate_HermitianBandMatrix_tileIsLocal_r32(slate_HermitianBandMatrix_r32 A, int64_t i, int64_t j);

slate_Tile_r32 slate_HermitianBandMatrix_at_r32(slate_HermitianBandMatrix_r32 A, int64_t i, int64_t j);

void slate_HermitianBandMatrix_transpose_in_place_r32(slate_HermitianBandMatrix_r32 A);

void slate_HermitianBandMatrix_conjTranspose_in_place_r32(slate_HermitianBandMatrix_r32 A);

/// slate::HermitianBandMatrix<double>
struct slate_HermitianBandMatrix_struct_r64;
typedef struct slate_HermitianBandMatrix_struct_r64* slate_HermitianBandMatrix_r64;

slate_HermitianBandMatrix_r64 slate_HermitianBandMatrix_create_r64(slate_Uplo uplo, int64_t n, int64_t kd, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_HermitianBandMatrix_destroy_r64(slate_HermitianBandMatrix_r64 A);

void slate_HermitianBandMatrix_insertLocalTiles_r64(slate_HermitianBandMatrix_r64 A);

int64_t slate_HermitianBandMatrix_mt_r64(slate_HermitianBandMatrix_r64 A);

int64_t slate_HermitianBandMatrix_nt_r64(slate_HermitianBandMatrix_r64 A);

int64_t slate_HermitianBandMatrix_m_r64(slate_HermitianBandMatrix_r64 A);

int64_t slate_HermitianBandMatrix_n_r64(slate_HermitianBandMatrix_r64 A);

bool slate_HermitianBandMatrix_tileIsLocal_r64(slate_HermitianBandMatrix_r64 A, int64_t i, int64_t j);

slate_Tile_r64 slate_HermitianBandMatrix_at_r64(slate_HermitianBandMatrix_r64 A, int64_t i, int64_t j);

void slate_HermitianBandMatrix_transpose_in_place_r64(slate_HermitianBandMatrix_r64 A);

void slate_HermitianBandMatrix_conjTranspose_in_place_r64(slate_HermitianBandMatrix_r64 A);

/// slate::HermitianBandMatrix<std::complex<float>>
struct slate_HermitianBandMatrix_struct_c32;
typedef struct slate_HermitianBandMatrix_struct_c32* slate_HermitianBandMatrix_c32;

slate_HermitianBandMatrix_c32 slate_HermitianBandMatrix_create_c32(slate_Uplo uplo, int64_t n, int64_t kd, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_HermitianBandMatrix_destroy_c32(slate_HermitianBandMatrix_c32 A);

void slate_HermitianBandMatrix_insertLocalTiles_c32(slate_HermitianBandMatrix_c32 A);

int64_t slate_HermitianBandMatrix_mt_c32(slate_HermitianBandMatrix_c32 A);

int64_t slate_HermitianBandMatrix_nt_c32(slate_HermitianBandMatrix_c32 A);

int64_t slate_HermitianBandMatrix_m_c32(slate_HermitianBandMatrix_c32 A);

int64_t slate_HermitianBandMatrix_n_c32(slate_HermitianBandMatrix_c32 A);

bool slate_HermitianBandMatrix_tileIsLocal_c32(slate_HermitianBandMatrix_c32 A, int64_t i, int64_t j);

slate_Tile_c32 slate_HermitianBandMatrix_at_c32(slate_HermitianBandMatrix_c32 A, int64_t i, int64_t j);

void slate_HermitianBandMatrix_transpose_in_place_c32(slate_HermitianBandMatrix_c32 A);

void slate_HermitianBandMatrix_conjTranspose_in_place_c32(slate_HermitianBandMatrix_c32 A);

/// slate::HermitianBandMatrix<std::complex<double>>
struct slate_HermitianBandMatrix_struct_c64;
typedef struct slate_HermitianBandMatrix_struct_c64* slate_HermitianBandMatrix_c64;

slate_HermitianBandMatrix_c64 slate_HermitianBandMatrix_create_c64(slate_Uplo uplo, int64_t n, int64_t kd, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_HermitianBandMatrix_destroy_c64(slate_HermitianBandMatrix_c64 A);

void slate_HermitianBandMatrix_insertLocalTiles_c64(slate_HermitianBandMatrix_c64 A);

int64_t slate_HermitianBandMatrix_mt_c64(slate_HermitianBandMatrix_c64 A);

int64_t slate_HermitianBandMatrix_nt_c64(slate_HermitianBandMatrix_c64 A);

int64_t slate_HermitianBandMatrix_m_c64(slate_HermitianBandMatrix_c64 A);

int64_t slate_HermitianBandMatrix_n_c64(slate_HermitianBandMatrix_c64 A);

bool slate_HermitianBandMatrix_tileIsLocal_c64(slate_HermitianBandMatrix_c64 A, int64_t i, int64_t j);

slate_Tile_c64 slate_HermitianBandMatrix_at_c64(slate_HermitianBandMatrix_c64 A, int64_t i, int64_t j);

void slate_HermitianBandMatrix_transpose_in_place_c64(slate_HermitianBandMatrix_c64 A);

void slate_HermitianBandMatrix_conjTranspose_in_place_c64(slate_HermitianBandMatrix_c64 A);

//------------------------------------------------------------------------------
/// slate::TriangularMatrix<float>
struct slate_TriangularMatrix_struct_r32;
typedef struct slate_TriangularMatrix_struct_r32* slate_TriangularMatrix_r32;

slate_TriangularMatrix_r32 slate_TriangularMatrix_create_r32(slate_Uplo uplo, slate_Diag diag, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_TriangularMatrix_r32 slate_TriangularMatrix_create_fromScaLAPACK_r32(slate_Uplo uplo, slate_Diag diag, int64_t n, float* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_TriangularMatrix_destroy_r32(slate_TriangularMatrix_r32 A);

void slate_TriangularMatrix_insertLocalTiles_r32(slate_TriangularMatrix_r32 A);

int64_t slate_TriangularMatrix_mt_r32(slate_TriangularMatrix_r32 A);

int64_t slate_TriangularMatrix_nt_r32(slate_TriangularMatrix_r32 A);

int64_t slate_TriangularMatrix_m_r32(slate_TriangularMatrix_r32 A);

int64_t slate_TriangularMatrix_n_r32(slate_TriangularMatrix_r32 A);

bool slate_TriangularMatrix_tileIsLocal_r32(slate_TriangularMatrix_r32 A, int64_t i, int64_t j);

slate_Tile_r32 slate_TriangularMatrix_at_r32(slate_TriangularMatrix_r32 A, int64_t i, int64_t j);

void slate_TriangularMatrix_transpose_in_place_r32(slate_TriangularMatrix_r32 A);

void slate_TriangularMatrix_conjTranspose_in_place_r32(slate_TriangularMatrix_r32 A);

/// slate::TriangularMatrix<double>
struct slate_TriangularMatrix_struct_r64;
typedef struct slate_TriangularMatrix_struct_r64* slate_TriangularMatrix_r64;

slate_TriangularMatrix_r64 slate_TriangularMatrix_create_r64(slate_Uplo uplo, slate_Diag diag, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_TriangularMatrix_r64 slate_TriangularMatrix_create_fromScaLAPACK_r64(slate_Uplo uplo, slate_Diag diag, int64_t n, double* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_TriangularMatrix_destroy_r64(slate_TriangularMatrix_r64 A);

void slate_TriangularMatrix_insertLocalTiles_r64(slate_TriangularMatrix_r64 A);

int64_t slate_TriangularMatrix_mt_r64(slate_TriangularMatrix_r64 A);

int64_t slate_TriangularMatrix_nt_r64(slate_TriangularMatrix_r64 A);

int64_t slate_TriangularMatrix_m_r64(slate_TriangularMatrix_r64 A);

int64_t slate_TriangularMatrix_n_r64(slate_TriangularMatrix_r64 A);

bool slate_TriangularMatrix_tileIsLocal_r64(slate_TriangularMatrix_r64 A, int64_t i, int64_t j);

slate_Tile_r64 slate_TriangularMatrix_at_r64(slate_TriangularMatrix_r64 A, int64_t i, int64_t j);

void slate_TriangularMatrix_transpose_in_place_r64(slate_TriangularMatrix_r64 A);

void slate_TriangularMatrix_conjTranspose_in_place_r64(slate_TriangularMatrix_r64 A);

/// slate::TriangularMatrix<std::complex<float>>
struct slate_TriangularMatrix_struct_c32;
typedef struct slate_TriangularMatrix_struct_c32* slate_TriangularMatrix_c32;

slate_TriangularMatrix_c32 slate_TriangularMatrix_create_c32(slate_Uplo uplo, slate_Diag diag, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_TriangularMatrix_c32 slate_TriangularMatrix_create_fromScaLAPACK_c32(slate_Uplo uplo, slate_Diag diag, int64_t n, float _Complex* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_TriangularMatrix_destroy_c32(slate_TriangularMatrix_c32 A);

void slate_TriangularMatrix_insertLocalTiles_c32(slate_TriangularMatrix_c32 A);

int64_t slate_TriangularMatrix_mt_c32(slate_TriangularMatrix_c32 A);

int64_t slate_TriangularMatrix_nt_c32(slate_TriangularMatrix_c32 A);

int64_t slate_TriangularMatrix_m_c32(slate_TriangularMatrix_c32 A);

int64_t slate_TriangularMatrix_n_c32(slate_TriangularMatrix_c32 A);

bool slate_TriangularMatrix_tileIsLocal_c32(slate_TriangularMatrix_c32 A, int64_t i, int64_t j);

slate_Tile_c32 slate_TriangularMatrix_at_c32(slate_TriangularMatrix_c32 A, int64_t i, int64_t j);

void slate_TriangularMatrix_transpose_in_place_c32(slate_TriangularMatrix_c32 A);

void slate_TriangularMatrix_conjTranspose_in_place_c32(slate_TriangularMatrix_c32 A);

/// slate::TriangularMatrix<std::complex<double>>
struct slate_TriangularMatrix_struct_c64;
typedef struct slate_TriangularMatrix_struct_c64* slate_TriangularMatrix_c64;

slate_TriangularMatrix_c64 slate_TriangularMatrix_create_c64(slate_Uplo uplo, slate_Diag diag, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_TriangularMatrix_c64 slate_TriangularMatrix_create_fromScaLAPACK_c64(slate_Uplo uplo, slate_Diag diag, int64_t n, double _Complex* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_TriangularMatrix_destroy_c64(slate_TriangularMatrix_c64 A);

void slate_TriangularMatrix_insertLocalTiles_c64(slate_TriangularMatrix_c64 A);

int64_t slate_TriangularMatrix_mt_c64(slate_TriangularMatrix_c64 A);

int64_t slate_TriangularMatrix_nt_c64(slate_TriangularMatrix_c64 A);

int64_t slate_TriangularMatrix_m_c64(slate_TriangularMatrix_c64 A);

int64_t slate_TriangularMatrix_n_c64(slate_TriangularMatrix_c64 A);

bool slate_TriangularMatrix_tileIsLocal_c64(slate_TriangularMatrix_c64 A, int64_t i, int64_t j);

slate_Tile_c64 slate_TriangularMatrix_at_c64(slate_TriangularMatrix_c64 A, int64_t i, int64_t j);

void slate_TriangularMatrix_transpose_in_place_c64(slate_TriangularMatrix_c64 A);

void slate_TriangularMatrix_conjTranspose_in_place_c64(slate_TriangularMatrix_c64 A);

//------------------------------------------------------------------------------
/// slate::TriangularBandMatrix<float>
struct slate_TriangularBandMatrix_struct_r32;
typedef struct slate_TriangularBandMatrix_struct_r32* slate_TriangularBandMatrix_r32;

slate_TriangularBandMatrix_r32 slate_TriangularBandMatrix_create_r32(slate_Uplo uplo, slate_Diag diag, int64_t n, int64_t kd, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_TriangularBandMatrix_destroy_r32(slate_TriangularBandMatrix_r32 A);

void slate_TriangularBandMatrix_insertLocalTiles_r32(slate_TriangularBandMatrix_r32 A);

int64_t slate_TriangularBandMatrix_mt_r32(slate_TriangularBandMatrix_r32 A);

int64_t slate_TriangularBandMatrix_nt_r32(slate_TriangularBandMatrix_r32 A);

int64_t slate_TriangularBandMatrix_m_r32(slate_TriangularBandMatrix_r32 A);

int64_t slate_TriangularBandMatrix_n_r32(slate_TriangularBandMatrix_r32 A);

bool slate_TriangularBandMatrix_tileIsLocal_r32(slate_TriangularBandMatrix_r32 A, int64_t i, int64_t j);

slate_Tile_r32 slate_TriangularBandMatrix_at_r32(slate_TriangularBandMatrix_r32 A, int64_t i, int64_t j);

void slate_TriangularBandMatrix_transpose_in_place_r32(slate_TriangularBandMatrix_r32 A);

void slate_TriangularBandMatrix_conjTranspose_in_place_r32(slate_TriangularBandMatrix_r32 A);

/// slate::TriangularBandMatrix<double>
struct slate_TriangularBandMatrix_struct_r64;
typedef struct slate_TriangularBandMatrix_struct_r64* slate_TriangularBandMatrix_r64;

slate_TriangularBandMatrix_r64 slate_TriangularBandMatrix_create_r64(slate_Uplo uplo, slate_Diag diag, int64_t n, int64_t kd, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_TriangularBandMatrix_destroy_r64(slate_TriangularBandMatrix_r64 A);

void slate_TriangularBandMatrix_insertLocalTiles_r64(slate_TriangularBandMatrix_r64 A);

int64_t slate_TriangularBandMatrix_mt_r64(slate_TriangularBandMatrix_r64 A);

int64_t slate_TriangularBandMatrix_nt_r64(slate_TriangularBandMatrix_r64 A);

int64_t slate_TriangularBandMatrix_m_r64(slate_TriangularBandMatrix_r64 A);

int64_t slate_TriangularBandMatrix_n_r64(slate_TriangularBandMatrix_r64 A);

bool slate_TriangularBandMatrix_tileIsLocal_r64(slate_TriangularBandMatrix_r64 A, int64_t i, int64_t j);

slate_Tile_r64 slate_TriangularBandMatrix_at_r64(slate_TriangularBandMatrix_r64 A, int64_t i, int64_t j);

void slate_TriangularBandMatrix_transpose_in_place_r64(slate_TriangularBandMatrix_r64 A);

void slate_TriangularBandMatrix_conjTranspose_in_place_r64(slate_TriangularBandMatrix_r64 A);

/// slate::TriangularBandMatrix<std::complex<float>>
struct slate_TriangularBandMatrix_struct_c32;
typedef struct slate_TriangularBandMatrix_struct_c32* slate_TriangularBandMatrix_c32;

slate_TriangularBandMatrix_c32 slate_TriangularBandMatrix_create_c32(slate_Uplo uplo, slate_Diag diag, int64_t n, int64_t kd, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_TriangularBandMatrix_destroy_c32(slate_TriangularBandMatrix_c32 A);

void slate_TriangularBandMatrix_insertLocalTiles_c32(slate_TriangularBandMatrix_c32 A);

int64_t slate_TriangularBandMatrix_mt_c32(slate_TriangularBandMatrix_c32 A);

int64_t slate_TriangularBandMatrix_nt_c32(slate_TriangularBandMatrix_c32 A);

int64_t slate_TriangularBandMatrix_m_c32(slate_TriangularBandMatrix_c32 A);

int64_t slate_TriangularBandMatrix_n_c32(slate_TriangularBandMatrix_c32 A);

bool slate_TriangularBandMatrix_tileIsLocal_c32(slate_TriangularBandMatrix_c32 A, int64_t i, int64_t j);

slate_Tile_c32 slate_TriangularBandMatrix_at_c32(slate_TriangularBandMatrix_c32 A, int64_t i, int64_t j);

void slate_TriangularBandMatrix_transpose_in_place_c32(slate_TriangularBandMatrix_c32 A);

void slate_TriangularBandMatrix_conjTranspose_in_place_c32(slate_TriangularBandMatrix_c32 A);

/// slate::TriangularBandMatrix<std::complex<double>>
struct slate_TriangularBandMatrix_struct_c64;
typedef struct slate_TriangularBandMatrix_struct_c64* slate_TriangularBandMatrix_c64;

slate_TriangularBandMatrix_c64 slate_TriangularBandMatrix_create_c64(slate_Uplo uplo, slate_Diag diag, int64_t n, int64_t kd, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_TriangularBandMatrix_destroy_c64(slate_TriangularBandMatrix_c64 A);

void slate_TriangularBandMatrix_insertLocalTiles_c64(slate_TriangularBandMatrix_c64 A);

int64_t slate_TriangularBandMatrix_mt_c64(slate_TriangularBandMatrix_c64 A);

int64_t slate_TriangularBandMatrix_nt_c64(slate_TriangularBandMatrix_c64 A);

int64_t slate_TriangularBandMatrix_m_c64(slate_TriangularBandMatrix_c64 A);

int64_t slate_TriangularBandMatrix_n_c64(slate_TriangularBandMatrix_c64 A);

bool slate_TriangularBandMatrix_tileIsLocal_c64(slate_TriangularBandMatrix_c64 A, int64_t i, int64_t j);

slate_Tile_c64 slate_TriangularBandMatrix_at_c64(slate_TriangularBandMatrix_c64 A, int64_t i, int64_t j);

void slate_TriangularBandMatrix_transpose_in_place_c64(slate_TriangularBandMatrix_c64 A);

void slate_TriangularBandMatrix_conjTranspose_in_place_c64(slate_TriangularBandMatrix_c64 A);

//------------------------------------------------------------------------------
/// slate::SymmetricMatrix<float>
struct slate_SymmetricMatrix_struct_r32;
typedef struct slate_SymmetricMatrix_struct_r32* slate_SymmetricMatrix_r32;

slate_SymmetricMatrix_r32 slate_SymmetricMatrix_create_r32(slate_Uplo uplo, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_SymmetricMatrix_r32 slate_SymmetricMatrix_create_fromScaLAPACK_r32(slate_Uplo uplo, int64_t n, float* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_SymmetricMatrix_destroy_r32(slate_SymmetricMatrix_r32 A);

void slate_SymmetricMatrix_insertLocalTiles_r32(slate_SymmetricMatrix_r32 A);

int64_t slate_SymmetricMatrix_mt_r32(slate_SymmetricMatrix_r32 A);

int64_t slate_SymmetricMatrix_nt_r32(slate_SymmetricMatrix_r32 A);

int64_t slate_SymmetricMatrix_m_r32(slate_SymmetricMatrix_r32 A);

int64_t slate_SymmetricMatrix_n_r32(slate_SymmetricMatrix_r32 A);

bool slate_SymmetricMatrix_tileIsLocal_r32(slate_SymmetricMatrix_r32 A, int64_t i, int64_t j);

slate_Tile_r32 slate_SymmetricMatrix_at_r32(slate_SymmetricMatrix_r32 A, int64_t i, int64_t j);

void slate_SymmetricMatrix_transpose_in_place_r32(slate_SymmetricMatrix_r32 A);

void slate_SymmetricMatrix_conjTranspose_in_place_r32(slate_SymmetricMatrix_r32 A);

/// slate::SymmetricMatrix<double>
struct slate_SymmetricMatrix_struct_r64;
typedef struct slate_SymmetricMatrix_struct_r64* slate_SymmetricMatrix_r64;

slate_SymmetricMatrix_r64 slate_SymmetricMatrix_create_r64(slate_Uplo uplo, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_SymmetricMatrix_r64 slate_SymmetricMatrix_create_fromScaLAPACK_r64(slate_Uplo uplo, int64_t n, double* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_SymmetricMatrix_destroy_r64(slate_SymmetricMatrix_r64 A);

void slate_SymmetricMatrix_insertLocalTiles_r64(slate_SymmetricMatrix_r64 A);

int64_t slate_SymmetricMatrix_mt_r64(slate_SymmetricMatrix_r64 A);

int64_t slate_SymmetricMatrix_nt_r64(slate_SymmetricMatrix_r64 A);

int64_t slate_SymmetricMatrix_m_r64(slate_SymmetricMatrix_r64 A);

int64_t slate_SymmetricMatrix_n_r64(slate_SymmetricMatrix_r64 A);

bool slate_SymmetricMatrix_tileIsLocal_r64(slate_SymmetricMatrix_r64 A, int64_t i, int64_t j);

slate_Tile_r64 slate_SymmetricMatrix_at_r64(slate_SymmetricMatrix_r64 A, int64_t i, int64_t j);

void slate_SymmetricMatrix_transpose_in_place_r64(slate_SymmetricMatrix_r64 A);

void slate_SymmetricMatrix_conjTranspose_in_place_r64(slate_SymmetricMatrix_r64 A);

/// slate::SymmetricMatrix<std::complex<float>>
struct slate_SymmetricMatrix_struct_c32;
typedef struct slate_SymmetricMatrix_struct_c32* slate_SymmetricMatrix_c32;

slate_SymmetricMatrix_c32 slate_SymmetricMatrix_create_c32(slate_Uplo uplo, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_SymmetricMatrix_c32 slate_SymmetricMatrix_create_fromScaLAPACK_c32(slate_Uplo uplo, int64_t n, float _Complex* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_SymmetricMatrix_destroy_c32(slate_SymmetricMatrix_c32 A);

void slate_SymmetricMatrix_insertLocalTiles_c32(slate_SymmetricMatrix_c32 A);

int64_t slate_SymmetricMatrix_mt_c32(slate_SymmetricMatrix_c32 A);

int64_t slate_SymmetricMatrix_nt_c32(slate_SymmetricMatrix_c32 A);

int64_t slate_SymmetricMatrix_m_c32(slate_SymmetricMatrix_c32 A);

int64_t slate_SymmetricMatrix_n_c32(slate_SymmetricMatrix_c32 A);

bool slate_SymmetricMatrix_tileIsLocal_c32(slate_SymmetricMatrix_c32 A, int64_t i, int64_t j);

slate_Tile_c32 slate_SymmetricMatrix_at_c32(slate_SymmetricMatrix_c32 A, int64_t i, int64_t j);

void slate_SymmetricMatrix_transpose_in_place_c32(slate_SymmetricMatrix_c32 A);

void slate_SymmetricMatrix_conjTranspose_in_place_c32(slate_SymmetricMatrix_c32 A);

/// slate::SymmetricMatrix<std::complex<double>>
struct slate_SymmetricMatrix_struct_c64;
typedef struct slate_SymmetricMatrix_struct_c64* slate_SymmetricMatrix_c64;

slate_SymmetricMatrix_c64 slate_SymmetricMatrix_create_c64(slate_Uplo uplo, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_SymmetricMatrix_c64 slate_SymmetricMatrix_create_fromScaLAPACK_c64(slate_Uplo uplo, int64_t n, double _Complex* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_SymmetricMatrix_destroy_c64(slate_SymmetricMatrix_c64 A);

void slate_SymmetricMatrix_insertLocalTiles_c64(slate_SymmetricMatrix_c64 A);

int64_t slate_SymmetricMatrix_mt_c64(slate_SymmetricMatrix_c64 A);

int64_t slate_SymmetricMatrix_nt_c64(slate_SymmetricMatrix_c64 A);

int64_t slate_SymmetricMatrix_m_c64(slate_SymmetricMatrix_c64 A);

int64_t slate_SymmetricMatrix_n_c64(slate_SymmetricMatrix_c64 A);

bool slate_SymmetricMatrix_tileIsLocal_c64(slate_SymmetricMatrix_c64 A, int64_t i, int64_t j);

slate_Tile_c64 slate_SymmetricMatrix_at_c64(slate_SymmetricMatrix_c64 A, int64_t i, int64_t j);

void slate_SymmetricMatrix_transpose_in_place_c64(slate_SymmetricMatrix_c64 A);

void slate_SymmetricMatrix_conjTranspose_in_place_c64(slate_SymmetricMatrix_c64 A);

//------------------------------------------------------------------------------
/// slate::TrapezoidMatrix<float>
struct slate_TrapezoidMatrix_struct_r32;
typedef struct slate_TrapezoidMatrix_struct_r32* slate_TrapezoidMatrix_r32;

slate_TrapezoidMatrix_r32 slate_TrapezoidMatrix_create_r32(slate_Uplo uplo, slate_Diag diag, int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_TrapezoidMatrix_r32 slate_TrapezoidMatrix_create_fromScaLAPACK_r32(slate_Uplo uplo, slate_Diag diag, int64_t m, int64_t n, float* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_TrapezoidMatrix_destroy_r32(slate_TrapezoidMatrix_r32 A);

void slate_TrapezoidMatrix_insertLocalTiles_r32(slate_TrapezoidMatrix_r32 A);

int64_t slate_TrapezoidMatrix_mt_r32(slate_TrapezoidMatrix_r32 A);

int64_t slate_TrapezoidMatrix_nt_r32(slate_TrapezoidMatrix_r32 A);

int64_t slate_TrapezoidMatrix_m_r32(slate_TrapezoidMatrix_r32 A);

int64_t slate_TrapezoidMatrix_n_r32(slate_TrapezoidMatrix_r32 A);

bool slate_TrapezoidMatrix_tileIsLocal_r32(slate_TrapezoidMatrix_r32 A, int64_t i, int64_t j);

slate_Tile_r32 slate_TrapezoidMatrix_at_r32(slate_TrapezoidMatrix_r32 A, int64_t i, int64_t j);

void slate_TrapezoidMatrix_transpose_in_place_r32(slate_TrapezoidMatrix_r32 A);

void slate_TrapezoidMatrix_conjTranspose_in_place_r32(slate_TrapezoidMatrix_r32 A);

/// slate::TrapezoidMatrix<double>
struct slate_TrapezoidMatrix_struct_r64;
typedef struct slate_TrapezoidMatrix_struct_r64* slate_TrapezoidMatrix_r64;

slate_TrapezoidMatrix_r64 slate_TrapezoidMatrix_create_r64(slate_Uplo uplo, slate_Diag diag, int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_TrapezoidMatrix_r64 slate_TrapezoidMatrix_create_fromScaLAPACK_r64(slate_Uplo uplo, slate_Diag diag, int64_t m, int64_t n, double* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_TrapezoidMatrix_destroy_r64(slate_TrapezoidMatrix_r64 A);

void slate_TrapezoidMatrix_insertLocalTiles_r64(slate_TrapezoidMatrix_r64 A);

int64_t slate_TrapezoidMatrix_mt_r64(slate_TrapezoidMatrix_r64 A);

int64_t slate_TrapezoidMatrix_nt_r64(slate_TrapezoidMatrix_r64 A);

int64_t slate_TrapezoidMatrix_m_r64(slate_TrapezoidMatrix_r64 A);

int64_t slate_TrapezoidMatrix_n_r64(slate_TrapezoidMatrix_r64 A);

bool slate_TrapezoidMatrix_tileIsLocal_r64(slate_TrapezoidMatrix_r64 A, int64_t i, int64_t j);

slate_Tile_r64 slate_TrapezoidMatrix_at_r64(slate_TrapezoidMatrix_r64 A, int64_t i, int64_t j);

void slate_TrapezoidMatrix_transpose_in_place_r64(slate_TrapezoidMatrix_r64 A);

void slate_TrapezoidMatrix_conjTranspose_in_place_r64(slate_TrapezoidMatrix_r64 A);

/// slate::TrapezoidMatrix<std::complex<float>>
struct slate_TrapezoidMatrix_struct_c32;
typedef struct slate_TrapezoidMatrix_struct_c32* slate_TrapezoidMatrix_c32;

slate_TrapezoidMatrix_c32 slate_TrapezoidMatrix_create_c32(slate_Uplo uplo, slate_Diag diag, int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_TrapezoidMatrix_c32 slate_TrapezoidMatrix_create_fromScaLAPACK_c32(slate_Uplo uplo, slate_Diag diag, int64_t m, int64_t n, float _Complex* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_TrapezoidMatrix_destroy_c32(slate_TrapezoidMatrix_c32 A);

void slate_TrapezoidMatrix_insertLocalTiles_c32(slate_TrapezoidMatrix_c32 A);

int64_t slate_TrapezoidMatrix_mt_c32(slate_TrapezoidMatrix_c32 A);

int64_t slate_TrapezoidMatrix_nt_c32(slate_TrapezoidMatrix_c32 A);

int64_t slate_TrapezoidMatrix_m_c32(slate_TrapezoidMatrix_c32 A);

int64_t slate_TrapezoidMatrix_n_c32(slate_TrapezoidMatrix_c32 A);

bool slate_TrapezoidMatrix_tileIsLocal_c32(slate_TrapezoidMatrix_c32 A, int64_t i, int64_t j);

slate_Tile_c32 slate_TrapezoidMatrix_at_c32(slate_TrapezoidMatrix_c32 A, int64_t i, int64_t j);

void slate_TrapezoidMatrix_transpose_in_place_c32(slate_TrapezoidMatrix_c32 A);

void slate_TrapezoidMatrix_conjTranspose_in_place_c32(slate_TrapezoidMatrix_c32 A);

/// slate::TrapezoidMatrix<std::complex<double>>
struct slate_TrapezoidMatrix_struct_c64;
typedef struct slate_TrapezoidMatrix_struct_c64* slate_TrapezoidMatrix_c64;

slate_TrapezoidMatrix_c64 slate_TrapezoidMatrix_create_c64(slate_Uplo uplo, slate_Diag diag, int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm);

slate_TrapezoidMatrix_c64 slate_TrapezoidMatrix_create_fromScaLAPACK_c64(slate_Uplo uplo, slate_Diag diag, int64_t m, int64_t n, double _Complex* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm);

void slate_TrapezoidMatrix_destroy_c64(slate_TrapezoidMatrix_c64 A);

void slate_TrapezoidMatrix_insertLocalTiles_c64(slate_TrapezoidMatrix_c64 A);

int64_t slate_TrapezoidMatrix_mt_c64(slate_TrapezoidMatrix_c64 A);

int64_t slate_TrapezoidMatrix_nt_c64(slate_TrapezoidMatrix_c64 A);

int64_t slate_TrapezoidMatrix_m_c64(slate_TrapezoidMatrix_c64 A);

int64_t slate_TrapezoidMatrix_n_c64(slate_TrapezoidMatrix_c64 A);

bool slate_TrapezoidMatrix_tileIsLocal_c64(slate_TrapezoidMatrix_c64 A, int64_t i, int64_t j);

slate_Tile_c64 slate_TrapezoidMatrix_at_c64(slate_TrapezoidMatrix_c64 A, int64_t i, int64_t j);

void slate_TrapezoidMatrix_transpose_in_place_c64(slate_TrapezoidMatrix_c64 A);

void slate_TrapezoidMatrix_conjTranspose_in_place_c64(slate_TrapezoidMatrix_c64 A);

//------------------------------------------------------------------------------
/// slate::TriangularFactors<float>
struct slate_TriangularFactors_struct_r32;
typedef struct slate_TriangularFactors_struct_r32* slate_TriangularFactors_r32;

slate_TriangularFactors_r32 slate_TriangularFactors_create_r32();
void slate_TriangularFactors_destroy_r32(slate_TriangularFactors_r32 T);

/// slate::TriangularFactors<double>
struct slate_TriangularFactors_struct_r64;
typedef struct slate_TriangularFactors_struct_r64* slate_TriangularFactors_r64;

slate_TriangularFactors_r64 slate_TriangularFactors_create_r64();
void slate_TriangularFactors_destroy_r64(slate_TriangularFactors_r64 T);

/// slate::TriangularFactors<std::complex<float>>
struct slate_TriangularFactors_struct_c32;
typedef struct slate_TriangularFactors_struct_c32* slate_TriangularFactors_c32;

slate_TriangularFactors_c32 slate_TriangularFactors_create_c32();
void slate_TriangularFactors_destroy_c32(slate_TriangularFactors_c32 T);

/// slate::TriangularFactors<std::complex<double>>
struct slate_TriangularFactors_struct_c64;
typedef struct slate_TriangularFactors_struct_c64* slate_TriangularFactors_c64;

slate_TriangularFactors_c64 slate_TriangularFactors_create_c64();
void slate_TriangularFactors_destroy_c64(slate_TriangularFactors_c64 T);

//------------------------------------------------------------------------------
/// slate::Pivots
struct slate_Pivots_struct;
typedef struct slate_Pivots_struct* slate_Pivots;

slate_Pivots slate_Pivots_create();
void slate_Pivots_destroy(slate_Pivots pivots);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // SLATE_C_API_MATRIX_H