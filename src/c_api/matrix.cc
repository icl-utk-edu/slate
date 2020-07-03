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

#include "slate/c_api/matrix.h"
#include "slate/c_api/util.hh"
#include "slate/slate.hh"

/// slate::Tile<float>::mb()
int64_t slate_Tile_mb_r32(slate_Tile_r32 T)
{
    assert(sizeof(slate_Tile_c64) == sizeof(slate::Tile<float>));
    slate::Tile<float> T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<float>));
    return(T_.mb());
}
/// slate::Tile<float>::nb()
int64_t slate_Tile_nb_r32(slate_Tile_r32 T)
{
    assert(sizeof(slate_Tile_c64) == sizeof(slate::Tile<float>));
    slate::Tile<float> T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<float>));
    return(T_.nb());
}
/// slate::Tile<float>::stride()
int64_t slate_Tile_stride_r32(slate_Tile_r32 T)
{
    assert(sizeof(slate_Tile_c64) == sizeof(slate::Tile<float>));
    slate::Tile<float> T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<float>));
    return(T_.stride());
}
/// slate::Tile<float>::data()
float* slate_Tile_data_r32(slate_Tile_r32 T)
{
    assert(sizeof(slate_Tile_c64) == sizeof(slate::Tile<float>));
    slate::Tile<float> T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<float>));
    return((float*)T_.data());
}
/// slate::Tile<double>::mb()
int64_t slate_Tile_mb_r64(slate_Tile_r64 T)
{
    assert(sizeof(slate_Tile_c64) == sizeof(slate::Tile<double>));
    slate::Tile<double> T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<double>));
    return(T_.mb());
}
/// slate::Tile<double>::nb()
int64_t slate_Tile_nb_r64(slate_Tile_r64 T)
{
    assert(sizeof(slate_Tile_c64) == sizeof(slate::Tile<double>));
    slate::Tile<double> T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<double>));
    return(T_.nb());
}
/// slate::Tile<double>::stride()
int64_t slate_Tile_stride_r64(slate_Tile_r64 T)
{
    assert(sizeof(slate_Tile_c64) == sizeof(slate::Tile<double>));
    slate::Tile<double> T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<double>));
    return(T_.stride());
}
/// slate::Tile<double>::data()
double* slate_Tile_data_r64(slate_Tile_r64 T)
{
    assert(sizeof(slate_Tile_c64) == sizeof(slate::Tile<double>));
    slate::Tile<double> T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<double>));
    return((double*)T_.data());
}
/// slate::Tile<std::complex<float>>::mb()
int64_t slate_Tile_mb_c32(slate_Tile_c32 T)
{
    assert(sizeof(slate_Tile_c64) == sizeof(slate::Tile<std::complex<float>>));
    slate::Tile<std::complex<float>> T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<float>>));
    return(T_.mb());
}
/// slate::Tile<std::complex<float>>::nb()
int64_t slate_Tile_nb_c32(slate_Tile_c32 T)
{
    assert(sizeof(slate_Tile_c64) == sizeof(slate::Tile<std::complex<float>>));
    slate::Tile<std::complex<float>> T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<float>>));
    return(T_.nb());
}
/// slate::Tile<std::complex<float>>::stride()
int64_t slate_Tile_stride_c32(slate_Tile_c32 T)
{
    assert(sizeof(slate_Tile_c64) == sizeof(slate::Tile<std::complex<float>>));
    slate::Tile<std::complex<float>> T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<float>>));
    return(T_.stride());
}
/// slate::Tile<std::complex<float>>::data()
float _Complex* slate_Tile_data_c32(slate_Tile_c32 T)
{
    assert(sizeof(slate_Tile_c64) == sizeof(slate::Tile<std::complex<float>>));
    slate::Tile<std::complex<float>> T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<float>>));
    return((float _Complex*)T_.data());
}
/// slate::Tile<std::complex<double>>::mb()
int64_t slate_Tile_mb_c64(slate_Tile_c64 T)
{
    assert(sizeof(slate_Tile_c64) == sizeof(slate::Tile<std::complex<double>>));
    slate::Tile<std::complex<double>> T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<double>>));
    return(T_.mb());
}
/// slate::Tile<std::complex<double>>::nb()
int64_t slate_Tile_nb_c64(slate_Tile_c64 T)
{
    assert(sizeof(slate_Tile_c64) == sizeof(slate::Tile<std::complex<double>>));
    slate::Tile<std::complex<double>> T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<double>>));
    return(T_.nb());
}
/// slate::Tile<std::complex<double>>::stride()
int64_t slate_Tile_stride_c64(slate_Tile_c64 T)
{
    assert(sizeof(slate_Tile_c64) == sizeof(slate::Tile<std::complex<double>>));
    slate::Tile<std::complex<double>> T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<double>>));
    return(T_.stride());
}
/// slate::Tile<std::complex<double>>::data()
double _Complex* slate_Tile_data_c64(slate_Tile_c64 T)
{
    assert(sizeof(slate_Tile_c64) == sizeof(slate::Tile<std::complex<double>>));
    slate::Tile<std::complex<double>> T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<double>>));
    return((double _Complex*)T_.data());
}
slate_Matrix_r32 slate_Matrix_create_r32(int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::Matrix<float>(m, n, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_Matrix_r32>(A);
}
slate_Matrix_r32 slate_Matrix_create_fromScaLAPACK_r32(int64_t m, int64_t n, float* A, int64_t lda, int64_t mb, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A_ = new slate::Matrix<float>();
    (*A_).fromScaLAPACK(m, n, (float*)A, lda, mb, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_Matrix_r32>(A_);
}
slate_Matrix_r32 slate_Matrix_create_slice_r32(slate_Matrix_r32 A, int64_t i1, int64_t i2, int64_t j1, int64_t j2)
{
    auto* A_ = reinterpret_cast<slate::Matrix<float>*>(A);
    auto* A_slice = new slate::Matrix<float>(A_->slice(i1, i2, j1, j2));
    return reinterpret_cast<slate_Matrix_r32>(A_slice);
}
void slate_Matrix_destroy_r32(slate_Matrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<float>*>(A);
    delete A_;
}
void slate_Matrix_insertLocalTiles_r32(slate_Matrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<float>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_Matrix_mt_r32(slate_Matrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<float>*>(A);
    return(A_->mt());
}
int64_t slate_Matrix_nt_r32(slate_Matrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<float>*>(A);
    return(A_->nt());
}
int64_t slate_Matrix_m_r32(slate_Matrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<float>*>(A);
    return(A_->m());
}
int64_t slate_Matrix_n_r32(slate_Matrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<float>*>(A);
    return(A_->n());
}
bool slate_Matrix_tileIsLocal_r32(slate_Matrix_r32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::Matrix<float>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_r32 slate_Matrix_at_r32(slate_Matrix_r32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::Matrix<float>*>(A);
    slate::Tile<float> T = A_->at(i, j);
    slate_Tile_r32 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<float>));
    return(T_);
}
void slate_Matrix_transpose_in_place_r32(slate_Matrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<float>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_Matrix_conjTranspose_in_place_r32(slate_Matrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<float>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_Matrix_r64 slate_Matrix_create_r64(int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::Matrix<double>(m, n, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_Matrix_r64>(A);
}
slate_Matrix_r64 slate_Matrix_create_fromScaLAPACK_r64(int64_t m, int64_t n, double* A, int64_t lda, int64_t mb, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A_ = new slate::Matrix<double>();
    (*A_).fromScaLAPACK(m, n, (double*)A, lda, mb, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_Matrix_r64>(A_);
}
slate_Matrix_r64 slate_Matrix_create_slice_r64(slate_Matrix_r64 A, int64_t i1, int64_t i2, int64_t j1, int64_t j2)
{
    auto* A_ = reinterpret_cast<slate::Matrix<double>*>(A);
    auto* A_slice = new slate::Matrix<double>(A_->slice(i1, i2, j1, j2));
    return reinterpret_cast<slate_Matrix_r64>(A_slice);
}
void slate_Matrix_destroy_r64(slate_Matrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<double>*>(A);
    delete A_;
}
void slate_Matrix_insertLocalTiles_r64(slate_Matrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<double>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_Matrix_mt_r64(slate_Matrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<double>*>(A);
    return(A_->mt());
}
int64_t slate_Matrix_nt_r64(slate_Matrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<double>*>(A);
    return(A_->nt());
}
int64_t slate_Matrix_m_r64(slate_Matrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<double>*>(A);
    return(A_->m());
}
int64_t slate_Matrix_n_r64(slate_Matrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<double>*>(A);
    return(A_->n());
}
bool slate_Matrix_tileIsLocal_r64(slate_Matrix_r64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::Matrix<double>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_r64 slate_Matrix_at_r64(slate_Matrix_r64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::Matrix<double>*>(A);
    slate::Tile<double> T = A_->at(i, j);
    slate_Tile_r64 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<double>));
    return(T_);
}
void slate_Matrix_transpose_in_place_r64(slate_Matrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<double>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_Matrix_conjTranspose_in_place_r64(slate_Matrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<double>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_Matrix_c32 slate_Matrix_create_c32(int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::Matrix<std::complex<float>>(m, n, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_Matrix_c32>(A);
}
slate_Matrix_c32 slate_Matrix_create_fromScaLAPACK_c32(int64_t m, int64_t n, float _Complex* A, int64_t lda, int64_t mb, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A_ = new slate::Matrix<std::complex<float>>();
    (*A_).fromScaLAPACK(m, n, (std::complex<float>*)A, lda, mb, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_Matrix_c32>(A_);
}
slate_Matrix_c32 slate_Matrix_create_slice_c32(slate_Matrix_c32 A, int64_t i1, int64_t i2, int64_t j1, int64_t j2)
{
    auto* A_ = reinterpret_cast<slate::Matrix<std::complex<float>>*>(A);
    auto* A_slice = new slate::Matrix<std::complex<float>>(A_->slice(i1, i2, j1, j2));
    return reinterpret_cast<slate_Matrix_c32>(A_slice);
}
void slate_Matrix_destroy_c32(slate_Matrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<std::complex<float>>*>(A);
    delete A_;
}
void slate_Matrix_insertLocalTiles_c32(slate_Matrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<std::complex<float>>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_Matrix_mt_c32(slate_Matrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<std::complex<float>>*>(A);
    return(A_->mt());
}
int64_t slate_Matrix_nt_c32(slate_Matrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<std::complex<float>>*>(A);
    return(A_->nt());
}
int64_t slate_Matrix_m_c32(slate_Matrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<std::complex<float>>*>(A);
    return(A_->m());
}
int64_t slate_Matrix_n_c32(slate_Matrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<std::complex<float>>*>(A);
    return(A_->n());
}
bool slate_Matrix_tileIsLocal_c32(slate_Matrix_c32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::Matrix<std::complex<float>>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_c32 slate_Matrix_at_c32(slate_Matrix_c32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::Matrix<std::complex<float>>*>(A);
    slate::Tile<std::complex<float>> T = A_->at(i, j);
    slate_Tile_c32 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<float>>));
    return(T_);
}
void slate_Matrix_transpose_in_place_c32(slate_Matrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<std::complex<float>>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_Matrix_conjTranspose_in_place_c32(slate_Matrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<std::complex<float>>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_Matrix_c64 slate_Matrix_create_c64(int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::Matrix<std::complex<double>>(m, n, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_Matrix_c64>(A);
}
slate_Matrix_c64 slate_Matrix_create_fromScaLAPACK_c64(int64_t m, int64_t n, double _Complex* A, int64_t lda, int64_t mb, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A_ = new slate::Matrix<std::complex<double>>();
    (*A_).fromScaLAPACK(m, n, (std::complex<double>*)A, lda, mb, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_Matrix_c64>(A_);
}
slate_Matrix_c64 slate_Matrix_create_slice_c64(slate_Matrix_c64 A, int64_t i1, int64_t i2, int64_t j1, int64_t j2)
{
    auto* A_ = reinterpret_cast<slate::Matrix<std::complex<double>>*>(A);
    auto* A_slice = new slate::Matrix<std::complex<double>>(A_->slice(i1, i2, j1, j2));
    return reinterpret_cast<slate_Matrix_c64>(A_slice);
}
void slate_Matrix_destroy_c64(slate_Matrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<std::complex<double>>*>(A);
    delete A_;
}
void slate_Matrix_insertLocalTiles_c64(slate_Matrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<std::complex<double>>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_Matrix_mt_c64(slate_Matrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<std::complex<double>>*>(A);
    return(A_->mt());
}
int64_t slate_Matrix_nt_c64(slate_Matrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<std::complex<double>>*>(A);
    return(A_->nt());
}
int64_t slate_Matrix_m_c64(slate_Matrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<std::complex<double>>*>(A);
    return(A_->m());
}
int64_t slate_Matrix_n_c64(slate_Matrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<std::complex<double>>*>(A);
    return(A_->n());
}
bool slate_Matrix_tileIsLocal_c64(slate_Matrix_c64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::Matrix<std::complex<double>>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_c64 slate_Matrix_at_c64(slate_Matrix_c64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::Matrix<std::complex<double>>*>(A);
    slate::Tile<std::complex<double>> T = A_->at(i, j);
    slate_Tile_c64 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<double>>));
    return(T_);
}
void slate_Matrix_transpose_in_place_c64(slate_Matrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<std::complex<double>>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_Matrix_conjTranspose_in_place_c64(slate_Matrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::Matrix<std::complex<double>>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
//------------------------------------------------------------------------------
slate_BandMatrix_r32 slate_BandMatrix_create_r32(int64_t m, int64_t n, int64_t kl, int64_t ku, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::BandMatrix<float>(m, n, kl, ku, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_BandMatrix_r32>(A);
}
void slate_BandMatrix_destroy_r32(slate_BandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<float>*>(A);
    delete A_;
}
int64_t slate_BandMatrix_mt_r32(slate_BandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<float>*>(A);
    return(A_->mt());
}
int64_t slate_BandMatrix_nt_r32(slate_BandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<float>*>(A);
    return(A_->nt());
}
int64_t slate_BandMatrix_m_r32(slate_BandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<float>*>(A);
    return(A_->m());
}
int64_t slate_BandMatrix_n_r32(slate_BandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<float>*>(A);
    return(A_->n());
}
bool slate_BandMatrix_tileIsLocal_r32(slate_BandMatrix_r32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<float>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_r32 slate_BandMatrix_at_r32(slate_BandMatrix_r32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<float>*>(A);
    slate::Tile<float> T = A_->at(i, j);
    slate_Tile_r32 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<float>));
    return(T_);
}
void slate_BandMatrix_transpose_in_place_r32(slate_BandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<float>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_BandMatrix_conjTranspose_in_place_r32(slate_BandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<float>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_BandMatrix_r64 slate_BandMatrix_create_r64(int64_t m, int64_t n, int64_t kl, int64_t ku, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::BandMatrix<double>(m, n, kl, ku, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_BandMatrix_r64>(A);
}
void slate_BandMatrix_destroy_r64(slate_BandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<double>*>(A);
    delete A_;
}
int64_t slate_BandMatrix_mt_r64(slate_BandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<double>*>(A);
    return(A_->mt());
}
int64_t slate_BandMatrix_nt_r64(slate_BandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<double>*>(A);
    return(A_->nt());
}
int64_t slate_BandMatrix_m_r64(slate_BandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<double>*>(A);
    return(A_->m());
}
int64_t slate_BandMatrix_n_r64(slate_BandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<double>*>(A);
    return(A_->n());
}
bool slate_BandMatrix_tileIsLocal_r64(slate_BandMatrix_r64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<double>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_r64 slate_BandMatrix_at_r64(slate_BandMatrix_r64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<double>*>(A);
    slate::Tile<double> T = A_->at(i, j);
    slate_Tile_r64 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<double>));
    return(T_);
}
void slate_BandMatrix_transpose_in_place_r64(slate_BandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<double>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_BandMatrix_conjTranspose_in_place_r64(slate_BandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<double>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_BandMatrix_c32 slate_BandMatrix_create_c32(int64_t m, int64_t n, int64_t kl, int64_t ku, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::BandMatrix<std::complex<float>>(m, n, kl, ku, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_BandMatrix_c32>(A);
}
void slate_BandMatrix_destroy_c32(slate_BandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<std::complex<float>>*>(A);
    delete A_;
}
int64_t slate_BandMatrix_mt_c32(slate_BandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<std::complex<float>>*>(A);
    return(A_->mt());
}
int64_t slate_BandMatrix_nt_c32(slate_BandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<std::complex<float>>*>(A);
    return(A_->nt());
}
int64_t slate_BandMatrix_m_c32(slate_BandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<std::complex<float>>*>(A);
    return(A_->m());
}
int64_t slate_BandMatrix_n_c32(slate_BandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<std::complex<float>>*>(A);
    return(A_->n());
}
bool slate_BandMatrix_tileIsLocal_c32(slate_BandMatrix_c32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<std::complex<float>>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_c32 slate_BandMatrix_at_c32(slate_BandMatrix_c32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<std::complex<float>>*>(A);
    slate::Tile<std::complex<float>> T = A_->at(i, j);
    slate_Tile_c32 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<float>>));
    return(T_);
}
void slate_BandMatrix_transpose_in_place_c32(slate_BandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<std::complex<float>>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_BandMatrix_conjTranspose_in_place_c32(slate_BandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<std::complex<float>>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_BandMatrix_c64 slate_BandMatrix_create_c64(int64_t m, int64_t n, int64_t kl, int64_t ku, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::BandMatrix<std::complex<double>>(m, n, kl, ku, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_BandMatrix_c64>(A);
}
void slate_BandMatrix_destroy_c64(slate_BandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<std::complex<double>>*>(A);
    delete A_;
}
int64_t slate_BandMatrix_mt_c64(slate_BandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<std::complex<double>>*>(A);
    return(A_->mt());
}
int64_t slate_BandMatrix_nt_c64(slate_BandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<std::complex<double>>*>(A);
    return(A_->nt());
}
int64_t slate_BandMatrix_m_c64(slate_BandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<std::complex<double>>*>(A);
    return(A_->m());
}
int64_t slate_BandMatrix_n_c64(slate_BandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<std::complex<double>>*>(A);
    return(A_->n());
}
bool slate_BandMatrix_tileIsLocal_c64(slate_BandMatrix_c64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<std::complex<double>>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_c64 slate_BandMatrix_at_c64(slate_BandMatrix_c64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<std::complex<double>>*>(A);
    slate::Tile<std::complex<double>> T = A_->at(i, j);
    slate_Tile_c64 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<double>>));
    return(T_);
}
void slate_BandMatrix_transpose_in_place_c64(slate_BandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<std::complex<double>>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_BandMatrix_conjTranspose_in_place_c64(slate_BandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::BandMatrix<std::complex<double>>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
//------------------------------------------------------------------------------
slate_HermitianMatrix_r32 slate_HermitianMatrix_create_r32(slate_Uplo uplo, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::HermitianMatrix<float>(slate::uplo2cpp(uplo), n, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_HermitianMatrix_r32>(A);
}
slate_HermitianMatrix_r32 slate_HermitianMatrix_create_fromScaLAPACK_r32(slate_Uplo uplo, int64_t n, float* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A_ = new slate::HermitianMatrix<float>();
    (*A_).fromScaLAPACK(slate::uplo2cpp(uplo), n, (float*)A, lda, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_HermitianMatrix_r32>(A_);
}
void slate_HermitianMatrix_destroy_r32(slate_HermitianMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<float>*>(A);
    delete A_;
}
void slate_HermitianMatrix_insertLocalTiles_r32(slate_HermitianMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<float>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_HermitianMatrix_mt_r32(slate_HermitianMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<float>*>(A);
    return(A_->mt());
}
int64_t slate_HermitianMatrix_nt_r32(slate_HermitianMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<float>*>(A);
    return(A_->nt());
}
int64_t slate_HermitianMatrix_m_r32(slate_HermitianMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<float>*>(A);
    return(A_->m());
}
int64_t slate_HermitianMatrix_n_r32(slate_HermitianMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<float>*>(A);
    return(A_->n());
}
bool slate_HermitianMatrix_tileIsLocal_r32(slate_HermitianMatrix_r32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<float>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_r32 slate_HermitianMatrix_at_r32(slate_HermitianMatrix_r32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<float>*>(A);
    slate::Tile<float> T = A_->at(i, j);
    slate_Tile_r32 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<float>));
    return(T_);
}
void slate_HermitianMatrix_transpose_in_place_r32(slate_HermitianMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<float>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_HermitianMatrix_conjTranspose_in_place_r32(slate_HermitianMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<float>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_HermitianMatrix_r64 slate_HermitianMatrix_create_r64(slate_Uplo uplo, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::HermitianMatrix<double>(slate::uplo2cpp(uplo), n, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_HermitianMatrix_r64>(A);
}
slate_HermitianMatrix_r64 slate_HermitianMatrix_create_fromScaLAPACK_r64(slate_Uplo uplo, int64_t n, double* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A_ = new slate::HermitianMatrix<double>();
    (*A_).fromScaLAPACK(slate::uplo2cpp(uplo), n, (double*)A, lda, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_HermitianMatrix_r64>(A_);
}
void slate_HermitianMatrix_destroy_r64(slate_HermitianMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<double>*>(A);
    delete A_;
}
void slate_HermitianMatrix_insertLocalTiles_r64(slate_HermitianMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<double>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_HermitianMatrix_mt_r64(slate_HermitianMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<double>*>(A);
    return(A_->mt());
}
int64_t slate_HermitianMatrix_nt_r64(slate_HermitianMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<double>*>(A);
    return(A_->nt());
}
int64_t slate_HermitianMatrix_m_r64(slate_HermitianMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<double>*>(A);
    return(A_->m());
}
int64_t slate_HermitianMatrix_n_r64(slate_HermitianMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<double>*>(A);
    return(A_->n());
}
bool slate_HermitianMatrix_tileIsLocal_r64(slate_HermitianMatrix_r64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<double>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_r64 slate_HermitianMatrix_at_r64(slate_HermitianMatrix_r64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<double>*>(A);
    slate::Tile<double> T = A_->at(i, j);
    slate_Tile_r64 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<double>));
    return(T_);
}
void slate_HermitianMatrix_transpose_in_place_r64(slate_HermitianMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<double>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_HermitianMatrix_conjTranspose_in_place_r64(slate_HermitianMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<double>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_HermitianMatrix_c32 slate_HermitianMatrix_create_c32(slate_Uplo uplo, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::HermitianMatrix<std::complex<float>>(slate::uplo2cpp(uplo), n, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_HermitianMatrix_c32>(A);
}
slate_HermitianMatrix_c32 slate_HermitianMatrix_create_fromScaLAPACK_c32(slate_Uplo uplo, int64_t n, float _Complex* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A_ = new slate::HermitianMatrix<std::complex<float>>();
    (*A_).fromScaLAPACK(slate::uplo2cpp(uplo), n, (std::complex<float>*)A, lda, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_HermitianMatrix_c32>(A_);
}
void slate_HermitianMatrix_destroy_c32(slate_HermitianMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<std::complex<float>>*>(A);
    delete A_;
}
void slate_HermitianMatrix_insertLocalTiles_c32(slate_HermitianMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<std::complex<float>>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_HermitianMatrix_mt_c32(slate_HermitianMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<std::complex<float>>*>(A);
    return(A_->mt());
}
int64_t slate_HermitianMatrix_nt_c32(slate_HermitianMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<std::complex<float>>*>(A);
    return(A_->nt());
}
int64_t slate_HermitianMatrix_m_c32(slate_HermitianMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<std::complex<float>>*>(A);
    return(A_->m());
}
int64_t slate_HermitianMatrix_n_c32(slate_HermitianMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<std::complex<float>>*>(A);
    return(A_->n());
}
bool slate_HermitianMatrix_tileIsLocal_c32(slate_HermitianMatrix_c32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<std::complex<float>>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_c32 slate_HermitianMatrix_at_c32(slate_HermitianMatrix_c32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<std::complex<float>>*>(A);
    slate::Tile<std::complex<float>> T = A_->at(i, j);
    slate_Tile_c32 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<float>>));
    return(T_);
}
void slate_HermitianMatrix_transpose_in_place_c32(slate_HermitianMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<std::complex<float>>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_HermitianMatrix_conjTranspose_in_place_c32(slate_HermitianMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<std::complex<float>>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_HermitianMatrix_c64 slate_HermitianMatrix_create_c64(slate_Uplo uplo, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::HermitianMatrix<std::complex<double>>(slate::uplo2cpp(uplo), n, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_HermitianMatrix_c64>(A);
}
slate_HermitianMatrix_c64 slate_HermitianMatrix_create_fromScaLAPACK_c64(slate_Uplo uplo, int64_t n, double _Complex* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A_ = new slate::HermitianMatrix<std::complex<double>>();
    (*A_).fromScaLAPACK(slate::uplo2cpp(uplo), n, (std::complex<double>*)A, lda, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_HermitianMatrix_c64>(A_);
}
void slate_HermitianMatrix_destroy_c64(slate_HermitianMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<std::complex<double>>*>(A);
    delete A_;
}
void slate_HermitianMatrix_insertLocalTiles_c64(slate_HermitianMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<std::complex<double>>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_HermitianMatrix_mt_c64(slate_HermitianMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<std::complex<double>>*>(A);
    return(A_->mt());
}
int64_t slate_HermitianMatrix_nt_c64(slate_HermitianMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<std::complex<double>>*>(A);
    return(A_->nt());
}
int64_t slate_HermitianMatrix_m_c64(slate_HermitianMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<std::complex<double>>*>(A);
    return(A_->m());
}
int64_t slate_HermitianMatrix_n_c64(slate_HermitianMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<std::complex<double>>*>(A);
    return(A_->n());
}
bool slate_HermitianMatrix_tileIsLocal_c64(slate_HermitianMatrix_c64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<std::complex<double>>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_c64 slate_HermitianMatrix_at_c64(slate_HermitianMatrix_c64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<std::complex<double>>*>(A);
    slate::Tile<std::complex<double>> T = A_->at(i, j);
    slate_Tile_c64 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<double>>));
    return(T_);
}
void slate_HermitianMatrix_transpose_in_place_c64(slate_HermitianMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<std::complex<double>>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_HermitianMatrix_conjTranspose_in_place_c64(slate_HermitianMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianMatrix<std::complex<double>>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
//------------------------------------------------------------------------------
slate_HermitianBandMatrix_r32 slate_HermitianBandMatrix_create_r32(slate_Uplo uplo, int64_t n, int64_t kd, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::HermitianBandMatrix<float>(slate::uplo2cpp(uplo), n, kd, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_HermitianBandMatrix_r32>(A);
}
void slate_HermitianBandMatrix_destroy_r32(slate_HermitianBandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<float>*>(A);
    delete A_;
}
void slate_HermitianBandMatrix_insertLocalTiles_r32(slate_HermitianBandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<float>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_HermitianBandMatrix_mt_r32(slate_HermitianBandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<float>*>(A);
    return(A_->mt());
}
int64_t slate_HermitianBandMatrix_nt_r32(slate_HermitianBandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<float>*>(A);
    return(A_->nt());
}
int64_t slate_HermitianBandMatrix_m_r32(slate_HermitianBandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<float>*>(A);
    return(A_->m());
}
int64_t slate_HermitianBandMatrix_n_r32(slate_HermitianBandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<float>*>(A);
    return(A_->n());
}
bool slate_HermitianBandMatrix_tileIsLocal_r32(slate_HermitianBandMatrix_r32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<float>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_r32 slate_HermitianBandMatrix_at_r32(slate_HermitianBandMatrix_r32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<float>*>(A);
    slate::Tile<float> T = A_->at(i, j);
    slate_Tile_r32 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<float>));
    return(T_);
}
void slate_HermitianBandMatrix_transpose_in_place_r32(slate_HermitianBandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<float>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_HermitianBandMatrix_conjTranspose_in_place_r32(slate_HermitianBandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<float>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_HermitianBandMatrix_r64 slate_HermitianBandMatrix_create_r64(slate_Uplo uplo, int64_t n, int64_t kd, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::HermitianBandMatrix<double>(slate::uplo2cpp(uplo), n, kd, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_HermitianBandMatrix_r64>(A);
}
void slate_HermitianBandMatrix_destroy_r64(slate_HermitianBandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<double>*>(A);
    delete A_;
}
void slate_HermitianBandMatrix_insertLocalTiles_r64(slate_HermitianBandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<double>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_HermitianBandMatrix_mt_r64(slate_HermitianBandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<double>*>(A);
    return(A_->mt());
}
int64_t slate_HermitianBandMatrix_nt_r64(slate_HermitianBandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<double>*>(A);
    return(A_->nt());
}
int64_t slate_HermitianBandMatrix_m_r64(slate_HermitianBandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<double>*>(A);
    return(A_->m());
}
int64_t slate_HermitianBandMatrix_n_r64(slate_HermitianBandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<double>*>(A);
    return(A_->n());
}
bool slate_HermitianBandMatrix_tileIsLocal_r64(slate_HermitianBandMatrix_r64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<double>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_r64 slate_HermitianBandMatrix_at_r64(slate_HermitianBandMatrix_r64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<double>*>(A);
    slate::Tile<double> T = A_->at(i, j);
    slate_Tile_r64 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<double>));
    return(T_);
}
void slate_HermitianBandMatrix_transpose_in_place_r64(slate_HermitianBandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<double>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_HermitianBandMatrix_conjTranspose_in_place_r64(slate_HermitianBandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<double>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_HermitianBandMatrix_c32 slate_HermitianBandMatrix_create_c32(slate_Uplo uplo, int64_t n, int64_t kd, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::HermitianBandMatrix<std::complex<float>>(slate::uplo2cpp(uplo), n, kd, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_HermitianBandMatrix_c32>(A);
}
void slate_HermitianBandMatrix_destroy_c32(slate_HermitianBandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<std::complex<float>>*>(A);
    delete A_;
}
void slate_HermitianBandMatrix_insertLocalTiles_c32(slate_HermitianBandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<std::complex<float>>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_HermitianBandMatrix_mt_c32(slate_HermitianBandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<std::complex<float>>*>(A);
    return(A_->mt());
}
int64_t slate_HermitianBandMatrix_nt_c32(slate_HermitianBandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<std::complex<float>>*>(A);
    return(A_->nt());
}
int64_t slate_HermitianBandMatrix_m_c32(slate_HermitianBandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<std::complex<float>>*>(A);
    return(A_->m());
}
int64_t slate_HermitianBandMatrix_n_c32(slate_HermitianBandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<std::complex<float>>*>(A);
    return(A_->n());
}
bool slate_HermitianBandMatrix_tileIsLocal_c32(slate_HermitianBandMatrix_c32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<std::complex<float>>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_c32 slate_HermitianBandMatrix_at_c32(slate_HermitianBandMatrix_c32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<std::complex<float>>*>(A);
    slate::Tile<std::complex<float>> T = A_->at(i, j);
    slate_Tile_c32 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<float>>));
    return(T_);
}
void slate_HermitianBandMatrix_transpose_in_place_c32(slate_HermitianBandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<std::complex<float>>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_HermitianBandMatrix_conjTranspose_in_place_c32(slate_HermitianBandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<std::complex<float>>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_HermitianBandMatrix_c64 slate_HermitianBandMatrix_create_c64(slate_Uplo uplo, int64_t n, int64_t kd, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::HermitianBandMatrix<std::complex<double>>(slate::uplo2cpp(uplo), n, kd, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_HermitianBandMatrix_c64>(A);
}
void slate_HermitianBandMatrix_destroy_c64(slate_HermitianBandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<std::complex<double>>*>(A);
    delete A_;
}
void slate_HermitianBandMatrix_insertLocalTiles_c64(slate_HermitianBandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<std::complex<double>>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_HermitianBandMatrix_mt_c64(slate_HermitianBandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<std::complex<double>>*>(A);
    return(A_->mt());
}
int64_t slate_HermitianBandMatrix_nt_c64(slate_HermitianBandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<std::complex<double>>*>(A);
    return(A_->nt());
}
int64_t slate_HermitianBandMatrix_m_c64(slate_HermitianBandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<std::complex<double>>*>(A);
    return(A_->m());
}
int64_t slate_HermitianBandMatrix_n_c64(slate_HermitianBandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<std::complex<double>>*>(A);
    return(A_->n());
}
bool slate_HermitianBandMatrix_tileIsLocal_c64(slate_HermitianBandMatrix_c64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<std::complex<double>>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_c64 slate_HermitianBandMatrix_at_c64(slate_HermitianBandMatrix_c64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<std::complex<double>>*>(A);
    slate::Tile<std::complex<double>> T = A_->at(i, j);
    slate_Tile_c64 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<double>>));
    return(T_);
}
void slate_HermitianBandMatrix_transpose_in_place_c64(slate_HermitianBandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<std::complex<double>>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_HermitianBandMatrix_conjTranspose_in_place_c64(slate_HermitianBandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::HermitianBandMatrix<std::complex<double>>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
//------------------------------------------------------------------------------
slate_TriangularMatrix_r32 slate_TriangularMatrix_create_r32(slate_Uplo uplo, slate_Diag diag, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::TriangularMatrix<float>(slate::uplo2cpp(uplo), slate::diag2cpp(diag), n, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_TriangularMatrix_r32>(A);
}
slate_TriangularMatrix_r32 slate_TriangularMatrix_create_fromScaLAPACK_r32(slate_Uplo uplo, slate_Diag diag, int64_t n, float* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A_ = new slate::TriangularMatrix<float>();
    (*A_).fromScaLAPACK(slate::uplo2cpp(uplo), slate::diag2cpp(diag), n, (float*)A, lda, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_TriangularMatrix_r32>(A_);
}
void slate_TriangularMatrix_destroy_r32(slate_TriangularMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<float>*>(A);
    delete A_;
}
void slate_TriangularMatrix_insertLocalTiles_r32(slate_TriangularMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<float>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_TriangularMatrix_mt_r32(slate_TriangularMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<float>*>(A);
    return(A_->mt());
}
int64_t slate_TriangularMatrix_nt_r32(slate_TriangularMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<float>*>(A);
    return(A_->nt());
}
int64_t slate_TriangularMatrix_m_r32(slate_TriangularMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<float>*>(A);
    return(A_->m());
}
int64_t slate_TriangularMatrix_n_r32(slate_TriangularMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<float>*>(A);
    return(A_->n());
}
bool slate_TriangularMatrix_tileIsLocal_r32(slate_TriangularMatrix_r32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<float>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_r32 slate_TriangularMatrix_at_r32(slate_TriangularMatrix_r32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<float>*>(A);
    slate::Tile<float> T = A_->at(i, j);
    slate_Tile_r32 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<float>));
    return(T_);
}
void slate_TriangularMatrix_transpose_in_place_r32(slate_TriangularMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<float>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_TriangularMatrix_conjTranspose_in_place_r32(slate_TriangularMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<float>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_TriangularMatrix_r64 slate_TriangularMatrix_create_r64(slate_Uplo uplo, slate_Diag diag, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::TriangularMatrix<double>(slate::uplo2cpp(uplo), slate::diag2cpp(diag), n, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_TriangularMatrix_r64>(A);
}
slate_TriangularMatrix_r64 slate_TriangularMatrix_create_fromScaLAPACK_r64(slate_Uplo uplo, slate_Diag diag, int64_t n, double* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A_ = new slate::TriangularMatrix<double>();
    (*A_).fromScaLAPACK(slate::uplo2cpp(uplo), slate::diag2cpp(diag), n, (double*)A, lda, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_TriangularMatrix_r64>(A_);
}
void slate_TriangularMatrix_destroy_r64(slate_TriangularMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<double>*>(A);
    delete A_;
}
void slate_TriangularMatrix_insertLocalTiles_r64(slate_TriangularMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<double>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_TriangularMatrix_mt_r64(slate_TriangularMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<double>*>(A);
    return(A_->mt());
}
int64_t slate_TriangularMatrix_nt_r64(slate_TriangularMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<double>*>(A);
    return(A_->nt());
}
int64_t slate_TriangularMatrix_m_r64(slate_TriangularMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<double>*>(A);
    return(A_->m());
}
int64_t slate_TriangularMatrix_n_r64(slate_TriangularMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<double>*>(A);
    return(A_->n());
}
bool slate_TriangularMatrix_tileIsLocal_r64(slate_TriangularMatrix_r64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<double>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_r64 slate_TriangularMatrix_at_r64(slate_TriangularMatrix_r64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<double>*>(A);
    slate::Tile<double> T = A_->at(i, j);
    slate_Tile_r64 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<double>));
    return(T_);
}
void slate_TriangularMatrix_transpose_in_place_r64(slate_TriangularMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<double>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_TriangularMatrix_conjTranspose_in_place_r64(slate_TriangularMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<double>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_TriangularMatrix_c32 slate_TriangularMatrix_create_c32(slate_Uplo uplo, slate_Diag diag, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::TriangularMatrix<std::complex<float>>(slate::uplo2cpp(uplo), slate::diag2cpp(diag), n, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_TriangularMatrix_c32>(A);
}
slate_TriangularMatrix_c32 slate_TriangularMatrix_create_fromScaLAPACK_c32(slate_Uplo uplo, slate_Diag diag, int64_t n, float _Complex* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A_ = new slate::TriangularMatrix<std::complex<float>>();
    (*A_).fromScaLAPACK(slate::uplo2cpp(uplo), slate::diag2cpp(diag), n, (std::complex<float>*)A, lda, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_TriangularMatrix_c32>(A_);
}
void slate_TriangularMatrix_destroy_c32(slate_TriangularMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<std::complex<float>>*>(A);
    delete A_;
}
void slate_TriangularMatrix_insertLocalTiles_c32(slate_TriangularMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<std::complex<float>>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_TriangularMatrix_mt_c32(slate_TriangularMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<std::complex<float>>*>(A);
    return(A_->mt());
}
int64_t slate_TriangularMatrix_nt_c32(slate_TriangularMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<std::complex<float>>*>(A);
    return(A_->nt());
}
int64_t slate_TriangularMatrix_m_c32(slate_TriangularMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<std::complex<float>>*>(A);
    return(A_->m());
}
int64_t slate_TriangularMatrix_n_c32(slate_TriangularMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<std::complex<float>>*>(A);
    return(A_->n());
}
bool slate_TriangularMatrix_tileIsLocal_c32(slate_TriangularMatrix_c32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<std::complex<float>>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_c32 slate_TriangularMatrix_at_c32(slate_TriangularMatrix_c32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<std::complex<float>>*>(A);
    slate::Tile<std::complex<float>> T = A_->at(i, j);
    slate_Tile_c32 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<float>>));
    return(T_);
}
void slate_TriangularMatrix_transpose_in_place_c32(slate_TriangularMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<std::complex<float>>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_TriangularMatrix_conjTranspose_in_place_c32(slate_TriangularMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<std::complex<float>>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_TriangularMatrix_c64 slate_TriangularMatrix_create_c64(slate_Uplo uplo, slate_Diag diag, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::TriangularMatrix<std::complex<double>>(slate::uplo2cpp(uplo), slate::diag2cpp(diag), n, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_TriangularMatrix_c64>(A);
}
slate_TriangularMatrix_c64 slate_TriangularMatrix_create_fromScaLAPACK_c64(slate_Uplo uplo, slate_Diag diag, int64_t n, double _Complex* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A_ = new slate::TriangularMatrix<std::complex<double>>();
    (*A_).fromScaLAPACK(slate::uplo2cpp(uplo), slate::diag2cpp(diag), n, (std::complex<double>*)A, lda, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_TriangularMatrix_c64>(A_);
}
void slate_TriangularMatrix_destroy_c64(slate_TriangularMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<std::complex<double>>*>(A);
    delete A_;
}
void slate_TriangularMatrix_insertLocalTiles_c64(slate_TriangularMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<std::complex<double>>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_TriangularMatrix_mt_c64(slate_TriangularMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<std::complex<double>>*>(A);
    return(A_->mt());
}
int64_t slate_TriangularMatrix_nt_c64(slate_TriangularMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<std::complex<double>>*>(A);
    return(A_->nt());
}
int64_t slate_TriangularMatrix_m_c64(slate_TriangularMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<std::complex<double>>*>(A);
    return(A_->m());
}
int64_t slate_TriangularMatrix_n_c64(slate_TriangularMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<std::complex<double>>*>(A);
    return(A_->n());
}
bool slate_TriangularMatrix_tileIsLocal_c64(slate_TriangularMatrix_c64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<std::complex<double>>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_c64 slate_TriangularMatrix_at_c64(slate_TriangularMatrix_c64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<std::complex<double>>*>(A);
    slate::Tile<std::complex<double>> T = A_->at(i, j);
    slate_Tile_c64 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<double>>));
    return(T_);
}
void slate_TriangularMatrix_transpose_in_place_c64(slate_TriangularMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<std::complex<double>>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_TriangularMatrix_conjTranspose_in_place_c64(slate_TriangularMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularMatrix<std::complex<double>>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
//------------------------------------------------------------------------------
slate_TriangularBandMatrix_r32 slate_TriangularBandMatrix_create_r32(slate_Uplo uplo, slate_Diag diag, int64_t n, int64_t kd, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::TriangularBandMatrix<float>(slate::uplo2cpp(uplo), slate::diag2cpp(diag), n, kd, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_TriangularBandMatrix_r32>(A);
}
void slate_TriangularBandMatrix_destroy_r32(slate_TriangularBandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<float>*>(A);
    delete A_;
}
void slate_TriangularBandMatrix_insertLocalTiles_r32(slate_TriangularBandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<float>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_TriangularBandMatrix_mt_r32(slate_TriangularBandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<float>*>(A);
    return(A_->mt());
}
int64_t slate_TriangularBandMatrix_nt_r32(slate_TriangularBandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<float>*>(A);
    return(A_->nt());
}
int64_t slate_TriangularBandMatrix_m_r32(slate_TriangularBandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<float>*>(A);
    return(A_->m());
}
int64_t slate_TriangularBandMatrix_n_r32(slate_TriangularBandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<float>*>(A);
    return(A_->n());
}
bool slate_TriangularBandMatrix_tileIsLocal_r32(slate_TriangularBandMatrix_r32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<float>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_r32 slate_TriangularBandMatrix_at_r32(slate_TriangularBandMatrix_r32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<float>*>(A);
    slate::Tile<float> T = A_->at(i, j);
    slate_Tile_r32 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<float>));
    return(T_);
}
void slate_TriangularBandMatrix_transpose_in_place_r32(slate_TriangularBandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<float>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_TriangularBandMatrix_conjTranspose_in_place_r32(slate_TriangularBandMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<float>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_TriangularBandMatrix_r64 slate_TriangularBandMatrix_create_r64(slate_Uplo uplo, slate_Diag diag, int64_t n, int64_t kd, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::TriangularBandMatrix<double>(slate::uplo2cpp(uplo), slate::diag2cpp(diag), n, kd, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_TriangularBandMatrix_r64>(A);
}
void slate_TriangularBandMatrix_destroy_r64(slate_TriangularBandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<double>*>(A);
    delete A_;
}
void slate_TriangularBandMatrix_insertLocalTiles_r64(slate_TriangularBandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<double>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_TriangularBandMatrix_mt_r64(slate_TriangularBandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<double>*>(A);
    return(A_->mt());
}
int64_t slate_TriangularBandMatrix_nt_r64(slate_TriangularBandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<double>*>(A);
    return(A_->nt());
}
int64_t slate_TriangularBandMatrix_m_r64(slate_TriangularBandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<double>*>(A);
    return(A_->m());
}
int64_t slate_TriangularBandMatrix_n_r64(slate_TriangularBandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<double>*>(A);
    return(A_->n());
}
bool slate_TriangularBandMatrix_tileIsLocal_r64(slate_TriangularBandMatrix_r64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<double>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_r64 slate_TriangularBandMatrix_at_r64(slate_TriangularBandMatrix_r64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<double>*>(A);
    slate::Tile<double> T = A_->at(i, j);
    slate_Tile_r64 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<double>));
    return(T_);
}
void slate_TriangularBandMatrix_transpose_in_place_r64(slate_TriangularBandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<double>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_TriangularBandMatrix_conjTranspose_in_place_r64(slate_TriangularBandMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<double>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_TriangularBandMatrix_c32 slate_TriangularBandMatrix_create_c32(slate_Uplo uplo, slate_Diag diag, int64_t n, int64_t kd, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::TriangularBandMatrix<std::complex<float>>(slate::uplo2cpp(uplo), slate::diag2cpp(diag), n, kd, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_TriangularBandMatrix_c32>(A);
}
void slate_TriangularBandMatrix_destroy_c32(slate_TriangularBandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<std::complex<float>>*>(A);
    delete A_;
}
void slate_TriangularBandMatrix_insertLocalTiles_c32(slate_TriangularBandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<std::complex<float>>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_TriangularBandMatrix_mt_c32(slate_TriangularBandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<std::complex<float>>*>(A);
    return(A_->mt());
}
int64_t slate_TriangularBandMatrix_nt_c32(slate_TriangularBandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<std::complex<float>>*>(A);
    return(A_->nt());
}
int64_t slate_TriangularBandMatrix_m_c32(slate_TriangularBandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<std::complex<float>>*>(A);
    return(A_->m());
}
int64_t slate_TriangularBandMatrix_n_c32(slate_TriangularBandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<std::complex<float>>*>(A);
    return(A_->n());
}
bool slate_TriangularBandMatrix_tileIsLocal_c32(slate_TriangularBandMatrix_c32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<std::complex<float>>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_c32 slate_TriangularBandMatrix_at_c32(slate_TriangularBandMatrix_c32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<std::complex<float>>*>(A);
    slate::Tile<std::complex<float>> T = A_->at(i, j);
    slate_Tile_c32 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<float>>));
    return(T_);
}
void slate_TriangularBandMatrix_transpose_in_place_c32(slate_TriangularBandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<std::complex<float>>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_TriangularBandMatrix_conjTranspose_in_place_c32(slate_TriangularBandMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<std::complex<float>>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_TriangularBandMatrix_c64 slate_TriangularBandMatrix_create_c64(slate_Uplo uplo, slate_Diag diag, int64_t n, int64_t kd, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::TriangularBandMatrix<std::complex<double>>(slate::uplo2cpp(uplo), slate::diag2cpp(diag), n, kd, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_TriangularBandMatrix_c64>(A);
}
void slate_TriangularBandMatrix_destroy_c64(slate_TriangularBandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<std::complex<double>>*>(A);
    delete A_;
}
void slate_TriangularBandMatrix_insertLocalTiles_c64(slate_TriangularBandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<std::complex<double>>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_TriangularBandMatrix_mt_c64(slate_TriangularBandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<std::complex<double>>*>(A);
    return(A_->mt());
}
int64_t slate_TriangularBandMatrix_nt_c64(slate_TriangularBandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<std::complex<double>>*>(A);
    return(A_->nt());
}
int64_t slate_TriangularBandMatrix_m_c64(slate_TriangularBandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<std::complex<double>>*>(A);
    return(A_->m());
}
int64_t slate_TriangularBandMatrix_n_c64(slate_TriangularBandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<std::complex<double>>*>(A);
    return(A_->n());
}
bool slate_TriangularBandMatrix_tileIsLocal_c64(slate_TriangularBandMatrix_c64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<std::complex<double>>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_c64 slate_TriangularBandMatrix_at_c64(slate_TriangularBandMatrix_c64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<std::complex<double>>*>(A);
    slate::Tile<std::complex<double>> T = A_->at(i, j);
    slate_Tile_c64 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<double>>));
    return(T_);
}
void slate_TriangularBandMatrix_transpose_in_place_c64(slate_TriangularBandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<std::complex<double>>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_TriangularBandMatrix_conjTranspose_in_place_c64(slate_TriangularBandMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TriangularBandMatrix<std::complex<double>>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
//------------------------------------------------------------------------------
slate_SymmetricMatrix_r32 slate_SymmetricMatrix_create_r32(slate_Uplo uplo, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::SymmetricMatrix<float>(slate::uplo2cpp(uplo), n, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_SymmetricMatrix_r32>(A);
}
slate_SymmetricMatrix_r32 slate_SymmetricMatrix_create_fromScaLAPACK_r32(slate_Uplo uplo, int64_t n, float* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A_ = new slate::SymmetricMatrix<float>();
    (*A_).fromScaLAPACK(slate::uplo2cpp(uplo), n, (float*)A, lda, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_SymmetricMatrix_r32>(A_);
}
void slate_SymmetricMatrix_destroy_r32(slate_SymmetricMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<float>*>(A);
    delete A_;
}
void slate_SymmetricMatrix_insertLocalTiles_r32(slate_SymmetricMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<float>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_SymmetricMatrix_mt_r32(slate_SymmetricMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<float>*>(A);
    return(A_->mt());
}
int64_t slate_SymmetricMatrix_nt_r32(slate_SymmetricMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<float>*>(A);
    return(A_->nt());
}
int64_t slate_SymmetricMatrix_m_r32(slate_SymmetricMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<float>*>(A);
    return(A_->m());
}
int64_t slate_SymmetricMatrix_n_r32(slate_SymmetricMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<float>*>(A);
    return(A_->n());
}
bool slate_SymmetricMatrix_tileIsLocal_r32(slate_SymmetricMatrix_r32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<float>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_r32 slate_SymmetricMatrix_at_r32(slate_SymmetricMatrix_r32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<float>*>(A);
    slate::Tile<float> T = A_->at(i, j);
    slate_Tile_r32 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<float>));
    return(T_);
}
void slate_SymmetricMatrix_transpose_in_place_r32(slate_SymmetricMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<float>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_SymmetricMatrix_conjTranspose_in_place_r32(slate_SymmetricMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<float>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_SymmetricMatrix_r64 slate_SymmetricMatrix_create_r64(slate_Uplo uplo, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::SymmetricMatrix<double>(slate::uplo2cpp(uplo), n, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_SymmetricMatrix_r64>(A);
}
slate_SymmetricMatrix_r64 slate_SymmetricMatrix_create_fromScaLAPACK_r64(slate_Uplo uplo, int64_t n, double* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A_ = new slate::SymmetricMatrix<double>();
    (*A_).fromScaLAPACK(slate::uplo2cpp(uplo), n, (double*)A, lda, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_SymmetricMatrix_r64>(A_);
}
void slate_SymmetricMatrix_destroy_r64(slate_SymmetricMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<double>*>(A);
    delete A_;
}
void slate_SymmetricMatrix_insertLocalTiles_r64(slate_SymmetricMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<double>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_SymmetricMatrix_mt_r64(slate_SymmetricMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<double>*>(A);
    return(A_->mt());
}
int64_t slate_SymmetricMatrix_nt_r64(slate_SymmetricMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<double>*>(A);
    return(A_->nt());
}
int64_t slate_SymmetricMatrix_m_r64(slate_SymmetricMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<double>*>(A);
    return(A_->m());
}
int64_t slate_SymmetricMatrix_n_r64(slate_SymmetricMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<double>*>(A);
    return(A_->n());
}
bool slate_SymmetricMatrix_tileIsLocal_r64(slate_SymmetricMatrix_r64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<double>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_r64 slate_SymmetricMatrix_at_r64(slate_SymmetricMatrix_r64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<double>*>(A);
    slate::Tile<double> T = A_->at(i, j);
    slate_Tile_r64 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<double>));
    return(T_);
}
void slate_SymmetricMatrix_transpose_in_place_r64(slate_SymmetricMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<double>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_SymmetricMatrix_conjTranspose_in_place_r64(slate_SymmetricMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<double>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_SymmetricMatrix_c32 slate_SymmetricMatrix_create_c32(slate_Uplo uplo, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::SymmetricMatrix<std::complex<float>>(slate::uplo2cpp(uplo), n, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_SymmetricMatrix_c32>(A);
}
slate_SymmetricMatrix_c32 slate_SymmetricMatrix_create_fromScaLAPACK_c32(slate_Uplo uplo, int64_t n, float _Complex* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A_ = new slate::SymmetricMatrix<std::complex<float>>();
    (*A_).fromScaLAPACK(slate::uplo2cpp(uplo), n, (std::complex<float>*)A, lda, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_SymmetricMatrix_c32>(A_);
}
void slate_SymmetricMatrix_destroy_c32(slate_SymmetricMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<std::complex<float>>*>(A);
    delete A_;
}
void slate_SymmetricMatrix_insertLocalTiles_c32(slate_SymmetricMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<std::complex<float>>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_SymmetricMatrix_mt_c32(slate_SymmetricMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<std::complex<float>>*>(A);
    return(A_->mt());
}
int64_t slate_SymmetricMatrix_nt_c32(slate_SymmetricMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<std::complex<float>>*>(A);
    return(A_->nt());
}
int64_t slate_SymmetricMatrix_m_c32(slate_SymmetricMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<std::complex<float>>*>(A);
    return(A_->m());
}
int64_t slate_SymmetricMatrix_n_c32(slate_SymmetricMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<std::complex<float>>*>(A);
    return(A_->n());
}
bool slate_SymmetricMatrix_tileIsLocal_c32(slate_SymmetricMatrix_c32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<std::complex<float>>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_c32 slate_SymmetricMatrix_at_c32(slate_SymmetricMatrix_c32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<std::complex<float>>*>(A);
    slate::Tile<std::complex<float>> T = A_->at(i, j);
    slate_Tile_c32 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<float>>));
    return(T_);
}
void slate_SymmetricMatrix_transpose_in_place_c32(slate_SymmetricMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<std::complex<float>>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_SymmetricMatrix_conjTranspose_in_place_c32(slate_SymmetricMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<std::complex<float>>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_SymmetricMatrix_c64 slate_SymmetricMatrix_create_c64(slate_Uplo uplo, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::SymmetricMatrix<std::complex<double>>(slate::uplo2cpp(uplo), n, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_SymmetricMatrix_c64>(A);
}
slate_SymmetricMatrix_c64 slate_SymmetricMatrix_create_fromScaLAPACK_c64(slate_Uplo uplo, int64_t n, double _Complex* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A_ = new slate::SymmetricMatrix<std::complex<double>>();
    (*A_).fromScaLAPACK(slate::uplo2cpp(uplo), n, (std::complex<double>*)A, lda, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_SymmetricMatrix_c64>(A_);
}
void slate_SymmetricMatrix_destroy_c64(slate_SymmetricMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<std::complex<double>>*>(A);
    delete A_;
}
void slate_SymmetricMatrix_insertLocalTiles_c64(slate_SymmetricMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<std::complex<double>>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_SymmetricMatrix_mt_c64(slate_SymmetricMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<std::complex<double>>*>(A);
    return(A_->mt());
}
int64_t slate_SymmetricMatrix_nt_c64(slate_SymmetricMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<std::complex<double>>*>(A);
    return(A_->nt());
}
int64_t slate_SymmetricMatrix_m_c64(slate_SymmetricMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<std::complex<double>>*>(A);
    return(A_->m());
}
int64_t slate_SymmetricMatrix_n_c64(slate_SymmetricMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<std::complex<double>>*>(A);
    return(A_->n());
}
bool slate_SymmetricMatrix_tileIsLocal_c64(slate_SymmetricMatrix_c64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<std::complex<double>>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_c64 slate_SymmetricMatrix_at_c64(slate_SymmetricMatrix_c64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<std::complex<double>>*>(A);
    slate::Tile<std::complex<double>> T = A_->at(i, j);
    slate_Tile_c64 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<double>>));
    return(T_);
}
void slate_SymmetricMatrix_transpose_in_place_c64(slate_SymmetricMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<std::complex<double>>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_SymmetricMatrix_conjTranspose_in_place_c64(slate_SymmetricMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::SymmetricMatrix<std::complex<double>>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
//------------------------------------------------------------------------------
slate_TrapezoidMatrix_r32 slate_TrapezoidMatrix_create_r32(slate_Uplo uplo, slate_Diag diag, int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::TrapezoidMatrix<float>(slate::uplo2cpp(uplo), slate::diag2cpp(diag), m, n, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_TrapezoidMatrix_r32>(A);
}
slate_TrapezoidMatrix_r32 slate_TrapezoidMatrix_create_fromScaLAPACK_r32(slate_Uplo uplo, slate_Diag diag, int64_t m, int64_t n, float* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A_ = new slate::TrapezoidMatrix<float>();
    (*A_).fromScaLAPACK(slate::uplo2cpp(uplo), slate::diag2cpp(diag), m, n, (float*)A, lda, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_TrapezoidMatrix_r32>(A_);
}
void slate_TrapezoidMatrix_destroy_r32(slate_TrapezoidMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<float>*>(A);
    delete A_;
}
void slate_TrapezoidMatrix_insertLocalTiles_r32(slate_TrapezoidMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<float>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_TrapezoidMatrix_mt_r32(slate_TrapezoidMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<float>*>(A);
    return(A_->mt());
}
int64_t slate_TrapezoidMatrix_nt_r32(slate_TrapezoidMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<float>*>(A);
    return(A_->nt());
}
int64_t slate_TrapezoidMatrix_m_r32(slate_TrapezoidMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<float>*>(A);
    return(A_->m());
}
int64_t slate_TrapezoidMatrix_n_r32(slate_TrapezoidMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<float>*>(A);
    return(A_->n());
}
bool slate_TrapezoidMatrix_tileIsLocal_r32(slate_TrapezoidMatrix_r32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<float>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_r32 slate_TrapezoidMatrix_at_r32(slate_TrapezoidMatrix_r32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<float>*>(A);
    slate::Tile<float> T = A_->at(i, j);
    slate_Tile_r32 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<float>));
    return(T_);
}
void slate_TrapezoidMatrix_transpose_in_place_r32(slate_TrapezoidMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<float>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_TrapezoidMatrix_conjTranspose_in_place_r32(slate_TrapezoidMatrix_r32 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<float>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_TrapezoidMatrix_r64 slate_TrapezoidMatrix_create_r64(slate_Uplo uplo, slate_Diag diag, int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::TrapezoidMatrix<double>(slate::uplo2cpp(uplo), slate::diag2cpp(diag), m, n, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_TrapezoidMatrix_r64>(A);
}
slate_TrapezoidMatrix_r64 slate_TrapezoidMatrix_create_fromScaLAPACK_r64(slate_Uplo uplo, slate_Diag diag, int64_t m, int64_t n, double* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A_ = new slate::TrapezoidMatrix<double>();
    (*A_).fromScaLAPACK(slate::uplo2cpp(uplo), slate::diag2cpp(diag), m, n, (double*)A, lda, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_TrapezoidMatrix_r64>(A_);
}
void slate_TrapezoidMatrix_destroy_r64(slate_TrapezoidMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<double>*>(A);
    delete A_;
}
void slate_TrapezoidMatrix_insertLocalTiles_r64(slate_TrapezoidMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<double>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_TrapezoidMatrix_mt_r64(slate_TrapezoidMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<double>*>(A);
    return(A_->mt());
}
int64_t slate_TrapezoidMatrix_nt_r64(slate_TrapezoidMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<double>*>(A);
    return(A_->nt());
}
int64_t slate_TrapezoidMatrix_m_r64(slate_TrapezoidMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<double>*>(A);
    return(A_->m());
}
int64_t slate_TrapezoidMatrix_n_r64(slate_TrapezoidMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<double>*>(A);
    return(A_->n());
}
bool slate_TrapezoidMatrix_tileIsLocal_r64(slate_TrapezoidMatrix_r64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<double>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_r64 slate_TrapezoidMatrix_at_r64(slate_TrapezoidMatrix_r64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<double>*>(A);
    slate::Tile<double> T = A_->at(i, j);
    slate_Tile_r64 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<double>));
    return(T_);
}
void slate_TrapezoidMatrix_transpose_in_place_r64(slate_TrapezoidMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<double>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_TrapezoidMatrix_conjTranspose_in_place_r64(slate_TrapezoidMatrix_r64 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<double>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_TrapezoidMatrix_c32 slate_TrapezoidMatrix_create_c32(slate_Uplo uplo, slate_Diag diag, int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::TrapezoidMatrix<std::complex<float>>(slate::uplo2cpp(uplo), slate::diag2cpp(diag), m, n, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_TrapezoidMatrix_c32>(A);
}
slate_TrapezoidMatrix_c32 slate_TrapezoidMatrix_create_fromScaLAPACK_c32(slate_Uplo uplo, slate_Diag diag, int64_t m, int64_t n, float _Complex* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A_ = new slate::TrapezoidMatrix<std::complex<float>>();
    (*A_).fromScaLAPACK(slate::uplo2cpp(uplo), slate::diag2cpp(diag), m, n, (std::complex<float>*)A, lda, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_TrapezoidMatrix_c32>(A_);
}
void slate_TrapezoidMatrix_destroy_c32(slate_TrapezoidMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<std::complex<float>>*>(A);
    delete A_;
}
void slate_TrapezoidMatrix_insertLocalTiles_c32(slate_TrapezoidMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<std::complex<float>>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_TrapezoidMatrix_mt_c32(slate_TrapezoidMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<std::complex<float>>*>(A);
    return(A_->mt());
}
int64_t slate_TrapezoidMatrix_nt_c32(slate_TrapezoidMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<std::complex<float>>*>(A);
    return(A_->nt());
}
int64_t slate_TrapezoidMatrix_m_c32(slate_TrapezoidMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<std::complex<float>>*>(A);
    return(A_->m());
}
int64_t slate_TrapezoidMatrix_n_c32(slate_TrapezoidMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<std::complex<float>>*>(A);
    return(A_->n());
}
bool slate_TrapezoidMatrix_tileIsLocal_c32(slate_TrapezoidMatrix_c32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<std::complex<float>>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_c32 slate_TrapezoidMatrix_at_c32(slate_TrapezoidMatrix_c32 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<std::complex<float>>*>(A);
    slate::Tile<std::complex<float>> T = A_->at(i, j);
    slate_Tile_c32 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<float>>));
    return(T_);
}
void slate_TrapezoidMatrix_transpose_in_place_c32(slate_TrapezoidMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<std::complex<float>>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_TrapezoidMatrix_conjTranspose_in_place_c32(slate_TrapezoidMatrix_c32 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<std::complex<float>>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
slate_TrapezoidMatrix_c64 slate_TrapezoidMatrix_create_c64(slate_Uplo uplo, slate_Diag diag, int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A = new slate::TrapezoidMatrix<std::complex<double>>(slate::uplo2cpp(uplo), slate::diag2cpp(diag), m, n, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_TrapezoidMatrix_c64>(A);
}
slate_TrapezoidMatrix_c64 slate_TrapezoidMatrix_create_fromScaLAPACK_c64(slate_Uplo uplo, slate_Diag diag, int64_t m, int64_t n, double _Complex* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm)
{
    auto* A_ = new slate::TrapezoidMatrix<std::complex<double>>();
    (*A_).fromScaLAPACK(slate::uplo2cpp(uplo), slate::diag2cpp(diag), m, n, (std::complex<double>*)A, lda, nb, p, q, mpi_comm);
    return reinterpret_cast<slate_TrapezoidMatrix_c64>(A_);
}
void slate_TrapezoidMatrix_destroy_c64(slate_TrapezoidMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<std::complex<double>>*>(A);
    delete A_;
}
void slate_TrapezoidMatrix_insertLocalTiles_c64(slate_TrapezoidMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<std::complex<double>>*>(A);
    A_->insertLocalTiles();
}
int64_t slate_TrapezoidMatrix_mt_c64(slate_TrapezoidMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<std::complex<double>>*>(A);
    return(A_->mt());
}
int64_t slate_TrapezoidMatrix_nt_c64(slate_TrapezoidMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<std::complex<double>>*>(A);
    return(A_->nt());
}
int64_t slate_TrapezoidMatrix_m_c64(slate_TrapezoidMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<std::complex<double>>*>(A);
    return(A_->m());
}
int64_t slate_TrapezoidMatrix_n_c64(slate_TrapezoidMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<std::complex<double>>*>(A);
    return(A_->n());
}
bool slate_TrapezoidMatrix_tileIsLocal_c64(slate_TrapezoidMatrix_c64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<std::complex<double>>*>(A);
    return(A_->tileIsLocal(i, j));
}
slate_Tile_c64 slate_TrapezoidMatrix_at_c64(slate_TrapezoidMatrix_c64 A, int64_t i, int64_t j)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<std::complex<double>>*>(A);
    slate::Tile<std::complex<double>> T = A_->at(i, j);
    slate_Tile_c64 T_;
    std::memcpy(&T_, &T, sizeof(slate::Tile<std::complex<double>>));
    return(T_);
}
void slate_TrapezoidMatrix_transpose_in_place_c64(slate_TrapezoidMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<std::complex<double>>*>(A);
    *A_ = slate::transpose(*A_);
}
void slate_TrapezoidMatrix_conjTranspose_in_place_c64(slate_TrapezoidMatrix_c64 A)
{
    auto* A_ = reinterpret_cast<slate::TrapezoidMatrix<std::complex<double>>*>(A);
    *A_ = slate::conjTranspose(*A_);
}
//------------------------------------------------------------------------------
/// slate::TriangularFactors<float>
slate_TriangularFactors_r32 slate_TriangularFactors_create_r32()
{
    auto* T = new slate::TriangularFactors<float>();
    return reinterpret_cast<slate_TriangularFactors_r32>(T);
}
void slate_TriangularFactors_destroy_r32(slate_TriangularFactors_r32 T)
{
    auto* T_ = reinterpret_cast<slate::TriangularFactors<float>*>(T);
    delete T_;
}
/// slate::TriangularFactors<double>
slate_TriangularFactors_r64 slate_TriangularFactors_create_r64()
{
    auto* T = new slate::TriangularFactors<double>();
    return reinterpret_cast<slate_TriangularFactors_r64>(T);
}
void slate_TriangularFactors_destroy_r64(slate_TriangularFactors_r64 T)
{
    auto* T_ = reinterpret_cast<slate::TriangularFactors<double>*>(T);
    delete T_;
}
/// slate::TriangularFactors<std::complex<float>>
slate_TriangularFactors_c32 slate_TriangularFactors_create_c32()
{
    auto* T = new slate::TriangularFactors<std::complex<float>>();
    return reinterpret_cast<slate_TriangularFactors_c32>(T);
}
void slate_TriangularFactors_destroy_c32(slate_TriangularFactors_c32 T)
{
    auto* T_ = reinterpret_cast<slate::TriangularFactors<std::complex<float>>*>(T);
    delete T_;
}
/// slate::TriangularFactors<std::complex<double>>
slate_TriangularFactors_c64 slate_TriangularFactors_create_c64()
{
    auto* T = new slate::TriangularFactors<std::complex<double>>();
    return reinterpret_cast<slate_TriangularFactors_c64>(T);
}
void slate_TriangularFactors_destroy_c64(slate_TriangularFactors_c64 T)
{
    auto* T_ = reinterpret_cast<slate::TriangularFactors<std::complex<double>>*>(T);
    delete T_;
}
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
