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

#ifndef SLATE_SIMPLIFIED_API_HH
#define SLATE_SIMPLIFIED_API_HH

namespace slate {

//------------------------------------------------------------------------------
// Level 3 BLAS and LAPACK auxiliary

//-----------------------------------------
// multiply()

// gbmm
template <typename scalar_t>
void multiply(
    scalar_t alpha, BandMatrix<scalar_t>& A,
                        Matrix<scalar_t>& B,
    scalar_t beta,      Matrix<scalar_t>& C,
    Options const& opts = Options())
{
    gbmm(alpha, A, B, beta, C, opts);
}

// gemm
template <typename scalar_t>
void multiply(
    scalar_t alpha, Matrix<scalar_t>& A,
                    Matrix<scalar_t>& B,
    scalar_t beta,  Matrix<scalar_t>& C,
    Options const& opts = Options())
{
    gemm(alpha, A, B, beta, C, opts);
}

// Left hbmm
template <typename scalar_t>
void multiply(
    scalar_t alpha, HermitianBandMatrix<scalar_t>& A,
                                 Matrix<scalar_t>& B,
    scalar_t beta,               Matrix<scalar_t>& C,
    Options const& opts = Options())
{
    hbmm(Side::Left, alpha, A, B, beta, C, opts);
}

// Right hbmm
template <typename scalar_t>
void multiply(
    scalar_t alpha,              Matrix<scalar_t>& A,
                    HermitianBandMatrix<scalar_t>& B,
    scalar_t beta,               Matrix<scalar_t>& C,
    Options const& opts = Options())
{
    hbmm(Side::Right, alpha, B, A, beta, C, opts);
}

// Left hemm
template <typename scalar_t>
void multiply(
    scalar_t alpha, HermitianMatrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,           Matrix<scalar_t>& C,
    Options const& opts = Options())
{
    hemm(Side::Left, alpha, A, B, beta, C, opts);
}

// Right hemm
template <typename scalar_t>
void multiply(
    scalar_t alpha,          Matrix<scalar_t>& A,
                    HermitianMatrix<scalar_t>& B,
    scalar_t beta,           Matrix<scalar_t>& C,
    Options const& opts = Options())
{
    hemm(Side::Right, alpha, B, A, beta, C, opts);
}

// Left symm
template <typename scalar_t>
void multiply(
    scalar_t alpha, SymmetricMatrix<scalar_t>& A,
                             Matrix<scalar_t>& B,
    scalar_t beta,           Matrix<scalar_t>& C,
    Options const& opts = Options())
{
    symm(Side::Left, alpha, A, B, beta, C, opts);
}

// Right symm
template <typename scalar_t>
void multiply(
    scalar_t alpha,          Matrix<scalar_t>& A,
                    SymmetricMatrix<scalar_t>& B,
    scalar_t beta,           Matrix<scalar_t>& C,
    Options const& opts = Options())
{
    symm(Side::Right, alpha, B, A, beta, C, opts);
}

//-----------------------------------------
// triangular_multiply()

// Left trmm
template <typename scalar_t>
void triangular_multiply(
    scalar_t alpha, TriangularMatrix<scalar_t>& A,
                              Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    trmm(Side::Left, alpha, A, B, opts);
}

// Right trmm
template <typename scalar_t>
void triangular_multiply(
    scalar_t alpha,           Matrix<scalar_t>& A,
                    TriangularMatrix<scalar_t>& B,
    Options const& opts = Options())
{
    trmm(Side::Right, alpha, B, A, opts);
}

//-----------------------------------------
// triangular_solve()

// Left tbsm
template <typename scalar_t>
void triangular_solve(
    scalar_t alpha, TriangularBandMatrix<scalar_t>& A,
                                  Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    tbsm(Side::Left, alpha, A, B, opts);
}

// Right tbsm
template <typename scalar_t>
void triangular_solve(
    scalar_t alpha,               Matrix<scalar_t>& A,
                    TriangularBandMatrix<scalar_t>& B,
    Options const& opts = Options())
{
    tbsm(Side::Right, alpha, B, A, opts);
}

// Left trsm
template <typename scalar_t>
void triangular_solve(
    scalar_t alpha, TriangularMatrix<scalar_t>& A,
                              Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    trsm(Side::Left, alpha, A, B, opts);
}

// Right trsm
template <typename scalar_t>
void triangular_solve(
    scalar_t alpha,           Matrix<scalar_t>& A,
                    TriangularMatrix<scalar_t>& B,
    Options const& opts = Options())
{
    trsm(Side::Right, alpha, B, A, opts);
}

//-----------------------------------------
// rank_k_update()

// herk
template <typename scalar_t>
void rank_k_update(
    blas::real_type<scalar_t> alpha,          Matrix<scalar_t>& A,
    blas::real_type<scalar_t> beta,  HermitianMatrix<scalar_t>& C,
    Options const& opts = Options())
{
    herk(alpha, A, beta, C, opts);
}

// syrk
template <typename scalar_t>
void rank_k_update(
    scalar_t alpha,           Matrix<scalar_t>& A,
    scalar_t beta,   SymmetricMatrix<scalar_t>& C,
    Options const& opts = Options())
{
    syrk(alpha, A, beta, C, opts);
}

//-----------------------------------------
// rank_2k_update()

// herk
template <typename scalar_t>
void rank_2k_update(
    scalar_t alpha,                           Matrix<scalar_t>& A,
                                              Matrix<scalar_t>& B,
    blas::real_type<scalar_t> beta,  HermitianMatrix<scalar_t>& C,
    Options const& opts = Options())
{
    her2k(alpha, A, B, beta, C, opts);
}

// syrk
template <typename scalar_t>
void rank_2k_update(
    scalar_t alpha,           Matrix<scalar_t>& A,
                              Matrix<scalar_t>& B,
    scalar_t beta,   SymmetricMatrix<scalar_t>& C,
    Options const& opts = Options())
{
    syr2k(alpha, A, B, beta, C, opts);
}

//------------------------------------------------------------------------------
// Linear systems

//-----------------------------------------
// LU

//-----------------------------------------
// lu_solve()

// gbsv
template <typename scalar_t>
void lu_solve(
    BandMatrix<scalar_t>& A,
        Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    Pivots pivots;
    gbsv(A, pivots, B, opts);
}

// gesv
template <typename scalar_t>
void lu_solve(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    Pivots pivots;
    gesv(A, pivots, B, opts);
}

//-----------------------------------------
// lu_solve_nopiv()

// todo
// gbsv_nopiv
// template <typename scalar_t>
// void lu_solve_nopiv(
//     BandMatrix<scalar_t>& A,
//         Matrix<scalar_t>& B,
//     Options const& opts = Options())
// {
//     gbsv_nopiv(A, B, opts);
// }

// gesv_nopiv
template <typename scalar_t>
void lu_solve_nopiv(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    gesv_nopiv(A, B, opts);
}

//-----------------------------------------
// lu_factor()

// gbtrf
template <typename scalar_t>
void lu_factor(
    BandMatrix<scalar_t>& A, Pivots& pivots,
    Options const& opts = Options())
{
    gbtrf(A, pivots, opts);
}

// getrf
template <typename scalar_t>
void lu_factor(
    Matrix<scalar_t>& A, Pivots& pivots,
    Options const& opts = Options())
{
    getrf(A, pivots, opts);
}

//-----------------------------------------
// lu_factor_nopiv()

// todo
// gbtrf_nopiv
// template <typename scalar_t>
// void lu_factor_nopiv(
//     BandMatrix<scalar_t>& A,
//     Options const& opts = Options())
// {
//     gbtrf_nopiv(A, opts);
// }

// getrf_nopiv
template <typename scalar_t>
void lu_factor_nopiv(
    Matrix<scalar_t>& A,
    Options const& opts = Options())
{
    getrf_nopiv(A, opts);
}

//-----------------------------------------
// lu_solve_using_factor()

// gbtrs
template <typename scalar_t>
void lu_solve_using_factor(
    BandMatrix<scalar_t>& A, Pivots& pivots,
        Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    gbtrs(A, pivots, B, opts);
}

// getrs
template <typename scalar_t>
void lu_solve_using_factor(
    Matrix<scalar_t>& A, Pivots& pivots,
    Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    getrs(A, pivots, B, opts);
}

//-----------------------------------------
// lu_solve_using_factor_nopiv()

// todo
// gbtrs_nopiv
// template <typename scalar_t>
// void lu_solve_using_factor_nopiv(
//     BandMatrix<scalar_t>& A,
//         Matrix<scalar_t>& B,
//     Options const& opts = Options())
// {
//     gbtrs_nopiv(A, B, opts);
// }

// getrs_nopiv
template <typename scalar_t>
void lu_solve_using_factor_nopiv(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    getrs_nopiv(A, B, opts);
}

//-----------------------------------------
// lu_inverse_using_factor()

// In-place getri
template <typename scalar_t>
void lu_inverse_using_factor(
    Matrix<scalar_t>& A, Pivots& pivots,
    Options const& opts = Options())
{
    getri(A, pivots, opts);
}

//-----------------------------------------
// lu_inverse_using_factor_out_of_place()

// Out-of-place getri
template <typename scalar_t>
void lu_inverse_using_factor_out_of_place(
    Matrix<scalar_t>& A, Pivots& pivots,
    Matrix<scalar_t>& A_inverse,
    Options const& opts = Options())

{
    getri(A, pivots, A_inverse, opts);
}

//-----------------------------------------
// Cholesky

//-----------------------------------------
// chol_solve()

// pbsv
template <typename scalar_t>
void chol_solve(
    HermitianBandMatrix<scalar_t>& A,
                 Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    pbsv(A, B, opts);
}

// posv
template <typename scalar_t>
void chol_solve(
    HermitianMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    posv(A, B, opts);
}

// forward real-symmetric matrices to posv;
// disabled for complex
template <typename scalar_t>
void chol_solve(
    SymmetricMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    posv(A, B, opts);
}

//-----------------------------------------
// chol_factor()

// pbtrf
template <typename scalar_t>
void chol_factor(
    HermitianBandMatrix<scalar_t>& A,
    Options const& opts = Options())
{
    pbtrf(A, opts);
}

// potrf
template <typename scalar_t>
void chol_factor(
    HermitianMatrix<scalar_t>& A,
    Options const& opts = Options())
{
    potrf(A, opts);
}

// forward real-symmetric matrices to potrf;
// disabled for complex
template <typename scalar_t>
void chol_factor(
    SymmetricMatrix<scalar_t>& A,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    potrf(A, opts);
}

//-----------------------------------------
// chol_solve_using_factor()

// pbtrs
template <typename scalar_t>
void chol_solve_using_factor(
    HermitianBandMatrix<scalar_t>& A,
                 Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    pbtrs(A, B, opts);
}

// potrs
template <typename scalar_t>
void chol_solve_using_factor(
    HermitianMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    potrs(A, B, opts);
}

// forward real-symmetric matrices to potrs;
// disabled for complex
template <typename scalar_t>
void chol_solve_using_factor(
    SymmetricMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    potrs(A, B, opts);
}

//-----------------------------------------
// chol_inverse_using_factor()

// potri
template <typename scalar_t>
void chol_inverse_using_factor(
    HermitianMatrix<scalar_t>& A,
    Options const& opts = Options())
{
    potri(A, opts);
}

//-----------------------------------------
// Symmetric indefinite -- block Aasen's

//-----------------------------------------
// indefinite_solve()

// hesv
template <typename scalar_t>
void indefinite_solve(
    HermitianMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    // auxiliary matrices
    auto H = slate::Matrix<scalar_t>::emptyLike(A);

    int64_t kl = A.tileNb(0);
    int64_t ku = A.tileNb(0);
    auto T = slate::BandMatrix<scalar_t>::emptyLike(A, kl, ku);

    Pivots pivots, pivots2;
    hesv(A, pivots, T, pivots2, H, B, opts);
}

// forward real-symmetric matrices to hesv;
// disabled for complex
template <typename scalar_t>
void indefinite_solve(
    SymmetricMatrix<scalar_t>& A,
             Matrix<scalar_t>& B,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    // auxiliary matrices
    auto H = slate::Matrix<scalar_t>::emptyLike(A);

    int64_t kl = A.tileNb(0);
    int64_t ku = A.tileNb(0);
    auto T = slate::BandMatrix<scalar_t>::emptyLike(A, kl, ku);

    Pivots pivots, pivots2;
    sysv(A, pivots, T, pivots2, H, B, opts);
}

//-----------------------------------------
// indefinite_factor()

// hetrf
template <typename scalar_t>
void indefinite_factor(
    HermitianMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& H,
    Options const& opts = Options())
{
    hetrf(A, pivots, T, pivots2, H, opts);
}

// forward real-symmetric matrices to hetrf;
// disabled for complex
template <typename scalar_t>
void indefinite_factor(
    SymmetricMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& H,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    sytrf(A, pivots, T, pivots2, H, opts);
}

//-----------------------------------------
// indefinite_solve_using_factor()

// hetrs
template <typename scalar_t>
void indefinite_solve_using_factor(
    HermitianMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& B,
    Options const& opts = Options())
{
    hetrs(A, pivots, T, pivots2, B, opts);
}
// forward real-symmetric matrices to hetrs;
// disabled for complex
template <typename scalar_t>
void indefinite_solve_using_factor(
    SymmetricMatrix<scalar_t>& A, Pivots& pivots,
         BandMatrix<scalar_t>& T, Pivots& pivots2,
             Matrix<scalar_t>& B,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    sytrs(A, pivots, T, pivots2, B, opts);
}

//------------------------------------------------------------------------------
// QR

//-----------------------------------------
// Least squares

//-----------------------------------------
// least_squares_solve()

// gels
template <typename scalar_t>
void least_squares_solve(
    Matrix<scalar_t>& A,
    Matrix<scalar_t>& BX,
    Options const& opts = Options())
{
    TriangularFactors<scalar_t> T;
    gels(A, T, BX, opts);
}

//-----------------------------------------
// QR

//-----------------------------------------
// qr_factor()

// geqrf
template <typename scalar_t>
void qr_factor(
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Options const& opts = Options())
{
    geqrf(A, T, opts);
}

//-----------------------------------------
// qr_multiply_by_q()

// unmqr
template <typename scalar_t>
void qr_multiply_by_q(
    Side side, Op op,
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& C,
    Options const& opts = Options())
{
    unmqr(side, op, A, T, C, opts);
}

//-----------------------------------------
// LQ

//-----------------------------------------
// lq_factor()

// gelqf
template <typename scalar_t>
void lq_factor(
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Options const& opts = Options())
{
    gelqf(A, T, opts);
}

//-----------------------------------------
// lq_multiply_by_q()

// unmlq
template <typename scalar_t>
void lq_multiply_by_q(
    Side side, Op op,
    Matrix<scalar_t>& A, TriangularFactors<scalar_t>& T,
    Matrix<scalar_t>& C,
    Options const& opts = Options())
{
    unmlq(side, op, A, T, C, opts);
}

//------------------------------------------------------------------------------
// SVD

//-----------------------------------------
// svd_vals()

// gesvd
template <typename scalar_t>
void svd_vals(
    Matrix<scalar_t> A,
    std::vector< blas::real_type<scalar_t> >& Sigma,
    Options const& opts = Options())
{
    gesvd(A, Sigma, opts);
}

//------------------------------------------------------------------------------
// Eigenvalue decomposition

//-----------------------------------------
// eig_vals()

//-----------------------------------------
// Symmetric/hermitian

// heev
template <typename scalar_t>
void eig_vals(
    HermitianMatrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Options const& opts = Options())
{
    Matrix<scalar_t> Z;
    heev(Job::NoVec, A, Lambda, Z, opts);
}

// syev
// forward real-symmetric matrices to heev;
// disabled for complex
template <typename scalar_t>
void eig_vals(
    SymmetricMatrix<scalar_t>& A,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    Matrix<scalar_t> Z;
    syev(Job::NoVec, A, Lambda, Z, opts);
}

//-----------------------------------------
// Generalized symmetric/hermitian

// hegv
template <typename scalar_t>
void eig_vals(
    int64_t itype,
    HermitianMatrix<scalar_t>& A,
    HermitianMatrix<scalar_t>& B,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Options const& opts = Options())
{
    Matrix<scalar_t> V;
    hegv(itype, Job::NoVec, A, B, Lambda, V, opts);
}

// sygv
// forward real-symmetric matrices to hegv;
// disabled for complex
template <typename scalar_t>
void eig_vals(
    int64_t itype,
    SymmetricMatrix<scalar_t>& A,
    SymmetricMatrix<scalar_t>& B,
    std::vector< blas::real_type<scalar_t> >& Lambda,
    Options const& opts = Options(),
    enable_if_t< ! is_complex<scalar_t>::value >* = nullptr)
{
    Matrix<scalar_t> V;
    sygv(itype, Job::NoVec, A, B, Lambda, V, opts);
}

} // namespace slate

#endif // SLATE_SIMPLIFIED_API_HH
