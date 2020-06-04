#ifndef SLATE_MATRIX_UTILS_HH
#define SLATE_MATRIX_UTILS_HH

#include "slate/slate.hh"

//------------------------------------------------------------------------------
// Zero out B, then copy band matrix B from A.
// B is stored as a non-symmetric matrix, so we can apply Q from left
// and right separately.
template <typename scalar_t>
void he2gb(slate::HermitianMatrix< scalar_t > A, slate::Matrix< scalar_t > B)
{
    // It must be defined here to avoid having numerical error with complex
    // numbers when calling conj();
    using blas::conj;
    const int64_t nt = A.nt();
    const scalar_t zero = 0;
    set(zero, B);
    for (int64_t i = 0; i < nt; ++i) {
        if (B.tileIsLocal(i, i)) {
            // diagonal tile
            auto Aii = A(i, i);
            auto Bii = B(i, i);
            Aii.uplo(slate::Uplo::Lower);
            Bii.uplo(slate::Uplo::Lower);
            tzcopy(Aii, Bii);
            // Symmetrize the tile.
            for (int64_t jj = 0; jj < Bii.nb(); ++jj)
                for (int64_t ii = jj; ii < Bii.mb(); ++ii)
                    Bii.at(jj, ii) = conj(Bii(ii, jj));
        }
        if (i+1 < nt && B.tileIsLocal(i+1, i)) {
            // sub-diagonal tile
            auto Ai1i = A(i+1, i);
            auto Bi1i = B(i+1, i);
            Ai1i.uplo(slate::Uplo::Upper);
            Bi1i.uplo(slate::Uplo::Upper);
            tzcopy(Ai1i, Bi1i);
            if (! B.tileIsLocal(i, i+1))
                B.tileSend(i+1, i, B.tileRank(i, i+1));
        }
        if (i+1 < nt && B.tileIsLocal(i, i+1)) {
            if (! B.tileIsLocal(i+1, i)) {
                // Remote copy-transpose B(i+1, i) => B(i, i+1);
                // assumes square tiles!
                B.tileRecv(i, i+1, B.tileRank(i+1, i), slate::Layout::ColMajor);
                deepConjTranspose(B(i, i+1));
            }
            else {
                // Local copy-transpose B(i+1, i) => B(i, i+1).
                deepConjTranspose(B(i+1, i), B(i, i+1));
            }
        }
    }
}

//------------------------------------------------------------------------------
// Convert a HermitianMatrix into a General Matrix, ConjTrans/Trans the opposite
// off-diagonal tiles
// todo: shouldn't assume the input HermitianMatrix has uplo=lower
template <typename scalar_t>
inline void he2ge(slate::HermitianMatrix<scalar_t> A, slate::Matrix<scalar_t> B)
{
    // todo:: shouldn't assume the input matrix has uplo=lower
    assert(A.uplo() == slate::Uplo::Lower);

    using blas::conj;
    const scalar_t zero = 0;
    set(zero, B);
    for (int64_t j = 0; j < A.nt(); ++j) {
        // todo: shouldn't assume uplo=lowwer
        for (int64_t i = j; i < A.nt(); ++i) {
            if (i == j) { // diagonal tiles
                if (B.tileIsLocal(i, j)) {
                    auto Aij = A(i, j);
                    auto Bij = B(i, j);
                    Aij.uplo(slate::Uplo::Lower);
                    Bij.uplo(slate::Uplo::Lower);
                    tzcopy(Aij, Bij);
                    for (int64_t jj = 0; jj < Bij.nb(); ++jj) {
                        for (int64_t ii = jj; ii < Bij.mb(); ++ii) {
                            Bij.at(jj, ii) = conj(Bij(ii, jj));
                        }
                    }
                }
            }
            else {
                if (B.tileIsLocal(i, j)) {
                    auto Aij = A(i, j);
                    auto Bij = B(i, j);
                    gecopy(Aij, Bij);
                    if (! B.tileIsLocal(j, i)) {
                        B.tileSend(i, j, B.tileRank(j, i));
                    }
                }
                if (B.tileIsLocal(j, i)) {
                    if (! B.tileIsLocal(i, j)) {
                        B.tileRecv(
                            j, i, B.tileRank(i, j), slate::Layout::ColMajor);
                        deepConjTranspose(B(j, i));
                    }
                    else {
                        deepConjTranspose(B(i, j), B(j, i));
                    }
                }
            }
        }
    }
}

#endif // SLATE_MATRIX_UTILS_HH
