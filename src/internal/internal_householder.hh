#ifndef SLATE_INTERNAL_HOUSEHOLDER_HH
#define SLATE_INTERNAL_HOUSEHOLDER_HH

#include "slate/Matrix.hh"
#include "slate/HermitianMatrix.hh"

namespace slate {
namespace internal {

// Defined in internal_gebr.cc.
template <typename scalar_t>
void gerfg(Matrix<scalar_t>& A, int64_t n, scalar_t* v);

template <typename scalar_t>
void gerf(int64_t n, scalar_t* v, Matrix<scalar_t>& A);

// Define in internal_hebr.cc.
template <typename scalar_t>
void herf(int64_t n, scalar_t* v, HermitianMatrix<scalar_t>& A);

}  // internal
}  // slate

#endif  // SLATE_INTERNAL_HOUSEHOLDER_HH
