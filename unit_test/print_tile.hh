#ifndef SLATE_PRINT_TILE_HH
#define SLATE_PRINT_TILE_HH

#include "slate/Tile.hh"

#include <stdio.h>
#include <complex>

//------------------------------------------------------------------------------
/// Print a Tile.
///
template <typename scalar_t>
void print(const char* name, slate::Tile<scalar_t>& A)
{
    printf("%s = [\n", name);
    for (int i = 0; i < A.mb(); ++i) {
        for (int64_t j = 0; j < A.nb(); ++j) {
            printf(" %9.4f", A(i, j));
        }
        printf("\n");
    }
    printf("];  %% op=%c, uplo=%c\n", char(A.op()), char(A.uplo()));
}

//------------------------------------------------------------------------------
/// Print a Tile, complex specialization.
///
template <typename scalar_t>
void print(const char* name, slate::Tile< std::complex<scalar_t> >& A)
{
    printf("%s = [\n", name);
    for (int i = 0; i < A.mb(); ++i) {
        for (int64_t j = 0; j < A.nb(); ++j) {
            printf(" %9.4f + %9.4fi", real(A(i, j)), imag(A(i, j)));
        }
        printf("\n");
    }
    printf("];  %% op=%c, uplo=%c\n", char(A.op()), char(A.uplo()));
}

#endif // SLATE_PRINT_TILE_HH
