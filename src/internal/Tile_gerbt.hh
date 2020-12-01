
#include "slate/Tile.hh"

namespace slate {

namespace internal {

template<typename scalar_t>
void gerbt(int64_t mb, int64_t nb,
           scalar_t* A11, int64_t A11_stride,
           scalar_t* A12, int64_t A12_stride,
           scalar_t* A21, int64_t A21_stride,
           scalar_t* A22, int64_t A22_stride,
           scalar_t* U1,
           scalar_t* U2,
           scalar_t* V1,
           scalar_t* V2)
{

    for (int64_t j = 0; j < nb; ++j) {
        const scalar_t v1 = V1[j];
        const scalar_t v2 = V2[j];

        for (int64_t i = 0; i < mb; ++i) {
            const scalar_t u1 = U1[i];
            const scalar_t u2 = U2[i];

            const scalar_t a11 = A11[i + j*A11_stride];
            const scalar_t a12 = A12[i + j*A12_stride];
            const scalar_t a21 = A21[i + j*A21_stride];
            const scalar_t a22 = A22[i + j*A22_stride];

            const scalar_t sum1 = a11 + a12;
            const scalar_t sum2 = a21 + a22;
            const scalar_t dif1 = a11 - a12;
            const scalar_t dif2 = a21 - a22;

            A11[i + j*A11_stride] = u1*v1*(sum1 + sum2);
            A12[i + j*A12_stride] = u1*v2*(dif1 + dif2);
            A21[i + j*A21_stride] = u2*v1*(sum1 - sum2);
            A22[i + j*A22_stride] = u2*v2*(dif1 - dif2);
        }
    }
}

template<typename scalar_t>
void gerbt_notrans(int64_t mb, int64_t nb,
                   scalar_t* B1, int64_t B1_stride,
                   scalar_t* B2, int64_t B2_stride,
                   scalar_t* U1,
                   scalar_t* U2)
{

    for (int64_t j = 0; j < nb; ++j) {
        for (int64_t i = 0; i < mb; ++i) {
            const scalar_t u1 = U1[i];
            const scalar_t u2 = U2[i];

            const scalar_t b1 = B1[i + j*B1_stride];
            const scalar_t b2 = B2[i + j*B2_stride];

            B1[i + j*B1_stride] = u1*b1 + u2*b2;
            B2[i + j*B2_stride] = u1*b1 - u2*b2;
        }
    }
}

template<typename scalar_t>
void gerbt_trans(int64_t mb, int64_t nb,
                 scalar_t* B1, int64_t B1_stride,
                 scalar_t* B2, int64_t B2_stride,
                 scalar_t* U1,
                 scalar_t* U2)
{

    for (int64_t j = 0; j < nb; ++j) {
        for (int64_t i = 0; i < mb; ++i) {
            const scalar_t u1 = U1[i];
            const scalar_t u2 = U2[i];

            const scalar_t b1 = B1[i + j*B1_stride];
            const scalar_t b2 = B2[i + j*B2_stride];

            B1[i + j*B1_stride] = u1*(b1 + b2);
            B2[i + j*B2_stride] = u2*(b1 - b2);
        }
    }
}

} // namespace internal

} // namespace slate

