
#include "slate/slate.hh"
#include "slate/types.hh"
#include "internal/internal.hh"


namespace slate {

template<typename scalar_t>
void gerbt(Matrix<scalar_t>& U_in,
           Matrix<scalar_t>& A,
           Matrix<scalar_t>& V)
{

    slate_assert(U_in.op() == Op::Trans);
    slate_assert(V.op() == Op::NoTrans);

    Matrix<scalar_t> U = transpose(U_in);

    slate_assert(A.op() == Op::NoTrans);
    slate_assert(U.op() == Op::NoTrans);
    slate_assert(A.layout() == Layout::ColMajor);
    slate_assert(U.layout() == Layout::ColMajor);
    slate_assert(V.layout() == Layout::ColMajor);

    slate_assert(U.n() == V.n());

    const int64_t d = U.n();
    const int64_t n = A.n();

    if (d == 0) {
        return;
    }

    slate_assert((n % (1<<d)) == 0);


    // 2-sided butterflies are applied smallest to largest
    for (int64_t k = d-1; k >= 0; --k) {
        const int64_t num_bt = 1 << k;
        const int64_t half_len = n >> (k+1);

        #pragma omp parallel
        #pragma omp master
        {
            omp_set_nested(1);

            for (int64_t bi = 0; bi < num_bt; ++bi) {
                const int64_t i1 = bi*2*half_len;
                const int64_t i2 = i1+half_len;
                const int64_t i3 = i2+half_len;
                for (int64_t bj = 0; bj < num_bt; ++bj) {
                    const int64_t j1 = bj*2*half_len;
                    const int64_t j2 = j1+half_len;
                    const int64_t j3 = j2+half_len;

                    auto A11 = A.slice(i1, i2-1, j1, j2-1);
                    auto A12 = A.slice(i1, i2-1, j2, j3-1);
                    auto A21 = A.slice(i2, i3-1, j1, j2-1);
                    auto A22 = A.slice(i2, i3-1, j2, j3-1);

                    auto U1 = U.slice(i1, i2-1, k, k);
                    auto U2 = U.slice(i2, i3-1, k, k);
                    auto V1 = V.slice(j1, j2-1, k, k);
                    auto V2 = V.slice(j2, j3-1, k, k);

                    internal::gerbt(A11, A12, A21, A22, U1, U2, V1, V2);
                    #pragma omp taskwait
                }
            }
        }
    }
}

template
void gerbt(Matrix<float>&,
           Matrix<float>&,
           Matrix<float>&);

template
void gerbt(Matrix<double>&,
           Matrix<double>&,
           Matrix<double>&);

template
void gerbt(Matrix<std::complex<float>>&,
           Matrix<std::complex<float>>&,
           Matrix<std::complex<float>>&);

template
void gerbt(Matrix<std::complex<double>>&,
           Matrix<std::complex<double>>&,
           Matrix<std::complex<double>>&);


template<typename scalar_t>
void gerbt(Matrix<scalar_t>& Uin,
           Matrix<scalar_t>& B)
{

    bool transposed = Uin.op() == Op::Trans;
    Matrix<scalar_t> U = transposed ? transpose(Uin) : Uin;

    slate_assert(B.op() == Op::NoTrans);
    slate_assert(U.op() == Op::NoTrans);
    slate_assert(B.layout() == Layout::ColMajor);
    slate_assert(U.layout() == Layout::ColMajor);

    slate_assert(B.mt() == U.mt());

    const int64_t d = U.n();
    const int64_t m = B.m();
    const int64_t n = B.n();

    if (d == 0) {
        return;
    }

    for (int64_t k_iter = 0; k_iter < d; ++k_iter) {
        // Regular butterflies are applied largest to smallest
        // Transposed butterflies are applied smallest to largest
        const int64_t k = transposed ? d-k_iter-1 : k_iter;

        const int64_t num_bt = 1 << k;
        const int64_t half_len = m >> (k+1);

        #pragma omp parallel
        #pragma omp master
        {
            omp_set_nested(1);

            for (int64_t bi = 0; bi < num_bt; ++bi) {
                const int64_t i1 = bi*2*half_len;
                const int64_t i2 = i1+half_len;
                const int64_t i3 = i2+half_len;

                auto B1 = B.slice(i1, i2-1, 0, n-1);
                auto B2 = B.slice(i2, i3-1, 0, n-1);

                auto U1 = U.slice(i1, i2-1, k, k);
                auto U2 = U.slice(i2, i3-1, k, k);

                internal::gerbt(transposed, B1, B2, U1, U2);
                #pragma omp taskwait
            }
        }
    }
}

template
void gerbt(Matrix<float>&,
           Matrix<float>&);

template
void gerbt(Matrix<double>&,
           Matrix<double>&);

template
void gerbt(Matrix<std::complex<float>>&,
           Matrix<std::complex<float>>&);

template
void gerbt(Matrix<std::complex<double>>&,
           Matrix<std::complex<double>>&);



} // namespace slate
