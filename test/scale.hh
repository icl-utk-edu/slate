#include <slate/slate.hh>

namespace slate {

//------------------------------------------------------------------------------
template <typename scalar_t>
void col_scale( Matrix<scalar_t>& A, std::vector<scalar_t>& x )
{
    #pragma omp parallel
    #pragma omp master
    {
        int64_t mt = A.mt();
        int64_t nt = A.nt();
        int64_t jj = 0;
        for (int64_t j = 0; j < nt; ++j) {
            for (int64_t i = 0; i < mt; ++i) {
                if (A.tileIsLocal( i, j )) {
                    #pragma omp task
                    {
                        auto Aij = A( i, j );
                        int64_t nb = Aij.nb();
                        int64_t mb = Aij.mb();
                        for (int64_t joff = 0; joff < nb; ++joff) {
                            blas::scal( mb, x[ jj + joff ], &Aij.at( 0, joff ), 1 );
                        }
                    }
                }
            }
            jj += A.tileNb( j );
        }
    }
}

}  // namespace slate
