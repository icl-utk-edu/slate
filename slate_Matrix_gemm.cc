
#include "slate_Matrix.hh"
#include "slate_types.hh"

namespace slate {

//------------------------------------------------------------------------------
template <typename FloatType>
template <Target target>
void Matrix<FloatType>::gemm(blas::Op opa, blas::Op opb,
                             FloatType alpha, Matrix &&a,
                                              Matrix &&b,
                             FloatType beta,  Matrix &&c)
{
    gemm(internal::TargetType<target>(),
         opa, opb,
         alpha, a,
                b,
         beta,  c);
}

//------------------------------------------------------------------------------
template <typename FloatType>
void Matrix<FloatType>::gemm(internal::TargetType<Target::HostTask>,
                             blas::Op opa, blas::Op opb,
                             FloatType alpha, Matrix &a,
                                              Matrix &b,
                             FloatType beta,  Matrix &c)
{
    // NoTrans, Trans
    for (int m = 0; m < c.mt_; ++m)
        for (int n = 0; n < c.nt_; ++n)
            if (c.tileIsLocal(m, n))
                #pragma omp task shared(a, b, c)
                {
                    c.tileMoveToHost(m, n, c.tileDevice(m, n));
                    Tile<FloatType>::gemm(opa, opb,
                                          alpha, a(m, 0),
                                                 b(n, 0),
                                          beta,  c(m, n));
                    a.tileTick(m, 0);
                    a.tileTick(n, 0);
                }

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
template
void Matrix<double>::gemm<Target::HostTask>(
    blas::Op opa, blas::Op opb,
    double alpha, Matrix &&a,
                  Matrix &&b,
    double beta,  Matrix &&c);

} // namespace slate
