
#include "slate_Matrix.hh"
#include "slate_types.hh"

namespace slate {

//------------------------------------------------------------------------------
template <typename FloatType>
template <Target target>
void Matrix<FloatType>::trsm(blas::Side side, blas::Uplo uplo,
                             blas::Op op, blas::Diag diag,
                             FloatType alpha, Matrix &&a,
                                              Matrix &&b)
{
    trsm(internal::TargetType<target>(),
        side, uplo, op, diag,
        alpha, a, b);
}

//------------------------------------------------------------------------------
template <typename FloatType>
void Matrix<FloatType>::trsm(internal::TargetType<Target::HostTask>,
                             blas::Side side, blas::Uplo uplo,
                             blas::Op op, blas::Diag diag,
                             FloatType alpha, Matrix &a,
                                              Matrix &b)
{
    // Right, Lower, Trans
    for (int64_t m = 0; m < b.mt_; ++m)
        if (b.tileIsLocal(m, 0))
            #pragma omp task shared(a, b)
            Tile<FloatType>::trsm(side, uplo, op, diag,
                                  alpha, a(0, 0),
                                         b(m, 0));

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
template
void Matrix<double>::trsm<Target::HostTask>(
    blas::Side side, blas::Uplo uplo,
    blas::Op op, blas::Diag diag,
    double alpha, Matrix &&a,
                  Matrix &&b);

} // namespace slate
