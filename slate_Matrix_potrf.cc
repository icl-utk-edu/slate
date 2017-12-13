
#include "slate_Matrix.hh"
#include "slate_types.hh"

namespace slate {

//------------------------------------------------------------------------------
template <typename FloatType>
template <Target target>
void Matrix<FloatType>::potrf(blas::Uplo uplo, Matrix &&a)
{
    potrf(internal::TargetType<target>(), uplo, a);
}

//------------------------------------------------------------------------------
template <typename FloatType>
void Matrix<FloatType>::potrf(internal::TargetType<Target::HostTask>,
                              blas::Uplo uplo, Matrix &a)
{

    if (a.tileIsLocal(0, 0))
        #pragma omp task shared(a)
        Tile<FloatType>::potrf(uplo, a(0, 0));

    #pragma omp taskwait
}

//------------------------------------------------------------------------------
template
void Matrix<double>::potrf<Target::HostTask>(
    blas::Uplo uplo, Matrix &&a);

} // namespace slate
