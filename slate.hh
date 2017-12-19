
#ifndef SLATE_HH
#define SLATE_HH

#include "slate_Matrix.hh"
#include "slate_types.hh"

namespace slate {

template <Target target = Target::HostTask>
void potrf(blas::Uplo uplo, Matrix<double> a, int64_t lookahead = 0);

} // namespace slate

#endif // SLATE_HH
