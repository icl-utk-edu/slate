
#include "slate_Matrix.hh"

namespace slate {

template <>
int Matrix<double>::host_num_ = omp_get_initial_device();

} // namespace slate
