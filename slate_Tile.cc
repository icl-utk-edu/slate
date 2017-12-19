
#include "slate_Tile.hh"

namespace slate {

template <>
int Tile<double>::host_num_ = omp_get_initial_device();

} // namespace slate
