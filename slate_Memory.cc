
#include "slate_Memory.hh"

namespace slate {

int Memory::host_num_ = omp_get_initial_device();

} // namespace slate
