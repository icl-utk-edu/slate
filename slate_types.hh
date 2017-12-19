
#ifndef SLATE_TYPES_HH
#define SLATE_TYPES_HH

namespace slate {

enum class Target {Devices, Host, HostTask, HostNest, HostBatch};

namespace internal {

template <Target> class TargetType {};

} // namespace internal

} // namespace slate

#endif // SLATE_TYPES_HH
