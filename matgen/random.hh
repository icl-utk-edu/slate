
#ifndef SLATE_RANDOM_HH
#define SLATE_RANDOM_HH

#include <cstdint>

namespace slate {
namespace random {

enum class Dist {
    Uniform=1,
    UniformSigned,
    Normal,
    UnitDisk,
    UnitCircle,
    Binary,
    BinarySigned
};

template<class scalar_t>
void generate(Dist dist, int64_t key,
                   int64_t m, int64_t n, int64_t ioffset, int64_t joffset,
                   scalar_t* A, int64_t lda);

} // namespace random
} // namespace slate

#endif // SLATE_RANDOM_HH
