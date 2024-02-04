// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

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
void generate( Dist dist, int64_t key,
               int64_t m, int64_t n, int64_t ioffset, int64_t joffset,
               scalar_t* A, int64_t lda );

} // namespace random
} // namespace slate

#endif // SLATE_RANDOM_HH
