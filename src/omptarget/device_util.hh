// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_OMPTARGET_UTIL_HH
#define SLATE_OMPTARGET_UTIL_HH

#include <math.h>

namespace slate {
namespace device {

//------------------------------------------------------------------------------
/// max that propogates nan consistently:
///     max_nan( 1,   nan ) = nan
///     max_nan( nan, 1   ) = nan
#pragma omp declare target
template <typename real_t>
inline real_t max_nan(real_t x, real_t y)
{
    return (isnan(y) || (y) >= (x) ? (y) : (x));
}
#pragma omp end declare target

// OpenMP reduction operation that allows for the nan propogation
#pragma omp declare reduction(max_nan_reduction: float, double: omp_out = max_nan(omp_out, omp_in))

//------------------------------------------------------------------------------
/// Square of number.
/// @return x^2
#pragma omp declare target
template <typename scalar_t>
inline scalar_t sqr(scalar_t x)
{
    return x*x;
}
#pragma omp end declare target

//------------------------------------------------------------------------------
/// Adds two scaled, sum-of-squares representations.
/// On exit, scale1 and sumsq1 are updated such that:
///     scale1^2 sumsq1 := scale1^2 sumsq1 + scale2^2 sumsq2.
#pragma omp declare target
template <typename real_t>
void combine_sumsq(
    real_t& scale1, real_t& sumsq1,
    real_t  scale2, real_t  sumsq2 )
{
    if (scale1 > scale2) {
        sumsq1 = sumsq1 + sumsq2*sqr(scale2 / scale1);
        // scale1 stays same
    }
    else if (scale2 != 0) {
        sumsq1 = sumsq1*sqr(scale1 / scale2) + sumsq2;
        scale1 = scale2;
    }
}
#pragma omp end declare target

//------------------------------------------------------------------------------
/// Adds new value to scaled, sum-of-squares representation.
/// On exit, scale and sumsq are updated such that:
///     scale^2 sumsq := scale^2 sumsq + (absx)^2
#pragma omp declare target
template <typename real_t>
inline void add_sumsq(
    real_t& scale, real_t& sumsq,
    real_t absx)
{
    if (scale < absx) {
        sumsq = 1 + sumsq * sqr(scale / absx);
        scale = absx;
    }
    else if (scale != 0) {
        sumsq = sumsq + sqr(absx / scale);
    }
}
#pragma omp end declare target

//------------------------------------------------------------------------------
/// Overloaded versions of absolute value on device.
#pragma omp declare target
inline float abs_val(float x)
{
    return fabsf(x);
}

inline double abs_val(double x)
{
    return fabs(x);
}

inline float abs_val(std::complex<float> x)
{
    // use our implementation that scales per LAPACK.
    // todo try std::abs(x)
    // todo try device specific abs eg for cuda
    float a = std::real(x);
    float b = std::imag(x);
    float z, w, t;
    if (isnan( a )) {
        return a;
    }
    else if (isnan( b )) {
        return b;
    }
    else {
        a = fabsf(a);
        b = fabsf(b);
        w = std::max(a, b);
        z = std::min(a, b);
        if (z == 0) {
            t = w;
        }
        else {
            t = z/w;
            t = 1 + t*t;
            t = w * sqrtf(t);
        }
        return t;
    }
}

inline double abs_val(std::complex<double> x)
{
    // use our implementation that scales per LAPACK.
    // todo try std::abs(x)
    // todo try device specific abs eg for cuda
    double a = std::real(x);
    double b = std::imag(x);
    double z, w, t;
    if (isnan( a )) {
        return a;
    }
    else if (isnan( b )) {
        return b;
    }
    else {
        a = fabs(a);
        b = fabs(b);
        w = std::max(a, b);
        z = std::min(a, b);
        if (z == 0) {
            t = w;
        }
        else {
            t = z/w;
            t = 1.0 + t*t;
            t = w * sqrt(t);
        }
        return t;
    }
}
#pragma omp end declare target

} // namespace device
} // namespace slate

#endif // SLATE_OMPTARGET_UTIL_HH
