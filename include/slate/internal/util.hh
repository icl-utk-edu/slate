// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
/// @file
///
#ifndef SLATE_UTIL_HH
#define SLATE_UTIL_HH

#include "slate/internal/mpi.hh"

#include <cmath>

#include <blas.hh>

namespace slate {

//------------------------------------------------------------------------------
/// max that propogates nan consistently:
///
///     max_nan( 1,   nan ) = nan
///     max_nan( nan, 1   ) = nan
///
template <typename real_t>
inline real_t max_nan(real_t x, real_t y)
{
    return (std::isnan(y) || (y) >= (x) ? (y) : (x));
}

//------------------------------------------------------------------------------
/// Square of number.
/// @return x^2
template <typename scalar_t>
inline scalar_t sqr(scalar_t x)
{
    return x*x;
}

//------------------------------------------------------------------------------
/// Adds two scaled, sum-of-squares representations.
/// On exit, scale1 and sumsq1 are updated such that:
///
///     scale1^2 sumsq1 := scale1^2 sumsq1 + scale2^2 sumsq2.
///
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

//------------------------------------------------------------------------------
/// Adds new value to scaled, sum-of-squares representation.
/// On exit, scale and sumsq are updated such that:
///
///     scale^2 sumsq := scale^2 sumsq + (absx)^2
///
template <typename real_t>
void add_sumsq(
    real_t& scale, real_t& sumsq,
    real_t absx)
{
    if (scale < absx) {
        sumsq = 1 + sumsq * sqr(scale / absx);
        scale = absx;
    }
    else {
        sumsq = sumsq + sqr(absx / scale);
    }
}

//------------------------------------------------------------------------------
/// @return ceil( x / y ), for integer type T.
template <typename T>
inline constexpr T ceildiv(T x, T y)
{
    return T((x + y - 1) / y);
}

/// @return ceil( x / y )*y, i.e., x rounded up to next multiple of y.
template <typename T>
inline constexpr T roundup(T x, T y)
{
    return T((x + y - 1) / y) * y;
}

//------------------------------------------------------------------------------
/// @return abs(r) + abs(i)
// std::abs is not yet labeled constexpr in C++ standard.
inline /*constexpr*/ float cabs1(float x)
{
    return std::abs(x);
}

inline /*constexpr*/ double cabs1(double x)
{
    return std::abs(x);
}

inline /*constexpr*/ float cabs1(std::complex<float> x)
{
    return float(std::abs(x.real()) + std::abs(x.imag()));
}

inline /*constexpr*/ double cabs1(std::complex<double> x)
{
    return double(std::abs(x.real()) + std::abs(x.imag()));
}

//------------------------------------------------------------------------------
class ThreadBarrier {
public:
    ThreadBarrier()
        : count_(0),
          passed_(0)
    {}

    void wait(int size)
    {
        int passed_old = passed_;

        __sync_fetch_and_add(&count_, 1);
        if (__sync_bool_compare_and_swap(&count_, size, 0))
            passed_++;
        else
            while (passed_ == passed_old);
    }

private:
    int count_;
    volatile int passed_;
};

//------------------------------------------------------------------------------
/// Use to silence compiler warnings regarding an unused variable var.
#define SLATE_UNUSED(var)  ((void)var)

} // namespace slate

#endif // SLATE_UTIL_HH
