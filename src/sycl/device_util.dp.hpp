// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_DEVICE_UTIL_CUH
#define SLATE_DEVICE_UTIL_CUH

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <complex>

namespace slate {
namespace device {

//------------------------------------------------------------------------------
/// max that propogates nan consistently:
///     max_nan( 1,   nan ) = nan
///     max_nan( nan, 1   ) = nan
template <typename real_t>

inline real_t max_nan(real_t x, real_t y)
{
    return (sycl::isnan(y) || (y) >= (x) ? (y) : (x));
}

//------------------------------------------------------------------------------
/// Max reduction of n-element array x, leaving total in x[0]. Propogates NaN
/// values consistently.
/// With k threads, can reduce array up to 2*k in size. Assumes number of
/// threads <= 1024, which is the current max number of CUDA threads.
///
/// @param[in] n
///     Size of array.
///
/// @param[in] tid
///     Thread id.
///
/// @param[in] x
///     Array of dimension n. On exit, x[0] = max(x[0], ..., x[n-1]);
///     the rest of x is overwritten.
///
template <typename real_t>

void max_nan_reduce(int n, int tid, real_t* x, const sycl::nd_item<3> &item_ct1)
{
    /*
      DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
    */
    if (n > 1024) {
        if (tid < 1024 && tid + 1024 < n) {
            x[tid] = max_nan(x[tid], x[tid + 1024]);
        } item_ct1.barrier();
    }
    /*
      DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
    */
    if (n > 512) {
        if (tid < 512 && tid + 512 < n) {
            x[tid] = max_nan(x[tid], x[tid + 512]);
        } item_ct1.barrier();
    }
    /*
      DPCT1065:2: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
    */
    if (n > 256) {
        if (tid < 256 && tid + 256 < n) {
            x[tid] = max_nan(x[tid], x[tid + 256]);
        } item_ct1.barrier();
    }
    /*
      DPCT1065:3: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
    */
    if (n > 128) {
        if (tid < 128 && tid + 128 < n) {
            x[tid] = max_nan(x[tid], x[tid + 128]);
        } item_ct1.barrier();
    }
    /*
      DPCT1065:4: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
    */
    if (n > 64) {
        if (tid < 64 && tid + 64 < n) {
            x[tid] = max_nan(x[tid], x[tid + 64]);
        } item_ct1.barrier();
    }
    /*
      DPCT1065:5: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
    */
    if (n > 32) {
        if (tid < 32 && tid + 32 < n) {
            x[tid] = max_nan(x[tid], x[tid + 32]);
        } item_ct1.barrier();
    }
    /*
      DPCT1065:6: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
    */
    if (n > 16) {
        if (tid < 16 && tid + 16 < n) {
            x[tid] = max_nan(x[tid], x[tid + 16]);
        } item_ct1.barrier();
    }
    /*
      DPCT1065:7: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
    */
    if (n > 8) {
        if (tid < 8 && tid + 8 < n) {
            x[tid] = max_nan(x[tid], x[tid + 8]);
        } item_ct1.barrier();
    }
    /*
      DPCT1065:8: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
    */
    if (n > 4) {
        if (tid < 4 && tid + 4 < n) {
            x[tid] = max_nan(x[tid], x[tid + 4]);
        } item_ct1.barrier();
    }
    /*
      DPCT1065:9: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
    */
    if (n > 2) {
        if (tid < 2 && tid + 2 < n) {
            x[tid] = max_nan(x[tid], x[tid + 2]);
        } item_ct1.barrier();
    }
    /*
      DPCT1065:10: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
    */
    if (n > 1) {
        if (tid < 1 && tid + 1 < n) {
            x[tid] = max_nan(x[tid], x[tid + 1]);
        } item_ct1.barrier();
    }
}

//------------------------------------------------------------------------------
/// Sum reduction of n-element array x, leaving total in x[0].
/// With k threads, can reduce array up to 2*k in size. Assumes number of
/// threads <= 1024 (which is current max number of CUDA threads).
///
/// @param[in] n
///     Size of array.
///
/// @param[in] tid
///     Thread id.
///
/// @param[in] x
///     Array of dimension n. On exit, x[0] = sum(x[0], ..., x[n-1]);
///     rest of x is overwritten.
///
template <typename real_t>

void sum_reduce(int n, int tid, real_t* x, const sycl::nd_item<3> &item_ct1)
{
    /*
      DPCT1065:11: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
    */
    if (n > 1024) {
        if (tid < 1024 && tid + 1024 < n) {
            x[tid] += x[tid + 1024];
        } item_ct1.barrier();
    }
    /*
      DPCT1065:12: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
    */
    if (n > 512) {
        if (tid < 512 && tid + 512 < n) {
            x[tid] += x[tid + 512];
        } item_ct1.barrier();
    }
    /*
      DPCT1065:13: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
    */
    if (n > 256) {
        if (tid < 256 && tid + 256 < n) {
            x[tid] += x[tid + 256];
        } item_ct1.barrier();
    }
    /*
      DPCT1065:14: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
    */
    if (n > 128) {
        if (tid < 128 && tid + 128 < n) {
            x[tid] += x[tid + 128];
        } item_ct1.barrier();
    }
    /*
      DPCT1065:15: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
    */
    if (n > 64) {
        if (tid < 64 && tid + 64 < n) {
            x[tid] += x[tid + 64];
        } item_ct1.barrier();
    }
    /*
      DPCT1065:16: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
    */
    if (n > 32) {
        if (tid < 32 && tid + 32 < n) {
            x[tid] += x[tid + 32];
        } item_ct1.barrier();
    }
    /*
      DPCT1065:17: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
    */
    if (n > 16) {
        if (tid < 16 && tid + 16 < n) {
            x[tid] += x[tid + 16];
        } item_ct1.barrier();
    }
    /*
      DPCT1065:18: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
    */
    if (n > 8) {
        if (tid < 8 && tid + 8 < n) {
            x[tid] += x[tid + 8];
        } item_ct1.barrier();
    }
    /*
      DPCT1065:19: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
    */
    if (n > 4) {
        if (tid < 4 && tid + 4 < n) {
            x[tid] += x[tid + 4];
        } item_ct1.barrier();
    }
    /*
      DPCT1065:20: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
    */
    if (n > 2) {
        if (tid < 2 && tid + 2 < n) {
            x[tid] += x[tid + 2];
        } item_ct1.barrier();
    }
    /*
      DPCT1065:21: Consider replacing sycl::nd_item::barrier() with
      sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
    */
    if (n > 1) {
        if (tid < 1 && tid + 1 < n) {
            x[tid] += x[tid + 1];
        } item_ct1.barrier();
    }
}

//==============================================================================
// real, imag, conj.

#ifdef DPCT_COMPATIBILITY_TEMP

// CUDA doesn't provide real, imag, conj.
inline double real(sycl::double2 x) { return x.x(); }
inline float real(sycl::float2 x) { return x.x(); }

inline double imag(sycl::double2 x) { return x.y(); }
inline float imag(sycl::float2 x) { return x.y(); }

inline sycl::double2 conj(sycl::double2 x) { return dpct::conj<double>(x); }
inline sycl::float2 conj(sycl::float2 x) { return dpct::conj<float>(x); }

#else

__host__ __device__ inline double real( rocblas_double_complex x ) { return x.real(); }
__host__ __device__ inline float  real( rocblas_float_complex  x ) { return x.real(); }

__host__ __device__ inline double imag( rocblas_double_complex x ) { return x.imag(); }
__host__ __device__ inline float  imag( rocblas_float_complex  x ) { return x.imag(); }

__host__ __device__ inline rocblas_double_complex conj( rocblas_double_complex x ) { return { x.real(), -x.imag() }; }
__host__ __device__ inline rocblas_float_complex  conj( rocblas_float_complex  x ) { return { x.real(), -x.imag() }; }

#endif

//----------------------------------------
// Overloads for real numbers.

/// @return real component of complex number x; x for real number.
/// @ingroup complex
inline double real( double x ) { return x; }
inline float  real( float  x ) { return x; }

/// @return imaginary component of complex number x; 0 for real number.
/// @ingroup complex
inline double imag( double x ) { return 0; }
inline float  imag( float  x ) { return 0; }

/// @return conjugate of complex number x; x for real number.
/// @ingroup complex
inline double conj( double x ) { return x; }
inline float  conj( float  x ) { return x; }

//------------------------------------------------------------------------------
/// Overloaded versions of absolute value on device.

inline float abs(float x)
{
    return sycl::fabs(x);
}

//----------------------------------------

inline double abs(double x)
{
    return sycl::fabs(x);
}

//----------------------------------------

inline float abs(sycl::float2 x)
{
#ifdef DPCT_COMPATIBILITY_TEMP
    // CUDA has a good implementation.
    return dpct::cabs<float>(x);
#else
    // For HIP, use our implementation that scales per LAPACK.
    float a = real( x );
    float b = imag( x );
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
        w = max(a, b);
        z = min(a, b);
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
#endif
}

//----------------------------------------

inline double abs(sycl::double2 x)
{
#ifdef DPCT_COMPATIBILITY_TEMP
    // CUDA has a good implementation.
    return dpct::cabs<double>(x);
#else
    // For HIP, use our implementation that scales per LAPACK.
    double a = real( x );
    double b = imag( x );
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
        w = max(a, b);
        z = min(a, b);
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
#endif
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
///     scale1^2 sumsq1 := scale1^2 sumsq1 + scale2^2 sumsq2.
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
///     scale^2 sumsq := scale^2 sumsq + (absx)^2
template <typename real_t>

void add_sumsq(
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

//------------------------------------------------------------------------------
/// @return ceil( x / y ), for integer type T.
template <typename T>

inline constexpr T ceildiv(T x, T y)
{
    return T((x + y - 1) / y);
}

//------------------------------------------------------------------------------
/// @return ceil( x / y )*y, i.e., x rounded up to next multiple of y.
template <typename T>

inline constexpr T roundup(T x, T y)
{
    return T((x + y - 1) / y) * y;
}

//------------------------------------------------------------------------------
/// Overloaded copy and precision conversion.
/// Sets b = a, converting from type TA to type TB.
template <typename TA, typename TB>

inline void copy(TA a, TB& b)
{
    b = a;
}

/// Sets b = a, converting from complex-float to complex-double.

inline void copy(sycl::float2 a, sycl::double2 &b)
{
    b = sycl::double2(real(a), imag(a));
}

/// Sets b = a, converting from complex-double to complex-float.

inline void copy(sycl::double2 a, sycl::float2 &b)
{
    b = sycl::float2(real(a), imag(a));
}

/// Sets b = a, converting from float to complex-float.

inline void copy(float a, sycl::float2 &b)
{
    b = sycl::float2(a, 0);
}

/// Sets b = a, converting from double to complex-double.

inline void copy(double a, sycl::double2 &b)
{
    b = sycl::double2(a, 0);
}

//==============================================================================
// CUDA doesn't provide operators, so define our own.
// rocBLAS provides operators.
//
// complex-double

#if defined( BLAS_HAVE_SYCL )

// ---------- negate
/*
DPCT1011:83: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::double2 operator-(const sycl::double2 &a)
{
    return sycl::double2(-real(a), -imag(a));
}
} // namespace dpct_operator_overloading

/*
DPCT1011:84: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::double2 operator+(const sycl::double2 a, const sycl::double2 b)
{
    return sycl::double2(real(a) + real(b), imag(a) + imag(b));
}
} // namespace dpct_operator_overloading

/*
DPCT1011:85: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::double2 operator+(const sycl::double2 a, const double s)
{
    return sycl::double2(real(a) + s, imag(a));
}
} // namespace dpct_operator_overloading

/*
DPCT1011:86: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::double2 operator+(const double s, const sycl::double2 b)
{
    return sycl::double2(s + real(b), imag(b));
}
} // namespace dpct_operator_overloading

/*
DPCT1011:87: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::double2 &operator += (sycl::double2 &a, const sycl::double2 b)
{
    a = sycl::double2(real(a) + real(b), imag(a) + imag(b));
    return a;
}
} // namespace dpct_operator_overloading

/*
DPCT1011:88: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::double2 &operator += (sycl::double2 &a, const double s)
{
    a = sycl::double2(real(a) + s, imag(a));
    return a;
}
} // namespace dpct_operator_overloading

// ---------- subtract
/*
DPCT1011:89: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::double2 operator-(const sycl::double2 a, const sycl::double2 b)
{
    return sycl::double2(real(a) - real(b), imag(a) - imag(b));
}
} // namespace dpct_operator_overloading

/*
DPCT1011:90: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::double2 operator-(const sycl::double2 a, const double s)
{
    return sycl::double2(real(a) - s, imag(a));
}
} // namespace dpct_operator_overloading

/*
DPCT1011:91: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::double2 operator-(const double s, const sycl::double2 b)
{
    return sycl::double2(s - real(b), -imag(b));
}
} // namespace dpct_operator_overloading

/*
DPCT1011:92: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::double2 &operator -= (sycl::double2 &a, const sycl::double2 b)
{
    a = sycl::double2(real(a) - real(b), imag(a) - imag(b));
    return a;
}
} // namespace dpct_operator_overloading

/*
DPCT1011:93: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::double2 &operator -= (sycl::double2 &a, const double s)
{
    a = sycl::double2(real(a) - s, imag(a));
    return a;
}
} // namespace dpct_operator_overloading

// ---------- multiply
/*
DPCT1011:94: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::double2 operator*(const sycl::double2 a, const sycl::double2 b)
{
    return sycl::double2(real(a) * real(b) - imag(a) * imag(b),
                         imag(a) * real(b) + real(a) * imag(b));
}
} // namespace dpct_operator_overloading

/*
DPCT1011:95: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::double2 operator*(const sycl::double2 a, const double s)
{
    return sycl::double2(real(a) * s, imag(a) * s);
}
} // namespace dpct_operator_overloading

/*
DPCT1011:96: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::double2 operator*(const sycl::double2 a, const float s)
{
    return sycl::double2(real(a) * s, imag(a) * s);
}
} // namespace dpct_operator_overloading

/*
DPCT1011:97: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::double2 operator*(const double s, const sycl::double2 a)
{
    return sycl::double2(real(a) * s, imag(a) * s);
}
} // namespace dpct_operator_overloading

/*
DPCT1011:98: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::double2 &operator *= (sycl::double2 &a, const sycl::double2 b)
{
    a = sycl::double2(real(a) * real(b) - imag(a) * imag(b),
                      imag(a) * real(b) + real(a) * imag(b));
    return a;
}
} // namespace dpct_operator_overloading

/*
DPCT1011:99: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::double2 &operator *= (sycl::double2 &a, const double s)
{
    a = sycl::double2(real(a) * s, imag(a) * s);
    return a;
}
} // namespace dpct_operator_overloading

// ---------- divide
/* From LAPACK DLADIV
 * Performs complex division in real arithmetic, avoiding unnecessary overflow.
 *
 *             a + i*b
 *  p + i*q = ---------
 *             c + i*d
 */
/*
DPCT1011:100: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::double2 operator/(const sycl::double2 x, const sycl::double2 y)
{
    double a = real(x);
    double b = imag(x);
    double c = real(y);
    double d = imag(y);
    double e, f, p, q;
    if (abs( d ) < abs( c )) {
        e = d / c;
        f = c + d*e;
        p = ( a + b*e ) / f;
        q = ( b - a*e ) / f;
    }
    else {
        e = c / d;
        f = d + c*e;
        p = (  b + a*e ) / f;
        q = ( -a + b*e ) / f;
    }
    return sycl::double2(p, q);
}
} // namespace dpct_operator_overloading

/*
DPCT1011:101: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::double2 operator/(const sycl::double2 a, const double s)
{
    return sycl::double2(real(a) / s, imag(a) / s);
}
} // namespace dpct_operator_overloading

/*
DPCT1011:102: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::double2 operator/(const double a, const sycl::double2 y)
{
    double c = real(y);
    double d = imag(y);
    double e, f, p, q;
    if (abs( d ) < abs( c )) {
        e = d / c;
        f = c + d*e;
        p =  a   / f;
        q = -a*e / f;
    }
    else {
        e = c / d;
        f = d + c*e;
        p =  a*e / f;
        q = -a   / f;
    }
    return sycl::double2(p, q);
}
} // namespace dpct_operator_overloading

/*
DPCT1011:103: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::double2 &operator /= (sycl::double2 &a, const sycl::double2 b)
{
    a = dpct_operator_overloading::operator/(a, b);
    return a;
}
} // namespace dpct_operator_overloading

/*
DPCT1011:104: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::double2 &operator /= (sycl::double2 &a, const double s)
{
    a = sycl::double2(real(a) / s, imag(a) / s);
    return a;
}
} // namespace dpct_operator_overloading

//==============================================================================
// complex-float

// ---------- negate
/*
DPCT1011:105: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator-(const sycl::float2 &a)
{
    return sycl::float2(-real(a), -imag(a));
}
} // namespace dpct_operator_overloading

// ---------- add
/*
DPCT1011:106: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator+(const sycl::float2 a, const sycl::float2 b)
{
    return sycl::float2(real(a) + real(b), imag(a) + imag(b));
}
} // namespace dpct_operator_overloading

/*
DPCT1011:107: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator+(const sycl::float2 a, const float s)
{
    return sycl::float2(real(a) + s, imag(a));
}
} // namespace dpct_operator_overloading

/*
DPCT1011:108: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator+(const float s, const sycl::float2 b)
{
    return sycl::float2(s + real(b), imag(b));
}
} // namespace dpct_operator_overloading

/*
DPCT1011:109: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 &operator += (sycl::float2 &a, const sycl::float2 b)
{
    a = sycl::float2(real(a) + real(b), imag(a) + imag(b));
    return a;
}
} // namespace dpct_operator_overloading

/*
DPCT1011:110: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 &operator += (sycl::float2 &a, const float s)
{
    a = sycl::float2(real(a) + s, imag(a));
    return a;
}
} // namespace dpct_operator_overloading

// ---------- subtract
/*
DPCT1011:111: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator-(const sycl::float2 a, const sycl::float2 b)
{
    return sycl::float2(real(a) - real(b), imag(a) - imag(b));
}
} // namespace dpct_operator_overloading

/*
DPCT1011:112: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator-(const sycl::float2 a, const float s)
{
    return sycl::float2(real(a) - s, imag(a));
}
} // namespace dpct_operator_overloading

/*
DPCT1011:113: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator-(const float s, const sycl::float2 b)
{
    return sycl::float2(s - real(b), -imag(b));
}
} // namespace dpct_operator_overloading

/*
DPCT1011:114: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 &operator -= (sycl::float2 &a, const sycl::float2 b)
{
    a = sycl::float2(real(a) - real(b), imag(a) - imag(b));
    return a;
}
} // namespace dpct_operator_overloading

/*
DPCT1011:115: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 &operator -= (sycl::float2 &a, const float s)
{
    a = sycl::float2(real(a) - s, imag(a));
    return a;
}
} // namespace dpct_operator_overloading

// ---------- multiply
/*
DPCT1011:116: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator*(const sycl::float2 a, const sycl::float2 b)
{
    return sycl::float2(real(a) * real(b) - imag(a) * imag(b),
                        imag(a) * real(b) + real(a) * imag(b));
}
} // namespace dpct_operator_overloading

/*
DPCT1011:117: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator*(const sycl::float2 a, const float s)
{
    return sycl::float2(real(a) * s, imag(a) * s);
}
} // namespace dpct_operator_overloading

/*
DPCT1011:118: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator*(const float s, const sycl::float2 a)
{
    return sycl::float2(real(a) * s, imag(a) * s);
}
} // namespace dpct_operator_overloading

/*
DPCT1011:119: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 &operator *= (sycl::float2 &a, const sycl::float2 b)
{
    a = sycl::float2(real(a) * real(b) - imag(a) * imag(b),
                     imag(a) * real(b) + real(a) * imag(b));
    return a;
}
} // namespace dpct_operator_overloading

/*
DPCT1011:120: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 &operator *= (sycl::float2 &a, const float s)
{
    a = sycl::float2(real(a) * s, imag(a) * s);
    return a;
}
} // namespace dpct_operator_overloading

// ---------- divide
/* From LAPACK DLADIV
 * Performs complex division in real arithmetic, avoiding unnecessary overflow.
 *
 *             a + i*b
 *  p + i*q = ---------
 *             c + i*d
 */
/*
DPCT1011:121: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator/(const sycl::float2 x, const sycl::float2 y)
{
    float a = real(x);
    float b = imag(x);
    float c = real(y);
    float d = imag(y);
    float e, f, p, q;
    if (abs( d ) < abs( c )) {
        e = d / c;
        f = c + d*e;
        p = ( a + b*e ) / f;
        q = ( b - a*e ) / f;
    }
    else {
        e = c / d;
        f = d + c*e;
        p = (  b + a*e ) / f;
        q = ( -a + b*e ) / f;
    }
    return sycl::float2(p, q);
}
} // namespace dpct_operator_overloading

/*
DPCT1011:122: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator/(const sycl::float2 a, const float s)
{
    return sycl::float2(real(a) / s, imag(a) / s);
}
} // namespace dpct_operator_overloading

/*
DPCT1011:123: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 operator/(const float a, const sycl::float2 y)
{
    float c = real(y);
    float d = imag(y);
    float e, f, p, q;
    if (abs( d ) < abs( c )) {
        e = d / c;
        f = c + d*e;
        p =  a   / f;
        q = -a*e / f;
    }
    else {
        e = c / d;
        f = d + c*e;
        p =  a*e / f;
        q = -a   / f;
    }
    return sycl::float2(p, q);
}
} // namespace dpct_operator_overloading

/*
DPCT1011:124: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 &operator /= (sycl::float2 &a, const sycl::float2 b)
{
    a = dpct_operator_overloading::operator/(a, b);
    return a;
}
} // namespace dpct_operator_overloading

/*
DPCT1011:125: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline sycl::float2 &operator /= (sycl::float2 &a, const float s)
{
    a = sycl::float2(real(a) / s, imag(a) / s);
    return a;
}
} // namespace dpct_operator_overloading

// ---------- equality
/*
DPCT1011:126: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline bool operator *= =(const sycl::float2 a, const sycl::float2 b)
{
    return ( real(a) == real(b) &&
             imag(a) == imag(b) );
}
} // namespace dpct_operator_overloading

/*
DPCT1011:127: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline bool operator *= =(const sycl::float2 a, const float s)
{
    return ( real(a) == s &&
             imag(a) == 0. );
}
} // namespace dpct_operator_overloading

/*
DPCT1011:128: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline bool operator *= =(const float s, const sycl::float2 a)
{
    return ( real(a) == s &&
             imag(a) == 0. );
}
} // namespace dpct_operator_overloading

// ---------- not equality
/*
DPCT1011:129: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline bool operator != (const sycl::float2 a, const sycl::float2 b)
{
    return !(dpct_operator_overloading::operator *= =(a, b));
}
} // namespace dpct_operator_overloading

/*
DPCT1011:130: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline bool operator != (const sycl::float2 a, const float s)
{
    return !(dpct_operator_overloading::operator *= =(a, s));
}
} // namespace dpct_operator_overloading

/*
DPCT1011:131: The tool detected overloaded operators for built-in vector types,
which may conflict with the SYCL 2020 standard operators (see 4.14.2.1 Vec
interface). The tool inserted a namespace to avoid the conflict. Use SYCL 2020
standard operators instead.
*/
namespace dpct_operator_overloading {

inline bool operator != (const float s, const sycl::float2 a)
{
    return !(dpct_operator_overloading::operator *= =(a, s));
}
} // namespace dpct_operator_overloading

#endif // BLAS_WITH_CUBLAS

} // namespace device
} // namespace slate

#endif // SLATE_DEVICE_UTIL_CUH
