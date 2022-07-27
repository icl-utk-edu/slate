// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_DEVICE_UTIL_CUH
#define SLATE_DEVICE_UTIL_CUH

#include <hip/hip_complex.h>

namespace slate {
namespace device {

//------------------------------------------------------------------------------
/// max that propogates nan consistently:
///     max_nan( 1,   nan ) = nan
///     max_nan( nan, 1   ) = nan
template <typename real_t>
__host__ __device__
inline real_t max_nan(real_t x, real_t y)
{
    return (isnan(y) || (y) >= (x) ? (y) : (x));
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
__device__
void max_nan_reduce(int n, int tid, real_t* x)
{
    if (n > 1024) { if (tid < 1024 && tid + 1024 < n) { x[tid] = max_nan(x[tid], x[tid+1024]); }  __syncthreads(); }
    if (n >  512) { if (tid <  512 && tid +  512 < n) { x[tid] = max_nan(x[tid], x[tid+ 512]); }  __syncthreads(); }
    if (n >  256) { if (tid <  256 && tid +  256 < n) { x[tid] = max_nan(x[tid], x[tid+ 256]); }  __syncthreads(); }
    if (n >  128) { if (tid <  128 && tid +  128 < n) { x[tid] = max_nan(x[tid], x[tid+ 128]); }  __syncthreads(); }
    if (n >   64) { if (tid <   64 && tid +   64 < n) { x[tid] = max_nan(x[tid], x[tid+  64]); }  __syncthreads(); }
    if (n >   32) { if (tid <   32 && tid +   32 < n) { x[tid] = max_nan(x[tid], x[tid+  32]); }  __syncthreads(); }
    if (n >   16) { if (tid <   16 && tid +   16 < n) { x[tid] = max_nan(x[tid], x[tid+  16]); }  __syncthreads(); }
    if (n >    8) { if (tid <    8 && tid +    8 < n) { x[tid] = max_nan(x[tid], x[tid+   8]); }  __syncthreads(); }
    if (n >    4) { if (tid <    4 && tid +    4 < n) { x[tid] = max_nan(x[tid], x[tid+   4]); }  __syncthreads(); }
    if (n >    2) { if (tid <    2 && tid +    2 < n) { x[tid] = max_nan(x[tid], x[tid+   2]); }  __syncthreads(); }
    if (n >    1) { if (tid <    1 && tid +    1 < n) { x[tid] = max_nan(x[tid], x[tid+   1]); }  __syncthreads(); }
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
__device__
void sum_reduce(int n, int tid, real_t* x)
{
    if (n > 1024) { if (tid < 1024 && tid + 1024 < n) { x[tid] += x[tid+1024]; }  __syncthreads(); }
    if (n >  512) { if (tid <  512 && tid +  512 < n) { x[tid] += x[tid+ 512]; }  __syncthreads(); }
    if (n >  256) { if (tid <  256 && tid +  256 < n) { x[tid] += x[tid+ 256]; }  __syncthreads(); }
    if (n >  128) { if (tid <  128 && tid +  128 < n) { x[tid] += x[tid+ 128]; }  __syncthreads(); }
    if (n >   64) { if (tid <   64 && tid +   64 < n) { x[tid] += x[tid+  64]; }  __syncthreads(); }
    if (n >   32) { if (tid <   32 && tid +   32 < n) { x[tid] += x[tid+  32]; }  __syncthreads(); }
    if (n >   16) { if (tid <   16 && tid +   16 < n) { x[tid] += x[tid+  16]; }  __syncthreads(); }
    if (n >    8) { if (tid <    8 && tid +    8 < n) { x[tid] += x[tid+   8]; }  __syncthreads(); }
    if (n >    4) { if (tid <    4 && tid +    4 < n) { x[tid] += x[tid+   4]; }  __syncthreads(); }
    if (n >    2) { if (tid <    2 && tid +    2 < n) { x[tid] += x[tid+   2]; }  __syncthreads(); }
    if (n >    1) { if (tid <    1 && tid +    1 < n) { x[tid] += x[tid+   1]; }  __syncthreads(); }
}

//------------------------------------------------------------------------------
/// Overloaded versions of absolute value on device.
__host__ __device__
inline float abs(float x)
{
    return fabsf(x);
}

__host__ __device__
inline double abs(double x)
{
    return fabs(x);
}

__host__ __device__
inline float abs(hipFloatComplex x)
{
    // CUDA has good implementation,
    // otherwise use our implementation that scales per LAPACK.
#ifdef __NVCC__
    return hipCabsf(x);
#else
    float a = hipCrealf(x);
    float b = hipCimagf(x);
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

__host__ __device__
inline double abs(hipDoubleComplex x)
{
    // CUDA has good implementation,
    // otherwise use our implementation that scales per LAPACK.
#ifdef __NVCC__
    return hipCabs(x);
#else
    double a = hipCreal(x);
    double b = hipCimag(x);
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
/// Overloaded versions of Ax+By on device.
template <typename T>
__host__ __device__
inline T axpby(T alpha, T x, T beta, T y)
{
    return alpha*x + beta*y;
}

__host__ __device__
inline hipFloatComplex axpby(hipFloatComplex alpha, hipFloatComplex x,
                            hipFloatComplex beta, hipFloatComplex y)
{
    return hipCaddf(hipCmulf(alpha, x), hipCmulf(beta, y));
}

__host__ __device__
inline hipDoubleComplex axpby(hipDoubleComplex alpha, hipDoubleComplex x,
                             hipDoubleComplex beta, hipDoubleComplex y)
{
    return hipCadd(hipCmul(alpha, x), hipCmul(beta, y));
}

//------------------------------------------------------------------------------
/// Overloaded copy and precision conversion.
/// Sets b = a, converting from type TA to type TB.
template <typename TA, typename TB>
__host__ __device__
inline void copy(TA a, TB& b)
{
    b = a;
}

/// Sets b = a, converting from complex-float to complex-double.
__host__ __device__
inline void copy(hipFloatComplex a, hipDoubleComplex& b)
{
    b.x = a.x;
    b.y = a.y;
}

/// Sets b = a, converting from complex-double to complex-float.
__host__ __device__
inline void copy(hipDoubleComplex a, hipFloatComplex& b)
{
    b.x = a.x;
    b.y = a.y;
}

//------------------------------------------------------------------------------
/// Square of number.
/// @return x^2
template <typename scalar_t>
__host__ __device__
inline scalar_t sqr(scalar_t x)
{
    return x*x;
}

//------------------------------------------------------------------------------
/// Adds two scaled, sum-of-squares representations.
/// On exit, scale1 and sumsq1 are updated such that:
///     scale1^2 sumsq1 := scale1^2 sumsq1 + scale2^2 sumsq2.
template <typename real_t>
__host__ __device__
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
__host__ __device__
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
__host__ __device__
inline constexpr T ceildiv(T x, T y)
{
    return T((x + y - 1) / y);
}

/// @return ceil( x / y )*y, i.e., x rounded up to next multiple of y.
template <typename T>
__host__ __device__
inline constexpr T roundup(T x, T y)
{
    return T((x + y - 1) / y) * y;
}
__host__ __device__  inline double real(hipDoubleComplex x) { return x.x; }
__host__ __device__  inline float  real(hipFloatComplex  x) { return  x.x; }

__host__ __device__  inline double imag(hipDoubleComplex x) { return  x.y; }
__host__ __device__  inline float  imag(hipFloatComplex  x) { return x.y;  }

__host__ __device__  inline hipDoubleComplex conj(hipDoubleComplex x) { return hipConj(x); }
__host__ __device__  inline hipFloatComplex  conj(hipFloatComplex  x) { return hipConjf(x); }
//#endif

__host__ __device__  inline double real(double             x) { return x; }
__host__ __device__  inline float  real(float              x) { return x; }

/// @return imaginary component of complex number x; 0 for real number.
/// @ingroup complex
__host__ __device__  inline double imag(double             x) { return 0.; }
__host__ __device__  inline float  imag(float              x) { return 0.f; }

/// @return conjugate of complex number x; x for real number.
/// @ingroup complex
__host__ __device__  inline double conj(double             x) { return x; }
__host__ __device__  inline float  conj(float              x) { return x; }

#if defined( BLAS_HAVE_CUBLAS )

// ---------- negate
__host__ __device__  inline hipDoubleComplex
operator - (const hipDoubleComplex& a)
{
    return make_hipDoubleComplex( -real(a),
                                 -imag(a) );
}


__host__ __device__  inline hipDoubleComplex
operator + (const hipDoubleComplex a, const hipDoubleComplex b)
{
    return make_hipDoubleComplex( real(a) + real(b),
                                 imag(a) + imag(b) );
}

__host__ __device__  inline hipDoubleComplex
operator + (const hipDoubleComplex a, const double s)
{
    return make_hipDoubleComplex( real(a) + s,
                                 imag(a) );
}

__host__ __device__  inline hipDoubleComplex
operator + (const double s, const hipDoubleComplex b)
{
    return make_hipDoubleComplex( s + real(b),
                                 imag(b) );
}

__host__ __device__  inline hipDoubleComplex&
operator += (hipDoubleComplex& a, const hipDoubleComplex b)
{
    a = make_hipDoubleComplex( real(a) + real(b),
                              imag(a) + imag(b) );
    return a;
}

__host__ __device__  inline hipDoubleComplex&
operator += (hipDoubleComplex& a, const double s)
{
    a = make_hipDoubleComplex( real(a) + s,
                              imag(a) );
    return a;
}

// ---------- subtract
__host__ __device__  inline hipDoubleComplex
operator - (const hipDoubleComplex a, const hipDoubleComplex b)
{
    return make_hipDoubleComplex( real(a) - real(b),
                                 imag(a) - imag(b) );
}

__host__ __device__  inline hipDoubleComplex
operator - (const hipDoubleComplex a, const double s)
{
    return make_hipDoubleComplex( real(a) - s,
                                 imag(a) );
}

__host__ __device__  inline hipDoubleComplex
operator - (const double s, const hipDoubleComplex b)
{
    return make_hipDoubleComplex( s - real(b),
                                 - imag(b) );
}

__host__ __device__  inline hipDoubleComplex&
operator -= (hipDoubleComplex& a, const hipDoubleComplex b)
{
    a = make_hipDoubleComplex( real(a) - real(b),
                              imag(a) - imag(b) );
    return a;
}

__host__ __device__  inline hipDoubleComplex&
operator -= (hipDoubleComplex& a, const double s)
{
    a = make_hipDoubleComplex( real(a) - s,
                              imag(a) );
    return a;
}

// ---------- multiply
__host__ __device__  inline hipDoubleComplex
operator * (const hipDoubleComplex a, const hipDoubleComplex b)
{
    return make_hipDoubleComplex( real(a)*real(b) - imag(a)*imag(b),
                                 imag(a)*real(b) + real(a)*imag(b) );
}

__host__ __device__  inline hipDoubleComplex
operator * (const hipDoubleComplex a, const double s)
{
    return make_hipDoubleComplex( real(a)*s,
                                 imag(a)*s );
}

__host__ __device__  inline hipDoubleComplex
operator * (const hipDoubleComplex a, const float s)
{
    return make_hipDoubleComplex( real(a)*s,
                                 imag(a)*s );
}



__host__ __device__  inline hipDoubleComplex
operator * (const double s, const hipDoubleComplex a)
{
    return make_hipDoubleComplex( real(a)*s,
                                 imag(a)*s );
}

__host__ __device__  inline hipDoubleComplex&
operator *= (hipDoubleComplex& a, const hipDoubleComplex b)
{
    a = make_hipDoubleComplex( real(a)*real(b) - imag(a)*imag(b),
                              imag(a)*real(b) + real(a)*imag(b) );
    return a;
}

__host__ __device__  inline hipDoubleComplex&
operator *= (hipDoubleComplex& a, const double s)
{
    a = make_hipDoubleComplex( real(a)*s,
                              imag(a)*s );
    return a;
}

// ---------- divide
/* From LAPACK DLADIV
 * Performs complex division in real arithmetic, avoiding unnecessary overflow.
 *
 *             a + i*b
 *  p + i*q = ---------
 *             c + i*d
 */
__host__ __device__  inline hipDoubleComplex
operator / (const hipDoubleComplex x, const hipDoubleComplex y)
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
    return make_hipDoubleComplex( p, q );
}

__host__ __device__  inline hipDoubleComplex
operator / (const hipDoubleComplex a, const double s)
{
    return make_hipDoubleComplex( real(a)/s,
                                 imag(a)/s );
}

__host__ __device__  inline hipDoubleComplex
operator / (const double a, const hipDoubleComplex y)
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
    return make_hipDoubleComplex( p, q );
}

__host__ __device__  inline hipDoubleComplex&
operator /= (hipDoubleComplex& a, const hipDoubleComplex b)
{
    a = a/b;
    return a;
}

__host__ __device__  inline hipDoubleComplex&
operator /= (hipDoubleComplex& a, const double s)
{
    a = make_hipDoubleComplex( real(a)/s,
                              imag(a)/s );
    return a;
}

// =============================================================================
// hipFloatComplex

// ---------- negate
__host__ __device__  inline hipFloatComplex
operator - (const hipFloatComplex& a)
{
    return make_hipFloatComplex( -real(a), -imag(a) );
}

// ---------- add
__host__ __device__  inline hipFloatComplex
operator + (const hipFloatComplex a, const hipFloatComplex b)
{
    return make_hipFloatComplex( real(a) + real(b),
                                imag(a) + imag(b) );
}

__host__ __device__  inline hipFloatComplex
operator + (const hipFloatComplex a, const float s)
{
    return make_hipFloatComplex( real(a) + s,
                                imag(a) );
}

__host__ __device__  inline hipFloatComplex
operator + (const float s, const hipFloatComplex b)
{
    return make_hipFloatComplex( s + real(b),
                                imag(b) );
}

__host__ __device__  inline hipFloatComplex&
operator += (hipFloatComplex& a, const hipFloatComplex b)
{
    a = make_hipFloatComplex( real(a) + real(b),
                             imag(a) + imag(b) );
    return a;
}

__host__ __device__  inline hipFloatComplex&
operator += (hipFloatComplex& a, const float s)
{
    a = make_hipFloatComplex( real(a) + s,
                             imag(a) );
    return a;
}


// ---------- subtract
__host__ __device__  inline hipFloatComplex
operator - (const hipFloatComplex a, const hipFloatComplex b)
{
    return make_hipFloatComplex( real(a) - real(b),
                                imag(a) - imag(b) );
}

__host__ __device__  inline hipFloatComplex
operator - (const hipFloatComplex a, const float s)
{
    return make_hipFloatComplex( real(a) - s,
                                imag(a) );
}

__host__ __device__  inline hipFloatComplex
operator - (const float s, const hipFloatComplex b)
{
    return make_hipFloatComplex( s - real(b),
                                - imag(b) );
}

__host__ __device__  inline hipFloatComplex&
operator -= (hipFloatComplex& a, const hipFloatComplex b)
{
    a = make_hipFloatComplex( real(a) - real(b),
                             imag(a) - imag(b) );
    return a;
}

__host__ __device__  inline hipFloatComplex&
operator -= (hipFloatComplex& a, const float s)
{
    a = make_hipFloatComplex( real(a) - s,
                             imag(a) );
    return a;
}


// ---------- multiply
__host__ __device__  inline hipFloatComplex
operator * (const hipFloatComplex a, const hipFloatComplex b)
{
    return make_hipFloatComplex( real(a)*real(b) - imag(a)*imag(b),
                                imag(a)*real(b) + real(a)*imag(b) );
}

__host__ __device__  inline hipFloatComplex
operator * (const hipFloatComplex a, const float s)
{
    return make_hipFloatComplex( real(a)*s,
                                imag(a)*s );
}

__host__ __device__  inline hipFloatComplex
operator * (const float s, const hipFloatComplex a)
{
    return make_hipFloatComplex( real(a)*s,
                                imag(a)*s );
}

__host__ __device__  inline hipFloatComplex&
operator *= (hipFloatComplex& a, const hipFloatComplex b)
{
    a = make_hipFloatComplex( real(a)*real(b) - imag(a)*imag(b),
                             imag(a)*real(b) + real(a)*imag(b) );
    return a;
}

__host__ __device__  inline hipFloatComplex&
operator *= (hipFloatComplex& a, const float s)
{
    a = make_hipFloatComplex( real(a)*s,
                             imag(a)*s );
    return a;
}


// ---------- divide
/* From LAPACK DLADIV
 * Performs complex division in real arithmetic, avoiding unnecessary overflow.
 *
 *             a + i*b
 *  p + i*q = ---------
 *             c + i*d
 */
__host__ __device__  inline hipFloatComplex
operator / (const hipFloatComplex x, const hipFloatComplex y)
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
    return make_hipFloatComplex( p, q );
}

__host__ __device__  inline hipFloatComplex
operator / (const hipFloatComplex a, const float s)
{
    return make_hipFloatComplex( real(a)/s,
                                imag(a)/s );
}

__host__ __device__  inline hipFloatComplex
operator / (const float a, const hipFloatComplex y)
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
    return make_hipFloatComplex( p, q );
}

__host__ __device__  inline hipFloatComplex&
operator /= (hipFloatComplex& a, const hipFloatComplex b)
{
    a = a/b;
    return a;
}

__host__ __device__  inline hipFloatComplex&
operator /= (hipFloatComplex& a, const float s)
{
    a = make_hipFloatComplex( real(a)/s,
                             imag(a)/s );
    return a;
}


// ---------- equality
__host__ __device__  inline bool
operator == (const hipFloatComplex a, const hipFloatComplex b)
{
    return ( real(a) == real(b) &&
             imag(a) == imag(b) );
}

__host__ __device__  inline bool
operator == (const hipFloatComplex a, const float s)
{
    return ( real(a) == s &&
             imag(a) == 0. );
}

__host__ __device__  inline bool
operator == (const float s, const hipFloatComplex a)
{
    return ( real(a) == s &&
             imag(a) == 0. );
}


// ---------- not equality
__host__ __device__  inline bool
operator != (const hipFloatComplex a, const hipFloatComplex b)
{
    return ! (a == b);
}

__host__ __device__  inline bool
operator != (const hipFloatComplex a, const float s)
{
    return ! (a == s);
}

__host__ __device__  inline bool
operator != (const float s, const hipFloatComplex a)
{
    return ! (a == s);
}

#endif // BLAS_WITH_CUBLAS

} // namespace device
} // namespace slate

#endif // SLATE_DEVICE_UTIL_CUH
