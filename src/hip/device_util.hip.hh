// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_DEVICE_UTIL_CUH
#define SLATE_DEVICE_UTIL_CUH

#include <hip/hip_complex.h>

namespace slate {
namespace device {

//------------------------------------------------------------------------------
/// max that propagates nan consistently:
///
///     max_nan( 1,   nan ) = nan
///     max_nan( nan, 1   ) = nan
///
template <typename real_t>
__host__ __device__
inline real_t max_nan(real_t x, real_t y)
{
    return (isnan(y) || (y) >= (x) ? (y) : (x));
}

//------------------------------------------------------------------------------
/// Max reduction of n-element array x, leaving total in x[0]. Propagates NaN
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

//==============================================================================
// real, imag, conj.

#ifdef __NVCC__

// CUDA doesn't provide real, imag, conj.
__host__ __device__ inline double real( rocblas_double_complex x ) { return x.x; }
__host__ __device__ inline float  real( rocblas_float_complex  x ) { return x.x; }

__host__ __device__ inline double imag( rocblas_double_complex x ) { return x.y; }
__host__ __device__ inline float  imag( rocblas_float_complex  x ) { return x.y; }

__host__ __device__ inline rocblas_double_complex conj( rocblas_double_complex x ) { return hipConj(x); }
__host__ __device__ inline rocblas_float_complex  conj( rocblas_float_complex  x ) { return hipConjf(x); }

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
__host__ __device__  inline double real( double x ) { return x; }
__host__ __device__  inline float  real( float  x ) { return x; }

/// @return imaginary component of complex number x; 0 for real number.
/// @ingroup complex
__host__ __device__  inline double imag( double x ) { return 0; }
__host__ __device__  inline float  imag( float  x ) { return 0; }

/// @return conjugate of complex number x; x for real number.
/// @ingroup complex
__host__ __device__  inline double conj( double x ) { return x; }
__host__ __device__  inline float  conj( float  x ) { return x; }

//------------------------------------------------------------------------------
/// Overloaded versions of absolute value on device.
__host__ __device__
inline float abs(float x)
{
    return fabsf(x);
}

//----------------------------------------
__host__ __device__
inline double abs(double x)
{
    return fabs(x);
}

//----------------------------------------
__host__ __device__
inline float abs(rocblas_float_complex x)
{
#ifdef __NVCC__
    // CUDA has a good implementation.
    return hipCabsf(x);
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
__host__ __device__
inline double abs(rocblas_double_complex x)
{
#ifdef __NVCC__
    // CUDA has a good implementation.
    return hipCabs(x);
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
__host__ __device__
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
__host__ __device__
void combine_sumsq(
    real_t& scale1, real_t& sumsq1,
    real_t  scale2, real_t  sumsq2 )
{
    if (scale1 >= scale2) {
        if (scale1 != 0 && ! isinf( scale1 ))
            sumsq1 += sumsq2*sqr( scale2 / scale1 );
        else
            sumsq1 += sumsq2;  // in case sumsq2 is NaN
        // scale1 stays same
    }
    else {
        sumsq1 = sumsq1*sqr( scale1 / scale2 ) + sumsq2;
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
__host__ __device__
void add_sumsq(
    real_t& scale, real_t& sumsq,
    real_t absx)
{
    if (absx > 0 || isnan( absx )) {
        if (isinf( scale )) {
            sumsq += absx;  // in case absx is NaN
        }
        else if (scale < absx) {
            sumsq = 1 + sumsq * sqr( scale / absx );
            scale = absx;
        }
        else {
            sumsq += sqr( absx / scale );
            // scale unchanged
        }
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

//------------------------------------------------------------------------------
/// @return ceil( x / y )*y, i.e., x rounded up to next multiple of y.
template <typename T>
__host__ __device__
inline constexpr T roundup(T x, T y)
{
    return T((x + y - 1) / y) * y;
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
inline void copy(rocblas_float_complex a, rocblas_double_complex& b)
{
    b = rocblas_double_complex( real( a ), imag( a ) );
}

/// Sets b = a, converting from complex-double to complex-float.
__host__ __device__
inline void copy(rocblas_double_complex a, rocblas_float_complex& b)
{
    b = rocblas_float_complex( real( a ), imag( a ) );
}

/// Sets b = a, converting from float to complex-float.
__host__ __device__
inline void copy( float a, rocblas_float_complex& b )
{
    b = rocblas_float_complex( a, 0 );
}

/// Sets b = a, converting from double to complex-double.
__host__ __device__
inline void copy( double a, rocblas_double_complex& b )
{
    b = rocblas_double_complex( a, 0 );
}

//==============================================================================
// CUDA doesn't provide operators, so define our own.
// rocBLAS provides operators.
//
// complex-double

#if defined( BLAS_HAVE_CUBLAS )

// ---------- negate
__host__ __device__  inline rocblas_double_complex
operator - (const rocblas_double_complex& a)
{
    return rocblas_double_complex( -real(a),
                                 -imag(a) );
}


__host__ __device__  inline rocblas_double_complex
operator + (const rocblas_double_complex a, const rocblas_double_complex b)
{
    return rocblas_double_complex( real(a) + real(b),
                                 imag(a) + imag(b) );
}

__host__ __device__  inline rocblas_double_complex
operator + (const rocblas_double_complex a, const double s)
{
    return rocblas_double_complex( real(a) + s,
                                 imag(a) );
}

__host__ __device__  inline rocblas_double_complex
operator + (const double s, const rocblas_double_complex b)
{
    return rocblas_double_complex( s + real(b),
                                 imag(b) );
}

__host__ __device__  inline rocblas_double_complex&
operator += (rocblas_double_complex& a, const rocblas_double_complex b)
{
    a = rocblas_double_complex( real(a) + real(b),
                              imag(a) + imag(b) );
    return a;
}

__host__ __device__  inline rocblas_double_complex&
operator += (rocblas_double_complex& a, const double s)
{
    a = rocblas_double_complex( real(a) + s,
                              imag(a) );
    return a;
}

// ---------- subtract
__host__ __device__  inline rocblas_double_complex
operator - (const rocblas_double_complex a, const rocblas_double_complex b)
{
    return rocblas_double_complex( real(a) - real(b),
                                 imag(a) - imag(b) );
}

__host__ __device__  inline rocblas_double_complex
operator - (const rocblas_double_complex a, const double s)
{
    return rocblas_double_complex( real(a) - s,
                                 imag(a) );
}

__host__ __device__  inline rocblas_double_complex
operator - (const double s, const rocblas_double_complex b)
{
    return rocblas_double_complex( s - real(b),
                                 - imag(b) );
}

__host__ __device__  inline rocblas_double_complex&
operator -= (rocblas_double_complex& a, const rocblas_double_complex b)
{
    a = rocblas_double_complex( real(a) - real(b),
                              imag(a) - imag(b) );
    return a;
}

__host__ __device__  inline rocblas_double_complex&
operator -= (rocblas_double_complex& a, const double s)
{
    a = rocblas_double_complex( real(a) - s,
                              imag(a) );
    return a;
}

// ---------- multiply
__host__ __device__  inline rocblas_double_complex
operator * (const rocblas_double_complex a, const rocblas_double_complex b)
{
    return rocblas_double_complex( real(a)*real(b) - imag(a)*imag(b),
                                 imag(a)*real(b) + real(a)*imag(b) );
}

__host__ __device__  inline rocblas_double_complex
operator * (const rocblas_double_complex a, const double s)
{
    return rocblas_double_complex( real(a)*s,
                                 imag(a)*s );
}

__host__ __device__  inline rocblas_double_complex
operator * (const rocblas_double_complex a, const float s)
{
    return rocblas_double_complex( real(a)*s,
                                 imag(a)*s );
}



__host__ __device__  inline rocblas_double_complex
operator * (const double s, const rocblas_double_complex a)
{
    return rocblas_double_complex( real(a)*s,
                                 imag(a)*s );
}

__host__ __device__  inline rocblas_double_complex&
operator *= (rocblas_double_complex& a, const rocblas_double_complex b)
{
    a = rocblas_double_complex( real(a)*real(b) - imag(a)*imag(b),
                              imag(a)*real(b) + real(a)*imag(b) );
    return a;
}

__host__ __device__  inline rocblas_double_complex&
operator *= (rocblas_double_complex& a, const double s)
{
    a = rocblas_double_complex( real(a)*s,
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
__host__ __device__  inline rocblas_double_complex
operator / (const rocblas_double_complex x, const rocblas_double_complex y)
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
    return rocblas_double_complex( p, q );
}

__host__ __device__  inline rocblas_double_complex
operator / (const rocblas_double_complex a, const double s)
{
    return rocblas_double_complex( real(a)/s,
                                 imag(a)/s );
}

__host__ __device__  inline rocblas_double_complex
operator / (const double a, const rocblas_double_complex y)
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
    return rocblas_double_complex( p, q );
}

__host__ __device__  inline rocblas_double_complex&
operator /= (rocblas_double_complex& a, const rocblas_double_complex b)
{
    a = a/b;
    return a;
}

__host__ __device__  inline rocblas_double_complex&
operator /= (rocblas_double_complex& a, const double s)
{
    a = rocblas_double_complex( real(a)/s,
                              imag(a)/s );
    return a;
}

//==============================================================================
// complex-float

// ---------- negate
__host__ __device__  inline rocblas_float_complex
operator - (const rocblas_float_complex& a)
{
    return rocblas_float_complex( -real(a), -imag(a) );
}

// ---------- add
__host__ __device__  inline rocblas_float_complex
operator + (const rocblas_float_complex a, const rocblas_float_complex b)
{
    return rocblas_float_complex( real(a) + real(b),
                                imag(a) + imag(b) );
}

__host__ __device__  inline rocblas_float_complex
operator + (const rocblas_float_complex a, const float s)
{
    return rocblas_float_complex( real(a) + s,
                                imag(a) );
}

__host__ __device__  inline rocblas_float_complex
operator + (const float s, const rocblas_float_complex b)
{
    return rocblas_float_complex( s + real(b),
                                imag(b) );
}

__host__ __device__  inline rocblas_float_complex&
operator += (rocblas_float_complex& a, const rocblas_float_complex b)
{
    a = rocblas_float_complex( real(a) + real(b),
                             imag(a) + imag(b) );
    return a;
}

__host__ __device__  inline rocblas_float_complex&
operator += (rocblas_float_complex& a, const float s)
{
    a = rocblas_float_complex( real(a) + s,
                             imag(a) );
    return a;
}


// ---------- subtract
__host__ __device__  inline rocblas_float_complex
operator - (const rocblas_float_complex a, const rocblas_float_complex b)
{
    return rocblas_float_complex( real(a) - real(b),
                                imag(a) - imag(b) );
}

__host__ __device__  inline rocblas_float_complex
operator - (const rocblas_float_complex a, const float s)
{
    return rocblas_float_complex( real(a) - s,
                                imag(a) );
}

__host__ __device__  inline rocblas_float_complex
operator - (const float s, const rocblas_float_complex b)
{
    return rocblas_float_complex( s - real(b),
                                - imag(b) );
}

__host__ __device__  inline rocblas_float_complex&
operator -= (rocblas_float_complex& a, const rocblas_float_complex b)
{
    a = rocblas_float_complex( real(a) - real(b),
                             imag(a) - imag(b) );
    return a;
}

__host__ __device__  inline rocblas_float_complex&
operator -= (rocblas_float_complex& a, const float s)
{
    a = rocblas_float_complex( real(a) - s,
                             imag(a) );
    return a;
}


// ---------- multiply
__host__ __device__  inline rocblas_float_complex
operator * (const rocblas_float_complex a, const rocblas_float_complex b)
{
    return rocblas_float_complex( real(a)*real(b) - imag(a)*imag(b),
                                imag(a)*real(b) + real(a)*imag(b) );
}

__host__ __device__  inline rocblas_float_complex
operator * (const rocblas_float_complex a, const float s)
{
    return rocblas_float_complex( real(a)*s,
                                imag(a)*s );
}

__host__ __device__  inline rocblas_float_complex
operator * (const float s, const rocblas_float_complex a)
{
    return rocblas_float_complex( real(a)*s,
                                imag(a)*s );
}

__host__ __device__  inline rocblas_float_complex&
operator *= (rocblas_float_complex& a, const rocblas_float_complex b)
{
    a = rocblas_float_complex( real(a)*real(b) - imag(a)*imag(b),
                             imag(a)*real(b) + real(a)*imag(b) );
    return a;
}

__host__ __device__  inline rocblas_float_complex&
operator *= (rocblas_float_complex& a, const float s)
{
    a = rocblas_float_complex( real(a)*s,
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
__host__ __device__  inline rocblas_float_complex
operator / (const rocblas_float_complex x, const rocblas_float_complex y)
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
    return rocblas_float_complex( p, q );
}

__host__ __device__  inline rocblas_float_complex
operator / (const rocblas_float_complex a, const float s)
{
    return rocblas_float_complex( real(a)/s,
                                imag(a)/s );
}

__host__ __device__  inline rocblas_float_complex
operator / (const float a, const rocblas_float_complex y)
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
    return rocblas_float_complex( p, q );
}

__host__ __device__  inline rocblas_float_complex&
operator /= (rocblas_float_complex& a, const rocblas_float_complex b)
{
    a = a/b;
    return a;
}

__host__ __device__  inline rocblas_float_complex&
operator /= (rocblas_float_complex& a, const float s)
{
    a = rocblas_float_complex( real(a)/s,
                             imag(a)/s );
    return a;
}


// ---------- equality
__host__ __device__  inline bool
operator == (const rocblas_float_complex a, const rocblas_float_complex b)
{
    return ( real(a) == real(b) &&
             imag(a) == imag(b) );
}

__host__ __device__  inline bool
operator == (const rocblas_float_complex a, const float s)
{
    return ( real(a) == s &&
             imag(a) == 0. );
}

__host__ __device__  inline bool
operator == (const float s, const rocblas_float_complex a)
{
    return ( real(a) == s &&
             imag(a) == 0. );
}


// ---------- not equality
__host__ __device__  inline bool
operator != (const rocblas_float_complex a, const rocblas_float_complex b)
{
    return ! (a == b);
}

__host__ __device__  inline bool
operator != (const rocblas_float_complex a, const float s)
{
    return ! (a == s);
}

__host__ __device__  inline bool
operator != (const float s, const rocblas_float_complex a)
{
    return ! (a == s);
}

#endif // BLAS_WITH_CUBLAS

} // namespace device
} // namespace slate

#endif // SLATE_DEVICE_UTIL_CUH
