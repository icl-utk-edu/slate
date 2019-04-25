//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the University of Tennessee nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------
// This research was supported by the Exascale Computing Project (17-SC-20-SC),
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//------------------------------------------------------------------------------
// For assistance with SLATE, email <slate-user@icl.utk.edu>.
// You can also join the "SLATE User" Google group by going to
// https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user,
// signing in with your Google credentials, and then clicking "Join group".
//------------------------------------------------------------------------------

#ifndef SLATE_DEVICE_UTIL_CUH
#define SLATE_DEVICE_UTIL_CUH

#include <cuComplex.h>

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
inline float abs(cuFloatComplex x)
{
    return cuCabsf(x);
}

__host__ __device__
inline double abs(cuDoubleComplex x)
{
    return cuCabs(x);
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
inline cuFloatComplex axpby(cuFloatComplex alpha, cuFloatComplex x,
                            cuFloatComplex beta, cuFloatComplex y)
{
    return cuCaddf(cuCmulf(alpha, x), cuCmulf(beta, y));
}

__host__ __device__
inline cuDoubleComplex axpby(cuDoubleComplex alpha, cuDoubleComplex x,
                             cuDoubleComplex beta, cuDoubleComplex y)
{
    return cuCadd(cuCmul(alpha, x), cuCmul(beta, y));
}

//------------------------------------------------------------------------------
/// Overloaded copy and precision conversion.
template <typename TA, typename TB>
__host__ __device__
inline void copy(TA a, TB& b)
{
    b = a;
}

__host__ __device__
inline void copy(cuFloatComplex a, cuDoubleComplex& b)
{
    b.x = a.x;
    b.y = a.y;
}

__host__ __device__
inline void copy(cuDoubleComplex a, cuFloatComplex& b)
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
void add_sumsq(
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
    else {
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

} // namespace device
} // namespace slate

#endif // SLATE_DEVICE_UTIL_CUH
