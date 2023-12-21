
#include "random.hh"

#include <array>
#include <complex>
#include "blas.hh"
#include "blas/util.hh"


namespace slate {
namespace random {

//------------------------------------------------------------------------------
/// Compute 128-bit product of 64-bit x 64-bit multiplication
inline std::array<uint64_t, 2> full_mul(uint64_t val1, uint64_t val2)
{
    #ifdef __SIZEOF_INT128__
        // Some compilers (e.g., GCC) support uint128_t
        // Such compilers can often compile it to 1-2 instructions
        auto product = __uint128_t(val1) * __uint128_t(val2);
        uint64_t product_lo = product;
        uint64_t product_hi = product >> 64;
    #else
        // Slower, portable implementation
        constexpr uint64_t mask_32 = (uint64_t(1)<<32)-1;
        uint64_t hi1 = (val1>>32) & mask_32;
        uint64_t lo1 = (val1    ) & mask_32;
        uint64_t hi2 = (val2>>32) & mask_32;
        uint64_t lo2 = (val2    ) & mask_32;

        uint64_t product_mi = hi1*lo2 + lo1*hi2;

        uint64_t product_hi = hi1*hi2 + (product_mi >> 32);
        uint64_t product_lo = val1*val2;
    #endif
    return {product_lo, product_hi};
}

//------------------------------------------------------------------------------
/// Generates 128 pseudorandom bits using the Philox-2x64 generator
/// Based on Salmon et al. "Parallel Random Numbers: As Easy as 1, 2, 3", 2011
///
/// @param[in] state
///     The counter indicating which entry of the sequence should be generated.
///
/// @param[in] seed
///     The seed indicating which pseudorandom sequence to generate.
///
/// @return 128 pseudorandom bits.bits.
inline std::array<uint64_t, 2> philox_2x64(std::array<uint64_t, 2> state, uint64_t seed)
{
    // Constants chosen emperically in the Salmon paper
    constexpr uint64_t seed_inc = UINT64_C(0xD2B74407B1CE6E93);
    constexpr uint64_t multiplier = UINT64_C(0x9E3779B97F4A7C15);
    constexpr int rounds = 10;

    for (int i = 0; i < rounds; ++i) {
        if (i != 0) {
            // bump seed
            seed += seed_inc;
        }

        // Philox S-Box
        uint64_t L = state[0];
        uint64_t R = state[1];

        auto product = full_mul(R, multiplier);

        state = {product[0], product[1]^seed^L};
    }
    return state;
}


//------------------------------------------------------------------------------
/// Helper function to make a real number in [0, 1) from psuedorandom bits.
template<typename real_t>
real_t rand_to_real(uint64_t bits)
{
    static_assert(std::numeric_limits<real_t>::is_iec559, "Assumes IEEE floats");

    constexpr int digits = std::numeric_limits<real_t>::digits;

    return real_t(bits>>(64-digits)) / real_t(UINT64_C(1)<<digits);
}

//------------------------------------------------------------------------------
/// Generates a single float for a specific distribution, seed, and (i,j) index.
template<typename scalar_t, Dist dist>
scalar_t generate_float(int64_t seed, int64_t i, int64_t j)
{
    using real_t = blas::real_type<scalar_t>;

    // C++20 has std::numbers::pi_v<real_t>
    constexpr real_t pi = 3.1415926535897932385;

    const auto bits = philox_2x64( {uint64_t(i), uint64_t(j)}, seed );
    // generate two floats in the range [0, 1)
    const auto raw_float1 = rand_to_real<real_t>(bits[0]);
    const auto raw_float2 = rand_to_real<real_t>(bits[1]);

    // unlike larnv, uniform generation is [0, 1)
    // This is a) easier b) more common in other libraries (e.g., C++, Java, Numpy)

    // Compiler won't compute imaginary part in the real case
    real_t re = 0, im = 0;
    switch (dist) {
        // first 5 are _larnv distributions
        case Dist::Uniform:
            re = raw_float1;
            im = raw_float2;
            break;
        case Dist::UniformSigned:
            re = 2*raw_float1-1;
            im = 2*raw_float2-1;
            break;
        case Dist::Normal:{
            // Box-Muller, per LAPACK
            // 1-raw_float1 switches from [0, 1) to (0, 1] to prevent log(0)
            real_t mag = std::sqrt(-2*std::log(1-raw_float1));
            real_t arg = 2 * pi * raw_float2;
            re = mag * std::cos(arg);
            im = mag * std::sin(arg);
            break;
        }
        case Dist::UnitDisk:{
            // per LAPACK
            real_t mag = std::sqrt(raw_float1);
            real_t arg = 2 * pi * raw_float2;
            re = mag * std::cos(arg);
            im = mag * std::sin(arg);
            break;
        }
        case Dist::UnitCircle:{
            // per LAPACK
            real_t arg = 2 * pi * raw_float2;
            re = std::cos(arg);
            im = std::sin(arg);
            break;
        }
        // Binary and Radamacher distributions
        case Dist::Binary:
            re = raw_float1 >= 0.5 ? 1.0 : 0.0;
            im = raw_float2 >= 0.5 ? 1.0 : 0.0;
            break;
        case Dist::BinarySigned:
            re = raw_float1 >= 0.5 ? 1.0 : -1.0;
            im = raw_float2 >= 0.5 ? 1.0 : -1.0;
            break;
        // default to nothing
        default:
            break;
    }

    return blas::make_scalar<scalar_t>(re, im);
}

//------------------------------------------------------------------------------
/// Helper function to convert the distribution to a template parameter.
template<Dist dist, typename scalar_t>
void generate_helper(int64_t seed,
                          int64_t m, int64_t n, int64_t ioffset, int64_t joffset,
                          scalar_t* A, int64_t lda)
{
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            A[i + j*lda] = generate_float<scalar_t, dist>( seed, i+ioffset, j+joffset );
        }
    }
}

//------------------------------------------------------------------------------
/// Generates a sub-matrix with random entries.
///
/// @param[in] dist
///     The distribution to take elements from.
///
/// @param[in] seed
///     The value to seed the random number generator.
///
/// @param[in] m
///     The number of rows.
///
/// @param[in] n
///     The number of columns.
///
/// @param[in] ioffset
///     The first row of the sub-matrix in the global matrix.
///
/// @param[in] joffset
///     The first column of the sub-matrix in the global matrix.
///
/// @param[out]A
///     The m-by-n, column-major sub-matrix to fill with random values.
///
/// @param[in]lda
///     The leading dimension of the matrix.
template<typename scalar_t>
void generate(Dist dist, int64_t seed,
              int64_t m, int64_t n, int64_t ioffset, int64_t joffset,
              scalar_t* A, int64_t lda)
{
    // dispatch on dist, to avoid large switch within inner loop
    if (dist == Dist::Uniform) {
        generate_helper<Dist::Uniform>( seed, m, n, ioffset, joffset, A, lda );
    }
    else if (dist == Dist::UniformSigned) {
        generate_helper<Dist::UniformSigned>( seed, m, n, ioffset, joffset, A, lda );
    }
    else if (dist == Dist::Normal) {
        generate_helper<Dist::Normal>( seed, m, n, ioffset, joffset, A, lda );
    }
    else if (dist == Dist::UnitDisk) {
        generate_helper<Dist::UnitDisk>( seed, m, n, ioffset, joffset, A, lda );
    }
    else if (dist == Dist::UnitCircle) {
        generate_helper<Dist::UnitCircle>( seed, m, n, ioffset, joffset, A, lda );
    }
    else if (dist == Dist::Binary) {
        generate_helper<Dist::Binary>( seed, m, n, ioffset, joffset, A, lda );
    }
    else if (dist == Dist::BinarySigned) {
        generate_helper<Dist::BinarySigned>( seed, m, n, ioffset, joffset, A, lda );
    }
    else {
        throw std::invalid_argument( "Unknown distribution" );
    }
}


template
void generate(
    Dist dist,
    int64_t seed,
    int64_t m,
    int64_t n,
    int64_t ioffset,
    int64_t joffset,
    float* A,
    int64_t lda);
template
void generate(
    Dist dist,
    int64_t seed,
    int64_t m,
    int64_t n,
    int64_t ioffset,
    int64_t joffset,
    double* A,
    int64_t lda);
template
void generate(
    Dist dist,
    int64_t seed,
    int64_t m,
    int64_t n,
    int64_t ioffset,
    int64_t joffset,
    std::complex<float>* A,
    int64_t lda);
template
void generate(
    Dist dist,
    int64_t seed,
    int64_t m,
    int64_t n,
    int64_t ioffset,
    int64_t joffset,
    std::complex<double>* A,
    int64_t lda);

} // namespace random
} // namespace slate
