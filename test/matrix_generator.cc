#include "slate/slate.hh"
#include "test.hh"
#include "print_matrix.hh"

#include <exception>
#include <string>
#include <vector>
#include <limits>
#include <complex>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include "matrix_params.hh"
#include "matrix_generator.hh"

// -----------------------------------------------------------------------------
// ANSI color codes
using testsweeper::ansi_esc;
using testsweeper::ansi_red;
using testsweeper::ansi_bold;
using testsweeper::ansi_normal;

// -----------------------------------------------------------------------------
/// Splits a string by any of the delimiters.
/// Adjacent delimiters will give empty tokens.
/// See https://stackoverflow.com/questions/53849
/// @ingroup util
std::vector< std::string >
    split( const std::string& str, const std::string& delims );

std::vector< std::string >
    split( const std::string& str, const std::string& delims )
{
    size_t npos = std::string::npos;
    std::vector< std::string > tokens;
    size_t start = (str.size() > 0 ? 0 : npos);
    while (start != npos) {
        size_t end = str.find_first_of( delims, start );
        tokens.push_back( str.substr( start, end - start ));
        start = (end == npos ? npos : end + 1);
    }
    return tokens;
}

// =============================================================================
namespace slate {

// -----------------------------------------------------------------------------
/// Generates sigma vector of singular or eigenvalues, according to distribution.
///
/// Internal function, called from generate_matrix().
///
/// @ingroup generate_matrix
template< typename scalar_t >
void generate_sigma(
    MatrixParams& params,
    TestMatrixDist dist, bool rand_sign,
    blas::real_type<scalar_t> cond,
    blas::real_type<scalar_t> sigma_max,
    slate::Matrix<scalar_t>& A,
    Vector< blas::real_type<scalar_t> >& sigma )
{
    using real_t = blas::real_type<scalar_t>;


    // locals
    int64_t minmn = std::min( A.m(), A.n() );
    assert( minmn == sigma.n );

    switch (dist) {
        case TestMatrixDist::arith:
            for (int64_t i = 0; i < minmn; ++i) {
                sigma[i] = 1 - i / real_t(minmn - 1) * (1 - 1/cond);
            }
            break;

        case TestMatrixDist::rarith:
            for (int64_t i = 0; i < minmn; ++i) {
                sigma[i] = 1 - (minmn - 1 - i) / real_t(minmn - 1) * (1 - 1/cond);
            }
            break;

        case TestMatrixDist::geo:
            for (int64_t i = 0; i < minmn; ++i) {
                sigma[i] = pow( cond, -i / real_t(minmn - 1) );
            }
            break;

        case TestMatrixDist::rgeo:
            for (int64_t i = 0; i < minmn; ++i) {
                sigma[i] = pow( cond, -(minmn - 1 - i) / real_t(minmn - 1) );
            }
            break;

        case TestMatrixDist::cluster0:
            sigma[0] = 1;
            for (int64_t i = 1; i < minmn; ++i) {
                sigma[i] = 1/cond;
            }
            break;

        case TestMatrixDist::rcluster0:
            for (int64_t i = 0; i < minmn-1; ++i) {
                sigma[i] = 1/cond;
            }
            sigma[minmn-1] = 1;
            break;

        case TestMatrixDist::cluster1:
            for (int64_t i = 0; i < minmn-1; ++i) {
                sigma[i] = 1;
            }
            sigma[minmn-1] = 1/cond;
            break;

        case TestMatrixDist::rcluster1:
            sigma[0] = 1/cond;
            for (int64_t i = 1; i < minmn; ++i) {
                sigma[i] = 1;
            }
            break;

        case TestMatrixDist::logrand: {
            real_t range = log( 1/cond );
            lapack::larnv( idist_rand, params.iseed, sigma.n, sigma(0) );
            for (int64_t i = 0; i < minmn; ++i) {
                sigma[i] = exp( sigma[i] * range );
            }
            // make cond exact
            if (minmn >= 2) {
                sigma[0] = 1;
                sigma[1] = 1/cond;
            }
            break;
        }

        case TestMatrixDist::randn:
        case TestMatrixDist::rands:
        case TestMatrixDist::rand: {
            int64_t idist = (int64_t) dist;
            lapack::larnv( idist, params.iseed, sigma.n, sigma(0) );
            break;
        }

        case TestMatrixDist::specified:
            // user-specified sigma values; don't modify
            sigma_max = 1;
            rand_sign = false;
            break;

        case TestMatrixDist::none:
            assert( false );
            break;
    }

    if (sigma_max != 1) {
        blas::scal( sigma.n, sigma_max, sigma(0), 1 );
    }

    if (rand_sign) {
        // apply random signs
        for (int64_t i = 0; i < minmn; ++i) {
            if (rand() > RAND_MAX/2) {
                sigma[i] = -sigma[i];
            }
        }
    }

    // copy sigma => A
    scalar_t zero = 0.0;
    const int64_t min_mt_nt = std::min(A.mt(), A.nt());
    set(zero, zero, A);
    int64_t S_index = 0;
    #pragma omp parallel for
    for (int64_t i = 0; i < min_mt_nt; ++i) {
        if (A.tileIsLocal(i, i)) {
            auto T = A(i, i);
            for (int ii = 0; ii < A.tileNb(i); ++ii) {
                T.at(ii, ii) = sigma[S_index + ii];
            }
        }
        S_index += A.tileNb(i);
    }
}


// -----------------------------------------------------------------------------
/// Given matrix A with singular values such that sum(sigma_i^2) = n,
/// returns A with columns of unit norm, with the same condition number.
/// see: Davies and Higham, 2000, Numerically stable generation of correlation
/// matrices and their factors.
///
/// Internal function, called from generate_matrix().
///
/// @ingroup generate_matrix
//template< typename scalar_t >
//void generate_correlation_factor( slate::Matrix<scalar_t>& A )
//{
//    //const scalar_t eps = std::numeric_limits<scalar_t>::epsilon();
//
//    Vector<scalar_t> x( A.n );
//    for (int64_t j = 0; j < A.n; ++j) {
//        x[j] = blas::dot( A.m, A(0,j), 1, A(0,j), 1 );
//    }
//
//    for (int64_t i = 0; i < A.n; ++i) {
//        for (int64_t j = 0; j < A.n; ++j) {
//            if ((x[i] < 1 && 1 < x[j]) || (x[i] > 1 && 1 > x[j])) {
//                scalar_t xij, d, t, c, s;
//                xij = blas::dot( A.m, A(0,i), 1, A(0,j), 1 );
//                d = sqrt( xij*xij - (x[i] - 1)*(x[j] - 1) );
//                t = (xij + std::copysign( d, xij )) / (x[j] - 1);
//                c = 1 / sqrt(1 + t*t);
//                s = c*t;
//                blas::rot( A.m, A(0,i), 1, A(0,j), 1, c, -s );
//                x[i] = blas::dot( A.m, A(0,i), 1, A(0,i), 1 );
//                //if (x[i] - 1 > 30*eps) {
//                //    printf( "i %d, x[i] %.6f, x[i] - 1 %.6e, 30*eps %.6e\n",
//                //            i, x[i], x[i] - 1, 30*eps );
//                //}
//                //assert( x[i] - 1 < 30*eps );
//                x[i] = 1;
//                x[j] = blas::dot( A.m, A(0,j), 1, A(0,j), 1 );
//                break;
//            }
//        }
//    }
//}
//
//
// -----------------------------------------------------------------------------
// specialization to complex
// can't use Higham's algorithm in complex
//template<>
//void generate_correlation_factor( slate::Matrix<std::complex<float>>& A )
//{
//    assert( false );
//}
//
//template<>
//void generate_correlation_factor( slate::Matrix<std::complex<double>>& A )
//{
//    assert( false );
//}


// -----------------------------------------------------------------------------
/// Generates matrix using SVD, $A = U Sigma V^H$.
///
/// Internal function, called from generate_matrix().
///
/// @ingroup generate_matrix
template< typename scalar_t >
void generate_svd(
    MatrixParams& params,
    TestMatrixDist dist,
    blas::real_type<scalar_t> cond,
    blas::real_type<scalar_t> condD,
    blas::real_type<scalar_t> sigma_max,
    slate::Matrix<scalar_t>& A,
    Vector< blas::real_type<scalar_t> >& sigma )
{
    using real_t = blas::real_type<scalar_t>;
    assert( A.m() >= A.n() );

    // locals
    const int64_t n = A.n();
    const int64_t mt = A.mt();
    const int64_t nt = A.nt();
    const int64_t min_mt_nt = std::min(mt, nt);

    slate::Matrix<scalar_t> U = A.emptyLike();
    U.insertLocalTiles();
    slate::TriangularFactors<scalar_t> T;

    // ----------
    generate_sigma( params, dist, false, cond, sigma_max, A, sigma );

    // for generate correlation factor, need sum sigma_i^2 = n
    // scaling doesn't change cond
    if (condD != 1) {
        real_t sum_sq = blas::dot( sigma.n, sigma(0), 1, sigma(0), 1 );
        real_t scale = sqrt( sigma.n / sum_sq );
        blas::scal( sigma.n, scale, sigma(0), 1 );

        // copy sigma to diag(A)
        int64_t S_index = 0;
        #pragma omp parallel for
        for (int64_t i = 0; i < min_mt_nt; ++i) {
            if (A.tileIsLocal(i, i)) {
                auto T = A(i, i);
                for (int ii = 0; ii < A.tileNb(i); ++ii) {
                    T.at(ii, ii) = sigma[S_index + ii];
                }
            }
            S_index += A.tileNb(i);
            //A.tileTick(i, i);
        }
    }

    // random U, m-by-minmn
    auto Tmp = U.emptyLike();
    #pragma omp parallel for collapse(2)
    for (int64_t j = 0; j < nt; ++j) {
        for (int64_t i = 0; i < mt; ++i) {
            if (U.tileIsLocal(i, j)) {
                // lapack assume input tile is contigous in memory
                // if the tiles are not contigous in memory, then
                // insert temperory tiles to be passed to lapack
                // then copy output tile to U.tile
                Tmp.tileInsert(i, j);
                auto Tmpij = Tmp(i, j);
                scalar_t* data = Tmpij.data();
                int64_t ldt = Tmpij.stride();
                params.iseed[0] = (rand() + i + j) % 256;
                //lapack::larnv( idist_randn, params.iseed,
                //    U.tileMb(i)*U.tileNb(j), Tmpij.data() );
                for (int64_t k = 0; k < Tmpij.nb(); ++k) {
                    lapack::larnv(idist_randn, params.iseed, Tmpij.mb(), &data[k*ldt]);
                }
                gecopy(Tmp(i, j), U(i, j));
                Tmp.tileErase(i, j);
            }
        }
    }
    // we need to make each random column into a Householder vector;
    // no need to update subsequent columns (as in geqrf).
    // However, currently we do geqrf here,
    // since we don’t have a way to make Householder vectors (no distributed larfg).
    slate::geqrf(U, T);

    // A = U*A
    slate::unmqr( slate::Side::Left, slate::Op::NoTrans, U, T, A);

    // random V, n-by-minmn (stored column-wise in U)
    auto V = U.sub(0, nt-1, 0, nt-1);
    #pragma omp parallel for collapse(2)
    for (int64_t j = 0; j < nt; ++j) {
        for (int64_t i = 0; i < mt; ++i) {
            if (V.tileIsLocal(i, j)) {
                Tmp.tileInsert(i, j);
                auto Tmpij = Tmp(i, j);
                scalar_t* data = Tmpij.data();
                int64_t ldt = Tmpij.stride();
                for (int64_t k = 0; k < Tmpij.nb(); ++k) {
                    lapack::larnv(idist_randn, params.iseed, Tmpij.mb(), &data[k*ldt]);
                }
                gecopy(Tmp(i, j), V(i, j));
                Tmp.tileErase(i, j);
            }
        }
    }
    slate::geqrf(V, T);

    // A = A*V^H
    slate::unmqr( slate::Side::Right, slate::Op::ConjTrans, V, T, A);

    if (condD != 1) {
        // A = A*W, W orthogonal, such that A has unit column norms
        // i.e., A'*A is a correlation matrix with unit diagonal
        // TODO: uncomment generate_correlation_factor
        //generate_correlation_factor( A );

        // A = A*D col scaling
        Vector<real_t> D( n );
        real_t range = log( condD );
        lapack::larnv( idist_rand, params.iseed, D.n, D(0) );
        for (int64_t i = 0; i < D.n; ++i) {
            D[i] = exp( D[i] * range );
        }
        // TODO: add argument to return D to caller?
        if (params.verbose) {
            printf( "D = [" );
            for (int64_t i = 0; i < D.n; ++i) {
                printf( " %11.8g", D[i] );
            }
            printf( " ];\n" );
        }

        int64_t J_index = 0;
        for (int64_t j = 0; j < nt; ++j) {
            for (int64_t i = 0; i < mt; ++i) {
                if (A.tileIsLocal(i, j)) {
                    auto T = A(i, j);
                    for (int jj = 0; jj < A.tileNb(j); ++jj) {
                        for (int ii = 0; ii < A.tileMb(i); ++ii) {
                            T.at(ii, jj) *= D[J_index + jj];
                        }
                    }
                }
            }
            J_index += A.tileNb(j);
        }
    }
}

// -----------------------------------------------------------------------------
/// Generates matrix using Hermitian eigenvalue decomposition, $A = U Sigma U^H$.
///
/// Internal function, called from generate_matrix().
///
/// @ingroup generate_matrix
template< typename scalar_t >
void generate_heev(
    MatrixParams& params,
    TestMatrixDist dist, bool rand_sign,
    blas::real_type<scalar_t> cond,
    blas::real_type<scalar_t> condD,
    blas::real_type<scalar_t> sigma_max,
    slate::Matrix<scalar_t>& A,
    Vector< blas::real_type<scalar_t> >& sigma )
{
    using real_t = blas::real_type<scalar_t>;

    // check inputs
    assert( A.m() == A.n() );

    // locals
    int64_t n = A.n();
    slate::Matrix<scalar_t> U = A.emptyLike();
    U.insertLocalTiles();
    slate::TriangularFactors<scalar_t> T;

    // ----------
    generate_sigma( params, dist, rand_sign, cond, sigma_max, A, sigma );

    // random U, m-by-minmn
    const int64_t nt = U.nt();
    const int64_t mt = U.mt();
    auto Tmp = U.emptyLike();

    #pragma omp parallel for collapse(2)
    for (int64_t j = 0; j < nt; ++j) {
        for (int64_t i = 0; i < mt; ++i) {
            if (U.tileIsLocal(i, j)) {
                Tmp.tileInsert(i, j);
                auto Tmpij = Tmp(i, j);
                scalar_t* data = Tmpij.data();
                int64_t ldt = Tmpij.stride();
                params.iseed[0] = (rand() + i + j) % 256;
                for (int64_t k = 0; k < Tmpij.nb(); ++k) {
                    lapack::larnv(idist_rand, params.iseed, Tmpij.mb(), &data[k*ldt]);
                }
                gecopy(Tmp(i, j), U(i, j));
                Tmp.tileErase(i, j);
            }
        }
    }
    // we need to make each random column into a Householder vector;
    // no need to update subsequent columns (as in geqrf).
    // However, currently we do geqrf here,
    // since we don’t have a way to make Householder vectors (no distributed larfg).
    slate::geqrf(U, T);

    // A = U*A
    slate::unmqr( slate::Side::Left, slate::Op::NoTrans, U, T, A);

    // A = A*U^H
    slate::unmqr( slate::Side::Right, slate::Op::ConjTrans, U, T, A);

    // make diagonal real
    // usually LAPACK ignores imaginary part anyway, but Matlab doesn't
    #pragma omp parallel for
    for (int64_t i = 0; i < nt; ++i) {
        if (A.tileIsLocal(i, i)) {
            auto T = A(i, i);
            for (int ii = 0; ii < A.tileMb(i); ++ii) {
                T.at(ii, ii) = std::real( T.at(ii, ii) );
            }
        }
    }

    if (condD != 1) {
        // A = D*A*D row & column scaling
        Vector<real_t> D( n );
        real_t range = log( condD );
        lapack::larnv( idist_rand, params.iseed, n, D(0) );
        for (int64_t i = 0; i < n; ++i) {
            D[i] = exp( D[i] * range );
        }

        int64_t J_index = 0;
        #pragma omp parallel for
        for (int64_t j = 0; j < nt; ++j) {
            int64_t I_index = 0;
            for (int64_t i = 0; i < mt; ++i) {
                if (A.tileIsLocal(i, j)) {
                    auto T = A(i, j);
                    for (int jj = 0; jj < A.tileMb(j); ++jj) {
                        for (int ii = 0; ii < A.tileMb(i); ++ii) {
                            T.at(ii, jj) *= D[I_index + ii] * D[J_index + jj];
                        }
                    }
                }
                I_index += A.tileNb(i);
            }
            J_index += A.tileMb(j);
        }
    }
}

// -----------------------------------------------------------------------------
/// Generates matrix using general eigenvalue decomposition, $A = V T V^H$,
/// with orthogonal eigenvectors.
/// Not yet implemented.
///
/// Internal function, called from generate_matrix().
///
/// @ingroup generate_matrix
template< typename scalar_t >
void generate_geev(
    MatrixParams& params,
    TestMatrixDist dist,
    blas::real_type<scalar_t> cond,
    blas::real_type<scalar_t> sigma_max,
    slate::Matrix<scalar_t>& A,
    Vector< blas::real_type<scalar_t> >& sigma )
{
    throw std::exception();  // not implemented
}

// -----------------------------------------------------------------------------
/// Generates matrix using general eigenvalue decomposition, $A = X T X^{-1}$,
/// with random eigenvectors.
/// Not yet implemented.
///
/// Internal function, called from generate_matrix().
///
/// @ingroup generate_matrix
template< typename scalar_t >
void generate_geevx(
    MatrixParams& params,
    TestMatrixDist dist,
    blas::real_type<scalar_t> cond,
    blas::real_type<scalar_t> sigma_max,
    slate::Matrix<scalar_t>& A,
    Vector< blas::real_type<scalar_t> >& sigma )
{
    throw std::exception();  // not implemented
}

// -----------------------------------------------------------------------------
void generate_matrix_usage()
{
    printf(
    "The --matrix, --cond, and --condD parameters specify a test matrix.\n"
    "See Test routines: generate_matrix in the HTML documentation for a\n"
    "complete description.\n"
    "\n"
    "%s--matrix%s is one of following:\n"
    "\n"
    "%sMatrix%s    |  %sDescription%s\n"
    "----------|-------------\n"
    "zero      |  all zero\n"
    "identity  |  ones on diagonal, rest zero\n"
    "jordan    |  ones on diagonal and first subdiagonal, rest zero\n"
    "          |  \n"
    "rand@     |  matrix entries random uniform on (0, 1)\n"
    "rands@    |  matrix entries random uniform on (-1, 1)\n"
    "randn@    |  matrix entries random normal with mean 0, std 1\n"
    "          |  \n"
    "diag^@    |  A = Sigma\n"
    "svd^@     |  A = U Sigma V^H\n"
    "poev^@    |  A = V Sigma V^H  (eigenvalues positive, i.e., matrix SPD)\n"
    "spd^@     |  alias for poev\n"
    "heev^@    |  A = V Lambda V^H (eigenvalues mixed signs)\n"
    "syev^@    |  alias for heev\n"
    "geev^@    |  A = V T V^H, Schur-form T                       [not yet implemented]\n"
    "geevx^@   |  A = X T X^{-1}, Schur-form T, X ill-conditioned [not yet implemented]\n"
    "\n"
    "^ and @ denote optional suffixes described below.\n"
    "\n"
    "%s^ Distribution%s  |  %sDescription%s\n"
    "----------------|-------------\n"
    "_logrand        |  log(sigma_i) random uniform on [ log(1/cond), log(1) ]; default\n"
    "_arith          |  sigma_i = 1 - frac{i - 1}{n - 1} (1 - 1/cond); arithmetic: sigma_{i+1} - sigma_i is constant\n"
    "_geo            |  sigma_i = (cond)^{ -(i-1)/(n-1) };             geometric:  sigma_{i+1} / sigma_i is constant\n"
    "_cluster0       |  sigma = [ 1, 1/cond, ..., 1/cond ];  1  unit value,  n-1 small values\n"
    "_cluster1       |  sigma = [ 1, ..., 1, 1/cond ];      n-1 unit values,  1  small value\n"
    "_rarith         |  _arith,    reversed order\n"
    "_rgeo           |  _geo,      reversed order\n"
    "_rcluster0      |  _cluster0,  reversed order\n"
    "_rcluster1      |  _cluster1, reversed order\n"
    "_specified      |  user specified sigma on input\n"
    "                |  \n"
    "_rand           |  sigma_i random uniform on (0, 1)\n"
    "_rands          |  sigma_i random uniform on (-1, 1)\n"
    "_randn          |  sigma_i random normal with mean 0, std 1\n"
    "\n"
    "%s@ Scaling%s       |  %sDescription%s\n"
    "----------------|-------------\n"
    "_ufl            |  scale near underflow         = 1e-308 for double\n"
    "_ofl            |  scale near overflow          = 2e+308 for double\n"
    "_small          |  scale near sqrt( underflow ) = 1e-154 for double\n"
    "_large          |  scale near sqrt( overflow  ) = 6e+153 for double\n"
    "\n"
    "%s@ Modifier%s      |  %sDescription%s\n"
    "----------------|-------------\n"
    "_dominant       |  make matrix diagonally dominant\n"
    "\n",
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal,
        ansi_bold, ansi_normal
    );
}

// -----------------------------------------------------------------------------
/// Generates an m-by-n test matrix.
/// Similar to LAPACK's libtmg functionality, but a level 3 BLAS implementation.
///
/// @param[in] params
///     Test matrix parameters. Uses matrix, cond, condD parameters;
///     see further details.
///
/// @param[out] A
///     Complex array, dimension (lda, n).
///     On output, the m-by-n test matrix A in an lda-by-n array.
///
/// @param[in,out] sigma
///     Real array, dimension (min(m,n))
///     - On input with matrix distribution "_specified",
///       contains user-specified singular or eigenvalues.
///     - On output, contains singular or eigenvalues, if known,
///       else set to NaN. sigma is not necesarily sorted.
///
/// ### Further Details
///
/// The **matrix** parameter specifies the matrix kind according to the
/// tables below. As indicated, kinds take an optional distribution suffix (^)
/// and an optional scaling and modifier suffix (@).
/// The default distribution is logrand.
/// Examples: rand, rand_small, svd_arith, heev_geo_small.
///
/// The **cond** parameter specifies the condition number $cond(S)$, where $S$ is either
/// the singular values $\Sigma$ or the eigenvalues $\Lambda$, as described by the
/// distributions below. It does not apply to some matrices and distributions.
/// For geev and geevx, cond(A) is generally much worse than cond(S).
/// If _dominant is applied, cond(A) generally improves.
/// By default, cond(S) = sqrt( 1/eps ) = 6.7e7 for double, 2.9e3 for single.
///
/// The **condD** parameter specifies the condition number cond(D), where D is
/// a diagonal scaling matrix [1]. By default, condD = 1. If condD != 1, then:
/// - For matrix = svd, set $A = A_0 K D$, where $A_0 = U \Sigma V^H$,
///   $D$ has log-random entries in $[ \log(1/condD), \log(1) ]$, and
///   $K$ is diagonal such that columns of $B = A_0 K$ have unit norm,
///   hence $B^T B$ has unit diagonal.
///
/// - For matrix = heev, set $A = D A_0 D$, where $A_0 = U \Lambda U^H$,
///   $D$ has log-random entries in $[ \log(1/condD), \log(1) ]$.
///   TODO: set $A = D K A_0 K D$ where
///   $K$ is diagonal such that $B = K A_0 K$ has unit diagonal.
///
/// Note using condD changes the singular or eigenvalues of $A$;
/// on output, sigma contains the singular or eigenvalues of $A_0$, not of $A$.
///
/// Notation used below:
/// $\Sigma$ is a diagonal matrix with entries $\sigma_i$ for $i = 1, \dots, n$;
/// $\Lambda$ is a diagonal matrix with entries $\lambda_i = \pm \sigma_i$,
/// with random sign;
/// $U$ and $V$ are random orthogonal matrices from the Haar distribution [2],
/// $X$ is a random matrix.
///
/// See LAPACK Working Note (LAWN) 41:\n
/// Table  5 (Test matrices for the nonsymmetric eigenvalue problem)\n
/// Table 10 (Test matrices for the symmetric eigenvalue problem)\n
/// Table 11 (Test matrices for the singular value decomposition)
///
/// Matrix   | Description
/// ---------|------------
/// zero     | all zero
/// identity | ones on diagonal, rest zero
/// jordan   | ones on diagonal and first subdiagonal, rest zero
/// --       | --
/// rand@    | matrix entries random uniform on (0, 1)
/// rands@   | matrix entries random uniform on (-1, 1)
/// randn@   | matrix entries random normal with mean 0, std 1
/// --       | --
/// diag^@   | $A = \Sigma$
/// svd^@    | $A = U \Sigma V^H$
/// poev^@   | $A = V \Sigma V^H$  (eigenvalues positive, i.e., matrix SPD)
/// spd^@    | alias for poev
/// heev^@   | $A = V \Lambda V^H$ (eigenvalues mixed signs)
/// syev^@   | alias for heev
/// geev^@   | $A = V T V^H$, Schur-form $T$                         [not yet implemented]
/// geevx^@  | $A = X T X^{-1}$, Schur-form $T$, $X$ ill-conditioned [not yet implemented]
///
/// Note for geev that $cond(\Lambda)$ is specified, where $\Lambda = diag(T)$;
/// while $cond(T)$ and $cond(A)$ are usually much worse.
///
/// ^ and @ denote optional suffixes described below.
///
/// ^ Distribution  |   Description
/// ----------------|--------------
/// _logrand        |  $\log(\sigma_i)$ random uniform on $[ \log(1/cond), \log(1) ]$; default
/// _arith          |  $\sigma_i = 1 - \frac{i - 1}{n - 1} (1 - 1/cond)$; arithmetic: $\sigma_{i+1} - \sigma_i$ is constant
/// _geo            |  $\sigma_i = (cond)^{ -(i-1)/(n-1) }$;              geometric:  $\sigma_{i+1} / \sigma_i$ is constant
/// _cluster0       |  $\Sigma = [ 1, 1/cond, ..., 1/cond ]$;  1     unit value,  $n-1$ small values
/// _cluster1       |  $\Sigma = [ 1, ..., 1, 1/cond ]$;       $n-1$ unit values, 1     small value
/// _rarith         |  _arith,    reversed order
/// _rgeo           |  _geo,      reversed order
/// _rcluster0      |  _cluster0,  reversed order
/// _rcluster1      |  _cluster1, reversed order
/// _specified      |  user specified sigma on input
/// --              |  --
/// _rand           |  $\sigma_i$ random uniform on (0, 1)
/// _rands          |  $\sigma_i$ random uniform on (-1, 1)
/// _randn          |  $\sigma_i$ random normal with mean 0, std 1
///
/// Note _rand, _rands, _randn do not use cond; the condition number is random.
///
/// Note for _rands and _randn, $\Sigma$ contains negative values.
/// This means poev_rands and poev_randn will not generate an SPD matrix.
///
/// @ Scaling       |  Description
/// ----------------|-------------
/// _ufl            |  scale near underflow         = 1e-308 for double
/// _ofl            |  scale near overflow          = 2e+308 for double
/// _small          |  scale near sqrt( underflow ) = 1e-154 for double
/// _large          |  scale near sqrt( overflow  ) = 6e+153 for double
///
/// Note scaling changes the singular or eigenvalues, but not the condition number.
///
/// @ Modifier      |  Description
/// ----------------|-------------
/// _dominant       |  diagonally dominant: set $A_{i,i} = \pm \max_i( \sum_j |A_{i,j}|, \sum_j |A_{j,i}| )$.
///
/// Note _dominant changes the singular or eigenvalues, and the condition number.
///
/// ### References
///
/// [1] Demmel and Veselic, Jacobi's method is more accurate than QR, 1992.
///
/// [2] Stewart, The efficient generation of random orthogonal matrices
///     with an application to condition estimators, 1980.
///
/// @ingroup generate_matrix
template< typename scalar_t >
void generate_matrix(
    MatrixParams& params,
    slate::Matrix<scalar_t>& A,
    Vector< blas::real_type<scalar_t> >& sigma )
{
    using real_t = blas::real_type<scalar_t>;

    // constants
    const real_t nan = std::numeric_limits<real_t>::quiet_NaN();
    const real_t d_zero = 0;
    const real_t d_one  = 1;
    const real_t ufl = std::numeric_limits< real_t >::min();      // == lamch("safe min")  ==  1e-38 or  2e-308
    const real_t ofl = 1 / ufl;                                   //                            8e37 or   4e307
    const real_t eps = std::numeric_limits< real_t >::epsilon();  // == lamch("precision") == 1.2e-7 or 2.2e-16
    const scalar_t c_zero = 0;
    const scalar_t c_one  = 1;

    // locals
    std::string kind = params.kind();
    std::vector< std::string > tokens = split( kind, "-_" );

    real_t cond = params.cond();
    bool cond_default = std::isnan( cond );
    if (cond_default) {
        cond = 1 / sqrt( eps );
    }

    real_t condD = params.condD();
    bool condD_default = std::isnan( condD );
    if (condD_default) {
        condD = 1;
    }

    real_t sigma_max = 1;

    // ----------
    // set sigma to unknown (nan)
    lapack::laset( lapack::MatrixType::General, sigma.n, 1,
        nan, nan, sigma(0), sigma.n );

    // ----- decode matrix type
    auto token = tokens.begin();
    if (token == tokens.end()) {
        throw std::runtime_error( "Error: empty matrix kind\n" );
    }
    std::string base = *token;
    ++token;
    TestMatrixType type = TestMatrixType::identity;
    if      (base == "zero"    ) { type = TestMatrixType::zero;     }
    else if (base == "identity") { type = TestMatrixType::identity; }
    else if (base == "jordan"  ) { type = TestMatrixType::jordan;   }
    else if (base == "randn"   ) { type = TestMatrixType::randn;    }
    else if (base == "rands"   ) { type = TestMatrixType::rands;    }
    else if (base == "rand"    ) { type = TestMatrixType::rand;     }
    else if (base == "diag"    ) { type = TestMatrixType::diag;     }
    else if (base == "svd"     ) { type = TestMatrixType::svd;      }
    else if (base == "poev" ||
             base == "spd"     ) { type = TestMatrixType::poev;     }
    else if (base == "heev" ||
             base == "syev"    ) { type = TestMatrixType::heev;     }
    else if (base == "geevx"   ) { type = TestMatrixType::geevx;    }
    else if (base == "geev"    ) { type = TestMatrixType::geev;     }
    else {
        fprintf( stderr, "%sUnrecognized matrix '%s'%s\n",
                 ansi_red, kind.c_str(), ansi_normal );
        throw std::exception();
    }

    // ----- decode distribution
    std::string suffix;
    TestMatrixDist dist = TestMatrixDist::none;
    if (token != tokens.end()) {
        suffix = *token;
        if      (suffix == "randn"    ) { dist = TestMatrixDist::randn;     }
        else if (suffix == "rands"    ) { dist = TestMatrixDist::rands;     }
        else if (suffix == "rand"     ) { dist = TestMatrixDist::rand;      }
        else if (suffix == "logrand"  ) { dist = TestMatrixDist::logrand;   }
        else if (suffix == "arith"    ) { dist = TestMatrixDist::arith;     }
        else if (suffix == "geo"      ) { dist = TestMatrixDist::geo;       }
        else if (suffix == "cluster1" ) { dist = TestMatrixDist::cluster1;  }
        else if (suffix == "cluster0" ) { dist = TestMatrixDist::cluster0;  }
        else if (suffix == "rarith"   ) { dist = TestMatrixDist::rarith;    }
        else if (suffix == "rgeo"     ) { dist = TestMatrixDist::rgeo;      }
        else if (suffix == "rcluster1") { dist = TestMatrixDist::rcluster1; }
        else if (suffix == "rcluster0") { dist = TestMatrixDist::rcluster0; }
        else if (suffix == "specified") { dist = TestMatrixDist::specified; }

        // if found, move to next token
        if (dist != TestMatrixDist::none) {
            ++token;

            // error if matrix type doesn't support it
            if (! (type == TestMatrixType::diag ||
                   type == TestMatrixType::svd  ||
                   type == TestMatrixType::poev ||
                   type == TestMatrixType::heev ||
                   type == TestMatrixType::geev ||
                   type == TestMatrixType::geevx))
            {
                fprintf( stderr, "%sError in '%s': matrix '%s' doesn't support"
                         " distribution suffix.%s\n",
                         ansi_red, kind.c_str(), base.c_str(), ansi_normal );
                throw std::exception();
            }
        }
    }
    if (dist == TestMatrixDist::none)
        dist = TestMatrixDist::logrand;  // default

    // ----- decode scaling
    sigma_max = 1;
    if (token != tokens.end()) {
        suffix = *token;
        if      (suffix == "small") { sigma_max = sqrt( ufl ); }
        else if (suffix == "large") { sigma_max = sqrt( ofl ); }
        else if (suffix == "ufl"  ) { sigma_max = ufl; }
        else if (suffix == "ofl"  ) { sigma_max = ofl; }

        // if found, move to next token
        if (sigma_max != 1) {
            ++token;

            // error if matrix type doesn't support it
            if (! (type == TestMatrixType::rand  ||
                   type == TestMatrixType::rands ||
                   type == TestMatrixType::randn ||
                   type == TestMatrixType::svd   ||
                   type == TestMatrixType::poev  ||
                   type == TestMatrixType::heev  ||
                   type == TestMatrixType::geev  ||
                   type == TestMatrixType::geevx))
            {
                fprintf( stderr, "%sError in '%s': matrix '%s' doesn't support"
                         " scaling suffix.%s\n",
                         ansi_red, kind.c_str(), base.c_str(), ansi_normal );
                throw std::exception();
            }
        }
    }

    // ----- decode modifier
    bool dominant = false;
    if (token != tokens.end()) {
        suffix = *token;
        if (suffix == "dominant") {
            dominant = true;

            // move to next token
            ++token;

            // error if matrix type doesn't support it
            if (! (type == TestMatrixType::rand  ||
                   type == TestMatrixType::rands ||
                   type == TestMatrixType::randn ||
                   type == TestMatrixType::svd   ||
                   type == TestMatrixType::poev  ||
                   type == TestMatrixType::heev  ||
                   type == TestMatrixType::geev  ||
                   type == TestMatrixType::geevx))
            {
                fprintf( stderr, "%sError in '%s': matrix '%s' doesn't support"
                         " modifier suffix.%s\n",
                         ansi_red, kind.c_str(), base.c_str(), ansi_normal );
                throw std::exception();
            }
        }
    }

    if (token != tokens.end()) {
        fprintf( stderr, "%sError in '%s': unknown suffix '%s'.%s\n",
                 ansi_red, kind.c_str(), token->c_str(), ansi_normal );
        throw std::exception();
    }

    // ----- check compatability of options
    if (A.m() != A.n() &&
        (type == TestMatrixType::jordan ||
         type == TestMatrixType::poev   ||
         type == TestMatrixType::heev   ||
         type == TestMatrixType::geev   ||
         type == TestMatrixType::geevx))
    {
        fprintf( stderr, "%sError: matrix '%s' requires m == n.%s\n",
                 ansi_red, kind.c_str(), ansi_normal );
        throw std::exception();
    }

    if (type == TestMatrixType::zero      ||
        type == TestMatrixType::identity  ||
        type == TestMatrixType::jordan    ||
        type == TestMatrixType::randn     ||
        type == TestMatrixType::rands     ||
        type == TestMatrixType::rand)
    {
        // warn first time if user set cond and matrix doesn't use it
        static std::string last;
        if (! cond_default && last != kind) {
            last = kind;
            fprintf( stderr, "%sWarning: matrix '%s' ignores cond %.2e.%s\n",
                     ansi_red, kind.c_str(), params.cond(), ansi_normal );
        }
        params.cond_used() = testsweeper::no_data_flag;
    }
    else if (dist == TestMatrixDist::randn ||
             dist == TestMatrixDist::rands ||
             dist == TestMatrixDist::rand)
    {
        // warn first time if user set cond and distribution doesn't use it
        static std::string last;
        if (! cond_default && last != kind) {
            last = kind;
            fprintf( stderr, "%sWarning: matrix '%s': rand, randn, and rands "
                     "singular/eigenvalue distributions ignore cond %.2e.%s\n",
                     ansi_red, kind.c_str(), params.cond(), ansi_normal );
        }
        params.cond_used() = testsweeper::no_data_flag;
    }
    else {
        params.cond_used() = cond;
    }

    if (! (type == TestMatrixType::svd ||
           type == TestMatrixType::heev ||
           type == TestMatrixType::poev))
    {
        // warn first time if user set condD and matrix doesn't use it
        static std::string last;
        if (! condD_default && last != kind) {
            last = kind;
            fprintf( stderr, "%sWarning: matrix '%s' ignores condD %.2e.%s\n",
                     ansi_red, kind.c_str(), params.condD(), ansi_normal );
        }
    }

    if (type == TestMatrixType::poev &&
        (dist == TestMatrixDist::rands ||
         dist == TestMatrixDist::randn))
    {
        fprintf( stderr, "%sWarning: matrix '%s' using rands or randn "
                 "will not generate SPD matrix; use rand instead.%s\n",
                 ansi_red, kind.c_str(), ansi_normal );
    }

    int n = (int)A.n();
    const int64_t nt = A.nt();
    const int64_t mt = A.mt();
    // ----- generate matrix
    switch (type) {
        case TestMatrixType::zero:
            set(c_zero, c_zero, A);
            lapack::laset( lapack::MatrixType::General, sigma.n, 1,
                d_zero, d_zero, sigma(0), sigma.n );
            break;

        case TestMatrixType::identity:
            set(c_zero, c_one, A);
            lapack::laset( lapack::MatrixType::General, sigma.n, 1,
                d_one, d_one, sigma(0), sigma.n );
            break;

        case TestMatrixType::jordan: {
            set(c_zero, c_one, A ); // ones on diagonal
            // ones on sub-diagonal
            for (int64_t i = 0; i < nt; ++i) {
                // Set 1 element from sub-diagonal tile to 1.
                if (i > 0) {
                    if (A.tileIsLocal(i, i-1)) {
                        auto T = A(i, i-1);
                        T.at(0, T.nb()-1) = 1.;
                        A.tileTick(i, i-1);
                    }
                }

                // Set 1 element from sub-diagonal tile to 1.
                if (A.tileIsLocal(i, i)) {
                    auto T = A(i, i);
                    auto len = T.nb();
                    for (int j = 0; j < len-1; ++j) {
                        T.at(j+1, j) = 1.;
                    }
                    A.tileTick(i, i);
                }
            }
            break;
        }

        case TestMatrixType::rand:
        case TestMatrixType::rands:
        case TestMatrixType::randn: {
            //int64_t idist = (int64_t) type;
            int64_t idist = 1;
            auto Tmp = A.emptyLike();
            #pragma omp parallel for collapse(2)
            for (int64_t j = 0; j < nt; ++j) {
                for (int64_t i = 0; i < mt; ++i) {
                    if (A.tileIsLocal(i, j)) {
                        Tmp.tileInsert(i, j);
                        auto Tmpij = Tmp(i, j);
                        scalar_t* data = Tmpij.data();
                        int64_t ldt = Tmpij.stride();
                        params.iseed[0] = (rand() + i + j) % 256;
                        for (int64_t k = 0; k < Tmpij.nb(); ++k) {
                            lapack::larnv(idist, params.iseed, Tmpij.mb(), &data[k*ldt]);
                        }

                        // Make it diagonally dominant
                        if (dominant) {
                            if (i == j) {
                                //auto T = A(i, i);
                                for (int ii = 0; ii < A.tileNb(i); ++ii) {
                                    Tmpij.at(ii, ii) += n;
                                }
                            }
                        }
                        gecopy(Tmp(i, j), A(i, j));
                        Tmp.tileErase(i, j);

                        // Scale the matrix
                        if (sigma_max != 1) {
                            scalar_t s = sigma_max;
                            scale(s, A(i, j));
                        }
                    }
                }
            }
            break;
        }

        case TestMatrixType::diag:
            generate_sigma( params, dist, false, cond, sigma_max, A, sigma );
            break;

        case TestMatrixType::svd:
            generate_svd( params, dist, cond, condD, sigma_max, A, sigma );
            break;

        case TestMatrixType::poev:
            generate_heev( params, dist, false, cond, condD, sigma_max, A, sigma );
            break;

        case TestMatrixType::heev:
            generate_heev( params, dist, true, cond, condD, sigma_max, A, sigma );
            break;

        case TestMatrixType::geev:
            generate_geev( params, dist, cond, sigma_max, A, sigma );
            break;

        case TestMatrixType::geevx:
            generate_geevx( params, dist, cond, sigma_max, A, sigma );
            break;
    }

    if (! (type == TestMatrixType::rand  ||
           type == TestMatrixType::rands ||
           type == TestMatrixType::randn) && dominant) {
           // make diagonally dominant; strict unless diagonal has zeros
           slate_error("Not implemented yet");
           throw std::exception();  // not implemented
    }
}


// -----------------------------------------------------------------------------
/// Generates an m-by-n test matrix.
/// Traditional interface with m, n, lda instead of Matrix object.
///
/// @see generate_matrix()
///
/// @ingroup generate_matrix
template< typename scalar_t >
void generate_matrix(
    MatrixParams& params,
    slate::Matrix<scalar_t>& A,
    blas::real_type<scalar_t>* sigma_ptr )
{
    using real_t = blas::real_type<scalar_t>;

    int64_t m = A.m();
    int64_t n = A.n();
    // vector & matrix wrappers
    // if sigma is null, create new vector; data is discarded later
    Vector<real_t> sigma( sigma_ptr, std::min( m, n ) );
    if (sigma_ptr == nullptr) {
        sigma = Vector<real_t>( std::min( m, n ) );
    }
    generate_matrix( params, A, sigma );
}


// -----------------------------------------------------------------------------
// explicit instantiations
template
void generate_matrix(
    MatrixParams& params,
    slate::Matrix<float>& A,
    float* sigma_ptr );

template
void generate_matrix(
    MatrixParams& params,
    slate::Matrix<double>& A,
    double* sigma_ptr );

template
void generate_matrix(
    MatrixParams& params,
    slate::Matrix< std::complex<float> >& A,
    float* sigma_ptr );

template
void generate_matrix(
    MatrixParams& params,
    slate::Matrix< std::complex<double> >& A,
    double* sigma_ptr );

} // namespace slate
