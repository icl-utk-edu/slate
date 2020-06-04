#ifndef SLATE_TEST_ERROR_HH
#define SLATE_TEST_ERROR_HH

#include "blas.hh"

#include <vector>

// -----------------------------------------------------------------------------
// returns absolute error, || x - xref ||_2
// TODO: generalize to more arbitrary iterators than std::vector.
// TODO: use LAPACK's lassq algorithm to avoid numerical issues in summing squares.
template< typename T1, typename T2 >
blas::real_type< T1, T2 >
abs_error(std::vector<T1>& x, std::vector<T2>& xref)
{
    using real_t = blas::real_type< T1, T2 >;

    if (x.size() != xref.size()) {
        return std::numeric_limits<T1>::quiet_NaN();
    }
    real_t tmp;
    real_t diff = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        tmp = std::abs(x[i] - xref[i]);
        diff += tmp*tmp;
    }
    diff = sqrt(diff);
    return diff;
}

// -----------------------------------------------------------------------------
// returns relative error, || x - xref ||_2 / || xref ||_2
template< typename T1, typename T2 >
blas::real_type< T1, T2 >
rel_error(std::vector<T1>& x, std::vector<T2>& xref)
{
    using real_t = blas::real_type< T1, T2 >;

    if (x.size() != xref.size()) {
        return std::numeric_limits<T1>::quiet_NaN();
    }
    real_t tmp;
    real_t diff = 0;
    real_t norm = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        tmp = std::abs(x[i] - xref[i]);
        diff += tmp*tmp;

        tmp = std::abs(xref[i]);
        norm += tmp*tmp;
    }
    diff = sqrt(diff);
    norm = sqrt(norm);
    return diff / norm;
}

#endif // SLATE_TEST_ERROR_HH
