#ifndef MATRIX_GENERATOR_HPP
#define MATRIX_GENERATOR_HPP

#include <exception>
#include <complex>
#include <ctype.h>

#include "testsweeper.hh"
#include "blas.hh"
#include "lapack.hh"
#include "slate/slate.hh"

#include "matrix_params.hh"

namespace lapack {

// -----------------------------------------------------------------------------
const int64_t idist_rand  = 1;
const int64_t idist_randu = 2;
const int64_t idist_randn = 3;

enum class TestMatrixType {
    rand      = 1,  // maps to larnv idist
    randu     = 2,  // maps to larnv idist
    randn     = 3,  // maps to larnv idist
    zero,
    identity,
    jordan,
    diag,
    svd,
    poev,
    heev,
    geev,
    geevx,
};

enum class Dist {
    rand      = 1,  // maps to larnv idist
    randu     = 2,  // maps to larnv idist
    randn     = 3,  // maps to larnv idist
    arith,
    geo,
    cluster,
    cluster2,
    rarith,
    rgeo,
    rcluster,
    rcluster2,
    logrand,
    specified,
    none,
};

// -----------------------------------------------------------------------------
/// Simple vector class that can wrap existing memory or allocate its own memory.
//
// Uses copy-and-swap idiom.
// https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
template< typename scalar_t >
class Vector
{
public:
    // constructor allocates new memory (unless n == 0)
    Vector( int64_t in_n=0 ):
        n    ( in_n ),
        data_( n > 0 ? new scalar_t[n] : nullptr ),
        own_ ( true )
    {
        if (n < 0) { throw std::exception(); }
    }

    // constructor wraps existing memory; caller maintains ownership
    Vector( scalar_t* data, int64_t in_n ):
        n    ( in_n ),
        data_( data ),
        own_ ( false )
    {
        if (n < 0) { throw std::exception(); }
    }

    // copy constructor
    Vector( Vector const &other ):
        n    ( other.n ),
        data_( nullptr ),
        own_ ( other.own_ )
    {
        if (other.own_) {
            if (n > 0) {
                data_ = new scalar_t[n];
                std::copy( other.data_, other.data_ + n, data_ );
            }
        }
        else {
            data_ = other.data_;
        }
    }

    // move constructor, using copy & swap idiom
    Vector( Vector&& other )
        : Vector()
    {
        swap( *this, other );
    }

    // assignment operator, using copy & swap idiom
    Vector& operator= (Vector other)
    {
        swap( *this, other );
        return *this;
    }

    // destructor deletes memory if constructor allocated it
    // (i.e., not if wrapping existing memory)
    ~Vector()
    {
        if (own_) {
            delete[] data_;
            data_ = nullptr;
        }
    }

    friend void swap( Vector& first, Vector& second )
    {
        using std::swap;
        swap( first.n,     second.n     );
        swap( first.data_, second.data_ );
        swap( first.own_,  second.own_  );
    }

    // returns pointer to element i, because that's what we normally need to
    // call BLAS / LAPACK, which avoids littering the code with &.
    scalar_t*       operator () ( int64_t i )       { return &data_[ i ]; }
    scalar_t const* operator () ( int64_t i ) const { return &data_[ i ]; }

    // return element i itself, as usual in C/C++.
    // unfortunately, this won't work for matrices.
    scalar_t&       operator [] ( int64_t i )       { return data_[ i ]; }
    scalar_t const& operator [] ( int64_t i ) const { return data_[ i ]; }

    int64_t size() const { return n; }
    bool        own()  const { return own_; }

public:
    int64_t n;

private:
    scalar_t *data_;
    bool own_;
};

// -----------------------------------------------------------------------------
/// Simple matrix class that can wrap existing memory or allocate its own memory.
/*
template< typename scalar_t >
class Matrix
{
public:
    // constructor allocates new memory
    // ld = m by default
    Matrix( int64_t in_m, int64_t in_n, int64_t in_ld=0 ):
        m( in_m ),
        n( in_n ),
        ld( in_ld == 0 ? m : in_ld ),
        data_( ld*n )
    {
        if (m  < 0) { throw std::exception(); }
        if (n  < 0) { throw std::exception(); }
        if (ld < m) { throw std::exception(); }
    }

    // constructor wraps existing memory; caller maintains ownership
    // ld = m by default
    Matrix( scalar_t* data, int64_t in_m, int64_t in_n, int64_t in_ld=0 ):
        m( in_m ),
        n( in_n ),
        ld( in_ld == 0 ? m : in_ld ),
        data_( data, ld*n )
    {
        if (m  < 0) { throw std::exception(); }
        if (n  < 0) { throw std::exception(); }
        if (ld < m) { throw std::exception(); }
    }

    int64_t size() const { return data_.size(); }
    bool        own()  const { return data_.own(); }

    // returns pointer to element (i,j), because that's what we normally need to
    // call BLAS / LAPACK, which avoids littering the code with &.
    scalar_t* operator () ( int i, int j )
        { return &data_[ i + j*ld ]; }

    scalar_t const* operator () ( int i, int j ) const
        { return &data_[ i + j*ld ]; }

public:
    int64_t m, n, ld;

protected:
    Vector<scalar_t> data_;
};
*/

// -----------------------------------------------------------------------------
template< typename scalar_t >
void generate_matrix(
    MatrixParams& params,
    slate::Matrix< scalar_t >& A,
    Vector< blas::real_type<scalar_t> >& sigma );

template< typename scalar_t >
void generate_matrix(
    MatrixParams& params,
    //int64_t m, int64_t n,
    //scalar_t* A, int64_t lda,
    slate::Matrix< scalar_t >& A,
    blas::real_type<scalar_t>* sigma=nullptr );

void generate_matrix_usage();

} // namespace lapack

#endif        // #ifndef MATRIX_GENERATOR_HPP
