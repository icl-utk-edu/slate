#ifndef TEST_HH
#define TEST_HH

#ifdef SLATE_WITH_MPI
    #include <mpi.h>
#else
    #include "slate_NoMpi.hh"
#endif

#ifdef _OPENMP
    #include <omp.h>
#else
    #include "slate_NoOpenmp.hh"
#endif

#include "slate_Matrix.hh"

#include <iostream>
#include <iomanip>

// -----------------------------------------------------------------------------
// global variables
MPI_Comm g_mpi_comm;
int  g_mpi_rank    = -1;
int  g_mpi_size    = -1;
bool g_verbose     = false;
int  g_num_devices = omp_get_num_devices();
int  g_host_num    = omp_get_initial_device();

// -----------------------------------------------------------------------------
// type_name<T>() returns string describing type of T.
// see https://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c

// for demangling on non-Microsoft platforms
#ifndef _MSC_VER
    #include <cxxabi.h>
#endif

template< typename T >
std::string type_name()
{
    typedef typename std::remove_reference<T>::type TR;

    std::unique_ptr< char, void(*)(void*) > own(
        #ifndef _MSC_VER
            abi::__cxa_demangle( typeid(TR).name(), nullptr, nullptr, nullptr ),
        #else
            nullptr,
        #endif
        std::free
    );

    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}

// -----------------------------------------------------------------------------
// Prints test name at start and end of test.
// Since destructors are called in reverse order of constructors,
// putting this first ensures that its destructor is called last when exiting
// the test function.
class Test {
public:
    // ----------------------------------------
    Test( const char* msg ):
        msg_( msg )
    {
        if (g_mpi_rank == 0) {
            std::cout << "%---------- " << msg_ << "\n" << std::flush;
        }
        MPI_Barrier( g_mpi_comm );
    }

    // ----------------------------------------
    ~Test()
    {
        std::cout << std::flush;
        MPI_Barrier( g_mpi_comm );

        if (g_mpi_rank == 0) {
            std::cout << "%---------- " << msg_ << " done\n\n" << std::flush;
        }
        MPI_Barrier( g_mpi_comm );
    }

    const char* msg_;
};

// -----------------------------------------------------------------------------
// Does barrier, then prints label on rank 0 for next sub-test.
void test_message( const char* format, ... )
{
    MPI_Barrier( g_mpi_comm );

    va_list ap;
    va_start( ap, format );
    char buf[ 1024 ];
    vsnprintf( buf, sizeof(buf), format, ap );
    va_end( ap );

    if (g_mpi_rank == 0) {
        std::cout << "%----- " << buf << "\n";
    }
}

// -----------------------------------------------------------------------------
// suppresses compiler warning if var is unused
#define unused( var ) \
    ((void)var)

// -----------------------------------------------------------------------------
// similar to assert(), but also prints out MPI rank.
#define test_assert( cond ) \
    do { \
        if (! (cond)) { \
            std::cerr << "rank " << g_mpi_rank \
                      << ": assertion failed at " \
                      << __FILE__ << ":" << __LINE__ << ": " \
                      << #cond << "\n"; \
            exit(1); \
        } \
    } while(0)

// -----------------------------------------------------------------------------
// executes expr; asserts that the given exception was thrown.
#define test_assert_throw( expr, exception ) \
    do { \
        try { \
            expr; \
            std::cerr << "rank " << g_mpi_rank \
                      << ": assertion failed at " \
                      << __FILE__ << ":" << __LINE__ << ": did not throw expected exception\n"; \
        } \
        catch( exception& e ) {} \
        catch( ... ) { \
            std::cerr << "rank " << g_mpi_rank \
                      << ": assertion failed at " \
                      << __FILE__ << ":" << __LINE__ << ": wrong exception thrown\n"; \
        } \
    } while(0)

// -----------------------------------------------------------------------------
// executes expr; asserts that no exception was thrown.
#define test_assert_no_throw( expr ) \
    do { \
        try { \
            expr; \
        } \
        catch( ... ) { \
            std::cerr << "rank " << g_mpi_rank \
                      << ": assertion failed at " \
                      << __FILE__ << ":" << __LINE__ << ": unexpected exception thrown\n"; \
        } \
    } while(0)

//------------------------------------------------------------------------------
template <typename scalar_t>
void print( slate::Tile< scalar_t >& A )
{
    printf( "[\n" );
    for (int i = 0; i < A.mb(); ++i) {
        for (int64_t j = 0; j < A.nb(); ++j) {
            printf( " %9.4f", A(i, j) );
        }
        printf( "\n" );
    }
    printf( "];  %% op=%c, uplo=%c\n", char(A.op()), char(A.uplo()) );
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void print( slate::Tile< std::complex< scalar_t > >& A )
{
    printf( "[\n" );
    for (int i = 0; i < A.mb(); ++i) {
        for (int64_t j = 0; j < A.nb(); ++j) {
            if (A.op() == blas::Op::ConjTrans) {
                printf( " %9.4f + %9.4fi", real( A(i, j) ), -imag( A(i, j) ) );
            }
            else {
                printf( " %9.4f + %9.4fi", real( A(i, j) ),  imag( A(i, j) ) );
            }
        }
        printf( "\n" );
    }
    printf( "];  %% op=%c, uplo=%c\n", char(A.op()), char(A.uplo()) );
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void print( slate::Matrix< scalar_t >& A )
{
    using blas::real;

    printf( "[\n" );
    // loop over block rows, then rows within block row
    for (int i = 0; i < A.mt(); ++i) {

        // loop over block cols, print out address
        for (int64_t j = 0; j < A.nt(); ++j) {
            int64_t jb = A.tileNb(i);

            if (A.tileIsLocal( i, j )) {
                if (j > 0)
                    printf( "   " );
                auto Aij = A(i, j);
                printf( "  %-18p", (void*) Aij.data() );
                for (int64_t jj = 2; jj < jb; ++jj) { // above pointer is 2 columns
                    printf( " %9s", "" );
                }
            }
            else {
                printf( " ... " );
            }
        }
        printf( "\n" );

        int64_t ib = A.tileMb(i);
        for (int64_t ii = 0; ii < ib; ++ii) {

            // loop over block cols, then cols within block col
            for (int64_t j = 0; j < A.nt(); ++j) {
                int64_t jb = A.tileNb(i);

                if (A.tileIsLocal( i, j )) {
                    if (j > 0)
                        printf( "   " );
                    auto Aij = A(i, j);
                    for (int64_t jj = 0; jj < jb; ++jj) {
                        printf( " %9.4f", real( Aij( ii, jj ) ) );
                    }
                }
                else {
                    printf( " ... " );
                }
            }
            printf( "\n" );
        }
        if (i < A.mt()-1) {
            printf( "\n" );
        }
    }
    printf( "];  %% op=%c\n", char(A.op()) );
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void print( slate::Matrix< std::complex< scalar_t > >& A )
{
    using blas::real;

    printf( "[\n" );
    // loop over block rows, then rows within block row
    for (int i = 0; i < A.mt(); ++i) {

        // loop over block cols, print out address
        for (int64_t j = 0; j < A.nt(); ++j) {
            int64_t jb = A.tileNb(i);

            if (A.tileIsLocal( i, j )) {
                if (j > 0)
                    printf( "   " );
                auto Aij = A(i, j);
                printf( "  %-21p", (void*) Aij.data() );
                for (int64_t jj = 1; jj < jb; ++jj) { // above pointer is 1 column
                    printf( " %22s", "" );
                }
            }
            else {
                printf( " ... " );
            }
        }
        printf( "\n" );

        int64_t ib = A.tileMb(i);
        for (int64_t ii = 0; ii < ib; ++ii) {

            // loop over block cols, then cols within block col
            for (int64_t j = 0; j < A.nt(); ++j) {
                int64_t jb = A.tileNb(i);

                if (A.tileIsLocal( i, j )) {
                    if (j > 0)
                        printf( "   " );
                    auto Aij = A(i, j);
                    for (int64_t jj = 0; jj < jb; ++jj) {
                        printf( " %9.4f + %9.4fi", real( Aij(ii, jj) ), imag( Aij(ii, jj) ) );
                    }
                }
                else {
                    printf( " ... " );
                }
            }
            printf( "\n" );
        }
        if (i < A.mt()-1) {
            printf( "\n" );
        }
    }
    printf( "];  %% op=%c\n", char(A.op()) );
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void print( slate::BaseTrapezoidMatrix< scalar_t >& A )
{
    assert( A.uplo() == blas::Uplo::Lower );

    printf( "[\n" );
    // loop over block rows, then rows within block row
    for (int i = 0; i < A.mt(); ++i) {

        // loop over block cols, print out address
        for (int64_t j = 0; j <= i && j < A.nt(); ++j) {  // lower
            int64_t jb = A.tileNb(i);

            if (A.tileIsLocal( i, j )) {
                if (j > 0)
                    printf( "   " );
                auto Aij = A(i, j);
                printf( "  %-18p", (void*) Aij.data() );
                for (int64_t jj = 2; jj < jb; ++jj) { // above pointer is 2 columns
                    printf( " %9s", "" );
                }
            }
            else {
                printf( " ... " );
            }
        }
        printf( "\n" );

        int64_t ib = A.tileMb(i);
        for (int64_t ii = 0; ii < ib; ++ii) {

            // loop over block cols, then cols within block col
            for (int64_t j = 0; j <= i && j < A.nt(); ++j) {  // lower
                int64_t jb = A.tileNb(i);

                if (A.tileIsLocal( i, j )) {
                    if (j > 0)
                        printf( "   " );
                    auto Aij = A(i, j);
                    for (int64_t jj = 0; jj < jb; ++jj) {
                        printf( " %9.4f", Aij(ii, jj) );
                    }
                }
                else {
                    printf( " ... " );
                }
            }
            printf( "\n" );
        }
        if (i < A.mt()-1) {
            printf( "\n" );
        }
    }
    printf( "];  %% op=%c, uplo=%c\n", char(A.op()), char(A.uplo()) );
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void print( slate::BaseTrapezoidMatrix< std::complex< scalar_t > >& A )
{
    assert( A.uplo() == blas::Uplo::Lower );

    printf( "[\n" );
    // loop over block rows, then rows within block row
    for (int i = 0; i < A.mt(); ++i) {

        // loop over block cols, print out address
        for (int64_t j = 0; j <= i && j < A.nt(); ++j) {  // lower
            int64_t jb = A.tileNb(i);

            if (A.tileIsLocal( i, j )) {
                if (j > 0)
                    printf( "   " );
                auto Aij = A(i, j);
                printf( "  %-21p", (void*) Aij.data() );
                for (int64_t jj = 1; jj < jb; ++jj) { // above pointer is 1 column
                    printf( " %22s", "" );
                }
            }
            else {
                printf( " ... " );
            }
        }
        printf( "\n" );

        int64_t ib = A.tileMb(i);
        for (int64_t ii = 0; ii < ib; ++ii) {

            // loop over block cols, then cols within block col
            for (int64_t j = 0; j <= i && j < A.nt(); ++j) {  // lower
                int64_t jb = A.tileNb(i);

                if (A.tileIsLocal( i, j )) {
                    if (j > 0)
                        printf( "   " );
                    auto Aij = A(i, j);
                    for (int64_t jj = 0; jj < jb; ++jj) {
                        printf( " %9.4f + %9.4fi", real( Aij(ii, jj) ), imag( Aij(ii, jj) ) );
                    }
                }
                else {
                    printf( " ... " );
                }
            }
            printf( "\n" );
        }
        if (i < A.mt()-1) {
            printf( "\n" );
        }
    }
    printf( "];  %% op=%c, uplo=%c\n", char(A.op()), char(A.uplo()) );
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void print( int64_t m, int64_t n, scalar_t* A, int64_t lda )
{
    printf( "[\n" );
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf( " %9.4f", A[ i + j*lda ] );
        }
        printf( "\n" );
    }
    printf( "];\n" );
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void print( int64_t m, int64_t n, std::complex< scalar_t >* A, int64_t lda )
{
    printf( "[\n" );
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf( " %9.4f + %9.4fi", real( A[ i + j*lda ] ), imag( A[ i + j*lda ] ) );
        }
        printf( "\n" );
    }
    printf( "];\n" );
}

#endif        //  #ifndef TEST_HH
