#ifndef SLATE_UNIT_TEST_HH
#define SLATE_UNIT_TEST_HH

#include <exception>
#include <stdexcept>
#include <string>
#include <memory> // unique_ptr

// For type_name<>() to demangle on non-Microsoft platforms.
#ifndef _MSC_VER
    #include <cxxabi.h>
#endif

#include "slate/internal/mpi.hh"

//==============================================================================
/// Exception class thrown by test_assert, test_assert_throw,
/// test_assert_no_throw.
class AssertError: public std::runtime_error {
public:
    AssertError(const char* what_arg, const char* file, int line);

    AssertError(const std::string& what_arg, const char* file, int line);

    std::string file_;
    int line_;
};

//------------------------------------------------------------------------------
/// Throws AssertError if cond is false.
#define test_assert(cond) \
    do { \
        if (! (cond)) { \
            throw AssertError(#cond, __FILE__, __LINE__); \
        } \
    } while(0)

//------------------------------------------------------------------------------
/// Executes expr; throws AssertError if the given exception was not thrown.
#define test_assert_throw(expr, expect) \
    do { \
        try { \
            expr; \
            throw AssertError( \
                "didn't throw exception; expected: " + std::string(#expect), \
                __FILE__, __LINE__); \
        } \
        catch (expect& e) {} \
        catch (AssertError& e) { \
            throw; \
        } \
        catch (std::exception& e) { \
            throw AssertError( \
                "threw wrong exception: " + std::string(e.what()) + \
                "; expected " + std::string(#expect), \
                __FILE__, __LINE__); \
        } \
        catch (...) { \
            throw AssertError( \
                "threw wrong exception; expected: " + std::string(#expect), \
                __FILE__, __LINE__); \
        } \
    } while(0)

//------------------------------------------------------------------------------
/// Executes expr; throws AssertError if std::exception was not thrown.
/// Similar to test_assert_throw(expr, std::exception),
/// but doesn't make two catch (std::exception) handlers.
#define test_assert_throw_std(expr) \
    do { \
        try { \
            expr; \
            throw AssertError( \
                "didn't throw exception; expected: std::exception", \
                __FILE__, __LINE__); \
        } \
        catch (AssertError& e) { \
            throw; \
        } \
        catch (std::exception& e) {} \
        catch (...) { \
            throw AssertError( \
                "threw wrong exception; expected: std::exception", \
                __FILE__, __LINE__); \
        } \
    } while(0)

//------------------------------------------------------------------------------
/// Executes expr; throws AssertError if an exception was thrown.
#define test_assert_no_throw(expr) \
    do { \
        try { \
            expr; \
        } \
        catch (std::exception& e) { \
            throw AssertError( \
                "threw unexpected exception: " + std::string(e.what()), \
                __FILE__, __LINE__); \
        } \
        catch (...) { \
            throw AssertError( \
                "threw unexpected exception: (unknown type)", \
                __FILE__, __LINE__); \
        } \
    } while(0)


//==============================================================================
/// Exception class thrown by test_skip.
class SkipException: public std::runtime_error {
public:
    SkipException(const char* what_arg, const char* file, int line);

    SkipException(const std::string& what_arg, const char* file, int line);

    std::string file_;
    int line_;
};

//------------------------------------------------------------------------------
/// Throws SkipException.
#define test_skip(msg) \
    throw SkipException(msg, __FILE__, __LINE__)

//------------------------------------------------------------------------------
/// type_name<T>() returns string describing the type of T.
///
//  see https://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c
template <typename T>
std::string type_name()
{
    typedef typename std::remove_reference<T>::type TR;

    std::unique_ptr< char, void(*)(void*) > own(
        #ifndef _MSC_VER
            abi::__cxa_demangle(typeid(TR).name(), nullptr, nullptr, nullptr),
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

//------------------------------------------------------------------------------
typedef void test_function(void);

void run_test(test_function* func, const char* name);
void run_test(test_function* func, const char* name, MPI_Comm comm);

int unit_test_main();
int unit_test_main(MPI_Comm comm);

std::string string_printf(const char* format, ...);
void printf_gather(int root, MPI_Comm comm, const std::string& str);
void printf_gather(int root, MPI_Comm comm, const char* format, ...);

/// To be implemented by user; called by unit_test_main().
void run_tests();

#endif // SLATE_UNIT_TEST_HH
