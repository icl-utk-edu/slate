#ifndef UNIT_HH
#define UNIT_HH

#include <exception>
#include <stdexcept>
#include <string>

#include "slate_mpi.hh"

//==============================================================================
/// Exception class thrown by test_assert, test_assert_throw,
/// test_assert_no_throw.
class AssertError: public std::runtime_error
{
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
class SkipException: public std::runtime_error
{
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

#endif // #ifndef UNIT_HH
