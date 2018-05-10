#include "unit_test.hh"

#include <exception>

#include <stdio.h>
#include <assert.h>

#include "slate_mpi.hh"

static int g_total = 0;
static int g_pass  = 0;
static int g_fail  = 0;
static int g_skip  = 0;

//------------------------------------------------------------------------------
// ANSI color codes
const char *ansi_esc     = "\x1b[";
const char *ansi_red     = "\x1b[31m";
const char *ansi_green   = "\x1b[92m";
const char *ansi_blue    = "\x1b[34m";
const char *ansi_cyan    = "\x1b[36m";
const char *ansi_magenta = "\x1b[35m";
const char *ansi_yellow  = "\x1b[33m";
const char *ansi_white   = "\x1b[37m";
const char *ansi_gray    = "\x1b[90m";  // "bright black"
const char *ansi_bold    = "\x1b[1m";
const char *ansi_normal  = "\x1b[0m";

//------------------------------------------------------------------------------
/// Returns a string that is sprintf formatted.
std::string string_printf(const char* format, ...)
{
    char buf[ 1024 ];
    va_list va;
    va_start(va, format);
    vsnprintf(buf, sizeof(buf), format, va);
    return std::string(buf);
}

//==============================================================================
AssertError::AssertError(
    const char* what_arg, const char* file, int line )
    : runtime_error(string_printf("%s at %s:%d",
                                  what_arg, file, line)),
      file_(file),
      line_(line)
{}

AssertError::AssertError(
    const std::string& what_arg, const char* file, int line )
    : runtime_error(string_printf("%s at %s:%d",
                                  what_arg.c_str(), file, line)),
      file_(file),
      line_(line)
{}

//==============================================================================
SkipException::SkipException(
    const char* what_arg, const char* file, int line )
    : runtime_error(what_arg),
      file_(file),
      line_(line)
{}

SkipException::SkipException(
    const std::string& what_arg, const char* file, int line )
    : runtime_error(what_arg),
      file_(file),
      line_(line)
{}

//------------------------------------------------------------------------------
/// Returns string for test label.
std::string output_test(const char* str)
{
    return string_printf("%-60s", str);
}

std::string output_test(const char* str, int rank)
{
    std::string tmp = string_printf("%s, rank %d", str, rank);
    return string_printf("%-60s", tmp.c_str());
}

//------------------------------------------------------------------------------
/// Returns string for "pass".
std::string output_pass()
{
    return std::string(ansi_blue) + "pass" + ansi_normal + "\n";
}

//------------------------------------------------------------------------------
/// Returns string for "failed" and error message.
std::string output_fail(AssertError& e)
{
    return std::string(ansi_bold) + ansi_red + "FAILED:" + ansi_normal
            + "\n\t" + ansi_gray + e.what() + ansi_normal + "\n";
}

//------------------------------------------------------------------------------
/// Returns string for "skipped" and message.
std::string output_skip(SkipException& e)
{
    return std::string(ansi_magenta) + "skipped: " + ansi_normal
            + ansi_gray + e.what() + ansi_normal + "\n";
}

//------------------------------------------------------------------------------
/// Root node prints string str from all MPI ranks.
void print_gather(const std::string& str, int root, MPI_Comm comm)
{
#ifdef SLATE_WITH_MPI
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

    if (mpi_rank == root) {
        char buf[ 1024 ];
        MPI_Status status;
        int bufsize;
        for (int rank = 0; rank < mpi_size; ++rank) {
            if (rank != root) {
                MPI_Recv(&bufsize, 1,  MPI_INT,  rank, 0, comm, &status);
                // todo: guard against buffer overflow
                assert(bufsize < int(sizeof(buf)));
                MPI_Recv(buf, bufsize, MPI_CHAR, rank, 0, comm, &status);
                printf(buf);
            }
            else {
                printf(str.c_str());
            }
        }
    }
    else {
        int bufsize = str.size() + 1;
        MPI_Send(&bufsize,    1,       MPI_INT,  0, 0, comm);
        MPI_Send(str.c_str(), bufsize, MPI_CHAR, 0, 0, comm);
    }
#endif
}

//------------------------------------------------------------------------------
/// Runs a single test. Prints label and either pass, failed, or skipped.
/// Catches and reports all exceptions.
void run_test(test_function* func, const char* name)
{
    printf(output_test(name).c_str());
    ++g_total;

    try {
        // run function
        func();
        printf(output_pass().c_str());
        ++g_pass;
    }
    catch (SkipException& e) {
        printf(output_skip(e).c_str());
        --g_total;
        ++g_skip;
    }
    catch (AssertError& e) {
        printf(output_fail(e).c_str());
        ++g_fail;
    }
    catch (std::exception& e) {
        AssertError err("unexpected exception: " + std::string(e.what()),
                        __FILE__, __LINE__);
        printf(output_fail(err).c_str());
        ++g_fail;
    }
    catch (...) {
        AssertError err("unexpected exception: (unknown type)",
                        __FILE__, __LINE__);
        printf(output_fail(err).c_str());
        ++g_fail;
    }
}

//------------------------------------------------------------------------------
/// Runs a single test using MPI. All ranks in comm must participate.
/// Rank 0 prints output from all ranks.
void run_test(test_function* func, const char* name, MPI_Comm comm)
{
#ifdef SLATE_WITH_MPI
    int mpi_rank;
    MPI_Comm_rank(comm, &mpi_rank);

    std::string output = output_test(name, mpi_rank);
    ++g_total;

    try {
        func();
        output += output_pass();
        ++g_pass;
    }
    catch (SkipException& e) {
        output += output_skip(e);
        --g_total;
        ++g_skip;
    }
    catch (AssertError& e) {
        output += output_fail(e);
        ++g_fail;
    }
    catch (std::exception& e) {
        AssertError err("unexpected exception: " + std::string(e.what()),
                        __FILE__, __LINE__);
        output += output_fail(err);
        ++g_fail;
    }
    catch (...) {
        AssertError err("unexpected exception: (unknown type)",
                        __FILE__, __LINE__);
        output += output_fail(err);
        ++g_fail;
    }
    print_gather(output, 0, comm);
#else
    run_test(func, name);
#endif // not SLATE_WITH_MPI
}

//------------------------------------------------------------------------------
/// Runs all tests and print summary of pass, failed, and skipped.
/// @retval  0 if all passed (or were skipped).
/// @retval -1 if any failed.
int unit_test_main()
{
    run_tests();

    if (g_pass == g_total) {
        printf("\n%spassed:  all tests (%d of %d)%s\n",
               ansi_blue, g_pass, g_total, ansi_normal);
        if (g_skip > 0) {
            printf("%sskipped: %d tests%s\n",
                   ansi_magenta, g_skip, ansi_normal);
        }
        return 0;
    }
    else {
        printf("\n%spassed:  %3d of %3d tests%s\n"
               "%s%sfailed:  %3d of %3d tests%s\n",
               ansi_blue, g_pass, g_total, ansi_normal,
               ansi_bold, ansi_red, g_fail, g_total, ansi_normal);
        if (g_skip > 0) {
            printf("%sskipped: %3d tests%s\n",
                   ansi_magenta, g_skip, ansi_normal);
        }
        return -1;
    }
}

//------------------------------------------------------------------------------
/// Runs all tests and prints summary of pass, failed, and skipped.
/// Rank 0 collects results and does printing. All ranks return same value.
/// @retval  0 if all passed (or were skipped).
/// @retval -1 if any failed.
int unit_test_main(MPI_Comm comm)
{
#ifdef SLATE_WITH_MPI
    run_tests();

    int mpi_rank;
    MPI_Comm_rank(comm, &mpi_rank);

    int sum_total, sum_pass, sum_fail, sum_skip;
    MPI_Allreduce(&g_total, &sum_total, 1, MPI_INT, MPI_SUM, comm);
    MPI_Allreduce(&g_pass,  &sum_pass,  1, MPI_INT, MPI_SUM, comm);
    MPI_Allreduce(&g_fail,  &sum_fail,  1, MPI_INT, MPI_SUM, comm);
    MPI_Allreduce(&g_skip,  &sum_skip,  1, MPI_INT, MPI_SUM, comm);

    if (sum_pass == sum_total) {
        if (mpi_rank == 0) {
            printf("\n%spassed all tests (%d of %d)%s\n",
                   ansi_blue, sum_pass, sum_total, ansi_normal);
            if (sum_skip > 0) {
                printf("%sskipped %d tests%s\n",
                       ansi_magenta, sum_skip, ansi_normal);
            }
        }
        return 0;
    }
    else {
        if (mpi_rank == 0) {
            printf("\n%spassed:  %3d of %3d tests%s\n"
                   "%s%sfailed:  %3d of %3d tests%s\n",
                   ansi_blue, sum_pass, sum_total, ansi_normal,
                   ansi_bold, ansi_red, sum_fail, sum_total, ansi_normal);
            if (sum_skip > 0) {
                printf("%sskipped: %3d tests%s\n",
                       ansi_magenta, sum_skip, ansi_normal);
            }
        }
        return -1;
    }
#else
    return unit_test_main();
#endif  // not SLATE_WITH_MPI
}
