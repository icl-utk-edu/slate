// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "unit_test.hh"

#include <exception>

#include <stdio.h>
#include <assert.h>
#include <stdarg.h>

#include "slate/internal/mpi.hh"
#include "testsweeper.hh"

int g_total = 0;
int g_pass  = 0;
int g_fail  = 0;
int g_skip  = 0;

//------------------------------------------------------------------------------
// ANSI color codes
using testsweeper::ansi_esc;
using testsweeper::ansi_red;
using testsweeper::ansi_green;
using testsweeper::ansi_blue;
using testsweeper::ansi_cyan;
using testsweeper::ansi_magenta;
using testsweeper::ansi_yellow;
using testsweeper::ansi_white;
using testsweeper::ansi_gray;
using testsweeper::ansi_bold;
using testsweeper::ansi_normal;

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

std::string output_test(const std::string str)
{
    return output_test( str.c_str() );
}

std::string output_test(const char* str, int rank)
{
    std::string tmp = string_printf("rank %2d, %s", rank, str);
    return string_printf("%-60s", tmp.c_str());
}

std::string output_test(const std::string str, int rank)
{
    std::string tmp = string_printf( "rank %2d, %s", rank, str.c_str() );
    return string_printf( "%-60s", tmp.c_str() );
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
void printf_gather(int root, MPI_Comm comm, const std::string& str)
{
#ifndef SLATE_NO_MPI
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
/// Root node prints output from all MPI ranks.
void printf_gather(int root, MPI_Comm comm, const char* format, ...)
{
    // same code in string_printf
    char buf[ 1024 ];
    va_list va;
    va_start(va, format);
    vsnprintf(buf, sizeof(buf), format, va);
    printf_gather(root, comm, std::string(buf));
}

//------------------------------------------------------------------------------
/// Runs a single test. Prints label and either pass, failed, or skipped.
/// Catches and reports all exceptions.
void run_test(test_function* func, const char* name)
{
    printf(output_test(name).c_str());
    fflush(stdout);
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
#ifndef SLATE_NO_MPI
    int mpi_rank, mpi_size;
    MPI_Comm_rank(comm, &mpi_rank);
    MPI_Comm_size(comm, &mpi_size);

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
    printf_gather(0, comm, output);
    MPI_Barrier(comm);
    if (mpi_size > 1 && mpi_rank == 0)
        printf( "\n" );
#else
    run_test(func, name);
#endif // SLATE_NO_MPI
}

//------------------------------------------------------------------------------
/// Runs all tests and print summary of pass, failed, and skipped.
/// @retval  0 if all passed (or were skipped).
/// @retval -1 if any failed.
int unit_test_main()
{
    test::run_tests();

    if (g_pass == g_total) {
        printf("\n%spassed all tests (%d of %d)%s\n",
               ansi_blue, g_pass, g_total, ansi_normal);
        if (g_skip > 0) {
            printf("%sskipped %d tests%s\n",
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
#ifndef SLATE_NO_MPI
    test::run_tests();

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
#endif  // SLATE_NO_MPI
}
