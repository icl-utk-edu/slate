// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/slate.hh"

#include "unit_test.hh"

namespace test {

//------------------------------------------------------------------------------
// global variables
int mpi_rank;
int mpi_size;
int verbose;
int num_devices;

//------------------------------------------------------------------------------
void test_uniform_blocksize()
{
    auto uni_100_16 = slate::func::uniform_blocksize(100, 16);
    for (int i = 0; i < 6; ++i) {
        test_assert(uni_100_16(i) == 16);
    }
    test_assert(uni_100_16(7) == 4);

    auto uni_75_25 = slate::func::uniform_blocksize(75, 25);
    for (int i = 0; i < 3; ++i) {
        test_assert(uni_75_25(i) == 25);
    }
}

//------------------------------------------------------------------------------
void test_grid_2d_block_cyclic()
{

    auto grid_col = slate::func::grid_2d_block_cyclic(
                            slate::Layout::ColMajor, 2, 3, 4, 5);

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {

            int ref_proc = 0;
            // Column major loop
            for (int jj = 0; jj < 5; ++jj) {
                for (int ii = 0; ii < 4; ++ii) {

                    for (int iii = 0; iii < 2; ++iii) {
                        for (int jjj = 0; jjj < 3; ++jjj) {

                            int global_i = iii + 2*(ii + 4*i);
                            int global_j = jjj + 3*(jj + 5*j);
                            test_assert(grid_col({global_i, global_j}) == ref_proc);

                        }
                    }
                    ref_proc++;
                }
            }
        }
    }

    auto grid_row = slate::func::grid_2d_block_cyclic(
                            slate::Layout::RowMajor, 2, 3, 4, 5);

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {

            int ref_proc = 0;
            // Row major loop
            for (int ii = 0; ii < 4; ++ii) {
                for (int jj = 0; jj < 5; ++jj) {

                    for (int iii = 0; iii < 2; ++iii) {
                        for (int jjj = 0; jjj < 3; ++jjj) {

                            int global_i = iii + 2*(ii + 4*i);
                            int global_j = jjj + 3*(jj + 5*j);
                            test_assert(grid_row({global_i, global_j}) == ref_proc);

                        }
                    }
                    ref_proc++;
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
void test_grid_2d_cyclic()
{
    auto grid_col = slate::func::grid_2d_cyclic(
                            slate::Layout::ColMajor, 4, 5);

    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 20; ++j) {

            int ref_proc = 0;
            // Column major loop
            for (int jj = 0; jj < 5; ++jj) {
                for (int ii = 0; ii < 4; ++ii) {

                    int global_i = ii + 4*i;
                    int global_j = jj + 5*j;
                    test_assert(grid_col({global_i, global_j}) == ref_proc);

                    ref_proc++;
                }
            }
        }
    }

    auto grid_row = slate::func::grid_2d_cyclic(
                            slate::Layout::RowMajor, 4, 5);

    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 20; ++j) {

            int ref_proc = 0;
            // Row major loop
            for (int ii = 0; ii < 4; ++ii) {
                for (int jj = 0; jj < 5; ++jj) {

                    int global_i = ii + 4*i;
                    int global_j = jj + 5*j;
                    test_assert(grid_row({global_i, global_j}) == ref_proc);

                    ref_proc++;
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
void test_grid_1d_block_cyclic()
{
    auto grid_col = slate::func::grid_1d_block_cyclic(
                            slate::Layout::ColMajor, 2, 4);

    for (int i = 0; i < 10; ++i) {
        int ref_proc = 0;
        for (int ii = 0; ii < 4; ++ii) {
            for (int iii = 0; iii < 2; ++iii) {
                int global_i = iii + 2*(ii + 4*i);

                for (int j = 0; j < 200; ++j) {

                    test_assert(grid_col({global_i, j}) == ref_proc);

                }
            }
            ref_proc++;
        } // ii loop
    }

    auto grid_row = slate::func::grid_1d_block_cyclic(
                            slate::Layout::RowMajor, 3, 5);

    for (int j = 0; j < 10; ++j) {
        int ref_proc = 0;
        for (int jj = 0; jj < 5; ++jj) {
            for (int jjj = 0; jjj < 3; ++jjj) {
                int global_j = jjj + 3*(jj + 5*j);

                for (int i = 0; i < 200; ++i) {
                    test_assert(grid_row({i, global_j}) == ref_proc);
                }
            }
            ref_proc++;
        } // jj loop
    }
}

//------------------------------------------------------------------------------
void test_grid_1d_cyclic()
{
    auto grid_col = slate::func::grid_1d_cyclic(
                            slate::Layout::ColMajor, 4);

    for (int i = 0; i < 20; ++i) {
        int ref_proc = 0;
        for (int ii = 0; ii < 4; ++ii) {
            int global_i = ii + 4*i;

            for (int j = 0; j < 200; ++j) {

                test_assert(grid_col({global_i, j}) == ref_proc);

            }
            ref_proc++;
        } // ii loop
    }

    auto grid_row = slate::func::grid_1d_cyclic(
                            slate::Layout::RowMajor, 5);

    for (int j = 0; j < 20; ++j) {
        int ref_proc = 0;
        for (int jj = 0; jj < 5; ++jj) {
            int global_j = jj + 5*j;

            for (int i = 0; i < 200; ++i) {
                test_assert(grid_row({i, global_j}) == ref_proc);
            }
            ref_proc++;
        } // jj loop
    }
}

//------------------------------------------------------------------------------
void test_grid_transpose()
{
    std::function<int( std::tuple<int64_t, int64_t> )>
    base = []( std::tuple<int64_t, int64_t> ij ) {
        int64_t i = std::get<0>( ij );
        int64_t j = std::get<1>( ij );
        return int(i + j*1000);
    };
    auto transposed = slate::func::grid_transpose(base);

    for (int i = 0; i < 500; ++i) {
        for (int j = 0; j < 500; ++j) {
            test_assert(transposed({i, j}) == j + i*1000);
        }
    }
}

//------------------------------------------------------------------------------
/// Runs all tests. Called by unit test main().
void run_tests()
{
    run_test(test_uniform_blocksize,    "test_uniform_blocksize");
    run_test(test_grid_2d_block_cyclic, "test_grid_2d_block_cyclic");
    run_test(test_grid_2d_cyclic,       "test_grid_2d_cyclic");
    run_test(test_grid_1d_block_cyclic, "test_grid_1d_block_cyclic");
    run_test(test_grid_1d_cyclic,       "test_grid_1d_cyclic");
    run_test(test_grid_transpose,       "test_grid_transpose");
}

}  // namespace test

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    return unit_test_main( );  // which calls run_tests()
}
