// Copyright (c) 2023, University of Tennessee. All rights reserved.
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
void test_process_2d_grid()
{
    auto grid_col = slate::func::process_2d_grid(slate::Layout::ColMajor, 4, 5);

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

    auto grid_row = slate::func::process_2d_grid(slate::Layout::RowMajor, 4, 5);

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
void test_process_1d_grid()
{
    auto grid_col = slate::func::process_1d_grid(slate::Layout::ColMajor, 4);

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

    auto grid_row = slate::func::process_1d_grid(slate::Layout::RowMajor, 5);

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
void test_device_2d_grid()
{
    auto grid_col = slate::func::device_2d_grid(slate::Layout::ColMajor, 2, 3, 4, 5);

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

    auto grid_row = slate::func::device_2d_grid(slate::Layout::RowMajor, 2, 3, 4, 5);

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
void test_device_1d_grid()
{
    auto grid_col = slate::func::device_1d_grid(slate::Layout::ColMajor, 2, 4);

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

    auto grid_row = slate::func::device_1d_grid(slate::Layout::RowMajor, 3, 5);

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
void test_grid_transpose()
{
    std::function<int( std::tuple<int64_t, int64_t> )>
    base = []( std::tuple<int64_t, int64_t> ij ) {
        int64_t i = std::get<0>( ij );
        int64_t j = std::get<1>( ij );
        return int(i + j*1000);
    };
    auto transposed = slate::func::transpose_grid(base);

    for (int i = 0; i < 500; ++i) {
        for (int j = 0; j < 500; ++j) {
            test_assert(transposed({i, j}) == j + i*1000);
        }
    }
}

//------------------------------------------------------------------------------
void test_is_same_map()
{
    // Check that is_same_map doesn't access an empty map
    std::function<int( std::tuple<int64_t, int64_t> )>
    error = []( std::tuple<int64_t, int64_t> ij ) {
        test_assert( false );
        return -1;
    };
    test_assert( slate::func::is_same_map(0, 0, error, error) );
    test_assert( slate::func::is_same_map(1000, 0, error, error) );
    test_assert( slate::func::is_same_map(0, 1000, error, error) );


    auto col_2d = slate::func::grid_2d_block_cyclic(
                            slate::Layout::ColMajor, 2, 3, 4, 5 );
    auto row_2d = slate::func::grid_2d_block_cyclic(
                            slate::Layout::RowMajor, 2, 3, 4, 5 );
    auto col_1d = slate::func::grid_2d_block_cyclic(
                            slate::Layout::ColMajor, 2, 1, 4, 1 );
    auto row_1d = slate::func::grid_2d_block_cyclic(
                            slate::Layout::ColMajor, 1, 3, 1, 5 );
    // equality
    test_assert( slate::func::is_same_map(100, 100, col_2d, col_2d) );
    test_assert( slate::func::is_same_map(100, 150, row_2d, row_2d) );
    test_assert( slate::func::is_same_map(100, 1, col_2d, col_1d) );
    test_assert( slate::func::is_same_map(100, 2, col_2d, col_1d) );
    test_assert( slate::func::is_same_map(100, 3, col_2d, col_1d) );
    test_assert( slate::func::is_same_map(1, 100, row_2d, row_1d) );
    test_assert( slate::func::is_same_map(2, 100, row_2d, row_1d) );

    //inequality
    test_assert( ! slate::func::is_same_map(150, 100, row_2d, col_2d) );
    test_assert( ! slate::func::is_same_map(34, 33, col_2d, row_2d) );
    test_assert( ! slate::func::is_same_map(100, 4, col_2d, col_1d) );
    test_assert( ! slate::func::is_same_map(3, 100, row_2d, row_1d) );
}

//------------------------------------------------------------------------------
void test_is_grid_2d_cyclic()
{
    slate::GridOrder out_o;
    int64_t out_p, out_q;

    // Private helpers to check order, p, and q
    #define test_opq( exp_o, exp_p, exp_q ) \
        do { \
            if ((exp_o) != out_o || (exp_p) != out_p || (exp_q) != out_q) { \
                std::cerr << "Computed grid is incorrect at " \
                          << __FILE__ << ":" << __LINE__ << "\n"; \
                exit(1); \
            } \
        } while(0)
    #define test_pq( exp_p, exp_q ) \
        do { \
            if (slate::GridOrder::Unknown == out_o \
                || (exp_p) != out_p || (exp_q) != out_q) { \
                std::cerr << "Computed grid is incorrect at " \
                          << __FILE__ << ":" << __LINE__ << "\n"; \
                exit(1); \
            } \
        } while(0)

    // Check that is_grid_2d_cyclic doesn't access an empty map
    std::function<int( std::tuple<int64_t, int64_t> )>
    error = []( std::tuple<int64_t, int64_t> ij ) {
        test_assert( false );
        return -1;
    };
    test_assert( slate::func::is_grid_2d_cyclic(0, 0, error) );
    test_assert( slate::func::is_grid_2d_cyclic(0, 0, error, out_o, out_p, out_q) );
    test_pq( 1, 1 );
    test_assert( slate::func::is_grid_2d_cyclic(1000, 0, error) );
    test_assert( slate::func::is_grid_2d_cyclic(1000, 0, error, out_o, out_p, out_q) );
    test_pq( 1, 1 );
    test_assert( slate::func::is_grid_2d_cyclic(0, 1000, error) );
    test_assert( slate::func::is_grid_2d_cyclic(1000, 0, error, out_o, out_p, out_q) );
    test_pq( 1, 1 );

    // Loop over arguments to grid_2d_block_cyclic
    std::vector<std::tuple<slate::Layout, int64_t, int64_t, int64_t, int64_t>>
        configs = {
            {slate::Layout::ColMajor, 2, 3, 4, 5},
            {slate::Layout::RowMajor, 2, 3, 4, 5},
            {slate::Layout::ColMajor, 2, 1, 4, 5},
            {slate::Layout::RowMajor, 2, 1, 4, 5},
            {slate::Layout::ColMajor, 1, 3, 4, 5},
            {slate::Layout::RowMajor, 1, 3, 4, 5},
            {slate::Layout::ColMajor, 2, 3, 4, 1},
            {slate::Layout::RowMajor, 2, 3, 4, 1},
            {slate::Layout::ColMajor, 2, 3, 1, 5},
            {slate::Layout::RowMajor, 2, 3, 1, 5},
            {slate::Layout::ColMajor, 2, 3, 1, 1},
            {slate::Layout::RowMajor, 2, 3, 1, 1},

            {slate::Layout::ColMajor, 1, 1, 4, 5},
            {slate::Layout::RowMajor, 1, 1, 4, 5},
            {slate::Layout::ColMajor, 1, 1, 1, 3},
            {slate::Layout::RowMajor, 1, 1, 1, 3},
            {slate::Layout::ColMajor, 1, 1, 10, 1},
            {slate::Layout::RowMajor, 1, 1, 10, 1},
            {slate::Layout::ColMajor, 1, 1, 1, 1},
            {slate::Layout::RowMajor, 1, 1, 1, 1},
    };
    for (auto config : configs) {
        auto func = std::apply(slate::func::grid_2d_block_cyclic, config );

        bool is_col_layout = std::get<0>(config) == slate::Layout::ColMajor;
        bool is_row_layout = ! is_col_layout;
        int64_t m = std::get<1>(config);
        int64_t n = std::get<2>(config);
        int64_t p = std::get<3>(config);
        int64_t q = std::get<4>(config);

        bool is_col_cyclic = p == 1 || (m == 1 && (is_col_layout || q == 1));
        bool is_row_cyclic = q == 1 || (n == 1 && (is_row_layout || p == 1));
        bool is_2d_cyclic  = (m == 1 || p == 1) && (n == 1 || q == 1);

        //std::cout << (is_col_layout ? "C, " : "R, " )
        //          << m << ", " << n << ", " << p << ", " << q << ": "
        //          << ", " << is_col_cyclic << ", " << is_row_cyclic << ", "
        //          << is_2d_cyclic << std::endl;


        test_assert( slate::func::is_grid_2d_cyclic( 1,  1, func) );
        test_assert( slate::func::is_grid_2d_cyclic( 1,  1, func, out_o, out_p, out_q) );
        test_pq( 1, 1 );

        test_assert( slate::func::is_grid_2d_cyclic( 40,  1, func )
                     == is_col_cyclic );
        test_assert( slate::func::is_grid_2d_cyclic( 40,  1, func, out_o, out_p, out_q )
                     == is_col_cyclic );
        if (is_col_cyclic) {
            test_pq( p, 1 );
        }
        else {
            test_opq( slate::GridOrder::Unknown, -1, -1 );
        }

        test_assert( slate::func::is_grid_2d_cyclic(  1, 40, func )
                     == is_row_cyclic );
        test_assert( slate::func::is_grid_2d_cyclic(  1, 40, func, out_o, out_p, out_q )
                     == is_row_cyclic );
        if (is_row_cyclic) {
            test_pq( 1, q );
        }
        else {
            test_opq( slate::GridOrder::Unknown, -1, -1 );
        }

        test_assert( slate::func::is_grid_2d_cyclic( 40, 40, func )
                     == is_2d_cyclic );
        test_assert( slate::func::is_grid_2d_cyclic( 40, 40, func, out_o, out_p, out_q )
                     == is_2d_cyclic );
        if (is_2d_cyclic) {
            if (p != 1 && q != 1) {
                // GridOrder only garunteed for non-1d grids
                test_opq( is_col_layout ? slate::GridOrder::Col : slate::GridOrder::Row,
                          p, q );
            }
            else {
                test_pq( p, q );
            }
        }
        else {
            test_opq( slate::GridOrder::Unknown, -1, -1 );
        }
    }

    // Test a map that's almost cyclic
    auto cyclic = slate::func::grid_2d_cyclic( slate::Layout::ColMajor, 4, 5 );
    std::function<int( std::tuple<int64_t, int64_t> )>
    tricky_func = [cyclic]( std::tuple<int64_t, int64_t> ij ) {
        int64_t i = std::get<0>( ij );
        int64_t j = std::get<1>( ij );
        if (i == 49 && j == 49) {
            return 21;
        }
        else {
            return cyclic( ij );
        }
    };
    test_assert( ! slate::func::is_grid_2d_cyclic(99, 99, tricky_func) );
    test_assert( ! slate::func::is_grid_2d_cyclic(99, 99, tricky_func, out_o, out_p, out_q) );
    test_opq( slate::GridOrder::Unknown, -1, -1 );
    test_assert( ! slate::func::is_grid_2d_cyclic(50, 99, tricky_func) );
    test_assert( ! slate::func::is_grid_2d_cyclic(50, 99, tricky_func, out_o, out_p, out_q) );
    test_opq( slate::GridOrder::Unknown, -1, -1 );
    test_assert( ! slate::func::is_grid_2d_cyclic(99, 50, tricky_func) );
    test_assert( ! slate::func::is_grid_2d_cyclic(99, 50, tricky_func, out_o, out_p, out_q) );
    test_opq( slate::GridOrder::Unknown, -1, -1 );
    test_assert( ! slate::func::is_grid_2d_cyclic(50, 50, tricky_func) );
    test_assert( ! slate::func::is_grid_2d_cyclic(50, 50, tricky_func, out_o, out_p, out_q) );
    test_opq( slate::GridOrder::Unknown, -1, -1 );

    test_assert( slate::func::is_grid_2d_cyclic(50, 49, tricky_func) );
    test_assert( slate::func::is_grid_2d_cyclic(50, 49, tricky_func, out_o, out_p, out_q) );
    test_opq( slate::GridOrder::Col, 4, 5 );
    test_assert( slate::func::is_grid_2d_cyclic(49, 50, tricky_func) );
    test_assert( slate::func::is_grid_2d_cyclic(49, 50, tricky_func, out_o, out_p, out_q) );
    test_opq( slate::GridOrder::Col, 4, 5 );
    test_assert( slate::func::is_grid_2d_cyclic(49, 49, tricky_func) );
    test_assert( slate::func::is_grid_2d_cyclic(49, 49, tricky_func, out_o, out_p, out_q) );
    test_opq( slate::GridOrder::Col, 4, 5 );

    #undef test_opq
    #undef test_pq
}

//------------------------------------------------------------------------------
/// Runs all tests. Called by unit test main().
void run_tests()
{
    run_test(test_uniform_blocksize, "test_uniform_blocksize");
    run_test(test_process_2d_grid,   "test_process_2d_grid");
    run_test(test_process_1d_grid,   "test_process_1d_grid");
    run_test(test_device_2d_grid,    "test_device_2d_grid");
    run_test(test_device_1d_grid,    "test_device_1d_grid");
    run_test(test_grid_transpose,    "test_transpose_grid");
    run_test(test_is_same_map,       "test_is_same_map");
    run_test(test_is_grid_2d_cyclic, "test_is_grid_2d_cyclic");
}

}  // namespace test

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    return unit_test_main( );  // which calls run_tests()
}
