// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// This test was adapted from
// OpenMP Examples Version 4.5.0 - November 2016 - Example icv.1.c

#include "slate/internal/openmp.hh"
#include "slate/internal/OmpSetMaxActiveLevels.hh"
#include "unit_test.hh"

#include <unistd.h>

namespace test {

void test_OmpSetMaxActiveLevels()
{
    // Save errors to info, since a thread throwing an exception causes an abort.
    int info = 0;

    int orig_levels = 0;
    omp_set_max_active_levels( orig_levels );
    test_assert( omp_get_max_active_levels() == orig_levels );

    {
        // Using slate::OmpSetMaxActiveLevels ensures a minimum value set
        // by omp_set_max_active_levels(), and resets to original when
        // this class leaves the scope.
        slate::OmpSetMaxActiveLevels set_active_levels( 2 );
        test_assert( omp_get_max_active_levels() == 2 );

        omp_set_num_threads( 2 );
        #pragma omp parallel
        {
            omp_set_num_threads( 3 );
            #pragma omp parallel
            {
                omp_set_num_threads( 4 );
                #pragma omp single
                {
                    // The following would print:
                    //     Inner: max_act_lev=4, num_thds=3, max_thds=4, curr_lvl 2, cur_active_lvl 2
                    //     Inner: max_act_lev=4, num_thds=3, max_thds=4, curr_lvl 2, cur_active_lvl 2
                    // printf( "Inner: max_act_lev=%d, num_thds=%d, max_thds=%d, curr_lvl %d, cur_active_lvl %d\n",
                    //         omp_get_max_active_levels(), omp_get_num_threads(),
                    //         omp_get_max_threads(), omp_get_level(), omp_get_active_level() );
                    if (omp_get_active_level() != 2)
                        info = __LINE__;

                    #pragma omp parallel
                    {
                        omp_set_num_threads( 4 );
                        #pragma omp single
                        {
                            // Even though this is the 3rd parallel level (omp_get_level() = 3)
                            // the set_active_levels( 2 ) call should restrict omp_get_active_level() to 2.
                            // The assertion below check that omp_get_active_level() is 2.

                            // The following would print:
                            //     Inner-2: max_act_lev=2, num_thds=1, max_thds=4, curr_lvl 3, cur_active_lvl 2
                            //     Inner-2: max_act_lev=2, num_thds=1, max_thds=4, curr_lvl 3, cur_active_lvl 2
                            // printf( "Inner-2: max_act_lev=%d, num_thds=%d, max_thds=%d, curr_lvl %d, cur_active_lvl %d\n",
                            //         omp_get_max_active_levels(), omp_get_num_threads(),
                            //         omp_get_max_threads(), omp_get_level(), omp_get_active_level() );
                            if (omp_get_active_level() != 2)
                                info = __LINE__;
                        }
                    }
                }
            }
            #pragma omp barrier
            #pragma omp single
            {
                // The following would print:
                //     Outer: max_act_lev=4, num_thds=2, max_thds=3, curr_lvl 1, cur_active_lvl 1
                // printf( "Outer: max_act_lev=%d, num_thds=%d, max_thds=%d, curr_lvl %d, cur_active_lvl %d\n",
                //         omp_get_max_active_levels(), omp_get_num_threads(),
                //         omp_get_max_threads(), omp_get_level(), omp_get_active_level() );
                if (omp_get_active_level() != 1)
                    info = __LINE__;
            }
        }

        test_assert( omp_get_max_active_levels() == 2 );
    }

    // Verify that OmpSetMaxActiveLevels restores it to original.
    test_assert( omp_get_max_active_levels() == orig_levels );

    // Verify that no threads had errors.
    test_assert( info == 0 );
}

//------------------------------------------------------------------------------
/// Runs all tests. Called by unit test main().
void run_tests()
{
    run_test( test_OmpSetMaxActiveLevels, "OmpSetMaxActiveLevels()" );
}

}  // namespace test

//------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    return unit_test_main();  // which calls run_tests()
}
