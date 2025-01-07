// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_OMPSETMAXACTIVELEVELS_HH
#define SLATE_OMPSETMAXACTIVELEVELS_HH

#include "slate/internal/openmp.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Minimum number of OpenMP active parallel levels to allow
/// multi-threaded panel implementation.
/// @ingroup enum
const int MinOmpActiveLevels = 4;

//------------------------------------------------------------------------------
/// Constructor ensures that OpenMP max-active-levels-var ICV has a
/// minimum value;  destructor resets original value.
/// This provides safety in case an exception is thrown, which would otherwise
/// by-pass the reset.
///
class OmpSetMaxActiveLevels {

private:
    int orig_max_active_levels_ = -1;

public:
    //----------------------------------------
    /// Require omp nested levels to have a minimum value.
    ///
    /// @param[in] min_active_levels
    ///     Ensure that OpenMP max-active-levels-var ICV has this minimum.
    OmpSetMaxActiveLevels(int min_active_levels)
    {
        int curr_max_active_levels = omp_get_max_active_levels();
        #if defined(_OPENMP) && _OPENMP < 200805
            // if OpenMP version < 5.0 then enable omp_set_nested
            omp_set_nested(1);
        #endif
        if (min_active_levels > curr_max_active_levels) {
            // record the original value
            orig_max_active_levels_ = curr_max_active_levels;
            omp_set_max_active_levels( min_active_levels );
        }
    }

    //----------------------------------------
    /// Reset omp nested levels variable.
    ~OmpSetMaxActiveLevels()
    {
        // if original was changed, reset it
        if (orig_max_active_levels_ != -1)
            omp_set_max_active_levels( orig_max_active_levels_ );
    }
};

}  // namespace slate

#endif // SLATE_OMPSETMAXACTIVELEVELS_HH
