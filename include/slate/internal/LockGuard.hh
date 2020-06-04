//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//------------------------------------------------------------------------------

#ifndef SLATE_LOCKGUARD_HH
#define SLATE_LOCKGUARD_HH

#include "slate/internal/openmp.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Constructor acquires lock; destructor releases lock.
/// This provides safety in case an exception is thrown, which would otherwise
/// by-pass the unlock. Like std::lock_guard, but for OpenMP nested locks.
///
class LockGuard {
public:
    //----------------------------------------
    /// Acquire nested lock.
    ///
    /// @param[in,out] lock
    ///     OpenMP nested lock. Must be initialized already.
    LockGuard(omp_nest_lock_t* lock)
        : lock_(lock)
    {
        omp_set_nest_lock(lock_);
    }

    //----------------------------------------
    /// Release nested lock.
    ~LockGuard()
    {
        omp_unset_nest_lock(lock_);
    }

private:
    omp_nest_lock_t* lock_;
};

}  // namespace slate

#endif // SLATE_LOCKGUARD_HH
