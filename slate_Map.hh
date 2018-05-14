//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the University of Tennessee nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------
// This research was supported by the Exascale Computing Project (17-SC-20-SC),
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//------------------------------------------------------------------------------
// Need assistance with the SLATE software? Join the "SLATE User" Google group
// by going to https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user
// and clicking "Apply to join group". Upon acceptance, email your questions and
// comments to <slate-user@icl.utk.edu>.
//------------------------------------------------------------------------------

#ifndef SLATE_MAP_HH
#define SLATE_MAP_HH

#include <map>

#include "slate_openmp.hh"

namespace slate {

//------------------------------------------------------------------------------
/// Constructor acquires lock; destructor releases lock.
/// This provides safety in case an exception is thrown, which would otherwise
/// by-pass the unlock. Like std::lock_guard, but for OpenMP locks.
///
class LockGuard {
public:
    LockGuard(omp_nest_lock_t* lock)
        : lock_(lock)
    {
        omp_set_nest_lock(lock_);
    }

    ~LockGuard()
    {
        omp_unset_nest_lock(lock_);
    }

private:
    omp_nest_lock_t* lock_;
};

// -----------------------------------------------------------------------------
/// Slate::Map
/// @brief Used for traversal of Matrix's tiles
/// @detailed Used by Slate::Storage to create mapping for each tile stored
///     in a matrix
/// @tparam KeyType Type for the key value
/// @tparam ValueType Type for the stored value
///
template <typename KeyType, typename ValueType>
class Map {
private:
    typedef std::map<KeyType, ValueType> stdMap;

    stdMap std_map_;
    mutable omp_nest_lock_t lock_;

public:
    using iterator = typename stdMap::iterator;
    using const_iterator = typename stdMap::const_iterator;

    /// Constructor for Map class
    Map() { omp_init_nest_lock(&lock_); }
    /// Destructor for Map class
    ~Map() { omp_destroy_nest_lock(&lock_); }

    omp_nest_lock_t* get_lock()
    {
        return &lock_;
    }

    //--------
    // begin()
    typename stdMap::iterator begin()
    {
        LockGuard guard(&lock_);
        typename stdMap::iterator begin = std_map_.begin();
        return begin;
    }
    typename stdMap::const_iterator begin() const
    {
        LockGuard guard(&lock_);
        typename stdMap::iterator begin = std_map_.begin();
        return begin;
    }

    //------
    // end()
    typename stdMap::iterator end()
    {
        LockGuard guard(&lock_);
        typename stdMap::iterator end = std_map_.end();
        return end;
    }
    typename stdMap::const_iterator end() const
    {
        LockGuard guard(&lock_);
        typename stdMap::iterator end = std_map_.end();
        return end;
    }

    //-------
    // find()
    typename stdMap::iterator find(const KeyType &key)
    {
        LockGuard guard(&lock_);
        typename stdMap::iterator element = std_map_.find(key);
        return element;
    }
    typename stdMap::const_iterator find(const KeyType &key) const
    {
        LockGuard guard(&lock_);
        typename stdMap::iterator element = std_map_.find(key);
        return element;
    }

    //--------
    // erase()
    typename stdMap::size_type erase(const KeyType &key)
    {
        LockGuard guard(&lock_);
        typename stdMap::size_type num_erased = std_map_.erase(key);
        return num_erased;
    }
    typename stdMap::iterator erase(typename stdMap::const_iterator position)
    {
        LockGuard guard(&lock_);
        typename stdMap::iterator next = std_map_.erase(position);
        return next;
    }

    //------------
    // operator []
    ValueType& operator[](const KeyType &key)
    {
        LockGuard guard(&lock_);
        ValueType& tile = std_map_[key];
        return tile;
    }
    ValueType& operator[](const KeyType &key) const
    {
        LockGuard guard(&lock_);
        ValueType& tile = std_map_[key];
        return tile;
    }

    //------------
    // at()
    ValueType& at(const KeyType &key)
    {
        LockGuard guard(&lock_);
        ValueType& tile = std_map_.at(key);
        return tile;
    }
    ValueType& at(const KeyType &key) const
    {
        LockGuard guard(&lock_);
        ValueType& tile = std_map_.at(key);
        return tile;
    }

    //------------
    // clear()
    void clear()
    {
        LockGuard guard(&lock_);
        std_map_.clear();
    }

    //------------
    // size()
    size_t size() const
    {
        LockGuard guard(&lock_);
        size_t size = std_map_.size();
        return size;
    }
};

} // namespace slate

#endif // SLATE_MAP_HH
