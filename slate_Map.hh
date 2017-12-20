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

#ifdef SLATE_WITH_OPENMP
    #include <omp.h>
#else
    #include "slate_NoOpenmp.hh"
#endif

namespace slate {

//------------------------------------------------------------------------------
template <typename KeyType, typename ValueType>
class Map {
private:
    typedef std::map<KeyType, ValueType> stdMap;

    stdMap std_map_;
    omp_lock_t lock_;

public:
    Map() { omp_init_lock(&lock_); }
    ~Map() { omp_destroy_lock(&lock_); }

    // begin()
    typename stdMap::iterator begin()
    {
        omp_set_lock(&lock_);
        typename stdMap::iterator begin = std_map_.begin();
        omp_unset_lock(&lock_);
        return begin;
    }
    typename stdMap::const_iterator begin() const
    {
        omp_set_lock(&lock_);
        typename stdMap::iterator begin = std_map_.begin();
        omp_unset_lock(&lock_);
        return begin;
    }

    // end()
    typename stdMap::iterator end()
    {
        omp_set_lock(&lock_);
        typename stdMap::iterator end = std_map_.end();
        omp_unset_lock(&lock_);
        return end;
    }
    typename stdMap::const_iterator end() const
    {
        omp_set_lock(&lock_);
        typename stdMap::iterator end = std_map_.end();
        omp_unset_lock(&lock_);
        return end;
    }

    // find()
    typename stdMap::iterator find(const KeyType &key)
    {
        omp_set_lock(&lock_);
        typename stdMap::iterator element = std_map_.find(key);
        omp_unset_lock(&lock_);
        return element;
    }
    typename stdMap::const_iterator find(const KeyType &key) const
    {
        omp_set_lock(&lock_);
        typename stdMap::iterator element = std_map_.find(key);
        omp_unset_lock(&lock_);
        return element;
    }

    // erase()
    typename stdMap::size_type erase(const KeyType &key)
    {
        omp_set_lock(&lock_);
        typename stdMap::size_type num_erased = std_map_.erase(key);
        omp_unset_lock(&lock_);
        return num_erased;
    }

    // [] operator
    ValueType &operator[](const KeyType &key)
    {
        omp_set_lock(&lock_);
        ValueType &tile = std_map_[key];
        omp_unset_lock(&lock_);
        return tile;
    }
    ValueType &operator[](const KeyType &key) const
    {
        omp_set_lock(&lock_);
        ValueType &tile = std_map_[key];
        omp_unset_lock(&lock_);
        return tile;
    }
};

} // namespace slate

#endif // SLATE_MAP_HH
