
#ifndef SLATE_TILES_HH
#define SLATE_TILES_HH

#include "slate_Tile.hh"

#include <map>
// #include <unordered_map>

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

    stdMap tiles_;
    omp_lock_t lock_;

public:
    Map() { omp_init_lock(&lock_); }
    ~Map() { omp_destroy_lock(&lock_); }

    // begin()
    typename stdMap::iterator begin()
    {
        omp_set_lock(&lock_);
        typename stdMap::iterator begin = tiles_.begin();
        omp_unset_lock(&lock_);
        return begin;
    }
    typename stdMap::const_iterator begin() const
    {
        omp_set_lock(&lock_);
        typename stdMap::iterator begin = tiles_.begin();
        omp_unset_lock(&lock_);
        return begin;
    }

    // end()
    typename stdMap::iterator end()
    {
        omp_set_lock(&lock_);
        typename stdMap::iterator end = tiles_.end();
        omp_unset_lock(&lock_);
        return end;
    }
    typename stdMap::const_iterator end() const
    {
        omp_set_lock(&lock_);
        typename stdMap::iterator end = tiles_.end();
        omp_unset_lock(&lock_);
        return end;
    }

    // find()
    typename stdMap::iterator find(const KeyType &key)
    {
        omp_set_lock(&lock_);
        typename stdMap::iterator element = tiles_.find(key);
        omp_unset_lock(&lock_);
        return element;
    }
    typename stdMap::const_iterator find(const KeyType &key) const
    {
        omp_set_lock(&lock_);
        typename stdMap::iterator element = tiles_.find(key);
        omp_unset_lock(&lock_);
        return element;
    }

    // erase()
    typename stdMap::size_type erase(const KeyType &key)
    {
        omp_set_lock(&lock_);
        typename stdMap::size_type num_erased = tiles_.erase(key);
        omp_unset_lock(&lock_);
        return num_erased;
    }

    // [] operator
    ValueType &operator[](const KeyType &key)
    {
        omp_set_lock(&lock_);
        ValueType &tile = tiles_[key];
        omp_unset_lock(&lock_);
        return tile;
    }
    ValueType &operator[](const KeyType &key) const
    {
        omp_set_lock(&lock_);
        ValueType &tile = tiles_[key];
        omp_unset_lock(&lock_);
        return tile;
    }
};

} // namespace slate

#endif // SLATE_TILES_HH
