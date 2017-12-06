
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
template <class FloatType>
class Tiles {
private:
    // struct TileHash {
    //     size_t operator()(const std::tuple<int64_t, int64_t, int> &key) const
    //     {
    //         size_t hash_i = std::hash<int64_t>()(std::get<0>(key));
    //         size_t hash_j = std::hash<int64_t>()(std::get<1>(key));
    //         size_t hash_k = std::hash<int>    ()(std::get<2>(key));
    //         return (hash_i*31 + hash_j)*31 + hash_k;
    //     }
    // };

    typedef std::tuple<int64_t, int64_t, int> TilesKey;
    typedef std::map<TilesKey, Tile<FloatType>*> TilesMap;
    // typedef std::unordered_map<TilesKey, Tile<FloatType>*, TileHash> TilesMap;

    TilesMap tiles_;
    omp_lock_t lock_;

public:
    Tiles() { omp_init_lock(&lock_); }
    ~Tiles() { omp_destroy_lock(&lock_); }

    // begin()
    typename TilesMap::iterator begin()
    {
        omp_set_lock(&lock_);
        typename TilesMap::iterator begin = tiles_.begin();
        omp_unset_lock(&lock_);
        return begin;
    }
    typename TilesMap::const_iterator begin() const
    {
        omp_set_lock(&lock_);
        typename TilesMap::iterator begin = tiles_.begin();
        omp_unset_lock(&lock_);
        return begin;
    }

    // end()
    typename TilesMap::iterator end()
    {
        omp_set_lock(&lock_);
        typename TilesMap::iterator end = tiles_.end();
        omp_unset_lock(&lock_);
        return end;
    }
    typename TilesMap::const_iterator end() const
    {
        omp_set_lock(&lock_);
        typename TilesMap::iterator end = tiles_.end();
        omp_unset_lock(&lock_);
        return end;
    }

    // find()
    typename TilesMap::iterator find(const TilesKey &key)
    {
        omp_set_lock(&lock_);
        typename TilesMap::iterator element = tiles_.find(key);
        omp_unset_lock(&lock_);
        return element;
    }
    typename TilesMap::const_iterator find(const TilesKey &key) const
    {
        omp_set_lock(&lock_);
        typename TilesMap::iterator element = tiles_.find(key);
        omp_unset_lock(&lock_);
        return element;
    }

    // erase()
    typename TilesMap::size_type erase(const TilesKey &key)
    {
        omp_set_lock(&lock_);
        typename TilesMap::size_type num_erased = tiles_.erase(key);
        omp_unset_lock(&lock_);
        return num_erased;
    }

    // [] operator
    Tile<FloatType>* &operator[](const TilesKey &key)
    {
        omp_set_lock(&lock_);
        Tile<FloatType>* &tile = tiles_[key];
        omp_unset_lock(&lock_);
        return tile;
    }
    Tile<FloatType>* &operator[](const TilesKey &key) const
    {
        omp_set_lock(&lock_);
        Tile<FloatType>* &tile = tiles_[key];
        omp_unset_lock(&lock_);
        return tile;
    }
};

} // namespace slate

#endif // SLATE_TILES_HH
