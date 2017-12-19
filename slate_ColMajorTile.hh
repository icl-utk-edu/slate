
#ifndef SLATE_COL_MAJOR_TILE_HH
#define SLATE_COL_MAJOR_TILE_HH

#include "slate_Tile.hh"

namespace slate {

//------------------------------------------------------------------------------
template <typename FloatType>
class ColMajorTile : public Tile<FloatType> {
public:
    ColMajorTile(int64_t mb, int64_t nb, Memory *memory)
        : Tile<FloatType>(mb, nb, memory)
    {
        this->stride_ = this->mb_;
        Tile<FloatType>::allocate();
    }
    ColMajorTile(const ColMajorTile<FloatType> *src_tile, int dst_device_num)
    {
        *this = *src_tile;
        this->stride_ = this->mb_;
        this->device_num_ = dst_device_num;
        Tile<FloatType>::allocate();
    }
    ColMajorTile(int64_t mb, int64_t nb, FloatType *a, int64_t lda, Memory *memory)
        : Tile<FloatType>(mb, nb, memory)
    {
        this->stride_ = this->mb_;
        Tile<FloatType>::allocate();
        copyTo(a, lda);
    }
    ~ColMajorTile() {
        Tile<FloatType>::deallocate();
    }

    //------------------------------------
    void copyTo(FloatType *a, int64_t lda)
    {
        auto mb_ = this->mb_;
        auto nb_ = this->nb_;
        auto data_ = this->data_;

        for (int64_t n = 0; n < nb_; ++n)
            memcpy(&data_[n*mb_], &a[n*lda], sizeof(FloatType)*mb_);
    }
    void copyFrom(FloatType *a, int64_t lda)
    {
        auto mb_ = this->mb_;
        auto nb_ = this->nb_;
        auto data_ = this->data_;

        for (int64_t n = 0; n < nb_; ++n)
            memcpy(&a[n*lda], &data_[n*mb_], sizeof(FloatType)*mb_);
    }

    //------------------------------------------------------
    ColMajorTile<FloatType>* copyToHost(cudaStream_t stream)
    {
        ColMajorTile<FloatType> *dst_tile =
            new ColMajorTile<FloatType>(this, this->host_num_);
        this->copyDataToHost(dst_tile, stream);
        return dst_tile;
    }
    ColMajorTile<FloatType>* copyToDevice(int device_num, cudaStream_t stream)
    {
        ColMajorTile<FloatType> *dst_tile =
            new ColMajorTile<FloatType>(this, device_num);
        this->copyDataToDevice(dst_tile, stream);
        return dst_tile;
    }
};

} // namespace slate

#endif // SLATE_COL_MAJOR_TILE_HH
