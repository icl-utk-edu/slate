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

#ifndef SLATE_COL_MAJOR_TILE_HH
#define SLATE_COL_MAJOR_TILE_HH

#include "slate_Tile.hh"

namespace slate {

///-----------------------------------------------------------------------------
/// \class
/// \brief
///
template <typename FloatType>
class ColMajorTile : public Tile<FloatType> {
public:
    ColMajorTile(int64_t mb, int64_t nb,
                 std::weak_ptr<Memory> memory);

    ColMajorTile(int64_t mb, int64_t nb,
                 FloatType *a, int64_t lda,
                 std::weak_ptr<Memory> memory);

    ColMajorTile(const ColMajorTile<FloatType> *src_tile, int dst_device_num);

    ~ColMajorTile() { Tile<FloatType>::deallocate(); }

    void copyTo(FloatType *a, int64_t lda);
    void copyFrom(FloatType *a, int64_t lda);

    ColMajorTile<FloatType>* copyToHost(cudaStream_t stream);
    ColMajorTile<FloatType>* copyToDevice(int device_num, cudaStream_t stream);

    void copyDataToHost(const Tile<FloatType> *dst_tile, cudaStream_t stream);
    void copyDataToDevice(const Tile<FloatType> *dst_tile, cudaStream_t stream);
};

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
ColMajorTile<FloatType>::ColMajorTile(int64_t mb, int64_t nb,
                                      std::weak_ptr<Memory> memory)
    : Tile<FloatType>(mb, nb, memory)
{
    this->stride_ = this->mb_;
    Tile<FloatType>::allocate();
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
ColMajorTile<FloatType>::ColMajorTile(int64_t mb, int64_t nb,
                                      FloatType *a, int64_t lda,
                                      std::weak_ptr<Memory> memory)
    : Tile<FloatType>(mb, nb,
                      a, lda,
                      memory) {}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
ColMajorTile<FloatType>::ColMajorTile(const ColMajorTile<FloatType> *src_tile,
                                      int dst_device_num)
{
    *this = *src_tile;
    this->origin_ = false;
    this->stride_ = this->mb_;
    this->device_num_ = dst_device_num;
    Tile<FloatType>::allocate();
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
void ColMajorTile<FloatType>::copyTo(FloatType *a, int64_t lda)
{
    auto mb_ = this->mb_;
    auto nb_ = this->nb_;
    auto data_ = this->data_;
    auto stride_ = this->stride_;

    for (int64_t n = 0; n < nb_; ++n)
        memcpy(&data_[n*stride_], &a[n*lda], sizeof(FloatType)*mb_);
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
void ColMajorTile<FloatType>::copyFrom(FloatType *a, int64_t lda)
{
    auto mb_ = this->mb_;
    auto nb_ = this->nb_;
    auto data_ = this->data_;
    auto stride_ = this->stride_;

    for (int64_t n = 0; n < nb_; ++n)
        memcpy(&a[n*lda], &data_[n*stride_], sizeof(FloatType)*mb_);
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
ColMajorTile<FloatType>*
ColMajorTile<FloatType>::copyToHost(cudaStream_t stream)
{
    ColMajorTile<FloatType> *dst_tile =
        new ColMajorTile<FloatType>(this, this->host_num_);
    this->copyDataToHost(dst_tile, stream);
    return dst_tile;
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
ColMajorTile<FloatType>*
ColMajorTile<FloatType>::copyToDevice(int device_num, cudaStream_t stream)
{
    ColMajorTile<FloatType> *dst_tile =
        new ColMajorTile<FloatType>(this, device_num);
    this->copyDataToDevice(dst_tile, stream);
    return dst_tile;
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
void ColMajorTile<FloatType>::copyDataToHost(
    const Tile<FloatType> *dest_tile, cudaStream_t stream)
{
    auto mb_ = this->mb_;
    auto nb_ = this->nb_;
    auto data_ = this->data_;
    auto stride_ = this->stride_;
    auto device_num_ = this->device_num_;

    const ColMajorTile<FloatType>* dst_tile =
        dynamic_cast<const ColMajorTile<FloatType>*>(dest_tile);

    trace_cpu_start();
    cudaError_t error;
    error = cudaSetDevice(device_num_);
    assert(error == cudaSuccess);

    // If no stride on both sides.
    if (stride_ == mb_ &&
        dst_tile->stride_ == dst_tile->mb_) {

        // Use simple copy.
        error = cudaMemcpyAsync(
            dst_tile->data_, data_, this->size(),
            cudaMemcpyDeviceToHost, stream);
        assert(error == cudaSuccess);
    }
    else {
        // Otherwise, use 2D copy.
        void* dst = dst_tile->data_;
        const void* src = data_;
        size_t dpitch = sizeof(FloatType)*dst_tile->stride_;
        size_t spitch = sizeof(FloatType)*stride_;
        size_t width = sizeof(FloatType)*mb_;
        size_t height = nb_;

        error = cudaMemcpy2DAsync(
            dst, dpitch,
            src, spitch,
            width, height,
            cudaMemcpyDeviceToHost, stream);
        assert(error == cudaSuccess);
    }

    error = cudaStreamSynchronize(stream);
    assert(error == cudaSuccess);
    trace_cpu_stop("Gray");
}

///-----------------------------------------------------------------------------
/// \brief
///
template <typename FloatType>
void ColMajorTile<FloatType>::copyDataToDevice(
    const Tile<FloatType> *dest_tile, cudaStream_t stream)
{
    auto mb_ = this->mb_;
    auto nb_ = this->nb_;
    auto data_ = this->data_;
    auto stride_ = this->stride_;
    auto device_num_ = this->device_num_;

    const ColMajorTile<FloatType>* dst_tile =
        dynamic_cast<const ColMajorTile<FloatType>*>(dest_tile);

    trace_cpu_start();
    cudaError_t error;
    error = cudaSetDevice(dst_tile->device_num_);
    assert(error == cudaSuccess);

    // If no stride on both sides.
    if (stride_ == mb_ &&
        dst_tile->stride_ == dst_tile->mb_) {

        // Use simple copy.
        error = cudaMemcpyAsync(
            dst_tile->data_, data_, this->size(),
            cudaMemcpyHostToDevice, stream);
        assert(error == cudaSuccess);
    }
    else {
        // Otherwise, use 2D copy.
        void* dst = dst_tile->data_;
        const void* src = data_;
        size_t dpitch = sizeof(FloatType)*dst_tile->stride_;
        size_t spitch = sizeof(FloatType)*stride_;
        size_t width = sizeof(FloatType)*mb_;
        size_t height = nb_;

        error = cudaMemcpy2DAsync(
            dst, dpitch,
            src, spitch,
            width, height,
            cudaMemcpyHostToDevice, stream);
        assert(error == cudaSuccess);
    }

    error = cudaStreamSynchronize(stream);
    assert(error == cudaSuccess);
    trace_cpu_stop("LightGray");
}

} // namespace slate

#endif // SLATE_COL_MAJOR_TILE_HH
