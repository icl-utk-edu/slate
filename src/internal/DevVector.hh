// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef SLATE_DEV_VECTOR_HH
#define SLATE_DEV_VECTOR_HH

namespace slate {

namespace internal {

//------------------------------------------------------------------------------
/// A simple vector class for GPU memory, loosely based on std::vector.
///
template <typename scalar_t>
class DevVector
{
public:
    /// Constructs empty vector.
    /// No memory is allocated, the data pointer is null.
    DevVector():
        data_( nullptr ),
        size_( 0 ),
        device_( -1 )
    {}

    /// Constructs vector, allocating in_size elements on device.
    /// Unlike std::vector, elements are uninitialized.
    /// todo: should this take queue instead of device, for SYCL?
    DevVector( size_t in_size, int device, blas::Queue &queue ):
        DevVector()
    {
        resize( in_size, device, queue );
    }

    /// Destroys vector, freeing memory.
    ~DevVector()
    {
        // This is just a check that a clear() was called before the
        // destructor happens.  The destructor does not maintain
        // access to the blas::Queue, so the user needs to call
        // clear(queue) explicity.
        assert(data_ == nullptr);
    }

    // Frees the memory, setting size to 0.
    void clear(blas::Queue &queue)
    {
        if (data_) {
            blas::device_free( data_, queue );
            data_ = nullptr;
            size_ = 0;
            device_ = -1;
        }
    }

    /// Frees existing memory and allocates new memory for in_size
    /// elements on device. Unlike std::vector, this does not copy old
    /// elements to new array.
    void resize( size_t in_size, int device, blas::Queue &queue )
    {
        clear( queue );
        data_ = blas::device_malloc<scalar_t>( in_size, queue );
        size_ = in_size;
    }

    /// @return reference to element i. Since the element is on the GPU,
    /// it can't be dereferenced; this is really only useful for taking
    /// addresses: &x[ i ].
    scalar_t& operator[] ( size_t i ) { return data_[ i ]; }

    /// @return underlying array.
    scalar_t* data() { return data_; }

    /// @return number of elements in the vector.
    size_t size() const { return size_; }

    /// @return whether the vector is empty (size == 0).
    bool empty() const { return size_ == 0; }

private:
    scalar_t* data_;
    size_t size_;
    int device_;
};

} // namespace internal

} // namespace slate

#endif // SLATE_DEV_VECTOR_HH
