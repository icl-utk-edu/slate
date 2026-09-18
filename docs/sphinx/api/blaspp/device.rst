Device (GPU) Operations
========================

GPU device management and asynchronous BLAS operations.

Queue Management
----------------

.. doxygenclass:: blas::Queue
   :project: BLASPP
   :members:

Device Memory and Batch Operations
-----------------------------------

See ``blaspp/include/blas/device.hh`` and ``blaspp/include/blas/batch_common.hh`` for:

**Device Memory Functions:**
- ``device_malloc()`` - Allocate device memory
- ``device_free()`` - Free device memory  
- ``device_memcpy()`` - Copy between host and device
- ``device_malloc_pinned()`` - Allocate pinned host memory
- ``device_free_pinned()`` - Free pinned host memory
- ``get_device_count()`` - Query number of GPU devices

**Batch Operation Validators:**
- ``gemm_check()`` - Validate batch GEMM parameters
- ``trsm_check()`` - Validate batch TRSM parameters
- ``trmm_check()`` - Validate batch TRMM parameters
- ``hemm_check()`` - Validate batch HEMM parameters
- ``herk_check()`` - Validate batch HERK parameters
- ``symm_check()`` - Validate batch SYMM parameters
- ``syrk_check()`` - Validate batch SYRK parameters
- ``her2k_check()`` - Validate batch HER2K parameters
- ``syr2k_check()`` - Validate batch SYR2K parameters

All batch operations execute asynchronously on device queues and support CUDA, ROCm/HIP, and SYCL backends.
