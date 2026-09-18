Device (GPU) Operations
========================

GPU device management and asynchronous LAPACK operations using cuSOLVER, rocSOLVER, or SYCL.

Queue Management
----------------

.. doxygenclass:: lapack::Queue
   :project: LAPACKPP
   :members:

GPU Operations
--------------

Cholesky Factorization
^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lapack::potrf(lapack::Uplo, int64_t, scalar_t*, int64_t, device_info_int*, lapack::Queue&)
   :project: LAPACKPP

LU Factorization
^^^^^^^^^^^^^^^^

.. doxygenfunction:: lapack::getrf_work_size_bytes
   :project: LAPACKPP

.. doxygenfunction:: lapack::getrf(int64_t, int64_t, scalar_t*, int64_t, device_pivot_int*, void*, size_t, void*, size_t, device_info_int*, lapack::Queue&)
   :project: LAPACKPP

QR Factorization
^^^^^^^^^^^^^^^^

.. doxygenfunction:: lapack::geqrf_work_size_bytes
   :project: LAPACKPP

.. doxygenfunction:: lapack::geqrf(int64_t, int64_t, scalar_t*, int64_t, scalar_t*, void*, size_t, void*, size_t, device_info_int*, lapack::Queue&)
   :project: LAPACKPP

Eigenvalue Decomposition
^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: lapack::heevd_work_size_bytes
   :project: LAPACKPP

.. doxygenfunction:: lapack::heevd(lapack::Job, lapack::Uplo, int64_t, scalar_t*, int64_t, blas::real_type<scalar_t>*, void*, size_t, void*, size_t, device_info_int*, lapack::Queue&)
   :project: LAPACKPP

Device Memory Types
-------------------

LAPACK++ defines integer types matching vendor GPU libraries:

- ``device_info_int``: Return status from device operations (int for CUDA/ROCm, int64_t otherwise)
- ``device_pivot_int``: Pivot indices (int64_t for cuSOLVER 11+, int for ROCm, int64_t otherwise)

These types ensure ABI compatibility with vendor-specific libraries while maintaining a unified interface.
