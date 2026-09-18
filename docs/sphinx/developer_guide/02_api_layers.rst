API Layers
==========

SLATE's API is composed of several layers: drivers, computational routines, internal task routines, and tile operations. The drivers and computational routines are the primary public API; the internal task and tile routines implement major (thread parallel) and minor (sequential) tasks, respectively. The LAPACK++ and BLAS++ packages, including Batched BLAS++, are independent packages developed for SLATE that provide a portability layer over vendor-optimized CPU and GPU LAPACK and BLAS routines.

Drivers
-------

As in LAPACK and ScaLAPACK, driver routines solve an entire problem, such as a linear system :math:`Ax = b` (routines ``gesv``, ``posv``), a least squares problem :math:`Ax \approx b` (``gels``), or a singular-value decomposition :math:`A = U \Sigma V^H` (``svd``). Drivers in turn call computational routines to solve sub-problems. Drivers are typically independent of the target (CPU or device), delegating those details to lower level routines.

Computational Routines
----------------------

As in LAPACK and ScaLAPACK, computational routines solve a sub-problem, such as computing an LU factorization (``getrf``), or solving a linear system given an LU factorization (``getrs``). In SLATE, these are templated on target (CPU or device), with the code typically independent of the device. Communication between processes and dependencies between tasks are managed at this level. SLATE's Parallel Basic Linear Algebra Subprograms (PBLAS) exists at this level.

Comments on the Code
^^^^^^^^^^^^^^^^^^^^

Normally, matrices are passed by reference (``Matrix<scalar_t>& A``), as this avoids invoking (shallow) copy constructors. For Cholesky, however, the matrix may get transposed to handle the ``uplo=Upper`` case, so it must be passed by value; see :doc:`04_side_uplo_trans`.

Dependencies are tracked via a dummy vector—not based on the actual data—unlike in pure dataflow implementations like PLASMA. For Cholesky, entries in the dummy vector represent each column. The dummy vector is allocated using ``std::vector`` for exception safety, but OpenMP needs a raw pointer to its data.

.. note::
   The tile life and tile tick mechanism is deprecated and has been removed in favor of an explicit release task. Previously, ``TileReleaseStrategy`` was used to control this behavior, but it has been deprecated.

Template Dispatch
^^^^^^^^^^^^^^^^^

The public routine that the user actually calls (e.g., ``slate::potrf``) dispatches to the target-specific versions. The user can specify the target as ``HostTask``, ``HostNest``, ``HostBatch``, or ``Devices`` via the options parameter.

Executing Multiple Internal Routines on Devices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Care must be taken when executing multiple internal routines simultaneously on a GPU to avoid data hazards and race conditions. Data hazards occur when two routines write to the same memory buffer, for instance the GPU batch array. To avoid data hazards in SLATE, we allocate multiple GPU queues and batch arrays, one for each task that will run simultaneously on the GPU.

.. warning::
   Implicit tile release is deprecated and error-prone. Race conditions can occur when internal routines do ``tileRelease`` to release local workspace tiles. The trailing matrix update task and the lookahead update task can both release the same tile. Whichever does so first removes the tile from memory, so the other routine is no longer able to access it. Setting the ``OnHold`` status in MOSI will disable ``tileRelease``, then the tile must be manually released later on.

Internal Routines for Major, Parallel Tasks
-------------------------------------------

SLATE adds a third layer of internal routines that generally perform one step or major task of a computational routine. These are typically executed in parallel across multiple CPU cores, or as a batch routine on the GPU. For instance, in the outer :math:`k` loop, ``slate::gemmC`` calls a sequence of ``slate::internal::gemm``, each of which performs one block outer product. Most internal routines consist of a set of independent tile operations that can be issued as a batched operation or an OpenMP parallel-for loop, with no task dependencies to track.

Internal routines provide device-specific implementations such as OpenMP nested tasks, parallel for-loops, or batched BLAS operations. In many linear algebra algorithms, these internal routines implement the trailing matrix update.

Batched GPU Tasks
^^^^^^^^^^^^^^^^^

Compared to the CPU implementation, the batched GPU implementation is significantly more complicated. Each device is handled by a separate task in parallel. After some initialization to deal with transposed and conjugate-transposed matrices, it loops over all the relevant tiles to copy them to the GPU device if they aren't already resident. This uses the MOSI tile coherency API to determine which tiles need to be transferred, then transfers them with a single call. Copying of the sets is launched as nested tasks for increased parallelism.

Then it constructs the batch arrays of pointers to tiles on the GPU and calls a batched BLAS++ routine. Each region with a different tile size is handled by a separate batch call.

Tile Operations for Small, Sequential Tasks
-------------------------------------------

Tile routines update one or a small number of individual tiles, generally sequentially on a single CPU core. For instance, a tile gemm takes three tiles, :math:`A`, :math:`B`, and :math:`C`, and updates :math:`C`. Transposition of individual tiles is resolved at this level when calling optimized BLAS. This allows higher-level operations to ignore whether a matrix is transposed or not. Currently, all tile operations are CPU-only, since accelerators use only batch operations.

BLAS++, Batched BLAS++, and LAPACK++
------------------------------------

At the lowest level, the BLAS++ and LAPACK++ packages provide thin, precision-independent, overloaded C++ wrappers around traditional BLAS, batched BLAS, and LAPACK routines. They use C++ calling conventions and enum values instead of character constants, but otherwise the calling sequence is similar to the standard BLAS and LAPACK routines. BLAS++ also includes batched BLAS, on both CPUs and GPUs.

Work Routines for Actual OpenMP Work
------------------------------------

In a couple instances, SLATE implements a middle "work" layer between the computational routines and the internal routines, which is called from within an OpenMP parallel region. This layer was introduced based on the needs of the generalized eigenvalue routine ``slate::hegst``. Outside this context, it has not been widely used in SLATE.

Some computational routines need to work on problem sizes larger than what internal routines can handle inside their parallel region. For instance, ``slate::hegst`` calls ``slate::trsm`` and ``slate::trmm`` that take a large triangular matrix, instead of a single tile triangular matrix. The ``internal::trsm`` and ``internal::trmm`` handle only the single tile triangular matrix case. Therefore, ``slate::work`` layer can allow ``slate::hegst`` to invoke a big triangular matrix solve and triangular matrix-matrix multiplication within ``slate::hegst``, without having multiple OpenMP parallel regions.
