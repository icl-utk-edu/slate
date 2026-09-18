ScaLAPACK Compatibility
=======================

SLATE provides compatibility APIs for easy transition from ScaLAPACK and LAPACK applications. 
This chapter covers both APIs and migration strategies.


Overview
--------

SLATE provides two compatibility layers:

1. **LAPACK Compatibility API**: For single-node LAPACK applications
2. **ScaLAPACK Compatibility API**: For distributed ScaLAPACK applications

Both APIs allow existing code to use SLATE with minimal changes.


LAPACK Compatibility API
------------------------

The LAPACK compatibility API provides routines with a ``slate_`` prefix that match standard 
LAPACK interfaces.


Using the API
~~~~~~~~~~~~~

**C Example:**

.. code-block:: c

    // Compile with:
    // mpicc -o example example.c -lslate_lapack_api
    
    // Original LAPACK call
    dgetrf_(&m, &n, A, &lda, ipiv, &info);
    
    // SLATE equivalent
    slate_dgetrf_(&m, &n, A, &lda, ipiv, &info);


**Fortran Example:**

.. code-block:: fortran

    !! Compile with:
    !! mpif90 -o example example.f90 -lslate_lapack_api
    
    !! Original LAPACK call
    call dgetrf(m, n, A, lda, ipiv, info)
    
    !! SLATE equivalent
    call slate_dgetrf(m, n, A, lda, ipiv, info)


How It Works
~~~~~~~~~~~~

The compatibility library:

1. Creates SLATE matrix from LAPACK data using ``fromLAPACK``
2. Sets execution target from environment variables
3. Calls the corresponding SLATE routine
4. Returns results in the original LAPACK arrays


Configuration
~~~~~~~~~~~~~

Set execution target via environment variables:

.. code-block:: bash

    # CPU execution (default)
    export SLATE_LAPACK_TARGET=HostTask
    
    # GPU execution
    export SLATE_LAPACK_TARGET=Devices


Tile Size
~~~~~~~~~

SLATE divides LAPACK matrices into tiles. The tile size can be configured:

.. code-block:: bash

    export SLATE_LAPACK_NB=256


Available Routines
~~~~~~~~~~~~~~~~~~

All routines have ``slate_`` prefix with standard LAPACK naming:

- Linear systems: ``slate_dgesv``, ``slate_dgetrf``, ``slate_dgetrs``
- Cholesky: ``slate_dposv``, ``slate_dpotrf``, ``slate_dpotrs``
- Least squares: ``slate_dgels``
- Eigenvalues: ``slate_dsyev``, ``slate_dheev``
- SVD: ``slate_dgesvd``
- BLAS: ``slate_dgemm``, ``slate_dtrsm``, etc.


ScaLAPACK Compatibility API
---------------------------

The ScaLAPACK compatibility API is **link-time compatible** with standard ScaLAPACK, 
using identical function names and parameters.


Using the API
~~~~~~~~~~~~~

Link with the SLATE ScaLAPACK library **before** the actual ScaLAPACK:

**C Example:**

.. code-block:: c

    // Compile with:
    // mpicc -o example example.c -lslate_scalapack_api -lscalapack
    
    // Standard ScaLAPACK call - automatically intercepted by SLATE
    pdgetrf_(&m, &n, A, &ia, &ja, descA, ipiv, &info);


**Fortran Example:**

.. code-block:: fortran

    !! Compile with:
    !! mpif90 -o example example.f90 -lslate_scalapack_api -lscalapack
    
    !! Standard ScaLAPACK call - automatically intercepted by SLATE
    call pdgetrf(m, n, A, ia, ja, descA, ipiv, info)


How It Works
~~~~~~~~~~~~

The compatibility library intercepts ScaLAPACK function calls:

1. Intercepts standard ScaLAPACK routine names (``pdgemm``, ``PDGEMM``, ``pdgemm_``, etc.)
2. Maps ScaLAPACK descriptors to SLATE matrices using ``fromScaLAPACK``
3. Uses ScaLAPACK blocking factor as SLATE tile size
4. Calls the SLATE implementation
5. Routines not implemented in SLATE fall through to actual ScaLAPACK


Configuration
~~~~~~~~~~~~~

Set execution target via environment variables:

.. code-block:: bash

    # CPU execution (default)
    export SLATE_SCALAPACK_TARGET=HostTask
    
    # GPU execution
    export SLATE_SCALAPACK_TARGET=Devices


Supported Routines
~~~~~~~~~~~~~~~~~~

Currently implemented ScaLAPACK routines:

**BLAS:**

- ``pdgemm``, ``psgemm``, ``pzgemm``, ``pcgemm``
- ``pdsymm``, ``pzhemm``, etc.
- ``pdsyrk``, ``pzherk``, etc.
- ``pdtrsm``, ``pstrsm``, etc.

**Linear Systems:**

- ``pdgesv``, ``psgesv``, etc.
- ``pdgetrf``, ``pdgetrs``
- ``pdposv``, ``pdpotrf``, ``pdpotrs``

**Eigenvalues:**

- ``pdsyev``, ``pzheev``

**SVD:**

- ``pdgesvd``, ``pzgesvd``

Routines not in the list pass through to the original ScaLAPACK.


Matrix Layout Compatibility
---------------------------

SLATE natively supports the ScaLAPACK 2D block-cyclic layout.


Creating from ScaLAPACK Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

    // ScaLAPACK-style allocation
    int myrow = mpi_rank % p;
    int mycol = mpi_rank / p;
    int64_t mlocal = slate::num_local_rows_cols(m, nb, myrow, 0, p);
    int64_t nlocal = slate::num_local_rows_cols(n, nb, mycol, 0, q);
    int64_t lld = mlocal;  // Local leading dimension
    double* A_data = new double[lld * nlocal];
    
    // Create SLATE matrix from ScaLAPACK data
    auto A = slate::Matrix<double>::fromScaLAPACK(
        m, n,                    // Global dimensions
        A_data,                  // Local data array
        lld,                     // Local leading dimension
        nb, nb,                  // Block size
        slate::GridOrder::Col,   // Column-major grid (ScaLAPACK default)
        p, q,                    // Process grid
        MPI_COMM_WORLD);


Grid Order
~~~~~~~~~~

ScaLAPACK typically uses column-major process grids:

.. code-block:: cpp

    // Column-major (ScaLAPACK default)
    slate::GridOrder::Col
    // Process 0 at (0,0), Process 1 at (1,0), ...
    
    // Row-major
    slate::GridOrder::Row
    // Process 0 at (0,0), Process 1 at (0,1), ...


Migration Strategies
--------------------

Gradual Migration
~~~~~~~~~~~~~~~~~

1. **Start with compatibility API**: Link with ``slate_scalapack_api``
2. **Verify correctness**: Run tests to ensure results match
3. **Benchmark performance**: Compare SLATE vs ScaLAPACK
4. **Enable GPU**: Set ``SLATE_SCALAPACK_TARGET=Devices``
5. **Migrate to native API**: For new code or performance-critical sections


Direct Migration
~~~~~~~~~~~~~~~~

Convert ScaLAPACK calls directly to SLATE:

**Before (ScaLAPACK):**

.. code-block:: fortran

    call descinit(descA, m, n, nb, nb, 0, 0, ictxt, lld, info)
    call pdgetrf(m, n, A, 1, 1, descA, ipiv, info)
    call pdgetrs('N', n, nrhs, A, 1, 1, descA, ipiv, 
                 B, 1, 1, descB, info)


**After (SLATE C++):**

.. code-block:: cpp

    auto A = slate::Matrix<double>::fromScaLAPACK(
        m, n, A_data, lld, nb, nb, 
        slate::GridOrder::Col, p, q, MPI_COMM_WORLD);
    auto B = slate::Matrix<double>::fromScaLAPACK(...);
    
    slate::Pivots pivots;
    slate::lu_factor(A, pivots);
    slate::lu_solve_using_factor(A, pivots, B);


Performance Considerations
--------------------------

Tile Size
~~~~~~~~~

ScaLAPACK blocking factor becomes SLATE tile size. Consider:

- SLATE performs better with larger tiles (especially on GPU)
- If ScaLAPACK uses small ``nb``, performance may be suboptimal
- For best performance, use larger ``nb`` (256-1024)


GPU Acceleration
~~~~~~~~~~~~~~~~

The compatibility API can run on GPUs:

.. code-block:: bash

    export SLATE_SCALAPACK_TARGET=Devices
    
    # Run existing ScaLAPACK code on GPU
    mpirun -n 4 ./my_scalapack_app

This provides GPU acceleration without code changes.


Limitations
-----------

- Not all ScaLAPACK routines are implemented in SLATE
- Some ScaLAPACK features (workspace queries) may behave differently
- Performance depends on tile size chosen in original code
- Complex workspace management may not translate directly


Troubleshooting
---------------

**Linker errors with compatibility API:**

Ensure correct link order (SLATE before ScaLAPACK):

.. code-block:: bash

    -lslate_scalapack_api -lscalapack -lblas


**Results differ from ScaLAPACK:**

Different algorithms may give different (but equally valid) results due to:

- Different pivoting strategies
- Different numerical ordering

Check that error is within tolerance, not for exact match.


**Routine not intercepted:**

Routine may not be implemented in SLATE. Check available routines or use 
native SLATE API.
