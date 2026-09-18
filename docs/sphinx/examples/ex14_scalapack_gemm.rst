Example 14: ScaLAPACK Compatibility
===================================

This example demonstrates SLATE's ScaLAPACK compatibility layer.

Key Concepts
------------

1.  **ScaLAPACK Interception**: SLATE can intercept standard ScaLAPACK calls (like ``pdgemm``) and execute them using SLATE algorithms.
2.  **Legacy Code Support**: Allows existing ScaLAPACK applications to benefit from SLATE performance without code changes (just linking).
3.  **BLACS Initialization**: The example sets up the BLACS grid and ScaLAPACK descriptors as usual.

C++ Example
-----------



**BLACS Initialization (Lines 45-52)**

.. code-block:: cpp

    Cblacs_pinfo( &iam, &nprocs );
    Cblacs_get( -1, 0, &ictxt );
    Cblacs_gridinit( &ictxt, "Col", grid_p, grid_q );

Standard setup for any ScaLAPACK program. This initializes the process grid.

**ScaLAPACK Descriptors (Lines 55-82)**

.. code-block:: cpp

    int mlocA = numroc( ... );
    descinit( descA, ... );

Allocates local memory (`mloc` * `nloc`) and initializes the array descriptor `descA` which describes the distributed matrix layout (dimensions, block size, process grid). This is standard ScaLAPACK boilerplate.

**PBLAS Call (Lines 88-111)**

.. code-block:: cpp

    psgemm( ... ); // float
    pdgemm( ... ); // double
    pcgemm( ... ); // complex<float>
    pzgemm( ... ); // complex<double>

The code calls the standard PBLAS functions (`p[sdcz]gemm`).
- **Crucial Point**: If this program is linked against the SLATE ScaLAPACK API library (`-lslate_scalapack_api`), these calls will be intercepted by SLATE.
- SLATE converts the ScaLAPACK descriptors to SLATE `Matrix` objects internally, executes the operation using SLATE's engine (potentially on GPUs), and then ensures the result is consistent with ScaLAPACK expectations.
- This allows drop-in acceleration for legacy codes.

.. literalinclude:: ../../../examples/ex14_scalapack_gemm.cc
   :language: cpp
   :linenos:
