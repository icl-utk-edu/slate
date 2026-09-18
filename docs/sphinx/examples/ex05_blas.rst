Example 05: BLAS Operations
===========================

This example demonstrates how to perform Basic Linear Algebra Subprograms (BLAS) operations in SLATE.

Key Concepts
------------

1.  **Matrix Multiplication**: Using ``slate::multiply`` (gemm, hemm, symm) for matrix products.
2.  **Rank Updates**: Performing rank-k (herk, syrk) and rank-2k (her2k, syr2k) updates.
3.  **Triangular Operations**: Triangular matrix multiplication (trmm) and solving triangular systems (trsm).
4.  **Simplified vs Traditional API**: Comparing the descriptive ``multiply`` API with the traditional BLAS-named API.

C++ Example
-----------

**General Matrix Multiplication (GEMM) (Lines 36-40)**

.. code-block:: cpp

    // C = alpha A B + beta C
    slate::multiply( alpha, A, B, beta, C );  // simplified API
    slate::gemm( alpha, A, B, beta, C );      // traditional API

Here we perform the standard operation :math:`C = \alpha AB + \beta C`.

- `A` is an `m` by `k` matrix.
- `B` is a `k` by `n` matrix.
- `C` is an `m` by `n` matrix.

SLATE provides both a descriptive `multiply` routine and the traditional BLAS-named `gemm`. They are equivalent.

**GPU Execution with Options (Lines 43-52)**

.. code-block:: cpp

    if (blas::get_device_count() > 0) {
        slate::Options opts = {
            { slate::Option::Lookahead, 2 },
            { slate::Option::Target, slate::Target::Devices },
        };
        slate::multiply( alpha, A, B, beta, C, opts );
    }

Most SLATE routines accept an `Options` map as the final argument. Here we:

- Set `Target::Devices` to offload computation to GPUs.
- Set `Lookahead` to 2 to overlap communication and computation.

**Transposed Multiplication (Lines 77-83)**

.. code-block:: cpp

    auto AT = transpose( A );
    auto BH = conj_transpose( B );
    slate::multiply( alpha, AT, BH, beta, C );

To compute :math:`C = \alpha A^T B^H + \beta C`, we simply create transposed views `AT` and `BH` and pass them to the multiply function. SLATE detects the transposition flags on the views and handles the logic internally.

**Symmetric/Hermitian Multiplication (SYMM/HEMM) (Lines 97-118)**

.. code-block:: cpp

    slate::multiply( alpha, A, B, beta, C );                  // simplified
    slate::symm( slate::Side::Left, alpha, A, B, beta, C );   // traditional

When `A` is a `SymmetricMatrix` (or `HermitianMatrix`), `multiply` automatically dispatches to the efficient symmetric/Hermitian algorithm (`symm`/`hemm`).

- `Side::Left` means :math:`C = \alpha A B + \beta C`.
- `Side::Right` means :math:`C = \alpha B A + \beta C` (demonstrated in lines 141-147).

**Rank-K Updates (SYRK/HERK) (Lines 230-241)**

.. code-block:: cpp

    slate::rank_k_update( alpha, A, beta, C );
    slate::syrk( alpha, A, beta, C );

Computes :math:`C = \alpha A A^T + \beta C` where `C` is symmetric. Only the designated triangle of `C` (Lower or Upper) is updated.

**Triangular Operations (TRMM/TRSM) (Lines 299-310)**

.. code-block:: cpp

    // B = alpha A B
    slate::triangular_multiply( alpha, A, B );       // trmm

    // B = alpha A^{-1} B (Solve AX = B)
    slate::triangular_solve( alpha, A, B );          // trsm

For triangular matrices, we can multiply (`trmm`) or solve (`trsm`). The simplified API names make the intent clear ("multiply" vs "solve").


.. literalinclude:: ../../../examples/ex05_blas.cc
   :language: cpp
   :linenos:


C API Example
-------------

.. literalinclude:: ../../../examples/c_api/ex05_blas.c
   :language: c
   :linenos:

Fortran API Example
-------------------

.. literalinclude:: ../../../examples/fortran/ex05_blas.f90
   :language: fortran
   :linenos:
