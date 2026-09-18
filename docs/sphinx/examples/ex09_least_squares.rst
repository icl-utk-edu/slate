Example 09: Least Squares
=========================

This example demonstrates solving overdetermined (:math:`m > n`) and underdetermined (:math:`m < n`) linear systems using least squares.

Key Concepts
------------

1.  **Overdetermined Systems**: Finding :math:`x` that minimizes :math:`\|Ax - B\|_2`.
2.  **Underdetermined Systems**: Finding the minimum norm solution :math:`x` that satisfies :math:`Ax = B`.
3.  **Simplified API**: Using ``slate::least_squares_solve`` which handles both cases automatically.
4.  **Traditional API**: Using ``slate::gels``.

C++ Example
-----------


**Overdetermined Least Squares (Lines 27-46)**

.. code-block:: cpp

    slate::Matrix<scalar_type> A( m, n, nb, ... );
    slate::Matrix<scalar_type> BX( max_mn, nrhs, nb, ... );

    // BX contains B on input
    auto B = BX; // View of top m rows
    auto X = BX.slice( 0, n-1, 0, nrhs-1 ); // View where X will be

    slate::least_squares_solve( A, BX );

For overdetermined systems (:math:`m \ge n`):

- `A` is `m` by `n`.
- The RHS matrix `BX` must be large enough to hold both the input `B` (`m` rows) and the result `X` (conceptually `n` rows, though in the algorithm `B` is overwritten in place). Since `m >= n`, `m` rows is sufficient.
- `least_squares_solve` (gels) overwrites `A` with QR factors and `BX` with the solution.

**Underdetermined Least Squares (Lines 59-82)**

.. code-block:: cpp

    // solve A^H X = B
    auto AH = conj_transpose( A );
    slate::least_squares_solve( AH, BX );

For underdetermined systems (:math:`m < n`), we typically solve :math:`A x = B` (minimum norm solution).
SLATE's `gels` routine expects an `m` by `n` matrix where `m >= n`. To solve the underdetermined case :math:`A x = B` where `A` is fat (`m < n`), we mathematically transform this into a problem involving :math:`A^H` (which is tall).

- The example demonstrates solving :math:`A^H X = B` where `A` is tall (`m > n`), which effectively simulates an underdetermined system from the perspective of the transposed matrix.
- `BX` must be size `max(m, n)` by `nrhs`. Since the solution vector `X` will be larger than the input `B`, `BX` provides the necessary space.

.. literalinclude:: ../../../examples/ex09_least_squares.cc
   :language: cpp
   :linenos:
