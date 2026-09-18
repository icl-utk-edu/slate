Example 15: Setting Matrix Elements
===================================

This example demonstrates advanced ways to set matrix elements using lambda functions.

Key Concepts
------------

1.  **Functional Initialization**: Using `slate::set` with a lambda function `f(i, j)` to set :math:`A_{ij}`.
2.  **Parallel Execution**: The lambda function is executed in parallel across tiles.
3.  **Use Cases**:
    *   Random initialization (non-deterministic if not careful with seeds).
    *   Coordinate-based initialization (e.g., :math:`A_{ij} = i + j`).
    *   Stencil generation (e.g., Laplacian).

C++ Example
-----------


**Setting Random Values (Lines 19-48)**

.. code-block:: cpp

    using entry_type = std::function< scalar_type (int64_t, int64_t) >;
    entry_type entry = [random_max]( int64_t i, int64_t j ) { ... };
    slate::set( entry, A );

`slate::set` iterates over the matrix `A` and for every element `(i, j)`, calls the provided function `entry(i, j)` to determine the value.
- **Note**: Since tiles are processed in parallel, using a global stateful random number generator (like `rand()`) inside the lambda is non-deterministic regarding the exact value pattern across runs or ranks unless synchronized or thread-local.

**Coordinate-based Initialization (Lines 98-112)**

.. code-block:: cpp

    entry_type entry = []( int64_t i, int64_t j ) {
        // Return value based on i and j
        return i + 1 + (j + 1)/1000.;
    };
    slate::set( entry, A );

This is useful for generating deterministic test matrices where the value depends on the position.

**Stencil Initialization (Lines 128-143)**

.. code-block:: cpp

    entry_type entry = [n]( int64_t i, int64_t j ) {
        if (i == j) return -3.0; // Diagonal
        else if (...) return 0.5; // Neighbors
        // ...
    };

This pattern allows initializing sparse or structured dense matrices, such as those arising from finite difference discretizations (e.g., a 9-point Laplacian stencil). The lambda defines the connectivity logic.

.. literalinclude:: ../../../examples/ex15_set_matrix.cc
   :language: cpp
   :linenos:
