Example 03: Submatrices and Slicing
===================================

This example demonstrates how to work with submatrices in SLATE.

Key Concepts
------------

1.  **Tile Indexing (sub)**: Creating a submatrix view using tile indices (block coordinates). This is the most efficient way to reference submatrices in SLATE.
2.  **Element Indexing (slice)**: Creating a submatrix view using global element indices (row/column coordinates). Note that slices must align with block boundaries if they are to be treated as standard distributed matrices in many operations.

C++ Example
-----------

**Tile-based Submatrices (Lines 36-39)**

.. code-block:: cpp

    // view of A( i1 : i2, j1 : j2 ) as tile indices, inclusive
    auto B = A.sub( i1, i2, j1, j2 );

The `sub` method creates a view into the matrix using **block (tile) coordinates**.

- `i1, i2`: Start and end block row indices (inclusive).
- `j1, j2`: Start and end block column indices (inclusive).
- If `A` has tiles of size `nb`, `sub(1, 1, ...)` starts at global row `nb`.
- This operation is very fast and simply adjusts internal offsets and dimensions. It creates a shallow copy.

**Common `sub` Use Cases (Lines 43-73)**

- `B = A`: Assigning a matrix to another creates a shallow copy view of the entire matrix.
- `B = A.sub(0, mt-1, 0, nt-1)`: Explicitly selecting the whole matrix range.
- `B = A.sub(0, mt-1, 0, 0)`: Selecting the first **block column**.
- `B = A.sub(0, 0, 0, nt-1)`: Selecting the first **block row**.

**Element-based Slicing (Lines 77-80)**

.. code-block:: cpp

    // view of A( row1 : row2, col1 : col2 ), inclusive
    B = A.slice( row1, row2, col1, col2 );

The `slice` method creates a view using **global element indices** (0-based row/column indices).

- `row1, row2`: Start and end row indices (inclusive).
- `col1, col2`: Start and end column indices (inclusive).
- **Important**: Slicing allows for arbitrary boundaries. However, many SLATE algorithms require matrix views to be aligned with tile boundaries. If you slice in the middle of a tile, you may be restricted in which operations you can perform on that view.

**Common `slice` Use Cases (Lines 84-106)**

- `B = A.slice(0, m-1, 0, n-1)`: Slice of the entire matrix dimensions.
- `B = A.slice(0, m-1, 0, 0)`: Slice of the first column (single vector).
- `B = A.slice(0, 0, 0, n-1)`: Slice of the first row.

.. literalinclude:: ../../../examples/ex03_submatrix.cc
   :language: cpp
   :linenos:
