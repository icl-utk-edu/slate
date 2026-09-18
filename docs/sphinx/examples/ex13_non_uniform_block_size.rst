Example 13: Non-uniform Block Sizes
===================================

This example demonstrates creating a matrix with non-uniform tile sizes.

Key Concepts
------------

1.  **Lambda Constructors**: Using lambda functions to define tile properties (`tileNb`, `tileRank`, `tileDevice`) instead of fixed values.
2.  **Custom Block Sizes**: Defining a function that returns the block size for a given block index :math:`j`.
3.  **Process/Device Mapping**: Using helper functions like `slate::func::process_2d_grid` and `slate::func::device_1d_grid` to define distribution.

C++ Example
-----------



**Custom Block Size Function (Lines 30-34)**

.. code-block:: cpp

    std::function< int64_t (int64_t j) >
    tileNb = [n, nb_](int64_t j)
    {
        return (j % 2 != 0 ? nb_/2 : nb_);
    };

Instead of passing a constant `nb`, we define a lambda function `tileNb`.
- Input: Block index `j` (0, 1, 2, ...).
- Output: The number of columns in block column `j`.
- This example alternates block sizes between `nb` and `nb/2`.

**Distribution Functions (Lines 36-38)**

.. code-block:: cpp

    auto tileRank = slate::func::process_2d_grid( slate::GridOrder::Col, p_, q_ );
    auto tileDevice = slate::func::device_1d_grid( slate::GridOrder::Col, p_, num_devices_ );

We can also customize how tiles map to MPI ranks and devices. Here we use standard helper functions from `slate::func`, but you could write your own lambdas (e.g., to force a specific block cyclic pattern or a custom distribution).

**Matrix Construction (Line 40)**

.. code-block:: cpp

    slate::Matrix<scalar_type> A( n, n, tileNb, tileNb, tileRank, tileDevice, MPI_COMM_WORLD );

The constructor takes these functions as arguments.
- `tileNb` is passed twice: once for row heights `tileMb` and once for column widths `tileNb`. This creates a square-tiled matrix (though the tiles themselves can vary in size).
- `tileRank` determines the MPI rank for tile `(i, j)`.
- `tileDevice` determines the GPU device ID for tile `(i, j)`.

**Verification (Lines 53-59)**

The code iterates through the matrix to verify that the block sizes match the logic defined in the lambda. Note that `A.tileNb(j)` handles the boundary condition at the end of the matrix automatically (clamping to `n`).

.. literalinclude:: ../../../examples/ex13_non_uniform_block_size.cc
   :language: cpp
   :linenos:
