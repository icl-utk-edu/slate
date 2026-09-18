Example 01: Matrix Construction
===============================

This example demonstrates the fundamental operations for creating and managing SLATE matrices.

Key Concepts
------------

1.  **Constructors**: Creating empty matrices with specific dimensions and distribution.
2.  **Tile Insertion**: Allocating memory for tiles on Host (CPU) or Devices (GPU).
3.  **User Data**: Wrapping existing data pointers (e.g., from ScaLAPACK) into a SLATE matrix.
4.  **Transposition**: Creating transposed views of matrices without copying data.
5.  **Element Access**: Iterating over tiles and elements efficiently.

C++ Example
-----------

**Matrix Construction (Lines 18-44)**

.. code-block:: cpp

    // Create an empty matrix (2D block cyclic layout, p x q grid,
    // no tiles allocated, square nb x nb tiles)
    slate::Matrix<scalar_type>
        A( m, n, nb, grid_p, grid_q, MPI_COMM_WORLD );

Here, we construct a standard general matrix `A`. Note that:

- `m` and `n` are the global dimensions (rows, columns).
- `nb` is the block size (tile size).
- `grid_p` and `grid_q` define the MPI process grid dimensions.
- `MPI_COMM_WORLD` is the communicator.
- **Important**: This constructor defines the matrix *structure* and *distribution*, but does **not** allocate memory for the matrix elements (tiles). The matrix is initially empty.

We can also create matrices with rectangular tiles (specifying `mb` and `nb`) or triangular matrices (`slate::TriangularMatrix`). `A.emptyLike()` (line 42) is a convenient way to create a new matrix `A2` with the same structure and distribution as `A` but without any data allocated.

**Allocating Memory (Lines 47-68)**

.. code-block:: cpp

    // Insert tiles on the CPU host.
    A.insertLocalTiles( slate::Target::Host );

Since the constructor creates an empty shell, we must explicitly allocate memory for the tiles that belong to this MPI rank. `insertLocalTiles` is the standard way to do this.

- `Target::Host` allocates memory on the CPU.
- `Target::Devices` (seen in lines 71-92) allocates memory on GPU devices if available.

The loop shown in lines 63-66 demonstrates what `insertLocalTiles` effectively does under the hood: it iterates over the matrix structure, checks if a tile is local to the current rank, and if so, inserts (allocates) it.

**Wrapping Existing Data (Lines 97-132)**

.. code-block:: cpp

    // Attach user allocated tiles, from pointers in data( i, j )
    // with local stride lld between columns.
    for (int64_t j = 0; j < A.nt(); ++j) {
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal( i, j ))
                A.tileInsert( i, j, data( i, j ), lld );
        }
    }

If you already have data allocated (e.g., from a legacy application), you can wrap it into a SLATE matrix without copying. The `data` lambda in this example simulates calculating the pointer to the start of each tile in a ScaLAPACK-style array. `tileInsert` is then called with the pointer and leading dimension (`lld`).

**ScaLAPACK Conversion (Lines 135-162)**

.. code-block:: cpp

    auto A = slate::Matrix<scalar_type>::fromScaLAPACK( ... );

This is the simplified, preferred way to wrap legacy ScaLAPACK data. It handles all the index calculations and tile insertions automatically.

**Transposition (Lines 165-185)**

.. code-block:: cpp

    auto AT = transpose( A );
    auto AH = conj_transpose( A );

SLATE operations like transposition are typically metadata operations. `AT` is a **view** of `A`. No data is copied or moved.

- `AT` has its operation flag set to `Op::Trans`.
- Accessing `AT(i, j)` effectively accesses `A(j, i)` with transposition applied on the fly.
- This allows efficient passing of transposed arguments to BLAS routines (like `gemm`) without overhead.

**Element Access (Lines 189-254)**

.. code-block:: cpp

    if (A.tileIsLocal( i, j )) {
        // ...
        slate::Tile<scalar_type> T = A( i, j, slate::HostNum );
        for (int64_t jj = 0; jj < T.nb(); ++jj) {
            for (int64_t ii = 0; ii < T.mb(); ++ii) {
                T.at( ii, jj ) = ...
            }
        }
    }

Direct element access in distributed memory requires care:

1.  We iterate over global tiles `(i, j)`.
2.  We check `A.tileIsLocal(i, j)` to ensure the current rank owns the data.
3.  We acquire the tile `T` on the host.
4.  We iterate within the tile using local indices `ii`, `jj`.
5.  `T.at(ii, jj)` provides safe access (with bounds checking in debug mode), while `T.data()` (shown in `elements2`) provides raw pointer access for maximum performance in inner loops.

.. literalinclude:: ../../../examples/ex01_matrix.cc
   :language: cpp
   :linenos:
