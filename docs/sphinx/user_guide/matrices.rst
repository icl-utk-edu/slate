Matrices in SLATE
=================

SLATE provides a rich hierarchy of matrix classes designed for efficient distributed-memory 
operations. This chapter covers matrix creation, manipulation, and the underlying design.


Matrix Hierarchy
----------------

SLATE matrices are organized in a class hierarchy that enables type-safe operations:

.. code-block:: text

    BaseMatrix (abstract)
    ├── Matrix                    (general m × n)
    └── BaseTrapezoidMatrix       (abstract, upper/lower storage)
        ├── TrapezoidMatrix       (m × n trapezoid)
        ├── TriangularMatrix      (n × n triangular)
        ├── SymmetricMatrix       (n × n symmetric)
        └── HermitianMatrix       (n × n Hermitian)
    
    BaseBandMatrix (abstract)
    ├── BandMatrix                (m × n band)
    └── BaseTriangularBandMatrix  (abstract)
        ├── TriangularBandMatrix  (n × n triangular band)
        ├── SymmetricBandMatrix   (n × n symmetric band)
        └── HermitianBandMatrix   (n × n Hermitian band)


Matrix Types
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Class
     - Description
   * - ``Matrix<T>``
     - General m × n matrix
   * - ``TrapezoidMatrix<T>``
     - Upper or lower trapezoid, m × n, with unit or non-unit diagonal
   * - ``TriangularMatrix<T>``
     - Upper or lower triangular, n × n
   * - ``SymmetricMatrix<T>``
     - Symmetric n × n, stored by upper or lower triangle
   * - ``HermitianMatrix<T>``
     - Hermitian n × n, stored by upper or lower triangle
   * - ``BandMatrix<T>``
     - General band m × n with bandwidths kl and ku
   * - ``TriangularBandMatrix<T>``
     - Triangular band n × n
   * - ``SymmetricBandMatrix<T>``
     - Symmetric band n × n
   * - ``HermitianBandMatrix<T>``
     - Hermitian band n × n


Creating Matrices
-----------------

Basic Creation
~~~~~~~~~~~~~~

Create an empty matrix with 2D block-cyclic distribution:

.. code-block:: cpp

    // Square tiles (nb × nb)
    slate::Matrix<double> A(m, n, nb, p, q, MPI_COMM_WORLD);
    
    // Rectangular tiles (mb × nb)
    slate::Matrix<double> B(m, n, mb, nb, p, q, MPI_COMM_WORLD);
    
    // Triangular matrix
    slate::TriangularMatrix<double> T(
        slate::Uplo::Lower, slate::Diag::NonUnit,
        n, nb, p, q, MPI_COMM_WORLD);
    
    // Hermitian matrix
    slate::HermitianMatrix<double> H(
        slate::Uplo::Lower, n, nb, p, q, MPI_COMM_WORLD);
    
    // Band matrix with kl sub-diagonals and ku super-diagonals
    slate::BandMatrix<double> Ab(m, n, kl, ku, nb, p, q, MPI_COMM_WORLD);


Creating from Existing Matrix Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

    // Create empty matrix with same structure
    auto A2 = A.emptyLike();


Allocating Tile Memory
----------------------

After creating a matrix, tiles must be allocated:

CPU Host Memory
~~~~~~~~~~~~~~~

.. code-block:: cpp

    // Automatic allocation
    A.insertLocalTiles(slate::Target::Host);
    
    // Manual allocation (equivalent)
    for (int64_t j = 0; j < A.nt(); ++j) {
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j)) {
                A.tileInsert(i, j, slate::HostNum);
            }
        }
    }


GPU Device Memory
~~~~~~~~~~~~~~~~~

.. code-block:: cpp

    // Allocate on GPU devices
    A.insertLocalTiles(slate::Target::Devices);


User-Provided Memory
~~~~~~~~~~~~~~~~~~~~

SLATE can wrap existing data arrays:

.. code-block:: cpp

    // Wrap user data
    for (int64_t j = 0; j < A.nt(); ++j) {
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal(i, j)) {
                A.tileInsert(i, j, user_data_ptr[i][j], lld);
            }
        }
    }


ScaLAPACK Layout
~~~~~~~~~~~~~~~~

SLATE natively supports ScaLAPACK's 2D block-cyclic layout:

.. code-block:: cpp

    // Create from ScaLAPACK-style data
    auto A = slate::Matrix<double>::fromScaLAPACK(
        m, n,                        // Global dimensions
        A_data,                      // Local data pointer
        lld,                         // Local leading dimension
        nb, nb,                      // Block size
        slate::GridOrder::Col,       // Column-major grid
        p, q,                        // Process grid
        MPI_COMM_WORLD);


Accessing Matrix Elements
-------------------------

Tile Access
~~~~~~~~~~~

Get a tile by block indices:

.. code-block:: cpp

    // Get tile at block position (i, j)
    slate::Tile<double> tile = A(i, j);
    
    // Check if tile is local
    if (A.tileIsLocal(i, j)) {
        // Get tile for writing
        A.tileGetForWriting(i, j, slate::HostNum, LayoutConvert::ColMajor);
        auto tile = A(i, j, slate::HostNum);
        // ...
    }


Element Access
~~~~~~~~~~~~~~

Access individual elements within a tile:

.. code-block:: cpp

    // Method 1: Using tile.at() - handles transposition
    tile.at(ii, jj) = value;
    
    // Method 2: Direct data access (more efficient in loops)
    double* data = tile.data();
    int64_t stride = tile.stride();
    int64_t mb = tile.mb();
    int64_t nb = tile.nb();
    
    for (int64_t jj = 0; jj < nb; ++jj) {
        for (int64_t ii = 0; ii < mb; ++ii) {
            data[ii + jj * stride] = value;
        }
    }


Matrix Properties
~~~~~~~~~~~~~~~~~

.. code-block:: cpp

    A.m()        // Number of rows
    A.n()        // Number of columns
    A.mt()       // Number of block rows
    A.nt()       // Number of block columns
    A.tileMb(i)  // Number of rows in block row i
    A.tileNb(j)  // Number of columns in block column j
    A.op()       // Transpose operation (NoTrans, Trans, ConjTrans)
    A.uplo()     // Upper/Lower storage (for triangular/symmetric)


Matrix Conversions
------------------

SLATE allows cheap "view" conversions between matrix types:

.. code-block:: cpp

    slate::Matrix<double> A(m, n, nb, p, q, MPI_COMM_WORLD);
    
    // View as lower trapezoid with unit diagonal
    auto Lz = slate::TrapezoidMatrix<double>(
        slate::Uplo::Lower, slate::Diag::Unit, A);
    
    // For triangular/symmetric/Hermitian, matrix must be square
    int64_t min_mn = std::min(m, n);
    auto A_square = A.slice(0, min_mn-1, 0, min_mn-1);
    
    // View as lower triangular
    auto L = slate::TriangularMatrix<double>(
        slate::Uplo::Lower, slate::Diag::Unit, A_square);
    
    // View as upper triangular
    auto U = slate::TriangularMatrix<double>(
        slate::Uplo::Upper, slate::Diag::NonUnit, A_square);
    
    // View as symmetric
    auto S = slate::SymmetricMatrix<double>(slate::Uplo::Upper, A_square);
    
    // View as Hermitian
    auto H = slate::HermitianMatrix<double>(slate::Uplo::Upper, A_square);

These conversions are shallow copies using C++ shared pointers—very efficient with no data copying.


Transpose and Conjugate Transpose
---------------------------------

SLATE supports matrix transposition as a view operation:

.. code-block:: cpp

    slate::Matrix<double> A(m, n, nb, p, q, MPI_COMM_WORLD);
    
    // Transpose view
    auto AT = slate::transpose(A);
    // AT.op() == slate::Op::Trans
    // AT(i, j) == transpose(A(j, i))
    
    // Conjugate transpose view
    auto AH = slate::conj_transpose(A);
    // AH.op() == slate::Op::ConjTrans
    // AH(i, j) == conj_transpose(A(j, i))


Submatrices and Slices
----------------------

Tile-Based Submatrices
~~~~~~~~~~~~~~~~~~~~~~

Create views based on tile indices:

.. code-block:: cpp

    // Submatrix A[i1:i2, j1:j2] (tile indices, inclusive)
    auto B = A.sub(i1, i2, j1, j2);
    
    // View of all of A
    auto B = A.sub(0, A.mt()-1, 0, A.nt()-1);
    
    // First block column
    auto B = A.sub(0, A.mt()-1, 0, 0);
    
    // First block row
    auto B = A.sub(0, 0, 0, A.nt()-1);


Element-Based Slices
~~~~~~~~~~~~~~~~~~~~

Create views based on row/column indices:

.. code-block:: cpp

    // Slice A[row1:row2, col1:col2] (element indices, inclusive)
    auto B = A.slice(row1, row2, col1, col2);
    
    // First column
    auto B = A.slice(0, A.m()-1, 0, 0);
    
    // First row
    auto B = A.slice(0, 0, 0, A.n()-1);

.. warning::

    Slice operations create matrices with non-uniform tile sizes, which may have limited 
    algorithm support, especially on GPUs.


Copying Matrices
----------------

Deep Copy
~~~~~~~~~

Create a full copy of matrix data:

.. code-block:: cpp

    auto A2 = A.emptyLike();
    A2.insertLocalTiles();
    slate::copy(A, A2);


Precision Conversion
~~~~~~~~~~~~~~~~~~~~

Copy with automatic precision conversion:

.. code-block:: cpp

    // Double to single precision
    slate::Matrix<double> A_hi(m, n, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<float> A_lo(m, n, nb, p, q, MPI_COMM_WORLD);
    A_hi.insertLocalTiles();
    A_lo.insertLocalTiles();
    
    // Copy with conversion
    slate::copy(A_hi, A_lo);  // double -> float
    slate::copy(A_lo, A_hi);  // float -> double


Tile Layout and Distribution
----------------------------

Data Distribution
~~~~~~~~~~~~~~~~~

By default, SLATE uses 2D block-cyclic distribution identical to ScaLAPACK. The mapping of tiles 
to processes can be customized using lambda functions.

.. code-block:: cpp

    // Default: 2D block-cyclic
    // Tile (i, j) owned by process (i % p, j % q)


Tile Sizes
~~~~~~~~~~

SLATE supports variable tile sizes, though most algorithms require square diagonal tiles:

- Off-diagonal tiles can have arbitrary dimensions
- Non-uniform sizes complicate batched BLAS operations
- GPU execution currently requires fixed tile sizes

.. note::

    Non-uniform tile sizes are useful for applications like Adaptive Cross Approximation (ACA) 
    where block structure is significant.


Memory Management
~~~~~~~~~~~~~~~~~

SLATE manages tile memory with reference counting:

- Matrix copies share tile data via shared pointers
- Tiles can exist on multiple devices (CPU and GPUs)
- SLATE handles data transfers between devices

.. code-block:: cpp

    // Move tile to GPU device
    A.tileGetForWriting(i, j, device_id, LayoutConvert::ColMajor);
    
    // Ensure tile is on CPU
    A.tileGetForWriting(i, j, slate::HostNum, LayoutConvert::ColMajor);


Best Practices
--------------

1. **Choose appropriate tile size**: 256-512 for CPU, 512-1024 for GPU
2. **Use near-square process grids**: Avoid 1D grids (p=1 or q=1)
3. **Prefer submatrices over slices**: Better algorithm support
4. **Reuse matrix structures**: Use ``emptyLike()`` for temporary matrices
5. **Consider memory layout**: ScaLAPACK layout for interoperability
