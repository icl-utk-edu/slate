Getting Started
===============

This chapter provides a quick introduction to SLATE through a complete example program that 
solves a linear system :math:`AX = B` using LU factorization.


Example Program: LU Solve
-------------------------

The following example demonstrates how to set up SLATE matrices and solve a linear system 
using the distributed ``lu_solve`` implementation (also known as ``gesv`` in traditional 
LAPACK naming).

.. code-block:: cpp

    #include <slate/slate.hh>
    #include <blas.hh>
    #include <mpi.h>
    #include <stdio.h>

    int main(int argc, char** argv)
    {
        // Initialize MPI with thread support
        int mpi_provided = 0;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_provided);
        if (mpi_provided != MPI_THREAD_MULTIPLE) {
            throw std::runtime_error("MPI_THREAD_MULTIPLE required");
        }

        int mpi_rank, mpi_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

        // Problem parameters
        int64_t n = 5000;      // Matrix dimension
        int64_t nrhs = 1;      // Number of right-hand sides
        int64_t nb = 256;      // Tile size
        int p = 2, q = 2;      // Process grid dimensions

        // Verify we have enough MPI processes
        if (mpi_size < p * q) {
            if (mpi_rank == 0) {
                printf("Need at least %d MPI processes\n", p * q);
            }
            MPI_Finalize();
            return 1;
        }

        // Create SLATE matrices
        slate::Matrix<double> A(n, n, nb, p, q, MPI_COMM_WORLD);
        slate::Matrix<double> B(n, nrhs, nb, p, q, MPI_COMM_WORLD);

        // Allocate local tiles on each process
        A.insertLocalTiles();
        B.insertLocalTiles();

        // Initialize with random data (each rank different seed)
        srand(100 * mpi_rank);
        // ... initialize A and B tiles ...

        // Solve AX = B using LU factorization
        // Solution X overwrites B
        slate::lu_solve(A, B);

        MPI_Finalize();
        return 0;
    }


Understanding the Example
-------------------------

MPI Initialization
~~~~~~~~~~~~~~~~~~

SLATE requires MPI to be initialized with ``MPI_THREAD_MULTIPLE`` support, as SLATE uses 
OpenMP threads internally that may make MPI calls:

.. code-block:: cpp

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_provided);


Creating Matrices
~~~~~~~~~~~~~~~~~

SLATE matrices are defined with:

- **Dimensions**: ``n × n`` for matrix A, ``n × nrhs`` for matrix B
- **Tile size**: ``nb × nb`` blocks (256 is a reasonable default)
- **Process grid**: ``p × q`` grid of MPI processes
- **Communicator**: MPI communicator for collective operations

.. code-block:: cpp

    slate::Matrix<double> A(n, n, nb, p, q, MPI_COMM_WORLD);

The matrices are distributed in a 2D block-cyclic pattern across the process grid, similar 
to ScaLAPACK.


Allocating Tiles
~~~~~~~~~~~~~~~~

After creating the matrix structure, local tiles must be allocated:

.. code-block:: cpp

    A.insertLocalTiles();

This allocates memory for tiles that belong to the current process based on the 2D block-cyclic 
distribution. SLATE can also work with user-provided memory or ScaLAPACK-style layouts.


Solving the System
~~~~~~~~~~~~~~~~~~

The ``lu_solve`` function factors A and solves the system in one call:

.. code-block:: cpp

    slate::lu_solve(A, B);

Internally, this performs:

1. LU factorization: :math:`PA = LU` with partial pivoting
2. Forward substitution: Solve :math:`LY = PB`
3. Back substitution: Solve :math:`UX = Y`

The solution X overwrites B.


Execution Options
~~~~~~~~~~~~~~~~~

SLATE routines accept optional parameters to control execution:

.. code-block:: cpp

    slate::Options opts = {
        {slate::Option::Target, slate::Target::HostTask},
        {slate::Option::Lookahead, 2}
    };
    slate::lu_solve(A, B, opts);

Common options include:

- ``Target``: Execution target (``HostTask``, ``Devices``, etc.)
- ``Lookahead``: Depth of lookahead for overlapping computation and communication
- ``InnerBlocking``: Inner blocking size for panel operations


Building the Example
--------------------

Compile with the MPI C++ wrapper and link against SLATE libraries:

.. code-block:: bash

    # Set paths
    export SLATE_ROOT=/path/to/slate
    export BLASPP_ROOT=${SLATE_ROOT}/blaspp
    export LAPACKPP_ROOT=${SLATE_ROOT}/lapackpp

    # Compile
    mpicxx -fopenmp -c example.cc \
        -I${SLATE_ROOT}/include \
        -I${BLASPP_ROOT}/include \
        -I${LAPACKPP_ROOT}/include

    # Link
    mpicxx -fopenmp -o example example.o \
        -L${SLATE_ROOT}/lib -Wl,-rpath,${SLATE_ROOT}/lib \
        -L${BLASPP_ROOT}/lib -Wl,-rpath,${BLASPP_ROOT}/lib \
        -L${LAPACKPP_ROOT}/lib -Wl,-rpath,${LAPACKPP_ROOT}/lib \
        -lslate -llapackpp -lblaspp

For CUDA support, add:

.. code-block:: bash

    -L${CUDA_HOME}/lib64 -Wl,-rpath,${CUDA_HOME}/lib64 \
    -lcusolver -lcublas -lcudart


Running the Example
-------------------

Run with MPI:

.. code-block:: bash

    export OMP_NUM_THREADS=8
    mpirun -n 4 ./example

Expected output:

.. code-block:: text

    lu_solve n 5000, nb 256, p-by-q 2-by-2, residual 8.41e-20, tol 2.22e-16, 
    time 7.65e-01 sec, pass


Simplifying Assumptions
-----------------------

The example uses several defaults that may need tuning for optimal performance:

**Tile Size (nb=256)**
    Should be tuned for the target architecture. Larger tiles (512-1024) often work better 
    for GPUs.

**Process Grid (p=2, q=2)**
    Square or near-square grids typically provide best performance. Avoid 1D grids (p=1 or q=1).

**Data Distribution**
    Default is 2D block-cyclic. Custom distributions can be specified using lambda functions.

**Execution Target**
    Default is ``HostTask`` (CPU with OpenMP tasks). Set to ``Devices`` for GPU acceleration.


Next Steps
----------

- :doc:`installation`: Detailed installation instructions
- :doc:`matrices`: Understanding SLATE matrix types and operations
- :doc:`operations`: Guide to all SLATE operations
- :doc:`../api/slate/index`: Complete API reference
