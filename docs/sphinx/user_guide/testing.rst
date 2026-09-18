Testing and Tuning
==================

SLATE includes a comprehensive testing suite for verifying correctness and measuring performance. 
This chapter covers the tester, performance tuning, and unit tests.


SLATE Tester
------------

The SLATE tester is built in the ``test/`` directory and exercises all library functionality.

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

    cd test
    
    # List available tests
    ./tester --help
    
    # Quick test of gemm with small defaults
    ./tester gemm
    
    # List options for a specific routine
    ./tester --help gemm


Single-Process Testing
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Sweep over matrix dimensions
    ./tester --nb 256 --dim 1000:5000:1000 gemm
    
    # Multiple data types
    ./tester --type s,d,c,z gemm
    
    # Different execution targets
    ./tester --target t gemm   # HostTask (CPU)
    ./tester --target d gemm   # Devices (GPU)


Multi-Process Testing
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Using mpirun
    mpirun -n 4 ./tester --nb 256 --dim 1000:5000:1000 --grid 2x2 gemm
    
    # Using Slurm
    srun --nodes=4 --ntasks=16 --cpus-per-task=8 ./tester --grid 4x4 gemm


Tester Parameters
~~~~~~~~~~~~~~~~~

Common parameters for the tester:

.. code-block:: text

    --check          check results (default: y)
    --ref            run ScaLAPACK reference (default: n)
    --tol            error tolerance (default: 50)
    --repeat         repetitions per test (default: 1)
    --verbose        output verbosity level (0-4)
    
    --type           data type: s, d, c, z (default: d)
    --origin         data origin: h=Host, s=ScaLAPACK, d=Devices
    --target         execution target: t=HostTask, n=HostNest, 
                     b=HostBatch, d=Devices (default: t)
    
    --transA/--transB   transpose: n, t, c (default: n)
    --uplo           upper/lower: u, l (default: l)
    --diag           diagonal: u=unit, n=non-unit (default: n)
    
    --dim            m x n x k dimensions
    --nb             tile size (default: 384)
    --grid           p x q MPI grid
    --lookahead      lookahead panels (default: 1)


Example Tester Output
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    % SLATE version 2023.08.25
    % input: ./tester gemm
    % MPI:   4 ranks, CPU-only MPI, 8 OpenMP threads per MPI rank
    
    gemm dtype=d, origin=h, target=t, transA=n, transB=n, m=100, n=100, k=100,
        alpha=1+0i, beta=1+0i, nb=100, grid=2x2, la=1
        time=0.00123, gflop/s=1.63, ref_time=0.00089, ref_gflop/s=2.24,
        error=1.2e-15, okay


Accuracy Verification
---------------------

SLATE uses backward error analysis for verification. Most routines check accuracy in two ways:

Without Reference (--ref=n)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fast check using algebraic identities. For gemm :math:`C = \alpha AB + \beta C`:

.. math::

    Y_1 = \alpha A(BX) + (\beta C_{in} X)
    
    Y_2 = C_{out} X
    
    \text{error} = \frac{\|Y_1 - Y_2\|}{\|Y_1\|}

With Reference (--ref=y)
~~~~~~~~~~~~~~~~~~~~~~~~

Compare against ScaLAPACK reference implementation. Slower but more robust.

.. code-block:: bash

    ./tester --ref=y gemm


Norm Verification
~~~~~~~~~~~~~~~~~

Matrix norms compare against ScaLAPACK. Note that older ScaLAPACK versions have accuracy 
issues in norm computation.


Full Testing Suite
------------------

The ``run_tests.py`` script runs comprehensive tests:

.. code-block:: bash

    cd test
    
    # View options
    python3 ./run_tests.py --help
    
    # Default full test suite
    python3 ./run_tests.py --xml ../report.xml
    
    # Small quick tests
    python3 ./run_tests.py --xsmall
    
    # Specific routines
    python3 ./run_tests.py --xsmall gesv potrf


Custom Test Commands
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Using Slurm
    python3 ./run_tests.py \
        --test "salloc -N 4 -n 4 -t 10 mpirun -n 4 ./tester" \
        --xsmall gesv
    
    # GPU execution
    python3 ./run_tests.py \
        --test "mpirun -n 4 ./tester" \
        --xsmall --target d gesv


Performance Tuning
------------------

Tile Size
~~~~~~~~~

Tile size (``nb``) significantly affects performance:

.. code-block:: bash

    # Sweep tile sizes for CPU
    ./tester --type d --target t --dim 3000 --nb 128:512:32 gesv
    
    # Sweep tile sizes for GPU (typically larger, multiple of 64)
    ./tester --type d --target d --dim 10000 --nb 192:1024:64 gesv

General guidelines:

- **CPU**: 128-512, depending on cache sizes
- **NVIDIA GPU**: 384-1024, multiples of 64
- **AMD GPU**: 256-768, multiples of 64


Process Grid
~~~~~~~~~~~~

Near-square grids usually provide best performance:

.. code-block:: bash

    # Test different grids
    mpirun -n 4 ./tester --nb 256 --dim 10000 --grid 1x4,2x2 gemm
    
    # 1D grids (1xq or px1) are typically slower

Avoid 1D grids as they lead to higher communication overhead.


Lookahead
~~~~~~~~~

Lookahead depth for overlapping communication and computation:

.. code-block:: bash

    ./tester --dim 10000 --lookahead 1,2,4 gesv

Default of 1 is usually sufficient. Higher values require more memory.


Panel Threads
~~~~~~~~~~~~~

Control threads used in panel operations:

.. code-block:: bash

    export OMP_NUM_THREADS=32
    # Panel uses min(OMP_NUM_THREADS/2, MaxPanelThreads)


Multi-threaded MPI Broadcast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For some systems, multi-threaded MPI broadcast improves performance. Enable by building with:

.. code-block:: make

    # In make.inc
    CXXFLAGS += -DSLATE_HAVE_MT_BCAST

.. warning::

    On some systems (e.g., Frontier with GPU-aware MPI), this can cause hangs. Test carefully.


GPU-Aware MPI
~~~~~~~~~~~~~

Enable for direct GPU-to-GPU transfers:

.. code-block:: bash

    export SLATE_GPU_AWARE_MPI=1
    
    # For Cray MPI
    export MPICH_GPU_SUPPORT_ENABLED=1


Performance Examples
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Tune tile size for double precision gemm on GPU
    mpirun -n 4 ./tester --type d --target d --dim 10000 \
           --nb 256,384,512,640,768 gemm
    
    # Compare process grids for Cholesky
    mpirun -n 16 ./tester --type d --target d --dim 20000 --nb 512 \
           --grid 1x16,2x8,4x4 potrf
    
    # Test lookahead for LU
    mpirun -n 16 ./tester --target d --dim 20000 --nb 512 \
           --grid 4x4 --lookahead 1,2,4 getrf


Unit Tests
----------

Unit tests verify individual SLATE components (matrix classes, memory manager, tiles):

.. code-block:: bash

    cd unit_test
    
    # Run default unit tests
    python3 ./run_tests.py --xml ../report_unit.xml
    
    # Specific test
    ./unit_test Matrix


Benchmark Suite
---------------

For production benchmarking:

.. code-block:: bash

    # Disable result checking for timing
    ./tester --check=n --repeat=3 gemm
    
    # Compare with ScaLAPACK
    ./tester --ref=y --repeat=3 gemm


Troubleshooting Tests
---------------------

Common Issues
~~~~~~~~~~~~~

**Test failures with old ScaLAPACK**

Older ScaLAPACK has accuracy bugs in norms. Update or use ``--ref=n``.

**GPU tests hang**

Try disabling GPU-aware MPI:

.. code-block:: bash

    unset SLATE_GPU_AWARE_MPI

**Memory errors**

Reduce problem size or number of OpenMP threads:

.. code-block:: bash

    export OMP_NUM_THREADS=4
    ./tester --dim 1000 gesv


Debugging
~~~~~~~~~

.. code-block:: bash

    # Verbose output
    ./tester --verbose=2 gemm
    
    # Print matrices (small problems only!)
    ./tester --verbose=4 --dim 10 gemm
    
    # Debug with specific rank
    ./tester --debug=0 gemm  # rank 0 waits for debugger
