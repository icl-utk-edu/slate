Parallelism Model
=================

SLATE utilizes three or four levels of parallelism: distributed parallelism between nodes using MPI, explicit thread parallelism using OpenMP, implicit thread parallelism within the vendor's node-level BLAS, and, at the lowest level, vector parallelism for the processor's single instruction, multiple data (SIMD) vector instructions. For multicore, SLATE typically uses all the threads explicitly, and uses the vendor's BLAS in sequential mode. For GPU accelerators, SLATE uses a batch BLAS call, utilizing the thread block parallelism built into the accelerator's BLAS.

The cornerstones of SLATE are:

1.  The SPMD programming model for productivity and maintainability.
2.  Dynamic task scheduling using OpenMP for maximum node-level parallelism and portability.
3.  The lookahead technique for prioritizing the critical path.
4.  Primary reliance on the 2D block cyclic distribution for scalability.
5.  Reliance on the ``gemm`` operation, specifically its batch rendition, for maximum hardware utilization.

Dataflow tasking (``omp task depend``) is used for scheduling operations with dependencies on large blocks of the matrix. Dependencies are performed on a dummy vector, representing each block column in the factorization, rather than on the matrix data itself. Within each large block, either nested tasking (``omp task``) or batch operations of independent tile operations are used for scheduling individual tile operations to individual cores, without dependencies. For accelerators, batched BLAS calls are used for fast processing of large blocks of the matrix using accelerators.

Compared to pure tile-by-tile dataflow scheduling, as used by DPLASMA and Chameleon, this approach minimizes the size of the task graph and number of dependencies to track. For a matrix of :math:`N \times N` tiles, tile-by-tile scheduling creates :math:`O(N^3)` tasks and dependencies, which can lead to significant scheduling overheads. In contrast, the SLATE approach creates :math:`O(N)` dependencies, eliminating the issue of scheduling overheads. At the same time, this approach is a necessity for scheduling a large set of independent tasks to accelerators, in order to fully occupy their massive compute resources.

At each step of Cholesky, one or more columns of the trailing submatrix are prioritized for processing, using the OpenMP priority clause, to facilitate faster advancement along the critical path, implementing a lookahead. At the same time, the lookahead depth needs to be limited, as it is proportional to the amount of extra memory required for storing temporary tiles. Deep lookahead translates to depth-first processing of the task graph, synonymous with left-looking algorithms, but can also lead to catastrophic memory overheads in distributed-memory environments.

Distributed-memory computing is implemented by filtering operations based on the matrix distribution function; in most cases, the owner of the output tile performs the computation to update the tile. Appropriate communication calls are issued to send tiles to where the computation will occur. Management of multiple accelerators is handled by a node-level memory consistency protocol.

The user can choose among various target implementations. In the case of accelerated execution, the updates are executed as calls to batched gemm (``Target::Devices``). In the case of multi-core execution, the updates can be executed as:

*   A set of OpenMP tasks (``Target::HostTask``).
*   A nested parallel for loop (``Target::HostNest``).
*   A call to batch gemm (``Target::HostBatch``).

To motivate our choices of CPU tasks on individual tiles and GPU tasks using batches of tiles, we examine the performance of ``dgemm``. Libraries such as DPLASMA and Chameleon have demonstrated that doing operations on a tile-by-tile basis can achieve excellent CPU performance. In contrast, accelerators would take much larger tiles to reach their maximum performance.

In SLATE, we observe that most ``gemm`` operations are block outer products, where :math:`A` is a block column and :math:`B` is a block row (e.g., the Schur complement in LU factorization), and that these can be implemented using a batch ``gemm``. This demonstrates that at specific sizes, the batched ``dgemm`` matches the performance of a regular ``dgemm``. Thus, with an appropriately chosen, modest block size, SLATE can achieve the maximum performance from accelerators.

SLATE intentionally relies on standards in MPI, OpenMP, and BLAS to maintain easy portability. Any CPU platform with good implementations of these standards should work well for SLATE. For accelerators, any platform that implements batched ``gemm``, on which SLATE relies, is a good target. Differences between vendors' BLAS implementations will be abstracted at a low level in the BLAS++ library to ease porting. There are very few accelerator (e.g., CUDA) kernels in SLATE—currently just matrix norms and transposition—so porting should be a lightweight task.
