Matrix Storage
==============

SLATE makes tiles first-class objects that can be individually allocated and passed to low-level tile routines. The matrix consists of a collection of individual tiles, with no correlation between their positions in the matrix and their memory locations. At the same time, SLATE also supports tiles pointing to data in a traditional ScaLAPACK matrix storage, easing an application's transition from ScaLAPACK to SLATE. Compared to other distributed dense linear algebra formats, SLATE's matrix structure offers numerous advantages.

First, the same structure can be used for holding many different matrix types: general, symmetric, triangular, band, symmetric band, etc. Little memory is wasted for storing parts of the matrix that hold no useful data (e.g., the upper triangle of a lower triangular matrix). Instead of wasting :math:`O(n^2)` memory as ScaLAPACK does, only :math:`O(n \cdot nb)` memory is wasted in the diagonal tiles for a block size :math:`nb`; all unused off-diagonal tiles are simply never allocated. There is no need for using complex matrix storage schemes such as the Recursive Packed Format (RPF) or Rectangular Full Packed (RFP) in order to save space.

Second, the matrix can be easily converted, in parallel, from one layout to another with :math:`O(P)` memory overhead for :math:`P` processors (cores/threads). Possible conversions include: changing tile layout from column-major to row-major, "packing" of tiles for efficient BLAS execution, and low-rank compression of tiles. Notably, transposition of the matrix can be accomplished by transposition of each tile and remapping of the indices. There is no need for complex in-place layout translation and transposition algorithms.

Also, tiles can be easily allocated and copied among different memory spaces. Both inter-node communication and intra-node communication are vastly simplified. Tiles can be easily and efficiently transferred between nodes using MPI. Tiles can be easily moved in and out of fast memory, such as the MCDRAM in Xeon Phi processors. Tiles can also be copied to one or more device memories in the case of GPU acceleration.

In practical terms, a SLATE matrix is implemented using the ``std::map`` container from the C++ standard library as:

.. code-block:: cpp

   std::map< std::tuple< int64_t, int64_t >,
             TileNode< scalar_t >* >

The map's key is a tuple consisting of the tile's :math:`(i, j)` block row and column indices in the matrix. The ``TileNode`` can then be indexed by the host or accelerator device ID to retrieve the corresponding ``Tile`` instance. SLATE relies on global indexing of tiles, meaning that each tile is identified by the same unique tuple across all processes. The lightweight ``Tile`` object stores a tile's data and properties such as MOSI state, dimensions, uplo, and transposition operation.

Note that to prevent SLATE from getting into an invalid state, the canonical ``Tile`` instances held by the ``TileNode``'s should never be directly accessible to users. Instead, modifications to a tile's properties should be done through the matrix object, and only copies should be returned.

In addition to facilitating the storage of different types of matrices, this structure also readily accommodates partitioning of the matrix to the nodes of a distributed-memory system. Each node stores only its local subset of tiles. Mapping of tiles to nodes is defined by a C++ lambda function, and set to 2D block cyclic mapping by default, but the user can supply an arbitrary mapping function. Similarly, distribution to accelerators within each node is 1D block cyclic by default, but the user can substitute an arbitrary function.

Remote access is realized by replicating remote tiles in the local matrix for the duration of the operation.

Finally, SLATE can support non-uniform tile sizes. Most factorizations require that the diagonal tiles are square, but the block row heights and block column widths can, in principle, be arbitrary. This will facilitate applications where the block structure is significant, for instance in Adaptive Cross Approximation (ACA) linear solvers.

Tile Management
---------------

A ``Tile`` can be one of three types, as denoted by the enum ``TileKind``:

.. code-block:: cpp

   enum class TileKind
   {
       Workspace,
       SlateOwned,
       UserOwned,
   };

UserOwned
    User allocated origin tile. This is the original instance of a tile initialized upon matrix creation. The tile's memory is managed by the user, not by SLATE. The tile has been initialized with a pre-existing data buffer. The tile's memory should not be freed by SLATE.

SlateOwned
    SLATE-allocated origin tile. This is the original instance of the tile received upon matrix creation or by ``tileInsert()``. The tile's memory is managed by SLATE, and is freed when the matrix is destructed.

Workspace
    SLATE-allocated workspace tile. This is an instance of the tile that is used as temporary workspace in a memory space different from that of the corresponding origin tile. The tile is created with ``tileInsertWorkspace()`` for receiving a remote tile copy or for computation on a different device (CPU or accelerator) than the origin. It should be released back to the matrix's memory pool after being used.

It is important to note that at most one instance of a tile per memory space (i.e., per CPU or accelerator device) is allowed.

An operation computing on a device needs to create copies of the involved tiles on the device as workspace tiles and purge them after usage in order to minimize memory consumption. On the other hand, certain algorithms may need to hold a set of tiles on the device for the duration of the algorithm to allow multiple accesses to these tiles and minimize the data traffic from/to host memory to/from device memory. These requirements necessitate the adoption of a coherency protocol that seamlessly manages the tile copies on various memory spaces, as described in :doc:`08_mosi`.
