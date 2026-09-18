Column Major and Row Major Layout
=================================

A tile's data can be stored in either column-major or row-major layout. In column-major layout, elements of a column have a memory stride of 1—that is, they are stored contiguously in memory, and elements of a row have a memory stride of at least the number of rows in the tile. In row-major layout, elements of a row have a memory stride of 1—that is, stored contiguously in memory, and elements of a column have a memory stride of at least the number of columns in the tile. Another representation where both the row and column strides are greater than one is possible; however, this later representation is not yet considered in SLATE, and is incompatible with the traditional BLAS.

SLATE supports converting tiles' layout for performance considerations. Layout conversion is mainly motivated by the fact that some algorithms perform much faster when access to a tile's element is contiguous in a row-major layout, or a column-major layout. The following sections explain the API and mechanisms used to establish layout conversion, especially tiles that cannot be transposed in-place.

Layout Representation and API
-----------------------------

The column-major or row-major layout (referred to as layout herein) is defined by the enum:

.. code-block:: cpp

   enum class Layout : char
   {
       ColMajor = 'C',
       RowMajor = 'R',
   };

The tile's layout is stored at the tile instance (indicating the Col/Row major storage of a tile's data) in the ``Tile::layout_`` member variable. Similarly, the matrix layout (defaulting to ``ColMajor``) is stored at the ``BaseMatrix::layout_`` member variable.

A MOSI operation (``tileGetForReading()``, ``tileGetForWriting()``, etc...) specifies the layout of the destination tile instance using the following enum:

.. code-block:: cpp

   enum class LayoutConvert : char
   {
       ColMajor = 'C',
       RowMajor = 'R',
       None     = 'N',
   };

Layout Conversion
-----------------

To foster high performance, algorithms in SLATE should operate in their preferred layout. For example, in LU factorization, row swapping during pivoting performs much better on devices when the tiles are in row-major. However, the panel factorization in the LU factorization prefers the col-major layout. As such, a runtime conversion between row-major and col-major layout is needed at the start of any computational or internal routine to ensure the tiles are in the needed layout. Obviously, the computational routine must reset the tiles layout when computations are done to the matrix original layout.

Layout conversion is implicitly handled at the MOSI calls by supplying the intended layout to the ``tileGet***()`` routines. As such, each computational routine sets a local variable indicating its preferred tile layout for computations, and passes this to any subroutine call. In turn, some internal routines can operate in both row-major or col-major tile layout, and receive a parameter to determine which layout to use, for example, ``internal::gemm``. However, other internal routines can operate only in one of the col-major or row-major layouts, and enforce it through the ``tileGet***()`` call. It is a general and preferred practice in SLATE to fetch the set of tiles to operate on at the beginning of each internal routine using the ``tileGet***()`` calls, which receive a parameter instructing it to convert the tiles to one of the layouts (``LayoutConvert::ColMajor`` or ``LayoutConvert::RowMajor``), or not to convert at all (``LayoutConvert::None``) because the routine is layout indifferent.

Inside ``tileGet***()``, the logic to copy and transpose is implemented within the ``BaseMatrix::tileCopyDataLayout()`` routine, which is a private function called only from the ``tileGet()`` routine. To avoid extra memory allocations, this routine checks if one if the tiles has a back buffer that can be used as workspace. Additionally, for performance purposes, out of place transposition is always done on device.

The routines ``BaseMatrix::tileLayoutConvert**()`` are available to convert the layout of a tile or set of tiles into the intended layout on a certain device, possibly in batch mode. However, it is important to note that these routines should rarely be needed and are best avoided. All layout conversions should be achievable through the MOSI ``tileGet***()`` routines, which in turn call the tile conversion routines.

Keep in mind that, as a tile can have instances in any of the memory spaces available at the hardware computation node, a tile instance layout is independent of the layout of other instances of the same tile. Additionally, conversion of a tile instance's layout does not change its MOSI state, i.e. a tile does not become ``MOSI::Modified`` by changing its layout since the data is still the same, only represented differently in memory.

Layout Conversion of Extended Tiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SLATE allocates and manages memory through the ``Memory`` class. At the construction of any matrix (``Matrix``, ``TriangularMatrix``, etc.), the parent ``BaseMatrix`` constructor instantiates a ``MatrixStorage`` object, which acts as an interface to the ``Memory`` object. Ideally, a large pool of memory is allocated at the matrix construction through the ``Memory`` object. Shallow copies of the matrix share the same ``MatrixStorage`` and ``Memory`` objects.

The tiles inserted at the matrix object may occupy memory provided by the user upon construction of the matrix, or otherwise occupy memory blocks provided by the ``Memory`` object. Memory provided by the user for a tile may be contiguous, or may be strided, while memory provided by the ``Memory`` object is provided in square contiguous blocks.

For converting a layout into the same memory, the tile's memory needs to be contiguous or square. Tiles whose memory is strided and are rectangular cannot be transposed into the same memory. To facilitate a seamless layout conversion of all tiles, a mechanism of extending the tiles memory is used. An extended tile has an extra memory buffer attached to it, which facilitates transposing the tiles data back and forth between the original memory buffer and the extended memory buffer. Auxiliary member variables of the ``Tile`` class help maintain consistent flags and memory buffer pointers of the extended tile. At any time, the front buffer of an extended tile (can be the original memory buffer referred to as ``Tile::user_data_``, or the extended buffer referred to as ``Tile::ext_data_``), holds the most up-to-date data and in the current layout. The logic to manage the buffers and stride is contained in the ``Tile::setLayout()`` routine. In order to ensure that tiles remain in consistent states, this is the only routine that should change the front buffer and stride.
