MOSI Coherency Protocol
=======================

Coherency Control
-----------------

We describe here the protocol used in SLATE to maintain coherency of tiles' instances among memory spaces (host memory, device memories). The protocol described here is inspired by known cache coherency protocols, but adapted to serve the needs of SLATE algorithms; specifically, no other memory exists as a backing store (as is the main memory in relation to a cache), nor auto eviction.

Concretely, this "coherency protocol" is used to maintain coherency between multiple copies of a tile in different memory spaces within one node (CPU memory, multiple GPU memories). Further, in this document, we will refer to this coherency protocol by the name MOSI (an acronym of the states we assign to the tiles: Modified, OnHold, Shared, Invalid).

The governing principles and requirements in MOSI protocol, besides maintaining tiles coherency, are:

*   Tile data can originate in either CPU or GPU memory.
*   Minimal memory occupation: workspace data to be purged when not in use.
*   Data can be held in a memory space for multiple accesses.
*   Minimal data transfers should be incurred across memory spaces.
*   Coherent states are to be maintained at any time, i.e., any function would assume a coherent state upon entry, and will maintain that coherency upon exit. Consequently, routines need not fix an incoherent state due to previous calls, but will make necessary and minimal validation to ensure it is being called without violating coherency.
*   The user/programmer shall be relieved, as much as possible, from thinking about tile state management (i.e., tile state management should be implicit).

Tile States
-----------

A tile's instance can be in one of three states: Modified, Shared, or Invalid. An additional OnHold flag can be set with any state. The states have the following meanings:

.. code-block:: cpp

   enum MOSI
   {
       Modified = 0x0100,
       OnHold   = 0x1000,
       Shared   = 0x0010,
       Invalid  = 0x0001,
   };

Modified (M)
    Tile's data is modified, other instances should be I; instance cannot be purged.

Shared (S)
    Tile's data is up to date, other instances may be in Shared or I; instance may be purged unless on hold.

Invalid (I)
    Tile's data is obsolete, other instances may be Modified, Shared, or I; instance may be purged unless on hold.

OnHold (O)
    A flag orthogonal to the three states above, indicating that a hold is set on this tile instance, thus it cannot be purged until the hold is unset.

.. warning::
   The ``OnHold`` state is deprecated. It disables the tile release mechanism, which is deprecated.

The state of a tile instance is associated with its pointer in the ``TilesMap`` of the ``MatrixStorage`` class. Recall that a map entry holds a key being a tuple of the tile's (row, col) position in the matrix, and a value being a ``TileNode`` containing pointers to tiles on the host and GPU devices. The MOSI state is stored in each tile itself, which allows tile routines to verify the correct MOSI status. For instance, ``tile::gemm`` can verify that :math:`A` and :math:`B` are at least Shared (readable), and :math:`C` is Modified (read/write).

Two instances of the same tile can be in any of (Invalid, Shared), (Invalid, Modified), (Invalid, Invalid), or (Shared, Shared). Coherence is maintained by enforcing these restrictions.

MOSI API
--------

The routines that control the tile state are the following member functions of the ``BaseMatrix`` class:

*   ``tileState(...)``
*   ``tileGetForReading(...)``
*   ``tileGetForWriting(...)``
*   ``tileModified(...)``
*   ``tileGetAndHold(...)``
*   ``tileUnsetHold(...)``
*   ``tileOnHold(...)``
*   ``tileRelease(...)``

Data Transfer
-------------

``tileGetForReading()``, ``tileGetForWriting()``, and ``tileGetAndHold()`` may initiate a data copy from a source memory space to the destination memory space. While the destination memory space is identified by the device id passed in as a parameter (could be host or GPU device), the source is automatically detected from existing instances of the same tile. SLATE searches for the first Modified or Shared instance, searching devices first, then the host. This ordering ensures it will prefer device-to-device copies over host-to-device copies.

Developer Hints
---------------

Acquiring tiles
    An operation that consumes tiles for reading or writing should acquire the tiles first. Tiles to be read-only should be acquired using the ``tileGetForReading()`` routine at the operation start on the intended device, which will ensure that the most up-to-date tile instance is brought into the device. Tiles to be modified should be acquired using the ``tileGetForWriting()`` routine at the operation start on the intended device, which will ensure that the most up-to-date tile instance is brought in, then marks it "Modified" and invalidates other instances.

Tile purging
    Tiles acquired for reading, unless origin, are placed in a workspace tile instance, and should be purged after the operation is over to make room on the device's memory. Purging is accomplished by calling the ``tileRelease()`` routine, which will delete a tile instance only if it is a workspace with no hold on it and not modified. ``tileErase()``, on the other hand, erases the indicated tile instance unconditionally, and should therefore be used carefully.

Modified tiles
    A tile instance that is acquired by ``tileGetForWriting()`` is marked Modified. However, a newly inserted tile instance may get updated without using the ``slate::internal`` routines, for example, by issuing lapack calls on them, or by direct editing. In addition, tiles acquired for reading (or for writing followed by a copy to other devices) may be updated similarly. In such cases, it is necessary to call ``tileModified()`` in order to mark a tile as Modified and maintain coherency. ``tileModified()`` will invalidate other tile instances, thus forcing them to update subsequently. ``tileModified()`` will check if other tile instances are already in Modified state, as a coherency check, since two instances may not be modified concurrently. However, in some cases, other modified instances may need to be ignored, which can be relayed to ``tileModified()`` by setting the permissive parameter to true.

Holding tiles in a memory space
    Some algorithms need to hold some tile instances with valid states in a certain memory space, and prevent them from being purged during workspace releasing. This can be accomplished using the ``tileGetAndHold()``, which will put a hold on the tile until ``tileUnsetHold()`` is called, at which time a ``tileRelease()`` should generally be invoked (unless the algorithm requires otherwise). This use is deprecated.
