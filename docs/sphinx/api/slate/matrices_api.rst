Matrix Classes
==============

SLATE's matrix classes provide distributed, tiled matrix storage with support for various matrix structures.

.. contents:: Contents
   :local:
   :depth: 2

General Matrices
----------------

.. doxygenclass:: slate::Matrix
   :project: SLATE
   :members:
   :protected-members:

.. doxygenclass:: slate::BaseMatrix
   :project: SLATE
   :members:
   :protected-members:

Structured Matrices
--------------------

Hermitian Matrices
^^^^^^^^^^^^^^^^^^

.. doxygenclass:: slate::HermitianMatrix
   :project: SLATE
   :members:

Symmetric Matrices
^^^^^^^^^^^^^^^^^^

.. doxygenclass:: slate::SymmetricMatrix
   :project: SLATE
   :members:

Triangular Matrices
^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: slate::TriangularMatrix
   :project: SLATE
   :members:

.. doxygenclass:: slate::TrapezoidMatrix
   :project: SLATE
   :members:

.. doxygenclass:: slate::BaseTrapezoidMatrix
   :project: SLATE
   :members:

Band Matrices
-------------

.. doxygenclass:: slate::BandMatrix
   :project: SLATE
   :members:

.. doxygenclass:: slate::BaseBandMatrix
   :project: SLATE
   :members:

.. doxygenclass:: slate::HermitianBandMatrix
   :project: SLATE
   :members:

.. doxygenclass:: slate::TriangularBandMatrix
   :project: SLATE
   :members:

.. doxygenclass:: slate::BaseTriangularBandMatrix
   :project: SLATE
   :members:

Tiles
-----

.. doxygenclass:: slate::Tile
   :project: SLATE
   :members:

Auxiliary Classes
-----------------

.. doxygenclass:: slate::Pivot
   :project: SLATE
   :members:

.. doxygentypedef:: slate::Pivots
   :project: SLATE

.. doxygentypedef:: slate::TriangularFactors
   :project: SLATE
