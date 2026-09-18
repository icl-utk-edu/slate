Eigenvalue Problems
===================

Eigenvalue and eigenvector computations for standard and generalized problems.

Overview
--------

LAPACK++ provides comprehensive eigenvalue solvers for:
- Symmetric/Hermitian matrices (real eigenvalues)
- Non-symmetric matrices (complex eigenvalues)
- Generalized eigenvalue problems
- Schur decompositions

Standard Eigenvalue Problems
-----------------------------

Symmetric/Hermitian Eigenvalues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**syev / heev** - Eigenvalues and eigenvectors (simple driver)

Computes all eigenvalues and optionally eigenvectors of symmetric/Hermitian matrix.
Uses QR iteration.

**syevd / heevd** - Eigenvalues using divide-and-conquer

Faster algorithm for large matrices. Recommended for most cases.

**syevr / heevr** - Eigenvalues using MRRR algorithm

Most accurate and efficient for computing subset of eigenvalues/vectors.
Uses Relatively Robust Representations.

**syevx / heevx** - Eigenvalues with subset selection

Computes selected eigenvalues/vectors by value range or index range.

**sygv / hegv** - Generalized symmetric eigenvalue problem

Solves :math:`Ax = \lambda Bx`, :math:`ABx = \lambda x`, or :math:`BAx = \lambda x`.

**sygvd / hegvd** - Generalized using divide-and-conquer

**sygvx / hegvx** - Generalized with subset selection

Non-Symmetric Eigenvalues
^^^^^^^^^^^^^^^^^^^^^^^^^^

**geev** - Eigenvalues and eigenvectors

Computes eigenvalues and optionally left/right eigenvectors.

**gees** - Schur decomposition

Computes Schur form :math:`A = QTQ^H` where T is upper triangular (or quasi-triangular for real).

**geesx** - Schur with condition estimates

Extended Schur decomposition with reciprocal condition numbers.

**geevx** - Eigenvalues with balancing and condition estimates

Tridiagonal/Banded Eigenproblems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**stev** - Symmetric tridiagonal eigenvalues (simple driver)

**stevd** - Symmetric tridiagonal using divide-and-conquer

**stevr** - Symmetric tridiagonal using MRRR

**stevx** - Symmetric tridiagonal with subset selection

**sbev / hbev** - Symmetric band eigenvalues

**sbevd / hbevd** - Symmetric band using divide-and-conquer

**sbevx / hbevx** - Symmetric band with subset selection

Generalized Eigenvalue Problems
--------------------------------

Symmetric/Hermitian
^^^^^^^^^^^^^^^^^^^

**sygv / hegv** - Types 1, 2, 3 (see itype)

Type 1: :math:`Ax = \lambda Bx` (most common)

Type 2: :math:`ABx = \lambda x`

Type 3: :math:`BAx = \lambda x`

**sygvd / hegvd** - Generalized using divide-and-conquer

**sygvx / hegvx** - Generalized with subset selection

**spgv / hpgv** - Packed storage format

**spgvd / hpgvd** - Packed storage with divide-and-conquer

**spgvx / hpgvx** - Packed storage with subset

**sbgv / hbgv** - Banded matrices

**sbgvd / hbgvd** - Banded with divide-and-conquer

**sbgvx / hbgvx** - Banded with subset

Non-Symmetric (Generalized Schur)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**gges** - Generalized Schur decomposition

Computes generalized Schur form of pair (A,B):
:math:`A = QSZ^H`, :math:`B = QTZ^H`

**ggesx** - Generalized Schur with condition estimates

**ggev** - Generalized eigenvalues and eigenvectors

Solves :math:`Ax = \lambda Bx`

**ggevx** - Generalized eigenvalues with balancing

Reduction to Standard Form
---------------------------

**hegst / sygst** - Reduce to standard form

Transforms :math:`Ax = \lambda Bx` to :math:`Cy = \lambda y` using Cholesky factorization of B.

**hpgst / spgst** - Packed storage variant

**hbgst / sbgst** - Band storage variant

Auxiliary Reductions
--------------------

**hetrd / sytrd** - Reduce to tridiagonal form

Computes :math:`A = Q T Q^H` where T is tridiagonal.

**orgtr / ungtr** - Generate Q from tridiagonal reduction

**ormtr / unmtr** - Multiply by Q from tridiagonal reduction

**gehrd** - Reduce to Hessenberg form

Computes :math:`A = Q H Q^H` where H is upper Hessenberg.

**orghr / unghr** - Generate Q from Hessenberg reduction

**ormhr / unmhr** - Multiply by Q from Hessenberg reduction

Schur Form Manipulation
------------------------

**hseqr** - Schur decomposition of Hessenberg

Computes eigenvalues from Hessenberg form.

**trexc** - Reorder Schur factorization

Reorders diagonal blocks of Schur form.

**trsen** - Reorder and compute condition numbers

**trsyl** - Sylvester equation :math:`AX \pm XB = C`

Eigenvector Computation
------------------------

**trevc** - Eigenvectors from Schur form

Computes right and/or left eigenvectors from triangular Schur form.

**hsein** - Eigenvectors by inverse iteration

**trsna** - Reciprocal condition numbers for eigenvectors

Balancing
---------

**gebal** - Balance non-symmetric matrix

Permutes and scales to improve eigenvalue accuracy.

**gebak** - Transform eigenvectors after balancing

**ggbal** - Balance generalized eigenvalue problem

**ggbak** - Transform eigenvectors after generalized balancing

Utility Functions
-----------------

**lacpy** - Copy matrix or submatrix

**laset** - Initialize matrix to constants

**disna** - Reciprocal condition numbers for eigenvectors

See Also
--------

- :doc:`svd` - Singular value decomposition
- :doc:`linear_systems` - Related factorizations
- :doc:`util` - Job and Range enumerations
