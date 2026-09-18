Utilities and Enumerations
===========================

Core utilities for LAPACK++ including error handling, enumerations, and type conversions.

Error Handling
--------------

.. doxygenclass:: lapack::Error
   :project: LAPACKPP
   :members:

Enumerations
------------

LAPACK++ provides type-safe enumerations for function parameters. All enumerations include:
- Conversion functions: ``to_char()``, ``to_string()``, ``from_string()``
- Character-based values matching LAPACK Fortran conventions

**Sides** - Extended side specification (Left, Right, Both)

Values: Left, Right, Both

Used in: trevc (eigenvector computation)

**Norm** - Matrix/vector norm types

Values:
- One (1-norm): Maximum absolute column sum
- Two (2-norm): Spectral norm (largest singular value)  
- Inf (infinity-norm): Maximum absolute row sum
- Fro (Frobenius): Square root of sum of squares
- Max: Maximum absolute element

Used in: lange, lansy, lanhe, gecon

**Job** - Eigenvector/singular vector computation options

Values:
- NoVec: Compute eigenvalues only
- Vec: Compute eigenvectors
- UpdateVec: Update eigenvectors
- AllVec: Compute all singular vectors
- SomeVec: Compute some singular vectors
- OverwriteVec: Overwrite with singular vectors
- CompactVec: Compact representation
- SomeVecTol: Some vectors with tolerance
- VecJacobi: Jacobi method vectors
- Workspace: Workspace query

Used in: geev, syev, gesvd, gesdd, gejsv, gesvj

**JobSchur** - Schur decomposition options

Values:
- Eigenvalues: Compute eigenvalues only
- Schur: Compute full Schur form

Used in: hseqr

**Sort** - Eigenvalue sorting

Values:
- NotSorted: Keep original order
- Sorted: Sort eigenvalues

Used in: gees, gges

**Range** - Eigenvalue/singular value selection

Values:
- All: Compute all values
- Value: Compute values in (vl, vu]
- Index: Compute values with indices [il, iu]

Used in: syevx, gesvdx, stebz

**Vect** - Orthogonal vector generation

Values:
- Q: Generate Q vectors
- P: Generate P vectors
- None: Don't generate vectors
- Both: Generate both Q and P

Used in: orgbr, ormbr, gbbrd

**Direction** - Block reflector direction

Values:
- Forward: Apply reflectors forward
- Backward: Apply reflectors backward

Used in: larfb, larft

**StoreV** - Reflector storage format

Values:
- Columnwise: Reflectors stored in columns
- Rowwise: Reflectors stored in rows

Used in: larfb

**MatrixType** - Matrix structure for scaling

Values:
- General: Full matrix
- Lower: Lower triangular
- Upper: Upper triangular
- Hessenberg: Upper Hessenberg
- LowerBand: Lower banded
- UpperBand: Upper banded
- Band: Full band

Used in: lascl, laset

**HowMany** - Eigenvector selection

Values:
- All: Compute all eigenvectors
- Backtransform: Backtransform eigenvectors
- Select: Compute selected eigenvectors

Used in: trevc

**Equed** - Equilibration status

Values:
- None: No equilibration
- Row: Row equilibration
- Col: Column equilibration
- Both: Row and column equilibration
- Yes: Equilibrated (porfsx)

Used in: gesvx, posvx, sysvx

**Factored** - Factorization status

Values:
- Factored: Matrix already factored
- NotFactored: Need factorization
- Equilibrate: Equilibrate then factor

Used in: gesvx, posvx, sysvx

**Sense** - Condition number computation

Values:
- None: Don't compute condition numbers
- Eigenvalues: Condition numbers for eigenvalues
- Subspace: Condition numbers for invariant subspace
- Both: Both eigenvalues and subspace

Used in: geesx, trsen

**JobCond** - Singular/eigenvector condition numbers

Values:
- EigenVec: Eigenvector condition numbers
- LeftSingularVec: Left singular vector conditions
- RightSingularVec: Right singular vector conditions

Used in: disna

**Balance** - Matrix balancing options

Values:
- None: No balancing
- Permute: Permute only
- Scale: Scale only
- Both: Permute and scale

Used in: gebal, ggbal, gebak, ggbak

**Order** - Eigenvalue ordering for tridiagonal solvers

Values:
- Block: By block
- Entire: For entire matrix

Used in: stebz, larrd, stein

**RowCol** - Row/column orientation

Values:
- Col: Column-oriented
- Row: Row-oriented

Used in: LAPACK testing utilities

**Pivot** - Pivoting strategy

Values:
- Variable: Algorithm-chosen pivots
- Top: Pivot at top
- Bottom: Pivot at bottom

Used in: geqp3 (rank-revealing QR)

Version Information
-------------------

LAPACK++ provides version query functions:

**lapackpp_version()** - Returns version as integer (YYYYMMDD format, e.g., 20250528)

**lapackpp_id()** - Returns version string with git commit (e.g., "2025.05.28.id123abc")

These functions are defined in ``lapack.hh``.
