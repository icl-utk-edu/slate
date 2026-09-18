Singular Value Decomposition (SVD)
===================================

Singular value decomposition and related matrix factorizations.

Overview
--------

SVD computes :math:`A = U \Sigma V^H` where:
- U: m×m orthogonal/unitary (left singular vectors)
- Σ: m×n diagonal (singular values in descending order)
- V: n×n orthogonal/unitary (right singular vectors)

LAPACK++ provides multiple SVD algorithms optimized for different scenarios.

Standard SVD
------------

**gesvd** - SVD (simple driver)

Classic SVD algorithm using bidiagonal reduction + QR iteration.
Most reliable but slower for large matrices.

Options for computing U and V:
- All vectors: AllVec
- Thin SVD (economy size): SomeVec
- Overwrite A with vectors: OverwriteVec
- No vectors: NoVec

**gesdd** - SVD using divide-and-conquer

Significantly faster for large matrices, especially when computing vectors.
Uses divide-and-conquer on bidiagonal SVD subproblem.

Recommended for most applications.

**gesvdx** - SVD with subset selection

Computes selected singular values/vectors by:
- Value range: [vl, vu]
- Index range: [il, iu]

Most efficient when only subset needed.

Specialized SVD Routines
------------------------

**gejsv** - Accurate SVD with additional options

Enhanced accuracy and control:
- Jacobi-based algorithm option
- Workspace query optimization
- Column scaling for better conditioning

**gesvj** - Two-sided Jacobi SVD

Uses Jacobi rotations. Excellent for high accuracy but slower.

**gesvdq** - SVD via QR with column pivoting

Efficient for tall-skinny matrices (m >> n).

Bidiagonal SVD
--------------

**bdsqr** - Bidiagonal SVD using QR iteration

Core SVD algorithm. Called internally by gesvd and gesdd.

**bdsdc** - Bidiagonal SVD using divide-and-conquer

Used internally by gesdd.

**bdsvdx** - Bidiagonal SVD with subset selection

Bidiagonal Reduction
--------------------

**gebrd** - Reduce to bidiagonal form

Computes :math:`A = U B V^H` where B is bidiagonal.
First step of SVD computation.

**ungbr / orgbr** - Generate U or V from bidiagonal reduction

Forms orthogonal matrices from elementary reflectors.

**ormbr / unmbr** - Multiply by U or V

Applies transformations without forming matrices explicitly.

Generalized SVD (GSVD)
-----------------------

**ggsvd3** - Generalized SVD

Computes SVD of matrix pair (A, B):
:math:`A = U \Sigma_1 [0 R] Q^H`
:math:`B = V \Sigma_2 [0 R] Q^H`

where R is upper triangular and Σ₁, Σ₂ satisfy :math:`\Sigma_1^T \Sigma_1 + \Sigma_2^T \Sigma_2 = I`.

Applications: Weighted least squares, regularization problems.

**ggsvp3** - Generalized SVD preprocessing

Computes orthogonal transformations for GSVD.

CS Decomposition (CSD)
----------------------

**orcsd / uncsd** - CS decomposition

Computes orthogonal matrices in simultaneous SVD of partitioned matrix:

.. math::

   Q = \\begin{bmatrix} Q_1 & 0 \\\\ 0 & Q_2 \\end{bmatrix}
   \\begin{bmatrix} U_1 & 0 \\\\ 0 & U_2 \\end{bmatrix}
   \\begin{bmatrix} V_1 & 0 \\\\ 0 & V_2 \\end{bmatrix}^T

Applications: Subspace angles, matrix separation.

**orcsd2by1 / uncsd2by1** - 2-by-1 CSD

Variant for 2-block column-partitioned matrices.

**bbcsd** - Bidiagonal block CSD

CS decomposition of bidiagonal blocks.

Utility Functions
-----------------

**bdsqr** - Bidiagonal singular values

Core iteration for computing singular values from bidiagonal form.

**bdsvdx** - Bidiagonal subset selection

Efficient computation of selected singular values.

Rank and Pseudoinverse
-----------------------

SVD is fundamental for:
- Rank determination (count non-zero singular values)
- Pseudoinverse: :math:`A^+ = V \Sigma^+ U^H`
- Condition number: :math:`\kappa(A) = \sigma_{max}/\sigma_{min}`
- Numerical rank: number of singular values above threshold

Applications
------------

**Low-rank approximation:**
:math:`A_k = U_k \Sigma_k V_k^H` (best rank-k approximation)

**Principal component analysis (PCA):**
Left singular vectors are principal components

**Total least squares:**
Accounts for errors in both A and b

**Matrix norms:**
- 2-norm: largest singular value
- Frobenius norm: :math:`\sqrt{\sum \sigma_i^2}`

**Regularization:**
Filter small singular values to solve ill-conditioned systems

See Also
--------

- :doc:`eigenvalues` - Eigenvalue problems
- :doc:`least_squares` - Least squares solvers
- :doc:`util` - Job and Range enumerations
