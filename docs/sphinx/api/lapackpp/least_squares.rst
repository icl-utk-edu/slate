Least Squares and Linear Regression
====================================

Solvers for overdetermined and underdetermined linear systems.

Overview
--------

Least squares problems minimize :math:`||b - Ax||_2` for overdetermined systems (m > n) or find minimum-norm solutions for underdetermined systems (m < n).

LAPACK++ provides QR and SVD-based methods with optional rank-revealing capabilities for ill-conditioned problems.

QR-Based Least Squares
-----------------------

**gels** - Least squares solve (simple driver)

Solves overdetermined or underdetermined systems using QR or LQ factorization:
- m ≥ n: QR factorization, minimizes :math:`||b - Ax||_2`
- m < n: LQ factorization, finds minimum-norm solution

**gelsy** - Least squares with complete orthogonal factorization

Uses rank-revealing QR with column pivoting. More robust than gels for rank-deficient problems.

**gelss** - Least squares using SVD (simple driver)

SVD-based solver with automatic rank detection via singular value threshold. Most robust but slowest.

**gelsd** - Least squares using SVD with divide-and-conquer

Faster SVD-based solver using divide-and-conquer algorithm for large problems.

**getsls** - Least squares with tall-skinny QR

Optimized for tall-skinny matrices (m >> n) using recursive algorithms.

QR Factorization
----------------

**geqrf** - QR factorization

Computes :math:`A = QR` where Q is orthogonal/unitary, R is upper triangular.

**geqp3** - QR with column pivoting

Rank-revealing QR: :math:`AP = QR` where P is permutation matrix.

**geqrt** - QR with tall-skinny optimization

Recursive blocked QR for tall-skinny matrices.

**geqr** - QR with optimal blocking

Adaptive blocked QR (LAPACK 3.9+).

LQ Factorization
----------------

**gelqf** - LQ factorization

Computes :math:`A = LQ` where L is lower triangular, Q is orthogonal/unitary.

QL and RQ Factorizations
-------------------------

**geqlf** - QL factorization

Computes :math:`A = QL`.

**gerqf** - RQ factorization

Computes :math:`A = RQ`.

Orthogonal Matrix Generation
-----------------------------

**ungqr / orgqr** - Generate Q from QR

Forms orthogonal/unitary matrix Q from elementary reflectors.

**unglq / orglq** - Generate Q from LQ

**ungql / orgql** - Generate Q from QL

**ungrq / orgrq** - Generate Q from RQ

Multiplication by Q
-------------------

**unmqr / ormqr** - Multiply by Q from QR

Computes :math:`QC`, :math:`Q^H C`, :math:`CQ`, or :math:`CQ^H` without forming Q explicitly.

**unmlq / ormlq** - Multiply by Q from LQ

**unmql / ormql** - Multiply by Q from QL

**unmrq / ormrq** - Multiply by Q from RQ

Complete Orthogonal Factorization
----------------------------------

**tzrzf** - RZ factorization

Reduces upper trapezoidal matrix to upper triangular form.

**unmrz / ormrz** - Multiply by Z from RZ

Constrained Least Squares
--------------------------

**gglse** - Linear equality-constrained least squares

Solves :math:`\min ||c - Ax||_2` subject to :math:`Bx = d`.

**ggglm** - General Gauss-Markov linear model

Solves :math:`\min ||y||_2` subject to :math:`Ax + By = d`.

Rank Estimation
---------------

**gelsy** includes rank estimation via:
- Column pivoting with threshold-based rank detection
- RCOND parameter controls singular value threshold

**gelss** / **gelsd** include:
- Singular value-based rank detection
- RCOND parameter for numerical rank determination

See Also
--------

- :doc:`linear_systems` - Square system solvers
- :doc:`svd` - Singular value decomposition
- :doc:`auxiliary` - QR/LQ utility functions
