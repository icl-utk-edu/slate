Linear Systems
==============

SLATE provides comprehensive routines for solving linear systems :math:`AX = B`.

.. contents:: Contents
   :local:
   :depth: 2

LU Factorization (General)
--------------------------

Simplified API
^^^^^^^^^^^^^^
.. doxygengroup:: lu_solve
   :project: SLATE
   :content-only:
   :members:

Standard API
^^^^^^^^^^^^
.. doxygengroup:: lu
   :project: SLATE
   :content-only:
   :members:

Cholesky Factorization (Positive Definite)
------------------------------------------

Simplified API
^^^^^^^^^^^^^^
.. doxygengroup:: chol_solve
   :project: SLATE
   :content-only:
   :members:

Standard API
^^^^^^^^^^^^
.. doxygengroup:: chol
   :project: SLATE
   :content-only:
   :members:

Indefinite Factorization (Symmetric/Hermitian)
-----------------------------------------------

Simplified API
^^^^^^^^^^^^^^
.. doxygengroup:: indefinite_solve
   :project: SLATE
   :content-only:
   :members:

Standard API
^^^^^^^^^^^^
.. doxygengroup:: sysv
   :project: SLATE
   :content-only:
   :members:
