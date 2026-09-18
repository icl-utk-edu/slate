Singular Value Decomposition
=============================

Compute the singular value decomposition :math:`A = U \Sigma V^H`.

.. contents:: Contents
   :local:
   :depth: 2

SVD Functions
-------------

.. doxygengroup:: svd
   :project: SLATE
   :content-only:
   :members:

Notes
-----

The SVD computes:

.. math::

   A = U \Sigma V^H

where:
- :math:`U` is m-by-m unitary
- :math:`\Sigma` is m-by-n diagonal with non-negative real entries
- :math:`V` is n-by-n unitary
