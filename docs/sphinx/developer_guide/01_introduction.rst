
The SLATE Developers' Guide provides in-depth information on SLATE's architecture, design principles, and implementation details. It is intended for developers contributing to SLATE or advanced users who want to understand the library's internal workings.

Introduction
============

SLATE (Software for Linear Algebra Targeting Exascale) [1]_ has been developed as part of the Exascale Computing Project (ECP) [2]_, which is a joint project of the U.S. Department of Energy's Office of Science and National Nuclear Security Administration (NNSA). The objective of SLATE is to provide fundamental dense linear algebra capabilities to the U.S. Department of Energy and to the high-performance computing (HPC) community at large.

SLATE provides coverage of existing LAPACK and ScaLAPACK functionality, including parallel implementations of basic linear algebra subprograms (BLAS), matrix norms, linear systems solvers, least squares solvers, and singular value and eigenvalue solvers. In this respect, SLATE will serve as a replacement for ScaLAPACK, which, after two decades of operation, cannot be adequately retrofitted for modern, GPU-accelerated architectures.

This Developers' Guide is intended to describe the internal workings of SLATE, to be of use for SLATE developers and contributors. A companion SLATE Users' Guide [3]_ is available for application end users, which focuses on the public SLATE API. These guides will be periodically revised as SLATE develops, with the revisions noted in the front matter notes and BibTeX.

Revision Notes
--------------

*   **12-2019**: First publication
*   **02-2020**: Executing multiple internal routines on devices subsection
*   **04-2020**: Add work routines
*   **08-2020**: Copy editing
*   **11-2023**: Major revision

.. [1] http://icl.utk.edu/slate/
.. [2] https://www.exascaleproject.org
.. [3] SLATE Users' Guide: https://www.icl.utk.edu/publications/swan-010
