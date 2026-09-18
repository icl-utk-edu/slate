Performance Counters
====================

FLOP (floating-point operation) and bandwidth counters for LAPACK routines.

Overview
--------

LAPACK++ provides theoretical operation counts based on LAWN 41 formulas to:

- Measure computational complexity
- Estimate performance 
- Calculate theoretical peak FLOP rates
- Analyze memory bandwidth requirements

All counts are templated on scalar type (float, double, complex<float>, complex<double>) to account for real vs. complex arithmetic overhead.

FLOP Counting
-------------

.. doxygenclass:: lapack::Gflop
   :project: LAPACKPP
   :members:

Bandwidth Analysis
------------------

.. doxygenclass:: lapack::Gbyte
   :project: LAPACKPP
   :members:

Usage Example
-------------

.. code-block:: cpp

    // Count FLOPs for double-precision LU factorization
    double m = 1000, n = 1000;
    double flops = lapack::Gflop<double>::getrf(m, n);
    std::cout << "LU factorization: " << flops << " GFLOPs" << std::endl;

    // Count FLOPs for complex QR factorization
    double gflops_complex = lapack::Gflop<std::complex<double>>::geqrf(m, n);
    
    // Data transfer for Cholesky
    double bytes = lapack::Gbyte<double>::potrf(n);
    std::cout << "Cholesky bandwidth: " << bytes << " GB" << std::endl;

Supported Operations
--------------------

**Linear Systems:**
- LU: gesv, getrf, getrs, getri
- Cholesky: posv, potrf, potrs, potri
- Band Cholesky: pbsv, pbtrf, pbtrs
- Symmetric: sysv, sytrf, sytrs, sytri, hesv, hetrf, hetrs, hetri

**QR Factorizations:**
- geqrf, geqlf, gerqf, gelqf, geqrt
- ungqr/orgqr, ungql/orgql, ungrq/orgrq, unglq/orglq
- unmqr/ormqr, unmql/ormql, unmrq/ormrq, unmlq/ormlq
- gels (least squares)

**Reductions:**
- gehrd (Hessenberg)
- hetrd/sytrd (tridiagonal)
- gebrd (bidiagonal)

**Utilities:**
- trtri (triangular inverse)
- lauum (U^H*U or L*L^T)
- larfg (Householder reflector)
- geadd (matrix addition)

**Norms:**
- lange (general matrix)
- lanhe/lansy (Hermitian/symmetric)

Notes
-----

- Formulas assume standard LAPACK calling conventions
- Some counts may be inaccurate for edge cases (m, n, or k = 0)
- Complex arithmetic counted as: 1 complex multiply = 6 real ops, 1 complex add = 2 real ops
- Based on LAWN 41: "Operation Count for the QR and Cholesky Factorizations" by Demmel and Hida
