SLATE - Software for Linear Algebra Targeting Exascale
======================================================

**SLATE** is a modern C++ library providing dense linear algebra capabilities for high-performance 
computing systems with multi-core processors and hardware accelerators. SLATE provides coverage of 
existing ScaLAPACK functionality and is designed as a replacement for ScaLAPACK, which cannot be 
adequately retrofitted for modern accelerated architectures.

.. grid:: 2

    .. grid-item-card:: 🚀 Quick Start
        :link: user_guide/getting_started
        :link-type: doc

        Get started with SLATE in minutes. Learn installation, basic usage, and run your first example.

    .. grid-item-card:: 📖 User Guide
        :link: user_guide/overview
        :link-type: doc

        Comprehensive guide covering matrices, operations, and all SLATE features.

    .. grid-item-card:: 📚 API Reference
        :link: api/slate/index
        :link-type: doc

        Complete API documentation for all SLATE functions and classes.

    .. grid-item-card:: 💡 Examples
        :link: examples/overview
        :link-type: doc

        Standalone examples demonstrating SLATE features and usage.

    .. grid-item-card:: 🧪 Testing & Tuning
        :link: user_guide/testing
        :link-type: doc

        Testing suite, benchmarking, and performance tuning guidance.


Key Features
------------

- **Distributed Computing**: Efficient 2D block-cyclic distribution using MPI
- **GPU Acceleration**: Support for NVIDIA (CUDA), AMD (ROCm), and Intel (SYCL) GPUs
- **Modern C++ API**: Clean, type-safe interface with simplified naming
- **ScaLAPACK Compatible**: Easy transition with compatible data layouts and API
- **Comprehensive Coverage**: Linear systems, least squares, eigenvalues, SVD, and more
- **Mixed Precision**: Iterative refinement with lower precision for performance


Quick Example
-------------

.. code-block:: cpp

    #include <slate/slate.hh>
    
    // Create matrices on 2x2 process grid
    slate::Matrix<double> A(n, n, nb, p, q, MPI_COMM_WORLD);
    slate::Matrix<double> B(n, nrhs, nb, p, q, MPI_COMM_WORLD);
    
    // Allocate and initialize tiles
    A.insertLocalTiles();
    B.insertLocalTiles();
    
    // Solve AX = B using LU factorization
    slate::lu_solve(A, B);


Resources
---------

* **GitHub**: https://github.com/icl-utk-edu/slate/
* **Website**: https://icl.utk.edu/slate/
* **User Forum**: https://groups.google.com/a/icl.utk.edu/g/slate-user


.. toctree::
  :maxdepth: 2
  :caption: USER GUIDE
  :hidden:

  user_guide/overview
  user_guide/getting_started
  user_guide/installation
  user_guide/matrices
  user_guide/operations
  user_guide/testing
  user_guide/compatibility


.. toctree::
  :maxdepth: 2
  :caption: API REFERENCE
  :hidden:

  api/slate/index
  api/blaspp/index
  api/lapackpp/index


.. toctree::
  :maxdepth: 1
  :caption: EXAMPLES
  :hidden:

  examples/overview
  examples/building
  examples/ex01_matrix
  examples/ex02_conversion
  examples/ex03_submatrix
  examples/ex04_norm
  examples/ex05_blas
  examples/ex06_linear_system_lu
  examples/ex07_linear_system_cholesky
  examples/ex08_linear_system_indefinite
  examples/ex09_least_squares
  examples/ex10_svd
  examples/ex11_hermitian_eig
  examples/ex12_generalized_hermitian_eig
  examples/ex13_non_uniform_block_size
  examples/ex14_scalapack_gemm
  examples/ex15_set_matrix


.. toctree::
  :maxdepth: 2
  :caption: DEVELOPER GUIDE
  :hidden:

  developer_guide/01_introduction
  developer_guide/02_api_layers
  developer_guide/03_matrix_storage
  developer_guide/04_side_uplo_trans
  developer_guide/05_precisions
  developer_guide/06_parallelism
  developer_guide/07_communication
  developer_guide/08_mosi
  developer_guide/09_layout
  developer_guide/10_bibliography


.. toctree::
   :maxdepth: 1
   :caption: ADDITIONAL INFO
   :hidden:

   about
   changelog
   license


Indices and Tables
------------------

* :ref:`genindex`
* :ref:`search`


Citation
--------

.. code-block:: bibtex

    @techreport{gates2020slate,
        author={Gates, Mark and Charara, Ali and Kurzak, Jakub and YarKhan, Asim
                and Al Farhan, Mohammed and Sukkari, Dalal and Burgess, Treece
                and Lindquist, Neil and Dongarra, Jack},
        title={{SLATE} Users' Guide, {SWAN} No. 10},
        institution={Innovative Computing Laboratory, University of Tennessee},
        year={2020},
        month={July},
        number={ICL-UT-19-01},
        note={revision 2023-11},
        url={https://www.icl.utk.edu/publications/swan-010},
    }
