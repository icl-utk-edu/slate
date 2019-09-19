![SLATE banner](http://icl.bitbucket.io/slate/artwork/Bitbucket/slate_banner.png)

**Software for Linear Algebra Targeting Exascale**

**Innovative Computing Laboratory**

**University of Tennessee**

* * *

[TOC]

* * *

About
=====

Software for Linear Algebra Targeting Exascale (SLATE) is being developed
as part of the Exascale Computing Project (ECP),
which is a joint project of the U.S. Department of Energy's Office of Science
and National Nuclear Security Administration (NNSA).
SLATE will deliver fundamental dense linear algebra capabilities
for current and upcoming distributed-memory systems,
including GPU-accelerated systems as well as more traditional multi core-only systems.

SLATE will provide coverage of existing LAPACK and ScaLAPACK functionality,
including parallel implementations of Basic Linear Algebra Subroutines (BLAS),
linear systems solvers, least squares solvers, and singular value and eigenvalue solvers.
In this respect, SLATE will serve as a replacement for LAPACK and ScaLAPACK,
which, after two decades of operation, cannot be adequately retrofitted
for modern, GPU-accelerated architectures.

The charts shows how heavily ECP applications depend
on dense linear algebra software.
A direct dependency means that the application's source code
contains calls to the library's routines.
An indirect dependency means that the application needs to be linked with the library
due to another component depending on it.
Out of 60 ECP applications, 38 depend on BLAS - either directly on indirectly -
40 depend on LAPACK, and 14 depend on ScaLAPACK.
In other words, the use of dense linear algebra software is ubiquitous
among ECP applications.

![dependency charts](http://icl.bitbucket.io/slate/artwork/Bitbucket/dependency_chart.png)

SLATE is built on top of standards, such as MPI and OpenMP,
and de facto-standard industry solutions such as NVIDIA CUDA and AMD HIP.
SLATE also relies on high performance implementations of numerical kernels from vendor libraries,
such as Intel MKL, IBM ESSL, NVIDIA cuBLAS, and AMD rocBLAS.
SLATE interacts with these libraries through a layer of C++ APIs.
This figure shows SLATE's position in the ECP software stack.

![dependency charts](http://icl.bitbucket.io/slate/artwork/Bitbucket/software_stack.png)

* * *

Documentation
=============

* [Building and Installing SLATE](https://bitbucket.org/icl/slate/wiki/Howto/Building_and_Installing_SLATE)
* [Simple code samples and Makefile for using SLATE](https://bitbucket.org/icl/slate-tutorial/src/default/)
* [SLATE Function Reference](https://icl.bitbucket.io/slate/doxygen/html/)
* [SLATE Website and Papers](http://icl.utk.edu/slate/)

* * *

Getting Help
============

For assistance with SLATE, email *slate-user@icl.utk.edu*.
You can also join the *SLATE User* Google group by going to
https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user,
signing in with your Google credentials, and then clicking `Join group`.

* * *

Contributing
============

The SLATE project welcomes contributions from new developers.
Contributions can be offered through the standard Bitbucket pull request model.
We ask that you complete and submit a contributor agreement.
There are two versions of the agreement,
one for [individuals](https://bitbucket.org/icl/slate/downloads/slate-individual-contributor-agreement-v02.doc),
and one for [organizations](https://bitbucket.org/icl/slate/downloads/slate-corporate-contributor-agreement-v02.doc).
Please look at both to determine which is right for you.
We strongly encourage you to coordinate large contributions with the SLATE development team early in the process.

* * *

Resources
=========

* Visit the [SLATE website](http://icl.utk.edu/slate/) for more information about the SLATE project.
* Visit the [SLATE Working Notes](http://www.icl.utk.edu/publications/series/swans) to find out more about ongoing SLATE developments.
* Visit the [BLAS++ repository](https://bitbucket.org/icl/blaspp) for more information about the C++ API for BLAS.
* Visit the [LAPACK++ repository](https://bitbucket.org/icl/lapackpp) for more information about the C++ API for LAPACK.
* Visit the [ECP website](https://exascaleproject.org) to find out more about the DOE Exascale Computing Initiative.

* * *

Acknowledgments
===============

This research was supported by the Exascale Computing Project (17-SC-20-SC),
a collaborative effort of two U.S. Department of Energy organizations
(Office of Science and the National Nuclear Security Administration)
responsible for the planning and preparation of a capable exascale ecosystem,
including software, applications, hardware, advanced system engineering
and early testbed platforms, in support of the nation's exascale computing imperative.

This research uses resources of the Oak Ridge Leadership Computing Facility,
which is a DOE Office of Science User Facility supported under Contract DE-AC05-00OR22725.
This research also uses resources of the Argonne Leadership Computing Facility,
which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357.

* * *

License
=======

    Copyright (c) 2017, University of Tennessee
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
        * Neither the name of the University of Tennessee nor the
          names of its contributors may be used to endorse or promote products
          derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
