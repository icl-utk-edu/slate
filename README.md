      __| |      \__ __| __|
    \__ \ |     _ \  |   _| 
    ____/____|_/  _\_|  ___|

**Software for Linear Algebra Targeting Exascale**

**Innovative Computing Laboratory**

**University of Tennessee**

![SLATE](http://icl.bitbucket.io/slate/artwork/ecp-slate.jpg)

* * *

[TOC]

* * *

About
=====

The objective of the Software for Linear Algebra Targeting Exascale (SLATE) project
is to provide fundamental dense linear algebra capabilities
to the US Department of Energy
and to the high-performance computing (HPC) community at large.
To this end, SLATE will provide basic dense matrix operations
(e.g., matrix multiplication, rank-k update, triangular solve),
linear systems solvers, least square solvers, singular value and eigenvalue solvers.

The ultimate objective of SLATE is to replace
the venerable Scalable Linear Algebra PACKage (ScaLAPACK) library,
which has become the industry standard for dense linear algebra operations
in distributed memory environments.
However, after two decades of operation, ScaLAPACK is past the end of its lifecycle
and overdue for a replacement, as it can hardly be retrofitted
to support hardware accelerators, which are an integral part
of today's HPC hardware infrastructure.

Primarily, SLATE aims to extract the full performance potential and maximum scalability
from modern, many-node HPC machines with large numbers of cores
and multiple hardware accelerators per node.
For typical dense linear algebra workloads, this means getting close
to the theoretical peak performance and scaling to the full size of the machine
(i.e., thousands to tens of thousands of nodes).
This is to be accomplished in a portable manner by relying on standards
like MPI and OpenMP.

SLATE functionalities will first be delivered to the ECP applications
that most urgently require SLATE capabilities
(e.g., EXascale Atomistics with Accuracy, Length, and Time [EXAALT],
NorthWest computational Chemistry for Exascale [NWChemEx],
Quantum Monte Carlo PACKage [QMCPACK],
General Atomic and Molecular Electronic Structure System [GAMESS],
CANcer Distributed Learning Environment [CANDLE])
and to other software libraries that rely on underlying dense linear algebra services
(e.g., Factorization Based Sparse Solvers and Preconditioners [FBSS]).
SLATE will also fill the void left by ScaLAPACK's inability
to utilize hardware accelerators, and it will ease the difficulties
associated with ScaLAPACK'slegacy matrix layout and Fortran API.

Also, part of the SLATE project is the development of C++ APIs
for [BLAS++](https://bitbucket.org/icl/blaspp)
and [LAPACK++](https://bitbucket.org/icl/lapackpp).

* * *

Documentation
=============

* [SLATE Users' Guide](https://icl.bitbucket.io/slate/sphinx/html/)
* [SLATE Function Reference](https://icl.bitbucket.io/slate/doxygen/html/)
* [SLATE Working Note 3: Designing SLATE: Software for Linear Algebra Targeting Exascale](http://www.icl.utk.edu/publications/swan-003)

* * *

Getting Help
============

Need assistance with the SLATE software?
Join the *SLATE User* Google group by going to
https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user
and clicking `Apply to join group`.
Upon acceptance, email your questions and comments to *slate-user@icl.utk.edu*.

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
