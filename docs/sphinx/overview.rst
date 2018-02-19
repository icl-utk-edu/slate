
Overview
========

About
-----

SLATE will offer a modern replacement for ScaLAPACK.
ScaLAPACK is a numerical software library of dense linear algebra routines
essential to the field of scientific and engineering computing,
with good asymptotic scaling properties, but lack of support for modern node architectures,
which are based on multicore processors and hardware accelerators,
and characterized by complex memory hierarchies.

SLATE will allow for the development of multicore and accelerator capabilities,
by leveraging recent progress and ongoing efforts in mainstream programming models
(MPI3 and beyond, OpenMP4 and beyond, OpenACC, etc.), and runtime scheduling systems
(PaRSEC, Legion, etc.).

As part of the SLATE projects, C++ APIs are developed for the
Basic Linear Algebra Subprograms
(`BLAS++ <https://bitbucket.org/icl/blaspp>`_) and for the Linear Algebra PACKage
(`LAPACK++ <https://bitbucket.org/icl/lapackpp>`_).


Resources
---------

* Visit the `SLATE website <http://icl.utk.edu/slate/>`_ for more information about the SLATE project.
* Visit the `SLATE Working Notes <http://www.icl.utk.edu/publications/series/swans>`_ to find out more about ongoing SLATE developments.
* Visit the `BLAS++ repository <https://bitbucket.org/icl/blaspp>`_ for more information about the C++ API for BLAS.
* Visit the `LAPACK++ repository <https://bitbucket.org/icl/lapackpp>`_ for more information about the C++ API for LAPACK.
* Visit the `ECP website <https://exascaleproject.org>`_ to find out more about the DOE Exascale Computing Initiative.

Getting Help
------------

Need assistance with the SLATE software?
Join the *SLATE User* Google group by going to
https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user
and clicking ``Apply to join group``.
Upon acceptance, email your questions and comments to *slate-user@icl.utk.edu*.

Acknowledgments
---------------

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

License
-------

.. code-block:: console

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
