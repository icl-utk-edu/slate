![SLATE banner](http://icl.bitbucket.io/slate/artwork/Bitbucket/slate_banner.png)

**Software for Linear Algebra Targeting Exascale**

**Innovative Computing Laboratory**

**University of Tennessee**

* * *

[TOC]

* * *

About
--------------------------------------------------------------------------------

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

SLATE is built on top of standards, such as MPI and OpenMP,
and de facto-standard industry solutions such as NVIDIA CUDA and AMD HIP.
SLATE also relies on high performance implementations of numerical kernels from vendor libraries,
such as Intel MKL, IBM ESSL, NVIDIA cuBLAS, and AMD rocBLAS.
SLATE interacts with these libraries through a layer of C++ APIs.
This figure shows SLATE's position in the ECP software stack.

![SLATE software stack](http://icl.bitbucket.io/slate/artwork/Bitbucket/software_stack.png)

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

![ECP dependency charts](http://icl.bitbucket.io/slate/artwork/Bitbucket/dependency_chart.png)

* * *

Documentation
--------------------------------------------------------------------------------

* [Building and Installing SLATE](https://bitbucket.org/icl/slate/wiki/Howto/Building_and_Installing_SLATE)
* [Tutorial with sample codes for using SLATE](https://bitbucket.org/icl/slate-tutorial/)
* [SLATE Function Reference](https://icl.bitbucket.io/slate/)
* [SLATE Website and Papers](http://icl.utk.edu/slate/)

* * *

Getting Help
--------------------------------------------------------------------------------

For assistance, visit the *SLATE User Forum* at
<https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user>.
Join by signing in with your Google credentials, then clicking
*Join group to post*.

Bug reports can be filed directly on Bitbucket's issue tracker:
<http://bitbucket.org/icl/lapackpp/issues>.

* * *

Resources
--------------------------------------------------------------------------------

* Visit the [SLATE website](http://icl.utk.edu/slate/)
  for more information about the SLATE project.
* Visit the [SLATE Working Notes](http://www.icl.utk.edu/publications/series/swans)
  to find out more about ongoing SLATE developments.
* Visit the [BLAS++ repository](https://bitbucket.org/icl/blaspp)
  for more information about the C++ API for BLAS.
* Visit the [LAPACK++ repository](https://bitbucket.org/icl/lapackpp)
  for more information about the C++ API for LAPACK.
* Visit the [ECP website](https://exascaleproject.org)
  to find out more about the DOE Exascale Computing Initiative.

* * *

Contributing
--------------------------------------------------------------------------------

The SLATE project welcomes contributions from new developers.
Contributions can be offered through the standard Bitbucket pull request model.
We strongly encourage you to coordinate large contributions with the SLATE
development team early in the process.

* * *

Acknowledgments
--------------------------------------------------------------------------------

<!--
https://www.exascaleproject.org/resources/
https://www.olcf.ornl.gov/olcf-media/media-assets/
https://www.alcf.anl.gov/support-center/facility-policies/alcf-acknowledgement-policy
-->

This research was supported by the Exascale Computing Project (17-SC-20-SC), a
joint project of the U.S. Department of Energy's Office of Science and National
Nuclear Security Administration, responsible for delivering a capable exascale
ecosystem, including software, applications, and hardware technology, to support
the nation’s exascale computing imperative.

This research uses resources of the Oak Ridge Leadership Computing Facility,
which is a DOE Office of Science User Facility supported under Contract DE-AC05-00OR22725.
This research also uses resources of the Argonne Leadership Computing Facility,
which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357.

* * *

License
--------------------------------------------------------------------------------

Copyright (c) 2017-2020, University of Tennessee. All rights reserved.

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

**This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall the copyright holders or contributors be liable
for any direct, indirect, incidental, special, exemplary, or consequential
damages (including, but not limited to, procurement of substitute goods or
services; loss of use, data, or profits; or business interruption) however
caused and on any theory of liability, whether in contract, strict liability, or
tort (including negligence or otherwise) arising in any way out of the use of
this software, even if advised of the possibility of such damage.**
