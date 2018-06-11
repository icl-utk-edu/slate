SLATE lapack_api

These routines are intended to simplify interfacing with Fortran-style
LAPACK and BLAS calls.  A Fortran call can be replaced fairly
directly. For example:

   GEMM( transa, transb, m, n, standard parameters ) 
   SLATE_GEMM( transa, transb, m, n, standard parameters ) 

The SLATE API will map the Fortran-style parameters as needed and call
the SLATE routine.  The SLATE API will need to make some choicesa that
are not part of the standard API.

SLATE execution target is set in this order: 

  if env SLATE_TARGET={HostTask,HostBatch,HostNest,Devices}, use it
  else if Devices are compiled in SLATE and available, use Devices
  else use HostTask

SLATE tile size (nb) is set in this order:

  if env SLATE_NB is set, use it
  else if Target=HostTask, nb=512
  else if Target=Devices, nb=1024
  else nb=256

Testing the SLATE lapack_api is done using the tests in the standard
lapack distribution (lapack/BLAS/TESTING).  The subroutine calls are
manually changed to use the SLATE API names (e.g. GEMM -> SLATE_GEMM).

Currently the tests and API handle numeric testing. 
These tests cannot catch parameter-errors.


