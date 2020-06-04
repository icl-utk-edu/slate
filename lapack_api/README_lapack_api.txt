SLATE BLAS AND LAPACK COMPATIBILITY API
=======================================

This API and routines are intended to simplify interfacing with
Fortran-style LAPACK and BLAS calls.  A Fortran call can be replaced
fairly directly. For example:

'''
   GEMM( transa, transb, m, n, standard parameters )
   SLATE_GEMM( transa, transb, m, n, standard parameters )
'''

The SLATE API will map the Fortran-style parameters as needed and call
the SLATE routine.  The SLATE API will need to make some choices that
are not part of the standard API.


USING THE COMPATIBILITY API
---------------------------

The lapack_api is used by prepending function calls with "SLATE_".
The lapack_api library must linked before standard blas or lapack
libraries.


ENVIRONMENT VARIABLES
---------------------

The SLATE execution target is set in this order:

*  if env SLATE_LAPACK_TARGET={HostTask,HostBatch,HostNest,Devices}, use it
*  else if Devices are compiled in SLATE and available, use Devices
*  else use HostTask

SLATE tile size (nb) is set in this order:

*  if env SLATE_LAPACK_NB is set, use it
*  else if Target=HostTask, nb=512
*  else if Target=Devices, nb=1024
*  else nb=256

SLATE_LAPACK_VERBOSE  0,1 (0: no output,  1: print some minor output)

SLATE_LAPACK_PANELTHREADS integer (number of threads to serve the panel, default (maximum omp threads)/2 )

SLATE_LAPACK_IB integer (inner blocking size useful for some routines, default 16)


TESTING
-------

Testing of the SLATE lapack_api is done using the tests in the
standard lapack distribution (lapack/BLAS/TESTING).  The subroutine
calls are manually changed to use the SLATE API names (e.g. GEMM ->
SLATE_GEMM).

Currently the tests and API handle numeric testing.
These tests cannot catch parameter-errors.

The following shell scripts were used to make alterations in the
standard lapack-release BLAS distribution so that it could be used to
test the SLATE lapack_api.  Your mileage may vary in using these
scripts.  Please consider them a starting point if you wish to to do
your own testing.


### TESTING BLAS3

'''shell

# Set up BLAS_DIR
export LAPACK_DIR=<$HOME/lapack-release>
export BLAS_DIR=<directory for lapack-release/BLAS>

# Prepend known routines with SLATE_ prefix
for precision in 'D' 'S' 'C' 'Z'; do
   for routine in 'GEMM' 'HEMM' 'HERK' 'HER2K' 'SYMM' 'SYR2K' 'SYRK' 'TRMM' 'TRSM' 'POTRF' 'LANGE' 'LANSY' 'LANTR'; do
       lcprec=`echo $precision | tr A-Z a-z`
       # Add slate prefix
       sed -i 's/\b$precision$routine\b/SLATE_$precision$routine/' $BLAS_DIR/TESTING/${lcprec}blat3.f
       # Remove prefix from string names that got prefix
       sed -i "s/'SLATE_$precision$routine/'$precision$routine/" $BLAS_DIR/TESTING/${lcprec}blat3.f
   done
done

# Modify the test input files to skip error exits and remove N=0
for precision in 'D' 'S' 'C' 'Z'; do
     lcprec=`echo $precision | tr A-Z a-z`
     # disable error exit tests
     sed -i 's/T        LOGICAL FLAG, T TO TEST ERROR EXITS./F        LOGICAL FLAG, T TO TEST ERROR EXITS./' $BLAS_DIR/TESTING/${lcprec}blat3.in
     # remove N=0
     sed -i 's/0 1 2/10 1 2/' $BLAS_DIR/TESTING/${lcprec}blat3.in
done

# Recompile. Manually change make.inc to disable fixed line length
# OPTS    = -O2 -frecursive -ffixed-line-length-none
# NOOPT   = -O0 -frecursive -ffixed-line-length-none
# Make sure the lapack_api library precedes blas and lapack libraries.
(cd $BLAS_DIR/TESTING && make clean)
(cd $BLAS_DIR/TESTING && make)

# Run tests for all precisions
# SLATE_LAPACK_VERBOSE=1 will cause a line to be printed on lapack_api call
for precision in 'D' 'S' 'C' 'Z'; do
    lcprec=`echo $precision | tr A-Z a-z`
    (cd $BLAS_DIR/TESTING && env SLATE_LAPACK_VERBOSE=1 ./xblat3$lcprec < ${lcprec}blat3.in 2>&1 | uniq )
done

'''


### TESTING LAPACK

'''shell
# Prepend known routines with SLATE_ prefix
for precision in 'D' 'S' 'C' 'Z'; do
   lcprec=`echo $precision | tr A-Z a-z`
   for routine in 'GEMM' 'HEMM' 'HERK' 'HER2K' 'SYMM' 'SYR2K' 'SYRK' 'TRMM' 'TRSM' 'POTRF' 'LANGE' 'LANSY' 'LANTR' 'GETRF' 'GETRS' 'GELS' 'GESV' 'POSV' 'POTRI'; do
     echo $precision$routine
     for file in $LAPACK_DIR/TESTING/LIN/${lcprec}*.f; do
       # Add slate prefix
       sed -i "s/\b${precision}${routine}\b/SLATE_${precision}${routine}/" ${file}
       # Remove prefix from string names that got prefix
       sed -i "s/'SLATE_${precision}${routine}/'${precision}${routine}/" ${file}
     done
   done
done

# Mixed precision
for precision in 'DS' 'ZC'; do
   for routine in 'GESV'; do
     echo $precision$routine
     for file in $LAPACK_DIR/TESTING/LIN/${lcprec}*.f; do
       # Add slate prefix
       sed -i "s/\b${precision}${routine}\b/SLATE_${precision}${routine}/" ${file}
       # Remove prefix from string names that got prefix
       sed -i "s/'SLATE_${precision}${routine}/'${precision}${routine}/" ${file}
     done
   done
done

# Modify the test input files to skip error exits and remove N=0
for precision in 'D' 'S' 'C' 'Z'; do
     lcprec=`echo $precision | tr A-Z a-z`
     # disable error exit tests
     sed -i 's/T                      Put T to test the error exits/F                      Put T to test the error exits/' $LAPACK_DIR/TESTING/${lcprec}test.in
     # remove N=0 as a choice
     sed -i 's/\b0 1 2/100 1 2/' $LAPACK_DIR/TESTING/${lcprec}test.in
done

# Recompile. Manually change make.inc to disable fixed line length
# OPTS    = -O2 -frecursive -ffixed-line-length-none
# NOOPT   = -O0 -frecursive -ffixed-line-length-none
(cd $LAPACK_DIR/TESTING/LIN && make clean )
(cd $LAPACK_DIR/TESTING/LIN && make )

# Run tests for all precisions
# SLATE_LAPACK_VERBOSE=1 will cause a line to be printed on lapack_api call
for precision in 'D' 'S' 'C' 'Z'; do
    lcprec=`echo $precision | tr A-Z a-z`
    (cd $LAPACK_DIR/TESTING && env SLATE_LAPACK_VERBOSE=1 ./LIN/xlintst$lcprec < ${lcprec}test.in 2>&1 | uniq )
done

'''
