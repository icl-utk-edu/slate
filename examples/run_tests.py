#!/usr/bin/env python3

import argparse
import subprocess
import io
import time
import sys

timer = time.time()

# ------------------------------------------------------------------------------
# command line arguments
parser = argparse.ArgumentParser()

parser.add_argument( '--run', action='store', default='mpirun -np',
                     help='command to run multiple tasks (i.e., processes, MPI ranks)' )

parser.add_argument( '--np', action='store', default='4',
                     help='number of MPI processes; default=%(default)s' )

parser.add_argument( '--type', action='store', default='s d c z',
                     help='data types, space separated; default=%(default)s' )

parser.add_argument( '--c_api', action='store_true',
                     help='run the C API examples' )

parser.add_argument( '--fortran', action='store_true',
                     help='run the Fortran API examples' )

parser.add_argument( 'tests', nargs=argparse.REMAINDER )

opts = parser.parse_args()

# ------------------------------------------------------------------------------
# When stdout is redirected to file instead of TTY console,
# and  stderr is still going to a TTY console,
# print extra summary messages to stderr.
output_redirected = sys.stderr.isatty() and not sys.stdout.isatty()

# ------------------------------------------------------------------------------
# if output is redirected, prints to both stderr and stdout;
# otherwise prints to just stdout.
def print_tee( *args ):
    global output_redirected
    print( *args )
    if (output_redirected):
        print( *args, file=sys.stderr )
# end

#-------------------------------------------------------------------------------
def run_test( cmd ):
    print_tee( '-' * 80 )
    print_tee( ' '.join( cmd ) )
    p = subprocess.Popen( cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT )
    p_out = io.TextIOWrapper( p.stdout, encoding='utf-8' )

    # Read unbuffered ("for line in p.stdout" will buffer).
    output = ''
    for line in iter(p_out.readline, ''):
        print( line, end='' )
        output += line
    err = p.wait()

    if (err != 0):
        print_tee( 'FAILED, exit code', err )
    else:
        print_tee( 'passed' )

    print( output )
    return err
# end

#-------------------------------------------------------------------------------
tests = [
    './ex01_matrix',
    './ex02_conversion',
    './ex03_submatrix',
    './ex04_norm',
    './ex05_blas',
    './ex06_linear_system_lu',
    './ex07_linear_system_cholesky',
    './ex08_linear_system_indefinite',
    './ex09_least_squares',
    './ex10_svd',
    './ex11_hermitian_eig',
    './ex12_generalized_hermitian_eig',
    './ex13_non_uniform_block_size',
    './ex14_scalapack_gemm',
    './ex15_set_matrix',
]

if opts.c_api:
    tests.extend([
        #'c_api/ex01_matrix',
        #'c_api/ex02_conversion',
        #'c_api/ex03_submatrix',
        #'c_api/ex04_norm',
        'c_api/ex05_blas',
        'c_api/ex06_linear_system_lu',
        #'c_api/ex07_linear_system_cholesky',
        #'c_api/ex08_linear_system_indefinite',
        #'c_api/ex09_least_squares',
        #'c_api/ex10_svd',
        #'c_api/ex11_hermitian_eig',
        #'c_api/ex12_generalized_hermitian_eig',
        #'c_api/ex13_non_uniform_block_size',
    ])

if opts.fortran:
    tests.extend([
        #'fortran/ex01_matrix',
        #'fortran/ex02_conversion',
        #'fortran/ex03_submatrix',
        #'fortran/ex04_norm',
        'fortran/ex05_blas',
        #'fortran/ex06_linear_system_lu',
        #'fortran/ex07_linear_system_cholesky',
        #'fortran/ex08_linear_system_indefinite',
        #'fortran/ex09_least_squares',
        #'fortran/ex10_svd',
        #'fortran/ex11_hermitian_eig',
        #'fortran/ex12_generalized_hermitian_eig',
        #'fortran/ex13_non_uniform_block_size',
    ])

if (opts.tests):
    tests = opts.tests

#-------------------------------------------------------------------------------
runner = opts.run.split() + [ opts.np ]
print( 'runner', runner )

failed_tests = []

types = opts.type.split()
for test in tests:
    t = time.time()
    cmd = runner + test.split() + types
    err = run_test( cmd )
    t = time.time() - t
    print_tee( 'Elapsed %02d:%05.2f mm:ss' % (t // 60, t % 60) )
    if (err):
        failed_tests.append( test )
# end

timer = time.time() - timer
print_tee( 'Total Elapsed %02d:%05.2f mm:ss' % (timer // 60, timer % 60) )

# print summary of failures
nfailed = len( failed_tests )
if (nfailed > 0):
    print( str(nfailed) + ' routines FAILED:',
               ', '.join( failed_tests ) )
else:
    print( 'All routines passed' )

exit( nfailed )
