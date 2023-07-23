#!/usr/bin/env python3

import argparse
import subprocess
import io
import time

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

parser.add_argument( 'tests', nargs=argparse.REMAINDER )

opts = parser.parse_args()

#-------------------------------------------------------------------------------
def run_test( cmd ):
    print( '-' * 80 )
    print( ' '.join( cmd ) )
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
        print( 'FAILED, exit code', err )
    else:
        print( 'passed' )

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
]

if (opts.tests):
    tests = opts.tests

#-------------------------------------------------------------------------------
runner = opts.run.split() + [ opts.np ]
print( 'runner', runner )

failed_tests = []

types = opts.type.split()
for test in tests:
    cmd = runner + test.split() + types
    err = run_test( cmd )
    if (err):
        failed_tests.append( test )
# end

timer = time.time() - timer
mins  = timer // 60
secs  = timer %  60
print( 'Elapsed %02d:%02d' % (mins, secs) )

# print summary of failures
nfailed = len( failed_tests )
if (nfailed > 0):
    print( str(nfailed) + ' routines FAILED:',
               ', '.join( failed_tests ) )
else:
    print( 'All routines passed' )

exit( nfailed )
