#!/usr/bin/env python
#
# Example usage:
# help
#     ./run_tests.py -h
#
# run everything with default sizes
# output is redirected; summary information is printed on stderr
#     ./run_tests.py > output.txt
#
# run LU (gesv, getrf, getri, ...), Cholesky (posv, potrf, potri, ...)
# with single, double and default sizes
#     ./run_tests.py --lu --chol --type s,d
#
# run getrf, potrf with small, medium sizes
#     ./run_tests.py -s -m getrf potrf

from __future__ import print_function

import sys
import os
import re
import argparse
import subprocess
import xml.etree.ElementTree as ET
import io

# ------------------------------------------------------------------------------
# command line arguments
parser = argparse.ArgumentParser()

group_test = parser.add_argument_group( 'test' )
group_test.add_argument( '-t', '--test', action='store',
    help='test command to run, e.g., --test "mpirun -np 4"; default "%(default)s"',
    default='' )
group_test.add_argument( '--xml', help='XML file to generate for jenkins' )

parser.add_argument( 'tests', nargs=argparse.REMAINDER )
opts = parser.parse_args()

# ------------------------------------------------------------------------------
cmds = [
    'test_BandMatrix',
    'test_HermitianMatrix',
    'test_LockGuard',
    'test_Matrix',
    'test_Memory',
    'test_SymmetricMatrix',
    'test_TrapezoidMatrix',
    'test_TriangularMatrix',
    'test_Tile',
    'test_Tile_kernels',
    'test_norm',
]

# ------------------------------------------------------------------------------
# when output is redirected to file instead of TTY console,
# print extra messages to stderr on TTY console.
output_redirected = not sys.stdout.isatty()

# ------------------------------------------------------------------------------
def run_test( cmd ):
    print( '-' * 80 )
    cmd = opts.test +' ./' + cmd
    print( cmd, file=sys.stderr )
    output = ''
    p = subprocess.Popen( cmd.split(), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT )
    p_out = p.stdout
    if (sys.version_info.major >= 3):
        p_out = io.TextIOWrapper(p.stdout, encoding='utf-8')
    # Read unbuffered ("for line in p.stdout" will buffer).
    for line in iter(p_out.readline, ''):
        print( line, end='' )
        output += line
    err = p.wait()
    if (err < 0):
        print( 'FAILED: exit with signal', -err )
    return (err, output)
# end

# ------------------------------------------------------------------------------
failed_tests = []
passed_tests = []
ntests = len(opts.tests)
run_all = (ntests == 0)

for cmd in cmds:
    if (run_all or cmd in opts.tests):
        if (not run_all):
            opts.tests.remove( cmd )
        (err, output) = run_test( cmd )
        if (err):
            failed_tests.append( (cmd, err, output) )
        else:
            passed_tests.append( cmd )
if (opts.tests):
    print( 'Warning: unknown tests:', ' '.join( opts.tests ))

# print summary of failures
nfailed = len( failed_tests )
if (nfailed > 0):
    print( '\n' + str(nfailed) + ' tests FAILED:',
           ', '.join( [x[0] for x in failed_tests] ),
           file=sys.stderr )

# generate jUnit compatible test report
if opts.xml:
    print( 'writing XML file', opts.xml )
    root = ET.Element("testsuites")
    doc = ET.SubElement(root, "testsuite",
                        name="slate_unit_test_suite",
                        tests=str(ntests),
                        errors="0",
                        failures=str(nfailed))

    for (test, err, output) in failed_tests:
        testcase = ET.SubElement(doc, "testcase", name=test)

        failure = ET.SubElement(testcase, "failure")
        if (err < 0):
            failure.text = "exit with signal " + str(-err)
        else:
            failure.text = str(err) + " tests failed"

        system_out = ET.SubElement(testcase, "system-out")
        system_out.text = output
    # end

    for test in passed_tests:
        testcase = ET.SubElement(doc, 'testcase', name=test)
        testcase.text = 'PASSED'

    tree = ET.ElementTree(root)
    tree.write( opts.xml )
# end

exit( nfailed )
