#!/usr/bin/env python3
#
# Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
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
import select
import time

# ------------------------------------------------------------------------------
# command line arguments
parser = argparse.ArgumentParser()

group_test = parser.add_argument_group( 'test' )
group_test.add_argument( '-t', '--test', action='store',
    help='test command to run, e.g., --test "mpirun -np 4"; default "%(default)s"',
    default='' )
group_test.add_argument( '--xml', help='XML file to generate for jenkins' )
group_test.add_argument( '--timeout', action='store', help='timeout in seconds for each routine', type=float )

parser.add_argument( 'tests', nargs=argparse.REMAINDER )
opts = parser.parse_args()

# ------------------------------------------------------------------------------
cmds = [
    'test_BandMatrix',
    'test_HermitianMatrix',
    'test_LockGuard',
    'test_OmpSetMaxActiveLevels',
    'test_Matrix',
    'test_Memory',
    'test_SymmetricMatrix',
    'test_TrapezoidMatrix',
    'test_TriangularBandMatrix',
    'test_TriangularMatrix',
    'test_Tile',
    'test_Tile_kernels',
    'test_geadd',
    'test_gecopy',
    'test_geset',
    'test_internal_blas',
    #'test_lq', todo hanging on dopamine
    'test_norm',
    #'test_qr',  # todo: failing
]

# ------------------------------------------------------------------------------
# when output is redirected to file instead of TTY console,
# print extra messages to stderr on TTY console.
output_redirected = not sys.stdout.isatty()

# ------------------------------------------------------------------------------
# if output is redirected, prints to both stderr and stdout;
# otherwise prints to just stdout.
def print_tee( *args ):
    global output_redirected
    print( *args )
    if (output_redirected):
        print( *args, file=sys.stderr )
# end

# ------------------------------------------------------------------------------
# cmd is a string: tester
def run_test( cmd ):
    print( '-' * 80 )
    cmd = opts.test +' ./' + cmd
    print_tee( cmd )
    failure_reason = 'FAILED'
    output = ''
    p = subprocess.Popen( cmd.split(), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT )
    p_out = p.stdout
    if (sys.version_info.major >= 3):
        p_out = io.TextIOWrapper(p.stdout, encoding='utf-8')

    if (opts.timeout is None):
        # Read unbuffered ("for line in p.stdout" will buffer).
        for line in iter(p_out.readline, ''):
            print( line, end='' )
            output += line
    else:
        killed = False
        poll_obj = select.poll()
        poll_obj.register(p_out, select.POLLIN)
        now = start = time.time()
        while (now - start) < opts.timeout:
            # 0 means do not wait in poll(), return immediately.
            poll_result = poll_obj.poll(0)
            if poll_result:
                # Assumed that tester prints new lines.
                out = p_out.readline()
                print( out, end='' )
                output += out
            # Check whether the process is still alive
            err = p.poll()
            if err is not None:
                break
            now = time.time()
        else:
            killed = True
            failure_reason = 'Timeout (limit=' + str(opts.timeout) + ')'
            output = output + '\n' + failure_reason
            p.kill()
    err = p.wait()

    if (err != 0):
        print_tee( failure_reason, ': exit code', err )
    else:
        print_tee( 'pass' )
    return (err, output)
# end

# ------------------------------------------------------------------------------
# run each test
failed_tests = []
passed_tests = []
ntests = len(opts.tests)
run_all = (ntests == 0)

seen = set()
for cmd in cmds:
    if (run_all or cmd in opts.tests):
        seen.add( cmd )
        (err, output) = run_test( cmd )
        if (err):
            failed_tests.append( (cmd, err, output) )
        else:
            passed_tests.append( cmd )
print( '-' * 80 )

not_seen = list( filter( lambda x: x not in seen, opts.tests ) )
if (not_seen):
    print_tee( 'Warning: unknown unit tests:', ' '.join( not_seen ))

# print summary of failures
nfailed = len( failed_tests )
if (nfailed > 0):
    print_tee( str(nfailed) + ' unit tests FAILED:',
               ', '.join( [x[0] for x in failed_tests] ) )
else:
    print_tee( 'All unit tests passed' )

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
