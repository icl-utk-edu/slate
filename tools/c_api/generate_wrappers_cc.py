#!/usr/bin/env python

import sys
import re

data_types = [
    ['float',           '_r32', 'float'],
    ['double',          '_r64', 'double'],
    ['float _Complex',  '_c32', 'std::complex<float>'],
    ['double _Complex', '_c64', 'std::complex<double>'],
]

matrix_types = [
    ['Matrix',               'int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm'],
    ['BandMatrix',           'int64_t m, int64_t n, int64_t kl, int64_t ku, int64_t nb, int p, int q, MPI_Comm mpi_comm'],
    ['HermitianMatrix',      'slate_Uplo uplo, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm'],
    ['HermitianBandMatrix',  'slate_Uplo uplo, int64_t n, int64_t kd, int64_t nb, int p, int q, MPI_Comm mpi_comm'],
    ['TriangularMatrix',     'slate_Uplo uplo, slate_Diag diag, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm'],
    ['TriangularBandMatrix', 'slate_Uplo uplo, slate_Diag diag, int64_t n, int64_t kd, int64_t nb, int p, int q, MPI_Comm mpi_comm'],
    ['SymmetricMatrix',      'slate_Uplo uplo, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm'],
    ['TrapezoidMatrix',      'slate_Uplo uplo, slate_Diag diag, int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm'],
]

file = open(sys.argv[1], 'r')

matrix_block_is_found = False
function_is_found = False
function_found = 0
function_counter = 0
functions = []
container = ''
container2 = ''
for line in file:
    if re.search(r'^ *// @begin matrix code block', line):
        matrix_block_is_found = True
        continue
    if re.search(r'^ *// @end matrix code block', line) and matrix_block_is_found:
        matrix_block_is_found = False
        for i in range(len(matrix_types)-1):
            instance = re.sub(r'%s' % matrix_types[0][0], r'%s' % matrix_types[i+1][0], container2)
            if re.search(r'\s*int64_t\s*m\s*,\s*int64_t\s*n\s*,\s*int64_t\s*nb\s*,\s*int\s*p\s*,\s*int\s*q\s*,\s*MPI_Comm\s*mpi_comm\s*', instance):
                instance = re.sub(r'\s*int64_t\s*m\s*,\s*int64_t\s*n\s*,\s*int64_t\s*nb\s*,\s*int\s*p\s*,\s*int\s*q\s*,\s*MPI_Comm\s*mpi_comm\s*', r'%s' % matrix_types[i+1][1], instance)
            if re.search(r'\s*m\s*,\s*n\s*,\s*nb\s*,\s*p\s*,\s*q\s*,\s*mpi_comm\s*', instance):
                s = re.sub('int64_t ', '', matrix_types[i+1][1])
                s = re.sub('MPI_Comm ', '', s)
                s = re.sub('int ', '', s)
                s = re.sub('slate_Uplo\s*uplo', 'slate::uplo2cpp(uplo)', s)
                s = re.sub('slate_Diag\s*diag', 'slate::diag2cpp(diag)', s)
                instance = re.sub(r'\s*m\s*,\s*n\s*,\s*nb\s*,\s*p\s*,\s*q\s*,\s*mpi_comm\s*', r'%s' % s, instance)
            functions.append(instance)
        container2 = ''
        continue
    if re.search(r'^ *// @begin function', line):
        function_is_found = True
        continue
    if re.search(r'^ *// @end function', line):
        function_is_found = False
        functions.append(container)
        container = ''
        continue
    if (function_is_found):
        container += line
    if (function_is_found and matrix_block_is_found):
        container2 += line

contents = ''
for function in functions:
    for i in range(len(data_types)-1):
        instance = re.sub(r'%s' % data_types[len(data_types)-1][1], r'%s' % data_types[i][1], function)
        instance = re.sub(r'%s' % data_types[len(data_types)-1][0], r'%s' % data_types[i][0], instance)
        instance = re.sub(r'%s' % data_types[len(data_types)-1][2], r'%s' % data_types[i][2], instance)
        if data_types[i][0] == 'float _Complex' or data_types[i][0] == 'float':
            instance = re.sub(r'double', r'float', instance)
        contents += instance
file.close()

print('''\
//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the University of Tennessee nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------
// This research was supported by the Exascale Computing Project (17-SC-20-SC),
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//------------------------------------------------------------------------------
// For assistance with SLATE, email <slate-user@icl.utk.edu>.
// You can also join the "SLATE User" Google group by going to
// https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user,
// signing in with your Google credentials, and then clicking "Join group".
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Auto-generated file by %s
''' % sys.argv[0])

print('''#include "slate/c_api/wrappers.h"''')
print('''#include "slate/c_api/util.hh"''')

print('')

print contents
