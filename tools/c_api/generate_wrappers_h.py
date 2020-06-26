#!/usr/bin/env python

import sys
import re

data_types = [
    ['float',           '_r32', 'float'],
    ['double',          '_r64', 'double'],
    ['float _Complex',  '_c32', 'std::complex<float>'],
    ['double _Complex', '_c64', 'std::complex<double>'],
]

file = open(sys.argv[1], 'r')

matrix_block_is_found = False
function_is_found = False
function_found = 0
function_counter = 0
functions = []
container = ''
for line in file:
    if re.search(r'^void', line):
        function_is_found = True
    if re.search(r'\);', line):
        function_is_found = False
        container += line
        functions.append(container)
        container = ''
    if (function_is_found):
        container += line

contents = ''
for function in functions:
    for i in range(len(data_types)-1):
        instance = re.sub(r'%s' % data_types[len(data_types)-1][1], r'%s' % data_types[i][1], function)
        instance = re.sub(r'%s' % data_types[len(data_types)-1][0], r'%s' % data_types[i][0], instance)
        instance = re.sub(r'%s' % data_types[len(data_types)-1][2], r'%s' % data_types[i][2], instance)
        if data_types[i][0] == 'float _Complex' or data_types[i][0] == 'float':
            instance = re.sub(r'double', r'float', instance)
        contents += instance + '\n'
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

#ifndef SLATE_C_API_WRAPPERS_PRECISIONS_H
#define SLATE_C_API_WRAPPERS_PRECISIONS_H
''' % sys.argv[0])

print('''#include "slate/c_api/types.h"''')
print('''#include "slate/c_api/Matrix.hh"''')

print('')

# print('''#include <stdbool.h>''')
# print('''#include <stdint.h>''')
# print('''#include <complex.h>''')

print('')

print('''#ifdef __cplusplus\nextern "C" {\n#endif\n''')

print('//' + ('-'*78))

print contents

print('''#ifdef __cplusplus\n}  // extern "C"\n#endif\n''')
print('''#endif // SLATE_C_API_WRAPPERS_PRECISIONS_H''')
