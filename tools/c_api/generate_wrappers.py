#!/usr/bin/env python

import sys
import re

data_types = [
    ['float',           '_r32', 'float'],
    ['double',          '_r64', 'double'],
    ['float _Complex',  '_c32', 'std::complex<float>'],
    ['double _Complex', '_c64', 'std::complex<double>'],
]

def gen_precisions(f):
    contents = ''
    for i in range(len(data_types)-1):
        instance = re.sub(r'%s' % data_types[len(data_types)-1][1], r'%s' % data_types[i][1], f)
        instance = re.sub(r'%s' % data_types[len(data_types)-1][0], r'%s' % data_types[i][0], instance)
        instance = re.sub(r'%s' % data_types[len(data_types)-1][2], r'%s' % data_types[i][2], instance)
        if data_types[i][0] == 'float _Complex' or data_types[i][0] == 'float':
            instance = re.sub(r'double', r'float', instance)
        contents += instance
    return contents

function_is_found = False
header_is_found   = False
container2        = ''
container         = ''
functions         = []
headers           = []

file = open(sys.argv[1], 'r')
for line in file:
    if re.search(r'^ *// @begin function', line):
        function_is_found = True
        continue
    if re.search(r'^void', line) or re.search(r'^double\s*slate_(.*?)norm_', line):
        header_is_found = True

    if re.search(r'\s*int\s*num_opts\s*,\s*slate_Options\s*opts\s*\[\s*\]\s*\)', line):
        header_is_found = False
        container2 += line.replace('\n', ';\n')
        headers.append(container2)
        container2 = ''
    if re.search(r'^ *// @end function', line):
        function_is_found = False
        functions.append(container)
        container = ''
        continue
    if (function_is_found):
        container += line
    if (header_is_found):
        container2 += line
file.close()

contents = ''
for function in functions:
    contents += gen_precisions(function)

contents2 = ''
for header in headers:
    contents2 += gen_precisions(header)
    contents2 += header + '\n'

file_hh = open('include/slate/c_api/wrappers.h',   'w')
file_cc = open('src/c_api/wrappers_precisions.cc', 'w')

copyright = '''\
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
// Auto-generated file by ''' + sys.argv[0] + '\n'

file_hh.write(copyright)
file_hh.write('''\
#ifndef SLATE_C_API_WRAPPERS_H
#define SLATE_C_API_WRAPPERS_H
\n''')

file_hh.write('#include "slate/c_api/matrix.h"\n')
file_hh.write('#include "slate/c_api/types.h"\n\n')

file_hh.write('#include <complex.h>\n')
file_hh.write('#include <stdbool.h>\n')
file_hh.write('#include <stdint.h>\n')
file_hh.write('\n')

file_cc.write(copyright + '\n')
file_cc.write('#include "slate/c_api/wrappers.h"\n')
file_cc.write('#include "slate/c_api/util.hh"\n')
file_cc.write('#include "slate/slate.hh"\n\n')

file_hh.write('''\
#ifdef __cplusplus
extern "C" {
#endif
\n''')

file_cc.write(contents)
file_hh.write(contents2)

file_hh.write('''\
#ifdef __cplusplus
}  // extern "C"
#endif
\n''')

file_hh.write('#endif // SLATE_C_API_WRAPPERS_H')

file_hh.close()
file_cc.close()
