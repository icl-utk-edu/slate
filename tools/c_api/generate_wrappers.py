#!/usr/bin/env python3

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

file_hh = open(sys.argv[2], 'w')
file_cc = open(sys.argv[3], 'w')

copyright = '''\
// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

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
