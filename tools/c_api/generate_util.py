#!/usr/bin/env python3

import sys
import re

enum       = []
enums      = []
enum_list  = []
enum_name  = ''
enum_start = False
pseudo_enum = False

file = open(sys.argv[1], 'r')
for line in file:
    s = re.search(r'^typedef\s*enum\s*(\w+)', line)
    if s and (not enum_start):
        enum_name  = s.group(1)
        enum_start = True
        pseudo_enum = False
        enum.append(enum_name)
        continue
    s = re.search(r'^typedef\s*\w+\s*(\w+)\s*;\s*/\* enum \*/', line)
    if s and (not enum_start):
        enum_name = s.group(1)
        enum_start = True
        pseudo_enum = True
        enum.append(enum_name)
        continue
    if enum_start:
        if pseudo_enum:
            end_regex = r'^//\s*end\s*%s'
        else:
            end_regex = r'^\}\s*%s\s*;'
        if re.search(end_regex % enum_name, line):
            enum_start = False
            enum.append(enum_list)
            enums.append(enum)
            enum      = []
            enum_list = []
            continue
        if pseudo_enum:
            line = re.search('^const\s*\w+\s*(\w+)\s*=', line).group(1)
        else:
            line = line.split(',')[0]
            line = line.split('=')[0]
            line = line.strip()
        enum_list.append(line)

file.close()

file_hh = open(sys.argv[2], 'w')
file_cc = open(sys.argv[3], 'w')

copyright = '''\
// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

//------------------------------------------------------------------------------
// Auto-generated file by ''' + sys.argv[0] + '\n'

file_hh.write(copyright)
file_hh.write('''\
#ifndef SLATE_C_API_UTIL_HH
#define SLATE_C_API_UTIL_HH
\n''')

file_hh.write('#include "slate/c_api/types.h"\n')
file_hh.write('#include "slate/c_api/matrix.h"\n')
file_hh.write('#include "slate/slate.hh"\n')
file_hh.write('\n')

file_cc.write(copyright + '\n')
file_cc.write('#include "c_api/util.hh"\n\n')

file_hh.write('namespace slate {\n\n')
file_cc.write('namespace slate {\n\n')

for e in enums:
    prefix = e[0].replace('slate_', '')
    var = e[0].replace('slate_', '').lower()

    # Write assertions
    file_cc.write('static_assert(sizeof(' + e[0] + ') == sizeof(' + prefix + '), "C API types are out of sync with C++ API types");\n')
    for i in e[1]:
        i_cpp = prefix + '::' + i.replace('slate_' + prefix + '_', '')
        file_cc.write('static_assert(' + i + ' == ' + e[0] + '(' + i_cpp + '), "C API constants are out of sync with C++ API constants");\n')

    # Write conversion functions
    instance  = prefix + ' '
    instance += var
    instance += '2cpp(' + e[0] + ' ' + var + ')'
    file_hh.write(instance + ';\n')
    file_cc.write(instance + '{\n')
    file_cc.write('    switch (' + var + ') {\n')
    for i in e[1]:
        file_cc.write('        case ' + i + ': return ' + prefix + '::' + i.replace('slate_' + prefix + '_', '') + ';\n')
    file_cc.write('        default: throw Exception("unknown %s");' % var)
    file_cc.write('\n    }\n}\n')

file_hh.write('Options options2cpp( slate_Options opts );\n')
file_cc.write('Options options2cpp( slate_Options opts )\n')
file_cc.write('{\n')
file_cc.write('    if (opts == nullptr) {\n')
file_cc.write('        return { };\n')
file_cc.write('    }\n')
file_cc.write('    else {\n')
file_cc.write('        return *static_cast<slate::Options*>( opts );\n')
file_cc.write('    }\n')
file_cc.write('}\n')

file_hh.write('\n} // namespace slate\n\n')
file_cc.write('\n} // namespace slate\n\n')

file_hh.write('#endif // SLATE_C_API_UTIL_HH\n')

file_hh.close()
file_cc.close()
