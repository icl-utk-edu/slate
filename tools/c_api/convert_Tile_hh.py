#!/usr/bin/env python

import sys
import re

templates = []
template  = 'typedef struct\n{\n'
typename  = None
name      = 'Tile'

typename_is_found     = False
data_members_is_found = False

for line in open(sys.argv[1]):
    if (typename_is_found):
        if (not re.search(r'^class\s*Tile\s*\{', line)):
            typename_is_found = False
            continue
    else:
        s = re.search(r'^template\s*<\s*typename\s*(\w+)\s*>', line)
        if (s):
            typename = s.group(1)
            typename_is_found = True
            continue

    if (re.search(r'^ *// @begin data members', line)):
        data_members_is_found = True
        continue

    if (re.search(r'^ *// @end data members', line)):
        data_members_is_found = False
        continue

    if (re.search(r'\S', line) and data_members_is_found):
        template += line

template += '} slate_' + name + '@SUFFIX;\n'
templates.append([name, typename, template])

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
/// @file
///
#ifndef SLATE_C_API_TILE_H
#define SLATE_C_API_TILE_H
''')

data_types = [
    ['float',           '_r32', 'float'],
    ['double',          '_r64', 'double'],
    ['float _Complex',  '_c32', 'std::complex<float>'],
    ['double _Complex', '_c64', 'std::complex<double>'],
]

routines = [
    ['int64_t', 'mb'],
    ['int64_t', 'nb'],
    ['int64_t', 'stride'],
    [typename,  'data'],
]

keywords = [
    'Op', 'Uplo', 'TileKind', 'Layout'
]

index_name          = 0
index_typename      = 1
index_template      = 2
index_data_type     = 0
index_suffix        = 1
index_data_type_cpp = 2
index_routine_ret   = 0
index_routine_name  = 1

print('''#include "slate/c_api/enums.h"\n''')
print('''#ifdef __cplusplus\nextern "C" {\n#endif\n''')

for template in templates:
    for type in data_types:
        instance = template[index_template]
        instance = re.sub(template[index_typename], type[index_data_type],
                          instance)
        for keyword in keywords:
            instance = re.sub(r'%s' % keyword, 'slate_' + keyword, instance)
        instance = re.sub(r'@SUFFIX', type[index_suffix], instance)
        print('//' + ('-'*78))
        print('// instantiate ' + template[index_name] + ' for '
              + template[index_typename] + ' = <' + type[index_data_type] + '>')
        print(instance)

        for routine in routines:
            print('/// slate::' + name + '<' + type[index_data_type_cpp]
                  + '>::' + routine[index_routine_name] + '()')
            if (routine[index_routine_ret] == typename):
                ret = routine[index_routine_ret].replace(
                      routine[index_routine_ret], type[index_data_type])
            else:
                ret = routine[index_routine_ret]
            print(ret + '* slate_' + name + '_' + routine[index_routine_name]
                  + type[index_suffix] + '(slate_' + name + type[index_suffix]
                  + ' T);' + '\n')

print('''#ifdef __cplusplus\n}  // extern "C"\n#endif\n''')

print('''#endif // SLATE_C_API_TILE_H''')
