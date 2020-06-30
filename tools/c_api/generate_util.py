#!/usr/bin/env python

import sys
import re

enum       = []
enums      = []
enum_list  = []
enum_name  = ''
enum_start = False

file = open(sys.argv[1], 'r')
for line in file:
    s = re.search(r'^typedef\s*enum\s*(\w+)', line)
    if s and (not enum_start):
        enum_name  = s.group(1)
        enum_start = True
        enum.append(enum_name)
        continue
    if enum_start:
        if re.search(r'^\}\s*%s\s*;' % enum_name, line):
            enum_start = False
            enum.append(enum_list)
            enums.append(enum)
            enum      = []
            enum_list = []
            continue
        line = line.split(',')[0]
        line = line.split('=')[0]
        line = line.strip()
        enum_list.append(line)
file.close()

file_hh = open('include/slate/c_api/util.hh', 'w')
file_cc = open('src/c_api/util.cc',           'w')

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
#ifndef SLATE_C_API_UTIL_HH
#define SLATE_C_API_UTIL_HH
\n''')

file_hh.write('#include "slate/c_api/types.h"\n')
file_hh.write('#include "slate/slate.hh"\n')
file_hh.write('\n')

file_cc.write(copyright + '\n')
file_cc.write('#include "slate/c_api/util.hh"\n\n')

file_hh.write('namespace slate {\n\n')
file_cc.write('namespace slate {\n\n')

for e in enums:
    prefix = e[0].replace('slate_', '')
    var = e[0].replace('slate_', '').lower()
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

file_hh.write('std::pair<Option, OptionValue> optionvalue2cpp(slate_Option option, slate_OptionValue optionvalue);\n')
file_cc.write('std::pair<Option, OptionValue> optionvalue2cpp(slate_Option option, slate_OptionValue optionvalue)\n')
file_cc.write('{\n')
file_cc.write('    switch (option) {\n')
file_cc.write('        case slate_Option_ChunkSize: return {Option::ChunkSize, optionvalue.chunk_size};\n')
file_cc.write('        case slate_Option_Lookahead: return {Option::Lookahead, optionvalue.lookahead};\n')
file_cc.write('        case slate_Option_BlockSize: return {Option::BlockSize, optionvalue.block_size};\n')
file_cc.write('        case slate_Option_InnerBlocking: return {Option::InnerBlocking, optionvalue.inner_blocking};\n')
file_cc.write('        case slate_Option_MaxPanelThreads: return {Option::MaxPanelThreads, optionvalue.max_panel_threads};\n')
file_cc.write('        case slate_Option_Tolerance: return {Option::Tolerance, optionvalue.tolerance};\n')
file_cc.write('        case slate_Option_Target: return {Option::Target, target2cpp(optionvalue.target)};\n')
file_cc.write('        default: throw Exception("unknown optionvalue");\n')
file_cc.write('    }\n}\n')

file_hh.write('void options2cpp(int num_options, slate_Options options[], Options& options_);\n')
file_cc.write('void options2cpp(int num_options, slate_Options options[], Options& options_)\n')
file_cc.write('{\n')
file_cc.write('    if (options !=  nullptr) {\n')
file_cc.write('        for(int i = 0; i < num_options; ++i) {\n')
file_cc.write('            options_.insert(optionvalue2cpp(options[i].option, options[i].value));\n')
file_cc.write('        }\n    }\n}\n')

file_hh.write('\n} // namespace slate\n\n')
file_cc.write('\n} // namespace slate\n\n')

file_hh.write('#endif // SLATE_C_API_UTIL_HH\n')

file_hh.close()
file_cc.close()
