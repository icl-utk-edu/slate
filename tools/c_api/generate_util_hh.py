#!/usr/bin/env python

import sys
import re

file = open(sys.argv[1], 'r')

typenames = []
for line in file:
        s = re.search(r'^typedef\s*enum\s*(\w+)', line)
        if (s):
            typenames.append(s.group(1))
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

#ifndef SLATE_C_API_UTIL_HH
#define SLATE_C_API_UTIL_HH
''' % sys.argv[0])

print('''#include "slate/c_api/types.h"''')
print('''#include "slate/slate.hh"''')

print('')

print('''namespace slate {\n''')

# print('//' + ('-'*78))

# print('')

for typename in typenames:
    instance  = typename.replace('slate_', '') + ' '
    instance += typename.replace('slate_', '').lower() + '2cpp('
    instance += typename + ' ' + typename.replace('slate_', '').lower() + ');'
    print instance + '\n'

# print('//' + ('-'*78))

# print('')

print('''std::pair<Option, OptionValue> optionvalue2cpp(slate_Option option, slate_OptionValue option_value);''')

print('')

print('''void options2cpp(int num_options, slate_Options options[], Options& options_);''')

# print('')

# print('//' + ('-'*78))

print('')

print('''} // namespace slate''')

print('')

print('''#endif // SLATE_C_API_UTIL_HH''')
