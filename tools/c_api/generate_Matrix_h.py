#!/usr/bin/env python

import sys
import re

scalar_types = [
    ['float',           '_r32', 'float'],
    ['double',          '_r64', 'double'],
    ['float _Complex',  '_c32', 'std::complex<float>'],
    ['double _Complex', '_c64', 'std::complex<double>'],
]

matrix_types = [
    ['Matrix',               '(int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)'],
    ['BandMatrix',           '(int64_t m, int64_t n, int64_t kl, int64_t ku, int64_t nb, int p, int q, MPI_Comm mpi_comm)'],
    ['HermitianMatrix',      '(slate_Uplo uplo, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)'],
    ['HermitianBandMatrix',  '(slate_Uplo uplo, int64_t n, int64_t kd, int64_t nb, int p, int q, MPI_Comm mpi_comm)'],
    ['TriangularMatrix',     '(slate_Uplo uplo, slate_Diag diag, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)'],
    ['TriangularBandMatrix', '(slate_Uplo uplo, slate_Diag diag, int64_t n, int64_t kd, int64_t nb, int p, int q, MPI_Comm mpi_comm)'],
    ['SymmetricMatrix',      '(slate_Uplo uplo, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)'],
    ['TrapezoidMatrix',      '(slate_Uplo uplo, slate_Diag diag, int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)'],
]

contents = ''
for matrix_type in matrix_types:
    for scalar_type in scalar_types:
        contents += '/// slate::' + matrix_type[0] + '<' + scalar_type[2] + '>\n'
        contents += 'struct slate_' + matrix_type[0] + '_struct' + scalar_type[1]
        contents += ';\ntypedef struct slate_' + matrix_type[0] + '_struct'
        contents += scalar_type[1] + '* slate_' + matrix_type[0] + scalar_type[1]
        contents += ';\n\n'
        contents += 'slate_' + matrix_type[0] + scalar_type[1] + ' slate_'
        contents += matrix_type[0] + '_create' + scalar_type[1] + matrix_type[1]
        contents += ';\n\n'
        if matrix_type[0].find("Band") < 0:
            contents += '/// slate::' + matrix_type[0] + '<' + scalar_type[2]
            contents += '>::fromScaLAPACK()\n'
            contents += 'slate_' + matrix_type[0] + scalar_type[1] + ' slate_'
            contents += matrix_type[0] + '_create_fromScaLAPACK' + scalar_type[1]
            contents += matrix_type[1][:matrix_type[1].find('int64_t n,') + len('int64_t n,') + 1]
            contents += scalar_type[0] + '* A, int64_t lda, '
            if matrix_type[0] == "Matrix":
                contents += 'int64_t mb, '
            contents += matrix_type[1][matrix_type[1].find('int64_t nb,'):]
            contents += ';\n\n'

        contents += 'void slate_' + matrix_type[0] + '_destroy' + scalar_type[1] + '(slate_' + matrix_type[0] + scalar_type[1] + ' A);\n\n'
        contents += '/// slate::' + matrix_type[0] + '<' + scalar_type[2]
        contents += '>::insertLocalTiles()\n'
        contents += 'void slate_' + matrix_type[0] + '_insertLocalTiles' + scalar_type[1] + '(slate_' + matrix_type[0] + scalar_type[1] + ' A);\n\n'
        contents += '/// slate::' + matrix_type[0] + '<' + scalar_type[2]
        contents += '>::mt()\n'
        contents += 'int64_t slate_' + matrix_type[0] + '_mt' + scalar_type[1] + '(slate_' + matrix_type[0] + scalar_type[1] + ' A);\n\n'
        contents += '/// slate::' + matrix_type[0] + '<' + scalar_type[2]
        contents += '>::nt()\n'
        contents += 'int64_t slate_' + matrix_type[0] + '_nt' + scalar_type[1] + '(slate_' + matrix_type[0] + scalar_type[1] + ' A);\n\n'
        contents += '/// slate::' + matrix_type[0] + '<' + scalar_type[2]
        contents += '>::m()\n'
        contents += 'int64_t slate_' + matrix_type[0] + '_m' + scalar_type[1] + '(slate_' + matrix_type[0] + scalar_type[1] + ' A);\n\n'
        contents += '/// slate::' + matrix_type[0] + '<' + scalar_type[2]
        contents += '>::n()\n'
        contents += 'int64_t slate_' + matrix_type[0] + '_n' + scalar_type[1] + '(slate_' + matrix_type[0] + scalar_type[1] + ' A);\n\n'
        contents += '/// slate::' + matrix_type[0] + '<' + scalar_type[2]
        contents += '>::tileIsLocal()\n'
        contents += 'bool slate_' + matrix_type[0] + '_tileIsLocal' + scalar_type[1] + '(slate_' + matrix_type[0] + scalar_type[1] + ' A, int64_t i, int64_t j);\n\n'
        contents += '/// slate::' + matrix_type[0] + '<' + scalar_type[2]
        contents += '>::at(i, j)\n'
        contents += 'slate_Tile' + scalar_type[1] + ' slate_' + matrix_type[0] + '_at' + scalar_type[1] + '(slate_' + matrix_type[0] + scalar_type[1] + ' A, int64_t i, int64_t j);\n\n'

    contents += '//' + ('-'*78) + '\n'

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

#ifndef SLATE_C_API_MATRIX_H
#define SLATE_C_API_MATRIX_H
''' % sys.argv[0])

print('''#include "slate/internal/mpi.hh"''')

print('''#include "slate/c_api/Tile.h"''')

print('')

print('''#include <stdbool.h>''')
# print('''#include <stdint.h>''')
# print('''#include <complex.h>''')

print('')

print('''#ifdef __cplusplus\nextern "C" {\n#endif\n''')

print('//' + ('-'*78))

print('/// slate::Pivots\n')
print('struct slate_Pivots_struct;')
print('typedef struct slate_Pivots_struct* slate_Pivots;\n')
print('slate_Pivots slate_Pivots_create();')
print('void slate_Pivots_destroy(slate_Pivots pivots);\n')

print('//' + ('-'*78) + '\n')

for scalar_type in scalar_types:
    print('/// slate::TriangularFactors<' + scalar_type[2] + '>')
    print('struct slate_TriangularFactors_struct' + scalar_type[1] + ';')
    print('typedef struct slate_TriangularFactors_struct' + scalar_type[1] + '* slate_TriangularFactors' + scalar_type[1] +';\n')
    print('slate_TriangularFactors' + scalar_type[1] + ' slate_TriangularFactors_create_'  + scalar_type[1] +'();')
    print('slate_TriangularFactors' + scalar_type[1] + ' slate_TriangularFactors_destroy_' + scalar_type[1] +'(slate_TriangularFactors' + scalar_type[1] + ' T);\n')

print('//' + ('-'*78) + '\n')

print contents

print('''#ifdef __cplusplus\n}  // extern "C"\n#endif\n''')
print('''#endif // SLATE_C_API_MATRIX_H''')
