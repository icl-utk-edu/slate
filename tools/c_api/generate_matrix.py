#!/usr/bin/env python3

import sys
import re

data_members_is_found = False
typename_is_found     = False
templates             = []
template              = 'typedef struct\n{\n'
typename              = None
name                  = 'Tile'

file = open(sys.argv[1], 'r')
for line in file:
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
    if re.search(r'^ *// @begin data members', line):
        data_members_is_found = True
        continue
    if re.search(r'^ *// @end data members', line):
        data_members_is_found = False
        continue
    if re.search(r'\S', line) and data_members_is_found:
        template += line
file.close()

template += '} slate_' + name + '@SUFFIX;\n'
templates.append([name, typename, template])

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
file_hh.write('''\n\
#ifndef SLATE_C_API_MATRIX_H
#define SLATE_C_API_MATRIX_H
\n''')

file_hh.write('#include "mpi.h"\n') # todo slate must be compiled with mpi
file_hh.write('#include "slate/c_api/types.h"\n\n')

file_hh.write('#include <complex.h>\n')
file_hh.write('#include <stdbool.h>\n')
file_hh.write('#include <stdint.h>\n')
file_hh.write('\n')

file_cc.write(copyright + '\n')
file_cc.write('#include "slate/c_api/matrix.h"\n')
file_cc.write('#include "slate/c_api/util.hh"\n')
file_cc.write('#include "slate/slate.hh"\n\n')

file_hh.write('''\
#ifdef __cplusplus
extern "C" {
#endif
\n''')

data_types = [
    ['float',           '_r32', 'float'],
    ['double',          '_r64', 'double'],
    ['float _Complex',  '_c32', 'std::complex<float>'],
    ['double _Complex', '_c64', 'std::complex<double>'],
]

tile_routines = [
    ['int64_t', 'mb'],
    ['int64_t', 'nb'],
    ['int64_t', 'stride'],
    [typename,  'data'],
]

keywords = [
    'Op', 'Uplo', 'TileKind', 'Layout'
]

index_data_type_cpp = 2
index_routine_name  = 1
index_routine_ret   = 0
index_data_type     = 0
index_typename      = 1
index_template      = 2
index_suffix        = 1
index_name          = 0
for template in templates:
    for type in data_types:
        instance = template[index_template]
        instance = re.sub(template[index_typename], type[index_data_type],
                          instance)
        for keyword in keywords:
            instance = re.sub(r'%s' % keyword, 'slate_' + keyword, instance)
        instance = re.sub(r'@SUFFIX', type[index_suffix], instance)
        file_hh.write('//' + ('-'*78) + '\n')
        file_hh.write('// instantiate ' + template[index_name] + ' for '
              + template[index_typename] + ' = <' + type[index_data_type] + '>\n')
        file_hh.write(instance + '\n')
        for routine in tile_routines:
            t = 'slate::' + name + '<' + type[index_data_type_cpp] + '>'
            s  = '/// ' + t
            s += '::' + routine[index_routine_name] + '()\n'
            if (routine[index_routine_ret] == typename):
                ret = routine[index_routine_ret].replace(
                      routine[index_routine_ret], type[index_data_type])
                ret += '*'
            else:
                ret = routine[index_routine_ret]
            s += ret + ' slate_' + name + '_' + routine[index_routine_name]
            s += type[index_suffix] + '(slate_' + name + type[index_suffix]
            s += ' T)'
            file_hh.write(s + ';\n\n')
            file_cc.write(s + '\n{\n')
            file_cc.write('    assert(sizeof(slate_Tile_c64) == sizeof(' + t +'));\n')
            file_cc.write('    ' + t + ' T_;\n')
            file_cc.write('    std::memcpy(&T_, &T, sizeof(' + t + '));\n')
            # file_cc.write('    auto T_ = *reinterpret_cast<' + t +'*>(&T);\n')
            file_cc.write('    return(')
            if (routine[index_routine_ret] == typename):
                file_cc.write('('+ type[index_data_type] +'*)')
            file_cc.write('T_.'+ routine[index_routine_name] + '());\n')
            file_cc.write('}\n')

file_hh.write('//' + ('-'*78) + '\n')

matrix_types = [
    ['Matrix',               '(int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)',                                    '(int64_t m, int64_t n, scalar_t* A, int64_t lda, int64_t mb, int64_t nb, int p, int q, MPI_Comm mpi_comm)'],
    ['BandMatrix',           '(int64_t m, int64_t n, int64_t kl, int64_t ku, int64_t nb, int p, int q, MPI_Comm mpi_comm)',            ''],
    ['HermitianMatrix',      '(slate_Uplo uplo, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)',                              '(slate_Uplo uplo, int64_t n, scalar_t* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm)'],
    ['HermitianBandMatrix',  '(slate_Uplo uplo, int64_t n, int64_t kd, int64_t nb, int p, int q, MPI_Comm mpi_comm)',                  ''],
    ['TriangularMatrix',     '(slate_Uplo uplo, slate_Diag diag, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)',             '(slate_Uplo uplo, slate_Diag diag, int64_t n, scalar_t* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm)'],
    ['TriangularBandMatrix', '(slate_Uplo uplo, slate_Diag diag, int64_t n, int64_t kd, int64_t nb, int p, int q, MPI_Comm mpi_comm)', ''],
    ['SymmetricMatrix',      '(slate_Uplo uplo, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)',                              '(slate_Uplo uplo, int64_t n, scalar_t* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm)'],
    ['TrapezoidMatrix',      '(slate_Uplo uplo, slate_Diag diag, int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)',  '(slate_Uplo uplo, slate_Diag diag, int64_t m, int64_t n, scalar_t* A, int64_t lda, int64_t nb, int p, int q, MPI_Comm mpi_comm)'],
]

matrix_routines = [
    ['slate_Matrix', '_create',                            '',                                                              ''],
    ['slate_Matrix', '_create_fortran',                    '',                                                              ''],
    ['slate_Matrix', '_create_fromScaLAPACK',              '',                                                              ''],
    ['slate_Matrix', '_create_fromScaLAPACK_fortran',      '',                                                              ''],
    ['slate_Matrix', '_create_slice',                      'slate_Matrix, int64_t i1, int64_t i2, int64_t j1, int64_t j2',  'slice(i1, i2, j1, j2)'],
    ['void',         '_destroy',                           'slate_Matrix',                                                  'delete'],
    ['void',         '_insertLocalTiles',                  'slate_Matrix',                                                  'insertLocalTiles()'],
    ['int64_t',      '_mt',                                'slate_Matrix',                                                  'mt()'],
    ['int64_t',      '_nt',                                'slate_Matrix',                                                  'nt()'],
    ['int64_t',      '_m',                                 'slate_Matrix',                                                  'm()'],
    ['int64_t',      '_n',                                 'slate_Matrix',                                                  'n()'],
    ['bool',         '_tileIsLocal',                       'slate_Matrix, int64_t i, int64_t j',                            'tileIsLocal(i, j)'],
    ['slate_Tile',   '_at',                                'slate_Matrix, int64_t i, int64_t j',                            'at(i, j)'],
    ['void',         '_transpose_in_place',                'slate_Matrix',                                                  'transpose( *A_ )'],
    ['void',         '_conj_transpose_in_place',           'slate_Matrix',                                                  'conj_transpose( *A_ )'],
]

contents = ''
contents0 = ''
for matrix_type in matrix_types:
    for data_type in data_types:
        contents += '/// slate::' + matrix_type[0] + '<' + data_type[2] + '>\n'
        contents += 'struct slate_' + matrix_type[0] + '_struct' + data_type[1]
        contents += ';\ntypedef struct slate_' + matrix_type[0] + '_struct'
        contents += data_type[1] + '* slate_' + matrix_type[0] + data_type[1]
        contents += ';\n\n'

        for routine in matrix_routines:
            # todo
            if matrix_type[0] == 'BandMatrix' and routine[1] == '_insertLocalTiles':
                continue
            # todo
            if matrix_type[0] != 'Matrix' and routine[1] == '_create_slice':
                continue
            # todo
            if 'Band' in matrix_type[0] and routine[1] == '_create_fromScaLAPACK':
                continue
            # todo
            if 'Band' in matrix_type[0] and routine[1] == '_create_fromScaLAPACK_fortran':
                continue
            ret = routine[0]
            if routine[0] == 'slate_Matrix':
                ret = 'slate_' + matrix_type[0] + data_type[1]
            elif routine[0] == 'slate_Tile':
                ret = routine[0] + data_type[1]
            routine_name = 'slate_' + matrix_type[0] + routine[1] + data_type[1]
            params = '(' + routine[2] + ')'
            if 'slate_Matrix' in routine[2]:
                s = re.sub('slate_Matrix', 'slate_' + matrix_type[0] + data_type[1] +  ' A', routine[2])
                params = '(' + s + ')'
            elif routine[1] == '_create':
                params = matrix_type[1]
            elif routine[1] == '_create_fortran':
                params = matrix_type[1]
                params = re.sub('MPI_Comm', 'MPI_Fint', params)
            elif routine[1] == '_create_fromScaLAPACK':
                params = matrix_type[2]
                params = re.sub('scalar_t', data_type[0], params)
            elif routine[1] == '_create_fromScaLAPACK_fortran':
                params = matrix_type[2]
                params = re.sub('scalar_t', data_type[0], params)
                params = re.sub('MPI_Comm', 'MPI_Fint', params)
            contents  += ret + ' ' + routine_name + params + ';\n\n'
            contents0 += ret + ' ' + routine_name + params + '\n{\n'
            if routine[1] != '_create' and routine[1] != '_create_fromScaLAPACK' and routine[1] != '_create_fortran' and routine[1] != '_create_fromScaLAPACK_fortran':
                contents0 += '    auto* A_ = reinterpret_cast<slate::' + matrix_type[0] + '<' + data_type[2] + '>' + '*>(A);\n'
                if routine[0] != 'void':
                    if routine[0] != 'slate_Tile' and routine[0] != 'slate_Matrix':
                        contents0 += '    return(A_->' + routine[3] + ');\n'
                    elif routine[0] == 'slate_Matrix' and routine[1] == '_create_slice':
                        contents0 += '    auto* A_slice = new slate::' + matrix_type[0] + '<' + data_type[2] + '>(A_->' + routine[3] + ');\n'
                        contents0 += '    return reinterpret_cast<slate_' + matrix_type[0] +  data_type[1] + '>(A_slice);\n'
                    else:
                        contents0 += '    slate::Tile<' +  data_type[2] + '> T = A_->at(i, j);\n'
                        contents0 += '    slate_Tile' +  data_type[1] + ' T_;\n'
                        contents0 += '    std::memcpy(&T_, &T, sizeof(slate::Tile<' +  data_type[2] + '>));\n'
                        contents0 += '    return(T_);\n'
                        # contents0 += '    slate::Tile<' +  data_type[2] + '> T = A_->at(i, j);\n'
                        # contents0 += '    return(*reinterpret_cast<slate_Tile' +  data_type[1] + '*>(&T));\n'
                else:
                    if routine[1] == '_destroy':
                        contents0 += '    ' + routine[3] + ' A_;\n'
                    elif (routine[1] == '_conj_transpose_in_place') or (routine[1] == '_transpose_in_place'):
                        contents0 += '    *A_ = slate::' + routine[3] + ';\n'
                    else:
                        contents0 += '    A_->' + routine[3] + ';\n'
            else:
                if routine[1] == '_create':
                    s = re.sub('int64_t ', '', matrix_type[1])
                    s = re.sub('MPI_Comm ', '', s)
                    s = re.sub('int ', '', s)
                    s = re.sub('slate_Uplo\s*uplo', 'slate::uplo2cpp(uplo)', s)
                    s = re.sub('slate_Diag\s*diag', 'slate::diag2cpp(diag)', s)
                    contents0 += '    auto* A = new ' + 'slate::' + matrix_type[0] + '<' + data_type[2] + '>' + s + ';\n'
                    contents0 += '    return reinterpret_cast<slate_' + matrix_type[0] + data_type[1] + '>(A);\n'
                if routine[1] == '_create_fortran':
                    s = re.sub('int64_t ', '', matrix_type[1])
                    s = re.sub('MPI_Comm ', '', s)
                    s = re.sub('int ', '', s)
                    s = re.sub('slate_Uplo\s*uplo', 'slate::uplo2cpp(uplo)', s)
                    s = re.sub('slate_Diag\s*diag', 'slate::diag2cpp(diag)', s)
                    s = re.sub('mpi_comm', 'MPI_Comm_f2c(mpi_comm)', s)
                    contents0 += '    auto* A = new ' + 'slate::' + matrix_type[0] + '<' + data_type[2] + '>' + s + ';\n'
                    contents0 += '    return reinterpret_cast<slate_' + matrix_type[0] + data_type[1] + '>(A);\n'
                elif routine[1] == '_create_fromScaLAPACK':
                    s = re.sub('int64_t ', '', matrix_type[2])
                    s = re.sub('MPI_Comm ', '', s)
                    s = re.sub('int ', '', s)
                    s = re.sub('slate_Uplo\s*uplo', 'slate::uplo2cpp(uplo)', s)
                    s = re.sub('slate_Diag\s*diag', 'slate::diag2cpp(diag)', s)
                    s = re.sub('scalar_t\*\s*A', '(' + data_type[2] +'*)A', s)
                    contents0 += '    auto* A_ = new ' + 'slate::' + matrix_type[0] + '<' + data_type[2] + '>();\n'
                    contents0 += '    (*A_) = slate::' + matrix_type[0] + '<' + data_type[2] + '>' + '::fromScaLAPACK' + s + ';\n'
                    contents0 += '    return reinterpret_cast<slate_' + matrix_type[0] + data_type[1] + '>(A_);\n'
                elif routine[1] == '_create_fromScaLAPACK_fortran':
                    s = re.sub('int64_t ', '', matrix_type[2])
                    s = re.sub('MPI_Comm ', '', s)
                    s = re.sub('int ', '', s)
                    s = re.sub('slate_Uplo\s*uplo', 'slate::uplo2cpp(uplo)', s)
                    s = re.sub('slate_Diag\s*diag', 'slate::diag2cpp(diag)', s)
                    s = re.sub('scalar_t\*\s*A', '(' + data_type[2] +'*)A', s)
                    s = re.sub('mpi_comm', 'MPI_Comm_f2c(mpi_comm)', s)
                    contents0 += '    auto* A_ = new ' + 'slate::' + matrix_type[0] + '<' + data_type[2] + '>();\n'
                    contents0 += '    (*A_) = slate::' + matrix_type[0] + '<' + data_type[2] + '>' + '::fromScaLAPACK' + s + ';\n'
                    contents0 += '    return reinterpret_cast<slate_' + matrix_type[0] + data_type[1] + '>(A_);\n'
            contents0 += '}\n'
        # if 'uplo' in matrix_type[1]:
        #     ret = 'slate_Uplo'
        #     routine_name = 'slate_' + matrix_type[0] + '_uplo' + data_type[1]
        #     params = '(' + 'slate_' + matrix_type[0] + data_type[1] +  ' A' + ')'
        #     contents  += ret + ' ' + routine_name + params + ';\n\n'
        #     contents0 += ret + ' ' + routine_name + params + '\n{\n'
        #     contents0 += '    auto* A_ = reinterpret_cast<slate::' + matrix_type[0] + '<' + data_type[2] + '>' + '*>(A);\n'
        #     contents0 += '    return(A_->' + 'uplo()' + ');\n'
        #     contents0 += '}\n'
        # if 'diag' in matrix_type[1]:
        #     ret = 'slate_Diag'
        #     routine_name = 'slate_' + matrix_type[0] + '_diag' + data_type[1]
        #     params = '(' + 'slate_' + matrix_type[0] + data_type[1] +  ' A' + ')'
        #     contents  += ret + ' ' + routine_name + params + ';\n\n'
        #     contents0 += ret + ' ' + routine_name + params + '\n{\n'
        #     contents0 += '    auto* A_ = reinterpret_cast<slate::' + matrix_type[0] + '<' + data_type[2] + '>' + '*>(A);\n'
        #     contents0 += '    return(A_->' + 'diag()' + ');\n'
        #     contents0 += '}\n'
    contents  += '//' + ('-'*78) + '\n'
    contents0 += '//' + ('-'*78) + '\n'

file_hh.write(contents)
file_cc.write(contents0)

for type in data_types:
    file_hh.write('/// slate::TriangularFactors<' + type[2] + '>\n')
    file_hh.write('struct slate_TriangularFactors_struct' + type[1] + ';\n')
    file_hh.write('typedef struct slate_TriangularFactors_struct' + type[1] + '* slate_TriangularFactors' + type[1] +';\n\n')
    file_hh.write('slate_TriangularFactors' + type[1] + ' slate_TriangularFactors_create'  + type[1] +'();\n')
    file_hh.write('void slate_TriangularFactors_destroy' + type[1] +'(slate_TriangularFactors' + type[1] + ' T);\n\n')

    file_cc.write('/// slate::TriangularFactors<' + type[2] + '>\n')
    file_cc.write('slate_TriangularFactors' + type[1] + ' slate_TriangularFactors_create'  + type[1] +'()\n')
    file_cc.write('{\n')
    file_cc.write('    auto* T = new slate::TriangularFactors<' + type[2] + '>();\n')
    file_cc.write('    return reinterpret_cast<slate_TriangularFactors' + type[1] + '>(T);\n')
    file_cc.write('}\n')
    file_cc.write('void slate_TriangularFactors_destroy' + type[1] +'(slate_TriangularFactors' + type[1] + ' T)\n')
    file_cc.write('{\n')
    file_cc.write('    auto* T_ = reinterpret_cast<slate::TriangularFactors<' + type[2] + '>*>(T);\n')
    file_cc.write('    delete T_;\n')
    file_cc.write('}\n')

file_hh.write('//' + ('-'*78) + '\n')
file_hh.write('/// slate::Pivots\n')
file_hh.write('struct slate_Pivots_struct;\n')
file_hh.write('typedef struct slate_Pivots_struct* slate_Pivots;\n\n')
file_hh.write('slate_Pivots slate_Pivots_create();\n')
file_hh.write('void slate_Pivots_destroy(slate_Pivots pivots);\n\n')

file_cc.write('/// slate::Pivots\n')
file_cc.write('slate_Pivots slate_Pivots_create()\n')
file_cc.write('{\n')
file_cc.write('    auto* pivots = new slate::Pivots();\n')
file_cc.write('    return reinterpret_cast<slate_Pivots>(pivots);\n')
file_cc.write('}\n')
file_cc.write('void slate_Pivots_destroy(slate_Pivots pivots)\n')
file_cc.write('{\n')
file_cc.write('    auto* pivots_ = reinterpret_cast<slate::Pivots*>(pivots);\n')
file_cc.write('    delete pivots_;\n')
file_cc.write('}\n')

file_hh.write('''\
#ifdef __cplusplus
}  // extern "C"
#endif
\n''')

file_hh.write('#endif // SLATE_C_API_MATRIX_H')

file_hh.close()
file_cc.close()
