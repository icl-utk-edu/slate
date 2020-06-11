#!/usr/bin/env python
#
# Example usage:
# help
#     ./run_tests.py -h
#
# run everything with default sizes
# output is redirected; summary information is printed on stderr
#     ./run_tests.py > output.txt
#
# run LU (gesv, getrf, getri, ...), Cholesky (posv, potrf, potri, ...)
# with single, double and default sizes
#     ./run_tests.py --lu --chol --type s,d
#
# run getrf, potrf with small, medium sizes
#     ./run_tests.py -s -m getrf potrf

from __future__ import print_function

import sys
import os
import re
import argparse
import subprocess
import xml.etree.ElementTree as ET
import io

# ------------------------------------------------------------------------------
# command line arguments
parser = argparse.ArgumentParser()

group_test = parser.add_argument_group( 'test' )
group_test.add_argument( '-t', '--test', action='store',
    help='test command to run, e.g., --test "mpirun -np 4 ./test"; default "%(default)s"; see also --np',
    default='./tester' )
group_test.add_argument( '--xml', help='generate report.xml for jenkins' )

group_size = parser.add_argument_group( 'matrix dimensions (default is medium)' )
group_size.add_argument( '-x', '--xsmall', action='store_true', help='run x-small tests' )
group_size.add_argument( '-s', '--small',  action='store_true', help='run small tests' )
group_size.add_argument( '-m', '--medium', action='store_true', help='run medium tests' )
group_size.add_argument( '-l', '--large',  action='store_true', help='run large tests' )
group_size.add_argument(       '--square', action='store_true', help='run square (m = n = k) tests', default=False )
group_size.add_argument(       '--tall',   action='store_true', help='run tall (m > n) tests', default=False )
group_size.add_argument(       '--wide',   action='store_true', help='run wide (m < n) tests', default=False )
group_size.add_argument(       '--mnk',    action='store_true', help='run tests with m, n, k all different', default=False )
group_size.add_argument(       '--dim',    action='store',      help='explicitly specify size', default='' )

group_cat = parser.add_argument_group( 'category (default is all)' )
categories = [
    group_cat.add_argument( '--blas3',         action='store_true', help='run Level 3 BLAS tests' ),
    group_cat.add_argument( '--lu',            action='store_true', help='run LU tests' ),
    group_cat.add_argument( '--lu-band',       action='store_true', help='run band LU tests' ),
    group_cat.add_argument( '--chol',          action='store_true', help='run Cholesky tests' ),
    group_cat.add_argument( '--chol-band',     action='store_true', help='run band Cholesky tests' ),
    group_cat.add_argument( '--sysv',          action='store_true', help='run symmetric indefinite (Aasen) tests' ),
    group_cat.add_argument( '--hesv',          action='store_true', help='run Hermetian indefinite (Aasen) tests' ),
    group_cat.add_argument( '--least-squares', action='store_true', help='run least squares tests' ),
    group_cat.add_argument( '--qr',            action='store_true', help='run QR tests' ),
    group_cat.add_argument( '--lq',            action='store_true', help='run LQ tests' ),
    group_cat.add_argument( '--ql',            action='store_true', help='run QL tests' ),
    group_cat.add_argument( '--rq',            action='store_true', help='run RQ tests' ),
    group_cat.add_argument( '--syev',          action='store_true', help='run symmetric/Hermitian eigenvalue tests' ),
    group_cat.add_argument( '--sygv',          action='store_true', help='run generalized symmetric/Hermitian eigenvalue tests' ),
    group_cat.add_argument( '--geev',          action='store_true', help='run non-symmetric eigenvalue tests' ),
    group_cat.add_argument( '--svd',           action='store_true', help='run SVD tests' ),
    group_cat.add_argument( '--aux',           action='store_true', help='run auxiliary routine tests' ),
    group_cat.add_argument( '--norms',         action='store_true', help='run norm tests' ),
]
# map category objects to category names: ['lu', 'chol', ...]
categories = list( map( lambda x: x.dest, categories ) )

group_opt = parser.add_argument_group( 'options' )
# BLAS and LAPACK
# Empty defaults (check, ref, etc.) use the default in test.cc.
group_opt.add_argument( '--type',   action='store', help='default=%(default)s', default='s,d,c,z' )
group_opt.add_argument( '--transA', action='store', help='default=%(default)s', default='n,t,c' )
group_opt.add_argument( '--transB', action='store', help='default=%(default)s', default='n,t,c' )
group_opt.add_argument( '--trans',  action='store', help='default=%(default)s', default='n,t,c' )
group_opt.add_argument( '--uplo',   action='store', help='default=%(default)s', default='l,u' )
group_opt.add_argument( '--diag',   action='store', help='default=%(default)s', default='n,u' )
group_opt.add_argument( '--side',   action='store', help='default=%(default)s', default='l,r' )
group_opt.add_argument( '--alpha',  action='store', help='', default='' )
group_opt.add_argument( '--beta',   action='store', help='', default='' )
group_opt.add_argument( '--incx',   action='store', help='default=%(default)s', default='1,2,-1,-2' )
group_opt.add_argument( '--incy',   action='store', help='default=%(default)s', default='1,2,-1,-2' )
group_opt.add_argument( '--check',  action='store', help='default=y', default='' )
group_opt.add_argument( '--ref',    action='store', help='default=%(default)s', default='n' )
group_opt.add_argument( '--tol',    action='store', help='default=%(default)s', default='' )

# LAPACK only
group_opt.add_argument( '--direct', action='store', help='default=%(default)s', default='f,b' )
group_opt.add_argument( '--storev', action='store', help='default=%(default)s', default='c,r' )
group_opt.add_argument( '--norm',   action='store', help='default=%(default)s', default='max,1,inf,fro' )
group_opt.add_argument( '--jobz',   action='store', help='default=%(default)s', default='n,v' )
group_opt.add_argument( '--jobvl',  action='store', help='default=%(default)s', default='n,v' )
group_opt.add_argument( '--jobvr',  action='store', help='default=%(default)s', default='n,v' )
group_opt.add_argument( '--jobu',   action='store', help='default=%(default)s', default='n,s,o,a' )
group_opt.add_argument( '--jobvt',  action='store', help='default=%(default)s', default='n,s,o,a' )
group_opt.add_argument( '--kd',     action='store', help='default=%(default)s', default='20,100' )
group_opt.add_argument( '--kl',     action='store', help='default=%(default)s', default='20,100' )
group_opt.add_argument( '--ku',     action='store', help='default=%(default)s', default='20,100' )
group_opt.add_argument( '--matrixtype', action='store', help='default=%(default)s', default='g,l,u' )
group_opt.add_argument( '--itype', action='store', help='default=%(default)s', default='1,2,3' )

# SLATE specific
group_opt.add_argument( '--origin', action='store', help='default=%(default)s', default='s' )
group_opt.add_argument( '--target', action='store', help='default=%(default)s', default='t' )
group_opt.add_argument( '--lookahead', action='store', help='default=%(default)s', default='1' )
group_opt.add_argument( '--dev-dist',  action='store', help='default=%(default)s', default='c,r' )
group_opt.add_argument( '--nb',     action='store', help='default=%(default)s', default='64,100' )
group_opt.add_argument( '--nt',     action='store', help='default=%(default)s', default='5,10,20' )
group_opt.add_argument( '--np',     action='store', help='number of MPI processes; default=%(default)s', default='1' )
group_opt.add_argument( '--p',      action='store', help='use p-by-q MPI process grid', default='' )
group_opt.add_argument( '--q',      action='store', help='use p-by-q MPI process grid', default='' )
group_opt.add_argument( '--repeat', action='store', help='times to repeat each test', default='' )

parser.add_argument( 'tests', nargs=argparse.REMAINDER )
opts = parser.parse_args()

for t in opts.tests:
    if (t.startswith('--')):
        print( 'Error: option', t, 'must come before any routine names' )
        print( 'usage:', sys.argv[0], '[options]', '[routines]' )
        print( '      ', sys.argv[0], '--help' )
        exit(1)

# by default, run medium sizes
if (not (opts.xsmall or opts.small or opts.medium or opts.large)):
    opts.medium = True

# by default, run all shapes
if (not (opts.square or opts.tall or opts.wide or opts.mnk)):
    opts.square = True
    opts.tall   = True
    opts.wide   = True
    opts.mnk    = True

# by default, run all categories
if (opts.tests or not any( map( lambda c: opts.__dict__[ c ], categories ))):
    for c in categories:
        opts.__dict__[ c ] = True

# ------------------------------------------------------------------------------
# parameters
# begin with space to ease concatenation

if (opts.np != '1'):
    if (opts.test != './tester'):
        print('--test overriding --np')
    else:
        opts.test = 'mpirun -np '+ opts.np +' '+ opts.test

# if given, use explicit dim
dim = ' --dim ' + opts.dim if (opts.dim) else ''
n        = dim
tall     = dim
wide     = dim
mn       = dim
mnk      = dim
nk_tall  = dim
nk_wide  = dim
nk       = dim

# for xsmall and small, use smaller nb, but not with medium or large
is_default_nb = (opts.nb == parser.get_default('nb'))

if (not opts.dim):
    if (opts.xsmall):
        n       += ' --dim 10'
        tall    += ' --dim 20x10'
        wide    += ' --dim 10x20'
        mnk     += ' --dim 10x15x20 --dim 15x10x20' \
                +  ' --dim 10x20x15 --dim 15x20x10' \
                +  ' --dim 20x10x15 --dim 20x15x10'
        nk_tall += ' --dim 1x20x10'
        nk_wide += ' --dim 1x10x20'
        if (is_default_nb):
            opts.nb = '5,8'

    if (opts.small):
        n       += ' --dim 25:100:25'
        tall    += ' --dim 50:200:50x25:100:25'  # 2:1
        wide    += ' --dim 25:100:25x50:200:50'  # 1:2
        mnk     += ' --dim 25x50x75 --dim 50x25x75' \
                +  ' --dim 25x75x50 --dim 50x75x25' \
                +  ' --dim 75x25x50 --dim 75x50x25'
        nk_tall += ' --dim 1x50:200:50x25:100:25'
        nk_wide += ' --dim 1x25:100:25x50:200:50'
        if (is_default_nb):
            opts.nb = '25,32'

    if (opts.medium):
        n       += ' --dim 100:500:100'
        tall    += ' --dim 200:1000:200x100:500:100'  # 2:1
        wide    += ' --dim 100:500:100x200:1000:200'  # 1:2
        mnk     += ' --dim 100x300x600 --dim 300x100x600' \
                +  ' --dim 100x600x300 --dim 300x600x100' \
                +  ' --dim 600x100x300 --dim 600x300x100'
        nk_tall += ' --dim 1x200:1000:200x100:500:100'
        nk_wide += ' --dim 1x100:500:100x200:1000:200'
        if (is_default_nb):
            opts.nb = parser.get_default('nb')

    if (opts.large):
        n       += ' --dim 1000:5000:1000'
        tall    += ' --dim 2000:10000:2000x1000:5000:1000'  # 2:1
        wide    += ' --dim 1000:5000:1000x2000:10000:2000'  # 1:2
        mnk     += ' --dim 1000x3000x6000 --dim 3000x1000x6000' \
                +  ' --dim 1000x6000x3000 --dim 3000x6000x1000' \
                +  ' --dim 6000x1000x3000 --dim 6000x3000x1000'
        nk_tall += ' --dim 1x2000:10000:2000x1000:5000:1000'
        nk_wide += ' --dim 1x1000:5000:1000x2000:10000:2000'
        if (is_default_nb):
            opts.nb = parser.get_default('nb')

    mn  = ''
    nk  = ''
    if (opts.square):
        mn = n
        nk = n
    if (opts.tall):
        mn += tall
        nk += nk_tall
    if (opts.wide):
        mn += wide
        nk += nk_wide
    if (opts.mnk):
        mnk = mn + mnk
    else:
        mnk = mn
# end

# BLAS and LAPACK
dtype  = ' --type '   + opts.type   if (opts.type)   else ''
transA = ' --transA ' + opts.transA if (opts.transA) else ''
transB = ' --transB ' + opts.transB if (opts.transB) else ''
trans  = ' --trans '  + opts.trans  if (opts.trans)  else ''
uplo   = ' --uplo '   + opts.uplo   if (opts.uplo)   else ''
diag   = ' --diag '   + opts.diag   if (opts.diag)   else ''
side   = ' --side '   + opts.side   if (opts.side)   else ''
a      = ' --alpha '  + opts.alpha  if (opts.alpha)  else ''
ab     = a+' --beta ' + opts.beta   if (opts.beta)   else a
incx   = ' --incx '   + opts.incx   if (opts.incx)   else ''
incy   = ' --incy '   + opts.incy   if (opts.incy)   else ''
check  = ' --check '  + opts.check  if (opts.check)  else ''
ref    = ' --ref '    + opts.ref    if (opts.ref)    else ''
tol    = ' --tol '    + opts.tol    if (opts.tol)    else ''

# LAPACK only
direct = ' --direct ' + opts.direct if (opts.direct) else ''
storev = ' --storev ' + opts.storev if (opts.storev) else ''
norm   = ' --norm '   + opts.norm   if (opts.norm)   else ''
jobz   = ' --jobz '   + opts.jobz   if (opts.jobz)   else ''
jobu   = ' --jobu '   + opts.jobu   if (opts.jobu)   else ''
jobvt  = ' --jobvt '  + opts.jobvt  if (opts.jobvt)  else ''
jobvl  = ' --jobvl '  + opts.jobvl  if (opts.jobvl)  else ''
jobvr  = ' --jobvr '  + opts.jobvr  if (opts.jobvr)  else ''
kd     = ' --kd '     + opts.kd     if (opts.kd)     else ''
kl     = ' --kl '     + opts.kl     if (opts.kl)     else ''
ku     = ' --ku '     + opts.ku     if (opts.ku)     else ''
mtype  = ' --matrixtype ' + opts.matrixtype if (opts.matrixtype) else ''
itype  = ' --itype '  + opts.itype  if (opts.itype)  else ''

# SLATE specific
origin = ' --origin ' + opts.origin if (opts.origin) else ''
target = ' --target ' + opts.target if (opts.target) else ''
la     = ' --lookahead ' + opts.lookahead if (opts.lookahead) else ''
ddist  = ' --dev-dist  ' + opts.dev_dist  if (opts.dev_dist)  else ''
nb     = ' --nb '     + opts.nb     if (opts.nb)     else ''
nt     = ' --nt '     + opts.nt     if (opts.nt)     else ''
p      = ' --p '      + opts.p      if (opts.p)      else ''
q      = ' --q '      + opts.q      if (opts.q)      else ''
repeat = ' --repeat ' + opts.repeat if (opts.repeat) else ''

# general options for all routines
gen       = origin + target + p + q + check + ref + tol + repeat + nb
gen_no_nb = origin + target + p + q + check + ref + tol + repeat

# ------------------------------------------------------------------------------
# filters a comma separated list csv based on items in list values.
# if no items from csv are in values, returns first item in values.
def filter_csv( values, csv ):
    f = list( filter( lambda x: x in values, csv.split( ',' ) ) )
    if (not f):
        return values[0]
    return ','.join( f )
# end

# ------------------------------------------------------------------------------
# limit options to specific values
dtype_real    = ' --type ' + filter_csv( ('s', 'd'), opts.type )
dtype_complex = ' --type ' + filter_csv( ('c', 'z'), opts.type )
dtype_double  = ' --type ' + filter_csv( ('d', 'z'), opts.type )

trans_nt = ' --trans ' + filter_csv( ('n', 't'), opts.trans )
trans_nc = ' --trans ' + filter_csv( ('n', 'c'), opts.trans )

# positive inc
incx_pos = ' --incx ' + filter_csv( ('1', '2'), opts.incx )
incy_pos = ' --incy ' + filter_csv( ('1', '2'), opts.incy )

# ------------------------------------------------------------------------------
cmds = []

# Level 3
if (opts.blas3):
    cmds += [
    [ 'gbmm',  gen + dtype + la + transA + transB + mnk + ab + kl + ku ],
    [ 'gemm',  gen + dtype + la + transA + transB + mnk + ab ],
    [ 'gemm',  origin + p + q + check + ref + tol + repeat + nb + dtype + la + transA + transB + mnk + ab + ' --gemm-variant=gemmA' + ' --target=t' ],

    [ 'hemm',  gen + dtype         + la + side + uplo     + mn + ab ],
    [ 'hbmm',  gen + dtype         + la + side + uplo     + mn + ab + kd ],
    [ 'herk',  gen + dtype_real    + la + uplo + trans    + mn + ab ],
    [ 'herk',  gen + dtype_complex + la + uplo + trans_nc + mn + ab ],
    [ 'her2k', gen + dtype_real    + la + uplo + trans    + mn + ab ],
    [ 'her2k', gen + dtype_complex + la + uplo + trans_nc + mn + ab ],

    [ 'symm',  gen + dtype         + la + side + uplo     + mn + ab ],
    [ 'syr2k', gen + dtype_real    + la + uplo + trans    + mn + ab ],
    [ 'syr2k', gen + dtype_complex + la + uplo + trans_nt + mn + ab ],
    [ 'syrk',  gen + dtype_real    + la + uplo + trans    + mn + ab ],
    [ 'syrk',  gen + dtype_complex + la + uplo + trans_nt + mn + ab ],

    [ 'tbsm',  gen + dtype + la + side + uplo + transA + diag + mn + a + kd ],
    [ 'trmm',  gen + dtype + la + side + uplo + transA + diag + mn + a ],
    [ 'trsm',  gen + dtype + la + side + uplo + transA + diag + mn + a ],
    ]

# LU
if (opts.lu):
    cmds += [
    [ 'gesv',  gen + dtype + la + n],
    [ 'gesv_nopiv',  gen + dtype + la + n + ' --matrix rand_dominant' + ' --nonuniform_nb y'],
    [ 'getrf', gen + dtype + la + n],  # todo: mn
    [ 'getrf_nopiv', target + p + q + ref + check + repeat + nb + dtype + la + n + ' --matrix rand_dominant'+ ' --nonuniform_nb y'],
    [ 'getrs', gen + dtype + la + n + trans],
    [ 'getrs_nopiv', gen + dtype + la + n + trans + ' --matrix rand_dominant' + ' --nonuniform_nb y'],
    [ 'getri', gen + dtype + la + n ],
    [ 'getriOOP', gen + dtype + la + n ],
    #[ 'gecon', gen + dtype + la + n ],
    #[ 'gerfs', gen + dtype + la + n + trans ],
    #[ 'geequ', gen + dtype + la + n ],
    [ 'gesvMixed',  gen + dtype_double + la + n ],
    ]

# LU banded
if (opts.lu_band):
    cmds += [
    [ 'gbsv',  gen + dtype + la + n  + kl + ku ],
    [ 'gbtrf', gen + dtype + la + n  + kl + ku ],  # todo: mn
    [ 'gbtrs', gen + dtype + la + n  + kl + ku + trans ],
    #[ 'gbcon', gen + dtype + la + n  + kl + ku ],
    #[ 'gbrfs', gen + dtype + la + n  + kl + ku + trans ],
    #[ 'gbequ', gen + dtype + la + n  + kl + ku ],
    ]

# Cholesky
if (opts.chol):
    cmds += [
    [ 'posv',  gen + dtype + la + n + uplo ],
    [ 'potrf', gen + dtype + la + n + uplo + ddist ],
    [ 'potrs', gen + dtype + la + n + uplo ],
    [ 'potri', gen + dtype + la + n + uplo ],
    #[ 'pocon', gen + dtype + la + n + uplo ],
    #[ 'porfs', gen + dtype + la + n + uplo ],
    #[ 'poequ', gen + dtype + la + n ],  # only diagonal elements (no uplo)
    [ 'posvMixed',  gen + dtype_double + la + n + uplo ],
    ]

# Cholesky banded
if (opts.chol):
    cmds += [
    [ 'pbsv',  gen + dtype + la + n + kd + uplo ],
    [ 'pbtrf', gen + dtype + la + n + kd + uplo ],
    [ 'pbtrs', gen + dtype + la + n + kd + uplo ],
    #[ 'pbcon', gen + dtype + la + n + kd + uplo ],
    #[ 'pbrfs', gen + dtype + la + n + kd + uplo ],
    #[ 'pbequ', gen + dtype + la + n + kd + uplo ],
    ]

# symmetric indefinite
if (opts.sysv):
    cmds += [
    #[ 'sysv',  gen + dtype + la + n + uplo ],
    #[ 'sytrf', gen + dtype + la + n + uplo ],
    #[ 'sytrs', gen + dtype + la + n + uplo ],
    #[ 'sytri', gen + dtype + la + n + uplo ],
    #[ 'sycon', gen + dtype + la + n + uplo ],
    #[ 'syrfs', gen + dtype + la + n + uplo ],
    ]

# Hermitian indefinite
if (opts.hesv):
    cmds += [
    [ 'hesv',  gen + dtype + la + n + uplo ],
    [ 'hetrf', gen + dtype + la + n + uplo ],
    [ 'hetrs', gen + dtype + la + n + uplo ],
    #[ 'hetri', gen + dtype + la + n + uplo ],
    #[ 'hecon', gen + dtype + la + n + uplo ],
    #[ 'herfs', gen + dtype + la + n + uplo ],
    ]

# least squares
if (opts.least_squares):
    cmds += [
    # todo: mn (i.e., add wide)
    [ 'gels',   gen + dtype + la + n + tall + trans_nc ],
    #[ 'gelsy',  gen + dtype + la + mn ],
    #[ 'gelsd',  gen + dtype + la + mn ],
    #[ 'gelss',  gen + dtype + la + mn ],
    #[ 'getsls', gen + dtype + la + mn + trans_nc ],

    # Generalized
    #[ 'gglse', gen + dtype + la + mnk ],
    #[ 'ggglm', gen + dtype + la + mnk ],
    ]

# QR
if (opts.qr):
    cmds += [
    [ 'geqrf', gen + dtype + la + mn ],
    #[ 'ggqrf', gen + dtype + la + mnk ],
    #[ 'ungqr', gen + dtype + la + mn ],  # m >= n
    #[ 'unmqr', gen + dtype_real    + la + mnk + side + trans    ],  # real does trans = N, T, C
    #[ 'unmqr', gen + dtype_complex + la + mnk + side + trans_nc ],  # complex does trans = N, C, not T
    ]

# LQ
if (opts.lq):
    cmds += [
    [ 'gelqf', gen + dtype + la + mn ],
    #[ 'gglqf', gen + dtype + la + mn ],
    #[ 'unglq', gen + dtype + la + mn ],  # m <= n, k <= m  TODO Fix the input sizes to match constraints
    #[ 'unmlq', gen + dtype_real    + la + mnk + side + trans    ],  # real does trans = N, T, C
    #[ 'unmlq', gen + dtype_complex + la + mnk + side + trans_nc ],  # complex does trans = N, C, not T
    ]

# QL
if (opts.ql):
    cmds += [
    #[ 'geqlf', gen + dtype + la + mn ],
    #[ 'ggqlf', gen + dtype + la + mn ],
    #[ 'ungql', gen + dtype + la + mn ],
    #[ 'unmql', gen + dtype_real    + la + mnk + side + trans    ],  # real does trans = N, T, C
    #[ 'unmql', gen + dtype_complex + la + mnk + side + trans_nc ],  # complex does trans = N, C, not T
    ]

# RQ
if (opts.rq):
    cmds += [
    #[ 'gerqf', gen + dtype + la + mn ],
    #[ 'ggrqf', gen + dtype + la + mnk ],
    #[ 'ungrq', gen + dtype + la + mnk ],
    #[ 'unmrq', gen + dtype_real    + la + mnk + side + trans    ],  # real does trans = N, T, C
    #[ 'unmrq', gen + dtype_complex + la + mnk + side + trans_nc ],  # complex does trans = N, C, not T
    ]

# symmetric/Hermitian eigenvalues
if (opts.syev):
    cmds += [
    # todo nb, uplo, jobz
    [ 'heev',  gen_no_nb + ' --nb 50' + dtype + la + n ],
    #[ 'ungtr', gen + dtype + la + n + uplo ],
    #[ 'unmtr', gen + dtype_real    + la + mn + uplo + side + trans    ],  # real does trans = N, T, C
    #[ 'unmtr', gen + dtype_complex + la + mn + uplo + side + trans_nc ],  # complex does trans = N, C, not T
    # todo nb, uplo, origin
    [ 'unmtr_he2hb', target + p + q + check + ref + tol + repeat + dtype_real    + ' --nb 50' + ' --origin s' + side + trans    ],  # real does trans = N, T, C
    # todo: include (side=l and trans=c) as well as (side=r and trans=n)
    # [ 'unmtr_he2hb', target + p + q + check + ref + tol + repeat + dtype_complex + ' --nb 50' + ' --origin s' + side + trans_nc ],  # complex does trans = N, C, not T
    [ 'unmtr_he2hb', target + p + q + check + ref + tol + repeat + dtype_complex + ' --nb 50' + ' --origin s' + ' --side l' + ' --trans n' ],
    [ 'unmtr_he2hb', target + p + q + check + ref + tol + repeat + dtype_complex + ' --nb 50' + ' --origin s' + ' --side r' + ' --trans c' ],
    # todo nb, uplo
    [ 'he2hb', gen_no_nb + ' --nb 50' + dtype + n ],
    # sterf doesn't take origin, target, nb, uplo
    [ 'sterf', p + q + check + ref + tol + repeat + dtype + n ],
    [ 'steqr2', p + q + check + ref + tol + repeat + dtype + n ],
    # todo: hb2st

    # Banded
    #[ 'hbev',  gen + dtype + la + n + jobz + uplo ],
    #[ 'ubgtr', gen + dtype + la + n + uplo ],
    #[ 'ubmtr', gen + dtype_real    + la + mn + uplo + side + trans    ],
    #[ 'ubmtr', gen + dtype_complex + la + mn + uplo + side + trans_nc ],
    ]

# generalized symmetric/Hermitian eigenvalues
if (opts.sygv):
    cmds += [
    # [ 'hegv',  gen + dtype + la + n + jobz + itype + uplo ], // todo
    [ 'hegv',  gen + dtype + la + n + itype ],
    [ 'hegst', gen + dtype + la + n + itype + uplo ],
    ]

# non-symmetric eigenvalues
if (opts.geev):
    cmds += [
    #[ 'geev',  gen + dtype + la + n + jobvl + jobvr ],
    #[ 'ggev',  gen + dtype + la + n + jobvl + jobvr ],
    #[ 'geevx', gen + dtype + la + n + balanc + jobvl + jobvr + sense ],
    #[ 'gehrd', gen + dtype + la + n ],
    #[ 'unghr', gen + dtype + la + n ],
    #[ 'unmhr', gen + dtype_real    + la + mn + side + trans    ],  # real does trans = N, T, C
    #[ 'unmhr', gen + dtype_complex + la + mn + side + trans_nc ],  # complex does trans = N, C, not T
    #[ 'trevc', gen + dtype + align + n + side + howmany + select ],
    ]

# svd
if (opts.svd):
    cmds += [
    # todo: mn (wide), nb, jobu, jobvt
    [ 'gesvd', gen_no_nb + ' --nb 50' + dtype + la + n + tall ],
    [ 'ge2tb', gen + dtype + n + tall ],
    # tb2bd, bdsqr don't take origin, target
    [ 'tb2bd', p + q + check + ref + tol + repeat + dtype + n ],
    [ 'bdsqr', p + q + check + ref + tol + repeat + dtype + n + uplo ],
    ]

# norms
if (opts.norms):
    cmds += [
    [ 'genorm', gen + dtype + mn + norm ],
    [ 'henorm', gen + dtype + n  + norm + uplo ],
    [ 'synorm', gen + dtype + n  + norm + uplo ],
    [ 'trnorm', gen + dtype + mn + norm + uplo + diag ],

    # Banded
    [ 'gbnorm', gen + dtype + mn  + kl + ku + norm ],
    [ 'hbnorm', gen + dtype + n   + kd      + norm + uplo ],
    #[ 'sbnorm', gen + dtype + la + n + kd + norm ],
    #[ 'tbnorm', gen + dtype + la + n + kd + norm ],
    ]

# ------------------------------------------------------------------------------
# when output is redirected to file instead of TTY console,
# print extra messages to stderr on TTY console.
output_redirected = not sys.stdout.isatty()

# ------------------------------------------------------------------------------
# if output is redirected, prints to both stderr and stdout;
# otherwise prints to just stdout.
def print_tee( *args ):
    global output_redirected
    print( *args )
    if (output_redirected):
        print( *args, file=sys.stderr )
# end

# ------------------------------------------------------------------------------
# cmd is a pair of strings: (function, args)

def run_test( cmd ):
    cmd = opts.test +' '+ cmd[0] +' '+ cmd[1]
    print_tee( cmd )
    output = ''
    p = subprocess.Popen( cmd.split(), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT )
    p_out = p.stdout
    if (sys.version_info.major >= 3):
        p_out = io.TextIOWrapper(p.stdout, encoding='utf-8')
    # Read unbuffered ("for line in p.stdout" will buffer).
    for line in iter(p_out.readline, ''):
        print( line, end='' )
        output += line
    err = p.wait()
    if (err != 0):
        print_tee( 'FAILED: exit code', err )
    else:
        print_tee( 'pass' )
    return (err, output)
# end

# ------------------------------------------------------------------------------
# run each test
failed_tests = []
passed_tests = []
ntests = len(opts.tests)
run_all = (ntests == 0)

seen = set()
for cmd in cmds:
    if (run_all or cmd[0] in opts.tests):
        seen.add( cmd[0] )
        (err, output) = run_test( cmd )
        if (err):
            failed_tests.append( (cmd[0], err, output) )
        else:
            passed_tests.append( cmd[0] )
not_seen = list( filter( lambda x: x not in seen, opts.tests ) )

if (not_seen):
    print_tee( 'Warning: unknown routines:', ' '.join( not_seen ))

# print summary of failures
nfailed = len( failed_tests )
if (nfailed > 0):
    print_tee( '\n' + str(nfailed) + ' routines FAILED:',
               ', '.join( [x[0] for x in failed_tests] ) )

# generate jUnit compatible test report
if opts.xml:
    print( 'writing XML file', opts.xml )
    root = ET.Element("testsuites")
    doc = ET.SubElement(root, "testsuite",
                        name="slate_suite",
                        tests=str(ntests),
                        errors="0",
                        failures=str(nfailed))

    for (test, err, output) in failed_tests:
        testcase = ET.SubElement(doc, "testcase", name=test)

        failure = ET.SubElement(testcase, "failure")
        if (err < 0):
            failure.text = "exit with signal " + str(-err)
        else:
            failure.text = str(err) + " tests failed"

        system_out = ET.SubElement(testcase, "system-out")
        system_out.text = output
    # end

    for test in passed_tests:
        testcase = ET.SubElement(doc, 'testcase', name=test)
        testcase.text = 'PASSED'

    tree = ET.ElementTree(root)
    tree.write( opts.xml )
# end

exit( nfailed )
