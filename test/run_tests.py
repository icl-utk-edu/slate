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

# ------------------------------------------------------------------------------
# command line arguments
parser = argparse.ArgumentParser()

group_test = parser.add_argument_group( 'test' )
group_test.add_argument( '-t', '--test', action='store',
    help='test command to run, e.g., --test "mpirun -np 4 ./test"; default "%(default)s"',
    default='./test' )
group_test.add_argument( '--xml', action='store_true', help='generate report.xml for jenkins' )

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
    group_cat.add_argument( '--gb',            action='store_true', help='run GB tests' ),
    group_cat.add_argument( '--gt',            action='store_true', help='run GT tests' ),
    group_cat.add_argument( '--chol',          action='store_true', help='run Cholesky tests' ),
    group_cat.add_argument( '--sysv',          action='store_true', help='run symmetric indefinite (Bunch-Kaufman) tests' ),
    group_cat.add_argument( '--rook',          action='store_true', help='run symmetric indefinite (rook) tests' ),
    group_cat.add_argument( '--aasen',         action='store_true', help='run symmetric indefinite (Aasen) tests' ),
    group_cat.add_argument( '--hesv',          action='store_true', help='run hermetian tests (FIXME more informationhere)' ),
    group_cat.add_argument( '--least-squares', action='store_true', help='run least squares tests' ),
    group_cat.add_argument( '--qr',            action='store_true', help='run QR tests' ),
    group_cat.add_argument( '--lq',            action='store_true', help='run LQ tests' ),
    group_cat.add_argument( '--ql',            action='store_true', help='run QL tests' ),
    group_cat.add_argument( '--rq',            action='store_true', help='run RQ tests' ),
    group_cat.add_argument( '--syev',          action='store_true', help='run symmetric eigenvalues tests' ),
    group_cat.add_argument( '--sygv',          action='store_true', help='run generalized symmetric eigenvalues tests' ),
    group_cat.add_argument( '--geev',          action='store_true', help='run non-symmetric eigenvalues tests' ),
    group_cat.add_argument( '--svd',           action='store_true', help='run svd tests' ),
    group_cat.add_argument( '--aux',           action='store_true', help='run auxiliary tests' ),
    group_cat.add_argument( '--aux-house',     action='store_true', help='run auxiliary Householder tests' ),
    group_cat.add_argument( '--aux-norm',      action='store_true', help='run auxiliary norm tests' ),
    group_cat.add_argument( '--blas',          action='store_true', help='run additional BLAS tests' ),
]
categories = map( lambda x: x.dest, categories ) # map to names: ['lu', 'chol', ...]

group_opt = parser.add_argument_group( 'options' )
# BLAS and LAPACK
group_opt.add_argument( '--type',   action='store', help='default=%(default)s', default='s,d,c,z' )
group_opt.add_argument( '--transA', action='store', help='default=%(default)s', default='n,t,c' )
group_opt.add_argument( '--transB', action='store', help='default=%(default)s', default='n,t,c' )
group_opt.add_argument( '--trans',  action='store', help='default=%(default)s', default='n,t,c' )
group_opt.add_argument( '--uplo',   action='store', help='default=%(default)s', default='l,u' )
group_opt.add_argument( '--diag',   action='store', help='default=%(default)s', default='n,u' )
group_opt.add_argument( '--side',   action='store', help='default=%(default)s', default='l,r' )
group_opt.add_argument( '--alpha',  action='store', help='default=%(default)s', default='' )
group_opt.add_argument( '--beta',   action='store', help='default=%(default)s', default='' )
group_opt.add_argument( '--incx',   action='store', help='default=%(default)s', default='1,2,-1,-2' )
group_opt.add_argument( '--incy',   action='store', help='default=%(default)s', default='1,2,-1,-2' )
group_opt.add_argument( '--check',  action='store', help='default=y', default='' )  # default in test.cc
group_opt.add_argument( '--ref',    action='store', help='default=y', default='' )  # default in test.cc

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

# SLATE specific
group_opt.add_argument( '--lookahead', action='store', help='default=%(default)s', default='1' )
group_opt.add_argument( '--nb',     action='store', help='default=%(default)s', default='10,100' )
group_opt.add_argument( '--nt',     action='store', help='default=%(default)s', default='5,10,20' )
group_opt.add_argument( '--p',      action='store', help='default=%(default)s', default='' )  # default in test.cc
group_opt.add_argument( '--q',      action='store', help='default=%(default)s', default='' )  # default in test.cc

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

    if (opts.small):
        n       += ' --dim 25:100:25'
        tall    += ' --dim 50:200:50x25:100:25'  # 2:1
        wide    += ' --dim 25:100:25x50:200:50'  # 1:2
        mnk     += ' --dim 25x50x75 --dim 50x25x75' \
                +  ' --dim 25x75x50 --dim 50x75x25' \
                +  ' --dim 75x25x50 --dim 75x50x25'
        nk_tall += ' --dim 1x50:200:50x25:100:25'
        nk_wide += ' --dim 1x25:100:25x50:200:50'

    if (opts.medium):
        n       += ' --dim 100:500:100'
        tall    += ' --dim 200:1000:200x100:500:100'  # 2:1
        wide    += ' --dim 100:500:100x200:1000:200'  # 1:2
        mnk     += ' --dim 100x300x600 --dim 300x100x600' \
                +  ' --dim 100x600x300 --dim 300x600x100' \
                +  ' --dim 600x100x300 --dim 600x300x100'
        nk_tall += ' --dim 1x200:1000:200x100:500:100'
        nk_wide += ' --dim 1x100:500:100x200:1000:200'

    if (opts.large):
        n       += ' --dim 1000:5000:1000'
        tall    += ' --dim 2000:10000:2000x1000:5000:1000'  # 2:1
        wide    += ' --dim 1000:5000:1000x2000:10000:2000'  # 1:2
        mnk     += ' --dim 1000x3000x6000 --dim 3000x1000x6000' \
                +  ' --dim 1000x6000x3000 --dim 3000x6000x1000' \
                +  ' --dim 6000x1000x3000 --dim 6000x3000x1000'
        nk_tall += ' --dim 1x2000:10000:2000x1000:5000:1000'
        nk_wide += ' --dim 1x1000:5000:1000x2000:10000:2000'

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
ab     = a + ' --beta ' + opts.beta if (opts.beta)   else ''
incx   = ' --incx '   + opts.incx   if (opts.incx)   else ''
incy   = ' --incy '   + opts.incy   if (opts.incy)   else ''
check  = ' --check '  + opts.check  if (opts.check)  else ''
if (opts.ref):
    check += ' --ref ' + opts.ref

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

# SLATE specific
la     = ' --lookahead ' + opts.lookahead if (opts.lookahead) else ''
nb     = ' --nb '     + opts.nb     if (opts.nb)     else ''
nt     = ' --nt '     + opts.nt     if (opts.nt)     else ''
p      = ' --p '      + opts.p      if (opts.p)      else ''
q      = ' --q '      + opts.q      if (opts.q)      else ''

# general options for all routines
gen = nb + p + q + check

# ------------------------------------------------------------------------------
# filters a comma separated list csv based on items in list values.
# if no items from csv are in values, returns first item in values.
def filter_csv( values, csv ):
    f = filter( lambda x: x in values, csv.split( ',' ))
    if (not f):
        return values[0]
    return ','.join( f )
# end

# ------------------------------------------------------------------------------
# limit options to specific values
dtype_real    = ' --type ' + filter_csv( ('s', 'd'), opts.type )
dtype_complex = ' --type ' + filter_csv( ('c', 'z'), opts.type )
dtype_double  = ' --type ' + filter_csv( ('d'), opts.type )

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
    [ 'gbmm',  gen + dtype + transA + transB + mnk + la + ab + kl + ku ],
    [ 'gemm',  gen + dtype + transA + transB + mnk + la + ab ],

    [ 'hemm',  gen + dtype         + side + uplo     + mn + la + ab ],
    [ 'herk',  gen + dtype_real    + uplo + trans    + mn + la + ab ],
    [ 'herk',  gen + dtype_complex + uplo + trans_nc + mn + la + ab ],
    [ 'her2k', gen + dtype_real    + uplo + trans    + mn + la + ab ],
    [ 'her2k', gen + dtype_complex + uplo + trans_nc + mn + la + ab ],

    [ 'symm',  gen + dtype         + side + uplo     + mn + la + ab ],
    [ 'syr2k', gen + dtype_real    + uplo + trans    + mn + la + ab ],
    [ 'syr2k', gen + dtype_complex + uplo + trans_nt + mn + la + ab ],
    [ 'syrk',  gen + dtype_real    + uplo + trans    + mn + la + ab ],
    [ 'syrk',  gen + dtype_complex + uplo + trans_nt + mn + la + ab ],

    [ 'tbsm',  gen + dtype + side + uplo + transA + diag + mn + la + a + kd ],
    [ 'trmm',  gen + dtype + side + uplo + transA + diag + mn + la + a ],
    [ 'trsm',  gen + dtype + side + uplo + transA + diag + mn + la + a ],
    ]

# LU
if (opts.lu):
    cmds += [
    [ 'gesv',  gen + dtype + n + la ],
    [ 'getrf', gen + dtype + n + la ],  # todo: mn
    [ 'getrs', gen + dtype + n + la + trans ],
    #[ 'getri', gen + dtype + n ],
    #[ 'gecon', gen + dtype + n ],
    #[ 'gerfs', gen + dtype + n + trans ],
    #[ 'geequ', gen + dtype + n ],
    ]

# General Banded
if (opts.gb):
    cmds += [
    [ 'gbsv',  gen + dtype + n + kl + ku + la ],
    [ 'gbtrf', gen + dtype + n + kl + ku + la ],  # todo: mn
    [ 'gbtrs', gen + dtype + n + kl + ku + la + trans ],
    #[ 'gbcon', gen + dtype + n + kl + ku ],
    #[ 'gbrfs', gen + dtype + n + kl + ku + trans ],
    #[ 'gbequ', gen + dtype + n + kl + ku ],
    ]

# General Tri-Diagonal
if (opts.gt):
    cmds += [
    #[ 'gtsv',  gen + dtype + n ],
    #[ 'gttrf', gen + dtype +         n ],
    #[ 'gttrs', gen + dtype + n + trans ],
    #[ 'gtcon', gen + dtype +         n ],
    #[ 'gtrfs', gen + dtype + n + trans ],
    ]

# Cholesky
if (opts.chol):
    cmds += [
    [ 'posv',  gen + dtype + n + uplo + la ],
    [ 'potrf', gen + dtype + n + uplo + la ],
    [ 'potrs', gen + dtype + n + uplo + la ],
    #[ 'potri', gen + dtype + n + uplo ],
    #[ 'pocon', gen + dtype + n + uplo ],
    #[ 'porfs', gen + dtype + n + uplo ],
    #[ 'poequ', gen + dtype + n ],  # only diagonal elements (no uplo)

    #[ 'ppsv',  gen + dtype + n + uplo ],
    #[ 'pptrf', gen + dtype +         n + uplo ],
    #[ 'pptrs', gen + dtype + n + uplo ],
    #[ 'pptri', gen + dtype +         n + uplo ],
    #[ 'ppcon', gen + dtype +         n + uplo ],
    #[ 'pprfs', gen + dtype + n + uplo ],
    #[ 'ppequ', gen + dtype +         n + uplo ],

    #[ 'pbsv',  gen + dtype + n + kd + uplo ],
    #[ 'pbtrf', gen + dtype + n + kd + uplo ],
    #[ 'pbtrs', gen + dtype + n + kd + uplo ],
    #[ 'pbcon', gen + dtype + n + kd + uplo ],
    #[ 'pbrfs', gen + dtype + n + kd + uplo ],
    #[ 'pbequ', gen + dtype + n + kd + uplo ],

    #[ 'ptsv',  gen + dtype + n ],
    #[ 'pttrf', gen + dtype         + n ],
    #[ 'pttrs', gen + dtype + n + uplo ],
    #[ 'ptcon', gen + dtype         + n ],
    #[ 'ptrfs', gen + dtype + n + uplo ],
    ]

# symmetric indefinite, Bunch-Kaufman
if (opts.sysv):
    cmds += [
    #[ 'sysv',  gen + dtype + n + uplo ],
    #[ 'sytrf', gen + dtype + n + uplo ],
    #[ 'sytrs', gen + dtype + n + uplo ],
    #[ 'sytri', gen + dtype + n + uplo ],
    #[ 'sycon', gen + dtype + n + uplo ],
    #[ 'syrfs', gen + dtype + n + uplo ],

    #[ 'spsv',  gen + dtype + n + uplo ],
    #[ 'sptrf', gen + dtype + n + uplo ],
    #[ 'sptrs', gen + dtype + n + uplo ],
    #[ 'sptri', gen + dtype + n + uplo ],
    #[ 'spcon', gen + dtype + n  + uplo ],
    #[ 'sprfs', gen + dtype + n  + uplo ],
    ]

# symmetric indefinite, rook
if (opts.rook):
    cmds += [
    #[ 'sysv_rook',  gen + dtype + n + uplo ],
    #[ 'sytrf_rook', gen + dtype + n + uplo ],
    #[ 'sytrs_rook', gen + dtype + n + uplo ],
    #[ 'sytri_rook', gen + dtype + n + uplo ],
    ]

# symmetric indefinite, Aasen
if (opts.aasen):
    cmds += [
    #[ 'sysv_aasen',  gen + dtype + n + uplo ],
    #[ 'sytrf_aasen', gen + dtype + n + uplo ],
    #[ 'sytrs_aasen', gen + dtype + n + uplo ],
    #[ 'sytri_aasen', gen + dtype + n + uplo ],
    #[ 'sysv_aasen_2stage',  gen + dtype + n + uplo ],
    #[ 'sytrf_aasen_2stage', gen + dtype + n + uplo ],
    #[ 'sytrs_aasen_2stage', gen + dtype + n + uplo ],
    #[ 'sytri_aasen_2stage', gen + dtype + n + uplo ],
    ]

# Hermitian indefinite
if (opts.hesv):
    cmds += [
    #[ 'hesv',  gen + dtype + n + uplo ],
    #[ 'hetrf', gen + dtype + n + uplo ],
    #[ 'hetrs', gen + dtype + n + uplo ],
    #[ 'hetri', gen + dtype + n + uplo ],
    #[ 'hecon', gen + dtype + n + uplo ],
    #[ 'herfs', gen + dtype + n + uplo ],

    #[ 'hpsv',  gen + dtype + n + uplo ],
    #[ 'hptrf', gen + dtype + n + uplo ],
    #[ 'hptrs', gen + dtype + n + uplo ],
    #[ 'hptri', gen + dtype + n + uplo ],
    #[ 'hpcon', gen + dtype + n + uplo ],
    #[ 'hprfs', gen + dtype + n + uplo ],
    ]

# least squares
if (opts.least_squares):
    cmds += [
    #[ 'gels',   gen + dtype + mn + trans_nc ],
    #[ 'gelsy',  gen + dtype + mn ],
    #[ 'gelsd',  gen + dtype + mn ],
    #[ 'gelss',  gen + dtype + mn ],
    #[ 'getsls', gen + dtype + mn + trans_nc ],
    ]

# QR
if (opts.qr):
    cmds += [
    #[ 'geqrf', gen + dtype + n + wide + tall ],
    #[ 'ggqrf', gen + dtype + mnk ],
    #[ 'ungqr', gen + dtype + mn ], # n<=m
    #[ 'unmqr', gen + dtype + mnk + side + trans_nc ],
    ]

# LQ
if (opts.lq):
    cmds += [
    #[ 'gelqf', gen + dtype + mn ],
    #[ 'gglqf', gen + dtype + mn ],
    #[ 'unglq', gen + dtype + mn ],  # m<=n, k<=m  TODO Fix the input sizes to match constraints
    #[ 'unmlq', gen + dtype + mn ],
    ]

# QL
if (opts.ql):
    cmds += [
    #[ 'geqlf', gen + dtype + mn ],
    #[ 'ggqlf', gen + dtype + mn ],
    #[ 'ungql', gen + dtype + mn ],
    #[ 'unmql', gen + dtype + mn ],
    ]

# RQ
if (opts.rq):
    cmds += [
    #[ 'gerqf', gen + dtype + mn ],
    #[ 'ggrqf', gen + dtype + mnk ],
    #[ 'ungrq', gen + dtype + mnk ],
    #[ 'unmrq', gen + dtype + mn ],
    ]

 #symmetric eigenvalues
 #todo: add jobs
if (opts.syev):
    cmds += [
    #[ 'heev',  gen + dtype + n + uplo + jobz ],
    #[ 'heevx', gen + dtype + n + uplo ],
    #[ 'heevd', gen + dtype + n + uplo + jobz ],
    #[ 'heevr', gen + dtype + n + uplo + jobz ],
    #[ 'hetrd', gen + dtype + n + uplo ],
    #[ 'ungtr', gen + dtype + n + uplo ],
    #[ 'unmtr', gen + dtype + n + uplo + side + trans_nc ],

    #[ 'hpev',  gen + dtype + n + uplo + jobz ],
    #[ 'hpevx', gen + dtype + n + uplo + jobz ],
    #[ 'hpevd', gen + dtype + n + uplo + jobz ],
    #[ 'hpevr', gen + dtype + n + uplo + jobz ],
    #[ 'hptrd', gen + dtype + n + uplo ],
    #[ 'upgtr', gen + dtype + n + uplo ],
    #[ 'upmtr', gen + dtype + n + uplo ],

    #[ 'hbev',  gen + dtype + n + uplo + jobz ],
    #[ 'hbevx', gen + dtype + n + uplo + jobz ],
    #[ 'hbevd', gen + dtype + n + uplo + jobz ],
    #[ 'hbevr', gen + dtype + n + uplo + jobz ],
    #[ 'hbtrd', gen + dtype + n + uplo ],
    #[ 'obgtr', gen + dtype + n + uplo ],
    #[ 'obmtr', gen + dtype + n + uplo ],
    ]

# generalized symmetric eigenvalues
# todo: add jobs
if (opts.sygv):
    cmds += [
    #[ 'sygv',  gen + dtype + n + uplo ],
    #[ 'sygvx', gen + dtype + n + uplo ],
    #[ 'sygvd', gen + dtype + n + uplo ],
    #[ 'sygvr', gen + dtype + n + uplo ],
    #[ 'sygst', gen + dtype + n + uplo ],
    ]

# non-symmetric eigenvalues
if (opts.geev):
    cmds += [
    #[ 'geev',  gen + dtype + n + jobvl + jobvr ],
    #[ 'geevx', gen + dtype + n + jobvl + jobvr ],
    #[ 'gehrd', gen + dtype + n ],
    #[ 'orghr', gen + dtype + n ],
    #[ 'unghr', gen + dtype + n ],
    #[ 'ormhr', gen + dtype + n + side + trans ],
    #[ 'unmhr', gen + dtype + n + side + trans ],
    ]

# svd
if (opts.svd):
    cmds += [
    #[ 'gesvd',         gen + dtype + mn + jobu + jobvt ],
    #[ 'gesdd',         gen + dtype + mn + jobu ],
    #[ 'gesvdx',        gen + dtype + mn ],
    #[ 'gesvd_2stage',  gen + dtype + mn ],
    #[ 'gesdd_2stage',  gen + dtype + mn ],
    #[ 'gesvdx_2stage', gen + dtype + mn ],
    #[ 'gejsv',         gen + dtype + mn ],
    #[ 'gesvj',         gen + dtype + mn ],
    ]

# auxilary - norms
if (opts.aux):
    cmds += [
    [ 'gbnorm', gen + dtype + mn + norm + kl + ku ],
    [ 'genorm', gen + dtype + mn + norm ],
    [ 'henorm', gen + dtype + n  + norm + uplo ],
    [ 'synorm', gen + dtype + n  + norm + uplo ],
    [ 'trnorm', gen + dtype + mn + norm + uplo + diag ],
    ]

# additional blas
if (opts.blas):
    cmds += [
    #[ 'syr',   gen + dtype + n + uplo ],
    ]

# ------------------------------------------------------------------------------
# when output is redirected to file instead of TTY console,
# print extra messages to stderr on TTY console.
output_redirected = not sys.stdout.isatty()

# ------------------------------------------------------------------------------
# cmd is a pair of strings: (function, args)

def run_test( cmd ):
    cmd = opts.test +' '+ cmd[0] +' '+ cmd[1]
    print( cmd, file=sys.stderr )
    output = ''
    p = subprocess.Popen( cmd.split(), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT )
    # Read unbuffered ("for line in p.stdout" will buffer).
    for line in iter(p.stdout.readline, b''):
        print( line, end='' )
        output += line
    err = p.wait()
    return (err, output)
# end

# ------------------------------------------------------------------------------
failed_tests = []
passed_tests = []
ntests = len(opts.tests)
run_all = (ntests == 0)

for cmd in cmds:
    if (run_all or cmd[0] in opts.tests):
        if (not run_all):
            opts.tests.remove( cmd[0] )
        (err, output) = run_test( cmd )
        if (err):
            failed_tests.append( (cmd[0], err, output) )
        else:
            passed_tests.append( cmd[0] )
if (opts.tests):
    print( 'Warning: unknown routines:', ' '.join( opts.tests ))

# print summary of failures
nfailed = len( failed_tests )
if (nfailed > 0):
    print( '\n' + str(nfailed) + ' routines FAILED:',
           ', '.join( [x[0] for x in failed_tests] ),
           file=sys.stderr )

# generate jUnit compatible test report
if opts.xml:
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
    tree.write("report.xml")
# end

exit( nfailed )
