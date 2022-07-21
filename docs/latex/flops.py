#!/usr/bin/env python3

'''!
Computes flop counts for LAPACK routines.
Outputs formulas in Latex format.
Has all formulas from LAWN 41, plus some derived formulas for gesvd, etc.

Command line usage:

    ./flops.py [sections] > flops.tex

By default, prints all sections. See g_sections for list of sections.

Interactive usage:

    > ipython
    Python 3.9.13 (main, May 19 2022, 14:10:54)
    In : import flops
    In : from flops import M, MgeN, N, NRHS
    In : expr = flops.geqrf( MgeN, N ) + flops.ormqr( 'L', MgeN, NRHS, N ) \
              + flops.trsm( 'L', N, NRHS )
    In : expr
    Out: 2*m_ge*n**2 + 4*m_ge*n*r + m_ge*n - 2*n**3/3 - n**2*r + n**2 + 3*n*r + 14*n/3

    In : flops.latex( expr )
    Out: '2 m_{ge} n^{2} + 4 m_{ge} n r + m_{ge} n - \\frac{2 n^{3}}{3} - n^{2} r + n^{2} + 3 n r + \\frac{14 n}{3}'

    In : flops.latex_clean( expr )
    Out: '2 m n^{2} + 4 m n r + m n - \\frac{2}{3} n^{3} - n^{2} r + n^{2} + 3 n r + \\frac{14}{3} n'

    In : flops.bigO( expr )
    Out: 2*m_ge*n**2 + 4*m_ge*n*r - 2*n**3/3 - n**2*r

    In : flops.latex( flops.bigO( expr ) )
    Out: '2 m_{ge} n^{2} + 4 m_{ge} n r - \\frac{2 n^{3}}{3} - n^{2} r'

    In : flops.latex_clean( flops.bigO( expr ) )
    Out: '2 m n^{2} + 4 m n r - \\frac{2}{3} n^{3} - n^{2} r'

Available sympy symbols: M, N, K, NRHS, MgeN, MleN, MggN, MllN.
These satisfy

    MgeN >= N >= MleN,
    MggN >> N >> MllN,

which is useful in routines like geqrf and gesvd that differentiate
such cases. These are printed as `m_ge, m_le, m_gg, m_ll`;
the subscript is stripped out by `latex_clean`.
'''

#===============================================================================
import sys
import re

import sympy
from sympy import symbols, Rational, latex, O

# If true, change fractions like \frac{2 n}{3} => \frac{2}{3} n
fix_frac = True

#-------------------------------------------------------------------------------
def die( *args ):
    print( 'Error:', *args )
    sys.exit()

#-------------------------------------------------------------------------------
# sympy variables

(M, N, K, NRHS) = symbols( 'm n k r', positive=True, integer=True )
(MgeN, MleN)    = symbols( 'm_ge m_le', positive=True, integer=True )
(MggN, MllN)    = symbols( 'm_gg m_ll', positive=True, integer=True )

#-------------------------------------------------------------------------------
def ge( x, y ):
    '''
    @return True if x >= y according to conditions:
        MgeN >= N >= MleN
    '''
    return ( ((x == MgeN or x == MggN) and y == N)
          or (x == N and (y == MleN or y == MllN)) )
# end

assert(     ge( MgeN, N ) )
assert( not ge( N, MgeN ) )
assert(     ge( N, MleN ) )
assert( not ge( MleN, N ) )

#-------------------------------------------------------------------------------
def gg( x, y ):
    '''
    @return True if x >> y according to conditions:
        MggN >> N >> MllN
    '''
    return ( (x == MggN and y == N)
          or (x == N and y == MllN) )
# end

assert(     ge( MggN, N ) )
assert( not ge( N, MggN ) )
assert(     ge( N, MllN ) )
assert( not ge( MllN, N ) )

assert(     gg( MggN, N ) )
assert( not gg( N, MggN ) )
assert(     gg( N, MllN ) )
assert( not gg( MllN, N ) )

#-------------------------------------------------------------------------------
def bigO( expr ):
    '''
    Takes sympy expression and returns its highest order terms.
    bigO( m*n**2 + m**2*n + n**2 + m*n ) => m*n**2 + m**2*n
    '''
    #print( '--- bigO(', expr, ')' )

    terms  = expr.as_ordered_terms()            # [ m**3, m**2, m ]
    polys  = [ t.as_poly()    for t in terms  ] # [ Poly(m**3), Poly(m**2), Poly(m) ]
    terms2 = [ p.terms()      for p in polys  ] # [ [((3,), 1)], [((2,), 1)], [((1,), 1)] ]
    powers = [ sum( x[0][0] ) for x in terms2 ] # [ 3, 2, 1 ]
    maxp   = max( powers )                      # 3

    #print( 'terms ', terms  )
    #print( 'polys ', polys  )
    #print( 'terms2', terms2 )
    #print( 'powers', powers )
    #print( 'maxp  ', maxp   )

    result = 0
    for (term, p) in zip( terms, powers ):
        if p == maxp:
            result += term

    #print( '==>', result )
    return result
# end

#-------------------------------------------------------------------------------
def latex_clean( expr ):
    '''
    Uses sympy latex, then applies some reformatting.
    - Remove m_xy subscripts.
    - Change \frac{n}{3}   => \frac{1}{3} n
    - Change \frac{2 n}{3} => \frac{2}{3} n
    '''
    s = latex( expr )
    s = re.sub( r'\bm_{\w\w}', r'm', s )

    if fix_frac:
        #print( s )
        # Without 1 in numer, do first.
        #                      1   2                3
        s = re.sub( r'\\frac\{(\w+(\^\{\d+\})?)\}\{(\d+)\}',
                    r'\\frac{1}{\3} \1', s )
        #print( s )
        #                      1     2   3                4
        s = re.sub( r'\\frac\{(\d+) (\w+(\^\{\d+\})?)\}\{(\d+)\}',
                    r'\\frac{\1}{\4} \2', s )
        #print( s )
    # end

    return s
# end

#-------------------------------------------------------------------------------
def output( name, expr ):
    '''
    Takes name and sympy expression to output.
    Returns tuple of 3 strings, e.g.,
        [ name, flops_latex, bigO_flops_latex ] = output( "gemv", gemv( M, N ) )
    '''
    e = expr.expand()
    latex_e = '$' + latex_clean( e ) + '$'

    b = bigO( e )
    eq = '=' if (e == b) else r'\approx'
    latex_b = '$' + eq + ' ' + latex_clean( b ) + '$'

    return ( name, latex_e, latex_b )
# end

#===============================================================================
# Latex strings
header =r'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% auto-generated by:
% ''' + ' '.join( sys.argv ) + r'''
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\documentclass{article}

\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{bm}       % bold symbol

\newcommand{\ph}{\phantom}

% -----
% set 1" margins on 8.5" x 11" paper
% top left is measured from 1", 1"
\topmargin       0in
\oddsidemargin   0in
\evensidemargin  0in
\headheight      0in
\headsep         0in
\topskip         0in
\textheight      9in
\textwidth       6.5in

\begin{document}

\def\arraystretch{1.3}  % 1 is the default, change as needed

\section*{LAPACK and BLAS floating point operation (flop) counts}

BLAS counts assume that $\alpha, \beta = 0$ or $1$,
otherwise a lower-order term is added.
\\
$m, n, k$ are the usual matrix dimensions.
\\
$r$ is the number of right-hand sides (\verb+nrhs+).
\\
L is left, R is right.
'''

#----------------------------------------
# Takes 1 argument: section name
#
table_header = r'''
\subsection*{%s}
\begin{tabular}{lll}
function  &  total flops  &  highest order  \\
\hline'''
# %-40s % ('total flops')

#----------------------------------------
# Takes 3 arguments: name, expr, bigO_expr
#
table_row = r'''%-20s  &  %s
                      &  %s  \\'''

#----------------------------------------
table_footer = r'\end{tabular}' + '\n'

#----------------------------------------
footer = r'\end{document}'

#===============================================================================
# Level 2 BLAS, from LAWN 41

def gemv( m=M, n=N ):
    return 2*m*n

def symv( n=N ):
    return 2*n*n

def sbmv( n=N, k=K ):
    return n*(4*k + 2) - 2*k*(k + 1)

def trmv( n=N ):
    return n**2

def tbmv( n=N, k=K ):
    return n*(2*k + 1) - k*(k + 1)

def trsv( n=N ):
    return n**2

def tbsv( n=N, k=K ):
    return n*(2*k + 1) - k*(k + 1)

def ger( m=M, n=N ):
    return 2*m*n

def syr( n=N ):
    return n*(n + 1)

def syr2( n=N ):
    return 2*n**2 + n

#-------------------------------------------------------------------------------
def blas2():
    print( table_header % ('Level 2 BLAS') )
    print( table_row % output( r'gemv', gemv( M, N )) )
    print( table_row % output( r'symv', symv( N    )) )
    print( table_row % output( r'sbmv', sbmv( N, K )) )
    print( table_row % output( r'trmv', trmv( N    )) )
    print( table_row % output( r'tbmv', tbmv( N, K )) )
    print( table_row % output( r'trsv', trsv( N    )) )
    print( table_row % output( r'tbsv', tbsv( N, K )) )
    print( table_row % output( r'ger',  ger(  M, N )) )
    print( table_row % output( r'syr',  syr(  N    )) )
    print( table_row % output( r'syr2', syr2( N    )) )
    print( table_footer )

#===============================================================================
# Level 3 BLAS, from LAWN 41

def gemm( m=M, n=N, k=K ):
    return 2*m*n*k

def symm( side='L', m=M, n=N ):
    if side == 'L':
        return 2 * m**2 * n
    elif side == 'R':
        return 2 * m * n**2
    else:
        raise ValueError( 'unknown side: ' + str( side ) )

def syrk( n=N, k=K ):
    return k*n*(n + 1)

def syr2k( n=N, k=K ):
    return 2*k*n**2 + n

def trmm( side='L', m=M, n=N ):
    if side == 'L':
        return m**2 * n
    elif side == 'R':
        return m * n**2
    else:
        raise ValueError( 'unknown side: ' + str( side ) )

def trsm( side='L', m=M, n=N ):
    if side == 'L':
        return m**2 * n
    elif side == 'R':
        return m * n**2
    else:
        raise ValueError( 'unknown side: ' + str( side ) )

#--------------------
# These are still the real flops, not complex flops.
def hemm( side='L', m=M, n=N ):
    return symm( side, m, n )

def herk( n=N, k=K ):
    return syrk( n, k )

def her2k( n=N, k=K ):
    return syr2k( n, k )

#-------------------------------------------------------------------------------
def blas3():
    print( table_header % ('Level 3 BLAS') )
    print( table_row % output( r'gemm',   gemm( M, N, K   ) ) )
    print( table_row % output( r'symm L', symm( 'L', M, N ) ) )
    print( table_row % output( r'symm R', symm( 'R', M, N ) ) )
    print( table_row % output( r'syrk',   syrk(  N, K     ) ) )
    print( table_row % output( r'syr2k',  syr2k( N, K     ) ) )
    print( table_row % output( r'trmm L', trmm( 'L', M, N ) ) )
    print( table_row % output( r'trmm R', trmm( 'R', M, N ) ) )
    print( table_row % output( r'trsm L', trsm( 'L', M, N ) ) )
    print( table_row % output( r'trsm R', trsm( 'R', M, N ) ) )
    print( table_footer )

def blas():
    blas2()
    blas3()


#===============================================================================
# LAPACK, from LAWN 41, plus some drivers

#---------------------------------------- Cholesky
def posv( n=N, nrhs=NRHS ):
    return potrf( n ) + potrs( n, nrhs )

def potrf( n=N ):
    return Rational(1,3)*n**3 + Rational(1,2)*n**2 + Rational(1,6)*n

def potri( n=N ):
    return Rational(2,3)*n**3 + Rational(1,2)*n**2 + Rational(5,6)*n

def potrs( n=N, nrhs=NRHS ):
    return nrhs * (2*n**2)

#---------------------------------------- Indefinite
def sysv( n=N, nrhs=NRHS ):
    return sytrf( n ) + sytrs( n, nrhs )

def sytrf( n=N ):
    return Rational(1,3)*n**3 + Rational(1,2)*n**2 + Rational(19,6)*n

def sytri( n=N ):
    return Rational(2,3)*n**3 + Rational(1,3)*n

def sytrs( n=N, nrhs=NRHS ):
    return nrhs * (2*n**2)

#--------------------
# These are still the real flops, not complex flops.
def hesv( n=N, nrhs=NRHS ):
    return sysv( n, nrhs )

def hetrf( n=N ):
    return sytrf( n )

def hetri( n=N ):
    return sytri( n )

def hetrs( n=N, nrhs=NRHS ):
    return sytrs( n, nrhs )

#---------------------------------------- LU
def gesv( n=N, nrhs=NRHS ):
    return getrf( n, n ) + getrs( n, nrhs )

def getrf( m=M, n=N ):
    return m*n**2 - Rational(1,3)*n**3 - Rational(1,2)*n**2 + Rational(5,6)*n

def getri( n=N ):
    return Rational(4,3)*n**3 - n**2 + Rational(5,3)*n

def getrs( n=N, nrhs=NRHS ):
    return nrhs * (2*n**2 - n)

#---------------------------------------- QR, LQ, RQ, QL
def geqrf( m=MgeN, n=N ):
    if ge( m, n ):
        return 2*m*n**2 - Rational(2,3)*n**3 + m*n + n**2 + Rational(14,3)*n
    else:
        return 2*n*m**2 - Rational(2,3)*m**3 + 3*m*n - m**2 + Rational(14,3)*m
        # typo in Lawn 41, 14/3n

geqlf = geqrf

#----------
def gerqf( m=MleN, n=N ):
    if ge( m, n ):
        return 2*m*n**2 - Rational(2,3)*n**3 + 2*m*n + Rational(17,3)*n
    else:
        return 2*n*m**2 - Rational(2,3)*m**3 + 2*m*n + Rational(17,3)*m
        # typo in Lawn 41, 17/3n

gelqf = gerqf

#----------
def orgqr( m=M, n=N, k=K ):
    return 4*m*n*k - 2*(m + n)*k**2 + Rational(4,3)*k**3 + 3*n*k - m*k - k**2 - Rational(4,3)*k

orgql = orgqr

#----------
def orglq( m=M, n=N, k=K ):
    return 4*m*n*k - 2*(m + n)*k**2 + Rational(4,3)*k**3 + 2*m*k - k**2 - Rational(1,3)*k

orgrq = orglq

#----------
def ormqr( side='L', m=M, n=N, k=K ):
    if side == 'L':
        return 4*m*n*k - 2*n*k**2 + 3*n*k
    elif side == 'R':
        return 4*m*n*k - 2*m*k**2 + 2*m*k + n*k - Rational(1,2)*k**2 + Rational(1,2)*k
    else:
        raise ValueError( 'unknown side: ' + str( side ) )

ormlq = ormqr
ormql = ormqr
ormrq = ormqr

#----------
def geqrs( m=M, n=N, nrhs=NRHS ):
    return nrhs * (4*m*n - n**2 + 3*n)

def geqrs2( m=M, n=N, nrhs=NRHS ):
    return ormqr( 'L', m, nrhs, n ) + trsm( 'L', n, nrhs )

def gels( m=M, n=N, nrhs=NRHS ):
    return geqrf( m, n ) + geqrs( m, n, nrhs )

#---------------------------------------- Eig, SVD
def gehrd( n=N ):
    return Rational(10,3)*n**3 - Rational(1,2)*n**2 - Rational(11,6)*n

def sytrd( n=N ):
    return Rational(4,3)*n**3 + 3*n**2 - Rational(17,6)*n

def gebrd( m=M, n=N ):
    # assume m >= n
    return 4*m*n**2 - Rational(4,3)*n**3 + 3*n**2 - m*n + Rational(25,3)*n

#--------------------
# These are still the real flops, not complex flops.
def hetrd( n=N ):
    return sytrd( n )

#---------------------------------------- other computational
def trtri( n=N ):
    return Rational(1,3)*n**3 + Rational(2,3)*n

#===============================================================================
# LAPACK, additional not in LAWN 41

#----------------------------------------
def cholqr( m=M, n=N ):
    #return gemm( m, n, n ) + potrf( n ) + trsm( 'R', m, n )
    return syrk( n, m ) + potrf( n ) + trsm( 'R', m, n )

#----------------------------------------
def gels_cholqr( m=M, n=N, nrhs=NRHS ):
    return cholqr( m, n ) + gemm( n, m, nrhs ) + trsm( 'L', n, nrhs )

#----------------------------------------
# from LAPACK source
def orgbr( vect='Q', m=M, n=N, k=K ):
    global M, N
    if vect == 'Q':
        if ge( m, k ):  # m >= k
            # assume m >= n >= k
            return orgqr( m, n, k )
        else:
            # assume m == n
            return orgqr( m, m, m )  # really m-1, m-1, m-1
    elif vect == 'P':
        if ge( n, k ):  # n >= k
            return orglq( m, n, k )
        else:
            return orglq( n, n, n )  # really n-1, n-1, n-1
    else:
        raise ValueError( 'unknown vect: ' + str( vect ) )
# end

#----------------------------------------
# from LAPACK source
def ormbr( vect='Q', side='L', m=M, n=N, k=K ):
    global M, N
    if side == 'L':
        nq = m
    elif side == 'R':
        nq = n
    else:
        raise ValueError( 'unknown side: ' + str( side ) )

    if vect == 'Q':
        if ge( nq, k ):  # nq = (left ? m : n) >= k
            return ormqr( side, m, n, k )
        else:
            return ormqr( side, m, n, nq )  # really m-1 (left) or n-1 (right), nq-1
    elif vect == 'P':
        if ge( nq, k ):  # nq = (left ? m : n) >= k
            return ormlq( side, m, n, k )
        else:
            return ormlq( side, m, n, nq )  # really m-1 (left) or n-1 (right), nq-1
    else:
        raise ValueError( 'unknown vect: ' + str( vect ) )
# end

#----------------------------------------
# tridiagonal eig D&C
def stedc( job='V', n=N ):
    if job == 'N':
        return n**2
    elif job == 'V':
        return Rational(4,3)*n**3
    else:
        raise ValueError( 'unknown job: ' + str( job ) )

#----------------------------------------
# SVD QR iteration (Golub & Reinsch)
# flop counts from Tony Chan (1982), based on 4 multiplies + 2 adds per rotation,
# 2 iterations per singular value, yielding
# 2 * sum( 1, ..., n ) = 2 * 0.5 n (n + 1) Givens rotation.
# Note the matrix A shrinks by 1 as each singular value is found.
# nru is rows of U, ncvt is columns of V
def bdsqr( job='V', n=N, ncvt=N, nru=N ):
    if job == 'N':
        return n**2
    elif job == 'V':
        return 6*nru*n**2 + 6*ncvt*n**2
    else:
        raise ValueError( 'unknown job: ' + str( job ) )

#----------------------------------------
# bidiagonal SVD D&C
def bdsdc( job='V', n=N ):
    if job == 'N':
        return n**2
    elif job == 'V':
        return Rational(8,3)*n**3
    else:
        raise ValueError( 'unknown job: ' + str( job ) )

#----------------------------------------
# SVD, QR iteration. Assumes m >= n.
def gesvd( job='V', m=MgeN, n=N ):
    ex = None
    if gg( m, n ):
        ex = (geqrf( m, n )
           +  gebrd( n, n ))

        if job == 'N':
            # Path 1: m >> n, no U vectors
            ex += bdsqr( 'N', n, 0, 0 )

            # Path 2: m >> n, U over A, no V
            # Path 3: m >> n, U over A, V
            # Path 4: m >> n, U, no V
            # Path 5: m >> n, U, V over A
            # todo

        elif job == 'S':
            # Path 6: m >> n, U, V
            ex += (orgqr( m, n, n )
                +  orgbr( 'Q', n, n, n )
                +  orgbr( 'P', n, n, n )
                +  bdsqr( 'V', n, n, n )
                +   gemm( m, n, n ))

            # Path 7: m >> n, full U, no V
            # Path 8: m >> n, full U, V over A
            # todo

        elif job == 'A':
            # Path 9: m >> n, full U, V
            ex += (orgqr( m, m, n )       # 2nd m
                +  orgbr( 'Q', n, n, n )
                +  orgbr( 'P', n, n, n )
                +  bdsqr( 'V', n, n, n )
                +   gemm( m, n, n ))
        else:
            raise ValueError( 'unknown job: ' + str( job ) )
    else:
        # Path 10: m >= n, but not m >> n
        ex = gebrd( m, n )

        if job == 'N':
            # Path 10n: no vectors
            ex += bdsqr( 'N', n, 0, 0 )

        elif job == 'S':
            # Path 10s: some vectors
            ex += (orgbr( 'Q', m, n, n )
                +  orgbr( 'P', n, n, n )
                +  bdsqr( 'V', n, n, m ))

        elif job == 'A':
            # Path 10a: all vectors
            ex += (orgbr( 'Q', m, m, n )
                +  orgbr( 'P', n, n, n )
                +  bdsqr( 'V', n, n, m ))
        else:
            raise ValueError( 'unknown job: ' + str( job ) )
    # end
    return ex
# end

#----------------------------------------
# SVD, divide and conquer. Assumes m >= n.
def gesdd( job='V', m=MgeN, n=N ):
    ex = None
    if gg( m, n ):
        ex = (geqrf( m, n )
           +  gebrd( n, n ))

        if job == 'N':
            # Path 1: m >> n, no U vectors
            ex += bdsdc( 'N', n )

            # Path 2: m >> n, U over A, V
            # todo

        elif job == 'S':
            # Path 3: m >> n, U, V
            ex += (orgqr( m, n, n )
                +  bdsdc( 'V', n )
                +  ormbr( 'Q', 'L', n, n, n )
                +  ormbr( 'P', 'R', n, n, n )
                +   gemm( m, n, n ))

        elif job == 'A':
            # Path 4: m >> n, full U, V
            ex += (orgqr( m, m, n )  # 2nd m
                +  bdsdc( 'V', n )
                +  ormbr( 'Q', 'L', n, n, n )
                +  ormbr( 'P', 'R', n, n, n )
                +   gemm( m, n, n ))
        else:
            raise ValueError( 'unknown job: ' + str( job ) )
    else:
        # Path 5: m >= n, but not m >> n
        ex = gebrd( m, n )

        if job == 'N':
            # Path 5n: no vectors
            ex += bdsdc( 'N', n )

        elif job == 'S':
            # Path 5s: some U, V vectors
            ex += (bdsdc( 'V', n )
                +  ormbr( 'Q', 'L', m, n, n )
                +  ormbr( 'P', 'R', n, n, n ))

        elif job == 'A':
            # Path 5a: all U, V vectors
            ex += (bdsdc( 'V', n )
                +  ormbr( 'Q', 'L', m, m, n )   # 2nd m
                +  ormbr( 'P', 'R', n, n, m ))  # 3rd m (seems odd)
        else:
            raise ValueError( 'unknown job: ' + str( job ) )
    # end
    return ex
# end


#===============================================================================
# LAPACK, sections

def chol():
    print( table_header % ('Cholesky') )
    print( table_row % output( r'posv',  posv ( N, NRHS ) ) )
    print( table_row % output( r'potrf', potrf( N ) ) )
    print( table_row % output( r'potrs', potrs( N, NRHS ) ) )
    print( table_row % output( r'potri', potri( N ) ) )
    print( table_footer )

def indefinite():
    print( table_header % ('Indefinite factorization') )
    print( table_row % output( r'sysv',  sysv ( N, NRHS ) ) )
    print( table_row % output( r'sytrf', sytrf( N ) ) )
    print( table_row % output( r'sytrs', sytrs( N, NRHS ) ) )
    print( table_row % output( r'sytri', sytri( N ) ) )
    print( table_footer )

def lu():
    print( table_header % ('LU') )
    print( table_row % output( r'gesv',  gesv ( N, NRHS ) ) )
    print( table_row % output( r'getrf', getrf( M, N ) ) )
    print( table_row % output( r'getrf, $m = n$', getrf( N, N ) ) )
    print( table_row % output( r'getrs', getrs( N, NRHS ) ) )
    print( table_row % output( r'getri', getri( N ) ) )
    print( table_footer )

def qr():
    print( table_header % ('QR') )
    print( table_row % output( r'gels, $m \ge n$', gels( MgeN, N, NRHS ) ) )
    print( table_row % output( r'\ph{gels,} $m = n$', gels( N, N, NRHS ) ) )
    #print( table_row % output( r'geqrs, $m \ge n$', geqrs ( MgeN, N, NRHS ) ) )
    #print( table_row % output( r'geqrs2, $m \ge n$', geqrs2( MgeN, N, NRHS ) ) )
    print( table_row % output( r'geqrf, $m \ge n$', geqrf( MgeN, N ) ) )
    print( table_row % output( r'\ph{geqrf,} $m \le n$', geqrf( MleN, N ) ) )
    print( table_row % output( r'\ph{geqrf,} $m = n$', geqrf( N, N ) ) )
    print( table_row % output( r'ormqr L', ormqr( 'L', M, NRHS, N ) ) )
    print( table_row % output( r'\ph{ormqr} R', ormqr( 'R', M, NRHS, N ) ) )
    print( table_row % output( r'orgqr', orgqr( M, N, K ) ) )
    print( table_row % output( r'\ph{orgqr,} $n = k$', orgqr( M, N, N ) ) )
    print( table_row % output( r'\ph{orgqr,} $m = n = k$', orgqr( N, N, N ) ) )
    print( table_footer )

    print( table_header % ('CholQR') )
    print( table_row % output( r'gels, $m \ge n$', gels_cholqr( MgeN, N, NRHS ) ) )
    print( table_row % output( r'\ph{gels,} $m = n$', gels_cholqr( N, N, NRHS ) ) )
    print( table_row % output( r'cholqr', cholqr( M, N ) ) )
    print( table_footer )

def lq():
    print( table_header % ('LQ') )
    print( table_row % output( r'gels, $m \le n$', gels( MleN, N, NRHS ) ) )
    print( table_row % output( r'gelqf, $m \ge n$', gelqf( MgeN, N ) ) )
    print( table_row % output( r'\ph{gelqf,} $m \le n$', gelqf( MleN, N ) ) )
    print( table_row % output( r'\ph{gelqf,} $m = n$', gelqf( N, N ) ) )
    print( table_row % output( r'ormlq L', ormlq( 'L', M, NRHS, N ) ) )
    print( table_row % output( r'ormlq R', ormlq( 'R', M, NRHS, N ) ) )
    print( table_row % output( r'orglq', orglq( M, N, K ) ) )
    print( table_row % output( r'orglq, $n = k$', orglq( M, N, N ) ) )
    print( table_row % output( r'orglq, $m = n = k$', orglq( N, N, N ) ) )
    #print( table_row % output( r'chollq', chollq( M, N ) ) )
    print( table_footer )

def rq():
    print( table_header % ('RQ') )
    print( table_row % output( r'gerqf, $m \ge n$', gerqf( MgeN, N ) ) )
    print( table_row % output( r'\ph{gerqf,} $m \le n$', gerqf( MleN, N ) ) )
    print( table_row % output( r'\ph{gerqf,} $m = n$', gerqf( N, N ) ) )
    print( table_row % output( r'ormrq L', ormrq( 'L', M, NRHS, N ) ) )
    print( table_row % output( r'\ph{ormrq} R', ormrq( 'R', M, NRHS, N ) ) )
    print( table_row % output( r'orgrq', orgrq( M, N, K ) ) )
    print( table_row % output( r'\ph{orgrq,} $n = k$', orgrq( M, N, N ) ) )
    print( table_row % output( r'\ph{orgrq,} $m = n = k$', orgrq( N, N, N ) ) )
    print( table_footer )

def ql():
    print( table_header % ('QL') )
    print( table_row % output( r'geqlf, $m \ge n$', geqlf( MgeN, N ) ) )
    print( table_row % output( r'\ph{geqlf,} $m \le n$', geqlf( MleN, N ) ) )
    print( table_row % output( r'\ph{geqlf,} $m = n$', geqlf( N, N ) ) )
    print( table_row % output( r'ormql L', ormql( 'L', M, NRHS, N ) ) )
    print( table_row % output( r'\ph{ormql} R', ormql( 'R', M, NRHS, N ) ) )
    print( table_row % output( r'orgql', orgql( M, N, K ) ) )
    print( table_row % output( r'\ph{orgql,} $n = k$', orgql( M, N, N ) ) )
    print( table_row % output( r'\ph{orgql,} $m = n = k$', orgql( N, N, N ) ) )
    print( table_footer )

def geev():
    print( table_header % ('Non-symmetric eig') )
    #print( table_row % output( r'geev',  geev ( N ) ) )
    print( table_row % output( r'gehrd', gehrd( N ) ) )
    #print( table_row % output( r'ormhr', ormhr( N ) ) )
    #print( table_row % output( r'orghr', orghr( N ) ) )
    print( table_footer )

def syev():
    print( table_header % ('Symmetric eig') )
    #print( table_row % output( r'syev',  syev ( N ) ) )
    #print( table_row % output( r'syevd', syevd( N ) ) )
    #print( table_row % output( r'syevr', syevr( N ) ) )
    print( table_row % output( r'sytrd', sytrd( N ) ) )
    #print( table_row % output( r'ormtr', ormtr( N ) ) )
    #print( table_row % output( r'orgtr', orgtr( N ) ) )
    print( table_footer )

def svd():
    print( table_header % ('SVD') )
    print( table_row % output( r'gesvd, N, $m = n$', gesvd( 'N', N, N ) ) )
    print( table_row % output( r'gesdd, N, $m = n$', gesdd( 'N', N, N ) ) )
    print( table_row % output( r'gesvd, N, $m \ge n$', gesvd( 'N', MgeN, N ) ) )
    print( table_row % output( r'gesdd, N, $m \ge n$', gesdd( 'N', MgeN, N ) ) )
    print( table_row % output( r'gesvd, N, $m \gg n$', gesvd( 'N', MggN, N ) ) )
    print( table_row % output( r'gesdd, N, $m \gg n$', gesdd( 'N', MggN, N ) ) )
    print( r'\hline' )
    print( table_row % output( r'gesvd, S, $m = n$', gesvd( 'S', N, N ) ) )
    print( table_row % output( r'gesdd, S, $m = n$', gesdd( 'S', N, N ) ) )
    print( table_row % output( r'gesvd, S, $m \ge n$', gesvd( 'S', MgeN, N ) ) )
    print( table_row % output( r'gesdd, S, $m \ge n$', gesdd( 'S', MgeN, N ) ) )
    print( table_row % output( r'gesvd, S, $m \gg n$', gesvd( 'S', MggN, N ) ) )
    print( table_row % output( r'gesdd, S, $m \gg n$', gesdd( 'S', MggN, N ) ) )
    print( r'\hline' )
    print( table_row % output( r'gesvd, A, $m = n$', gesvd( 'A', N, N ) ) )
    print( table_row % output( r'gesdd, A, $m = n$', gesdd( 'A', N, N ) ) )
    print( table_row % output( r'gesvd, A, $m \ge n$', gesvd( 'A', MgeN, N ) ) )
    print( table_row % output( r'gesdd, A, $m \ge n$', gesdd( 'A', MgeN, N ) ) )
    print( table_row % output( r'gesvd, A, $m \gg n$', gesvd( 'A', MggN, N ) ) )
    print( table_row % output( r'gesdd, A, $m \gg n$', gesdd( 'A', MggN, N ) ) )
    print( r'\hline' )
    print( table_row % output( r'gebrd', gebrd( M, N ) ) )
    print( table_row % output( r'gebrd, $m = n$', gebrd( N, N ) ) )
    #print( table_row % output( r'ormbr', ormbr( M, N ) ) )
    #print( table_row % output( r'orgbr', orgbr( M, N ) ) )
    print( table_footer )
    print( r'\vspace{1em} \noindent For $m < n$, swap $m$ and $n$ in above equations.' )

def computational():
    print( table_header % ('LAPACK: other computational') )
    print( table_row % output( r'trtri', trtri( N ) ) )
    print( table_footer )

def auxiliary():
    #print( table_header % ('LAPACK auxiliary') )
    #print( table_row % output( r'lascl', lascl( M, N ) ) )
    #print( table_footer )
    pass

def lapack():
    chol()
    indefinite()
    lu()
    qr()
    lq()
    rq()
    ql()
    geev()
    syev()
    svd()
    computational()
    auxiliary()

def all():
    blas()
    lapack()

#===============================================================================
# main

g_sections = [
    'blas2', 'blas3', 'blas',
    'chol', 'indefinite', 'lu', 'qr', 'lq', 'rq', 'ql',
    'geev', 'syev', 'svd',
    'computational', 'auxiliary',
    'lapack',
    'all',
]

#-------------------------------------------------------------------------------
def flops( sections ):
    '''
    Print Latex document with flop counts for given sections.
    '''
    print( header )
    for section in sections:
        if section not in g_sections:
            die( 'Unknown section', section )

        xsection = globals()[ section ]
        xsection()
    # end
    print( footer )
# end

#-------------------------------------------------------------------------------
def update_src( routines ):
    '''
    Adds Complexity line to src file, right before tparam line.
    Experimental.
    '''
    for routine in routines:
        try:
            xroutine = globals()[ routine ]
            (name, expr, bigO_expr) = output( routine, xroutine() )
            comp = '/// Complexity: ' + bigO_expr + ' flops (in real)\n///\n'
            #print( 'comp:\n', comp )
            comp = re.sub( r'\\', r'\\\\', comp )

            file = 'src/' + routine + '.cc'
            infile = open( file )
            txt = infile.read()
            infile.close()

            txt = re.sub( r'(//--+\n/// @tparam)', comp + r'\1', txt )
            #print( 'txt:\n', txt )

            outfile = open( file, 'w' )
            outfile.write( txt )
            outfile.close()
        except Exception as ex:
            print( 'Error processing', routine + ':', ex )
# end

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    if len( sys.argv ) > 1:
        sections = sys.argv[ 1: ]
    else:
        sections = ['all']
    flops( sections )
    #update_src( sections )
