#!/usr/bin/env python
#
# Rejects commits that violate basic style rules:
#   - no trailing whitespace
#   - no tabs in source code
#   - unix newlines
# See https://bitbucket.org/icl/style/wiki/ICL_C_CPP_Coding_Style_Guide
# for additional rules.
#
# Enable by adding to .hg/hgrc:
# [hooks]
# pretxncommit.whitespace = hg export tip | tools/check-style-hook.py
#
# Run standalone:
# tools/check-style-hook.py files
# Files must be under Mercurial control so they show up in 'hg diff'.
#
# Adapted from:
# http://hgbook.red-bean.com/read/handling-repository-events-with-hooks.html

from __future__ import print_function

import re
import os
import sys

# .sh -- issues with ${foo}
src_ext = (
    '.c', '.h', '.hh', '.cc', '.hpp', '.cpp', '.cu', '.cuh',
    '.py', '.pl',
    '.f', '.f90', '.F', '.F90',
)

# ------------------------------------------------------------------------------
# raises exception if line matches regexp.
def check( regexp, line, msg, exclude=None ):
    if (re.search( regexp, line ) and
            (exclude is None or not re.search( exclude, line ))):
        raise Exception(msg)
# end

# ------------------------------------------------------------------------------
# check diff output for style
# lines is an iterator of 'hg diff' or 'hg export' output.
# returns number of errors found.
def check_style( lines ):
    errors = 0
    linenum = 0
    header = False
    filename = None
    is_src = False
    for line in lines:
        ##print( 'line:', line, end='' )
        # header - get filename
        if (header):
            m = re.search( r'^(?:---|\+\+\+) ([^\t\n]+)', line )
            if (m and m.group(1) != '/dev/null'):
                filename = m.group(1)
                (base, ext) = os.path.splitext( filename )
                is_src = (ext in src_ext)
            if (line.startswith('+++ ')):
                header = False
            continue
        # end

        # new diff starts new header
        if (line.startswith( 'diff ' )):
            header = True
            continue

        # hunk header - save line number
        m = re.search( r'^@@ -\d+,\d+ \+(\d+),', line )
        if (m):
            linenum = int( m.group(1) )
            continue

        # hunk body - check added lines for style errors
        if (line.startswith( '+' )):
            line = line[1:]  # chop off '+'
            try:
                check( r'[ \t]$', line, 'remove trailing whitespace' )
                check( r'\r',     line, 'Unix newlines only; no Windows returns!' )
                if (is_src):
                    check( r'\t', line, 'remove tab' )
            except Exception as ex:
                print( '%s, line %d: %s\n>>%s' %
                    (filename, linenum, str(ex), line), file=sys.stderr )
                errors += 1
        # end

        # increment line number on unchanged (' ') or added ('+') lines
        if (line and line[0] in ' +'):
            linenum += 1
    # end

    return errors
# end

# ------------------------------------------------------------------------------
# called with files, runs 'hg diff' on them
# otherwise, reads from stdin
if (__name__ == '__main__'):
    if (len(sys.argv) > 1):
        errors = check_style( os.popen( 'hg diff ' + ' '.join( sys.argv[1:] )))
    else:
        # assume stdin is a diff, e.g., from hg export tip
        errors = check_style( sys.stdin )

    if (errors):
        os.system( 'hg tip --template "{desc}" > .hg/commit.save' )
        print( 'message saved; use `hg commit -l .hg/commit.save`' )
        sys.exit(1)
    # end
# end
