# Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

'''
Tags project with version based on current date, and creates tar file.
Tag is yyyy.mm.rr, where yyyy.mm is current year and month,
and rr is a release counter within current month, starting at 0.
Version is an integer yyyymmrr, to allow simple comparisons.

Requires Python >= 3.7.

Usage:

    #!/usr/bin/env python3
    import release
    release.copyright()
    release.make( 'project', 'version.h', 'version.c' )

'project' is the name of the project, used for the tar filename.

'version.h' is a header containing the following #define for the version,
with PROJECT changed to the project's name.

    // Version is updated by make_release.py; DO NOT EDIT.
    // Version 2020.02.00
    #define PROJECT_VERSION 20200200

'version.c' is a source file containing the following #define for the id:

    // PROJECT_ID is the Mercurial or git commit hash ID, either
    // defined by `hg id` or `git rev-parse --short HEAD` in Makefile,
    // or defined here by make_release.py for release tar files. DO NOT EDIT.
    #ifndef PROJECT_ID
    #define PROJECT_ID "unknown"
    #endif

    const char* id() {
        return PROJECT_ID;
    }

    int version() {
        return PROJECT_VERSION;
    }

Steps this takes:

1. Marks version in repo.
   - Saves the Version to version.h.
   - Updates copyright year in all files.
   - Commits that change.
   - Tags that commit.
2. Prepares archive in directory project-tag.
   - Saves the `git rev-parse --short HEAD` to version.c.
   - Generates Doxygen docs.
3. Generates tar file project-tag.tar.gz
'''

from __future__ import print_function

import sys
MIN_PYTHON = (3, 7)
assert sys.version_info >= MIN_PYTHON, "requires Python >= %d.%d" % MIN_PYTHON

import os
import datetime
import re
import subprocess
from   subprocess import PIPE

#-------------------------------------------------------------------------------
def myrun( cmd, **kwargs ):
    '''
    Simple wrapper around subprocess.run(), with check=True.
    If cmd is a str, it is split on spaces before being passed to run.
    Prints cmd.
    kwargs are passed to run(). Set `stdout=PIPE, text=True` if you want the
    output returned.
    '''
    if (type(cmd) is str):
        cmd = cmd.split(' ')
    print( '\n>>', ' '.join( cmd ) )
    return subprocess.run( cmd, check=True, **kwargs ).stdout
# end

#-------------------------------------------------------------------------------
def file_sub( filename, search, replace, **kwargs ):
    '''
    Replaces search regexp with replace in file filename.
    '''
    #print( 'reading', filename )
    txt = open( filename ).read()
    txt2 = re.sub( search, replace, txt, **kwargs )
    if (txt != txt2):
        #print( 'writing', filename )
        open( filename, mode='w' ).write( txt2 )
# end

#-------------------------------------------------------------------------------
def copyright():
    '''
    Update copyright in all files.
    '''
    today = datetime.date.today()
    year  = today.year

    files = myrun( 'git ls-tree -r master --name-only',
                   stdout=PIPE, text=True ).rstrip().split( '\n' )
    print( '\n>> Updating copyright in:', end=' ' )
    for file in files:
        if (re.search( r'^(old/|src/hip/)', file ) or os.path.isdir( file )):
            continue

        print( file, end=', ' )
        file_sub( file,
                  r'Copyright \(c\) (\d+)(-\d+)?, University of Tennessee',
                  r'Copyright (c) \1-%04d, University of Tennessee' % (year) )
    # end
    print()

    myrun( 'make hipify' )

    myrun( 'git diff' )
    print( '>> Commit changes [yn]? ', end='' )
    response = input()
    if (response != 'y'):
        print( '>> Release aborted. Please revert changes as desired.' )
        exit(1)

    myrun( ['git', 'commit', '-m', 'copyright '+ str(year), '.'] )
# end

#-------------------------------------------------------------------------------
def make( project, version_h, version_c ):
    '''
    Makes project release.
    '''
    today = datetime.date.today()
    year  = today.year
    month = today.month
    release = 0

    top_dir = os.getcwd()

    # Search for latest tag this month and increment release if found.
    tags = myrun( 'git tag', stdout=PIPE, text=True ).rstrip().split( '\n' )
    tags.sort( reverse=True )
    pattern = r'%04d\.%02d\.(\d+)' % (year, month)
    for tag in tags:
        s = re.search( pattern, tag )
        if (s):
            release = int( s.group(1) ) + 1
            break

    tag = '%04d.%02d.%02d' % (year, month, release)
    version = '%04d%02d%02d' % (year, month, release)
    print( '\n>> Tag '+ tag +', Version '+ version )

    #--------------------
    # Check change log
    txt = open( 'CHANGELOG.md' ).read()
    if (not re.search( tag, txt )):
        print( '>> Add', tag, 'to CHANGELOG.md. Release aborted.' )
        exit(1)

    #--------------------
    # Update version in version_h.
    print( '\n>> Updating version in:', version_h )
    file_sub( version_h,
              r'// Version \d\d\d\d.\d\d.\d\d\n(#define \w+_VERSION) \d+',
              r'// Version %s\n\1 %s' % (tag, version), count=1 )

    print( '\n>> Updating version in: GNUmakefile' )
    file_sub( 'GNUmakefile',
              r'VERSION:\d\d\d\d.\d\d.\d\d',
              r'VERSION:%s' % (tag), count=1 )

    print( '\n>> Updating version in: CMakeLists.txt' )
    file_sub( 'CMakeLists.txt',
              r'VERSION \d\d\d\d.\d\d.\d\d',
              r'VERSION %s' % (tag), count=1 )

    print( '\n>> Updating version in: doxyfile.conf' )
    file_sub( 'docs/doxygen/doxyfile.conf',
              r'(PROJECT_NUMBER *=) *"\d+\.\d+\.\d+"',
              r'\1 "%s"' % (tag), count=1 )

    myrun( 'git diff' )
    myrun( 'git diff --staged' )
    print( '>> Do changes look good? Continue building release [yn]? ', end='' )
    response = input()
    if (response != 'y'):
        print( '>> Release aborted. Please revert changes as desired.' )
        exit(1)

    myrun( ['git', 'commit', '-m', 'Version '+ tag, '.'] )
    myrun( ['git', 'tag', tag, '-a', '-m', 'Version '+ tag] )

    #--------------------
    # Prepare tar file.
    dir = project +'-'+ tag
    print( '\n>> Preparing files in', dir )

    # Move any existing dir to dir-#; maximum # is 100.
    if (os.path.exists( dir )):
        for index in range( 1, 100 ):
            backup = '%s-%d' % (dir, index)
            if (not os.path.exists( backup )):
                os.rename( dir, backup )
                print( 'backing up', dir, 'to', backup )
                break
    # end

    # git archive doesn't recurse submodule; do it with rsync.
    os.mkdir( dir )
    subprocess.run( 'git ls-files --recurse-submodule | '
                    + 'rsync -av --files-from=- ./ ' + dir + '/', shell=True )
    os.chdir( dir )

    # Update hash ID in version_c.
    id = myrun( 'git rev-parse --short HEAD', stdout=PIPE, text=True ).strip()
    print( '\n>> Setting ID in:', version_c )
    file_sub( version_c,
              r'^(#define \w+_ID) "unknown"',
              r'\1 "'+ id +'"', count=1, flags=re.M )

    # Build Doxygen docs. Create dummy 'make.inc' to avoid 'make config'.
    open( 'make.inc', mode='a' ).close()
    myrun( 'make docs blas=openblas' )
    os.unlink( 'make.inc' )

    os.chdir( '..' )

    tar = dir + '.tar.gz'
    print( '\n>> Creating tar file', tar )
    myrun( 'tar -zcvf '+ tar +' '+ dir )

    #--------------------
    # Update online docs.
    myrun( ['rsync', '-av', '--delete',
            '--exclude', 'artwork',  # keep artwork on icl.bitbucket.io
            dir + '/docs/html/',
            'icl.bitbucket.io/' + project + '/'] )

    os.chdir( 'icl.bitbucket.io' )
    myrun( 'git add ' + project )
    myrun( 'git status' )
    print( '>> Do changes look good? Commit docs [yn]? ', end='' )
    response = input()
    if (response != 'y'):
        print( '>> Doc update aborted. Please revert changes as desired.' )
        exit(1)

    # Commit staged files.
    myrun( ['git', 'commit', '-m', project + ' version ' + tag] )

    print( '>> Run `git push` to make changes live [yn]? ', end='' )
    response = input()
    if (response == 'y'):
        myrun( 'git push' )
    else:
        print( '>> Doc update aborted. Please revert changes as desired.' )
        exit(1)

    os.chdir( top_dir )
# end
