#!/usr/bin/perl
#
# changes $...$ and \[...\] Latex-style syntax
# to \f$...\f$ and \f[...\f] Doxygen-style syntax.

use strict;

# takes   \f$, \$, or   $
# returns \f$,  $, or \f$, respectively
sub dollar
{
    my( $pre ) = @_;
    if ($pre eq '\\') {
        # change \$ to $
        return '$';
    }
    elsif ($pre eq '\\f') {
        # don't change \f$
        return '\\f$';
    }
    else {
        # change $ to \f$
        return $pre . '\\f$';
    }
}

while (<>) {
    # replace \[ and \] by \f[ and \f]
    s/\\([\[\]])/\\f$1/g;

    # replace         $  by  \f$
    # replace        \$  by  $
    # don't change  \f$
    s/(\\f|\\|)\$/dollar($1)/eg;

    print
}
