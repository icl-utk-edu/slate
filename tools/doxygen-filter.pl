#!/usr/bin/perl

use strict;

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
