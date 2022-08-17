#!/usr/bin/perl
#
# Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

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
