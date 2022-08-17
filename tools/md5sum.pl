#!/usr/bin/perl
#
# Generate md5 sums of files, with output compatible with md5sum.
# Doesn't support any options of md5sum, though (--check, etc.).

use strict;
use Digest::MD5;

foreach my $filename (@ARGV) {
    my $file;
    if (not open( $file, '<', $filename )) {
        warn "$0: $filename: $!\n";
        next;
    }
    binmode( $file );
    print Digest::MD5->new->addfile( $file )->hexdigest, "  $filename\n";
}
