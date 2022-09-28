#!/usr/bin/perl
#
# Catches certain common style issues and rejects them, returning non-zero exit code.
# Can be used for pre-commit and pre-push hook to prevent errors from coming in.

use strict;
use Getopt::Std;

my %opts = ();
getopts( 'vl', \%opts ) || exit(-1);

my $verbose = $opts{v};
my $list    = $opts{l};

# escape characters for ANSI colors
# see http://en.wikipedia.org/wiki/ANSI_escape_code
my $esc     = chr(0x1B) . "[";
my $red     = "${esc}31m";
my $green   = "${esc}32m";
my $yellow  = "${esc}33m";
my $blue    = "${esc}34m";
my $magenta = "${esc}35m";
my $cyan    = "${esc}36m";
my $white   = "${esc}37m";
my $black   = "${esc}0m";

my $result = 0;
for my $ARG (@ARGV) {
    open( my $fileh, "<", $ARG ) or die( "Can't open $ARG: $!\n" );
    my $file_result = 0;
    while (<$fileh>) {
        chomp;
        my $orig = $_;
        s@//.*\S@//@;     # ignore C++ comments
        s/".*?"/"..."/g;  # ignore strings

        my $line = 0;
        if (m/\t/) {
            print( "$red$ARG:$.$black: tab\n" ) if (not $list);
            $line = 1;
        }

        if (m/ $/) {
            print( "$red$ARG:$.$black: trailing space\n" ) if (not $list);
            $line = 1;
        }

        if (m/^ *(if|for|while|switch|else if)\(/) {
            print( "$red$ARG:$.$black: missing space after `$1`\n" ) if (not $list);
            $line = 1;
        }

        if (m/^ *(if|for|while|switch|else if) \( +[^ ;]/) {
            print( "$red$ARG:$.$black: excess space inside parens after `$1`\n" ) if (not $list);
            $line = 1;
        }

        if (m/^ *\} *else/) {
            print( "$red$ARG:$.$black: don't cuddle } and else on same line\n" ) if (not $list);
            $line = 1;
        }

        if (m/\)\{/) {
            print( "$red$ARG:$.$black: missing space before { brace\n" ) if (not $list);
            $line = 1;
        }

        if (m/  +\\$/) {
            print( "$red$ARG:$.$black: excess space before line continuation\n" ) if (not $list);
            $line = 1;
        }

        # This checks 2-character operators.
        # It's hard to check < > = w/o full parser.
        if (m/[^ =](&&|\|\||==|<=|>=|!=|\+=|-=|\*=|\/=|\|=|\&=)[^ =]/) {
            print( "$red$ARG:$.$black: missing space around boolean operator\n" ) if (not $list);
            $line = 1;
        }

        # Prohibit space before , or ; unless at the beginning of a line.
        # Sometimes with #if conditions, the comma has to start the line.
        if (m/\S +[,;]/) {
            print( "$red$ARG:$.$black: excess space before comma or semi-colon\n" ) if (not $list);
            $line = 1;
        }

        # semi-colon ; must be at end, followed by space,
        # or followed by \n string as in printf( "];\n" ).
        if (m/(;(?!$|\s|\\n))/) {
            print( "$red$ARG:$.$black: missing space after semi-colon: <<$1>>\n" ) if (not $list);
            $line = 1;
        }

        # It's hard to check indentation w/o full parser, but at least
        # preprocessor, comments, and control keywords should be
        # indented correctly.
        if (m@^(    )* {1,3}(#|//|if|else|for|while|switch|case|default|break|throw)@) {
            print( "$red$ARG:$.$black: not 4-space indent\n" ) if (not $list);
            $line = 1;
        }

        # if (m/.{85}/) {
        #     print( "$red$ARG:$.$black: longer than 85 char hard limit\n" ) if (not $list);
        #     $line = 1;
        # }

        $file_result |= $line;
        if ($line and $verbose) {
            print( "<$orig>\n\n" );
        }
    }
    if ($file_result and $list) {
        print( "$ARG\n" );
    }
    $result |= $file_result;
}

#print( "result $result\n" );
exit( $result );
